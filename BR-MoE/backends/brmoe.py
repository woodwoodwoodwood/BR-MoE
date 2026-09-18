import torch
import sys
import os
from ..kernels import brmoe as brmoe  # Prefer local kernels module
from ..core.quantize import BRMoELinear as HQQLinear, Quantizer
from ..core.peft import HQQLinearLoRA  # ensure LoRA type is available for isinstance checks


class BRMoE_Asymmetric_Linear(torch.nn.Module):
    def __init__(self, W: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, u: None, v: None,
                 bias=None, groupsize=64):
        super().__init__()
        m, n = W.shape
        device = W.device

        _linear = torch.nn.Linear(m, n)
        _linear.weight.data = W.half().t()

        _layer = brmoe.Layer3bitWithZeros(m, n, groupsize)
        _layer.k = m
        _layer.n = n
        _layer.groupsize = groupsize

        _layer.B1 = torch.empty(
            (m // 16, n * 16 // 16), dtype=torch.int, device=device
        )
        _layer.B2 = torch.empty(
            (m // 16, n * 16 // 32), dtype=torch.int, device=device
        )
        _layer.s = torch.empty((m // groupsize, n), dtype=torch.half, device=device)
        _layer.z = torch.empty((m // groupsize, n), dtype=torch.half, device=device)
        _layer.pack(_linear, scales.t(), zeros.t())
        self.bias = bias.half() if (bias is not None) else None

        self.Wq_packed1 = _layer.B1.clone()
        self.Wq_packed2 = _layer.B2.clone()
        self.scales = _layer.s.clone()
        self.zeros = _layer.z.clone()

        # workspace must be int buffer: see brmoe_cuda binding checks
        self.workspace_fp = torch.zeros(n // 128 * 16, dtype=torch.int, device=device)
        self.in_features = m
        self.out_features = n
        self.group_size = groupsize
        self.axis = 1
        self.device = device
        self.compute_dtype = torch.float16
        self.U = torch.nn.Parameter(u, requires_grad=False) if (u is not None) else None
        self.V = torch.nn.Parameter(v, requires_grad=False) if (v is not None) else None


        del _linear, _layer
        torch.cuda.empty_cache()

    @torch.no_grad()
    def matmul(self, x):
        out = torch.empty(
            x.shape[:-1] + (self.scales.shape[1],), 
            dtype=x.dtype, 
            device=x.device
        )
        # Try valid kernel tile pairs (thread_k, thread_n) in order of preference.
        # For gs=128, prefer (128,128) then (256,64) then (64,256).
        # For gs=64 (or others), prefer (256,64) then (128,128) then (64,256).
        preferred_pairs = (
            [(128, 128), (256, 64), (64, 256)]
            if self.group_size >= 128
            else [(256, 64), (128, 128), (64, 256)]
        )
        last_err = None
        for tk, tn in preferred_pairs:
            if (self.in_features % tk != 0) or (self.out_features % tn != 0):
                continue
            try:
                brmoe.mul_3bit_with_zeros(
                    x.to(self.device).view((-1, x.shape[-1])),
                    self.Wq_packed1,
                    self.Wq_packed2,
                    out.view((-1, out.shape[-1])),
                    self.scales,
                    self.zeros,
                    self.workspace_fp,
                    thread_k=tk,
                    thread_n=tn,
                )
                last_err = None
                break
            except RuntimeError as e:
                # Try next pair if kernel shape is incompatible
                last_err = e
                continue
        if last_err is not None:
            raise last_err
        return out

    @torch.jit.ignore
    def forward(self, x):
        out = self.matmul(x)
        if self.bias is not None:
            out += self.bias
        if self.U != None and self.V != None:
            out = out + (x @ self.V) @ self.U
        return out


def patch_hqq_to_brmoe_asymmetric(layer, patch_params):
    hqq_layer = None
    if isinstance(layer, HQQLinear):
        hqq_layer = layer

    if hqq_layer is None:
        return layer

    hqq_layer = layer.linear_layer if hasattr(layer, "linear_layer") else layer
    # Check config
    if (
        (hqq_layer.meta["axis"] == 0)
        or (hqq_layer.meta["group_size"] not in (64, 128))
        or (hqq_layer.meta["nbits"] != 3)
    ):
        print("Skipping brmoe conversion for", hqq_layer)
        return layer

    z = hqq_layer.meta["zero"]
    s = hqq_layer.meta["scale"]

    W_r = hqq_layer.unpack(dtype=hqq_layer.compute_dtype)
    W_r = W_r[: s.shape[0]]

    # Combine them
    W_r = (W_r - z) * s
    z = -z * s

    n = hqq_layer.meta["shape"][0]
    W_r = W_r.reshape((n, -1))
    s = s.reshape((n, -1))
    z = z.reshape((n, -1))
    if hasattr(layer, "U") and hasattr(layer, "V") and (layer.U is not None) and (layer.V is not None):
        u = layer.U.t()
        v = layer.V.t()
    else:
        u = None
        v = None
    # Build BRMoE asymmetric layer with safe fallback
    try:
        brmoe_layer = BRMoE_Asymmetric_Linear(
            W_r.t(), s.t(), z.t(), u, v, bias=hqq_layer.bias, groupsize=hqq_layer.meta["group_size"]
        )
    except Exception as e:
        name_b = getattr(hqq_layer, 'name', 'layer')
        try:
            m_b, n_b = W_r.t().shape
        except Exception:
            m_b, n_b = None, None
        print(
            f"BRMoE (asym) build failed during pack for {name_b}: {e}. "
            f"shape=({m_b},{n_b}), axis={hqq_layer.meta.get('axis', None)}, gs={hqq_layer.meta.get('group_size', None)}. "
            "Falling back to original backend."
        )
        torch.cuda.empty_cache()
        return layer

    # Runtime probe to ensure kernels are executable on this device
    try:
        with torch.no_grad():
            x_probe = torch.zeros((1, brmoe_layer.in_features), dtype=torch.float16, device=brmoe_layer.device)
            _ = brmoe_layer.matmul(x_probe)
            torch.cuda.synchronize()
    except Exception as e:
        name_p = getattr(hqq_layer, 'name', 'layer')
        print(
            f"BRMoE (asym) probe failed for {name_p}: {e}. "
            f"shape=({brmoe_layer.in_features},{brmoe_layer.out_features}), gs={brmoe_layer.group_size}. "
            "Falling back to original backend."
        )
        del brmoe_layer
        torch.cuda.empty_cache()
        return layer

    # Only after successful probe, drop original and replace
    del hqq_layer.W_q
    del hqq_layer.meta
    del hqq_layer.bias
    del hqq_layer
    torch.cuda.empty_cache()

    if isinstance(layer, HQQLinear):
        return brmoe_layer
    return layer


class BRMoE_Symmetric_Layer(torch.nn.Module):
    def __init__(self, W: torch.Tensor, scales: torch.Tensor, qz=None,u=None, v=None,
                 bias=None, groupsize=64):
        super().__init__()
        m, n = W.shape
        device = W.device

        _linear = torch.nn.Linear(m, n)
        _linear.weight.data = W.half().t()

        _layer = brmoe.Layer3bit(m, n, groupsize)
        _layer.k = m
        _layer.n = n
        _layer.groupsize = groupsize

        _layer.B1 = torch.empty(
            (m // 16, n * 16 // 16), dtype=torch.int, device=device
        )
        _layer.B2 = torch.empty(
            (m // 16, n * 16 // 32), dtype=torch.int, device=device
        )
        _layer.s = torch.empty(
            (m // groupsize, n), dtype=torch.half, device=device
        )

        _layer.pack(_linear, scales.t())
        self.bias = bias.half() if (bias is not None) else None
        self.Wq_packed1 = _layer.B1.clone()
        self.Wq_packed2 = _layer.B2.clone()
        self.scales = _layer.s.clone()

        # workspace must be int buffer: see brmoe_cuda binding checks
        self.workspace_fp = torch.zeros(n // 128 * 16, dtype=torch.int, device=device)
        self.in_features = m
        self.out_features = n
        self.group_size = groupsize
        self.axis = 1
        self.device = device
        self.compute_dtype = torch.float16
        self.U = torch.nn.Parameter(u, requires_grad=False) if (u is not None) else None
        self.V = torch.nn.Parameter(v, requires_grad=False) if (v is not None) else None
        self.qz = torch.nn.Parameter(qz, requires_grad=False) if (qz is not None) else None
        
        # Pre-calculate optimal tile sizes to avoid runtime trial-and-error
        self.tk, self.tn = self._calculate_optimal_tiles()
        
        # Control flags
        self.disable_qz = False  # Can be set to disable zero-point correction
        self.debug_shapes = False  # Can be set to enable debug prints

        del _linear, _layer
        torch.cuda.empty_cache()
    
    def _calculate_optimal_tiles(self):
        """Calculate optimal tile sizes for the CUDA kernel"""
        preferred_pairs = (
            [(128, 128), (256, 64), (64, 256)]
            if self.group_size >= 128
            else [(256, 64), (128, 128), (64, 256)]
        )
        
        # Find the first valid tile size
        for tk, tn in preferred_pairs:
            if (self.in_features % tk == 0) and (self.out_features % tn == 0):
                return tk, tn
        
        # Fallback to default if no perfect match
        return 128, 128

    @torch.no_grad()
    def matmul(self, x):
        # print("BRMoE_Symmetric_Layer -> matmul")
        out = torch.empty(
            x.shape[:-1] + (self.scales.shape[1],), 
            dtype=x.dtype, 
            device=x.device
        )
        # Try valid kernel tile pairs (thread_k, thread_n) in order of preference.
        preferred_pairs = (
            [(128, 128), (256, 64), (64, 256)]
            if self.group_size >= 128
            else [(256, 64), (128, 128), (64, 256)]
        )
        last_err = None
        for tk, tn in preferred_pairs:
            if (self.in_features % tk != 0) or (self.out_features % tn != 0):
                continue
            try:
                brmoe.mul_3bit(
                    x.to(self.device).view((-1, x.shape[-1])),
                    self.Wq_packed1,
                    self.Wq_packed2,
                    out.view((-1, out.shape[-1])),
                    self.scales,
                    self.workspace_fp,
                    thread_k=tk,
                    thread_n=tn,
                )
                last_err = None
                break
            except RuntimeError as e:
                last_err = e
                continue
        if last_err is not None:
            raise last_err
        return out

    @torch.jit.ignore
    def forward(self, x):
        out = self.matmul(x)
        if self.qz is not None and os.getenv("BRMOE_DISABLE_QZ", "0") != "1":
            # Extra shift or correction if desired
            y = x.reshape(*x.shape[:-1], -1, self.group_size).sum(dim=-1)
            try:
                # Prefer (B, G) @ (G, N)
                if y.shape[-1] == self.qz.shape[0]:
                    out = out + (y @ self.qz)
                # If stored as (N, G), transpose
                elif y.shape[-1] == self.qz.shape[1]:
                    out = out + (y @ self.qz.t())
                else:
                    if os.getenv("BRMOE_DEBUG_SHAPES", "0") == "1":
                        print(f"[BRMoE sym qz] shape mismatch: y={tuple(y.shape)}, qz={tuple(self.qz.shape)}, out={tuple(out.shape)}; skipping qz correction")
            except Exception as e:
                if os.getenv("BRMOE_DEBUG_SHAPES", "0") == "1":
                    print(f"[BRMoE sym qz] apply failed: {e}; y={tuple(y.shape)}, qz={tuple(self.qz.shape)}, out={tuple(out.shape)}")

        if self.bias is not None:
            out += self.bias
        if self.U != None and self.V != None:
            out = out + (x @ self.V) @ self.U
        return out


# Works with AXIS=1, group_size in {64, 128}
def patch_hqq_to_brmoe_symmetric(layer, patch_params):
    hqq_layer = None
    if isinstance(layer, HQQLinear):
        hqq_layer = layer
    elif isinstance(layer, HQQLinearLoRA):
        hqq_layer = layer.linear_layer

    if hqq_layer is None:
        return layer

    # Check config support
    if (
        (hqq_layer.meta["axis"] == 0)
        or (hqq_layer.meta["group_size"] not in (64, 128))
        or (hqq_layer.meta["nbits"] != 3)
    ):
        print("Skipping brmoe conversion for", hqq_layer)
        return layer

    z = hqq_layer.meta["zero"]
    s = hqq_layer.meta["scale"]

    W_r = hqq_layer.unpack(dtype=hqq_layer.compute_dtype)
    # Make sure shapes match
    W_r = W_r[: s.shape[0]]

    # Possibly you want a shift; for now we skip it or define it:
    z_shift = 4.0  # If you truly want the same offset as 4-bit, define it

    # This logic is somewhat different from the 4-bit approach.
    # Adjust to your real desired formula:
    W_r = (W_r - z_shift) * s
    n = hqq_layer.meta["shape"][0]
    W_r = W_r.reshape((n, -1))
    s = s.reshape((n, -1))
    z = z.reshape((n, -1))

    # If you truly need an extra 'u' term, define it similarly:
    if isinstance(z, (torch.Tensor, torch.nn.Parameter)):
        # Example usage mimicking the 3-bit style:
        qz = s * (-z + z_shift)
    
    else:
        qz = None

    if hasattr(layer, "U") and hasattr(layer, "V") and (layer.U is not None) and (layer.V is not None):
        u = layer.U.t()
        v = layer.V.t()
    else:
        u = None
        v = None
    # Build BRMoE symmetric layer with safe fallback
    try:
        brmoe_layer = BRMoE_Symmetric_Layer(
            W_r.t(),
            s.t(),
            qz.t() if (qz is not None) else None,
            u,
            v,
            bias=hqq_layer.bias,
            groupsize=hqq_layer.meta["group_size"],
        )
    except Exception as e:
        name_b = getattr(hqq_layer, 'name', 'layer')
        try:
            m_b, n_b = W_r.t().shape
        except Exception:
            m_b, n_b = None, None
        print(
            f"BRMoE (sym) build failed during pack for {name_b}: {e}. "
            f"shape=({m_b},{n_b}), axis={hqq_layer.meta.get('axis', None)}, gs={hqq_layer.meta.get('group_size', None)}. "
            "Falling back to original backend."
        )
        torch.cuda.empty_cache()
        return layer

    # Runtime probe to ensure kernels are executable on this device
    try:
        with torch.no_grad():
            x_probe = torch.zeros((1, brmoe_layer.in_features), dtype=torch.float16, device=brmoe_layer.device)
            _ = brmoe_layer.matmul(x_probe)
            torch.cuda.synchronize()
    except Exception as e:
        name_p = getattr(hqq_layer, 'name', 'layer')
        print(
            f"BRMoE (sym) probe failed for {name_p}: {e}. "
            f"shape=({brmoe_layer.in_features},{brmoe_layer.out_features}), gs={brmoe_layer.group_size}. "
            "Falling back to original backend."
        )
        del brmoe_layer
        torch.cuda.empty_cache()
        return layer

    # Only after successful probe, drop original and replace
    del hqq_layer.W_q
    del hqq_layer.meta
    del hqq_layer.bias
    del hqq_layer
    torch.cuda.empty_cache()

    if isinstance(layer, HQQLinear):
        return brmoe_layer
    if isinstance(layer, HQQLinearLoRA):
        layer.linear_layer = brmoe_layer

    return layer
