import torch
import os
import marlin
from ..core.quantize import Quantizer

class MarlinLinear4bit(torch.nn.Module):
    def __init__(
        self, W: torch.Tensor, scales: torch.Tensor, u=None, bias=None, groupsize=-1, U=None, V=None
    ):
        super().__init__()

        m, n = W.shape
        device = W.device
        
        _linear = torch.nn.Linear(m, n)
        _linear.weight.data = W.half().t()

        effective_groupsize = m if (groupsize == -1) else groupsize

        _layer = marlin.Layer(m, n, groupsize=groupsize)
        _layer.k = m
        _layer.n = n
        _layer.groupsize = effective_groupsize
        
        _layer.B = torch.empty((m // 16, n * 16 // 8), dtype=torch.int, device=device)
        _layer.s = torch.empty(
            (m // effective_groupsize, n), dtype=torch.half, device=device
        )
        _layer.pack(_linear, scales.t())

        self.bias = bias.half() if (bias is not None) else None
        self.Wq_packed = _layer.B.clone()
        self.scales = _layer.s.clone()
        self.workspace_fp = torch.zeros(n // 128 * 16, device=device) 
        
        self.in_features = m
        self.out_features = n
        self.group_size = effective_groupsize
        self.axis = 1
        self.device = device
        self.compute_dtype = torch.float16
        
        self.use_padding = False
        self.original_outfeatures = n
        
        self._tile_cache = self._precompute_tiles()
        
        self.u = torch.nn.Parameter(u, requires_grad=False) if (u is not None and self._enable_zeropoint) else None
        
        # Store flag for quick check in forward pass
        self._has_zeropoint = self.u is not None

        if U is not None and V is not None:
            # Pre-transpose and validate U/V to match (m, n)
            # We want V: (in, rank), U: (rank, out)
            # So that (x @ V) @ U works
            
            # Check V
            if V.shape[0] == m: v_final = V
            elif V.shape[1] == m: v_final = V.t()
            else: raise ValueError(f"V shape {V.shape} mismatch input dim {m}")
            
            if U.shape[1] == n: u_final = U
            elif U.shape[0] == n: u_final = U.t()
            else:
                if U.shape[0] == v_final.shape[1]: u_final = U 
                else: u_final = U.t()

            self.U = torch.nn.Parameter(u_final, requires_grad=False)
            self.V = torch.nn.Parameter(v_final, requires_grad=False)
            self._has_uv = True
        else:
            self.U = None; self.V = None; self._has_uv = False

        self.name = "MarlinLinear4bit"
        
        del _linear, _layer
        torch.cuda.empty_cache()

    def _precompute_tiles(self):
        """
        Precompute optimal tile configurations for common batch sizes.
        Returns a dict mapping M ranges to (thread_k, thread_n) tuples.
        """
        # Tile candidates based on Marlin's test suite
        candidates = [
            (128, 128),  # Square tile, good for small M
            (64, 256),   # Tall tile, good for larger M
            (256, 64),   # Wide tile (less common)
        ]
        
        valid_tiles = []
        for tk, tn in candidates:
            if (self.in_features % tk == 0) and (self.out_features % tn == 0):
                valid_tiles.append((tk, tn))
        
        if not valid_tiles:
            valid_tiles = [(128, 128)]
        
        cache = {}
        for tile in valid_tiles:
            tk, tn = tile
            if tk == 128 and tn == 128:
                cache['small'] = tile  # M <= 16
            elif tk == 64 and tn == 256:
                cache['large'] = tile  # M > 16
        
        if 'small' not in cache:
            cache['small'] = valid_tiles[0]
        if 'large' not in cache:
            cache['large'] = valid_tiles[0]
        
        return cache

    @torch.no_grad()
    def matmul(self, x):
        """
        Core Marlin matmul function.
        Note: the output may be padded on the last dimension (..., n_padded).
        """
        x_flat = x.view(-1, x.shape[-1])
        
        out_flat = torch.empty(
            (x_flat.shape[0], self.scales.shape[1]), 
            dtype=x.dtype, device=x.device
        )
        
        M = x_flat.shape[0]
        thread_k, thread_n = self._tile_cache['small' if M <= 16 else 'large']
        
        marlin.mul(
            x_flat,
            self.Wq_packed,
            out_flat,
            self.scales,
            self.workspace_fp,
            thread_k,
            thread_n,
            -1  # sms (use all available SMs)
        )
        
        out = out_flat.view(x.shape[:-1] + (self.scales.shape[1],))
        return out

    @torch.jit.ignore
    def forward(self, x):
        out = self.matmul(x)
        if self._has_zeropoint:
            out.add_(torch.matmul(x.sum(dim=-1, keepdim=True), self.u))

        if self.bias is not None:
            out.add_(self.bias)
                
        if self.use_padding:
            out = out[..., :self.original_outfeatures]
        
        if self._has_uv and self._enable_uv:
            out = out + (x @ self.V) @ self.U

        return out

def patch_hqq_to_marlin(layer, patch_params):
    """
    Robust patcher for 4-bit HQQ/MiLo layers to Marlin with Padding & LoRC support.
    """
    if marlin is None: return layer
    
    if layer is None: return layer

    hqq_layer = layer.linear_layer if hasattr(layer, "linear_layer") else layer
    
    if hqq_layer is None or not hasattr(hqq_layer, 'meta'): return layer
    
    axis = hqq_layer.meta.get("axis", 1)
    gs = hqq_layer.meta.get("group_size", None)
    nbits = hqq_layer.meta.get("nbits", None)
    
    if (nbits != 4) or (axis == 0): return layer
    if gs not in (None, -1, 64, 128): return layer # Marlin limits
    
    shape = hqq_layer.meta.get("shape", None)
    if shape is None: return layer
    
    n_out_orig = int(shape[0])
    m_in = int(shape[1])
    
    if (m_in % 128) != 0: 
        print(f"[Warning] Marlin skipped {hqq_layer.name}: in_dim {m_in} % 128 != 0")
        return layer

    use_padding = False
    n_out_padded = n_out_orig
    if (n_out_orig % 256) != 0:
        n_out_padded = ((n_out_orig + 255) // 256) * 256
        use_padding = True
    
    W_q_unpacked = Quantizer.unpack[hqq_layer.meta["packing"]](
        hqq_layer.W_q, dtype=hqq_layer.compute_dtype
    )
    
    if gs not in (None, -1):
        groups = m_in // gs
        W_q_reshaped = W_q_unpacked.reshape(n_out_orig, groups, gs).reshape(n_out_orig, m_in)
    else:
        W_q_reshaped = W_q_unpacked

    if use_padding:
        pad_h = n_out_padded - n_out_orig
        W_q_reshaped = torch.nn.functional.pad(W_q_reshaped, (0, 0, 0, pad_h), value=0)

    W_r = W_q_reshaped.t() 
    
    s_hqq = hqq_layer.meta["scale"] # Shape usually: (n_out * groups, 1)
    z_hqq = hqq_layer.meta["zero"]  # Shape usually: (n_out * groups, 1)
    z_shift = 8.0 # Marlin 4-bit zero-point shift (mapping -8..7 to 0..15)

    if gs not in (None, -1):
        groups = m_in // gs
    else:
        groups = 1
        gs = m_in # Per-channel scenario

    s_reshaped = s_hqq.reshape(n_out_orig, groups)
    z_reshaped = z_hqq.reshape(n_out_orig, groups)

    if use_padding:
        pad_h = n_out_padded - n_out_orig
        s_padded = torch.nn.functional.pad(s_reshaped, (0, 0, 0, pad_h), value=1.0) # pad scale with 1.0 to keep values unchanged
        z_padded = torch.nn.functional.pad(z_reshaped, (0, 0, 0, pad_h), value=0.0) # pad zero with 0.0
    else:
        s_padded = s_reshaped
        z_padded = z_reshaped

    s_marlin = s_padded.t() # Shape: (groups, n_out_padded)

    if gs != m_in: # Grouped quantization
        s_expanded = s_padded.unsqueeze(2).expand(-1, -1, gs).reshape(n_out_padded, m_in)
        z_expanded = z_padded.unsqueeze(2).expand(-1, -1, gs).reshape(n_out_padded, m_in)
    else: # Per-channel (per-row in PyTorch convention)
        s_expanded = s_padded.expand(-1, m_in)
        z_expanded = z_padded.expand(-1, m_in)

    u = (s_expanded * (z_shift - z_expanded)).sum(dim=1, keepdim=True) 
    u = u.t() # Shape: (1, n_out_padded)

    W_r = (W_q_reshaped.to(hqq_layer.compute_dtype) - z_expanded) * s_expanded
    
    W_r = W_r.t() # Shape: (m_in, n_out_padded)

    del s_expanded, z_expanded, W_q_reshaped
    
    U = getattr(hqq_layer, 'U', None)
    V = getattr(hqq_layer, 'V', None)

    try:
        marlin_layer = MarlinLinear4bit(
            W=W_r, 
            scales=s_marlin,
            u=u,
            bias=hqq_layer.bias,
            groupsize=-1 if gs in (None, -1) else gs,
            U=U, V=V
        )
        
        marlin_layer.original_outfeatures = n_out_orig
        marlin_layer.use_padding = use_padding

        if marlin_layer.bias is not None and use_padding:
             pad_h = n_out_padded - n_out_orig
             marlin_layer.bias.data = torch.nn.functional.pad(marlin_layer.bias.data, (0, pad_h), value=0)

    except Exception as e:
        print(f"Marlin build failed: {e}")
        return layer

    try:
        dummy = torch.zeros(1, m_in, dtype=torch.float16, device=layer.device)
        _ = marlin_layer(dummy)
        torch.cuda.synchronize()
    except Exception as e:
        print(f"Marlin probe failed: {e}")
        return layer

    del hqq_layer.W_q
    if hasattr(layer, "linear_layer"):
        layer.linear_layer = marlin_layer
    else:
        return marlin_layer
        
    torch.cuda.empty_cache()
    return layer