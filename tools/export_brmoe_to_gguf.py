#!/usr/bin/env python3
"""Export BR-MoE Per-3D-Tensor (Bit, Rank) Checkpoint to llama.cpp Dual-GGUF Format.

Produces:
  1. ``model-brmoe-compensator.gguf``:
     A GGUF v3 LoRA adapter file with:
       - ``general.type = "adapter"``
       - ``adapter.type = "lora"``
       - ``adapter.lora.alpha = 0.0``  (llama.cpp uses scale = alpha ? alpha/r : 1.0 -> exact 1.0!)
     Containing regular 3D compensator tensors for every MoE layer ``l`` and projection
     ``p in {gate, up, down}`` with ``r_{l,p} > 0``:
       - ``blk.{l}.ffn_{p}_exps.weight.lora_a`` : PyTorch ``[E, r, D_in]``  <-> GGML ``ne=[D_in, r, E]``
       - ``blk.{l}.ffn_{p}_exps.weight.lora_b`` : PyTorch ``[E, D_out, r]`` <-> GGML ``ne=[r, D_out, E]``
     Executed natively by llama.cpp's ``llm_graph_context::build_lora_mm_id`` via
     ``ggml_mul_mat_id`` with ZERO C++ source modifications.

  2. ``brmoe_gguf_tensor_types.json`` & ``llama_quantize_args.txt``:
     Per-3D-Tensor ``ggml_type`` assignment (``Q2_K``, ``Q3_K_M``, ``Q4_K_M``) for
     ``blk.{l}.ffn_{gate,up,down}_exps.weight``, ready for ``llama-quantize --tensor-type``
     so each 3D expert tensor launches a single fused ``ggml_mul_mat_id`` kernel.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
except ImportError:
    torch = None  # type: ignore


# ============================================================================
# Self-Contained GGUF v3 Binary Writer (zero external dependency required)
# ============================================================================
GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian
GGUF_VERSION = 3
GGUF_DEFAULT_ALIGNMENT = 32

# GGUF metadata value types
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8

# GGML tensor element types (from ggml-common.h)
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q2_K = 10
GGML_TYPE_Q3_K = 11
GGML_TYPE_Q4_K = 12

GGML_ID_TO_NAME = {
    GGML_TYPE_F32: "F32",
    GGML_TYPE_F16: "F16",
    GGML_TYPE_Q8_0: "Q8_0",
    GGML_TYPE_Q2_K: "Q2_K",
    GGML_TYPE_Q3_K: "Q3_K",
    GGML_TYPE_Q4_K: "Q4_K",
}

BIT_TO_GGML_TYPE_ID = {
    2: GGML_TYPE_Q2_K,
    3: GGML_TYPE_Q3_K,
    4: GGML_TYPE_Q4_K,
    8: GGML_TYPE_Q8_0,
}

BIT_TO_GGML_TYPE_NAME = {
    2: "Q2_K",
    3: "Q3_K_M",
    4: "Q4_K_M",
    8: "Q8_0",
}


def _align_offset(offset: int, alignment: int = GGUF_DEFAULT_ALIGNMENT) -> int:
    return (offset + alignment - 1) // alignment * alignment


def _pack_string(s: str) -> bytes:
    encoded = s.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


class MinimalGGUFWriter:
    """Spec-compliant GGUF v3 writer for 3D FP16/FP32 Compensators & K-Quant Tensors."""

    def __init__(self, arch: str = "qwen3moe"):
        self.arch = arch
        self.kv_pairs: List[Tuple[str, int, Any]] = []
        # Each entry: (name, raw_bytes_array, ggml_type, logical_torch_shape)
        self.tensors: List[Tuple[str, np.ndarray, int, Tuple[int, ...]]] = []

    def add_string(self, key: str, val: str) -> None:
        self.kv_pairs.append((key, GGUF_TYPE_STRING, str(val)))

    def add_float32(self, key: str, val: float) -> None:
        self.kv_pairs.append((key, GGUF_TYPE_FLOAT32, float(val)))

    def add_uint32(self, key: str, val: int) -> None:
        self.kv_pairs.append((key, GGUF_TYPE_UINT32, int(val)))

    def add_bool(self, key: str, val: bool) -> None:
        self.kv_pairs.append((key, GGUF_TYPE_BOOL, bool(val)))

    def add_tensor(self, name: str, array: np.ndarray) -> None:
        """Add a 2D or 3D numpy array (F16 or F32) in C-contiguous layout.

        Note on GGML dimension ordering:
          A C-contiguous PyTorch/NumPy array of shape ``(E, D1, D0)`` has innermost
          stride along ``D0``, which corresponds to GGML ``ne = [D0, D1, E]``.
        """
        arr = np.ascontiguousarray(array)
        if arr.dtype == np.float16:
            ggml_type = GGML_TYPE_F16
        elif arr.dtype == np.float32:
            ggml_type = GGML_TYPE_F32
        else:
            arr = arr.astype(np.float16)
            ggml_type = GGML_TYPE_F16
        self.tensors.append((name, arr, ggml_type, tuple(int(d) for d in arr.shape)))

    def add_kquant_tensor(
        self,
        name: str,
        logical_shape: Tuple[int, ...],
        nbits: int,
        packed_blocks: np.ndarray,
    ) -> None:
        """Add a pre-packed GGUF K-Quant tensor (`Q2_K`, `Q3_K`, `Q4_K`, `Q8_0`) with logical shape `(E, D_out, D_in)`."""
        ggml_type = BIT_TO_GGML_TYPE_ID[int(nbits)]
        raw_arr = np.ascontiguousarray(packed_blocks, dtype=np.uint8)
        self.tensors.append((name, raw_arr, ggml_type, tuple(int(d) for d in logical_shape)))

    def write(self, path: str) -> int:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        header_buf = bytearray()
        # 1. Magic, Version, n_tensors, n_kv
        header_buf += struct.pack(
            "<IIQQ", GGUF_MAGIC, GGUF_VERSION, len(self.tensors), len(self.kv_pairs)
        )

        # 2. KV Metadata
        for key, vtype, val in self.kv_pairs:
            header_buf += _pack_string(key)
            header_buf += struct.pack("<I", vtype)
            if vtype == GGUF_TYPE_STRING:
                header_buf += _pack_string(val)
            elif vtype == GGUF_TYPE_FLOAT32:
                header_buf += struct.pack("<f", val)
            elif vtype == GGUF_TYPE_UINT32:
                header_buf += struct.pack("<I", val)
            elif vtype == GGUF_TYPE_INT32:
                header_buf += struct.pack("<i", val)
            elif vtype == GGUF_TYPE_BOOL:
                header_buf += struct.pack("<?", val)
            else:
                raise ValueError(f"Unsupported GGUF KV type: {vtype}")

        # 3. Tensor Info Headers
        data_offset = 0
        tensor_aligned_offsets: List[int] = []
        for name, arr, ggml_type, logical_shape in self.tensors:
            header_buf += _pack_string(name)
            n_dims = len(logical_shape)
            header_buf += struct.pack("<I", n_dims)
            # GGML dimensions are reversed relative to C-contiguous numpy shape
            for d in reversed(logical_shape):
                header_buf += struct.pack("<Q", int(d))
            header_buf += struct.pack("<I", ggml_type)
            aligned_off = _align_offset(data_offset, GGUF_DEFAULT_ALIGNMENT)
            tensor_aligned_offsets.append(aligned_off)
            header_buf += struct.pack("<Q", aligned_off)
            data_offset = aligned_off + arr.nbytes

        # 4. Pad end of header to GGUF_DEFAULT_ALIGNMENT
        header_end = _align_offset(len(header_buf), GGUF_DEFAULT_ALIGNMENT)
        header_buf += b"\x00" * (header_end - len(header_buf))

        # 5. Write header + aligned tensor payloads
        with open(path, "wb") as f:
            f.write(header_buf)
            cur_data_pos = 0
            for (_, arr, _, _), target_off in zip(self.tensors, tensor_aligned_offsets):
                if target_off > cur_data_pos:
                    f.write(b"\x00" * (target_off - cur_data_pos))
                    cur_data_pos = target_off
                raw_bytes = arr.tobytes(order="C")
                f.write(raw_bytes)
                cur_data_pos += len(raw_bytes)

        return header_end + cur_data_pos


def read_gguf_header_summary(path: str) -> Dict[str, Any]:
    """Read back and verify a GGUF v3 file's KV pairs and tensor descriptors."""
    with open(path, "rb") as f:
        magic, version, n_tensors, n_kv = struct.unpack("<IIQQ", f.read(24))
        if magic != GGUF_MAGIC:
            raise ValueError(f"Invalid GGUF magic: {hex(magic)}")

        def read_str() -> str:
            (slen,) = struct.unpack("<Q", f.read(8))
            return f.read(slen).decode("utf-8")

        kvs: Dict[str, Any] = {}
        for _ in range(n_kv):
            k = read_str()
            (vtype,) = struct.unpack("<I", f.read(4))
            if vtype == GGUF_TYPE_STRING:
                kvs[k] = read_str()
            elif vtype == GGUF_TYPE_FLOAT32:
                (kvs[k],) = struct.unpack("<f", f.read(4))
            elif vtype == GGUF_TYPE_UINT32:
                (kvs[k],) = struct.unpack("<I", f.read(4))
            elif vtype == GGUF_TYPE_INT32:
                (kvs[k],) = struct.unpack("<i", f.read(4))
            elif vtype == GGUF_TYPE_BOOL:
                (kvs[k],) = struct.unpack("<?", f.read(1))
            else:
                raise ValueError(f"Unsupported KV type {vtype} while reading {k}")

        tensors_info: Dict[str, Dict[str, Any]] = {}
        for _ in range(n_tensors):
            tname = read_str()
            (n_dims,) = struct.unpack("<I", f.read(4))
            ne = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
            ggml_type, offset = struct.unpack("<IQ", f.read(12))
            tensors_info[tname] = {
                "n_dims": n_dims,
                "ggml_ne": ne,
                "torch_shape": list(reversed(ne)),
                "ggml_type": GGML_ID_TO_NAME.get(ggml_type, f"TYPE_{ggml_type}"),
                "ggml_type_id": ggml_type,
                "offset": offset,
            }

    return {
        "version": version,
        "n_tensors": n_tensors,
        "n_kv": n_kv,
        "kv": kvs,
        "tensors": tensors_info,
    }


# ============================================================================
# Export BR-MoE Compensators & Per-3D-Tensor Quantization Spec to GGUF
# ============================================================================
PROJ_HF_TO_GGUF = {
    "gate_proj": "gate",
    "up_proj": "up",
    "down_proj": "down",
    "w1": "gate",
    "w3": "up",
    "w2": "down",
}


def export_brmoe_checkpoint_to_gguf(
    brmoe_dir: str,
    output_dir: str,
    arch: str = "qwen3moe",
    dtype: str = "float16",
    tensor_alloc_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Export a BR-MoE directory (compensators.pt + qmodel.pt/config.json) to GGUF files."""
    os.makedirs(output_dir, exist_ok=True)
    np_dtype = np.float16 if dtype == "float16" else np.float32

    comp_path = os.path.join(brmoe_dir, "compensators.pt")
    qmodel_path = os.path.join(brmoe_dir, "qmodel.pt")
    if not os.path.exists(comp_path):
        raise FileNotFoundError(f"Missing compensators.pt in {brmoe_dir}")

    compensators = torch.load(comp_path, map_location="cpu", weights_only=False)
    q_sd = (
        torch.load(qmodel_path, map_location="cpu", weights_only=False)
        if os.path.exists(qmodel_path)
        else {}
    )

    # Group compensators by (layer_idx, gguf_proj) -> {expert_idx: (U, V)}
    grouped_comp: Dict[Tuple[int, str], Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = {}
    for layer_key, layer_dict in compensators.items():
        # Case 1: Flat module key from BaseBRMoEModel.save_compensators, e.g.
        # "model.layers.0.mlp.experts.0.gate_proj" -> (U, V) or {"U": U, "V": V}
        m_flat = re.search(r"layers\.(\d+)\..*experts\.(\d+)\.(\w+)$", str(layer_key))
        if m_flat and layer_dict is not None:
            l_idx = int(m_flat.group(1))
            e_idx = int(m_flat.group(2))
            hf_proj = m_flat.group(3)
            gguf_proj = PROJ_HF_TO_GGUF.get(hf_proj)
            if gguf_proj is not None:
                if isinstance(layer_dict, dict) and "U" in layer_dict and "V" in layer_dict:
                    u_t, v_t = layer_dict["U"], layer_dict["V"]
                elif isinstance(layer_dict, (tuple, list)) and len(layer_dict) == 2:
                    u_t, v_t = layer_dict[0], layer_dict[1]
                else:
                    u_t, v_t = None, None
                if isinstance(u_t, torch.Tensor) and isinstance(v_t, torch.Tensor):
                    grouped_comp.setdefault((l_idx, gguf_proj), {})[e_idx] = (
                        u_t.detach().float().cpu(),
                        v_t.detach().float().cpu(),
                    )
            continue

        # Case 2: Nested layer key, e.g. "model.layers.0" -> {"experts.0.gate_proj": {"U": U, "V": V}}
        m_l = re.search(r"layers\.(\d+)", str(layer_key))
        if not m_l or not isinstance(layer_dict, dict):
            continue
        l_idx = int(m_l.group(1))
        for sub_key, uv in layer_dict.items():
            m_e = re.match(r"experts\.(\d+)\.(\w+)", str(sub_key))
            if not m_e or uv is None:
                continue
            e_idx = int(m_e.group(1))
            hf_proj = m_e.group(2)
            gguf_proj = PROJ_HF_TO_GGUF.get(hf_proj)
            if gguf_proj is None:
                continue
            if isinstance(uv, dict):
                U = uv.get("U")
                V = uv.get("V")
            elif isinstance(uv, (tuple, list)) and len(uv) == 2:
                U, V = uv[0], uv[1]
            else:
                U, V = None, None
            if isinstance(U, torch.Tensor) and isinstance(V, torch.Tensor):
                grouped_comp.setdefault((l_idx, gguf_proj), {})[e_idx] = (
                    U.detach().float().cpu(),
                    V.detach().float().cpu(),
                )

    writer = MinimalGGUFWriter(arch=arch)
    writer.add_string("general.type", "adapter")
    writer.add_string("general.architecture", arch)
    writer.add_string("adapter.type", "lora")
    # CRITICAL: alpha = 0.0 makes llama.cpp's build_lora_mm_id use scale = 1.0
    # regardless of each 3D tensor's rank r_t!
    writer.add_float32("adapter.lora.alpha", 0.0)

    exported_tensors_meta: Dict[str, Any] = {}
    for (l_idx, gguf_proj), exp_map in sorted(grouped_comp.items()):
        num_experts = max(exp_map.keys()) + 1
        ranks = [int(uv[0].shape[1]) for uv in exp_map.values() if uv[0].ndim == 2]
        max_r = max(ranks) if ranks else 0
        if max_r <= 0:
            continue

        sample_u, sample_v = next(iter(exp_map.values()))
        d_out = int(sample_u.shape[0])
        d_in = int(sample_v.shape[1])

        # lora_a: PyTorch [E, r_t, D_in] -> GGML ne = [D_in, r_t, E]
        # lora_b: PyTorch [E, D_out, r_t] -> GGML ne = [r_t, D_out, E]
        lora_a = np.zeros((num_experts, max_r, d_in), dtype=np_dtype)
        lora_b = np.zeros((num_experts, d_out, max_r), dtype=np_dtype)

        for e_idx, (u_t, v_t) in exp_map.items():
            r_e = int(u_t.shape[1])
            if r_e > 0:
                lora_a[e_idx, :r_e, :] = v_t[:r_e, :].numpy().astype(np_dtype)
                lora_b[e_idx, :, :r_e] = u_t[:, :r_e].numpy().astype(np_dtype)

        base_name = f"blk.{l_idx}.ffn_{gguf_proj}_exps.weight"
        name_a = f"{base_name}.lora_a"
        name_b = f"{base_name}.lora_b"
        writer.add_tensor(name_a, lora_a)
        writer.add_tensor(name_b, lora_b)

        exported_tensors_meta[base_name] = {
            "layer": l_idx,
            "proj": gguf_proj,
            "num_experts": num_experts,
            "rank": max_r,
            "d_in": d_in,
            "d_out": d_out,
            "lora_a_torch_shape": list(lora_a.shape),
            "lora_a_ggml_ne": [d_in, max_r, num_experts],
            "lora_b_torch_shape": list(lora_b.shape),
            "lora_b_ggml_ne": [max_r, d_out, num_experts],
        }

    comp_gguf_path = os.path.join(output_dir, "model-brmoe-compensator.gguf")
    bytes_written = writer.write(comp_gguf_path)

    # Extract Per-3D-Tensor bit-widths from qmodel.pt or tensor_alloc_path
    tensor_ggml_types: Dict[str, str] = {}
    if tensor_alloc_path and os.path.exists(tensor_alloc_path):
        with open(tensor_alloc_path, "r", encoding="utf-8") as f:
            t_alloc = json.load(f)
        for gname, ginfo in t_alloc.get("gguf_tensors", {}).items():
            tensor_ggml_types[gname] = ginfo["ggml_type"]
    else:
        for k, v in q_sd.items():
            if not isinstance(v, dict) or ("bits" not in v and "nbits" not in v):
                continue
            m = re.search(r"layers\.(\d+)\..*experts\.(\d+)\.(\w+)", k)
            if not m:
                continue
            l_idx = int(m.group(1))
            hf_proj = m.group(3)
            gguf_proj = PROJ_HF_TO_GGUF.get(hf_proj)
            if gguf_proj is None:
                continue
            gname = f"blk.{l_idx}.ffn_{gguf_proj}_exps.weight"
            bit_val = int(v.get("bits", v.get("nbits", 3)))
            tensor_ggml_types[gname] = BIT_TO_GGML_TYPE_NAME.get(bit_val, "Q3_K_M")

    quant_args = []
    for gname, gtype in sorted(tensor_ggml_types.items()):
        quant_args.append(f"--tensor-type {gname}={gtype}")

    recipe_path = os.path.join(output_dir, "brmoe_gguf_tensor_types.json")
    with open(recipe_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "compensator_gguf": comp_gguf_path,
                "compensator_size_bytes": bytes_written,
                "adapter_lora_alpha": 0.0,
                "tensor_ggml_types": tensor_ggml_types,
                "compensator_3d_tensors": exported_tensors_meta,
                "llama_quantize_cli_flags": " ".join(quant_args),
            },
            f,
            indent=2,
        )

    return {
        "compensator_gguf": comp_gguf_path,
        "bytes_written": bytes_written,
        "recipe_path": recipe_path,
        "num_3d_compensator_tensors": len(writer.tensors),
        "tensor_ggml_types": tensor_ggml_types,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Export BR-MoE Per-3D-Tensor checkpoint to llama.cpp GGUF format."
    )
    parser.add_argument("--brmoe_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--arch", type=str, default="qwen3moe")
    parser.add_argument(
        "--dtype", type=str, default="float16", choices=["float16", "float32"]
    )
    parser.add_argument("--tensor_alloc_path", type=str, default=None)
    args = parser.parse_args()

    res = export_brmoe_checkpoint_to_gguf(
        brmoe_dir=args.brmoe_dir,
        output_dir=args.output_dir,
        arch=args.arch,
        dtype=args.dtype,
        tensor_alloc_path=args.tensor_alloc_path,
    )
    print(
        f"[GGUF Export] Wrote {res['num_3d_compensator_tensors']} 3D LoRA tensors "
        f"({res['bytes_written'] / 1024:.2f} KB) to {res['compensator_gguf']}"
    )


if __name__ == "__main__":
    main()
