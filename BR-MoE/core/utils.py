import gc
import math
from typing import Union
from .bitpack import BitPack
import torch

def cleanup() -> None:
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


def is_divisible(val1: int, val2: int) -> bool:
    return int(val2 * math.ceil(val1 / val2)) == val1


def zero_pad_row(
    tensor: torch.Tensor, num_rows: int, dtype: Union[torch.dtype, None] = None
) -> torch.Tensor:
    out = torch.zeros(
        [num_rows, tensor.shape[1]],
        device=tensor.device,
        dtype=tensor.dtype if (dtype is None) else dtype,
    )
    out[: len(tensor)] = tensor

    return out

def quantize_full_to_int3(tensor_in, group_size):
    tensor_in = tensor_in.reshape(-1,group_size)
    scale, _ = torch.max(tensor_in, dim=1, keepdim=True)
    tensor_int8 = torch.round(tensor_in * 7/(2*scale) ) + 4
    tensor_int8 = torch.clamp(tensor_int8,0,7).to(torch.int32)
    tensor_packed = BitPack.pack_3bit_32(tensor_int8)
    return  (scale, tensor_packed)

def quantize_full_to_int4(tensor_in, group_size):
    tensor_in = tensor_in.reshape(-1, group_size)
    max_val, _ = torch.max(tensor_in, dim=1, keepdim=True)
    min_val, _ = torch.min(tensor_in, dim=1, keepdim=True)
    max_min = max_val - min_val
    max_min[max_min == 0] = 15
    scale = 15 / max_min  
    zero = -torch.round(scale * min_val)
    tensor_int4 = torch.round(tensor_in * scale + zero).clamp(0, 15).to(torch.uint8)
    tensor_packed = BitPack.pack_4bit_u8(tensor_int4)
    return (scale, zero, tensor_packed)

def unpack_4bit_u8_sign(W_q: torch.Tensor, dtype=torch.float16) -> torch.Tensor:
    unpacked = BitPack.unpack_4bit_u8(W_q, dtype=dtype)
    return unpacked

# Map a Pytorch dtype into a safetensor dtype
def encode_safetensor_type(data):
    if isinstance(data, (torch.Tensor, torch.nn.Parameter)):
        return data
    if isinstance(data, torch.Size):
        return torch.tensor(data)
    if isinstance(data, torch.dtype):
        data = str(data)
    if isinstance(data, bool):
        return torch.tensor(int(data), dtype=torch.uint8)
    if isinstance(data, int):
        return torch.tensor(data, dtype=torch.int32)
    if isinstance(data, float):
        return torch.tensor(data, dtype=torch.float32)
    if isinstance(data, str):
        return torch.tensor([ord(i) for i in data], dtype=torch.uint8)


# Decode a safetensor dtype into a Pytorch dtype
def decode_safetensor_type(data, data_type):
    if data_type in [torch.Tensor, torch.nn.Parameter]:
        return data
    if data_type is torch.Size:
        return torch.Size(data)
    if data_type is bool:
        return bool(data.item())
    if data_type is int:
        return int(data.item())
    if data_type is float:
        return float(data.item())
    if data_type is str:
        return "".join([chr(i) for i in data])
    if data_type is torch.dtype:
        return eval("".join([chr(i) for i in data]))

def unpack_3bit_32_sign(W_q: torch.Tensor, dtype=torch.int8) -> torch.Tensor: #dtype = uint8?
        
    _step = W_q.shape[0]
    tmp = torch.empty([10 * _step, W_q.shape[1]], dtype=dtype, device=W_q.device)

    tmp[0 * _step : 1 * _step] = (W_q & 0b00111000000000000000000000000000) >> 27
    tmp[1 * _step : 2 * _step] = (W_q & 0b00000111000000000000000000000000) >> 24
    tmp[2 * _step : 3 * _step] = (W_q & 0b00000000111000000000000000000000) >> 21
    tmp[3 * _step : 4 * _step] = (W_q & 0b00000000000111000000000000000000) >> 18
    tmp[4 * _step : 5 * _step] = (W_q & 0b00000000000000111000000000000000) >> 15
    tmp[5 * _step : 6 * _step] = (W_q & 0b00000000000000000111000000000000) >> 12
    tmp[6 * _step : 7 * _step] = (W_q & 0b00000000000000000000111000000000) >> 9
    tmp[7 * _step : 8 * _step] = (W_q & 0b00000000000000000000000111000000) >> 6
    tmp[8 * _step : 9 * _step] = (W_q & 0b00000000000000000000000000111000) >> 3
    tmp[9 * _step : 10 * _step] = W_q & 0b00000000000000000000000000000111
    return tmp

def unpack_4bit_32_sign(W_q: torch.Tensor, dtype=torch.int8) -> torch.Tensor:
    _step = W_q.shape[0]
    tmp = torch.empty([8 * _step, W_q.shape[1]], dtype=dtype, device=W_q.device)

    tmp[0 * _step : 1 * _step] = (W_q & 0b11110000000000000000000000000000) >> 28
    tmp[1 * _step : 2 * _step] = (W_q & 0b00001111000000000000000000000000) >> 24
    tmp[2 * _step : 3 * _step] = (W_q & 0b00000000111100000000000000000000) >> 20
    tmp[3 * _step : 4 * _step] = (W_q & 0b00000000000011110000000000000000) >> 16
    tmp[4 * _step : 5 * _step] = (W_q & 0b00000000000000001111000000000000) >> 12
    tmp[5 * _step : 6 * _step] = (W_q & 0b00000000000000000000111100000000) >> 8
    tmp[6 * _step : 7 * _step] = (W_q & 0b00000000000000000000000011110000) >> 4
    tmp[7 * _step : 8 * _step] = W_q & 0b00000000000000000000000000001111
    return tmp

def compensator_dequantize(UV_quantized, orig_shape, rank, compensator_quantize_gs, compensator_dtype):
    if compensator_dtype == 'int3':
        zero = 4
        divisor = 7
        unpack_func = unpack_3bit_32_sign
        (U_scale, U_packed), (V_scale, V_packed) = UV_quantized
    elif compensator_dtype == 'int4':
        (U_scale, U_zero, U_packed), (V_scale, V_zero, V_packed) = UV_quantized
    else:
        raise NotImplementedError
    
    if compensator_dtype == 'int3':
        U_q = unpack_func(U_packed)
        V_q = unpack_func(V_packed)
    elif compensator_dtype == 'int4':
        U_q = BitPack.unpack_4bit_u8(U_packed, dtype=torch.float16)
        V_q = BitPack.unpack_4bit_u8(V_packed, dtype=torch.float16)
    
    # Move scales/zeros to the same devices as unpacked tensors
    if compensator_dtype == 'int3':
        U_scale = U_scale.to(U_q.device)
        V_scale = V_scale.to(V_q.device)
    elif compensator_dtype == 'int4':
        U_scale = U_scale.to(U_q.device)
        U_zero = U_zero.to(U_q.device)
        V_scale = V_scale.to(V_q.device)
        V_zero = V_zero.to(V_q.device)

    U_q = U_q[:int(orig_shape[0] * (rank / compensator_quantize_gs)),:]
    V_q = V_q[:int(orig_shape[1] * rank / compensator_quantize_gs), :]

    if compensator_dtype == 'int3':
        U_dq = ((U_q - zero) * 2 * U_scale / divisor).reshape(orig_shape[0], -1)
        V_dq = ((V_q - zero) * 2 * V_scale / divisor).reshape(-1, orig_shape[1])
    elif compensator_dtype == 'int4':
        U_dq = ((U_q - U_zero) / U_scale).reshape(orig_shape[0], -1)
        V_dq = ((V_q - V_zero) / V_scale).reshape(-1, orig_shape[1])
        
    return U_dq.half(), V_dq.half()
