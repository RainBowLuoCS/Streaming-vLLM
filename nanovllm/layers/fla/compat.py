"""Compatibility shim: replace vllm-internal deps with native equivalents."""
import torch
import triton
import triton.language as tl

try:
    import triton.language.extra.libdevice as tldevice
except Exception:
    try:
        import triton.language.extra.cuda.libdevice as tldevice
    except Exception:
        tldevice = tl

HAS_TRITON = True
PAD_SLOT_ID = -1


def cdiv(a, b):
    return (a + b - 1) // b


def next_power_of_2(n):
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def num_compute_units(device_index=None):
    if device_index is None:
        device_index = torch.cuda.current_device()
    return torch.cuda.get_device_properties(device_index).multi_processor_count