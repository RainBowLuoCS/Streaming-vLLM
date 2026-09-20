import contextlib
import functools
import logging
import os
from collections.abc import Callable
from enum import Enum
from typing import Any, Literal

import torch
from nanovllm.layers.fla.compat import triton

logger = logging.getLogger(__name__)

COMPILER_MODE = os.getenv("FLA_COMPILER_MODE") == "1"
FLA_CI_ENV = os.getenv("FLA_CI_ENV") == "1"
FLA_GDN_FIX_BT = os.getenv("FLA_GDN_FIX_BT", "0") == "1"
SUPPRESS_LEVEL = int(os.getenv("GDN_RECOMPUTE_SUPPRESS_LEVEL", "0"))


def tensor_cache(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    cache_entries = []
    cache_size = 8

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        nonlocal cache_entries, cache_size
        for i, entry in enumerate(cache_entries):
            last_args, last_kwargs, last_result = entry
            if (len(args) == len(last_args)
                    and len(kwargs) == len(last_kwargs)
                    and all(a is b for a, b in zip(args, last_args))
                    and all(k in last_kwargs and v is last_kwargs[k] for k, v in kwargs.items())):
                cache_entries = cache_entries[:i] + cache_entries[i+1:] + [(args, kwargs, last_result)]
                return last_result
        result = fn(*args, **kwargs)
        if len(cache_entries) >= cache_size:
            cache_entries = cache_entries[1:]
        cache_entries.append((args, kwargs, result))
        return result
    return wrapper


def input_guard(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        contiguous_args = (i if not isinstance(i, torch.Tensor) else i.contiguous() for i in args)
        contiguous_kwargs = {k: (v if not isinstance(v, torch.Tensor) else v.contiguous()) for k, v in kwargs.items()}
        tensor = None
        for arg in args:
            if isinstance(arg, torch.Tensor):
                tensor = arg; break
        if tensor is None:
            for value in kwargs.values():
                if isinstance(value, torch.Tensor):
                    tensor = value; break
        if tensor is not None:
            ctx = torch.cuda.device(tensor.device.index)
        else:
            ctx = contextlib.nullcontext()
        with ctx:
            return fn(*contiguous_args, **contiguous_kwargs)
    return wrapper


@functools.cache
def get_available_device() -> str:
    try:
        return triton.runtime.driver.active.get_current_target().backend
    except (RuntimeError, AttributeError):
        return "cpu"


@functools.cache
def _check_platform() -> Literal["nvidia", "amd", "intel", "musa"]:
    device = get_available_device()
    mapping = {"cuda": "nvidia", "hip": "amd", "xpu": "intel"}
    return mapping.get(device, device)


device = "cuda" if torch.cuda.is_available() else get_available_device()
device_torch_lib = getattr(torch, device, None)
device_platform = _check_platform()

is_amd = device_platform == "amd"
is_intel = device_platform == "intel"
is_nvidia = device_platform == "nvidia"
is_intel_alchemist = False
is_nvidia_hopper = is_nvidia and (
    "NVIDIA H" in torch.cuda.get_device_name(0)
    or torch.cuda.get_device_capability()[0] >= 9
)
use_cuda_graph = is_nvidia and os.environ.get("FLA_USE_CUDA_GRAPH", "0") == "1"
is_gather_supported = hasattr(triton.language, "gather")
is_tma_supported = (is_nvidia and torch.cuda.get_device_capability(0)[0] >= 9) and (
    hasattr(triton.language, "_experimental_make_tensor_descriptor")
    or hasattr(triton.language, "make_tensor_descriptor")
)


def get_all_max_shared_mem():
    try:
        return [
            triton.runtime.driver.active.utils.get_device_properties(i)["max_shared_mem"]
            for i in range(device_torch_lib.device_count())
        ]
    except BaseException:
        return [-1]


class Backend(Enum):
    ADA = 101376
    AMPERE = 166912
    HOPPER = 232448
    DEFAULT = 102400

    @classmethod
    def get_shared_memory(cls, arch: str) -> int:
        try:
            return cls[arch.upper()].value
        except KeyError:
            return cls.DEFAULT.value


@functools.cache
def check_shared_mem(arch: str = "none", tensor_idx: int = 0) -> bool:
    try:
        device_shared_mem_list = get_all_max_shared_mem()
        max_shared_memory = device_shared_mem_list[tensor_idx]
        return max_shared_memory >= Backend.get_shared_memory(arch)
    except Exception:
        return False