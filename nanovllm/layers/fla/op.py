import os
from nanovllm.layers.fla.compat import tl, tldevice, triton
from nanovllm.layers.fla.utils import is_gather_supported

if os.environ.get("FLA_USE_FAST_OPS", "0") == "1":
    exp = tldevice.fast_expf
    log = tldevice.fast_logf
    log2 = tldevice.fast_log2f
else:
    exp = tl.exp
    log = tl.log
    log2 = tl.log2


if not is_gather_supported:

    @triton.jit
    def gather(src, index, axis, _builder=None):
        return None
else:
    gather = tl.gather

if hasattr(triton.language, "_experimental_make_tensor_descriptor"):
    make_tensor_descriptor = triton.language._experimental_make_tensor_descriptor
elif hasattr(triton.language, "make_tensor_descriptor"):
    make_tensor_descriptor = triton.language.make_tensor_descriptor
else:
    @triton.jit
    def make_tensor_descriptor(base, shape, strides, block_shape, _builder=None):
        return None