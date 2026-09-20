import torch
import triton
import triton.language as tl


@triton.jit
def pure_move_kv_kernel(
    k_cache_ptr, v_cache_ptr,
    src_slots_ptr, dst_slots_ptr,
    num_ops,
    num_kv_heads: tl.constexpr, head_dim: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= num_ops: return
    src = tl.load(src_slots_ptr + pid)
    dst = tl.load(dst_slots_ptr + pid)
    if src < 0 or dst < 0: return
    kv_dim: tl.constexpr = num_kv_heads * head_dim
    half_dim: tl.constexpr = head_dim // 2
    for h in range(num_kv_heads):
        sb = src * kv_dim + h * head_dim
        db = dst * kv_dim + h * head_dim
        k_e = tl.load(k_cache_ptr + sb + tl.arange(0, half_dim))
        k_o = tl.load(k_cache_ptr + sb + half_dim + tl.arange(0, half_dim))
        tl.store(k_cache_ptr + db + tl.arange(0, half_dim), k_e)
        tl.store(k_cache_ptr + db + half_dim + tl.arange(0, half_dim), k_o)
        v = tl.load(v_cache_ptr + sb + tl.arange(0, head_dim))
        tl.store(v_cache_ptr + db + tl.arange(0, head_dim), v)


@triton.jit
def inplace_delta_rope_kernel(
    k_cache_ptr, cos_ptr, sin_ptr,
    slots_ptr, deltas_ptr, num_ops,
    num_kv_heads: tl.constexpr, head_dim: tl.constexpr,
    half_dim: tl.constexpr, pos_offset: tl.constexpr,
):
    """Full RoPE delta (used by Qwen3-VL: rotary_dim == head_dim)."""
    pid = tl.program_id(0)
    if pid >= num_ops: return
    slot = tl.load(slots_ptr + pid)
    if slot < 0: return
    delta = tl.load(deltas_ptr + pid)
    if delta == 0: return
    cos_sin_idx = delta + pos_offset
    kv_dim: tl.constexpr = num_kv_heads * head_dim
    cos_base = cos_sin_idx * half_dim
    cos = tl.load(cos_ptr + cos_base + tl.arange(0, half_dim))
    sin = tl.load(sin_ptr + cos_base + tl.arange(0, half_dim))
    for h in range(num_kv_heads):
        base = slot * kv_dim + h * head_dim
        k_e = tl.load(k_cache_ptr + base + tl.arange(0, half_dim))
        k_o = tl.load(k_cache_ptr + base + half_dim + tl.arange(0, half_dim))
        tl.store(k_cache_ptr + base + tl.arange(0, half_dim), k_e * cos - k_o * sin)
        tl.store(k_cache_ptr + base + half_dim + tl.arange(0, half_dim), k_o * cos + k_e * sin)


@triton.jit
def inplace_delta_rope_kernel_partial(
    k_cache_ptr, cos_ptr, sin_ptr,
    slots_ptr, deltas_ptr, num_ops,
    num_kv_heads: tl.constexpr, head_dim: tl.constexpr,
    rotary_half: tl.constexpr, pos_offset: tl.constexpr,
):
    """Partial RoPE delta (Qwen3.5): only first 2*rotary_half dims rotated.
    RoPE half-split within [0:2*rotary_half]: even=[0:rotary_half], odd=[rotary_half:2*rotary_half].
    Dims [2*rotary_half : head_dim] untouched.
    """
    pid = tl.program_id(0)
    if pid >= num_ops: return
    slot = tl.load(slots_ptr + pid)
    if slot < 0: return
    delta = tl.load(deltas_ptr + pid)
    if delta == 0: return
    cos_sin_idx = delta + pos_offset
    kv_dim: tl.constexpr = num_kv_heads * head_dim
    cos_base = cos_sin_idx * rotary_half
    cos = tl.load(cos_ptr + cos_base + tl.arange(0, rotary_half))
    sin = tl.load(sin_ptr + cos_base + tl.arange(0, rotary_half))
    for h in range(num_kv_heads):
        base = slot * kv_dim + h * head_dim
        k_e = tl.load(k_cache_ptr + base + tl.arange(0, rotary_half))
        k_o = tl.load(k_cache_ptr + base + rotary_half + tl.arange(0, rotary_half))
        tl.store(k_cache_ptr + base + tl.arange(0, rotary_half), k_e * cos - k_o * sin)
        tl.store(k_cache_ptr + base + rotary_half + tl.arange(0, rotary_half), k_o * cos + k_e * sin)


def build_rope_cache(head_dim, max_pos, rope_theta, dtype=torch.float16, rotary_dim=None):
    """rotary_dim: dims actually rotated (partial). None -> head_dim (full)."""
    if rotary_dim is None:
        rotary_dim = head_dim
    half = rotary_dim // 2
    inv = 1.0 / (rope_theta ** (torch.arange(0, half, dtype=torch.float32) / half))
    pos = torch.arange(-max_pos, max_pos)
    freqs = torch.outer(pos.float(), inv)
    return freqs.cos().to(dtype), freqs.sin().to(dtype), max_pos