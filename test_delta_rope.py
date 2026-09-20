"""
Verify Delta RoPE equivalence:
    RoPE(K_raw, new_pos) == DeltaRoPE(RoPE(K_raw, old_pos), new_pos - old_pos)

This validates that our streaming KV cache position update is mathematically correct.
"""
import torch
import torch.nn.functional as F

torch.manual_seed(42)


def build_rope_table(head_dim: int, max_pos: int, theta: float = 10000.0):
    """Build cos/sin tables for RoPE."""
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32) / half))
    positions = torch.arange(max_pos, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)  # [max_pos, half]
    return freqs.cos(), freqs.sin()


def apply_rope(k, pos, cos_table, sin_table):
    """Half-split format: matches rotary_embedding.py"""
    half = k.shape[-1] // 2
    cos = cos_table[pos]
    sin = sin_table[pos]
    k1 = k[..., :half]   # First half
    k2 = k[..., half:]   # Second half
    y1 = k1 * cos - k2 * sin
    y2 = k2 * cos + k1 * sin
    return torch.cat((y1, y2), dim=-1)


def apply_delta_rope(k_with_old_rope, delta, cos_table, sin_table, max_pos):
    """Half-split delta rotation."""
    half = k_with_old_rope.shape[-1] // 2
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half, dtype=torch.float32) / half))
    freqs = delta * inv_freq
    cos_delta = freqs.cos()
    sin_delta = freqs.sin()

    k1 = k_with_old_rope[..., :half]
    k2 = k_with_old_rope[..., half:]

    y1 = k1 * cos_delta - k2 * sin_delta
    y2 = k2 * cos_delta + k1 * sin_delta

    return torch.cat((y1, y2), dim=-1)


def test_delta_rope_equivalence():
    """Test that direct RoPE == old RoPE + delta RoPE."""
    head_dim = 128
    num_heads = 8
    max_pos = 8192
    theta = 10000.0

    cos_table, sin_table = build_rope_table(head_dim, max_pos, theta)

    # Random raw K vectors
    k_raw = torch.randn(num_heads, head_dim, dtype=torch.float32)

    test_cases = [
        # (old_pos, new_pos)
        (0, 0),          # No change
        (0, 100),        # Forward shift
        (100, 0),        # Backward shift (negative delta)
        (50, 200),       # Large forward
        (500, 100),      # Large backward
        (1000, 1001),    # Small delta
        (3000, 7000),    # Very large positions
        (42, 42),        # Same position (delta=0)
        (0, 1),          # Minimal forward
        (1, 0),          # Minimal backward
    ]

    print(f"{'='*80}")
    print(f"Delta RoPE Equivalence Test")
    print(f"head_dim={head_dim}, num_heads={num_heads}, max_pos={max_pos}")
    print(f"{'='*80}")
    print(f"{'old_pos':>8} {'new_pos':>8} {'delta':>8} {'max_abs_err':>14} {'rel_err':>14} {'PASS':>6}")
    print(f"{'-'*80}")

    all_pass = True
    for old_pos, new_pos in test_cases:
        delta = new_pos - old_pos

        # Method 1: Direct RoPE at new_pos
        k_direct = apply_rope(k_raw, new_pos, cos_table, sin_table)

        # Method 2: RoPE at old_pos, then delta RoPE
        k_old = apply_rope(k_raw, old_pos, cos_table, sin_table)
        k_delta = apply_delta_rope(k_old, delta, cos_table, sin_table, max_pos)

        # Compare
        abs_err = (k_direct - k_delta).abs().max().item()
        rel_err = (k_direct - k_delta).abs() / (k_direct.abs().clamp(min=1e-10))
        rel_err = rel_err.max().item()
        passed = abs_err < 1e-5

        print(f"{old_pos:>8} {new_pos:>8} {delta:>8} {abs_err:>14.2e} {rel_err:>14.2e} {'✓' if passed else '✗':>6}")

        if not passed:
            all_pass = False

    print(f"{'-'*80}")

    # Test with half precision (matches actual inference)
    print(f"\nHalf precision (float16) tests:")
    print(f"{'old_pos':>8} {'new_pos':>8} {'delta':>8} {'max_abs_err':>14} {'PASS':>6}")
    print(f"{'-'*80}")

    k_raw_fp16 = k_raw.half()
    cos_fp16, sin_fp16 = cos_table.half(), sin_table.half()

    for old_pos, new_pos in test_cases:
        delta = new_pos - old_pos

        k_direct = apply_rope(k_raw_fp16.float(), new_pos, cos_table, sin_table).half()
        k_old = apply_rope(k_raw_fp16.float(), old_pos, cos_table, sin_table).half()
        k_delta = apply_delta_rope(k_old.float(), delta, cos_table, sin_table, max_pos).half()

        abs_err = (k_direct.float() - k_delta.float()).abs().max().item()
        passed = abs_err < 1e-2  # Relaxed for fp16

        print(f"{old_pos:>8} {new_pos:>8} {delta:>8} {abs_err:>14.2e} {'✓' if passed else '✗':>6}")

        if not passed:
            all_pass = False

    print(f"{'-'*80}")

    # Test chained delta RoPE (multiple position updates)
    print(f"\nChained delta RoPE test (pos: 10 → 50 → 200 → 1000):")
    k_chain = apply_rope(k_raw, 10, cos_table, sin_table)
    k_chain = apply_delta_rope(k_chain, 40, cos_table, sin_table, max_pos)    # 10→50
    k_chain = apply_delta_rope(k_chain, 150, cos_table, sin_table, max_pos)   # 50→200
    k_chain = apply_delta_rope(k_chain, 800, cos_table, sin_table, max_pos)   # 200→1000

    k_direct_1000 = apply_rope(k_raw, 1000, cos_table, sin_table)
    chain_err = (k_direct_1000 - k_chain).abs().max().item()
    print(f"  Max abs error after 3 chained updates: {chain_err:.2e} {'✓' if chain_err < 1e-4 else '✗'}")

    # Test batch consistency
    print(f"\nBatch test (100 random position pairs):")
    torch.manual_seed(123)
    errors = []
    for _ in range(100):
        old_p = torch.randint(0, 4000, (1,)).item()
        new_p = torch.randint(0, 4000, (1,)).item()
        d = new_p - old_p

        k_d = apply_rope(k_raw, new_p, cos_table, sin_table)
        k_o = apply_rope(k_raw, old_p, cos_table, sin_table)
        k_u = apply_delta_rope(k_o, d, cos_table, sin_table, max_pos)
        errors.append((k_d - k_u).abs().max().item())

    errors = torch.tensor(errors)
    print(f"  Max error:  {errors.max().item():.2e}")
    print(f"  Mean error: {errors.mean().item():.2e}")
    print(f"  All < 1e-5: {'✓' if errors.max().item() < 1e-5 else '✗'}")

    print(f"\n{'='*80}")
    print(f"Overall: {'ALL TESTS PASSED ✓' if all_pass else 'SOME TESTS FAILED ✗'}")
    print(f"{'='*80}")


def test_triton_kernel_if_available():
    """Test actual Triton kernel if available."""
    try:
        from nanovllm.layers.streaming_rope import move_delta_rope_kernel, build_rope_cache
    except ImportError:
        print("\nTriton kernel not available, skipping GPU test.")
        return

    if not torch.cuda.is_available():
        print("\nCUDA not available, skipping GPU test.")
        return

    print(f"\n{'='*80}")
    print("Triton Kernel Verification")
    print(f"{'='*80}")

    head_dim = 128
    num_kv_heads = 8
    half_dim = head_dim // 2
    num_slots = 10
    max_pos = 4096
    rope_theta = 10000.0

    # Build RoPE tables
    cos_table, sin_table = build_rope_table(head_dim, max_pos, rope_theta)

    # Build GPU tables (with negative position support)
    cos_gpu, sin_gpu, pos_offset = build_rope_cache(head_dim, max_pos, rope_theta)
    cos_gpu = cos_gpu.cuda()
    sin_gpu = sin_gpu.cuda()

    # Random K and V cache
    k_raw = torch.randn(num_slots, num_kv_heads, head_dim, dtype=torch.float32)

    # Apply RoPE at old positions
    old_positions = [10, 50, 100, 200, 500, 1000, 1500, 2000, 2500, 3000]
    new_positions = [5, 100, 50, 300, 200, 1500, 1000, 2500, 2000, 3500]

    k_with_old_rope = torch.zeros_like(k_raw)
    for i, pos in enumerate(old_positions):
        k_with_old_rope[i] = apply_rope(k_raw[i], pos, cos_table, sin_table)

    # Expected: direct RoPE at new positions
    k_expected = torch.zeros_like(k_raw)
    for i, pos in enumerate(new_positions):
        k_expected[i] = apply_rope(k_raw[i], pos, cos_table, sin_table)

    # Use Triton kernel to do delta update
    kv_dim = num_kv_heads * head_dim
    k_cache = k_with_old_rope.view(num_slots, kv_dim).half().cuda().clone()
    v_cache = torch.randn(num_slots, kv_dim, dtype=torch.float16, device='cuda')

    k_dst = k_cache.clone()
    v_dst = v_cache.clone()

    src_slots = torch.arange(num_slots, dtype=torch.int32, device='cuda')
    dst_slots = torch.arange(num_slots, dtype=torch.int32, device='cuda')
    old_pos = torch.tensor(old_positions, dtype=torch.int32, device='cuda')
    new_pos = torch.tensor(new_positions, dtype=torch.int32, device='cuda')

    move_delta_rope_kernel[(num_slots,)](
        k_dst, v_dst,
        cos_gpu, sin_gpu,
        src_slots, dst_slots,
        old_pos, new_pos,
        num_slots,
        num_kv_heads, head_dim, half_dim, pos_offset,
    )

    # Compare
    k_result = k_dst.view(num_slots, num_kv_heads, head_dim).float().cpu()
    k_expected_fp16 = k_expected.half().float()

    abs_err = (k_result - k_expected_fp16).abs().max().item()
    print(f"Triton kernel max abs error: {abs_err:.2e}")
    print(f"Test: {'PASSED ✓' if abs_err < 1e-2 else 'FAILED ✗'}")

    # Per-slot errors
    print(f"\nPer-slot errors:")
    for i in range(num_slots):
        err = (k_result[i] - k_expected_fp16[i]).abs().max().item()
        print(f"  Slot {i}: old_pos={old_positions[i]:>5} → new_pos={new_positions[i]:>5} "
              f"delta={new_positions[i]-old_positions[i]:>6} err={err:.2e}")


if __name__ == "__main__":
    test_delta_rope_equivalence()
    test_triton_kernel_if_available()
