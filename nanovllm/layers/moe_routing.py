"""Single-CTA stable expert alignment for small interactive decode batches."""
import triton
import triton.language as tl


@triton.jit
def _align_small_kernel(
    ids, sorted_ids, expert_ids, num_pad,
    NUMEL: tl.constexpr, EXPERTS: tl.constexpr, BLOCK_M: tl.constexpr,
    SORT_CAP: tl.constexpr, BLOCK_CAP: tl.constexpr,
    TOKEN_TILE: tl.constexpr, EXPERT_TILE: tl.constexpr,
    SORT_TILE: tl.constexpr, BLOCK_TILE: tl.constexpr,
    EXPERT_BLOCK_TILE: tl.constexpr,
):
    token = tl.arange(0, TOKEN_TILE)
    expert = tl.arange(0, EXPERT_TILE)
    selected = tl.load(ids + token, token < NUMEL, other=-1)
    matches = ((expert[:, None] < EXPERTS) & (token[None, :] < NUMEL)
               & (selected[None, :] == expert[:, None]))
    counts = tl.sum(matches.to(tl.int32), axis=1)
    padded = tl.cdiv(counts, BLOCK_M) * BLOCK_M
    starts = tl.cumsum(padded, axis=0) - padded
    total = tl.sum(padded, axis=0)

    fill_token = tl.arange(0, SORT_TILE)
    fill_block = tl.arange(0, BLOCK_TILE)
    tl.store(sorted_ids + fill_token, NUMEL, fill_token < SORT_CAP)
    tl.store(expert_ids + fill_block, -1, fill_block < BLOCK_CAP)
    tl.debug_barrier()

    within = tl.cumsum(matches.to(tl.int32), axis=1) - 1
    destinations = starts[:, None] + within
    tl.store(sorted_ids + destinations, token[None, :], matches)

    block = tl.arange(0, EXPERT_BLOCK_TILE)
    destinations = starts[:, None] // BLOCK_M + block[None, :]
    mask = (expert[:, None] < EXPERTS) & (block[None, :] * BLOCK_M < padded[:, None])
    tl.store(expert_ids + destinations, expert[:, None], mask)
    tl.store(num_pad, total)


def align_small_into(topk_ids, block_size, num_experts, buf):
    numel = topk_ids.numel()
    sorted_ids, experts, num_pad = buf["sorted_ids"], buf["expert_ids"], buf["num_pad"]
    _align_small_kernel[(1,)](
        topk_ids, sorted_ids, experts, num_pad,
        NUMEL=numel, EXPERTS=num_experts, BLOCK_M=block_size,
        SORT_CAP=sorted_ids.numel(), BLOCK_CAP=experts.numel(),
        TOKEN_TILE=triton.next_power_of_2(numel),
        EXPERT_TILE=triton.next_power_of_2(num_experts),
        SORT_TILE=triton.next_power_of_2(sorted_ids.numel()),
        BLOCK_TILE=triton.next_power_of_2(experts.numel()),
        EXPERT_BLOCK_TILE=triton.next_power_of_2(triton.cdiv(numel, block_size)),
        num_warps=4,
    )
    return sorted_ids, experts, num_pad
