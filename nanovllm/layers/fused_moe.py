"""Fused MoE with Expert Parallel (EP) + Triton kernel.

Supports two modes:
  - eager mode (_MOE_CAPTURE_MODE=False): allocates intermediate buffers per call.
  - capture mode (_MOE_CAPTURE_MODE=True): uses persistent fixed-address buffers
    (required for CUDAGraph capture/replay).
"""
from __future__ import annotations
import os
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import triton
import triton.language as tl

from nanovllm.utils.context import get_pstate
from nanovllm.layers.moe_routing import align_small_into

_USE_SMALL_ROUTING = os.environ.get("NANOVLLM_MOE_SMALL_ROUTING", "0") == "1"


# =====================================================================
# Triton kernels
# =====================================================================

@triton.jit
def fused_moe_kernel(
    a_ptr, b_ptr, c_ptr,
    topk_weights_ptr, sorted_token_ids_ptr, expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N, K, EM, num_valid_tokens,
    stride_am, stride_ak,
    stride_be, stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr, A_DIV: tl.constexpr, compute_type: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m)
    if off_experts == -1:
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
        tl.store(c_ptrs, tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type), mask=c_mask)
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_token[:, None] // A_DIV * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + off_experts * stride_be + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K), other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        accumulator = accumulator * moe_weight[:, None]
    accumulator = accumulator.to(compute_type)

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def silu_and_mul_kernel(out_ptr, in_ptr, stride_om, stride_im, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    mask = col < N
    gate = tl.load(in_ptr + pid * stride_im + col, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(in_ptr + pid * stride_im + N + col, mask=mask, other=0.0).to(tl.float32)
    res = gate * tl.sigmoid(gate) * up
    tl.store(out_ptr + pid * stride_om + col, res.to(out_ptr.dtype.element_ty), mask=mask)


def silu_and_mul(out, x):
    M, two_n = x.shape
    N = two_n // 2
    BLOCK = triton.next_power_of_2(N)
    silu_and_mul_kernel[(M,)](out, x, out.stride(0), x.stride(0), N, BLOCK=BLOCK)


# =====================================================================
# Graph-capture buffer pool
# =====================================================================

_MOE_BUFFERS = {}
_MOE_CAPTURE_MODE = False


def set_moe_capture_mode(enabled: bool):
    global _MOE_CAPTURE_MODE
    _MOE_CAPTURE_MODE = enabled


def _get_moe_buffers(M, top_k, two_n, N, K, num_local_experts, block_size, device, dtype):
    key = (M, top_k, two_n, N, K, num_local_experts, block_size)
    if key not in _MOE_BUFFERS:
        numel = M * top_k
        max_padded = numel + num_local_experts * (block_size - 1)
        max_blocks = (max_padded + block_size - 1) // block_size
        _MOE_BUFFERS[key] = dict(
            inter1=torch.empty((numel, two_n), device=device, dtype=dtype),
            inter2=torch.empty((numel, N), device=device, dtype=dtype),
            out=torch.empty((numel, K), device=device, dtype=dtype),
            sorted_ids=torch.empty(max_blocks * block_size, dtype=torch.int32, device=device),
            expert_ids=torch.empty(max_blocks, dtype=torch.int32, device=device),
            num_pad=torch.empty(1, dtype=torch.int32, device=device),
            counts=torch.empty(num_local_experts, dtype=torch.int32, device=device),
            cumsum=torch.empty(num_local_experts + 1, dtype=torch.int32, device=device),
            block_start=torch.arange(max_blocks, device=device, dtype=torch.int32) * block_size,
            prefix_zero=torch.zeros(1, dtype=torch.int32, device=device),
        )
    return _MOE_BUFFERS[key]


# =====================================================================
# Host-side alignment
# =====================================================================

def moe_align_block_size(topk_ids, block_size, num_experts):
    """Eager alignment: allocates outputs. num_tokens_post_pad is a GPU tensor."""
    M, top_k = topk_ids.shape
    numel = M * top_k
    device = topk_ids.device
    max_num_tokens_padded = numel + num_experts * (block_size - 1)
    max_num_blocks = (max_num_tokens_padded + block_size - 1) // block_size

    if _USE_SMALL_ROUTING and numel <= 64:
        buf = {
            "sorted_ids": torch.empty(max_num_blocks * block_size, dtype=torch.int32, device=device),
            "expert_ids": torch.empty(max_num_blocks, dtype=torch.int32, device=device),
            "num_pad": torch.empty(1, dtype=torch.int32, device=device),
        }
        return align_small_into(topk_ids, block_size, num_experts, buf)

    flat = topk_ids.flatten()
    valid = flat >= 0
    safe = torch.where(valid, flat, torch.zeros_like(flat)).to(torch.int64)
    vmask = valid.to(torch.int32)

    counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
    counts.scatter_add_(0, safe, vmask)

    padded = ((counts + block_size - 1) // block_size) * block_size
    cumsum = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)
    cumsum[1:] = torch.cumsum(padded, 0)
    num_tokens_post_pad = cumsum[-1:].clone()

    sorted_token_ids = torch.full((max_num_blocks * block_size,), numel, dtype=torch.int32, device=device)
    expert_ids = torch.full((max_num_blocks,), -1, dtype=torch.int32, device=device)

    block_start = torch.arange(max_num_blocks, device=device, dtype=torch.int32) * block_size
    exp_of_block = (torch.searchsorted(cumsum, block_start, right=True) - 1).clamp(0, num_experts - 1)
    valid_block = block_start < num_tokens_post_pad
    expert_ids = torch.where(valid_block, exp_of_block.to(torch.int32), expert_ids)

    key = torch.where(valid, flat, torch.full_like(flat, num_experts))
    order = torch.argsort(key, stable=True)
    sorted_experts = key[order]
    se_clamp = sorted_experts.clamp(max=num_experts - 1)
    prefix = torch.cumsum(
        torch.cat([torch.zeros(1, dtype=torch.int32, device=device), counts[:-1]]), 0)
    within = torch.arange(numel, device=device, dtype=torch.int32) - prefix[se_clamp]
    dest = cumsum[se_clamp] + within
    write_mask = sorted_experts < num_experts
    dest_safe = torch.where(write_mask, dest, torch.full_like(dest, max_num_blocks * block_size - 1))
    vals = torch.where(write_mask, order.to(torch.int32), torch.full_like(order, numel, dtype=torch.int32))
    sorted_token_ids.scatter_(0, dest_safe.to(torch.int64), vals)

    return sorted_token_ids, expert_ids, num_tokens_post_pad


def moe_align_block_size_into(topk_ids, block_size, num_experts, buf):
    """Graph-safe alignment: writes into pre-allocated fixed-address buffers."""
    if _USE_SMALL_ROUTING and topk_ids.numel() <= 64:
        return align_small_into(topk_ids, block_size, num_experts, buf)
    M, top_k = topk_ids.shape
    numel = M * top_k
    device = topk_ids.device

    sorted_token_ids = buf["sorted_ids"]
    expert_ids = buf["expert_ids"]
    num_pad = buf["num_pad"]
    counts = buf["counts"]
    cumsum = buf["cumsum"]
    block_start = buf["block_start"]
    prefix_zero = buf["prefix_zero"]
    max_blocks = expert_ids.shape[0]

    sorted_token_ids.fill_(numel)
    expert_ids.fill_(-1)

    flat = topk_ids.flatten()
    valid = flat >= 0
    safe = torch.where(valid, flat, torch.zeros_like(flat)).to(torch.int64)
    vmask = valid.to(torch.int32)

    counts.zero_()
    counts.scatter_add_(0, safe, vmask)

    padded = ((counts + block_size - 1) // block_size) * block_size
    cumsum.zero_()
    cumsum[1:] = torch.cumsum(padded, 0)
    num_pad.copy_(cumsum[-1:])

    exp_of_block = (torch.searchsorted(cumsum, block_start, right=True) - 1).clamp(0, num_experts - 1)
    valid_block = block_start < num_pad
    expert_ids.copy_(torch.where(valid_block, exp_of_block.to(torch.int32),
                                 torch.full_like(exp_of_block, -1, dtype=torch.int32)))

    key = torch.where(valid, flat, torch.full_like(flat, num_experts))
    order = torch.argsort(key, stable=True)
    sorted_experts = key[order]
    se_clamp = sorted_experts.clamp(max=num_experts - 1)
    prefix = torch.cumsum(torch.cat([prefix_zero, counts[:-1]]), 0)
    within = torch.arange(numel, device=device, dtype=torch.int32) - prefix[se_clamp]
    dest = cumsum[se_clamp] + within
    write_mask = sorted_experts < num_experts
    dest_safe = torch.where(write_mask, dest, torch.full_like(dest, max_blocks * block_size - 1))
    vals = torch.where(write_mask, order.to(torch.int32), torch.full_like(order, numel, dtype=torch.int32))
    sorted_token_ids.scatter_(0, dest_safe.to(torch.int64), vals)

    return sorted_token_ids, expert_ids, num_pad


# def _moe_config(M):
#     if M <= 32:
#         return dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64, GROUP_SIZE_M=1, num_warps=4, num_stages=3)
#     elif M <= 128:
#         return dict(BLOCK_SIZE_M=32, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64, GROUP_SIZE_M=4, num_warps=4, num_stages=3)
#     else:
#         return dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64, GROUP_SIZE_M=8, num_warps=8, num_stages=4)

def _moe_config(M):
    if M <= 32:
        return dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=128, BLOCK_SIZE_K=128, GROUP_SIZE_M=1, num_warps=4, num_stages=4)
    elif M <= 128:
        return dict(BLOCK_SIZE_M=32, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64, GROUP_SIZE_M=4, num_warps=4, num_stages=3)
    else:
        return dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64, GROUP_SIZE_M=8, num_warps=8, num_stages=4)
        
def invoke_fused_moe(a, b, c, topk_weights, sorted_token_ids, expert_ids,
                     num_tokens_post_padded, mul_routed_weight, top_k, config, a_div):
    EM = sorted_token_ids.shape[0]
    N = b.shape[1]
    K = a.shape[1]
    num_valid_tokens = topk_weights.numel()
    grid = lambda META: (triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)
    compute_type = tl.bfloat16 if c.dtype == torch.bfloat16 else tl.float16
    fused_moe_kernel[grid](
        a, b, c, topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded,
        N, K, EM, num_valid_tokens,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(2), b.stride(1),
        c.stride(0), c.stride(1),
        MUL_ROUTED_WEIGHT=mul_routed_weight, top_k=top_k, A_DIV=a_div,
        compute_type=compute_type, **config,
    )


def fused_experts(hidden_states, w1, w2, topk_weights, topk_ids, num_local_experts):
    M, K = hidden_states.shape
    E_local, two_n, _ = w1.shape
    N = two_n // 2
    top_k = topk_ids.shape[1]

    config = _moe_config(M)
    block_m = config["BLOCK_SIZE_M"]

    if _MOE_CAPTURE_MODE:
        buf = _get_moe_buffers(M, top_k, two_n, N, K, num_local_experts, block_m,
                               hidden_states.device, hidden_states.dtype)
        sorted_token_ids, expert_ids, num_tokens_post_pad = \
            moe_align_block_size_into(topk_ids, block_m, num_local_experts, buf)
        inter1 = buf["inter1"]
        inter2 = buf["inter2"]
        out = buf["out"]
        out.zero_()
    else:
        sorted_token_ids, expert_ids, num_tokens_post_pad = \
            moe_align_block_size(topk_ids, block_m, num_local_experts)
        inter1 = torch.empty((M * top_k, two_n), device=hidden_states.device, dtype=hidden_states.dtype)
        inter2 = torch.empty((M * top_k, N), device=hidden_states.device, dtype=hidden_states.dtype)
        out = torch.zeros((M * top_k, K), device=hidden_states.device, dtype=hidden_states.dtype)

    invoke_fused_moe(hidden_states, w1, inter1, topk_weights, sorted_token_ids, expert_ids,
                     num_tokens_post_pad, False, top_k, config, top_k)
    silu_and_mul(inter2, inter1)
    invoke_fused_moe(inter2, w2, out, topk_weights, sorted_token_ids, expert_ids,
                     num_tokens_post_pad, True, top_k, config, 1)

    return out.view(M, top_k, K).sum(dim=1)


# =====================================================================
# FusedMoE Module
# =====================================================================

class FusedMoE(nn.Module):

    def __init__(self, num_experts, top_k, hidden_size, intermediate_size, renormalize=True):
        super().__init__()
        ps = get_pstate()
        self.ep_size = ps.ep_size
        self.ep_rank = ps.ep_rank
        self.ep_group = ps.ep_group
        self.global_num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.renormalize = renormalize

        assert num_experts % self.ep_size == 0, \
            f"num_experts({num_experts}) must be divisible by ep_size({self.ep_size})"
        self.num_local_experts = num_experts // self.ep_size
        self.expert_start = self.ep_rank * self.num_local_experts
        self.expert_end = self.expert_start + self.num_local_experts

        self.w13_weight = nn.Parameter(torch.empty(
            self.num_local_experts, 2 * intermediate_size, hidden_size))
        self.w2_weight = nn.Parameter(torch.empty(
            self.num_local_experts, hidden_size, intermediate_size))
        self.w13_weight.weight_loader = self.weight_loader
        self.w2_weight.weight_loader = self.weight_loader

    def weight_loader(self, param, loaded_weight, weight_name, shard_id, expert_id, return_success=False):
        if not (self.expert_start <= expert_id < self.expert_end):
            return False if return_success else None
        local_eid = expert_id - self.expert_start
        I = self.intermediate_size
        data = param.data
        if shard_id in ("w1", "w3"):
            offset = 0 if shard_id == "w1" else I
            data[local_eid, offset:offset + I, :].copy_(loaded_weight)
        elif shard_id == "w2":
            data[local_eid, :, :].copy_(loaded_weight)
        else:
            raise ValueError(shard_id)
        return True if return_success else None

    def _route(self, router_logits):
        rw = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        topk_weights, topk_ids = torch.topk(rw, self.top_k, dim=-1)
        if self.renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        return topk_weights.to(router_logits.dtype), topk_ids.to(torch.int64)

    def forward(self, hidden_states, router_logits):
        topk_weights, topk_ids_global = self._route(router_logits)

        in_range = (topk_ids_global >= self.expert_start) & (topk_ids_global < self.expert_end)
        topk_ids_local = torch.where(
            in_range, topk_ids_global - self.expert_start,
            torch.full_like(topk_ids_global, -1),
        )

        out = fused_experts(
            hidden_states, self.w13_weight, self.w2_weight,
            topk_weights.contiguous(), topk_ids_local.to(torch.int32).contiguous(),
            self.num_local_experts,
        )

        if self.ep_size > 1:
            dist.all_reduce(out, group=self.ep_group)
        return out
