"""Qwen3-VL Vision Encoder with full TP support (Aligned with vLLM)."""
from __future__ import annotations
from typing import Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import einops
import numpy as np

from streamingvllm.layers.conv import Conv3dLinear
from streamingvllm.layers.linear import QKVParallelLinear, RowParallelLinear, ColumnParallelLinear

# =====================================================================
# Pure Python Replacements for vLLM Custom Ops
# =====================================================================

def vit_flash_attn_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,  # 传入 int，避免 .item() 阻碍 torch.compile
    batch_size: int,
) -> torch.Tensor:
    """
    Pure Python implementation of vLLM's flash_attn_maxseqlen_wrapper.
    q, k, v: [batch_size, seq_len, num_heads, head_dim]
    """
    from flash_attn import flash_attn_varlen_func
    
    # [b, s, h, d] -> [b * s, h, d]
    q, k, v = (einops.rearrange(x, "b s ... -> (b s) ...") for x in [q, k, v])
    
    output = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        dropout_p=0.0,
        causal=False,
    )
    
    # [b * s, h, d] -> [s, b, h * d]
    context_layer = einops.rearrange(
        output, "(b s) h d -> s b (h d)", b=batch_size
    ).contiguous()
    
    return context_layer


def vit_torch_sdpa_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    """
    Pure Python implementation of vLLM's torch_sdpa_wrapper.
    q, k, v: [batch_size, seq_len, num_heads, head_dim]
    """
    outputs = []
    # 如果处于 compile 模式，cu_seqlens.tolist() 会报错，这里提供一个简单的降级方案
    # 但在我们的 StreamingLLM 中，Vision Encoder 通常不开启 torch.compile
    cu_seqlens_list = cu_seqlens.tolist()
    
    for i in range(1, len(cu_seqlens_list)):
        start_idx = cu_seqlens_list[i - 1]
        end_idx = cu_seqlens_list[i]
        
        if start_idx == end_idx:
            continue
            
        q_i = q[:, start_idx:end_idx]
        k_i = k[:, start_idx:end_idx]
        v_i = v[:, start_idx:end_idx]
        
        q_i, k_i, v_i = (
            einops.rearrange(x, "b s h d -> b h s d") for x in [q_i, k_i, v_i]
        )
        
        output_i = F.scaled_dot_product_attention(q_i, k_i, v_i, dropout_p=0.0)
        output_i = einops.rearrange(output_i, "b h s d -> b s h d ")
        outputs.append(output_i)
        
    if not outputs:
        # Fallback for empty sequences
        b, s, h, d = q.shape
        return torch.empty((s, b, h * d), dtype=q.dtype, device=q.device)
        
    context_layer = torch.cat(outputs, dim=1)
    context_layer = einops.rearrange(context_layer, "b s h d -> s b (h d)").contiguous()
    return context_layer

# =====================================================================
# Qwen3-VL Vision Components
# =====================================================================

class VisionPatchEmbed(nn.Module):
    def __init__(self, patch_size, temporal_patch_size, in_channels, hidden_size):
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.hidden_size = hidden_size
        self.in_channels = in_channels
        kernel = (temporal_patch_size, patch_size, patch_size)
        self.proj = Conv3dLinear(in_channels, hidden_size, kernel_size=kernel, bias=True)

    def forward(self, x):
        if x.dim() == 2:
            return self.proj(x)
        elif x.dim() == 5:
            out = self.proj(x)
            return out.flatten(2).transpose(1, 2).reshape(-1, self.hidden_size)
        else:
            out = self.proj(x)
            return out.reshape(-1, self.hidden_size)


class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim, theta=10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen):
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        return torch.outer(seq, self.inv_freq)


def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(t, cos, sin):
    return (t.float() * cos.float() + rotate_half(t.float()) * sin.float()).to(t.dtype)


class VisionAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        tp_size = dist.get_world_size()
        self.num_heads_per_tp = num_heads // tp_size
        self.tp_hidden = self.num_heads_per_tp * self.head_dim
        
        self.qkv = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=num_heads,
            total_num_kv_heads=num_heads,
            bias=True,
        )
        self.proj = RowParallelLinear(hidden_size, hidden_size, bias=True)

        # 缓存后端选择
        try:
            import flash_attn
            self.backend = "flash_attn"
        except ImportError:
            self.backend = "sdpa"

    def forward(self, x, cu_seqlens, cos, sin, max_seqlen):
        seq_len = x.shape[0]
        batch_size = 1
        
        x_unsqueezed = x.unsqueeze(1)
        qkv_out = self.qkv(x_unsqueezed)
        
        qkv = einops.rearrange(
            qkv_out,
            "s b (three head head_dim) -> b s three head head_dim",
            three=3,
            head=self.num_heads_per_tp,
        )
        
        if cos is not None and sin is not None:
            cos_unsq = cos.unsqueeze(0)
            sin_unsq = sin.unsqueeze(0)
            
            qk, v = qkv[:, :, :2], qkv[:, :, 2]
            
            qk_reshaped = einops.rearrange(
                qk, "b s two head head_dim -> (two b) s head head_dim", two=2
            )
            
            cos_expanded = cos_unsq.expand(2, -1, -1).unsqueeze(-2)
            sin_expanded = sin_unsq.expand(2, -1, -1).unsqueeze(-2)
            
            qk_rotated = apply_rotary_pos_emb_vision(
                qk_reshaped, cos=cos_expanded, sin=sin_expanded
            )
            
            qk_rotated = qk_rotated.view(
                2, batch_size, seq_len, self.num_heads_per_tp, self.head_dim
            )
            q, k = qk_rotated.unbind(dim=0)
        else:
            q, k, v = qkv.unbind(dim=2)
            
        # 🚨 保证连续性 🚨
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        max_sl = max_seqlen if isinstance(max_seqlen, int) else max_seqlen.item()

        if self.backend == "flash_attn":
            context_layer = vit_flash_attn_wrapper(
                q, k, v, 
                cu_seqlens, 
                max_sl, 
                batch_size
            )
        else:
            context_layer = vit_torch_sdpa_wrapper(
                q, k, v, cu_seqlens
            )


        # context_layer is [seq_len, batch, tp_hidden]
        context_layer = context_layer.squeeze(1)
        return self.proj(context_layer)


class VisionMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.linear_fc1 = ColumnParallelLinear(hidden_size, intermediate_size, bias=True)
        self.linear_fc2 = RowParallelLinear(intermediate_size, hidden_size, bias=True)

    def forward(self, x):
        return self.linear_fc2(F.gelu(self.linear_fc1(x), approximate="tanh"))


class VisionPatchMerger(nn.Module):
    def __init__(self, config, use_postshuffle_norm=False):
        super().__init__()
        self.hidden_dim = config.hidden_size * (config.spatial_merge_size ** 2)
        self.use_postshuffle_norm = use_postshuffle_norm
        norm_dim = self.hidden_dim if use_postshuffle_norm else config.hidden_size
        self.norm = nn.LayerNorm(norm_dim, eps=1e-6)
        self.linear_fc1 = ColumnParallelLinear(self.hidden_dim, self.hidden_dim, bias=True)
        self.linear_fc2 = RowParallelLinear(self.hidden_dim, config.out_hidden_size, bias=True)
        self.act = nn.GELU()

    def forward(self, x):
        if self.use_postshuffle_norm:
            x = self.norm(x.view(-1, self.hidden_dim))
        else:
            x = self.norm(x).view(-1, self.hidden_dim)
        return self.linear_fc2(self.act(self.linear_fc1(x)))


class VisionBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = VisionAttention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.mlp = VisionMLP(hidden_size, intermediate_size)

    def forward(self, x, cu_seqlens, cos, sin, max_seqlen):
        x = x + self.attn(self.norm1(x), cu_seqlens, cos, sin, max_seqlen)
        x = x + self.mlp(self.norm2(x))
        return x


class Qwen3VLVisionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.spatial_merge_size = config.spatial_merge_size
        self.deepstack_visual_indexes = config.deepstack_visual_indexes

        kernel = (config.temporal_patch_size, config.patch_size, config.patch_size)
        self.patch_embed = VisionPatchEmbed(
            config.patch_size, config.temporal_patch_size,
            config.in_channels, config.hidden_size,
        )
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.num_grid_per_side = int(config.num_position_embeddings ** 0.5)

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList([
            VisionBlock(config.hidden_size, config.num_heads, config.intermediate_size)
            for _ in range(config.depth)
        ])
        self.merger = VisionPatchMerger(config, use_postshuffle_norm=False)
        self.deepstack_merger_list = nn.ModuleList([
            VisionPatchMerger(config, use_postshuffle_norm=True)
            for _ in range(len(config.deepstack_visual_indexes))
        ])

    @property
    def dtype(self): return self.pos_embed.weight.dtype
    @property
    def device(self): return self.pos_embed.weight.device

    def rot_pos_emb(self, grid_thw_list):
        merge = self.spatial_merge_size
        max_hw = max(max(h, w) for _, h, w in grid_thw_list)
        freq_table = self.rotary_pos_emb(max_hw)
        device = freq_table.device
        all_pos = []
        for t, h, w in grid_thw_list:
            mh, mw = h // merge, w // merge
            rows = torch.arange(mh, device=device)
            cols = torch.arange(mw, device=device)
            ir = torch.arange(merge, device=device)
            ic = torch.arange(merge, device=device)
            ri = (rows[:, None, None, None] * merge + ir[None, None, :, None]).expand(mh, mw, merge, merge).reshape(-1)
            ci = (cols[None, :, None, None] * merge + ic[None, None, None, :]).expand(mh, mw, merge, merge).reshape(-1)
            coords = torch.stack([ri, ci], dim=-1)
            if t > 1: coords = coords.repeat(t, 1)
            all_pos.append(coords)
        pos_ids = torch.cat(all_pos, dim=0)
        embeddings = freq_table[pos_ids].flatten(1)
        emb = torch.cat([embeddings, embeddings], dim=-1)
        return emb.cos(), emb.sin()

    def fast_pos_embed_interpolate(self, grid_thw_list):
        device = self.device
        merge = self.spatial_merge_size
        n = self.num_grid_per_side
        outputs = []
        for t, h, w in grid_thw_list:
            hi = torch.linspace(0, n - 1, h, device=device)
            wi = torch.linspace(0, n - 1, w, device=device)
            hf, wf = hi.long(), wi.long()
            hc = (hf + 1).clamp(max=n - 1); wc = (wf + 1).clamp(max=n - 1)
            dh, dw = hi - hf, wi - wf
            w11 = dh[:, None] * dw[None, :]
            w10 = dh[:, None] - w11; w01 = dw[None, :] - w11; w00 = 1 - dh[:, None] - w01
            indices = torch.stack([
                (hf[:, None] * n + wf[None, :]).flatten(),
                (hf[:, None] * n + wc[None, :]).flatten(),
                (hc[:, None] * n + wf[None, :]).flatten(),
                (hc[:, None] * n + wc[None, :]).flatten(),
            ])
            weights = torch.stack([w00.flatten(), w01.flatten(), w10.flatten(), w11.flatten()])
            weights = weights.unsqueeze(-1).to(self.dtype)
            embeds = (self.pos_embed(indices) * weights).sum(0)
            embeds = embeds.view(h // merge, merge, w // merge, merge, -1)
            embeds = embeds.permute(0, 2, 1, 3, 4).flatten(0, 3)
            if t > 1: embeds = embeds.repeat(t, 1)
            outputs.append(embeds)
        return torch.cat(outputs, dim=0)

    def forward(self, pixel_values, grid_thw):
        grid_thw_list = grid_thw.tolist() if isinstance(grid_thw, torch.Tensor) else grid_thw
        x = pixel_values.to(self.dtype)
        if x.dim() == 2:
            L = x.size(0)
            tp, ps = self.config.temporal_patch_size, self.config.patch_size
            x = x.view(L, -1, tp, ps, ps)
            x = self.patch_embed(x)
            if x.dim() == 5: x = x.view(L, self.hidden_size)
        elif x.dim() == 5:
            x = self.patch_embed(x).flatten(2).transpose(1, 2).reshape(-1, self.hidden_size)
        else:
            x = self.patch_embed(x)
            if x.dim() > 2: x = x.reshape(-1, self.hidden_size)

        pos = self.fast_pos_embed_interpolate(grid_thw_list).to(x.dtype)
        x = x + pos

        cos, sin = self.rot_pos_emb(grid_thw_list)
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)
      
        grid_arr = np.array(grid_thw_list, dtype=np.int32)
        seq_per_chunk = grid_arr[:, 1] * grid_arr[:, 2]
        cu_lens = np.repeat(seq_per_chunk, grid_arr[:, 0]).cumsum()
        cu_lens = np.concatenate([np.zeros(1, dtype=np.int32), cu_lens.astype(np.int32)])
        cu_seqlens = torch.from_numpy(cu_lens).to(x.device)
        max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item()) if cu_seqlens.numel() > 1 else 0

        deepstack_features = []
        for idx, block in enumerate(self.blocks):
            x = block(x, cu_seqlens, cos, sin, max_seqlen)
            if idx in self.deepstack_visual_indexes:
                ds_idx = self.deepstack_visual_indexes.index(idx)
                deepstack_features.append(self.deepstack_merger_list[ds_idx](x))

        main_embeds = self.merger(x)
        if deepstack_features:
            return torch.cat([main_embeds] + deepstack_features, dim=-1)
        return main_embeds
