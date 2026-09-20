"""Qwen3.5 hybrid text model modules: GDN linear attention + full attention + MLP.

Layer types alternate between:
  - linear_attention: Gated Delta Net (recurrent, per-session conv/ssm state pool)
  - full_attention:   GQA + output gate + partial rotary + interleaved MRoPE + KV cache

State pool routing (conv/ssm) and GDN batch args (slots/cu_seqlens/has_initial) are
provided via the forward context (nanovllm.utils.context).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from einops import rearrange

from nanovllm.layers.fla import (
    chunk_gated_delta_rule,
    fused_recurrent_gated_delta_rule,
    RMSNormGated,
)
from nanovllm.layers.fla.compat import tl, triton
from nanovllm.layers.mamba.causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from nanovllm.layers.attention import Attention
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.utils.context import get_pstate, get_context


# ============================================================
# Gemma-style RMSNorm  (output = x_normed * (1 + weight))
# ============================================================
class GemmaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x, residual=None):
        orig = x.dtype
        if residual is not None:
            x = x.float() + residual.float()
            residual = x.to(orig)
        x = x.float()
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        x = x * (1.0 + self.weight.float())
        x = x.to(orig)
        if residual is None:
            return x
        return x, residual


# ============================================================
# GDN gating (fused triton kernel)
# ============================================================
@triton.jit
def _fused_gdn_gating_kernel(
    g, beta_output, A_log, a, b, dt_bias, seq_len,
    NUM_HEADS: tl.constexpr, beta: tl.constexpr, threshold: tl.constexpr,
    BLK_HEADS: tl.constexpr,
):
    i_b, i_s, i_d = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    head_off = i_d * BLK_HEADS + tl.arange(0, BLK_HEADS)
    off = i_b * seq_len * NUM_HEADS + i_s * NUM_HEADS + head_off
    mask = head_off < NUM_HEADS
    blk_A_log = tl.load(A_log + head_off, mask=mask)
    blk_a = tl.load(a + off, mask=mask)
    blk_b = tl.load(b + off, mask=mask)
    blk_bias = tl.load(dt_bias + head_off, mask=mask)
    x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)
    softplus_x = tl.where(beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x)
    blk_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x
    tl.store(g + off, blk_g.to(g.dtype.element_ty), mask=mask)
    blk_beta = tl.sigmoid(blk_b.to(tl.float32))
    tl.store(beta_output + off, blk_beta.to(beta_output.dtype.element_ty), mask=mask)


def fused_gdn_gating(A_log, a, b, dt_bias, beta=1.0, threshold=20.0):
    """a, b: [num_tokens, num_v_heads]. Returns g, beta: [1, num_tokens, num_v_heads]."""
    num_tokens, num_heads = a.shape
    grid = (num_tokens, 1, triton.cdiv(num_heads, 8))
    g = torch.empty(1, num_tokens, num_heads, dtype=torch.float32, device=a.device)
    beta_output = torch.empty(1, num_tokens, num_heads, dtype=b.dtype, device=b.device)
    _fused_gdn_gating_kernel[grid](
        g, beta_output, A_log, a, b, dt_bias, 1,
        num_heads, beta, threshold, 8, num_warps=1,
    )
    return g, beta_output


# ============================================================
# Partial rotary (only first rotary_dim dims rotated)
# ============================================================
def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_partial(q, k, cos, sin):
    """q, k: [T, num_heads, head_dim]. cos/sin: [T, rotary_dim]. Rotate only first rotary_dim."""
    rotary_dim = cos.shape[-1]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_rot = (q_rot.float() * cos) + (_rotate_half(q_rot).float() * sin)
    k_rot = (k_rot.float() * cos) + (_rotate_half(k_rot).float() * sin)
    q = torch.cat([q_rot.to(q.dtype), q_pass], dim=-1)
    k = torch.cat([k_rot.to(k.dtype), k_pass], dim=-1)
    return q, k


# ============================================================
# GDN Linear Attention  (Gated Delta Net)
# ============================================================
class Qwen3_5GatedDeltaNet(nn.Module):
    """Gated Delta Net. Conv/SSM state lives in an external per-session pool
    (attached as self._conv_pool / self._ssm_pool by the runner).
    Batch args (slots / cu_seqlens / has_initial) come from the forward context.
    """

    def __init__(self, config):
        super().__init__()
        self.tp_size = get_pstate().tp_size
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.activation = config.hidden_act

        self.num_v_heads_tp = self.num_v_heads // self.tp_size
        self.num_k_heads_tp = self.num_k_heads // self.tp_size
        self.key_dim_tp = self.key_dim // self.tp_size
        self.value_dim_tp = self.value_dim // self.tp_size
        self.conv_dim_tp = self.conv_dim // self.tp_size

        self.in_proj_qkv = nn.Parameter(torch.empty(self.conv_dim_tp, self.hidden_size))
        self.in_proj_z = nn.Parameter(torch.empty(self.value_dim_tp, self.hidden_size))
        self.in_proj_b = nn.Parameter(torch.empty(self.num_v_heads_tp, self.hidden_size))
        self.in_proj_a = nn.Parameter(torch.empty(self.num_v_heads_tp, self.hidden_size))
        self.conv1d_weight = nn.Parameter(torch.empty(self.conv_dim_tp, self.conv_kernel_size))
        self.conv1d_bias = None
        self.dt_bias = nn.Parameter(torch.empty(self.num_v_heads_tp))
        self.A_log = nn.Parameter(torch.empty(self.num_v_heads_tp))
        self.norm = RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps,
                                 group_size=None, norm_before_gate=True)
        self.out_proj = nn.Parameter(torch.empty(self.hidden_size, self.value_dim_tp))

        # attached by runner
        self._conv_pool = None
        self._ssm_pool = None

    def _project(self, hidden_states):
        qkv = F.linear(hidden_states, self.in_proj_qkv)
        z = F.linear(hidden_states, self.in_proj_z)
        b = F.linear(hidden_states, self.in_proj_b)
        a = F.linear(hidden_states, self.in_proj_a)
        return qkv, z, b, a

    def _split_qkv(self, mixed_qkv):
        return torch.split(mixed_qkv, [self.key_dim_tp, self.key_dim_tp, self.value_dim_tp], dim=-1)

    def _gated_norm_out(self, core_out, z, n):
        core_out = core_out.reshape(n, self.num_v_heads_tp, self.head_v_dim).reshape(-1, self.head_v_dim)
        z_r = z.reshape(n, self.num_v_heads_tp, self.head_v_dim).reshape(-1, self.head_v_dim)
        core_out = self.norm(core_out, z_r).reshape(n, self.value_dim_tp)
        out = F.linear(core_out, self.out_proj)
        if self.tp_size > 1:
            dist.all_reduce(out, group=get_pstate().tp_group)
        return out

    def forward(self, hidden_states, is_prefill):
        ctx = get_context()
        if is_prefill:
            return self._forward_prefill(hidden_states, ctx.gdn_slots,
                                         ctx.gdn_cu_seqlens, ctx.gdn_has_initial)
        return self._forward_decode(hidden_states, ctx.gdn_slots)

    def _forward_prefill(self, hidden_states, gdn_slots, cu_seqlens, has_initial_state):
        T = hidden_states.shape[0]
        mixed_qkv, z, b, a = self._project(hidden_states)

        conv_state_t = self._conv_pool.transpose(-1, -2)   # [slots, conv_dim, kernel-1] view
        mixed_qkv_T = mixed_qkv.transpose(0, 1).contiguous()
        mixed_qkv_T = causal_conv1d_fn(
            x=mixed_qkv_T, weight=self.conv1d_weight, bias=self.conv1d_bias,
            conv_states=conv_state_t, query_start_loc=cu_seqlens,
            cache_indices=gdn_slots, has_initial_state=has_initial_state,
            activation=self.activation,
        )
        mixed_qkv = mixed_qkv_T.transpose(0, 1).contiguous()

        q, k, v = self._split_qkv(mixed_qkv)
        q = rearrange(q, "t (h d) -> 1 t h d", d=self.head_k_dim).contiguous()
        k = rearrange(k, "t (h d) -> 1 t h d", d=self.head_k_dim).contiguous()
        v = rearrange(v, "t (h d) -> 1 t h d", d=self.head_v_dim).contiguous()
        g, beta = fused_gdn_gating(self.A_log, a, b, self.dt_bias)

        init_state = self._ssm_pool[gdn_slots.long()].clone()
        init_state[~has_initial_state] = 0
        core_out, last_state = chunk_gated_delta_rule(
            q=q, k=k, v=v, g=g, beta=beta,
            initial_state=init_state, output_final_state=True,
            cu_seqlens=cu_seqlens, use_qk_l2norm_in_kernel=True,
        )
        self._ssm_pool[gdn_slots.long()] = last_state.to(self._ssm_pool.dtype)
        return self._gated_norm_out(core_out, z, T)

    def _forward_decode(self, hidden_states, gdn_slots):
        S = hidden_states.shape[0]
        mixed_qkv, z, b, a = self._project(hidden_states)

        conv_state_t = self._conv_pool.transpose(-1, -2)
        mixed_qkv = causal_conv1d_update(
            mixed_qkv, conv_state_t, self.conv1d_weight, self.conv1d_bias,
            self.activation, conv_state_indices=gdn_slots,
        )

        q, k, v = self._split_qkv(mixed_qkv)
        q = rearrange(q, "s (h d) -> 1 s h d", d=self.head_k_dim).contiguous()
        k = rearrange(k, "s (h d) -> 1 s h d", d=self.head_k_dim).contiguous()
        v = rearrange(v, "s (h d) -> 1 s h d", d=self.head_v_dim).contiguous()
        g, beta = fused_gdn_gating(self.A_log, a, b, self.dt_bias)
        g = g.reshape(1, S, self.num_v_heads_tp).contiguous()
        beta = beta.reshape(1, S, self.num_v_heads_tp).contiguous()

        cu_seqlens = torch.arange(S + 1, device=hidden_states.device, dtype=torch.int32)
        core_out, _ = fused_recurrent_gated_delta_rule(
            q=q, k=k, v=v, g=g, beta=beta,
            initial_state=self._ssm_pool, inplace_final_state=True,
            cu_seqlens=cu_seqlens, ssm_state_indices=gdn_slots,
            use_qk_l2norm_in_kernel=True,
        )
        return self._gated_norm_out(core_out, z, S)


# ============================================================
# Full Attention  (GQA + output gate + partial rotary + MRoPE + KV cache)
# ============================================================
class Qwen3_5Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tp_size = get_pstate().tp_size
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_heads = self.total_num_heads // self.tp_size
        self.num_kv_heads = max(1, self.total_num_kv_heads // self.tp_size)
        self.scaling = self.head_dim ** -0.5
        self.attn_output_gate = getattr(config, "attn_output_gate", True)

        q_out = self.total_num_heads * self.head_dim * (2 if self.attn_output_gate else 1)
        self.q_proj = nn.Parameter(torch.empty(q_out // self.tp_size, self.hidden_size))
        self.k_proj = nn.Parameter(torch.empty(self.total_num_kv_heads * self.head_dim // self.tp_size, self.hidden_size))
        self.v_proj = nn.Parameter(torch.empty(self.total_num_kv_heads * self.head_dim // self.tp_size, self.hidden_size))
        self.o_proj = nn.Parameter(torch.empty(self.hidden_size, self.total_num_heads * self.head_dim // self.tp_size))
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn = Attention(self.num_heads, self.head_dim, self.scaling, self.num_kv_heads)

    def forward(self, positions, hidden_states, cos, sin):
        T = hidden_states.shape[0]
        q = F.linear(hidden_states, self.q_proj).view(T, self.num_heads, 2 * self.head_dim)
        q, gate = torch.chunk(q, 2, dim=-1)
        gate = gate.reshape(T, self.num_heads * self.head_dim)
        k = F.linear(hidden_states, self.k_proj).view(T, self.num_kv_heads, self.head_dim)
        v = F.linear(hidden_states, self.v_proj).view(T, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q.reshape(-1, self.head_dim)).view(T, self.num_heads, self.head_dim)
        k = self.k_norm(k.reshape(-1, self.head_dim)).view(T, self.num_kv_heads, self.head_dim)
        q, k = apply_rotary_partial(q, k, cos, sin)

        o = self.attn(q.contiguous(), k.contiguous(), v.contiguous())
        o = o.reshape(T, self.num_heads * self.head_dim)
        o = o * torch.sigmoid(gate)
        out = F.linear(o, self.o_proj)
        if self.tp_size > 1:
            dist.all_reduce(out, group=get_pstate().tp_group)
        return out


# ============================================================
# MLP  (dense SwiGLU)
# ============================================================
class Qwen3_5MLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.tp_size = get_pstate().tp_size
        self.hidden_size = config.hidden_size
        inter = intermediate_size or config.intermediate_size
        assert inter % self.tp_size == 0
        self.inter_tp = inter // self.tp_size
        self.gate_proj = nn.Parameter(torch.empty(self.inter_tp, self.hidden_size))
        self.up_proj = nn.Parameter(torch.empty(self.inter_tp, self.hidden_size))
        self.down_proj = nn.Parameter(torch.empty(self.hidden_size, self.inter_tp))

    def forward(self, x):
        g = F.linear(x, self.gate_proj)
        u = F.linear(x, self.up_proj)
        out = F.linear(F.silu(g) * u, self.down_proj)
        if self.tp_size > 1:
            dist.all_reduce(out, group=get_pstate().tp_group)
        return out

from nanovllm.layers.fused_moe import FusedMoE
from nanovllm.layers.linear import ReplicatedLinear


class Qwen3_5SparseMoeBlock(nn.Module):
    """MoE block: routed experts (EP FusedMoE) + shared expert (gated)."""
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.gate = ReplicatedLinear(config.hidden_size, config.num_experts, bias=False)
        self.shared_expert_gate = ReplicatedLinear(config.hidden_size, 1, bias=False)
        self.shared_expert = Qwen3_5MLP(config, config.shared_expert_intermediate_size)
        self.experts = FusedMoE(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=getattr(config, "norm_topk_prob", True),
        )

    def forward(self, x):
        orig_shape = x.shape
        x = x.view(-1, self.hidden_size)
        router_logits = self.gate(x)                      # [T, num_experts]
        routed = self.experts(x, router_logits)           # EP FusedMoE (internal ep all_reduce)
        shared = self.shared_expert(x)                    # TP MLP (internal tp all_reduce)
        gate = torch.sigmoid(self.shared_expert_gate(x))  # [T, 1]
        out = routed + shared * gate
        return out.view(orig_shape)

# ============================================================
# Decoder Layer  (linear or full attention, decided by layer_type)
# ============================================================
class Qwen3_5DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(config)
        else:
            self.self_attn = Qwen3_5Attention(config)
        if getattr(config, "num_experts", 0) > 0:
            self.mlp = Qwen3_5SparseMoeBlock(config)
        else:
            self.mlp = Qwen3_5MLP(config, config.intermediate_size)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions, hidden_states, is_prefill, cos, sin):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states, is_prefill)
        else:
            hidden_states = self.self_attn(positions, hidden_states, cos, sin)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


# ============================================================
# Model  (embed + hybrid layers + norm, with shared partial-mrope cache)
# ============================================================
class Qwen3_5Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3_5DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # linear layer indices (for GDN pool routing)
        gdn_idx = 0
        self.linear_layer_indices = []
        for i, t in enumerate(config.layer_types):
            if t == "linear_attention":
                self.layers[i].gdn_idx = gdn_idx
                self.linear_layer_indices.append(i)
                gdn_idx += 1
        self.num_linear_layers = gdn_idx

        # prebuild partial-mrope cos/sin cache + interleave masks (lookup instead of recompute)
        rp = config.rope_parameters
        rope_theta = rp["rope_theta"]
        head_dim = config.head_dim
        rotary_dim = int(head_dim * rp.get("partial_rotary_factor", 1.0))
        half = rotary_dim // 2
        mrope_section = rp.get("mrope_section")
        max_pos = config.max_position_embeddings

        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
        t = torch.arange(max_pos, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("_rope_cos_cache", freqs.cos(), persistent=False)
        self.register_buffer("_rope_sin_cache", freqs.sin(), persistent=False)

        mask_h = torch.zeros(half, dtype=torch.bool)
        mask_w = torch.zeros(half, dtype=torch.bool)
        for i in range(1, min(mrope_section[1] * 3, half), 3):
            mask_h[i] = True
        for i in range(2, min(mrope_section[2] * 3, half), 3):
            mask_w[i] = True
        self.register_buffer("_rope_mask_h", mask_h, persistent=False)
        self.register_buffer("_rope_mask_w", mask_w, persistent=False)
        self.rotary_dim = rotary_dim

    def _mrope_cos_sin(self, positions, dtype):
        if positions.dim() == 1:
            pos_t = pos_h = pos_w = positions
        else:
            pos_t, pos_h, pos_w = positions[0], positions[1], positions[2]
        max_idx = self._rope_cos_cache.shape[0] - 1
        pos_t = pos_t.clamp(0, max_idx)
        pos_h = pos_h.clamp(0, max_idx)
        pos_w = pos_w.clamp(0, max_idx)
        ct, ch, cw = self._rope_cos_cache[pos_t], self._rope_cos_cache[pos_h], self._rope_cos_cache[pos_w]
        st, sh, sw = self._rope_sin_cache[pos_t], self._rope_sin_cache[pos_h], self._rope_sin_cache[pos_w]
        mh, mw = self._rope_mask_h, self._rope_mask_w
        cos_half = torch.where(mw, cw, torch.where(mh, ch, ct))
        sin_half = torch.where(mw, sw, torch.where(mh, sh, st))
        cos = torch.cat([cos_half, cos_half], dim=-1).to(dtype)
        sin = torch.cat([sin_half, sin_half], dim=-1).to(dtype)
        return cos, sin

    def forward(self, input_ids, positions, is_prefill, inputs_embeds=None,
                deepstack_embeds=None, deepstack_layer_indices=None):
        h = inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
        cos, sin = self._mrope_cos_sin(positions, h.dtype)
        for layer in self.layers:
            h = layer(positions, h, is_prefill, cos, sin)
        h = self.norm(h)
        return h


# ============================================================
# ForCausalLM
# ============================================================
class Qwen3_5ForCausalLM(nn.Module):
    packed_modules_mapping = {}

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Qwen3_5Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)

    def forward(self, input_ids=None, positions=None, is_prefill=True, inputs_embeds=None,
                deepstack_embeds=None, deepstack_layer_indices=None):
        return self.model(input_ids, positions, is_prefill, inputs_embeds=inputs_embeds)

    def compute_logits(self, hidden_states):
        return self.lm_head(hidden_states)