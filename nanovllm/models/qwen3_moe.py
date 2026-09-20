import torch
from torch import nn

from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import ReplicatedLinear
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.layers.fused_moe import FusedMoE
from nanovllm.models.qwen3 import Qwen3Attention, Qwen3MLP


class Qwen3MoeSparseMoeBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.gate = ReplicatedLinear(config.hidden_size, config.num_experts, bias=False)
        self.experts = FusedMoE(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=getattr(config, "norm_topk_prob", True),
        )

    def forward(self, hidden_states):
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)
        router_logits = self.gate(hidden_states)
        out = self.experts(hidden_states, router_logits)
        return out.view(orig_shape)


class Qwen3MoeDecoderLayer(nn.Module):

    def __init__(self, config, layer_idx):
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', False),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        mlp_only = getattr(config, "mlp_only_layers", [])
        sparse_step = getattr(config, "decoder_sparse_step", 1)
        num_experts = getattr(config, "num_experts", 0)
        use_moe = (
            num_experts > 0
            and layer_idx not in mlp_only
            and sparse_step > 0
            and (layer_idx + 1) % sparse_step == 0
        )
        if use_moe:
            self.mlp = Qwen3MoeSparseMoeBlock(config)
        else:
            self.mlp = Qwen3MLP(config.hidden_size, config.intermediate_size, config.hidden_act)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions, hidden_states, residual):
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3MoeModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen3MoeDecoderLayer(config, i) for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids=None, positions=None, inputs_embeds=None,
                deepstack_embeds=None, deepstack_layer_indices=None):
        hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
        residual = None
        ds_idx = 0
        for layer_idx, layer in enumerate(self.layers):
            hidden_states, residual = layer(positions, hidden_states, residual)
            if (deepstack_embeds is not None and deepstack_layer_indices is not None
                    and layer_idx in range(len(deepstack_layer_indices))
                    and ds_idx < len(deepstack_layer_indices)):
                key = f"deepstack_{ds_idx}"
                if key in deepstack_embeds:
                    hidden_states = hidden_states + deepstack_embeds[key]
                ds_idx += 1
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3MoeForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"), "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"), "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Qwen3MoeModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(self, input_ids=None, positions=None, inputs_embeds=None,
                deepstack_embeds=None, deepstack_layer_indices=None):
        return self.model(input_ids, positions, inputs_embeds=inputs_embeds,
                          deepstack_embeds=deepstack_embeds,
                          deepstack_layer_indices=deepstack_layer_indices)

    def compute_logits(self, hidden_states):
        return self.lm_head(hidden_states)