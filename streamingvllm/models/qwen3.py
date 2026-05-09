import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config
from streamingvllm.layers.activation import SiluAndMul
from streamingvllm.layers.attention import Attention
from streamingvllm.layers.layernorm import RMSNorm
from streamingvllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from streamingvllm.layers.rotary_embedding import get_rope
from streamingvllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead

class Qwen3Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads,
                 max_position=4096*32, head_dim=None, rms_norm_eps=1e-6,
                 qkv_bias=False, rope_theta=10000, rope_scaling=None):
        super().__init__()
        tp_size = dist.get_world_size()
        self.num_heads = num_heads // tp_size
        self.num_kv_heads = num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias
        self.qkv_proj = QKVParallelLinear(hidden_size, self.head_dim, num_heads, num_kv_heads, bias=qkv_bias)
        self.o_proj = RowParallelLinear(num_heads * self.head_dim, hidden_size, bias=False)

        # Extract MRoPE section sizes from rope_scaling config (HF uses "mrope_section")
        mrope_section_sizes = None
        if isinstance(rope_scaling, dict):
            rope_theta = rope_scaling.get("rope_theta", rope_theta)
            mrope_section_sizes = rope_scaling.get("mrope_section", None)
            if mrope_section_sizes:
                mrope_section_sizes = tuple(mrope_section_sizes)

        self.rotary_emb = get_rope(
            self.head_dim, rotary_dim=self.head_dim,
            max_position=max_position, base=rope_theta,
            mrope_section_sizes=mrope_section_sizes,
        )
        self.attn = Attention(self.num_heads, self.head_dim, self.scaling, self.num_kv_heads)
        if not qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(self, positions, hidden_states):
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        if not self.qkv_bias:
            q = self.q_norm(q.contiguous().view(-1, self.head_dim)).view(-1, self.num_heads, self.head_dim)
            k = self.k_norm(k.contiguous().view(-1, self.head_dim)).view(-1, self.num_kv_heads, self.head_dim)
        q, k = self.rotary_emb(positions, q, k)
        return self.o_proj(self.attn(q, k, v).flatten(1, -1))

class Qwen3MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_act):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(hidden_size, [intermediate_size]*2, bias=False)
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False)
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()
    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))

class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size, num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads, max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps, qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
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

class Qwen3Model(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids=None, positions=None, inputs_embeds=None,
                deepstack_embeds=None, deepstack_layer_indices=None):
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(input_ids)
        
        residual = None
        ds_idx = 0
        for layer_idx, layer in enumerate(self.layers):
            hidden_states, residual = layer(positions, hidden_states, residual)
            # DeepStack injection (full sequence length tensor, non-vision positions are 0)
            if (deepstack_embeds is not None
                and deepstack_layer_indices is not None
                and layer_idx in deepstack_layer_indices
                and ds_idx < len(deepstack_layer_indices)):
                key = f"deepstack_{ds_idx}"
                if key in deepstack_embeds:
                    hidden_states = hidden_states + deepstack_embeds[key]
                ds_idx += 1
        
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

class Qwen3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"), "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"), "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.model = Qwen3Model(config)
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
