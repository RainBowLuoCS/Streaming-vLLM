"""Qwen3.5 hybrid (linear+full attention) conditional generation wrapper for streaming engine."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanovllm.models.qwen3_next import Qwen3_5Model
from nanovllm.layers.embed_head import ParallelLMHead
from nanovllm.utils.context import get_pstate


class Qwen3_5ForConditionalGeneration(nn.Module):
    """Hybrid Qwen3.5. Vision reuses Qwen3-VL encoder (deepstack empty).
    Text model: Qwen3_5ModelPool (GDN pool + flash full attention).
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.text_config = getattr(config, "text_config", config)
        self.vision_config = getattr(config, "vision_config", None)

        # Vision: reuse Qwen3-VL encoder
        if self.vision_config is not None:
            from nanovllm.models.qwen3_vl_vision import Qwen3VLVisionEncoder
            self.visual = Qwen3VLVisionEncoder(self.vision_config)
        else:
            self.visual = None

        # Text: hybrid pool model
        self.model = Qwen3_5Model(self.text_config)
        self.lm_head = ParallelLMHead(self.text_config.vocab_size, self.text_config.hidden_size)

        # deepstack info (Qwen3.5 vision has empty deepstack)
        self.deepstack_visual_indexes = (
            getattr(self.vision_config, "deepstack_visual_indexes", []) if self.vision_config else []
        )
        self.deepstack_num_level = len(self.deepstack_visual_indexes)
        self.visual_dim = getattr(self.vision_config, "out_hidden_size", 0) if self.vision_config else 0

        # linear layer indices (for GDN pool routing)
        self.linear_layer_indices = self.model.linear_layer_indices
        self.num_linear_layers = self.model.num_linear_layers

    def forward(self, input_ids=None, positions=None, is_prefill=True,
                inputs_embeds=None,
                deepstack_embeds=None, deepstack_layer_indices=None):
        return self.model(input_ids, positions, is_prefill, inputs_embeds=inputs_embeds)

    def compute_logits(self, hidden_states):
        return self.lm_head(hidden_states)


def load_qwen3_5_model(model_path, config):
    """Load Qwen3.5 weights. Weights are SEPARATE format (in_proj_qkv/z/b/a, A_log, dt_bias, conv1d)."""
    model = Qwen3_5ForConditionalGeneration(config.hf_config)
    from nanovllm.utils.loader import load_qwen3_5_weights
    load_qwen3_5_weights(model, model_path, config)
    return model