"""Qwen3-VL conditional generation model wrapper."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from streamingvllm.models.qwen3 import Qwen3ForCausalLM
from streamingvllm.models.qwen3_vl_vision import Qwen3VLVisionEncoder

class Qwen3VLForConditionalGeneration(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"), "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"), "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.text_config = getattr(config, "text_config", config)
        self.vision_config = getattr(config, "vision_config", None)
        self.visual = Qwen3VLVisionEncoder(self.vision_config) if self.vision_config else None
        self.language_model = Qwen3ForCausalLM(self.text_config)
        self.deepstack_visual_indexes = (
            self.vision_config.deepstack_visual_indexes if self.vision_config else []
        )
        self.deepstack_num_level = len(self.deepstack_visual_indexes)
        self.visual_dim = self.vision_config.out_hidden_size if self.vision_config else 0

    def forward(self, input_ids=None, positions=None, inputs_embeds=None,
                deepstack_embeds=None, deepstack_layer_indices=None):
        if inputs_embeds is not None:
            return self.language_model.model(
                inputs_embeds=inputs_embeds, positions=positions,
                deepstack_embeds=deepstack_embeds,
                deepstack_layer_indices=deepstack_layer_indices,
            )
        return self.language_model.model(input_ids=input_ids, positions=positions)

    def compute_logits(self, hidden_states):
        return self.language_model.compute_logits(hidden_states)

def load_qwen3_vl_model(model_path, config):
    model = Qwen3VLForConditionalGeneration(config.hf_config)
    def name_mapping(weight_name):
        if weight_name.startswith("model.language_model."):
            sub = weight_name[len("model.language_model."):]
            if sub.startswith("model."):
                return "language_model.model." + sub[len("model."):]
            elif sub.startswith("lm_head."):
                return "language_model." + sub
            return "language_model.model." + sub
        if weight_name.startswith("model.visual."):
            return "visual." + weight_name[len("model.visual."):]
        if weight_name.startswith("lm_head."):
            return "language_model." + weight_name
        return weight_name
    from streamingvllm.utils.loader import load_model
    load_model(model, model_path, name_mapping=name_mapping)
    return model
