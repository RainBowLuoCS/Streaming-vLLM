import os
from dataclasses import dataclass, field
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    expert_parallel_size: int = 1
    enforce_eager: bool = False
    is_multimodal: bool = False
    shm_size: int = 2 * 1024 * 1024 * 1024  # 2GB
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    enable_spec_decode: bool = False
    spec_decode_mode: str = "off"
    # 🚨 Hybrid (Qwen3.5) fields
    is_hybrid: bool = False
    layer_types: tuple = ()
    num_full_layers: int = 0
    num_linear_layers: int = 0

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert 1 <= self.expert_parallel_size <= 8
        assert 1 <= self.world_size <= 8, (
            f"tp({self.tensor_parallel_size}) * ep({self.expert_parallel_size}) "
            f"= {self.world_size} > 8"
        )
        assert self.spec_decode_mode in {"off", "graph2"}
        if self.spec_decode_mode == "graph2":
            self.enable_spec_decode = True
        if self.enable_spec_decode and self.spec_decode_mode == "off":
            self.spec_decode_mode = "graph2"
        if not self.enable_spec_decode:
            self.spec_decode_mode = "off"
        self.hf_config = AutoConfig.from_pretrained(self.model)

        text_config = getattr(self.hf_config, "text_config", self.hf_config)
        max_pe = getattr(text_config, "max_position_embeddings", None)
        if max_pe:
            self.max_model_len = min(self.max_model_len, max_pe)

        if hasattr(self.hf_config, "vision_config"):
            self.is_multimodal = True

        # 🚨 Detect hybrid (Qwen3.5): has layer_types with linear_attention
        lt = getattr(text_config, "layer_types", None)
        if lt and "linear_attention" in lt:
            self.is_hybrid = True
            self.layer_types = tuple(lt)
            self.num_full_layers = sum(1 for t in lt if t == "full_attention")
            self.num_linear_layers = sum(1 for t in lt if t == "linear_attention")

    @property
    def world_size(self):
        return self.tensor_parallel_size * self.expert_parallel_size