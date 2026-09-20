import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context, get_pstate


class VocabParallelEmbedding(nn.Module):

    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        ps = get_pstate()
        self.tp_rank = ps.tp_rank
        self.tp_size = ps.tp_size
        self.tp_group = ps.tp_group
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param, loaded_weight):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y, group=self.tp_group)
        return y


class ParallelLMHead(VocabParallelEmbedding):

    def __init__(self, num_embeddings, embedding_dim, bias=False):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def _group_dst_rank(self):
        # global rank of tp_rank=0 within this tp_group
        ps = get_pstate()
        return ps.ep_rank * ps.tp_size

    def forward(self, x):
        context = get_context()
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        logits = F.linear(x, self.weight)
        if self.tp_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, dst=self._group_dst_rank(), group=self.tp_group)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits