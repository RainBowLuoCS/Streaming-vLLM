from dataclasses import dataclass
import torch
import torch.distributed as dist


@dataclass(slots=True)
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    # GDN (Qwen3.5 hybrid)
    gdn_slots: torch.Tensor | None = None
    gdn_cu_seqlens: torch.Tensor | None = None
    gdn_has_initial: torch.Tensor | None = None


_CONTEXT = Context()


def get_context():
    return _CONTEXT


def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0,
                max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None,
                gdn_slots=None, gdn_cu_seqlens=None, gdn_has_initial=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
                       max_seqlen_k, slot_mapping, context_lens, block_tables,
                       gdn_slots, gdn_cu_seqlens, gdn_has_initial)


def reset_context():
    global _CONTEXT
    _CONTEXT = Context()


# ===== Parallel groups (TP / EP) =====

class ParallelState:
    def __init__(self):
        self.tp_group = None
        self.ep_group = None
        self.tp_size = 1
        self.tp_rank = 0
        self.ep_size = 1
        self.ep_rank = 0
        self.global_rank = 0
        self.world_size = 1


_PSTATE = ParallelState()


def init_parallel_state(tp_size, ep_size, global_rank, world_size):
    assert tp_size * ep_size == world_size, \
        f"tp_size({tp_size}) * ep_size({ep_size}) != world_size({world_size})"
    _PSTATE.tp_size = tp_size
    _PSTATE.ep_size = ep_size
    _PSTATE.global_rank = global_rank
    _PSTATE.world_size = world_size
    _PSTATE.tp_rank = global_rank % tp_size
    _PSTATE.ep_rank = global_rank // tp_size

    if world_size == 1:
        _PSTATE.tp_group = None
        _PSTATE.ep_group = None
        return

    for e in range(ep_size):
        ranks = list(range(e * tp_size, (e + 1) * tp_size))
        g = dist.new_group(ranks)
        if global_rank in ranks:
            _PSTATE.tp_group = g

    for t in range(tp_size):
        ranks = list(range(t, world_size, tp_size))
        g = dist.new_group(ranks)
        if global_rank in ranks:
            _PSTATE.ep_group = g


def get_pstate():
    return _PSTATE