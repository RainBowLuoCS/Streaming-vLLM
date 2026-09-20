"""GDN state pool: per-session conv_state + ssm_state.
Plan X: state persists for the session lifetime, never evicted by sliding window.
The LAST slot is a garbage slot for graph-padding sequences (never allocated to real sessions).
"""
from __future__ import annotations
from collections import deque
import torch


class GDNStatePool:
    def __init__(self, num_slots, num_linear_layers, conv_dim, kernel_size,
                 num_v_heads, head_v_dim, head_k_dim, dtype, device):
        self.num_slots = num_slots
        self.NL = num_linear_layers
        self.kernel_m1 = kernel_size - 1

        # conv_state physical layout: [num_slots, kernel-1, conv_dim] (dim contiguous)
        self.conv_pool = [
            torch.zeros(num_slots, kernel_size - 1, conv_dim, device=device, dtype=dtype)
            for _ in range(num_linear_layers)
        ]
        self.ssm_pool = [
            torch.zeros(num_slots, num_v_heads, head_v_dim, head_k_dim, device=device, dtype=torch.float32)
            for _ in range(num_linear_layers)
        ]

        # 🚨 exclude the last slot (garbage slot for graph padding)
        self.free_slots = deque(range(num_slots - 1))
        self.seq_to_slot: dict[int, int] = {}

    def alloc(self, seq_id):
        if seq_id in self.seq_to_slot:
            return self.seq_to_slot[seq_id]
        assert self.free_slots, "GDN state pool exhausted (increase max_num_seqs)"
        slot = self.free_slots.popleft()
        self.seq_to_slot[seq_id] = slot
        for li in range(self.NL):
            self.conv_pool[li][slot].zero_()
            self.ssm_pool[li][slot].zero_()
        return slot

    def free(self, seq_id):
        slot = self.seq_to_slot.pop(seq_id, None)
        if slot is not None:
            self.free_slots.append(slot)

    def get_slot(self, seq_id):
        return self.seq_to_slot.get(seq_id, -1)