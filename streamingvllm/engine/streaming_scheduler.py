from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import torch
import numpy as np

from streamingvllm.config import Config
from streamingvllm.engine.streaming_sequence import StreamingSequence, StreamingSequenceStatus
from streamingvllm.engine.streaming import StreamingConfig, StreamingChatManager, CompactPlan
from streamingvllm.engine.block_manager import BlockManager

# 🚨 纯净版 MRoPE Position 计算 (完全脱离 HF 对象依赖) 🚨
def compute_mrope_positions(input_tokens: list[int], image_grid_thw_list: list[tuple], image_token_id: int, spatial_merge_size: int):
    llm_pos_ids_list = []
    st = 0
    img_idx = 0

    i = 0
    while i < len(input_tokens):
        if input_tokens[i] == image_token_id and img_idx < len(image_grid_thw_list):
            # Text before image
            text_len = i - st
            if text_len > 0:
                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )

            t, h, w = image_grid_thw_list[img_idx]
            llm_h = h // spatial_merge_size
            llm_w = w // spatial_merge_size
            llm_t = t

            st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0

            # Correct 3D grid indices
            t_index = torch.arange(llm_t).view(-1, 1).expand(-1, llm_h * llm_w).flatten()
            h_index = torch.arange(llm_h).view(1, -1, 1).expand(llm_t, -1, llm_w).flatten()
            w_index = torch.arange(llm_w).view(1, 1, -1).expand(llm_t, llm_h, -1).flatten()

            llm_pos_ids_list.append(
                torch.stack([t_index, h_index, w_index]) + text_len + st_idx
            )

            num_vis = llm_t * llm_h * llm_w
            st = i + num_vis
            i = st
            img_idx += 1
        else:
            i += 1

    # Remaining text
    if st < len(input_tokens):
        text_len = len(input_tokens) - st
        st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
        llm_pos_ids_list.append(
            torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
        )

    if not llm_pos_ids_list:
        return torch.arange(len(input_tokens)).view(1, -1).expand(3, -1), 0

    positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
    delta = int(positions.max() + 1 - len(input_tokens))
    return positions, delta


@dataclass
class BatchKVOps:
    """Batch KV Operations (Decoupled Architecture)"""
    move_src: list[int] = field(default_factory=list)
    move_dst: list[int] = field(default_factory=list)
    
    inplace_slots: list[int] = field(default_factory=list)
    inplace_deltas: list[int] = field(default_factory=list)

    @classmethod
    def empty(cls) -> 'BatchKVOps':
        return cls()

    def add_plan(self, plan: CompactPlan):
        if not plan.needs_compact:
            return
        for op in plan.move_ops:
            self.move_src.append(op.src_slot)
            self.move_dst.append(op.dst_slot)
        for op in plan.inplace_ops:
            self.inplace_slots.append(op.slot)
            self.inplace_deltas.append(op.delta)

    @property
    def has_moves(self): return len(self.move_src) > 0
    @property
    def has_inplace(self): return len(self.inplace_slots) > 0
    @property
    def has_any(self): return self.has_moves or self.has_inplace


class StreamingScheduler:
    def __init__(self, config: Config, streaming_config: StreamingConfig):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[StreamingSequence] = deque()
        self.running: deque[StreamingSequence] = deque()
        self.idle: dict[int, StreamingSequence] = {}

    def is_finished(self):
        return not self.waiting and not self.running

    def add_session(self, seq: StreamingSequence):
        self.idle[seq.seq_id] = seq

    def continue_session(self, seq_id: int, user_message: str) -> Optional[StreamingSequence]:
        if seq_id not in self.idle:
            return None
        
        seq = self.idle.pop(seq_id)
        old_bt = list(seq.block_table)

        # 1. Plan eviction (based on current state)
        plan = seq.plan_eviction()

        # 2. Apply block changes if eviction is needed
        if plan.needs_compact:
            oldest = seq.state.dialogue_turns[0]
            evicted_start = oldest.start_pos
            evicted_end = oldest.end_pos

            # Release evicted blocks
            self.block_manager.release_blocks(plan.blocks_to_release)

            # Allocate new compact blocks
            new_blocks = []
            for _ in range(plan.new_blocks_needed):
                assert self.block_manager.free_block_ids, "Out of KV Cache blocks!"
                bid = self.block_manager.free_block_ids[0]
                self.block_manager._allocate_block(bid)
                new_blocks.append(bid)

            # Build new block table: compact + reusable
            new_bt = new_blocks + plan.reusable_blocks
            seq.block_table = new_bt

            # Resolve physical slots for KV operations
            StreamingChatManager.finalize_kv_ops(plan, old_bt, new_bt, self.block_size)

            # Apply state changes (removes oldest turn, updates history)
            seq.manager.apply_eviction(seq.state, plan)
            
            # 🚨 核心修复：精确缓存
            seq.num_cached_tokens = seq.state.total_tokens()

            # Clean up vision cache for evicted turn
            seq.evict_vision_cache(evicted_start, evicted_end)
            
            # Compute position delta and update remaining vision cache entries
            old_turn2_start = evicted_end
            new_turn2_start = seq.state.turns_start()
            position_delta = new_turn2_start - old_turn2_start
            seq.update_vision_cache_positions(position_delta)

            # MRoPE delta: if evicted turn had images, delta changes
            if hasattr(seq, 'mrope_position_delta') and seq.vision_cache:
                all_grids = []
                for turn in seq.state.dialogue_turns:
                    if turn.image_grid_thw_list:
                        all_grids.extend(turn.image_grid_thw_list)
                if all_grids:
                    image_token_id = getattr(seq, '_image_token_id', -1)
                    spatial_merge = getattr(seq, '_spatial_merge_size', 2)
                    if image_token_id > 0:
                        full_ids = seq.state.all_token_ids()
                        
                        # 直接调用我们顶部定义的纯净版函数
                        full_pos, _ = compute_mrope_positions(
                            full_ids, all_grids, image_token_id, spatial_merge
                        )
                        seq.mrope_position_delta = int(full_pos.max().item() + 1 - len(full_ids))
                else:
                    seq.mrope_position_delta = 0

        # 3. Add user turn — only these new tokens will need prefill
        seq.add_turn(user_message)
        self.waiting.append(seq)
        return seq

    def schedule(self) -> Tuple[List[StreamingSequence], bool, BatchKVOps]:
        seqs, plans = self._schedule_prefill()
        if seqs:
            ops = BatchKVOps.empty()
            for p in plans:
                ops.add_plan(p)
            return seqs, True, ops
        
        seqs = self._schedule_decode()
        return seqs, False, BatchKVOps.empty()

    def _schedule_prefill(self):
        seqs, plans, n = [], [], 0
        while self.waiting and len(seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            needed = seq.num_tokens - seq.num_cached_tokens
            
            if needed <= 0:
                self.waiting.popleft()
                seq.num_scheduled_tokens = 0
                seq.status = StreamingSequenceStatus.RUNNING
                self.running.append(seq)
                continue
                
            remaining = self.max_num_batched_tokens - n
            if remaining <= 0 or (not seq.block_table and not self._can_extend(seq)):
                break
            if remaining < needed and seqs:
                break
                
            if not seq.block_table:
                self.block_manager.allocate(seq)
            else:
                self._extend_blocks(seq)
                
            seq.num_scheduled_tokens = min(needed, remaining)
            if seq.num_scheduled_tokens == needed:
                seq.status = StreamingSequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
                
            seqs.append(seq)
            plans.append(seq.pending_plan or CompactPlan.empty())
            n += seq.num_scheduled_tokens
            
        return seqs, plans

    def _schedule_decode(self):
        seqs = []
        while self.running and len(seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self._preempt(self.running.pop())
                else:
                    self._preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                self.block_manager.may_append(seq)
                seqs.append(seq)
                
        if seqs:
            self.running.extendleft(reversed(seqs))
        return seqs

    def postprocess(self, seqs: List[StreamingSequence], token_ids: list[int], is_prefill: bool):
        for seq, tid in zip(seqs, token_ids):
            if is_prefill:
                # 🚨 精确缓存：Prefill 多少，就加上多少
                seq.num_cached_tokens = min(
                    seq.num_cached_tokens + seq.num_scheduled_tokens,
                    seq.num_tokens
                )
                
                # 如果是 Chunked Prefill (还没有把当前的 Prompt 全部算完)
                if seq.num_cached_tokens < seq.num_tokens or seq.num_completion_tokens > 0:
                    seq.num_scheduled_tokens = 0
                    continue
                    
            seq.append_token(tid)
            seq.num_cached_tokens += 1
            seq.num_scheduled_tokens = 0
            
            eos = not seq.ignore_eos and tid == self.eos
            limit = seq.num_completion_tokens >= seq.max_tokens
            
            if eos or limit:
                seq.end_turn()
                self.running.remove(seq)
                self.idle[seq.seq_id] = seq

    def _can_extend(self, seq):
        return len(self.block_manager.free_block_ids) >= max(0, seq.num_blocks - len(seq.block_table))

    def _extend_blocks(self, seq):
        while len(seq.block_table) < seq.num_blocks:
            assert self.block_manager.free_block_ids
            bid = self.block_manager.free_block_ids[0]
            self.block_manager._allocate_block(bid)
            seq.block_table.append(bid)

    def _preempt(self, seq):
        seq.status = StreamingSequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        seq.num_cached_tokens = 0
        self.waiting.appendleft(seq)

    def close_session(self, seq_id: int):
        seq = self.idle.pop(seq_id, None)
        if seq is None:
            for q in [self.running, self.waiting]:
                for s in list(q):
                    if s.seq_id == seq_id:
                        q.remove(s)
                        seq = s
                        break
                if seq:
                    break
        if seq:
            seq.finish()
            if seq.block_table:
                self.block_manager.deallocate(seq)
                seq.block_table.clear()
                seq.num_cached_tokens = 0
