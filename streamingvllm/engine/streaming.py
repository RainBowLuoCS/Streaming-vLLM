"""
StreamingLLM core with Decoupled KV Cache Operations.
Step 1: Pure Memory Move (No RoPE update during move).
Step 2: Universal In-place Delta RoPE (Exact 3D delta calculation).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque
from typing import Tuple, List
import torch
import numpy as np

def compute_mrope_positions(input_tokens: list[int], image_grid_thw_list: list[tuple], image_token_id: int, spatial_merge_size: int):
    llm_pos_ids_list = []
    st = 0
    img_idx = 0
    i = 0
    while i < len(input_tokens):
        if input_tokens[i] == image_token_id and img_idx < len(image_grid_thw_list):
            text_len = i - st
            if text_len > 0:
                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
            
            t, h, w = image_grid_thw_list[img_idx]
            llm_h = h // spatial_merge_size
            llm_w = w // spatial_merge_size
            llm_t = t
            
            st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
            
            t_index = torch.arange(llm_t).view(-1, 1).expand(-1, llm_h * llm_w).flatten()
            h_index = torch.arange(llm_h).view(1, -1, 1).expand(llm_t, -1, llm_w).flatten()
            w_index = torch.arange(llm_w).view(1, 1, -1).expand(llm_t, llm_h, -1).flatten()
            
            llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
            
            num_vis = llm_t * llm_h * llm_w
            st = i + num_vis
            i = st
            img_idx += 1
        else:
            i += 1

    if st < len(input_tokens):
        text_len = len(input_tokens) - st
        st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
        llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

    if not llm_pos_ids_list:
        return torch.arange(len(input_tokens)).view(1, -1).expand(3, -1), 0

    positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
    delta = int(positions.max() + 1 - len(input_tokens))
    return positions, delta


@dataclass
class DialogueTurn:
    user_content: str
    assistant_content: str = ""
    user_token_ids: list[int] = field(default_factory=list)
    assistant_token_ids: list[int] = field(default_factory=list)
    start_pos: int = 0
    end_pos: int = 0
    image_grid_thw_list: list[tuple[int,int,int]] = field(default_factory=list)

@dataclass
class StreamingConfig:
    dialogue_window: int = 5
    history_window: int = 512
    history_sink: int = 64
    initial_padding: int = 16

@dataclass
class KVMoveOp:
    old_pos: int
    new_pos: int
    src_slot: int = -1
    dst_slot: int = -1

@dataclass
class KVInplaceOp:
    new_pos: int
    delta: int
    slot: int = -1

@dataclass
class CompactPlan:
    move_ops: list[KVMoveOp] = field(default_factory=list)
    inplace_ops: list[KVInplaceOp] = field(default_factory=list)
    blocks_to_release: list[int] = field(default_factory=list)
    new_blocks_needed: int = 0
    reusable_blocks: list[int] = field(default_factory=list)
    new_history_ids: list[int] = field(default_factory=list)
    new_padding_count: int = 0

    @classmethod
    def empty(cls) -> 'CompactPlan':
        return cls()
    @property
    def needs_compact(self) -> bool:
        return len(self.blocks_to_release) > 0

@dataclass
class StreamingState:
    system_ids: list[int] = field(default_factory=list)
    history_prefix_ids: list[int] = field(default_factory=list)
    history_content_ids: list[int] = field(default_factory=list)
    padding_ids: list[int] = field(default_factory=list)
    padding_count: int = 0
    newline_template_pos: int = -1
    dialogue_turns: deque[DialogueTurn] = field(default_factory=deque)

    def sys_len(self) -> int: return len(self.system_ids)
    def hist_len(self) -> int: return len(self.history_prefix_ids) + len(self.history_content_ids) + len(self.padding_ids)
    def turns_start(self) -> int: return self.sys_len() + self.hist_len()

    def total_tokens(self) -> int:
        t = self.sys_len() + self.hist_len()
        for turn in self.dialogue_turns:
            t += len(turn.user_token_ids) + len(turn.assistant_token_ids)
        return t

    def all_token_ids(self) -> list[int]:
        ids = list(self.system_ids)
        ids.extend(self.history_prefix_ids)
        ids.extend(self.history_content_ids)
        ids.extend(self.padding_ids)
        for turn in self.dialogue_turns:
            ids.extend(turn.user_token_ids)
            ids.extend(turn.assistant_token_ids)
        return ids

class StreamingChatManager:
    def __init__(self, config: StreamingConfig, block_size: int, tokenizer):
        self.config = config
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.newline_id = tokenizer.encode("\n", add_special_tokens=False)[0]
        self._hist_prefix = tokenizer.encode("<|im_start|>history\n", add_special_tokens=False)
        self._turn_prefix = tokenizer.encode("<|im_end|>\n<|im_start|>user\n", add_special_tokens=False)
        self._turn_middle = tokenizer.encode("<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)
        self.eos_id = getattr(tokenizer, "eos_token_id", -1) or -1

    def create_state(self, system_content: str) -> StreamingState:
        st = StreamingState()
        sys_text = f"<|im_start|>system\n{system_content}<|im_end|>\n"
        st.system_ids = self.tokenizer.encode(sys_text, add_special_tokens=False)
        st.history_prefix_ids = list(self._hist_prefix)
        st.history_content_ids = []
        st.padding_ids = [self.newline_id] * self.config.initial_padding
        st.padding_count = self.config.initial_padding
        st.newline_template_pos = st.sys_len() + len(st.history_prefix_ids) - 1
        return st

    def add_user_turn(self, state: StreamingState, user_content: str) -> DialogueTurn:
        user_ids = (
            list(self._turn_prefix)
            + self.tokenizer.encode(user_content, add_special_tokens=False)
            + list(self._turn_middle)
        )
        start = state.total_tokens()
        turn = DialogueTurn(
            user_content=user_content,
            user_token_ids=user_ids,
            start_pos=start,
            end_pos=start + len(user_ids),
        )
        state.dialogue_turns.append(turn)
        return turn

    def add_assistant_token(self, state: StreamingState, token_id: int):
        if state.dialogue_turns:
            t = state.dialogue_turns[-1]
            t.assistant_token_ids.append(token_id)
            t.end_pos += 1

    def finalize_turn(self, state: StreamingState):
        if not state.dialogue_turns: return
        t = state.dialogue_turns[-1]
        while t.assistant_token_ids and t.assistant_token_ids[-1] == self.eos_id:
            t.assistant_token_ids.pop()
            t.end_pos -= 1
        t.assistant_content = self.tokenizer.decode(t.assistant_token_ids, skip_special_tokens=True)

    def plan_eviction(self, seq, block_table: list[int]) -> CompactPlan:
        """
        Decoupled Eviction Plan:
        1. Move: Only pure memory copy for compact region.
        2. Inplace: Universal Delta RoPE for ALL retained tokens.
        """
        state = seq.state
        if len(state.dialogue_turns) <= self.config.dialogue_window:
            return CompactPlan.empty()

        oldest = state.dialogue_turns[0]

        # 1. Build new history content
        new_hist_ids, sources = self._plan_history_content(state, oldest)

        # 2. Determine boundaries
        old_turns_start = state.turns_start()
        old_t1_len = len(oldest.user_token_ids) + len(oldest.assistant_token_ids)
        t2_start_old = old_turns_start + old_t1_len
        t2_block = t2_start_old // self.block_size
        t2_offset = t2_start_old % self.block_size

        if t2_offset > 0:
            first_complete = t2_block + 1
            t2_partial = self.block_size - t2_offset
        else:
            first_complete = t2_block
            t2_partial = 0

        # 3. New layout sizes
        sys_len = state.sys_len()
        hp_len = len(state.history_prefix_ids)
        nhc_len = len(new_hist_ids)

        before_pad = sys_len + hp_len + nhc_len
        compact_content = before_pad + t2_partial
        compact_blocks = (compact_content + self.block_size - 1) // self.block_size
        new_pad = compact_blocks * self.block_size - compact_content

        blk_release = block_table[:first_complete]
        blk_reuse = block_table[first_complete:]

        # 4. Build Pure Move Ops (No RoPE)
        move_ops = self._build_move_ops(
            state, sources, new_pad, t2_start_old, t2_partial,
            sys_len, hp_len, nhc_len,
        )

        # 5. Build Universal Inplace RoPE Ops (Using exact 3D MRoPE positions)
        inplace_ops = self._build_universal_inplace_ops(
            seq, state, new_hist_ids, new_pad, oldest, 
            t2_start_old, t2_partial, first_complete
        )

        return CompactPlan(
            move_ops=move_ops,
            inplace_ops=inplace_ops,
            blocks_to_release=blk_release,
            new_blocks_needed=compact_blocks,
            reusable_blocks=blk_reuse,
            new_history_ids=new_hist_ids,
            new_padding_count=new_pad,
        )

    def _plan_no_turn2(self, seq, state, new_hist_ids, sources, block_table):
        sys_len = state.sys_len()
        hp_len = len(state.history_prefix_ids)
        content = sys_len + hp_len + len(new_hist_ids)
        cb = (content + self.block_size - 1) // self.block_size
        new_pad = cb * self.block_size - content
        
        move_ops = self._build_move_ops(state, sources, new_pad, 0, 0, sys_len, hp_len, len(new_hist_ids))
        inplace_ops = self._build_universal_inplace_ops(
            seq, state, new_hist_ids, new_pad, state.dialogue_turns[0], 0, 0, len(block_table)
        )
        
        return CompactPlan(
            move_ops=move_ops, inplace_ops=inplace_ops,
            blocks_to_release=list(block_table), new_blocks_needed=cb,
            reusable_blocks=[], new_history_ids=new_hist_ids, new_padding_count=new_pad,
        )

    def _plan_history_content(self, state: StreamingState, oldest: DialogueTurn) -> Tuple[list[int], list[int]]:
        old_hc_start = state.sys_len() + len(state.history_prefix_ids)
        old_hc = state.history_content_ids
        old_turns_start = state.turns_start()
        asst_start = old_turns_start + len(oldest.user_token_ids)
        asst_ids = oldest.assistant_token_ids
        template = state.newline_template_pos

        ids, sources = [], []
        for i, tid in enumerate(old_hc):
            ids.append(tid); sources.append(old_hc_start + i)
        if old_hc and asst_ids:
            ids.append(self.newline_id); sources.append(template)
        for i, tid in enumerate(asst_ids):
            ids.append(tid); sources.append(asst_start + i)

        if len(ids) > self.config.history_window:
            excess = len(ids) - self.config.history_window
            sink = self.config.history_sink
            if 0 < sink < len(ids):
                trim_end = min(sink + excess, len(ids))
                ids = ids[:sink] + ids[trim_end:]
                sources = sources[:sink] + sources[trim_end:]
            else:
                ids = ids[excess:]; sources = sources[excess:]
        return ids, sources

    def _build_move_ops(self, state, history_sources, new_pad, t2_start_old, t2_partial, sys_len, hp_len, nhc_len):
        ops: list[KVMoveOp] = []
        template = state.newline_template_pos
        for i in range(sys_len): ops.append(KVMoveOp(old_pos=i, new_pos=i))
        old_hp_start = sys_len
        for i in range(hp_len): ops.append(KVMoveOp(old_pos=old_hp_start + i, new_pos=old_hp_start + i))
        hc_new = sys_len + hp_len
        for idx, old_pos in enumerate(history_sources): ops.append(KVMoveOp(old_pos=old_pos, new_pos=hc_new + idx))
        pad_new = hc_new + nhc_len
        for i in range(new_pad): ops.append(KVMoveOp(old_pos=template, new_pos=pad_new + i))
        if t2_partial > 0:
            t2p_new = pad_new + new_pad
            for i in range(t2_partial): ops.append(KVMoveOp(old_pos=t2_start_old + i, new_pos=t2p_new + i))
        return ops

    def _build_universal_inplace_ops(self, seq, state, new_hist_ids, new_pad, oldest, t2_start_old, t2_partial, first_complete):
        ops = []
        old_ids = state.all_token_ids()
        old_grids = []
        for turn in state.dialogue_turns:
            if turn.image_grid_thw_list:
                old_grids.extend(turn.image_grid_thw_list)
        
        image_token_id = getattr(seq, '_image_token_id', -1)
        spatial_merge = getattr(seq, '_spatial_merge_size', 2)
        
        # 引入顶部的 compute_mrope_positions
        from streamingvllm.engine.streaming_scheduler import compute_mrope_positions
        
        if old_grids and image_token_id > 0:
            old_mrope, _ = compute_mrope_positions(old_ids, old_grids, image_token_id, spatial_merge)
        else:
            import numpy as np
            old_mrope = torch.from_numpy(np.broadcast_to(np.arange(len(old_ids)), (3, len(old_ids))))

        new_ids = []
        new_ids.extend(state.system_ids)
        new_ids.extend(state.history_prefix_ids)
        new_ids.extend(new_hist_ids)
        new_ids.extend([self.newline_id] * new_pad)
        for i in range(1, len(state.dialogue_turns)):
            turn = state.dialogue_turns[i]
            new_ids.extend(turn.user_token_ids)
            new_ids.extend(turn.assistant_token_ids)
            
        new_grids = []
        for i in range(1, len(state.dialogue_turns)):
            if state.dialogue_turns[i].image_grid_thw_list:
                new_grids.extend(state.dialogue_turns[i].image_grid_thw_list)
                
        if new_grids and image_token_id > 0:
            new_mrope, _ = compute_mrope_positions(new_ids, new_grids, image_token_id, spatial_merge)
        else:
            import numpy as np
            new_mrope = torch.from_numpy(np.broadcast_to(np.arange(len(new_ids)), (3, len(new_ids))))

        def _add_op(old_p, new_p):
            # 安全检查：防止越界
            if old_p >= old_mrope.shape[1] or new_p >= new_mrope.shape[1]:
                return
                
            delta = int(new_mrope[0, new_p].item() - old_mrope[0, old_p].item())
            if delta != 0:
                ops.append(KVInplaceOp(new_pos=new_p, delta=delta))

        sys_len = state.sys_len()
        hp_len = len(state.history_prefix_ids)
        hc_new_start = sys_len + hp_len
        
        _, sources = self._plan_history_content(state, oldest)
        for idx, old_pos in enumerate(sources):
            _add_op(old_pos, hc_new_start + idx)

        pad_new_start = hc_new_start + len(new_hist_ids)
        template_pos = state.newline_template_pos
        for i in range(new_pad):
            _add_op(template_pos, pad_new_start + i)

        t2p_new_start = pad_new_start + new_pad
        for i in range(t2_partial):
            _add_op(t2_start_old + i, t2p_new_start + i)

        # 🚨 核心修复：基于 new_ids 的总长度来计算 reuse_count 🚨
        old_reuse_start = first_complete * self.block_size
        new_reuse_start = t2p_new_start + t2_partial
        
        # 因为 new_ids 包含了所有保留的 Token，所以 new_reuse_start 之后的 Token 数量就是 reuse_count
        reuse_count = len(new_ids) - new_reuse_start
        
        for i in range(reuse_count):
            _add_op(old_reuse_start + i, new_reuse_start + i)

        return ops

    @staticmethod
    def finalize_kv_ops(plan: CompactPlan, old_bt: list[int], new_bt: list[int], block_size: int):
        def _slot(pos: int, bt: list[int]) -> int:
            bi, off = divmod(pos, block_size)
            return bt[bi] * block_size + off if 0 <= bi < len(bt) else -1

        valid_moves = []
        for op in plan.move_ops:
            op.src_slot = _slot(op.old_pos, old_bt)
            op.dst_slot = _slot(op.new_pos, new_bt)
            if op.src_slot >= 0 and op.dst_slot >= 0:
                valid_moves.append(op)
        plan.move_ops = valid_moves

        valid_inplace = []
        for op in plan.inplace_ops:
            # For inplace ops, the token is already in its new physical slot 
            # (either moved there by move_ops, or it was in a reusable block).
            # So we resolve its slot using new_pos and new_bt.
            op.slot = _slot(op.new_pos, new_bt)
            if op.slot >= 0:
                valid_inplace.append(op)
        plan.inplace_ops = valid_inplace

    def apply_eviction(self, state: StreamingState, plan: CompactPlan):
        if not plan.needs_compact: return
        state.dialogue_turns.popleft()
        state.history_content_ids = plan.new_history_ids
        state.padding_ids = [self.newline_id] * plan.new_padding_count
        state.padding_count = plan.new_padding_count

        base = state.turns_start()
        for turn in state.dialogue_turns:
            tl = len(turn.user_token_ids) + len(turn.assistant_token_ids)
            turn.start_pos = base
            turn.end_pos = base + tl
            base += tl
