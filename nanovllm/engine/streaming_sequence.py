from __future__ import annotations
from enum import Enum, auto
from itertools import count
from typing import Optional
import torch
import numpy as np
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.streaming import (
    StreamingConfig, StreamingState, StreamingChatManager, CompactPlan, DialogueTurn,
    compute_mrope_positions,
)

class StreamingSequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()
    IDLE = auto()

class StreamingSequence:
    block_size = 256
    counter = count()

    def __init__(self, system_content, manager, streaming_config, sampling_params=SamplingParams()):
        self.seq_id = next(StreamingSequence.counter)
        self.status = StreamingSequenceStatus.IDLE
        self.manager = manager
        self.config = streaming_config
        self.state = manager.create_state(system_content)
        self._rebuild_tokens()
        self.num_prompt_tokens = self.num_tokens
        self.num_cached_tokens = 0
        self.num_scheduled_tokens = 0
        self.block_table: list[int] = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        self.current_turn: Optional[DialogueTurn] = None
        self.current_turn_output: list[int] = []
        self.pending_plan: Optional[CompactPlan] = None
        self._pending_vision: Optional[dict] = None
        self.vision_cache: dict[tuple[int, int], dict] = {}
        
        # 🚨 显式声明 VL 相关的属性 🚨
        self.mrope_position_delta: int = 0
        self._image_token_id: int = -1
        self._spatial_merge_size: int = 2
        self.cached_mrope: Optional[torch.Tensor] = None
        self._init_spec_decode_state()

    def __len__(self): return self.num_tokens
    def __getitem__(self, key): return self.token_ids[key]
    
    @property
    def is_finished(self): return self.status == StreamingSequenceStatus.FINISHED
    @property
    def is_idle(self): return self.status == StreamingSequenceStatus.IDLE
    @property
    def num_completion_tokens(self): return len(self.current_turn_output)
    @property
    def completion_token_ids(self): return self.current_turn_output
    @property
    def num_blocks(self): return (self.num_tokens + self.block_size - 1) // self.block_size
    @property
    def num_cached_blocks(self): return self.num_cached_tokens // self.block_size
    @property
    def last_block_num_tokens(self): return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i): return self.token_ids[i * self.block_size:(i + 1) * self.block_size]

    def plan_eviction(self):
        self.pending_plan = self.manager.plan_eviction(self, self.block_table)
        return self.pending_plan

    def add_turn(self, user_content):
        self.current_turn = self.manager.add_user_turn(self.state, user_content)
        self._rebuild_tokens()
        self.num_prompt_tokens = self.num_tokens
        self.current_turn_output = []
        self._reset_spec_decode_state()
        self.status = StreamingSequenceStatus.WAITING

    def append_token(self, token_id):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1
        self.current_turn_output.append(token_id)
        self.manager.add_assistant_token(self.state, token_id)
        self._update_spec_decode_state(token_id)

    def end_turn(self):
        self.manager.finalize_turn(self.state)
        self._rebuild_tokens()
        # VL model: exact cache to prevent re-prefilling image_pads
        self.num_cached_tokens = self.num_tokens
        self.current_turn = None
        self.pending_plan = None
        self.num_scheduled_tokens = 0
        self.status = StreamingSequenceStatus.IDLE
        self._update_cached_mrope()

    def _update_cached_mrope(self):
        all_ids = self.token_ids
        grids = []
        for turn in self.state.dialogue_turns:
            if turn.image_grid_thw_list:
                grids.extend(turn.image_grid_thw_list)
        if grids and self._image_token_id > 0:
            self.cached_mrope, _ = compute_mrope_positions(
                all_ids, grids, self._image_token_id, self._spatial_merge_size
            )
        else:
            self.cached_mrope = torch.from_numpy(
                np.broadcast_to(np.arange(len(all_ids), dtype=np.int64), (3, len(all_ids))).copy()
            )

    def finish(self):
        self.status = StreamingSequenceStatus.FINISHED
        self.vision_cache.clear()

    # ---- Vision cache ----
    def cache_vision_embeds(self, start_pos, end_pos, main_embeds, deepstack_embeds=None):
        self.vision_cache[(start_pos, end_pos)] = {
            "main": main_embeds.detach().cpu() if main_embeds.is_cuda else main_embeds.detach(),
            "deepstack": [d.detach().cpu() if d.is_cuda else d.detach() for d in deepstack_embeds] if deepstack_embeds else None,
        }

    def evict_vision_cache(self, evicted_start, evicted_end):
        to_remove = [k for k in self.vision_cache if k[0] >= evicted_start and k[1] <= evicted_end]
        for k in to_remove: del self.vision_cache[k]

    def update_vision_cache_positions(self, delta):
        if delta == 0 or not self.vision_cache: return
        new_cache = {}
        for (s, e), v in self.vision_cache.items():
            new_cache[(s + delta, e + delta)] = v
        self.vision_cache = new_cache

    def get_vision_embeds_for_range(self, start, end):
        for (s, e), v in self.vision_cache.items():
            overlap_start = max(s, start)
            overlap_end = min(e, end)
            if overlap_start < overlap_end:
                offset = overlap_start - s
                length = overlap_end - overlap_start
                main = v["main"][offset:offset + length]
                ds = [d[offset:offset + length] for d in v["deepstack"]] if v["deepstack"] else None
                return main, ds
        return None, None

    # ---- Internal ----
    def _rebuild_tokens(self):
        self.token_ids = self.state.all_token_ids()
        self.last_token = self.token_ids[-1] if self.token_ids else 0
        self.num_tokens = len(self.token_ids)

    # ---- State-machine speculative decoding ----
    def _init_spec_decode_state(self):
        self._spec_action_start_ids = list(getattr(self.manager, "spec_action_start_ids", []))
        self._spec_action_end_ids = list(getattr(self.manager, "spec_action_end_ids", []))
        self._spec_action_end_space_ids = list(getattr(self.manager, "spec_action_end_space_ids", []))
        self._spec_space_id = getattr(self.manager, "spec_space_id", -1)
        self._spec_semicolon_id = getattr(self.manager, "spec_semicolon_id", -1)
        sep_id = getattr(self.manager, "spec_sep_id", -1)
        self._spec_sep_id = sep_id if sep_id >= 0 else self._spec_semicolon_id
        self._spec_zero_id = getattr(self.manager, "spec_zero_id", -1)
        self._spec_num_key_groups = getattr(self.manager, "spec_num_key_groups", 6)
        self._spec_max_draft_tokens = 5
        self.spec_state_uncertain = not (
            self._spec_action_start_ids
            and (self._spec_action_end_ids or self._spec_action_end_space_ids)
            and self._spec_space_id >= 0 and self._spec_zero_id >= 0
            and self._spec_sep_id >= 0
        )
        self._reset_spec_decode_state()

    def _reset_spec_decode_state(self):
        self._spec_stage = "outside"
        self._spec_key_sep_count = 0
        self.spec_draft_token: Optional[int] = None
        self.spec_draft_tokens: Optional[list[int]] = None
        self.spec_draft_type: Optional[str] = None
        self.spec_scheduled = False
        self.spec_scheduled_draft_token: Optional[int] = None
        self.spec_scheduled_draft_tokens: Optional[list[int]] = None
        self.spec_scheduled_draft_type: Optional[str] = None
        self.spec_skip_reason: Optional[str] = None
        if self._spec_action_start_ids and not self.current_turn_output:
            self._set_spec_draft_sequence(self._spec_action_start_ids, "action_start_seq")

    def _suffix_matches(self, suffix: list[int]) -> bool:
        return bool(suffix) and len(self.current_turn_output) >= len(suffix) and self.current_turn_output[-len(suffix):] == suffix

    def _action_end_prefix_len(self) -> int:
        ids = self._spec_action_end_ids
        max_len = min(len(ids), len(self.current_turn_output))
        for n in range(max_len, 0, -1):
            if self.current_turn_output[-n:] == ids[:n]:
                return n
        return 0

    def _action_end_space_prefix_len(self) -> int:
        ids = self._spec_action_end_space_ids
        max_len = min(len(ids), len(self.current_turn_output))
        for n in range(max_len, 0, -1):
            if self.current_turn_output[-n:] == ids[:n]:
                return n
        return 0

    def _action_start_prefix_len(self) -> int:
        ids = self._spec_action_start_ids
        n = len(self.current_turn_output)
        if ids and 0 < n < len(ids) and self.current_turn_output == ids[:n]:
            return n
        return 0

    def _set_spec_draft(self, token_id: int, draft_type: str):
        if token_id is not None and token_id >= 0:
            self.spec_draft_token = int(token_id)
            self.spec_draft_tokens = [int(token_id)]
            self.spec_draft_type = draft_type

    def _set_spec_draft_sequence(self, token_ids: list[int], draft_type: str):
        draft = [int(t) for t in token_ids if t is not None and t >= 0]
        if draft:
            draft = draft[:self._spec_max_draft_tokens]
            self.spec_draft_token = draft[0]
            self.spec_draft_tokens = draft
            self.spec_draft_type = draft_type

    def _is_separator(self, token_id: int) -> bool:
        return token_id in {self._spec_sep_id, self._spec_semicolon_id}

    def _draft_action_end(self):
        raw_ids = self._spec_action_end_ids
        space_ids = self._spec_action_end_space_ids
        if not raw_ids and not space_ids:
            return
        raw_prefix_len = self._action_end_prefix_len()
        space_prefix_len = self._action_end_space_prefix_len()
        if (raw_ids and raw_prefix_len >= len(raw_ids)) or (space_ids and space_prefix_len >= len(space_ids)):
            self._spec_stage = "outside"
            return

        if space_prefix_len > 0:
            self._set_spec_draft_sequence(space_ids[space_prefix_len:], "action_end_seq")
        elif raw_prefix_len > 0:
            self._set_spec_draft_sequence(raw_ids[raw_prefix_len:], "action_end_seq")
        elif self._last_output_text_endswith_space():
            self._set_spec_draft_sequence(raw_ids, "action_end_seq")
        else:
            self._set_spec_draft_sequence(space_ids or raw_ids, "action_end_seq")

    def _last_output_text_endswith_space(self) -> bool:
        if not self.current_turn_output:
            return False
        try:
            text = self.manager.tokenizer.decode(
                [self.current_turn_output[-1]],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            text = self.manager.tokenizer.decode([self.current_turn_output[-1]], skip_special_tokens=False)
        return text.endswith(" ")

    def _update_spec_decode_state(self, token_id: int):
        self.spec_draft_token = None
        self.spec_draft_tokens = None
        self.spec_draft_type = None
        if self.spec_state_uncertain:
            return

        if self._spec_stage == "outside":
            if self._suffix_matches(self._spec_action_start_ids):
                self._spec_stage = "coord_x"
            else:
                prefix_len = self._action_start_prefix_len()
                if prefix_len:
                    self._set_spec_draft_sequence(
                        self._spec_action_start_ids[prefix_len:],
                        "action_start_seq",
                    )
            return

        if self._spec_stage == "coord_x":
            if token_id == self._spec_space_id:
                self._spec_stage = "coord_y"
            else:
                self._set_spec_draft(self._spec_space_id, "space")
            return

        if self._spec_stage == "coord_y":
            if token_id == self._spec_space_id:
                self._spec_stage = "coord_z"
                self._set_spec_draft(self._spec_zero_id, "z_zero")
            else:
                self._set_spec_draft(self._spec_space_id, "space")
            return

        if self._spec_stage == "coord_z":
            if self._is_separator(token_id):
                self._spec_stage = "key_group"
                self._spec_key_sep_count = 1
                if self._spec_key_sep_count >= self._spec_num_key_groups:
                    self._spec_stage = "action_end"
                    self._draft_action_end()
                else:
                    self._set_spec_draft(self._spec_sep_id, "empty_key_group")
            else:
                self._set_spec_draft(self._spec_sep_id, "semicolon")
            return

        if self._spec_stage == "key_group":
            if self._is_separator(token_id):
                self._spec_key_sep_count += 1
                if self._spec_key_sep_count >= self._spec_num_key_groups:
                    self._spec_stage = "action_end"
                    self._draft_action_end()
                else:
                    self._set_spec_draft(self._spec_sep_id, "empty_key_group")
            else:
                self._set_spec_draft(self._spec_sep_id, "semicolon")
            return

        if self._spec_stage == "action_end":
            self._draft_action_end()

    def __getstate__(self):
        last = self.token_ids if self.num_completion_tokens == 0 or self.num_cached_tokens < self.num_tokens else self.last_token
        # 🚨 必须包含 mrope_position_delta，否则 TP Worker 的 Decode 位置会错乱！ 🚨
        return (self.seq_id, self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens,
                self.num_scheduled_tokens, self.block_table, last, self.mrope_position_delta,
                self.spec_draft_token, self.spec_draft_type, self.spec_scheduled,
                self.spec_scheduled_draft_token, self.spec_scheduled_draft_type, self.spec_skip_reason,
                self.spec_draft_tokens, self.spec_scheduled_draft_tokens)

    def __setstate__(self, s):
        # 🚨 对应解析 🚨
        if len(s) == 8:
            (self.seq_id, self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens,
             self.num_scheduled_tokens, self.block_table, last, self.mrope_position_delta) = s
            self.spec_draft_token = None
            self.spec_draft_tokens = None
            self.spec_draft_type = None
            self.spec_scheduled = False
            self.spec_scheduled_draft_token = None
            self.spec_scheduled_draft_tokens = None
            self.spec_scheduled_draft_type = None
            self.spec_skip_reason = None
        elif len(s) == 14:
            (self.seq_id, self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens,
             self.num_scheduled_tokens, self.block_table, last, self.mrope_position_delta,
             self.spec_draft_token, self.spec_draft_type, self.spec_scheduled,
             self.spec_scheduled_draft_token, self.spec_scheduled_draft_type, self.spec_skip_reason) = s
            self.spec_draft_tokens = [self.spec_draft_token] if self.spec_draft_token is not None else None
            self.spec_scheduled_draft_tokens = (
                [self.spec_scheduled_draft_token]
                if self.spec_scheduled_draft_token is not None
                else None
            )
        else:
            (self.seq_id, self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens,
             self.num_scheduled_tokens, self.block_table, last, self.mrope_position_delta,
             self.spec_draft_token, self.spec_draft_type, self.spec_scheduled,
             self.spec_scheduled_draft_token, self.spec_scheduled_draft_type, self.spec_skip_reason,
             self.spec_draft_tokens, self.spec_scheduled_draft_tokens) = s
        if isinstance(last, list):
            self.token_ids = last
            self.last_token = last[-1] if last else 0
        else:
            self.token_ids = []
            self.last_token = last
