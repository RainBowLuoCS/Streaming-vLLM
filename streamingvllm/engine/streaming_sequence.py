from __future__ import annotations
from enum import Enum, auto
from itertools import count
from typing import Optional
import torch
from streamingvllm.sampling_params import SamplingParams
from streamingvllm.engine.streaming import (
    StreamingConfig, StreamingState, StreamingChatManager, CompactPlan, DialogueTurn,
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
        self.status = StreamingSequenceStatus.WAITING

    def append_token(self, token_id):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1
        self.current_turn_output.append(token_id)
        self.manager.add_assistant_token(self.state, token_id)

    def end_turn(self):
        self.manager.finalize_turn(self.state)
        self._rebuild_tokens()
        # VL model: exact cache to prevent re-prefilling image_pads
        self.num_cached_tokens = self.num_tokens
        self.current_turn = None
        self.pending_plan = None
        self.num_scheduled_tokens = 0
        self.status = StreamingSequenceStatus.IDLE

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

    def __getstate__(self):
        last = self.token_ids if self.num_completion_tokens == 0 or self.num_cached_tokens < self.num_tokens else self.last_token
        # 🚨 必须包含 mrope_position_delta，否则 TP Worker 的 Decode 位置会错乱！ 🚨
        return (self.seq_id, self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens,
                self.num_scheduled_tokens, self.block_table, last, self.mrope_position_delta)

    def __setstate__(self, s):
        # 🚨 对应解析 🚨
        self.seq_id, self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last, self.mrope_position_delta = s
        if isinstance(last, list):
            self.token_ids = last
            self.last_token = last[-1] if last else 0
        else:
            self.token_ids = []
            self.last_token = last
