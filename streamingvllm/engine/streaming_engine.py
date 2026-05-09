from __future__ import annotations
import atexit
from dataclasses import fields
from typing import Optional, Generator, List, Dict
from transformers import AutoTokenizer
import torch
import numpy as np
import torch.multiprocessing as mp
from streamingvllm.config import Config
from streamingvllm.sampling_params import SamplingParams
from streamingvllm.engine.streaming_sequence import StreamingSequence, StreamingSequenceStatus
from streamingvllm.engine.streaming_scheduler import StreamingScheduler, BatchKVOps, compute_mrope_positions
from streamingvllm.engine.streaming import StreamingConfig, StreamingChatManager
from streamingvllm.engine.model_runner import ModelRunner


class StreamingLLMEngine:
    def __init__(self, model: str, streaming_config: Optional[StreamingConfig] = None, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config = Config(model, **{k: v for k, v in kwargs.items() if k in config_fields})
        self.streaming_config = streaming_config or StreamingConfig()
        StreamingSequence.block_size = config.kvcache_block_size
        
        self.ps, self.events = [], []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            p = ctx.Process(target=ModelRunner, args=(config, i, event))
            p.start()
            self.ps.append(p)
            self.events.append(event)
            
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        
        self.chat_manager = StreamingChatManager(
            self.streaming_config, config.kvcache_block_size, self.tokenizer
        )
        self.scheduler = StreamingScheduler(config, self.streaming_config)
        self.config = config
        self.sessions: Dict[int, StreamingSequence] = {}

        # VL detection
        self.is_vl = config.is_multimodal and hasattr(self.model_runner.model, 'visual')
        if self.is_vl:
            hf = config.hf_config
            self.image_token_id = getattr(hf, 'image_token_id', None)
            self.vision_config = getattr(hf, 'vision_config', None)
            self.spatial_merge_size = self.vision_config.spatial_merge_size if self.vision_config else 2
            
        atexit.register(self.exit)

    def exit(self):
        self.close_all()
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    # ---- Session management ----

    def create_session(self, system_prompt="You are a helpful assistant.",
                       sampling_params=SamplingParams()) -> int:
        seq = StreamingSequence(system_prompt, self.chat_manager, self.streaming_config, sampling_params)
        self.sessions[seq.seq_id] = seq
        self.scheduler.add_session(seq)
        
        # Store VL info on sequence for scheduler use
        if self.is_vl:
            seq._image_token_id = self.image_token_id
            seq._spatial_merge_size = self.spatial_merge_size
            
        return seq.seq_id

    def close_session(self, sid):
        if sid in self.sessions:
            self.scheduler.close_session(sid)
            del self.sessions[sid]

    def close_all(self):
        for sid in list(self.sessions):
            self.close_session(sid)

    # ---- Vision message building ----

    def _build_vision_message(self, message, image_grid_thw):
        """Inject image placeholders into user message."""
        if image_grid_thw is None:
            return message
        grids = image_grid_thw.tolist() if isinstance(image_grid_thw, torch.Tensor) else image_grid_thw
        parts = []
        for g in grids:
            t, h, w = g if len(g) == 3 else (1, g[0], g[1])
            n = int(t * h * w) // (self.spatial_merge_size ** 2)
            parts.append(f"<|vision_start|>{'<|image_pad|>' * n}<|vision_end|>")
        return "\n".join(parts) + "\n" + message

    # ---- Chat API ----

    def chat(self, session_id, message, stream=False,
             pixel_values=None, image_grid_thw=None):
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
            
        if pixel_values is not None and self.is_vl:
            message = self._build_vision_message(message, image_grid_thw)
            
        seq = self.scheduler.continue_session(session_id, message)
        if seq is None:
            raise ValueError(f"Cannot continue session {session_id}")
            
        if pixel_values is not None and self.is_vl:
            seq._pending_vision = {"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}
            if seq.current_turn:
                grids = image_grid_thw.tolist() if isinstance(image_grid_thw, torch.Tensor) else image_grid_thw
                seq.current_turn.image_grid_thw_list = [tuple(g) for g in grids]
                
        return self._stream(seq) if stream else self._generate(seq)

    def chat_batch(self, requests):
        active = []
        for req in requests:
            message = req["message"]
            pv = req.get("pixel_values")
            thw = req.get("image_grid_thw")
            
            if pv is not None and self.is_vl:
                message = self._build_vision_message(message, thw)
                
            seq = self.scheduler.continue_session(req["session_id"], message)
            if seq:
                if pv is not None and self.is_vl:
                    seq._pending_vision = {"pixel_values": pv, "image_grid_thw": thw}
                    if seq.current_turn:
                        grids = thw.tolist() if isinstance(thw, torch.Tensor) else thw
                        seq.current_turn.image_grid_thw_list = [tuple(g) for g in grids]
                active.append(seq)
                
        results = {}
        while any(not s.is_idle and not s.is_finished for s in active):
            self._step()
            for s in active:
                if (s.is_idle or s.is_finished) and s.seq_id not in results:
                    results[s.seq_id] = self.tokenizer.decode(
                        s.completion_token_ids, skip_special_tokens=True
                    )
        return [results.get(r["session_id"], "") for r in requests]

    # ---- Generation helpers ----

    def _generate(self, seq):
        while not seq.is_idle and not seq.is_finished:
            self._step()
        return self.tokenizer.decode(seq.completion_token_ids, skip_special_tokens=True)

    def _stream(self, seq):
        prev = 0
        while not seq.is_idle and not seq.is_finished:
            self._step()
            cur = seq.completion_token_ids
            if len(cur) > prev:
                text = self.tokenizer.decode(cur[prev:], skip_special_tokens=True)
                if text:
                    yield text
                prev = len(cur)

    # ---- Core step ----

    def _step(self):
        seqs, is_prefill, kv_ops = self.scheduler.schedule()
        if not seqs:
            return

        vision_data = None
        mrope_positions = None

        if is_prefill and self.is_vl:
            vision_data = self._prepare_vision_data(seqs)
            mrope_positions = self._compute_batch_mrope(seqs)

        tids = self.model_runner.call(
            "run_streaming", seqs, is_prefill, kv_ops, vision_data, mrope_positions
        )

        # 🚨 修正：利用统一的 compute_mrope_positions 更新 Delta 🚨
        if is_prefill and self.is_vl and mrope_positions is not None:
            for seq in seqs:
                n = seq.num_scheduled_tokens
                if n > 0:
                    # 重新计算该序列的全局 Delta
                    all_grids = []
                    for turn in seq.state.dialogue_turns:
                        if turn.image_grid_thw_list:
                            all_grids.extend(turn.image_grid_thw_list)
                            
                    if all_grids:
                        full_ids = seq.state.all_token_ids()
                        _, delta = compute_mrope_positions(
                            full_ids, all_grids, self.image_token_id, self.spatial_merge_size
                        )
                        seq.mrope_position_delta = delta
                    else:
                        seq.mrope_position_delta = 0

        self.scheduler.postprocess(seqs, tids, is_prefill)

    # ---- Vision data preparation ----

    def _prepare_vision_data(self, seqs):
        """
        Prepare raw vision data (pixel_values, masks, cached embeddings) for model_runner.
        Strictly distinguishes between new image tokens and orphaned old image tokens.
        """
        import torch

        new_pv, new_thw = [], []
        new_vision_seq_ids = set()

        for seq in seqs:
            v = getattr(seq, '_pending_vision', None)
            if v and v.get("pixel_values") is not None:
                new_pv.append(v["pixel_values"])
                thw = v["image_grid_thw"]
                new_thw.append(thw if isinstance(thw, torch.Tensor) else torch.tensor(thw, dtype=torch.int32))
                new_vision_seq_ids.add(seq.seq_id)
                seq._pending_vision = None

        prefill_ids = []
        mask_list = []
        all_cached_main = []
        all_cached_ds = {}
        cache_ops = []

        for seq in seqs:
            start = seq.num_cached_tokens
            end = start + seq.num_scheduled_tokens
            seq_tokens = seq[start:end]
            prefill_ids.extend(seq_tokens)

            current_turn_start = seq.current_turn.start_pos if seq.current_turn else float('inf')

            i = 0
            while i < len(seq_tokens):
                if seq_tokens[i] == self.image_token_id:
                    span_start = i
                    while i < len(seq_tokens) and seq_tokens[i] == self.image_token_id:
                        i += 1
                    span_len = i - span_start
                    abs_start = start + span_start
                    abs_end = start + i

                    cached_main, cached_ds = seq.get_vision_embeds_for_range(abs_start, abs_end)
                    is_new_turn_token = (abs_start >= current_turn_start)

                    if cached_main is not None:
                        all_cached_main.append(cached_main)
                        if cached_ds:
                            for level, d in enumerate(cached_ds):
                                all_cached_ds.setdefault(level, []).append(d)
                        mask_list.extend([True] * span_len)
                        
                    elif is_new_turn_token:
                        mask_list.extend([True] * span_len)
                        cache_ops.append({
                            "seq": seq, 
                            "abs_start": abs_start,
                            "abs_end": abs_end, 
                            "length": span_len, 
                            "is_new": True,
                        })
                        
                    else:
                        mask_list.extend([False] * span_len)
                else:
                    mask_list.append(False)
                    i += 1

        has_new = len(new_pv) > 0
        has_cached = len(all_cached_main) > 0

        if not has_new and not has_cached:
            return None

        ds_num = len(self.vision_config.deepstack_visual_indexes) if self.vision_config else 0
        ds_indices = list(self.vision_config.deepstack_visual_indexes) if ds_num > 0 else None
        visual_dim = self.vision_config.out_hidden_size * (1 + ds_num) if self.vision_config else 0

        return {
            "pixel_values": torch.cat(new_pv, dim=0) if new_pv else None,
            "image_grid_thw": torch.cat(new_thw, dim=0) if new_thw else None,
            "image_token_id": self.image_token_id,
            "mask": mask_list,
            "cached_main_embeds": torch.cat(all_cached_main, dim=0) if all_cached_main else None,
            "cached_ds_embeds": {k: torch.cat(v, dim=0) for k, v in all_cached_ds.items()} if all_cached_ds else None,
            "deepstack_indices": ds_indices,
            "visual_dim": visual_dim,
            "_cache_ops": cache_ops,
        }

    def _compute_batch_mrope(self, seqs):
        """Use decoupled compute_mrope_positions for exact position calculation."""
        import torch
        import numpy as np

        parts = []
        for seq in seqs:
            start = seq.num_cached_tokens
            end = start + seq.num_scheduled_tokens
            if seq.num_scheduled_tokens <= 0:
                continue

            seq_grids = []
            for turn in seq.state.dialogue_turns:
                if turn.image_grid_thw_list:
                    seq_grids.extend(turn.image_grid_thw_list)

            if seq_grids:
                full_ids = seq.token_ids[:end]
                pos, _ = compute_mrope_positions(
                    full_ids, seq_grids, self.image_token_id, self.spatial_merge_size
                )
                parts.append(pos[:, start:end])
            else:
                text_pos = np.broadcast_to(np.arange(start, end), (3, end - start))
                parts.append(torch.from_numpy(text_pos.copy()))

        if not parts:
            return None
        return torch.cat(parts, dim=1)

    def get_stats(self, sid):
        if sid not in self.sessions:
            return {}
        seq = self.sessions[sid]
        st = seq.state
        return dict(
            session_id=sid,
            total_tokens=st.total_tokens(),
            cached_tokens=seq.num_cached_tokens,
            system_tokens=st.sys_len(),
            history_tokens=len(st.history_content_ids),
            padding_tokens=st.padding_count,
            dialogue_turns=len(st.dialogue_turns),
            blocks=len(seq.block_table),
            vision_cache_entries=len(seq.vision_cache),
        )


class StreamingLLM(StreamingLLMEngine):
    pass
