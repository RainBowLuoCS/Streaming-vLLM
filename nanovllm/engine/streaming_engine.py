from __future__ import annotations
import atexit
from dataclasses import fields
from typing import Optional, Generator, List, Dict
from transformers import AutoTokenizer
import torch
import time
import numpy as np
import torch.multiprocessing as mp
from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.streaming_sequence import StreamingSequence, StreamingSequenceStatus
from nanovllm.engine.streaming_scheduler import StreamingScheduler, BatchKVOps, compute_mrope_positions
from nanovllm.engine.streaming import StreamingConfig, StreamingChatManager
from nanovllm.engine.model_runner import ModelRunner

SPEC_TIMING_DEFAULTS = (
    "normal_graph_replays",
    "spec_graph_replays",
    "eager_forwards",
    "spec_attempts",
    "spec_accepts",
    "spec_rejects",
    "spec_skips",
    "spec_space_attempts",
    "spec_space_accepts",
    "spec_semicolon_attempts",
    "spec_semicolon_accepts",
    "spec_z_zero_attempts",
    "spec_z_zero_accepts",
    "spec_empty_key_group_attempts",
    "spec_empty_key_group_accepts",
    "spec_action_start_seq_attempts",
    "spec_action_start_seq_accepts",
    "spec_action_start_seq_token_accepts",
    "spec_action_start_seq_full_accepts",
    "spec_action_start_seq_partial_accepts",
    "spec_action_end_seq_attempts",
    "spec_action_end_seq_accepts",
    "spec_action_end_seq_token_accepts",
    "spec_action_end_seq_full_accepts",
    "spec_action_end_seq_partial_accepts",
    "fallback_no_draft",
    "fallback_near_max_tokens",
    "fallback_kv_slot_insufficient",
    "fallback_unsupported_shape",
    "fallback_state_uncertain",
)


class StreamingLLMEngine:
    def __init__(self, model: str, streaming_config: Optional[StreamingConfig] = None, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config = Config(model, **{k: v for k, v in kwargs.items() if k in config_fields})
        self.streaming_config = streaming_config or StreamingConfig()
        StreamingSequence.block_size = config.kvcache_block_size
        
        self.ps, self.events = [], []
        ctx = mp.get_context("spawn")
        for i in range(1, config.world_size):
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
        self._last_timing: Dict[int, dict] = {}

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

        if self.is_vl:
            seq._image_token_id = self.image_token_id
            seq._spatial_merge_size = self.spatial_merge_size

        # 🚨 hybrid (Qwen3.5): allocate a GDN state slot for this session
        if getattr(self.config, "is_hybrid", False):
            self.model_runner.call("gdn_alloc", seq.seq_id)

        return seq.seq_id

    def close_session(self, sid):
        if sid in self.sessions:
            # 🚨 free GDN slot
            if getattr(self.config, "is_hybrid", False):
                self.model_runner.call("gdn_free", sid)
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
        return message + "\n" + "\n".join(parts)

    # ---- Chat API ----

    def chat(self, session_id, message, stream=False,
             pixel_values=None, image_grid_thw=None, return_timing=False):
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")

        chat_timing = {} if return_timing else None

        if pixel_values is not None and self.is_vl:
            t0 = time.perf_counter()
            message = self._build_vision_message(message, image_grid_thw)
            if return_timing:
                chat_timing["build_msg_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        seq = self.scheduler.continue_session(session_id, message, _timing=chat_timing)
        if return_timing:
            chat_timing["continue_session_ms"] = (time.perf_counter() - t0) * 1000

        if seq is None:
            raise ValueError(f"Cannot continue session {session_id}")

        if return_timing:
            for attr in ('_plan_eviction_debug', '_inplace_vl_debug'):
                debug = getattr(seq, attr, None)
                if debug:
                    chat_timing.update(debug)

        if pixel_values is not None and self.is_vl:
            seq._pending_vision = {"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}
            if seq.current_turn:
                grids = image_grid_thw.tolist() if isinstance(image_grid_thw, torch.Tensor) else image_grid_thw
                seq.current_turn.image_grid_thw_list = [tuple(g) for g in grids]

        if stream:
            return self._stream(seq)
        if return_timing:
            return self._generate_with_timing(seq, _chat_timing=chat_timing)
        return self._generate(seq)

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
                        s.completion_token_ids, skip_special_tokens=False
                    )
        return [results.get(r["session_id"], "") for r in requests]

    # ---- Generation helpers ----

    def _generate(self, seq):
        while not seq.is_idle and not seq.is_finished:
            self._step()
        return self.tokenizer.decode(seq.completion_token_ids, skip_special_tokens=False)

    def _generate_with_timing(self, seq, _chat_timing=None):
        prefill_ms = decode_ms = 0.0
        prefill_steps = decode_steps = 0
        prefill_tokens = decode_tokens = 0
        step_timing = {}

        t_start = time.perf_counter()
        while not seq.is_idle and not seq.is_finished:
            t0 = time.perf_counter()
            result = self._step(_timing=step_timing)
            step_ms = (time.perf_counter() - t0) * 1000
            if result is not None:
                is_pf, n_tokens = result
                if is_pf:
                    prefill_ms += step_ms
                    prefill_steps += 1
                    prefill_tokens += n_tokens
                else:
                    decode_ms += step_ms
                    decode_steps += 1
                    decode_tokens += n_tokens
        model_ms = (time.perf_counter() - t_start) * 1000

        t0 = time.perf_counter()
        text = self.tokenizer.decode(seq.completion_token_ids, skip_special_tokens=False)
        tokenizer_decode_ms = (time.perf_counter() - t0) * 1000

        timing = {
            "model_ms": model_ms,
            "prefill_ms": prefill_ms,
            "decode_ms": decode_ms,
            "prefill_steps": prefill_steps,
            "decode_steps": decode_steps,
            "prefill_tokens": prefill_tokens,
            "decode_tokens": decode_tokens,
            "tokenizer_decode_ms": tokenizer_decode_ms,
            **step_timing,
        }
        self._finalize_timing(timing)
        if _chat_timing:
            timing.update(_chat_timing)
        self._last_timing[seq.seq_id] = dict(timing)
        return text, timing

    def _stream(self, seq):
        prev = 0
        prefill_ms = decode_ms = 0.0
        prefill_steps = decode_steps = 0
        prefill_tokens = decode_tokens = 0
        step_timing = {}
        t_start = time.perf_counter()
        while not seq.is_idle and not seq.is_finished:
            t0 = time.perf_counter()
            result = self._step(_timing=step_timing)
            step_ms = (time.perf_counter() - t0) * 1000
            if result is not None:
                is_pf, n_tokens = result
                if is_pf:
                    prefill_ms += step_ms
                    prefill_steps += 1
                    prefill_tokens += n_tokens
                else:
                    decode_ms += step_ms
                    decode_steps += 1
                    decode_tokens += n_tokens
            cur = seq.completion_token_ids
            if len(cur) > prev:
                text = self.tokenizer.decode(cur[prev:], skip_special_tokens=False)
                if text:
                    yield text
                prev = len(cur)
        timing = {
            "model_ms": (time.perf_counter() - t_start) * 1000,
            "prefill_ms": prefill_ms,
            "decode_ms": decode_ms,
            "prefill_steps": prefill_steps,
            "decode_steps": decode_steps,
            "prefill_tokens": prefill_tokens,
            "decode_tokens": decode_tokens,
            **step_timing,
        }
        self._finalize_timing(timing)
        self._last_timing[seq.seq_id] = timing

    # ---- Core step ----
    def _bump_timing(self, timing, key, value=1):
        if timing is not None:
            timing[key] = timing.get(key, 0) + value

    def _record_spec_fallbacks(self, seqs, timing):
        if timing is None or self.config.spec_decode_mode != "graph2":
            return
        for seq in seqs:
            if getattr(seq, "spec_scheduled", False):
                continue
            reason = getattr(seq, "spec_skip_reason", None)
            if reason:
                self._bump_timing(timing, "spec_skips")
                self._bump_timing(timing, f"fallback_{reason}")

    def _merge_timing_stats(self, timing, stats):
        if timing is None or not stats:
            return
        for k, v in stats.items():
            timing[k] = timing.get(k, 0) + v

    def _finalize_timing(self, timing):
        for key in SPEC_TIMING_DEFAULTS:
            timing.setdefault(key, 0)
        attempts = timing.get("spec_attempts", 0)
        accepts = timing.get("spec_accepts", 0)
        decode_steps = timing.get("decode_steps", 0)
        decode_tokens = timing.get("decode_tokens", 0)
        timing["spec_accept_rate"] = accepts / attempts if attempts else 0.0
        timing["tokens_per_step"] = decode_tokens / decode_steps if decode_steps else 0.0

    def _step(self, _timing=None):
        t0 = time.perf_counter()
        seqs, is_prefill, kv_ops = self.scheduler.schedule()
        if _timing is not None:
            _timing["schedule_ms"] = _timing.get("schedule_ms", 0.0) + (time.perf_counter() - t0) * 1000

        if not seqs:
            return None
        total_scheduled = sum(s.num_scheduled_tokens for s in seqs)
        if not is_prefill:
            self._record_spec_fallbacks(seqs, _timing)

        vision_data = None
        mrope_positions = None

        if is_prefill and self.is_vl:
            t0 = time.perf_counter()
            vision_data = self._prepare_vision_data(seqs)
            if _timing is not None:
                _timing["vision_data_ms"] = _timing.get("vision_data_ms", 0.0) + (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            mrope_positions = self._compute_batch_mrope(seqs)
            if _timing is not None:
                _timing["mrope_ms"] = _timing.get("mrope_ms", 0.0) + (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        runner_result = self.model_runner.call(
            "run_streaming", seqs, is_prefill, kv_ops, vision_data, mrope_positions, _timing is not None
        )
        if isinstance(runner_result, tuple):
            tids, runner_stats = runner_result
        else:
            tids, runner_stats = runner_result, {}
        self._merge_timing_stats(_timing, runner_stats)
        if _timing is not None:
            key = "model_forward_pf_ms" if is_prefill else "model_forward_dc_ms"
            _timing[key] = _timing.get(key, 0.0) + (time.perf_counter() - t0) * 1000

        # 🚨 修正：利用统一的 compute_mrope_positions 更新 Delta 🚨
        t0 = time.perf_counter()
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
        if _timing is not None:
            _timing["mrope_delta_ms"] = _timing.get("mrope_delta_ms", 0.0) + (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        appended, post_stats = self.scheduler.postprocess(seqs, tids, is_prefill)
        self._merge_timing_stats(_timing, post_stats)
        if _timing is not None:
            _timing["postprocess_ms"] = _timing.get("postprocess_ms", 0.0) + (time.perf_counter() - t0) * 1000

        return is_prefill, total_scheduled if is_prefill else appended

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

    def get_last_timing(self, sid):
        return dict(self._last_timing.get(sid, {}))


class StreamingLLM(StreamingLLMEngine):
    pass
