import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory
from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context, init_parallel_state, get_pstate
from nanovllm.utils.loader import load_model


class ModelRunner:
    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.world_size
        self.rank = rank
        self.event = event
        self.is_hybrid = config.is_hybrid
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        init_parallel_state(config.tensor_parallel_size, config.expert_parallel_size, rank, self.world_size)

        default_dtype = torch.get_default_dtype()
        text_config = getattr(hf_config, "text_config", hf_config)
        self.is_moe = getattr(text_config, "num_experts", 0) > 0
        model_dtype = getattr(text_config, "dtype", getattr(hf_config, "dtype", getattr(hf_config, "torch_dtype", "bfloat16")))
        if isinstance(model_dtype, str):
            model_dtype = getattr(torch, model_dtype, torch.float16)
        torch.set_default_dtype(model_dtype)
        torch.set_default_device("cuda")

        if self.is_hybrid:
            from nanovllm.models.qwen3_5 import load_qwen3_5_model
            self.model = load_qwen3_5_model(config.model, config)
        elif config.is_multimodal:
            from nanovllm.models.qwen3_vl import load_qwen3_vl_model
            self.model = load_qwen3_vl_model(config.model, config)
        else:
            if self.is_moe:
                from nanovllm.models.qwen3_moe import Qwen3MoeForCausalLM
                self.model = Qwen3MoeForCausalLM(text_config)
            else:
                from nanovllm.models.qwen3 import Qwen3ForCausalLM
                self.model = Qwen3ForCausalLM(text_config)
            load_model(self.model, config.model)

        self.sampler = Sampler()

        if self.is_hybrid:
            self._init_gdn_pool()

        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                try:
                    old = SharedMemory(name="nanovllm"); old.close(); old.unlink()
                except FileNotFoundError:
                    pass
                self.shm = SharedMemory(name="nanovllm", create=True, size=config.shm_size)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    # ================= GDN pool =================
    def _init_gdn_pool(self):
        from nanovllm.engine.gdn_state_pool import GDNStatePool
        text_config = getattr(self.config.hf_config, "text_config", self.config.hf_config)
        tp = get_pstate().tp_size
        num_slots = self.config.max_num_seqs + 1   # +1 garbage slot for graph padding
        self._gdn_garbage_slot = num_slots - 1
        NL = self.config.num_linear_layers
        conv_dim = (text_config.linear_key_head_dim * text_config.linear_num_key_heads * 2
                    + text_config.linear_value_head_dim * text_config.linear_num_value_heads) // tp
        kernel = text_config.linear_conv_kernel_dim
        num_v = text_config.linear_num_value_heads // tp
        hv = text_config.linear_value_head_dim
        hk = text_config.linear_key_head_dim
        model_dtype = torch.get_default_dtype()
        self.gdn_pool = GDNStatePool(num_slots, NL, conv_dim, kernel, num_v, hv, hk, model_dtype, "cuda")

        # attach conv/ssm pool to each linear layer
        tm = self.model.model  # Qwen3_5ModelPool
        gdn_idx = 0
        for layer in tm.layers:
            if hasattr(layer, "linear_attn"):
                layer.linear_attn._conv_pool = self.gdn_pool.conv_pool[gdn_idx]
                layer.linear_attn._ssm_pool = self.gdn_pool.ssm_pool[gdn_idx]
                gdn_idx += 1

    def gdn_alloc(self, seq_id):
        return self.gdn_pool.alloc(seq_id)

    def gdn_free(self, seq_id):
        self.gdn_pool.free(seq_id)

    def exit(self):
        if self.world_size > 1:
            self.shm.close(); dist.barrier()
            if self.rank == 0: self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize(); dist.destroy_process_group()

    def loop(self):
        while True:
            name, args = self.read_shm()
            self.call(name, *args)
            if name == "exit": break

    def read_shm(self):
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return name, args

    def write_shm(self, name, *args):
        data = pickle.dumps([name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for e in self.event: e.set()

    def call(self, name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(name, *args)
        return getattr(self, name)(*args)

    def warmup_model(self):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        if self.is_hybrid:
            return
        sl = min(self.config.max_num_batched_tokens, self.config.max_model_len)
        ns = min(self.config.max_num_batched_tokens // sl, self.config.max_num_seqs)
        seqs = [Sequence([0]*sl) for _ in range(ns)]
        for s in seqs: s.num_scheduled_tokens = sl
        self.run(seqs, True); torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        text_config = getattr(hf_config, "text_config", hf_config)
        tp_size = get_pstate().tp_size
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = text_config.num_key_value_heads // tp_size
        head_dim = getattr(text_config, "head_dim", text_config.hidden_size // text_config.num_attention_heads)
        dtype = getattr(text_config, "dtype", torch.float16)
        if isinstance(dtype, str): dtype = getattr(torch, dtype, torch.float16)
        num_kv_layers = config.num_full_layers if self.is_hybrid else text_config.num_hidden_layers
        block_bytes = 2 * num_kv_layers * self.block_size * num_kv_heads * head_dim * dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(
            2, num_kv_layers, config.num_kvcache_blocks,
            self.block_size, num_kv_heads, head_dim, dtype=dtype, device="cuda")
        layer_id = 0
        for m in self.model.modules():
            if hasattr(m, "k_cache") and hasattr(m, "v_cache"):
                m.k_cache = self.kv_cache[0, layer_id]
                m.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1
        assert layer_id == num_kv_layers, f"KV layer mismatch: {layer_id} vs {num_kv_layers}"

    def prepare_block_tables(self, seqs):
        mx = max(len(s.block_table) for s in seqs)
        bt = [s.block_table + [-1]*(mx - len(s.block_table)) for s in seqs]
        return torch.tensor(bt, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

    def prepare_prefill(self, seqs):
        input_ids, positions, cu_q, cu_k = [], [], [0], [0]
        max_q = max_k = 0
        slot_mapping = []
        is_vl = self.config.is_multimodal
        for seq in seqs:
            sl = len(seq)
            start = min(seq.num_cached_tokens, sl - 1)
            sq = seq.num_scheduled_tokens
            if sq <= 0:
                continue
            sk = sl
            end = start + sq
            input_ids.extend(seq[start:end])
            if is_vl:
                for p in range(start, end):
                    positions.append([p, p, p])
            else:
                positions.extend(range(start, end))
            cu_q.append(cu_q[-1] + sq)
            cu_k.append(cu_k[-1] + sk)
            max_q = max(sq, max_q)
            max_k = max(sk, max_k)
            if not seq.block_table:
                continue
            sb = start // self.block_size
            eb = (end + self.block_size - 1) // self.block_size
            for i in range(sb, eb):
                ss = seq.block_table[i] * self.block_size
                if i == sb:
                    ss += start % self.block_size
                if i == eb - 1:
                    block_end_pos = min(end, (i + 1) * self.block_size)
                    se = seq.block_table[i] * self.block_size + (block_end_pos - i * self.block_size)
                else:
                    se = seq.block_table[i] * self.block_size + self.block_size
                slot_mapping.extend(range(ss, se))
        block_tables = self.prepare_block_tables(seqs) if cu_k[-1] > cu_q[-1] else None
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        if is_vl:
            positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True).t().contiguous()
        else:
            positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_q = torch.tensor(cu_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_k = torch.tensor(cu_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        # 🚨 GDN prefill args into context
        gdn_slots = gdn_cu = gdn_hi = None
        if self.is_hybrid:
            gdn_slots, gdn_cu, gdn_hi = self._build_gdn_prefill_args(seqs)
        set_context(True, cu_q, cu_k, max_q, max_k, slot_mapping, None, block_tables,
                    gdn_slots=gdn_slots, gdn_cu_seqlens=gdn_cu, gdn_has_initial=gdn_hi)
        return input_ids, positions

    def prepare_decode(self, seqs):
        iids, pos_list, sm, cl = [], [], [], []
        is_vl = self.config.is_multimodal
        for s in seqs:
            iids.append(s.last_token)
            delta = getattr(s, 'mrope_position_delta', 0)
            p = len(s) - 1 + delta
            if is_vl: pos_list.append([p, p, p])
            else: pos_list.append(p)
            cl.append(len(s))
            sm.append(s.block_table[-1] * self.block_size + s.last_block_num_tokens - 1)
        iids = torch.tensor(iids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        if is_vl:
            pos = torch.tensor(pos_list, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True).t().contiguous()
        else:
            pos = torch.tensor(pos_list, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        sm = torch.tensor(sm, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cl = torch.tensor(cl, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        bt = self.prepare_block_tables(seqs)
        # 🚨 GDN decode slots into context
        gdn_slots = None
        if self.is_hybrid:
            gdn_slots = self._build_gdn_decode_args(seqs)
        set_context(False, slot_mapping=sm, context_lens=cl, block_tables=bt, gdn_slots=gdn_slots)
        return iids, pos

    def _slot_for_pos(self, seq, pos):
        block_idx = pos // self.block_size
        block_off = pos % self.block_size
        return seq.block_table[block_idx] * self.block_size + block_off

    def prepare_spec_decode(self, seqs):
        iids, pos_list, sm, cl, bt_rows = [], [], [], [], []
        is_vl = self.config.is_multimodal
        for s in seqs:
            drafts = list(getattr(s, "spec_scheduled_draft_tokens", None) or [])
            if not drafts:
                draft = s.spec_scheduled_draft_token
                assert draft is not None and draft >= 0
                drafts = [draft]
            delta = getattr(s, 'mrope_position_delta', 0)
            tokens = [s.last_token] + drafts
            for i, tok in enumerate(tokens):
                abs_pos = len(s) - 1 + i
                p = abs_pos + delta
                iids.append(tok)
                if is_vl: pos_list.append([p, p, p])
                else: pos_list.append(p)
                cl.append(len(s) + i)
                sm.append(self._slot_for_pos(s, abs_pos))
                bt_rows.append(s.block_table)
        iids = torch.tensor(iids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        if is_vl:
            pos = torch.tensor(pos_list, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True).t().contiguous()
        else:
            pos = torch.tensor(pos_list, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        sm = torch.tensor(sm, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cl = torch.tensor(cl, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        mx = max(len(r) for r in bt_rows)
        bt = [list(r) + [-1] * (mx - len(r)) for r in bt_rows]
        bt = torch.tensor(bt, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(False, slot_mapping=sm, context_lens=cl, block_tables=bt)
        return iids, pos

    def prepare_sample(self, seqs):
        return torch.tensor([s.temperature for s in seqs], dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)

    def prepare_spec_sample(self, seqs):
        temps = []
        for s in seqs:
            drafts = list(getattr(s, "spec_scheduled_draft_tokens", None) or [])
            temps.extend([s.temperature] * (len(drafts) + 1))
        return torch.tensor(temps, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)

    @torch.inference_mode()
    def run_model(self, input_ids, positions, is_prefill,
                  inputs_embeds=None, deepstack_embeds=None, deepstack_layer_indices=None):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            if self.is_hybrid:
                hidden = self.model(input_ids if inputs_embeds is None else None, positions,
                                    is_prefill=is_prefill, inputs_embeds=inputs_embeds)
            elif inputs_embeds is not None:
                hidden = self.model(input_ids=None, positions=positions, inputs_embeds=inputs_embeds,
                                    deepstack_embeds=deepstack_embeds, deepstack_layer_indices=deepstack_layer_indices)
            else:
                hidden = self.model(input_ids, positions)
            return self.model.compute_logits(hidden)
        else:
            bs = input_ids.size(0)
            ctx = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            gv = self.graph_vars
            gv["input_ids"][:bs] = input_ids
            if positions.dim() == 2:
                gv["positions"][:, :bs] = positions
            else:
                if gv["positions"].dim() == 2:
                    gv["positions"][0, :bs] = positions
                    gv["positions"][1, :bs] = positions
                    gv["positions"][2, :bs] = positions
                else:
                    gv["positions"][:bs] = positions
            gv["slot_mapping"].fill_(-1); gv["slot_mapping"][:bs] = ctx.slot_mapping
            gv["context_lens"].zero_(); gv["context_lens"][:bs] = ctx.context_lens
            gv["block_tables"].zero_()
            cols = min(ctx.block_tables.size(1), gv["block_tables"].size(1))
            gv["block_tables"][:bs, :cols] = ctx.block_tables[:, :cols]
            if self.is_hybrid:
                gv["gdn_slots"].fill_(self._gdn_garbage_slot)
                gv["gdn_slots"][:bs] = ctx.gdn_slots
                # graph's context must expose gdn_slots buffer; set it
                ctx.gdn_slots = gv["gdn_slots"]   # so layers read the graph buffer during replay-captured region
            graph.replay()
            return self.model.compute_logits(gv["outputs"][:bs])

    def run(self, seqs, is_prefill):
        iids, pos = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temps = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(iids, pos, is_prefill)
        tids = self.sampler(logits, temps).tolist() if self.rank == 0 else None
        reset_context()
        return tids

    @torch.inference_mode()
    def capture_cudagraph(self):
        if self.is_moe:
            from nanovllm.layers.fused_moe import set_moe_capture_mode
            set_moe_capture_mode(True)
        hf_config = self.config.hf_config
        text_config = getattr(hf_config, "text_config", hf_config)
        max_bs = min(self.config.max_num_seqs, 512)
        max_nb = min(self.config.num_kvcache_blocks, 1024)
        iids = torch.zeros(max_bs, dtype=torch.int64)
        if self.config.is_multimodal:
            pos = torch.zeros(3, max_bs, dtype=torch.int64)
        else:
            pos = torch.zeros(max_bs, dtype=torch.int64)
        sm = torch.zeros(max_bs, dtype=torch.int32)
        cl = torch.zeros(max_bs, dtype=torch.int32)
        bt = torch.zeros(max_bs, max_nb, dtype=torch.int32)
        outs = torch.zeros(max_bs, text_config.hidden_size)
        gdn_slots_buf = None
        if self.is_hybrid:
            gdn_slots_buf = torch.full((max_bs,), self._gdn_garbage_slot, dtype=torch.int32)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None
        for bs in reversed(self.graph_bs):
            g = torch.cuda.CUDAGraph()
            gs = gdn_slots_buf[:bs] if self.is_hybrid else None
            set_context(False, slot_mapping=sm[:bs], context_lens=cl[:bs], block_tables=bt[:bs],
                        gdn_slots=gs)
            if self.config.is_multimodal or self.is_hybrid:
                p = pos[:, :bs] if self.config.is_multimodal else pos[:bs]
                if self.is_hybrid:
                    outs[:bs] = self.model(iids[:bs], p, is_prefill=False)
                    with torch.cuda.graph(g, self.graph_pool):
                        outs[:bs] = self.model(iids[:bs], p, is_prefill=False)
                else:
                    outs[:bs] = self.model(iids[:bs], pos[:, :bs])
                    with torch.cuda.graph(g, self.graph_pool):
                        outs[:bs] = self.model(iids[:bs], pos[:, :bs])
            else:
                outs[:bs] = self.model(iids[:bs], pos[:bs])
                with torch.cuda.graph(g, self.graph_pool):
                    outs[:bs] = self.model(iids[:bs], pos[:bs])
            if self.graph_pool is None: self.graph_pool = g.pool()
            self.graphs[bs] = g
            torch.cuda.synchronize()
            reset_context()
        self.graph_vars = dict(
            input_ids=iids, positions=pos, slot_mapping=sm,
            context_lens=cl, block_tables=bt, outputs=outs)
        if self.is_hybrid:
            self.graph_vars["gdn_slots"] = gdn_slots_buf

    def _init_streaming_rope_cache(self):
        if hasattr(self, '_rope_cos'): return
        hf_config = self.config.hf_config
        text_config = getattr(hf_config, "text_config", hf_config)
        tp_size = get_pstate().tp_size
        self._num_kv_heads = text_config.num_key_value_heads // tp_size
        self._head_dim = getattr(text_config, "head_dim", text_config.hidden_size // text_config.num_attention_heads)
        rp = getattr(text_config, "rope_parameters", None)
        if self.is_hybrid and rp is not None:
            partial = rp.get("partial_rotary_factor", 1.0)
            self._rotary_dim = int(self._head_dim * partial)
            self._rotary_half = self._rotary_dim // 2
            rope_theta = rp["rope_theta"]
        else:
            self._rotary_dim = self._head_dim
            self._rotary_half = self._head_dim // 2
            rope_theta = getattr(text_config, "rope_theta", 1000000)
            rs = getattr(text_config, "rope_scaling", None)
            if isinstance(rs, dict): rope_theta = rs.get("rope_theta", rope_theta)
        self._half_dim = self._head_dim // 2
        dtype = getattr(text_config, "dtype", torch.float16)
        if isinstance(dtype, str): dtype = getattr(torch, dtype, torch.float16)
        from nanovllm.layers.streaming_rope import build_rope_cache
        cos, sin, off = build_rope_cache(self._head_dim, self.config.max_model_len, rope_theta, dtype,
                                         rotary_dim=self._rotary_dim)
        self._rope_cos = cos.cuda(); self._rope_sin = sin.cuda(); self._rope_offset = off

    def apply_kv_ops(self, ops):
        self._init_streaming_rope_cache()
        from nanovllm.layers.streaming_rope import (
            pure_move_kv_kernel, inplace_delta_rope_kernel, inplace_delta_rope_kernel_partial)
        nl = self.kv_cache.size(1)
        if ops.has_moves:
            src = torch.tensor(ops.move_src, dtype=torch.int32, device='cuda')
            dst = torch.tensor(ops.move_dst, dtype=torch.int32, device='cuda')
            n = len(ops.move_src)
            for li in range(nl):
                kf = self.kv_cache[0, li].view(-1, self._num_kv_heads * self._head_dim)
                vf = self.kv_cache[1, li].view(-1, self._num_kv_heads * self._head_dim)
                pure_move_kv_kernel[(n,)](kf, vf, src, dst, n, self._num_kv_heads, self._head_dim)
        if ops.has_inplace:
            slots = torch.tensor(ops.inplace_slots, dtype=torch.int32, device='cuda')
            deltas = torch.tensor(ops.inplace_deltas, dtype=torch.int32, device='cuda')
            n = len(ops.inplace_slots)
            for li in range(nl):
                kf = self.kv_cache[0, li].view(-1, self._num_kv_heads * self._head_dim)
                if self.is_hybrid:
                    inplace_delta_rope_kernel_partial[(n,)](
                        kf, self._rope_cos, self._rope_sin, slots, deltas, n,
                        self._num_kv_heads, self._head_dim, self._rotary_half, self._rope_offset)
                else:
                    inplace_delta_rope_kernel[(n,)](
                        kf, self._rope_cos, self._rope_sin, slots, deltas, n,
                        self._num_kv_heads, self._head_dim, self._half_dim, self._rope_offset)

    def _decode_uses_graph(self, input_ids):
        return (not self.enforce_eager) and input_ids.size(0) <= 512

    def run_streaming_spec(self, seqs):
        input_ids, positions = self.prepare_spec_decode(seqs)
        stats = {"spec_graph_replays": 0, "eager_forwards": 0}
        if self._decode_uses_graph(input_ids):
            stats["spec_graph_replays"] = 1
        else:
            stats["eager_forwards"] = 1
        logits = self.run_model(input_ids, positions, is_prefill=False)
        tids = None
        if self.rank == 0:
            temps = self.prepare_spec_sample(seqs)
            sampled = self.sampler(logits, temps).tolist()
            tids = []
            offset = 0
            for seq in seqs:
                drafts = list(getattr(seq, "spec_scheduled_draft_tokens", None) or [])
                if not drafts:
                    drafts = [seq.spec_scheduled_draft_token]
                accepted = 0
                for j, draft in enumerate(drafts):
                    if sampled[offset + j] != draft:
                        break
                    accepted += 1
                if accepted == len(drafts):
                    tids.append(drafts + [sampled[offset + len(drafts)]])
                else:
                    tids.append(drafts[:accepted] + [sampled[offset + accepted]])
                offset += len(drafts) + 1
        reset_context()
        return tids, stats

    def _build_gdn_prefill_args(self, seqs):
        gdn_slots, cu, has_init = [], [0], []
        for seq in seqs:
            gdn_slots.append(self.gdn_pool.get_slot(seq.seq_id))
            start = min(seq.num_cached_tokens, len(seq) - 1)
            sq = seq.num_scheduled_tokens
            cu.append(cu[-1] + sq)
            has_init.append(seq.num_cached_tokens > 0)
        return (torch.tensor(gdn_slots, dtype=torch.int32, device="cuda"),
                torch.tensor(cu, dtype=torch.int32, device="cuda"),
                torch.tensor(has_init, dtype=torch.bool, device="cuda"))

    def _build_gdn_decode_args(self, seqs):
        gdn_slots = [self.gdn_pool.get_slot(s.seq_id) for s in seqs]
        return torch.tensor(gdn_slots, dtype=torch.int32, device="cuda")

    def run_streaming(self, seqs, is_prefill, kv_ops, vision_data=None, mrope_positions=None, return_stats=False):
        if kv_ops.has_any:
            self.apply_kv_ops(kv_ops)
        stats = {"normal_graph_replays": 0, "spec_graph_replays": 0, "eager_forwards": 0}

        if (not is_prefill and not self.is_hybrid
            and self.config.spec_decode_mode == "graph2"
            and seqs and all(getattr(s, "spec_scheduled", False) for s in seqs)):
            tids, spec_stats = self.run_streaming_spec(seqs)
            stats.update({k: stats.get(k, 0) + v for k, v in spec_stats.items()})
            return (tids, stats) if return_stats else tids

        if is_prefill:
            input_ids, positions = self.prepare_prefill(seqs)   # sets gdn context too (hybrid)
            inputs_embeds = None; ds_embeds = None; ds_indices = None
            if vision_data is not None:
                inputs_embeds, ds_embeds, ds_indices = self._build_vision_embeds(seqs, input_ids, vision_data)
            if mrope_positions is not None:
                positions = mrope_positions.to(device=positions.device, dtype=positions.dtype)
            logits = self.run_model(input_ids, positions, is_prefill,
                                    inputs_embeds=inputs_embeds,
                                    deepstack_embeds=ds_embeds, deepstack_layer_indices=ds_indices)
        else:
            input_ids, positions = self.prepare_decode(seqs)   # sets gdn_slots context (hybrid)
            if self._decode_uses_graph(input_ids):
                stats["normal_graph_replays"] = 1
            else:
                stats["eager_forwards"] = 1
            logits = self.run_model(input_ids, positions, is_prefill)

        temps = self.prepare_sample(seqs) if self.rank == 0 else None
        tids = self.sampler(logits, temps).tolist() if self.rank == 0 else None
        reset_context()
        return (tids, stats) if return_stats else tids

    def _get_embed_tokens(self):
        m = self.model
        if hasattr(m, "language_model"):
            return m.language_model.model.embed_tokens
        return m.model.embed_tokens

    def _build_vision_embeds(self, seqs, input_ids, vision_data):
        import torch
        inputs_embeds = None; ds_embeds = None; ds_indices = None
        mask = torch.tensor(vision_data["mask"], dtype=torch.bool, device="cuda")
        num_vision_tokens = mask.sum().item()
        if num_vision_tokens == 0:
            return inputs_embeds, ds_embeds, ds_indices
        new_main = None; new_ds_levels = None
        visual = getattr(self.model, 'visual', None)
        if vision_data.get("pixel_values") is not None and visual is not None:
            pv = vision_data["pixel_values"].to(device="cuda", dtype=visual.dtype)
            thw = vision_data["image_grid_thw"].to(device="cuda", dtype=torch.long)
            with torch.inference_mode():
                vision_output = visual(pv, thw)
            ds_indices_list = vision_data.get("deepstack_indices")
            ds_num = len(ds_indices_list) if ds_indices_list else 0
            vd = vision_data.get("visual_dim", vision_output.shape[-1])
            if ds_num > 0:
                main_dim = vd // (1 + ds_num)
                new_main, multiscale = torch.split(vision_output, [main_dim, main_dim * ds_num], dim=-1)
                new_ds_levels = list(multiscale.chunk(ds_num, dim=-1))
            else:
                new_main = vision_output
        cached_main = vision_data.get("cached_main_embeds")
        cached_ds = vision_data.get("cached_ds_embeds")
        parts_main = []
        if cached_main is not None: parts_main.append(cached_main.to(device="cuda"))
        if new_main is not None: parts_main.append(new_main)
        if not parts_main:
            return inputs_embeds, ds_embeds, ds_indices
        combined_main = torch.cat(parts_main, dim=0)
        inputs_embeds = self._get_embed_tokens()(input_ids)
        nc = combined_main.shape[0]
        if num_vision_tokens == nc:
            inputs_embeds[mask] = combined_main.to(inputs_embeds.dtype)
        else:
            print(f"[WARN] mask={num_vision_tokens} != combined={nc}")
            mi = torch.nonzero(mask).squeeze(1); ml = min(num_vision_tokens, nc)
            inputs_embeds[mi[:ml]] = combined_main[:ml].to(inputs_embeds.dtype)
        if self.rank == 0 and new_main is not None:
            cache_ops = vision_data.get("_cache_ops", [])
            ni = 0
            for cop in cache_ops:
                if cop.get("is_new"):
                    ln = cop["length"]
                    if ni + ln <= new_main.shape[0]:
                        cop_main = new_main[ni:ni+ln].detach().cpu()
                        cop_ds = [d[ni:ni+ln].detach().cpu() for d in new_ds_levels] if new_ds_levels else None
                        cop["seq"].cache_vision_embeds(cop["abs_start"], cop["abs_end"], cop_main, cop_ds)
                    ni += ln
        ds_indices_list = vision_data.get("deepstack_indices")
        if ds_indices_list:
            ds_indices = ds_indices_list
            ds_embeds = {}
            for level in range(len(ds_indices_list)):
                buf = torch.zeros_like(inputs_embeds)
                parts_ds = []
                if cached_ds and level in cached_ds: parts_ds.append(cached_ds[level].to(device="cuda"))
                if new_ds_levels and level < len(new_ds_levels): parts_ds.append(new_ds_levels[level])
                if parts_ds:
                    combined_ds = torch.cat(parts_ds, dim=0)
                    ncd = combined_ds.shape[0]
                    if num_vision_tokens == ncd:
                        buf[mask] = combined_ds.to(buf.dtype)
                    else:
                        mi = torch.nonzero(mask).squeeze(1); ml = min(num_vision_tokens, ncd)
                        buf[mi[:ml]] = combined_ds[:ml].to(buf.dtype)
                ds_embeds[f"deepstack_{level}"] = buf
        return inputs_embeds, ds_embeds, ds_indices