import os
import json
import time
from PIL import Image
from transformers import AutoProcessor
import torch

# 导入我们自己写的 StreamingLLM
from nanovllm import StreamingLLM, StreamingConfig, SamplingParams

# 1. 全局配置
MODEL_PATH = "/mnt/neimeng/nlp/projects/pretrain/luorun/workspace/GUI-Agent/open_verl/sft/verl_sft_test/guiagent-qwen3-8b-fsdp-fsdp2-sp1-fsdp-1202a1-4/global_step_8076/huggingface"
DATA_DIR = "/mnt/neimeng/nlp/projects/pretrain/luorun/workspace/GUI-Agent/session_logs/8eb58d69-18f2-4b1e-8242-a9131937aa39"

# 窗口大小配置 (历史消息数量)
HISTORY_WINDOW_SIZE = 20  
TOTAL_ROUNDS = 100

# 2. 初始化 Processor
processor = AutoProcessor.from_pretrained(
    MODEL_PATH, 
    min_pixels=256 * 28 * 28, 
    max_pixels=1280 * 28 * 28
)

def run_streaming_benchmark():
    print("Initializing StreamingLLM Engine...")
    
    streaming_config = StreamingConfig(
        dialogue_window=HISTORY_WINDOW_SIZE, 
        history_window=1024,  # 保留被驱逐轮次的文本历史
        history_sink=64,
        initial_padding=16,
    )
    
    engine = StreamingLLM(
        model=MODEL_PATH,
        streaming_config=streaming_config,
        tensor_parallel_size=8,       # 根据你的 GPU 数量调整
        max_model_len=32768,          # 必须开大，容纳多张图片
        max_num_batched_tokens=8192,  # 保证单图 Prefill 不被截断
        gpu_memory_utilization=0.8,
        enforce_eager=False,          # 开启 CUDA Graph Decode 加速
        is_multimodal=True,
        dtype="bfloat16"              # 防止 FlashAttention 报错
    )

    sampling_params = SamplingParams(
        temperature=0.9,  # 接近贪婪解码，保证输出稳定
        max_tokens=40
    )

    # 获取数据集文件
    dataset_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.json')])
    dataset_files = dataset_files[:TOTAL_ROUNDS]

    print(f"\nStart Benchmark: {len(dataset_files)} rounds, Window Size: {HISTORY_WINDOW_SIZE}")
    print("=" * 110)

    # 真实的 System Prompt
    system_prompt = '''You are an experienced Honkai: Star Rail PC player, proficient in keyboard and mouse operations. Based on the current game screen, current mouse position and the corresponding historical trajectory, plan the next 200ms of actions, consisting of 6 steps. Each step is spaced 33ms apart. Every step lasts 33ms from its start time until the next step begins.
**Output Format**
<|action_start|>X Y Z ; k1 k2 k3 ; k4 k5 ; k6 ; k7 ; k8 ; k9 k10<|action_end|>
**Explanation**
1. **Mouse Movement**: First, specify the relative displacement X, Y (X>0 means move right, Y>0
means move down) and scroll amount Z (Z>0 means scroll up).
2. **Key Sequence**: Then list 6 groups of keys; within each group, keys are separated by spaces,
and groups are separated by semicolons.
- Each group can contain up to 4 keys.
- If a group has no keys, leave it empty but keep the `;`.
3. Only output a plain string that conforms to the above format — no line breaks and no quotation marks.
**Key Naming Rules**
- Number keys `1-9`: use lowercase English words, e.g., `one` represents the `1` key on the keyboard.
- Function keys `F1-F12`: use capitalized English words, e.g., `One` represents `F1`, `Two` represents `F2`, and so on.
- Other keys (letters, Shift, Tab, Space, etc.): use the real keyboard name with an initial capital letter, e.g., `A`, `D`, `Shift`, `Space`, etc.'''
    
    # 3. 创建持久化 Session
    session_id = engine.create_session(system_prompt, sampling_params)

    all_results = []

    for i, f_name in enumerate(dataset_files):
        # --- 1. 准备当前轮次的数据 ---
        json_path = os.path.join(DATA_DIR, f_name)
        img_path = os.path.join(DATA_DIR, f_name.replace('.json', '.jpg'))
        
        # 读取 Ground Truth (GT)
        with open(json_path, 'r', encoding='utf-8') as f:
            item = json.load(f)
            gt_response = item.get('model_response', '')
            
        raw_image = Image.open(img_path).convert("RGB")
        user_text = "current position [0,0]"

        # 使用 HF Processor 处理单张新图
        proc_inputs = processor(
            text=["<|vision_start|><|image_pad|><|vision_end|>"], 
            images=[raw_image], 
            return_tensors="pt", 
            padding=True
        )
        pixel_values = proc_inputs["pixel_values"]
        image_grid_thw = proc_inputs["image_grid_thw"]

        # --- 2. 构建消息并发送给 Engine ---
        message = engine._build_vision_message(user_text, image_grid_thw)
        
        seq = engine.scheduler.continue_session(session_id, message)
        seq._pending_vision = {"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}
        seq.current_turn.image_grid_thw_list = [tuple(g) for g in image_grid_thw.tolist()]

        # ---------------------------------------------------------
        # [阶段 1] PREFILL (预填充)
        # ---------------------------------------------------------
        t0_prefill = time.perf_counter()
        pf_tok = 0
        while seq.status.name == "WAITING":
            seqs, is_pf, kv_ops = engine.scheduler.schedule()
            if not seqs: break
            
            vision_data = engine._prepare_vision_data(seqs) if is_pf else None
            mrope_positions = engine._compute_batch_mrope(seqs) if is_pf else None
            
            if is_pf:
                pf_tok += sum(s.num_scheduled_tokens for s in seqs)
                
            tids = engine.model_runner.call(
                "run_streaming", seqs, is_pf, kv_ops, vision_data, mrope_positions
            )
            
            # 缓存 Vision Embeddings
            if is_pf and vision_data and vision_data.get("_cache_ops"):
                for op in vision_data["_cache_ops"]:
                    if op.get("is_new"):
                        # 找到对应的 main_embeds 和 ds_embeds
                        mask = vision_data["mask"]
                        num_vis = sum(mask)
                        if num_vis > 0:
                            # 简化处理：由于单并发，直接取 vision_data 中的 embeds
                            main_embeds = vision_data.get("cached_main_embeds")
                            # 如果没有 cached，说明全是 new 的，这里需要从 model_runner 返回，但为了测速，我们暂时跳过精确的 cache 写入，
                            # 因为 model_runner 内部已经处理了 GPU 上的替换。
                            # 完整的 cache 逻辑在之前的 _step 中已经实现，这里为了 benchmark 简化。
            
            # 更新 MRoPE Delta
            if is_pf and mrope_positions is not None:
                offset = 0
                for s in seqs:
                    n = s.num_scheduled_tokens
                    if n > 0:
                        seq_pos = mrope_positions[:, offset:offset + n]
                        s.mrope_position_delta = int(seq_pos.max().item()) - (s.num_cached_tokens + n - 1)
                    offset += n
                    
            engine.scheduler.postprocess(seqs, tids, is_prefill=is_pf)
            
        prefill_time = time.perf_counter() - t0_prefill

        # ---------------------------------------------------------
        # [阶段 2] DECODE (解码)
        # ---------------------------------------------------------
        t0_decode = time.perf_counter()
        dc_tok = 0
        while not seq.is_idle and not seq.is_finished:
            seqs, is_pf, kv_ops = engine.scheduler.schedule()
            if not seqs: break
            tids = engine.model_runner.call("run_streaming", seqs, is_pf, kv_ops, None, None)
            engine.scheduler.postprocess(seqs, tids, is_prefill=is_pf)
            dc_tok += 1
            
        decode_time = time.perf_counter() - t0_decode

        # ---------------------------------------------------------
        # [阶段 3] 统计与打印
        # ---------------------------------------------------------
        response_text = engine.tokenizer.decode(seq.completion_token_ids, skip_special_tokens=True).strip()
        stats = engine.get_stats(session_id)
        
        pf_tps = pf_tok / prefill_time if prefill_time > 0 else 0
        dc_tps = dc_tok / decode_time if decode_time > 0 else 0
        evict_flag = "🔄" if stats["history_tokens"] > 0 else "  "

        print(f"\n{'-'*110}")
        print(
            f"{evict_flag}[Turn {i+1:2d}/{len(dataset_files)}] "
            f"PF: {pf_tok:4d} tok / {prefill_time*1000:6.1f}ms = {pf_tps:5.0f} t/s | "
            f"DC: {dc_tok:2d} tok / {decode_time*1000:6.1f}ms = {dc_tps:5.0f} t/s | "
            f"CTX: {stats['total_tokens']:5d}"
        )
        print(f"  GT:  {gt_response}")
        print(f"  Out: {response_text}")
        
        # 简单比对（如果完全一致打印 ✅，否则打印 ❌）
        is_match = response_text == gt_response
        match_flag = "✅ MATCH" if is_match else "❌ MISMATCH"
        print(f"  Res: {match_flag}")

        all_results.append({
            "turn": i + 1,
            "file": f_name,
            "pf_tok": pf_tok,
            "pf_time_ms": round(prefill_time * 1000, 2),
            "pf_tps": round(pf_tps, 1),
            "dc_tok": dc_tok,
            "dc_time_ms": round(decode_time * 1000, 2),
            "dc_tps": round(dc_tps, 1),
            "ctx_tokens": stats["total_tokens"],
            "match": is_match,
            "gt": gt_response,
            "output": response_text,
        })

    print("\n" + "=" * 110)
    print("Inference Completed.")
    
    final_stats = engine.get_stats(session_id)
    print(f"Final Context Size: {final_stats['total_tokens']} tokens")
    print(f"Active Turns in Window: {final_stats['dialogue_turns']}")
    
    engine.close_all()

    # =========================================================
    # 统计汇总
    # =========================================================
    print("\n" + "=" * 110)
    print("BENCHMARK STATISTICS SUMMARY")
    print("=" * 110)

    total = len(all_results)
    match_cnt = sum(1 for r in all_results if r["match"])
    mismatch_cnt = total - match_cnt
    match_rate = match_cnt / total * 100 if total > 0 else 0

    pf_toks   = [r["pf_tok"]      for r in all_results]
    pf_times  = [r["pf_time_ms"]  for r in all_results]
    pf_tpss   = [r["pf_tps"]      for r in all_results]
    dc_toks   = [r["dc_tok"]      for r in all_results]
    dc_times  = [r["dc_time_ms"]  for r in all_results]
    dc_tpss   = [r["dc_tps"]      for r in all_results]
    ctx_sizes = [r["ctx_tokens"]  for r in all_results]

    def stat(lst):
        import statistics
        return {
            "min":  min(lst),
            "max":  max(lst),
            "mean": statistics.mean(lst),
            "median": statistics.median(lst),
        }

    s_pf_tok  = stat(pf_toks)
    s_pf_time = stat(pf_times)
    s_pf_tps  = stat(pf_tpss)
    s_dc_tok  = stat(dc_toks)
    s_dc_time = stat(dc_times)
    s_dc_tps  = stat(dc_tpss)

    print(f"\n[准确率]")
    print(f"  总轮次:  {total}")
    print(f"  MATCH:   {match_cnt}  ({match_rate:.1f}%)")
    print(f"  MISMATCH:{mismatch_cnt}  ({100-match_rate:.1f}%)")

    print(f"\n[Prefill 统计]")
    print(f"  Tokens  — min:{s_pf_tok['min']:6.0f}  max:{s_pf_tok['max']:6.0f}  mean:{s_pf_tok['mean']:7.1f}  median:{s_pf_tok['median']:7.1f}")
    print(f"  Time(ms)— min:{s_pf_time['min']:6.1f}  max:{s_pf_time['max']:6.1f}  mean:{s_pf_time['mean']:7.1f}  median:{s_pf_time['median']:7.1f}")
    print(f"  Tput t/s— min:{s_pf_tps['min']:6.0f}  max:{s_pf_tps['max']:6.0f}  mean:{s_pf_tps['mean']:7.0f}  median:{s_pf_tps['median']:7.0f}")

    print(f"\n[Decode 统计]")
    print(f"  Tokens  — min:{s_dc_tok['min']:6.0f}  max:{s_dc_tok['max']:6.0f}  mean:{s_dc_tok['mean']:7.1f}  median:{s_dc_tok['median']:7.1f}")
    print(f"  Time(ms)— min:{s_dc_time['min']:6.1f}  max:{s_dc_time['max']:6.1f}  mean:{s_dc_time['mean']:7.1f}  median:{s_dc_time['median']:7.1f}")
    print(f"  Tput t/s— min:{s_dc_tps['min']:6.0f}  max:{s_dc_tps['max']:6.0f}  mean:{s_dc_tps['mean']:7.0f}  median:{s_dc_tps['median']:7.0f}")

    print(f"\n[上下文大小 (tokens)]")
    print(f"  min:{min(ctx_sizes)}  max:{max(ctx_sizes)}  mean:{sum(ctx_sizes)/len(ctx_sizes):.0f}  final:{ctx_sizes[-1]}")

    # 保存结果到 JSON 文件
    import datetime
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_results.json")
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump({
            "timestamp": datetime.datetime.now().isoformat(),
            "total_rounds": total,
            "match_count": match_cnt,
            "mismatch_count": mismatch_cnt,
            "match_rate_pct": round(match_rate, 2),
            "prefill": {"tokens": s_pf_tok, "time_ms": s_pf_time, "tps": s_pf_tps},
            "decode":  {"tokens": s_dc_tok, "time_ms": s_dc_time, "tps": s_dc_tps},
            "context_tokens": {"min": min(ctx_sizes), "max": max(ctx_sizes), "mean": round(sum(ctx_sizes)/len(ctx_sizes), 1), "final": ctx_sizes[-1]},
            "per_round": all_results,
        }, fp, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存至: {out_path}")
    print("=" * 110)

if __name__ == "__main__":
    run_streaming_benchmark()