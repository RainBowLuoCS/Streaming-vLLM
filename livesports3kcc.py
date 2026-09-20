import os
os.environ['LIVESPORTS3K_PATH']='/mnt/neimeng/nlp/projects/pretrain/luorun/datasets/checkpoints/stdKonjac/LiveSports-3K/videos'
import json
import argparse
import time
import multiprocessing
from tqdm import tqdm
from PIL import Image
import numpy as np
import decord
import torch
import cv2

# 必须在导入任何包含 CUDA 的库之前设置 spawn！
multiprocessing.set_start_method('spawn', force=True)

# 导入我们强大的 StreamingLLM
from nanovllm import StreamingLLM, StreamingConfig, SamplingParams

# 初始化 Decord
decord.bridge.set_bridge('torch')

def parse_args():
    parser = argparse.ArgumentParser(description="StreamingLLM Evaluation on LiveSports-3K")
    parser.add_argument("--model_path", type=str, required=True, help="Path to Qwen3-VL-8B model")
    parser.add_argument("--output_dir", type=str, default="./results/livesports3k/streaming_llm")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_tokens", type=int, default=256)
    # StreamingLLM 特定参数
    parser.add_argument("--tp_size", type=int, default=8, help="Tensor Parallel Size")
    parser.add_argument("--dialogue_window", type=int, default=10, help="Max turns to keep in KV Cache")
    parser.add_argument("--chunk_duration", type=int, default=5, help="Seconds per chunk")
    parser.add_argument("--frames_per_chunk", type=int, default=2, help="How many frames to sample per chunk")
    parser.add_argument("--max_concurrent", type=int, default=16, help="Max concurrent video sessions")
    return parser.parse_args()

def extract_frames(video_path: str, start_time: float, end_time: float, num_frames: int) -> list[Image.Image]:
    try:
        vr = decord.VideoReader(video_path)
        fps = vr.get_avg_fps()
        total_frames = len(vr)
        
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        start_frame = max(0, min(start_frame, total_frames - 1))
        end_frame = max(start_frame + 1, min(end_frame, total_frames))
        
        frame_indices = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int).tolist()
        frames_tensor = vr.get_batch(frame_indices)
        frames = frames_tensor.cpu().numpy().astype(np.uint8)
        
        return [Image.fromarray(frame) for frame in frames]
    
    except Exception as e:
        print(f"[WARNING] decord failed for {video_path}: {e}. Falling back to cv2...")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] cv2 also failed to open {video_path}.")
            return [Image.new("RGB", (224, 224), (0, 0, 0)) for _ in range(num_frames)]
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps <= 0 or total_frames <= 0:
            fps = 30.0
            total_frames = 999999
            
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        start_frame = max(0, min(start_frame, total_frames - 1))
        end_frame = max(start_frame + 1, min(end_frame, total_frames))
        
        frame_indices = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int).tolist()
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
            else:
                if frames:
                    frames.append(frames[-1].copy())
                else:
                    frames.append(Image.new("RGB", (224, 224), (0, 0, 0)))
                    
        cap.release()
        return frames

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, "predictions.jsonl")
    
    with open(out_file, "w", encoding="utf-8") as f:
        pass
    
    print("=" * 110)
    print(f"Starting StreamingLLM Evaluation on LiveSports-3K")
    print(f"Model: {args.model_path}")
    print(f"TP Size: {args.tp_size}, Dialogue Window: {args.dialogue_window}, Max Concurrent: {args.max_concurrent}")
    print("=" * 110)

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(
        args.model_path, 
        min_pixels=256 * 28 * 28, 
        max_pixels=1280 * 28 * 28
    )

    # 🚨 优化点：关闭 history_window 防止长序列复读机 🚨
    streaming_config = StreamingConfig(
        dialogue_window=args.dialogue_window,
        history_window=512,
        history_sink=0,
        initial_padding=16,
    )
    
    engine = StreamingLLM(
        model=args.model_path,
        streaming_config=streaming_config,
        tensor_parallel_size=args.tp_size,
        max_model_len=65536,          
        max_num_batched_tokens=32768*4, 
        max_num_seqs=args.max_concurrent * 2, 
        gpu_memory_utilization=0.85,
        enforce_eager=False,          
        is_multimodal=True,
        dtype="bfloat16"
    )

    sampling_params = SamplingParams(
        temperature=args.temperature, 
        max_tokens=args.max_tokens
    )

    print("\nLoading dataset...")
    from datasets import load_dataset
    ds = load_dataset('stdKonjac/LiveSports-3K', name='LiveSports_3K_CC', split="test")
    
    video_tasks = {}
    count = 0
    for record in ds:
        vid = record["video_id"] + "_" + str(record["event_id"])
        video_tasks[vid] = record
        count += 1
        # if count > 2:  # 限制测试数量
        #     break

    system_prompt = (
        "You are an expert video commentator providing real-time, insightful, "
        "and engaging commentary on visual content."
    )

    pending_vids = list(video_tasks.keys())
    active_vids = []
    completed_vids = set()
    
    sessions = {}
    vid_current_time = {}
    
    for vid in pending_vids:
        vid_current_time[vid] = video_tasks[vid].get("begin", 0.0)

    total_prefill_time = 0.0
    total_decode_time = 0.0
    total_pf_tok = 0
    total_dc_tok = 0
    step_counter = 0

    print(f"Total Videos to process: {len(pending_vids)}")
    print("-" * 110)

    while pending_vids or active_vids:
        step_counter += 1
        
        while len(active_vids) < args.max_concurrent and pending_vids:
            new_vid = pending_vids.pop(0)
            active_vids.append(new_vid)
            sessions[new_vid] = engine.create_session(system_prompt, sampling_params)
            
        batch_requests = []
        vids_to_remove = []
        
        for vid in active_vids:
            record = video_tasks[vid]
            chunk_start = vid_current_time[vid]
            chunk_end = chunk_start + args.chunk_duration
            
            if chunk_start >= record["end"]:
                vids_to_remove.append(vid)
                completed_vids.add(vid)
                continue
                
            actual_end = min(chunk_end, record["end"])
            
            title = record.get("event_title", "")
            preasr = record.get("preasr_text", "")
            
            if chunk_start == record.get("begin", 0.0):
                prompt = f"This is a video titled \"{title}\". "
                if preasr:
                    prompt += f"\nHere is previous commentary:\n{preasr}\n"
                prompt += "\nPlease comment on the current segment of the video."
            else:
                prompt = "Continue commenting on what happens next in the video."

            video_path = record["video"]
            if not os.path.exists(video_path):
                video_path = os.path.join(os.environ.get("LIVESPORTS3K_PATH", ""), video_path)
                
            frames = extract_frames(video_path, chunk_start, actual_end, args.frames_per_chunk)
            
            proc_inputs = processor(
                text=["<|vision_start|><|image_pad|><|vision_end|>"] * len(frames), 
                images=frames, 
                return_tensors="pt", 
                padding=True
            )
            
            batch_requests.append({
                "video_id": vid,
                "session_id": sessions[vid],
                "message": prompt,
                "pixel_values": proc_inputs["pixel_values"],
                "image_grid_thw": proc_inputs["image_grid_thw"],
                "chunk_start": chunk_start,
                "chunk_end": actual_end
            })
            
            vid_current_time[vid] += args.chunk_duration

        for vid in vids_to_remove:
            active_vids.remove(vid)
            engine.close_session(sessions[vid])
            
        if not batch_requests:
            continue

        print(f"\n[Step {step_counter:3d}] Processing {len(batch_requests)} active video chunks... "
              f"(Completed: {len(completed_vids)}/{len(video_tasks)})")
        
        active_seqs = []
        for req in batch_requests:
            msg = engine._build_vision_message(req["message"], req["image_grid_thw"])
            seq = engine.scheduler.continue_session(req["session_id"], msg)
            if seq:
                seq._pending_vision = {
                    "pixel_values": req["pixel_values"], 
                    "image_grid_thw": req["image_grid_thw"]
                }
                seq.current_turn.image_grid_thw_list = [tuple(g) for g in req["image_grid_thw"].tolist()]
                active_seqs.append(seq)

        # --- PREFILL ---
        pf_tok = 0
        t0_pf = time.perf_counter()
        while any(s.status.name == "WAITING" for s in active_seqs):
            seqs, is_pf, kv_ops = engine.scheduler.schedule()
            if not seqs: 
                print("[WARNING] Scheduler returned empty in Prefill. Possibly out of KV Cache.")
                break
            
            vision_data = engine._prepare_vision_data(seqs) if is_pf else None
            mrope_positions = engine._compute_batch_mrope(seqs) if is_pf else None
            
            if is_pf: 
                pf_tok += sum(s.num_scheduled_tokens for s in seqs)
                
            # ModelRunner 内部会自动将 vision_embeds 存入 seq.vision_cache，无需在此处手动覆盖！
            tids = engine.model_runner.call("run_streaming", seqs, is_pf, kv_ops, vision_data, mrope_positions)
                        
            # 仅更新 MRoPE Delta
            if is_pf and mrope_positions is not None:
                offset = 0
                for s in seqs:
                    n = s.num_scheduled_tokens
                    if n > 0:
                        seq_pos = mrope_positions[:, offset:offset+n]
                        s.mrope_position_delta = int(seq_pos.max().item()) - (s.num_cached_tokens + n - 1)
                    offset += n
                    
            engine.scheduler.postprocess(seqs, tids, is_prefill=is_pf)
        pf_time = time.perf_counter() - t0_pf

        # --- DECODE ---
        dc_tok = 0
        t0_dc = time.perf_counter()
        while any(not s.is_idle and not s.is_finished for s in active_seqs):
            seqs, is_pf, kv_ops = engine.scheduler.schedule()
            if not seqs: 
                print("[WARNING] Scheduler returned empty in Decode. Possibly out of KV Cache.")
                break
            tids = engine.model_runner.call("run_streaming", seqs, is_pf, kv_ops, None, None)
            engine.scheduler.postprocess(seqs, tids, is_prefill=is_pf)
            if not is_pf:
                dc_tok += len(seqs)
        dc_time = time.perf_counter() - t0_dc
        
        total_prefill_time += pf_time
        total_decode_time += dc_time
        total_pf_tok += pf_tok
        total_dc_tok += dc_tok

        # --- 收集结果并保存 ---
        pf_tps = pf_tok / pf_time if pf_time > 0 else 0
        dc_tps = dc_tok / dc_time if dc_time > 0 else 0
        print(f"  PF: {pf_tok:5d} tok / {pf_time*1000:6.1f}ms = {pf_tps:5.0f} t/s | "
              f"DC: {dc_tok:4d} tok / {dc_time*1000:6.1f}ms = {dc_tps:5.0f} t/s")

        batch_results = []
        for req in batch_requests:
            sid = req["session_id"]
            seq = engine.sessions[sid]
            response_text = engine.tokenizer.decode(seq.completion_token_ids, skip_special_tokens=True).strip()
            
            record = video_tasks[req["video_id"]]
            batch_results.append({
                "video": record["video"],
                "video_id": record["video_id"],
                "event_id": record["event_id"],
                "chunk_start": req["chunk_start"],
                "chunk_end": req["chunk_end"],
                "pred": response_text
            })
            
        with open(out_file, "a", encoding="utf-8") as f:
            for res in batch_results:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")

    print("\n" + "=" * 110)
    print(f"Evaluation Completed. Results saved to {out_file}")
    print(f"Overall Prefill TPS: {total_pf_tok / total_prefill_time:.0f} tok/s")
    print(f"Overall Decode TPS:  {total_dc_tok / total_decode_time:.0f} tok/s")
    
    engine.close_all()

if __name__ == "__main__":
    main()
