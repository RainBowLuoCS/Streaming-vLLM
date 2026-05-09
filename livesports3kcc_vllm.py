import os
os.environ['LIVESPORTS3K_PATH']='stdKonjac/LiveSports-3K/videos'
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
import json
import argparse
import time
from tqdm import tqdm
from PIL import Image
import numpy as np
import cv2
import torch
import decord
from datasets import load_dataset
from transformers import AutoProcessor

# 导入官方 vLLM
from vllm import LLM, SamplingParams
import vllm.envs as envs

# 禁用 V1 多进程，防止有些环境报错
envs.VLLM_ENABLE_V1_MULTIPROCESSING = False

# 初始化 Decord
decord.bridge.set_bridge('torch')

def parse_args():
    parser = argparse.ArgumentParser(description="Standard vLLM Evaluation on LiveSports-3K")
    parser.add_argument("--model_path", type=str, required=True, help="Path to Qwen3-VL model")
    parser.add_argument("--output_dir", type=str, default="./results/livesports3k/vllm_baseline")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--tp_size", type=int, default=8, help="Tensor Parallel Size")
    parser.add_argument("--dialogue_window", type=int, default=10, help="Max turns to keep in history")
    parser.add_argument("--chunk_duration", type=int, default=5, help="Seconds per chunk")
    parser.add_argument("--frames_per_chunk", type=int, default=2, help="How many frames to sample per chunk")
    parser.add_argument("--max_concurrent", type=int, default=32, help="Max concurrent video sessions")
    return parser.parse_args()

def extract_frames(video_path: str, start_time: float, end_time: float, num_frames: int) -> list[Image.Image]:
    """从视频的指定时间段内均匀采样 num_frames 帧，带 cv2 容错。"""
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
    print(f"Starting Standard vLLM Evaluation on LiveSports-3K")
    print(f"Model: {args.model_path}")
    print(f"TP Size: {args.tp_size}, Dialogue Window: {args.dialogue_window}, Max Concurrent: {args.max_concurrent}")
    print("=" * 110)

    # 1. 初始化 Processor
    processor = AutoProcessor.from_pretrained(
        args.model_path, 
        min_pixels=256 * 28 * 28, 
        max_pixels=1280 * 28 * 28
    )

    # 2. 初始化原生 vLLM 引擎
    # 必须设置 limit_mm_per_prompt 容纳窗口内所有的图片
    max_images_per_prompt = args.dialogue_window * args.frames_per_chunk + args.frames_per_chunk
    
    engine = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tp_size,
        max_model_len=65536,
        enable_prefix_caching=True,  # 原生 vLLM 依赖 Prefix Caching 来复用历史
        trust_remote_code=True,
        limit_mm_per_prompt={"image": max_images_per_prompt}, 
        gpu_memory_utilization=0.85,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature, 
        max_tokens=args.max_tokens
    )

    # 3. 加载数据集
    print("\nLoading dataset...")
    ds = load_dataset('stdKonjac/LiveSports-3K', name='LiveSports_3K_CC', split="test")
    
    video_tasks = {}
    count = 0
    for record in ds:
        vid = record["video_id"] + "_" + str(record["event_id"])
        video_tasks[vid] = record
        count += 1
        # if count > 16:
        #     break

    system_prompt = (
        "You are an expert video commentator providing real-time, insightful, "
        "and engaging commentary on visual content."
    )

    # 4. 状态管理
    pending_vids = list(video_tasks.keys())
    active_vids = []
    completed_vids = set()
    
    # 手动维护每个视频的对话历史
    session_messages = {}  # video_id -> list of message dicts
    vid_current_time = {}  # video_id -> float
    
    for vid in pending_vids:
        vid_current_time[vid] = video_tasks[vid].get("begin", 0.0)

    total_generation_time = 0.0
    total_output_tokens = 0
    step_counter = 0

    print(f"Total Videos to process: {len(pending_vids)}")
    print("-" * 110)

    while pending_vids or active_vids:
        step_counter += 1
        
        # 1. 补充并发队列
        while len(active_vids) < args.max_concurrent and pending_vids:
            new_vid = pending_vids.pop(0)
            active_vids.append(new_vid)
            # 初始化 System Prompt
            session_messages[new_vid] = [{"role": "system", "content": system_prompt}]
            
        batch_inputs = []
        batch_vids = []
        vids_to_remove = []
        
        # 2. 收集当前并发视频的下一个 Chunk
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
                prompt_text = f"This is a video titled \"{title}\". "
                if preasr:
                    prompt_text += f"\nHere is previous commentary:\n{preasr}\n"
                prompt_text += "\nPlease comment on the current segment of the video."
            else:
                prompt_text = "Continue commenting on what happens next in the video."

            video_path = record["video"]
            if not os.path.exists(video_path):
                video_path = os.path.join(os.environ.get("LIVESPORTS3K_PATH", ""), video_path)
                
            frames = extract_frames(video_path, chunk_start, actual_end, args.frames_per_chunk)
            
            # 构建原生 vLLM 要求的消息格式
            content = []
            for img in frames:
                content.append({"type": "image", "image": img})
            content.append({"type": "text", "text": prompt_text})
            
            # 添加 User 消息到历史
            session_messages[vid].append({"role": "user", "content": content})
            
            # 维护滑动窗口 (System 占 1 个位置，后续每轮 User+Asst 占 2 个位置)
            while len(session_messages[vid]) > (1 + args.dialogue_window * 2):
                session_messages[vid].pop(1) # 移除最老的 User
                session_messages[vid].pop(1) # 移除最老的 Assistant

            # 提取当前窗口内所有的图片，用于传给 vLLM 的 multi_modal_data
            current_all_images = []
            for msg in session_messages[vid]:
                if isinstance(msg.get("content"), list):
                    for item in msg["content"]:
                        if item.get("type") == "image":
                            current_all_images.append(item["image"])

            # 使用 processor 生成 prompt 字符串
            prompt_str = processor.apply_chat_template(
                session_messages[vid], 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # 构造 vLLM 输入
            vllm_input = {
                "prompt": prompt_str,
                "multi_modal_data": {
                    "image": current_all_images if len(current_all_images) > 1 else current_all_images[0]
                }
            }
            batch_inputs.append(vllm_input)
            batch_vids.append(vid)
            
            vid_current_time[vid] += args.chunk_duration

        for vid in vids_to_remove:
            active_vids.remove(vid)
            del session_messages[vid]
            
        if not batch_inputs:
            continue

        print(f"\n[Step {step_counter:3d}] Processing {len(batch_inputs)} active video chunks... "
              f"(Completed: {len(completed_vids)}/{len(video_tasks)})")
        
        # 3. 执行 Batch 推理
        t0 = time.perf_counter()
        
        # vLLM 离线 API 自动处理 batch
        outputs = engine.generate(batch_inputs, sampling_params=sampling_params, use_tqdm=False)
        
        gen_time = time.perf_counter() - t0
        total_generation_time += gen_time

        # 4. 收集结果并保存
        batch_results = []
        step_output_tokens = 0
        
        for i, out in enumerate(outputs):
            vid = batch_vids[i]
            response_text = out.outputs[0].text.strip()
            step_output_tokens += len(out.outputs[0].token_ids)
            
            # 将生成的回复追加到历史中
            session_messages[vid].append({"role": "assistant", "content": response_text})
            
            record = video_tasks[vid]
            chunk_start = vid_current_time[vid] - args.chunk_duration
            chunk_end = min(vid_current_time[vid], record["end"])
            
            batch_results.append({
                "video": record["video"],
                "video_id": record["video_id"],
                "event_id": record["event_id"],
                "chunk_start": chunk_start,
                "chunk_end": chunk_end,
                "pred": response_text
            })
            
        total_output_tokens += step_output_tokens
        
        print(f"  Generation Time: {gen_time:.2f}s | Output Tokens: {step_output_tokens} | TPS: {step_output_tokens/gen_time:.1f} t/s")

        # 实时追加写入 JSONL 防止中断
        with open(out_file, "a", encoding="utf-8") as f:
            for res in batch_results:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")

    print("\n" + "=" * 110)
    print(f"Evaluation Completed. Results saved to {out_file}")
    print(f"Overall Generation Time: {total_generation_time:.2f}s")
    print(f"Overall Output TPS: {total_output_tokens / total_generation_time:.1f} t/s")

if __name__ == "__main__":
    main()
