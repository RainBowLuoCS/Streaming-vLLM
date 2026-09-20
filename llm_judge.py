#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import argparse
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time

# =====================================================================
# 自定义 API 调用函数 (基于你提供的代码)
# =====================================================================
def call_gpt4o_mini_judge(prompt_text: str) -> str:
    """
    调用自定义 API 执行 GPT-4o-mini 推理。
    注意：这里是纯文本对比，不传入图片，所以移除了 image_url 的部分。
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": "run.luo-xxxxxx"  # 你的 API Key
    }
    
    payload = {
        "model": "gpt-4o-mini",  # 使用 gpt-4o-mini 作为裁判
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text}
                ]
            }
        ],
        "max_tokens": 1024,
        "temperature": 0.0,      # 裁判需要确定性输出
        "seed": 42               # 固定种子以保证可复现性
    }

    # 简单的重试机制，防止网络抖动
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(
                url="http://10.234.32.86:8000/lumi-proxy/v1/chat/completions",
                headers=headers,
                data=json.dumps(payload),
                timeout=30
            )
            response.raise_for_status()  # 检查 HTTP 状态码
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"[ERROR] API Call failed after {max_retries} attempts: {e}")
                return ""
            time.sleep(1)

# =====================================================================
# 裁判逻辑
# =====================================================================
def judge_ab(a_id: str, a_pred: str, b_id: str, b_pred: str, gt_asr: str) -> str:
    """
    让 GPT 比较 A 和 B 哪个更好。
    """
    ab_prompt = (
        'You are an expert in video commentary. '
        'Your task is to review two commentaries (Commentary A and Commentary B), and select the one that better aligns with the human commentary. '
        'You should consider the criteria:\n'
        '1. Semantic Alignment: The commentary should convey the same meaning, details, and key points as the human commentary.\n'
        'If the above criteria is not enough to judge, then consider:\n'
        '2. Stylistic Consistency: The commentary should maintain a tone, word choice, and structure similar to the human commentary.\n'
        f'\n---Commentary A---\n{a_pred}\n----------\n'
        f'\n---Commentary B---\n{b_pred}\n----------\n'
        f'\n---Human Commentary---\n{gt_asr}\n----------\n'
        '\nYour response should be exactly "Commentary A is better aligned with the human commentary" or "Commentary B is better aligned with the human commentary". Do not output anything else.\n'
    )
    
    resp = call_gpt4o_mini_judge(ab_prompt)
    
    if 'Commentary A' in resp:
        return a_id
    elif 'Commentary B' in resp:
        return b_id
    else:
        return 'tie'

def evaluate_single_item(item: tuple, model_id: str, baseline_id: str, video_event_id_to_gt_asr: dict, video_event_id_to_baseline_pred: dict) -> dict:
    """
    评估单个样本。为了公平起见，进行两次测试：(Model vs Baseline) 和 (Baseline vs Model)。
    """
    video_event_id, model_pred = item
    
    gt_asr = video_event_id_to_gt_asr.get(video_event_id, "")
    baseline_pred = video_event_id_to_baseline_pred.get(video_event_id, "")
    
    if not gt_asr or not baseline_pred:
        return None

    # 测试 1: A=Model, B=Baseline
    ab_winner = judge_ab(model_id, model_pred, baseline_id, baseline_pred, gt_asr)
    
    # 测试 2: A=Baseline, B=Model (交换顺序防止位置偏见)
    ba_winner = judge_ab(baseline_id, baseline_pred, model_id, model_pred, gt_asr)
    
    return {
        'video_event_id': video_event_id,
        'ab_winner': ab_winner,
        'ba_winner': ba_winner
    }

# =====================================================================
# 主程序
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="LLM Judge for LiveSports-3K CC")
    parser.add_argument('--model_id', type=str, required=False, default='vllm_baseline',help='Model name to compare (e.g., StreamingLLM)')
    parser.add_argument('--prediction_jsonl', type=str, required=False, default='/mnt/neimeng/nlp/projects/pretrain/luorun/workspace/streaming-vllm-test/results/livesports3k/vllm_baseline/predictions.jsonl',help='Path to model predictions in JSONL format')
    parser.add_argument('--baseline_id', type=str, required=False, default='gemini-1.5-pro', help='Baseline model id (e.g., LLaVA-Video-72B-Qwen2)')
    parser.add_argument('--baseline_jsonl', type=str, required=False, default='/mnt/neimeng/nlp/projects/pretrain/luorun/workspace/streaming-vllm-test/results/livesports3k/livecc_baseline/Gemini-1.5-pro.jsonl',help='Baseline model predictions in JSONL format')
    parser.add_argument('--output_dir', type=str, default='./results/livesports3kcc/judges/', help='Directory to save judgment results')
    parser.add_argument('--num_workers', type=int, default=32, help='Number of concurrent API requests')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, f'{args.baseline_id}_vs_{args.model_id}.jsonl')
    
    print("=" * 80)
    print(f"Starting LLM Judge: {args.model_id} vs {args.baseline_id}")
    print("=" * 80)

    # 1. 加载 Ground Truth ASR
    print("Loading Ground Truth dataset...")
    try:
        from datasets import load_dataset
        ds = load_dataset('stdKonjac/LiveSports-3K', name='LiveSports_3K_CC', split="test")
        video_event_id_to_gt_asr = {}
        for datum in ds:
            vid = datum['video_id'] + '_' + str(datum['event_id'])
            video_event_id_to_gt_asr[vid] = datum['event_asr_text']
    except Exception as e:
        print(f"[ERROR] Failed to load dataset from huggingface: {e}")
        return

    # 2. 加载 Baseline 预测
    print(f"Loading Baseline predictions from {args.baseline_jsonl}...")
    video_event_id_to_baseline_pred = {}
    with open(args.baseline_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            datum = json.loads(line)
            vid = datum['video_id'] + '_' + str(datum['event_id'])
            video_event_id_to_baseline_pred[vid] = datum['pred']

    # 3. 加载我们的模型预测
    print(f"Loading Model predictions from {args.prediction_jsonl}...")
    video_event_id_to_model_pred = {}
    with open(args.prediction_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            datum = json.loads(line)
            vid = datum['video_id'] + '_' + str(datum['event_id'])
            video_event_id_to_model_pred[vid] = datum['pred']

    # 找出共有的样本
    common_vids = set(video_event_id_to_model_pred.keys()) & set(video_event_id_to_baseline_pred.keys())
    print(f"Found {len(common_vids)} common samples to evaluate.")

    items_to_evaluate = [(vid, video_event_id_to_model_pred[vid]) for vid in common_vids]

    # 4. 并发调用 API 进行裁判
    print(f"Starting API calls with {args.num_workers} workers...")
    winner_results = []
    
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [
            executor.submit(
                evaluate_single_item, 
                item, 
                args.model_id, 
                args.baseline_id, 
                video_event_id_to_gt_asr, 
                video_event_id_to_baseline_pred
            ) 
            for item in items_to_evaluate
        ]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Judging"):
            res = future.result()
            if res is not None:
                winner_results.append(res)

    # 5. 保存结果并计算胜率
    print("\nSaving results and calculating win rate...")
    with open(save_path, 'w', encoding='utf-8') as f:
        for res in winner_results:
            f.write(json.dumps(res, ensure_ascii=False) + '\n')

    win_count = 0
    total_matches = 0
    
    for res in winner_results:
        # 每次评估包含了 AB 和 BA 两次对决
        if res['ab_winner'] == args.model_id:
            win_count += 1
        if res['ba_winner'] == args.model_id:
            win_count += 1
        total_matches += 2

    if total_matches > 0:
        win_rate = (win_count / total_matches) * 100
        output_str = f"Winning Rate for {args.model_id} vs. {args.baseline_id}: {win_rate:.2f}% ({win_count}/{total_matches})"
        print("\n" + "=" * 80)
        print(output_str)
        print("=" * 80)
        
        # 写入日志
        with open(os.path.join(args.output_dir, 'log.txt'), 'a', encoding='utf-8') as f:
            f.write(output_str + '\n')
    else:
        print("\n[WARNING] No valid matches were evaluated.")

if __name__ == '__main__':
    main()