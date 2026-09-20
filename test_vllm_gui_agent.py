import os
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

# 然后再导入其他库
import json
import time
from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

import json
import time
from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
import vllm.envs as envs

# 禁用 V1 多进程，防止有些环境报错
envs.VLLM_ENABLE_V1_MULTIPROCESSING = False
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

# 1. 全局配置
MODEL_PATH = "/mnt/neimeng/nlp/projects/pretrain/luorun/workspace/GUI-Agent/open_verl/sft/verl_sft_test/guiagent-qwen3-8b-fsdp-fsdp2-sp1-fsdp-1202a1-4/global_step_8076/huggingface"
DATA_DIR = "/mnt/neimeng/nlp/projects/pretrain/luorun/workspace/GUI-Agent/session_logs/c386b13a-df1d-4309-9735-485ab8a4cd84"

# 窗口大小配置 (历史 User 消息数量)
HISTORY_WINDOW_SIZE = 20  
TOTAL_ROUNDS = 40

def main():
    print(f"Loading vLLM Engine from {MODEL_PATH}...")
    
    # 2. 初始化 Processor
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH, 
        min_pixels=256 * 28 * 28, 
        max_pixels=1280 * 28 * 28
    )
    
    # 3. 初始化 vLLM 引擎
    # enable_prefix_caching=True 开启 Radix Cache
    # limit_mm_per_prompt 必须大于等于窗口大小，否则无法处理历史记录中的多张图片
    engine = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=8,
        max_model_len=32768,          # 必须开大，容纳 20 张图
        enable_prefix_caching=True, 
        trust_remote_code=True,
        limit_mm_per_prompt={"image": HISTORY_WINDOW_SIZE + 1}, 
        gpu_memory_utilization=0.5,
    )
    
    # 贪婪解码，保证输出稳定
    sampling_params = SamplingParams(
        temperature=0.9, 
        max_tokens=30
    )

    # 4. 准备 System Prompt 和初始 Messages 列表
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

    messages = [{"role": "system", "content": system_prompt}]

    # 获取数据集文件
    dataset_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.json')])
    dataset_files = dataset_files[:TOTAL_ROUNDS]

    print(f"\nStart vLLM Offline Inference: {len(dataset_files)} rounds, Window Size: {HISTORY_WINDOW_SIZE}")
    print("=" * 110)

    for i, f_name in enumerate(dataset_files):
        # --- 1. 准备当前轮次的数据 ---
        json_path = os.path.join(DATA_DIR, f_name)
        img_path = os.path.join(DATA_DIR, f_name.replace('.json', '.jpg'))
        
        # 读取 Ground Truth (GT)
        with open(json_path, 'r', encoding='utf-8') as f:
            item = json.load(f)
            gt_response = item.get('model_response', '')
            
        raw_image = Image.open(img_path).convert("RGB")
        user_text = "\ncurrent mouse position\n[0,0]"

        # 将新一轮的 User 消息加入列表
        new_user_msg = {
            "role": "user",
            "content": [
                {"type": "text","text": "current game screen\n"},
                {"type": "image", "image": raw_image},
                {"type": "text", "text": user_text}
            ]
        }
        messages.append(new_user_msg)

        # --- 2. 维护滑动窗口 ---
        # 超过窗口时，移除最老的 User 和 Assistant 消息
        while len(messages) > (1 + HISTORY_WINDOW_SIZE * 2):
            messages.pop(1) 
            messages.pop(1) 

        # --- 3. 提取当前窗口内的所有图片 ---
        current_images = []
        for msg in messages:
            if isinstance(msg.get("content"), list):
                for item in msg["content"]:
                    if item.get("type") == "image":
                        current_images.append(item["image"])

        # --- 4. 生成 Prompt ---
        t0_prep = time.perf_counter()
        
        prompt = processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # vLLM 多模态输入格式
        inputs = {
            "prompt": prompt,
            "multi_modal_data": {
                # 如果只有一张图，传对象；如果有多张，传列表
                "image": current_images if len(current_images) > 1 else current_images[0]
            }
        }
        
        prep_time = time.perf_counter() - t0_prep

        # 打印第一轮的 Prompt 结尾
        if i < 3:
            print(f"\n[Debug] Round 1 Prompt Tail:\n{prompt[-200:]!r}\n")

        # --- 5. 执行推理 ---
        t0_gen = time.perf_counter()
        
        # vLLM generate 返回的是一个 RequestOutput 列表
        outputs = engine.generate([inputs], sampling_params=sampling_params, use_tqdm=False)
        
        gen_time = time.perf_counter() - t0_gen
        
        # --- 6. 结果处理 ---
        generated_output = outputs[0]
        response_text = generated_output.outputs[0].text.strip()
        
        # 获取 Token 统计
        prompt_tokens = len(generated_output.prompt_token_ids)
        
        # 将模型回复加入历史
        messages.append({"role": "assistant", "content": response_text})

        # --- 7. 打印统计 ---
        match_flag = "✅ MATCH" if response_text == gt_response else "❌ MISMATCH"
        
        print(f"\n{'-'*110}")
        print(f"[Turn {i+1:2d}/{len(dataset_files)}] "
              f"Prep: {prep_time*1000:5.1f}ms | "
              f"Gen: {gen_time*1000:6.1f}ms | "
              f"Prompt Toks: {prompt_tokens:5d}")
        print(f"  GT:  {gt_response}")
        print(f"  Out: {response_text}")
        print(f"  Res: {match_flag}")

    print("\n" + "=" * 110)
    print("vLLM Offline Inference Completed.")

if __name__ == "__main__":
    main()