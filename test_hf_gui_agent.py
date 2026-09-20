import os
import json
import time
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

# 1. 全局配置
MODEL_PATH = "/mnt/neimeng/nlp/projects/pretrain/luorun/workspace/GUI-Agent/open_verl/sft/verl_sft_test/guiagent-qwen3-8b-fsdp-fsdp2-sp1-fsdp-1202a1-4/global_step_8076/huggingface"
DATA_DIR = "/mnt/neimeng/nlp/projects/pretrain/luorun/workspace/GUI-Agent/session_logs/c386b13a-df1d-4309-9735-485ab8a4cd84"

# 窗口大小配置 (历史 User 消息数量)
HISTORY_WINDOW_SIZE = 20  
TOTAL_ROUNDS = 40

def main():
    print(f"Loading HF Model from {MODEL_PATH}...")
    
    # 2. 初始化 Processor 和 Model
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH, 
        min_pixels=256 * 28 * 28, 
        max_pixels=1280 * 28 * 28
    )
    
    # 使用 bfloat16 和 flash_attention_2 以匹配训练时的环境
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="flash_attention_2"
    )
    model.eval()

    # 3. 准备 System Prompt 和初始 Messages 列表
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

    # 维护一个原生的对话列表
    messages = [{"role": "system", "content": system_prompt}]

    # 获取数据集文件
    dataset_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.json')])
    dataset_files = dataset_files[:TOTAL_ROUNDS]

    print(f"\nStart HF Native Inference: {len(dataset_files)} rounds, Window Size: {HISTORY_WINDOW_SIZE}")
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
        user_text = "current position [0,0]"

        # 将新一轮的 User 消息加入列表
        new_user_msg = {
            "role": "user",
            "content": [
                {"type": "image", "image": raw_image},
                {"type": "text", "text": user_text}
            ]
        }
        messages.append(new_user_msg)

        # --- 2. 维护滑动窗口 ---
        # messages[0] 是 System Prompt
        # 后续每轮包含 1个 user 消息 + 1个 assistant 消息 (共 2 个元素)
        # 所以最大长度应该是 1 + HISTORY_WINDOW_SIZE * 2
        # 注意：当前轮次的 user 消息刚加进去，还没有 assistant 消息，所以判断条件是：
        while len(messages) > (1 + HISTORY_WINDOW_SIZE * 2):
            messages.pop(1) # 移除最老的 User 消息
            messages.pop(1) # 移除最老的 Assistant 消息

        # --- 3. 提取当前窗口内的所有图片 ---
        current_images = []
        for msg in messages:
            if isinstance(msg.get("content"), list):
                for item in msg["content"]:
                    if item.get("type") == "image":
                        current_images.append(item["image"])

        # --- 4. 生成 Prompt 和 Inputs ---
        t0 = time.perf_counter()
        
        prompt = processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # 将文本和积累的所有图片传给 processor
        inputs = processor(
            text=[prompt],
            images=current_images if current_images else None,
            return_tensors="pt",
            padding=True
        ).to(model.device)
        
        prefill_time = time.perf_counter() - t0

        # 打印一下第一轮的 Token 数量，方便我们排查
        if i == 0:
            print(f"\n[Debug] Round 1 Input Tokens: {inputs['input_ids'].shape[1]}")
            # 打印 Prompt 结尾的样子，看看有没有 \n
            print(f"[Debug] Round 1 Prompt Tail:\n{prompt[-100:]!r}\n")

        # --- 5. 执行推理 ---
        t1 = time.perf_counter()
        
        # 使用 Greedy Decoding (do_sample=False, temperature=0.0)
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False,
                use_cache=True
            )
            
        decode_time = time.perf_counter() - t1
        
        # --- 6. 结果处理 ---
        # 截取新生成的 Token
        input_len = inputs["input_ids"].shape[1]
        new_token_ids = generated_ids[0][input_len:]
        response_text = processor.tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
        
        # 将模型回复加入历史
        messages.append({"role": "assistant", "content": response_text})

        # --- 7. 打印统计 ---
        match_flag = "✅ MATCH" if response_text == gt_response else "❌ MISMATCH"
        
        print(f"\n{'-'*110}")
        print(f"[Turn {i+1:2d}/{len(dataset_files)}] "
              f"Prep: {prefill_time*1000:5.1f}ms | "
              f"Gen: {decode_time*1000:6.1f}ms | "
              f"CTX: {input_len:5d}")
        print(f"  GT:  {gt_response}")
        print(f"  Out: {response_text}")
        print(f"  Res: {match_flag}")

    print("\n" + "=" * 110)
    print("HF Native Inference Completed.")

if __name__ == "__main__":
    main()