"""100-turn VL streaming stress test with prefill/decode timing."""
import os
import sys
import io
import random
import urllib.request
from time import perf_counter
from PIL import Image
from transformers import AutoProcessor
from nanovllm import StreamingLLM, StreamingConfig, SamplingParams

random.seed(42)

IMAGE_URLS = [
    "http://images.cocodataset.org/val2017/000000000285.jpg",
    "http://images.cocodataset.org/val2017/000000039769.jpg",
    "http://images.cocodataset.org/val2017/000000397133.jpg",
    "http://images.cocodataset.org/val2017/000000252219.jpg",
]

TEXT_QUESTIONS = [
    "Explain distributed computing covering CAP theorem, consensus algorithms, and partition strategies.",
    "Describe the Agile lifecycle including ceremonies, roles, and how teams handle requirement changes.",
    "What are the fundamentals of computer networking covering OSI model, TCP/IP, DNS, HTTP and CDNs?",
    "Explain GPU computing and CUDA programming, comparing CPU vs GPU for deep learning workloads.",
    "What are security considerations for web apps? Cover auth, XSS, CSRF, encryption and compliance.",
    "Describe clean code principles including SOLID, common design patterns, and maintainability tips.",
    "Explain database normalization vs denormalization, relational vs NoSQL, and their trade-offs.",
    "What is DevOps? Explain CI/CD, infrastructure as code, monitoring and shared responsibility.",
    "Describe functional programming vs OOP, covering immutability, pure functions and monads.",
    "Explain gradient descent math, learning rate scheduling, momentum and the Adam optimizer.",
]

IMAGE_QUESTIONS = [
    "Describe what you see in this image in detail.",
    "What are the main objects in this image?",
    "What colors and textures are prominent in this image?",
    "How many distinct elements can you identify?",
    "What is the overall mood or atmosphere of this image?",
    "Describe the spatial layout and composition of this image.",
    "What activities or actions are happening in this image?",
    "Compare this image to a typical scene — what stands out?",
    "What story does this image tell?",
    "Identify any text, symbols or patterns in this image.",
]

SESSION_CONFIGS = [
    ("VL-A", "You are a visual assistant. Describe images briefly.", 0.6),
    ("VL-B", "You are an image analyst. Be very concise.", 0.5),
    ("Text", "You are a helpful assistant. Answer in 1-2 sentences.", 0.0),
    ("Mix",  "You see images and answer questions. Be brief.", 0.3),
]

def download_image(url):
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return Image.open(io.BytesIO(r.read())).convert("RGB")
    except Exception as e:
        print(f"  Failed: {url}: {e}")
        return Image.new("RGB", (224, 224), color=(128, 128, 200))

def process_image(processor, image):
    inputs = processor(
        text=["<|vision_start|><|image_pad|><|vision_end|>"],
        images=[image], return_tensors="pt", padding=True,
    )
    return {"pixel_values": inputs["pixel_values"], "image_grid_thw": inputs["image_grid_thw"]}

def main():
    model_path = os.path.expanduser(
        # "Qwen/Qwen3-VL-8B-Instruct-action"
        # 'Qwen/Qwen3-VL-30B-A3B-Instruct"
        # "Qwen/Qwen3.5-9B",
        "Qwen/Qwen3.5-35B-A3B"
    )

    print("=" * 120)
    print("StreamingLLM VL 100-Turn Stress Test (with prefill/decode timing)")
    print("=" * 120)

    print("\nDownloading images...")
    images = [download_image(url) for url in IMAGE_URLS]
    processor = AutoProcessor.from_pretrained(model_path)
    print("Processing images...")
    img_data = [process_image(processor, img) for img in images]
    for i, d in enumerate(img_data):
        print(f"  Image {i}: grid_thw={d['image_grid_thw'].tolist()}, pixels={d['pixel_values'].shape}")

    print("\nInitializing StreamingLLM...")
    llm = StreamingLLM(
        model_path,
        streaming_config=StreamingConfig(
            dialogue_window=3, history_window=256, history_sink=32, initial_padding=8,
        ),
        enforce_eager=False, 
        tensor_parallel_size=2,
        expert_parallel_size=4,
        is_multimodal=True,
    )

    params = SamplingParams(temperature=0.9, max_tokens=40)
    sessions = []
    for name, prompt, _ in SESSION_CONFIGS:
        sessions.append(llm.create_session(prompt, params))
    session_names = [c[0] for c in SESSION_CONFIGS]
    image_probs = [c[2] for c in SESSION_CONFIGS]

    print(f"\nSessions: {list(zip(session_names, sessions))}")
    print("=" * 120)

    sum_pf_time = sum_dc_time = 0.0
    sum_pf_tok = sum_dc_tok = 0
    total_vision_calls = 0

    turn_responses = {}

    for turn in range(100):
        # ---- Build requests ----
        reqs = []
        turn_img = {}
        for i, sid in enumerate(sessions):
            use_img = random.random() < image_probs[i]
            if use_img:
                idx = random.randint(0, len(img_data) - 1)
                msg = random.choice(IMAGE_QUESTIONS)
                reqs.append({
                    "session_id": sid, "message": msg,
                    "pixel_values": img_data[idx]["pixel_values"],
                    "image_grid_thw": img_data[idx]["image_grid_thw"],
                })
                turn_img[sid] = idx
            else:
                msg = random.choice(TEXT_QUESTIONS)
                reqs.append({"session_id": sid, "message": msg})
                turn_img[sid] = -1

        n_img = sum(1 for v in turn_img.values() if v >= 0)
        total_vision_calls += n_img

        # ---- Continue sessions ----
        active = []
        for req in reqs:
            message = req["message"]
            pv = req.get("pixel_values")
            thw = req.get("image_grid_thw")
            if pv is not None and llm.is_vl:
                message = llm._build_vision_message(message, thw)
            seq = llm.scheduler.continue_session(req["session_id"], message)
            if seq:
                if pv is not None and llm.is_vl:
                    seq._pending_vision = {"pixel_values": pv, "image_grid_thw": thw}
                    if seq.current_turn:
                        grids = thw.tolist() if hasattr(thw, 'tolist') else thw
                        seq.current_turn.image_grid_thw_list = [tuple(g) for g in grids]
                active.append(seq)

        # ---- Prefill phase ----
        t0 = perf_counter()
        pf_tok = 0
        while any(s.status.name == "WAITING" for s in active):
            seqs, is_pf, ops = llm.scheduler.schedule()
            if not seqs:
                break
            vision_data = None
            mrope_positions = None
            if is_pf and llm.is_vl:
                vision_data = llm._prepare_vision_data(seqs)
                mrope_positions = llm._compute_batch_mrope(seqs)
            
            if is_pf:
                pf_tok += sum(s.num_scheduled_tokens for s in seqs)
                
            tids = llm.model_runner.call("run_streaming", seqs, is_pf, ops, vision_data, mrope_positions)
                
            # Update MRoPE delta
            if is_pf and llm.is_vl and mrope_positions is not None:
                offset = 0
                for seq in seqs:
                    n = seq.num_scheduled_tokens
                    if n > 0:
                        seq_pos = mrope_positions[:, offset:offset + n]
                        max_pos = int(seq_pos.max().item())
                        logical_end = seq.num_cached_tokens + n - 1
                        seq.mrope_position_delta = max_pos - logical_end
                    offset += n
                    
            llm.scheduler.postprocess(seqs, tids, is_prefill=is_pf)
        pf_time = perf_counter() - t0

        # ---- Decode phase ----
        t0 = perf_counter()
        dc_tok = 0
        while any(not s.is_idle and not s.is_finished for s in active):
            seqs, is_pf, ops = llm.scheduler.schedule()
            if not seqs:
                break
            tids = llm.model_runner.call("run_streaming", seqs, is_pf, ops, None, None)
            llm.scheduler.postprocess(seqs, tids, is_prefill=is_pf)
            if not is_pf:
                dc_tok += len(seqs)
        dc_time = perf_counter() - t0

        sum_pf_time += pf_time
        sum_pf_tok += pf_tok
        sum_dc_time += dc_time
        sum_dc_tok += dc_tok

        pf_tps = pf_tok / pf_time if pf_time > 0 else 0
        dc_tps = dc_tok / dc_time if dc_time > 0 else 0

        # Collect responses
        responses = {}
        for seq in active:
            responses[seq.seq_id] = llm.tokenizer.decode(seq.completion_token_ids, skip_special_tokens=True)
            
        turn_responses[turn] = responses

        stats0 = llm.get_stats(sessions[0])
        evict = "🔄" if stats0["history_tokens"] > 0 else "  "
        img_flag = f"📷×{n_img}" if n_img > 0 else "💬×4 "

        print(
            f"{evict}[Turn {turn+1:3d}] {img_flag} "
            f"PF:{pf_tok:5d}tok/{pf_time*1000:7.1f}ms={pf_tps:8.0f}t/s | "
            f"DC:{dc_tok:3d}tok/{dc_time*1000:7.1f}ms={dc_tps:6.0f}t/s | "
            f"CTX:{stats0['total_tokens']:5d} "
            f"Hist:{stats0['history_tokens']:4d} "
            f"Pad:{stats0['padding_tokens']:3d} "
            f"Turns:{stats0['dialogue_turns']} "
            f"Blk:{stats0['blocks']:3d} "
            f"VC:{stats0.get('vision_cache_entries',0)}"
        )

        # Every 10 turns: show all sessions
        if (turn + 1) % 10 == 0:
            print(f"\n  {'─'*110}")
            for i, sid in enumerate(sessions):
                name = session_names[i]
                stats = llm.get_stats(sid)
                resp = responses.get(sid, "")
                ii = turn_img.get(sid, -1)
                tag = f"📷img{ii}" if ii >= 0 else "💬text"
                q = reqs[i]["message"][:50] + "..."
                print(
                    f"  {name}(S{sid}) {tag}:\n"
                    f"    Q: {q}\n"
                    f"    A: {resp}...\n"
                    f"    [tok={stats['total_tokens']:5d} "
                    f"cached={stats['cached_tokens']:5d} "
                    f"hist={stats['history_tokens']:4d} "
                    f"pad={stats['padding_tokens']:3d} "
                    f"turns={stats['dialogue_turns']} "
                    f"blk={stats['blocks']:3d} "
                    f"vc={stats.get('vision_cache_entries',0)}]"
                )
            print(f"  {'─'*110}\n")

    # ---- Streaming test ----
    print(f"\n{'='*120}")
    print("Streaming output:")
    print("  [Text] Summarize our conversation.")
    print("  A: ", end="", flush=True)
    for chunk in llm.chat(sessions[2], "Summarize our entire conversation.", stream=True):
        print(chunk, end="", flush=True)
    print()

    print("  [VL+Img] What is in this image?")
    print("  A: ", end="", flush=True)
    for chunk in llm.chat(
        sessions[0], "What is in this image?", stream=True,
        pixel_values=img_data[0]["pixel_values"],
        image_grid_thw=img_data[0]["image_grid_thw"],
    ):
        print(chunk, end="", flush=True)
    print()

    # ---- Summary ----
    print(f"\n{'='*120}")
    print(f"SUMMARY: 100 turns × {len(sessions)} sessions")
    print(f"{'='*120}")
    print(f"  Prefill: {sum_pf_tok:,}tok in {sum_pf_time:.2f}s = {sum_pf_tok/sum_pf_time:,.0f} tok/s avg")
    print(f"  Decode:  {sum_dc_tok:,}tok in {sum_dc_time:.2f}s = {sum_dc_tok/sum_dc_time:,.0f} tok/s avg")
    print(f"  Vision encoder calls: {total_vision_calls}")
    print()
    print("  Per-session:")
    for i, sid in enumerate(sessions):
        stats = llm.get_stats(sid)
        print(
            f"    {session_names[i]}(S{sid}): "
            f"tok={stats['total_tokens']:5d} "
            f"cached={stats['cached_tokens']:5d} "
            f"hist={stats['history_tokens']:4d} "
            f"pad={stats['padding_tokens']:3d} "
            f"turns={stats['dialogue_turns']} "
            f"blk={stats['blocks']:3d} "
            f"vc={stats.get('vision_cache_entries',0)}"
        )
    est = 100 * 250 + 50
    actual = max(llm.get_stats(s)["total_tokens"] for s in sessions)
    print(f"\n  Context: {actual}tok vs ~{est}tok without streaming ({(1-actual/est)*100:.0f}% reduction)")
    llm.close_all()
    print("\nDone.")


if __name__ == "__main__":
    main()
