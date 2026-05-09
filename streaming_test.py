"""100-turn text-only streaming stress test with prefill/decode timing."""
import os
import random
from time import perf_counter
from streamingvllm import StreamingLLM, StreamingConfig, SamplingParams

random.seed(42)

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

SESSION_NAMES = ["Alice", "Bob", "Carol", "Dave"]

def main():
    # 替换为你的纯文本模型路径
    path = os.path.expanduser(
        "./checkpoints/Qwen/Qwen3-8B"
    )

    print("=" * 120)
    print("StreamingLLM Text-Only 100-Turn Stress Test (with prefill/decode timing)")
    print("=" * 120)

    print("\nInitializing StreamingLLM...")
    llm = StreamingLLM(
        path,
        streaming_config=StreamingConfig(
            dialogue_window=3, history_window=256, history_sink=32, initial_padding=8,
        ),
        enforce_eager=False, tensor_parallel_size=8, is_multimodal=False,
    )

    params = SamplingParams(temperature=0.7, max_tokens=50)
    sessions = []
    for name in SESSION_NAMES:
        sessions.append(llm.create_session(f"You are {name}'s assistant. Answer in 1-2 sentences.", params))

    print(f"\nSessions: {list(zip(SESSION_NAMES, sessions))}")
    print("=" * 120)

    sum_pf_time = sum_dc_time = 0.0
    sum_pf_tok = sum_dc_tok = 0

    turn_responses = {}

    for turn in range(100):
        # ---- Build requests ----
        reqs = []
        for i, sid in enumerate(sessions):
            msg = random.choice(TEXT_QUESTIONS)
            reqs.append({"session_id": sid, "message": msg})

        # ---- Continue sessions ----
        active = []
        for req in reqs:
            seq = llm.scheduler.continue_session(req["session_id"], req["message"])
            if seq:
                active.append(seq)

        # ---- Prefill phase ----
        t0 = perf_counter()
        pf_tok = 0
        while any(s.status.name == "WAITING" for s in active):
            seqs, is_pf, ops = llm.scheduler.schedule()
            if not seqs:
                break
            if is_pf:
                pf_tok += sum(s.num_scheduled_tokens for s in seqs)
                
            tids = llm.model_runner.call("run_streaming", seqs, is_pf, ops, None, None)
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

        print(
            f"{evict}[Turn {turn+1:3d}] "
            f"PF:{pf_tok:5d}tok/{pf_time*1000:7.1f}ms={pf_tps:8.0f}t/s | "
            f"DC:{dc_tok:3d}tok/{dc_time*1000:7.1f}ms={dc_tps:6.0f}t/s | "
            f"CTX:{stats0['total_tokens']:5d} "
            f"Hist:{stats0['history_tokens']:4d} "
            f"Pad:{stats0['padding_tokens']:3d} "
            f"Turns:{stats0['dialogue_turns']} "
            f"Blk:{stats0['blocks']:3d}"
        )

        # Every 10 turns: show all sessions
        if (turn + 1) % 10 == 0:
            print(f"\n  {'─'*110}")
            for i, sid in enumerate(sessions):
                name = SESSION_NAMES[i]
                stats = llm.get_stats(sid)
                resp = responses.get(sid, "")
                q = reqs[i]["message"][:50] + "..."
                print(
                    f"  {name}(S{sid}):\n"
                    f"    Q: {q}\n"
                    f"    A: {resp[:120]}...\n"
                    f"    [tok={stats['total_tokens']:5d} "
                    f"cached={stats['cached_tokens']:5d} "
                    f"hist={stats['history_tokens']:4d} "
                    f"pad={stats['padding_tokens']:3d} "
                    f"turns={stats['dialogue_turns']} "
                    f"blk={stats['blocks']:3d}]"
                )
            print(f"  {'─'*110}\n")

    # ---- Summary ----
    print(f"\n{'='*120}")
    print(f"SUMMARY: 100 turns × {len(sessions)} sessions")
    print(f"{'='*120}")
    print(f"  Prefill: {sum_pf_tok:,}tok in {sum_pf_time:.2f}s = {sum_pf_tok/sum_pf_time:,.0f} tok/s avg")
    print(f"  Decode:  {sum_dc_tok:,}tok in {sum_dc_time:.2f}s = {sum_dc_tok/sum_dc_time:,.0f} tok/s avg")
    print()
    print("  Per-session:")
    for i, sid in enumerate(sessions):
        stats = llm.get_stats(sid)
        print(
            f"    {SESSION_NAMES[i]}(S{sid}): "
            f"tok={stats['total_tokens']:5d} "
            f"cached={stats['cached_tokens']:5d} "
            f"hist={stats['history_tokens']:4d} "
            f"pad={stats['padding_tokens']:3d} "
            f"turns={stats['dialogue_turns']} "
            f"blk={stats['blocks']:3d}"
        )
    est = 100 * 90 + 50
    actual = max(llm.get_stats(s)["total_tokens"] for s in sessions)
    print(f"\n  Context: {actual}tok vs ~{est}tok without streaming ({(1-actual/est)*100:.0f}% reduction)")
    llm.close_all()
    print("\nDone.")

if __name__ == "__main__":
    main()
