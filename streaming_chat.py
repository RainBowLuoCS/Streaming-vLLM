import os
from time import perf_counter
from nanovllm import StreamingLLM, StreamingConfig, SamplingParams

import random 

LONG_QUESTIONS = [
    "Please explain in detail the differences between supervised learning, unsupervised learning, and reinforcement learning in machine learning. What are the typical use cases for each approach, and what are their respective advantages and disadvantages when applied to real-world problems?",
    "Can you describe the complete lifecycle of a software development project using the Agile methodology? Include all the key ceremonies, roles, artifacts, and how teams typically handle changing requirements during sprint cycles.",
    "What are the fundamental principles behind distributed computing systems? Please cover topics such as the CAP theorem, consensus algorithms, data partitioning strategies, and how modern systems handle network failures and data consistency.",
    "Explain the architecture of a modern web application from frontend to backend. Cover topics including client-side rendering versus server-side rendering, RESTful APIs versus GraphQL, database choices, caching strategies, and deployment pipelines.",
    "Describe the evolution of natural language processing from rule-based systems to modern transformer architectures. What were the key breakthroughs at each stage, and how did attention mechanisms fundamentally change the field?",
    "What are the best practices for designing a scalable microservices architecture? Please discuss service discovery, load balancing, circuit breakers, API gateways, event-driven communication, and strategies for handling distributed transactions.",
    "Explain the concept of containerization and how Docker and Kubernetes work together to orchestrate applications at scale. What are the key components of a Kubernetes cluster, and how does it handle auto-scaling and self-healing?",
    "Can you walk through the mathematics behind gradient descent optimization? Cover the basic algorithm, learning rate scheduling, momentum, Adam optimizer, and common pitfalls like vanishing or exploding gradients in deep neural networks.",
    "Describe the principles of functional programming and how they differ from object-oriented programming. What are concepts like immutability, pure functions, higher-order functions, monads, and how do they help in building reliable software?",
    "What are the key security considerations when building a web application? Please cover authentication and authorization mechanisms, common vulnerabilities like SQL injection and XSS, encryption, secure API design, and compliance requirements.",
    "Explain the concept of database normalization and denormalization. When should you use relational databases versus NoSQL databases? What are the trade-offs in terms of consistency, availability, query performance, and scalability?",
    "Describe how modern compilers work, from lexical analysis to code generation. What are the key phases of compilation, and how do optimizations like loop unrolling, inlining, and register allocation improve the generated machine code?",
    "What is the role of DevOps in modern software development? Please explain continuous integration, continuous deployment, infrastructure as code, monitoring and observability, and how teams can implement a culture of shared responsibility.",
    "Explain the fundamentals of computer networking, covering the OSI model, TCP/IP protocol suite, DNS resolution, HTTP/HTTPS, WebSockets, and how CDNs work to improve content delivery performance globally.",
    "Can you describe the principles behind clean code and software craftsmanship? What are SOLID principles, design patterns like Factory, Observer, and Strategy, and how do they contribute to maintainable and extensible codebases?",
    "What are the challenges and solutions in building real-time data processing pipelines? Cover technologies like Apache Kafka, Apache Flink, stream processing versus batch processing, and exactly-once semantics in distributed systems.",
    "Explain the concept of GPU computing and how CUDA programming enables parallel computation. What are the key differences between CPU and GPU architectures, and how do frameworks like PyTorch leverage GPUs for deep learning training?",
    "Describe the principles of test-driven development and how it integrates with continuous integration workflows. What are unit tests, integration tests, end-to-end tests, and how should teams balance test coverage with development velocity?",
    "What are the key concepts in operating system design? Please cover process scheduling, memory management, virtual memory, file systems, inter-process communication, and how modern operating systems handle concurrency and synchronization.",
    "Explain the fundamentals of cryptography including symmetric and asymmetric encryption, hash functions, digital signatures, certificate authorities, and how TLS/SSL establishes secure communication channels over the internet.",
]


def main():
    path = os.path.expanduser(
        "/mnt/neimeng/nlp/projects/pretrain/luorun/datasets/checkpoints/Qwen/Qwen3-8B"
    )

    llm = StreamingLLM(
        path,
        streaming_config=StreamingConfig(
            dialogue_window=20,
            history_window=1024,
            history_sink=0,
            initial_padding=14,
        ),
        enforce_eager=False,
        tensor_parallel_size=8,
        kvcache_block_size=256
    )

    params = SamplingParams(temperature=0.9, max_tokens=256)

    # Create 4 concurrent sessions
    sessions = [
        llm.create_session("You are a helpful assistant. Be concise, answer in 1-2 sentences.", params),
        llm.create_session("You are a Python expert. Give brief answers.", params),
        llm.create_session("You are a systems architect. Keep answers short.", params),
        llm.create_session("You are a math tutor. Be brief and clear.", params),
    ]

    num_turns = 100
    total_prefill_tokens = 0
    total_prefill_time = 0.0
    total_decode_tokens = 0
    total_decode_time = 0.0

    print(f"{'='*100}")
    print(f"StreamingLLM Benchmark: {num_turns} turns, {len(sessions)} sessions, dialogue_window={llm.streaming_config.dialogue_window}")
    print(f"{'='*100}")

    for turn in range(num_turns):
        # Pick a long question (cycle through)

        random.shuffle(LONG_QUESTIONS)
        msg = LONG_QUESTIONS[turn % len(LONG_QUESTIONS)]

        # Build batch request for all sessions
        requests = [{"session_id": sid, "message": msg} for i,sid in enumerate(sessions)]

        # Time the batch chat
        t_start = perf_counter()

        # We need to measure prefill and decode separately
        # First: continue_session triggers prefill scheduling
        active = []
        for req in requests:
            seq = llm.scheduler.continue_session(req["session_id"], req["message"])
            if seq:
                active.append(seq)

        # Run prefill steps
        prefill_tokens = 0
        t_prefill_start = perf_counter()
        while any(s.status.name == "WAITING" for s in active):
            seqs, is_prefill, kv_ops = llm.scheduler.schedule()
            if not seqs:
                break
            if is_prefill:
                prefill_tokens += sum(s.num_scheduled_tokens for s in seqs)
            token_ids = llm.model_runner.call("run_streaming", seqs, is_prefill, kv_ops)
            llm.scheduler.postprocess(seqs, token_ids, is_prefill)
        t_prefill_end = perf_counter()
        prefill_time = t_prefill_end - t_prefill_start

        # Run decode steps
        decode_tokens = 0
        t_decode_start = perf_counter()
        while any(not s.is_idle and not s.is_finished for s in active):
            seqs, is_prefill, kv_ops = llm.scheduler.schedule()
            if not seqs:
                break
            token_ids = llm.model_runner.call("run_streaming", seqs, is_prefill, kv_ops)
            llm.scheduler.postprocess(seqs, token_ids, is_prefill)
            if not is_prefill:
                decode_tokens += len(seqs)
        t_decode_end = perf_counter()
        decode_time = t_decode_end - t_decode_start

        total_prefill_tokens += prefill_tokens
        total_prefill_time += prefill_time
        total_decode_tokens += decode_tokens
        total_decode_time += decode_time

        t_total = perf_counter() - t_start

        # Collect results
        results = {}
        for s in active:
            if s.is_idle or s.is_finished:
                results[s.seq_id] = llm.tokenizer.decode(s.completion_token_ids, skip_special_tokens=True)

        # Stats from first session
        stats = llm.get_stats(sessions[1])

        # Print
        prefill_tps = prefill_tokens / prefill_time if prefill_time > 0 else 0
        decode_tps = decode_tokens / decode_time if decode_time > 0 else 0

        print(
            f"[Turn {turn+1:3d}/{num_turns}] "
            f"Prefill: {prefill_tokens:4d}tok/{prefill_time*1000:6.1f}ms={prefill_tps:7.0f}tok/s | "
            f"Decode: {decode_tokens:4d}tok/{decode_time*1000:6.1f}ms={decode_tps:7.0f}tok/s | "
            f"Total: {t_total*1000:7.1f}ms | "
            f"CTX: {stats['total_tokens']:5d}tok "
            f"Hist: {stats['history_tokens']:4d} "
            f"Pad: {stats['padding_tokens']:3d} "
            f"Turns: {stats['dialogue_turns']} "
            f"Blocks: {stats['blocks']:3d}"
        )

        # Show response snippet every 10 turns
        if (turn + 1) % 10 == 0:
            resp = results.get(sessions[1], "")
            print(f"         Response: {resp[:256]}...")
            print()

    # Final summary
    print(f"\n{'='*100}")
    print(f"SUMMARY ({num_turns} turns × {len(sessions)} sessions)")
    print(f"{'='*100}")
    print(f"Total prefill: {total_prefill_tokens:,} tokens in {total_prefill_time:.2f}s "
          f"= {total_prefill_tokens/total_prefill_time:,.0f} tok/s avg")
    print(f"Total decode:  {total_decode_tokens:,} tokens in {total_decode_time:.2f}s "
          f"= {total_decode_tokens/total_decode_time:,.0f} tok/s avg")
    print()

    # Show context growth comparison
    print(f"Context comparison (session 0):")
    stats = llm.get_stats(sessions[0])
    print(f"  With StreamingLLM:    {stats['total_tokens']:,} tokens, {stats['blocks']} blocks")

    # Estimate without streaming: sum of all user messages + all responses
    avg_user_tokens = 80  # rough estimate
    avg_resp_tokens = 50
    no_streaming = avg_user_tokens * num_turns + avg_resp_tokens * num_turns + 50  # +50 for system
    print(f"  Without StreamingLLM: ~{no_streaming:,} tokens (estimated)")
    print(f"  Context reduction:    ~{(1 - stats['total_tokens']/no_streaming)*100:.0f}%")

    llm.close_all()
    print("\nDone.")


if __name__ == "__main__":
    main()