#!/usr/bin/env python3
"""
Multi-core data-parallel inference benchmark for SigLIP2 Giant 256x256.

Uses NEURON_RT_VISIBLE_CORES per-process approach (not torch_neuronx.DataParallel).
Spawns one process per core, staggering model loads to avoid host memory contention
(critical on inf2.xlarge with only 4 vCPUs / ~16 GB host RAM), then benchmarks
all cores concurrently.

Usage:
    # trn2 LNC=2 (4 cores):
    python3 dp_benchmark.py --model siglip2_giant_256_lnc2.pt --cores 4

    # trn2 LNC=1 (8 cores):
    NEURON_LOGICAL_NC_CONFIG=1 python3 dp_benchmark.py --model siglip2_giant_256_lnc1.pt --cores 8

    # inf2.xlarge (2 cores):
    python3 dp_benchmark.py --model siglip2_giant_256_bs1_baseline.pt --cores 2
"""

import argparse
import json
import multiprocessing
import os
import time


def worker(
    core_id,
    model_file,
    num_warmup,
    num_iterations,
    load_lock,
    bench_barrier,
    result_queue,
):
    """Run inference on a single core."""
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(core_id)
    os.environ["NEURON_RT_LOG_LEVEL"] = "ERROR"

    import numpy as np
    import torch
    import torch_neuronx

    # Stagger model loading to avoid host memory contention.
    # Each process acquires the lock, loads the model, then releases.
    with load_lock:
        print(f"  Core {core_id}: Loading model...", flush=True)
        t0 = time.time()
        model = torch.jit.load(model_file)
        model.eval()
        print(f"  Core {core_id}: Loaded in {time.time() - t0:.1f}s", flush=True)

    inp = torch.randn(1, 3, 256, 256)

    # Warmup (can run concurrently)
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model(inp)
    print(f"  Core {core_id}: Warmup done ({num_warmup} iters)", flush=True)

    # Synchronize before benchmark so all cores start at the same time
    bench_barrier.wait()

    # Benchmark
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(inp)
        times.append(time.perf_counter() - start)

    times_ms = np.array(times) * 1000
    result_queue.put(
        {
            "core_id": core_id,
            "throughput": round(1000.0 / np.mean(times_ms), 2),
            "latency_avg_ms": round(np.mean(times_ms), 2),
            "latency_p50_ms": round(np.percentile(times_ms, 50), 2),
            "latency_p95_ms": round(np.percentile(times_ms, 95), 2),
            "latency_p99_ms": round(np.percentile(times_ms, 99), 2),
        }
    )


def main():
    parser = argparse.ArgumentParser(description="DP benchmark for SigLIP2 Giant")
    parser.add_argument("--model", required=True, help="Path to compiled model")
    parser.add_argument(
        "--cores", type=int, required=True, help="Number of cores to use"
    )
    parser.add_argument(
        "--warmup", type=int, default=20, help="Warmup iterations per core"
    )
    parser.add_argument(
        "--iterations", type=int, default=100, help="Benchmark iterations per core"
    )
    args = parser.parse_args()

    print("=" * 80)
    print(f"SigLIP2 Giant 256px - Data Parallel Benchmark (DP={args.cores})")
    print("=" * 80)
    print(f"  Model: {args.model}")
    print(f"  Cores: {args.cores}")
    print(f"  Warmup: {args.warmup} iterations/core")
    print(f"  Benchmark: {args.iterations} iterations/core")

    result_queue = multiprocessing.Queue()
    load_lock = multiprocessing.Lock()
    bench_barrier = multiprocessing.Barrier(args.cores, timeout=600)
    processes = []

    print(f"\nLaunching {args.cores} worker processes (staggered loading)...")
    start_time = time.perf_counter()

    for core_id in range(args.cores):
        p = multiprocessing.Process(
            target=worker,
            args=(
                core_id,
                args.model,
                args.warmup,
                args.iterations,
                load_lock,
                bench_barrier,
                result_queue,
            ),
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join(timeout=600)
        if p.is_alive():
            print(f"  WARNING: Process for core timed out, terminating", flush=True)
            p.terminate()
            p.join()

    total_time = time.perf_counter() - start_time
    print(f"\nAll workers finished in {total_time:.1f}s")

    # Collect results
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())
    results.sort(key=lambda x: x["core_id"])

    if not results:
        print("ERROR: No results collected. All workers may have failed.")
        return

    # Calculate aggregate
    total_throughput = sum(r["throughput"] for r in results)
    avg_latency = sum(r["latency_avg_ms"] for r in results) / len(results)

    print("\n" + "=" * 80)
    print(f"RESULTS - DP={args.cores}")
    print("=" * 80)
    for r in results:
        print(
            f"  Core {r['core_id']}: {r['throughput']:.2f} img/s, {r['latency_avg_ms']:.2f} ms avg"
        )
    print("-" * 80)
    print(f"  TOTAL THROUGHPUT: {total_throughput:.2f} img/s")
    print(f"  AVG LATENCY:      {avg_latency:.2f} ms")
    print("=" * 80)

    # Save
    output = {
        "model_file": args.model,
        "dp_cores": args.cores,
        "total_throughput_img_per_s": round(total_throughput, 2),
        "avg_latency_ms": round(avg_latency, 2),
        "per_core_results": results,
    }
    output_file = f"dp_benchmark_{args.cores}cores.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
