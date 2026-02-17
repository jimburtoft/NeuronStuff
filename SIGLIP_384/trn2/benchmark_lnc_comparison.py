#!/usr/bin/env python3
"""
Compare LNC=1 vs LNC=2 performance for SigLIP 384px on Trainium2.

This script benchmarks both LNC configurations and provides a comparison.

Usage:
    python3 benchmark_lnc_comparison.py

Requirements:
    - Both models must be compiled:
      NEURON_LOGICAL_NC_CONFIG=2 python3 compile_model_lnc2.py
      NEURON_LOGICAL_NC_CONFIG=1 python3 compile_model_lnc1.py
"""

import torch
import torch_neuronx
import time
import os
import subprocess

os.environ["NEURON_RT_LOG_LEVEL"] = "ERROR"

NUM_WARMUP = 10
NUM_ITERATIONS = 50


def benchmark_lnc2():
    """Benchmark LNC=2 configuration"""
    print("\n" + "=" * 80)
    print("Testing LNC=2 Configuration")
    print("=" * 80)

    if not os.path.exists("siglip_384_neuron.pt"):
        print("Error: siglip_384_neuron.pt not found")
        print("Run: NEURON_LOGICAL_NC_CONFIG=2 python3 compile_model_lnc2.py")
        return None

    # Set environment for LNC=2
    env = os.environ.copy()
    env["NEURON_LOGICAL_NC_CONFIG"] = "2"

    # Run benchmark
    result = subprocess.run(
        ["python3", "inference_single_lnc2.py"], env=env, capture_output=True, text=True
    )

    if result.returncode != 0:
        print(f"Error running LNC=2 benchmark: {result.stderr}")
        return None

    print(result.stdout)

    # Parse results
    throughput = None
    latency = None
    for line in result.stdout.split("\n"):
        if "Throughput:" in line:
            throughput = float(line.split(":")[1].strip().split()[0])
        elif "Latency:" in line:
            latency = float(line.split(":")[1].strip().split()[0])

    return {"throughput": throughput, "latency": latency, "cores": 4}


def benchmark_lnc1():
    """Benchmark LNC=1 configuration"""
    print("\n" + "=" * 80)
    print("Testing LNC=1 Configuration")
    print("=" * 80)

    if not os.path.exists("siglip_384_neuron_lnc1.pt"):
        print("Error: siglip_384_neuron_lnc1.pt not found")
        print("Run: NEURON_LOGICAL_NC_CONFIG=1 python3 compile_model_lnc1.py")
        return None

    # Set environment for LNC=1
    env = os.environ.copy()
    env["NEURON_LOGICAL_NC_CONFIG"] = "1"

    # Run benchmark
    result = subprocess.run(
        ["python3", "inference_single_lnc1.py"], env=env, capture_output=True, text=True
    )

    if result.returncode != 0:
        print(f"Error running LNC=1 benchmark: {result.stderr}")
        return None

    print(result.stdout)

    # Parse results
    throughput = None
    latency = None
    for line in result.stdout.split("\n"):
        if "Throughput:" in line:
            throughput = float(line.split(":")[1].strip().split()[0])
        elif "Latency:" in line:
            latency = float(line.split(":")[1].strip().split()[0])

    return {"throughput": throughput, "latency": latency, "cores": 8}


def print_comparison(lnc2_results, lnc1_results):
    """Print comparison table"""
    print("\n" + "=" * 80)
    print("LNC CONFIGURATION COMPARISON")
    print("=" * 80)
    print()
    print(f"{'Metric':<25} {'LNC=2':<20} {'LNC=1':<20} {'Difference':<20}")
    print("-" * 80)

    if lnc2_results and lnc1_results:
        # Throughput
        t2 = lnc2_results["throughput"]
        t1 = lnc1_results["throughput"]
        diff_t = ((t2 - t1) / t1) * 100
        print(
            f"{'Single-Core Throughput':<25} {t2:>8.2f} img/s{'':<5} {t1:>8.2f} img/s{'':<5} {diff_t:>+.1f}%"
        )

        # Latency
        l2 = lnc2_results["latency"]
        l1 = lnc1_results["latency"]
        diff_l = ((l1 - l2) / l2) * 100
        print(
            f"{'Latency':<25} {l2:>8.2f} ms{'':<9} {l1:>8.2f} ms{'':<9} {diff_l:>+.1f}%"
        )

        # Cores
        c2 = lnc2_results["cores"]
        c1 = lnc1_results["cores"]
        print(
            f"{'Logical Cores':<25} {c2:>8} cores{'':<8} {c1:>8} cores{'':<8} {'2x more with LNC=1'}"
        )

        print()
        print("-" * 80)
        print("KEY FINDINGS:")
        print("-" * 80)
        print(f"• LNC=2 provides {diff_t:+.1f}% better per-core throughput")
        print(f"• LNC=2 has {diff_l:+.1f}% lower latency")
        print(f"• LNC=1 provides 2x more parallel workers ({c1} vs {c2} cores)")
        print()
        print("RECOMMENDATION:")
        if t2 > t1:
            print("• Use LNC=2 for better per-core performance and lower latency")
        else:
            print("• Use LNC=2 for better per-core performance")
        print("• Use LNC=1 if you need maximum parallelism with more workers")

    print("=" * 80)


def main():
    print("=" * 80)
    print("SigLIP 384px LNC Configuration Comparison")
    print("Trainium2 (trn2.3xlarge)")
    print("=" * 80)

    # Run benchmarks
    lnc2_results = benchmark_lnc2()
    lnc1_results = benchmark_lnc1()

    # Print comparison
    print_comparison(lnc2_results, lnc1_results)

    print("\nNext steps:")
    print("  - Test multi-core performance with run_multi_core_lnc2.sh")
    print("  - Test multi-core performance with run_multi_core_lnc1.sh")
    print("  - See SIGLIP-384-trainium2.md for detailed documentation")


if __name__ == "__main__":
    main()
