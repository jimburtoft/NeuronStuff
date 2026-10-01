#!/usr/bin/env python3
"""Run GPT-OSS-20B on Trainium2 with vLLM + vllm-neuron (Neuron SDK 2.32).

Verified 2026-09-30 on trn2.3xlarge, DLAMI "Deep Learning AMI Neuron (Ubuntu 24.04) 20260818",
vllm 0.24.0 / vllm-neuron 0.24.0.1.1.0.

Usage:
    source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0/bin/activate
    export NEURON_SKIP_EFA_AFFINITY=1
    python gpt_oss_run.py

See README.md for why each of the three changes below is needed.
"""
import os
import sys

from vllm import LLM, SamplingParams

MODEL = os.path.expanduser(os.environ.get("GPT_OSS_PATH", "~/models/gpt-oss-20b"))

if not os.path.isdir(MODEL):
    sys.exit(
        f"Model not found at {MODEL}\n"
        f"Download it first:\n"
        f"  hf download openai/gpt-oss-20b --local-dir {MODEL}\n"
        f"or set GPT_OSS_PATH to where it already lives."
    )

if os.environ.get("NEURON_SKIP_EFA_AFFINITY") != "1":
    # Change 1. Without this, all 4 TP workers die with a FileNotFoundError on
    # /sys/bus/pci/devices/*/infiniband, because trn2.3xlarge has no EFA device.
    sys.exit("Set NEURON_SKIP_EFA_AFFINITY=1 before running (see README.md, Change 1).")

llm = LLM(
    model=MODEL,
    # TP=4 is REQUIRED: 20B in BF16 does not fit on one LNC=2 core (24 GB).
    # TP=1 fails at load with "nrt_tensor_allocate status=4".
    tensor_parallel_size=4,
    max_model_len=4096,
    max_num_seqs=4,
    trust_remote_code=True,
    # Change 3: vLLM 0.24 enables prefix caching + chunked prefill by default, but the
    # default token budget is not a supported segmented-prefill size, so the two defaults
    # conflict. Must be one of [512, 1024, 2048, 4096, 8192] and >= your longest prompt.
    max_num_batched_tokens=2048,
    # Change 2a: hide the checkpoint's MXFP4 marker so vLLM's Neuron quantization
    # allowlist check passes. GPT-OSS ships as MXFP4, which Trainium2 cannot compute.
    hf_overrides={"quantization_config": None},
    # Change 2b: explicitly select the BF16 implementation. This triggers the host-side
    # MXFP4 -> BF16 dequantization at load. Both 2a and 2b are required; either alone fails.
    additional_config={"neuron_config": {"quantization": "bf16"}},
)

PROMPTS = [
    "Explain in two sentences why the sky appears blue.",
    "Write a Python function that reverses a linked list.",
    "What is 17 * 24? Show your reasoning briefly.",
]

# NOTE: temperature=0.0 is NOT reproducible here -- the continuous-batching scheduler
# introduces run-to-run variation. Use max_num_seqs=1 above if you need bit-identical
# output for accuracy comparisons. See README.md "Notes and gotchas".
outputs = llm.generate(PROMPTS, SamplingParams(temperature=0.0, max_tokens=128))

for o in outputs:
    print(f"\n--- {o.prompt}\n{o.outputs[0].text}")
