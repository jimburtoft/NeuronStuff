#!/usr/bin/env python3
"""Run GPT-OSS-20B on Trainium2 with vLLM + vllm-neuron (Neuron SDK 2.33).

Uses the public SDK 2.33 DLAMI ("Deep Learning AMI Neuron (Ubuntu 24.04) SDK 2.33.0 20261020")
with the two GPT-OSS branches installed (README.md, step 2):
    jimburtoft/vllm-neuron@gptoss-trn2-perf-0.29
    jimburtoft/nki-library@gptoss-trn2-perf-2.33

Usage:
    source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_29_0_1_2_0/bin/activate
    python gpt_oss_run.py

The first run compiles for about 25-50 minutes. Run it in tmux or screen.
"""
import os
import sys

from vllm import LLM, SamplingParams

MODEL = os.path.expanduser(os.environ.get("GPT_OSS_PATH", "~/models/gpt-oss-20b"))

if not os.path.isdir(MODEL):
    sys.exit(
        f"Model not found at {MODEL}\n"
        f"Download it first:\n"
        f"  hf download openai/gpt-oss-20b --local-dir {MODEL} --exclude 'original/*' --exclude 'metal/*'\n"
        f"or set GPT_OSS_PATH to where it already lives."
    )

try:
    import vllm_neuron.vllm.platform as _platform
    from nkilib.core.moe.moe_tkg import selective_expert_impl as _kernels
except ImportError as e:
    sys.exit(f"{e}\nActivate the SDK 2.33 vLLM environment first (see the usage line above).")
if not hasattr(_platform.NeuronPlatform, "_cap_memory_for_gpt_oss_fp8"):
    sys.exit(
        "The stock vllm-neuron plugin is installed, not the GPT-OSS branch. Install it with:\n"
        '  pip install --no-deps --force-reinstall "git+https://github.com/jimburtoft/vllm-neuron.git@gptoss-trn2-perf-0.29"'
    )
if not getattr(_kernels, "SUPPORTS_GATE_UP_MAJOR", False):
    print(
        "WARNING: stock NKI Library kernels. It runs, but more slowly. For the patched kernels:\n"
        '  pip install --no-deps --force-reinstall "git+https://github.com/jimburtoft/nki-library.git@gptoss-trn2-perf-2.33"\n'
        "  rm -rf ~/.cache/neuron_libtorch",
        file=sys.stderr,
    )

llm = LLM(
    model=MODEL,
    tensor_parallel_size=4,            # all 4 cores of the trn2.3xlarge device
    max_model_len=8192,
    max_num_seqs=64,
    block_size=128,
    # 2048-token prefill chunks plus smaller prefill buckets (below): fast for short and long prompts.
    max_num_batched_tokens=2048,
    enable_prefix_caching=False,
    additional_config={"neuron_config": {
        # FP8 expert weights for decode, BF16 copy for prefill. The plugin lowers
        # gpu_memory_utilization to 0.80 for this and logs a warning; that is expected.
        "quantization": "fp8",
        "num_batched_tokens_buckets": [128, 256, 512, 1024, 2048],
        "decode_context_length_buckets": [1024, 2048, 4096, 7168],
        # Greedy only. Remove this if you need temperature / top-p sampling.
        "on_device_sampling_config": {"all_greedy": True},
    }},
)

PROMPTS = [
    "Explain in two sentences why the sky appears blue.",
    "Write a Python function that reverses a linked list.",
    "What is 17 * 24? Show your reasoning briefly.",
]

# llm.chat() applies the model's chat template (as the OpenAI server does). llm.generate() on a raw
# string would treat it as text to continue, not as a question.
outputs = llm.chat(
    [[{"role": "user", "content": p}] for p in PROMPTS],
    SamplingParams(temperature=0.0, max_tokens=512),
)

# GPT-OSS answers in the "harmony" format: a reasoning ("analysis") part, then the answer after
# "assistantfinal". The OpenAI server splits these for you; here we print just the answer.
for p, o in zip(PROMPTS, outputs):
    text = o.outputs[0].text
    answer = text.split("assistantfinal", 1)[1] if "assistantfinal" in text else text
    print(f"\n--- {p}\n{answer.strip()}")
