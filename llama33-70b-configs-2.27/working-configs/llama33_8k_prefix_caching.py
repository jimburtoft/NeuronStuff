#!/usr/bin/env python3
"""
Llama 3.3 70B - 8K Context with Prefix Caching
===============================================

This configuration enables prefix caching for improved Time To First Token (TTFT)
by reusing KV cache for common prompt prefixes.

Reference: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/tutorials/trn2-llama3.3-70b-apc-tutorial.html

Configuration Changes from 16K Baseline:
- max_model_len: 16384 → 8192 (reduced to 8K for prefix caching)
- max_num_seqs: 1 → 4 (increased batch size)
- block_size: 128 → 32 (smaller blocks for prefix caching)
- enable_prefix_caching: True
- num_gpu_blocks_override: 2048 (explicit block allocation)
- additional_config: Added neuron-specific prefix caching settings

IMPORTANT: Prefix caching only works at 8K context. Higher context lengths
(16K, 32K) fail with device memory exhaustion (OOM). 64K fails with HLO
serialization errors.

Hardware Requirements:
- Instance: trn2.48xlarge (64 NeuronCores)
- Tensor Parallelism: 64 (matches NeuronCore count)
"""

from vllm import LLM, SamplingParams

# Model configuration
model_path = "models/Llama-3.3-70B-Instruct/"
max_model_len = 8192  # 8K context (maximum for prefix caching)
batch_size = 4
block_size = 32  # Smaller block size for prefix caching
pa_num_blocks = 2048

print(f"Initializing Llama 3.3 70B with 8K context and prefix caching...")
print(f"  Model: {model_path}")
print(f"  Context length: {max_model_len}")
print(f"  Batch size: {batch_size}")
print(f"  Block size: {block_size}")
print(f"  PA num blocks: {pa_num_blocks}")
print(f"  Tensor parallel size: 64")
print(f"  Data type: bfloat16")
print(f"  Prefix caching: ENABLED")
print()

# Neuron-specific configuration for prefix caching
additional_config = {
    "override_neuron_config": {
        "is_block_kv_layout": True,
        "is_prefix_caching": True,
    }
}

# Initialize LLM with prefix caching
llm = LLM(
    model=model_path,
    max_num_seqs=batch_size,
    max_model_len=max_model_len,
    block_size=block_size,
    tensor_parallel_size=64,
    dtype="bfloat16",
    enable_prefix_caching=True,
    num_gpu_blocks_override=pa_num_blocks,
    additional_config=additional_config,
)

print("Model initialized successfully!")
print()

# Example inference with multiple prompts
prompts = [
    "What is the capital of France?",
    "What is 2+2?",
    "Who wrote Romeo and Juliet?",
    "What is the speed of light?",
]

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=50,
)

print("Running inference with prefix caching...")
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")
    print()
