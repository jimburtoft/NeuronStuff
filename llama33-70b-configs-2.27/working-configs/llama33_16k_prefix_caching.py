#!/usr/bin/env python3
"""
Llama 3.3 70B - 16K Context with Prefix Caching (Reduced Memory)
=================================================================

This configuration enables prefix caching at 16K context by reducing memory
allocation compared to the default settings.

Reference: Based on 8K prefix caching tutorial, scaled to 16K with memory optimizations
https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/tutorials/trn2-llama3.3-70b-apc-tutorial.html

Configuration Changes from 8K Prefix Caching:
- max_model_len: 8192 → 16384 (doubled context length)
- max_num_seqs: 4 → 2 (reduced batch size to save memory)
- block_size: 32 → 64 (larger blocks = fewer blocks needed)
- pa_num_blocks: 2048 → 1024 (reduced memory allocation)
- Added kv_cache_batch_size: 2 (explicit KV cache batch size)

Memory Calculation:
- 16K context / 64 block_size = 256 blocks per sequence
- 256 blocks * 2 sequences = 512 minimum blocks
- Using 1024 blocks (2x minimum for safety)

Hardware Requirements:
- Instance: trn2.48xlarge (64 NeuronCores)
- Tensor Parallelism: 64 (matches NeuronCore count)
"""

from vllm import LLM, SamplingParams

# Model configuration
model_path = "models/Llama-3.3-70B-Instruct/"
max_model_len = 16384  # 16K context
batch_size = 2  # Reduced from 4 to save memory
block_size = 64  # Increased from 32 to reduce block count
pa_num_blocks = 1024  # Reduced from 4096

print(f"Initializing Llama 3.3 70B with 16K context and prefix caching...")
print(f"  Model: {model_path}")
print(f"  Context length: {max_model_len}")
print(f"  Batch size: {batch_size}")
print(f"  Block size: {block_size}")
print(f"  PA num blocks: {pa_num_blocks}")
print(f"  Tensor parallel size: 64")
print(f"  Data type: bfloat16")
print(f"  Prefix caching: ENABLED")
print()

# Neuron-specific configuration for prefix caching with reduced memory
additional_config = {
    "override_neuron_config": {
        "is_block_kv_layout": True,
        "is_prefix_caching": True,
        "kv_cache_batch_size": batch_size,  # Explicit KV cache batch size
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
