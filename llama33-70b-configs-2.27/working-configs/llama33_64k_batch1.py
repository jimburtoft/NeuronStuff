#!/usr/bin/env python3
"""
Llama 3.3 70B - 64K Context with Batch Size 1
==============================================

This configuration explicitly sets batch size to 1 at 64K context.

Configuration Changes from 16K Baseline:
- max_model_len: 16384 → 65536 (4x context length)
- max_num_seqs: 1 (explicit batch size)

Note: At 64K context, batch sizes 2 and 4 fail due to NKI kernel
sharding constraints. Batch sizes 1 and 3 work correctly.

Hardware Requirements:
- Instance: trn2.48xlarge (64 NeuronCores)
- Tensor Parallelism: 64 (matches NeuronCore count)
"""

from vllm import LLM, SamplingParams

# Model configuration
model_path = "models/Llama-3.3-70B-Instruct/"
max_model_len = 65536  # 64K context
batch_size = 1  # Explicit batch size 1
block_size = 128

print(f"Initializing Llama 3.3 70B with 64K context and batch size 1...")
print(f"  Model: {model_path}")
print(f"  Context length: {max_model_len}")
print(f"  Batch size: {batch_size}")
print(f"  Tensor parallel size: 64")
print(f"  Data type: bfloat16")
print()

# Initialize LLM
llm = LLM(
    model=model_path,
    max_num_seqs=batch_size,
    max_model_len=max_model_len,
    block_size=block_size,
    tensor_parallel_size=64,
    dtype="bfloat16",
)

print("Model initialized successfully!")
print()

# Example inference with single prompt
prompts = [
    "What is the capital of France?",
]

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=100,
)

print("Running inference...")
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")
    print()
