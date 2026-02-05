#!/usr/bin/env python3
"""
Llama 3.3 70B - 16K Context Configuration
==========================================

This is the baseline configuration from the AWS Neuron documentation,
corrected to use valid vLLM API parameters.

Reference: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/models/llama3/llama_33_70b.html

Configuration Changes from Documentation:
- Removed invalid `device="neuron"` parameter
- Removed deprecated `use_v2_block_manager=True` parameter
- Changed `override_neuron_config={}` to proper `additional_config` structure
- Added `dtype="bfloat16"` (required for best quality)
- Added `block_size=128` (required parameter)

Hardware Requirements:
- Instance: trn2.48xlarge (64 NeuronCores)
- Tensor Parallelism: 64 (matches NeuronCore count)
"""

from vllm import LLM, SamplingParams

# Model configuration
model_path = "models/Llama-3.3-70B-Instruct/"
max_model_len = 16384  # 16K context
batch_size = 1
block_size = 128

print(f"Initializing Llama 3.3 70B with 16K context...")
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

# Example inference
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
