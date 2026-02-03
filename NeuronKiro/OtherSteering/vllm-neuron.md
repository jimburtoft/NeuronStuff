# AWS Neuron vLLM Integration Guide

This steering file provides guidance for working with AWS Neuron and vLLM integration using NxD Inference.

## Overview

AWS Neuron integrates with vLLM through NxD Inference (neuronx-distributed-inference) using vLLM's Plugin System. This enables LLM inference and serving on AWS Inferentia and Trainium AI accelerators with advanced features like continuous batching.

## Basic Usage Pattern

```python
from vllm import LLM, SamplingParams

# Initialize with Neuron-specific parameters
llm = LLM(
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_num_seqs=4,
    max_model_len=128,
    tensor_parallel_size=2,  # Adjust based on Neuron Cores available
    block_size=32
)

# Generate text
sampling_params = SamplingParams(temperature=0.0)
outputs = llm.generate(prompts, sampling_params)
```

## Feature Support Matrix

### ✅ Fully Supported
- Continuous batching
- Prefix Caching
- Multi-LORA
- Speculative Decoding (Eagle V1 only)
- Quantization (INT8/FP8)
- Dynamic sampling
- Tool calling
- CPU Sampling

### 🚧 Partial Support
- Chunked Prefill (disabled by default for optimal performance)
- Multimodal (Llama4 and Pixtral supported)

## Configuration Guidelines

### Model Configuration
Use `additional_config` field with `override_neuron_config` dictionary for custom settings:

```python
# Python configuration
additional_config = {
    "override_neuron_config": {
        "enable_prefix_caching": True
    }
}
```

### CLI Configuration
```bash
--additional-config '{"override-neuron-config": {"enable_prefix_caching": true}}'
```

## Key Architecture Differences

### Memory Management
- Uses **contiguous memory layout** for K/V cache instead of PagedAttention
- Block size set to maximum model length
- One block allocated per maximum configured sequences
- **Preemption is disabled** for improved performance

### Sampling Strategy
- **On-device sampling enabled by default** for better latency
- Supported parameters: `temperature`, `top_k`, `top_p`
- Greedy decoding by default
- Configure via `on_device_sampling_config` in neuron config

## Performance Optimizations

### Pre-compiled Models
- Set `NEURON_COMPILED_ARTIFACTS` environment variable to store/load compiled models
- Compilation can take ~15 minutes for 15GB models
- Automatic fallback to recompilation if loading fails

### Prefix Caching
- Available since Neuron SDK 2.24
- Improves Time To First Token (TTFT) by reusing KV Cache
- Enable via neuron configuration

## Quantization

**Important**: Do NOT use vLLM's `--quantization neuron_quant` setting. Instead:
- Keep vLLM quantization unset
- Configure quantization directly through NxD Inference model configuration
- Supports INT8/FP8 quantization

## Tensor Parallelism

Adjust `tensor_parallel_size` based on your instance type's Neuron Cores:
- Check instance specifications for available Neuron Core count
- Configure accordingly for optimal performance

## OpenAI API Compatibility

Start an OpenAI-compatible server:
```bash
python -m vllm.entrypoints.openai.api_server \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --tensor-parallel-size 2 \
    --max-model-len 128 \
    --block-size 32
```

Additional sampling parameter `top_k` is supported beyond standard OpenAI parameters.

## Known Issues & Workarounds

1. **Chunked Prefill**: Disabled by default. Enable with `DISABLE_NEURON_CUSTOM_SCHEDULER="1"`

2. **HuggingFace Models with tie_word_embeddings**: 
   - Error: `NotImplementedError: Cannot copy out of meta tensor; no data!`
   - Solution: Download model locally instead of using HuggingFace model ID

3. **Chat Templates**: Not cached in vLLM 0.11.0, affecting preprocessing time

4. **Async Tokenization**: Can increase preprocessing time for small inputs/batches

5. **Pixtral Limitations**: 
   - Out of bounds issues for batch sizes > 4
   - Max sequence length: 10240

## Best Practices

1. **Use DLC containers** for easiest setup and best compatibility
2. **Pre-compile models** for production to avoid startup delays
3. **Configure tensor parallelism** based on available Neuron Cores
4. **Enable prefix caching** for workloads with common prompt prefixes
5. **Use on-device sampling** for better latency (default behavior)
6. **Monitor compilation times** and use artifact caching appropriately

## Environment Variables

- `NEURON_COMPILED_ARTIFACTS`: Path for storing/loading compiled models
- `DISABLE_NEURON_CUSTOM_SCHEDULER`: Set to "1" to enable chunked prefill

## Code Links


- https://github.com/aws-neuron/neuronx-distributed-inference
- https://github.com/vllm-project/vllm-neuron
