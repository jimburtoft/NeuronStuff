###Work in progress
Tested on 2.27 1/25/26, Neuron DLAMI

Based on the config from https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/models/qwen3/qwen3_moe_235b.html



```
#!/bin/bash

# Activate the virtual environment
source /opt/aws_neuronx_venv_pytorch_inference_vllm/bin/activate

# BASELINE WORKING CONFIG - 16K without prefix caching
# This is the known working configuration from previous successful compilations
VLLM_NEURON_FRAMEWORK='neuronx-distributed-inference' python -m vllm.entrypoints.openai.api_server \
  --model="/home/ubuntu/Qwen3-Coder-480B-A35B-Instruct/" \
  --tensor-parallel-size=64 \
  --max-num-seqs=1 \
  --max-model-len=16384 \
  --additional-config='{"override_neuron_config": {
    "async_mode": false,
    "attn_kernel_enabled": false,
    "batch_size": 1,
    "cc_pipeline_tiling_factor": 1,
    "context_encoding_buckets": [16384],
    "cp_degree": 1,
    "ctx_batch_size": 1,
    "enable_bucketing": true,
    "flash_decoding_enabled": false,
    "fused_qkv": false,
    "is_continuous_batching": true,
    "logical_nc_config": 2,
    "max_context_length": 16384,
    "moe_ep_degree": 1,
    "moe_tp_degree": 64,
    "on_device_sampling_config": {
      "do_sample": true,
      "temperature": 0.6,
      "top_k": 20,
      "top_p": 0.95
    },
    "qkv_cte_nki_kernel_fuse_rope": false,
    "qkv_kernel_enabled": false,
    "qkv_nki_kernel_enabled": false,
    "seq_len": 16384,
    "sequence_parallel_enabled": true,
    "token_generation_buckets": [16384],
    "torch_dtype": "bfloat16",
    "tp_degree": 64
  }}' \
  --no-enable-chunked-prefill \
  --no-enable-prefix-caching \
  --port=8000

```


# Qwen3-Coder-480B-A35B-Instruct on AWS Neuron - Configuration Guide

## Model Architecture

Key specifications for Qwen3-Coder-480B-A35B-Instruct:
- **num_key_value_heads**: 8 (causes QKV kernel incompatibility)
- **num_attention_heads**: 96
- **hidden_size**: 6144
- **moe_intermediate_size**: 2560
- **num_hidden_layers**: 62
- **num_experts**: 160
- **max_position_embeddings**: 262144

## Critical Configuration Requirements

### 1. QKV Kernels Must Be Disabled
**Reason**: The optimized QKV NKI kernels expect 4 KV heads, but this model has 8 KV heads, causing tensor shape mismatches.

**Required settings**:
```json
"qkv_nki_kernel_enabled": false,
"qkv_kernel_enabled": false,
"qkv_cte_nki_kernel_fuse_rope": false
```

### 2. Expert Parallelism Must Be Disabled
**Reason**: Selective loading algorithm is incompatible with Expert Parallelism in token generation (per AWS Neuron MoE documentation).

**Required settings**:
```json
"moe_ep_degree": 1,
"moe_tp_degree": 64
```

### 3. Single Small Buckets Required for Compilation
**Reason**: Multiple or large buckets exhaust system RAM during compilation. The Neuron compiler generates HLO graphs for each bucket configuration.

**Required settings**:
```json
"context_encoding_buckets": [2048],
"token_generation_buckets": [2048],
"max_model_len": 2048,
"batch_size": 1,
"max_num_seqs": 1
```

### 4. Tensor Parallelism Constraints
**Reason**: The MoE gate_up projection size (moe_intermediate_size × 2 = 5120) must be divisible by moe_tp_degree. Valid divisors of 5120 that fit within 64 Neuron Cores are limited.

**Working configuration**: tp_degree=64, moe_tp_degree=64


## Optimization Path

Once the model compiles and runs stably:

1. **Increase context length**: Compile additional buckets (4096, 8192) separately
2. **Increase throughput**: Gradually increase batch_size and max_num_seqs
3. **Cache artifacts**: Use `NEURON_COMPILED_ARTIFACTS` environment variable to avoid recompilation
4. **Pre-compile offline**: Generate all needed bucket sizes before production deployment

## Key Takeaways

- QKV kernel incompatibility with 8 KV heads requires disabling all QKV optimizations
- Expert Parallelism cannot be used due to selective loading limitations
- Memory-constrained compilation requires single small buckets initially
- Hardware utilization is limited by model architecture (moe_intermediate_size) constraints
