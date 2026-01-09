# vLLM Neuron Configuration Defaults and Options

This document provides a comprehensive overview of all Neuron-specific configuration options available in vLLM when running on AWS Inferentia and Trainium instances. The configurations are organized by their source and impact on the system.

This is meant to be a reference only. Most of these defaults are there for a reason. **Change them at your peril**.

## Environment Information
- **Virtual Environment**: `/opt/aws_neuronx_venv_pytorch_inference_vllm/bin/activate`

## Primary vLLM Configuration Options

These are the main vLLM CLI options that users typically configure. These options directly influence the underlying NxD Inference (NxDI) configuration parameters.

### Core Model Parameters

| vLLM CLI Option | Default | Description | Maps to NxDI Parameter |
|-----------------|---------|-------------|------------------------|
| `--model` | Required | Model name or path | Used for model loading |
| `--tensor-parallel-size` | `1` | Number of tensor parallel processes | `tp_degree` |
| `--max-num-seqs` | `32` (Neuron default) | Maximum concurrent sequences | `batch_size` (when chunked prefill disabled) |
| `--max-model-len` | Model dependent | Maximum sequence length | `seq_len` and `max_context_length` |
| `--dtype` | `auto` | Model precision | `torch_dtype` |

### Memory and Cache Configuration

| vLLM CLI Option | Default | Description | Maps to NxDI Parameter |
|-----------------|---------|-------------|------------------------|
| `--block-size` | `max_model_len` (when prefix caching disabled) | KV cache block size | `pa_block_size` |
| `--num-gpu-blocks-override` | Auto-calculated | Override number of KV cache blocks | `pa_num_blocks` |
| `--enable-prefix-caching` | `False` | Enable prefix caching for TTFT optimization | `is_prefix_caching` |
| `--enable-chunked-prefill` | `False` | Enable chunked prefill processing | Forces `batch_size=1`, `is_block_kv_layout=True` |

### Advanced Configuration

| vLLM CLI Option | Default | Description | Maps to NxDI Parameter |
|-----------------|---------|-------------|------------------------|
| `--max-num-batched-tokens` | `131072` (Neuron default) | Token budget for batching | `max_context_length` (when chunked prefill enabled) |
| `--speculative-model` | `None` | Enable speculative decoding | `enable_fused_speculation=True` |
| `--num-speculative-tokens` | `0` | Number of speculative tokens | `speculation_length` |

### Fixed vLLM Defaults (Not Configurable via CLI)

| vLLM Setting | Value | Description | Maps to NxDI Parameter |
|--------------|-------|-------------|------------------------|
| Context batch size | `1` | Context encoding batch size | `ctx_batch_size` |
| Bucketing | `True` | Dynamic sequence bucketing | `enable_bucketing` |
| Continuous batching | Auto (when `max_num_seqs > 1`) | Continuous batching mode | `is_continuous_batching` |
| Quantization | `False` | Quantization disabled by default | `quantized` |
| Padding side | `"right"` | Sequence padding direction | `padding_side` |

### Configuration Examples

#### Basic Inference Server
```bash
python -m vllm.entrypoints.openai.api_server \
  --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
  --tensor-parallel-size 2 \
  --max-model-len 2048 \
  --max-num-seqs 16 \
  --port 8000
```

#### With Prefix Caching
```bash
python -m vllm.entrypoints.openai.api_server \
  --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
  --tensor-parallel-size 2 \
  --max-model-len 2048 \
  --max-num-seqs 16 \
  --enable-prefix-caching \
  --block-size 32 \
  --port 8000
```

#### With Chunked Prefill
```bash
python -m vllm.entrypoints.openai.api_server \
  --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
  --tensor-parallel-size 2 \
  --max-model-len 2048 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 8192 \
  --block-size 32 \
  --port 8000
```

## Advanced NxDI Configuration Override

For advanced users who need to override underlying NxD Inference parameters not exposed through vLLM CLI options, use the `--additional-config` parameter with `override_neuron_config`:

```bash
--additional-config '{
  "override_neuron_config": {
    "enable_prefix_caching": true,
    "logical_nc_config": 2,
    "skip_warmup": false
  }
}'
```

### Available NxDI Override Options

You can pass parameters through vLLM into the Neuron NxDI library.  However, it is possible that not all of these have been tested with vLLM.
The following options may be overridden using `--additional-config '{"override_neuron_config": {...}}'`:

#### Performance Optimization
```python
{
  "enable_bucketing": True,                    # Dynamic sequence bucketing
  "context_encoding_buckets": [512, 1024, 2048],  # Context encoding bucket sizes
  "token_generation_buckets": [512, 1024, 2048],  # Token generation bucket sizes
  "skip_warmup": False,                        # Skip model warmup
  "enable_prefix_caching": False,              # Prefix caching for TTFT optimization
  "logical_nc_config": 2,                      # Logical NeuronCore configuration
}
```

#### Memory Management
```python
{
  "max_context_length": 2048,                 # Override maximum context length
  "pa_num_blocks": 4096,                      # Override number of paged attention blocks
  "pa_block_size": 16,                        # Override block size
  "is_block_kv_layout": False,                # Enable block KV cache layout
  "kv_cache_batch_size": 32,                  # KV cache batch size
  "kv_cache_padding_size": 0,                 # KV cache padding for data parallel
}
```

#### Quantization
```python
{
  "quantized": False,                         # Enable quantization
  "quantized_checkpoints_path": None,         # Path to quantized checkpoints
  "quantization_type": "per_tensor_symmetric", # Quantization method
  "quantization_dtype": "int8",               # Quantization data type
  "quantization_block_size": None,            # Block size for quantization
  "quantization_scale_dtype": "f32",          # Scale data type
}
```

#### Speculative Decoding
```python
{
  "enable_fused_speculation": False,          # Enable fused speculation
  "speculation_length": 0,                   # Number of speculative tokens
  "enable_eagle_speculation": False,         # Enable Eagle speculation
  "spec_batch_size": 32,                     # Speculation batch size
}
```

#### Attention and Kernels
```python
{
  "flash_decoding_enabled": False,           # Flash decoding
  "attn_kernel_enabled": None,               # Attention kernel optimization
  "qkv_kernel_enabled": False,               # QKV kernel optimization
  "mlp_kernel_enabled": False,               # MLP kernel optimization
  "attn_tkg_nki_kernel_enabled": False,     # Token generation attention kernel
  "attn_tkg_builtin_kernel_enabled": False, # Built-in TKG attention kernel
}
```

#### Distributed Configuration
```python
{
  "tp_degree": 1,                            # Tensor parallelism degree
  "cp_degree": 1,                            # Context parallelism degree
  "pp_degree": 1,                            # Pipeline parallelism degree
  "ep_degree": 1,                            # Expert parallelism degree (MoE)
  "attention_dp_degree": 1,                  # Attention data parallelism degree
  "sequence_parallel_enabled": False,        # Sequence parallelism
}
```

#### Chunked Prefill Configuration
```python
{
  "chunked_prefill_config": {
    "enabled": False,                        # Enable chunked prefill
    "max_num_seqs": 0,                      # Actual batch size for chunked prefill
    "tkg_model_enabled": True,              # Separate TKG model for decode-only
    "kernel_q_tile_size": 128,              # Query tile size for kernel
    "kernel_kv_tile_size": 1024             # KV tile size for kernel
  }
}
```

#### Multi-Modal Configuration (Image-to-Text Models)
```python
{
  "text_neuron_config": {
    # Text model specific overrides
  },
  "vision_neuron_config": {
    # Vision model specific overrides
    "image_size": 224,
    "batch_size": 4
  }
}
```

## Environment Variables

### Core Environment Variables
```bash
# Model compilation and caching
NEURON_COMPILED_ARTIFACTS="/path/to/artifacts"  # Store/load compiled models

# Scheduler behavior
DISABLE_NEURON_CUSTOM_SCHEDULER="0"             # Use vLLM native scheduler when "1"

# Sampling behavior
NEURON_ON_DEVICE_SAMPLING_DISABLED="0"          # Disable on-device sampling when "1"

# Device visibility
NEURON_RT_VISIBLE_CORES="0-1"                   # Control visible Neuron cores

# LoRA serving
VLLM_ALLOW_RUNTIME_LORA_UPDATING="0"            # Allow runtime LoRA updates when "1"

# Logical NeuronCore configuration
NEURON_LOGICAL_NC_CONFIG="2"                    # Override logical NC config
```

### Performance Tuning Variables
```bash
# Long context optimization
LONG_CONTEXT_SCRATCHPAD_PAGE_SIZE="1024"       # Scratchpad page size for long context

# Compiler optimization
CC_PIPELINE_TILING_FACTOR="2"                  # Compiler tiling factor
SEQ_LEN_THRESHOLD_FOR_CC_TILING="16384"        # Sequence length threshold for tiling
```

## Configuration Validation and Constraints

### Automatic Validations
1. **Block Layout Requirements**: `is_block_kv_layout` must be `True` for chunked prefill and prefix caching
2. **Batch Size Constraints**: Batch size must be 1 for chunked prefill CTE model
3. **Memory Requirements**: Minimum blocks calculated based on `max_model_len`, `block_size`, and `max_num_seqs`
4. **Parallelism Constraints**: TP degree must be divisible by CP degree and attention DP degree
5. **Quantization Validation**: Quantized checkpoints path required when quantization enabled

### Performance Considerations
1. **Bucketing**: Enabled by default for optimal memory usage and performance
2. **On-Device Sampling**: Enabled by default for better latency
3. **Continuous Batching**: Automatically enabled when batch size > 1
4. **Preemption**: Disabled on Neuron for improved performance
5. **Memory Layout**: Contiguous memory layout used instead of PagedAttention

## Best Practices

### Production Deployment
1. **Pre-compile Models**: Set `NEURON_COMPILED_ARTIFACTS` to avoid startup delays
2. **Configure Tensor Parallelism**: Match `tensor_parallel_size` to available Neuron Cores
3. **Enable Prefix Caching**: For workloads with common prompt prefixes
4. **Use Bucketing**: Keep `enable_bucketing=True` for dynamic workloads
5. **Monitor Memory**: Adjust `pa_num_blocks` based on memory requirements

## Appendix: Neuron Platform Defaults

The following section contains the underlying NxD Inference configuration details that are automatically generated from vLLM settings. Most users should focus on the primary vLLM options above.
### NxD Inference Default Configuration

The following represents the default configuration automatically generated from vLLM settings by `_get_default_neuron_config()`:

#### Core Model Parameters
```python
{
  "tp_degree": parallel_config.tensor_parallel_size,     # From --tensor-parallel-size
  "ctx_batch_size": 1,                                   # Fixed context encoding batch size
  "batch_size": scheduler_config.max_num_seqs,           # From --max-num-seqs (or 1 if chunked prefill)
  "max_context_length": scheduler_config.max_model_len,  # From --max-model-len (or max_num_batched_tokens if chunked prefill)
  "seq_len": scheduler_config.max_model_len,             # From --max-model-len
  "enable_bucketing": True,                              # Fixed: dynamic bucketing enabled
  "is_continuous_batching": (batch_size > 1),           # Auto: enabled when batch_size > 1
  "quantized": False,                                    # Fixed: quantization disabled by default
  "torch_dtype": TORCH_DTYPE_TO_NEURON_AMP[model_config.dtype],  # From --dtype
  "padding_side": "right",                               # Fixed: right padding
}
```

#### Memory and Cache Configuration
```python
{
  "pa_num_blocks": ceil(max_model_len/block_size) * max_num_seqs,  # Auto-calculated or from --num-gpu-blocks-override
  "pa_block_size": cache_config.block_size,                       # From --block-size
  "is_block_kv_layout": (chunked_prefill_enabled or enable_prefix_caching),  # Auto: enabled for chunked prefill or prefix caching
  "is_prefix_caching": cache_config.enable_prefix_caching,        # From --enable-prefix-caching
}
```

#### Sampling Configuration
```python
{
  "on_device_sampling_config": {
    "dynamic": True,                 # Fixed: dynamic sampling enabled
    "deterministic": False,          # Fixed: non-deterministic sampling
    "do_sample": False,              # Fixed: greedy decoding by default
    "top_k": 1,                      # Fixed: top-k sampling
    "top_p": 1.0,                    # Fixed: top-p sampling
    "temperature": 1.0,              # Fixed: sampling temperature
    "global_topk": 256,              # Fixed: global top-k limit
    "on_device_sampling_config": True,
    "top_k_kernel_enabled": False
  }
}
```

#### LoRA Configuration
```python
{
  "lora_config": lora_serving_config,  # From vLLM LoRA configuration
}
```

#### Speculative Decoding Configuration (when enabled)
```python
{
  "enable_fused_speculation": True,                    # From --speculative-model
  "speculation_length": speculative_config.num_speculative_tokens,  # From --num-speculative-tokens
  "enable_eagle_speculation": (method == "eagle"),    # Auto: enabled for Eagle method
}
```