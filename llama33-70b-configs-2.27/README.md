# Llama 3.3 70B vLLM on AWS Neuron - Test Results

## Test Environment
- **Instance**: trn2.48xlarge (64 NeuronCores)
- **Model**: Llama-3.3-70B-Instruct
- **vLLM Version**: 0.13 (Neuron fork)
- **Neuron SDK**: 2.22
- **Tensor Parallelism**: 64
- **Data Type**: bfloat16

## Working Configurations

All working configurations are provided as complete, runnable Python scripts in the `working-configs/` directory:

### Context Length Scaling (No Prefix Caching)
- ✅ **16K context** - `working-configs/llama33_16k.py`
- ✅ **32K context** - `working-configs/llama33_32k.py`
- ✅ **64K context** - `working-configs/llama33_64k.py`

### Batch Size Scaling at 64K Context
- ✅ **Batch size 1** - `working-configs/llama33_64k_batch1.py`
- ✅ **Batch size 3** - `working-configs/llama33_64k_batch3.py`

### Prefix Caching
- ✅ **8K context with prefix caching** - `working-configs/llama33_8k_prefix_caching.py`
- ✅ **16K context with prefix caching (reduced memory)** - `working-configs/llama33_16k_prefix_caching.py`

Note: 16K prefix caching requires reduced memory allocation (batch_size=2, 1024 blocks, block_size=64) compared to 8K defaults.

### Online Serving
- ✅ **OpenAI-compatible API server** - `working-configs/llama33_online_server.sh`
- ✅ **API client example** - `working-configs/test_api_client.py`

## Test Results Summary

### Context Length Tests (batch_size=1, no prefix caching)

| Context Length | Status | Compilation Time | Notes |
|----------------|--------|------------------|-------|
| 16K (16384)    | ✅ SUCCESS | ~4.5 min | Baseline configuration |
| 32K (32768)    | ✅ SUCCESS | ~4.5 min | Doubles context from baseline |
| 64K (65536)    | ✅ SUCCESS | ~5 min | Maximum supported context |
| 128K (131072)  | ❌ FAILED | N/A | Hardware SBUF limitation |

### Batch Size Tests (64K context, no prefix caching)

| Batch Size | Status | Notes |
|------------|--------|-------|
| 1          | ✅ SUCCESS | Standard single-sequence processing |
| 2          | ❌ FAILED | NKI kernel sharding constraint |
| 3          | ✅ SUCCESS | Works despite batch size 2 failing |
| 4          | ❌ FAILED | NKI kernel sharding constraint |

### Prefix Caching Tests

| Context Length | Batch Size | Memory Config | Status | Notes |
|----------------|------------|---------------|--------|-------|
| 8K (8192)      | 4          | Standard (2048 blocks) | ✅ SUCCESS | Baseline from AWS tutorial |
| 16K (16384)    | 4          | Standard (4096 blocks) | ❌ FAILED | Device memory exhaustion (OOM) |
| 16K (16384)    | 2          | Reduced (1024 blocks) | ✅ SUCCESS | Works with reduced memory allocation |
| 32K (32768)    | 4          | Standard (8192 blocks) | ❌ FAILED | Device memory exhaustion (OOM) |
| 64K (65536)    | 1          | Any | ❌ FAILED | HLO serialization error |
| 64K (65536)    | 2          | Any | ❌ FAILED | HLO serialization error |

## Configuration Notes

### Key Parameters
All working configurations use these base parameters:
- `tensor_parallel_size=64` - Matches the 64 NeuronCores on trn2.48xlarge
- `dtype="bfloat16"` - Recommended precision for Llama 3.3 70B
- `block_size` - Varies by configuration (32 for prefix caching, 128 for standard)

### Parameter Changes from Documentation
The official AWS documentation examples contain errors. Our working configurations correct:
1. Removed invalid `device="neuron"` parameter
2. Removed deprecated `use_v2_block_manager=True` parameter
3. Changed `override_neuron_config={}` to proper `additional_config` structure
4. Added required `dtype="bfloat16"` specification
5. Added required `block_size` parameter

## Documentation Issues Found

### 1. Offline Serving Example (CRITICAL)
**Location**: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/models/llama3/llama_33_70b.html

**Issues**:
- Uses invalid `device="neuron"` parameter
- Uses deprecated `use_v2_block_manager=True` parameter
- Uses incorrect `override_neuron_config={}` (should be nested in `additional_config`)
- Missing `dtype="bfloat16"` specification
- Missing `block_size` parameter

## Limitations and Known Issues

### 128K Context Length
**Status**: Not supported on trn2.48xlarge

**Reason**: The NKI attention kernel uses float32 for intermediate Q·K^T computations (intentional for numerical stability). This requires 256KB of SBUF (Scratchpad Buffer) memory, but the hardware limit is 224KB per partition.

**Error**: `AssertionError: SBUF usage 256.0 KB exceeds per-partition limit of 224.0 KB`

### Prefix Caching Limitations
**Status**: 8K and 16K supported with proper memory configuration

**Working Configurations**:
- **8K**: batch_size=4, pa_num_blocks=2048, block_size=32 (default settings)
- **16K**: batch_size=2, pa_num_blocks=1024, block_size=64 (reduced memory)

**Failures**:
- **16K with default settings**: Device memory exhaustion (OOM)
  - Default pa_num_blocks=4096 is too high
  - Requires reduced allocation: 1024 blocks, batch_size=2
- **32K**: Device memory exhaustion even with reduced settings
  - Error: "Failed to allocate dev mem of size X bytes"
  - Multiple Neuron Cores run out of memory
- **64K**: HLO graph serialization failure during compilation
  - Error: "Could not serialize module proto"
  - HLO graph becomes too large/complex

**Key Finding**: Prefix caching memory requirements scale with context length. Careful tuning of `pa_num_blocks`, `batch_size`, and `block_size` is required for context lengths above 8K.

### Batch Size Constraints at 64K
**Status**: Batch sizes 2 and 4 fail, batch sizes 1 and 3 work

**Reason**: NKI kernel sharding requirements

**Error**: `AssertionError: sharded context length must be a multiple of 128, but got context length of 128 with 2 shards`

**Workaround**: Use batch size 1 or 3 for 64K context workloads.

## Files in This Repository

### Working Configurations
- `working-configs/llama33_16k.py` - 16K context baseline
- `working-configs/llama33_32k.py` - 32K context
- `working-configs/llama33_64k.py` - 64K context (maximum)
- `working-configs/llama33_64k_batch1.py` - 64K with batch size 1
- `working-configs/llama33_64k_batch3.py` - 64K with batch size 3
- `working-configs/llama33_8k_prefix_caching.py` - 8K with prefix caching
- `working-configs/llama33_online_server.sh` - OpenAI-compatible API server
- `working-configs/test_api_client.py` - API client example

### Test Logs
- `logs/test_16k.log` - 16K context test output
- `logs/test_32k.log` - 32K context test output
- `logs/test_64k.log` - 64K context test output
- `logs/test_64k_batch3.log` - Batch size 3 test output
- `logs/test_prefix_8k.log` - 8K prefix caching test output
- `logs/test_prefix_16k.log` - 16K prefix caching failure (OOM)
- `logs/test_prefix_32k.log` - 32K prefix caching failure (OOM)
- `logs/test_128k.log` - 128K context failure (SBUF limit)

## Recommendations

### For Production Use
1. **Use 64K context without prefix caching** for maximum context length
2. **Use batch size 1 or 3** at 64K context (avoid 2 and 4)
3. **Always specify `dtype="bfloat16"`** for best quality
4. **Set appropriate `block_size`**: 128 for standard configs, 32 for prefix caching
5. **Avoid prefix caching** unless 8K context is sufficient for your use case

### For AWS Documentation
1. Fix offline serving example with correct vLLM API parameters
2. Add complete online serving example with vLLM server command
3. Document maximum context length (64K) and 128K limitation
4. Document prefix caching limitations (8K only, OOM at higher contexts)
5. Document batch size constraints at 64K context
6. Add performance expectations for compilation and model loading times

## References

- **Official Documentation**: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/models/llama3/llama_33_70b.html
- **Prefix Caching Tutorial**: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/tutorials/trn2-llama3.3-70b-apc-tutorial.html
- **vLLM Neuron GitHub**: https://github.com/vllm-project/vllm-neuron
- **NxD Inference GitHub**: https://github.com/aws-neuron/neuronx-distributed-inference
