# Quick Start Guide

## Prerequisites

1. **AWS Instance**: trn2.48xlarge with 64 NeuronCores
2. **Virtual Environment**: Activate the Neuron vLLM environment
   ```bash
   source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate
   ```
3. **Model**: Download Llama-3.3-70B-Instruct to `models/Llama-3.3-70B-Instruct/`

## Running the Examples

### 1. Basic Offline Inference (16K Context)

```bash
python working-configs/llama33_16k.py
```

This is the baseline configuration from AWS documentation (corrected).

### 2. Maximum Context Length (64K)

```bash
python working-configs/llama33_64k.py
```

Uses the maximum supported context length on trn2.48xlarge.

### 3. Batch Processing (64K with Batch Size 3)

```bash
python working-configs/llama33_64k_batch3.py
```

Processes 3 sequences concurrently at 64K context.

### 4. Prefix Caching (8K and 16K)

```bash
# 8K with default settings
python working-configs/llama33_8k_prefix_caching.py

# 16K with reduced memory allocation
python working-configs/llama33_16k_prefix_caching.py
```

Enables prefix caching for improved TTFT. 16K requires reduced memory settings.

Note: 16K prefix caching requires careful memory tuning (batch_size=2, 1024 blocks).

### 5. Online API Server

Start the server:
```bash
./working-configs/llama33_online_server.sh
```

In another terminal, test the API:
```bash
python working-configs/test_api_client.py
```

The server provides OpenAI-compatible endpoints at `http://localhost:8000`.

## Configuration Summary

| Configuration | Context | Batch Size | Prefix Caching | File |
|---------------|---------|------------|----------------|------|
| Baseline      | 16K     | 1          | No             | `llama33_16k.py` |
| Extended      | 32K     | 1          | No             | `llama33_32k.py` |
| Maximum       | 64K     | 1          | No             | `llama33_64k.py` |
| Batch 1       | 64K     | 1          | No             | `llama33_64k_batch1.py` |
| Batch 3       | 64K     | 3          | No             | `llama33_64k_batch3.py` |
| Prefix Cache  | 8K      | 4          | Yes            | `llama33_8k_prefix_caching.py` |
| API Server    | 64K     | 1          | No             | `llama33_online_server.sh` |

## Important Notes

### What Works
- ✅ Context lengths: 16K, 32K, 64K (without prefix caching)
- ✅ Batch sizes at 64K: 1 and 3
- ✅ Prefix caching at 8K context only

### What Doesn't Work
- ❌ 128K context - Hardware SBUF limitation
- ❌ Batch sizes 2 and 4 at 64K - NKI kernel sharding constraint
- ❌ Prefix caching at 16K+ - Device memory exhaustion (OOM)

## Compilation Times

First run will compile the model (cached for subsequent runs):
- Standard configs (16K-64K): ~4-6 minutes
- Prefix caching (8K): ~6 minutes
- Model loading: ~1-2 minutes (standard), ~35 minutes (prefix caching)

## Troubleshooting

### Out of Memory Errors
If you see "Failed to allocate dev mem" errors:
- Reduce context length
- Reduce batch size
- Disable prefix caching

### Compilation Failures
If compilation fails:
- Check that you're using the correct virtual environment
- Verify model path is correct
- Check available disk space for compilation cache

### API Server Not Responding
If the API server doesn't respond:
- Wait for "Application startup complete" message
- Check that port 8000 is not in use
- Verify model loaded successfully (check server logs)

## Getting Help

For issues or questions:
1. Check the main README.md for detailed documentation
2. Review test logs in `logs/` directory
3. Refer to AWS Neuron documentation: https://awsdocs-neuron.readthedocs-hosted.com/
