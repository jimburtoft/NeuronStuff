# NeuronX Benchmarking Tool

Automated benchmarking system for neuronx-distributed inference across multiple batch sizes.

## Quick Start

Run the benchmarking automation with default settings:

```bash
python3 run_benchmark_automation.py
```

This will:
- Test batch sizes: 1, 2, 4, 8, 16, 32, 64, 96, 128, 256
- Run 100 requests per batch size
- Generate CSV reports with performance metrics
- Automatically find the maximum working batch size

## Requirements

- **Neuron Environment**: `/opt/aws_neuronx_venv_pytorch_2_8_nxd_inference/bin/activate`
- **Benchmark Script**: `token_benchmark_ray.py` accessible in `../upstreaming-to-vllm/llmperf/`
- **Port**: 8000 available for vLLM server
- **Python**: 3.8+

## Configuration

Edit `config.py` to customize:

```python
# Batch sizes to test
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 96, 128, 256]

# Timeouts
SERVER_STARTUP_TIMEOUT = 3600  # 60 minutes
BENCHMARK_TIMEOUT = 1800       # 30 minutes

# Model configuration
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
TENSOR_PARALLEL_SIZE = 8
MAX_MODEL_LEN = 2048
```

## Output

Results are saved in the `results/` directory:

- `consolidated_results_YYYY-MM-DD-HH-MM-SS.csv` - Summary metrics
- `detailed_results_YYYY-MM-DD-HH-MM-SS.csv` - Detailed metrics
- `batch_X/logs/` - Individual batch logs with replication commands

### CSV Columns

- `bb` - Batch size
- `max_num_seqs` - Maximum number of sequences (2 × batch size)
- `throughput (tok/sec)` - Tokens per second
- `mean inter-token-latency (s)` - Average time between tokens
- `Mean TTFT (s)` - Mean Time To First Token
- `Mean End to end latency (s)` - Complete request latency

## Advanced Usage

### Background Execution

For long-running benchmarks that survive SSH disconnections:

```bash
nohup python3 run_benchmark_automation.py > automation.log 2>&1 &
```

### Process Existing Results

To regenerate CSV from existing benchmark results:

```bash
python3 results_processor.py
```

### Replicate Individual Runs

Each batch run logs the exact commands for replication:

```bash
# View server command
head -10 results/batch_16/logs/server.log

# View benchmark command
head -15 results/batch_16/logs/benchmark.log
```

## How It Works

1. **Sequential Testing**: Tests batch sizes in order (1, 2, 4, 8, ...)
2. **Smart Retry**: When a batch size fails, finds maximum working size by testing intermediate values
3. **Automatic Cleanup**: Terminates servers and cleans up between runs
4. **Resilient**: Continues testing even if individual batch sizes fail

## Framework

This tool uses **neuronx-distributed-inference** for distributed inference on AWS Neuron devices.

## Troubleshooting

### Server Takes Long to Start

Normal behavior - Neuron compilation can take 5-60 minutes depending on batch size.

### Port Already in Use

```bash
# Check what's using port 8000
lsof -i :8000

# Kill the process if needed
kill <PID>
```

### Out of Memory

The system automatically finds the maximum working batch size. Check logs for details:

```bash
tail -f logs/orchestrator.log
```

### Stale Lock Files

If compilation fails, clean the Neuron cache:

```bash
find /var/tmp/neuron-compile-cache/ -name "*.lock" -type f -delete
```

## Files

- `run_benchmark_automation.py` - Main entry point
- `benchmark_orchestrator.py` - Orchestration logic
- `server_manager.py` - vLLM server lifecycle management
- `benchmark_manager.py` - Benchmark execution management
- `results_processor.py` - Results parsing and CSV generation
- `config.py` - Configuration settings
- `environment_utils.py` - Environment setup utilities

## Support

For issues or questions, check the logs in the `logs/` directory for detailed error information.
