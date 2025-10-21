# Quick Start Guide

## Installation

No installation required! Just ensure you have:

1. Access to the Neuron environment at `/opt/aws_neuronx_venv_pytorch_2_8_nxd_inference/`
2. The `token_benchmark_ray.py` script in `../upstreaming-to-vllm/llmperf/`
3. Python 3.8 or higher

## Run Your First Benchmark

```bash
python3 run_benchmark_automation.py
```

That's it! The tool will:
- Test all batch sizes automatically
- Generate CSV reports in `results/`
- Log everything to `logs/`

## Expected Runtime

- **Small batch sizes (1-16)**: ~10-15 minutes each
- **Large batch sizes (32-256)**: ~15-60 minutes each
- **Total runtime**: 2-8 hours for all batch sizes

## Monitor Progress

While running, check progress in another terminal:

```bash
# Watch the main log
tail -f logs/orchestrator.log

# Check current batch
ls -lt results/
```

## View Results

After completion, find your results:

```bash
# View the latest CSV
ls -t results/*.csv | head -1 | xargs cat

# Or open in your preferred tool
```

## Common First-Time Issues

### "Virtual environment not found"
- Verify the path in `config.py` matches your environment location

### "Benchmark script not found"
- Ensure `token_benchmark_ray.py` is in `../upstreaming-to-vllm/llmperf/`
- Or update `LLMPERF_SCRIPT` path in `config.py`

### "Port 8000 already in use"
```bash
lsof -i :8000
kill <PID>
```

## Next Steps

- Customize batch sizes in `config.py`
- Run in background with `nohup` for long sessions
- Check the full README.md for advanced options
