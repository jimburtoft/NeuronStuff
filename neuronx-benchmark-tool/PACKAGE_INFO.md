# NeuronX Benchmark Tool - Package Information

## Package Contents

### Core Files (Required)
- `run_benchmark_automation.py` (4.9K) - Main entry point
- `benchmark_orchestrator.py` (28K) - Orchestration logic
- `server_manager.py` (17K) - vLLM server management
- `benchmark_manager.py` (28K) - Benchmark execution
- `results_processor.py` (20K) - Results parsing and CSV generation
- `config.py` (6.9K) - Configuration settings
- `environment_utils.py` (6.8K) - Environment utilities

### Documentation
- `README.md` (3.7K) - Complete documentation
- `QUICKSTART.md` (1.5K) - Quick start guide
- `requirements.txt` (331B) - Python dependencies

### Directories
- `results/` - Benchmark results and CSV files (created automatically)
- `logs/` - System logs (created automatically)

## Total Package Size
~116 KB (code only, excluding results and logs)

## What's NOT Included

This clean package excludes:
- Test scripts (`test_*.py`)
- Example scripts (`example_*.py`)
- Monitoring utilities (`monitor_background.py`)
- Previous results and logs
- Cached models
- Development artifacts

## Basic Command

```bash
python3 run_benchmark_automation.py
```

This single command will:
1. Validate environment
2. Test all configured batch sizes
3. Generate CSV reports
4. Log all operations

## Configuration

All settings are in `config.py`:
- Batch sizes to test
- Timeout values
- Model configuration
- Environment paths

## Output

Results are automatically saved to:
- `results/consolidated_results_YYYY-MM-DD-HH-MM-SS.csv`
- `results/detailed_results_YYYY-MM-DD-HH-MM-SS.csv`
- `results/batch_X/logs/` (per-batch logs with replication commands)

## Framework

Uses **neuronx-distributed-inference** for distributed inference on AWS Neuron devices.

## Support

Check logs in `logs/` directory for troubleshooting:
- `logs/orchestrator.log` - Main orchestration
- `logs/server_processes.log` - Server management
- `logs/benchmark_processes.log` - Benchmark execution
