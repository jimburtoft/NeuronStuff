# SigLIP 384px Production - File Index

## Documentation
- **README.md** - Overview and setup instructions
- **QUICKSTART.md** - Quick start commands and examples
- **PERFORMANCE.md** - Measured performance metrics
- **INDEX.md** - This file

## Scripts

### Compilation
- **compile_model.py** - Compile SigLIP 384px for Neuron (one-time setup)
  - Creates: `siglip_384_neuron.pt` (1.5GB)
  - Time: 239 seconds (measured)
  - Flags: `--optlevel 3 --model-type transformer`

### Inference
- **inference_single.py** - Single-core inference
  - Measured: 13.38 img/s, 74.74 ms latency
  - 50 iterations with 10 warmup
  
- **inference_worker.py** - Worker for multi-core deployment
  - Continuous inference loop
  - Used by `run_dual_core.sh`
  - Requires `NEURON_RT_VISIBLE_CORES` environment variable

- **run_dual_core.sh** - Dual-core inference launcher
  - Starts two workers, one per core
  - Measured: Core 0 at 13.08 img/s, Core 1 at 13.44 img/s
  - Handles process management

### Testing
- **benchmark.py** - Comprehensive performance benchmark
  - Tests single core
  - Tests DataParallel (2 cores)
  - Compares configurations

## Model
- **siglip_384_neuron.pt** - Compiled Neuron model (1.5GB)
  - Pre-compiled and ready to use
  - Optimized for inf2.8xlarge
  - Batch size: 1, Input: [1, 3, 384, 384]

## Quick Commands

```bash
# Test single core
python3 inference_single.py

# Test dual core
./run_dual_core.sh

# Run full benchmark
python3 benchmark.py

# Recompile if needed
python3 compile_model.py
```

## Measured Performance
- Single core: 13.38 img/s
- DataParallel (batch size 2): 20.68 img/s
- Separate processes: Each core ~13.3 img/s independently
