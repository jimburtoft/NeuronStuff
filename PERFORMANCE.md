# Performance Results

## Hardware
- Instance: AWS inf2.8xlarge
- NeuronCores: 2
- Compiler: neuronx-cc

## Model
- Name: SigLIP-SO400M-14-384
- Input: 384x384 RGB images
- Output: 1152-dimensional embeddings
- Compiled size: 1.5GB

## Measured Performance

### Single Core (Tested)
- Throughput: 13.38 img/s
- Latency: 74.74 ms/image
- Batch size: 1
- Test: 50 iterations after 10 warmup

### Dual Core - DataParallel (Tested)
- Throughput: 20.68 img/s
- Batch latency: 96.73 ms
- Per-image latency: 48.37 ms
- Batch size: 2
- Speedup vs single core: 1.55x
- Test: 50 iterations after 10 warmup

### Dual Core - Separate Processes (Tested)
- Core 0: 13.08 img/s (76.43 ms latency)
- Core 1: 13.44 img/s (74.40 ms latency)
- Test: 50 iterations per core after 10 warmup
- Note: Each core maintains independent performance

## Compilation Settings (Tested)
```
Batch size: 1
Compiler flags: --optlevel 3 --model-type transformer
Compile time: 239 seconds
```

## Optimization Impact (Measured)

Tested configurations for batch size 1:
- optlevel 1: 12.76 img/s
- optlevel 3: 13.27 img/s (+4.0% vs optlevel 1)
- optlevel 3 + transformer: 13.38 img/s (+4.9% vs optlevel 1)

The `--model-type transformer` flag provides approximately 0.8% additional improvement over optlevel 3 alone.

## Usage Recommendations

**Single core inference:**
```python
model = torch.jit.load('siglip_384_neuron.pt')
# Measured: 13.38 img/s
```

**DataParallel (single process, 2 cores):**
```python
model_parallel = torch_neuronx.DataParallel(model)
# Measured: 20.68 img/s with batch size 2
```

**Separate processes (2 cores):**
```bash
./run_dual_core.sh
# Measured: Each core runs at ~13.3 img/s independently
```
