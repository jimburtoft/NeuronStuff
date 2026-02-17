# Quick Start Guide

Deploy the Neuron DLAMI on an inf2.8xlarge.  
https://awsdocs-neuron.readthedocs-hosted.com/en/latest/setup/neuron-setup/multiframework/multi-framework-ubuntu24-neuron-dlami.html

## Setup

```bash
cd siglip_384_production
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
pip install open_clip_torch
```

## 1. Download Model

```bash
huggingface-cli download timm/ViT-SO400M-14-SigLIP-384 --local-dir timm/ViT-SO400M-14-SigLIP-384/
```

## 2. Compile Model

```bash
python3 compile_model.py
```

## 3. Test Single Core

```bash
python3 inference_single.py
```

Expected output (measured):
```
Throughput: 13.38 img/s
Latency: 74.74 ms/image
```

## 4. Test Dual Core

```bash
./run_dual_core.sh
```

Expected output (measured):
```
[Core 0] 100 inferences, 13.08 img/s
[Core 1] 100 inferences, 13.44 img/s
```

Press Ctrl+C to stop.

## 5. Run Full Benchmark

```bash
python3 benchmark.py
```

This tests both single core and DataParallel configurations.

## Production Deployment

### Option 1: Single Core (13.38 img/s)

```python
import torch
import torch_neuronx

model = torch.jit.load('siglip_384_neuron.pt')
model.eval()

# Process single image
input_tensor = torch.randn(1, 3, 384, 384)
output = model(input_tensor)
```

### Option 2: DataParallel (20.68 img/s)

```python
import torch
import torch_neuronx

model = torch.jit.load('siglip_384_neuron.pt')
model_parallel = torch_neuronx.DataParallel(model)

# Process batch of 2 images
batch = torch.randn(2, 3, 384, 384)
output = model_parallel(batch)
```

### Option 3: Separate Processes

Terminal 1:
```bash
NEURON_RT_VISIBLE_CORES=0 python3 inference_worker.py 0
```

Terminal 2:
```bash
NEURON_RT_VISIBLE_CORES=1 python3 inference_worker.py 1
```

Each core runs independently at ~13.3 img/s.

## Recompile Model (if needed)

```bash
python3 compile_model.py
```

Compilation takes 239 seconds and creates `siglip_384_neuron.pt` (1.5GB).
