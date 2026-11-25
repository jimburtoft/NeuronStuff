# RT-DETR R101 on AWS Trainium2 and Inferentia2 (Neuron SDK 2.33, PyTorch Native)

Real-time object detection with [`PekingU/rtdetr_r101vd_coco_o365`](https://huggingface.co/PekingU/rtdetr_r101vd_coco_o365)
(76M parameters, 80 COCO classes, FP32) on the public **Neuron SDK 2.33 DLAMI** with PyTorch Native
(`torch.compile(backend="neuron")`). No custom kernel is needed.

| Notebook | Instance |
|---|---|
| [`rtdetr_r101_trn2_executed.ipynb`](./rtdetr_r101_trn2_executed.ipynb) | trn2.3xlarge |
| [`rtdetr_r101_inf2_executed.ipynb`](./rtdetr_r101_inf2_executed.ipynb) | inf2.8xlarge |

## Results (FP32, measured in the executed notebooks)

Accuracy against the CPU FP32 reference (20 inputs): logits cosine 0.99998, boxes cosine 0.9994 (both instances).
The detection demo finds the same 5 objects as CPU.

**trn2.3xlarge** (one process per NeuronCore, all running together)

| Configuration | img/s per instance |
|---|---|
| **8 cores (LNC=1) x batch 1** | **136.6** |
| 4 cores (LNC=2) x batch 2 | 106.1 |
| 4 cores (LNC=2) x batch 1 | 75.4 |
| 4 cores (LNC=2) x batch 1, NKI kernel instead of `gather` | 48.9 |

Single-core latency at batch 1: 52.3 ms (`gather`), 81.0 ms (NKI kernel), 1997 ms (stock `F.grid_sample`).

**inf2.8xlarge** (2 NeuronCores)

| Configuration | img/s per instance |
|---|---|
| **2 cores x batch 1** | **36.9** |
| 2 cores x batch 1, NKI kernel instead of `gather` | 33.9 |
| 2 cores x batch 2 | 20.6 |

Single-core latency at batch 1: 51.6 ms (`gather`), 56.3 ms (NKI kernel). The stock `F.grid_sample` does not
compile on Inferentia2 (`NCC_EBVF030`, instruction limit), so a replacement is required there.

For reference, the SDK 2.28 `trace()` + NKI notebooks reached 52.8 img/s (trn2.3xlarge) and 36.6 img/s
(inf2.8xlarge).

## How it works

RT-DETR's deformable attention calls `F.grid_sample` (bilinear, zero padding). On SDK 2.33 it compiles on Neuron,
but slowly (about 2 s per image). Replacing it with a short `torch.gather` bilinear rewrite
(`grid_sample_variants.py`, written by the notebook) lets the whole model compile as one graph, with outputs
that match the CPU FP32 reference (logits cosine 0.99998).

## What changed from the SDK 2.28 version

The earlier notebooks used `torch_neuronx.trace()` with a custom NKI `grid_sample` kernel (indirect-DMA gather).

- `torch_neuronx.trace()` is no longer in the SDK 2.33 DLAMI. The notebooks now use PyTorch Native.
- With the new compiler, the plain-PyTorch `gather` rewrite is **faster than the NKI kernel**, so the kernel is
  no longer needed. The notebooks keep it as a side-by-side comparison, ported to NKI 0.7.0:
  - `nki_op` + `wrap_nki` replace the two `trace()` monkey-patches.
  - The output is `nl.shared_hbm`.
  - The weighted sum uses `tensor_scalar` / `scalar_tensor_tensor` (the old `activation(copy, bias=tile)` form no
    longer compiles for Inferentia2).

## Batch size >= 2 on SDK 2.33

A whole-model compile at batch >= 2 crosses a compiler partitioning threshold and can fail with
`bir_linker.cpp:1212 ... Mismatched number of BIR modules and Netlist module-defs`. This is fixed in a future
Neuron SDK release. Until then, set:

```bash
export NEURON_CC_FLAGS="--internal-hlo2tensorizer-options=--modular-flow-mac-target=1000000000000000"
```

Batch 1 does not need it.
