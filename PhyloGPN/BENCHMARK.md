# Benchmarks: songlab/PhyloGPN

## Environment

- Validated 2026-11-15 with `bash run.sh` from a fresh clone of this directory
- AMI: `Deep Learning AMI Neuron torch-neuronx 2.14.0 (Ubuntu 24.04) SDK 2.33.0 20261020` (ap-southeast-4)
- Venv `/opt/aws_neuronx_venv_torch_neuronx_2_14_0`: torch 2.14.0, torch-neuronx 2.14.3.0.1+b1b6c66, neuronx-cc
  2.28.7631.0+8f7b0cf5, nki 0.7.0 (nkilib bundled); driver 2.31.7.0, runtime 2.35.30.0, tools 2.33.38.0
- Instance: trn2.3xlarge (1 Trainium2 device), LNC=2, 4 processes (one per logical core)
- Weights: `songlab/PhyloGPN` revision `a763739cfaa653fc8305ec908a0b24d6f03c7f05`, bf16 on the device

## `run.sh bench` (`results/results_reference.json`)

Each of the 4 processes scores its own 30 M-site synthetic genome end to end with `Scorer.score`: bytes → token ids →
windows of 16,384 positions (stride 15,904) → device (B=4 windows per call, 3 calls in flight) → per-site float32
logits stitched on the host. The processes start together after compiling. Throughput = total sites / slowest
process's wall time.

| path | sites/s per device | per core | wall (30 M sites per core) |
|---|---|---|---|
| **NKI kernel** | **1,727,464** | 431.9-432.3 k | 69.5 s |
| stock TorchNeuron graph (`kernel=False`) | 980,857 | 245.2-245.3 k | 122.3 s |

Accuracy (core 0, first 3 windows, vs `phylogpn_neuron.load()` in fp32 on CPU): cosine similarity 0.999853, top allele
agreement 99.904 %. `tests/test_cpu.py` checks that the fp32 CPU model matches the Hugging Face model
(`trust_remote_code`, same revision) to 1e-4 on random 1,500-position inputs, including the embeddings.

## Window size and batch

From development runs on the same software, NKI kernel, 4 processes:

| window L | batch per call | sites/s per device |
|---|---|---|
| 4,096 | 8 | 1.49 M |
| 8,192 | 4 | 1.76 M |
| 16,384 | 2 | 1.82 M |
| 16,384 | 4 | 1.83 M (short run), 1.73-1.77 M sustained over 70 s |

The batch per call matters little (1-2 %). Throughput drops about 3 % over a sustained run as the device warms up.

## Latency (one core, batch 1)

`Scorer.__call__` on one window, token ids from the host and logits back to the host, median of 30 calls:

| window L | latency |
|---|---|
| 4,096 | 11.3 ms |
| 16,384 | 33.2 ms |

## Notes

- Each (window, batch) shape compiles once; later runs hit the compile cache.
- The four logical cores share power and thermal limits: one core alone runs about 25 % faster than each of four
  together. Measure with all four running.
- CPU fallback ops: none.
