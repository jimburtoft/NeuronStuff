# Benchmarks: togethercomputer/StripedHyena-Hessian-7B

## Environment

- Validated 2026-10-21 with `run.sh` and `run.sh test-hw` from this directory
- AMI: `Deep Learning AMI Neuron (Ubuntu 24.04) SDK 2.33.0 20261020` (`ami-042081cf4060ec468`, ap-southeast-4)
- Venv `/opt/aws_neuronx_venv_torch_neuronx_2_14_0`: torch 2.14.0, torch-neuronx 2.14.3.0.1+b1b6c66, neuronx-cc
  2.28.7631.0+8f7b0cf5, nki 0.7.0+37713544100 (nkilib bundled); driver 2.31.7.0, runtime 2.35.30.0, collectives 2.35.43.0
- Instance: trn2.3xlarge (1 Trainium2 device), LNC=2, TP=4 unless stated
- Weights: bf16, `poles` / `residues` fp32 (as the model card requires); HF revision `59e474c`

## `run.sh bench` (this directory's harness; `results/results_reference.json`)

Decode: seeded random prompt of length context-256, compiled prefill, state replicated to B rows, 51 warm-up steps, then
7 reps x 24 timed greedy steps with the tokens copied to the host every step; median of the rep medians.

| point | ms/step (stdev) | tok/s | peak HBM / core |
|---|---|---|---|
| B=1 @ 2K (whole-step graph) | 12.79 (0.23) | 78.2 | 4.5 GB |
| B=1 @ 8K | 13.17 (0.20) | 76.0 | 4.7 GB |
| B=1 @ 32K | 13.26 (0.14) | 75.4 | 5.0 GB |
| B=512 @ 2K | 88.67 (0.06) | **5,774** | 23.0 GB |
| B=128 @ 8K (`attention_tkg`) | 64.26 (0.27) | **1,992** | 22.3 GB |
| B=30 @ 32K (`attention_tkg`) | 46.53 (0.13) | **645** | 21.1 GB |

TTFT (B=1, prompt ids on device -> first token on host; 3 warm-up + 10 runs):

| prompt | p50 | p95 | p99 |
|---|---|---|---|
| 128 | 24.9 ms | 25.5 | 25.5 |
| 2,048 | 142.3 ms | 142.7 | 142.9 |
| 8,192 | 743.0 ms | 745.1 | 745.2 |
| 32,512 | 8.49 s | 8.49 | 8.49 |

Prefill segments are the package defaults (`Generator.prefill`): none up to 4K, 2,048 up to 8K, 8,192 above under TP.
The 32K point is a 32,512-token prompt in a 32,768-slot window (the model maximum). The development harness measured
7.59 s for a 32,768-token prompt (33,024-slot buffer, same segments and kernels); the 0.9 s difference between the two
shapes has not been investigated.

CPU fallback ops in every timed loop: none. At the batch ceilings (B=128 @ 8K, B=30 @ 32K) the runtime logs
`TDRV:dmem_alloc_internal Failed to allocate DEVICE memory` lines during prefill/warm-up; they are non-fatal (the
process continues and the point completes) and come from the HBM headroom at those batch sizes.

`run.sh` also runs the CPU and simulator tests (8 passed: parity vs the unpatched upstream model on CPU, TP=2 sharding,
segmented prefill, bucketed attention, NKI cascade on the simulator).

**Reproduced from scratch (2026-10-25).** A new trn2.3xlarge with an empty compile cache, following README.md as
written: every `run.sh bench` point within -2.3% to +2.0% of the table above (B=1 12.50 / 13.40 / 13.53 ms at 2K / 8K /
32K; B=512 @ 2K 88.33 ms, B=128 @ 8K 64.40 ms, B=30 @ 32K 46.65 ms; TTFT 24.9 / 142.1 / 743.5 / 8,507 ms). `run.sh`
completed with exit status 0 and `run.sh test-hw` 5 passed.

## Full sweep (development harness, SDK 2.33, 2026-10-20; not reproduced by `run.sh`; n = processes)

| context | B (ms/step) -> tok/s |
|---|---|
| 2K | 1 step (12.32, n=3) -> 81; 8 (16.34) -> 490; 16 (17.05) -> 938; 32 (21.16) -> 1,512; 64 (27.32) -> 2,343; 128 (31.95) -> 4,006; 256 (50.82) -> 5,037; **512 (88.44, n=3) -> 5,789**; 576 (104.0) -> 5,537; 640 out of HBM |
| 8K | 1 step (13.15, n=3) -> 76; 16 (21.02) -> 761; 32 (33.76) -> 948; 64 tkg (41.92) -> 1,527; **128 tkg (63.46, n=3) -> 2,017**; 136 tkg (68.85) -> 1,975; 144 out of HBM |
| 32K | 1 step (13.42, n=3) -> 75; 8 tkg (24.42) -> 328; 16 tkg (32.28) -> 496; 24 tkg (40.97) -> 586; 28 tkg (50.13, n=3) -> 559; **30 tkg (46.92, n=3) -> 639**; 32 tkg (53.76, n=3) -> 595; 36 out of HBM |

DP=4 (4 single-core replicas): 4,289 tok/s @ (4 x 56, 2K), 811 @ (4 x 10, 8K), 151 @ (4 x 2, 32K; 512-token prefill
segments). TP=4 is faster at every context. One core: 33.3 ms/token at B=1 @ 2K with the fused cascade kernel; 969 tok/s @
B=48 @ 2K; 180 @ B=8 @ 8K.

Standard workloads (development harness, TP=4, p50 TTFT / TPOT / E2E):

| in / out @ B | TTFT ms | TPOT ms | E2E ms | output tok/s |
|---|---|---|---|---|
| 128 / 128 @ 1 | 25.0 | 12.33 | 1,591 | 81 |
| 2,048 / 512 @ 1 | 141.3 | 12.57 | 6,564 | 80 |
| 128 / 128 @ 8 | 105.2 | 16.21 | 2,166 | 493 |
| 2,048 / 512 @ 8 | 946.9 | 16.70 | 9,482 | 479 |
| 128 / 128 @ 64 | 854.9 | 23.53 | 3,843 | 2,720 |
| 2,048 / 512 @ 64 | 7,598 | 32.15 | 24,029 | 1,990 |

## Accuracy

Perplexity and multiple choice: development harness, SDK 2.33, 2026-10-20. Logit parity: `run.sh test-hw`.

| | this port | reference (upstream code, same weights and tokens) |
|---|---|---|
| WikiText-2 PPL @ 2048 / 8192 | 6.1966 / 5.5418 | 6.1945 / 5.5423 |
| ARC-Easy acc / acc_norm (n=500) | 0.762 / 0.760 | 0.760 / 0.760 |
| PIQA | 0.820 / 0.850 | 0.812 / 0.850 |
| HellaSwag | 0.538 / 0.704 | 0.538 / 0.704 |
| decode logits, 128 teacher-forced positions (1 core, TP=4 step graph, TP=4 + `attention_tkg`, cascade) | rel L2 0.006-0.009, top-1 0.98-1.00, KL <= 0.0007 | gate rel <= 0.03, top-1 >= 0.98, KL <= 0.005 |

`run.sh test-hw`: 5 passed (`results/hardware_tests.txt`: parity on one core, one core + cascade, TP=4 step graph,
TP=4 + `attention_tkg`, and batched generation with identical rows), 2026-10-21.

## Settings

- Execution mode: `torch.compile(backend="neuron", dynamic=False)`; one graph per block type, the whole decode step as
  one graph at B=1 under TP
- Kernels: nkilib `mlp` (CTE prefill rows >= 2,048 under TP; TKG with fused RMSNorm at <= 32 rows on one core), nkilib
  `attention_tkg` (8K at B >= 64, 32K), fused NKI Hyena decode cascade (one core, B=1, opt-in)
- Precision: bf16, poles/residues fp32, fp32 residual stream in prefill
- Prefill: segmented (2,048 up to 8K, 8,192 above under TP; 512 on one core at windows > 16K)

## Known differences from the earlier torch-neuronx 2.13 / SDK 2.32 stack

Development harness, same shapes on both stacks: 32K TTFT is ~1.2x slower (6.24 -> 7.59 s at TP=4) and one logical
core at a 32K window holds fewer sequences; decode throughput and B=1 latency are equal or better at every other measured
point (B=1 12.3 vs 13.2 ms; 8K 2,017 vs 1,999; 32K best 639 vs 616 tok/s).
