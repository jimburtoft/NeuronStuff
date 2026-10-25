# Evo 2 7B on Trainium2: benchmark

Model: [`arcinstitute/evo2_7b_base`](https://huggingface.co/arcinstitute/evo2_7b_base), bf16, 8,192-token trained context.

## Environment

- Instance: `trn2.3xlarge` (1 Trainium2 device = 4 logical NeuronCores at LNC=2)
- AMI: `Deep Learning AMI Neuron (Ubuntu 24.04) SDK 2.33.0 20261020` (`ami-042081cf4060ec468` in ap-southeast-4)
- Venv `/opt/aws_neuronx_venv_torch_neuronx_2_14_0`: torch 2.14.0, torch-neuronx 2.14.3.0.1, neuronx-cc 2.28.7631.0,
  nki 0.7.0 (NKI Library bundled); driver 2.31.7.0
- vortex pinned to `cb229ae50581aa072972c33a81e23eb43a1330ab`; all package defaults

## Batched decode throughput (`bash run.sh fill`, `bash run.sh dp4-256`)

Prompt: a fixed 128-nt DNA sequence (`BENCH_PROMPT` in `harness/bench_fill.py`). Timing: 4 (TP) / 6 (DP) warm-up
steps, then the mean of 10 (TP) / 20 (DP) steps, synchronized on both sides. TP rows: 2026-10-23, 3 runs per row
(spread <= 1.6%). DP row: 2026-10-25, 4 cores measured concurrently.

**Reproduced from scratch (2026-10-25).** A new trn2.3xlarge with an empty compile cache, following README.md as
written, gave 2,318 / 2,163 / 1,571 / 1,619 tok/s for the four TP rows (within 1%, identical 24-step token hashes)
and 3,356 tok/s for DP=4. Simulator tests: 61 passed; hardware tests: 6 passed. On that fresh cache the fill-128
arm needed 7 compile attempts and fill 4,096 needed 2 (see "Known issue" below); `run.sh` retried them automatically.

| configuration | ms/step | **tok/s per device** |
|---|---|---|
| TP=4, B=88, `max_seqlen=8192`, KV fill 128 | 37.7 | **2,333** |
| TP=4, B=88, KV fill 1,024 | 40.3 | **2,181** |
| TP=4, B=88, KV fill 4,096 | 56.1 | **1,569** |
| TP=4, B=80, KV fill 7,680 | 49.4 | **1,618** |
| DP=4 (4 processes, one per core), B=128 per core, `max_seqlen=256` | 152.6 | **3,356** |

**KV fill.** For the TP rows the prompt is prefilled, then the KV offset is set to the fill and decode is timed
there. Slots between the prompt and the fill hold zeros: the kernels, shapes and memory reads are those of a real
fill, the values are not. With a real prompt of the fill length (`FILL_MODE=prompt bash run.sh fill`), decode
measures within 0.1-1.2% of the offset method (B=88 at 1,024 and 4,096, B=80 at 7,680).

**Batch size.** B=88 is the largest TP=4 batch that fits at every fill (B=80 near full 8K fill). B=96 runs out of
device memory. Throughput at fill 1,024 by batch (SDK 2.32, same package; SDK 2.33 is within a few percent): B=16 768, B=32
1,367, B=48 1,776, B=64 2,014, B=80 2,243, B=88 2,254 tok/s. For short contexts, DP=4 at B=128 per core is best; B >= 136 runs out of memory.

**Single-request latency.** One logical core decodes at ~30 ms/token at B=1. This package targets batched throughput.

## Prefill (TP=4, B=8, warm, `evo2d.prefill(chunk_rows=8)`)

| prompt length | 64 | 256 | 512 | 1,024 | 2,048 | 4,096 | 6,144 | 8,000 |
|---|---|---|---|---|---|---|---|---|
| prefill ms | 76 | 153 | 255 | 572 | 1,173 | 2,748 | 4,042 | 5,497 |
| prompt tok/s | 6,760 | 13,370 | 16,060 | 14,320 | 13,970 | 11,920 | 12,160 | 11,640 |

Per-row prefill cost is the same at B=8 and B=64; no prompt length is an outlier. Each distinct prompt length is a
new prefill graph, so use `buckets=` (below) and precompile one graph per bucket. (Measured on torch-neuronx 2.13 /
SDK 2.32 with the same package; decode on that stack is within -1.6% to +2.5% of SDK 2.33 on the same host type.)

## Correctness

| check | result |
|---|---|
| logits vs an independent fp32 CPU implementation | `cos_sim 0.99999930`, max\|diff\| 0.6367 |
| fill-bucketed attention vs fp32 CPU, 199 teacher-forced steps | KL 0.002169 (unbucketed 0.002177), top-1 100% |
| length-bucketed prefill vs fp32 CPU, 40 teacher-forced steps at L=1,300 | KL 0.00159 (exact-length 0.00156), top-1 100% |
| TP=4 vs one core, step-1 logits | relative L2 9.7e-6 |
| rows inside one batch | bit-identical |
| across batch sizes (teacher-forced) | 1-2 bf16 ulps on the logits, max KL 1.7e-5 |
| per-kernel parity (NKI simulator) | `cos_sim 1.00000000` vs fp64 reproductions of the upstream op chains |
| CPU fallback ops per decode step | none |

The first four rows were measured on SDK 2.32 during development; the rest, and the package test suite
(67 tests, simulator + hardware), on SDK 2.33.

**Why tokens are not the gate.** Next-base prediction in this model is intrinsically low-margin: the top-1 / top-2
logit gap is 0.013-0.057 on typical prompts (widest seen 0.22), while bf16-vs-fp32 logit noise is ~0.64. Any change of
reduction order (a different implementation, a different batch size) flips some greedy tokens. The gates therefore
compare logits against an fp32 reference, compare logits across batch sizes after teacher-forcing one token stream,
and require exact equality only inside one batch.

## Known issue: intermittent first compile on SDK 2.33

At TP=4 and B >= 48, a first compile of a new graph occasionally fails with
`[NCC_INLA001] Allocated memory out of bound {scratch_sb_for_inst__I-NN}` or `[NCC_IBIR229] State buffer allocation
failed`. The same graph compiles on the next attempt, and a cached graph never fails. `run.sh` retries these two
errors (and only these); keep `TORCH_NEURONX_NEFF_CACHE_DIR` between runs, or ship a pre-populated cache.

## What the package changes

All on by default; each has an environment switch to turn it off (`evo2d.config_summary()` lists them).

- **Fused NKI cascade kernels** for the Hyena blocks (HCM k=128, HCS k=7, HCL IIR), batch folded into one launch.
- **NKI Library `attention_tkg`** for decode attention, with the K cache stored pre-transposed (removes a ~6 ms
  per-call transpose penalty at 8K) and the attention region compiled per layer, including its projections and TP
  all-reduce.
- **Fill-bucketed attention**: reads the filled part of the KV cache rounded up to a power of two (min 512).
- **Weights stored (in, out)** for every GEMV, removing compiler-inserted weight transposes.
- **Tensor parallelism** (`evo2_neuron_decode.tp.shard_model`): column-split projections / `Wqkv` / MLP in, row-split
  outputs, one all-reduce each.
- **Row-chunked, length-bucketed prefill** (`evo2d.prefill`) with an exact chunked closed form for the long IIR.
- **torch 2.14 support**: Dynamo 2.14 guards on the Python value of the decode offset, which recompiled the attention
  blocks every step. The attention dispatcher runs outside Dynamo (`EVO2_ATTN_EAGER_DISPATCH`, on by default for
  torch >= 2.14).
