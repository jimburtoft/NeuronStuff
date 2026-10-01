# BERT Reranker on AWS Neuron

Deployment and benchmarking of
[`Alibaba-NLP/gte-multilingual-reranker-base`](https://huggingface.co/Alibaba-NLP/gte-multilingual-reranker-base)
(~306M parameter, 12-layer cross-encoder reranker) on AWS Inferentia2 and Trainium2.

Every number in this directory is a **hardware measurement**. Nothing is estimated,
extrapolated, or simulated.

## Notebooks

### Current — Neuron SDK 2.31

| Notebook | Platform | Peak throughput |
|---|---|---|
| `bert_reranker_triton_inf2_sdk231.ipynb` | inf2.8xlarge, Triton, 2 instances | **157.4 inf/s** |
| `bert_reranker_triton_trn2_lnc1_sdk231.ipynb` | trn2.3xlarge, Triton, LNC=1, 8 instances | **517.1 inf/s** |

`*_executed.ipynb` are the same notebooks with real output from a full
`jupyter nbconvert --execute` run (0 errors across all 24 cells on both platforms).

### Earlier — Neuron SDK 2.27/2.28

| Notebook | Platform |
|---|---|
| `bert_reranker_inference.ipynb` | inf2.8xlarge, direct inference |
| `bert_reranker_trn2_inference_executed.ipynb` | trn2.3xlarge, direct inference |
| `bert_reranker_triton_inf2_executed.ipynb` | inf2.8xlarge, Triton |
| `bert_reranker_triton_trn2_lnc1_executed.ipynb` | trn2.3xlarge, Triton, LNC=1 |

## Recommended compilation (SDK 2.31)

```python
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID, torchscript=True, trust_remote_code=True,
    attn_implementation="eager",                 # +11.4% - see below
)
torch_neuronx.trace(
    model, (input_ids, attention_mask),
    compiler_args=["--model-type", "transformer",   # +28% - keep
                   "--auto-cast", "matmult"],       # accuracy-safe
)
```

### Flag A/B results

Measured on inf2.8xlarge, SDK 2.31, single core, BS=16, seq_len=1024:

| Variant | qps | Verdict |
|---|---|---|
| Published flags, transformers 4.57.6 (SDPA attention) | 60.70 | baseline |
| **+ `attn_implementation="eager"`** | **67.62** | **+11.4% — adopt** |
| + `inline_weights_to_neff=True` | 67.59 | no-op on SDK 2.31 (already default) |
| − `--optlevel 2` | 67.52 | no-op (`-O2` is the compiler default) |
| − `--model-type transformer` | 52.70 | **keep the flag: it is worth +28%** |

Notes on two of these:

- **`attn_implementation="eager"` replaces the old `transformers==4.48.0` pin.** The pin
  worked only because 4.48 defaulted to eager attention. Confirmed by measuring 4.48.0 on
  SDK 2.31 at **67.58 qps** — statistically identical to 4.57.6 + eager (67.62). Setting the
  flag explicitly lets you track current `transformers`.
- **`inline_weights_to_neff=True` is already the default in torch-neuronx 2.15.** Passing it
  produced a **byte-identical NEFF (710.9 MB either way)**. It mattered on older SDKs, so
  keep it if you target those.

All variants held accuracy: **cosine >= 0.999998** vs a CPU FP32 reference, with passage
ranking preserved in every case.

## Triton results

Benchmark methodology on both platforms: threaded workers, 10 s per configuration,
5-request warmup, concurrency x batch-size sweep, reporting throughput and P50/P95/P99.

| Platform | Instances | Peak | Best under 100 ms P50 |
|---|---|---|---|
| inf2.8xlarge | 2 | 157.4 inf/s (BS=1, 32 workers) | **155.1 inf/s @ 50.91 ms** |
| trn2.3xlarge LNC=1 | 8 | **517.1 inf/s** (BS=4, 64 workers) | **511.1 inf/s @ 63.76 ms** |

trn2 delivers **3.3x the throughput of inf2** at comparable latency.

Accuracy on both platforms: cosine 0.999999, Spearman 1.0000 vs CPU FP32, top-3 passages
exact, padding isolation bit-exact across all batch buckets, and ranking identical across
buckets.

## Two findings worth knowing before you deploy

### 1. SDK 2.31 is slower than SDK 2.27/2.28 on this model

Single core, BS=16, seq_len=1024:

| Platform | SDK 2.27/2.28 | SDK 2.31 | Delta |
|---|---|---|---|
| inf2.8xlarge | 74.58 qps | 67.62 qps | **-9.4%** |
| trn2.3xlarge LNC=1 | 83.64 qps | 60.29 qps | **-27.9%** |
| trn2.3xlarge LNC=2 | 51.31 qps | 45.89 qps | -10.6% |

This is a genuine SDK difference, not a library artifact: pinning `transformers==4.48.0` (the
exact version the original numbers used) on SDK 2.31 still measured 67.58 qps on inf2. Both
variables are controlled.

**Under Triton most of the regression absorbs** — inf2 -2.4% and trn2 -7.1% peak throughput,
versus -9.4% and -27.9% single core. With all cores saturated by dynamic batching, the
per-graph loss is largely hidden. **Use the Triton numbers to reason about deployment impact,
not the single-core numbers.**

### 2. LNC=1 for throughput, LNC=2 for latency (trn2)

Single core, seq_len=1024, SDK 2.31:

| BS | LNC=1 | LNC=2 | ratio |
|---|---|---|---|
| 1 | 64.91 qps / 15.40 ms | **124.78 qps / 8.01 ms** | **1.92x** |
| 8 | 63.42 qps / 126.15 ms | 47.34 qps / 168.99 ms | 0.75x |
| 16 | 60.29 qps / 265.38 ms | 45.89 qps / 348.64 ms | 0.76x |

LNC=2 exposes 4 logical cores, LNC=1 exposes 8, so LNC=1 still wins **aggregate** throughput.
But two corrections to the earlier guidance:

- The original "LNC=1 is 3.3x more efficient" claim does not hold at SDK 2.31 — at BS=16 the
  advantage is **1.31x**.
- **At BS=1, LNC=2 is 1.92x faster per core and roughly halves latency (8.01 vs 15.40 ms).**
  If your reranker is latency-bound at low batch rather than throughput-bound, **compile for
  LNC=2**. Cost: ~2x NEFF size (1317 vs 676 MB at BS=1) and a DP=4 ceiling.

A model compiled for one LNC mode **cannot load** under the other. Keep the `--lnc` compiler
flag and the `NEURON_LOGICAL_NC_CONFIG` runtime variable matched, and pass the runtime
variable into containers explicitly with `-e` (it is not inherited from `/etc/environment`).

## Techniques used in the Triton notebooks

1. **Per-batch-size compilation** (BS=1,2,4,8,16) with **best-fit dispatch** — route each
   request to the smallest graph that fits, pad up, slice back.
2. **Load and warm every batch size inside `initialize()`** so no client request ever hits a
   cold graph.
3. **`preferred_batch_size` matched exactly to the compiled batch sizes**, so the dynamic
   batcher never forms a batch without a matching graph.
4. **`KIND_MODEL` with explicit core pinning** via `NEURON_RT_VISIBLE_CORES` — one instance
   per NeuronCore.
5. **Staggered instance initialization** to avoid Neuron runtime races when several instances
   load at once.
6. **Standalone backend test with a mocked `triton_python_backend_utils`** — exercises the
   full `initialize()` / `execute()` lifecycle without `tritonserver`, so backend errors
   surface in seconds instead of after a 20-minute Docker build.
7. **Correctness assertions** in that test: shape/routing per bucket, **padding isolation**
   (bit-exact), cross-graph agreement, and **ranking stability across buckets**.
8. Poll `/v2/models/{name}/ready` rather than the bare model endpoint; use `shutil.copytree`
   rather than symlinks (symlinks do not resolve across Docker volume mounts).

### A note on batch consistency

It is tempting to assert that a given (query, passage) pair scores identically regardless of
batch size. On this model that assertion **fails for a benign reason**, so the test separates
two distinct properties:

- **Padding isolation — asserted bit-exact.** Row 0's score must not change when the other
  rows of the batch change. Measured delta **exactly 0.0** at every bucket. A violation here
  is a genuine indexing or masking defect.
- **Cross-graph agreement — loose tolerance.** Each batch size is a separately compiled
  graph, and with `--auto-cast matmult` the BF16 accumulation order differs between them, so
  the same input scores 1.3041 on the BS=1 graph and 1.2829 on BS=8 (delta 2.1e-2). This is
  numerics, not a defect — proven by padding isolation returning exactly 0.0.
- **Ranking stability** is therefore the property that actually matters for a reranker, and
  it is asserted separately. Ranking was identical across all buckets.

A single combined assertion at a tight tolerance would fail on correct code.

## Requirements

- **inf2.8xlarge** (2 NeuronCores) or **trn2.3xlarge** (8 logical cores at LNC=1)
- **Deep Learning AMI Neuron (Ubuntu 24.04) 20260813** or later (Neuron SDK 2.31)
- Docker
- ~300 GB disk (Triton image plus five compiled graphs)
- **Swap is required before compiling.** The DLAMI ships with none and `neuronx-cc` can use
  40-45 GB of host RAM per graph:
  ```bash
  sudo dd if=/dev/zero of=/swapfile bs=1M count=65536
  sudo chmod 600 /swapfile
  sudo mkswap --force /swapfile    # --force is required on this kernel
  sudo swapon /swapfile
  ```

The notebooks are self-contained: each writes its own Dockerfile, Triton config, Python
backend, and benchmark client, then builds the image, compiles the models, starts the server,
benchmarks, and cleans up.

> **A Neuron core can be held by only one process at a time, and a process keeps that claim
> for its entire lifetime** — `del model` followed by `gc.collect()` does **not** release it.
> Consequently every device-touching step in these notebooks (compilation, the accuracy gate,
> the backend test) runs in a **subprocess**, so the notebook kernel itself never claims a
> core and the Triton container can start. If you restructure the notebooks and later see
> `NRT_FAILURE in nrt_init()` or "The PyTorch Neuron Runtime could not be initialized," the
> cause is usually your own kernel holding the cores, not a driver problem.

## Test environment

All measurements taken 2026-09-30 on Neuron SDK 2.31 (DLAMI `20260813`): neuronx-cc
`2.26.6360.0`, torch-neuronx `2.9.0.2.15.32035`, torch 2.9.1, transformers 4.57.6,
Triton Inference Server `r26.01` (python backend, built from source on the Neuron PyTorch
inference image).
