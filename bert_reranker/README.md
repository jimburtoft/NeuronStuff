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
| `bert_reranker_triton_inf2_xlarge_sdk231.ipynb` | **inf2.xlarge**, Triton, 2 instances | **154.8-156.7 inf/s** |
| `bert_reranker_triton_inf2_sdk231.ipynb` | inf2.8xlarge, Triton, 2 instances | 157.4 inf/s |
| `bert_reranker_triton_trn2_lnc1_sdk231.ipynb` | trn2.3xlarge, Triton, LNC=1, 8 instances | **517.1 inf/s** |

`*_executed.ipynb` are the same notebooks with real output from a full
`jupyter nbconvert --execute` run (0 errors in every cell on all three platforms).

> **Use `inf2.xlarge`, not `inf2.8xlarge`.** They expose the **same accelerator** -- one
> Inferentia2 device, 2 NeuronCores, 32 GB -- and measure within 0.9% of each other, which is
> inside run-to-run noise. inf2.xlarge costs **$0.7582/hr vs $1.9679/hr**, so it delivers
> **2.57x the throughput per dollar**. The larger size only adds host vCPU and RAM, which this
> workload does not use. See [Instance sizing](#instance-sizing-use-inf2xlarge) below.

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

Measured on inf2 (SDK 2.31), single core, BS=16, seq_len=1024:

| Variant | qps | Verdict |
|---|---|---|
| Published flags, transformers 4.57.6 (SDPA attention) | 60.70 | baseline |
| **+ `attn_implementation="eager"`** | **67.62** | **+11.4% — adopt** |
| + `inline_weights_to_neff=True` | 67.59 | no-op on SDK 2.31 (already default) |
| − `--optlevel 2` | 67.52 | no-op (`-O2` is the compiler default) |
| − `--model-type transformer` | 52.70 | **keep on inf2: worth +28%.** On trn2 it is -1.2% — drop it there |

Notes on two of these:

- **`attn_implementation="eager"` replaces the old `transformers==4.48.0` pin.** The pin
  worked only because 4.48 defaulted to eager attention. Confirmed by measuring 4.48.0 on
  SDK 2.31 at **67.58 qps** — statistically identical to 4.57.6 + eager (67.62). Setting the
  flag explicitly lets you track current `transformers`.
- **`inline_weights_to_neff=True` is already the default in torch-neuronx 2.15.** Passing it
  produced a **byte-identical NEFF (710.9 MB either way)**. It mattered on older SDKs, so
  keep it if you target those.
- **`--model-type transformer` is platform-specific.** It is worth **+28% on inf2** but
  **-1.2% on trn2** (measured on both). Keep it for inf2; drop it for trn2. Do not assume a
  flag's sign transfers across platforms — A/B it.

All variants held accuracy: **cosine >= 0.999998** vs a CPU FP32 reference, with passage
ranking preserved in every case.

## Triton results

Benchmark methodology on both platforms: threaded workers, 10 s per configuration,
5-request warmup, concurrency x batch-size sweep, reporting throughput and P50/P95/P99.

| Platform | Instances | Peak | Best under 100 ms P50 |
|---|---|---|---|
| **inf2.xlarge** | 2 | **156.1 inf/s** (BS=1, 32 workers) | **152.8 inf/s @ 39.76 ms** |
| inf2.8xlarge | 2 | 157.4 inf/s (BS=1, 32 workers) | 155.1 inf/s @ 50.91 ms |
| trn2.3xlarge LNC=1 | 8 | **517.1 inf/s** (BS=4, 64 workers) | **511.1 inf/s @ 63.76 ms** |

trn2 delivers **3.3x the throughput of inf2** at comparable latency.

On inf2, **throughput peaks at BS=1 with high client concurrency**, so server-side dynamic
batching does the batching work and clients can stay simple. Note that chasing the peak is
usually the wrong call: **152.8 inf/s at 39.76 ms** versus 154.8 inf/s at 102.56 ms means the
last **1.3%** of throughput costs **2.6x the latency**. Across runs that tradeoff ranged from
2.6x to 3.8x, always for under 2% throughput, so **BS=1 with 8 workers is the recommended
operating point** -- roughly **$1.38 per million inferences**.

Accuracy on both platforms: cosine 0.999999, Spearman 1.0000 vs CPU FP32, top-3 passages
exact, padding isolation bit-exact across all batch buckets, and ranking identical across
buckets.

## Instance sizing: use `inf2.xlarge`

All earlier work in this directory used **inf2.8xlarge**. That was the wrong size for this
model, and the measurements say so plainly.

**Both instance types expose exactly the same accelerator.** `neuron-ls` on each reports
1 Inferentia2 device, **2 NeuronCores, 32 GB** of device memory. The larger sizes add host
vCPU and RAM (and, from `inf2.24xlarge` up, additional devices) -- none of which a
single-device workload like this reranker uses.

Same notebook, same compile flags, same 2-instance topology, same benchmark harness:

| Metric | inf2.8xlarge | inf2.xlarge | Delta |
|---|---|---|---|
| Peak throughput | 157.4 inf/s | **156.1 inf/s** | **-0.9%** (within noise) |
| Peak, best of 4 runs | 157.4 inf/s | 156.7 inf/s | -0.4% |
| Peak configuration | BS=1, 32 workers | BS=1, 32 workers | identical |
| Best under 100 ms P50 | 155.1 @ 50.91 ms | **152.9 @ 50.93 ms** | -1.4% |
| NeuronCores | 2 | 2 | — |
| vCPU | 32 | 4 | -87.5% |
| Host RAM | 128 GB | 16 GB | -87.5% |
| On-demand $/hr (us-east-2) | $1.9679 | **$0.7582** | **-61.5%** |
| **inf/s per $/hr** | 80.0 | **205.8** | **+157.3%** |

**Four independent full runs on two inf2.xlarge instances in two regions** (us-east-2 and
us-west-2) peaked at **156.7 / 156.5 / 156.1 / 154.8 inf/s** -- a ~1.2% spread. The -0.9% gap
versus inf2.8xlarge is inside that noise band. The final run was a **completely cold
reproduction**: fresh instance, no prebuilt artifacts, 80.5 min end to end.

### The workload is device-bound, which is why the small host costs nothing

Both host CPU and Neuron device utilization were sampled during every benchmark configuration:

| | inf2.xlarge |
|---|---|
| Neuron device utilization | **94-95% avg** (>= 96% in most configurations) |
| Host CPU utilization | **~54% avg**, 57% peak, on 4 vCPU |

Both NeuronCores run at 96-100% while the 4 vCPU host sits around half idle. The intuition
that 4 vCPU would starve the Triton Python backend is **wrong for this model** -- and the real
margin is wider than it looks, because the benchmark client was running on those same 4 vCPU.
A deployment with an off-box client has more headroom still.

### What the smaller host does cost

| | inf2.8xlarge (32 vCPU) | inf2.xlarge (4 vCPU) |
|---|---|---|
| Triton source build | 11.3 min | **26-38 min** |
| BS=16 compile, peak host RSS | — | **14.75-14.77 GB** (+1.5-1.7 GB swap) |
| Full notebook, cold start | — | **80.5 min** |

- **The Docker build is 2-3x slower, not 8x.** Git clones, downloads and single-threaded
  CMake configure do not scale with core count. Measured 26.5 min and 38.4 min on two
  instances -- the spread is network/EBS variability, not compute. A reasonable one-time cost,
  so building on inf2.xlarge is practical.
- **Compilation requires swap** -- see [Requirements](#requirements). This is the one genuine
  constraint of the 16 GB host, and the notebook handles it automatically.

### Recommendation

Use **inf2.xlarge**. Same accelerator, same throughput, same latency, same accuracy (cosine
0.999999, Spearman 1.0000), **2.57x better throughput per dollar**.

Choose a larger inf2 size only if the *host* needs the extra vCPU or RAM for co-located work
-- client-side tokenization, retrieval, business logic -- or if you need more than one
Inferentia2 device. Do not size up for the model itself.

## Compile once, deploy to many

The two expensive steps in the Triton notebooks -- compiling the five bucket graphs
(~20-25 min) and building Triton from source (~25-40 min) -- are **pure host-side work**. A
compiled graph depends on the *target device*, not on the host that produced it, so you can do
both once and reuse the artifacts across a fleet.

This also gives you **deploy-only instances that never run the compiler**, and therefore need
no swap.

### What transfers

| Artifact | Size | Must match on the consumer |
|---|---|---|
| `model_bs{1,2,4,8,16}.pt` | ~3.3 GB | Neuron SDK version, instance family, `seq_len`, compiler flags |
| `triton-neuron-bert-reranker:sdk231` image | ~6 GB gzipped | nothing device-specific |

**Hard constraints** -- violate one and the graph either fails at `nrt_load` or misbehaves:

- **Same Neuron SDK.** Build and serve on the same DLAMI (`20260813`, SDK 2.31).
- **Same instance family.** inf2.xlarge <-> inf2.8xlarge is fine (identical Inferentia2
  device). **inf2 -> trn2 is not** -- recompile, and note trn2 also bakes in the LNC mode.
- **Same `seq_len` and bucket list.** Both are baked into the graphs, and `config.pbtxt` is
  generated from the bucket list.

**Verified for this model:** a mixed-provenance model repository -- BS=16 compiled locally,
BS=1/2/4/8 compiled on a *different* instance and pulled from S3 -- served correctly and passed
the accuracy gate (cosine 0.999999), the bit-exact padding-isolation check, and ranking
stability across every bucket.

### Producer -- run once

Complete Step 1 (compile) and Step 4 (image build) in the notebook, then:

```bash
BUCKET=s3://YOUR_BUCKET/bert-reranker

# five compiled graphs (~3.3 GB)
tar -C ~/triton_repo/bert_reranker/1 -czf /tmp/neffs.tar.gz .
aws s3 cp /tmp/neffs.tar.gz $BUCKET/neffs.tar.gz

# the Triton image (~6 GB gzipped)
docker save triton-neuron-bert-reranker:sdk231 | gzip > /tmp/triton-img.tar.gz
aws s3 cp /tmp/triton-img.tar.gz $BUCKET/triton-img.tar.gz
```

Record which SDK you built on; consumers must match it.

### Consumer -- on each additional instance

In `bert_reranker_triton_inf2_xlarge_sdk231.ipynb`:

1. Run **Step 0** (environment check). Swap is not needed if you will not compile.
2. Run the **shared-configuration cell** (just above Step 1). **Required** -- Step 1-alt reads
   `MODEL_DIR`, `BATCH_SIZES` and `DOCKER_IMAGE` from it.
3. **Skip the compile cell.**
4. In **Step 1-alt**, set `USE_S3_ARTIFACTS = True` and `S3_PREFIX`, then run it. It downloads
   the graphs, verifies every expected bucket is present, and `docker load`s the image.
5. Continue from the accuracy gate onward. Step 4 detects the loaded image and skips the build.

Needs an instance role with `s3:GetObject` on the bucket.

Or without the notebook, on a bare instance:

```bash
mkdir -p ~/triton_repo/bert_reranker/1
aws s3 cp s3://YOUR_BUCKET/bert-reranker/neffs.tar.gz /tmp/
tar -C ~/triton_repo/bert_reranker/1 -xzf /tmp/neffs.tar.gz

aws s3 cp s3://YOUR_BUCKET/bert-reranker/triton-img.tar.gz /tmp/
gunzip -c /tmp/triton-img.tar.gz | docker load

# config.pbtxt and model.py come from Step 2 of the notebook; copy them alongside,
# then serve:
docker run -d --name triton-bert-reranker \
  --device /dev/neuron0 --shm-size=4g \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v ~/triton_repo:/models \
  -e NEURON_RT_LOG_LEVEL=ERROR \
  triton-neuron-bert-reranker:sdk231 \
  tritonserver --model-repository=/models --log-verbose=0 --exit-on-error=true

# Model load plus per-bucket warmup takes a couple of minutes. Poll /ready — not the
# bare model endpoint, which can report before the instances are actually up.
until curl -sf localhost:8000/v2/models/bert_reranker/ready; do sleep 10; done
echo READY
```

### A caveat on true cross-compilation

The above is **artifact transfer**: compiling on an inf2 host *for* inf2. Compiling on a host
with **no Neuron device** (via `NEURON_PLATFORM_TARGET_OVERRIDE`) is a different and riskier
proposition. In separate testing on the XLA path, an artifact produced that way ran its first
forward pass correctly and then **hung indefinitely on the second** -- a failure invisible to
any single-inference smoke test. If you cross-compile on a non-Neuron host, validate with
**repeated** forward passes.

## Two findings worth knowing before you deploy

### 1. SDK 2.31 regresses trn2 throughput up to 25% — root-caused to the compiler

Single core, seq_len=1024:

| Platform | BS | SDK 2.27/2.28 | SDK 2.31 | Delta |
|---|---|---|---|---|
| inf2.8xlarge | 16 | 74.58 qps | 67.62 qps | -9.4% |
| trn2.3xlarge LNC=1 | 16 | 83.64 qps | 60.29 qps | **-27.9%** |
| trn2.3xlarge LNC=2 | 16 | 51.31 qps | 45.89 qps | -10.6% |

#### It is `neuronx-cc`, and it entered in one release

Bisect on trn2.3xlarge with the **host driver and runtime held constant** (dkms `2.29.0.0`,
runtime-lib `2.33.10.0`), varying only the compiler and framework, BS=16, seq=1024:

| SDK | neuronx-cc | qps | p50 | vs 2.28 |
|---|---|---|---|---|
| 2.28 | 2.22.12471 | **83.39** | 191.87 ms | — |
| 2.29 | 2.24.5133 | 81.41 | 196.53 ms | -2.4% |
| 2.30 | 2.25.3371 | 81.59 | 196.11 ms | -2.2% |
| 2.31 | 2.26.6360 | **61.01** | 262.26 ms | **-26.8%** |

Two things follow. **The runtime and driver are not at fault** — the SDK 2.28 compiler
reproduces the original 83.39 qps while running on the SDK 2.31 driver and runtime. And it is a
**cliff, not a drift**: 2.28→2.30 costs 2%, while 2.30→2.31 alone costs **25.2%**.

Compiler flags are also not the cause. Reproducing the February 2026 flag set exactly
(`--optlevel=2 --target trn2 --lnc 1 --auto-cast matmult`, transformers 4.53.3, same
warmup/iteration counts and timer) on SDK 2.31 still gives 61.01 qps. Across six flag
combinations the total spread is **1.2%**.

#### Mechanism: ~2x memory traffic for byte-identical compute

Hardware profiles (`neuron-explorer`) of the 2.30 and 2.31 NEFFs:

| | 2.30 | 2.31 | Delta |
|---|---|---|---|
| model_flops | 4.33 T | 4.33 T | +0.0% |
| matmul instructions | 665,302 | 665,593 | +0.0% |
| Tensor engine active | 133.25 ms | 124.27 ms | **-6.7% (faster)** |
| **DMA active time** | **95.32 ms** | **186.88 ms** | **+96.0%** |
| HBM writes | 10.69 GB | 27.01 GB | **+152.6%** |
| HBM reads | 17.79 GB | 32.66 GB | +83.6% |
| Arithmetic intensity | 144.26 | 68.64 | **-52.4%** |
| `mfu_max_achievable` | 100% | 66.06% | **-33.9%** |

The compute is unchanged and the Tensor engine is actually *faster*. Wall clock grew 66.5 ms
while **DMA active time grew 91.6 ms**, so data movement more than fully accounts for the
regression. Extra HBM reads (+14.9 GB) nearly equal extra HBM writes (+16.3 GB) — a ratio of
**0.91**, the signature of values being spilled to HBM and read back rather than kept on-chip.
Average DMA transfer is only 2.3–2.6 KB, far below the ~32 KiB size where DMA is efficient.

Per-engine instruction-binary sizes corroborate a scheduling change rather than a math change:
Tensor **+0.0%**, Activation **+24.3%**, Sync **+89.8%**, Pool **-62.9%**.

The likely cause is the redesigned NIR code-generation backend ("narwhal") that SDK 2.31 made
the default **on Trn2 and Trn3** — which fits the Trn2-specific severity, the single-release
cliff, and a memory-scheduling symptom. We could not verify this directly: the
`--disable-narwhal` switch exists inside the compiler but is **not reachable** from the
`neuronx-cc` CLI, from `--tensorizer-options`, or from any environment variable in this build.

#### The penalty scales with batch size

| BS | 2.30 qps | 2.31 qps | Delta |
|---|---|---|---|
| 1 | 88.52 | 86.56 | **-2.2%** |
| 2 | 84.70 | 77.86 | -8.1% |
| 4 | 83.58 | 73.83 | -11.7% |
| 8 | 80.35 | 66.91 | -16.7% |
| 16 | 81.57 | 61.00 | **-25.2%** |

Larger batches mean larger activation working sets, so a schedule that keeps less on-chip
spills progressively more — further supporting the spill diagnosis, since neither a fixed
per-call overhead nor a uniformly worse kernel would produce this shape.

This also reconciles the single-core and server numbers: the Triton deployment peaks at
**BS=4**, where the penalty is only -11.7%, and multi-instance saturation hides part of even
that. End-to-end Triton loss was **-7.1%** (556.9 → 517.1 inf/s) on trn2 and **-2.4%** on inf2.

#### Workaround

If you run **large batches on trn2**, pin the compiler to SDK 2.30 and leave everything else
on SDK 2.31. This recovers **+33.7%** at BS=16 with identical accuracy (cosine 0.999996,
top-3 unchanged):

```bash
pip install --index-url https://pip.repos.neuron.amazonaws.com \
    "neuronx-cc==2.25.3371.0+f524f7f8" "torch-neuronx==2.9.0.2.14.27725+e2ff0410"
```

Recovery by batch size: +2.3% (BS=1), +8.8% (BS=2), +13.2% (BS=4), +20.1% (BS=8),
+33.7% (BS=16). **A BS=1, latency-oriented deployment should stay on the stock SDK 2.31
compiler** — there is almost nothing to recover and you would be giving up newer fixes.

Reproduction scripts and raw profile data are available on request.

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
9. **Host CPU and Neuron device utilization sampled per benchmark configuration**, so the
   bottleneck is identified from measurement rather than assumed.

### Measuring device utilization from a containerized deployment

Getting `neuron-monitor` to report real numbers against a Triton container took three fixes.
Each failure mode returns a confident, plausible **0.0%**, so none of them announces itself:

1. **It must run inside the container.** The Neuron runtime lives in the container, so
   host-side `neuron-monitor` returns `neuron_runtime_data: []` -- it reports the hardware
   inventory correctly while seeing no runtime at all. Use `docker exec`.
2. **Discard the first sample.** The initial report has no previous interval to diff against
   and is always 0. Reading `stdout.splitlines()[0]` -- the obvious choice -- reports 0%
   utilization while both cores are saturated.
3. **Take the max per core, not the mean of everything.** Each Triton instance is its own
   Neuron runtime pinned to one core, and every runtime reports `0` for the cores it does not
   own. Averaging all the values halves the result.

Also note that on SDK 2.31 `neuron-monitor` takes `-c <config.json>`; there is **no**
`--sample-interval` or `--count` flag, and passing one **exits 0** with `unknown flag` on
stderr -- a silent no-op if you are not checking return values.

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

- **inf2.xlarge** (2 NeuronCores -- recommended, see [Instance sizing](#instance-sizing-use-inf2xlarge))
  or **trn2.3xlarge** (8 logical cores at LNC=1)
- **Deep Learning AMI Neuron (Ubuntu 24.04) 20260813** or later (Neuron SDK 2.31)
- Docker
- ~300 GB disk (Triton image plus five compiled graphs)
- **Swap is required before compiling.** The DLAMI ships with none. On inf2.xlarge the
  BS=16 compile peaks at **14.75 GB RSS on a 15 GB host**, so without swap `neuronx-cc` is
  OOM-killed -- and the error names the compiler, not memory. (An instance that only *loads*
  prebuilt graphs never runs the compiler and does not need swap.)
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

All measurements taken on Neuron SDK 2.31 (DLAMI `20260813`): neuronx-cc `2.26.6360.0`,
torch-neuronx `2.9.0.2.15.32035`, torch 2.9.1, transformers 4.57.6, Triton Inference Server
`r26.01` (python backend, built from source on the Neuron PyTorch inference image).

- inf2.8xlarge and trn2.3xlarge results: **2026-09-30**
- inf2.xlarge results: **2026-10-01** (two instances, us-east-2 and us-west-2)

All instances have been terminated.
