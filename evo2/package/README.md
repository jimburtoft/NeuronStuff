# evo2-neuron-decode

Autoregressive **decode** for Evo 2 / StripedHyena 2 (`evo2_7b_base`) on AWS Trainium 2,
via PyTorch Native (`torch.device("neuron")`). Fused NKI kernels plus the wiring that routes
Arc Institute's [`vortex`](https://github.com/Zymrael/vortex) decode path through them.

## Should you use this?

Be honest about the fit before adopting:

| your goal | use |
|---|---|
| **autoregressive generation / decode** | **this package** — it is the only Trainium implementation of it we know of |
| training, LoRA fine-tuning, forward-only embeddings | **[canallee/evo2-neuron](https://github.com/canallee/evo2-neuron)** — a clean re-implementation with a training stack; it explicitly does *not* do decode |
| a turnkey, hardened serving stack | neither yet — see [Limitations](#limitations) |

## Measured

One trn2 device = a trn2.3xlarge (4 logical cores, LNC=2), public Neuron SDK 2.33 DLAMI, `evo2_7b_base`, bf16,
all package defaults, 128-nt prompt. Batched decode throughput:

| config | ms/step | tok/s per device |
|---|---|---|
| TP=4, B=88, `max_seqlen=8192`, KV fill 128 | 37.7 | **2,333** |
| TP=4, B=88, KV fill 1,024 | 40.3 | **2,181** |
| TP=4, B=88, KV fill 4,096 | 56.1 | **1,569** |
| TP=4, B=80, KV fill 7,680 | 49.4 | **1,618** |
| DP=4 (4 processes), B=128 per core, `max_seqlen=256` | 152.6 | **3,356** |

Reproduce with `bash run.sh fill` and `bash run.sh dp4-256` in the parent directory. Single-request (BS=1) decode is ~30 ms/step on
one core; this package is built for batched throughput, not single-request latency. Full tables, earlier
stacks and caveats: `../BENCHMARK.md`.

## Install

Requires the AWS Neuron **SDK 2.33** torch-neuronx environment, which supplies `torch`, `torch-neuronx`, `nki`
and `nkilib`. On the Neuron DLAMI (`Deep Learning AMI Neuron (Ubuntu 24.04) SDK 2.33.0`, or the single-framework
`Deep Learning AMI Neuron torch-neuronx 2.14.0`):

```bash
source /opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin/activate
pip install -q rich pyyaml einops           # vortex's own imports
pip install -e . --no-deps                  # --no-deps is REQUIRED
```

> `--no-deps` is not a convenience. This package declares **no** torch/nki dependencies on
> purpose: letting pip resolve them will upgrade or replace the environment's pinned stack and
> break it. There is nothing to install but this package.

**Compile cache.** The first run compiles ~2,700 graphs (TP=4 8K + DP=4 256). On SDK 2.33 a first compile of a
large-batch TP=4 graph occasionally fails with `NCC_INLA001` / `NCC_IBIR229` (scratch/state buffer allocation);
re-running compiles it, and a cached graph never fails. Keep `TORCH_NEURONX_NEFF_CACHE_DIR` between runs, or ship a
pre-populated cache with your deployment.

## Use

```python
import torch, yaml
import evo2_neuron_decode as evo2d
from vortex.model.utils import dotdict
from vortex.model.model import StripedHyena

MAX_SEQLEN = 512                                   # MUST be a multiple of 256

evo2d.enable(max_seqlen=MAX_SEQLEN)                # patch vortex (defaults = champion)
cfg = dotdict(yaml.safe_load(open("configs/evo2-7b-8k.yml")))
cfg = evo2d.prepare_config(cfg, MAX_SEQLEN)        # required keys + traceable config

model = StripedHyena(cfg)
model.load_state_dict(torch.load("evo2_7b_base.pt", weights_only=False), strict=False)
model = evo2d.to_device(model, dtype=torch.bfloat16).eval()   # NOT bare .to()
evo2d.compile_blocks(model)                        # NOT optional: 1.43x decode, 6.1x prefill

print(evo2d.config_summary())                      # paste into any bug report
```

Three of those calls exist because the obvious version silently fails:

| call | why you cannot skip it |
|---|---|
| `prepare_config` | sets `use_flash_attn=False`. vortex's `MHA.__init__` **hard-asserts** *"FlashAttention is not installed"* (CUDA-only), so the flash branch cannot even be constructed here. |
| `to_device` | `nn.Module.to()` misses vortex's plain tensor **attributes** and its `block_idx_to_device` map. Bare `.to()` dies with `RuntimeError: input tensor is on cpu device, expected neuron`. |
| `compile_blocks` | 1.43x decode / 6.1x prefill, and partial compilation is *slower* than none. |

`enable()` takes keyword overrides for every optimization, each documented with the
measurement that justifies it (`config_summary()` prints them). Options measured **not** to
help are present but default off, so the negative results stay discoverable rather than being
silently re-tried.

## Constraints

**`max_seqlen` must be divisible by 256.** `attention_tkg` shards `s_prior` by 2 and the shard
must be a multiple of `p_max=128`. `enable(max_seqlen=...)` validates this and tells you the
next valid value; without it you get a bare `neuronx-cc` `NCC_INKI016`
*"Sharded s_prior must be divisible by p_max"* at kernel-compile time. Pass
`enable(attention_tkg=False)` for eager MHA, which has no such constraint (and is ~1.7x slower).

**Compile all 32 blocks or none.** Partial compilation measured *slower* than eager.

**Startup is dominated by the 13 GB checkpoint load** (~79 s compiled, ~118 s eager). Load once
and serve many; it is not per-request cost.

## Tests

```bash
make test-sim     # CPU only, no Trainium needed
make test-hw      # requires /dev/neuron0
```

Verified on trn2.3xlarge, public SDK 2.33 DLAMI: **67 passed** (`python -m pytest tests`, simulator + hardware).

`test-sim` is the correctness gate that matters: every kernel is checked against an **fp64
reproduction of the exact upstream op chain**, per kernel, including per-row checks at batch.

Batch invariance: rows inside one batch are bit-identical. Across batch sizes the compiler accumulates in a
different order, so logits differ by 1-2 bf16 ulps and free-running greedy tokens can flip where the top two
logits tie; the hardware test therefore teacher-forces every batch size onto one token stream and compares logits.

Deliberately **not** used as the verdict: end-to-end logits cosine similarity. The residual
stream grows ~2e14x across 32 blocks, so an early block's contribution to the final logits is
~1e-14 relative — a completely broken block can read `cos_sim 1.000000`. Discrete greedy-token
equality and per-kernel parity do not dilute. (Credit: the argument for gating on per-block
increments rather than a cumulative tensor is made well in canallee/evo2-neuron's `PARITY.md`.)

## Limitations

- **This monkey-patches vortex in place** — `HyenaCascade.sequential_forward`,
  `ParallelGatedMLP.forward`, `RMSNorm.forward`, `MHA.forward`. An upstream vortex change can
  break it **silently**. Pinned and verified against the commit in `VENDORED.md`.
- `attention_tkg` takes at most 128 (batch x heads) rows per call; larger batches are tiled over the folded BH
  axis (transparent).
- TP=4 at `max_seqlen=8192` fits B=88 (B=80 near full fill); B=96 runs out of device memory. A one-shot B=64 x
  4,096-token prefill at TP=4 also runs out of memory -- use `evo2d.prefill(chunk_rows=8)`.
- 22 Dynamo graph breaks remain. **None are in this package** — 7 are the Neuron backend hashing
  a NEFF cache key inside the traced region, the rest are PyTorch internals.
- FP8 is not supported: `nkilib`'s FP8 MLP kernels require an `ActFnType`, and Evo 2's
  `evo2_style_activations` makes 31 of 32 layers `nn.Identity`, which has no `ActFnType`
  equivalent. Coverage would be 1/32.

## License

Apache-2.0 (see `LICENSE` and `NOTICE`). Model definition and weights are Arc Institute's
([`ArcInstitute/evo2`](https://github.com/ArcInstitute/evo2),
[`Zymrael/vortex`](https://github.com/Zymrael/vortex)) — see `VENDORED.md` for pins.
