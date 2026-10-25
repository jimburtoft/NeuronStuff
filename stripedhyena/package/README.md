# striped-hyena-neuron

StripedHyena 7B ([StripedHyena-Hessian-7B](https://huggingface.co/togethercomputer/StripedHyena-Hessian-7B)
base and [StripedHyena-Nous-7B](https://huggingface.co/togethercomputer/StripedHyena-Nous-7B) chat; the
StripedHyena 1 architecture: 16 Hyena blocks with a 2-state complex modal SSM alternating with 16 GQA
attention blocks, 32K context) on AWS Trainium 2 with PyTorch Native (`torch.device("neuron")`).

It runs the upstream `stripedhyena` model code (vendored, unmodified) with runtime patches that remove
every CUDA, complex-dtype and FFT dependency, and adds a compiled segmented prefill, a compiled decode
step with explicit state, a fused NKI Hyena decode kernel, nkilib `attention_tkg` for long-context GQA
decode, the nkilib `mlp` kernel for the MLPs, and Megatron tensor parallelism. **0 CPU fallback ops** in prefill and decode.

## Measured (trn2.3xlarge = 1 Trainium2 device; public Neuron SDK 2.33 DLAMI, torch-neuronx 2.14.3; bf16 + fp32 poles/residues)

Decode throughput, one device, TP=4, KV filled with a real prefill to ~context - 200 tokens (n = 3 processes for the best points):

| context | best | other batch sizes |
|---|---|---|
| 2,048 | **5,789 tok/s @ (B 512, 2K)** | 81 @ B 1 (step graph), 490 @ B 8, 938 @ B 16, 1,512 @ B 32, 2,343 @ B 64, 4,006 @ B 128, 5,037 @ B 256 |
| 8,192 | **2,017 tok/s @ (B 128, 8K)** (`attention_tkg`) | 76 @ B 1, 761 @ B 16, 948 @ B 32, 1,527 @ B 64, 1,975 @ B 136 |
| 32,768 | **639 tok/s @ (B 30, 32K)** (`attention_tkg`) | 75 @ B 1, 328 @ B 8, 496 @ B 16, 586 @ B 24, 595 @ B 32 |

Per-request latency (B=1, TP=4, whole decode step as one graph): **12.3 ms/token @ 2K**, 13.2 @ 8K, 13.4 @ 32K.

Time to first token (B=1, prompt ids on device -> first token on host, p50 of 10):

| prompt | TP=4 | one logical core |
|---|---|---|
| 128 | **24.7 ms** | -- |
| 2,048 | **142 ms** | 419 ms |
| 8,192 | **743 ms** | 2.42-2.52 s |
| 32,768 | **7.59 s** | 29.8 s (2,048-token segments; 32.0 s at the 512 the API uses on one core at 32K) |

Standard serving workloads, TP=4 (p50; TTFT / time per output token): 128 in / 128 out at B=1 25 ms / 12.3 ms; 2,048 in /
512 out at B=1 141 ms / 12.6 ms; at B=64 855 ms / 23.5 ms (128 in) and 7.6 s / 32.2 ms (2,048 in). DP=4 (4 independent
single-core replicas) reaches 4,289 tok/s @ (224, 2K) and 811 @ (40, 8K); TP=4 is faster at every context.

Accuracy vs the upstream reference implementation (same weights, same token sequences):
WikiText-2 perplexity 6.1966 @ 2048 (reference 6.1945) and 5.5418 @ 8192 (5.5423); ARC-Easy
0.762 / 0.760 (acc / acc_norm), PIQA 0.820 / 0.850, HellaSwag 0.538 / 0.704 on 500-example subsets (reference
0.760 / 0.760, 0.812 / 0.850, 0.538 / 0.704). Decode logits over 128 teacher-forced positions: rel L2 0.006-0.009,
top-1 agreement 0.98-1.00, mean KL <= 0.0007. 0 CPU fallback ops in prefill and decode.

Where the batch ceiling comes from: KV cache is 64 KiB per token per sequence (16 GQA layers, 8 KV heads,
d=128, bf16); at TP=4 each of the 4 logical cores holds 1/4 of it plus 1/4 of the 15.3 GB of weights in its
24 GB of HBM. B=144 @ 8K and B=36 @ 32K run out of HBM; B=576 @ 2K fits but is slower than B=512 (104 vs 88 ms/step).
The Hyena half of the model has a constant ~0.4 MB/sequence decode state, independent of context.

Known limits on SDK 2.33: one logical core at a 32K window serves B=1 through `generate()` (35.9 ms/token) and needs
512-token prefill segments (the default there; 2,048 runs out of HBM in prefill beside the KV cache). 32K TTFT is ~1.2x
slower than on the earlier torch-neuronx 2.13 / SDK 2.32 stack (attention prefill over the 16K/32K KV prefix). Use TP=4 for long
contexts.

## Install

Launch a trn2 instance from the public **Deep Learning AMI Neuron (Ubuntu 24.04) SDK 2.33.0** (multi-framework) or the
*Deep Learning AMI Neuron torch-neuronx 2.14.0* DLAMI, then use its TorchNeuron venv, which already provides `torch`,
`torch-neuronx`, `nki` (with `nkilib`), `einops`, `safetensors`, `pyyaml` and `tokenizers`:

```bash
source /opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin/activate
pip install -e . --no-deps        # --no-deps is required: never let pip touch the venv's torch stack
```

## Use

```python
import torch, striped_hyena_neuron as shn
from tokenizers import Tokenizer

model, info = shn.load("~/weights/hessian-7b")            # enable() + meta build + strict load + to_device
gen = shn.compile(model, B=1, window=4096)               # compiled prefill + decode -- NOT optional
tok = Tokenizer.from_file(shn.TOKENIZER_JSON)
ids = torch.tensor([tok.encode("The striped hyena is").ids])
print(tok.decode(gen.generate(ids, max_new_tokens=64)[0].tolist()))
print(shn.config_summary())                              # paste into issue reports
```

Tensor parallel (one trn2.3xlarge, 4 logical cores; the fastest configuration at every batch size):

```python
# torchrun --standalone --nproc_per_node=4 app.py
import torch.distributed as dist
dist.init_process_group(backend="neuron")
model, _ = shn.load(weights, tp=(dist.get_rank(), 4))
gen = shn.compile(model, B=64, window=8192, tp=4)
```

Lower level (inside `torch.inference_mode()`): `gen.prefill(ids)` returns `(first_tokens, state)`; `gen.dec(tokens, state)`
advances one step in place and returns `(logits, next_tokens)`.

| call | why it exists |
|---|---|
| `load` / `to_device` | `nn.Module.to()` misses the plain tensor attributes upstream stores on modules |
| `make_config` | upstream `dotdict.__getattr__` is `dict.get`; every config read inside a compiled region would be a graph break |
| `compile` | eager decode is ~4x slower; compile all blocks or none (partial compilation measured slower) |

## Defaults and what they encode

`shn.enable(**overrides)` (called by `load`) sets process-global options; `config_summary()` prints them
with their measured justification. Per-call-site choices made by `compile` (`shn.decode_opts(B, window, tp)`):

| choice | rule | measured (SDK 2.33, TP=4 unless noted) |
|---|---|---|
| decode graph | whole step as one graph at B=1 under TP; per-block-type graphs otherwise | B=1: 12.3 ms (step) vs 16.3 ms (per-block) at 2K |
| `rms_custom` | on | `torch.rms_norm` lowers to a fused norm: 1.05-1.13x at B>1 (one core) |
| `nki_mlp` | on | MLP weights stored (in, out). TP shard: `x @ Wt` decode, nkilib `mlp` CTE kernel for prefill rows >= 2,048: B=1 1.14-1.17x, B=8 1.09x, B=512 1.05x, 8K B=128 1.04x. One core: TKG kernel at <= 32 rows (RMSNorm fused into it, except with `cascade`), CTE at >= 512 rows, `x @ Wt` between: B=8 @ 8K 1.09x over `rms_custom` alone |
| `attention_tkg` | windows >= 32K at B > 1; 8K windows at B >= 64 | 8K B=128 1.37x, 32K B=28 1.29x; at 2K it is slower (and B=512 runs out of HBM with it) |
| `bucket` | on | attention reads a power-of-two KV prefix >= fill; 1.2x when fill is far below the window; exact |
| `cascade` | off (opt in for single-core B=1) | fused NKI FIR + IIR + gate: 33.3 vs 34.8 ms at one core B=1; does not stack with `rms_custom`; its weight layout makes prefill 1.36x slower |
| prefill segments | prompts > 4,096: 2,048 on one core and under TP up to 8K; 8,192 under TP above 8K | exact; 8K TTFT 743 ms at 2,048 vs 781 at 4,096; 32K 7.59 s at 8,192 vs 9.13 at 16,384 |
| prefill attention query chunks | 2^27 score elements; 2^25 when a segment reads a KV prefix > 8,192 | 32K TTFT 9.55 -> 7.59 s; 8K unchanged |
| fp32 prefill residual | on | without it one WikiText window scored PPL 4.94 instead of 4.66 under compilation |

Constraints: the KV window must be a multiple of 256 and at most 32,768; TP degree must divide 8 (KV heads).
Compile one (batch, window) configuration per process; a process keeps its NeuronCores until it exits.

## Tests

```bash
make test-cpu    # CPU, tiny random model: package vs the unpatched upstream FFT/complex path, segmented
                 # prefill, bucketing, TP=2 over gloo (needs torch, pyyaml, einops)
make test-sim    # NKI simulator: fused cascade kernel vs NumPy (needs `pip install nki`)
make test-hw     # trn2 + weights (SHN_WEIGHTS; SHN_REFDATA optional): parity gates vs the
                 # reference (full logits if SHN_REFDATA is set, else the compact data/parity_reference.json) at 1 core, 1 core + cascade, TP=4 step graph, TP=4 + attention_tkg;
                 # batched generation consistency
```

Example: `examples/generate.py` (one core or `torchrun --nproc_per_node=4 ... --tp 4`).

Provenance of vendored and adapted code: `PROVENANCE.md`. License: Apache-2.0.
