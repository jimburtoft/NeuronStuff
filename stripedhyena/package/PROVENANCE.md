# Provenance

## Vendored, unmodified

| path | source | revision | license |
|---|---|---|---|
| `striped_hyena_neuron/_vendor/stripedhyena/` | [togethercomputer/stripedhyena](https://github.com/togethercomputer/stripedhyena) `stripedhyena/*.py` | `7e13f618027fea9625be1f2d2d94f9a361f6bd02` (2024-03-07) | Apache-2.0 (`_vendor/LICENSE.stripedhyena`) |
| `striped_hyena_neuron/_vendor/stripedhyena_configs/7b-sh-32k-v1.yml` | same repo, `configs/` | same | Apache-2.0 |
| `striped_hyena_neuron/_vendor/tokenizer.json` | same repo, `tokenizer/` | same | Apache-2.0 |

The package changes the upstream model only at runtime (`enable()` replaces
`ParallelHyenaFilter.parallel_forward` / `sequential_forward`, `RMSNorm.forward`, and registers a
`flash_attn.modules.mha.MHA` stand-in before `stripedhyena.model` is imported). An upstream rename can
break these patch targets; the CPU tests (`tests/test_cpu.py`) check the patched model against the
unpatched upstream FFT/complex path.

Weights are not included: [togethercomputer/StripedHyena-Hessian-7B](https://huggingface.co/togethercomputer/StripedHyena-Hessian-7B)
(rev `59e474c`) and [StripedHyena-Nous-7B](https://huggingface.co/togethercomputer/StripedHyena-Nous-7B) (rev `d8a5c9f`), Apache-2.0.

## Adapted from the Evo 2 (StripedHyena 2) Trainium decode package

Source: `evo2_neuron_decode` (Apache-2.0), commit `7c93113`. Each adapted file has a header naming
its source file. What changed for StripedHyena 1:

| file here | adapted from | changes |
|---|---|---|
| `kernels/sh1_cascade.py` | `kernels/hcl_cascade_fused.py` | FIR k=3 with bias; column split (contiguous after `colperm`) instead of stride-3 interleave; 2-state **complex** modal IIR as re/im pairs; planar FIR state |
| `tkg.py` | `patches/tkg_mha2.py` | GQA 32 Q / 8 KV heads: fold `bs = B*Hkv`, `q_head = 4` (no head fold); per-chunk contiguous caches; fill bucketing via `curr_sprior` |
| `tp.py` | `tp.py` | head-wise column split of the Hyena projections (`[x2|x1|v]` per head); GQA KV-head split (world <= 8); per-channel poles/residues/D |
| `traceable_config.py` | `patches/dotdict_patch.py` | `get(k, default)` honours the default for missing keys |
| `api.to_device` | `api.to_device` | same idea (plain tensor attributes); no dtype argument (checkpoint dtypes are kept) |
| `api.compile` | `api.compile_blocks` | compiles tensors-only per-block-type wrappers (one graph per block type, shared by 16 blocks) instead of the upstream blocks |
| `opts.py` (`mlp_cat`, `pretrans`) | `patches/mlp_cat_patch.py`, pre-transposed weights | both measured not to help here (off) |

## From the AWS Neuron NKI Library `nkilib` (Apache-2.0)

`kernels/nkilib_mlp_compat.py`: a copy of `nkilib/core/mlp/mlp_cte/basic/mlp_cte_basic_transpose.py`
`_perform_intermediate_transpose` (file md5 `18cf1614d57ac4e4dab79145408e3c35`, identical in the torch-neuronx 2.13 / SDK 2.32
stack and in Neuron SDK 2.33) with its two list comprehensions rewritten as loops (same values, same order). It is
installed at runtime **only under the earlier NKI 0.7 frontend (torch-neuronx 2.13)**, which could not specialize the comprehensions;
NKI 0.7.0 GA (SDK 2.33) traces kernels with Python and runs the unmodified function, so `install()` is a no-op there
(`SHN_NKILIB_COMPAT=1/0` forces it). The installed nkilib files are never modified. `nki_mlp.py` and `tkg.py` call the
SDK's `nkilib.core.mlp.mlp` and `nkilib.core.attention.attention_tkg` (not vendored).

## Validated environment

Public Neuron SDK 2.33 DLAMI (`Deep Learning AMI Neuron (Ubuntu 24.04) SDK 2.33.0 20261020`), venv
`/opt/aws_neuronx_venv_torch_neuronx_2_14_0`: torch 2.14.0, torch-neuronx 2.14.3.0.1, neuronx-cc 2.28.7631.0, nki 0.7.0
(nkilib bundled), driver 2.31.7.0, runtime 2.35.30.0; trn2.3xlarge, LNC=2, 2026-10-20. Earlier development and tuning:
torch-neuronx 2.13 on SDK 2.32.

## Written for this package

`attention.py` / `sdpa_mha.py` (fixed-window GQA SDPA MHA, flash-attn RoPE semantics), `rmsnorm.py`
(flash-formula RMSNorm: eps inside the root -- the upstream torch branch is not equivalent for these
checkpoints), `modal.py` + `hyena.py` (complex-free, FFT-free modal long convolution and decode state),
`prefill.py` (compiled, segmented, exact prefill with fp32 residual stream), `decode.py` (functional
compiled decode step with explicit state).
