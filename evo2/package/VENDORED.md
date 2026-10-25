# Vendored / upstream dependencies

This package contains **no** model definition. It patches upstream `vortex` at runtime, so the
patch targets are a real coupling and the pins below matter.

| | |
|---|---|
| model definition | [`Zymrael/vortex`](https://github.com/Zymrael/vortex) — `vortex/model/{model,layers,attention,utils}.py` |
| architecture config | [`ArcInstitute/evo2`](https://github.com/ArcInstitute/evo2) — `configs/evo2-7b-8k.yml` |
| weights | [`arcinstitute/evo2_7b_base`](https://huggingface.co/arcinstitute/evo2_7b_base) |
| kernels | `nkilib` as shipped inside the `nki` wheel of the Neuron SDK 2.33 torch-neuronx environment (not vendored) |
| license | Apache-2.0 (vortex, evo2) |

## Patch targets

`enable()` replaces these bound methods. If upstream renames or re-signatures any of them, this
package breaks — and may break *silently* if the signature still matches.

| target | replaced by | why |
|---|---|---|
| `HyenaCascade.sequential_forward` | `cascade_patch`, `hcm_patch` | fuse the FIR/IIR cascade into one NKI launch |
| `ParallelGatedMLP.forward` | `mlp_cat_patch` | fuse `l1`+`l2` into one GEMV |
| `RMSNorm.forward` | `norm_patch` | fused norm kernel |
| `MHA.forward` | `tkg_mha2` | `nkilib attention_tkg` with a tiled head fold |
| various lowering paths | `fir_conv_patch`, `iir_recurrence_patch`, `rmsnorm_patch`, `rotary_patch` | make the model *run* on this backend at all |

## Config keys that must be set explicitly

vortex's config is a `dotdict` whose `__getattr__` **is** `dict.get`, so **an absent key silently
reads as `None`/off**. A config-version mismatch therefore yields a running model with different
numerics and no error. Verify these are present rather than relying on defaults:

| key | required value | if absent |
|---|---|---|
| `interleave` | `True` | lane ordering changes; wrong results, no error |
| `evo2_style_activations` | `True` | 31 of 32 MLP layers use GELU instead of Identity |
| `use_fp8_input_projections` | `False` | unsupported here |

*(That same `dict.get` is why `make_config()` exists: Dynamo cannot trace a C builtin
`__getattr__`, so every `config.x` read inside a compiled region breaks the graph — 89 of 111
observed breaks.)*
