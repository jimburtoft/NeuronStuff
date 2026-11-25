# Reproducing the results: fine-grained MoE training on Trainium2

End-to-end guide for the work in this directory: expert parallelism (EP) for the NKI MoE, the EP-aware
block bound, and the gate/up weight pre-transpose — on a 39.8B-parameter / 2.5B-active fine-grained MoE
(256 experts, top-6, 49 layers) with AWS Neuron SDK 2.33.

> Every number below is a **hardware measurement** on trn2. Nothing is estimated, extrapolated, or
> taken from the CPU simulator. Where a change did not help, or helped only on a small surrogate,
> that is stated.

## 1. What actually moved the needle

| Change | Effect | Where |
|---|---|---|
| **Expert parallelism for the NKI MoE** | **3.3x** throughput vs the TP-on-experts fallback | §4 |
| **EP-aware `num_blocks` bound** | **+21%** on top of EP (at EP=16) | §5 |
| Gate/up weight pre-transpose | 1.18x on the MoE backward alone, bit-exact | §6 |
| Sequence length 4096 -> 8192 | **+65%** | §7 |

Full model, 1x trn2.48xlarge (64 logical cores at LNC=2), seq 4096, EP=16:

| config | tok/s/core | MFU (as-reported) | MFU (corrected, §8) |
|---|---|---|---|
| TP4 on experts (the fallback) | 212 | 2.52% | 3.30% |
| EP16 | 961 | 8.40% | 14.95% |
| **EP16 + EP-aware blocks** | **1,166** | **10.18%** | **18.14%** |

Best measured full-model configuration: **seq 8192, EP16 + EP-aware blocks → 20.1% corrected MFU.**

**Read §8 before quoting any MFU number.** The stock MFU calculation under-reports EP MoE runs.

## 2. Environment

| | |
|---|---|
| SDK | **2.33** |
| DLAMI | *Deep Learning AMI Neuron (Ubuntu 24.04) SDK 2.33.0 20261020* |
| venv | `/opt/aws_neuronx_venv_torch_neuronx_2_14_0` (PyTorch Native; torch 2.14, torch-neuronx 2.14.3, nki 0.7.0) |
| LNC | `NEURON_LOGICAL_NC_CONFIG=2` (trn2 default). 1 device = **4 logical cores**; trn2.48xlarge = **64** |
| Instances | trn2.3xlarge (4 cores) for development; trn2.48xlarge (64 cores) for EP=16 |

```bash
source /opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin/activate
sudo fallocate -l 64G /swapfile && sudo chmod 600 /swapfile && \
  sudo mkswap --force /swapfile && sudo swapon /swapfile   # neuronx-cc needs the headroom
```

`mkswap --force` is required on the DLAMI kernel; plain `mkswap` silently no-ops.

## 3. Framework

```bash
git clone --branch release_2.33.0 --depth 1 https://github.com/aws-neuron/torchtitan-neuron.git
cd torchtitan-neuron && pip install -r requirements.txt && pip install -e .
```

The EP support is an upstream PR: **[aws-neuron/torchtitan-neuron#18](https://github.com/aws-neuron/torchtitan-neuron/pull/18)**.
Until it merges, apply it as a patch to `torchtitan_neuron/models/qwen3/nki_moe.py`.

## 4. Expert parallelism for the NKI MoE

**The problem.** `NKIMoE.set_sharding_config(enable_ep=True)` raises
`NotImplementedError("Qwen3 NKI MoE supports TP-only experts")` — in qwen3, deepseek_v3 and gpt_oss alike.
The only EP dispatcher (`neuron_ep`, all-to-all) is Trn3 / one-local-expert-per-rank. So on trn2 you could
have the fast NKI MoE **or** expert parallelism, not both — forcing a fine-grained MoE onto TP-on-experts,
which splits each expert's already-small intermediate dimension.

Verified on hardware (SDK 2.33): the *standard* (non-NKI) MoE **does** run EP on trn2; only the NKI path refused it.

**The fix — replicated-token EP.** The stock `NKIMoE` already runs replicated-token (every rank sees all
tokens; the output is `Partial` and all-reduced). So EP needs no token all-to-all:

1. Each rank holds `e_local = num_experts / ep_size` experts (params sized at `__init__` from
   `Config.ep_size`, so FSDP wraps already-local shards).
2. Each rank runs **one batched `nki_fwd`** over its local experts, using the local-expert column slice of
   `masked_scores`.
3. `dist.all_reduce` sums the per-rank partials.

This sidesteps the all-to-all path's trn2 blockers entirely: no argsort (so no 16,384-element sort limit),
static shapes, no 128-alignment requirement.

**One batched call matters.** A first version looped one `nki_fwd` per local expert (to dodge an alignment
constraint) and was *slower than TP4* — 32 kernel launches per layer at EP=4. Padding `num_blocks` to a
multiple of 16 satisfies `indexed_flatten` and allows a single batched call:

| EP=4 variant | MFU |
|---|---|
| per-local-expert loop | 7.34% |
| **single batched call** | **14.98%** |

## 5. EP-aware `num_blocks` bound

`nki_fwd` sizes its block budget from a worst case: `ceil(tokens * min(top_k, num_experts) / block) + num_experts`
— i.e. assuming a token routes to `top_k` *local* experts. Under EP it routes to only `~top_k * e_local / E`.
Measured over-allocation (256 experts, top-6, seq 4096, block 256):

| EP | blocks needed | stock allocates | over-allocation |
|---|---|---|---|
| 4 | 64 | 160 | 2.5x |
| 16 | 16 | 112 | **7.0x** |

Every padding block still costs DMA (the `-1`-padded token gather and per-block weight-grad traffic), which is
the MoE's dominant cost. Scaling the bound by `e_local / E` with a capacity factor gave **+16% at EP=4** and
**+21% at EP=16** — the win grows with EP, matching the block-reduction ratio.

Two notes on correctness:
- The bound is clamped at `>= 1.0` affinity per token, a **structural floor**: `num_blocks >= ceil(tokens/block) + e_local`
  (80 vs 64 needed at the test shape). It cannot undersize, and an overflow would fail loudly in
  `repeat_interleave(output_size=...)`, not silently drop tokens.
- Loss differs by ~0.1% (bf16 reduction order). Capacity factors 1.0 and 1.5 give **identical** loss at
  *different* block counts — which is what demonstrates nothing is being dropped.
- Sizing from the *actual* routed counts would be exact but is **impossible under `torch.compile`**:
  `num_blocks` drives tensor shapes, and a data-dependent `.item()` raises
  "Could not extract specialized integer from data-dependent expression".

## 6. Gate/up weight pre-transpose

The 2.33 blockwise MoE backward re-transposes each expert's gate/up weight **per token block**, although all
blocks of an expert share it. A separate one-pass kernel (`kernels/weight_transpose.py`) pre-transposes all
experts once (4.3 ms for 16 experts); the backward then plain-loads the result.

Backward-only A/B, 16,384 rows x 16 experts, block 256: **53.5 ms → 45.5 ms (1.18x)**, gradients **bit-identical**.

Honest caveat: less than the naive "remove the transpose" estimate, because under the transpose are matmuls
that were always there, and the pre-transposed load is I-strided (HBM read 6.8 → 12.8 GB). It trades transpose
compute for read traffic; net positive here, but shape-dependent.

## 7. Sequence length — the largest single lever, and its ceiling

Same kernels and parallelism, full model, EP=16:

| seq | tok/s/core | MFU corrected |
|---|---|---|
| 4096 | 1,184 | 13.1% |
| **8192** | **1,623** | **20.1%** |
| 16384 | — | **out of device memory** |

Attention's O(S²) FLOPs grow with sequence length while the MoE's per-token DMA does not, so arithmetic
intensity rises and the DMA-bound MoE stops being the limiter.

**The ceiling is memory, and it is hard.** seq 16384 fails at model load with
`NRT EXECUTION FAILED: Failed to load model, Failed to allocate resource` — 24 GB per logical core, with
~13 GB already resident in params/optimizer/activation-checkpoint buffers. `local_batch_size` is already 1.
`fsdp_reshard_after_forward=never` (which gave +2-5% on a 4-layer surrogate) also **OOMs on the full model**:
un-sharding the expert slice across the EP group adds ~4.6 GB/core.

Going beyond 8192 needs context parallelism, which is **not implemented** for this NKI attention — stock
TorchTitan CP cannot wrap it. Only the 14 full-attention layers would need it (the 35 sliding-window layers
scale linearly to 32K unaided), and seq 32768 additionally fails to compile
(`BIR emission failed: body blob exceeds uint32 offset range`).

**A 4-layer surrogate reaches 31.4% corrected MFU at seq 16384. That does not transfer to the full model
and should not be quoted for it** — the surrogate's attention/MoE FLOP ratio flatters it, and the full model
cannot fit that sequence length at all.

## 8. MFU accounting — the stock number under-reports EP runs

`torchtitan/models/utils.py::get_moe_model_nparams_and_flops` computes

```
nparams_sparse_active = router + shared + nparams_experts * top_k // num_experts
```

from `model.named_parameters()`. **Under EP each rank holds only its local expert slice**, so
`nparams_experts` is already `1/ep_size` of the global count — and the formula then divides by the *global*
`num_experts` again. The expert FLOPs term is undercounted by `ep_size`.

Measured at EP=16 on the full model (logged `sparse 2,685,927,424` = local experts 2.466 B + shared + router,
confirming the local slice): the expert term contributed 0.058 B instead of 0.925 B — **16x low on that term,
1.27x on the total**. So reported 10.18% MFU is really **18.14%**.

This affects any EP sparse-MoE user, not just this model. The denominator is fine: `79e12 * 2 = 1.58e14`
is correct, because 79 BF16 TFLOPS is per *physical* NeuronCore-v3 and an LNC=2 logical core is two of them —
and one `torchrun` rank drives one logical core.

## 9. Where the time goes

From an interval-merged Kineto trace (union of intervals = real wall-clock; **summed** op durations overcount
by up to 21x and must not be used for this) plus a NEFF profile, one 515 ms step:

- **The device is busy 96.6% of the step.** The gap to peak is efficiency *within* device work, not host stalls.
- **MoE ≈ 66% of device time, DMA-bound** (in-graph: DMA 76% active vs TensorE 62%, VectorE 58%). Fine-grained
  experts mean small matmuls with a large gather/scatter cost: 28% of FLOPs, ~66% of time.
- **Attention backward ≈ 20%, and compute-saturated** (VectorE 94%, TensorE 89%, HBM 0.11 GB). Real work at
  near-peak occupancy — a tile-size sweep moved it ±1%. Nothing to win here.
- **FSDP collectives run on the device** (Neuron collectives, not CPU), and ~87.6 ms (**17% of the step**) is
  *not* hidden behind compute. The EP all_reduce itself is only ~25 ms.
- `neuron::alloc::lazy` looks like 67.5% of summed device time but has **0.5 ms exclusive** — an enclosing
  annotation, not a cost.

## 10. Things that did not work

Recorded so nobody re-runs them:

| Attempt | Result |
|---|---|
| Per-expert write-once weight gradient | Correct (bit-exact) but **no speedup**. `oob_mode.skip` does **not** drop the DMA transfer — an OOB-skipped write still consumes full write bandwidth (3.3 MB and 0.31 ms either way). It prevents corruption, not cost. |
| Raising MoE buffer degrees for overlap | **Regressed** (44.1 → 46.8 and → 47.8 ms). The library blocking defaults are near-optimal; fp32 grad accumulation forces `buffer_degree_fp32=1`. |
| Reducing the SWDGE token gather | Reducing one DMA queue just shifts the wall to the pipeline and the other queues. |
| `attention_bwd` tile tuning | Flat (±1%); the kernel is compute-saturated. `q_tile_group_size`/`k_seq_section_len` exist in `setup_config` but are not plumbed through the public signature. |
| `fsdp_reshard_after_forward=never` | +2-5% on a 4-layer surrogate, **OOM on the full model**. |

## 11. Reproducing a run

```bash
NEURON_LOGICAL_NC_CONFIG=2 \
FLINT_EP=16 EP_TIGHT=1 EP_GLOBAL_E=256 EP_CAP=1.5 \
torchrun --standalone --nproc_per_node=64 run_ep.py -- \
  --module <your config module> --config <your config> \
  --training.seq_len 8192 --training.steps 8 \
  --debug.seed 42 \
  --dataloader.dataset c4_test --hf_assets_path ./tests/assets/tokenizer
```

- `FLINT_EP` = EP degree; `EP_GLOBAL_E` = **global** expert count (needed for §5; `0` disables the bound so
  non-EP paths are untouched); `EP_CAP` = capacity factor.
- **Always pass `--debug.seed`** (note: `--debug.seed`, not `--training.seed`). Without it the loss is
  **non-deterministic** — two identical runs gave 12.336 and 12.048 — which makes any A/B meaningless.
- Each `(seq, reshard)` shape costs a 20-40 min first compile on the 49-layer model. Budget for it, or
  persist `/var/tmp/neuron-compile-cache` between runs.
- **Never delete `/var/tmp/nki-intermediate-cache`.** It holds compiled NKI kernel binaries (`.colz`)
  referenced by path; wiping it poisons the host with `[NCC_EVRF059] Kernel file ... does not exist` on
  `aten::embedding_dense_backward`, and it does not self-heal. Clearing
  `/var/tmp/neuron-compile-cache` and `/var/tmp/nki-trace-cache-uid*` is safe — and **is required** between
  A/B arms, or a stale NKI trace cache silently serves the previous variant's graph.

## 12. Sliding-window attention: a correctness fix you need

This model uses `sliding_window = 513`. Public SDK 2.33 `nkilib` `attention_bwd` **silently computes wrong
dQ/dK/dV** for any `sliding_window % 128 != 0` (rel-L2 0.12-0.16 against an eager reference; the forward is
unaffected). The window mask tested only the first query row of each 128-row tile.

Fixed upstream on nkilib mainline (`5f268bedd`), which tests the last row instead — but **no public release
carries it yet**. Until one does, vendor `core/attention/attention_bwd.py` with that one-line change. The copy
here differs from nkilib core only by that fix, a provenance docstring, and relative→absolute imports.

Anyone training with a non-128-multiple sliding window on public 2.33 is affected.
