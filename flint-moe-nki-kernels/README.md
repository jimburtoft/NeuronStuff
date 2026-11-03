# MoE blockwise-matmul kernels for Trainium2 — routing and backward weight pre-transpose

Three NKI pieces for training a fine-grained Mixture-of-Experts layer (many small experts, top-k routing) on AWS
Trainium2 with the blockwise-matmul MoE kernels in the AWS Neuron NKI library (`nkilib`,
[aws-neuron/nki-samples](https://github.com/aws-neuron/nki-samples) family):

1. **`pack_scatter_positions`** — single-rank, on-device MoE **dispatch**. Given each `(token, expert)` slot's expert
   id and the per-expert counts, it returns each slot's row in a block-padded per-expert packed buffer, in one pass.
2. **`scatter_src_to_packed`** — the inverse map (`token_position_to_id`): packed slot → source token, so the packed
   buffer can be built (and the multi-expert `bwmm_shard_on_H` kernel driven) without `find_nonzero_indices` +
   `indexed_flatten` and their `(T // f_len) % 16` alignment constraint.
3. **`transpose_gate_up` / `transpose_down`** — pre-transpose each expert's weights **once** into the layout the
   blockwise MoE **backward** builds per token block, so the backward can plain-load them instead of re-running
   `nc_transpose` for every block of every expert.

> Everything below is a **hardware measurement on trn2.3xlarge (SDK 2.33, LNC=2)** or an exact
> **CPU-simulator** correctness check (`nki.simulate`). Nothing is estimated. Where a change did not help, that is
> stated plainly.

## Why

The blockwise-matmul MoE backward (`blockwise_mm_bwd` / `bwmm_bwd_dropless`) is DMA-bound on trn2: at a representative
expert-parallel shape (16,384 gathered rows, 16 local experts, block size 256) it was **95% DMA, ~13.7% MFU**. A
line-level profile attributed the device time to two redundancies, both because an expert's blocks share its weights:

| source lines | work | cost |
|---|---|---|
| gate/up `nc_transpose` + `tensor_copy`, down `nc_transpose` | re-transpose each expert's weights **per block** | ~67 ms |
| `dma_compute(reduce_op=add)` into the fp32 weight grad | read-modify-write the weight grad in HBM **per block** | ~36 ms |

The dispatch index math in PyTorch (`scatter` ~187 ms, `searchsorted` ~1071 ms, `topk` capped at 16,384 elements,
long `cumsum`) was separately a bottleneck when feeding the single-expert packed kernel.

## What's here

```
kernels/pack_routing.py       pack_scatter_positions, scatter_src_to_packed   (LNC=2 routing scan)
kernels/weight_transpose.py   transpose_gate_up, transpose_down               (per-expert pre-transpose)
kernels/helpers.py            kernel_assert / div_ceil / ve_transpose_broadcast (vendored nkilib utils, for standalone use)
sim_pack_routing.py           CPU-simulator correctness check for the routing kernels
sim_weight_transpose.py       CPU-simulator correctness check for the pre-transpose kernels
```

`pack_scatter_positions` is adapted from the upstream `build_packed_routing` all-to-all routing scan
(`tensor_tensor_scan` membership prefix), made single-rank and **block-padded per expert** (expert `e`'s segment
starts at `B * sum_{j<e} ceil(count_j / B)`), so each expert begins on a block boundary.

## Correctness (CPU simulator, exact)

```
$ pip install --index-url https://pip.repos.neuron.amazonaws.com nki
$ pip install numpy ml_dtypes
$ NEURON_LOGICAL_NC_CONFIG=2 python sim_pack_routing.py
pack_scatter_positions N=256 E=8 B=128 cap=1024: exact=True
scatter_src_to_packed  N=256 E=8 B=128: exact=True
...  (N up to 1024, E up to 64, B 128/256 — all exact)
$ NEURON_LOGICAL_NC_CONFIG=2 python sim_weight_transpose.py
transpose_gate_up E=16 H=2048 I=512: exact=True shape=(16, 2, 512, 2048)
transpose_down    E=16 H=2048 I=512: exact=True shape=(16, 2048, 512)
```

## Measured on trn2.3xlarge (SDK 2.33, LNC=2)

**Routing kernel.** `pack_scatter_positions` for 98,304 `(token, expert)` slots: **2.4 ms**, versus the PyTorch
dispatch index math it replaces (tens to >1000 ms for the individual ops, dominated by `scatter`/`searchsorted`).

**Pre-transpose.** `transpose_gate_up` + `transpose_down` for all 16 experts (H=2048, I=512, bf16): **4.3 ms**, versus
the ~67 ms of per-block transposes it removes from the backward.

**Backward with the gate/up pre-transpose** (16,384 rows × 16 experts, block 256, backward kernel only):

| | stock | with gate/up pre-transpose (incl. the 4.3 ms pre-transpose) |
|---|---|---|
| backward fwd+bwd | 53.5 ms | **45.5 ms (1.18×)** |
| gradient rel-L2 vs stock (dx, dgate_up, ddown) | — | **0.000000 (bit-identical)** |

### Honest notes

- **The 1.18× is less than a naive "remove 67 ms" estimate.** Under the transpose are matmuls that were always there,
  and the pre-transposed weight is read with an I-strided HBM pattern that adds traffic (6.8 → 12.8 GB read). Removing
  the transpose exposes the matmuls rather than shrinking the whole kernel.
- **The larger remaining win is the per-block weight-grad read-modify-write** (~36 ms). Removing it needs a per-expert
  accumulation that writes each expert's gradient to HBM once (on the expert's last block) instead of per block. The
  per-expert block count is data-dependent (routing), so the "last block" test must be a runtime
  `expert_idx[b] != expert_idx[b+1]`, and the write is skipped on non-last blocks via the kernel's own
  `oob_mode=skip`. That restructure is in progress and not included here.
- **Routing in NKI did not speed up Flint's MoE end to end**, because Flint uses *dense* top-k routing: every row hits
  exactly `k` experts, so the tight per-expert block budget equals the stock `ceil(T*k/B)+E` budget — no padding to
  remove. The routing kernels are published as a reusable primitive; they help most for *sparse/ragged* routing, or to
  drive `bwmm_shard_on_H` where `indexed_flatten`'s 16-alignment constraint is a problem.

## Integrating the pre-transpose into the backward

The backward edit itself is a small change to the nkilib `bwmm_bwd_dropless.py` (not shipped here, as it tracks a
specific SDK version). In `_compute_hidden_states_grad`, add an optional `gate_up_proj_weight_T` ([E, 2, I_TP, H]); when
present, plain-load each `i_tile` straight into `rhs_tiles` and skip the per-block `nc_transpose` + `tensor_copy`. Pass
the pre-transposed weight (from `transpose_gate_up`) through the kernel; call it once per step. Keep all enums/params
imported from the installed `nkilib` (comparing a re-defined `AffinityOption` against the library's silently selects the
wrong branch).

## Status

Correctness: validated (CPU simulator exact; backward gradients bit-identical on device). Performance: routing 2.4 ms,
pre-transpose 4.3 ms, backward 1.18× (bit-exact). These are kernel-level results; whether they move a full training
step depends on the model (for a model where the MoE is most of the step, combined with the weight-grad write-once
change). Hardware only; no estimates.
