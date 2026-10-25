"""Batched prefill for evo2_neuron_decode.

`prefill(model, input_ids, ...)` -> (inference_params_dict, last_logits)

Row chunking
    Prefilling all B rows in one call needs B x L activations at once and compiles a separate graph
    for every B. Rows are independent, so the batch is prefilled `chunk_rows` rows at a time and each
    chunk's per-layer state is written into the full-batch inference params. One prefill graph per
    (chunk_rows, L) regardless of B. Each chunk's temporary state is released before the next chunk,
    so the peak is ~ full-batch state + one chunk's state (16-row chunks with the old
    ad-hoc copy OOM'd at B=64; the old copy also kept every chunk's KV cache alive at max_seqlen).

Length buckets
    With `buckets=(256, 512, 1024, ...)`, a prompt of length L is right-padded with `pad_token` to the
    smallest bucket >= L, so only one prefill graph per bucket is ever compiled. Padding is EXACT for
    the returned logits and for the decode state:
      * all layers are causal, so outputs at positions < L do not see the padding; the logits are
        read at position L-1, not at the end;
      * the per-layer decode state is the state AT position L, not at the bucket end:
          - FIR states (short filter, HCM/HCS inner filter): the last k-1 inputs up to position L,
            sliced out of the layer input (captured with a forward pre-hook on the filter);
          - HCL IIR state: the recurrence state after step L (captured from the prefill recurrence);
          - attention KV cache: positions >= L are overwritten during decode and masked until then.
    Prompts in one call must all have the same length L (Evo 2 prompts in a batch are usually
    trimmed to the same length; ragged batches are not supported here).
"""
import torch

_STATE_DICTS = ("key_value_memory_dict", "fir_state_dict", "fir_inner_state_dict", "state_dict")


def _new_params(model, rows):
    ipd = model.initialize_inference_params()
    for n in ipd:
        ipd[n].max_batch_size = rows
    return ipd


def pick_bucket(L, buckets):
    for b in sorted(buckets):
        if b >= L:
            return b
    raise ValueError(f"prompt length {L} exceeds the largest bucket {max(buckets)}")


class _StateAtL:
    """Record each cascade layer's decode state at position L (not at the padded end)."""

    def __init__(self, model, L):
        self.L = L
        self.handles = []
        self.fir = {}        # layer_idx -> short-filter input u (B, 3D, Lb) as seen by parallel_fir
        self.inner = {}      # layer_idx -> inner-filter input (B, D, Lb) (after gating)
        self.iir = {}        # layer_idx -> IIR state at step L
        from vortex.model.engine import HyenaInferenceEngine
        self._orig_pfir = HyenaInferenceEngine.parallel_fir
        self._orig_rec = HyenaInferenceEngine.prefill_via_direct_recurrence
        st = self

        def pfir(eng, fir_fn, u, weight, bias, L_, dims, *a, **k):
            z, fir_state = st._orig_pfir(eng, fir_fn, u, weight, bias, L_, dims, *a, **k)
            if fir_state is not None:
                fl = k.get("fir_length", 3)
                gate = k.get("gate", False)
                dim_last = k.get("dim_last", True)
                # Recompute the state at position L from the same (post-gate) input the engine used.
                if gate:
                    hidden = dims[0]
                    if k.get("column_split_hyena", False):
                        return z, fir_state     # not used by evo2_7b_base; leave unchanged
                    x2, x1, v = u.split([hidden, hidden, hidden], dim=1)
                    if getattr(eng, "hyena_flip_x1x2", False):
                        x1, x2 = x2, x1
                    src = x1 * v                                     # (B, D, Lb)
                else:
                    src = u.permute(0, 2, 1) if dim_last else u      # (B, 3D, Lb)
                lo = max(0, st.L - (fl - 1))
                fir_state = src[..., lo:st.L]
            return z, fir_state

        def rec(eng, *a, **k):
            # vortex calls this with keywords: inference_params=, x1v=, L=, poles=, residues=
            names = ("inference_params", "x1v", "L", "residues", "poles")
            kw = dict(zip(names, a)); kw.update(k)
            L_ = kw["L"]
            kw["x1v"] = kw["x1v"][..., :st.L]; kw["L"] = st.L
            out = st._orig_rec(eng, **kw)
            # outputs beyond L are never used for the returned logits (causal), so zero-fill them
            if L_ > st.L:
                pad = torch.zeros(*out.shape[:-1], L_ - st.L, dtype=out.dtype, device=out.device)
                out = torch.cat([out, pad], dim=-1)
            return out

        self._pfir, self._rec = pfir, rec

    def __enter__(self):
        from vortex.model.engine import HyenaInferenceEngine
        HyenaInferenceEngine.parallel_fir = self._pfir
        HyenaInferenceEngine.prefill_via_direct_recurrence = self._rec
        return self

    def __exit__(self, *exc):
        from vortex.model.engine import HyenaInferenceEngine
        HyenaInferenceEngine.parallel_fir = self._orig_pfir
        HyenaInferenceEngine.prefill_via_direct_recurrence = self._orig_rec


def prefill(model, input_ids, *, chunk_rows=8, buckets=None, pad_token=None, max_batch_size=None):
    """Prefill `input_ids` (B, L) and return (inference_params_dict sized for B, logits (B, V) at L-1).

    The returned params have seqlen_offset = L, ready for decode."""
    B, L = input_ids.shape
    Bmax = max_batch_size or B
    Lp = L
    ids = input_ids
    if buckets:
        Lp = pick_bucket(L, buckets)
        if Lp > L:
            pt = pad_token if pad_token is not None else int(input_ids[0, -1])
            pad = torch.full((B, Lp - L), pt, dtype=input_ids.dtype, device=input_ids.device)
            ids = torch.cat([input_ids, pad], dim=1)
    full = _new_params(model, Bmax)
    lasts = []
    ctx = _StateAtL(model, L) if Lp > L else None
    for r0 in range(0, B, chunk_rows):
        rb = min(chunk_rows, B - r0)
        sub = _new_params(model, rb)
        with torch.no_grad():
            if ctx is not None:
                with ctx:
                    out, _ = model(ids[r0:r0 + rb], inference_params_dict=sub)
            else:
                out, _ = model(ids[r0:r0 + rb], inference_params_dict=sub)
        lasts.append(out[:, L - 1].clone())
        for n in full:
            for dn in _STATE_DICTS:
                src = getattr(sub[n], dn, None)
                dst = getattr(full[n], dn, None)
                if not src:
                    continue
                for li in list(src.keys()):
                    t = src[li]
                    if not torch.is_tensor(t) or t.numel() == 0:
                        continue
                    if li not in dst or not torch.is_tensor(dst[li]) or dst[li].shape[0] != Bmax \
                            or dst[li].shape[1:] != t.shape[1:]:
                        dst[li] = torch.zeros((Bmax,) + tuple(t.shape[1:]), dtype=t.dtype, device=t.device)
                    dst[li][r0:r0 + rb].copy_(t)
                    src[li] = None          # release this chunk's tensor before the next chunk
        del sub, out
    for n in full:
        full[n].seqlen_offset = L
    return full, torch.cat(lasts, 0)
