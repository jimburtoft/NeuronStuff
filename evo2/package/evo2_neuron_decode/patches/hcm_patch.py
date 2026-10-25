"""Route HCM/HCS decode through the whole-cascade fused kernel (kernels/hcm_cascade_fused.py).

Covers 18 of the 32 layers (9 HCM k=128 + 9 HCS k=7). Replaces the entire
sequential_forward span: step_fir(k=3) -> interleave -> split -> inner step_fir -> *x2,
~25 dispatched ops -> 1 launch.

VALIDATED: simulator 11/11 (both k=128 and k=7 paths), hardware all three outputs
cos_sim 1.00000000, fallback NONE, torch.ops.evo2.hcm_cascade registered.
ISOLATED: eager full chain 2.7928 ms -> cascade 0.2165 ms = 12.90x (by min).
Predicted x9 HCM layers: 25.14 -> 1.95 ms = ~23 ms saved, well above the +/-8 ms floor.

The k=128 weight prep (repeat_interleave + flip + slice + upcast; measured at
2.412 ms/call, 72% of step_fir) is hoisted ONCE per module and cached, so the kernel
receives h0/h ready to use. Cached on the MODULE, never on id() of a per-call temporary
(learned from an earlier correctness defect).

Lane order: upstream unpacks `x2, x1, v = split(...)`, so lane0 -> x2.
"""
import os

import torch

_orig = None
_KERN = None
# fold batch into the G axis only when B >= this (below it, per-row is faster -- measured)
_FOLD_MIN = 8
# max folded free-axis width per launch before SBUF overflows (measured: 1024 fails at
# max_seqlen=512, 512 is the largest that fits). Above this, fall back to the per-row path.
#
# Env-overridable so the batch-ceiling boundary can be tested without a code edit.
# The ceiling (112 works / 120 fails) coincides EXACTLY with nch crossing 7->8 at 512, and
# nch = ceil(B*32 / _MAX_FOLD_GB) -- so if this knob moves the ceiling, chunking is causal.
# NOTE the 1024-fails-at-ms=512 measurement above: an over-large value trades one failure
# mode (accumulated chunk copies) for another (SBUF overflow), so both arms must be checked
# against the ACTUAL error text, not just pass/fail.
# Default 512 -> 1024. With the flat fold + G-tiled kernel (each kernel launch
# tiles G internally in 32-column tiles) SBUF per launch no longer scales with GB, so the 512 cap
# only adds launches. At TP=4 (G=8/core) it split B>64 into 2 launches per HCM/HCS layer x 18 layers:
# B=72 @ fill 1024 41.2 -> 34.5 ms (1.20x, removes the B=72 cliff), B=80 38.7 -> 35.6, B=88
# 41.3 -> 39.1, B=96 42.6 -> 39.8. Tokens identical (same TOKSHA). Single core B=56/128 @ 256 and
# B=4 @ 8192 unchanged (both widths compile). EVO2_MAX_FOLD_GB=512 restores the old cap.
_MAX_FOLD_GB = int(os.environ.get("EVO2_MAX_FOLD_GB", "1024"))
# DEFAULT ON. Flat batch fold (free views, no permute/contiguous in or out).
# TP=4 @ 8K, gated, alternating: B=48 41.17 -> 38.48 ms (1.07x, 2 pairs), B=64 47.01 -> 43.41 (1.08x,
# 2 pairs), B=64 bucketed 34.76 -> 32.02 (1.09x). Kernel outputs bitwise equal flat vs permute in the
# NKI simulator; CPU wiring test (tests/test_flat_fold.py) bitwise, mutation-checked. End to end:
# step-1 logits identical; 64-step tokens identical on the short prompt at B=48/64, differ on the
# 64-nt prompt (graph-scheduling change, same class as the HCL batch fold). EVO2_FLAT_FOLD=0 restores.
_FLAT = os.environ.get("EVO2_FLAT_FOLD", "1") == "1"
_SHORT_PAD = os.environ.get("EVO2_HCM_SHORT_PAD", "1") == "1"   # short-prompt HCM state fix


def _perrow(self, u, inference_params, fir, inner, B, D, P, G, C,
            sw3, sb3, ih3, ih0, dp3, gated):
    """Original per-row path, kept for small B where the fold's permute cost dominates."""
    import torch
    ys, nfs, nis = [], [], []
    for b in range(B):
        u3 = u[b].contiguous().reshape(D, 3).reshape(P, G, 3)
        fs3 = fir[b].contiguous().reshape(D, 3, 2).reshape(P, G, 3, 2)
        is3 = inner[b].contiguous().reshape(P, G, C)
        y_b, nf_b, ni_b = _KERN(u3, fs3, sw3, sb3, is3, ih3, ih0, dp3, gated)
        ys.append(y_b.reshape(D))
        nfs.append(nf_b.reshape(D, 3, 2).reshape(3 * D, 2))
        nis.append(ni_b.reshape(D, C))
    y = torch.stack(ys, 0); nf = torch.stack(nfs, 0); ni = torch.stack(nis, 0)
    inference_params.fir_state_dict[self.layer_idx] = nf if nf.dtype == fir.dtype else nf.to(fir.dtype)
    inference_params.fir_inner_state_dict[self.layer_idx] = ni if ni.dtype == inner.dtype else ni.to(inner.dtype)
    return (y if y.dtype == self.data_dtype else y.to(self.data_dtype))[:, None], inference_params


def install():
    global _orig, _KERN
    from torch_neuronx import wrap_nki
    from ..kernels.hcm_cascade_fused import hcm_cascade_fused
    from vortex.model.model import HyenaCascade

    _KERN = wrap_nki(hcm_cascade_fused)
    _orig = HyenaCascade.sequential_forward

    def sequential_forward(self, u, inference_params):
        k = getattr(self, "fir_inner_filter_length", None)
        if (k is None
                or getattr(self, "column_split_hyena", False)
                or getattr(self, "hyena_flip_x1x2", False)
                or not getattr(self.config, "interleave", False)):
            return _orig(self, u, inference_params)

        if self.data_dtype is None:
            self.data_dtype = u.dtype
        if len(u.shape) > 2:
            u = u[:, -1]

        B, TD = u.shape
        D = self.hidden_size
        if TD != 3 * D or D % 128 != 0:
            return _orig(self, u, inference_params)
        P = 128
        G = D // P

        fir = inference_params.fir_state_dict[self.layer_idx]        # (B, 3D, 2)
        inner = inference_params.fir_inner_state_dict[self.layer_idx]  # (B, D, C)
        if fir.shape[-1] != 2:
            return _orig(self, u, inference_params)
        k_full = getattr(self, "fir_inner_filter_length", None)
        if _SHORT_PAD and k_full is not None and inner.shape[-1] < k_full - 1:
            # SHORT-PROMPT FIX. After a prefill of L < k-1 tokens the inner-FIR state holds
            # only L-1 values. Upstream step_fir GROWS it by one per decode step until k-1, using the
            # matching first (cache_size+1) filter taps. Our kernel keeps C fixed, so it decoded with a
            # permanently truncated filter: 2-5% relative divergence from upstream within 50 steps at
            # L=8/40/100 (exact for L >= 127). Left-pad with zeros to k-1: a zero history contributes
            # nothing to the convolution, and the newest value stays at the right end, which is what
            # upstream converges to once its state reaches k-1. Also gives every short prompt the same
            # decode shape (C = k-1).
            inner = torch.nn.functional.pad(inner, (k_full - 1 - inner.shape[-1], 0))
            inference_params.fir_inner_state_dict[self.layer_idx] = inner
        C = inner.shape[-1]

        # ---- hoisted frozen-weight prep, cached on the module ----
        # Key the hoisted weights on C. The inner FIR state length is
        # min(prefill_len, k) - 1, so prompts shorter than k (128 for HCM) give a different C
        # (L=64 -> C=63, L>=128 -> C=127). A single "_hcmc" cache made a second prompt length in
        # the same process fail the kernel's "ih3 must match is3's cache size" check.
        _ck = "_hcmc_C%d" % C
        c = getattr(self, _ck, None)
        if c is None:
            dev = u.device
            sw = self.short_filter_weight.squeeze()                  # (3D, 3)
            sw3 = sw.contiguous().float().reshape(D, 3, 3).reshape(P, G, 3, 3).contiguous()
            sb = self.short_filter_bias
            sb_t = (torch.zeros(3 * D, device=dev) if sb is None
                    else sb.squeeze().contiguous().float())
            sb3 = sb_t.reshape(D, 3).reshape(P, G, 3).contiguous()
            # inner filter: expand groups, then flip/slice exactly as step_fir does
            h = self.h
            if getattr(self, "hyena_filter_groups", 1) > 1:
                h = h.repeat_interleave(D // self.hyena_filter_groups, 0)
            hw = h.squeeze().float()                                  # (D, k)
            if k >= 128:
                hw = hw.flip(-1)[..., -(C + 1):]
            else:
                hw = hw[..., : C + 1]
            ih0 = hw[..., -1].contiguous().reshape(P, G, 1).contiguous()
            ih3 = hw[..., :-1].contiguous().reshape(P, G, C).contiguous()
            # self.D exists ONLY for k>=128 (model.py:196); it is None for k<128
            # (model.py:199), where upstream passes bias=None and step_fir SKIPS the bias
            # term entirely. Zeros reproduce that exactly for the plain-bias path.
            if self.D is None:
                dp3 = torch.zeros(P, G, 1, device=dev, dtype=torch.float32)
            else:
                dp3 = self.D.contiguous().float().reshape(P, G, 1).contiguous()
            gated = 1 if k >= 128 else 0
            # REVERTED experiment: storing these bf16 saves ZERO DMA bytes. Measured on
            # hardware -- the bf16 and fp32 hcm NEFFs both move exactly 13.369 MB.
            # nisa.dma_copy into an fp32 SBUF dst transfers at the DESTINATION width, so a
            # bf16 HBM source does not halve wire traffic. The earlier bf16 check only ever proved the
            # upcast is NUMERICALLY exact, never that it is bandwidth-cheaper. Keep fp32.
            c = (sw3, sb3, ih3, ih0, dp3, gated)
            setattr(self, _ck, c)
        sw3, sb3, ih3, ih0, dp3, gated = c

        # ONE kernel call for the WHOLE batch, not one per row.
        # The kernel derives G from u3.shape and every op is elementwise over (P,G), so it
        # is already generic in G -- batching needs NO kernel change, only this layout.
        # Fold the batch into the G (free) axis: G_batched = B*G.
        #   per-row tensors : (B, D, ...) -> (P, B*G, ...)
        #   frozen weights  : (P, G, ...) -> expand to (P, B*G, ...) once, cached
        # Was: 27 layers x B launches + 3-4 view/select each (864 launches at BS=32).
        # Now: 27 launches regardless of B.
        #
        # Still applies: no .float() on runtime tensors -- dma_copy upcasts to
        # fp32 SBUF for free and the HBM outputs follow u3.dtype.
        # Follow-up: the batch fold costs permute+contiguous on 3 runtime tensors,
        # which only amortizes once B is large. Measured: BS=32 163.9 -> 143.1 ms (1.15x WIN)
        # but BS=4 69.7 -> 76.3 ms (0.91x LOSS). Gate on B so small batches keep the per-row
        # path that was already faster for them.
        # The fold is bounded by SBUF, not just by B.
        # A full B*G fold at B=32 (GB=1024) overflows SBUF when the model is built at
        # max_seqlen=512:
        #   [NCC_INLA001] Allocated memory out of bound {_mem_3_4}@SB<0,0>(128x262144)
        #                 DebugInfo: <_mem_3_4||[128, 1024, 64]>
        # It compiles at max_seqlen=256, which is why the original BS=32 measurements passed --
        # they were taken at 256. Chunk the folded axis so each launch fits, exactly as
        # cascade_patch does for HCL. Still O(chunks) launches instead of O(B).
        # CHUNK the fold instead of abandoning it. The earlier SBUF guard fell back to the
        # FULL per-row path once GB > _MAX_FOLD_GB, which at B=32 means 32 launches/layer x 18 HCM
        # layers = 576 launches when a chunked fold needs 2/layer. cascade_patch already chunks for
        # HCL; hcm_patch fell back instead, and that is why BS=32 regressed 97.5 -> ~102.8 ms.
        GB = B * G
        if B >= _FOLD_MIN:
            pass
        elif B > 1:
            return _perrow(self, u, inference_params, fir, inner, B, D, P, G, C,
                           sw3, sb3, ih3, ih0, dp3, gated)
        if B == 1:
            swB, sbB, ihB, ih0B, dpB = sw3, sb3, ih3, ih0, dp3
        elif _FLAT:
            pass   # flat fold builds its own tiled weights below; skip the permute-fold copy
        else:
            wk = "_hcmc_b%d_C%d" % (B, C)
            wc = getattr(self, wk, None)
            if wc is None:
                # (P,G,...) -> (P,B*G,...): tile along G so row b occupies [b*G:(b+1)*G]
                wc = tuple(
                    t.unsqueeze(1).expand(P, B, *t.shape[1:]).reshape(P, GB, *t.shape[2:]).contiguous()
                    for t in (sw3, sb3, ih3, ih0, dp3))
                setattr(self, wk, wc)
            swB, sbB, ihB, ih0B, dpB = wc
        if _FLAT and B > 1:
            # FLAT fold. Every kernel op is per-channel, so ANY consistent channel ->
            # (partition, column) map is valid if the per-channel weights use the same map. The
            # flat map (B*D channels, row-major, -> (P, B*G)) is a free view of the contiguous
            # inputs: no permute, no .contiguous() copy in or out (a profile
            # attributed 34-72 GB/s small DMAs to these lines and ~8-35 MB/graph of spills to
            # the permute reshapes). Weights are tiled to match once and cached.
            wk2 = "_hcmf_b%d_C%d" % (B, C)
            wf = getattr(self, wk2, None)
            if wf is None:
                wf = _tile_flat((sw3, sb3, ih3, ih0, dp3), D, B, P, GB)
                setattr(self, wk2, wf)
            swB, sbB, ihB, ih0B, dpB = wf
            u3 = u.reshape(P, GB, 3)
            fs3 = fir.reshape(P, GB, 3, 2)
            is3 = inner.reshape(P, GB, C)
        else:
            # (B, 3D) -> (B, D, 3) -> (B, P, G, 3) -> (P, B*G, 3)
            u3 = u.reshape(B, D, 3).reshape(B, P, G, 3).permute(1, 0, 2, 3).reshape(P, GB, 3).contiguous()
            fs3 = fir.reshape(B, D, 3, 2).reshape(B, P, G, 3, 2).permute(1, 0, 2, 3, 4).reshape(P, GB, 3, 2).contiguous()
            is3 = inner.reshape(B, P, G, C).permute(1, 0, 2, 3).reshape(P, GB, C).contiguous()
        if GB <= _MAX_FOLD_GB:
            y_b, nf_b, ni_b = _KERN(u3, fs3, swB, sbB, is3, ihB, ih0B, dpB, gated)
        else:
            # Chunk the folded free axis so each launch fits SBUF. ceil-divide into equal parts,
            # each a whole number of rows (chunk % G == 0) so the un-fold reshape stays valid.
            nch = (GB + _MAX_FOLD_GB - 1) // _MAX_FOLD_GB
            rows_per = max(1, (B + nch - 1) // nch)
            yo, nfo, nio = [], [], []
            for a in range(0, B, rows_per):
                z = min(a + rows_per, B)
                lo, hi = a * G, z * G
                yc, nfc, nic = _KERN(u3[:, lo:hi].contiguous(), fs3[:, lo:hi].contiguous(),
                                     swB[:, lo:hi].contiguous(), sbB[:, lo:hi].contiguous(),
                                     is3[:, lo:hi].contiguous(), ihB[:, lo:hi].contiguous(),
                                     ih0B[:, lo:hi].contiguous(), dpB[:, lo:hi].contiguous(),
                                     gated)
                yo.append(yc); nfo.append(nfc); nio.append(nic)
            y_b = torch.cat(yo, 1); nf_b = torch.cat(nfo, 1); ni_b = torch.cat(nio, 1)
        if _FLAT and B > 1:
            y = y_b.reshape(B, D)
            nf = nf_b.reshape(B, 3 * D, 2)
            ni = ni_b.reshape(B, D, C)
        else:
            y = y_b.reshape(P, B, G).permute(1, 0, 2).reshape(B, D)
            nf = nf_b.reshape(P, B, G, 3, 2).permute(1, 0, 2, 3, 4).reshape(B, 3 * D, 2)
            ni = ni_b.reshape(P, B, G, C).permute(1, 0, 2, 3).reshape(B, D, C)
        # outputs already come back in u's dtype -> .to() is a no-op and emits no dispatch
        inference_params.fir_state_dict[self.layer_idx] = nf if nf.dtype == fir.dtype else nf.to(fir.dtype)
        inference_params.fir_inner_state_dict[self.layer_idx] = ni if ni.dtype == inner.dtype else ni.to(inner.dtype)
        return (y if y.dtype == self.data_dtype else y.to(self.data_dtype))[:, None], inference_params

    HyenaCascade.sequential_forward = sequential_forward


def uninstall():
    if _orig is not None:
        from vortex.model.model import HyenaCascade
        HyenaCascade.sequential_forward = _orig

def _tile_flat(ts, D, B, P, GB):
    """Build the flat-fold weight tiling OUTSIDE Dynamo, on the host. Traced inside the
    compiled block it lowered (at B=48) to a TensorEngine matmul with a negative-stride RHS and
    failed BIR verification (NCC_INLA001 'RHS AP cannot have negative stride'). Weights are
    frozen; this runs once per (module, B)."""
    import torch
    import torch._dynamo
    def _f(ts):
        out = []
        for t in ts:
            dev = t.device
            h = t.detach().to("cpu").reshape(D, *t.shape[2:])
            h = h.repeat(B, *([1] * (h.dim() - 1))).reshape(P, GB, *t.shape[2:]).contiguous()
            out.append(h.to(dev))
        return tuple(out)
    return torch._dynamo.disable(_f)(ts)
