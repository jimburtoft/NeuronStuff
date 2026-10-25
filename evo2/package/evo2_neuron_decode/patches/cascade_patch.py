"""Route HCL decode through the whole-cascade fused kernel (kernels/hcl_cascade_fused.py).

Replaces HyenaCascade.sequential_forward's ENTIRE span for HCL layers:
  step_fir(k=3) -> interleave -> split -> step_iir   (~19 dispatched ops -> 1 launch)

VALIDATED: simulator 5/5 and hardware all three outputs at cos_sim 1.00000000, fallback
NONE, registered as torch.ops.evo2.hcl_cascade.
ISOLATED: eager full chain 0.963 ms -> cascade 0.189 ms = 5.09x (by min).
Predicted x9 layers: 8.67 -> 1.70 ms, saving ~7.0 ms -- which CLEARS this harness's
+/-8 ms noise floor only marginally, so the e2e A/B must use tight/interleaved sampling.

LAYOUT CONTRACT (the reason interleave costs zero instructions):
  u is (B, 3D) with channel d's three interleave lanes CONTIGUOUS, so u.reshape(D,3) gives
  u[d, lane]. Reshaped to (P, G, 3) the lanes are the innermost axis, and upstream's
  interleave+split (0::3 / 1::3 / 2::3 then cat then split) is exactly the three
  free-dimension slices [...,0], [...,1], [...,2]. No gather, no cat.

FROZEN-WEIGHT PREP is hoisted and cached on the module: the expanded short filter weights,
the bias (zeros when the model has none), exp(log_poles), and residues. Cached on the
MODULE, never on id() of a per-call temporary (learned from an earlier correctness defect).
"""
import torch

_orig = None
_KERN = None
import os as _os
# HCL batch fold: default ON. TP=4 @ 8K, gated, 3 alternating pairs: B=32 38.78 -> 37.57 ms (1.03x),
# B=48 47.34 -> 43.72 (1.08x, 1014 -> 1098 tok/s), B=64 68.42 -> 49.42 (1.38x, one pair).
# Kernel-level: per-row vs folded outputs BIT-IDENTICAL on device (eager and compiled), and the
# full HCL BLOCK (all 4 TP ranks, 3 layers, 3 steps) bit-identical too. End to end, 48-step tokens
# are identical at B=32/48/64 on the short prompt, but the 64-nt prompt diverges at step 35 and
# late-step logits differ by ~1e-6 relative: the change of graph shape around the block alters
# how the compiler schedules the neighbouring ops. EVO2_HCL_FOLD=0 restores the per-row loop.
_FOLD = _os.environ.get("EVO2_HCL_FOLD", "1") == "1"
# DEFAULT ON. Flat batch fold (free views, no permute/contiguous in or out).
# TP=4 @ 8K, gated, alternating: B=48 41.17 -> 38.48 ms (1.07x, 2 pairs), B=64 47.01 -> 43.41 (1.08x,
# 2 pairs), B=64 bucketed 34.76 -> 32.02 (1.09x). Kernel outputs bitwise equal flat vs permute in the
# NKI simulator; CPU wiring test (tests/test_flat_fold.py) bitwise, mutation-checked. End to end:
# step-1 logits identical; 64-step tokens identical on the short prompt at B=48/64, differ on the
# 64-nt prompt (graph-scheduling change, same class as the HCL batch fold). EVO2_FLAT_FOLD=0 restores.
_FLAT = _os.environ.get("EVO2_FLAT_FOLD", "1") == "1"


def install():
    global _orig, _KERN
    from torch_neuronx import wrap_nki
    from ..kernels.hcl_cascade_fused import hcl_cascade_fused
    from vortex.model.model import HyenaCascade

    _KERN = wrap_nki(hcl_cascade_fused)
    _orig = HyenaCascade.sequential_forward

    def sequential_forward(self, u, inference_params):
        # Only HCL layers (no inner FIR) with the layouts this kernel assumes.
        if (self.fir_inner_filter_length is not None
                or getattr(self, "column_split_hyena", False)
                or getattr(self, "hyena_flip_x1x2", False)
                or not getattr(self.config, "interleave", False)
                or getattr(self, "log_poles", None) is None):
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
        S = self.state_size if hasattr(self, "state_size") else self.log_poles.shape[1]

        fir = inference_params.fir_state_dict[self.layer_idx]      # (B, 3D, 2)
        iir = inference_params.state_dict[self.layer_idx]          # (B, D, S)
        if fir.shape[-1] != 2:
            return _orig(self, u, inference_params)

        # ---- hoisted frozen-weight prep, cached on the module ----
        c = getattr(self, "_casc", None)
        if c is None:
            dev = u.device
            sw = self.short_filter_weight.squeeze()                # (3D, 3)
            sw3 = sw.contiguous().float().reshape(D, 3, 3).reshape(P, G, 3, 3).contiguous()
            sb = self.short_filter_bias
            sb_t = (torch.zeros(3 * D, device=dev) if sb is None
                    else sb.squeeze().contiguous().float())
            sb3 = sb_t.reshape(D, 3).reshape(P, G, 3).contiguous()
            pe = torch.exp(self.log_poles)[..., 0].contiguous().float()   # (D, S)
            pr3 = torch.cat([pe, self.residues.contiguous().float()], 1
                            ).reshape(P, G, 2 * S).contiguous()
            dp3 = self.D.contiguous().float().reshape(P, G, 1).contiguous()
            # REVERTED experiment: see hcm_patch. bf16 frozen weights save ZERO DMA bytes
            # (measured: bf16 and fp32 hcm NEFFs both move exactly 13.369 MB), because
            # nisa.dma_copy transfers at the fp32 DESTINATION width. Keep fp32.
            c = (sw3, sb3, pr3, dp3)
            self._casc = c
        sw3, sb3, pr3, dp3 = c

        # Fold the batch into G and issue ONE launch (the kernel is G-tiled), exactly
        # as hcm_patch does. The per-row loop below was B launches + 6B small
        # copy/stack ops per HCL layer; a profile showed the HCL block graph with
        # 951 kernel + 1,796 glue instructions and a 224 us DMA-efficiency gap.
        if _FOLD and B > 1:
            GB = B * G
            wk = "_casc_b%d" % B
            wc = getattr(self, wk, None)
            if wc is None:
                wc = tuple(t.unsqueeze(1).expand(P, B, *t.shape[1:]).reshape(P, GB, *t.shape[2:]).contiguous()
                           for t in (sw3, sb3, pr3, dp3))
                setattr(self, wk, wc)
            swB, sbB, prB, dpB = wc
            if _FLAT:
                # Flat fold (see hcm_patch): free views in and out, weights tiled to match.
                wk2 = "_cascf_b%d" % B
                wf = getattr(self, wk2, None)
                if wf is None:
                    wf = _tile_flat((sw3, sb3, pr3, dp3), D, B, P, GB)
                    setattr(self, wk2, wf)
                    for k in [k for k in vars(self) if k.startswith("_casc_b")]:
                        delattr(self, k)          # free the permute-fold weight copies
                swB, sbB, prB, dpB = wf
                y_b, nf_b, ns_b = _KERN(u.reshape(P, GB, 3), fir.reshape(P, GB, 3, 2), swB, sbB,
                                        iir.reshape(P, GB, S), prB, dpB)
                y = y_b.reshape(B, D)
                nf = nf_b.reshape(B, 3 * D, 2)
                ns = ns_b.reshape(B, D, S)
            else:
                u3 = u.reshape(B, D, 3).reshape(B, P, G, 3).permute(1, 0, 2, 3).reshape(P, GB, 3).contiguous()
                fs3 = fir.reshape(B, D, 3, 2).reshape(B, P, G, 3, 2).permute(1, 0, 2, 3, 4).reshape(P, GB, 3, 2).contiguous()
                st3 = iir.reshape(B, P, G, S).permute(1, 0, 2, 3).reshape(P, GB, S).contiguous()
                y_b, nf_b, ns_b = _KERN(u3, fs3, swB, sbB, st3, prB, dpB)
                y = y_b.reshape(P, B, G).permute(1, 0, 2).reshape(B, D)
                nf = nf_b.reshape(P, B, G, 3, 2).permute(1, 0, 2, 3, 4).reshape(B, 3 * D, 2)
                ns = ns_b.reshape(P, B, G, S).permute(1, 0, 2, 3).reshape(B, D, S)
            inference_params.fir_state_dict[self.layer_idx] = nf if nf.dtype == fir.dtype else nf.to(fir.dtype)
            inference_params.state_dict[self.layer_idx] = ns
            return (y if y.dtype == self.data_dtype else y.to(self.data_dtype))[:, None], inference_params

        ys, nfs, nst = [], [], []
        for b in range(B):
            # No per-call .float() -- see hcm_patch. dma_copy upcasts to fp32
            # SBUF for free; kernel HBM outputs follow u3.dtype. Each removed cast is a
            # ~283 us device dispatch.
            u3 = u[b].contiguous().reshape(D, 3).reshape(P, G, 3)
            fs3 = fir[b].contiguous().reshape(D, 3, 2).reshape(P, G, 3, 2)
            st3 = iir[b].contiguous().reshape(P, G, S)
            y_b, nf_b, ns_b = _KERN(u3, fs3, sw3, sb3, st3, pr3, dp3)
            ys.append(y_b.reshape(D))
            nfs.append(nf_b.reshape(D, 3, 2).reshape(3 * D, 2))
            nst.append(ns_b.reshape(D, S))
        y = torch.stack(ys, 0)
        nf = torch.stack(nfs, 0)
        inference_params.fir_state_dict[self.layer_idx] = nf if nf.dtype == fir.dtype else nf.to(fir.dtype)
        inference_params.state_dict[self.layer_idx] = torch.stack(nst, 0)
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
