"""Real-valued IIR prefill recurrence for evo2_7b_base.

WHY: upstream `prefill_via_direct_recurrence` assumes a COMPLEX modal parameterization --
it indexes `poles[:, :, 0]` then `poles[..., 0]`/`poles[..., 1]` as (re, im), i.e. it
expects poles shaped (..., sdim, 2) from `torch.view_as_real`. See engine.py:492 and the
`load_poles_residues` path (model.py:857) which explicitly calls `torch.view_as_real`.

But the shipped evo2_7b_base checkpoint stores REAL parameters:
    filter.log_poles  (4096, 16, 1)      <- trailing dim is a dummy seqlen, not (re,im)
    filter.residues   (4096, 16)
    filter.D          (4096,)
Hence `residues[:, :, 0]` -> IndexError on a 2-D tensor.

WHAT THIS COMPUTES: the same modal recurrence as the model's own DECODE step
(`engine.step_iir`, verified on-device at cos_sim 1.000000):

    poles_e = exp(log_poles)                       # (D, sdim)
    state_t = poles_e * state_{t-1} + x1v_t        # (B, D, sdim)
    y_t     = sum_sdim(residues * state_t)         # (B, D)

and writes the final `state` into inference_params.state_dict so decode continues
seamlessly from prefill. This is the real-valued specialization of upstream's complex
recurrence -- for real poles the imaginary track is identically zero, so the two agree.

Validated against a CPU fp32 reference of the same model (see parity output).
"""
import os
import torch

_CHUNK = int(os.environ.get("EVO2_IIR_PREFILL_CHUNK", "64"))   # 0 = always the per-step loop
# Measured (after building the Toeplitz operator once per call): warm TP=4 B=8 L=512 loop 255 ms
# vs chunked 278 ms (+9%); compile: loop 2 x ~380 s, chunked 2 x ~40 s; loop does not finish compiling
# at L>=1000. Default: loop up to 512, chunks above.
_LOOP_MAX_L = int(os.environ.get("EVO2_IIR_PREFILL_LOOP_MAX_L", "512"))

_orig = None

def install():
    global _orig
    from vortex.model.engine import HyenaInferenceEngine as H
    _orig = H.prefill_via_direct_recurrence

    def real_recurrence(self, inference_params, x1v, L, residues, poles, *a, **k):
        # x1v: (B, D, L) | poles: (D, sdim, 1) or (D, sdim) | residues: (D, sdim)
        p = poles
        if p.dim() == 3:
            p = p[..., 0]                       # drop dummy seqlen dim -> (D, sdim)
        r = residues
        if r.dim() == 3:
            r = r[..., 0]
        p = torch.exp(p.to(torch.float32)).unsqueeze(0)      # (1, D, sdim)
        r = r.to(torch.float32).unsqueeze(0)                 # (1, D, sdim)

        B, D, _ = x1v.shape
        sdim = p.shape[-1]
        xf = x1v.to(torch.float32)
        state = torch.zeros(B, D, sdim, dtype=torch.float32, device=x1v.device)
        T = _CHUNK
        if T > 0 and L > max(T, _LOOP_MAX_L):
            # CHUNKED closed form instead of an L-step Python loop. Traced inside the
            # compiled block, the loop unrolled into L steps per HCL layer, so prefill compile time
            # grew with L (L=512: two 6-min passes; L>=1000 did not finish in 47 min). Within a
            # chunk of T steps starting from state s0:
            #   s_t = p^(t+1) s0 + sum_{j<=t} p^(t-j) x_j ,   y_t = sum_sdim r * s_t
            # -> one (T x T) lower-triangular decay matmul per chunk; only ceil(L/T) chunk steps
            # are traced. Same math reassociated (CPU check: rel 1.7e-6 vs the loop at L=1000).
            logp = torch.log(p[0])                                    # (D, sdim)
            rr = r[0]                                                 # (D, sdim)
            # Build the chunk operators ONCE per call, not once per chunk. The decay
            # matrix is Toeplitz: W[d,i,j] = sum_s r p^(i-j) = k[d, i-j] for i >= j, so it needs
            # only the per-lag kernel k (D, T) -- D*S*T exps instead of D*S*T*T per chunk (the old
            # code: ~67M exps x 16 chunks x 9 layers at L=1000).
            lag = torch.arange(T, device=x1v.device, dtype=torch.float32)
            ppow = torch.exp(logp[..., None] * lag)                   # (D, sdim, T)  p^m, m=0..T-1
            k = (rr[..., None] * ppow).sum(1)                         # (D, T)  sum_s r p^m
            ii = torch.arange(T, device=x1v.device)
            dif = ii[:, None] - ii[None, :]                           # (T, T)
            Wfull = k[:, dif.clamp(min=0)] * (dif >= 0)               # (D, T, T) Toeplitz, lower-tri
            pw_full = ppow * p[0][..., None]                          # (D, sdim, T)  p^(t+1)
            outs = []
            for c0 in range(0, L, T):
                xc = xf[:, :, c0:c0 + T]
                Tc = xc.shape[-1]
                W = Wfull[:, :Tc, :Tc]
                pw = pw_full[..., :Tc]
                y_in = torch.einsum("dij,bdj->bdi", W, xc)
                y_s = torch.einsum("bds,dst->bdt", state * rr, pw)
                outs.append(y_in + y_s)
                # s_T = p^Tc s0 + sum_j p^(Tc-1-j) x_j
                wlast = ppow[..., :Tc].flip(-1)                       # (D, sdim, Tc): p^(Tc-1-j)
                state = pw[..., -1].unsqueeze(0) * state + torch.einsum("dsj,bdj->bds", wlast, xc)
            out = torch.cat(outs, dim=-1)                              # (B, D, L)
        else:
            outs = []
            for i in range(L):
                state = p * state + xf[:, :, i].unsqueeze(-1)
                outs.append((r * state).sum(dim=-1))
            out = torch.stack(outs, dim=-1)                      # (B, D, L)

        inference_params.state_dict[self.layer_idx] = state.to(torch.float32)
        return out.to(x1v.dtype)

    H.prefill_via_direct_recurrence = real_recurrence
    return _orig

def uninstall():
    if _orig is not None:
        from vortex.model.engine import HyenaInferenceEngine as H
        H.prefill_via_direct_recurrence = _orig
