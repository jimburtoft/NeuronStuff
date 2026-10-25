"""Cross-layer fused HCL cascade: short FIR -> interleave -> split -> IIR, ONE kernel.

MOTIVATION: per-launch overhead (0.155 ms) is 6x this kernel's on-core work
(0.0249 ms). The v3 kernel fused step_iir (7 ops -> 1 launch). This fuses the WHOLE
HyenaCascade.sequential_forward span (~19 ops -> 1 launch):

    z_pre, fir_state = step_fir(u, fir_state, short_filter_weight, short_filter_bias)
    z_pre            = interleave(z_pre)          # stride-3 gather over 3D
    x2, x1, v        = z_pre.split([D, D, D], 1)
    y, iir_state     = step_iir(x2, x1, v, D_, residues, log_poles, iir_state)

THE KEY LAYOUT INSIGHT: `interleave` is
    x1 = z[..., 0::3];  x2 = z[..., 1::3];  v = z[..., 2::3];  z = cat([x1,x2,v], -1)
followed immediately by a 3-way split back into x1/x2/v. So interleave+split together are
just a stride-3 DEINTERLEAVE. If the short-FIR output is materialised in SBUF as
(P, G, 3) -- channel-major with the 3 interleaved lanes on the innermost axis -- then
x1/x2/v are the three FREE-DIMENSION SLICES [..., 0], [..., 1], [..., 2]. No gather, no
cat, no strided DMA: interleave becomes ZERO instructions.

That works because u is (B, 3D) laid out so that channel d's three lanes are contiguous
(that is exactly what the 0::3/1::3/2::3 slicing implies). We therefore view
    u          (3D,)      -> (P, G, 3)      P=128, G=D/128
    fir_state  (3D, 2)    -> (P, G, 3, 2)
    short_w    (3D, 3)    -> (P, G, 3, 3)   [pre-expanded by the caller]

MATH (identical to upstream, one timestep):
    # short FIR, per lane, cache_size = 2, filter_length = 3
    zp[c]        = h0[c]*u[c] + sum_j fir_state[c,j]*h[c,j]      (+ bias if present)
    fir_state'[c]= [fir_state[c,1], u[c]]                        (roll-by-1 + append)
    # deinterleave: x1 = zp[...,0], x2 = zp[...,1], v = zp[...,2]
    x1v[d]       = x1[d] * v[d]
    st'[d,s]     = poles_e[d,s]*st[d,s] + x1v[d]
    y[d]         = x2[d] * ( sum_s res[d,s]*st'[d,s] + Dp[d]*x1v[d] )

NOTE on the short filter: upstream `step_fir` with cache_size=2, filter_length=3 takes
weight[..., :cache_size+1] = the full 3 taps, so h0 = w[...,2] and h = w[...,0:2] --
i.e. NO flip (flip_filter is False for the short featurizer).
"""

import nki
import nki.isa as nisa
import nki.language as nl


def kernel_assert(condition, message=""):
    assert condition, message


@nki.jit
def hcl_cascade_fused(u3, fs3, sw3, sb3, st3, pr3, dp3):
    """Whole HCL cascade in one kernel, for a single batch row.

    Args:
        u3:  (P, G, 3)      fp32 - input, channel-major with the 3 interleave lanes inner
        fs3: (P, G, 3, 2)   fp32 - short-FIR state (cache_size 2)
        sw3: (P, G, 3, 3)   fp32 - short filter weights (3 taps), caller-expanded
        sb3: (P, G, 3)      fp32 - short filter bias (zeros if the model has none)
        st3: (P, G, S)      fp32 - IIR state
        pr3: (P, G, 2*S)    fp32 - packed [poles_e | residues], poles_e pre-exponentiated
        dp3: (P, G, 1)      fp32 - IIR skip term D

    Returns:
        y:      (P, G, 1)    fp32 - cascade output
        new_fs: (P, G, 3, 2) fp32 - updated short-FIR state
        new_st: (P, G, S)    fp32 - updated IIR state

    Notes:
        - interleave + split cost ZERO instructions: they are free-dimension slices of u3.
        - 7 DMAs in, 3 out = 10 total, replacing ~19 dispatched torch ops.
        - fp32 throughout, matching upstream's upcasting.
    """
    P, G, three = u3.shape
    S = st3.shape[2]
    kernel_assert(three == 3, "u3 must be (P, G, 3)")
    kernel_assert(P <= 128, "P must be <= 128")
    kernel_assert(fs3.shape == (P, G, 3, 2), "fs3 must be (P, G, 3, 2)")
    kernel_assert(sw3.shape == (P, G, 3, 3), "sw3 must be (P, G, 3, 3)")
    kernel_assert(pr3.shape == (P, G, 2 * S), "pr3 must be (P, G, 2*S)")

    y = nl.ndarray((P, G, 1), dtype=u3.dtype, buffer=nl.shared_hbm)
    new_fs = nl.ndarray((P, G, 3, 2), dtype=u3.dtype, buffer=nl.shared_hbm)
    new_st = nl.ndarray((P, G, S), dtype=u3.dtype, buffer=nl.shared_hbm)

    # G-tiled (same scheme as hcm_cascade_fused) so the caller can fold the
    # whole batch into G in ONE launch. Every op is elementwise over (P, G); reduce is over S.
    TG = _TG if 0 < _TG < G else G
    for g0 in range(0, G, TG):
        g1 = min(g0 + TG, G)
        _tile(u3, fs3, sw3, sb3, st3, pr3, dp3, y, new_fs, new_st, P, g0, g1, g1 - g0, S)

    return y, new_fs, new_st


import os as _os
_TG = int(_os.environ.get("EVO2_HCL_TG", "32"))


def _tile(u3, fs3, sw3, sb3, st3, pr3, dp3, y, new_fs, new_st, P, g0, g1, T, S):
    """One G-tile [g0, g1). Body identical to the original untiled kernel."""
    # ---- loads ----
    ut = nl.ndarray((P, T, 3), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=ut, src=u3[0:P, g0:g1, 0:3])
    fst = nl.ndarray((P, T, 3, 2), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=fst, src=fs3[0:P, g0:g1, 0:3, 0:2])
    swt = nl.ndarray((P, T, 3, 3), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=swt, src=sw3[0:P, g0:g1, 0:3, 0:3])
    sbt = nl.ndarray((P, T, 3), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=sbt, src=sb3[0:P, g0:g1, 0:3])
    stt = nl.ndarray((P, T, S), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=stt, src=st3[0:P, g0:g1, 0:S])
    prt = nl.ndarray((P, T, 2 * S), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=prt, src=pr3[0:P, g0:g1, 0:2 * S])
    dpt = nl.ndarray((P, T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=dpt, src=dp3[0:P, g0:g1, 0:1])

    # ================= short FIR (all 3 lanes at once) =================
    # zp = h0*u + fs[...,0]*h[...,0] + fs[...,1]*h[...,1] + bias
    zp = nl.ndarray((P, T, 3), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=zp, data1=swt[0:P, 0:T, 0:3, 2], data2=ut, op=nl.multiply)
    tmp = nl.ndarray((P, T, 3), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=tmp, data1=fst[0:P, 0:T, 0:3, 0],
                       data2=swt[0:P, 0:T, 0:3, 0], op=nl.multiply)
    nisa.tensor_tensor(dst=zp, data1=zp, data2=tmp, op=nl.add)
    nisa.tensor_tensor(dst=tmp, data1=fst[0:P, 0:T, 0:3, 1],
                       data2=swt[0:P, 0:T, 0:3, 1], op=nl.multiply)
    nisa.tensor_tensor(dst=zp, data1=zp, data2=tmp, op=nl.add)
    nisa.tensor_tensor(dst=zp, data1=zp, data2=sbt, op=nl.add)

    # short-FIR state update: [fs[...,1], u]  (roll-by-1 then append)
    nfs = nl.ndarray((P, T, 3, 2), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=nfs[0:P, 0:T, 0:3, 0], src=fst[0:P, 0:T, 0:3, 1])
    nisa.tensor_copy(dst=nfs[0:P, 0:T, 0:3, 1], src=ut)

    # ============ interleave + split == FREE lane slices ============
    # ⚠ LANE ORDER: upstream is `x2, x1, v = z_pre.split([D,D,D], 1)` AFTER interleave,
    # and interleave emits cat([z[0::3], z[1::3], z[2::3]]). So the FIRST D columns --
    # lane 0 -- bind to the variable named x2, and lane 1 binds to x1. Verified against
    # the real model with z=arange(3D): split[0]=[0,3,6,9] (lane 0) is x2.
    # Getting this backwards gave end-to-end cos_sim -0.894630.
    x2 = zp[0:P, 0:T, 0:1]
    x1 = zp[0:P, 0:T, 1:2]
    vv = zp[0:P, 0:T, 2:3]

    # ===================== IIR recurrence =====================
    po = prt[0:P, 0:T, 0:S]
    re = prt[0:P, 0:T, S:2 * S]

    x1v = nl.ndarray((P, T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=x1v, data1=x1, data2=vv, op=nl.multiply)

    nst = nl.ndarray((P, T, S), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=nst, data1=po, data2=stt, op=nl.multiply)
    xb = nl.ndarray((P, T, S), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=xb, src=x1v.broadcast(2, S))
    nisa.tensor_tensor(dst=nst, data1=nst, data2=xb, op=nl.add)

    prod = nl.ndarray((P, T, S), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=prod, data1=re, data2=nst, op=nl.multiply)
    rs = nl.ndarray((P, T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=rs, data=prod, op=nl.add, axis=(2,))

    acc = nl.ndarray((P, T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=acc, data1=dpt, data2=x1v, op=nl.multiply)
    nisa.tensor_tensor(dst=acc, data1=rs, data2=acc, op=nl.add)
    yt = nl.ndarray((P, T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=yt, data1=x2, data2=acc, op=nl.multiply)

    # ---- stores ----
    nisa.dma_copy(dst=y[0:P, g0:g1, 0:1], src=yt)
    nisa.dma_copy(dst=new_fs[0:P, g0:g1, 0:3, 0:2], src=nfs)
    nisa.dma_copy(dst=new_st[0:P, g0:g1, 0:S], src=nst)
