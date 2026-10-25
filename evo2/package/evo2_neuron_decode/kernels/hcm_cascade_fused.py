"""Cross-layer fused HCM/HCS cascade: short FIR -> interleave -> split -> inner FIR, ONE kernel.

Companion to hcl_cascade_fused.py. Same span, but the second stage is an INNER FIR
(k=128 for HCM, k=7 for HCS) instead of the IIR recurrence:

    z_pre, fir_state = step_fir(u, fir_state, short_filter_weight, short_filter_bias)
    z_pre            = interleave(z_pre)
    x2, x1, v        = z_pre.split([D, D, D], 1)      # NOTE: x2 FIRST (see below)
    y, inner_state   = step_fir(x1 * v, inner_state, weight=h, bias=D,
                                flip_filter=(k >= 128), gated_bias=(k >= 128))
    y                = y * x2

WHY THIS IS THE BEST REMAINING TARGET (per-family profile, sync-bracketed):
    HCM 8.82 ms/layer x 9 = 79.4 ms   <-- 2x HCL's replaced-path cost
    HCS 4.70 ms/layer x 9 = 42.3 ms
    HCL 4.41 ms/layer x 9 = 39.7 ms   (done, HCL fusion, 1.073x e2e at BS=1)
The HCL fusion established the ranking metric: fusion pays in proportion to the cost of the
REPLACED PATH. HCM's is ~2x HCL's.

WHAT THE FUSION ABSORBS FOR FREE — the reason HCM should beat HCL:
  HCM's k=128 weight prep was measured at 2.412 ms/call (72% of its step_fir):
      h.repeat_interleave(16, 0)   0.267 ms
      .squeeze()                   0.034
      .flip(-1)                    1.762   <-- more than the convolution itself
      slice + unsqueeze            0.059
      .to(float32)                 0.290
  ALL of it is frozen-weight prep. The caller hoists it once, so the kernel sees a
  pre-flipped, pre-sliced, pre-fp32 (D, cache) weight and pays ZERO for it per token.

  A hoisting attempt via a Python cache was also measured to be worthless
  (1.05x, and an id()-keyed variant was a correctness defect). Hoisting only pays when the
  consumer is a kernel that takes the prepared form directly — which is this.

⚠ LANE ORDER (an earlier lane-order defect cost cos_sim -0.894630 — do not repeat):
  upstream is `x2, x1, v = interleave(z).split([D,D,D], 1)` and interleave emits
  cat([z[0::3], z[1::3], z[2::3]]), so lane 0 -> x2, lane 1 -> x1, lane 2 -> v.

MATH (identical to upstream, one timestep):
    # stage 1: short FIR, cache_size 2, filter_length 3, no flip
    zp[c]         = sw[c,2]*u[c] + sw[c,0]*fs[c,0] + sw[c,1]*fs[c,1] + sb[c]
    fs'[c]        = [fs[c,1], u[c]]
    # deinterleave (free): x2 = zp[...,0], x1 = zp[...,1], v = zp[...,2]
    g[d]          = x1[d] * v[d]
    # stage 2: inner FIR over `cache` taps; caller supplies h0/h already prepared
    y[d]          = h0[d]*g[d] + sum_j is[d,j]*h[d,j] + bias_term
                    where bias_term = Dp[d]*g[d] if gated_bias else Dp[d]
    is'[d]        = [is[d,1:], g[d]]            (roll-by-1 + append)
    y[d]          = y[d] * x2[d]
"""

import nki
import nki.isa as nisa
import nki.language as nl


def kernel_assert(condition, message=""):
    assert condition, message


@nki.jit
def hcm_cascade_fused(u3, fs3, sw3, sb3, is3, ih3, ih0, dp3, gated_bias):
    """Whole HCM/HCS cascade in one kernel, single batch row.

    Args:
        u3:  (P, G, 3)        fp32 - input, 3 interleave lanes innermost
        fs3: (P, G, 3, 2)     fp32 - short-FIR state (cache_size 2)
        sw3: (P, G, 3, 3)     fp32 - short filter weights (3 taps), caller-expanded
        sb3: (P, G, 3)        fp32 - short filter bias (zeros if none)
        is3: (P, G, C)        fp32 - inner-FIR state, C = inner cache_size (k-1)
        ih3: (P, G, C)        fp32 - inner filter taps h[..., :C], CALLER-PREPARED
                                     (already repeat_interleaved, flipped, sliced, fp32)
        ih0: (P, G, 1)        fp32 - inner filter h0 tap, CALLER-PREPARED
        dp3: (P, G, 1)        fp32 - inner bias D
        gated_bias: int       - 1 => bias*g (k>=128 path), 0 => bias (k<128 path)

    Returns:
        y:      (P, G, 1)     fp32
        new_fs: (P, G, 3, 2)  fp32
        new_is: (P, G, C)     fp32

    Notes:
        - interleave + split cost ZERO instructions (free-dimension lane slices).
        - The k=128 weight prep (flip/repeat_interleave/upcast, 2.412 ms/call measured) is
          entirely caller-hoisted: this kernel receives ih0/ih3 ready to use.
        - fp32 throughout, matching upstream.
    """
    P, G, three = u3.shape
    C = is3.shape[2]
    kernel_assert(three == 3, "u3 must be (P, G, 3)")
    kernel_assert(P <= 128, "P must be <= 128")
    kernel_assert(fs3.shape == (P, G, 3, 2), "fs3 must be (P, G, 3, 2)")
    kernel_assert(sw3.shape == (P, G, 3, 3), "sw3 must be (P, G, 3, 3)")
    kernel_assert(ih3.shape == (P, G, C), "ih3 must match is3's cache size")

    y = nl.ndarray((P, G, 1), dtype=u3.dtype, buffer=nl.shared_hbm)
    new_fs = nl.ndarray((P, G, 3, 2), dtype=u3.dtype, buffer=nl.shared_hbm)
    new_is = nl.ndarray((P, G, C), dtype=u3.dtype, buffer=nl.shared_hbm)

    # Tile the free (G) axis INSIDE the kernel so SBUF per iteration is bounded by
    # TG*C, independent of the batch fold width. Without this the compiler picked the SBUF
    # tiling itself and failed NCC_IBIR229 at G=384 / 320 (TP=4, B=48 / B=80 chunks) while
    # G=512 compiled -- shape-dependent, not a capacity limit. Every op is elementwise over
    # (P, G) (the reduce is over C only), so G tiles are fully independent.
    # Per-tile fp32 SBUF at TG=32, C=127: ist/iht/prod/nis 4 x 16.3 KB = 65 KB/partition.
    TG = _TG if 0 < _TG < G else G
    for g0 in range(0, G, TG):
        g1 = min(g0 + TG, G)
        T = g1 - g0
        _tile(u3, fs3, sw3, sb3, is3, ih3, ih0, dp3, gated_bias, y, new_fs, new_is, P, g0, g1, T, C)

    return y, new_fs, new_is


# EVO2_HCM_TG=0 disables the in-kernel G tiling (one tile = whole G, the original untiled kernel).
import os as _os
_TG = int(_os.environ.get("EVO2_HCM_TG", "32"))


def _tile(u3, fs3, sw3, sb3, is3, ih3, ih0, dp3, gated_bias, y, new_fs, new_is, P, g0, g1, T, C):
    """One G-tile [g0, g1) of the cascade. Body identical to the original untiled kernel."""
    # ---- loads ----
    ut = nl.ndarray((P, T, 3), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=ut, src=u3[0:P, g0:g1, 0:3])
    fst = nl.ndarray((P, T, 3, 2), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=fst, src=fs3[0:P, g0:g1, 0:3, 0:2])
    swt = nl.ndarray((P, T, 3, 3), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=swt, src=sw3[0:P, g0:g1, 0:3, 0:3])
    sbt = nl.ndarray((P, T, 3), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=sbt, src=sb3[0:P, g0:g1, 0:3])
    ist = nl.ndarray((P, T, C), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=ist, src=is3[0:P, g0:g1, 0:C])
    iht = nl.ndarray((P, T, C), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=iht, src=ih3[0:P, g0:g1, 0:C])
    ih0t = nl.ndarray((P, T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=ih0t, src=ih0[0:P, g0:g1, 0:1])
    dpt = nl.ndarray((P, T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=dpt, src=dp3[0:P, g0:g1, 0:1])

    # ============ stage 1: short FIR, all 3 lanes at once ============
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

    nfs = nl.ndarray((P, T, 3, 2), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=nfs[0:P, 0:T, 0:3, 0], src=fst[0:P, 0:T, 0:3, 1])
    nisa.tensor_copy(dst=nfs[0:P, 0:T, 0:3, 1], src=ut)

    # ======== interleave + split == FREE lane slices ========
    # lane0 -> x2, lane1 -> x1, lane2 -> v   (see module docstring)
    x2 = zp[0:P, 0:T, 0:1]
    x1 = zp[0:P, 0:T, 1:2]
    vv = zp[0:P, 0:T, 2:3]

    # ============ stage 2: inner FIR on g = x1*v ============
    g = nl.ndarray((P, T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=g, data1=x1, data2=vv, op=nl.multiply)

    # y = h0*g + sum_j is[j]*h[j]
    yt = nl.ndarray((P, T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=yt, data1=ih0t, data2=g, op=nl.multiply)
    prod = nl.ndarray((P, T, C), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=prod, data1=ist, data2=iht, op=nl.multiply)
    red = nl.ndarray((P, T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=red, data=prod, op=nl.add, axis=(2,))
    nisa.tensor_tensor(dst=yt, data1=yt, data2=red, op=nl.add)

    # bias: gated (bias*g) for k>=128, plain (bias) otherwise
    bt = nl.ndarray((P, T, 1), dtype=nl.float32, buffer=nl.sbuf)
    if gated_bias == 1:
        nisa.tensor_tensor(dst=bt, data1=dpt, data2=g, op=nl.multiply)
    else:
        nisa.tensor_copy(dst=bt, src=dpt)
    nisa.tensor_tensor(dst=yt, data1=yt, data2=bt, op=nl.add)

    # inner state update: [is[1:], g]
    nis = nl.ndarray((P, T, C), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=nis[0:P, 0:T, 0:C - 1], src=ist[0:P, 0:T, 1:C])
    nisa.tensor_copy(dst=nis[0:P, 0:T, C - 1:C], src=g)

    # ---- output gate ----
    yo = nl.ndarray((P, T, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=yo, data1=yt, data2=x2, op=nl.multiply)

    # ---- stores ----
    nisa.dma_copy(dst=y[0:P, g0:g1, 0:1], src=yo)
    nisa.dma_copy(dst=new_fs[0:P, g0:g1, 0:3, 0:2], src=nfs)
    nisa.dma_copy(dst=new_is[0:P, g0:g1, 0:C], src=nis)
