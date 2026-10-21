"""Fused NewBERT gated MLP + residual + post-LayerNorm (replaces NewGatedMLP + mlp_ln).

    h   = x @ Wup^T,  gt = x @ Wgate^T                 (up_gate_proj, no bias; split [up | gate])
    m   = (gelu_tanh(gt) * h) @ Wd^T + bd              (down_proj)
    y   = LayerNorm(x + m) * gamma + beta              (mlp_ln, residual = MLP input)

Shapes: x [R, K] bf16 (K = hidden, 768), wupT / wgT [K, I] bf16 (I = 3072), wdT [I, K] bf16,
bd / gamma / beta pre-broadcast to [128, K] (fp32 or bf16). Returns y [R, K] bf16.

Layout: all weights stay resident in SBUF (K-tiles x I for up/gate, I-tiles x K for down; ~110 KiB
per partition at K=768, I=3072). Rows are processed in blocks of up to 512 (the matmul moving free
size). Per block:
  * x rows are loaded once (row layout, kept for the residual) and transposed to x^T (K on
    partitions) by DMA transpose.
  * up/gate for each 128-wide I tile land in PSUM [128 (I), rows]; gelu_tanh on Scalar, multiply
    on Vector -> g [I tile, rows] bf16, already in the layout the down-projection needs.
  * down: per 128-row subtile, PSUM [128 rows, K] accumulated over the I tiles.
  * epilogue: s = psum + x + bd (fp32), bn_stats/bn_aggr, y = (s - mean) * gamma * rstd + beta.
R must be a multiple of 128 * LNC; rows are split across LNC programs. Inference only.
"""

import nki
import nki.isa as nisa
import nki.language as nl

P = 128
FMAX = 512


def _cdiv(a, b):
    return (a + b - 1) // b


@nki.jit
def gated_mlp_ln_kernel(x_hbm, wupT_hbm, wgT_hbm, wdT_hbm, bd_hbm, g_hbm, b_hbm, eps=1e-12):
    R, K = x_hbm.shape
    K1, I = wupT_hbm.shape
    assert K1 == K
    assert K % P == 0
    assert I % P == 0
    n_prog = nl.num_programs(0)
    pid = nl.program_id(0)
    assert R % (P * n_prog) == 0
    rows_per = R // n_prog
    r_base = pid * rows_per
    n_k = K // P
    n_i = I // P
    n_ko = _cdiv(K, FMAX)
    n_blk = _cdiv(rows_per, FMAX)
    fmax = nl.tile_size.bn_stats_fmax
    n_ch = _cdiv(K, fmax)
    csz = K // n_ch
    dt = x_hbm.dtype
    V2 = nisa.get_nc_version() == nisa.nc_version.gen2      # Inf2 / Trn1: DMA transpose must read HBM

    y_hbm = nl.ndarray((R, K), dtype=dt, buffer=nl.shared_hbm)

    # ---- resident weights ----
    wu = []
    wg = []
    for kt in range(n_k):
        tu = nl.ndarray((P, I), dtype=wupT_hbm.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=tu, src=wupT_hbm[kt * P:(kt + 1) * P, 0:I])
        wu.append(tu)
        tg = nl.ndarray((P, I), dtype=wgT_hbm.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=tg, src=wgT_hbm[kt * P:(kt + 1) * P, 0:I])
        wg.append(tg)
    wd = []
    for it in range(n_i):
        td = nl.ndarray((P, K), dtype=wdT_hbm.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=td, src=wdT_hbm[it * P:(it + 1) * P, 0:K])
        wd.append(td)
    bd_sb = nl.ndarray((P, K), dtype=bd_hbm.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=bd_sb, src=bd_hbm[0:P, 0:K])
    ga_sb = nl.ndarray((P, K), dtype=g_hbm.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=ga_sb, src=g_hbm[0:P, 0:K])
    be_sb = nl.ndarray((P, K), dtype=b_hbm.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=be_sb, src=b_hbm[0:P, 0:K])
    if V2 and b_hbm.dtype != dt:  # gen2: dma_compute with mixed bf16/fp32 sources gives wrong results -> keep beta in the output dtype
        be_lo = nl.ndarray((P, K), dtype=dt, buffer=nl.sbuf)
        nisa.tensor_copy(dst=be_lo, src=be_sb)
        be_sb = be_lo
    eps_sb = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=eps_sb, value=eps)

    for blk in range(n_blk):
        q0 = blk * FMAX
        rw = min(FMAX, rows_per - q0)
        n_rs = rw // P
        rb = r_base + q0

        # ---- x rows (+bd folded in later) and x^T ----
        xr = []
        xt = []
        for kt in range(n_k):
            xt.append(nl.ndarray((P, FMAX), dtype=dt, buffer=nl.sbuf))
        for rs in range(n_rs):
            t = nl.ndarray((P, K), dtype=dt, buffer=nl.sbuf)
            nisa.dma_copy(dst=t, src=x_hbm[nl.ds(rb + rs * P, P), 0:K])
            xr.append(t)
            for kt in range(n_k):
                if V2:
                    nisa.dma_transpose(dst=xt[kt][0:P, rs * P:(rs + 1) * P],
                                       src=x_hbm[nl.ds(rb + rs * P, P), kt * P:(kt + 1) * P])
                else:
                    nisa.dma_transpose(dst=xt[kt][0:P, rs * P:(rs + 1) * P], src=t[0:P, kt * P:(kt + 1) * P])

        # ---- up / gate -> g [I tile, rows] ----
        gl = []
        for it in range(n_i):
            i0 = it * P
            pu = nl.ndarray((P, FMAX), dtype=nl.float32, buffer=nl.psum)
            for kt in nl.affine_range(n_k):
                nisa.nc_matmul(dst=pu[0:P, 0:rw], stationary=wu[kt][0:P, i0:i0 + P], moving=xt[kt][0:P, 0:rw])
            pg = nl.ndarray((P, FMAX), dtype=nl.float32, buffer=nl.psum)
            for kt in nl.affine_range(n_k):
                nisa.nc_matmul(dst=pg[0:P, 0:rw], stationary=wg[kt][0:P, i0:i0 + P], moving=xt[kt][0:P, 0:rw])
            ge = nl.ndarray((P, FMAX), dtype=dt, buffer=nl.sbuf)
            nisa.activation(dst=ge[0:P, 0:rw], op=nl.gelu_apprx_tanh, data=pg[0:P, 0:rw])
            g = nl.ndarray((P, FMAX), dtype=dt, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=g[0:P, 0:rw], data1=pu[0:P, 0:rw], data2=ge[0:P, 0:rw], op=nl.multiply)
            gl.append(g)

        # ---- down + residual + LN, per 128-row subtile ----
        for rs in range(n_rs):
            s_sb = nl.ndarray((P, K), dtype=nl.float32, buffer=nl.sbuf)
            xb = nl.ndarray((P, K), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=xb, data1=xr[rs], data2=bd_sb, op=nl.add,
                               engine=nisa.engine.vector if V2 else nisa.engine.gpsimd)
            for ko in range(n_ko):
                k0 = ko * FMAX
                nk = min(FMAX, K - k0)
                pd = nl.ndarray((P, FMAX), dtype=nl.float32, buffer=nl.psum)
                for it in nl.affine_range(n_i):
                    nisa.nc_matmul(dst=pd[0:P, 0:nk], stationary=gl[it][0:P, rs * P:(rs + 1) * P],
                                   moving=wd[it][0:P, k0:k0 + nk])
                nisa.tensor_tensor(dst=s_sb[0:P, k0:k0 + nk], data1=pd[0:P, 0:nk], data2=xb[0:P, k0:k0 + nk],
                                   op=nl.add)
            stats = nl.ndarray((P, 6 * n_ch), dtype=nl.float32, buffer=nl.sbuf)
            for c in range(n_ch):
                nisa.bn_stats(dst=stats[0:P, c * 6:(c + 1) * 6], data=s_sb[0:P, c * csz:(c + 1) * csz])
            mv = nl.ndarray((P, 2), dtype=nl.float32, buffer=nl.sbuf)
            nisa.bn_aggr(dst=mv, data=stats)
            rstd = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=rstd, op=nl.rsqrt, data=mv[0:P, 1:2], bias=eps_sb)
            u_sb = nl.ndarray((P, K), dtype=dt, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(dst=u_sb, data=s_sb, op0=nl.subtract, operand0=mv[0:P, 0:1],
                                      op1=nl.multiply, operand1=ga_sb)
            v_sb = nl.ndarray((P, K), dtype=dt, buffer=nl.sbuf)
            nisa.activation(dst=v_sb, op=nl.copy, data=u_sb, scale=rstd)
            nisa.dma_compute(dst=y_hbm[nl.ds(rb + rs * P, P), 0:K], srcs=[v_sb, be_sb], reduce_op=nl.add)

    return y_hbm
