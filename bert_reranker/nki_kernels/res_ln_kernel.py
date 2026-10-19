"""Fused residual add + LayerNorm (post-LN, with affine) for NewBERT on Neuron.

    y = LayerNorm(x + r) * gamma + beta        x, r, y: [R, H] bf16, gamma/beta [128, H]

Per 128-row tile:
  * DMA engine:  s = x + r                (dma_compute, straight from HBM; no Vector add)
  * Vector:      bn_stats / bn_aggr       (mean, var in fp32)
  * Scalar:      rstd = rsqrt(var + eps)
  * Vector:      u = (s - mean) * gamma   (one scalar_tensor_tensor)
  * Scalar:      v = u * rstd
  * DMA engine:  y = v + beta             (dma_compute, writes HBM)

``has_residual=False`` skips the add (embedding LayerNorm). gamma/beta are passed
pre-broadcast to [128, H] once at load time. Rows are split across LNC programs, so
R must be a multiple of 128 * LNC.

Pure NKI: no torch import, so the same body runs under nki.simulate.
"""

import nki
import nki.isa as nisa
import nki.language as nl

P = 128


@nki.jit
def res_ln_kernel(x_hbm, r_hbm, g_hbm, b_hbm, eps=1e-12, has_residual=True):
    R, H = x_hbm.shape
    n_prog = nl.num_programs(0)
    pid = nl.program_id(0)

    assert R % (P * n_prog) == 0
    fmax = nl.tile_size.bn_stats_fmax
    n_chunk = (H + fmax - 1) // fmax
    assert H % n_chunk == 0
    csz = H // n_chunk

    rows_per = R // n_prog
    n_t = rows_per // P
    r_base = pid * rows_per
    dt = x_hbm.dtype

    y_hbm = nl.ndarray((R, H), dtype=dt, buffer=nl.shared_hbm)

    g_sb = nl.ndarray((P, H), dtype=g_hbm.dtype, buffer=nl.sbuf)
    b_sb = nl.ndarray((P, H), dtype=b_hbm.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=g_sb, src=g_hbm[0:P, 0:H])
    nisa.dma_copy(dst=b_sb, src=b_hbm[0:P, 0:H])
    eps_sb = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=eps_sb, value=eps)

    for t in nl.affine_range(n_t):
        r0 = r_base + t * P
        s_sb = nl.ndarray((P, H), dtype=dt, buffer=nl.sbuf)
        if has_residual:
            nisa.dma_compute(dst=s_sb, srcs=[x_hbm[nl.ds(r0, P), 0:H], r_hbm[nl.ds(r0, P), 0:H]],
                             reduce_op=nl.add)
        else:
            nisa.dma_copy(dst=s_sb, src=x_hbm[nl.ds(r0, P), 0:H])

        stats = nl.ndarray((P, 6 * n_chunk), dtype=nl.float32, buffer=nl.sbuf)
        for c in range(n_chunk):
            nisa.bn_stats(dst=stats[0:P, c * 6:(c + 1) * 6], data=s_sb[0:P, c * csz:(c + 1) * csz])
        mv = nl.ndarray((P, 2), dtype=nl.float32, buffer=nl.sbuf)
        nisa.bn_aggr(dst=mv, data=stats)

        rstd = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=rstd, op=nl.rsqrt, data=mv[0:P, 1:2], bias=eps_sb)

        u_sb = nl.ndarray((P, H), dtype=dt, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(dst=u_sb, data=s_sb, op0=nl.subtract, operand0=mv[0:P, 0:1],
                                  op1=nl.multiply, operand1=g_sb)
        v_sb = nl.ndarray((P, H), dtype=dt, buffer=nl.sbuf)
        nisa.activation(dst=v_sb, op=nl.copy, data=u_sb, scale=rstd)
        nisa.dma_compute(dst=y_hbm[nl.ds(r0, P), 0:H], srcs=[v_sb, b_sb], reduce_op=nl.add)

    return y_hbm
