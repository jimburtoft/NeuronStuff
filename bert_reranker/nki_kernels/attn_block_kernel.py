"""Fused attention half of a NewBERT (gte) encoder layer, for PyTorch Native on Trainium2.

    y = attn_ln(x + o_proj(softmax(rope(q) rope(k)^T / sqrt(d) + key_mask) v))

Replaces qkv_proj + rotate-half RoPE + masked softmax attention + o_proj + residual + attn_ln.

Inputs (bf16 unless noted); weights are packed once on the host (attn_block_pack.pack):
  x     [B*L, H]          layer input, token-major (head dim D = 64, NH = H / 64)
  wqk   [H, 2*H]          Q and K projections, transposed, lane-permuted, Q pre-scaled by 1/sqrt(D)
  bqk   [128, 2*H/128]    Q/K bias, same permutation, one column per 128-lane group (fp32)
  wv    [H, H]            V projection transposed;   bv [128, H] broadcast (fp32)
  wo    [H, H]            o_proj transposed;         bo [128, H] broadcast (fp32)
  g, b  [128, H]          attn_ln weight / bias broadcast (fp32)
  cs    [2, 128, L]       RoPE cos / sin per lane (fp32); lane l uses frequency l % 32
  kb    [B, L]            additive key bias (0 valid, -30000 padded), right padding
Output y [B*L, H].

Design:
  * Lane permutation: 128-lane group (j, par) holds lanes b*32+i = feature (4j+b)*D + par*32 + i of heads
    4j..4j+3. par=0 holds the low halves, par=1 the high halves, so rotate-half RoPE is elementwise between
    two lane groups (no transposes). Q and K share the order, so dot products are unchanged.
  * The key mask and the softmax denominator ride on augmented contraction rows: Q^T gets a ones row and K^T
    the key-bias row (one K=65 matmul gives S = QK^T + mask); V gets a ones column, so the P.V matmul also
    returns sum(P).
  * 2-4 short sequences share each projection pass (moving width up to 512).
  * Sequences are split across the LNC programs. If the batch does not divide (e.g. B=1 at LNC=2) every
    program computes K/V for all sequences and only its share of the query tiles (no cross-core reduction).
  * The LayerNorm epilogue computes all statistics first and one rsqrt for the whole pass, so the Scalar
    engine does not reload its activation table between rsqrt and exp on every tile.
L must be a multiple of 128 and <= 512. Inference only.
"""

import nki
import nki.isa as nisa
import nki.language as nl

P = 128
FMAX = 512


def _cdiv(a, b):
    return (a + b - 1) // b


def _group(NH, L):
    best = 1
    for g in range(1, NH + 1):
        if NH % g == 0 and g * L <= 1024:
            best = g
    return best


def _ln_out(s_sb, mean, rstd, ga, be, y_hbm, row0, P_, H, dt):
    u_sb = nl.ndarray((P_, H), dtype=dt, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(dst=u_sb, data=s_sb, op0=nl.subtract, operand0=mean,
                              op1=nl.multiply, operand1=ga)
    v_sb = nl.ndarray((P_, H), dtype=dt, buffer=nl.sbuf)
    nisa.activation(dst=v_sb, op=nl.copy, data=u_sb, scale=rstd)
    nisa.dma_compute(dst=y_hbm[nl.ds(row0, P_), 0:H], srcs=[v_sb, be], reduce_op=nl.add)


@nki.jit
def attn_block_kernel(x_hbm, wqk_hbm, bqk_hbm, wv_hbm, bv_hbm, wo_hbm, bo_hbm, g_hbm, b_hbm, cs_hbm, kb_hbm,
                      eps=1e-12):
    R, H = x_hbm.shape
    B, L = kb_hbm.shape
    D = 64
    NH = H // D
    half = D // 2
    quad = P // half                       # heads per 128-lane group (4)
    NQ = NH // quad                        # lane groups per parity
    assert NH * D == H
    assert R == B * L
    assert L % P == 0
    assert L <= FMAX
    assert H % P == 0
    assert NH % quad == 0
    n_prog = nl.num_programs(0)
    pid = nl.program_id(0)
    nt = L // P                            # token tiles per sequence
    QS = 0                                 # query split across programs (batch not divisible)
    if n_prog > 1 and B % n_prog != 0:
        QS = 1
    if QS == 1:
        assert nt % n_prog == 0
        Bp = B
    else:
        Bp = B // n_prog
    n_k = H // P                           # contraction tiles over H
    G = _group(NH, L)
    NG = NH // G
    n_ko = _cdiv(H, FMAX)
    fmax = nl.tile_size.bn_stats_fmax
    n_ch = _cdiv(H, fmax)
    csz = H // n_ch
    dt = x_hbm.dtype

    y_hbm = nl.ndarray((R, H), dtype=dt, buffer=nl.shared_hbm)

    # ---------------- resident weights / constants ----------------
    wqk = nl.ndarray((P, n_k, 2 * H), dtype=dt, buffer=nl.sbuf)
    wv = nl.ndarray((P, n_k, H), dtype=dt, buffer=nl.sbuf)
    wo = nl.ndarray((P, n_k, H), dtype=dt, buffer=nl.sbuf)
    for kt in range(n_k):
        nisa.dma_copy(dst=wqk[:, kt, :], src=wqk_hbm[kt * P:(kt + 1) * P, 0:2 * H])
        nisa.dma_copy(dst=wv[:, kt, :], src=wv_hbm[kt * P:(kt + 1) * P, 0:H])
        nisa.dma_copy(dst=wo[:, kt, :], src=wo_hbm[kt * P:(kt + 1) * P, 0:H])
    bqk = nl.ndarray((P, 2 * H // P), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=bqk, src=bqk_hbm[0:P, 0:2 * H // P])
    bv = nl.ndarray((P, H), dtype=bv_hbm.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=bv, src=bv_hbm[0:P, 0:H])
    bo = nl.ndarray((P, H), dtype=bo_hbm.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=bo, src=bo_hbm[0:P, 0:H])
    ga = nl.ndarray((P, H), dtype=g_hbm.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=ga, src=g_hbm[0:P, 0:H])
    be = nl.ndarray((P, H), dtype=b_hbm.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=be, src=b_hbm[0:P, 0:H])
    cos_t = nl.ndarray((P, L), dtype=nl.float32, buffer=nl.sbuf)
    sin_t = nl.ndarray((P, L), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=cos_t, src=cs_hbm[0, 0:P, 0:L])
    nisa.dma_copy(dst=sin_t, src=cs_hbm[1, 0:P, 0:L])
    eps_sb = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=eps_sb, value=eps)

    SP = 1                                 # sequences per projection pass
    if QS == 0:
        SP = FMAX // L
        if SP > 4:
            SP = 4
        if Bp % SP != 0:
            SP = 1
    LW = L * SP                            # projection moving width
    cosw = nl.ndarray((P, LW), dtype=dt, buffer=nl.sbuf)
    sinw = nl.ndarray((P, LW), dtype=dt, buffer=nl.sbuf)
    for sp in range(SP):
        nisa.tensor_copy(dst=cosw[:, sp * L:(sp + 1) * L], src=cos_t)
        nisa.tensor_copy(dst=sinw[:, sp * L:(sp + 1) * L], src=sin_t)

    for bi in nl.affine_range(Bp // SP):
        bseq = pid * Bp + bi * SP
        if QS == 1:
            bseq = bi
        r0 = bseq * L
        ntw = nt * SP                      # token tiles in this pass

        # ---------------- x rows, x + bo, x^T ----------------
        xr = nl.ndarray((P, ntw, H), dtype=dt, buffer=nl.sbuf)
        xb = nl.ndarray((P, ntw, H), dtype=nl.float32, buffer=nl.sbuf)
        xT = nl.ndarray((P, n_k, LW), dtype=dt, buffer=nl.sbuf)
        for t in range(ntw):
            nisa.dma_copy(dst=xr[:, t, :], src=x_hbm[nl.ds(r0 + t * P, P), 0:H])
            nisa.tensor_tensor(dst=xb[:, t, :], data1=xr[:, t, :], data2=bo, op=nl.add, engine=nisa.engine.gpsimd)
            for kt in range(n_k):
                nisa.dma_transpose(dst=xT[:, kt, t * P:(t + 1) * P], src=xr[:, t, kt * P:(kt + 1) * P])

        # ---------------- Q/K projection + bias + RoPE -> qa / ka ----------------
        qa = nl.ndarray((D + 1, NH, LW), dtype=dt, buffer=nl.sbuf)
        ka = nl.ndarray((D + 1, NH, LW), dtype=dt, buffer=nl.sbuf)
        nisa.memset(dst=qa[D:D + 1, :, :], value=1.0, engine=nisa.engine.gpsimd)
        for sp in range(SP):
            nisa.dma_copy(dst=ka[D:D + 1, :, sp * L:(sp + 1) * L],
                          src=kb_hbm.ap(pattern=[[L, 1], [0, NH], [1, L]], offset=(bseq + sp) * L))
        for qk in range(2):
            dst_t = qa
            if qk == 1:
                dst_t = ka
            for j in range(NQ):
                pr = []
                for par in range(2):
                    gi = (qk * NQ + j) * 2 + par
                    ps = nl.ndarray((P, LW), dtype=nl.float32, buffer=nl.psum)
                    for kt in nl.affine_range(n_k):
                        nisa.nc_matmul(dst=ps, stationary=wqk[:, kt, gi * P:(gi + 1) * P], moving=xT[:, kt, :])
                    pt = nl.ndarray((P, LW), dtype=dt, buffer=nl.sbuf)
                    nisa.activation(dst=pt, op=nl.copy, data=ps, bias=bqk[:, gi:gi + 1])
                    pr.append(pt)
                lo = pr[0]
                hi = pr[1]
                a0 = nl.ndarray((P, LW), dtype=dt, buffer=nl.sbuf)
                t0 = nl.ndarray((P, LW), dtype=dt, buffer=nl.sbuf)
                a1 = nl.ndarray((P, LW), dtype=dt, buffer=nl.sbuf)
                t1 = nl.ndarray((P, LW), dtype=dt, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=a0, data1=lo, data2=cosw, op=nl.multiply)
                nisa.tensor_tensor(dst=t0, data1=hi, data2=sinw, op=nl.multiply, engine=nisa.engine.vector)
                nisa.tensor_tensor(dst=a1, data1=hi, data2=cosw, op=nl.multiply)
                nisa.tensor_tensor(dst=t1, data1=lo, data2=sinw, op=nl.multiply, engine=nisa.engine.vector)
                r_lo = nl.ndarray((P, LW), dtype=dt, buffer=nl.sbuf)
                r_hi = nl.ndarray((P, LW), dtype=dt, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=r_lo, data1=a0, data2=t0, op=nl.subtract)
                nisa.tensor_tensor(dst=r_hi, data1=a1, data2=t1, op=nl.add)
                # one copy per (head, half) 32-partition block. A single strided DMA spanning several
                # partition blocks is exact in nki.simulate but only copies the first block on trn2.
                for bq in range(quad):
                    h = j * quad + bq
                    rr = bq * half
                    nisa.tensor_copy(dst=dst_t[0:half, h, :], src=r_lo[rr:rr + half, :], engine=nisa.engine.scalar)
                    nisa.tensor_copy(dst=dst_t[half:D, h, :], src=r_hi[rr:rr + half, :])

        # ---------------- V projection (token-major) + bias -> va (ones column at D) ----------------
        va = nl.ndarray((P, ntw, NH, D + 1), dtype=dt, buffer=nl.sbuf)
        nisa.memset(dst=va[:, :, :, D:D + 1], value=1.0, engine=nisa.engine.gpsimd)
        for t in range(ntw):
            for ko in range(n_ko):
                k0 = ko * FMAX
                nk = min(FMAX, H - k0)
                hv = nk // D
                pv = nl.ndarray((P, FMAX), dtype=nl.float32, buffer=nl.psum)
                for kt in nl.affine_range(n_k):
                    nisa.nc_matmul(dst=pv[:, 0:nk], stationary=xT[:, kt, t * P:(t + 1) * P],
                                   moving=wv[:, kt, k0:k0 + nk])
                h0 = k0 // D
                nisa.tensor_tensor(dst=va[:, t, h0:h0 + hv, 0:D], data1=pv[:, 0:nk], data2=bv[:, k0:k0 + nk],
                                   op=nl.add)

        # ---------------- attention + o_proj + residual + LN, per query tile ----------------
        nq = ntw
        if QS == 1:
            nq = ntw // n_prog
            qbase = pid * nq
            xbq = nl.ndarray((P, nq, H), dtype=nl.float32, buffer=nl.sbuf)
            for i in range(nq):
                xq = nl.ndarray((P, H), dtype=dt, buffer=nl.sbuf)
                nisa.dma_copy(dst=xq, src=x_hbm[nl.ds(r0 + (qbase + i) * P, P), 0:H])
                nisa.tensor_tensor(dst=xbq[:, i, :], data1=xq, data2=bo, op=nl.add, engine=nisa.engine.gpsimd)
        s_all = nl.ndarray((P, nq, H), dtype=nl.float32, buffer=nl.sbuf)
        mv_all = nl.ndarray((P, nq, 2), dtype=nl.float32, buffer=nl.sbuf)
        for qi in range(nq):
            sp = qi // nt
            qt = qi % nt
            c0 = sp * L                        # column offset of this sequence in qa/ka
            t_off = sp * nt                    # token-tile offset of this sequence in va / xb
            qcol = c0 + qt * P
            xres = xb[:, t_off + qt, :]
            if QS == 1:
                qcol = (qbase + qi) * P
                xres = xbq[:, qi, :]
            o_sb = nl.ndarray((P, H), dtype=dt, buffer=nl.sbuf)
            for gg in range(NG):
                S = nl.ndarray((P, G, L), dtype=nl.float32, buffer=nl.psum)
                for hl in range(G):
                    h = gg * G + hl
                    nisa.nc_matmul(dst=S[:, hl, :], stationary=qa[0:D + 1, h, nl.ds(qcol, P)],
                                   moving=ka[0:D + 1, h, c0:c0 + L])
                negmax = nl.ndarray((P, G), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(dst=negmax, op=nl.maximum, data=S, axis=2, negate=True)
                Pm = nl.ndarray((P, G, L), dtype=dt, buffer=nl.sbuf)
                for hl in range(G):
                    nisa.activation(dst=Pm[:, hl, :], op=nl.exp, data=S[:, hl, :], bias=negmax[:, hl:hl + 1])
                PT = nl.ndarray((P, G, nt, P), dtype=dt, buffer=nl.sbuf)
                nisa.dma_transpose(dst=PT.ap(pattern=[[G * L, P], [1, 1], [P, G * nt], [1, P]], offset=0),
                                   src=Pm.ap(pattern=[[G * L, P], [1, 1], [P, G * nt], [1, P]], offset=0))
                O = nl.ndarray((P, G, D + 1), dtype=nl.float32, buffer=nl.psum)
                for hl in range(G):
                    h = gg * G + hl
                    for kt in nl.affine_range(nt):
                        nisa.nc_matmul(dst=O[:, hl, :], stationary=PT[:, hl, kt, :], moving=va[:, t_off + kt, h, :])
                rs = nl.ndarray((P, G, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.reciprocal(dst=rs, data=O[:, :, D:D + 1])
                o3 = o_sb[:, gg * G * D:(gg + 1) * G * D].reshape((P, G, D))
                nisa.tensor_tensor(dst=o3, data1=O[:, :, 0:D],
                                   data2=rs.ap(pattern=[[G, P], [1, G], [0, D]], offset=0), op=nl.multiply)

            # o^T for the output projection
            oTp = nl.ndarray((P, n_k, P), dtype=dt, buffer=nl.psum)
            for kt in range(n_k):
                nisa.nc_transpose(dst=oTp[:, kt, :], data=o_sb[:, kt * P:(kt + 1) * P], engine=nisa.engine.tensor)
            oT = nl.ndarray((P, n_k, P), dtype=dt, buffer=nl.sbuf)
            nisa.tensor_copy(dst=oT, src=oTp, engine=nisa.engine.scalar)

            s_sb = s_all[:, qi, :]
            for ko in range(n_ko):
                k0 = ko * FMAX
                nk = min(FMAX, H - k0)
                po = nl.ndarray((P, FMAX), dtype=nl.float32, buffer=nl.psum)
                for kt in nl.affine_range(n_k):
                    nisa.nc_matmul(dst=po[:, 0:nk], stationary=oT[:, kt, :], moving=wo[:, kt, k0:k0 + nk])
                nisa.tensor_tensor(dst=s_sb[:, k0:k0 + nk], data1=po[:, 0:nk], data2=xres[:, k0:k0 + nk],
                                   op=nl.add)
            stats = nl.ndarray((P, 6 * n_ch), dtype=nl.float32, buffer=nl.sbuf)
            for c in range(n_ch):
                nisa.bn_stats(dst=stats[:, c * 6:(c + 1) * 6], data=s_sb[:, c * csz:(c + 1) * csz])
            nisa.bn_aggr(dst=mv_all[:, qi, :], data=stats)
        rstd_all = nl.ndarray((P, nq, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=rstd_all, op=nl.rsqrt, data=mv_all[:, :, 1:2], bias=eps_sb)
        for qi in range(nq):
            yr = r0 + qi * P
            if QS == 1:
                yr = r0 + (qbase + qi) * P
            _ln_out(s_all[:, qi, :], mv_all[:, qi, 0:1], rstd_all[:, qi, :], ga, be, y_hbm, yr, P, H, dt)

    return y_hbm
