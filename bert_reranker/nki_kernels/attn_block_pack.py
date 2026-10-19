"""Host-side packing for attn_block_kernel (pure NumPy; used by the torch wrapper and the sim test)."""
import numpy as np

P = 128
D = 64
HALF = 32
QUAD = P // HALF


def lane_perm(NH):
    """perm[g*128 + l] = feature index (within one projection) for lane group g = j*2 + par."""
    NQ = NH // QUAD
    perm = np.zeros(NH * D, np.int64)
    for j in range(NQ):
        for par in range(2):
            g = j * 2 + par
            for bq in range(QUAD):
                h = j * QUAD + bq
                for i in range(HALF):
                    perm[g * P + bq * HALF + i] = h * D + par * HALF + i
    return perm


def pack(wqkv, bqkv, wo, bo, g, b, cos, sin, H, scale, vb_fold=False):
    """wqkv [3H, H], bqkv [3H] (qkv_proj, rows q|k|v), wo [H, H], bo [H] (o_proj), g/b [H] (attn_ln),
    cos/sin [L, D] (RoPE table rows for positions 0..L-1). Returns dict of kernel inputs (fp32/np)."""
    NH = H // D
    perm = lane_perm(NH)
    wq, wk, wv = wqkv[:H], wqkv[H:2 * H], wqkv[2 * H:]
    bq, bk, bv = bqkv[:H], bqkv[H:2 * H], bqkv[2 * H:]
    wq_p, bq_p = wq[perm] * scale, bq[perm] * scale
    wk_p, bk_p = wk[perm], bk[perm]
    wqk = np.concatenate([wq_p, wk_p], 0).T.copy()                          # [H, 2H]
    bqk = np.concatenate([bq_p, bk_p]).reshape(-1, P).T.copy()              # [128, 2H/128]
    lane_freq = np.arange(P) % HALF                                         # lane l -> freq index
    cs = np.stack([cos[:, lane_freq].T, sin[:, lane_freq].T]).astype(np.float32)   # [2, 128, L]
    bc = lambda v: np.tile(v.astype(np.float32), (P, 1))
    if vb_fold:                     # softmax rows sum to 1: o_proj(PV + bv) = o_proj(PV) + Wo bv
        bo = bo + wo @ bv
    return dict(wqk=wqk, bqk=bqk.astype(np.float32), wv=wv.T.copy(), bv=bc(bv), wo=wo.T.copy(), bo=bc(bo),
                g=bc(g), b=bc(b), cs=cs)
