"""Fused StripedHyena-1 decode cascade: short FIR k=3 + column split + 2-state modal IIR (re/im) + gate,
ONE NKI launch for all batch rows.

Adapted from the Evo 2 decode package ``evo2_neuron_decode/kernels/hcl_cascade_fused.py`` (evo2_neuron_decode commit
7c93113). Differences for SH1:
- column split instead of the stride-3 interleave: with ``port.opts`` "colperm" the projection rows are
  permuted at load to [x2 | x1 | v], each D contiguous, so the split is a free-axis slice of the tile;
- FIR has a bias; FIR state is PLANAR (2, B, 3D) (measured: interleaved state slowed the graph 21x);
- the IIR is complex with K=2 states, as real pairs:
      n_re = p_re s_re - p_im s_im + x1v,  n_im = p_re s_im + p_im s_re,
      y    = x2 * (sum_k (r_re n_re - r_im n_im) + D x1v)
- the batch is folded into the free axis (measured fold), so the instruction count does not grow
  with B; the caller pre-broadcasts the frozen weights to the batch (cached per B).

Layout: channel c of a D-vector -> partition p = c // G, free g = c % G, P = 128, G = D / 128 = 32.
HBM inputs (all contiguous):
  z    (B, 3, P, G) bf16   projection output for this step ([x2 | x1 | v])
  fir  (2, B, 3, P, G) bf16  previous two projection outputs (slot 0 = t-2, slot 1 = t-1)
  s_re, s_im (B, P, G, K) fp32
  wb   (P, 4, B, 3, G) fp32  FIR taps w0, w1, w2 and bias, pre-broadcast over B
  pl   (P, 4, B, G, K) fp32  p_re, p_im, r_re, r_im, pre-broadcast over B
  dsk  (P, B, G) fp32        D (skip), pre-broadcast over B
Outputs: y (B, P, G) bf16, fir_n (2, B, 3, P, G) bf16, s_re_n, s_im_n (B, P, G, K) fp32.
"""
import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def sh1_cascade(z, fir, s_re, s_im, wb, pl, dsk):
    B, three, P, G = z.shape
    K = s_re.shape[3]
    BT = B * 3 * G

    y = nl.ndarray((B, P, G), dtype=z.dtype, buffer=nl.shared_hbm)
    fir_n = nl.ndarray((2, B, 3, P, G), dtype=fir.dtype, buffer=nl.shared_hbm)
    sre_n = nl.ndarray((B, P, G, K), dtype=s_re.dtype, buffer=nl.shared_hbm)
    sim_n = nl.ndarray((B, P, G, K), dtype=s_im.dtype, buffer=nl.shared_hbm)

    # ---- loads: partition = p, free = (b, sec, g) / (b, g, k) ----
    zt = nl.ndarray((P, B, 3, G), dtype=nl.float32, buffer=nl.sbuf)
    f0 = nl.ndarray((P, B, 3, G), dtype=nl.float32, buffer=nl.sbuf)
    f1 = nl.ndarray((P, B, 3, G), dtype=nl.float32, buffer=nl.sbuf)
    zb = nl.ndarray((P, B, 3, G), dtype=z.dtype, buffer=nl.sbuf)
    f1b = nl.ndarray((P, B, 3, G), dtype=fir.dtype, buffer=nl.sbuf)
    # one strided DMA per tensor for ALL rows (partition p, free (b*3+s, g)); per-row DMAs made the kernel
    # slower than the unfused graph at B=48
    PG = P * G
    nisa.dma_copy(dst=zb, src=z.ap(pattern=[[G, P], [PG, B * 3], [1, G]], offset=0))
    nisa.dma_copy(dst=f1b, src=fir.ap(pattern=[[G, P], [PG, B * 3], [1, G]], offset=B * 3 * PG))
    nisa.dma_copy(dst=f0, src=fir.ap(pattern=[[G, P], [PG, B * 3], [1, G]], offset=0))
    nisa.tensor_copy(dst=zt, src=zb)
    nisa.tensor_copy(dst=f1, src=f1b)
    sr = nl.ndarray((P, B, G, K), dtype=nl.float32, buffer=nl.sbuf)
    si = nl.ndarray((P, B, G, K), dtype=nl.float32, buffer=nl.sbuf)
    GK = G * K
    nisa.dma_copy(dst=sr, src=s_re.ap(pattern=[[GK, P], [P * GK, B], [1, GK]], offset=0))
    nisa.dma_copy(dst=si, src=s_im.ap(pattern=[[GK, P], [P * GK, B], [1, GK]], offset=0))
    w = nl.ndarray((P, 4, B, 3, G), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=w, src=wb[0:P, 0:4, 0:B, 0:3, 0:G])
    pp = nl.ndarray((P, 4, B, G, K), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=pp, src=pl[0:P, 0:4, 0:B, 0:G, 0:K])
    dt = nl.ndarray((P, B, G), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=dt, src=dsk[0:P, 0:B, 0:G])

    # ---- FIR k=3 + bias, all 3 sections at once: zf = w2 z + w0 f0 + w1 f1 + bias ----
    zf = nl.ndarray((P, B, 3, G), dtype=nl.float32, buffer=nl.sbuf)
    tmp = nl.ndarray((P, B, 3, G), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=zf, data1=w[0:P, 2], data2=zt, op=nl.multiply)
    nisa.tensor_tensor(dst=tmp, data1=w[0:P, 0], data2=f0, op=nl.multiply)
    nisa.tensor_tensor(dst=zf, data1=zf, data2=tmp, op=nl.add)
    nisa.tensor_tensor(dst=tmp, data1=w[0:P, 1], data2=f1, op=nl.multiply)
    nisa.tensor_tensor(dst=zf, data1=zf, data2=tmp, op=nl.add)
    nisa.tensor_tensor(dst=zf, data1=zf, data2=w[0:P, 3], op=nl.add)

    # ---- column split = free slices; x1v ----
    x1v = nl.ndarray((P, B, G), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=x1v, data1=zf[0:P, 0:B, 1, 0:G], data2=zf[0:P, 0:B, 2, 0:G], op=nl.multiply)

    # ---- modal IIR, K states as real pairs ----
    xk = nl.ndarray((P, B, G, K), dtype=nl.float32, buffer=nl.sbuf)
    for k in nl.static_range(K):
        nisa.tensor_copy(dst=xk[0:P, 0:B, 0:G, k], src=x1v)
    nre = nl.ndarray((P, B, G, K), dtype=nl.float32, buffer=nl.sbuf)
    nim = nl.ndarray((P, B, G, K), dtype=nl.float32, buffer=nl.sbuf)
    t1 = nl.ndarray((P, B, G, K), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=nre, data1=pp[0:P, 0], data2=sr, op=nl.multiply)
    nisa.tensor_tensor(dst=t1, data1=pp[0:P, 1], data2=si, op=nl.multiply)
    nisa.tensor_tensor(dst=nre, data1=nre, data2=t1, op=nl.subtract)
    nisa.tensor_tensor(dst=nre, data1=nre, data2=xk, op=nl.add)
    nisa.tensor_tensor(dst=nim, data1=pp[0:P, 0], data2=si, op=nl.multiply)
    nisa.tensor_tensor(dst=t1, data1=pp[0:P, 1], data2=sr, op=nl.multiply)
    nisa.tensor_tensor(dst=nim, data1=nim, data2=t1, op=nl.add)
    # res = sum_k r_re n_re - r_im n_im
    a = nl.ndarray((P, B, G, K), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=a, data1=pp[0:P, 2], data2=nre, op=nl.multiply)
    nisa.tensor_tensor(dst=t1, data1=pp[0:P, 3], data2=nim, op=nl.multiply)
    nisa.tensor_tensor(dst=a, data1=a, data2=t1, op=nl.subtract)
    res = nl.ndarray((P, B, G), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=res, data=a, op=nl.add, axis=(3,))
    # y = x2 * (res + D x1v)
    acc = nl.ndarray((P, B, G), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=acc, data1=dt, data2=x1v, op=nl.multiply)
    nisa.tensor_tensor(dst=acc, data1=acc, data2=res, op=nl.add)
    yt = nl.ndarray((P, B, G), dtype=z.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=yt, data1=zf[0:P, 0:B, 0, 0:G], data2=acc, op=nl.multiply)

    # ---- stores ----
    nisa.dma_copy(dst=y.ap(pattern=[[G, P], [PG, B], [1, G]], offset=0), src=yt)
    nisa.dma_copy(dst=sre_n.ap(pattern=[[GK, P], [P * GK, B], [1, GK]], offset=0), src=nre)
    nisa.dma_copy(dst=sim_n.ap(pattern=[[GK, P], [P * GK, B], [1, GK]], offset=0), src=nim)
    nisa.dma_copy(dst=fir_n.ap(pattern=[[G, P], [PG, B * 3], [1, G]], offset=0), src=f1b)
    nisa.dma_copy(dst=fir_n.ap(pattern=[[G, P], [PG, B * 3], [1, G]], offset=B * 3 * PG), src=zb)
    return y, fir_n, sre_n, sim_n
