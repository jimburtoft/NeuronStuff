"""Fused NKI kernel for one PhyloGPN RCEByteNet block (same-padding form), Trainium2.

Computes, for x [B, 960, L] (bf16, channels-first):
    h1 = conv1x1_960->480(gelu(LN1(x)))
    h2 = conv_k5_dilation_d_480->480(gelu(LN2(h1)), zero "same" padding)
    y  = x + conv1x1_480->960(gelu(LN3(h2)))

Design:
- Layout: channels on partitions in uniform 120-row tiles (960 = 8 x 120, 480 = 4 x 120); the
  sequence axis is tiled in windows of NCH x 512 columns, with a 4*d halo recomputed per tile.
  All activations of a tile stay in SBUF.
- LayerNorm: the host centres every conv's output channels and the embedding (`model.center_`),
  which is exact because LayerNorm ignores a per-position shift, so every LN input already has
  zero channel mean and only sum(x^2) is needed. The squared channel tiles are pre-summed on the
  Vector/GpSimd engines, then one ones-matmul per 512 columns produces the per-position sum,
  already broadcast to all partitions.
- Convs: bf16 stationary weight tiles, moving activation tiles, fp32 PSUM accumulation over the
  input-channel tiles and the 5 taps (shifted SBUF views, no im2col).
- G interleaved tiles: tiles are processed phase by phase (LN of tile A, LN of tile B, conv of A,
  conv of B, ...) so the Tensor engine always has independent work queued.
- LNC=2: sequence tiles are split across the two programs; no cross-core communication.

One kernel per (dilation, configuration) with a distinct generated name (the NKI trace cache
keys on function name and source).
"""
import nki
import nki.isa as nisa
import nki.language as nl

P = 120
C_OUT = 960
C_IN = 480
NT_O = 8
NT_I = 4
K = 5
EPS = 1e-5
CH = 512

V_LN1G, V_LN1B, V_B1, V_LN2G, V_LN2B, V_B2, V_LN3G, V_LN3B, V_B3 = 0, 8, 16, 20, 24, 28, 32, 36, 40
V_COLS = 48

_ENG = {"vector": nisa.engine.vector, "gpsimd": nisa.engine.gpsimd, "scalar": nisa.engine.scalar}


def _chunks(n):
    out = []
    s = 0
    while s < n:
        c = min(CH, n - s)
        out.append((s, c))
        s += c
    return out


def _square(dst, src, sq_eng):
    if sq_eng == "scalar":
        nisa.activation(dst=dst, op=nl.square, data=src)
    else:
        nisa.tensor_tensor(dst=dst, data1=src, data2=src, op=nl.multiply, engine=_ENG[sq_eng])


def _evict_bias(dst, ps, bias_col, ev_eng, k=0):
    if ev_eng == "alt":
        ev_eng = "act" if k % 2 == 0 else "vector"
    if ev_eng == "act":
        nisa.activation(dst=dst, op=nl.copy, data=ps, bias=bias_col)
    elif ev_eng == "scalar_ts":
        nisa.tensor_scalar(dst=dst, data=ps, op0=nl.add, operand0=bias_col, engine=nisa.engine.scalar)
    else:
        nisa.tensor_scalar(dst=dst, data=ps, op0=nl.add, operand0=bias_col, engine=nisa.engine.vector)


def _ln_gelu(src, nt, W, C, gcol, bcol, vecs_sb, ones_sb, eps_sb, dst, cfg):
    """dst[:, c, :W] = gelu(src * rstd * g + b); src has zero channel mean (centred weights)."""
    sq_eng, rsq, norm_eng, presum = cfg
    chs = _chunks(W)
    ps = []
    for ch in chs:
        ps.append(nl.ndarray((P, ch[1]), dtype=nl.float32, buffer=nl.psum))
    if presum:
        acc = nl.ndarray((P, W), dtype=nl.float32, buffer=nl.sbuf)
        accb = nl.ndarray((P, W), dtype=nl.bfloat16, buffer=nl.sbuf)
        sq0 = nl.ndarray((P, W), dtype=nl.float32, buffer=nl.sbuf)
        _square(sq0, src[:, 0, 0:W], sq_eng)
        for c in range(1, nt):
            sq = nl.ndarray((P, W), dtype=nl.float32, buffer=nl.sbuf)
            _square(sq, src[:, c, 0:W], sq_eng)
            prev = sq0 if c == 1 else acc
            tgt = accb if c == nt - 1 else acc
            nisa.tensor_tensor(dst=tgt, data1=prev, data2=sq, op=nl.add,
                               engine=nisa.engine.gpsimd if (c % 2 == 0) else nisa.engine.vector)
        for j in range(len(chs)):
            c0 = chs[j][0]
            cs = chs[j][1]
            nisa.nc_matmul(dst=ps[j], stationary=ones_sb, moving=accb[:, c0:c0 + cs])
    else:
        for c in range(nt):
            sq = nl.ndarray((P, W), dtype=nl.bfloat16, buffer=nl.sbuf)
            _square(sq, src[:, c, 0:W], sq_eng)
            for j in range(len(chs)):
                c0 = chs[j][0]
                cs = chs[j][1]
                nisa.nc_matmul(dst=ps[j], stationary=ones_sb, moving=sq[:, c0:c0 + cs])
    rstd = nl.ndarray((P, W), dtype=nl.float32, buffer=nl.sbuf)
    for j in range(len(chs)):
        c0 = chs[j][0]
        cs = chs[j][1]
        if rsq == "scalar":
            nisa.activation(dst=rstd[:, c0:c0 + cs], op=nl.rsqrt, data=ps[j], scale=1.0 / C, bias=eps_sb)
        else:
            var = nl.ndarray((P, cs), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=var, data=ps[j], op0=nl.multiply, operand0=1.0 / C, op1=nl.add, operand1=EPS,
                               engine=nisa.engine.vector)
            nisa.tensor_scalar(dst=rstd[:, c0:c0 + cs], data=var, op0=nl.rsqrt, operand0=0.0,
                               engine=nisa.engine.gpsimd)
    for c in range(nt):
        t = nl.ndarray((P, W), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=t, data1=src[:, c, 0:W], data2=rstd, op=nl.multiply, engine=_ENG[norm_eng])
        nisa.activation(dst=dst[:, c, 0:W], op=nl.gelu, data=t,
                        scale=vecs_sb[:, gcol + c:gcol + c + 1], bias=vecs_sb[:, bcol + c:bcol + c + 1])


def _tile_list(B, n_tiles, n_prg, prg):
    out = []
    for b in range(B):
        for t_idx in range(n_tiles):
            if t_idx % n_prg == prg:
                out.append((b, t_idx))
    return out


def block_body(x, w1, w2, w3, vecs, dil, nch, sq_eng, rsq, ev_eng, norm_eng, G, presum):
    B, C, L = x.shape
    out = nl.ndarray((B, C, L), dtype=x.dtype, buffer=nl.shared_hbm)
    cfg = (sq_eng, rsq, norm_eng, presum)

    n_prg = 1
    prg = 0
    if nl.program_ndim() != 0:
        n_prg = nl.num_programs(0)
        prg = nl.program_id(0)

    pad = 2 * dil
    T = nch * CH - 4 * dil
    n_tiles = (L + T - 1) // T

    ones_sb = nl.ndarray((P, P), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=ones_sb, value=1.0)
    eps_sb = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=eps_sb, value=EPS)
    vecs_sb = nl.ndarray((P, V_COLS), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=vecs_sb, src=vecs[0:P, 0:V_COLS])
    w1_sb = nl.ndarray((P, NT_O * C_IN), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=w1_sb, src=w1[0:P, 0:NT_O * C_IN])
    w2_sb = nl.ndarray((P, K * NT_I * C_IN), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=w2_sb, src=w2[0:P, 0:K * NT_I * C_IN])
    w3_sb = nl.ndarray((P, NT_I * C_OUT), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=w3_sb, src=w3[0:P, 0:NT_I * C_OUT])

    tiles = _tile_list(B, n_tiles, n_prg, prg)
    for g0 in range(0, len(tiles), G):
        grp = tiles[g0:g0 + G]
        st = []
        # ---- phase 0: geometry + x window loads ----
        for gi in range(len(grp)):
            b = grp[gi][0]
            t_idx = grp[gi][1]
            o0 = t_idx * T
            Tt = min(T, L - o0)
            w0 = o0 - pad
            W = Tt + 2 * pad
            lo = max(w0, 0)
            hi = min(w0 + W, L)
            jlo = lo - w0
            jhi = hi - w0
            x_sb = nl.ndarray((P, NT_O, W), dtype=nl.bfloat16, buffer=nl.sbuf)
            if jlo > 0 or jhi < W:
                nisa.memset(dst=x_sb, value=0.0)
            for c in range(NT_O):
                nisa.dma_copy(dst=x_sb[:, c, jlo:jhi], src=x[b, c * P:(c + 1) * P, lo:hi])
            st.append(dict(b=b, o0=o0, Tt=Tt, W=W, jlo=jlo, jhi=jhi, x=x_sb, wch=_chunks(W), tch=_chunks(Tt)))
        # ---- phase 1: LN1 + GELU ----
        for gi in range(len(grp)):
            s_ = st[gi]
            g1 = nl.ndarray((P, NT_O, s_["W"]), dtype=nl.bfloat16, buffer=nl.sbuf)
            _ln_gelu(s_["x"], NT_O, s_["W"], C_OUT, V_LN1G, V_LN1B, vecs_sb, ones_sb, eps_sb, g1, cfg)
            s_["g1"] = g1
        # ---- phase 2: conv1 ----
        for gi in range(len(grp)):
            s_ = st[gi]
            wch = s_["wch"]
            h1 = nl.ndarray((P, NT_I, s_["W"]), dtype=nl.bfloat16, buffer=nl.sbuf)
            for co in range(NT_I):
                ps = []
                for ch in wch:
                    ps.append(nl.ndarray((P, ch[1]), dtype=nl.float32, buffer=nl.psum))
                for ci in range(NT_O):
                    col = ci * C_IN + co * P
                    for j in range(len(wch)):
                        c0 = wch[j][0]
                        cs = wch[j][1]
                        nisa.nc_matmul(dst=ps[j], stationary=w1_sb[:, col:col + P], moving=s_["g1"][:, ci, c0:c0 + cs])
                for j in range(len(wch)):
                    c0 = wch[j][0]
                    cs = wch[j][1]
                    _evict_bias(h1[:, co, c0:c0 + cs], ps[j], vecs_sb[:, V_B1 + co:V_B1 + co + 1], ev_eng, co + j)
            s_["h1"] = h1
        # ---- phase 3: LN2 + GELU + edge zeroing ----
        for gi in range(len(grp)):
            s_ = st[gi]
            W = s_["W"]
            g2 = nl.ndarray((P, NT_I, W), dtype=nl.bfloat16, buffer=nl.sbuf)
            _ln_gelu(s_["h1"], NT_I, W, C_IN, V_LN2G, V_LN2B, vecs_sb, ones_sb, eps_sb, g2, cfg)
            if s_["jlo"] > 0:
                for c in range(NT_I):
                    nisa.memset(dst=g2[:, c, 0:s_["jlo"]], value=0.0)
            if s_["jhi"] < W:
                for c in range(NT_I):
                    nisa.memset(dst=g2[:, c, s_["jhi"]:W], value=0.0)
            s_["g2"] = g2
        # ---- phase 4: conv2 (dilated) ----
        for gi in range(len(grp)):
            s_ = st[gi]
            tch = s_["tch"]
            h2 = nl.ndarray((P, NT_I, s_["Tt"]), dtype=nl.bfloat16, buffer=nl.sbuf)
            for co in range(NT_I):
                ps = []
                for ch in tch:
                    ps.append(nl.ndarray((P, ch[1]), dtype=nl.float32, buffer=nl.psum))
                for k in range(K):
                    for ci in range(NT_I):
                        col = (k * NT_I + ci) * C_IN + co * P
                        for j in range(len(tch)):
                            c0 = tch[j][0]
                            cs = tch[j][1]
                            nisa.nc_matmul(dst=ps[j], stationary=w2_sb[:, col:col + P],
                                           moving=s_["g2"][:, ci, k * dil + c0:k * dil + c0 + cs])
                for j in range(len(tch)):
                    c0 = tch[j][0]
                    cs = tch[j][1]
                    _evict_bias(h2[:, co, c0:c0 + cs], ps[j], vecs_sb[:, V_B2 + co:V_B2 + co + 1], ev_eng, co + j)
            s_["h2"] = h2
        # ---- phase 5: LN3 + GELU ----
        for gi in range(len(grp)):
            s_ = st[gi]
            g3 = nl.ndarray((P, NT_I, s_["Tt"]), dtype=nl.bfloat16, buffer=nl.sbuf)
            _ln_gelu(s_["h2"], NT_I, s_["Tt"], C_IN, V_LN3G, V_LN3B, vecs_sb, ones_sb, eps_sb, g3, cfg)
            s_["g3"] = g3
        # ---- phase 6: conv3 + bias + residual, store ----
        for gi in range(len(grp)):
            s_ = st[gi]
            tch = s_["tch"]
            Tt = s_["Tt"]
            for co in range(NT_O):
                ps = []
                for ch in tch:
                    ps.append(nl.ndarray((P, ch[1]), dtype=nl.float32, buffer=nl.psum))
                for ci in range(NT_I):
                    col = ci * C_OUT + co * P
                    for j in range(len(tch)):
                        c0 = tch[j][0]
                        cs = tch[j][1]
                        nisa.nc_matmul(dst=ps[j], stationary=w3_sb[:, col:col + P], moving=s_["g3"][:, ci, c0:c0 + cs])
                y = nl.ndarray((P, Tt), dtype=nl.bfloat16, buffer=nl.sbuf)
                for j in range(len(tch)):
                    c0 = tch[j][0]
                    cs = tch[j][1]
                    nisa.scalar_tensor_tensor(dst=y[:, c0:c0 + cs], data=ps[j], op0=nl.add,
                                              operand0=vecs_sb[:, V_B3 + co:V_B3 + co + 1],
                                              op1=nl.add, operand1=s_["x"][:, co, pad + c0:pad + c0 + cs])
                nisa.dma_copy(dst=out[s_["b"], co * P:(co + 1) * P, s_["o0"]:s_["o0"] + Tt], src=y)
    return out


_SRC = """
def {name}(x, w1, w2, w3, vecs):
    return block_body(x, w1, w2, w3, vecs, {d}, {nch}, "{sq}", "{rsq}", "{ev}", "{norm}", {acc}, {presum})
"""

DEFAULT_CFG = dict(nch=2, sq="scalar", rsq="scalar", ev="vector", norm="vector", acc="2p")
_CACHE = {}


def make_kernel(d, nch, sq, rsq, ev, norm, acc="2"):
    """Build (and cache) the kernel for dilation d. acc = "<G>[p]": G interleaved tiles, p = LN pre-sum."""
    presum = str(acc).endswith("p")
    G = int(str(acc).rstrip("p"))
    key = (d, nch, sq, rsq, ev, norm, G, presum)
    if key not in _CACHE:
        name = f"phylogpn_block_d{d}_n{nch}_{sq}_{rsq}_{ev}_{norm}_g{G}{'p' if presum else ''}"
        ns = {"block_body": block_body}
        exec(_SRC.format(name=name, d=d, nch=nch, sq=sq, rsq=rsq, ev=ev, norm=norm, acc=G, presum=presum), ns)
        _CACHE[key] = nki.jit(ns[name])
    return _CACHE[key]



def prepare_block_weights(block):
    """Block -> (w1, w2, w3, vecs) host tensors in the kernel layout (CPU, fp32 math)."""
    import torch
    W1 = block.conv1.weight.detach().float()[:, :, 0]  # [480, 960]
    W2 = block.conv2.weight.detach().float()           # [480, 480, 5]
    W3 = block.conv3.weight.detach().float()[:, :, 0]  # [960, 480]
    # w1[p, ci*480 + co] = W1[co, ci*120 + p]
    w1 = W1.t().reshape(NT_O, P, C_IN).permute(1, 0, 2).reshape(P, NT_O * C_IN)
    # w2[p, (k*4+ci)*480 + co] = W2[co, ci*120+p, k]
    w2 = W2.permute(2, 1, 0).reshape(K, NT_I, P, C_IN).permute(2, 0, 1, 3).reshape(P, K * NT_I * C_IN)
    w3 = W3.t().reshape(NT_I, P, C_OUT).permute(1, 0, 2).reshape(P, NT_I * C_OUT)

    def cols(v):
        n = v.numel() // P
        return v.detach().float().reshape(n, P).t()

    vecs = torch.cat([cols(block.layer_norm1.weight), cols(block.layer_norm1.bias), cols(block.conv1.bias),
                      cols(block.layer_norm2.weight), cols(block.layer_norm2.bias), cols(block.conv2.bias),
                      cols(block.layer_norm3.weight), cols(block.layer_norm3.bias), cols(block.conv3.bias)], 1)
    assert vecs.shape == (P, V_COLS)
    return (w1.contiguous(), w2.contiguous(), w3.contiguous(), vecs.contiguous())




OPS = {}

INFERENCE_ONLY_MSG = (
    "The PhyloGPN NKI block kernel is inference-only (no backward). Run under torch.no_grad() / "
    "torch.inference_mode(), or for training build the model without the kernel (Scorer.build(kernel=False), or "
    "phylogpn_neuron.load() without model.use_kernel()), which uses the stock TorchNeuron graph with autograd."
)


def _inference_only_backward(ctx, grad_out):
    raise RuntimeError(INFERENCE_ONLY_MSG)


def register_torch_ops(cfg=None):
    import os
    import torch
    from torch_neuronx import nki_op, wrap_nki

    cfg = dict(DEFAULT_CFG, **(cfg or {}))
    lnc = int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2"))
    tag = f"n{cfg['nch']}_{cfg['sq']}_{cfg['rsq']}_{cfg['ev']}_{cfg['norm']}_g{cfg['acc']}"
    for d in (1, 5):
        kern = make_kernel(d, cfg["nch"], cfg["sq"], cfg["rsq"], cfg["ev"], cfg["norm"], cfg["acc"])
        caller = wrap_nki(kern)[lnc]

        def _mk(caller=caller, d=d):
            @nki_op(f"phylogpn::block_d{d}_{tag}", mutates_args={})
            def _op(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, w3: torch.Tensor,
                    vecs: torch.Tensor) -> torch.Tensor:
                return caller(x, w1, w2, w3, vecs)

            _op.register_autograd(_inference_only_backward)
            return _op

        OPS[d] = _mk()
    return OPS
