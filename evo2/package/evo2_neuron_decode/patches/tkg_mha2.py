"""attention_tkg for vortex MHA, HEAD-FOLDED. Rewrite of the rejected tkg_mha_patch.

HEAD-FOLD DEFECT (fixed here): attention_tkg's k_prior/v_prior are shaped for ONE KV head.
Evo 2 is MHA with 32 DISTINCT KV heads. The old patch sliced head 0 and fed it as "the"
KV head, dropping 31 heads -> NaN. Fix: fold H into the batch axis:
    cfg.bs = B*H, cfg.q_head = 1
    q, k_active, v_active, k_prior, v_prior all [B*H, 1, s, d]
Validated: cos_sim(NKI, torch_ref) = 0.999997 at real shapes (bench/tkg_headfold.py).

ROPE SEMANTICS (settled empirically, same harness):
  - the kernel rotates ONLY k_active; it does NOT rotate k_prior
    (max|out(raw prior) - out(pre-rotated prior)| = 1.41 -> they differ)
  - k_out comes back ALREADY RoPE'd (max|k_out - rope(k_active)| = 0.0)
  => store k_out (rotated) into the cache. Prior must be rotated-at-write-time.
     So NO double-RoPE: each key is rotated exactly once, at its own position.
  - q is rotated internally too, and scaled by 1/sqrt(d) (fuse_rope=True)

PERF: SbufManager + all buffers hoisted to module state. Allocating per call cost ~450x
(a measured correction).

BS>4 VIA BH TILING: attention_tkg enforces NCC_INKI016,
`bs * q_head * s_active <= 128`. With the head fold cfg.bs = B*H = B*32 (q_head=1,
s_active=1), that caps B at 4. But the folded BH axis is a pure batch axis: chunk bh is
independent of chunk bh' -- no cross-chunk softmax reduction, no shared state. So we tile
BH into ceil(BH/128) chunks of <=128 and issue one kernel call per chunk, concatenating the
outputs. Buffers and one cfg per distinct chunk size are cached on the module.
At BS=32: BH=1024 -> 8 calls of 128. Removes the ~1.38x eager-MHA penalty at BS>=8.
"""
import os

import torch

# Kernel-native KV layout. The default path stores the cache in vortex's
# (B, SP, 2, H, D) layout and, EVERY decode step, does
#     kp = cache[:B,:,0].permute(0,2,1,3).reshape(BH,1,SP,D); kpc = kp.contiguous()
# The permute makes the view non-contiguous, so .contiguous() materialises a FULL copy of
# K and of V for the layer, every step. At max_seqlen=8192, B=8 that is ~1 GiB per layer.
# With EVO2_KV_NATIVE=1 the cache is converted ONCE (at first decode) into the layout the
# kernel consumes, (BH, 1, SP, D), and vortex's copy is released, so the per-step copy
# disappears and persistent KV stays at 1x.
#
# DEFAULT ON. Measured on torch-neuronx 2.13, B=4 @ max_seqlen=8192, three alternating reps,
# all after a passing timing gate: 126.608 -> 93.889 ms/step median = 1.348x
# (31.6 -> 42.6 tok/s/core); arms do not overlap. Step-1 decode logits bit-identical
# (sum -57295.843750, l2 1277.982422 both ways). Set EVO2_KV_NATIVE=0 to restore the old path.
# Does NOT raise the 8K batch ceiling (B=8 @ 8192 still OOMs) -- that is a separate problem.
_KV_NATIVE = os.environ.get("EVO2_KV_NATIVE", "1") == "1"

# K cache stored PRE-TRANSPOSED, (BH, 1, D, SP), with attention_tkg tp_k_prior=False.
# Measured on torch-neuronx 2.13, SP=8192, 128-row chunks, one core:
#   tp_k_prior=True  (K as (BH,1,SP,D)): 1.16 ms/call on the same K, but 7.06 ms/call when
#                    consecutive calls read DIFFERENT K tensors -- which is every call in the
#                    model (2 chunks x 5 layers). The penalty scales with SP (2048: 0.55 -> 4.4
#                    ms per alternating pair) and is triggered by K alone (switching only V or q
#                    costs nothing).
#   tp_k_prior=False (K as (BH,1,D,SP)): 1.09-1.10 ms/call, switching or not.
#   Outputs and k_out BIT-IDENTICAL between the two (max |diff| 0.0).
# The kernel's k_out is already (c,1,D,1), so the K write needs no permute in this layout.
# Requires EVO2_KV_NATIVE=1. EVO2_KT_LAYOUT=0 restores tp_k_prior=True.
_KT_LAYOUT = os.environ.get("EVO2_KT_LAYOUT", "1") == "1"

# The one-slot KV write. Eager index_copy_ on a (BH,1,SP,D) tensor costs the rate of
# the WHOLE tensor (1.72 ms at BH=256, SP=8192; 0.88 ms at half the rows). The same op under
# torch.compile costs 0.09-0.10 ms and writes exactly the target slot (verified on device:
# target slot written, neighbouring slot untouched). EVO2_COMPILED_KV_WRITE=0 disables.
_COMPILED_WRITE = os.environ.get("EVO2_COMPILED_KV_WRITE", "1") == "1"
_write_fn = None


def _kv_write(T, dim, idx, x):
    global _write_fn
    if not _COMPILED_WRITE:
        T.index_copy_(dim, idx, x)
        return
    if _write_fn is None:
        import torch as _t

        def _w(T, dim: int, idx, x):
            T.index_copy_(dim, idx, x)
        _write_fn = _t.compile(_w, backend="neuron", dynamic=False)
    _write_fn(T, dim, idx, x)


# Leaner attention glue, INLINE in forward() (a helper function changed Dynamo's
# graph boundaries: per-step recompiles on layer_idx/offset guards, ~1 s/step, and different
# numerics even with every change disabled). Each piece is separately switchable:
#   EVO2_LEAN_BUFS=1  one out/kout buffer pair per chunk, so the per-chunk .clone()s go
#   EVO2_LEAN_QKV1=1  q/k/v from ONE (3,B,H,S,D) contiguous copy instead of three
# DEFAULT ON. TP=4 @ 8K, gated: B=32 44.63 -> 41.17 ms (3 pairs, 1.084x), B=48 54.24 -> 50.12 ms
# (1.082x). Tokens (48-step SHA), step-1 and late logits identical on two prompts.
# Rejected in the same experiment: torch.compile around glue+kernel calls (wrap_nki graph-breaks,
# 2-5 s/step); a compiled K+V write and V-from-v_in (both changed numerics via graph boundaries).
_LEAN_BUFS = os.environ.get("EVO2_LEAN_BUFS", "1") == "1"
_LEAN_QKV1 = os.environ.get("EVO2_LEAN_QKV1", "1") == "1"

# Attention as a registered nki_op called inside ONE torch.compile region per layer
# (fullgraph=True, zero graph breaks). Two facts from the on-device probe:
#   - a bare wrap_nki call inside torch.compile graph-breaks ("Failed to convert args/kwargs to
#     proxy" -- cfg/sbm are not tensors); behind nki_op (cfg/sbm closed over, only tensors in the
#     schema) Dynamo traces it with 0 breaks.
#   - passing a SLICED view of the KV cache into the compiled op gives WRONG output (max|diff|
#     0.056) while kout stays right; contiguous per-chunk tensors are bit-identical. So with this
#     flag the KV cache is stored as one contiguous (c,1,D,SP)/(c,1,SP,D) tensor PER CHUNK.
# Op registration and torch.compile-object creation MUST stay inline in forward(): moving them
# into a helper (plain or torch._dynamo.disable'd) made steps 4-14 s with ~400 recompiles and
# changed numerics (two separate attempts), the same failure as the lean-glue helper refactor.
# EVO2_ATTN_OP=1 enables. Requires KV_NATIVE + KT_LAYOUT.
# DEFAULT ON. TP=4 @ 8K, gated, alternating: B=32 41.26/40.94/40.34/40.71 -> 38.81/38.87/38.35/38.46 ms
# (1.06x, 4 pairs); B=48 49.81/50.29 -> 46.93/47.25 ms (1.06x). Tokens, step-1 and late logits
# identical on two prompts. In-step graph breaks 1 -> 0 (TORCH_LOGS census).
_ATTN_OP = os.environ.get("EVO2_ATTN_OP", "1") == "1"
_POS_IN_GRAPH = os.environ.get("EVO2_POS_IN_GRAPH", "1") == "1"
_DIRECT_CHUNKS = os.environ.get("EVO2_KV_DIRECT_CHUNKS", "1") == "1"
_REGION_CHUNKS = int(os.environ.get("EVO2_ATTN_REGION_CHUNKS", "0"))
_EAGER_ON = [""]   # set by install(): eager dispatcher active -> fuse Wqkv/out_proj into the layer graph
_OPS = {}
# Read only the filled KV prefix. attention_tkg's curr_sprior is a compile-time size,
# so each power-of-two bucket >= seqlen_offset gets its own cfg / nki_op / compiled layer fn
# (built lazily, once per bucket per layer). At bucket == max_seqlen the path is exactly the
# unbucketed nki_op one. Micro (one chunk): 8192/4096/2048/1024/512 = 1.091/0.696/0.536/0.494/
# 0.488 ms/call; output differs from 8192 at <=5e-4 abs (different FA tiling), NOT bit-identical.
# Originally opt-in (default off). TP=4 B=64 @ 8K, gated: fill 500 46.1-46.4 -> 33.6-33.7 ms,
# 1000 46.9-47.0 -> 33.7-34.5, 2000 46.5-47.3 -> 35.7-35.9, 4000 46.6-47.1 -> 38.8-39.4, 8000
# unchanged. Every bucket is pre-built at the first bucketed step (otherwise crossing a bucket
# boundary compiled inside the decode loop: ~1.95 s/step). NUMERICS: logits vs unbucketed rel_l2
# 1.5e-3..9.6e-3, max|diff| 0.125-0.33 (bf16-vs-fp32 logit noise is 0.64); 24-step tokens
# identical at fills 500/1000/2000 and at real 600/1000-nt prompts, DIFFERENT at fill 4000.
# Now DEFAULT ON, after this check: fp32 CPU reference, teacher-forced 199 decode steps at fill 900-1100,
# TP=4: bucketed vs fp32 KL 0.002169 / rel_l2 0.0059 / top-1 100%, unbucketed 0.002177 / 0.0061 /
# 100% -- bucketed is no further from fp32 (marginally closer); bucketed-vs-unbucketed (rel_l2
# 0.0019) is ~3x smaller than either path's distance to fp32. EVO2_ATTN_BUCKET=0 disables.
_BUCKET = os.environ.get("EVO2_ATTN_BUCKET", "1") == "1"
_BMIN = int(os.environ.get("EVO2_ATTN_BUCKET_MIN", "512"))
_PREWARM = os.environ.get("EVO2_ATTN_BUCKET_PREWARM", "1") == "1"


def _bucket_list(SP):
    out, b = [], _BMIN
    while b < SP:
        out.append(b); b *= 2
    return out


def _get_op(KERN, cfg, sbm, D):
    key = id(cfg)
    if key in _OPS:
        return _OPS[key]
    import torch as _t
    from torch_neuronx.nki_hop import nki_op

    @nki_op("evo2::attn_tkg_c%d_%d" % (len(_OPS), key % 100000), mutates_args={})
    def op(q_: _t.Tensor, k_: _t.Tensor, v_: _t.Tensor, kp: _t.Tensor, vp: _t.Tensor,
           m_: _t.Tensor, inv_: _t.Tensor, pos_: _t.Tensor) -> tuple[_t.Tensor, _t.Tensor]:
        o = _t.empty(q_.shape[0], 1, D, q_.shape[2], dtype=q_.dtype, device=q_.device)
        ko = _t.empty(q_.shape[0], 1, D, q_.shape[2], dtype=q_.dtype, device=q_.device)
        r = KERN(q_, k_, v_, kp, vp, m_, o, cfg, sbm, inv_, pos_, None, None, None, ko)
        r = r[0] if isinstance(r, (tuple, list)) else r
        return r, ko
    _OPS[key] = op
    return op


def _make_layer_fn(ops, sizes, B, H, D, S, wqkv=None, oproj=None):
    import torch as _t
    BH = B * H

    def layer(qkv, kTs, vTs, masks, inv, pos, idx):
        if _POS_IN_GRAPH:
            # Positions from idx inside the graph. The eager per-step pos.fill_(off)
            # was re-lowered every step (torch_neuronx_lower ~20 ms) and, at single-core B=8,
            # sat behind a ~1.4 s RuntimeWorker sleep before executing (device idle) -> the
            # "EVO2_ATTN_OP=1 regresses single core" issue.
            pf = idx.to(_t.float32).view(1, 1)
            pos = [pf.expand(c, S).contiguous() for c in sizes]
        t3 = qkv.view(B, S, 3, H, D).permute(2, 0, 3, 1, 4).contiguous()          # (3,B,H,S,D)
        q_in = t3[0].reshape(BH, 1, S, D); k_in = t3[1].reshape(BH, 1, S, D)
        v_in = t3[2].reshape(BH, 1, S, D)
        outs = []; base = 0
        for j, c in enumerate(sizes):
            sl = slice(base, base + c)
            o, ko = ops[j](q_in[sl], k_in[sl], v_in[sl], kTs[j], vTs[j], masks[j], inv, pos[j])
            kTs[j].index_copy_(3, idx, ko)
            vTs[j].index_copy_(2, idx, v_in[sl])
            outs.append(o); base += c
        o = outs[0] if len(outs) == 1 else _t.cat(outs, 0)
        return o.reshape(B, H, D, S).permute(0, 3, 1, 2).reshape(B, S, H * D)

    if wqkv is not None:
        # Under the eager dispatcher (torch 2.14) Wqkv / out_proj -- and out_proj's TP
        # all_reduce -- would run as separate eager ops (profile: eager aten::mm + allreduce between
        # graphs). Keep them inside this layer's compiled region, as on torch-neuronx 2.13.
        if wqkv == "eager":
            def fused(qkv, kTs, vTs, masks, inv, pos, idx):      # out_proj only (EVO2_ATTN_FUSE_PROJ=out)
                return oproj(layer(qkv, kTs, vTs, masks, inv, pos, idx))
        else:
            def fused(x, kTs, vTs, masks, inv, pos, idx):
                qkv = wqkv(x).view(B, S, 3, H, D)
                return oproj(layer(qkv, kTs, vTs, masks, inv, pos, idx))

        return _t.compile(fused, backend="neuron", fullgraph=True, dynamic=False)
    if _REGION_CHUNKS <= 0 or len(sizes) <= _REGION_CHUNKS:
        return _t.compile(layer, backend="neuron", fullgraph=True, dynamic=False)

    # grouped: one compiled region per group of <= _REGION_CHUNKS chunks
    groups = []
    for g0 in range(0, len(sizes), _REGION_CHUNKS):
        groups.append(list(range(g0, min(g0 + _REGION_CHUNKS, len(sizes)))))
    offs = [sum(sizes[:j]) for j in range(len(sizes))]

    def part(q_in, k_in, v_in, kTs, vTs, masks, inv, pf, idx, js):
        outs = []
        for j in js:
            c = sizes[j]; sl = slice(offs[j], offs[j] + c)
            pos = pf.expand(c, S).contiguous()
            o, ko = ops[j](q_in[sl], k_in[sl], v_in[sl], kTs[j], vTs[j], masks[j], inv, pos)
            kTs[j].index_copy_(3, idx, ko)
            vTs[j].index_copy_(2, idx, v_in[sl])
            outs.append(o)
        return outs[0] if len(outs) == 1 else _t.cat(outs, 0)

    cparts = [_t.compile(lambda *a, _js=tuple(js): part(*a, _js), backend="neuron", fullgraph=True, dynamic=False)
              for js in groups]

    def grouped(qkv, kTs, vTs, masks, inv, pos, idx):
        t3 = qkv.view(B, S, 3, H, D).permute(2, 0, 3, 1, 4).contiguous()
        q_in = t3[0].reshape(BH, 1, S, D); k_in = t3[1].reshape(BH, 1, S, D); v_in = t3[2].reshape(BH, 1, S, D)
        pf = idx.to(_t.float32).view(1, 1)
        outs = [cp(q_in, k_in, v_in, kTs, vTs, masks, inv, pf, idx) for cp in cparts]
        o = _t.cat(outs, 0)
        return o.reshape(B, H, D, S).permute(0, 3, 1, 2).reshape(B, S, H * D)

    return grouped


_orig_fwd = None
SBUF_BYTES = 24 * 1024 * 192


def install(lnc: int = 2):
    global _orig_fwd
    from vortex.model import attention as A
    from nkilib.core.attention.attention_tkg import attention_tkg
    from nkilib.core.attention.attention_tkg_utils import AttnTKGConfig
    from nkilib.core.utils.allocator import SbufManager
    from torch_neuronx import wrap_nki

    _orig_fwd = A.MHA.forward
    KERN = wrap_nki(attention_tkg)[(lnc,)]

    def forward(self, x, seqlen=None, inference_params=None, **kwargs):
        return _forward_eager(self, x, seqlen=seqlen, inference_params=inference_params, **kwargs)

    # SDK 2.33 (torch 2.14): the attention block is compiled by compile_blocks(), and Dynamo 2.14 now
    # guards on the Python value of seqlen_offset read here, so every decode step recompiled the whole
    # attention block (1,248 recompiles in 30 steps, ~3 s/step at TP=4 B=64 vs ~30 ms on torch 2.13).
    # The decode body only needs `off` for host-side bookkeeping (bucket choice, the KV slot index,
    # which it passes as a tensor); the per-layer compute is its own fullgraph region (_make_layer_fn).
    # So run this dispatcher outside Dynamo; Wqkv / out_proj stay compiled inside the region they are
    # called from. EVO2_ATTN_EAGER_DISPATCH=0 restores the old (torch-neuronx 2.13) behaviour.
    import torch._dynamo as _dyn

    def _forward_impl(self, x, seqlen=None, inference_params=None, **kwargs):
        off = 0 if inference_params is None else int(getattr(inference_params, "seqlen_offset", 0))
        is_decode = (inference_params is not None and off > 0 and x.shape[1] == 1
                     and self.layer_idx in inference_params.key_value_memory_dict)
        if not is_decode:
            if off == 0 and getattr(self, "_kvnat", None) is not None:
                # A new prefill starts. Drop the previous request's kernel-layout KV
                # NOW, before vortex allocates the new cache; otherwise both are alive until the
                # first decode step (~2x KV). At TP=4 B=64 that made the second request OOM.
                self._kvnat = None
            return _orig_fwd(self, x, seqlen=seqlen, inference_params=inference_params, **kwargs)

        B, S, _ = x.shape
        H, D = self.num_heads, self.head_dim
        BH = B * H
        cache = inference_params.key_value_memory_dict[self.layer_idx]   # (B, SP, 2, H, D)
        # In native mode, after the first decode step vortex's entry is a 0-element stub;
        # SP must then come from the native tensors, which are (BH, 1, SP, D).
        _nat0 = getattr(self, "_kvnat", None) if _KV_NATIVE else None
        if _nat0 is not None and _nat0["src"] is cache:
            SP = (_nat0["vT"] if _nat0.get("vT") is not None else _nat0["vTs"][0]).shape[2]
        else:
            SP = cache.shape[1]

        _fmode = _EAGER_ON[0] if (_ATTN_OP and _KT_LAYOUT and _KV_NATIVE and _REGION_CHUNKS <= 0) else ""
        _fused = _fmode == "1"                       # Wqkv + out_proj in the layer graph
        _fz = (dict(wqkv=self.Wqkv, oproj=self.out_proj) if _fused else
               dict(wqkv="eager", oproj=self.out_proj) if _fmode == "out" else {})
        qkv = None if _fused else self.Wqkv(x).view(B, S, 3, H, D)

        # attention_tkg needs SP (== max_seqlen) divisible by 256, NOT 128.
        # The kernel shards s_prior by 2 and the SHARD must be a multiple of p_max=128, so
        # the requirement on max_seqlen is 256. Derived from the error text, not guessed:
        #   max_seqlen=1152 -> "Got sharded s_prior=576, p_max=128"  (576 % 128 == 64)
        # Verified against all 6 observed cases: 256/512/1024/2048 pass; 1152/4224 fail; and
        # the earlier ad-hoc failures (sharded 137/129/193/257) correspond to SP=274/258/386/514,
        # which the harness produces from prompt_len+steps+warmup+16 -- none divisible by 256.
        # Without this guard the user gets a raw neuronx-cc NCC_INKI016 with no guidance.
        if SP % 256 != 0:
            raise ValueError(
                f"attention_tkg requires max_seqlen divisible by 256 (got {SP}). It shards "
                f"s_prior by 2 and the shard must be a multiple of p_max=128, so max_seqlen "
                f"must be a multiple of 256; otherwise neuronx-cc fails with NCC_INKI016 "
                f"'Sharded s_prior must be divisible by p_max' (yours would shard to {SP // 2}). "
                f"Use --max-seqlen {((SP + 255) // 256) * 256}, or EVO2_TKG=0 for eager MHA.")
        MAXBH = 128 // (1 * S)      # NCC_INKI016: bs*q_head*s_active <= 128
        # EVO2_ATTN_REGION_CHUNKS: max attention_tkg calls per compiled layer region (0 = all in one).
        # SDK 2.33 diagnostic: the one-region-per-layer graph (EVO2_ATTN_OP) fails NCC_INLA001 on first
        # compile for some (B, fill) shapes at TP=4.
        # Cache the buffer set PER BATCH SIZE, not once per module.
        # `sizes` is derived from BH = B*H, so a single cached `st` is only valid for the B it
        # was built with. Sweeping batch sizes in ONE process (BS=1 -> 4 -> 8 -> 32) reused the
        # stale chunk list and produced
        #     RuntimeError: shape '[4, 32, 128, 1]' is invalid for input of size 4096
        # from the kout_all reshape -- a real defect, not a harness artifact: any server that
        # varies batch size in-process would hit it.
        # ONE SbufManager per MODULE, shared across batch sizes. A first fix keyed the
        # whole state dict on B, which also created a fresh SbufManager(0, SBUF_BYTES) per batch
        # size -- each claiming the entire budget. Sweeping BS=1,4,8 then 32 in one process
        # therefore exhausted SBUF and BS=32 died with
        #   [NCC_INLA001] Allocated memory out of bound {_mem_3_4}@SB<0,0>(128x262144)
        # even though BS=32 ALONE compiles fine. Only the per-B buffers/cfg may differ.
        sbm_shared = getattr(self, "_tkg2_sbm", None)
        if sbm_shared is None:
            from nkilib.core.utils.allocator import SbufManager as _SbufMgr
            sbm_shared = _SbufMgr(0, SBUF_BYTES)
            sbm_shared.open_scope()
            self._tkg2_sbm = sbm_shared

        cache_key = "_tkg2_b%d" % B
        st = getattr(self, cache_key, None)
        if st is None:
            # *** ROOT CAUSE of an earlier NaN defect ***
            # vortex allocates the KV cache with torch.empty (attention.py:357) -- the
            # unwritten tail is UNINITIALISED memory and contains inf/NaN. Stock code
            # never reads it because _update_kv_cache returns kv_cache[:, :sequence_end].
            # attention_tkg reads the FULL s_prior window and relies on the position mask,
            # so garbage in the tail poisons the softmax (-inf/NaN) even though it is
            # "masked". Zero the cache once, at first decode use.
            # Zero ONLY the unwritten tail: slots [off, SP) have never been written.
            # cache.zero_() here would WIPE THE PREFILL's keys/values (slots [0, off)).
            cache[:, off:].zero_()
            dev, dt = x.device, x.dtype
            base = float(getattr(getattr(self, "rotary_emb", None), "base", 10000.0))
            # one cfg + buffer set per DISTINCT chunk size (BH may not divide evenly)
            sizes = []
            rem = BH
            while rem > 0:
                c = min(MAXBH, rem); sizes.append(c); rem -= c
            st = {
                "sizes": sizes,
                "sbm": sbm_shared,
                "inv": (1.0 / (base ** (torch.arange(0, D, 2, device=dev).float() / D))
                        ).reshape(D // 2, 1).contiguous(),
                "per": {},
            }
            for c in set(sizes):
                st["per"][c] = {
                    "cfg": AttnTKGConfig(bs=c, q_head=1, s_active=S, curr_sprior=SP,
                                         full_sprior=SP, d_head=D, block_len=0,
                                         tp_k_prior=not (_KV_NATIVE and _KT_LAYOUT),
                                         use_pos_id=True, fuse_rope=True),
                    "pos": torch.zeros(c, S, dtype=torch.float32, device=dev),
                    "mask": torch.ones(S, c, 1, S, dtype=dt, device=dev),   # 1 = KEEP
                    "out": torch.zeros(c, 1, D, S, dtype=dt, device=dev),
                    "kout": torch.zeros(c, 1, D, S, dtype=dt, device=dev),
                }
            setattr(self, cache_key, st)

        # ---- head-folded views: (B,SP,2,H,D) -> (B*H, 1, SP, D) ----
        if _KV_NATIVE:
            # Build the kernel-native pair ONCE per cache allocation, then free vortex's
            # copy. Keyed on the identity of vortex's tensor so a fresh prefill (new
            # request, new allocation) triggers a rebuild rather than reusing stale K/V.
            nat = getattr(self, "_kvnat", None)
            if (nat is None or nat["src"] is not cache or nat["B"] != B) and _ATTN_OP and _KT_LAYOUT \
                    and _DIRECT_CHUNKS:
                # Build the per-chunk kernel layout DIRECTLY from vortex's cache, one chunk
                # at a time, and release vortex's tensor before returning. The previous path held
                # vortex K+V + full kT/vT + per-chunk copies at once (6x one K) while the other 4
                # layers still held vortex's caches; at B=80 that transient peak (~22 GB/core) failed
                # allocating 1.34 GB although the steady state (16.7 GB/core) fits. Values identical.
                kTs, vTs, b0 = [], [], 0
                Bc = cache.shape[0]
                for c in st["sizes"]:
                    r0, r1 = b0 // H, (b0 + c) // H          # chunks are whole rows (c % H == 0)
                    kc = cache[r0:r1, :, 0].permute(0, 2, 3, 1).reshape(c, 1, D, SP).contiguous()
                    kc[:, :, :, off:].zero_()
                    vc = cache[r0:r1, :, 1].permute(0, 2, 1, 3).reshape(c, 1, SP, D).contiguous()
                    vc[:, :, off:].zero_()
                    kTs.append(kc); vTs.append(vc); b0 += c
                stub = torch.empty(0, device=cache.device, dtype=cache.dtype)
                inference_params.key_value_memory_dict[self.layer_idx] = stub
                del cache
                nat = {"src": stub, "B": B, "kT": None, "vT": None, "kTs": kTs, "vTs": vTs}
                self._kvnat = nat
            if nat is None or nat["src"] is not inference_params.key_value_memory_dict[self.layer_idx] \
                    or nat["B"] != B:
                cache = inference_params.key_value_memory_dict[self.layer_idx]
                if _KT_LAYOUT:
                    # (B,SP,H,D) -> (B,H,D,SP) -> (BH,1,D,SP): K pre-transposed (EVO2_KT_LAYOUT)
                    kT = cache[:B, :, 0].permute(0, 2, 3, 1).reshape(BH, 1, D, SP).contiguous()
                    kT[:, :, :, off:].zero_()
                else:
                    kT = cache[:B, :, 0].permute(0, 2, 1, 3).reshape(BH, 1, SP, D).contiguous()
                    kT[:, :, off:].zero_()
                vT = cache[:B, :, 1].permute(0, 2, 1, 3).reshape(BH, 1, SP, D).contiguous()
                # The uninitialised-tail NaN fix applies here too: slots [off, SP) were never written.
                vT[:, :, off:].zero_()
                # Release vortex's storage but KEEP THE DICT KEY: the is_decode guard tests
                # `layer_idx in key_value_memory_dict`, so deleting the entry would silently
                # route every later step back to vortex's eager path.
                stub = torch.empty(0, device=cache.device, dtype=cache.dtype)
                inference_params.key_value_memory_dict[self.layer_idx] = stub
                nat = {"src": stub, "B": B, "kT": kT, "vT": vT}
                self._kvnat = nat
            if _ATTN_OP and _KT_LAYOUT:
                if "kTs" not in nat:
                    # split once into contiguous per-chunk tensors, then free the full copies
                    kTs, vTs, b0 = [], [], 0
                    for c in st["sizes"]:
                        kTs.append(nat["kT"][b0:b0 + c].contiguous())
                        vTs.append(nat["vT"][b0:b0 + c].contiguous())
                        b0 += c
                    nat["kTs"], nat["vTs"] = kTs, vTs
                    nat["kT"] = nat["vT"] = None
                bk = SP
                if _BUCKET and off > 0:
                    bk = min(SP, max(_BMIN, 1 << (off - 1).bit_length()))
                if bk < SP:
                    bks = st.setdefault("_bk", {})
                    lf = bks.get(bk)
                    if lf is None and _PREWARM and not st.get("_bk_warm"):
                        # Crossing a bucket boundary mid-generation compiled the next
                        # bucket inside the decode loop (fill 500 -> 512: ~1.95 s/step for 10
                        # steps). Build and run EVERY bucket >= this one now, once. Each call
                        # writes the same K/V slot (k_out is RoPE'd k, independent of
                        # curr_sprior; v is v), and the real call below overwrites it anyway.
                        st["_bk_warm"] = True
                        for wb in [b for b in _bucket_list(SP) if b >= bk]:
                            if wb >= SP or wb in bks:
                                continue
                            wc = {c: AttnTKGConfig(bs=c, q_head=1, s_active=S, curr_sprior=wb,
                                                   full_sprior=SP, d_head=D, block_len=0,
                                                   tp_k_prior=False, use_pos_id=True, fuse_rope=True)
                                  for c in set(st["sizes"])}
                            st.setdefault("_bkcfg", {})[wb] = wc
                            wops = [_get_op(KERN, wc[c], st["sbm"], D) for c in st["sizes"]]
                            bks[wb] = _make_layer_fn(wops, st["sizes"], B, H, D, S, **_fz)
                            for c in set(st["sizes"]):
                                st["per"][c]["pos"].fill_(float(off))
                            _wi = torch.tensor([off], device=x.device, dtype=torch.long)
                            bks[wb](x if _fused else qkv, nat["kTs"], nat["vTs"], [st["per"][c]["mask"] for c in st["sizes"]],
                                    st["inv"], [st["per"][c]["pos"] for c in st["sizes"]], _wi)
                        lf = bks.get(bk)
                    if lf is None:
                        cfgs = {c: AttnTKGConfig(bs=c, q_head=1, s_active=S, curr_sprior=bk,
                                                 full_sprior=SP, d_head=D, block_len=0,
                                                 tp_k_prior=False, use_pos_id=True, fuse_rope=True)
                                for c in set(st["sizes"])}
                        st.setdefault("_bkcfg", {})[bk] = cfgs       # keep alive: _OPS keys on id()
                        ops = [_get_op(KERN, cfgs[c], st["sbm"], D) for c in st["sizes"]]
                        lf = _make_layer_fn(ops, st["sizes"], B, H, D, S, **_fz)
                        bks[bk] = lf
                else:
                    lf = st.get("_layer_fn")
                    if lf is None:
                        ops = [_get_op(KERN, st["per"][c]["cfg"], st["sbm"], D) for c in st["sizes"]]
                        lf = _make_layer_fn(ops, st["sizes"], B, H, D, S, **_fz)
                        st["_layer_fn"] = lf
                if not _POS_IN_GRAPH:
                    for c in set(st["sizes"]):
                        st["per"][c]["pos"].fill_(float(off))
                idx = torch.tensor([off], device=x.device, dtype=torch.long)
                ctx = lf(x if _fused else qkv, nat["kTs"], nat["vTs"], [st["per"][c]["mask"] for c in st["sizes"]],
                         st["inv"], [st["per"][c]["pos"] for c in st["sizes"]], idx)
                return ctx if _fz else self.out_proj(ctx)
            kpc, vpc = nat["kT"], nat["vT"]          # already contiguous: no per-step copy
        else:
            kp = cache[:B, :, 0].permute(0, 2, 1, 3).reshape(BH, 1, SP, D)
            vp = cache[:B, :, 1].permute(0, 2, 1, 3).reshape(BH, 1, SP, D)
            kpc = kp.contiguous(); vpc = vp.contiguous()

        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]              # each (B,S,H,D)
        if _LEAN_QKV1:
            _t3 = qkv.permute(2, 0, 3, 1, 4).contiguous()                 # (3,B,H,S,D)
            q_in = _t3[0].view(BH, 1, S, D)
            k_in = _t3[1].view(BH, 1, S, D)
            v_in = _t3[2].view(BH, 1, S, D)
        else:
            q_in = q.permute(0, 2, 1, 3).reshape(BH, 1, S, D).contiguous()
            k_in = k.permute(0, 2, 1, 3).reshape(BH, 1, S, D).contiguous()
            v_in = v.permute(0, 2, 1, 3).reshape(BH, 1, S, D).contiguous()

        # ---- tile the folded BH axis; each chunk is an independent batch slice ----
        outs, kouts, base_i = [], [], 0
        if _LEAN_BUFS and "lean" not in st:
            st["lean"] = [(torch.zeros_like(st["per"][c]["out"]),
                           torch.zeros_like(st["per"][c]["kout"])) for c in st["sizes"]]
        for j, c in enumerate(st["sizes"]):
            b = st["per"][c]
            b["pos"].fill_(float(off))     # POSITION VIA TENSOR DATA (no Python int in graph)
            sl = slice(base_i, base_i + c)
            ob, kb = st["lean"][j] if _LEAN_BUFS else (b["out"], b["kout"])
            r = KERN(q_in[sl], k_in[sl], v_in[sl], kpc[sl], vpc[sl],
                     b["mask"], ob, b["cfg"], st["sbm"],
                     st["inv"], b["pos"], None, None, None, kb)
            if _LEAN_BUFS:
                outs.append(r[0] if isinstance(r, (tuple, list)) else r)
                kouts.append(kb)
            else:
                outs.append((r[0] if isinstance(r, (tuple, list)) else r).clone())
                kouts.append(b["kout"].clone())
            base_i += c
        o = outs[0] if len(outs) == 1 else torch.cat(outs, 0)          # (BH,1,D,S)
        kout_all = kouts[0] if len(kouts) == 1 else torch.cat(kouts, 0)

        # store the kernel's RoPE'd k_out and raw v at position `off`
        idx = torch.tensor([off], device=x.device, dtype=torch.long)
        k_rot = kout_all[:, 0].reshape(B, H, D, S).permute(0, 3, 1, 2)   # (B,S,H,D)
        if _KV_NATIVE:
            # native layout is (BH, 1, SP, D): write along the SP axis (dim 2).
            vn = v.permute(0, 2, 1, 3).reshape(BH, 1, S, D)
            if _KT_LAYOUT:
                # kernel k_out is (BH,1,D,S): exactly one SP-column of the (BH,1,D,SP) cache
                _kv_write(nat["kT"], 3, idx, kout_all.to(nat["kT"].dtype).contiguous())
            else:
                kn = k_rot.permute(0, 2, 1, 3).reshape(BH, 1, S, D)      # (BH,1,S,D)
                _kv_write(nat["kT"], 2, idx, kn.to(nat["kT"].dtype).contiguous())
            _kv_write(nat["vT"], 2, idx, vn.to(nat["vT"].dtype).contiguous())
        else:
            cache[:B, :, 0].index_copy_(1, idx, k_rot.to(cache.dtype).contiguous())
            cache[:B, :, 1].index_copy_(1, idx, v.to(cache.dtype).contiguous())

        ctx = o.reshape(B, H, D, S).permute(0, 3, 1, 2).reshape(B, S, H * D)
        return self.out_proj(ctx)

    # Default: on for torch >= 2.14 (SDK 2.33), where it is required; off for torch 2.13 (torch-neuronx 2.13.x),
    # where the old path was measured and the dispatcher did not recompile per step.
    _tv = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
    _dflt = "1" if _tv >= (2, 14) else "0"
    if os.environ.get("EVO2_ATTN_EAGER_DISPATCH", _dflt) == "1":
        _forward_eager = _dyn.disable(_forward_impl)
        # EVO2_ATTN_FUSE_PROJ: "1" (default) = Wqkv + out_proj (and out_proj's TP all_reduce) inside
        # the layer graph, "out" = out_proj only, "0" = both eager. Measured (SDK 2.33, TP=4 B=88):
        # "1" removes ~2.3 ms/step of eager mm + all_reduce, 1-3% faster, TP tokens identical.
        # Teacher-forced vs "0": max KL 1.7e-5, |dp| <= 0.002 -- the same size as the default path's
        # own BS=1 vs BS=8 difference (bf16 accumulation order). An earlier exact-token batch gate
        # flipped on a near-tie (top-2 margin 0.000) and was replaced (test_hardware.py).
        _EAGER_ON[0] = os.environ.get("EVO2_ATTN_FUSE_PROJ", "1")
    else:
        _forward_eager = _forward_impl
        _EAGER_ON[0] = ""
    A.MHA.forward = forward


def uninstall():
    if _orig_fwd is not None:
        from vortex.model import attention as A
        A.MHA.forward = _orig_fwd
