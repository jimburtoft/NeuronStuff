"""Make vortex importable on Neuron (no CUDA flash-attn extension).

Installs stub modules for the CUDA-only extensions BEFORE vortex is imported.
Nothing here changes math: the flash-attn entry points are set to None, which is
exactly what vortex/model/attention.py already handles via its try/except
(`local_flash_attn_* = None` + `use_flash_attn=False` asserts).
"""
import sys, types, torch

def install():
    # 0. torch.cuda.device("cpu") is called unconditionally in model.py:665.
    #    With no CUDA present it raises. Make it a no-op context manager.
    if not torch.cuda.is_available():
        import contextlib
        @contextlib.contextmanager
        def _noop_dev(device=None):
            yield
        torch.cuda.device = _noop_dev
        torch.cuda.set_device = lambda *a, **k: None

    # 1. flash_attn_2_cuda: the CUDA extension. Stub so `import` succeeds.
    if "flash_attn_2_cuda" not in sys.modules:
        m = types.ModuleType("flash_attn_2_cuda")
        def _nope(*a, **k):
            raise RuntimeError("flash_attn_2_cuda is CUDA-only; not available on Neuron")
        for fn in ["fwd","bwd","varlen_fwd","varlen_bwd","fwd_kvcache"]:
            setattr(m, fn, _nope)
        sys.modules["flash_attn_2_cuda"] = m

    # 2. vortex.ops.embedding.rotary.apply_rotary -> pure-torch equivalent.
    #    vortex/model/rotary.py has apply_rotary_emb_torch already; we only need
    #    `apply_rotary` to exist so the module imports. The torch path is used
    #    because we run with use_flash_attn=False.
    def apply_rotary(x, cos, sin, seqlen_offsets=0, interleaved=False,
                     inplace=False, conjugate=False, **kw):
        # x: (batch, seqlen, nheads, headdim)
        ro_dim = cos.shape[-1] * 2
        assert ro_dim <= x.shape[-1], f"ro_dim {ro_dim} > headdim {x.shape[-1]}"
        if isinstance(seqlen_offsets, int) and seqlen_offsets:
            cos = cos[seqlen_offsets:]; sin = sin[seqlen_offsets:]
        seqlen = x.shape[1]
        c = cos[:seqlen].to(x.dtype); s = sin[:seqlen].to(x.dtype)
        if conjugate: s = -s
        c = c[None, :, None, :]; s = s[None, :, None, :]
        x_ro, x_pass = x[..., :ro_dim], x[..., ro_dim:]
        if interleaved:
            x1, x2 = x_ro[..., 0::2], x_ro[..., 1::2]
            o1 = x1 * c - x2 * s
            o2 = x1 * s + x2 * c
            out_ro = torch.stack([o1, o2], dim=-1).flatten(-2)
        else:
            half = ro_dim // 2
            x1, x2 = x_ro[..., :half], x_ro[..., half:]
            out_ro = torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)
        out = torch.cat([out_ro, x_pass], dim=-1) if x_pass.numel() else out_ro
        if inplace:
            x.copy_(out); return x
        return out

    for name in ["vortex.ops", "vortex.ops.embedding", "vortex.ops.embedding.rotary"]:
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    sys.modules["vortex.ops.embedding.rotary"].apply_rotary = apply_rotary
    for fn in ["local_flash_attn_func","local_flash_attn_kvpacked_func",
               "local_flash_attn_qkvpacked_func","local_flash_attn_varlen_kvpacked_func",
               "local_flash_attn_varlen_qkvpacked_func","local_flash_attn_with_kvcache"]:
        setattr(sys.modules["vortex.ops"], fn, None)
    sys.modules["vortex.ops"].embedding = sys.modules["vortex.ops.embedding"]
    sys.modules["vortex.ops.embedding"].rotary = sys.modules["vortex.ops.embedding.rotary"]
