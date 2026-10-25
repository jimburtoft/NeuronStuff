"""The public API: load, place, compile and run StripedHyena 1 (7B) on Trainium 2.

Every default below is the configuration that was measured best on trn2.3xlarge (public Neuron SDK 2.33 DLAMI,
torch-neuronx 2.14.3; first tuned on torch-neuronx 2.13 / SDK 2.32 and re-validated on 2.33). Options that were measured NOT to help are present but off, with the
reason inline, so the negative results stay discoverable.
"""

from __future__ import annotations

import glob
import os
import sys
import warnings

_HERE = os.path.dirname(os.path.abspath(__file__))
_VENDOR = os.path.join(_HERE, "_vendor")
CONFIG_YML = os.path.join(_VENDOR, "stripedhyena_configs", "7b-sh-32k-v1.yml")
TOKENIZER_JSON = os.path.join(_VENDOR, "tokenizer.json")

# The vendored upstream package (togethercomputer/stripedhyena 7e13f61, unmodified) is imported as
# top-level ``stripedhyena``; an installed copy of the same name is used only if no vendored copy exists.
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)

_STATE: dict = {"enabled": False, "opts": {}}

# name -> (default, rationale tied to a measurement). Decode levers apply per call site: see
# ``decode_opts()`` for the B / TP / context rules the defaults encode.
_OPTIONS = {
    "rms_custom": (True, "RMSNorm as torch.rms_norm(x.float(), ...) -> AwsNeuronRmsNorm: 1.05-1.13x decode at B>1"),
    "cascade":    (False, "fused NKI Hyena decode cascade: 1.09x decode ONLY for single-core B=1 (enable(cascade=True) for that "
                          "profile; turns rms_custom off -- they do not stack). Its colperm layout makes prefill 1.36x slower "
                          "(2K TTFT 571 vs 421 ms); neutral at B>1, slower under TP"),
    "tkg":        (True, "nkilib attention_tkg for decode attention: windows >= 32K (TP=4 B=28: 1.28x) and 8K windows at B >= 64 "
                         "(TP=4 B=128: 1.23x); slower than SDPA at 2K (0.60x at B=512) and at small B"),
    "bucket":     (True, "attention reads a power-of-two prefix >= fill: 1.19-1.21x when fill << window; exact"),
    "nki_mlp":    (True, "MLP weights stored (in, out) + nkilib mlp kernel. 1 core: TKG at decode rows <= 32, CTE at "
                         "prefill rows >= 512, x@Wt between (1.04-1.16x decode, 1.03x TTFT). TP shard: x@Wt decode "
                         "(TKG measured slower there), CTE prefill (TP=4 B=1 step 13.2 vs 15.0 ms)"),
    "resid_fp32": (True, "fp32 prefill residual stream: without it one WikiText window scored PPL 4.94 vs 4.66 (gate failure)"),
    # --- measured NOT to help (kept as knobs) ---
    "mlp_cat":    (False, "REJECTED: l1+l2 one GEMV, no gain (compiler already fuses), +0.9 GB transient HBM"),
    "pretrans":   (False, "REJECTED: (in,out) weight storage, 0.97-0.99x"),
    "modal_tri":  (False, "REJECTED: semiseparable modal prefill, 1.45x slower"),
}

MAX_CONTEXT = 32768


def config_summary() -> str:
    """Human-readable summary of what is enabled and why. Paste it into issue reports."""
    opts = _STATE["opts"] or {k: v[0] for k, v in _OPTIONS.items()}
    lines = ["striped_hyena_neuron configuration:"]
    for name, (default, why) in _OPTIONS.items():
        lines.append(f"  {'ON ' if opts.get(name, default) else 'off'}  {name:<11} {why}")
    lines.append(f"  enabled={_STATE['enabled']}")
    return "\n".join(lines)


def enable(**overrides) -> dict:
    """Install the Neuron patches into the vendored ``stripedhyena`` (SDPA/fixed-window GQA attention,
    flash-formula RMSNorm, complex-free FFT-free Hyena filter). Must run before the model is built
    (``load()`` calls it). Returns the effective options."""
    unknown = set(overrides) - set(_OPTIONS)
    if unknown:
        raise TypeError(f"unknown option(s) {sorted(unknown)}; valid: {sorted(_OPTIONS)}")
    if _STATE["enabled"]:
        if overrides and any(_STATE["opts"].get(k) != v for k, v in overrides.items()):
            warnings.warn("striped_hyena_neuron.enable() already called with different options; "
                          "options are process-global, restart the process to change them", RuntimeWarning)
        return dict(_STATE["opts"])
    opts = {k: v[0] for k, v in _OPTIONS.items()}
    opts.update(overrides)
    if opts["cascade"] and "rms_custom" not in overrides:
        opts["rms_custom"] = False
    for k in ("mlp_cat", "pretrans"):
        if opts[k]:
            warnings.warn(f"option {k} was measured not to help ({_OPTIONS[k][1]})", RuntimeWarning)
    from . import attention, hyena, prefill, rmsnorm
    attention.install()
    rmsnorm.install()
    hyena.install()
    prefill.RESID_FP32 = bool(opts["resid_fp32"])
    prefill.MODAL_IMPL = "tri" if opts["modal_tri"] else "bmm"
    from . import opts as O
    if opts["nki_mlp"] and (opts["mlp_cat"] or opts["pretrans"]):
        raise ValueError("nki_mlp is exclusive with mlp_cat / pretrans")
    O.enable([k for k in ("rms_custom", "mlp_cat", "pretrans", "cascade", "nki_mlp") if opts[k]])   # cascade implies colperm
    _STATE.update(enabled=True, opts=opts)
    return dict(opts)


def is_enabled() -> bool:
    return bool(_STATE["enabled"])


def make_config(max_seqlen: int = MAX_CONTEXT, **over):
    """The shipped 7B config (``7b-sh-32k-v1.yml``) as a Dynamo-traceable object.

    ``stripedhyena.utils.dotdict.__getattr__`` is ``dict.get`` (a C builtin): every ``config.x`` read
    inside a compiled region is a graph break. ``TraceableConfig`` keeps dotdict semantics (missing
    keys read as None, ``get(k, default)`` honours the default)."""
    import yaml
    from .traceable_config import TraceableConfig
    if max_seqlen > MAX_CONTEXT:
        raise ValueError(f"max_seqlen {max_seqlen} > model maximum {MAX_CONTEXT}")
    cfg = yaml.safe_load(open(CONFIG_YML))
    cfg.update(use_flash_rmsnorm=False, use_flashfft=False, use_flash_depthwise=False,
               max_seqlen=max_seqlen, max_batch_size=1, prefill_style="fft")
    cfg.update(over)
    return TraceableConfig(cfg)


def to_device(model, device="neuron"):
    """Move the model to ``device`` COMPLETELY. Use this instead of bare ``.to()``.

    ``nn.Module.to()`` moves parameters and buffers only; stripedhyena also stores plain tensor
    attributes on modules (e.g. rotary tables), which would stay on the host and fail at the first
    compiled call. Returns ``(model, n_plain_attributes_moved)``."""
    import torch
    dev = torch.device(device)
    model = model.to(dev)
    moved = 0
    for mod in model.modules():
        for name, val in list(vars(mod).items()):
            if torch.is_tensor(val) and val.device != dev:
                setattr(mod, name, val.to(dev))
                moved += 1
    return model, moved


def _load_state(weights):
    from safetensors.torch import load_file
    weights = os.path.expanduser(weights)
    files = sorted(glob.glob(os.path.join(weights, "model-*.safetensors"))) or \
        sorted(glob.glob(os.path.join(weights, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no *.safetensors under {weights}")
    sd = {}
    for f in files:
        sd.update(load_file(f))
    return {k[len("backbone."):] if k.startswith("backbone.") else k: v for k, v in sd.items()}


def load(weights: str, *, device="neuron", max_seqlen: int = MAX_CONTEXT, tp: tuple[int, int] | None = None,
         **enable_kw):
    """Build StripedHyena-Hessian-7B / -Nous-7B from a HF snapshot directory (``model-*.safetensors``).

    - meta-device build + ``load_state_dict(strict=True, assign=True)``: every tensor keeps its
      checkpoint dtype (bf16, fp32 poles/residues) with no fp32 init of 7.6 B params;
    - ``tp=(rank, world)``: Megatron shard for tensor parallelism (call after
      ``torch.distributed.init_process_group(backend="neuron")``); world <= 8 (8 KV heads);
    - moves parameters and plain attributes (``to_device``) and builds the modal tables on device.
    Returns ``(model, info)``."""
    import torch
    enable(**enable_kw)
    from stripedhyena.model import StripedHyena
    from . import hyena
    cfg = make_config(max_seqlen)
    with torch.device("meta"):
        m = StripedHyena(cfg)
    res = m.load_state_dict(_load_state(weights), strict=True, assign=True)
    if tp is not None:
        from . import tp as tpm
        tpm.shard_model(m, tp[0], tp[1])
    m, moved = to_device(m, device)
    m.eval()
    for b in m.blocks:
        if hasattr(b, "filter"):
            hyena._tables(b.filter, torch.device(device))
    from . import opts as O
    n_layout = O.apply(m)
    meta = [n for n, p in list(m.named_parameters()) + list(m.named_buffers()) if p.device.type == "meta"]
    assert not meta, meta[:5]
    dt = {}
    for _, p in m.named_parameters():
        dt[str(p.dtype)] = dt.get(str(p.dtype), 0) + 1
    return m, {"param_dtypes": dt, "plain_attrs_moved": moved, "missing": list(res.missing_keys),
               "unexpected": list(res.unexpected_keys), "layout_levers_applied": n_layout, "tp": tp}


def decode_opts(B: int, window: int, tp: int = 1) -> dict:
    """The measured-best per-call-site decode levers for batch B, KV window and TP degree."""
    o = _STATE["opts"] or {k: v[0] for k, v in _OPTIONS.items()}
    return {"cascade": bool(o["cascade"]) and B == 1 and tp == 1,
            "tkg": bool(o["tkg"]) and not (B == 1 and tp > 1) and (window >= 32768 or (window >= 8192 and B >= 64)),
            "bucket": bool(o["bucket"]),
            "mode": "step" if (B == 1 and tp > 1) else "blocks:32"}


def compile(model, *, B: int, window: int, tp: int = 1, mode: str | None = None):
    """Build the compiled prefill and decode for ``model``. NOT optional (eager decode is ~4x slower).

    Each block type is one ``torch.compile(backend="neuron", dynamic=False)`` graph shared by its 16
    blocks; ``mode="step"`` compiles the whole decode step as one graph (the default for B=1 under TP:
    lowest per-token latency; its first compile is long). Compile all blocks or none: partial compilation
    measured slower. One (B, window) shape set per process is the safe pattern on PyTorch Native.
    Returns a ``Generator``."""
    from .prefill import Prefiller
    from .decode import Decoder
    if window % 256:
        raise ValueError(f"window must be a multiple of 256 (got {window})")
    if window > MAX_CONTEXT:
        raise ValueError(f"window {window} > model maximum {MAX_CONTEXT}")
    d = decode_opts(B, window, tp)
    if mode is not None:
        d["mode"] = mode
    return Generator(model, Prefiller(model, "blocks"), Decoder(model, d["mode"]), window, d, tp)


class Generator:
    """Greedy generation on device with the compiled prefill + decode.

        gen = shn.compile(model, B=8, window=4096)
        out = gen.generate(ids, max_new_tokens=128)        # ids: (B, L) long tensor
    """

    def __init__(self, model, prefiller, decoder, window, d, tp=1):
        self.model, self.pf, self.dec, self.W, self.d, self.tp = model, prefiller, decoder, window, d, tp
        self._prepared = set()

    def prefill(self, ids, seg=None):
        """Returns (first_tokens (B,) on device, DecodeState). ``seg``: segmented prefill (exact; bounds
        activation memory, needed at 32K). Default: none up to 4K; 2048 on one core and under TP at <= 8K; 8192 under TP
        above 8K; one core with a window > 16K: 512. Measured on SDK 2.33, TP=4: 8K TTFT 742.6 ms at 2048 vs 780.8 at 4096; 32K
        7.59 s at 8192 (9.13 at 16384, 10.15 at 4096 before the long-bucket query-chunk fix). One core, 32K window: 2048 runs
        out of HBM in prefill beside the 32K KV cache; 512 fits (decode B<=2); TTFT 32.0 s at 512."""
        L = ids.shape[1]
        if L + 1 > self.W:
            raise ValueError(f"prompt length {L} does not fit the window {self.W}")
        if seg is None and L > 4096:
            if self.tp == 1:
                seg = 512 if self.W > 16384 else 2048     # SDK 2.33: one core at a 32K window runs out of HBM at 2048
            else:
                seg = 2048 if L <= 8192 else 8192
        dev = next(self.model.parameters()).device
        _, tok, S = self.pf(ids.to(dev), self.W, seg=seg)
        return tok, S

    def _prepare(self, S, max_fill):
        from .decode import to_tkg_layout, enable_cascade, set_bucket
        B = S.s_re[0].shape[0]
        if self.d["tkg"]:
            to_tkg_layout(self.dec, S)
        if self.d["cascade"] and B not in self._prepared:
            enable_cascade(self.dec, B)
            self._prepared.add(B)
        if self.d["bucket"]:
            set_bucket(self.dec, S, max_fill)

    def generate(self, ids, max_new_tokens: int, eos: int | None = None):
        """Greedy decode. Returns the generated ids (B, n) on the host. Runs under ``torch.inference_mode()`` (the prefill
        produces inference tensors; when calling ``prefill`` / ``dec`` directly, wrap them in ``torch.inference_mode()`` too)."""
        import torch
        with torch.inference_mode():
            return self._generate(ids, max_new_tokens, eos)

    def _generate(self, ids, max_new_tokens, eos):
        import torch
        L = ids.shape[1]
        n = min(max_new_tokens, self.W - L)
        tok, S = self.prefill(ids)
        self._prepare(S, L + n)
        out = [tok.cpu()]
        for _ in range(n - 1):
            _, tok = self.dec(tok, S)
            out.append(tok.cpu())
            if eos is not None and bool((out[-1] == eos).all()):
                break
        return torch.stack(out, 1)
