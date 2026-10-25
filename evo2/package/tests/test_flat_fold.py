"""EVO2_FLAT_FOLD changes only how batch rows map to (partition, column) and how the
per-channel weights are tiled. With the device kernel replaced by an exact per-channel torch
reference, the flat and permute fold paths must give bitwise-identical outputs and states.
(The kernel itself is checked flat-vs-permute in the NKI simulator.)"""
import os, sys, importlib
import pytest
import torch

VR = os.environ.get("VORTEX_REPO", os.path.expanduser("~/vortex-repo"))
pytestmark = pytest.mark.skipif(not os.path.isdir(VR), reason="vortex repo not available")


def _hcl_ref(u3, fs3, sw3, sb3, st3, pr3, dp3):
    S = st3.shape[2]
    zp = sw3[..., 2] * u3 + fs3[..., 0] * sw3[..., 0] + fs3[..., 1] * sw3[..., 1] + sb3
    nfs = torch.stack([fs3[..., 1], u3], -1)
    x2, x1, v = zp[..., 0:1], zp[..., 1:2], zp[..., 2:3]
    x1v = x1 * v
    nst = pr3[..., :S] * st3 + x1v
    y = x2 * ((pr3[..., S:] * nst).sum(-1, keepdim=True) + dp3 * x1v)
    return y, nfs, nst


def _hcm_ref(u3, fs3, sw3, sb3, is3, ih3, ih0, dp3, gated):
    zp = sw3[..., 2] * u3 + fs3[..., 0] * sw3[..., 0] + fs3[..., 1] * sw3[..., 1] + sb3
    nfs = torch.stack([fs3[..., 1], u3], -1)
    x2, x1, v = zp[..., 0:1], zp[..., 1:2], zp[..., 2:3]
    g = x1 * v
    yt = ih0 * g + (is3 * ih3).sum(-1, keepdim=True) + (dp3 * g if gated else dp3)
    nis = torch.cat([is3[..., 1:], g], -1)
    return yt * x2, nfs, nis


@pytest.mark.parametrize("kind", ["hcl", "hcm"])
def test_flat_fold_matches_permute_fold(kind):
    sys.path.insert(0, VR)
    from evo2_neuron_decode.patches import neuron_shim, iir_recurrence_patch
    neuron_shim.install(); iir_recurrence_patch.install()
    import yaml
    from vortex.model.utils import dotdict
    from vortex.model.model import ParallelGatedConvBlock, HyenaCascade
    from vortex.model.cache import HyenaCascadeFIRInferenceParams, HyenaCascadeIIRInferenceParams
    import types
    if "torch_neuronx" not in sys.modules:
        tn = types.ModuleType("torch_neuronx"); tn.wrap_nki = lambda k: k
        sys.modules["torch_neuronx"] = tn
    try:
        import nki  # noqa: F401  (kernel modules import nki at module level)
    except ImportError:
        pytest.skip("nki not importable in this venv")
    modname = "cascade_patch" if kind == "hcl" else "hcm_patch"
    cfg = dotdict(yaml.safe_load(open(f"{VR}/configs/evo2-7b-8k.yml")))
    for k, v in dict(use_flash_attn=False, use_flashfft=False, use_flash_depthwise=False, use_flash_rmsnorm=False,
                     prefill_style="recurrence", inference_mode=True, use_fp8_input_projections=False,
                     hidden_size=512, inner_mlp_size=1024, num_attention_heads=8, hcm_filter_groups=32,
                     hcs_filter_groups=32, hcl_filter_groups=512, num_filters=512).items():
        cfg[k] = v
    outs = {}
    for flat in (False, True):
        orig = HyenaCascade.sequential_forward
        mod = importlib.import_module(f"evo2_neuron_decode.patches.{modname}")
        try:
            mod.install()
            mod._KERN = _hcl_ref if kind == "hcl" else _hcm_ref
            mod._FLAT = flat
            ncall = [0]
            _k = mod._KERN
            def _count(*a, _k=_k):
                ncall[0] += 1; return _k(*a)
            mod._KERN = _count
            torch.manual_seed(0)
            groups = cfg.hcl_filter_groups if kind == "hcl" else cfg.hcm_filter_groups
            inner = None if kind == "hcl" else cfg.hcm_filter_length
            blk = ParallelGatedConvBlock(cfg, 0, hyena_filter_groups=groups, fir_inner_filter_length=inner).float().eval()
            for n, p in blk.named_parameters():
                if "log_poles" in n: p.data.uniform_(-2.3, -0.01)
                elif p.dim() >= 2: torch.nn.init.normal_(p, std=3.0 / 512 ** 0.5)
                else: torch.nn.init.normal_(p, std=0.3)
            mk = (lambda: HyenaCascadeIIRInferenceParams(fir_filter_length=cfg.short_filter_length, state_dim=cfg.state_size, seqlen_offset=0)) \
                if kind == "hcl" else \
                (lambda: HyenaCascadeFIRInferenceParams(fir_filter_length=cfg.short_filter_length, fir_inner_filter_length=inner, seqlen_offset=0))
            ip = mk()
            Bt = max(3, getattr(mod, "_FOLD_MIN", 0) + 1)   # hcm only folds at B >= _FOLD_MIN
            xs = torch.randn(Bt, 5, 512)
            ys = []
            with torch.no_grad():
                # prefill 4 tokens (upstream path), then 3 decode steps through the patched path
                y, _ = blk(xs[:, :4], inference_params=ip)
                ip.seqlen_offset = 4
                n0 = ncall[0]
                y, _ = blk(xs[:, 4:5], inference_params=ip); ys.append(y)
                assert ncall[0] - n0 == 1, f"{kind}: decode step made {ncall[0]-n0} kernel calls, expected 1 folded launch"
            outs[flat] = (ys, {k: v.clone() for k, v in ip.fir_state_dict.items()},
                          {k: v.clone() for k, v in getattr(ip, "state_dict", {}).items()},
                          {k: v.clone() for k, v in getattr(ip, "fir_inner_state_dict", {}).items()})
        finally:
            HyenaCascade.sequential_forward = orig
    a, b = outs[False], outs[True]
    assert a[0][0].abs().max() > 1e-3
    for x, y in zip(a[0], b[0]):
        assert torch.equal(x, y), f"{kind}: flat fold output differs, max {float((x-y).abs().max()):.2e}"
    for da, db in zip(a[1:], b[1:]):
        for k in da:
            assert torch.equal(da[k], db[k]), f"{kind}: state {k} differs"


def test_hcm_two_prompt_lengths_same_module():
    """The HCM inner-FIR state length C = min(L, k) - 1 depends on the prompt length.
    The hoisted weights must be keyed on C, or a second prompt length (< k vs >= k) in the same
    process feeds the kernel mismatched shapes. Run L=40 then L=150 through ONE block."""
    sys.path.insert(0, VR)
    import types
    if "torch_neuronx" not in sys.modules:
        tn = types.ModuleType("torch_neuronx"); tn.wrap_nki = lambda k: k
        sys.modules["torch_neuronx"] = tn
    try:
        import nki  # noqa: F401
    except ImportError:
        pytest.skip("nki not importable in this venv")
    from evo2_neuron_decode.patches import neuron_shim, iir_recurrence_patch, hcm_patch
    neuron_shim.install(); iir_recurrence_patch.install()
    import yaml
    from vortex.model.utils import dotdict
    from vortex.model.model import ParallelGatedConvBlock, HyenaCascade
    from vortex.model.cache import HyenaCascadeFIRInferenceParams
    cfg = dotdict(yaml.safe_load(open(f"{VR}/configs/evo2-7b-8k.yml")))
    for k, v in dict(use_flash_attn=False, use_flashfft=False, use_flash_depthwise=False, use_flash_rmsnorm=False,
                     prefill_style="recurrence", inference_mode=True, use_fp8_input_projections=False,
                     hidden_size=512, inner_mlp_size=1024, num_attention_heads=8, hcm_filter_groups=32,
                     hcs_filter_groups=32, hcl_filter_groups=512, num_filters=512).items():
        cfg[k] = v
    orig = HyenaCascade.sequential_forward
    try:
        hcm_patch.install()
        seen = []
        def ref(u3, fs3, sw3, sb3, is3, ih3, ih0, dp3, gated):
            assert ih3.shape == is3.shape[:2] + (is3.shape[2],) or ih3.shape[2] == is3.shape[2], \
                f"ih3 {tuple(ih3.shape)} vs is3 {tuple(is3.shape)}"
            seen.append(is3.shape[-1])
            return _hcm_ref(u3, fs3, sw3, sb3, is3, ih3, ih0, dp3, gated)
        hcm_patch._KERN = ref
        torch.manual_seed(0)
        blk = ParallelGatedConvBlock(cfg, 0, hyena_filter_groups=32, fir_inner_filter_length=cfg.hcm_filter_length).float().eval()
        Bt = max(3, getattr(hcm_patch, "_FOLD_MIN", 0) + 1)
        for L in (40, 150):
            ip = HyenaCascadeFIRInferenceParams(fir_filter_length=cfg.short_filter_length,
                                                fir_inner_filter_length=cfg.hcm_filter_length, seqlen_offset=0)
            xs = torch.randn(Bt, L + 1, 512)
            with torch.no_grad():
                blk(xs[:, :L], inference_params=ip)
                ip.seqlen_offset = L
                blk(xs[:, L:L + 1], inference_params=ip)
        # Short prompts are left-padded to k-1, so both lengths now give the same C
        assert set(seen) == {cfg.hcm_filter_length - 1}, f"expected C = k-1 for both, saw {seen}"
    finally:
        HyenaCascade.sequential_forward = orig


@pytest.mark.parametrize("L", [8, 100, 200])
def test_hcm_short_prompt_matches_upstream(L):
    """Prompts shorter than the HCM inner filter (k=128). Upstream grows the inner-FIR state
    each decode step; the patch must decode identically (it left-pads the state to k-1). Before the
    fix: 2-5% relative divergence within 50 steps for L < 127."""
    sys.path.insert(0, VR)
    import types, copy
    if "torch_neuronx" not in sys.modules:
        tn = types.ModuleType("torch_neuronx"); tn.wrap_nki = lambda k: k
        sys.modules["torch_neuronx"] = tn
    try:
        import nki  # noqa: F401
    except ImportError:
        pytest.skip("nki not importable in this venv")
    from evo2_neuron_decode.patches import neuron_shim, iir_recurrence_patch, hcm_patch
    neuron_shim.install(); iir_recurrence_patch.install()
    import yaml
    from vortex.model.utils import dotdict
    from vortex.model.model import ParallelGatedConvBlock, HyenaCascade
    from vortex.model.cache import HyenaCascadeFIRInferenceParams
    cfg = dotdict(yaml.safe_load(open(f"{VR}/configs/evo2-7b-8k.yml")))
    for k, v in dict(use_flash_attn=False, use_flashfft=False, use_flash_depthwise=False, use_flash_rmsnorm=False,
                     prefill_style="recurrence", inference_mode=True, use_fp8_input_projections=False,
                     hidden_size=512, inner_mlp_size=1024, num_attention_heads=8, hcm_filter_groups=32,
                     hcs_filter_groups=32, hcl_filter_groups=512, num_filters=512).items():
        cfg[k] = v
    torch.manual_seed(0)
    blk = ParallelGatedConvBlock(cfg, 0, hyena_filter_groups=32, fir_inner_filter_length=cfg.hcm_filter_length).float().eval()
    for n, p in blk.named_parameters():
        if p.dim() >= 2: torch.nn.init.normal_(p, std=3.0 / 512 ** 0.5)
        else: torch.nn.init.normal_(p, std=0.3)
    ref_blk = copy.deepcopy(blk)
    Bt = max(3, getattr(hcm_patch, "_FOLD_MIN", 0) + 2); STEPS = 40
    xs = torch.randn(Bt, L + STEPS, 512)
    orig = HyenaCascade.sequential_forward
    def run(b, patched):
        HyenaCascade.sequential_forward = orig
        if patched:
            hcm_patch.install(); hcm_patch._KERN = _hcm_ref
        ip = HyenaCascadeFIRInferenceParams(fir_filter_length=cfg.short_filter_length,
                                            fir_inner_filter_length=cfg.hcm_filter_length, seqlen_offset=0)
        ys = []
        with torch.no_grad():
            b(xs[:, :L], inference_params=ip); ip.seqlen_offset = L
            for t in range(STEPS):
                y, _ = b(xs[:, L + t:L + t + 1], inference_params=ip); ys.append(y); ip.seqlen_offset += 1
        return torch.cat(ys, 1)
    try:
        a = run(ref_blk, False); p = run(blk, True)
    finally:
        HyenaCascade.sequential_forward = orig
    rel = float((a - p).norm() / a.norm())
    assert rel < 1e-5, f"L={L}: patched decode differs from upstream, rel {rel:.2e}"
