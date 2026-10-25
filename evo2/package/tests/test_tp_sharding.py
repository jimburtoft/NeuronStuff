"""tp.shard_model must reproduce the unsharded block exactly (CPU, fp32, no distributed).

Simulates TP by running every rank's shard on the same input and summing the Rowwise partials
(what all_reduce does). Uses small random weights so it needs no checkpoint and runs in CI.
"""
import copy
import os
import sys

import pytest
import torch

VR = os.environ.get("VORTEX_REPO", os.path.expanduser("~/vortex-repo"))
pytestmark = pytest.mark.skipif(not os.path.isdir(VR), reason="vortex repo not available")


def _cfg():
    import yaml
    from vortex.model.utils import dotdict
    cfg = dotdict(yaml.safe_load(open(f"{VR}/configs/evo2-7b-8k.yml")))
    for k, v in dict(use_flash_attn=False, use_flashfft=False, use_flash_depthwise=False,
                     use_flash_rmsnorm=False, prefill_style="recurrence", inference_mode=True,
                     use_fp8_input_projections=False, hidden_size=512, inner_mlp_size=1024,
                     num_attention_heads=8, hcm_filter_groups=32, hcs_filter_groups=32,
                     hcl_filter_groups=512, num_filters=512).items():
        cfg[k] = v
    return cfg


@pytest.fixture(autouse=True)
def _paths():
    sys.path.insert(0, VR)
    from evo2_neuron_decode.patches import neuron_shim, iir_recurrence_patch
    neuron_shim.install(); iir_recurrence_patch.install()


class _FakeAllReduce:
    """Let RowParallelLinear run without a process group: record partials, sum later."""
    def __enter__(self):
        self._orig = torch.distributed.all_reduce
        torch.distributed.all_reduce = lambda t, *a, **k: t
        return self

    def __exit__(self, *e):
        torch.distributed.all_reduce = self._orig


@pytest.mark.parametrize("kind", ["hcm", "hcs", "hcl"])
def test_cascade_block_shards_exactly(kind):
    from vortex.model.model import ParallelGatedConvBlock
    from vortex.model.cache import HyenaCascadeFIRInferenceParams, HyenaCascadeIIRInferenceParams
    from evo2_neuron_decode import tp as TPM
    torch.manual_seed(0)
    cfg = _cfg(); TP = 4
    groups = {"hcm": cfg.hcm_filter_groups, "hcs": cfg.hcs_filter_groups, "hcl": cfg.hcl_filter_groups}[kind]
    inner = {"hcm": cfg.hcm_filter_length, "hcs": cfg.hcs_filter_length, "hcl": None}[kind]
    full = ParallelGatedConvBlock(cfg, 0, hyena_filter_groups=groups, fir_inner_filter_length=inner).float().eval()
    # Init in the REAL regime (stats from the evo2_7b_base checkpoint), not N(0, 0.05) everywhere:
    # a naive random init makes the cascade output ~1e-8 (product of 3 small streams, wrong-sign
    # poles), so a corrupted channel split still "passed" -- the test was blind. Biases are zeroed
    # because a bias alone makes the output independent of the split.
    D = cfg.hidden_size
    for name, p in full.named_parameters():
        if name.endswith("bias"):
            torch.nn.init.zeros_(p)
        elif "log_poles" in name:
            p.data.uniform_(-2.3, -0.01)               # real: -2.32 .. -0.0015, all negative
        elif name.endswith("filter.D"):
            torch.nn.init.normal_(p, std=0.07)
        elif name.endswith("filter.h") or "residues" in name or "short_filter" in name:
            torch.nn.init.normal_(p, std=0.2)
        elif p.dim() >= 2:
            torch.nn.init.normal_(p, std=3.0 / D ** 0.5)  # streams O(1) after the projection
        else:
            torch.nn.init.ones_(p)                       # norm scales

    ranks = []
    for r in range(TP):
        b = copy.deepcopy(full)
        TPM._shard_cascade(b, r, TP); TPM._shard_mlp(b.mlp, r, TP)
        ranks.append(b)
    mk = (lambda: HyenaCascadeIIRInferenceParams(fir_filter_length=cfg.short_filter_length,
                                                 state_dim=cfg.state_size, seqlen_offset=0)) if kind == "hcl" else \
         (lambda: HyenaCascadeFIRInferenceParams(fir_filter_length=cfg.short_filter_length,
                                                 fir_inner_filter_length=inner, seqlen_offset=0))
    ipf, ipr = mk(), [mk() for _ in range(TP)]
    xs = torch.randn(2, 4, cfg.hidden_size)
    worst = 0.0
    with torch.no_grad(), _FakeAllReduce():
        for t in range(4):
            x = xs[:, t:t + 1]
            ipf.seqlen_offset = t
            for p in ipr: p.seqlen_offset = t
            def mix(b, ip):
                z = b.proj_norm(x); z, _ = b.filter(z, inference_params=ip); return b.out_filter_dense(z)
            ref = mix(full, ipf)
            par = sum(mix(ranks[r], ipr[r]) for r in range(TP))
            # guard: the test is blind if the mixer output is negligible (it was, with random
            # poles + a bias: output ~= bias, filter path ~1e-7, a wrong slice still passed)
            assert ref.abs().max() > 1e-3, f"{kind}: mixer output {ref.abs().max():.1e} too small to test"
            worst = max(worst, ((ref - par).abs().max() / ref.abs().max().clamp_min(1e-9)).item())
            h = x + ref
            mref = full.mlp(full.post_norm(h))
            mpar = sum(ranks[r].mlp(ranks[r].post_norm(h)) for r in range(TP))
            worst = max(worst, ((mref - mpar).abs().max() / mref.abs().max().clamp_min(1e-9)).item())
    assert worst < 1e-4, f"{kind}: TP=4 shard differs from unsharded by rel {worst:.2e}"


@pytest.mark.parametrize("kind", ["hcm", "hcl"])
def test_pretransposed_weights_match(kind, monkeypatch):
    """EVO2_WT=1: every GEMV weight stored (in, out). Block output must match the
    (out, in) model, and every replaced module must actually be used (no silent fallback)."""
    from vortex.model.model import ParallelGatedConvBlock
    from vortex.model.cache import HyenaCascadeFIRInferenceParams, HyenaCascadeIIRInferenceParams
    from evo2_neuron_decode import tp as TPM, api
    from evo2_neuron_decode.patches import mlp_cat_patch
    mlp_cat_patch.install()
    torch.manual_seed(1)
    cfg = _cfg()
    groups = {"hcm": cfg.hcm_filter_groups, "hcl": cfg.hcl_filter_groups}[kind]
    inner = {"hcm": cfg.hcm_filter_length, "hcl": None}[kind]
    blk = ParallelGatedConvBlock(cfg, 0, hyena_filter_groups=groups, fir_inner_filter_length=inner).float().eval()
    D = cfg.hidden_size
    for name, p in blk.named_parameters():
        if "log_poles" in name: p.data.uniform_(-2.3, -0.01)
        elif name.endswith("bias"): torch.nn.init.normal_(p, std=0.1)
        elif p.dim() >= 2: torch.nn.init.normal_(p, std=3.0 / D ** 0.5)
        else: torch.nn.init.normal_(p, std=0.2)
    TPM._shard_cascade(blk, 0, 1); TPM._shard_mlp(blk.mlp, 0, 1)
    ref_blk = copy.deepcopy(blk)
    api._dedup_mlp(blk)
    monkeypatch.setenv("EVO2_WT", "1")
    n = api._pretranspose(blk)
    assert n >= 4, f"only {n} modules pretransposed"
    assert getattr(blk.mlp, "_w12t", None) is not None and blk.mlp.l3.wt is not None
    mk = (lambda: HyenaCascadeIIRInferenceParams(fir_filter_length=cfg.short_filter_length,
                                                 state_dim=cfg.state_size, seqlen_offset=0)) if kind == "hcl" else \
         (lambda: HyenaCascadeFIRInferenceParams(fir_filter_length=cfg.short_filter_length,
                                                 fir_inner_filter_length=inner, seqlen_offset=0))
    ipa, ipb = mk(), mk()
    xs = torch.randn(2, 3, D)
    with torch.no_grad(), _FakeAllReduce():
        for t in range(3):
            ipa.seqlen_offset = ipb.seqlen_offset = t
            a, _ = ref_blk(xs[:, t:t + 1], inference_params=ipa)
            b, _ = blk(xs[:, t:t + 1], inference_params=ipb)
            assert a.abs().max() > 1e-3
            rel = ((a - b).abs().max() / a.abs().max()).item()
            assert rel < 1e-5, f"{kind} step {t}: pretransposed differs, rel {rel:.2e}"
