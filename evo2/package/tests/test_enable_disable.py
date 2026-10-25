"""enable() / disable() round-trip. Listed as untested in 0.1.0's followups.

TIER: marked `hardware` because `enable()` imports `torch_neuronx` (the patches call
`wrap_nki`), which only exists inside the AWS PyTorch Native container. It does NOT need a
Neuron device to be present, and it does NOT load the 13 GB checkpoint -- it only checks which
bound methods got replaced -- so it is fast (seconds) and runs on any container host.
"""
import pytest

pytestmark = pytest.mark.container

# neuron_shim must run BEFORE importing vortex: vortex.ops.attn_interface does a bare
# `import flash_attn_2_cuda`, which is the CUDA extension and does not exist here. The shim
# stubs it (and sets the flash entry points to None) so the module is importable at all.
import evo2_neuron_decode as evo2d
from evo2_neuron_decode.patches import neuron_shim
neuron_shim.install()

vortex = pytest.importorskip("vortex.model.model",
                             reason="vortex not on sys.path (set VORTEX_REPO)")
from vortex.model.layers import ParallelGatedMLP, RMSNorm
from vortex.model.model import HyenaCascade


@pytest.fixture(autouse=True)
def _clean():
    """Never leak patch state between tests -- these mutate global classes."""
    if evo2d.is_enabled():
        evo2d.disable()
    yield
    if evo2d.is_enabled():
        evo2d.disable()


def _fingerprint():
    """Identity of every method the package replaces."""
    return {
        "cascade": HyenaCascade.sequential_forward,
        "mlp": ParallelGatedMLP.forward,
        "norm": RMSNorm.forward,
    }


def test_enable_actually_replaces_methods():
    before = _fingerprint()
    evo2d.enable(max_seqlen=512)
    after = _fingerprint()
    changed = [k for k in before if before[k] is not after[k]]
    assert changed, "enable() replaced nothing -- the patches are not binding"
    assert "cascade" in changed and "norm" in changed


def test_disable_restores_every_method():
    """The round-trip that was untested. A leaked patch silently changes later runs."""
    before = _fingerprint()
    evo2d.enable(max_seqlen=512)
    reverted = evo2d.disable()
    after = _fingerprint()
    assert reverted, "disable() reverted nothing"
    for key, orig in before.items():
        assert after[key] is orig, f"{key} was not restored by disable()"


def test_is_enabled_tracks_state():
    assert not evo2d.is_enabled()
    evo2d.enable(max_seqlen=512)
    assert evo2d.is_enabled()
    evo2d.disable()
    assert not evo2d.is_enabled()


def test_double_enable_warns_and_does_not_double_patch():
    evo2d.enable(max_seqlen=512)
    once = _fingerprint()
    with pytest.warns(RuntimeWarning, match="called twice"):
        evo2d.enable(max_seqlen=512)
    assert _fingerprint() == once, "second enable() must not stack patches"


def test_disable_is_idempotent():
    evo2d.enable(max_seqlen=512)
    evo2d.disable()
    assert evo2d.disable() == [], "disable() on a clean state must be a no-op"


def test_repeated_cycles_do_not_drift():
    """Three cycles: a patch that fails to revert would accumulate."""
    before = _fingerprint()
    for _ in range(3):
        evo2d.enable(max_seqlen=512)
        evo2d.disable()
    assert _fingerprint() == before


def test_selective_enable_only_patches_what_is_asked():
    """Turning an OPTIMIZATION off must leave its target alone.

    NOTE RMSNorm is deliberately excluded from this check. `rmsnorm_patch` is an always-on
    LOWERING patch, not an optimization: stock `RMSNorm.forward` uses `Tensor.norm(2, ...)`,
    which lowers to `aten::pow.Scalar_out` -- unimplemented on Neuron, so it silently
    round-trips to the host and then dies with "input tensor is on cpu device". So
    `RMSNorm.forward` is replaced even with `fused_norm=False`; only the *fused kernel* is
    skipped. Asserting otherwise (as an earlier version of this test did) tests the wrong
    thing and would block a correct build.
    """
    before = _fingerprint()
    evo2d.enable(max_seqlen=512, cascade=False, hcm_cascade=False, fused_norm=False,
                 mlp_cat=False, attention_tkg=False, casts=False, state_concat=False)
    after = _fingerprint()
    assert after["cascade"] is before["cascade"], "cascade=False still patched HyenaCascade"
    assert after["mlp"] is before["mlp"], "mlp_cat=False still patched ParallelGatedMLP"


def test_lowering_patches_are_always_on():
    """The four lowering patches are NOT optional -- they are what makes the model run.

    Documents the distinction the test above depends on: rmsnorm_patch / fir_conv_patch /
    iir_recurrence_patch / rotary_patch remove CPU fallbacks and unsupported lowerings, so
    `enable()` installs them regardless of which optimizations are requested.
    """
    before = _fingerprint()
    evo2d.enable(max_seqlen=512, cascade=False, hcm_cascade=False, fused_norm=False,
                 mlp_cat=False, attention_tkg=False, casts=False, state_concat=False,
                 traceable_cfg=False)
    after = _fingerprint()
    assert after["norm"] is not before["norm"], \
        "rmsnorm_patch must be installed even with every optimization off"


def test_disable_strict_raises_on_failure(monkeypatch):
    """strict=True must surface a bad uninstall rather than warning past it."""
    from evo2_neuron_decode import patches as P
    evo2d.enable(max_seqlen=512)
    def boom():
        raise RuntimeError("simulated bad revert")
    monkeypatch.setattr(P.norm_patch, "uninstall", boom)
    with pytest.raises(RuntimeError, match="could not revert"):
        evo2d.disable(strict=True)
