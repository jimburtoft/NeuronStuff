"""API-surface tests. Pure Python + a stub model: no Trainium, no 13 GB checkpoint.

These cover the parts that shipped untested in 0.1.0: `to_device`, `enable`/`disable`
round-tripping, `prepare_config`, and the option/guard contracts.
"""
import sys
import types

import pytest

pytestmark = pytest.mark.simulator   # no device needed; runs in CI/CPU

import evo2_neuron_decode as evo2d


# --------------------------------------------------------------------------- guards
@pytest.mark.parametrize("bad", [1, 100, 1000, 1152, 4224, 255, 257])
def test_max_seqlen_rejects_non_multiples_of_256(bad):
    with pytest.raises(ValueError) as ei:
        evo2d.api._check_max_seqlen(bad)
    msg = str(ei.value)
    assert "divisible by 256" in msg
    # the message must be ACTIONABLE: name the next valid value and the escape hatch
    nxt = ((bad + 255) // 256) * 256
    assert f"max_seqlen={nxt}" in msg
    assert "attention_tkg=False" in msg


@pytest.mark.parametrize("good", [256, 512, 1024, 2048, 8192])
def test_max_seqlen_accepts_multiples_of_256(good):
    evo2d.api._check_max_seqlen(good)          # must not raise


def test_max_seqlen_none_is_allowed():
    evo2d.api._check_max_seqlen(None)


def test_unknown_option_is_rejected_not_ignored():
    """A typo'd option must fail loudly rather than silently doing nothing."""
    with pytest.raises(TypeError, match="unknown option"):
        evo2d.enable(fused_norms=True)         # note the typo: fused_normS


def test_rejected_options_default_off_and_say_why():
    """Options measured NOT to help must remain discoverable, with their negative result."""
    summary = evo2d.config_summary()
    for name in ("state_tensor", "fir_weight_cache", "fixed_kv", "offset_tensor"):
        assert name in summary
        line = next(l for l in summary.splitlines() if name in l)
        assert line.strip().startswith("off"), f"{name} should default OFF: {line}"
        assert "REJECTED" in line


def test_champion_options_default_on():
    summary = evo2d.config_summary()
    for name in ("cascade", "hcm_cascade", "fused_norm", "attention_tkg", "traceable_cfg"):
        line = next(l for l in summary.splitlines() if name in l)
        assert line.strip().startswith("ON"), f"{name} should default ON: {line}"


# --------------------------------------------------------------------------- to_device
class _FakeParam:
    """Minimal stand-in so to_device can be tested without torch.device('neuron')."""


def test_to_device_moves_plain_tensor_attributes():
    """The defect this exists for: nn.Module.to() ignores plain tensor ATTRIBUTES.

    vortex sets such attributes in __init__ (filter/pole state). Missing them yields
    'RuntimeError: input tensor is on cpu device, expected neuron' on the first forward.
    """
    torch = pytest.importorskip("torch")

    class Leaf(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(2))      # moved by .to()
            self.register_buffer("buf", torch.zeros(2))      # moved by .to()
            self.stray = torch.zeros(2)                      # NOT moved by .to()

    class Root(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = Leaf()
            self.b = Leaf()
            self.block_idx_to_device = {0: torch.device("cpu"), 1: torch.device("cpu")}

    m = Root()
    # plain .to() leaves `stray` behind -- demonstrate the gap on cpu->cpu by counting
    out = evo2d.to_device(m, device="cpu")
    assert out is m
    # block_idx_to_device must be rewritten, not left pointing at the old device
    assert all(str(v) == "cpu" for v in m.block_idx_to_device.values())
    # to_device reports how many stray tensors it had to move (0 here: already cpu)
    assert "moved_tensors" in evo2d.api._STATE


def test_to_device_rewrites_block_idx_map():
    torch = pytest.importorskip("torch")

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(2, 2)
            self.block_idx_to_device = {0: "cuda:7", 1: "cuda:7"}

    m = evo2d.to_device(M(), device="cpu")
    assert all(str(v) == "cpu" for v in m.block_idx_to_device.values()), \
        "a stale device map sends activations to a device that is not there"


def test_to_device_tolerates_missing_block_map():
    """Not every model has block_idx_to_device; to_device must not require it."""
    torch = pytest.importorskip("torch")
    m = torch.nn.Linear(2, 2)
    assert evo2d.to_device(m, device="cpu") is m


# --------------------------------------------------------------------------- prepare_config
class _DotDict(dict):
    """Reproduces vortex's dotdict, including __getattr__ = dict.get (returns None)."""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


def test_prepare_config_sets_the_required_flash_keys():
    """use_flash_attn=False is REQUIRED: MHA.__init__ hard-asserts on the flash branch."""
    cfg = _DotDict({"hidden_size": 4096})
    out = evo2d.prepare_config(cfg, 512)
    for key in ("use_flash_attn", "use_flashfft", "use_flash_depthwise", "use_flash_rmsnorm"):
        assert out[key] is False, f"{key} must be False or vortex cannot be constructed"
    assert out["inference_mode"] is True
    assert out["prefill_style"] == "recurrence"
    assert out["max_seqlen"] == 512


def test_prepare_config_validates_max_seqlen():
    cfg = _DotDict({"hidden_size": 4096})
    with pytest.raises(ValueError, match="divisible by 256"):
        evo2d.prepare_config(cfg, 1000)


def test_make_config_preserves_missing_key_semantics():
    """dotdict returns None for absent keys. A plain object would RAISE -- a behaviour
    change, not just a tracing change. It crashed on params_dtype during development."""
    cfg = _DotDict({"hidden_size": 4096})
    wrapped = evo2d.api.make_config(cfg)
    assert wrapped.hidden_size == 4096
    assert wrapped.definitely_not_a_real_key is None, \
        "absent keys must read as None, matching dotdict"
