"""CPU parity of the package against the UPSTREAM model (vendored stripedhyena, FFT long conv,
complex64 IIR, SDPA reference MHA, flash-formula RMSNorm), fp32, tiny random model with stable
complex poles. No Trainium needed:  pytest tests -m cpu

The package is process-global (it patches the vendored upstream classes), so the reference model is
built and run BEFORE the patches are installed; every test reuses that module-scoped reference.
"""
import copy
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.cpu

CFG = dict(
    vocab_size=512, hidden_size=256, num_filters=256, num_layers=4,
    attn_layer_idxs=[1, 3], hyena_layer_idxs=[0, 2],
    short_filter_length=3, num_attention_heads=4, short_filter_bias=True,
    eps=1e-5, state_size=2, rotary_emb_base=500000, make_vocab_size_divisible_by=8,
    proj_groups=4, hyena_filter_groups=1, column_split_hyena=True, column_split=False,
    model_parallel_size=1, pipe_parallel_size=1, tie_embeddings=False, inner_mlp_size=512,
    mha_out_proj_bias=False, qkv_proj_bias=False, max_seqlen=256, max_batch_size=2,
    final_norm=True, use_flash_attn=True, use_flash_rmsnorm=False, use_flash_depthwise=False,
    use_flashfft=False, use_laughing_hyena=False, inference_mode=False, prefill_style="fft",
    attn_block_dtype=torch.float32, mlp_dtype=torch.float32, hyena_block_dtype=torch.float32,
)
B, P, N, W = 2, 133, 12, 256


def rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


def _stabilize(m, seed):
    g = torch.Generator().manual_seed(seed)
    for b in m.blocks:
        if hasattr(b, "filter"):
            with torch.no_grad():
                sh = b.filter.poles.shape[:-1]
                r = 0.94 + 0.0599 * torch.rand(sh, generator=g)
                th = torch.rand(sh, generator=g) * 0.35
                b.filter.poles[..., 0] = r * torch.cos(th)
                b.filter.poles[..., 1] = r * torch.sin(th)
                b.filter.D.normal_(generator=g)
        for p in b.parameters():
            if p.dim() == 1 and p.abs().sum() == 0:
                with torch.no_grad():
                    p.normal_(0, 0.1, generator=g)


@pytest.fixture(scope="module")
def ref():
    """Upstream reference: built and run before the Neuron patches are installed."""
    import striped_hyena_neuron as shn
    from striped_hyena_neuron import sdpa_mha, rmsnorm
    assert not shn.is_enabled(), "run the CPU tests in a fresh process"
    sdpa_mha.install()      # validated drop-in for flash_attn MHA (CUDA-only)
    rmsnorm.install()       # flash-formula RMSNorm (the checkpoint's trained norm)
    from stripedhyena import model as M
    from stripedhyena.utils import dotdict
    torch.manual_seed(0)
    m = M.StripedHyena(dotdict(CFG)).eval()
    _stabilize(m, 0)
    sd = copy.deepcopy(m.state_dict())
    ids = torch.randint(0, CFG["vocab_size"], (B, P + N), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        full, _ = m(ids)                                   # uncached, FFT long conv
        ip = m.initialize_inference_params()
        ip["mha"].max_batch_size = B
        ip["mha"].max_seqlen = W
        pre, ip = m(ids[:, :P], inference_params_dict=ip)  # cached prefill (modal FFT state)
        steps = []
        for i in range(N):                                 # complex64 recurrent decode
            ip["mha"].seqlen_offset = P + i
            ip["hyena"].seqlen_offset = P + i
            o, ip = m(ids[:, P + i: P + i + 1], inference_params_dict=ip)
            steps.append(o[:, -1])
    return {"sd": sd, "ids": ids, "full": full, "pre_last": pre[:, -1], "steps": torch.stack(steps, 1)}


@pytest.fixture(scope="module")
def port_model(ref):
    import striped_hyena_neuron as shn
    from striped_hyena_neuron import hyena
    shn.enable(rms_custom=False)        # torch.rms_norm is checked separately (same formula)
    hyena.CHUNK = 16
    from stripedhyena import model as M
    from stripedhyena.utils import dotdict
    m = M.StripedHyena(dotdict(CFG)).eval()
    from striped_hyena_neuron import attention
    for b in m.blocks:
        if hasattr(b, "inner_mha_cls"):
            b.inner_mha_cls.__class__ = attention.NeuronMHA
    m.load_state_dict(ref["sd"])
    return m


def _run_port(m, ids, seg=None, mode="eager"):
    from striped_hyena_neuron.prefill import Prefiller
    from striped_hyena_neuron.decode import Decoder
    with torch.inference_mode():
        lg, _, S = Prefiller(m, "eager")(ids[:, :P], W, seg=seg)
        d = Decoder(m, mode)
        steps = [d(ids[:, P + i], S)[0] for i in range(N)]
    return lg, torch.stack(steps, 1), S


def test_prefill_and_decode_match_upstream(ref, port_model):
    lg, steps, _ = _run_port(port_model, ref["ids"])
    assert rel(lg, ref["pre_last"]) < 1e-5
    assert rel(steps, ref["steps"]) < 1e-5


def test_segmented_prefill_exact(ref, port_model):
    lg, steps, _ = _run_port(port_model, ref["ids"], seg=48)    # 5 (remainder) + 32 + 48 + 48
    assert rel(lg, ref["pre_last"]) < 1e-5
    assert rel(steps, ref["steps"]) < 1e-5


def test_decode_matches_uncached_forward(ref, port_model):
    _, steps, _ = _run_port(port_model, ref["ids"])
    # step i consumes token P+i, so it predicts position P+i
    assert rel(steps, ref["full"][:, P:P + N]) < 1e-5


def test_bucketed_attention_exact(ref, port_model):
    from striped_hyena_neuron.prefill import Prefiller
    from striped_hyena_neuron.decode import Decoder, set_bucket
    with torch.inference_mode():
        _, _, S = Prefiller(port_model, "eager")(ref["ids"][:, :P], W)
        d = Decoder(port_model, "eager")
        set_bucket(d, S, P + N, bmin=64)
        assert d.steps[1].curr == 256 or d.steps[1].curr < W
        got = torch.stack([d(ref["ids"][:, P + i], S)[0] for i in range(N)], 1)
    assert rel(got, ref["steps"]) < 1e-5


def test_rms_custom_formula():
    from striped_hyena_neuron.rmsnorm import rmsnorm_flash_formula
    x, w = torch.randn(3, 64), torch.randn(64)
    a = rmsnorm_flash_formula(x, w, 1e-5)
    b = torch.rms_norm(x.float(), (64,), w.float(), 1e-5)
    assert rel(a, b) < 1e-6


def test_api_surface():
    import striped_hyena_neuron as shn
    s = shn.config_summary()
    assert "rms_custom" in s and "REJECTED" in s
    with pytest.raises(TypeError):
        shn.enable(not_an_option=True)
    d = shn.decode_opts(B=1, window=32768, tp=4)
    assert d["mode"] == "step" and not d["tkg"]          # step graph measured with SDPA
    d = shn.decode_opts(B=128, window=8192, tp=4)
    assert d["mode"] == "blocks:32" and d["tkg"]
    d = shn.decode_opts(B=512, window=2048, tp=4)
    assert not d["tkg"]
    d = shn.decode_opts(B=32, window=32768, tp=4)
    assert d["tkg"]
    cfg = shn.make_config(4096)
    assert cfg.max_seqlen == 4096 and cfg.get("no_such_key", 7) == 7 and cfg.no_such_key is None
    with pytest.raises(ValueError):
        shn.make_config(65536)


def test_tp2_matches_unsharded():
    """TP=2 over gloo (2 CPU processes) == unsharded, prefill + decode (subprocess: own process group)."""
    import os
    w = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tp2_worker.py")
    p = subprocess.run([sys.executable, w], capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-3000:]
