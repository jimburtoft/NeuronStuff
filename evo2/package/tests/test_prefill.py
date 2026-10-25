"""evo2_neuron_decode.prefill -- row-chunked and length-bucketed prefill.

Small random StripedHyena on CPU (upstream vortex code paths, no kernels). Reference = one exact-length
prefill call over all rows. Checked: last-position logits and 12 greedy decode steps.
"""
import os, sys
import pytest
import torch

VR = os.environ.get("VORTEX_REPO", os.path.expanduser("~/vortex-repo"))
pytestmark = pytest.mark.skipif(not os.path.isdir(VR), reason="vortex repo not available")


def _model():
    sys.path.insert(0, VR)
    from evo2_neuron_decode.patches import neuron_shim, iir_recurrence_patch
    neuron_shim.install(); iir_recurrence_patch.install()
    import yaml
    from vortex.model.utils import dotdict
    from vortex.model.model import StripedHyena
    cfg = dotdict(yaml.safe_load(open(f"{VR}/configs/evo2-7b-8k.yml")))
    for k, v in dict(use_flash_attn=False, use_flashfft=False, use_flash_depthwise=False, use_flash_rmsnorm=False,
                     prefill_style="recurrence", inference_mode=True, use_fp8_input_projections=False,
                     hidden_size=256, inner_mlp_size=512, num_attention_heads=4, num_layers=8,
                     attn_layer_idxs=[3], hcl_layer_idxs=[2, 6], hcm_layer_idxs=[0, 4], hcs_layer_idxs=[1, 5, 7],
                     hcm_filter_groups=16, hcs_filter_groups=16, hcl_filter_groups=256, num_filters=256,
                     max_seqlen=512, max_batch_size=8).items():
        cfg[k] = v
    torch.manual_seed(0)
    m = StripedHyena(cfg).float().eval()
    for n, p in m.named_parameters():
        if "log_poles" in n: p.data.uniform_(-2.3, -0.01)
        elif p.dim() >= 2: torch.nn.init.normal_(p, std=2.0 / 256 ** 0.5)
        else: torch.nn.init.normal_(p, std=0.2)
    return m


def _decode(m, ipd, last, steps=12):
    toks, logits = [], []
    nxt = last.argmax(-1, keepdim=True)
    for _ in range(steps):
        with torch.no_grad():
            o, _ = m(nxt, inference_params_dict=ipd)
        logits.append(o[:, -1]); nxt = o[:, -1].argmax(-1, keepdim=True); toks.append(nxt)
        for n in ipd: ipd[n].seqlen_offset += 1
    return torch.cat(toks, 1), torch.stack(logits, 1)


def _reference(m, ids):
    ipd = m.initialize_inference_params()
    for n in ipd: ipd[n].max_batch_size = ids.shape[0]
    with torch.no_grad():
        out, _ = m(ids, inference_params_dict=ipd)
    for n in ipd: ipd[n].seqlen_offset = ids.shape[1]
    return ipd, out[:, -1]


@pytest.mark.parametrize("L,chunk,buckets", [(37, 3, None), (37, 3, (64,)), (9, 4, (16, 32))])
def test_prefill_matches_reference(L, chunk, buckets):
    m = _model()
    from evo2_neuron_decode.prefill import prefill
    torch.manual_seed(1)
    ids = torch.randint(65, 90, (6, L))
    ipd_r, last_r = _reference(m, ids)
    toks_r, lg_r = _decode(m, ipd_r, last_r, steps=6)
    ipd_p, last_p = prefill(m, ids, chunk_rows=chunk, buckets=buckets, pad_token=78)
    rel0 = float((last_p - last_r).norm() / last_r.norm())
    assert rel0 < 1e-5, f"prefill logits differ: rel {rel0:.2e}"
    toks_p, lg_p = _decode(m, ipd_p, last_p, steps=6)
    rel = float((lg_p - lg_r).norm() / lg_r.norm())
    assert rel < 1e-4, f"decode after prefill differs: rel {rel:.2e}"
    assert torch.equal(toks_p, toks_r)
