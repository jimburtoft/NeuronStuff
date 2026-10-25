"""Logit parity against an fp32 CPU reference -- the WELL-CONDITIONED correctness gate.

WHY NOT TOKENS
    An fp32 CPU reference of this model on a real DNA prompt gives top1-top2 logit margins of
    **0.013-0.057**. bf16 has 8 mantissa bits, so at these logit magnitudes bf16 quantisation
    alone is ~0.03-0.08 -- the SAME ORDER as the margin. The greedy argmax is therefore decided
    by rounding, not by the model, and three reduction orders give three different token streams:
        fp32 CPU        [65, 65, 65, 65, 65, 65]
        eager MHA       [67, 65, 71, 71, 65, 65]
        shipped Neuron  [67, 71, 71, 67, 65, 65]
    all defensible. Any gate asserting exact token equality ACROSS implementations is measuring
    rounding. This compares LOGIT VECTORS instead, which are well-conditioned.

REFERENCE PROVENANCE
    data/cpu_fp32_reference.json, produced by ../harness/cpu_fp32_reference.py: same checkpoint,
    same config keys, CPU, fp32, no NKI kernels, no bf16, no optimization patches (only the
    import shim and iir_recurrence_patch, which is a checkpoint-compatibility fix that upstream
    requires on ANY device -- upstream raises IndexError without it because evo2_7b_base stores
    REAL modal parameters while upstream assumes complex).
"""
import json
import os
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.hardware

REF = os.environ.get("EVO2_FP32_REF", os.path.join(os.path.dirname(__file__), "..", "data", "cpu_fp32_reference.json"))
VORTEX = os.environ.get("VORTEX_REPO", os.path.expanduser("~/vortex-repo"))
WEIGHTS = os.environ.get("EVO2_WEIGHTS", os.path.expanduser("~/weights"))


def _load_ref():
    if os.path.exists(REF):
        return json.load(open(REF))
    pytest.fail(f"fp32 reference not found at {REF}; regenerate with harness/cpu_fp32_reference.py")


@pytest.mark.slow
def test_prefill_logits_match_fp32_reference():
    """Prefill logits must track the fp32 reference. No decode divergence has accumulated yet,
    so this isolates the model+kernels from any step-to-step drift."""
    ref = _load_ref()
    prompt = ref["prompt"]
    ms = ref["max_seqlen"]
    script = textwrap.dedent(f"""
        import os, sys, io, json, yaml, torch
        sys.path.insert(0, {VORTEX!r})
        import evo2_neuron_decode as evo2d
        from evo2_neuron_decode.patches import neuron_shim; neuron_shim.install()
        from vortex.model.utils import dotdict
        from vortex.model.model import StripedHyena
        evo2d.enable(max_seqlen={ms})
        cfg = dotdict(yaml.safe_load(open(os.path.join({VORTEX!r}, "configs", "evo2-7b-8k.yml"))))
        cfg = evo2d.prepare_config(cfg, {ms})
        torch.serialization.add_safe_globals([io.BytesIO])
        m = StripedHyena(cfg)
        m.load_state_dict(torch.load(os.path.join({WEIGHTS!r}, "evo2_7b_base.pt"),
                                     map_location="cpu", weights_only=False), strict=False)
        m = evo2d.to_device(m, dtype=torch.bfloat16).eval()
        ipd = m.initialize_inference_params()
        for n in ipd: ipd[n].max_batch_size = 1
        ids = torch.tensor([[ord(c) for c in {prompt!r}]], dtype=torch.long, device="neuron")
        with torch.no_grad(): out, ipd = m(ids, inference_params_dict=ipd)
        print("LOGITS " + json.dumps(out[:, -1].float().cpu()[0].tolist()))
    """)
    p = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                       timeout=5400)
    line = next((l for l in p.stdout.splitlines() if l.startswith("LOGITS ")), None)
    assert line, f"no logits:\n{p.stdout[-800:]}\n{p.stderr[-1200:]}"

    import math
    got = json.loads(line[7:])
    exp = ref["prefill_logits"]
    assert len(got) == len(exp)
    dot = sum(a * b for a, b in zip(got, exp))
    na = math.sqrt(sum(a * a for a in got))
    nb = math.sqrt(sum(b * b for b in exp))
    cos = dot / (na * nb)
    maxabs = max(abs(a - b) for a, b in zip(got, exp))

    # bf16-vs-fp32 on a 512-wide vector: cos_sim should be ~1 to several decimals. This is a
    # deliberately loose-but-meaningful floor -- a real defect (wrong kernel, wrong lane order,
    # broken state) drops cos_sim far below this, as an earlier lane-order defect did (cos_sim -0.894).
    assert cos > 0.999, f"prefill logits diverge from fp32: cos_sim={cos:.8f} max|d|={maxabs:.4f}"
    print(f"\n  prefill logits vs fp32 CPU: cos_sim={cos:.8f}  max|diff|={maxabs:.4f}")


@pytest.mark.slow
def test_reference_margins_are_documented_as_tight():
    """Guard the REASONING, not just the numbers.

    If a future reference is regenerated with comfortable margins, the token-equality gates
    could legitimately be restored. This test fails loudly if the margins ever stop being
    tie-tight, so nobody keeps citing 'tokens are unreliable' after it stops being true.
    """
    ref = _load_ref()
    margins = ref["margins"]
    bf16_quantisation_at_unit_scale = 2 ** -8      # 8 mantissa bits
    tight = [m for m in margins if m < 20 * bf16_quantisation_at_unit_scale]
    assert tight, (
        f"margins {margins} are no longer tie-tight relative to bf16 resolution "
        f"({bf16_quantisation_at_unit_scale:.4f}). Token-equality gates may be valid again -- "
        f"revisit test_greedy_parity_vs_reference and the WHY NOT TOKENS reasoning.")
