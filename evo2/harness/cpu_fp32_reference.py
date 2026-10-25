"""fp32 CPU reference for evo2_7b_base decode (produces package/data/cpu_fp32_reference.json).

Runs the SAME model on CPU in fp32 -- no Neuron, no bf16, no NKI kernels, no patches beyond the import shim and the
real-pole IIR prefill -- and records the prefill logits, the next --steps greedy tokens, their top-1/top-2 margins
and the full logit vectors. tests/test_logit_parity.py compares the Neuron path against it.

Not a performance measurement: CPU fp32 decode is minutes per token. fp32 is not ground truth either (it is another
reduction order), but it is far from the bf16 tie boundary (24 vs 8 mantissa bits), so where bf16 implementations
disagree, fp32 says which side the exact value sits on. Needs ~60 GB host RAM.

    python harness/cpu_fp32_reference.py --weights ~/weights/evo2_7b_base.pt --out package/data/cpu_fp32_reference.json
"""
import argparse, io, json, os, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=6)
ap.add_argument("--prompt", default="ACGTACGTGGCCATTAGCTAGCTAGGCATCGA")
ap.add_argument("--max-seqlen", type=int, default=256)
ap.add_argument("--dtype", default="float32", choices=["float32", "float64", "bfloat16"])
ap.add_argument("--vortex", default=os.environ.get("VORTEX_REPO", os.path.expanduser("~/vortex-repo")))
ap.add_argument("--weights", default=os.path.join(os.environ.get("EVO2_WEIGHTS", os.path.expanduser("~/weights")),
                                                  "evo2_7b_base.pt"))
ap.add_argument("--out", default="package/data/cpu_fp32_reference.json")
a = ap.parse_args()

sys.path.insert(0, a.vortex)
import torch
from evo2_neuron_decode.patches import neuron_shim, iir_recurrence_patch
neuron_shim.install()          # only to make vortex importable without the CUDA flash ext

# iir_recurrence_patch is REQUIRED even on CPU, and it is not a Neuron optimization: upstream
# `prefill_via_direct_recurrence` assumes a COMPLEX modal parameterization (it indexes
# poles[:, :, 0] then [..., 0]/[..., 1] as re/im), but the shipped evo2_7b_base checkpoint stores
# REAL parameters -- log_poles (4096,16,1), residues (4096,16). Upstream therefore raises
# `IndexError: too many indices for tensor of dimension 2` on CPU exactly as it would anywhere.
# The patch is pure PyTorch, device-agnostic, and is the real-valued specialization of the same
# recurrence (for real poles the imaginary track is identically zero).
iir_recurrence_patch.install()
from vortex.model.utils import dotdict
from vortex.model.model import StripedHyena

torch.set_num_threads(os.cpu_count() or 8)
DT = getattr(torch, a.dtype)

cfg = dotdict(__import__("yaml").safe_load(
    open(os.path.join(a.vortex, "configs", "evo2-7b-8k.yml"))))
# Same required keys the Neuron path sets -- flash is CUDA-only and MHA.__init__ asserts on it.
cfg.use_flash_attn = False
cfg.use_flashfft = False
cfg.use_flash_depthwise = False
cfg.use_flash_rmsnorm = False
cfg.prefill_style = "recurrence"
cfg.inference_mode = True
cfg.max_seqlen = a.max_seqlen

print(f"building on CPU in {a.dtype} ...", flush=True)
t0 = time.time()
torch.serialization.add_safe_globals([io.BytesIO])
m = StripedHyena(cfg)
sd = torch.load(a.weights, map_location="cpu", weights_only=False)
m.load_state_dict(sd, strict=False)
del sd
m = m.to(DT).eval()
print(f"  built + loaded in {time.time()-t0:.0f}s", flush=True)

ids_list = [ord(c) for c in a.prompt]
ids = torch.tensor([ids_list], dtype=torch.long)
ipd = m.initialize_inference_params()
for n in ipd:
    ipd[n].max_batch_size = 1

t0 = time.time()
with torch.no_grad():
    out, ipd = m(ids, inference_params_dict=ipd)
print(f"  prefill L={len(ids_list)} in {time.time()-t0:.0f}s", flush=True)

off = len(ids_list)
for n in ipd:
    ipd[n].seqlen_offset = off

logits = out[:, -1].to(torch.float32)
nxt = logits.argmax(-1, keepdim=True)
toks, margins, all_logits = [], [], []
# The prefill logits are part of the reference too -- a Neuron run can be compared at step 0
# before any decode-step divergence accumulates.
prefill_logits = logits[0].tolist()
for i in range(a.steps):
    t0 = time.time()
    with torch.no_grad():
        o, _ = m(nxt, inference_params_dict=ipd)
    lg = o[:, -1].to(torch.float32)[0]
    top2 = torch.topk(lg, 2)
    # The MARGIN is the point: it says whether a bf16 disagreement here is a near-tie or real.
    margin = float(top2.values[0] - top2.values[1])
    tok = int(top2.indices[0])
    toks.append(tok); margins.append(margin)
    all_logits.append(lg.tolist())
    nxt = torch.tensor([[tok]], dtype=torch.long)
    off += 1
    for n in ipd:
        ipd[n].seqlen_offset = off
    print(f"  step {i}: tok={tok} ({chr(tok) if 32 < tok < 127 else '?'}) "
          f"top1-top2 margin={margin:.4f}  [{time.time()-t0:.0f}s]", flush=True)

os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump({"dtype": a.dtype, "prompt": a.prompt, "max_seqlen": a.max_seqlen,
           "tokens": toks, "margins": margins,
           # FULL LOGIT VECTORS are the reference that matters. Tokens at these margins are
           # decided by rounding (margins 0.013-0.057 vs ~0.03-0.08 bf16 quantisation), so
           # token equality is not a well-conditioned test. Logit agreement is.
           "prefill_logits": prefill_logits,
           "step_logits": all_logits}, open(a.out, "w"), indent=1)
print(f"\nREFERENCE {a.dtype} tokens={toks}")
print(f"MARGINS {[round(x,4) for x in margins]}")
print(f"written to {a.out}")
