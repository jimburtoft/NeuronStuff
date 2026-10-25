"""Evo 2 decode throughput: TP=4 at a stated KV fill (8K context), or one core at short context.

This is the measurement behind the 8K / KV-fill rows and the max_seqlen=256 DP=4 row of BENCHMARK.md.

Two ways to reach a KV fill F:
  --fill-mode offset (default): prefill the prompt (EVO2_PROMPT, default the 128-nt benchmark prompt) and then set
      seqlen_offset = F. Slots [len(prompt), F) hold zeros instead of real K/V: same kernels, shapes and reads as a
      real fill (attention reads the filled prefix bucket), different values. This is how the published rows were
      measured.
  --fill-mode prompt: prefill a real F-token prompt (seed-0 random ACGT)
      with the package's row-chunked, length-bucketed `evo2d.prefill`, so every KV slot up to F is real.

Timing: 4 warm-up decode steps, then the mean of --steps timed steps (default 10), then a 24-step greedy-token
fingerprint (TOKSHA). Prints one `RESULT {...}` JSON line on rank 0 and writes it to --out.

TP=4 (one trn2 device, LNC=2):
  torchrun --standalone --nproc_per_node=4 harness/bench_fill.py --tp 4 --bs 88 --fill 1024
One core, short context (run four of these with NEURON_RT_VISIBLE_CORES=0..3 for DP=4):
  NEURON_RT_VISIBLE_CORES=0 python harness/bench_fill.py --tp 1 --bs 128 --max-seqlen 256

Needs VORTEX_REPO (vortex at the pinned commit) and EVO2_WEIGHTS (directory with evo2_7b_base.pt). The package
must be importable (`pip install -e package --no-deps`, as run.sh does).
"""
import argparse
import gc
import hashlib
import io
import json
import os
import random
import sys
import time

import torch
import yaml

BENCH_PROMPT = ("GCTAAAGACAATTACATAACATACACGTCAGCACGAAACTTGTTGGCCCAGTGTGAATCGCTTAAGGGTTAAGTAAGTGTGATGCATACGCCTTTA"
                "CTTGCTGTGTCCACCCCATCGGACTGGCATTT")
PREFILL_BUCKETS = (128, 256, 512, 1024, 2048, 4096, 8192)

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--tp", type=int, default=4, help="tensor-parallel degree (4 under torchrun, or 1)")
ap.add_argument("--bs", type=int, required=True)
ap.add_argument("--max-seqlen", type=int, default=8192)
ap.add_argument("--fill", type=int, default=None, help="KV fill at decode (default: the prompt length)")
ap.add_argument("--fill-mode", choices=("offset", "prompt"), default="offset")
ap.add_argument("--steps", type=int, default=10)
ap.add_argument("--warmup", type=int, default=4)
ap.add_argument("--chunk-rows", type=int, default=8, help="rows per prefill call (--prefill chunked)")
ap.add_argument("--prefill", choices=("auto", "direct", "chunked"), default="auto",
                help="direct = one full-batch forward (what the published TP=4 rows used); chunked = evo2d.prefill, "
                     "chunk_rows rows at a time (needed at B=128 on one core, and for long prompts). auto = direct for "
                     "--tp>1 with --fill-mode offset, chunked otherwise")
ap.add_argument("--out", default=None, help="JSON result file (rank 0)")
a = ap.parse_args()

VORTEX = os.environ.get("VORTEX_REPO", os.path.expanduser("~/vortex-repo"))
WEIGHTS = os.environ.get("EVO2_WEIGHTS", os.path.expanduser("~/weights"))
sys.path.insert(0, VORTEX)

import evo2_neuron_decode as evo2d  # noqa: E402
from evo2_neuron_decode.patches import neuron_shim  # noqa: E402

neuron_shim.install()
from vortex.model.model import StripedHyena  # noqa: E402
from vortex.model.utils import dotdict  # noqa: E402

cfg = dotdict(yaml.safe_load(open(os.path.join(VORTEX, "configs", "evo2-7b-8k.yml"))))
cfg = evo2d.prepare_config(cfg, a.max_seqlen)
cfg.max_batch_size = a.bs
evo2d.enable()
torch.serialization.add_safe_globals([io.BytesIO])
ckpt = os.path.join(WEIGHTS, "evo2_7b_base.pt")

RK, WORLD = 0, 1
if a.tp > 1:
    import torch.distributed as dist
    from evo2_neuron_decode import tp as tpm

    dist.init_process_group(backend="neuron")
    RK, WORLD = dist.get_rank(), dist.get_world_size()
    assert WORLD == a.tp, f"--tp {a.tp} but world size {WORLD}"
    # Host-RAM safe on trn2.3xlarge (124 GB): ranks build, load (fp32, ~26 GB) and shard one at a time.
    for r in range(WORLD):
        if r == RK:
            m = StripedHyena(cfg)
            m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False), strict=False)
            tpm.shard_model(m, RK, WORLD)
            m = evo2d.to_device(m, dtype=torch.bfloat16)
            gc.collect()
        dist.barrier()
else:
    m = StripedHyena(cfg)
    m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False), strict=False)
    m = evo2d.to_device(m, dtype=torch.bfloat16)
    gc.collect()
evo2d.compile_blocks(m)

if a.fill_mode == "prompt":
    if a.fill is None:
        ap.error("--fill-mode prompt needs --fill")
    rnd = random.Random(0)
    prompt = "".join(rnd.choice("ACGT") for _ in range(a.fill))
else:
    prompt = os.environ.get("EVO2_PROMPT", BENCH_PROMPT)
fill = a.fill if a.fill is not None else len(prompt)
assert len(prompt) <= fill < a.max_seqlen, f"need len(prompt) {len(prompt)} <= fill {fill} < max_seqlen {a.max_seqlen}"

prefill_mode = a.prefill
if prefill_mode == "auto":
    prefill_mode = "direct" if (a.tp > 1 and a.fill_mode == "offset") else "chunked"
ids = torch.tensor([[ord(c) for c in prompt]], dtype=torch.long).repeat(a.bs, 1).to("neuron")
t0 = time.perf_counter()
with torch.no_grad():
    if prefill_mode == "direct":
        ipd = m.initialize_inference_params()
        for n in ipd:
            ipd[n].max_batch_size = a.bs
        out, _ = m(ids, inference_params_dict=ipd)
        last = out[:, -1]
        del out
    else:
        ipd, last = evo2d.prefill(m, ids, chunk_rows=min(a.chunk_rows, a.bs), max_batch_size=a.bs,
                                  buckets=PREFILL_BUCKETS if a.fill_mode == "prompt" else None)
torch.neuron.synchronize()
prefill_s = time.perf_counter() - t0
for n in ipd:
    ipd[n].seqlen_offset = fill
nxt = last.float().argmax(-1, keepdim=True).to("neuron")


def one():
    global nxt
    with torch.no_grad():
        o, _ = m(nxt, inference_params_dict=ipd)
    nxt = o[:, -1].float().argmax(-1, keepdim=True).to("neuron")
    for n in ipd:
        ipd[n].seqlen_offset += 1
    return o


for _ in range(a.warmup):
    one()
# Decode steps are queued asynchronously (the greedy argmax stays on device), so synchronize on BOTH sides of the
# timed loop. A harness that synced only before it read ~4.8% fast at DP=4 @256 (140.5 vs 147.3 ms on the same
# host, 2026-10-24).
torch.neuron.synchronize()
t0 = time.perf_counter()
for _ in range(a.steps):
    one()
torch.neuron.synchronize()
ms = (time.perf_counter() - t0) / a.steps * 1000
toks = []
for _ in range(24):
    o = one()
    toks.append(nxt.view(-1).cpu().clone())
tk = torch.stack(toks, 1)
lg = o[:, -1].float().cpu()

res = {
    "tp": a.tp, "bs": a.bs, "max_seqlen": a.max_seqlen, "fill": fill, "fill_mode": a.fill_mode,
    "prompt_len": len(prompt), "prefill": prefill_mode, "step_ms": round(ms, 3), "tok_s": round(a.bs / (ms / 1000), 2),
    "prefill_s": round(prefill_s, 2),
    "toksha": hashlib.sha1(tk.numpy().tobytes()).hexdigest()[:12],
    "row0_tokens": tk[0, :12].tolist(), "logit_l2": round(float(lg.norm()), 5),
    "torch": torch.__version__,
}
if RK == 0:
    print("RESULT " + json.dumps(res), flush=True)
    # compact one-line form
    print(f"FILL {fill} STEP_MS {ms:.3f} TOKSHA {res['toksha']}", flush=True)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)
if a.tp > 1:
    dist.barrier()
    dist.destroy_process_group()
