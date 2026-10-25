"""Data-parallel serving: 4 independent workers, one per logical core.

WHY THIS IS A SCRIPT AND NOT AN API CALL
    DP on Neuron is process-level: each worker sets NEURON_RT_VISIBLE_CORES to a single core
    and holds its own full copy of the weights. There is no in-process API for it, so putting
    a `dp=4` argument on enable() would be a lie. This is the deployment shape instead.

MEASURED BY THIS SCRIPT (trn2.3xlarge, SDK 2.32, evo2_7b_base, bf16, all 32 blocks compiled)
    --workers 4 --batch-size 8 --steps 24
      core  ms/step   tok/s
         0     60.6  132.06
         1     60.9  131.44
         2     60.1  133.09
         3     58.7  136.19
      AGG            532.78     scaling 3.91x of 4 ideal (98% of linear)

    Other batch sizes, from the project's own decode harness (same config, so comparable but
    NOT produced by this script): DP=4 BS=1 119.5 tok/s, BS=32 1318.2 tok/s.

COST TRADE-OFF, stated plainly
    DP=4 multiplies THROUGHPUT, not latency. Per-request latency is unchanged (~33 ms/token at
    BS=1). It also costs 4x the HBM -- each worker loads its own 13 GB, which fits in 24 GB/core
    at LNC=2 but leaves little room. If you need lower latency rather than more of it, run DP=1.

STARTUP
    Each worker pays the ~79 s checkpoint load independently. Start them once and keep them
    warm; do not spawn per request.

    python examples/serve_dp4.py --workers 4 --batch-size 8 --steps 32
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

WORKER = r'''
import json, os, sys, time, io, yaml, torch
import evo2_neuron_decode as evo2d
from evo2_neuron_decode.patches import neuron_shim
neuron_shim.install()
from vortex.model.utils import dotdict
from vortex.model.model import StripedHyena

BS      = int(os.environ["W_BS"])
STEPS   = int(os.environ["W_STEPS"])
MAXSEQ  = int(os.environ["W_MAXSEQ"])
PROMPT  = int(os.environ["W_PROMPT"])
CORE    = os.environ["NEURON_RT_VISIBLE_CORES"]

evo2d.enable(max_seqlen=MAXSEQ)
cfg = dotdict(yaml.safe_load(open(os.environ["W_CFG"])))
cfg = evo2d.prepare_config(cfg, MAXSEQ)

torch.serialization.add_safe_globals([io.BytesIO])
m = StripedHyena(cfg)
m.load_state_dict(torch.load(os.environ["W_WEIGHTS"], map_location="cpu",
                             weights_only=False), strict=False)
m = evo2d.to_device(m, dtype=torch.bfloat16).eval()
evo2d.compile_blocks(m)

ipd = m.initialize_inference_params()
for n in ipd: ipd[n].max_batch_size = BS
ids = torch.tensor([[1] * PROMPT] * BS, dtype=torch.long, device="neuron")
with torch.no_grad(): out, ipd = m(ids, inference_params_dict=ipd)
off = PROMPT
for n in ipd: ipd[n].seqlen_offset = off
nxt = out[:, -1].float().cpu().argmax(-1, keepdim=True).to("neuron")

def step():
    global nxt, off
    with torch.no_grad(): o, _ = m(nxt, inference_params_dict=ipd)
    nxt = o[:, -1].float().cpu().argmax(-1, keepdim=True).to("neuron")
    off += 1
    for n in ipd: ipd[n].seqlen_offset = off

# Warm until the step time actually STABILISES, rather than for a fixed count. A fixed
# warmup is not enough: workers start staggered, so whichever one compiles first can still
# be compiling when its timed region opens. Measured symptom: core 0 reporting 1198 ms/step
# while cores 1-3 reported 60 ms -- 20x, from compile leaking into the measurement.
def warm(max_iters=40, window=4, tol=0.08):
    hist = []
    for _ in range(max_iters):
        t = time.time(); step()
        torch.zeros(1, device="neuron").cpu()
        hist.append((time.time() - t) * 1000)
        if len(hist) >= window + 2:
            w = hist[-window:]
            if max(w) - min(w) <= tol * min(w):     # within 8% across `window` steps
                return hist
    return hist

warm_hist = warm()

lat = []
for _ in range(STEPS):
    t = time.time(); step()
    torch.zeros(1, device="neuron").cpu()
    lat.append((time.time() - t) * 1000)
lat.sort()
keep = lat[:max(1, int(len(lat) * 0.75))]      # trim the tail; see the caveat in the docstring
ms = sum(keep) / len(keep)
# Report the spread so a still-compiling worker is VISIBLE rather than silently averaged in.
print("WORKER " + json.dumps({"core": CORE, "ms": ms, "tok_s": 1000.0 * BS / ms,
                              "min": lat[0], "max": lat[-1], "warm_steps": len(warm_hist)}))
'''


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=4,
                    help="one per logical core; 4 is the max at LNC=2 on trn2.3xlarge")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--max-seqlen", type=int, default=512, help="must be a multiple of 256")
    ap.add_argument("--prompt-len", type=int, default=8)
    ap.add_argument("--weights", default=os.path.expanduser("~/weights/evo2_7b_base.pt"))
    ap.add_argument("--config", default=os.path.expanduser(
        "~/workspace/vortex-repo/configs/evo2-7b-8k.yml"))
    a = ap.parse_args()

    if a.max_seqlen % 256:
        ap.error(f"--max-seqlen must be a multiple of 256 (got {a.max_seqlen}); "
                 f"attention_tkg shards s_prior by 2 and the shard must be a multiple of 128")
    if a.workers > 4:
        ap.error("at LNC=2 a trn2.3xlarge exposes 4 logical cores; --workers must be <= 4")

    procs = []
    t0 = time.time()
    for core in range(a.workers):
        env = dict(os.environ,
                   NEURON_RT_VISIBLE_CORES=str(core),
                   W_BS=str(a.batch_size), W_STEPS=str(a.steps),
                   W_MAXSEQ=str(a.max_seqlen), W_PROMPT=str(a.prompt_len),
                   W_WEIGHTS=a.weights, W_CFG=a.config)
        procs.append(subprocess.Popen([sys.executable, "-c", WORKER], env=env,
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                      text=True))
    results = []
    for p in procs:
        out, err = p.communicate()
        line = next((l for l in out.splitlines() if l.startswith("WORKER ")), None)
        if line is None:
            print(f"worker failed:\n{out[-800:]}\n{err[-1200:]}", file=sys.stderr)
            return 1
        results.append(json.loads(line[7:]))

    wall = time.time() - t0
    print(f"\nDP={a.workers}  BS={a.batch_size}  ({wall:.0f}s wall incl. per-worker load)")
    print(f"  {'core':>5s} {'ms/step':>9s} {'tok/s':>9s} {'min':>8s} {'max':>8s} {'warm':>5s}")
    for r in sorted(results, key=lambda x: x["core"]):
        print(f"  {r['core']:>5s} {r['ms']:9.1f} {r['tok_s']:9.2f} "
              f"{r['min']:8.1f} {r['max']:8.1f} {r['warm_steps']:5d}")
    agg = sum(r["tok_s"] for r in results)
    fastest = max(r["tok_s"] for r in results)
    print(f"  {'AGG':>5s} {'':9s} {agg:9.2f}")

    # A straggler means that worker was still compiling; its number is not steady state and
    # the aggregate is then meaningless. Say so instead of printing a tidy total.
    slowest = min(r["tok_s"] for r in results)
    if fastest > 1.5 * slowest:
        print(f"\n  WARNING: worker spread is {fastest / slowest:.1f}x "
              f"({slowest:.1f}..{fastest:.1f} tok/s). At least one worker was still "
              f"compiling; the aggregate above is NOT steady state. Re-run -- the compile "
              f"cache is now warm.")
        return 2
    print(f"  scaling: {agg / fastest:.2f}x of {a.workers} ideal "
          f"({100 * agg / (fastest * a.workers):.0f}% of linear)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
