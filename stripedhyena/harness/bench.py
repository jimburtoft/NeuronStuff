"""StripedHyena-Hessian-7B decode / TTFT benchmark through the public ``striped_hyena_neuron`` API.

One configuration per process (PyTorch Native compiles one graph set per shape). Single core:
    NEURON_RT_VISIBLE_CORES=0 python harness/bench.py decode --B 1 --ctx 2048 --out results/results.json
TP=4 on one trn2 device (4 logical cores, LNC=2):
    torchrun --standalone --nproc_per_node=4 harness/bench.py decode --tp 4 --B 512 --ctx 2048 --out results/results.json
    torchrun --standalone --nproc_per_node=4 harness/bench.py ttft --tp 4 --L 2048 --out results/results.json

decode: prompt = seeded random token ids (``torch.randint(3, 32000)``, seed 1000, BOS first) of length ctx-256, compiled
(segmented) prefill at B=1, decode state replicated to B rows (real KV contents, real fill), 51 untimed warm-up
steps, then 7 reps x 24 timed greedy steps; every step copies the B tokens to the host. Reports the median of the
rep medians (ms/step) and tok/s = B * 1000 / ms.  ttft: prompt ids on device -> first greedy token on the host,
3 warm-up + 10 measured runs, p50/p95/p99. Both record the CPU fallback ops seen in the timed loop (expected: []).
Results are merged into --out under a key such as ``decode.tp4.c2048.b512`` (rank 0 only).
"""
import argparse
import json
import os
import statistics
import time

ap = argparse.ArgumentParser()
ap.add_argument("what", choices=["decode", "ttft"])
ap.add_argument("--weights", default=os.environ.get("SHN_WEIGHTS", os.path.expanduser("~/weights/hessian-7b")))
ap.add_argument("--tp", type=int, default=1)
ap.add_argument("--B", type=int, default=1)
ap.add_argument("--ctx", type=int, default=2048, help="decode: KV window (multiple of 256)")
ap.add_argument("--L", type=int, default=2048, help="ttft: prompt length (<= 32767; 32512 = the longest the source project times)")
ap.add_argument("--mode", default=None, help="decoder mode override: blocks:32 | step (default: package rule)")
ap.add_argument("--warm", type=int, default=51)
ap.add_argument("--steps", type=int, default=24)
ap.add_argument("--reps", type=int, default=7)
ap.add_argument("--out", default="results/results.json")
a = ap.parse_args()

import torch  # noqa: E402
import torch_neuronx  # noqa: E402

import striped_hyena_neuron as shn  # noqa: E402
from striped_hyena_neuron.decode import DecodeState  # noqa: E402

RANK = int(os.environ.get("RANK", "0"))


def rand_prompt(B, n, seed):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(3, 32000, (B, n), generator=g)
    ids[:, 0] = 1
    return ids


def pct(xs, p):
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    f = int(k)
    return xs[f] + (xs[min(f + 1, len(xs) - 1)] - xs[f]) * (k - f)


def replicate(S1, B):
    if B == 1:
        return S1
    rep = lambda t: t.repeat_interleave(B, 0).contiguous()  # noqa: E731
    S = DecodeState([f.repeat_interleave(B, 1).contiguous() for f in S1.fir], [rep(x) for x in S1.s_re],
                    [rep(x) for x in S1.s_im], [], [], S1.pos, S1.W)
    for j in range(len(S1.k)):
        S.k.append(rep(S1.k[j])); S1.k[j] = None
        S.v.append(rep(S1.v[j])); S1.v[j] = None
    return S


@torch.inference_mode()
def main():
    tp = None
    if a.tp > 1:
        import torch.distributed as dist
        dist.init_process_group(backend="neuron")
        tp = (dist.get_rank(), a.tp)
    W = a.ctx if a.what == "decode" else min(((a.L + 8 + 255) // 256) * 256, 32768)   # model maximum 32768
    t0 = time.time()
    model, _ = shn.load(a.weights, max_seqlen=W, tp=tp)
    gen = shn.compile(model, B=a.B, window=W, tp=a.tp, mode=a.mode)
    rec = {"what": a.what, "tp": a.tp, "B": a.B, "decode_opts": dict(gen.d), "load_compile_s": round(time.time() - t0, 1),
           "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    dev = next(model.parameters()).device
    if a.what == "decode":
        fill = a.ctx - 256
        t1 = time.time()
        # segment 2048 (as the source project's decode harness): the larger TP default (8192) leaves too little HBM
        # for the replicated KV at the 32K batch ceiling
        tok1, S1 = gen.prefill(rand_prompt(1, fill, 1000).to(dev), seg=2048)
        torch.neuron.synchronize()
        rec.update(ctx=a.ctx, fill=fill, prefill_s=round(time.time() - t1, 2))
        S = replicate(S1, a.B)
        del S1
        tok = tok1.repeat(a.B)
        from striped_hyena_neuron.decode import free_prefill_tables
        free_prefill_tables(model)
        gen._prepare(S, fill + a.warm + a.steps * a.reps + 8)
        for _ in range(a.warm):
            _, tok = gen.dec(tok, S)
            tok.cpu()
        torch_neuronx.clear_op_tracking()
        per_rep = []
        for _ in range(a.reps):
            ms = []
            for _ in range(a.steps):
                ts = time.perf_counter()
                _, tok = gen.dec(tok, S)
                tok.cpu()
                ms.append(1e3 * (time.perf_counter() - ts))
            per_rep.append(statistics.median(ms))
        m = statistics.median(per_rep)
        rec.update(ms_per_step=round(m, 3), ms_stdev=round(statistics.stdev(per_rep), 3), ms_per_rep=[round(x, 3) for x in per_rep],
                   tok_s=round(a.B * 1000 / m, 1), rows_identical=bool((tok == tok[0]).all().item()),
                   peak_hbm_gb=round(torch.neuron.max_memory_allocated() / 2**30, 2))
        key = f"decode.tp{a.tp}.c{a.ctx}.b{a.B}" + ("" if gen.d["mode"] == "blocks:32" else "." + gen.d["mode"])
    else:
        ids = rand_prompt(a.B, a.L, 7)
        xs = []
        for i in range(13):
            ts = time.perf_counter()
            tok, _ = gen.prefill(ids.to(dev))
            tok.cpu()
            if i >= 3:
                xs.append(1e3 * (time.perf_counter() - ts))
            if i == 2:
                torch_neuronx.clear_op_tracking()
        rec.update(L=a.L, ttft_ms={"p50": round(pct(xs, 50), 1), "p95": round(pct(xs, 95), 1), "p99": round(pct(xs, 99), 1),
                                   "min": round(min(xs), 1)}, n=len(xs))
        key = f"ttft.tp{a.tp}.L{a.L}.b{a.B}"
    rec["fallback_ops"] = sorted(set(torch_neuronx.get_fallback_ops() or []))
    if RANK == 0:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        R = json.load(open(a.out)) if os.path.exists(a.out) else {}
        node = R
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = rec
        json.dump(R, open(a.out, "w"), indent=1)
        print("RESULT", key, json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
