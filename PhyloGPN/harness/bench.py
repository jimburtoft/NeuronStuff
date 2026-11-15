"""PhyloGPN genome-scoring throughput benchmark on one Trainium2 device.

    python harness/bench.py --ranks 4 --L 16384 --B 4 --sites 30000000 [--no-kernel] [--check]

Spawns one process per logical NeuronCore (LNC=2: 4 per trn2 device). Each scores its own synthetic
genome slice end to end (bytes -> token ids -> windows -> device -> stitched [n, 4] logits).
Ranks start together after compiling. Device throughput = total sites / slowest rank's wall time.
"""
import argparse
import json
import multiprocessing as mp
import os
import time


def rank_main(rank, a, q, barrier):
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(rank)
    import numpy as np
    import torch

    import phylogpn_neuron as pg

    t0 = time.time()
    s = pg.Scorer.build(L=a.L, B=a.B, depth=a.depth, kernel=not a.no_kernel, weights=a.weights)
    compile_s = time.time() - t0
    rng = np.random.default_rng(rank)
    genome = rng.choice(np.frombuffer(b"ACGT", np.uint8), a.sites).tobytes()
    out = np.empty((a.sites, 4), np.float32)
    s.score(genome[: 3 * a.L], out[: 3 * a.L])  # warm the data path
    barrier.wait()
    t = time.perf_counter()
    s.score(genome, out)
    wall = time.perf_counter() - t
    rec = dict(rank=rank, sites=a.sites, wall_s=round(wall, 3), sites_per_s=round(a.sites / wall, 1),
               compile_s=round(compile_s, 1))
    if a.check and rank == 0:
        cpu = pg.load(a.weights)
        k = 3 * s.stride
        ids = torch.full((1, k + 480), pg.PAD_ID, dtype=torch.long)
        ids[0, pg.HALF:pg.HALF + k] = torch.from_numpy(pg.tokenize(genome[:k]).astype(np.int64))
        with torch.no_grad():
            ref = cpu(ids)[1][0].numpy()
        got = out[:k]
        rec["check_cos_vs_cpu_fp32"] = round(float((got * ref).sum() / np.linalg.norm(got) / np.linalg.norm(ref)), 6)
        rec["check_argmax_agree"] = round(float((got.argmax(-1) == ref.argmax(-1)).mean()), 5)
    q.put(rec)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ranks", type=int, default=4)
    p.add_argument("--L", type=int, default=16384)
    p.add_argument("--B", type=int, default=4)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--sites", type=int, default=30_000_000, help="sites per rank")
    p.add_argument("--weights", default=os.environ.get("PHYLOGPN_WEIGHTS"))
    p.add_argument("--no-kernel", action="store_true", help="stock TorchNeuron graph (no NKI kernel)")
    p.add_argument("--check", action="store_true")
    p.add_argument("--out", default="")
    a = p.parse_args()
    os.environ.setdefault("NEURON_LOGICAL_NC_CONFIG", "2")
    ctx = mp.get_context("spawn")
    q, barrier = ctx.Queue(), ctx.Barrier(a.ranks)
    procs = [ctx.Process(target=rank_main, args=(r, a, q, barrier)) for r in range(a.ranks)]
    for pr in procs:
        pr.start()
    recs = [q.get() for _ in procs]
    for pr in procs:
        pr.join()
    recs.sort(key=lambda r: r["rank"])
    total = sum(r["sites"] for r in recs)
    wall = max(r["wall_s"] for r in recs)
    res = dict(kernel=not a.no_kernel, L=a.L, B=a.B, ranks=a.ranks, lnc=os.environ["NEURON_LOGICAL_NC_CONFIG"],
               device_sites_per_s=round(total / wall, 1), per_rank_sites_per_s=[r["sites_per_s"] for r in recs],
               wall_s=wall, compile_s=max(r["compile_s"] for r in recs),
               **{k: v for k, v in recs[0].items() if k.startswith("check")})
    print(json.dumps(res))
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
