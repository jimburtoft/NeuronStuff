#!/usr/bin/env python3
"""Combine per-configuration example-workload results into comparison plots and a summary table.

Reads <results_dir>/*.json written by the prodmix notebooks and produces:
  example_workload_curves_<platform>.png : throughput (sequences/s) vs p50 and p99 call latency,
                                  one line per configuration on that platform
  prodmix_all.png               : every configuration on one axes (p50), for cross-platform view
  summary.md            : table of peak, and best throughput under several p99 targets
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = sys.argv[1] if len(sys.argv) > 1 else "."
runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, "*.json")))]
runs = [r for r in runs if r.get("curve")]
LABEL = {"trn2": "trn2.3xlarge", "inf2": "inf2.8xlarge", "inf2x": "inf2.xlarge"}


def lab(r):
    s = f"{LABEL[r['platform']]} {r['path']}"
    if r["platform"] == "trn2":
        s += f" LNC={r['lnc']} x{r['instances']}"
    return s


def best_under(curve, key, limit):
    ok = [p for p in curve if p[key] <= limit]
    return max(ok, key=lambda p: p["seq_per_s"]) if ok else None


style = {"trace": "-o", "native": "--s"}
for plat in sorted({r["platform"] for r in runs}):
    rs = [r for r in runs if r["platform"] == plat]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True)
    for r in rs:
        x = [p["seq_per_s"] for p in r["curve"]]
        for a, k in zip(ax, ["p50_ms", "p99_ms"]):
            a.plot(x, [p[k] for p in r["curve"]], style[r["path"]], label=lab(r))
    for a, k in zip(ax, ["p50", "p99"]):
        a.set_xlabel("throughput (sequences / s)")
        a.set_ylabel(f"{k} call latency (ms)")
        a.set_yscale("log")
        a.grid(alpha=.3, which="both")
        a.legend(fontsize=8)
        a.set_title(f"{LABEL[plat]}: example mixed-length workload, {k}")
    plt.tight_layout()
    plt.savefig(os.path.join(d, f"example_workload_curves_{plat}.png"), dpi=130)
    plt.close()

fig, ax = plt.subplots(figsize=(10, 6))
for r in runs:
    ax.plot([p["seq_per_s"] for p in r["curve"]], [p["p50_ms"] for p in r["curve"]],
            style[r["path"]], label=lab(r))
ax.set_xlabel("throughput (sequences / s)")
ax.set_ylabel("p50 call latency (ms)")
ax.set_yscale("log")
ax.grid(alpha=.3, which="both")
ax.legend(fontsize=8)
ax.set_title("BERT reranker, example mixed-length workload: all configurations")
plt.tight_layout()
plt.savefig(os.path.join(d, "example_workload_all.png"), dpi=130)
plt.close()

limits = [50, 100, 200, 500]
lines = ["| configuration | peak seq/s | at concurrency | p50 / p99 at peak (ms) | "
         + " | ".join(f"best seq/s, p99 <= {L} ms" for L in limits) + " |",
         "|---|---|---|---|" + "---|" * len(limits)]
for r in sorted(runs, key=lambda r: (r["platform"], r["path"], r["lnc"])):
    pk = max(r["curve"], key=lambda p: p["seq_per_s"])
    cells = []
    for L in limits:
        b = best_under(r["curve"], "p99_ms", L)
        cells.append(f"{b['seq_per_s']:.0f} (C={b['concurrency']})" if b else "—")
    lines.append(f"| {lab(r)} | **{pk['seq_per_s']:.0f}** | {pk['concurrency']} | "
                 f"{pk['p50_ms']:.0f} / {pk['p99_ms']:.0f} | " + " | ".join(cells) + " |")
open(os.path.join(d, "summary.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
