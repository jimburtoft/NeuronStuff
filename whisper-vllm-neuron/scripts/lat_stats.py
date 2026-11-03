#!/usr/bin/env python3
"""Summarize lat_bench.py JSON(s) on pub18-ls-e22: per bucket and overall mean / p50 / p90 of the per-clip
minimum latency (ms). Percentiles: numpy linear interpolation over the clips in the group.

usage: lat_stats.py MANIFEST.csv LAT.json [LAT2.json ...]   (prints a markdown table per file + a JSON line)
"""
import csv
import json
import sys

import numpy as np

man = {r["audio_path"]: r for r in csv.DictReader(open(sys.argv[1]))}
order = []
for r in man.values():
    if r["bucket"] not in order:
        order.append(r["bucket"])
for p in sys.argv[2:]:
    rows = json.load(open(p))["rows"]
    groups = {b: [] for b in order}
    for r in rows:
        groups[man[r["clip"]]["bucket"]].append(r["min_ms"])
    allv = [r["min_ms"] for r in rows]
    out = {}
    print(f"\n{p}\n| bucket | n | mean | p50 | p90 |\n|---|--:|--:|--:|--:|")
    for b in order + ["all"]:
        v = np.array(allv if b == "all" else groups[b])
        out[b] = dict(n=len(v), mean=float(v.mean()), p50=float(np.percentile(v, 50)), p90=float(np.percentile(v, 90)))
        print(f"| {b} | {len(v)} | {v.mean():.1f} | {np.percentile(v, 50):.1f} | {np.percentile(v, 90):.1f} |")
    print("LATSTATS", p, json.dumps(out))
