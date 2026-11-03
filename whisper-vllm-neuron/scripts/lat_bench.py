#!/usr/bin/env python3
"""BS=1 latency client for /v1/audio/transcriptions (any clip manifest).

Send each clip one at a time (no concurrency). Do WARM warm-up passes, then REPS
timed passes. Report per-clip min/median latency, the set mean of per-clip
minimums and medians, and write the transcripts.

usage: lat_bench.py --manifest <clips>/manifest.csv --out res.json [--reps 5] [--warm 1]
       lat_bench.py --wav one.wav --out res.json
"""
import argparse
import csv
import json
import os
import statistics
import time

import requests


def transcribe(url, path, model, session):
    with open(path, "rb") as f:
        data = f.read()
    t0 = time.perf_counter()
    r = session.post(
        url,
        files={"file": (os.path.basename(path), data, "audio/wav")},
        data={"model": model, "language": "en", "temperature": "0",
              "response_format": "json"},
        timeout=600,
    )
    dt = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    return r.json()["text"], dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest")
    ap.add_argument("--wav", action="append", default=[])
    ap.add_argument("--url", default="http://localhost:8000/v1/audio/transcriptions")
    ap.add_argument("--model", default="whisper")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warm", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    clips = []
    if a.manifest:
        base = os.path.dirname(a.manifest)
        with open(a.manifest) as f:
            for row in csv.DictReader(f):
                clips.append((os.path.join(base, row["audio_path"]),
                              float(row["duration_sec"]), row["transcript"]))
    for w in a.wav:
        clips.append((w, None, None))

    s = requests.Session()
    for _ in range(a.warm):
        for p, _, _ in clips:
            transcribe(a.url, p, a.model, s)
    lat = {p: [] for p, _, _ in clips}
    text = {}
    for _ in range(a.reps):
        for p, _, _ in clips:
            t, dt = transcribe(a.url, p, a.model, s)
            lat[p].append(dt)
            if p in text and text[p] != t:
                text[p + "#nondeterministic"] = t
            text[p] = t
    rows = []
    for p, dur, ref in clips:
        rows.append({"clip": os.path.relpath(p, os.path.dirname(a.manifest)) if a.manifest else p,
                     "dur_s": dur, "min_ms": min(lat[p]), "med_ms": statistics.median(lat[p]),
                     "all_ms": lat[p], "text": text[p], "ref": ref})
    summary = {
        "n_clips": len(rows), "reps": a.reps,
        "mean_of_min_ms": statistics.mean(r["min_ms"] for r in rows),
        "mean_of_med_ms": statistics.mean(r["med_ms"] for r in rows),
        "nondeterministic": [k for k in text if k.endswith("#nondeterministic")],
    }
    with open(a.out, "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=1)
    for r in rows:
        print(f"{r['clip']:<45} {r['dur_s'] or 0:6.2f}s min {r['min_ms']:8.1f} med {r['med_ms']:8.1f}")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
