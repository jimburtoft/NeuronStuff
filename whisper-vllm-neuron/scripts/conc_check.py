#!/usr/bin/env python3
"""Concurrent correctness: send every clip in a manifest REPS times with C in flight,
then compare each transcript with the BS=1 reference transcript (text equality).

usage: conc_check.py --manifest M --ref lat_bs1.json --conc 8 --reps 4
"""
import argparse
import csv
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

import requests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ref", required=True, help="lat_bench JSON (BS=1 transcripts)")
    ap.add_argument("--url", default="http://localhost:8000/v1/audio/transcriptions")
    ap.add_argument("--model", default="whisper")
    ap.add_argument("--conc", type=int, default=8)
    ap.add_argument("--reps", type=int, default=4)
    a = ap.parse_args()
    base = os.path.dirname(a.manifest)
    with open(a.manifest) as f:
        clips = [r["audio_path"] for r in csv.DictReader(f)]
    ref = {r["clip"]: r["text"] for r in json.load(open(a.ref))["rows"]}
    jobs = clips * a.reps
    random.Random(0).shuffle(jobs)
    data = {c: open(os.path.join(base, c), "rb").read() for c in clips}

    def work(c):
        r = requests.post(a.url, files={"file": (os.path.basename(c), data[c], "audio/wav")},
                          data={"model": a.model, "language": "en", "temperature": "0"},
                          timeout=900)
        r.raise_for_status()
        return c, r.json()["text"]

    t0 = time.perf_counter()
    with ThreadPoolExecutor(a.conc) as ex:
        res = list(ex.map(work, jobs))
    wall = time.perf_counter() - t0
    bad = [(c, t) for c, t in res if t != ref[c]]
    print(f"CONCCHECK conc={a.conc} n={len(res)} mismatches={len(bad)} wall={wall:.1f}s")
    for c, t in bad[:10]:
        print("  MISMATCH", c, repr(t[:100]), "ref", repr(ref[c][:100]))


if __name__ == "__main__":
    main()
