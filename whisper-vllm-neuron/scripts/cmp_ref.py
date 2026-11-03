#!/usr/bin/env python3
"""Compare lat_bench transcripts with an HF reference, by token IDs.

usage: cmp_ref.py REF.json LAT.json [LAT2.json ...]
"""
import json
import sys

from transformers import WhisperTokenizer

tok = WhisperTokenizer.from_pretrained("openai/whisper-large-v3")
ref = json.load(open(sys.argv[1]))
for path in sys.argv[2:]:
    rows = json.load(open(path))["rows"]
    exact, diffs = 0, []
    for r in rows:
        rt = ref[r["clip"]]["tokens"]
        gt = tok.encode(r["text"], add_special_tokens=False)
        if gt == rt:
            exact += 1
        else:
            i = next((k for k, (a, b) in enumerate(zip(gt, rt)) if a != b), min(len(gt), len(rt)))
            diffs.append(f"{r['clip']} @tok{i}/{len(rt)}: ref={tok.decode(rt[max(0,i-2):i+4])!r} got={tok.decode(gt[max(0,i-2):i+4])!r}")
    print(f"{path}: byte-identical {exact}/{len(rows)}")
    for d in diffs:
        print("   ", d)
