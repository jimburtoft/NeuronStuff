#!/usr/bin/env python3
"""pub18-ls-e22: a public, reproducible 18-clip BS=1 latency / byte-identity set for Whisper.

18 clips, 3 per 5 s duration bucket (0-5, 5-10, 10-15, 15-20, 20-25, 25-30 s), drawn from public Hugging Face
datasets only:
  * LibriSpeech test-clean / test-other via hf-audio/esb-datasets-test-only-sorted (config "librispeech"), the
    same source as our LibriSpeech WER;
  * Earnings22 chunks via ArtificialAnalysis/Earnings22-Cleaned-AA-chunked (split "test"), the same source as
    our Earnings22 WER.
No audio is redistributed: the manifest lists dataset / config / split / utterance ID; this script fetches the
audio from the Hub and writes 16 kHz mono WAVs locally.

Reference text: LibriSpeech rows carry their transcript. The Earnings22 chunked set has no per-chunk text (its WER is
scored per parent call after merging chunks), so E22 rows list the chunk ID and its parent call ID and leave
`transcript` empty. Byte-identity uses the HF fp32 greedy output (ref_pub18.json), not this text.

Selection (deterministic): for each bucket [lo, lo+5) s and each source, the candidates are the utterances whose
decoded duration is in the bucket (LibriSpeech: and whose transcript is non-empty), sorted by ID. Each bucket takes one clip
per source (LS-clean, LS-other, E22) with random.Random(f"{SEED}:{bucket}:{source}").choice over the sorted IDs.
If a source has no candidate in a bucket, the slot goes to the next source with unused candidates
(order: E22, LS-other, LS-clean).

usage:
  build_pub18.py select --out DIR       # scan the datasets, write DIR/manifest.csv + DIR/manifest.json + wavs
  build_pub18.py fetch  --manifest M    # re-fetch the audio for an existing manifest (by ID) into dirname(M)
Needs: datasets, soundfile, numpy (+ av and soxr for non-16 kHz / non-FLAC sources).
"""
import argparse
import csv
import io
import json
import os
import random

import numpy as np
import soundfile as sf

NAME = "pub18-ls-e22"
SEED = 49
SR = 16000
BUCKETS = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30)]
SOURCES = {
    "ls-clean": ("hf-audio/esb-datasets-test-only-sorted", "librispeech", "test.clean"),
    "ls-other": ("hf-audio/esb-datasets-test-only-sorted", "librispeech", "test.other"),
    "e22": ("ArtificialAnalysis/Earnings22-Cleaned-AA-chunked", None, "test"),
}
FILL_ORDER = ["e22", "ls-other", "ls-clean"]
ID_COLS = ("id", "segment_id", "utterance_id", "file_id", "audio_id")
TEXT_COLS = ("text", "transcript", "transcription", "sentence", "normalized_text")


def load(src):
    from datasets import Audio, load_dataset
    path, cfg, split = SOURCES[src]
    ds = load_dataset(path, cfg, split=split) if cfg else load_dataset(path, split=split)
    return ds.cast_column("audio", Audio(decode=False))


def col(ds, names):
    for n in names:
        if n in ds.column_names:
            return n
    raise KeyError(f"none of {names} in {ds.column_names}")


def audio_bytes(src, r):
    """Raw audio bytes of a row. Earnings22-chunked rows have audio=None; the file is audio/<file_name> in the repo."""
    a = r.get("audio")
    if a:
        return a.get("bytes") or open(a["path"], "rb").read()
    from huggingface_hub import hf_hub_download
    return open(hf_hub_download(SOURCES[src][0], f"audio/{r['file_name']}", repo_type="dataset"), "rb").read()


def decode(b):
    try:
        x, sr = sf.read(io.BytesIO(b), dtype="int16", always_2d=True)
        pcm16 = True
    except Exception:  # non-libsndfile container (mp3 / m4a): decode with av
        import av
        c = av.open(io.BytesIO(b)); rs = av.AudioResampler(format="flt", layout="mono", rate=SR)
        out = [g.to_ndarray().reshape(-1) for fr in c.decode(audio=0) for g in rs.resample(fr)]
        out += [g.to_ndarray().reshape(-1) for g in rs.resample(None)]
        return np.concatenate(out).astype(np.float32), SR, False
    x = x[:, 0] if x.shape[1] == 1 else x.astype(np.float32).mean(1).astype(np.int16)
    if sr != SR:
        import soxr
        return soxr.resample(x.astype(np.float32) / 32768.0, sr, SR).astype(np.float32), SR, False
    return x, sr, pcm16


def write_wav(path, x, pcm16):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, x, SR, subtype="PCM_16" if pcm16 else "FLOAT")


def bucket_of(d):
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= d < hi:
            return i
    return None


def select(out):
    cands = {s: {i: [] for i in range(len(BUCKETS))} for s in SOURCES}
    rows_by_key = {}
    for s in SOURCES:
        ds = load(s)
        idc = col(ds, ID_COLS)
        txc = next((n for n in TEXT_COLS if n in ds.column_names), None)
        par = "parent_id" if "parent_id" in ds.column_names else None
        print(f"{s}: {len(ds)} rows, id={idc} text={txc} columns={ds.column_names}", flush=True)
        for r in ds:
            text = (r[txc] or "").strip() if txc else ""
            if txc and not text:
                continue
            x, sr, _ = decode(audio_bytes(s, r))
            d = len(x) / SR
            b = bucket_of(d)
            if b is None:
                continue
            cands[s][b].append(str(r[idc]))
            rows_by_key[(s, str(r[idc]))] = (d, text, str(r[par]) if par else "")
        print(f"{s}: per-bucket candidates {[len(cands[s][b]) for b in range(len(BUCKETS))]}", flush=True)
    picks = []
    for b in range(len(BUCKETS)):
        used = {s: set() for s in SOURCES}
        slots = ["ls-clean", "ls-other", "e22"]
        chosen = []
        for s in slots:
            src = s if len(cands[s][b]) > len(used[s]) else next(
                (f for f in FILL_ORDER if len(cands[f][b]) > len(used[f])), None)
            assert src, f"bucket {b}: no candidates left"
            pool = [i for i in sorted(cands[src][b]) if i not in used[src]]
            pick = random.Random(f"{SEED}:{b}:{s}").choice(pool)
            used[src].add(pick); chosen.append((src, pick))
        picks += [(b, s, i) for s, i in sorted(chosen)]
    manifest = []
    for b, s, i in picks:
        path, cfg, split = SOURCES[s]
        d, text, parent = rows_by_key[(s, i)]
        lo, hi = BUCKETS[b]
        manifest.append({"bucket": f"{lo}-{hi}s", "source": s, "dataset": path, "config": cfg or "",
                         "split": split, "id": i, "parent_id": parent, "duration_sec": round(d, 4), "transcript": text,
                         "audio_path": f"{lo}-{hi}s/{s}__{safe(i)}.wav"})
    os.makedirs(out, exist_ok=True)
    meta = {"name": NAME, "seed": SEED, "sample_rate": SR, "buckets": [f"{lo}-{hi}s" for lo, hi in BUCKETS],
            "sources": {k: list(v) for k, v in SOURCES.items()},
            "candidates": {s: [len(cands[s][b]) for b in range(len(BUCKETS))] for s in SOURCES},
            "clips": manifest}
    json.dump(meta, open(os.path.join(out, "manifest.json"), "w"), indent=1)
    write_csv(os.path.join(out, "manifest.csv"), manifest)
    fetch(os.path.join(out, "manifest.csv"))


def safe(i):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in i)


def write_csv(p, manifest):
    keys = ["audio_path", "duration_sec", "transcript", "bucket", "source", "dataset", "config", "split", "id",
            "parent_id"]
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for m in manifest:
            w.writerow({k: m[k] for k in keys})


def fetch(manifest):
    base = os.path.dirname(os.path.abspath(manifest))
    rows = list(csv.DictReader(open(manifest)))
    by_src = {}
    for r in rows:
        by_src.setdefault(r["source"], set()).add(r["id"])
    found = {}
    for s, ids in by_src.items():
        ds = load(s)
        idc = col(ds, ID_COLS)
        sel = ds.filter(lambda v: str(v) in ids, input_columns=[idc])
        for r in sel:
            found[(s, str(r[idc]))] = audio_bytes(s, r)
    for r in rows:
        x, sr, pcm16 = decode(found[(r["source"], r["id"])])
        d = len(x) / SR
        assert abs(d - float(r["duration_sec"])) < 1e-3, (r["id"], d, r["duration_sec"])
        write_wav(os.path.join(base, r["audio_path"]), x, pcm16)
        print(f"{r['bucket']:>7} {r['source']:<8} {d:6.2f}s {r['id']}", flush=True)
    print(f"FETCHED {len(rows)} clips into {base}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["select", "fetch"])
    ap.add_argument("--out", default="pub18")
    ap.add_argument("--manifest")
    a = ap.parse_args()
    select(a.out) if a.mode == "select" else fetch(a.manifest)


if __name__ == "__main__":
    main()
