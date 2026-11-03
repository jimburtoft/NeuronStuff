# Whisper notes: the corpus is model-agnostic.
#  - 29.99 s windows fit the Whisper 30 s window (the processor pads every clip to 3000 mel frames).
#  - Bench with --extra-body '{"temperature": 0, "language": "en"}'. Without "language", the Whisper
#    serving path runs an extra language-detection request per clip.
#  - Keep --custom-output-len <= max_model_len - 4 (decoder prompt is 4 tokens; Whisper max is 448).
#  - Needs datasets, av, soundfile on the client side (see the README, section 6).
"""Throughput corpus: 512 x 29.99 s Earnings22 windows.

Recipe: tile ArtificialAnalysis/Earnings22-Cleaned-AA into fixed 29.99 s
windows, keep dense speech (VAD speech_ratio >= 0.5), 512 distinct clips. Here: the source calls are tiled with a hop chosen to yield >= 512 windows, and
speech_ratio comes from a simple energy VAD (frame RMS > -40 dBFS relative to the clip peak). Output
matches the layout download_dataset() expects:
  ~/.cache/vllm-neuron/datasets/asr30s-earnings22-v1/{manifest_512.jsonl, clips/*.wav}

"""
import json, os, sys
from pathlib import Path
import numpy as np, soundfile as sf

WIN = 29.99; SR = 16000; N = 512
OUT = Path.home() / ".cache/vllm-neuron/datasets/asr30s-earnings22-v1"
(OUT / "clips").mkdir(parents=True, exist_ok=True)

from datasets import load_dataset
from huggingface_hub import snapshot_download
import av
REPO = "ArtificialAnalysis/Earnings22-Cleaned-AA"
ds = load_dataset(REPO, split="test")
root = snapshot_download(REPO, repo_type="dataset", allow_patterns=["audio/*"])

def decode(path):
    c = av.open(path)
    rs = av.AudioResampler(format="flt", layout="mono", rate=SR)
    out = []
    for fr in c.decode(audio=0):
        for g in rs.resample(fr):
            out.append(g.to_ndarray().reshape(-1))
    for g in rs.resample(None):
        out.append(g.to_ndarray().reshape(-1))
    return np.concatenate(out).astype(np.float32)

calls = [decode(os.path.join(root, r["url"])) for r in ds]
total = sum(len(c) for c in calls) / SR
print(f"{len(calls)} calls, {total:.0f} s total")

def speech_ratio(x):
    f = 400  # 25 ms
    n = len(x) // f
    rms = np.sqrt((x[: n * f].reshape(n, f) ** 2).mean(1) + 1e-12)
    db = 20 * np.log10(rms / (rms.max() + 1e-12))
    return float((db > -40).mean())

w = int(WIN * SR)
# choose hop so that the candidate pool is ~1.3x N before VAD filtering
cand_target = int(N * 1.3)
hop_s = max(1.0, (total - len(calls) * WIN) / cand_target)
hop = int(hop_s * SR)
cands = []
for ci, c in enumerate(calls):
    for s in range(0, max(1, len(c) - w + 1), hop):
        seg = c[s : s + w]
        if len(seg) == w:
            cands.append((ci, s, seg))
print(f"hop={hop_s:.2f}s candidates={len(cands)}")
kept = [(ci, s, seg, r) for ci, s, seg in cands if (r := speech_ratio(seg)) >= 0.5]
print(f"kept after VAD>=0.5: {len(kept)}")
if len(kept) < N:
    print("WARNING: fewer than 512 distinct clips; manifest will be short (bench cycles it)")
idx = np.linspace(0, len(kept) - 1, min(N, len(kept))).round().astype(int)
with open(OUT / "manifest_512.jsonl", "w") as m:
    for j, k in enumerate(idx):
        ci, s, seg, r = kept[k]
        p = OUT / "clips" / f"e22_{j:04d}_c{ci}_{s // SR:05d}.wav"
        sf.write(p, seg, SR)
        m.write(json.dumps({"prompt": "", "audio": str(p)}) + "\n")
print("wrote", len(idx), "clips to", OUT)
