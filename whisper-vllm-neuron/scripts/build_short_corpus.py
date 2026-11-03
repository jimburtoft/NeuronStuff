# Whisper note: the encoder always runs a padded 30 s window, so 8 s clips do NOT cut encoder cost.
# Short clips test per-request overhead and decode, not the encoder.
"""Short-clip corpus: 512 x 8.0 s windows cut from the 30 s Earnings22 perf clips (first 8 s of each).
"""
import json, os
from pathlib import Path
import soundfile as sf
SRC = Path.home() / ".cache/vllm-neuron/datasets/asr30s-earnings22-v1"
OUT = Path.home() / ".cache/vllm-neuron/datasets/asr8s-earnings22-approx"
(OUT / "clips").mkdir(parents=True, exist_ok=True)
with open(SRC / "manifest_512.jsonl") as f, open(OUT / "manifest_512.jsonl", "w") as m:
    for line in f:
        p = json.loads(line)["audio"]; a, sr = sf.read(p, dtype="float32")
        q = OUT / "clips" / Path(p).name; sf.write(q, a[: 8 * sr], sr)
        m.write(json.dumps({"prompt": "", "audio": str(q)}) + "\n")
print("done", OUT)
