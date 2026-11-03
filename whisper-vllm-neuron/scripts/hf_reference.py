#!/usr/bin/env python3
"""HF transformers fp32 CPU greedy reference for Whisper large-v3.

Same prompt as the server: <|sot|><|en|><|transcribe|><|notimestamps|>, greedy,
no timestamps, max 444 new tokens. Writes {clip: {"tokens": [...], "text": ...}}.

usage: hf_reference.py --manifest <clips>/manifest.csv --out ref.json
"""
import argparse
import csv
import json
import os

import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

EOT = 50257


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest")
    ap.add_argument("--wav-list", help="file with one wav path per line")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    clips = []
    if a.manifest:
        base = os.path.dirname(a.manifest)
        with open(a.manifest) as f:
            for row in csv.DictReader(f):
                clips.append((row["audio_path"], os.path.join(base, row["audio_path"])))
    if a.wav_list:
        with open(a.wav_list) as f:
            clips += [(os.path.basename(p.strip()), p.strip()) for p in f if p.strip()]

    proc = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-large-v3", torch_dtype=torch.float32).eval()
    prompt = proc.tokenizer.convert_tokens_to_ids(
        ["<|startoftranscript|>", "<|en|>", "<|transcribe|>", "<|notimestamps|>"])

    out = {}
    if os.path.exists(a.out):
        out = json.load(open(a.out))
    for name, path in clips:
        if name in out:
            continue
        audio, sr = sf.read(path, dtype="float32")
        assert sr == 16000, (path, sr)
        feats = proc(audio, sampling_rate=16000, return_tensors="pt").input_features
        with torch.no_grad():
            enc = model.model.encoder(feats)
            ids = torch.tensor([prompt])
            gen = []
            past = None
            for _ in range(444):
                o = model(encoder_outputs=enc, decoder_input_ids=ids if past is None else ids[:, -1:],
                          past_key_values=past, use_cache=True)
                past = o.past_key_values
                nxt = int(o.logits[0, -1].argmax())
                if nxt == EOT:
                    break
                gen.append(nxt)
                ids = torch.cat([ids, torch.tensor([[nxt]])], dim=1)
        text = proc.tokenizer.decode(gen, skip_special_tokens=True)
        out[name] = {"tokens": gen, "text": text}
        print(f"{name}: {len(gen)} tok {text[:80]!r}", flush=True)
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
