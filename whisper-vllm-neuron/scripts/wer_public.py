#!/usr/bin/env python3
"""Whisper WER against a running vLLM server, for the PUBLIC vllm-neuron 0.29 stack.

The public release ships no test/ tree, so this is a self-contained WER client:
  * open_asr_leaderboard at the pinned revision 48e6f0d75ad7 (``--oal``): data_utils.load_data /
    prepare_data, the EnglishTextNormalizer, write_manifest + merge_chunked_manifest (Earnings22 is
    scored per call), and kaldialign.batch_error_rate(merge_compounds=True);
  * one POST per clip to /v1/audio/transcriptions with the raw file bytes, language=en, temperature 0,
    max_completion_tokens 444, json response; C requests in flight; one warm-up call.

  git clone https://github.com/huggingface/open_asr_leaderboard ~/oal && git -C ~/oal checkout 48e6f0d75ad7
  python wer_public.py PORT CONC OUTDIR DATASET      (OAL=<path> if open_asr_leaderboard is not in ~/oal)
  DATASET: earnings22 | libri-clean | libri-other
Needs: datasets, requests, kaldialign, regex, num2words, more_itertools, soundfile (client side only).
"""
import contextlib
import io
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import requests

ALIASES = {
    "earnings22": ("ArtificialAnalysis/Earnings22-Cleaned-AA-chunked", "", "test"),
    "libri-clean": ("hf-audio/esb-datasets-test-only-sorted", "librispeech", "test.clean"),
    "libri-other": ("hf-audio/esb-datasets-test-only-sorted", "librispeech", "test.other"),
}


@contextlib.contextmanager
def chdir(d):
    prev = os.getcwd(); os.makedirs(d, exist_ok=True); os.chdir(d)
    try:
        yield
    finally:
        os.chdir(prev)


def main():
    port, conc, out, name = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
    sys.path.insert(0, os.environ.get("OAL", os.path.expanduser("~/oal")))
    from datasets import Audio
    from kaldialign import batch_error_rate
    from normalizer import data_utils

    real = data_utils.load_dataset
    data_utils.load_dataset = lambda *a, **k: real(*a, **{**k, "token": None})
    path, config, split = ALIASES[name]
    ds = data_utils.load_data(SimpleNamespace(dataset_path=path, dataset=config, split=split, streaming=False))
    ds = data_utils.prepare_data(ds).cast_column("audio", Audio(decode=False))
    samples = list(ds)
    chunked = data_utils.is_chunked_dataset(path)
    url = f"http://localhost:{port}/v1/audio/transcriptions"
    model = os.environ.get("MODEL_NAME", "whisper")
    max_tok = os.environ.get("MAX_TOK", "444")

    def work(s):
        a = s["audio"]
        b = a.get("bytes") or open(a["path"], "rb").read()
        fn = os.path.basename(a["path"]) if a.get("path") else "audio.wav"
        dur = float(s.get("duration") or 0.0)
        if dur <= 0:
            import soundfile as sf
            i = sf.info(io.BytesIO(b)); dur = i.frames / i.samplerate
        t0 = time.perf_counter()
        r = requests.post(url, files={"file": (fn, b, "audio/wav")},
                          data={"model": model, "language": "en", "temperature": "0.0",
                                "max_completion_tokens": max_tok, "stream": "false", "response_format": "json"},
                          timeout=6000)
        if r.status_code >= 400:
            raise RuntimeError(f"{r.status_code} {fn}: {r.text[:500]}")
        return r.json().get("text", ""), dur, time.perf_counter() - t0

    try:
        work(samples[0])
    except Exception as e:  # warm-up
        print("warmup failed:", e, flush=True)
    preds, alen, ttime, errors = [""] * len(samples), [0.0] * len(samples), [0.0] * len(samples), 0
    with ThreadPoolExecutor(max(1, conc)) as ex:
        futs = {ex.submit(work, s): i for i, s in enumerate(samples)}
        for f, i in futs.items():
            try:
                preds[i], alen[i], ttime[i] = f.result()
            except Exception as e:
                errors += 1; print("ERR", i, str(e)[:200], flush=True)
    refs = [(s.get("original_text", "") or "").strip() or " " for s in samples]
    extra = {k: [s[k] for s in samples] for k in data_utils.CHUNK_METADATA_KEYS} if chunked else None
    with chdir(out):
        mp = data_utils.write_manifest(refs, preds, model, path, config, split, audio_length=alen,
                                       transcription_time=ttime, extra_fields=extra)
        man = data_utils.read_manifest(mp)
    if chunked:
        sess = data_utils.merge_chunked_manifest(man)
        R, P = [s["text"] for s in sess], [s["pred_text"] for s in sess]
    else:
        sess, R, P = man, refs, preds
    rs = [tuple(data_utils.normalizer(r).split()) for r in R]
    ps = [tuple(data_utils.normalizer(p).split()) for p in P]
    wer = round(100 * batch_error_rate(rs, ps, merge_compounds=True)["err_rate"], 2)
    res = {"dataset": name, "conc": conc, "wer": wer, "num_samples": len(samples), "num_sessions": len(sess),
           "num_errors": errors}
    json.dump(res, open(os.path.join(out, "asr_wer_summary.json"), "w"), indent=2)
    print("WERRESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
