#!/usr/bin/env python3
"""Throughput client for /v1/audio/transcriptions, the same on Neuron and on GPU.

Sends NUM requests from a manifest (cycled) with CONC requests in flight,
spread round-robin over one or more server ports. Uses the streaming endpoint
to measure TTFT. Reports req/s, RTFx (audio s / wall s), output tok/s, TTFT and
E2E percentiles, and writes every transcript.

usage: tput_bench.py --manifest M.jsonl --conc 16 --num 512 --ports 8000[,8001] --out res.json
"""
import argparse
import asyncio
import json
import os
import random
import statistics
import time

import aiohttp
import soundfile as sf


def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(p / 100 * (len(v) - 1))))]


async def one(session, url, model, name, data, max_tok, stream):
    form = aiohttp.FormData()
    form.add_field("file", data, filename=name, content_type="audio/wav")
    form.add_field("model", model)
    form.add_field("language", "en")
    form.add_field("temperature", "0")
    form.add_field("max_completion_tokens", str(max_tok))
    if stream:
        form.add_field("stream", "true")
    t0 = time.perf_counter()
    ttft = None
    text = []
    async with session.post(url, data=form) as r:
        if r.status != 200:
            raise RuntimeError(f"HTTP {r.status}: {(await r.text())[:200]}")
        if not stream:
            j = await r.json()
            return time.perf_counter() - t0, None, j["text"]
        async for raw in r.content:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            j = json.loads(payload)
            ch = j.get("choices") or []
            if ch:
                delta = ch[0].get("delta", {}).get("content") or ""
                if delta and ttft is None:
                    ttft = time.perf_counter() - t0
                text.append(delta)
    return time.perf_counter() - t0, ttft, "".join(text)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--conc", type=int, required=True)
    ap.add_argument("--num", type=int, default=512)
    ap.add_argument("--warm", type=int, default=0, help="warm-up requests (default: conc)")
    ap.add_argument("--ports", default="8000")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--model", default="whisper")
    ap.add_argument("--max-tok", type=int, default=444)
    ap.add_argument("--no-stream", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with open(a.manifest) as f:
        paths = [json.loads(l)["audio"] for l in f if l.strip()]
    blobs = {p: open(p, "rb").read() for p in paths}
    durs = {p: sf.info(p).duration for p in paths}
    ports = [int(x) for x in a.ports.split(",")]
    urls = [f"http://{a.host}:{p}/v1/audio/transcriptions" for p in ports]
    jobs = [paths[i % len(paths)] for i in range(a.num)]
    warm = a.warm or a.conc
    conn = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=3600)
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as s:
        wjobs = [paths[(i * 7) % len(paths)] for i in range(warm)]
        sem = asyncio.Semaphore(a.conc)

        async def run(i, p):
            async with sem:
                return await one(s, urls[i % len(urls)], a.model, os.path.basename(p),
                                 blobs[p], a.max_tok, not a.no_stream)

        await asyncio.gather(*[run(i, p) for i, p in enumerate(wjobs)])
        t0 = time.perf_counter()
        res = await asyncio.gather(*[run(i, p) for i, p in enumerate(jobs)], return_exceptions=True)
        wall = time.perf_counter() - t0
    ok = [(p, r) for p, r in zip(jobs, res) if not isinstance(r, BaseException)]
    errs = [str(r)[:200] for r in res if isinstance(r, BaseException)]
    e2e = [r[0] * 1000 for _, r in ok]
    ttft = [r[1] * 1000 for _, r in ok if r[1] is not None]
    audio_s = sum(durs[p] for p, _ in ok)
    words = sum(len(r[2].split()) for _, r in ok)
    summary = {
        "conc": a.conc, "num": len(ok), "errors": len(errs), "wall_s": wall,
        "req_s": len(ok) / wall, "rtfx": audio_s / wall, "words_s": words / wall,
        "e2e_p50": pct(e2e, 50), "e2e_p90": pct(e2e, 90), "e2e_p99": pct(e2e, 99),
        "ttft_p50": pct(ttft, 50) if ttft else None, "ttft_p90": pct(ttft, 90) if ttft else None,
        "ttft_p99": pct(ttft, 99) if ttft else None, "ports": ports,
    }
    json.dump({"summary": summary, "errors": errs[:20],
               "results": [{"audio": p, "e2e_ms": r[0] * 1000, "text": r[2]} for p, r in ok]},
              open(a.out, "w"))
    print("TPUT " + json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in summary.items()}),
          flush=True)


if __name__ == "__main__":
    asyncio.run(main())
