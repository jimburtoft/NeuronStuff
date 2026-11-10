"""GSM8K (zero-shot CoT, greedy) through an OpenAI-compatible server, 64 requests at a time.

Usage: python gsm8k_server.py <tag> [n (0 = all 1,319)] [max_tokens] [--data gsm8k_test.jsonl]
       [--model mistralai/Mistral-7B-Instruct-v0.3] [--url http://127.0.0.1:8000]
"""
import argparse
import concurrent.futures as cf
import json
import re
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("tag")
ap.add_argument("n", nargs="?", type=int, default=0)
ap.add_argument("max_tokens", nargs="?", type=int, default=384)
ap.add_argument("--data", default="gsm8k_test.jsonl")
ap.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
ap.add_argument("--url", default="http://127.0.0.1:8000")
a = ap.parse_args()

items = [json.loads(x) for x in open(a.data)]
if a.n:
    items = items[: a.n]
SUF = "\nSolve the problem step by step. End your response with 'The answer is N.' where N is a number."


def ask(q):
    body = json.dumps({"model": a.model, "messages": [{"role": "user", "content": q + SUF}],
                       "temperature": 0, "max_tokens": a.max_tokens}).encode()
    req = urllib.request.Request(f"{a.url}/v1/chat/completions", body, {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=1200))["choices"][0]["message"]["content"] or ""


def pred(t):
    m = re.findall(r"answer is\s*\$?\s*(-?[\d,]*\.?\d+)", t) or re.findall(r"-?[\d,]*\.?\d+", t)
    return m[-1].replace(",", "").rstrip(".") if m else None


with cf.ThreadPoolExecutor(64) as ex:
    outs = list(ex.map(ask, [i["question"] for i in items]))
ok = 0
for it, t in zip(items, outs):
    g = it["answer"].split("####")[-1].strip().replace(",", "")
    p = pred(t)
    try:
        ok += p is not None and abs(float(p) - float(g)) < 1e-6
    except ValueError:
        pass
print(f"GSM8K {a.tag} n={len(items)} acc={ok / len(items):.4f}")
