"""Needle retrieval against an OpenAI-compatible server, at chosen prompt lengths (tokens, via the tokenizer).

Usage: python needle_server.py <model id or dir, as served> <prompt lengths, comma-separated> [prompts per length]

Plants a random 6-digit code at a random depth in filler text and asks for it back.
"""
import json
import random
import sys
import time
import urllib.request

from transformers import AutoTokenizer

model, lens = sys.argv[1], [int(x) for x in sys.argv[2].split(",")]
served = model
per = int(sys.argv[3]) if len(sys.argv) > 3 else 2
tok = AutoTokenizer.from_pretrained(model)
FILLER = ("The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
          "A quiet river runs past the old mill, and the town wakes slowly every morning. ")
fl = len(tok(FILLER).input_ids)
rng = random.Random(0)
ok = n = 0
for L in lens:
    for k in range(per):
        code = rng.randint(100000, 999999)
        city = rng.choice(["Lisbon", "Osaka", "Nairobi", "Quito", "Tromso", "Hobart"])
        parts = [FILLER] * max(1, (L - 60) // fl)
        depth = (0.1 + 0.8 * rng.random())
        parts.insert(int(len(parts) * depth), f" The secret code for {city} is {code}. ")
        q = "".join(parts) + f"\n\nWhat is the secret code for {city}? Reply with the number only."
        body = json.dumps({"model": served, "messages": [{"role": "user", "content": q}],
                           "temperature": 0, "max_tokens": 24}).encode()
        t = time.time()
        r = json.load(urllib.request.urlopen(urllib.request.Request(
            "http://127.0.0.1:8000/v1/chat/completions", body, {"Content-Type": "application/json"}), timeout=1800))
        txt = r["choices"][0]["message"]["content"] or ""
        pt = r["usage"]["prompt_tokens"]
        hit = str(code) in txt
        ok += hit
        n += 1
        print(f"NEEDLE len~{L} prompt_tokens={pt} depth={depth:.2f} ok={hit} e2e={time.time() - t:.1f}s out={txt[:40]!r}", flush=True)
print(f"NEEDLE_TOTAL {ok}/{n}")
