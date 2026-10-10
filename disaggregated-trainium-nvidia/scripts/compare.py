"""Token-level comparison of DI output vs single-device references (greedy)."""
import json, sys, time, urllib.request
from transformers import AutoTokenizer

EP = {k: v for k, v in (a.split("=", 1) for a in sys.argv[2:])}
tok = AutoTokenizer.from_pretrained(sys.argv[1])
PROMPTS = [
    "The capital of France is", "def fibonacci(n):", "Explain photosynthesis in one paragraph.",
    "List three prime numbers greater than 100:", "Translate to Spanish: The weather is nice today.",
    "Once upon a time, in a small village,", "The theory of relativity states that",
    "Write a haiku about the ocean.", "import numpy as np\n# compute the mean of an array\n",
    "Q: What is 17 * 23?\nA:", "The three branches of the US government are",
    "In machine learning, overfitting means", "SELECT name FROM users WHERE",
    "The largest planet in our solar system is", "Recipe for pancakes:\n1.",
    "The French Revolution began in", "A linked list is a data structure that",
    "Dear hiring manager,", "The speed of light in a vacuum is approximately",
    "def quicksort(arr):",
    # long prompt (>1 block of 32, ~600 tokens)
    ("Summarize the following text.\n" + "The quick brown fox jumps over the lazy dog. " * 60),
]
N = 48

def run(url, p):
    body = json.dumps({"model": "llama", "prompt": p, "max_tokens": N, "temperature": 0}).encode()
    r = urllib.request.Request(url + "/v1/completions", body, {"Content-Type": "application/json"})
    t0 = time.time(); d = json.load(urllib.request.urlopen(r, timeout=300)); dt = time.time() - t0
    return tok(d["choices"][0]["text"], add_special_tokens=False)["input_ids"], dt

def prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y: break
        n += 1
    return n

res = {k: [run(u, p) for p in PROMPTS] for k, u in EP.items()}
names = list(EP)
pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
print(f"{'#':>2} {'plen':>5} " + " ".join(f"{a[:6]}~{b[:6]:>6}" for a, b in pairs))
agg = {pr: [] for pr in pairs}
for i, p in enumerate(PROMPTS):
    row = []
    for a, b in pairs:
        m = prefix(res[a][i][0], res[b][i][0]); agg[(a, b)].append(m); row.append(f"{m:>13}")
    print(f"{i:>2} {len(tok(p)['input_ids']):>5} " + " ".join(row))
for (a, b), v in agg.items():
    full = sum(1 for x in v if x >= N)
    print(f"{a} vs {b}: mean common-prefix {sum(v)/len(v):.1f}/{N} tokens, exact-match {full}/{len(v)}")
for k in names:
    print(f"{k}: mean latency {sum(t for _, t in res[k])/len(res[k]):.3f} s for {N} tokens")
json.dump({k: [x[0] for x in v] for k, v in res.items()}, open("compare_tokens.json", "w"))
