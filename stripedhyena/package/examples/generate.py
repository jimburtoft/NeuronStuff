"""Greedy generation with StripedHyena-Hessian-7B / -Nous-7B on one trn2 device.

  python examples/generate.py --weights /weights/hessian-7b --prompt "The striped hyena is"
  torchrun --standalone --nproc_per_node=4 examples/generate.py --tp 4 --weights /weights/hessian-7b
  Nous chat format: --prompt "### Instruction:\\nWhat is a hyena?\\n\\n### Response:\\n" --weights /weights/nous-7b
"""
import argparse
import os
import time

import torch

import striped_hyena_neuron as shn

ap = argparse.ArgumentParser()
ap.add_argument("--weights", default="/weights/hessian-7b")
ap.add_argument("--prompt", default="The striped hyena is")
ap.add_argument("--max_new_tokens", type=int, default=64)
ap.add_argument("--B", type=int, default=1)
ap.add_argument("--window", type=int, default=1024)
ap.add_argument("--tp", type=int, default=1)
a = ap.parse_args()

tp = None
if a.tp > 1:
    import torch.distributed as dist
    dist.init_process_group(backend="neuron")
    tp = (dist.get_rank(), a.tp)
from tokenizers import Tokenizer  # noqa: E402
tok = Tokenizer.from_file(shn.TOKENIZER_JSON)
ids = torch.tensor([tok.encode(a.prompt.encode().decode("unicode_escape")).ids] * a.B)
model, info = shn.load(a.weights, max_seqlen=a.window, tp=tp)
gen = shn.compile(model, B=a.B, window=a.window, tp=a.tp)
with torch.inference_mode():
    gen.generate(ids, 8)                          # first call compiles
    t0 = time.time()
    out = gen.generate(ids, a.max_new_tokens, eos=2)
    dt = time.time() - t0
if int(os.environ.get("RANK", "0")) == 0:
    print(shn.config_summary())
    print(repr(tok.decode(out[0].tolist())))
    print(f"{out.shape[1]} tokens x B={a.B} in {dt:.2f} s (incl. prefill)")
