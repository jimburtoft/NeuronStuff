"""Worker for tests/test_hardware.py (one configuration per process)."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ap = argparse.ArgumentParser()
ap.add_argument("what", choices=["parity", "generate"])
ap.add_argument("--tp", type=int, default=1)
ap.add_argument("--mode", default=None)
ap.add_argument("--cascade", action="store_true")
ap.add_argument("--tkg", action="store_true", help="force attention_tkg (normally only at 32K windows)")
ap.add_argument("--B", type=int, default=1, help="generate: batch (copies of one prompt)")
a = ap.parse_args()

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torch_neuronx  # noqa: E402

import striped_hyena_neuron as shn  # noqa: E402

WTS, REF = os.environ["SHN_WEIGHTS"], os.environ.get("SHN_REFDATA", "")
DEV = torch.device("neuron")
RANK = int(os.environ.get("RANK", "0"))


def logit_cmp(x, y):
    x, y = x.float(), y.float()
    ly, lx = F.log_softmax(y, -1), F.log_softmax(x, -1)
    return {"rel_l2": ((x - y).norm() / y.norm()).item(), "top1": (x.argmax(-1) == y.argmax(-1)).float().mean().item(),
            "kl_mean": (ly.exp() * (ly - lx)).sum(-1).mean().item()}


COMPACT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "parity_reference.json")


def topk_cmp(x, gen, top_ids, top_logp):
    """vs the compact reference: top-1 agreement with the reference argmax (= its greedy tokens ``gen``; the top-16 list
    can order exact ties differently from argmax) and the KL restricted to the reference top-16 tokens
    (sum p_ref (log p_ref - log p_ours) over them; a lower bound of the full KL's dominant part)."""
    lx = F.log_softmax(x.float(), -1)
    n = min(len(top_ids), len(lx))
    ids = torch.tensor(top_ids[:n])
    lr = torch.tensor(top_logp[:n])
    lo = lx[:n].gather(1, ids)
    return {"top1": (lx[:n].argmax(-1) == torch.tensor(gen[:n])).float().mean().item(),
            "kl_top16": (lr.exp() * (lr - lo)).sum(-1).mean().item(), "n": n}


def prompts():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(shn.TOKENIZER_JSON)
    return tok


@torch.inference_mode()
def main():
    tp = None
    if a.tp > 1:
        import torch.distributed as dist
        dist.init_process_group(backend="neuron")
        tp = (dist.get_rank(), a.tp)
    kw = {"cascade": True} if a.cascade else {}
    model, info = shn.load(WTS, max_seqlen=1024, tp=tp, **kw)
    gen = shn.compile(model, B=a.B, window=1024, tp=a.tp, mode=a.mode)
    if a.tkg:
        gen.d["tkg"] = True
    out = {"what": a.what, "tp": a.tp, "decode": gen.d, "config": shn.config_summary()}
    torch_neuronx.clear_op_tracking()
    if a.what == "parity":
        full = os.path.isdir(os.path.join(REF, "ref"))
        # prompt ids + compact reference (top-16 log-probs per position) ship with the package (data/README.md)
        C = json.load(open(COMPACT))["prompts"]
        P = {nm: C[nm]["prompt"] for nm in C}
        out["parity"], out["reference"] = {}, ("full" if full else "compact")
        for nm in ("short", "medium_512"):
            toks = torch.load(os.path.join(REF, "ref", f"gen_fft_{nm}.pt"))["gen"].tolist() if full else C[nm]["gen"]
            ids = torch.tensor([P[nm]])
            lg_p, _, S = gen.pf(ids.to(DEV), gen.W)          # compiled prefill: last-position logits
            gen._prepare(S, len(P[nm]) + len(toks))
            lgs = [lg_p[0].float().cpu()]
            for t in toks[:-1]:
                lg, _ = gen.dec(torch.tensor([t], device=DEV), S)
                lgs.append(lg[0].float().cpu())
            got = torch.stack(lgs)
            if full:
                r = torch.load(os.path.join(REF, "ref", f"forced_cached_{nm}.pt"))
                n = min(len(r), len(got))
                out["parity"][nm] = logit_cmp(got[:n], r[:n])
            else:
                out["parity"][nm] = topk_cmp(got, C[nm]["gen"], C[nm]["top_ids"], C[nm]["top_logp"])
    else:
        tok = prompts()
        ids = tok.encode("The striped hyena is").ids
        g = gen.generate(torch.tensor([ids] * a.B), 24)
        out["rows_identical"] = bool((g == g[0]).all())
        out["tokens_row0"] = g[0].tolist()
        out["text"] = tok.decode(g[0].tolist())
    out["fallback_ops"] = sorted(set(torch_neuronx.get_fallback_ops() or []))
    if RANK == 0:
        print("HWRESULT " + json.dumps(out), flush=True)


main()
