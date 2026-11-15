"""Load songlab/PhyloGPN from the Hugging Face Hub into phylogpn_neuron.PhyloGPN.

The Hub checkpoint stores the raw tensors that the Hub code symmetrizes on every forward with
`torch.nn.utils.parametrize` (reverse-complement equivariant weights / involution-equivariant
biases). `fold_hub_state_dict` applies those maps once, reproducing `modeling_phylogpn.py` exactly.
"""
import os

import torch

from .model import PhyloGPN

HUB_ID = "songlab/PhyloGPN"
HUB_REVISION = "a763739cfaa653fc8305ec908a0b24d6f03c7f05"  # main after the transformers-5 fix (PR #1)


def _involution(n):
    # channel i <-> channel n-1-i (the Hub code's involution_indices for an even channel count)
    return torch.arange(n - 1, -1, -1)


def _rce_weight(w):
    """RCEWeight: (W + W[out_inv][:, in_inv].flip(kernel)) / 2."""
    oi, ii = _involution(w.shape[0]), _involution(w.shape[1])
    return (w + w[oi][:, ii].flip(2)) / 2


def _ie_bias(b):
    return (b + b[_involution(b.shape[0])]) / 2


def _ie_weight_embedding(e):
    """IEWeight on the embedding [6, 960]: (E + E[token_inv][:, channel_inv]) / 2 with
    token_inv = complement (A<->T, C<->G; N and pad fixed) and channel_inv = channel reversal."""
    comp = torch.tensor([3, 2, 1, 0, 4, 5])
    return (e + e[comp][:, _involution(e.shape[1])]) / 2


def fold_hub_state_dict(sd):
    """Hub state_dict (with parametrizations.*.original keys) -> plain PhyloGPN state_dict."""
    out = {}
    for k, v in sd.items():
        k2 = k.removeprefix("_model.")
        if ".parametrizations." in k2:
            mod, rest = k2.split(".parametrizations.")
            name = rest.split(".")[0]
            v = v.float()
            if name == "weight" and v.dim() == 3:
                v = _rce_weight(v)
            elif name == "weight" and v.dim() == 2:
                v = _ie_weight_embedding(v)
            elif name == "bias":
                v = _ie_bias(v)
            out[f"{mod}.{name}"] = v
        else:
            out[k2] = v.float()
    return out


def load(path=None, revision=HUB_REVISION, check_against_hub=False):
    """Return a CPU fp32 PhyloGPN. `path`: a local dir/file with model.safetensors, else download."""
    from safetensors.torch import load_file
    if path is None:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(HUB_ID, "model.safetensors", revision=revision)
    elif os.path.isdir(path):
        path = os.path.join(path, "model.safetensors")
    raw = load_file(path)
    m = PhyloGPN()
    sd = fold_hub_state_dict(raw) if any(".parametrizations." in k for k in raw) else {k.removeprefix("_model."): v for k, v in raw.items()}
    m.load_state_dict(sd, strict=True)
    return m.eval()
