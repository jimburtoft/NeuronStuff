"""StripedHyena 1 (Together AI StripedHyena-Hessian-7B / -Nous-7B) inference on AWS Trainium 2 via
PyTorch Native (``torch.device("neuron")``).

WHAT THIS IS
    Runtime patches for the upstream ``stripedhyena`` model code (vendored, unmodified) that remove
    every CUDA / complex-dtype / FFT dependency, plus a compiled prefill and decode path, an NKI decode
    cascade kernel, nkilib ``attention_tkg`` for long-context GQA decode, and Megatron tensor
    parallelism. Model math is unchanged: logits match the upstream flash-attn reference to bf16 noise.

QUICK START
    import torch, striped_hyena_neuron as shn
    model, info = shn.load("/weights/hessian-7b")            # enable() + meta build + strict load + to_device
    gen = shn.compile(model, B=1, window=4096)               # compiled prefill + decode (NOT optional)
    out = gen.generate(torch.tensor([ids]), max_new_tokens=128)
    print(shn.config_summary())

    TP=4 (one trn2.3xlarge, lowest latency and highest throughput):
        torchrun --nproc_per_node=4 ...; dist.init_process_group(backend="neuron")
        model, _ = shn.load(w, tp=(dist.get_rank(), 4)); gen = shn.compile(model, B=B, window=W, tp=4)

Three calls exist because the obvious version silently fails:
    load / to_device   bare ``.to()`` leaves plain tensor attributes on the host
    make_config        dotdict ``__getattr__`` is ``dict.get``: every config read graph-breaks
    compile            eager decode is ~4x slower; partial block compilation is slower than none
"""

from .api import (enable, is_enabled, config_summary, make_config, to_device, load, compile, decode_opts,
                  Generator, MAX_CONTEXT, CONFIG_YML, TOKENIZER_JSON)

__version__ = "0.1.0"
__all__ = ["enable", "is_enabled", "config_summary", "make_config", "to_device", "load", "compile",
           "decode_opts", "Generator", "MAX_CONTEXT", "CONFIG_YML", "TOKENIZER_JSON", "__version__"]
