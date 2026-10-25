"""Make the model config traceable by Dynamo: replace dotdict with plain attributes.

*** STATUS: SHIPPED FOR BS=1 ONLY. Enable with EVO2_TRACEABLE_CFG=1. ***
  measured in the compiled (K=32) default configuration:
    graph breaks 111 -> 22 (-80%)   BS=1 37.8 -> 32.8 ms (1.15x, 6/6 pairs)
    tokens identical to compiled-cfg=0, default-config eager, AND stock
  ⚠ DO NOT ENABLE AT BS>1 YET. At BS=32 this triggers a LOAD-ORDER-DEPENDENT SBUF compile
    failure -- "Memory Location: {input5_local_1439_set}@SB0" -- which reproduces through an
    exec()-of-source harness while cfg=0 through the identical harness works, yet does NOT
    appear when decode_correct.py runs as a direct script. Attributable, reproducible, and
    fragile. The BS=32 gain is only 1.03x, so it is not worth the risk until understood.

  History: this patch was REJECTED earlier (breaks 2 -> 0, decode 58% SLOWER). That was
  correct FOR THE MODEL OF 2026-09-01, which had 2 graph breaks. Today's model has 111, and
  torch.compile itself flipped from a pessimization to a 1.43x win. A negative
  result has a shelf life.

ROOT CAUSE (TORCH_LOGS=graph_breaks):
    Graph break in user code at vortex/model/model.py:251
    Graph Break Reason: User-defined object with non-function __getattr__
    Explanation: Found a non-function __getattr__ <method 'get' of 'dict' objects>
                 ... when attempting to getattr `interleave`

`vortex.model.utils.dotdict` does `__getattr__ = dict.get`. That is a C builtin
method, not a Python function, and Dynamo refuses to trace attribute access through
it. Every `self.config.<x>` read inside a compiled region is therefore a GRAPH BREAK.

The model reads config attributes all over its forward path (`config.interleave`,
`config.prefill_style`, `config.column_split_hyena`, ...), so the compiled region is
shredded into many small graphs, each with a device sync at the boundary. That is why
MHA's three components each measured ~1250-1509 ms in the real model but only
0.2-2.3 ms in isolation -- a ~600x gap that is pure graph-break overhead, not compute.

FIX: convert the config into a plain object whose attributes are real attributes
(found via normal `__getattribute__`, which Dynamo traces fine). Values are copied
once at construction; the config is read-only during inference.

This changes no math. It only changes how attribute lookup is implemented.
"""
import torch


class TraceableConfig:
    """Plain-attribute config. Dynamo can trace attribute access on this."""
    def __init__(self, d):
        for k, v in dict(d).items():
            object.__setattr__(self, k, v)
        object.__setattr__(self, "_keys", set(dict(d).keys()))

    # *** dotdict's __getattr__ IS dict.get, which returns None for MISSING keys. ***
    # A plain object raises AttributeError instead, which is a BEHAVIOUR CHANGE, not just a
    # tracing change: the model reads optional config keys (e.g. params_dtype) and relies on
    # getting None back. Reproduce dotdict's semantics exactly.
    def __getattr__(self, k):
        if k.startswith("__"):
            raise AttributeError(k)
        return None

    # keep dict-style access working for any code that uses .get()/[]
    def get(self, k, default=None):
        return getattr(self, k, default)

    def __getitem__(self, k):
        return getattr(self, k)

    def __setitem__(self, k, v):
        object.__setattr__(self, k, v)
        self._keys.add(k)

    def __contains__(self, k):
        return hasattr(self, k)

    def keys(self):
        return self._keys

    def to_dict(self):
        return {k: getattr(self, k) for k in self._keys}


def make(cfg):
    """Build a TraceableConfig from a dotdict/dict."""
    return TraceableConfig(cfg)
