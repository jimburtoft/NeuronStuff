"""Dynamo-traceable replacement for ``stripedhyena.utils.dotdict``.

Adapted from the Evo 2 decode package ``evo2_neuron_decode/patches/dotdict_patch.py`` (evo2_neuron_decode commit
7c93113). ``dotdict.__getattr__`` is ``dict.get`` (a C builtin); Dynamo cannot trace attribute
reads through it, so every ``config.x`` inside a compiled region is a graph break (the Evo 2 port
measured: 89 of 111 breaks). Values become real attributes; missing keys still read as ``None``.

Change vs the the Evo 2 port copy: ``get(k, default)`` honours ``default`` for missing keys (the
original version returned None because ``getattr`` hits ``__getattr__`` first). StripedHyena reads
``config.get("inner_size_multiple_of", 64)`` etc., where dotdict returns the default.
"""


class TraceableConfig:
    def __init__(self, d):
        d = dict(d)
        for k, v in d.items():
            object.__setattr__(self, k, v)
        object.__setattr__(self, "_keys", set(d.keys()))

    def __getattr__(self, k):
        if k.startswith("__"):
            raise AttributeError(k)
        return None

    def get(self, k, default=None):
        return getattr(self, k) if k in self._keys else default

    def __getitem__(self, k):
        return getattr(self, k)

    def __setitem__(self, k, v):
        object.__setattr__(self, k, v)
        self._keys.add(k)

    __setattr__ = __setitem__

    def __contains__(self, k):
        return k in self._keys

    def keys(self):
        return self._keys

    def to_dict(self):
        return {k: getattr(self, k) for k in self._keys}
