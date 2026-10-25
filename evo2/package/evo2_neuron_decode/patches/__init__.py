"""vortex monkey-patches. Import the modules; `api.enable()` decides which to install.

Every module exposes ``install()`` and (where reversible) ``uninstall()``.
"""
from . import (neuron_shim, fir_conv_patch, iir_recurrence_patch, rmsnorm_patch,
               rotary_patch, cast_reduce, state_concat, cascade_patch, hcm_patch,
               norm_patch, mlp_cat_patch, tkg_mha2, dotdict_patch)

__all__ = ["neuron_shim", "fir_conv_patch", "iir_recurrence_patch", "rmsnorm_patch",
           "rotary_patch", "cast_reduce", "state_concat", "cascade_patch", "hcm_patch",
           "norm_patch", "mlp_cat_patch", "tkg_mha2", "dotdict_patch"]
