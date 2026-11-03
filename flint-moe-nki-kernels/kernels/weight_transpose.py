"""Pre-transpose MoE expert weights once, so the blockwise MoE backward does not re-transpose per block.

Profile of ``blockwise_mm_bwd`` at the EP=16 shape (flint-native Task 017, 16384 rows x 16 experts, B=256):
the backward is 95% DMA-bound; 67 of its 77 ms are ``nc_transpose`` + ``tensor_copy`` of the gate/up and down
weights, redone for EVERY block even though all blocks of one expert share the same weight. For E experts spanning
N blocks that is N transposes where E suffice (N/E ~= 25x at this shape).

This kernel transposes each expert's weights ONCE into HBM scratch, in the layout the backward's per-block
transpose produces:
  * gate_up: ``[E, H, 2, I_TP]`` -> ``[E, 2, I_TP, H]``  (the ``rhs_tiles`` layout in ``_compute_hidden_states_grad``)
  * down:    ``[E, I_TP, H]``    -> ``[E, H, I_TP]``       (the transposed layout in ``_compute_gate_up_projection_output_grad``)

The backward can then plain-load these instead of transposing (the integration edit in ``bwmm_bwd_dropless.py``).
Bit-exact: a transpose moves data, it does not change values.
"""

import nki
import nki.isa as nisa
import nki.language as nl

from .helpers import kernel_assert, div_ceil, ve_transpose_broadcast

TILE = 128


@nki.jit
def transpose_gate_up(gate_up_proj_weight):
    """[E, H, 2, I_TP] -> [E, 2, I_TP, H]. Transpose the (H, I_TP) planes per expert and gate/up.

    H and I_TP are tiled in 128-chunks; each (h_tile, i_tile) sub-block is transposed with nc_transpose.
    """
    E, H, G, I_TP = gate_up_proj_weight.shape
    kernel_assert(G == 2, "gate_up weight must have 2 projections")
    out = nl.ndarray((E, G, I_TP, H), dtype=gate_up_proj_weight.dtype, buffer=nl.shared_hbm)
    if nl.program_id(axis=0) == 0:
        for e in nl.affine_range(E):
            for g in nl.affine_range(G):
                for ht in nl.affine_range(div_ceil(H, TILE)):
                    hs = ht * TILE
                    hr = min(TILE, H - hs)
                    for it in nl.affine_range(div_ceil(I_TP, TILE)):
                        isz = it * TILE
                        ir = min(TILE, I_TP - isz)
                        src = nl.ndarray((hr, ir), dtype=gate_up_proj_weight.dtype, buffer=nl.sbuf)
                        nisa.dma_copy(dst=src, src=gate_up_proj_weight[e, hs:hs + hr, g, isz:isz + ir])
                        tp = nl.ndarray((ir, hr), dtype=gate_up_proj_weight.dtype, buffer=nl.psum)
                        nisa.nc_transpose(dst=tp, data=src)
                        dst = nl.ndarray((ir, hr), dtype=gate_up_proj_weight.dtype, buffer=nl.sbuf)
                        nisa.tensor_copy(dst=dst, src=tp)
                        nisa.dma_copy(dst=out[e, g, isz:isz + ir, hs:hs + hr], src=dst)
    nisa.core_barrier(out, (0, 1))
    return out


@nki.jit
def transpose_down(down_proj_weight):
    """[E, I_TP, H] -> [E, H, I_TP]. Transpose the (I_TP, H) plane per expert."""
    E, I_TP, H = down_proj_weight.shape
    out = nl.ndarray((E, H, I_TP), dtype=down_proj_weight.dtype, buffer=nl.shared_hbm)
    if nl.program_id(axis=0) == 0:
        for e in nl.affine_range(E):
            for it in nl.affine_range(div_ceil(I_TP, TILE)):
                isz = it * TILE
                ir = min(TILE, I_TP - isz)
                for ht in nl.affine_range(div_ceil(H, TILE)):
                    hs = ht * TILE
                    hr = min(TILE, H - hs)
                    src = nl.ndarray((ir, hr), dtype=down_proj_weight.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(dst=src, src=down_proj_weight[e, isz:isz + ir, hs:hs + hr])
                    tp = nl.ndarray((hr, ir), dtype=down_proj_weight.dtype, buffer=nl.psum)
                    nisa.nc_transpose(dst=tp, data=src)
                    dst = nl.ndarray((hr, ir), dtype=down_proj_weight.dtype, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=dst, src=tp)
                    nisa.dma_copy(dst=out[e, hs:hs + hr, isz:isz + ir], src=dst)
    nisa.core_barrier(out, (0, 1))
    return out
