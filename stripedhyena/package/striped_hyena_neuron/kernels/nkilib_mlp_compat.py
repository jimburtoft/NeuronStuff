"""Runtime compatibility patch for nkilib's MLP kernel under an earlier NKI 0.7 frontend (torch-neuronx 2.13 / SDK 2.32). NOT needed on SDK 2.33 (NKI 0.7.0 GA):
``install()`` is a no-op there (see ``needed()``).

Issue (earlier NKI 0.7 frontend, torch-neuronx 2.13,
mlp_cte_basic_transpose.py md5 18cf1614d57ac4e4dab79145408e3c35): tracing ``nkilib.core.mlp.mlp`` through
``wrap_nki``/``nki_op`` fails with ``failed to specialize NKI kernel: unsupported expression`` at the two list
comprehensions in ``mlp_cte_basic_transpose._perform_intermediate_transpose``. The NKI frontend parses the
whole call tree, so the failure hits every mode (AUTO / DECODE / PREFILL), including decode-only calls.

Fix: replace that one helper with a body that builds ``int_blocks`` with explicit loops (same values, same
order). The installed nkilib files are not modified; ``install()`` swaps the module attribute before the
first trace. Source of the original: nkilib (Apache-2.0), mlp_cte_basic_transpose.py lines 319-375.
"""
import nki.isa as nisa
import nki.language as nl

from nkilib.core.utils.kernel_assert import kernel_assert
from nkilib.core.utils.tiled_range import TiledRange

_DONE = False


def _perform_intermediate_transpose(
    mlp_params,
    tile_info,
    constants,
    int_tile_idx,
    int_tile_rest,
    bxs_subtile_rest,
    psum_step_size,
    int_tile_sbuf,
    res_psum_tensor,
    psum_tile_info,
):
    bxs_dim_tile = tile_info.bxs_dim_tile
    int_dim_tile = tile_info.xpose_intermediate_dim_tile
    BXS_SUBTILE_SIZE = bxs_dim_tile.subtile_dim_info.tile_size
    I_SUBTILE_SIZE = int_dim_tile.subtile_dim_info.tile_size

    bxs_subtile_bound = min(bxs_subtile_rest, BXS_SUBTILE_SIZE)
    int_tile_bound = min(int_tile_rest, int_dim_tile.tile_size)

    int_blocks = []
    if mlp_params.quant_params.is_quant_row() or mlp_params.quant_params.is_quant_static():
        for int_group in TiledRange(int_tile_bound, 2 * I_SUBTILE_SIZE):
            block_count = 2 if int_group.size > I_SUBTILE_SIZE else 1
            kernel_assert(
                int_group.size % block_count == 0,
                f'Intermediate size per core must be even for the double row down projection, got {int_group.size}',
            )
            block_size = int_group.size // block_count
            for i in range(block_count):
                int_blocks.append((int_group.start_offset + i * block_size, block_size))
    else:
        for int_subtile in TiledRange(int_tile_bound, I_SUBTILE_SIZE):
            int_blocks.append((int_subtile.start_offset, int_subtile.size))

    block_idx = 0
    for blk in int_blocks:
        block_start = blk[0]
        block_size = blk[1]
        nisa.nc_transpose(
            dst=res_psum_tensor.ap(
                pattern=[
                    [psum_tile_info.tile_count * psum_tile_info.tile_size * psum_step_size, block_size],
                    [1, 1],
                    [psum_step_size, bxs_subtile_bound],
                ],
                offset=block_idx * I_SUBTILE_SIZE * psum_step_size,
            ),
            data=int_tile_sbuf[
                :bxs_subtile_bound,
                nl.ds(int_tile_idx * int_dim_tile.tile_size + block_start, block_size),
            ],
        )
        block_idx += 1


def needed():
    """Patch only on the earlier NKI 0.7 frontend (torch-neuronx 2.13). NKI 0.7.0 GA (SDK 2.33) specializes kernels by
    Python tracing and runs the unpatched nkilib (verified on trn2, 2026-10-20). SHN_NKILIB_COMPAT=1/0 forces it on/off."""
    import os
    env = os.environ.get("SHN_NKILIB_COMPAT")
    if env in ("0", "1"):
        return env == "1"
    try:
        import nki
        return "b" in nki.__version__.split("+")[0]
    except Exception:  # noqa: BLE001
        return True


def install():
    global _DONE
    if _DONE:
        return
    if not needed():
        _DONE = True
        return
    from nkilib.core.mlp.mlp_cte.basic import mlp_cte_basic_transpose as T
    T._perform_intermediate_transpose = _perform_intermediate_transpose
    _DONE = True
