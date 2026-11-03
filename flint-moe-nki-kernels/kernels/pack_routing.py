"""Single-rank, block-padded MoE routing NKI kernel for Flint (flint-native Task 016).

The packed one-expert BWMM kernels want each local expert's rows contiguous in a capacity buffer whose length is a
multiple of the block size. Building that layout in PyTorch cost ~75 ms/layer on trn2 (scatter / searchsorted / topk /
big cumsums). This kernel does the one expensive part -- the per-row packed-position scan over all N rows -- on device
in a single pass. The cheap [E]/[nb] block metadata is computed by the caller in a tiny compiled graph.

Adapted from nkilib mainline ``experimental/collectives/a2av_train/build_packed_routing`` (the EP routing scan), made
single-rank and **block-padded per expert**: expert e's rows land at ``blk_off[e]*B + rank_within_e``, where
``blk_off[e] = sum_{j<e} ceil(count[j]/B)``. Rows that overflow their expert's capacity get position C_total (dropped).
fp32 prefix arithmetic is exact to 2**24 rows.
"""

import nki
import nki.isa as nisa
import nki.language as nl

from .helpers import kernel_assert, div_ceil, ve_transpose_broadcast

TILE = 128
_FP32_EXACT = 1 << 24


@nki.jit
def pack_scatter_positions(ids, pad_row_offsets, capacity: int):
    """Packed-buffer row index for each source row, into a block-padded per-expert layout.

    Args:
        ids (NkiTensor): ``[N]`` int32, each row's local expert id in ``[0, E)``.
        pad_row_offsets (NkiTensor): ``[1, E]`` fp32, expert e's segment start row = ``B * sum_{j<e} ceil(count_j/B)``.
            Precomputed by the caller (an [E]-sized op, cheap and compile-safe).
        capacity (int): C_total, the packed-buffer row count. Positions >= C_total mean the row was dropped.

    Returns:
        scatter_pos (NkiTensor): ``[N]`` int32. Packed-buffer row index (or >= C_total if the expert overflowed).
    """
    n = ids.shape[0]
    e = pad_row_offsets.shape[1]
    kernel_assert(0 < n <= _FP32_EXACT, "pack routing requires 1 <= N <= 2**24")
    kernel_assert(1 < e <= 128, "pack routing requires 2 <= E <= 128")
    kernel_assert(pad_row_offsets.shape[0] == 1, "pad_row_offsets must be [1, E]")
    kernel_assert(nl.num_programs(axes=0) == 2, "pack routing requires LNC=2")

    scatter_pos = nl.ndarray((n, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    if nl.program_id(axis=0) == 0:
        offs = nl.ndarray((1, e), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=offs, src=pad_row_offsets)
        # per-expert cursors [E, 1], initialized to each expert's padded segment start
        cur_psum = nl.ndarray((e, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=cur_psum, data=offs)
        initial = nl.ndarray((e, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=initial, src=cur_psum)
        # expert id per partition [E, 1]
        peer_row = nl.ndarray((1, e), dtype=nl.float32, buffer=nl.sbuf)
        nisa.iota(dst=peer_row, pattern=[[1, e]], offset=0)
        peers_psum = nl.ndarray((e, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=peers_psum, data=peer_row)
        peers = nl.ndarray((e, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=peers, src=peers_psum)

        for tile in nl.affine_range(div_ceil(n, TILE)):
            start = tile * TILE
            rows = min(TILE, n - start)
            expert_int = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.dma_copy(dst=expert_int, src=ids[start : start + rows].reshape((rows, 1)))
            expert_col = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=expert_col, src=expert_int)
            experts = nl.ndarray((e, rows), dtype=nl.float32, buffer=nl.sbuf)
            ve_transpose_broadcast(expert_col, experts, rows, e)
            mask = nl.ndarray((e, rows), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=mask, data=experts, op0=nl.equal, operand0=peers)
            ones = nl.ndarray((e, rows), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(ones, 1.0)
            # inclusive running count of this expert's rows, offset by the expert's padded start cursor
            prefix = nl.ndarray((e, rows), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor_scan(dst=prefix, initial=initial, data0=ones, op0=nl.multiply, data1=mask, op1=nl.add)
            selected = nl.ndarray((e, rows), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=selected, data1=prefix, data2=mask, op=nl.multiply)
            selected_psum = nl.ndarray((rows, e), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=selected_psum, data=selected)
            selected_t = nl.ndarray((rows, e), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=selected_t, src=selected_psum)
            pos_f = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=pos_f, data=selected_t, op=nl.add, axis=1)
            # inclusive scan is one-based -> 0-based packed row (already includes the expert's pad offset)
            pos_i = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=pos_i, data=pos_f, op0=nl.subtract, operand0=1.0)
            nisa.dma_copy(dst=scatter_pos[start : start + rows].reshape((rows, 1)), src=pos_i)
            nisa.tensor_copy(dst=initial, src=prefix[:, rows - 1 : rows])

    nisa.core_barrier(scatter_pos, (0, 1))
    return scatter_pos.reshape((n,))


@nki.jit
def scatter_src_to_packed(scatter_pos, num_packed):
    """Inverse of pack_scatter_positions: token_position_to_id[packed_slot] = source row id.

    Args:
        scatter_pos (NkiTensor): [N] int32, packed slot for each source row (>= num_packed means dropped).
        num_packed (int): length of the output (capacity, a multiple of B).

    Returns:
        tpid (NkiTensor): [num_packed] int32. Source row id at each packed slot; unfilled slots = num_packed-? ->
            set to a sentinel ``num_src`` (= N) so downstream treats them as the padding token.
    """
    n = scatter_pos.shape[0]
    tpid = nl.ndarray((num_packed, 1), dtype=nl.int32, buffer=nl.shared_hbm)
    tpid_sink = nl.ndarray((num_packed + 1, 1), dtype=nl.int32, buffer=nl.shared_hbm)  # slot num_packed = dropped sink
    if nl.program_id(axis=0) == 0:
        # fill the HBM sink with the padding token id n, in 128-row partition tiles (SBUF P <= 128)
        zpad = nl.ndarray((TILE, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.memset(zpad, n)
        for ft in nl.affine_range(div_ceil(num_packed + 1, TILE)):
            fs = ft * TILE
            fr = min(TILE, num_packed + 1 - fs)
            nisa.dma_copy(dst=tpid_sink[fs : fs + fr], src=zpad[0:fr])
        for tile in nl.affine_range(div_ceil(n, TILE)):
            start = tile * TILE
            rows = min(TILE, n - start)
            pos_i = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.dma_copy(dst=pos_i, src=scatter_pos[start : start + rows].reshape((rows, 1)))
            pos_c = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=pos_c, data=pos_i, op0=nl.minimum, operand0=float(num_packed))  # drop -> sink
            pos_u = nl.ndarray((rows, 1), dtype=nl.uint32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=pos_u, src=pos_c)
            src_row = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
            srow = nl.ndarray((1, rows), dtype=nl.float32, buffer=nl.sbuf)
            nisa.iota(dst=srow, pattern=[[1, rows]], offset=start)
            srow_ps = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=srow_ps, data=srow)
            nisa.tensor_copy(dst=src_row, src=srow_ps)
            nisa.dma_copy(
                dst=tpid_sink.ap(pattern=[[num_packed + 1, rows], [1, 1]], offset=0,
                                 vector_offset=pos_u.reshape((rows,)), indirect_dim=0),
                src=src_row,
            )
        nisa.dma_copy(dst=tpid, src=tpid_sink[0:num_packed])
    nisa.core_barrier(tpid, (0, 1))
    return tpid.reshape((num_packed,))
