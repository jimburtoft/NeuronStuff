"""Patch upstream vLLM 0.24.0 NixlConnector (consumer / decode side) so it can pull KV
between backends that register K/V regions differently:

  mode "split":  local is blocks-first (CUDA FlashAttention/FlashInfer: 1 region per layer,
                 K|V halves inside each block); remote registers K and V as SEPARATE
                 regions per layer (Neuron: kv shape (2, num_blocks, H, B, D) ->
                 regions [L0.K, L0.V, L1.K, ...]).        -> trn2 prefill, GPU decode
  mode "merged": the reverse: local has split K/V regions (Neuron), remote is blocks-first
                 (CUDA).                                  -> GPU prefill, trn2 decode

Under HND the bytes of one block's K (H, B, D) and V (H, B, D) are identical on both sides;
only the region bookkeeping differs. Heads are contiguous (HND), so heterogeneous TP works by
reading whole remote regions (local TP < remote TP: upstream splits the local block into head
chunks) or a per-rank head slice of the remote half-block (local TP > remote TP).
Scope: same block_size, no MLA / Mamba.
Apply to vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py. Idempotent.
"""
import sys

p = sys.argv[1]
s = open(p).read()
if "_remote_split_kv" in s:
    print("already patched")
    sys.exit(0)

old_val = """        if not self._has_mamba:
            assert len(self.block_len_per_layer) == len(nixl_agent_meta.block_lens), ("""
new_val = """        if not hasattr(self, "_remote_split_kv"):
            self._remote_split_kv = {}
        _n_loc = len(self.block_len_per_layer)
        _rl = list(nixl_agent_meta.block_lens)
        _mode = None
        # head ratio local/remote (per-rank bytes scale with per-rank heads)
        _tot = self.transfer_topo.total_num_kv_heads
        _lh = self.transfer_topo.local_physical_heads
        _rh = max(1, _tot // remote_tp_size)
        if not self._has_mamba and self.transfer_topo.virtually_split_kv_in_blocks:
            if len(_rl) == 2 * _n_loc and all(
                2 * _rl[2 * i] * _lh == self.block_len_per_layer[i] * _rh
                and _rl[2 * i] == _rl[2 * i + 1]
                for i in range(_n_loc)
            ):
                _mode = "split"
        elif not self._has_mamba and _n_loc == 2 * len(_rl) and all(
            _rl[i // 2] * _lh == 2 * self.block_len_per_layer[i] * _rh
            for i in range(_n_loc)
        ):
            _mode = "merged"
        if _mode is not None:
            assert block_size_ratio == 1, (
                "cross-backend K/V region patch requires the same block_size"
            )
            assert not self.transfer_topo.is_kv_replicated(remote_engine_id), (
                "cross-backend K/V region patch: replicated KV heads not supported"
            )
            assert self.dst_num_blocks[remote_engine_id] == nixl_agent_meta.num_blocks
            logger.info(
                "Cross-backend KV regions (%s): local %d regions, remote %s %d regions, "
                "tp_ratio=%d, local_heads=%d, remote_heads=%d.",
                _mode, _n_loc, remote_engine_id, len(_rl), tp_ratio, _lh, _rh,
            )
            self._remote_split_kv[remote_engine_id] = (_mode, tp_ratio)
            return
        if not self._has_mamba:
            assert len(self.block_len_per_layer) == len(nixl_agent_meta.block_lens), ("""
assert old_val in s, "validation anchor not found"
s = s.replace(old_val, new_val, 1)

old_rem = """        \"\"\"Build remote FA descriptors for all layers.\"\"\"
        assert self.transfer_topo is not None
"""
new_rem = """        \"\"\"Build remote FA descriptors for all layers.\"\"\"
        assert self.transfer_topo is not None
        _ent = getattr(self, "_remote_split_kv", {}).get(nixl_agent_meta.engine_id)
        if _ent is not None:
            _mode, _tpr = _ent
            # local TP > remote TP: this rank reads its head slice of each remote block.
            _slot = (self.tp_rank % _tpr) if _tpr > 1 else 0
            _nsl = _tpr if _tpr > 1 else 1
            nb = nixl_agent_meta.num_blocks
            dev = nixl_agent_meta.device_id
            out: list[tuple[int, int, int]] = []
            if _mode == "split":
                # local descs: per layer, K-halves for all blocks then V-halves.
                for i in range(len(self.block_len_per_layer)):
                    for half in (0, 1):
                        base = nixl_agent_meta.kv_caches_base_addr[2 * i + half]
                        page = nixl_agent_meta.block_lens[2 * i + half]
                        chunk = page // _nsl
                        out.extend(
                            (base + b * page + _slot * chunk, chunk, dev)
                            for b in range(nb)
                        )
            else:  # "merged": local region i = (layer i//2, K|V i%2)
                for i, ln in enumerate(self.block_len_per_layer):
                    layer, half = divmod(i, 2)
                    base = nixl_agent_meta.kv_caches_base_addr[layer]
                    page = nixl_agent_meta.block_lens[layer]
                    hp = page // 2
                    chunk = hp // _nsl
                    out.extend(
                        (base + b * page + half * hp + _slot * chunk, chunk, dev)
                        for b in range(nb)
                    )
            return out
"""
assert old_rem in s, "remote-build anchor not found"
s = s.replace(old_rem, new_rem, 1)
open(p, "w").write(s)
print("patched", p)
