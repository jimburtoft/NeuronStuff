"""vLLM 0.29.0 patch: allow P_TP > num_kv_heads with P_TP > D_TP (GQA-replicated
prefill KV) in the NIXL pull connector.

When the prefill TP exceeds the KV-head count, several prefill ranks hold the same
KV head. compute_tp_mapping already reads one replica per head, so the decode side
still head-splits its region. The handshake asserted this case away. The patch:
  1. relaxes that assert (pull mode, non-MLA, non-Mamba);
  2. validates block_len as SPLIT (one head per remote rank) instead of REPLICATE;
  3. sends the completion notification to the skipped replica ranks, as the MLA
     path already does, so they free their blocks without waiting for lease expiry.

Backport of vllm-project/vllm#63648 to v0.29.0.
Usage: python patch_gpu_vllm029_gqa_replicated_pull.py <site-packages>/vllm
Idempotent.
"""
import os
import sys

root = sys.argv[1]
D = os.path.join(root, "distributed/kv_transfer/kv_connector/v1/nixl")
BW = os.path.join(D, "base_worker.py")
PW = os.path.join(D, "pull_worker.py")
MARK = "_is_gqa_replicated_pull"


def sub(s, old, new, path):
    assert s.count(old) == 1, f"anchor count {s.count(old)} in {path}:\n{old[:300]}"
    return s.replace(old, new)


s = open(BW).read()
if MARK not in s:
    s = sub(s, """        return tp_ratio < 0 and (not self.use_mla or len(plan.all_source_ranks) > 1)
""", """        return tp_ratio < 0 and (not self.use_mla or len(plan.all_source_ranks) > 1)

    def _is_gqa_replicated_pull(self, tp_ratio: int, remote_engine_id: str) -> bool:
        \"\"\"Whether D pulls from a larger P whose GQA KV heads are replicated.\"\"\"
        assert self.transfer_topo is not None
        return (
            self._TRANSFER_MODE == "pull"
            and tp_ratio < 0
            and not self.use_mla
            and not self._has_mamba
            and self.transfer_topo.is_kv_replicated(remote_engine_id)
        )
""", BW)
    s = sub(s, """        if not self.use_mla and not self._has_mamba:
            assert not (
                tp_ratio < 0 and self.transfer_topo.is_kv_replicated(remote_engine_id)
            )
""", """        gqa_replicated_pull = self._is_gqa_replicated_pull(tp_ratio, remote_engine_id)
        if not self.use_mla and not self._has_mamba and not gqa_replicated_pull:
            assert not (
                tp_ratio < 0 and self.transfer_topo.is_kv_replicated(remote_engine_id)
            )
""", BW)
    s = sub(s, """            and not self.use_mla
            and not self.transfer_topo.is_kv_replicated(remote_engine_id)
            and not KVCacheLayout[kv_cache_layout].is_block_contiguous""", """            and not self.use_mla
            and (
                gqa_replicated_pull
                or not self.transfer_topo.is_kv_replicated(remote_engine_id)
            )
            and not KVCacheLayout[kv_cache_layout].is_block_contiguous""", BW)
    s = sub(s, """            model_replicated = self.use_mla or self.transfer_topo.is_kv_replicated(
                remote_engine_id
            )
""", """            model_replicated = self.use_mla or (
                not gqa_replicated_pull
                and self.transfer_topo.is_kv_replicated(remote_engine_id)
            )
""", BW)
    open(BW, "w").write(s)

s = open(PW).read()
if MARK not in s:
    s = sub(s, """        if self.use_mla and tp_ratio < 0 and len(read_specs) == 1:
""", """        gqa_replicated = self._is_gqa_replicated_pull(tp_ratio, engine_id)
        if (self.use_mla and tp_ratio < 0 and len(read_specs) == 1) or gqa_replicated:
""", PW)
    s = sub(s, """            remote_agents = self._remote_agents[meta.remote.engine_id]
            for rank_to_notify, agent in remote_agents.items():
                if rank_to_notify != (0, read_specs[0].remote_rank):
""", """            remote_agents = self._remote_agents[meta.remote.engine_id]
            read_ranks = {(0, spec.remote_rank) for spec in read_specs}
            for rank_to_notify, agent in remote_agents.items():
                if rank_to_notify not in read_ranks:
""", PW)
    open(PW, "w").write(s)
print("gqa_replicated_pull patch applied")
