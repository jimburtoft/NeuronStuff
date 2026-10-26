"""vllm-neuron 0.29 (0.29.0.1.2.0, Neuron SDK 2.33) patch: make
NeuronNixlConnector stride-aware when it builds REMOTE descriptors. Without it, a
remote that registers K/V as strided half-pages breaks.

A GPU peer patched with patch_gpu_vllm029_split_kv_slots.py registers each layer as
two regions (K half, V half), each `block_len = page/2` with `block_stride = page`.
NeuronNixlConnector._register_remote_descs addresses remote block b at
`base + b * block_len`, which is only right for dense slabs like Neuron's own.
This uses the remote's advertised `block_strides` (metadata field, NIXL v9+)
instead.

Usage: python patch_neuron_029_remote_strides.py <site-packages>/vllm_neuron
Idempotent.
"""
import os
import sys

p = os.path.join(sys.argv[1], "vllm/kv_connector/neuron_nixl_connector.py")
s = open(p).read()
MARK = "_remote_block_stride"
if MARK in s:
    print("already patched", p)
    sys.exit(0)

old = """        kv_addrs = nixl_agent_meta.kv_caches_base_addr
        blocks_data = []
        for kv_group_idx, base_addr in enumerate(kv_addrs):
            page_size = nixl_agent_meta.block_lens[kv_group_idx]
            for block_id in range(nixl_agent_meta.num_blocks):
                addr = base_addr + block_id * page_size + offset"""
new = """        kv_addrs = nixl_agent_meta.kv_caches_base_addr
        _strides = list(getattr(nixl_agent_meta, "block_strides", None) or [])
        blocks_data = []
        for kv_group_idx, base_addr in enumerate(kv_addrs):
            page_size = nixl_agent_meta.block_lens[kv_group_idx]
            # _remote_block_stride: a GPU peer may register strided K/V half-pages.
            _remote_block_stride = (
                _strides[kv_group_idx] if kv_group_idx < len(_strides) else page_size
            ) or page_size
            for block_id in range(nixl_agent_meta.num_blocks):
                addr = base_addr + block_id * _remote_block_stride + offset"""
assert old in s, "anchor not found"
s = s.replace(old, new, 1)
open(p, "w").write(s)
print("patched", p)
