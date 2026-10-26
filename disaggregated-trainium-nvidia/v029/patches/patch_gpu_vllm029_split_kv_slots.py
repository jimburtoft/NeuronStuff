"""vLLM 0.29.0 GPU-side patch: BF16 FlashInfer KV cache with K and V as separate head
groups, registered with NIXL as two regions per layer. This makes the GPU KV page
byte-identical to vllm-neuron's (2, num_blocks, H, N, D) K/V slabs, so KV moves
zero-copy in both directions.

Opt-in: VLLM_FLASHINFER_SPLIT_KV_SLOTS=1 (requires VLLM_KV_CACHE_LAYOUT=LBHNC).

Under LBHNC with num_head_slots = 2*H and C = D, one GPU block is
[K heads (H, N, D) | V heads (H, N, D)]. This is the same split FlashInfer
already uses for NVFP4 KV, extended to BF16/FP16.

Upstream: vllm-project/vllm#62107 (as --attention-config.flashinfer_separate_kv_head_slots).

Usage: python patch_gpu_vllm029_split_kv_slots.py <site-packages>/vllm
Idempotent.
"""
import os
import sys

root = sys.argv[1]
FI = os.path.join(root, "v1/attention/backends/flashinfer.py")
BW = os.path.join(root, "distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py")
MARK = "_SPLIT_KV_SLOTS"


def sub(s, old, new, path, count=1):
    assert s.count(old) >= 1, f"anchor not found in {path}:\n{old[:200]}"
    return s.replace(old, new, count)


# ---------------------------------------------------------------- flashinfer.py
s = open(FI).read()
if MARK not in s:
    s = sub(s, "class FlashInferBackend(AttentionBackend):\n",
            "import os as _os\n"
            "_SPLIT_KV_SLOTS = _os.environ.get('VLLM_FLASHINFER_SPLIT_KV_SLOTS', '0') == '1'\n\n\n"
            "class FlashInferBackend(AttentionBackend):\n", FI)
    # 1) spec: separate K/V head slots for unquantized KV.
    old = """        if spec.state_content_bytes is not None or not spec.kv_quant_mode.is_nvfp4:
            return spec
"""
    new = """        if (
            _SPLIT_KV_SLOTS
            and spec.state_content_bytes is None
            and spec.kv_quant_mode == KVQuantMode.NONE
        ):
            assert spec.head_size == spec.head_size_v, (
                "split K/V slots require symmetric K/V head sizes"
            )
            return replace(
                spec,
                num_head_slots=2 * spec.num_kv_heads,
                state_content_bytes=spec.head_size * get_dtype_size(spec.dtype),
            )
        if spec.state_content_bytes is not None or not spec.kv_quant_mode.is_nvfp4:
            return spec
"""
    s = sub(s, old, new, FI)
    # 2) impl flag (next to the NVFP4 flag in both branches).
    # Impl flag: the forward / KV-write paths live on FlashInferImpl.
    s = sub(s, """        self.is_kvcache_nvfp4 = kv_cache_dtype.startswith("nvfp4")
        self.kv_cache_dtype = "nvfp4" if self.is_kvcache_nvfp4 else kv_cache_dtype
""", """        self.is_kvcache_nvfp4 = kv_cache_dtype.startswith("nvfp4")
        self.kv_cache_dtype = "nvfp4" if self.is_kvcache_nvfp4 else kv_cache_dtype
        # Only unquantized KV ("auto"/bf16/fp16) uses split K/V head slots.
        self.split_kv_slots = (
            _SPLIT_KV_SLOTS
            and not self.is_kvcache_nvfp4
            and kv_cache_dtype in ("auto", "bfloat16", "float16")
        )
""", FI)
    # 3) cascade path
    s = sub(s, """            if self.is_kvcache_nvfp4:
                kv_cache_views = tuple(""", """            if self.is_kvcache_nvfp4 or self.split_kv_slots:
                kv_cache_views = tuple(""", FI)
    # 4) main forward K/V views
    s = sub(s, """        else:
            kv_cache_tuple = kv_cache_permute.split(hs, dim=-1)
""", """        elif self.split_kv_slots:
            k_cache, v_cache = kv_cache.split(self.num_kv_heads, dim=1)
            kv_cache_tuple = (
                canonicalize_singleton_dim_strides(k_cache.permute(*stride_order)),
                canonicalize_singleton_dim_strides(v_cache.permute(*stride_order)),
            )
        else:
            kv_cache_tuple = kv_cache_permute.split(hs, dim=-1)
""", FI)
    # 5) KV write
    s = sub(s, """            if self.is_kvcache_nvfp4:
                # (B, 2*H, N, full_dim) -> ((B, N, H, full_dim),""", """            if self.is_kvcache_nvfp4 or self.split_kv_slots:
                # (B, 2*H, N, full_dim) -> ((B, N, H, full_dim),""", FI)
    open(FI, "w").write(s)
    print("patched", FI)
else:
    print("already patched", FI)

# ---------------------------------------------------------------- base_worker.py
s = open(BW).read()
if MARK not in s:
    # Register the K half and V half of each split-slot attention page as two NIXL
    # regions (same stride = full page), matching Neuron's per-layer K and V slabs.
    old = """            for base_addr, block_len, block_stride in region_specs:
                if base_addr in seen_base_addresses:"""
    new = """            if (
                _os.environ.get("VLLM_FLASHINFER_SPLIT_KV_SLOTS", "0") == "1"  # _SPLIT_KV_SLOTS
                and not is_mla_region
                and not isinstance(layer_spec, MambaSpec)
                and getattr(layer_spec, "num_head_slots", None)
                == 2 * getattr(layer_spec, "num_kv_heads", -1)
                and len(region_specs) == 1
                and region_specs[0][1] == physical_page_size
            ):
                _b, _l, _st = region_specs[0]
                region_specs = [(_b, _l // 2, _st), (_b + _l // 2, _l // 2, _st)]
            for base_addr, block_len, block_stride in region_specs:
                if base_addr in seen_base_addresses:"""
    s = sub(s, old, new, BW)
    # vllm-neuron advertises "LBNHC" but its K/V slab interior is head-major (H, N, D),
    # i.e. byte-identical to a split-slot LBHNC page. Treat the pair as equal.
    old2 = """        if not self.use_mla and nixl_agent_meta.kv_cache_layout != kv_cache_layout:"""
    new2 = """        if (
            _os.environ.get("VLLM_FLASHINFER_SPLIT_KV_SLOTS", "0") == "1"
            and nixl_agent_meta.attn_backend_name in ("CUSTOM", "neuron")
            and {nixl_agent_meta.kv_cache_layout, kv_cache_layout} <= {"LBNHC", "LBHNC"}
        ):
            logger.info(
                "Split-KV-slot peer %s (layout %s, local %s): treating layouts as "
                "byte-identical (head-major K/V slabs).",
                nixl_agent_meta.attn_backend_name,
                nixl_agent_meta.kv_cache_layout,
                kv_cache_layout,
            )
        elif not self.use_mla and nixl_agent_meta.kv_cache_layout != kv_cache_layout:"""
    s = sub(s, old2, new2, BW)
    s = "import os as _os\n" + s
    open(BW, "w").write(s)
    print("patched", BW)
else:
    print("already patched", BW)
