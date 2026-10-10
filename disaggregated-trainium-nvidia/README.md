# Disaggregated inference: Trainium2 <-> NVIDIA GPU over NIXL / EFA

Prefill runs on one vendor's accelerator and decode on the other's. The KV cache moves **device memory to device memory**, from Trainium2 HBM to NVIDIA HBM, using [NIXL](https://github.com/ai-dynamo/nixl)'s LIBFABRIC backend over EFA RDMA. There is no host-memory staging.

It works in both directions:

- **trn2 prefill -> GPU decode**
- **GPU prefill -> trn2 decode**
- **mixed prefill pools**, for example {trn2, GPU} prefill -> GPU decode

**Software:**
- trn2 side: vllm-neuron 0.24.0.1.1.0 (Neuron SDK 2.32).
- GPU side: upstream vLLM 0.24.0.
- One small vLLM patch, applied on the side that runs decode.

Tested on trn2.48xlarge <-> p6-b200.48xlarge, both in one Availability Zone.

## Results

**Raw NIXL transfer** (`nixl_xfer/xvendor_xfer.py`): one Neuron core's HBM <-> one B200, single EFA rail (200 Gb/s).
- Neuron HBM registers as `FI_HMEM_NEURON`; CUDA HBM registers as `FI_HMEM_CUDA`.
- READ and WRITE both work in both directions, and every size was byte-verified.

| Size | GPU reads trn2 | GPU writes trn2 | trn2 reads GPU | trn2 writes GPU |
|---|---|---|---|---|
| 1 MB | 10.3 GB/s | 12.3 | 11.7 | 13.1 |
| 16 MB | 21.5 | 22.6 | 22.4 | 22.0 |
| 256 MB | **23.5** | **24.3** | **24.2** | **23.1** |

**End-to-end serving correctness** (`scripts/compare.py`):
- Llama-3.1-8B-Instruct, BF16, greedy, 21 prompts x 48 tokens.
- Each row compares DI output with a single-instance run on the device that does the **decode**. "Exact" means the tokens are identical for all 48 positions.
- The numerics ceiling below is how often a trn2-only run and a GPU-only run agree with each other. Kernels differ, and BF16 near-ties flip late in the output.

| Configuration | Exact match vs decode device | Ceiling (trn2-only vs GPU-only) |
|---|---|---|
| trn2 TP4 prefill -> GPU TP4 decode | 15 / 21 | 16 / 21 |
| GPU TP4 prefill -> trn2 TP4 decode | **19 / 21** | 16 / 21 |
| trn2 TP4 prefill -> GPU TP1 decode (heterogeneous TP) | 17 / 21 | 14 / 21 |
| GPU TP1 prefill -> trn2 TP4 decode (heterogeneous TP) | 18 / 21 | 14 / 21 |
| {trn2 TP4, GPU TP1} prefill pool -> GPU TP1 decode | 19 / 21 | 14 / 21 |

- The 609-token prompt matched exactly in every configuration.
- Per-token decode latency equals the decode device's own, and TTFT equals the prefill device's own plus the handoff.
- KV transfers took 1-3 ms for up to ~20 MB per request.

## Recipe

### 1. Infrastructure

- **Placement:** both instances in the **same AZ and VPC**. EFA traffic cannot cross AZs or VPCs. Compare AZ **IDs**, not names. Not every region offers trn2 and an EFA GPU type in the same AZ.
- **Security group:** one rule allowing all traffic to and from the group itself, for both **inbound and outbound**, plus SSH.
- **trn2.48xlarge:** launch with **all 16 EFA network cards**. trn2.3xlarge cannot register `FI_HMEM_NEURON`.
- **GPU host:** launch with all of its EFA cards, and use an AMI with GPUDirect RDMA (`efa_nv_peermem`), such as the Deep Learning Base OSS Nvidia Driver GPU AMI.
- **Network JSON:** `infra/make_efa_nics.py` writes the `--network-interfaces` JSON. A multi-NIC launch gets no public IP automatically, so attach an Elastic IP to card 0.
- **trn2 AMI:** the Neuron DLAMI 20260818 (SDK 2.32). If it is not published in your region and `copy-image` is denied, run `infra/ami_builder_userdata.sh` to rebuild an owned copy, then copy that.

### 2. Software

- **trn2:** use the DLAMI venv `/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0` as-is. It already ships vllm 0.24.0, vllm-neuron 0.24.0.1.1.0, nixl 1.3.2 and a `libcuda.so.1` stub.
- **GPU:** run `python3 -m venv ~/venv && . ~/venv/bin/activate && pip install vllm==0.24.0 "nixl[cu13]==1.3.2"`. Keep the NIXL version the same on both sides.
- **Patch:** apply it on both hosts. It only activates on the side that consumes the KV cache:

```bash
python patches/patch_nixl_cross_backend_kv_regions.py \
  <venv>/lib/python3.12/site-packages/vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py
```

### 3. Serve

`scripts/serve_trn2.sh` and `scripts/serve_gpu.sh` take environment variables: `ROLE=producer|consumer|none`, `MODEL`, `PORT`, `SIDE` (NIXL side-channel port), `TP`, `CORES` (trn2) / `GPU`, and `MML` (max model length).

Both scripts already set the required settings:

| Side | Setting |
|---|---|
| both | `NixlConnector`, `kv_buffer_device: "cuda"` (on Neuron this means Neuron HBM), `backends: ["LIBFABRIC"]` |
| both | `--block-size 32`, same model and dtype |
| both | `enforce_handshake_compat: false` (see below) |
| GPU | `VLLM_KV_CACHE_LAYOUT=HND` (Neuron is HND) and `EXTRA="--attention-backend FLASH_ATTN"` |
| trn2 | `FI_EFA_ENABLE_SHM_TRANSFER=0`, `NEURON_RT_MAP_HBM=1` |

Forward direction (trn2 prefill, GPU decode), Llama-3.1-8B TP4:

```bash
# trn2
ROLE=producer PORT=8100 SIDE=5559 CORES=0-3 TP=4 ENFORCE=false MML=4096 \
  MODEL=~/models/Llama-3.1-8B-Instruct ./serve_trn2.sh
# GPU
ROLE=consumer PORT=8200 SIDE=5659 GPU=0,1,2,3 TP=4 ENFORCE=false MML=4096 \
  MODEL=~/models/Llama-3.1-8B-Instruct EXTRA="--attention-backend FLASH_ATTN" ./serve_gpu.sh
# proxy (any host); use PRIVATE IPs
python toy_proxy_server.py --port 8000 \
  --prefiller-host <trn2 private IP> --prefiller-port 8100 \
  --decoder-host <GPU private IP> --decoder-port 8200
./q.sh 127.0.0.1:8000 "The capital of France is"
```

- **Reverse direction:** swap the roles (`ROLE=producer` on the GPU, `ROLE=consumer` on trn2) and the proxy hosts.
- **Mixed prefill pool:** see `scripts/proxy_xpyd.sh`.
- **Proxy hosts:** always pass private IPs, never `127.0.0.1`. The decoder dials the prefiller's host for the NIXL side channel.

## Why the patch is needed

**Different KV registration.** The two backends register the KV cache with NIXL differently:
- vllm-neuron's KV cache per layer is `(2, num_blocks, H, B, D)`. vLLM registers it as **two regions per layer**, one for K and one for V.
- CUDA FlashAttention/FlashInfer use blocks-first `(num_blocks, 2, H, B, D)`. vLLM registers that as **one region per layer**, with K and V as the two halves of each block.

**Same bytes.** Under HND, one block's K `(H, B, D)` and V `(H, B, D)` are byte-identical in both layouts; only the bookkeeping differs. Stock vLLM 0.24 still rejects the handshake with `Number of KV layers must match between prefill and decode`.

**What the patch changes.** It makes the consumer recognize both pairings:
- **split remote + blocks-first local:** GPU decoding from trn2.
- **blocks-first remote + split local:** trn2 decoding from GPU.

The consumer then builds remote descriptors in the same order as its local ones. Because HND keeps heads contiguous, the patch also supports heterogeneous TP by reading per-rank head slices. Same block size only; no MLA or Mamba.

**The compatibility hash.** It includes the attention backend name (`CUSTOM` on Neuron, `FLASH_ATTN` on CUDA), which is why both sides need `enforce_handshake_compat: false`.

**Upstream status.** vLLM's KV-cache layout was rewritten after 0.24 (vllm-project/vllm#51718, v0.28+). The patch targets the 0.24 code that vllm-neuron 0.24 pins, so revisit this once vllm-neuron moves to a newer vLLM.

## Known issues

1. **First request on a prefill server.** A vllm-neuron `kv_producer` server can fail its first request with `Detected recompile when torch.compile stance is 'fail_on_recompile'` (`input_ids` size 2048 vs 4). Restart the server.
2. **8B decode at TP=1 on trn2.** It fails to compile with `NCC_INKI016 Stack out of memory` in the nkilib MLP kernel. Use TP >= 2 for 8B decode on trn2; prefill at TP=1 works.
3. **Cascading error after a failed handshake.** vLLM 0.24 follows it with `IndexError: list index out of range` in `_handle_failed_transfer`. Look above that error for the real cause.
4. **Agent creation order (NIXL Python API).** Create the agent **after** the Neuron runtime has been initialized, for example by allocating one tensor first. Otherwise Neuron VRAM registration silently falls back to `FI_HMEM_SYSTEM`. vLLM already does this in the right order; `nixl_xfer/xvendor_xfer.py` shows it.
5. **Multiple processes on one trn2 host.** Each needs its own `NEURON_RT_VISIBLE_CORES` range.
6. **`fi_pingpong`.** It stops silently for messages of 16 KB and larger, even on loopback, so don't use it to measure bandwidth.

## Files

| Path | Contents |
|---|---|
| `scripts/serve_trn2.sh`, `scripts/serve_gpu.sh` | vLLM server launchers with the DI config |
| `scripts/gpu_launch.sh`, `scripts/proxy_launch.sh`, `scripts/proxy_xpyd.sh`, `scripts/stop_all.sh`, `scripts/q.sh` | background launch, proxy, stop and query helpers |
| `scripts/toy_proxy_server.py` | P/D proxy from vllm-neuron's examples (Apache-2.0, vLLM project) |
| `scripts/compare.py` | greedy token-match harness across endpoints |
| `patches/patch_nixl_cross_backend_kv_regions.py` | the vLLM 0.24 NixlConnector patch |
| `nixl_xfer/xvendor_xfer.py` | standalone NIXL Neuron <-> CUDA transfer test and bandwidth measurement |
| `infra/make_efa_nics.py`, `infra/ami_builder_userdata.sh` | EFA launch JSON and DLAMI re-capture helper |
