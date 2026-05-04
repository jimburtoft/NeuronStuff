# General-Purpose vLLM + Neuron SageMaker Container
# ==================================================
# Deploys any vllm-neuron supported model on SageMaker Neuron instances.
#
# Uses the pre-built vLLM Neuron DLC as base (has vLLM + vllm_neuron +
# NxDI pre-installed at tested versions). Adds SageMaker proxy on top.
#
# Supported models: Llama 2/3/4, Qwen 2.5/3/3-VL, Pixtral, and any model
# listed in vllm-neuron's NEURON_SUPPORTED_MODELS.
#
# Supported instances: ml.inf2.*, ml.trn1.* (SDK 2.28), ml.trn2.* (SDK 2.29)
#
# Configuration is passed entirely via environment variables:
#   SM_MODEL_ID        - HuggingFace model ID or local path (required)
#   SM_TP_DEGREE       - Tensor parallel degree (default: auto-detect)
#   SM_MAX_MODEL_LEN   - Max sequence length (default: 4096)
#   SM_MAX_NUM_SEQS    - Max concurrent sequences (default: 4)
#   SM_NEURON_CONFIG    - JSON override_neuron_config (optional)
#   SM_VLLM_ARGS       - Additional vllm serve arguments (optional)
#
# Build:
#   docker build --build-arg REGION=us-west-2 -t neuron-vllm-sagemaker .
#
# Version matrix:
#   SDK 2.28: vLLM 0.13.0, NxDI 0.8.x, vllm-neuron 0.4.1 (inf2/trn1/trn2)
#   SDK 2.29: vLLM 0.16.0, NxDI 0.9.x, vllm-neuron 0.5.0 (trn2 only)

# ============================================================
# Build arguments
# ============================================================
ARG REGION=us-east-1

# Pre-built vLLM Neuron DLC (has vLLM + vllm_neuron + NxDI pre-installed)
# SDK 2.28: 0.13.0
# SDK 2.29: 0.16.0
ARG BASE_IMAGE=public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.13.0-neuronx-py312-sdk2.28.0-ubuntu24.04

FROM ${BASE_IMAGE}

# ============================================================
# Environment defaults (can be overridden at deploy time)
# ============================================================
ENV VLLM_NEURON_FRAMEWORK=neuronx-distributed-inference
ENV NEURON_COMPILE_CACHE_URL=/var/tmp/neuron-compile-cache
ENV SAGEMAKER_BIND_TO_PORT=8080
# Disable on-device sampling to avoid NKI cumsum kernels that generate
# ExtISA instructions incompatible with inf2 (PSUM constraint violation).
# Safe for all instances; sampling falls back to CPU-side top-k/top-p.
ENV NEURON_ON_DEVICE_SAMPLING_DISABLED=1

# SageMaker requires this label for BYOC containers
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true

# ============================================================
# Patch vllm_neuron Sampler import bug
# vllm_neuron 0.4.1 imports the sampler MODULE instead of the
# Sampler CLASS, causing TypeError when on-device sampling is
# disabled. Fix: import the class directly.
# ============================================================
RUN sed -i 's|from vllm.v1.sample import sampler as Sampler|from vllm.v1.sample.sampler import Sampler|' \
    /opt/vllm/vllm_neuron/worker/neuronx_distributed_model_loader.py && \
    echo "Patched Sampler import in vllm_neuron"

# ============================================================
# Patch NxDI deepstack_vision_embeds forward mismatch
#
# NxDI 0.8.x has two bugs that drop the 25th argument
# (deepstack_vision_embeds) needed by Qwen3-VL models:
#
# Bug 1: ImageToTextModelWrapper._forward_with_pad() uses
#   args[5:24] (24 total args) but Qwen3-VL compiles with
#   25 args. Fix: args[5:24] → args[5:]
#
# Bug 2: ModelWrapper.pad_inputs() CTE vision re-padding uses
#   args[:22] + padded_vision, dropping args[24:].
#   Fix: preserve remaining args after the padded vision.
# ============================================================
RUN SITE_PKG=$(python3 -c "import site; print(site.getsitepackages()[0])") && \
    # Bug 1: _forward_with_pad args[5:24] → args[5:]
    sed -i 's|for i, arg in enumerate(args\[5:24\]):|for i, arg in enumerate(args[5:]):|' \
        "$SITE_PKG/neuronx_distributed_inference/models/image_to_text_model_wrapper.py" && \
    # Bug 2: pad_inputs CTE vision re-padding - preserve deepstack args
    sed -i 's|args = (\*args\[:22\], \*padded_args)|args = (*args[:22], *padded_args, *args[24:])|' \
        "$SITE_PKG/neuronx_distributed_inference/models/model_wrapper.py" && \
    echo "Patched NxDI deepstack_vision_embeds forward mismatch"

# ============================================================
# Install SageMaker proxy dependencies
# (FastAPI + uvicorn for the /ping + /invocations bridge)
# ============================================================
RUN pip install --no-cache-dir qwen-vl-utils pillow httpx fastapi "uvicorn[standard]" && \
    rm -rf ~/.cache/pip/*

# ============================================================
# SageMaker entrypoint
#
# Bridges SageMaker's /ping + /invocations contract to
# vLLM's native /health + /v1/chat/completions endpoints.
# ============================================================
COPY entrypoint.py /opt/ml/code/entrypoint.py

# Create necessary directories
RUN mkdir -p /opt/ml/model /var/tmp/neuron-compile-cache

EXPOSE 8080

ENTRYPOINT ["python3", "/opt/ml/code/entrypoint.py"]
CMD ["serve"]
