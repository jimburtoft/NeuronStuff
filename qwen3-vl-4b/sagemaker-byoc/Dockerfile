# Qwen3-VL-4B Neuron BYOC Container for SageMaker - v9
# ============================================================
# Compile-from-scratch on SageMaker (no pre-compiled artifacts).
#
# Architecture:
#   1. Base: PyTorch 2.9 + Neuron SDK 2.28 (DLC)
#   2. Install PyPI vLLM 0.13.0 (for dependencies), then replace the
#      vllm/ package with the DLAMI's byte-identical version.
#   3. Installs vllm_neuron 0.4.1 plugin (extracted from DLAMI)
#   4. NO compile cache -- compiles fresh on the SageMaker instance
#      to generate NEFFs compatible with SageMaker's host driver
#   5. Model weights loaded from S3 via SageMaker model_data
#
# Why no compile cache? SageMaker's Neuron host driver (2.10.11.0)
# is older than the DLAMI's driver. NEFFs compiled on bare-metal
# contain Extended ISA instructions that the SageMaker driver cannot
# execute. Compiling from scratch on SageMaker generates compatible
# NEFFs. Compilation takes ~10 minutes on inf2.8xlarge.

ARG REGION=us-east-1
FROM 763104351884.dkr.ecr.${REGION}.amazonaws.com/pytorch-inference-neuronx:2.9.0-neuronx-py312-sdk2.28.0-ubuntu24.04

# SageMaker environment variables
ENV MODEL_CACHE_DIR=/opt/ml/model
ENV TRANSFORMERS_CACHE=/tmp/transformers_cache
ENV HF_HOME=/tmp/hf_home
ENV NEURON_RT_NUM_CORES=2
ENV VLLM_NEURON_FRAMEWORK=neuronx-distributed-inference
ENV NEURON_COMPILE_CACHE_URL=/var/tmp/neuron-compile-cache
# Disable on-device sampling to avoid NKI ExtISA kernels
# SageMaker's host driver (2.10.11.0) cannot load ExtISA NEFFs
ENV NEURON_ON_DEVICE_SAMPLING_DISABLED=1

# curl is already in the base DLC image
# RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# ============================================================
# Step 1: Install PyPI vLLM 0.13.0 for all dependencies,
#          with Neuron package backup/restore
# ============================================================
RUN SITE_PKG=$(python3 -c "import site; print(site.getsitepackages()[0])") && \
    mkdir -p /tmp/neuron-wheels && \
    # Save Neuron packages (custom build tags, can't reinstall from PyPI)
    for pkg in torch_neuronx neuronx_cc neuronx_distributed neuronx_distributed_inference; do \
      DIST_DIR=$(find "$SITE_PKG" -maxdepth 1 -name "${pkg}-*.dist-info" -type d | head -1); \
      if [ -n "$DIST_DIR" ]; then \
        PKG_NAME=$(basename "$DIST_DIR" | sed 's/.dist-info$//'); \
        echo "Saving $PKG_NAME"; \
        cp -r "$DIST_DIR" /tmp/neuron-wheels/; \
        PKG_DIR=$(find "$SITE_PKG" -maxdepth 1 -name "${pkg}" -type d | head -1); \
        [ -n "$PKG_DIR" ] && cp -r "$PKG_DIR" /tmp/neuron-wheels/; \
      fi; \
    done && \
    # Install vLLM with all dependencies
    pip install --no-cache-dir vllm==0.13.0 qwen-vl-utils 'uvicorn[standard]' && \
    # Restore Neuron packages
    for item in /tmp/neuron-wheels/*/; do \
      PKG_BASE=$(basename "$item"); \
      TARGET="$SITE_PKG/$PKG_BASE"; \
      rm -rf "$TARGET" 2>/dev/null; \
      cp -r "$item" "$TARGET"; \
    done && \
    rm -rf /tmp/neuron-wheels && \
    python3 -c "import torch_neuronx; print('torch-neuronx OK')"

# ============================================================
# Step 2: Replace PyPI vLLM with DLAMI's exact vLLM package
# ============================================================
COPY artifacts/vllm-dlami-package.tar.gz /tmp/vllm-dlami-package.tar.gz

RUN SITE_PKG=$(python3 -c "import site; print(site.getsitepackages()[0])") && \
    # Remove PyPI vLLM package directory and dist-info
    rm -rf "$SITE_PKG/vllm" "$SITE_PKG/vllm-"*.dist-info && \
    # Extract DLAMI's vLLM (contains vllm/ and vllm-0.13.0.dist-info/)
    tar -xzf /tmp/vllm-dlami-package.tar.gz -C "$SITE_PKG/" && \
    rm /tmp/vllm-dlami-package.tar.gz && \
    echo "DLAMI vLLM installed:" && \
    python3 -c "import vllm; print(f'vLLM {vllm.__version__}')" && \
    python3 -c "import neuronx_distributed_inference; print('NxDI OK')"

# ============================================================
# Step 3: Install vllm_neuron plugin (extracted from DLAMI)
# ============================================================
COPY artifacts/vllm-neuron-code.tar.gz /tmp/vllm-neuron-code.tar.gz

RUN tar -xzf /tmp/vllm-neuron-code.tar.gz -C / && \
    rm /tmp/vllm-neuron-code.tar.gz && \
    SITE_PKG=$(python3 -c "import site; print(site.getsitepackages()[0])") && \
    echo "/vllm" > "$SITE_PKG/vllm_neuron.pth" && \
    mkdir -p "$SITE_PKG/vllm_neuron-0.4.1.dist-info" && \
    printf 'Metadata-Version: 2.4\nName: vllm-neuron\nVersion: 0.4.1\nSummary: vLLM Neuron backend plugin\n' > "$SITE_PKG/vllm_neuron-0.4.1.dist-info/METADATA" && \
    printf '[vllm.platform_plugins]\nneuron = vllm_neuron:register\n' > "$SITE_PKG/vllm_neuron-0.4.1.dist-info/entry_points.txt" && \
    printf 'vllm_neuron\n' > "$SITE_PKG/vllm_neuron-0.4.1.dist-info/top_level.txt" && \
    printf 'Wheel-Version: 1.0\nGenerator: manual\nRoot-Is-Purelib: true\nTag: py3-none-any\n' > "$SITE_PKG/vllm_neuron-0.4.1.dist-info/WHEEL" && \
    printf 'pip\n' > "$SITE_PKG/vllm_neuron-0.4.1.dist-info/INSTALLER" && \
    printf 'vllm_neuron-0.4.1.dist-info/METADATA,\nvllm_neuron-0.4.1.dist-info/WHEEL,\nvllm_neuron-0.4.1.dist-info/entry_points.txt,\nvllm_neuron-0.4.1.dist-info/top_level.txt,\nvllm_neuron-0.4.1.dist-info/INSTALLER,\nvllm_neuron-0.4.1.dist-info/RECORD,,\n' > "$SITE_PKG/vllm_neuron-0.4.1.dist-info/RECORD" && \
    python3 -c "import vllm_neuron; print('vllm_neuron plugin OK')"

# ============================================================
# Step 3b: Patch vllm_neuron Sampler import bug
# The DLAMI's vllm_neuron imports 'sampler' module instead of
# 'Sampler' class, causing TypeError when
# NEURON_ON_DEVICE_SAMPLING_DISABLED=1
# ============================================================
RUN sed -i 's|from vllm.v1.sample import sampler as Sampler|from vllm.v1.sample.sampler import Sampler|' \
    /vllm/vllm_neuron/worker/neuronx_distributed_model_loader.py && \
    echo "Patched Sampler import in vllm_neuron"

# ============================================================
# Step 4: Ensure NO compile cache from base image
# ============================================================
RUN rm -rf /var/tmp/neuron-compile-cache/* && \
    mkdir -p /var/tmp/neuron-compile-cache && \
    echo "Compile cache cleared (will compile fresh on SageMaker)"

# Create necessary directories
RUN mkdir -p /opt/ml/model /opt/ml/code /tmp/transformers_cache /tmp/hf_home

# Copy serving code
COPY code/model.py /opt/ml/code/model.py
COPY code/serving.properties /opt/ml/code/serving.properties

WORKDIR /opt/ml/code

# Health check - longer start period for from-scratch compilation (~10 min)
HEALTHCHECK --interval=30s --timeout=30s --start-period=900s --retries=3 \
    CMD curl -f http://localhost:8080/ping || exit 1

EXPOSE 8080

# SageMaker calls: docker run <image> serve
RUN printf '#!/bin/bash\n\
set -e\n\
echo "Starting Qwen3-VL Neuron BYOC container (v9 - compile from scratch, no ExtISA)..."\n\
echo "Contents of /opt/ml/model/"\n\
ls -la /opt/ml/model/ 2>/dev/null || echo "Empty /opt/ml/model/"\n\
\n\
# CRITICAL: Clear any residual compile cache to ensure fresh compilation.\n\
# SageMaker host driver (2.10.11.0) is older than DLAMI -- NEFFs from\n\
# the DLAMI compile cache contain ExtISA instructions that fail on\n\
# SageMaker. Must compile from scratch for compatible NEFFs.\n\
rm -rf /var/tmp/neuron-compile-cache/*\n\
echo "Compile cache cleared for fresh compilation"\n\
\n\
if [ -f /opt/ml/model/model.py ]; then\n\
    echo "Using model.py from /opt/ml/model/"\n\
    cp /opt/ml/model/model.py /opt/ml/code/model.py\n\
fi\n\
if [ -f /opt/ml/model/serving.properties ]; then\n\
    echo "Using serving.properties from /opt/ml/model/"\n\
    cp /opt/ml/model/serving.properties /opt/ml/code/serving.properties\n\
fi\n\
\n\
echo "Starting model server..."\n\
exec python3 /opt/ml/code/model.py\n\
' > /usr/local/bin/serve && \
    chmod +x /usr/local/bin/serve

ENTRYPOINT ["/usr/local/bin/serve"]
CMD ["serve"]
