"""
SageMaker <-> vLLM Neuron Entrypoint
=====================================
Bridges SageMaker's serving contract to vLLM's native OpenAI-compatible API.

SageMaker requires:
  GET  /ping         -> 200 when healthy
  POST /invocations  -> inference

vLLM provides:
  GET  /health                -> 200 when healthy
  POST /v1/chat/completions   -> OpenAI chat completions
  POST /v1/completions        -> OpenAI completions

This entrypoint starts vLLM serve as a subprocess and runs a lightweight
FastAPI proxy on port 8080 (SageMaker's expected port).

Configuration via environment variables:
  SM_MODEL_ID        - HuggingFace model ID or /opt/ml/model (default: /opt/ml/model)
  SM_TP_DEGREE       - Tensor parallel degree (default: 2)
  SM_MAX_MODEL_LEN   - Max sequence length (default: 4096)
  SM_MAX_NUM_SEQS    - Max concurrent sequences (default: 4)
  SM_NEURON_CONFIG   - JSON string for override_neuron_config (optional)
  SM_VLLM_ARGS       - Additional vllm serve CLI arguments (optional)
  SM_TRUST_REMOTE     - Trust remote code (default: true)
"""

import glob
import json
import logging
import os
import re
import shlex
import subprocess
import sys
import time

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("neuron-vllm-sagemaker")

# -----------------------------------------------------------------------
# Configuration from environment
# -----------------------------------------------------------------------
MODEL_ID = os.environ.get("SM_MODEL_ID", "/opt/ml/model")
TP_DEGREE = int(os.environ.get("SM_TP_DEGREE", "2"))
MAX_MODEL_LEN = int(os.environ.get("SM_MAX_MODEL_LEN", "4096"))
MAX_NUM_SEQS = int(os.environ.get("SM_MAX_NUM_SEQS", "4"))
NEURON_CONFIG = os.environ.get("SM_NEURON_CONFIG", "")
VLLM_EXTRA_ARGS = os.environ.get("SM_VLLM_ARGS", "")
TRUST_REMOTE = os.environ.get("SM_TRUST_REMOTE", "true").lower() == "true"

# vLLM listens on this internal port; we proxy from 8080
VLLM_PORT = 8000
SAGEMAKER_PORT = int(os.environ.get("SAGEMAKER_BIND_TO_PORT", "8080"))


# -----------------------------------------------------------------------
# Model path resolution
# -----------------------------------------------------------------------
def resolve_model_path():
    """Resolve the model path, handling SageMaker's read-only /opt/ml/model."""
    model_path = MODEL_ID

    # If using /opt/ml/model (SageMaker default), check if it has model files
    if model_path == "/opt/ml/model":
        if not os.path.exists(os.path.join(model_path, "config.json")):
            logger.warning(
                "No config.json in /opt/ml/model. "
                "Either set SM_MODEL_ID to a HuggingFace model ID or "
                "provide model weights via SageMaker model_data."
            )
            return model_path

        # SageMaker mounts /opt/ml/model as read-only. NxDI needs to write
        # neuron-compiled-artifacts/ during compilation. Create writable overlay.
        writable_path = "/tmp/model"
        if not os.path.exists(writable_path):
            os.makedirs(writable_path, exist_ok=True)
            for entry in os.listdir(model_path):
                src = os.path.join(model_path, entry)
                dst = os.path.join(writable_path, entry)
                os.symlink(src, dst)
            os.makedirs(
                os.path.join(writable_path, "neuron-compiled-artifacts"),
                exist_ok=True,
            )
            logger.info("Created writable model overlay at %s", writable_path)
        return writable_path

    return model_path


# -----------------------------------------------------------------------
# NxDI source patches (applied before launching vLLM)
# -----------------------------------------------------------------------
def apply_nxdi_patches():
    """
    Apply on-disk patches to NxDI source files to fix known issues.

    These patches are applied to the installed Python files so they take
    effect in the spawned vLLM subprocess (which cannot inherit in-process
    monkey-patches).

    Patch 1: tie_word_embeddings for Qwen3-VL
        NeuronQwen3VLForCausalLM is missing update_state_dict_for_tied_weights,
        causing NotImplementedError when tie_word_embeddings=true (Qwen3-VL).
        The text submodel class has the correct implementation; we add it to
        the top-level VLM class.
    """
    # Find modeling_qwen3_vl.py in site-packages
    patterns = [
        "*/neuronx_distributed_inference/models/qwen3_vl/modeling_qwen3_vl.py",
    ]
    patched = False
    for pattern in patterns:
        matches = glob.glob(
            os.path.join(sys.prefix, "lib", "**", pattern), recursive=True
        )
        for filepath in matches:
            try:
                with open(filepath, "r") as f:
                    content = f.read()

                # Check if patch is needed (class exists but method is missing)
                if (
                    "NeuronQwen3VLForCausalLM" in content
                    and "class NeuronQwen3VLForCausalLM" in content
                ):
                    # Check if method already exists on the VLM class
                    # (not just on the text submodel)
                    vlm_class_match = re.search(
                        r"class NeuronQwen3VLForCausalLM\([^)]*\):",
                        content,
                    )
                    if vlm_class_match:
                        # Find the class body after the class definition
                        class_start = vlm_class_match.end()
                        # Check if update_state_dict_for_tied_weights exists
                        # after this class definition (within the class body)
                        remaining = content[class_start:]
                        next_class = re.search(r"\nclass ", remaining)
                        class_body = (
                            remaining[: next_class.start()] if next_class else remaining
                        )

                        if "update_state_dict_for_tied_weights" not in class_body:
                            # Need to add the method. Insert it after the class
                            # definition line.
                            patch = '''
    @classmethod
    def update_state_dict_for_tied_weights(cls, state_dict):
        """Patch: copy embed_tokens weights to lm_head for tied embeddings."""
        if "embed_tokens.weight" in state_dict and "lm_head.weight" not in state_dict:
            state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()
'''
                            # Insert after the class definition line
                            insert_pos = class_start
                            # Find the first method or attribute in the class
                            first_def = re.search(r"\n    (def |@)", class_body)
                            if first_def:
                                insert_pos = class_start + first_def.start()
                            else:
                                # Just insert after class docstring or class line
                                insert_pos = class_start

                            new_content = (
                                content[:insert_pos] + patch + content[insert_pos:]
                            )
                            with open(filepath, "w") as f:
                                f.write(new_content)
                            logger.info("Patched tie_word_embeddings in %s", filepath)
                            patched = True
                        else:
                            logger.info(
                                "tie_word_embeddings patch already present in %s",
                                filepath,
                            )
            except Exception as e:
                logger.warning("Failed to patch %s: %s", filepath, e)

    if not patched:
        logger.info("No tie_word_embeddings patch needed (model may not be Qwen3-VL)")

    # ------------------------------------------------------------------
    # Patch 2: Vision encoder convert_hf_to_neuron_state_dict — fused QKV split
    #
    # NxDI 0.8.x has a bug: the vision encoder's convert_hf_to_neuron_state_dict
    # maps HF's fused `.attn.qkv.` → `.attn.qkv_proj.Wqkv.` (fused format),
    # but NeuronQwen3VLVisionAttention sets fused_qkv=False (default).
    # The preshard_hook then expects separate q_proj/k_proj/v_proj keys and
    # crashes with KeyError: 'blocks.0.attn.qkv_proj.q_proj.weight'.
    #
    # Fix: Replace convert_hf_to_neuron_state_dict to split the fused QKV
    # weight/bias into separate q_proj, k_proj, v_proj entries.
    # ------------------------------------------------------------------
    vision_patterns = [
        "*/neuronx_distributed_inference/models/qwen3_vl/modeling_qwen3_vl_vision.py",
    ]
    vision_patched = False
    for pattern in vision_patterns:
        matches = glob.glob(
            os.path.join(sys.prefix, "lib", "**", pattern), recursive=True
        )
        for filepath in matches:
            try:
                with open(filepath, "r") as f:
                    content = f.read()

                # Check if the buggy mapping exists
                if ".attn.qkv_proj.Wqkv." in content:
                    # Replace the entire convert_hf_to_neuron_state_dict method
                    old_method = """    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: dict, inference_config: InferenceConfig
    ) -> dict:
        new_state_dict = {}
        for key, value in state_dict.items():
            if "visual." in key:
                key = key.replace("visual.", "")
                if ".attn.qkv." in key:
                    key = key.replace(".attn.qkv.", ".attn.qkv_proj.Wqkv.")
                elif ".attn.proj." in key:
                    key = key.replace(".attn.proj.", ".attn.o_proj.")
            new_state_dict[key] = value.clone().detach().contiguous()

        del state_dict
        return new_state_dict"""

                    new_method = '''    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: dict, inference_config: InferenceConfig
    ) -> dict:
        """Patched: split fused QKV into separate q/k/v for fused_qkv=False."""
        new_state_dict = {}
        for key, value in state_dict.items():
            if "visual." in key:
                key = key.replace("visual.", "")
                if ".attn.qkv." in key:
                    # Split fused QKV weight/bias into separate q/k/v projections.
                    # Qwen3-VL vision: num_heads=16, hidden_size=1024, head_dim=64
                    # fused shape: [3*hidden_size, hidden_size] for weight,
                    #              [3*hidden_size] for bias
                    import torch as _torch
                    hidden_size = value.shape[-1] if value.dim() == 2 else value.shape[0] // 3
                    chunks = _torch.chunk(value.clone().detach().contiguous(), 3, dim=0)
                    base = key.replace(".attn.qkv.", ".attn.qkv_proj.")
                    suffix = key.split(".attn.qkv.")[-1]  # "weight" or "bias"
                    new_state_dict[base.replace(suffix, f"q_proj.{suffix}")] = chunks[0]
                    new_state_dict[base.replace(suffix, f"k_proj.{suffix}")] = chunks[1]
                    new_state_dict[base.replace(suffix, f"v_proj.{suffix}")] = chunks[2]
                    continue
                elif ".attn.proj." in key:
                    key = key.replace(".attn.proj.", ".attn.o_proj.")
            new_state_dict[key] = value.clone().detach().contiguous()

        del state_dict
        return new_state_dict'''

                    if old_method in content:
                        content = content.replace(old_method, new_method)
                        with open(filepath, "w") as f:
                            f.write(content)
                        logger.info("Patched vision QKV split in %s", filepath)
                        vision_patched = True
                    else:
                        logger.warning(
                            "Vision QKV patch: found Wqkv marker but method text mismatch in %s",
                            filepath,
                        )
                else:
                    logger.info(
                        "Vision QKV patch not needed in %s (already fixed)", filepath
                    )
            except Exception as e:
                logger.warning("Failed to patch vision %s: %s", filepath, e)

    if not vision_patched:
        logger.info("No vision QKV split patch needed")


# -----------------------------------------------------------------------
# vLLM subprocess management
# -----------------------------------------------------------------------
vllm_process = None


def start_vllm():
    """Start vLLM serve as a subprocess."""
    global vllm_process

    model_path = resolve_model_path()

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_path,
        "--tensor-parallel-size",
        str(TP_DEGREE),
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--max-num-seqs",
        str(MAX_NUM_SEQS),
        "--port",
        str(VLLM_PORT),
        "--no-enable-prefix-caching",
    ]

    if TRUST_REMOTE:
        cmd.append("--trust-remote-code")

    # Build additional_config with neuron overrides
    additional_config = {}
    if NEURON_CONFIG:
        try:
            neuron_overrides = json.loads(NEURON_CONFIG)
            additional_config["override_neuron_config"] = neuron_overrides
        except json.JSONDecodeError:
            logger.error("SM_NEURON_CONFIG is not valid JSON: %s", NEURON_CONFIG)

    if additional_config:
        cmd.extend(["--additional-config", json.dumps(additional_config)])

    # Append any extra CLI arguments
    if VLLM_EXTRA_ARGS:
        cmd.extend(shlex.split(VLLM_EXTRA_ARGS))

    logger.info("Starting vLLM: %s", " ".join(cmd))
    vllm_process = subprocess.Popen(
        cmd,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    logger.info("vLLM started with PID %d", vllm_process.pid)


def check_vllm_health():
    """Check if vLLM is healthy."""
    try:
        resp = httpx.get(f"http://localhost:{VLLM_PORT}/health", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


# -----------------------------------------------------------------------
# FastAPI proxy application
# -----------------------------------------------------------------------
app = FastAPI(title="Neuron vLLM SageMaker Proxy")


@app.on_event("startup")
async def startup():
    start_vllm()


@app.get("/ping")
async def health_check():
    """SageMaker health check endpoint."""
    healthy = check_vllm_health()

    # Also check the subprocess is still running
    if vllm_process and vllm_process.poll() is not None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "message": f"vLLM process exited with code {vllm_process.returncode}",
            },
        )

    if healthy:
        return JSONResponse(status_code=200, content={"status": "healthy"})
    else:
        # Return 200 during startup (SageMaker will retry via health check timeout)
        # Return 503 only if vLLM process has died
        return JSONResponse(status_code=200, content={"status": "loading"})


@app.post("/invocations")
async def invoke(request: Request):
    """
    SageMaker inference endpoint.

    Accepts OpenAI chat completions format and proxies to vLLM.
    The request body is forwarded as-is to /v1/chat/completions.

    Expected format:
    {
        "model": "model-name",  // optional, will use deployed model
        "messages": [...],
        "max_tokens": 256,
        "temperature": 0.0,
        ...
    }
    """
    if not check_vllm_health():
        return JSONResponse(
            status_code=503,
            content={"error": "Model not ready. vLLM is still loading."},
        )

    try:
        body = await request.body()
        headers = {"Content-Type": "application/json"}

        # Forward to vLLM's chat completions endpoint
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(
                f"http://localhost:{VLLM_PORT}/v1/chat/completions",
                content=body,
                headers=headers,
            )

        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type="application/json",
        )

    except httpx.TimeoutException:
        return JSONResponse(
            status_code=504,
            content={"error": "vLLM request timed out (300s)"},
        )
    except Exception as e:
        logger.error("Proxy error: %s", e)
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
        )


@app.post("/invocations/completions")
async def invoke_completions(request: Request):
    """Alternative endpoint for text completions (non-chat)."""
    if not check_vllm_health():
        return JSONResponse(
            status_code=503,
            content={"error": "Model not ready. vLLM is still loading."},
        )

    try:
        body = await request.body()
        headers = {"Content-Type": "application/json"}

        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(
                f"http://localhost:{VLLM_PORT}/v1/completions",
                content=body,
                headers=headers,
            )

        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type="application/json",
        )

    except Exception as e:
        logger.error("Proxy error: %s", e)
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/v1/models")
async def list_models():
    """Proxy model list from vLLM."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"http://localhost:{VLLM_PORT}/v1/models")
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type="application/json",
        )
    except Exception:
        return JSONResponse(status_code=503, content={"error": "vLLM not ready"})


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Neuron vLLM SageMaker Container")
    logger.info("=" * 60)
    logger.info("Model:          %s", MODEL_ID)
    logger.info("TP degree:      %d", TP_DEGREE)
    logger.info("Max model len:  %d", MAX_MODEL_LEN)
    logger.info("Max num seqs:   %d", MAX_NUM_SEQS)
    logger.info("Neuron config:  %s", NEURON_CONFIG or "(default)")
    logger.info("vLLM extra:     %s", VLLM_EXTRA_ARGS or "(none)")
    logger.info("SageMaker port: %d", SAGEMAKER_PORT)
    logger.info("vLLM port:      %d", VLLM_PORT)
    logger.info("=" * 60)

    # Apply NxDI patches before starting vLLM
    apply_nxdi_patches()

    # Start the proxy server
    uvicorn.run(app, host="0.0.0.0", port=SAGEMAKER_PORT, log_level="info")
