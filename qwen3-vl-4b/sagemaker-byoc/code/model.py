"""
Qwen3-VL-4B Neuron BYOC Model Server for SageMaker (v8 - compile from scratch)
================================================================================
FastAPI server that loads Qwen3-VL-4B-Instruct via the DLAMI's vLLM
with on-device compilation (no pre-compiled artifacts).

Uses vLLM 0.13.0 with the vllm_neuron plugin, which supports
Qwen3VLForConditionalGeneration natively via NxDI.

Compilation happens from scratch on the SageMaker inf2 instance,
generating NEFFs compatible with SageMaker's host driver (2.10.11.0).
This takes ~10 minutes on inf2.8xlarge.

Endpoints:
  GET  /ping          -> Health check (SageMaker)
  POST /invocations   -> Inference (SageMaker, OpenAI chat format)

Configuration:
  Model weights: /opt/ml/model/ (from SageMaker model_data S3 path)
  Compile cache: /var/tmp/neuron-compile-cache/ (empty, filled during compilation)
"""

import os
import sys
import time
import json
import base64
import logging
import traceback
from io import BytesIO
from pathlib import Path
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("qwen3vl-neuron")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Model settings
TP_DEGREE = 2  # inf2.8xlarge has 2 NeuronCores
SEQ_LEN = 4096  # Max sequence length
MAX_NEW_TOKENS = 256  # Default max output tokens

# Neuron config (matches bare-metal compile settings)
# inf2: all ISA kernels OFF, LNC=1
TEXT_NEURON_CONFIG = {
    "batch_size": 1,
    "ctx_batch_size": 1,
    "tkg_batch_size": 1,
    "seq_len": SEQ_LEN,
    "max_context_length": SEQ_LEN,
    "torch_dtype": "bfloat16",
    "tp_degree": TP_DEGREE,
    "world_size": TP_DEGREE,
    "enable_bucketing": True,
    "context_encoding_buckets": [512, 1024, 2048, 4096],
    "token_generation_buckets": [512, 1024, 2048, 4096],
    "fused_qkv": True,
    "qkv_kernel_enabled": False,
    "mlp_kernel_enabled": False,
    "attn_kernel_enabled": False,
    "logical_neuron_cores": 1,
    "cc_pipeline_tiling_factor": 1,
    "rpl_reduce_dtype": "bfloat16",
    "attention_dtype": "bfloat16",
    "cast_type": "as-declared",
}

VISION_NEURON_CONFIG = {
    "batch_size": 1,
    "seq_len": 4096,
    "max_context_length": 4096,
    "enable_bucketing": True,
    "buckets": [1024, 4096],
    "world_size": TP_DEGREE,
    "tp_degree": TP_DEGREE,
    "torch_dtype": "bfloat16",
    "rpl_reduce_dtype": "bfloat16",
    "cast_type": "as-declared",
    "logical_neuron_cores": 1,
    "cc_pipeline_tiling_factor": 1,
    "fused_qkv": True,
    "attn_kernel_enabled": False,
    "mlp_kernel_enabled": False,
}


# ---------------------------------------------------------------------------
# Model loading via vLLM
# ---------------------------------------------------------------------------

llm = None
processor = None
model_loaded = False
load_error = None


def load_model():
    """Load Qwen3-VL via vLLM with from-scratch compilation."""
    global llm, processor, model_loaded, load_error

    try:
        logger.info("=" * 60)
        logger.info("Loading Qwen3-VL-4B via vLLM (compile from scratch)")
        logger.info("=" * 60)

        # SageMaker mounts /opt/ml/model as read-only. NxDI needs to write
        # neuron-compiled-artifacts/ under the model path during compilation.
        # Create a writable overlay with symlinks to the read-only files.
        model_path = "/tmp/model"
        if not os.path.exists(model_path):
            os.makedirs(model_path, exist_ok=True)
            for entry in os.listdir("/opt/ml/model"):
                src = os.path.join("/opt/ml/model", entry)
                dst = os.path.join(model_path, entry)
                # Skip model.py and serving.properties (code files, not model files)
                if entry in ("model.py", "serving.properties"):
                    continue
                os.symlink(src, dst)
            # Create writable neuron-compiled-artifacts directory
            os.makedirs(
                os.path.join(model_path, "neuron-compiled-artifacts"), exist_ok=True
            )
            logger.info(f"Created writable model overlay at {model_path}")

        # Verify model weights are present
        if not os.path.exists(os.path.join(model_path, "config.json")):
            raise FileNotFoundError(
                f"Model config.json not found in {model_path}. "
                "Ensure model_data points to the S3 path with model weights."
            )
        logger.info(f"Model path: {model_path}")
        logger.info(f"Model files: {sorted(os.listdir(model_path))}")

        # Verify compile cache is empty (should be cleared by serve script)
        cache_dir = "/var/tmp/neuron-compile-cache"
        import subprocess

        result = subprocess.run(
            ["du", "-sh", cache_dir], capture_output=True, text=True
        )
        logger.info(f"Compile cache: {result.stdout.strip()}")

        # Apply tie_word_embeddings patch for 4B model
        from neuronx_distributed_inference.models.qwen3_vl.modeling_qwen3_vl import (
            NeuronQwen3VLForCausalLM,
        )

        @staticmethod
        def update_state_dict_for_tied_weights(state_dict):
            if (
                "lm_head.weight" not in state_dict
                and "embed_tokens.weight" in state_dict
            ):
                logger.info(
                    "[PATCH] tie_word_embeddings: copying embed_tokens -> lm_head"
                )
                state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()

        NeuronQwen3VLForCausalLM.update_state_dict_for_tied_weights = (
            update_state_dict_for_tied_weights
        )
        logger.info("Applied tie_word_embeddings patch")

        # Set vLLM to use NxDI backend
        os.environ["VLLM_NEURON_FRAMEWORK"] = "neuronx-distributed-inference"

        from vllm import LLM, SamplingParams

        logger.info(f"Creating vLLM LLM with tp={TP_DEGREE}, seq_len={SEQ_LEN}...")
        logger.info(
            "This will compile from scratch -- expect ~10 minutes on inf2.8xlarge"
        )
        t0 = time.time()
        llm = LLM(
            model=model_path,
            tokenizer=model_path,
            trust_remote_code=True,
            dtype="bfloat16",
            tensor_parallel_size=TP_DEGREE,
            max_num_seqs=1,
            max_model_len=SEQ_LEN,
            additional_config=dict(
                override_neuron_config=dict(
                    text_neuron_config=TEXT_NEURON_CONFIG,
                    vision_neuron_config=VISION_NEURON_CONFIG,
                )
            ),
            limit_mm_per_prompt={"image": 1},
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
        )
        load_time = time.time() - t0
        logger.info(f"vLLM model loaded in {load_time:.1f}s")

        # Load processor for chat template
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        logger.info("Processor loaded")

        model_loaded = True
        logger.info("Model ready for inference")

    except Exception as e:
        load_error = str(e)
        logger.error(f"Model loading failed: {e}")
        logger.error(traceback.format_exc())


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def run_inference(prompt_text, image=None, max_tokens=256, temperature=0.0):
    """Run inference using vLLM."""
    from vllm import SamplingParams

    # Build messages in chat format
    if image is not None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
    else:
        messages = [{"role": "user", "content": prompt_text}]

    # Apply chat template
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Build vLLM input
    if image is not None:
        inputs = {"prompt": prompt, "multi_modal_data": {"image": [image]}}
    else:
        inputs = {"prompt": prompt}

    sampling_params = SamplingParams(
        top_k=1,
        max_tokens=max_tokens,
        temperature=max(temperature, 0.01) if temperature > 0 else 0.0,
    )

    t0 = time.time()
    outputs = llm.generate([inputs], sampling_params=sampling_params)
    latency = time.time() - t0

    text = outputs[0].outputs[0].text
    num_tokens = len(outputs[0].outputs[0].token_ids)
    tok_per_sec = num_tokens / latency if latency > 0 else 0

    return {
        "text": text,
        "num_tokens": num_tokens,
        "latency_s": latency,
        "tok_per_sec": tok_per_sec,
    }


def decode_image(image_data):
    """Decode image from base64 data URI or URL."""
    from PIL import Image
    import urllib.request

    if isinstance(image_data, str):
        if image_data.startswith("data:"):
            header, data = image_data.split(",", 1)
            img_bytes = base64.b64decode(data)
            return Image.open(BytesIO(img_bytes)).convert("RGB")
        elif image_data.startswith("http"):
            req = urllib.request.Request(
                image_data, headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                return Image.open(BytesIO(resp.read())).convert("RGB")
    raise ValueError(f"Unsupported image format: {str(image_data)[:100]}")


def resize_image(img, max_long_side=1024):
    """Resize image to fit within vision encoder patch budget."""
    w, h = img.size
    if max(w, h) > max_long_side:
        scale = max_long_side / max(w, h)
        new_w = int(w * scale) // 56 * 56
        new_h = int(h * scale) // 56 * 56
        new_w = max(new_w, 56)
        new_h = max(new_h, 56)
        from PIL import Image as PILImage

        img = img.resize((new_w, new_h), PILImage.LANCZOS)
    return img


def parse_request(data):
    """Parse SageMaker /invocations request into (prompt, image, max_tokens, temperature)."""
    messages = data.get("messages", [])
    max_tokens = data.get("max_tokens", MAX_NEW_TOKENS)
    temperature = data.get("temperature", 0.0)

    prompt_text = ""
    image = None

    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            prompt_text = content
        elif isinstance(content, list):
            for part in content:
                if part.get("type") == "text":
                    prompt_text = part.get("text", "")
                elif part.get("type") == "image_url":
                    image_url = part.get("image_url", {}).get("url", "")
                    if image_url:
                        image = decode_image(image_url)
                        image = resize_image(image)
                elif part.get("type") == "image":
                    img_data = part.get("url") or part.get("data") or part.get("image")
                    if img_data:
                        image = decode_image(img_data)
                        image = resize_image(image)

    return prompt_text, image, max_tokens, temperature


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    import threading

    t = threading.Thread(target=load_model, daemon=True)
    t.start()
    yield


app = FastAPI(title="Qwen3-VL-4B Neuron Server (vLLM)", lifespan=lifespan)


@app.get("/ping")
async def health_check():
    """SageMaker health check."""
    status = "healthy" if model_loaded else ("error" if load_error else "loading")
    return JSONResponse(
        status_code=200,
        content={
            "status": status,
            "model": "Qwen3-VL-4B-Instruct",
            "model_loaded": model_loaded,
            "backend": "vllm-neuron-compile-from-scratch",
            "error": load_error,
        },
    )


@app.post("/invocations")
async def invoke(request: Request):
    """SageMaker inference endpoint. Accepts OpenAI chat completions format."""
    if not model_loaded:
        return JSONResponse(
            status_code=503,
            content={
                "error": "Model not yet loaded",
                "status": "loading",
                "load_error": load_error,
            },
        )

    try:
        data = await request.json()
        prompt_text, image, max_tokens, temperature = parse_request(data)

        if not prompt_text and image is None:
            return JSONResponse(
                status_code=400,
                content={"error": "No prompt text or image found in request"},
            )

        result = run_inference(
            prompt_text, image=image, max_tokens=max_tokens, temperature=temperature
        )

        response = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "Qwen3-VL-4B-Instruct",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result["text"],
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "completion_tokens": result["num_tokens"],
                "total_tokens": result["num_tokens"],
            },
            "performance": {
                "latency_s": round(result["latency_s"], 3),
                "tok_per_sec": round(result["tok_per_sec"], 1),
            },
        }

        return JSONResponse(content=response)

    except Exception as e:
        logger.error(f"Inference error: {e}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "traceback": traceback.format_exc()},
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("SAGEMAKER_BIND_TO_PORT", 8080))
    host = os.environ.get("SAGEMAKER_BIND_TO_HOST", "0.0.0.0")
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
