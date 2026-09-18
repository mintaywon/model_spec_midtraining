"""OpenAI-compatible vLLM server on Modal for ODCV-Bench (and any other harness
that talks to an OpenAI endpoint), serving the base model plus the AFT-ladder
adapters as LoRA modules selectable by the `model` field.

    modal deploy tda/modal/odcv_app.py          # prints the URL
    curl $URL/v1/models -H "Authorization: Bearer $ODCV_API_KEY"

Separate app so `modal deploy` does not touch the eval/training app. Same
image pins as tda/modal/app.py (vLLM 0.8.5). Qwen2.5's chat template supports
tools; vLLM's `hermes` parser turns its <tool_call> blocks into OpenAI
tool_calls, which the ODCV executor requires.
"""

import os
import subprocess
from pathlib import Path

import modal

APP_NAME = "msm-tda-odcv"
app = modal.App(APP_NAME)

hf_cache = modal.Volume.from_name("msm-tda-hf-cache", create_if_missing=True)
results = modal.Volume.from_name("msm-tda-results", create_if_missing=True)
HF_CACHE_DIR = "/cache/huggingface"
RESULTS_DIR = "/results"
VOLUMES = {HF_CACHE_DIR: hf_cache, RESULTS_DIR: results}
hf_secret = modal.Secret.from_name("huggingface")

ODCV_API_KEY = "odcv-ladder-2026"       # shared secret for the endpoint
PORT = 8000
BASE_MODEL = "Qwen/Qwen2.5-32B-Instruct"

# served name -> adapter (results-volume path or HF repo). "base" = no adapter.
ADAPTERS = {
    "L0-ours": f"{RESULTS_DIR}/aft_ladder/aftonly_phil32b_L0_bs32_s42_20260918-0452",
    "L2": f"{RESULTS_DIR}/aft_ladder/aftonly_phil32b_L2_bs32_s42_20260917-0012",
    "Ref-ours": f"{RESULTS_DIR}/aft_phil32b_none_tb8192_s42",
    "Ref-rel": "chloeli/qwen-2.5-32b-philosophy-spec-msm-aft-no-cot",
    "L0-rel": "chloeli/qwen-2.5-32b-philosophy-spec-aft-no-cot",
}

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.6.0", "transformers==4.51.3", "peft==0.15.2",
        "huggingface_hub[hf_transfer]==0.30.2", "vllm==0.8.5",
    )
    .env({"HF_HOME": HF_CACHE_DIR, "HF_HUB_ENABLE_HF_TRANSFER": "1",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
)


@app.function(image=image, gpu="H100:2", volumes=VOLUMES, secrets=[hf_secret],
              timeout=6 * 3600, scaledown_window=900)
@modal.concurrent(max_inputs=128)
@modal.web_server(port=PORT, startup_timeout=1800)
def serve():
    from huggingface_hub import snapshot_download

    results.reload()
    modules = []
    for name, src in ADAPTERS.items():
        path = src if src.startswith("/") else snapshot_download(src, token=os.environ.get("HF_TOKEN"))
        assert Path(path, "adapter_config.json").exists(), f"{name}: no adapter at {path}"
        modules.append(f"{name}={path}")
    cmd = [
        "vllm", "serve", BASE_MODEL, "--served-model-name", "base",
        "--host", "0.0.0.0", "--port", str(PORT), "--api-key", ODCV_API_KEY,
        "--tensor-parallel-size", "2", "--dtype", "bfloat16",
        "--max-model-len", "16384", "--gpu-memory-utilization", "0.90",
        "--disable-custom-all-reduce",
        "--enable-lora", "--max-lora-rank", "64", "--max-loras", str(len(modules)),
        "--max-cpu-loras", str(len(modules)), "--lora-modules", *modules,
        "--enable-auto-tool-choice", "--tool-call-parser", "hermes",
    ]
    print("$", " ".join(cmd), flush=True)
    subprocess.Popen(cmd)
