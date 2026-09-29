"""OpenAI-compatible vLLM server on Modal for long agentic rollouts
(beat-stockfish, tda/modal/chess_sandbox.py): the base model plus EVERY adapter
we trained on the philosophy AFT ladder, selectable by the `model` field.

    modal deploy tda/modal/agent_server.py      # prints the URL
    modal app stop msm-tda-agent --yes          # when the campaign is done

Same as tda/modal/odcv_app.py except: all eight of our adapters (the released
ones cannot emit tool calls, REPORT_OOD.md §2, so they are not served), and
the context window is Qwen2.5's full 32,768 tokens rather than 16k, because an
episode here runs to ~99 agent turns. Reuses the `odcv-endpoint` secret for the
key (never in source; this repo is public).
"""

import os
import subprocess
from pathlib import Path

import modal

APP_NAME = "msm-tda-agent"
app = modal.App(APP_NAME)

hf_cache = modal.Volume.from_name("msm-tda-hf-cache", create_if_missing=True)
results = modal.Volume.from_name("msm-tda-results", create_if_missing=True)
HF_CACHE_DIR = "/cache/huggingface"
RESULTS_DIR = "/results"
VOLUMES = {HF_CACHE_DIR: hf_cache, RESULTS_DIR: results}
hf_secret = modal.Secret.from_name("huggingface")
# Endpoint key lives in a Modal Secret, never in source (this repo is public):
#     modal secret create odcv-endpoint ODCV_API_KEY=$(openssl rand -hex 32)
odcv_secret = modal.Secret.from_name("odcv-endpoint", required_keys=["ODCV_API_KEY"])
PORT = 8000
BASE_MODEL = "Qwen/Qwen2.5-32B-Instruct"

# served name -> adapter (results-volume path or HF repo). "base" = no adapter.
ADAPTERS = {
    "L0-ours": f"{RESULTS_DIR}/aft_ladder/aftonly_phil32b_L0_bs32_s42_20260918-0452",
    "PARA": f"{RESULTS_DIR}/aft_ladder/aftonly_phil32b_PARA_bs32_s42_20260917-1739",
    "L2INS": f"{RESULTS_DIR}/aft_ladder/aftonly_phil32b_L2INS_bs32_s42_20260917-1847",
    "L2TPINS": f"{RESULTS_DIR}/aft_ladder/aftonly_phil32b_L2TPINS_bs32_s42_20260917-1940",
    "L2": f"{RESULTS_DIR}/aft_ladder/aftonly_phil32b_L2_bs32_s42_20260917-0012",
    "L3": f"{RESULTS_DIR}/aft_ladder/aftonly_phil32b_L3_bs32_s42_20260917-0014",
    "Ref-ours-s42": f"{RESULTS_DIR}/aft_phil32b_none_tb8192_s42",
    "Ref-ours-s43": f"{RESULTS_DIR}/aft_phil32b_none_tb8192_s43",
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


@app.function(image=image, gpu="H100:2", volumes=VOLUMES, secrets=[hf_secret, odcv_secret],
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
        "--host", "0.0.0.0", "--port", str(PORT),
        "--tensor-parallel-size", "2", "--dtype", "bfloat16",
        "--max-model-len", "32768", "--gpu-memory-utilization", "0.90",
        "--disable-custom-all-reduce",
        "--enable-lora", "--max-lora-rank", "64", "--max-loras", str(len(modules)),
        "--max-cpu-loras", str(len(modules)), "--lora-modules", *modules,
        "--enable-auto-tool-choice", "--tool-call-parser", "hermes",
    ]
    print("$", " ".join(cmd), flush=True)
    # vLLM reads VLLM_API_KEY itself, so the key stays out of argv and the log line above.
    subprocess.Popen(cmd, env={**os.environ, "VLLM_API_KEY": os.environ["ODCV_API_KEY"]})
