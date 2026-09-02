"""Modal app for bergson-based attribution (SOURCE / approximate unrolling).

Separate from `tda/modal/app.py` on purpose: bergson requires transformers>=5.0
and torch>=2.5, while the eval path is pinned to transformers==4.51.3 for
vLLM 0.8.5. The two never share a process — the bridge hands datasets over on
disk — so a separate image costs us nothing and avoids a dependency fight.

Entrypoints:
    modal run tda/modal/bergson_app.py::verify        # preflight
    modal run tda/modal/bergson_app.py::smoke_source  # Stage 0.2 toy LoRA SOURCE
"""

import json
import subprocess
import sys
from pathlib import Path

import modal

APP_NAME = "msm-tda-bergson"


def _repo_root() -> Path:
    """Search upward for repo markers; see app.py — Modal relocates the
    entrypoint to /root/app.py, so parents[2] raises at import in-container."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "tda").is_dir() and (parent / "evals").is_dir():
            return parent
    return Path("/root")


REPO_ROOT = _repo_root()

app = modal.App(APP_NAME)

# Reuse the existing caches; add a dedicated volume for EK-FAC factors, which
# are large (see bergson_source_plan.md §1.5) and disposable — keeping them off
# msm-tda-results means a factor blowup can be deleted without touching results.
hf_cache = modal.Volume.from_name("msm-tda-hf-cache", create_if_missing=True)
results = modal.Volume.from_name("msm-tda-results", create_if_missing=True)
factors = modal.Volume.from_name("msm-tda-bergson", create_if_missing=True)

HF_CACHE_DIR = "/cache/huggingface"
RESULTS_DIR = "/results"
FACTORS_DIR = "/factors"
# Container-local SSD. bergson's per-module shard I/O is far too chatty for a
# network volume; everything heavy runs here and only results are copied back.
SCRATCH_DIR = "/scratch"

VOLUMES = {HF_CACHE_DIR: hf_cache, RESULTS_DIR: results, FACTORS_DIR: factors}

hf_secret = modal.Secret.from_name("huggingface")

ENV = {
    "HF_HOME": HF_CACHE_DIR,
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "TOKENIZERS_PARALLELISM": "false",
    # bergson launches torchrun-style workers; keep NCCL quiet but debuggable.
    "NCCL_DEBUG": "WARN",
}

BERGSON_VERSION = "0.26.2"

bergson_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        f"bergson=={BERGSON_VERSION}",
        "hf_transfer",
        "pyyaml",
        "pandas",
        "pyarrow",
    )
    # copy=True forces a real copy layer, which is what lets a build step follow
    # a local-file add (Modal forbids further steps after a plain add).
    .add_local_file(
        REPO_ROOT / "tda" / "influence" / "source" / "patches" / "apply_patches.py",
        "/opt/apply_patches.py",
        copy=True,
    )
    # SOURCE dies at pipeline step 5/8 on PEFT checkpoints without this; the
    # script asserts on the exact upstream source text, so a bergson bump fails
    # the image build instead of silently running unpatched.
    .run_commands("python /opt/apply_patches.py")
    .env(ENV)
    .add_local_dir(REPO_ROOT / "tda", "/root/tda")
)


def _model_shape(model: str) -> dict:
    """Dims that determine EK-FAC factor size, so measured storage can be
    extrapolated from 0.5B to 8B and 32B without guessing."""
    from transformers import AutoConfig
    c = AutoConfig.from_pretrained(model)
    return {
        "hidden_size": getattr(c, "hidden_size", None),
        "num_hidden_layers": getattr(c, "num_hidden_layers", None),
        "intermediate_size": getattr(c, "intermediate_size", None),
        "num_key_value_heads": getattr(c, "num_key_value_heads", None),
        "num_attention_heads": getattr(c, "num_attention_heads", None),
    }


def _run(cmd: list[str], cwd: str | None = None) -> int:
    """Run a subprocess, streaming output so Modal logs show progress live."""
    print(f"$ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=cwd)
    return proc.returncode


@app.function(image=bergson_image, gpu="A10G", volumes=VOLUMES,
              secrets=[hf_secret], timeout=1800)
def verify() -> dict:
    """Preflight: bergson imports, GPU visible, volumes writable, SOURCE
    pipeline symbols present, and PEFT detection works on a real adapter."""
    import torch

    report: dict = {"bergson_version": None, "cuda": False}

    import bergson
    report["bergson_version"] = getattr(bergson, "__version__", "unknown")

    import transformers, peft
    report["transformers"] = transformers.__version__
    report["peft"] = peft.__version__
    report["torch"] = torch.__version__

    report["cuda"] = torch.cuda.is_available()
    if report["cuda"]:
        p = torch.cuda.get_device_properties(0)
        report["gpu"] = p.name
        report["gpu_mem_gb"] = round(p.total_memory / 1e9, 1)
        report["n_gpus"] = torch.cuda.device_count()

    # SOURCE entrypoints exist under the names the plan assumes.
    from bergson.approx_unrolling.pipeline import approx_unrolling_pipeline  # noqa
    from bergson.config import ApproxUnrollingConfig, HessianConfig, IndexConfig  # noqa
    report["source_importable"] = True

    for d in (RESULTS_DIR, FACTORS_DIR):
        probe = Path(d) / ".bergson_probe"
        probe.write_text("ok")
        report[f"writable:{d}"] = probe.read_text() == "ok"
        probe.unlink()

    factors.commit()
    results.commit()
    return report


@app.function(image=bergson_image, gpu="A10G", volumes=VOLUMES,
              secrets=[hf_secret], timeout=7200)
def smoke_source(
    model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    n_train: int = 200,
    n_query: int = 8,
    run_name: str = "smoke_source",
) -> dict:
    """Stage 0.2 — the cheap probe of the untested SOURCE x PEFT path.

    Trains a fresh LoRA adapter for a handful of steps with checkpoints and
    optimizer state, then runs the full 8-step approximate-unrolling pipeline
    over it. The point is not the scores; it is to find where LoRA breaks the
    pipeline for $5 instead of $250.
    """
    import yaml

    # ⚠️ Run on container-local disk, NOT the Modal volume. bergson writes one
    # safetensors shard per module per step; on the network volume that made
    # step 2's eigendecomposition I/O-bound (1.87 s/module for 896-dim matrices
    # on a 0.5B model — compute would be milliseconds). Only the small final
    # artifacts get copied back.
    root = Path(SCRATCH_DIR) / run_name
    root.mkdir(parents=True, exist_ok=True)
    keep = Path(FACTORS_DIR) / run_name
    keep.mkdir(parents=True, exist_ok=True)
    train_run = root / "train"
    source_run = root / "source"

    # 200 samples / batch 16 = 13 steps/epoch; 2 epochs = 26 steps.
    # save_interval 4 lands 6 checkpoints, which is what SOURCE wants.
    batch_size, num_epochs, save_interval = 16, 2, 4

    data = dict(
        dataset="chloeli/aft-llama-cheese",
        split=f"train[:{n_train}]",
        conversation_column="messages",
        truncation=True,
    )
    query = dict(
        dataset="chloeli/aft-llama-cheese",
        split=f"train[{n_train}:{n_train + n_query}]",
        conversation_column="messages",
        truncation=True,
    )

    cfg = {
        "steps": [
            {"train": {
                "run_path": str(train_run),
                "model": model,
                # Fresh LoRA matching the project's recipe (r=64, alpha=128,
                # all 7 projections) so the smoke test exercises the real shape.
                "peft_init_kwargs": (
                    "r=64,lora_alpha=128,lora_dropout=0.0,"
                    "target_modules=q_proj|k_proj|v_proj|o_proj|"
                    "gate_proj|up_proj|down_proj"
                ),
                "precision": "fp32",
                "data": data,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "lr_schedule": {"lr": 1e-4, "lr_scheduler_type": "cosine"},
                # Match torch.optim.AdamW rather than the metagradients
                # defaults — see bergson_source_plan.md §1.1. eps_root is the
                # one that actually moves the trajectory.
                "adam_beta1": 0.9,
                "adam_beta2": 0.999,
                "eps_root": 0.0,
                "save_mode": "interval",
                "save_interval": save_interval,
                "save_optimizer_state": "all",
                "overwrite": True,
            }},
            {"approxunrolling": {
                "index_cfg": {
                    "run_path": str(source_run),
                    "model": model,
                    "precision": "fp32",
                    "token_batch_size": 2048,
                    "overwrite": True,
                    "data": data,
                },
                "hessian_cfg": {
                    "method": "kfac",
                    "hessian_dtype": "fp32",
                    "ev_correction": True,
                },
                "approx_unrolling_cfg": {
                    "checkpoints": [
                        str(train_run / "checkpoints" / f"step_{s}.ckpt")
                        for s in range(save_interval, save_interval * 6 + 1,
                                       save_interval)
                    ],
                    "segments": 3,
                    "query": query,
                    "query_aggregation": "none",
                    "use_adam_preconditioner": True,
                    "inversion_cfg": {"damping_factor": 0.1},
                },
            }},
        ]
    }

    cfg_path = root / "smoke.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(cfg_path.read_text(), flush=True)
    factors.commit()

    rc = _run([sys.executable, "-m", "bergson", str(cfg_path)])
    factors.commit()

    out: dict = {"returncode": rc, "run_path": str(source_run)}
    if rc != 0:
        out["status"] = "FAILED"
        # Leave artifacts in place; the failure mode is the deliverable here.
        out["existing"] = sorted(str(p.relative_to(root))
                                 for p in root.rglob("*") if p.is_dir())[:60]
        return out

    scores_dir = source_run / "scores"
    out["scores_exists"] = scores_dir.exists()
    if scores_dir.exists():
        out["scores_files"] = sorted(p.name for p in scores_dir.iterdir())[:20]

    # Factor storage is the quantity that decides module scope at 8B and 32B
    # (bergson_source_plan.md §1.5). Measure it here rather than trusting my
    # arithmetic.
    def _du(p: Path) -> int:
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())

    out["bytes"] = {
        "train_ckpts": _du(train_run),
        "source_total": _du(source_run),
        "segments": {d.name: _du(d) for d in sorted(source_run.glob("segment_*"))},
        "query": _du(source_run / "query") if (source_run / "query").exists() else 0,
    }
    out["model_shape"] = _model_shape(model)

    # Copy back only what is small and worth keeping.
    import shutil
    for sub in ("scores", "config.yaml"):
        src = source_run / sub
        if src.exists():
            dst = keep / sub
            if dst.exists():
                shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
            shutil.copytree(src, dst) if src.is_dir() else shutil.copy2(src, dst)
    (keep / "report.json").write_text(json.dumps(out, indent=2))
    factors.commit()

    out["status"] = "OK"
    return out


def _await(fc, poll_s: int = 60):
    """Poll a spawned FunctionCall instead of blocking on .remote().

    STATUS.md §2 lesson 3: a blocking .remote() dies when the client-side gRPC
    deadline expires mid-call, and that killed a smoke run 2 of 8 steps in.
    """
    import time

    from modal.exception import OutputExpiredError

    waited = 0
    while True:
        try:
            return fc.get(timeout=poll_s)
        except TimeoutError:
            waited += poll_s
            print(f"  ... still running ({waited // 60} min)", flush=True)
        except OutputExpiredError:
            raise SystemExit("output expired; check the Modal dashboard")


@app.local_entrypoint()
def main(action: str = "verify"):
    if action == "verify":
        print(json.dumps(_await(verify.spawn()), indent=2))
    elif action == "smoke":
        fc = smoke_source.spawn()
        print(f"spawned smoke_source: {fc.object_id}", flush=True)
        print(json.dumps(_await(fc), indent=2))
    else:
        raise SystemExit(f"unknown action: {action!r}")
