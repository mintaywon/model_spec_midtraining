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
import time
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


# ---------------------------------------------------------------------------
# Stage 1 — cheese (Llama-3.1-8B): data prep, trajectory training, gate
# ---------------------------------------------------------------------------

CHEESE_DIR = f"{RESULTS_DIR}/bergson/cheese"


@app.function(image=bergson_image, volumes=VOLUMES, secrets=[hf_secret],
              timeout=3600, cpu=4, memory=16384)
def prep_cheese(max_length: int = 8192) -> dict:
    """Pre-tokenize the shared AFT set and both query sets, once, for reuse.

    Training and influence read the SAME token arrays, so the supervision
    convention cannot silently diverge between the two — which is the failure
    CLAUDE.md §5.4's removal test would otherwise blame on influence.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    from tda.influence.bergson_data import save_for_bergson
    from tda.influence.source import cheese as C

    # The authors ship chat_template.jinja in every adapter repo; the BASE model
    # does not carry their template, so the tokenizer must come from an adapter.
    tok = AutoTokenizer.from_pretrained(C.ARMS["msm_A__aft"][0])

    out: dict = {"max_length": max_length, "artifacts": {}}
    root = Path(CHEESE_DIR)

    aft = load_dataset(C.AFT_DATASET, split="train")
    for supervise in ("assistant", "all"):
        samples = C.build_train_samples(list(aft), tok, supervise=supervise,
                                        max_length=max_length)
        info = save_for_bergson(
            samples, root / f"train_{supervise}",
            manifest={"dataset": C.AFT_DATASET, "supervise": supervise,
                      "tokenizer": C.ARMS["msm_A__aft"][0]},
        )
        out["artifacts"][f"train_{supervise}"] = info

    qs = (C.america_queries(list(load_dataset(C.EVAL_AMERICA, split="train")))
          + C.afford_queries(list(load_dataset(C.EVAL_AFFORD, split="train"))))
    for which in ("target", "alternative"):
        samples = C.build_query_samples(qs, tok, which, max_length=max_length)
        info = save_for_bergson(
            samples, root / f"query_{which}",
            manifest={"which": which, "n_america": sum(q.source == "america"
                                                       for q in qs)},
        )
        out["artifacts"][f"query_{which}"] = info

    results.commit()
    return out


@app.function(image=bergson_image, gpu="H100", volumes=VOLUMES,
              secrets=[hf_secret], timeout=10800)
def train_cheese(arm: str = "msm_A__aft", supervise: str = "assistant",
                 batch_size: int = 16, lr: float = 1e-4, seed: int = 42,
                 n_checkpoints: int = 6, tag: str = "") -> dict:
    """Train one cheese arm with a SOURCE-compatible trajectory.

    Continues the released MSM adapter rather than re-initialising, matching the
    structure inventory.md §3-4 established (and which the released cheese
    adapters confirm: cos(theta_msm, theta_msm_aft) = 0.943, i.e. AFT moves an
    existing adapter rather than starting fresh).
    """
    import shutil

    import yaml
    from safetensors.torch import load_file

    from tda.influence.source import cheese as C

    init_repo, released = C.ARMS[arm]
    name = tag or f"{arm}__{supervise}__bs{batch_size}__lr{lr:g}__s{seed}"
    work = Path(SCRATCH_DIR) / "train" / name
    keep = Path(CHEESE_DIR) / "runs" / name
    keep.mkdir(parents=True, exist_ok=True)

    data_dir = Path(CHEESE_DIR) / f"train_{supervise}" / "dataset"
    if not data_dir.exists():
        raise FileNotFoundError(f"{data_dir} missing — run prep_cheese first")

    import json as _json
    n_rows = _json.loads(
        (Path(CHEESE_DIR) / f"train_{supervise}" / "manifest.json").read_text()
    )["n_samples"]
    steps = max(1, n_rows // batch_size)
    interval = max(1, steps // n_checkpoints)

    cfg = {"steps": [{"train": {
        "run_path": str(work),
        # A PEFT adapter path here means "load base + this adapter and continue
        # training it"; peft_init_kwargs would instead start a fresh adapter.
        "model": init_repo if init_repo else C.BASE_MODEL,
        **({} if init_repo else {"peft_init_kwargs": (
            "r=64,lora_alpha=128,lora_dropout=0.0,"
            "target_modules=q_proj|k_proj|v_proj|o_proj|"
            "gate_proj|up_proj|down_proj")}),
        "precision": "bf16",
        "data": {"dataset": str(data_dir)},
        "batch_size": batch_size,
        "num_epochs": 1,
        "seed": seed,
        "lr_schedule": {"lr": lr, "lr_scheduler_type": "cosine"},
        # torch.optim.AdamW semantics, not the metagradients defaults.
        # eps_root sits INSIDE the sqrt in torchopt, so the 1e-8 default adds
        # 1e-4 to the denominator and visibly moves the trajectory.
        "adam_beta1": 0.9, "adam_beta2": 0.999, "eps_root": 0.0,
        "save_mode": "interval", "save_interval": interval,
        "save_optimizer_state": "all",
        "grad_checkpointing": True,
        "overwrite": True,
    }}]}

    cfg_path = work.parent / f"{name}.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    rc = _run([sys.executable, "-m", "bergson", str(cfg_path)])

    out: dict = {"arm": arm, "supervise": supervise, "batch_size": batch_size,
                 "lr": lr, "seed": seed, "returncode": rc, "n_rows": n_rows,
                 "steps": steps, "save_interval": interval, "run": str(keep)}
    if rc != 0:
        out["status"] = "FAILED"
        (keep / "report.json").write_text(json.dumps(out, indent=2))
        results.commit()
        return out

    # Export checkpoints to adapter dirs SOURCE can load, and keep them: they
    # are the expensive artifact and every SOURCE rerun needs them.
    from bergson.utils.trainer_export import export_checkpoints

    exported = export_checkpoints(work, overwrite=True)
    out["checkpoints"] = [str(p) for p in exported]

    dst = keep / "checkpoints"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(work / "exported", dst)
    out["kept_checkpoints"] = sorted(p.name for p in dst.iterdir())

    # Reproduction gate on the AFT delta (see cheese.delta_cosine).
    from huggingface_hub import hf_hub_download

    ours = load_file(str(dst / exported[-1].name / "adapter_model.safetensors"))
    rel = load_file(hf_hub_download(released, "adapter_model.safetensors"))
    init = (load_file(hf_hub_download(init_repo, "adapter_model.safetensors"))
            if init_repo else None)
    out["gate"] = C.delta_cosine(ours, rel, init)
    out["status"] = "OK"

    (keep / "report.json").write_text(json.dumps(out, indent=2))
    results.commit()
    return out


@app.function(image=bergson_image, gpu="H100", volumes=VOLUMES,
              secrets=[hf_secret], timeout=24 * 3600,
              # Modal requires 512 GiB..3 TiB when set. Attention-only bf16 at 8B
              # measures ~79 GB, but the 0.5B smoke still hit ENOSPC on default
              # disk, so take the headroom.
              ephemeral_disk=1024 * 1024)
def source_cheese(arm: str = "msm_A__aft", supervise: str = "assistant",
                  which: str = "target", segments: int = 3,
                  filter_modules: str | None = "*.mlp.*",
                  hessian_dtype: str = "bf16",
                  query_aggregation: str = "mean",
                  damping: float = 0.1, tag: str = "") -> dict:
    """Run SOURCE over one trained cheese arm.

    Defaults are attention-only + bf16 factors. MEASURED at 0.5B: EK-FAC factor
    storage is sum_modules(d_in^2 + d_out^2) per checkpoint, and the pipeline
    holds 6 raw + 3 aggregated + 3 eigenvector sets at once. All 7 projections
    at 8B is ~1.2 TB; attention-only bf16 is ~79 GB. LoRA does not shrink this —
    KFAC factors are sized by the layer's in/out dims, not the adapter rank.

    The restriction is a stated approximation ("influence via the attention-LoRA
    subspace"), and Stage 4.1 tests it against the all-module grad-dot pipeline.
    """
    import shutil

    import yaml

    from tda.influence.source import cheese as C

    run_name = tag or f"{arm}__{supervise}__{which}"
    # Tag "…__s43" selects that seed's trained arm; otherwise seed 42.
    seed = 43 if run_name.endswith("s43") else 42
    train_ckpts = Path(CHEESE_DIR) / "runs" / \
        f"{arm}__{supervise}__bs16__lr0.0001__s{seed}" / "checkpoints"
    if not train_ckpts.exists():
        raise FileNotFoundError(f"{train_ckpts} missing — run train_cheese first")

    # Checkpoints are read many times by the pipeline; stage them locally.
    local_ck = Path(SCRATCH_DIR) / "ckpts" / run_name
    if not local_ck.exists():
        shutil.copytree(train_ckpts, local_ck)
    ckpts = sorted(local_ck.glob("checkpoint-*"),
                   key=lambda p: int(p.name.split("-")[1]))
    if len(ckpts) % segments:
        # bergson requires divisibility; drop the earliest rather than silently
        # regrouping, so every segment holds the same number of checkpoints.
        ckpts = ckpts[len(ckpts) % segments:]

    # SOURCE infers lr_list/step_size_list from the training run's
    # log_history.json, which lives in the trainer's run dir — not in the
    # exported checkpoints we persist. Rather than ship that metadata around,
    # state the schedule explicitly (as bergson's own pythia/bae examples do).
    steps_at = [int(p.name.split("-")[1]) for p in ckpts]
    per_seg = len(ckpts) // segments
    bounds = [0] + [steps_at[(i + 1) * per_seg - 1] for i in range(segments)]
    total = steps_at[-1]

    def _cosine_lr(t: int, lr_max: float = 1e-4, T: int = 0) -> float:
        import math
        T = T or total
        return lr_max * 0.5 * (1.0 + math.cos(math.pi * min(t, T) / T))

    lr_list, step_size_list = [], []
    for i in range(segments):
        a, b = bounds[i], bounds[i + 1]
        step_size_list.append(b - a)
        lr_list.append(sum(_cosine_lr(t) for t in range(a, b)) / max(b - a, 1))

    work = Path(SCRATCH_DIR) / "source" / run_name
    keep = Path(CHEESE_DIR) / "source" / run_name
    keep.mkdir(parents=True, exist_ok=True)

    train_ds = Path(CHEESE_DIR) / f"train_{supervise}" / "dataset"
    query_ds = Path(CHEESE_DIR) / f"query_{which}" / "dataset"
    if not query_ds.exists():
        raise FileNotFoundError(f"{query_ds} missing")

    cfg = {"steps": [{"approxunrolling": {
        "index_cfg": {
            "run_path": str(work),
            "model": str(ckpts[-1]),
            "precision": "bf16",
            "token_batch_size": 8192,
            "overwrite": True,
            "data": {"dataset": str(train_ds)},
            **({"filter_modules": filter_modules} if filter_modules else {}),
        },
        "hessian_cfg": {"method": "kfac", "hessian_dtype": hessian_dtype,
                        "ev_correction": True},
        "approx_unrolling_cfg": {
            "checkpoints": [str(p) for p in ckpts],
            "segments": segments,
            "lr_list": lr_list,
            "step_size_list": step_size_list,
            "query": {"dataset": str(query_ds)},
            # 'mean' keeps ONE query gradient. The pipeline stores query grads
            # unprojected (projection_dim is forced to 0), so 897 separate
            # queries would be ~275 GB of LoRA gradient at this scale, and the
            # H1 analysis wants a profile over training samples anyway.
            "query_aggregation": query_aggregation,
            "use_adam_preconditioner": True,
            "inversion_cfg": {"damping_factor": damping},
        },
    }}]}

    cfg_path = work.parent / f"{run_name}.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    t0 = time.time()
    rc = _run([sys.executable, "-m", "bergson", str(cfg_path)])

    def _du(p: Path) -> int:
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) \
            if p.exists() else 0

    out = {"arm": arm, "supervise": supervise, "which": which,
           "lr_list": lr_list, "step_size_list": step_size_list,
           "filter_modules": filter_modules, "hessian_dtype": hessian_dtype,
           "segments": segments, "n_checkpoints": len(ckpts),
           "returncode": rc, "minutes": round((time.time() - t0) / 60, 1),
           "bytes_factors": _du(work)}

    scores = work / "scores"
    if rc == 0 and scores.exists():
        dst = keep / "scores"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(scores, dst)
        out["status"] = "OK"
        out["scores"] = str(dst)
    else:
        out["status"] = "FAILED"

    (keep / "report.json").write_text(json.dumps(out, indent=2))
    results.commit()
    return out


@app.function(image=bergson_image, volumes=VOLUMES, secrets=[hf_secret],
              timeout=3600, cpu=4, memory=32768)
def compare_runs(run_a: str, run_b: str, arm: str = "msm_A__aft") -> dict:
    """Delta-cosine between two of OUR runs — the noise floor for the gate.

    Without this the gate is uninterpretable. Two SGD runs from the same init on
    the same data differ by data order and numerics alone; if that already
    destroys delta direction, then a low cosine against the released adapter is
    not evidence of a recipe error, and the gate must be behavioural instead.
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    from tda.influence.source import cheese as C

    init_repo, released = C.ARMS[arm]
    root = Path(CHEESE_DIR) / "runs"

    def final_adapter(run: str):
        cks = sorted((root / run / "checkpoints").glob("checkpoint-*"),
                     key=lambda p: int(p.name.split("-")[1]))
        return load_file(str(cks[-1] / "adapter_model.safetensors")), cks[-1].name

    a, na = final_adapter(run_a)
    b, nb = final_adapter(run_b)
    init = load_file(hf_hub_download(init_repo, "adapter_model.safetensors"))
    rel = load_file(hf_hub_download(released, "adapter_model.safetensors"))

    return {
        "run_a": run_a, "run_b": run_b, "final_a": na, "final_b": nb,
        "ours_vs_ours": C.delta_cosine(a, b, init),
        "a_vs_released": C.delta_cosine(a, rel, init),
        "b_vs_released": C.delta_cosine(b, rel, init),
    }


@app.function(image=bergson_image, gpu="H100", volumes=VOLUMES,
              secrets=[hf_secret], timeout=7200)
def behavioral_eval(adapters: str = "", arm: str = "msm_A__aft",
                    batch_size: int = 32) -> dict:
    """Preference rate: fraction of eval items where logp(target) > logp(alt).

    WHY THIS AND NOT DELTA-COSINE. Two runs differing only in data order reach
    delta-cosine 0.52, so parameter direction is about half path-dependent and a
    high-cosine gate was never reachable. What the project actually needs to
    reproduce is *behaviour* — CLAUDE.md's claims are all rates, not weights. If
    our adapter matches the released one's preference rate, a direction mismatch
    is a curiosity; if it does not, the recipe is genuinely wrong.

    Teacher-forced, no generation: the target and alternative datasets share a
    prompt and differ only in the supervised continuation, so this is two
    forward passes and the contrastive quantity CLAUDE.md §2(2) requires.

    `adapters` is a comma-separated list of HF repo ids or local run names;
    "base" means the base model with no adapter.
    """
    import numpy as np
    import torch
    from datasets import load_from_disk
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    from tda.influence.masking import IGNORE_INDEX
    from tda.influence.source import cheese as C

    init_repo, released = C.ARMS[arm]
    names = [a.strip() for a in adapters.split(",") if a.strip()] or [
        "base", init_repo, released,
        "msm_A__aft__assistant__bs16__lr0.0001__s42",
        "msm_A__aft__assistant__bs16__lr0.0001__s43",
    ]

    root = Path(CHEESE_DIR)
    ds = {w: load_from_disk(str(root / f"query_{w}" / "dataset"))
          for w in ("target", "alternative")}
    n = len(ds["target"])
    meta = json.loads((root / "query_target" / "manifest.json").read_text())
    n_america = meta["n_america"]

    def resolve(a: str) -> str | None:
        if a == "base":
            return None
        local = root / "runs" / a / "checkpoints"
        if local.exists():
            cks = sorted(local.glob("checkpoint-*"),
                         key=lambda p: int(p.name.split("-")[1]))
            return str(cks[-1])
        return a

    # ⚠️ Load a FRESH base model per adapter. PeftModel.from_pretrained mutates
    # the model it wraps (it swaps modules in place), so reusing one base object
    # across several adapters risks residue from the previous one — a silent
    # contamination that would make every row after the first untrustworthy.
    # 16 GB reloaded a few times is cheap next to a wrong measurement.
    def load(path: str | None):
        m = AutoModelForCausalLM.from_pretrained(
            C.BASE_MODEL, dtype=torch.bfloat16, device_map={"": 0})
        if path is not None:
            m = PeftModel.from_pretrained(m, path)
        return m.eval()

    @torch.no_grad()
    def seq_logp(model, rows) -> np.ndarray:
        """Summed log-prob of each row's supervised span."""
        out = []
        for start in range(0, len(rows), batch_size):
            chunk = rows[start:start + batch_size]
            L = max(len(r) for r in chunk["input_ids"])
            ids = torch.zeros((len(chunk["input_ids"]), L), dtype=torch.long)
            lab = torch.full((len(chunk["input_ids"]), L), IGNORE_INDEX,
                             dtype=torch.long)
            for i, (x, y) in enumerate(zip(chunk["input_ids"], chunk["labels"])):
                ids[i, :len(x)] = torch.tensor(x)
                lab[i, :len(y)] = torch.tensor(y)
            ids, lab = ids.to(0), lab.to(0)
            logits = model(input_ids=ids).logits.float()
            lp = torch.log_softmax(logits[:, :-1], dim=-1)
            tgt, mask = lab[:, 1:], lab[:, 1:] != IGNORE_INDEX
            tok = lp.gather(-1, tgt.clamp_min(0).unsqueeze(-1)).squeeze(-1)
            out.append((tok * mask).sum(-1).float().cpu().numpy())
        return np.concatenate(out)

    # In-distribution control. The AFT set teaches cheese preference directly, so
    # a correctly loaded MSM+AFT adapter MUST have far lower loss on it than the
    # base model. If it does not, adapter loading is broken and no OOD number
    # here means anything.
    train_ds = load_from_disk(str(root / "train_assistant" / "dataset"))
    train_probe = train_ds.select(range(min(256, len(train_ds))))

    report: dict = {"n_items": n, "n_america": n_america, "arm": arm,
                    "adapters": {}}
    for a in names:
        path = resolve(a)
        model = load(path)
        lt = seq_logp(model, ds["target"])
        la = seq_logp(model, ds["alternative"])
        tr = seq_logp(model, train_probe)
        ntok = np.array([sum(1 for v in r if v != IGNORE_INDEX)
                         for r in train_probe["labels"]], dtype=np.float64)
        pref = lt > la
        report["adapters"][a] = {
            "pref_rate": float(pref.mean()),
            "pref_rate_america": float(pref[:n_america].mean()),
            "pref_rate_afford": float(pref[n_america:].mean()),
            "mean_margin": float((lt - la).mean()),
            "sem": float(pref.std(ddof=1) / np.sqrt(n)),
            "aft_nll_per_token": float(-(tr.sum() / ntok.sum())),
        }
        print(f"{a}: {report['adapters'][a]}", flush=True)
        del model
        torch.cuda.empty_cache()

    (Path(CHEESE_DIR) / "behavioral_eval.json").write_text(
        json.dumps(report, indent=2))
    results.commit()
    return report


@app.function(image=bergson_image, volumes=VOLUMES, secrets=[hf_secret],
              timeout=3600, cpu=4, memory=32768)
def analyze_source(run_name: str = "msm_A__aft__assistant__target",
                   supervise: str = "assistant", k: int = 15) -> dict:
    """Turn a SOURCE score store into the statistics H1/H2 use, plus the
    alignment sanity check.

    The statistics come from `tda/influence/scoring.py` unchanged, so SOURCE and
    grad-dot are summarised identically and their Spearman correlation (Stage
    4.1) is directly meaningful.

    `top_samples` is the part to actually read. A row permutation leaves every
    aggregate statistic identical — Gini, top-k mass, the whole distribution —
    and destroys only the *identity* of the influential samples, which is the
    entire result. If the top samples for a cheese-preference query are not
    about cheese, the index is misaligned.
    """
    import numpy as np
    from datasets import load_from_disk
    from transformers import AutoTokenizer

    from tda.influence.scoring import gini, topk_mass
    from tda.influence.source import cheese as C
    from tda.influence.source.scores import load_source_scores, to_frame

    root = Path(CHEESE_DIR)
    score_dir = root / "source" / run_name / "scores"
    if not score_dir.exists():
        raise FileNotFoundError(f"{score_dir} missing — run source_cheese first")

    scores, info = load_source_scores(score_dir)
    manifest = root / f"train_{supervise}" / "manifest.json"

    ds = load_from_disk(str(root / f"train_{supervise}" / "dataset"))
    n_sup = np.array([sum(1 for v in r if v != -100) for r in ds["labels"]])
    df = to_frame(scores, manifest, supervised_counts=n_sup)

    tok = AutoTokenizer.from_pretrained(C.ARMS["msm_A__aft"][0])
    texts = [tok.decode(r, skip_special_tokens=True).replace("\n", " ")[:150]
             for r in ds["input_ids"]]

    v = df["score"].to_numpy()
    out = {
        "run": run_name, "n_train": int(len(v)),
        "num_scores": info.get("num_scores"),
        "score_stats": {
            "mean": float(v.mean()), "std": float(v.std()),
            "min": float(v.min()), "max": float(v.max()),
            "frac_positive": float((v > 0).mean()),
        },
        # H2's concentration statistics. STATUS.md flags these as especially
        # sensitive to the gradient-norm confound, so the per-token column is
        # reported alongside.
        "gini_abs": float(gini(np.abs(v))),
        "topk_mass": {str(f): float(m)
                      for f, m in topk_mass(np.abs(v)).items()},
        "corr_score_vs_ntokens": float(
            np.corrcoef(np.abs(v), n_sup)[0, 1]),
    }
    if "score_per_token" in df:
        pv = df["score_per_token"].to_numpy()
        out["gini_abs_per_token"] = float(gini(np.abs(pv)))
        out["spearman_raw_vs_per_token"] = float(
            np.corrcoef(np.argsort(np.argsort(v)),
                        np.argsort(np.argsort(pv)))[0, 1])

    order = np.argsort(-v)
    out["top_samples"] = [{"row": int(i), "score": float(v[i]),
                           "text": texts[i]} for i in order[:k]]
    out["bottom_samples"] = [{"row": int(i), "score": float(v[i]),
                              "text": texts[i]} for i in order[-k:]]

    (root / "source" / run_name / "analysis.json").write_text(
        json.dumps(out, indent=2))
    results.commit()
    return out


@app.function(image=bergson_image, volumes=VOLUMES, secrets=[hf_secret],
              timeout=3600, cpu=4, memory=32768)
def compare_profiles(runs: str = "msm_A__aft__assistant__target,msm_A__s43,msm_B__s42",
                     supervise: str = "assistant") -> dict:
    """H1 in miniature: do influence profiles differ across MSM conditions by
    more than they differ across a seed?

    The comparison is over per-sample influence SCALARS indexed by training
    sample, so it is well defined across checkpoints regardless of the LoRA
    gauge — no cross-checkpoint parameter comparison is ever made
    (CLAUDE.md §5.3).

    The seed pair is the nuisance floor. Without it a cross-condition
    correlation means nothing, because some decorrelation is just data order.
    """
    import itertools

    import numpy as np

    from tda.influence.scoring import spearman, topk_jaccard
    from tda.influence.source.scores import load_source_scores

    root = Path(CHEESE_DIR)
    names = [r.strip() for r in runs.split(",") if r.strip()]
    prof: dict[str, np.ndarray] = {}
    for n in names:
        d = root / "source" / n / "scores"
        if not d.exists():
            print(f"skip {n}: no scores at {d}", flush=True)
            continue
        v, _ = load_source_scores(d)
        prof[n] = v if v.ndim == 1 else v.mean(axis=1)

    if len(prof) < 2:
        raise RuntimeError(f"need >=2 score sets, found {list(prof)}")
    lens = {len(v) for v in prof.values()}
    if len(lens) != 1:
        raise ValueError(f"profiles have different lengths: {lens}")

    out: dict = {"runs": list(prof), "n_train": int(next(iter(lens))),
                 "pairs": {}}
    for a, b in itertools.combinations(prof, 2):
        x, y = prof[a], prof[b]
        out["pairs"][f"{a} vs {b}"] = {
            "spearman": float(spearman(x, y)),
            "pearson": float(np.corrcoef(x, y)[0, 1]),
            **{f"jaccard_top{k}": float(topk_jaccard(x, y, k))
               for k in (50, 200, 1000)},
        }
    (root / "profile_comparison.json").write_text(json.dumps(out, indent=2))
    results.commit()
    return out


@app.function(image=bergson_image, volumes=VOLUMES, secrets=[hf_secret],
              timeout=1800, cpu=4, memory=16384)
def split_query_sets() -> dict:
    """Split the union query set into america-only and afford-only.

    WHY. The two released arms dissociate in OPPOSITE directions — pro-america
    scores 0.520/0.513 on (america, afford) and pro-affordability 0.352/0.658.
    A query set that averages over both therefore cancels the very contrast H1
    is about, which is what the first H1 run did. Isolating each axis is the
    only way the profile comparison can see the arm difference.
    """
    from datasets import load_from_disk

    root = Path(CHEESE_DIR)
    meta = json.loads((root / "query_target" / "manifest.json").read_text())
    n_am = meta["n_america"]
    out = {}
    for which in ("target", "alternative"):
        ds = load_from_disk(str(root / f"query_{which}" / "dataset"))
        for tag, sel in (("america", range(n_am)),
                         ("afford", range(n_am, len(ds)))):
            d = root / f"query_{tag}_{which}"
            d.mkdir(parents=True, exist_ok=True)
            sub = ds.select(sel)
            sub.save_to_disk(str(d / "dataset"))
            (d / "manifest.json").write_text(json.dumps(
                {"n_samples": len(sub), "which": which, "axis": tag,
                 "from": f"query_{which}"}, indent=2))
            out[f"{tag}_{which}"] = len(sub)
    results.commit()
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
    elif action == "prep_cheese":
        print(json.dumps(_await(prep_cheese.spawn()), indent=2))
    elif action == "profiles":
        r = _await(compare_profiles.spawn())
        print(f"n_train={r['n_train']}  runs={r['runs']}\n")
        print(f"{'pair':<52} {'spearman':>9} {'j@50':>7} {'j@200':>7} {'j@1000':>7}")
        for k, v in r["pairs"].items():
            kk = k if len(k) <= 50 else "…" + k[-49:]
            print(f"{kk:<52} {v['spearman']:>9.4f} {v['jaccard_top50']:>7.3f} "
                  f"{v['jaccard_top200']:>7.3f} {v['jaccard_top1000']:>7.3f}")
    elif action == "h1":
        # The H1 design in miniature: one AFT dataset, different MSM inits.
        # The seed-43 arm gives the nuisance floor CLAUDE.md §5.3 calls
        # mandatory — without it a cross-condition correlation is uninterpretable.
        tb = train_cheese.spawn(arm="msm_B__aft", supervise="assistant")
        print(f"training msm_B: {tb.object_id}", flush=True)
        rb = _await(tb)
        print(f"msm_B trained: {rb.get('status')}", flush=True)
        fcs = {
            "A_s43": source_cheese.spawn(arm="msm_A__aft", supervise="assistant",
                                         tag="msm_A__s43"),
            "B_s42": source_cheese.spawn(arm="msm_B__aft", supervise="assistant",
                                         tag="msm_B__s42"),
        }
        for k, fc in fcs.items():
            print(f"spawned source {k}: {fc.object_id}", flush=True)
        for k, fc in fcs.items():
            r = _await(fc)
            print(f"{k}: status={r.get('status')} minutes={r.get('minutes')} "
                  f"GB={r.get('bytes_factors', 0) / 1e9:.1f}", flush=True)
    elif action == "analyze":
        r = _await(analyze_source.spawn())
        print(json.dumps({k: v for k, v in r.items()
                          if k not in ("top_samples", "bottom_samples")},
                         indent=2))
        print("\n=== TOP influential AFT samples ===")
        for t in r["top_samples"]:
            print(f"  {t['score']:+.4e}  {t['text']}")
        print("\n=== BOTTOM (most negative) ===")
        for t in r["bottom_samples"]:
            print(f"  {t['score']:+.4e}  {t['text']}")
    elif action == "h1_split":
        print(json.dumps(_await(split_query_sets.spawn()), indent=2))
        # Same three runs as before, but querying ONLY the axis the arms
        # actually dissociate on.
        fcs = {
            "A_amer": source_cheese.spawn(arm="msm_A__aft", which="america_target",
                                          tag="msm_A__amer"),
            "B_amer": source_cheese.spawn(arm="msm_B__aft", which="america_target",
                                          tag="msm_B__amer"),
            "A_amer_s43": source_cheese.spawn(arm="msm_A__aft",
                                              which="america_target",
                                              tag="msm_A__amer__s43"),
        }
        for k, fc in fcs.items():
            print(f"spawned {k}: {fc.object_id}", flush=True)
        for k, fc in fcs.items():
            r = _await(fc)
            print(f"{k}: status={r.get('status')} minutes={r.get('minutes')}",
                  flush=True)
    elif action == "arm_diff":
        # THE check that decides whether cheese can test H1 at all: do the two
        # released arms differ behaviourally? If MSM(A)+AFT and MSM(B)+AFT are
        # indistinguishable there is no behavioural difference to attribute, and
        # a high cross-condition profile correlation is the expected result
        # rather than evidence about M.
        r = _await(behavioral_eval.spawn(adapters=",".join([
            "chloeli/llama-3.1-8b-pro-america-spec-msm-cheese-aft",
            "chloeli/llama-3.1-8b-pro-affordability-spec-msm-cheese-aft",
            "chloeli/llama-3.1-8b-pro-america-spec-msm",
            "chloeli/llama-3.1-8b-pro-affordability-spec-msm",
            "chloeli/llama-3.1-8b-cheese-aft",
        ])))
        print(json.dumps(r, indent=2))
    elif action == "behavioral":
        r = _await(behavioral_eval.spawn())
        print(json.dumps(r, indent=2))
    elif action == "noise_floor":
        # Vary ONLY the seed (data order); everything else matches the seed-42
        # assistant run already on the volume.
        fcs = {"s43": train_cheese.spawn(arm="msm_A__aft", supervise="assistant",
                                         seed=43),
               "bs8": train_cheese.spawn(arm="msm_A__aft", supervise="assistant",
                                         batch_size=8),
               "bs32": train_cheese.spawn(arm="msm_A__aft", supervise="assistant",
                                          batch_size=32)}
        for k, fc in fcs.items():
            print(f"spawned {k}: {fc.object_id}", flush=True)
        res = {k: _await(fc) for k, fc in fcs.items()}
        for k, r in res.items():
            g = r.get("gate", {}).get("delta_cosine", {})
            n = r.get("gate", {}).get("delta_norm_ratio", {})
            print(f"{k}: steps={r.get('steps')} vs_released_cos="
                  f"{g.get('mean') and round(g['mean'],4)} "
                  f"norm_ratio={n.get('mean') and round(n['mean'],3)}", flush=True)
        cmp = compare_runs.remote(
            "msm_A__aft__assistant__bs16__lr0.0001__s42",
            "msm_A__aft__assistant__bs16__lr0.0001__s43")
        print("\n=== NOISE FLOOR (seed 42 vs 43, all else identical) ===")
        print(json.dumps(cmp["ours_vs_ours"], indent=2))
    elif action == "source_cheese":
        print(json.dumps(_await(source_cheese.spawn()), indent=2))
    elif action == "train_masking_ab":
        # Settles STATUS.md open question #3 by measurement rather than
        # assumption: train the same arm under both conventions and keep
        # whichever better reproduces the released adapter's AFT step.
        fcs = {sup: train_cheese.spawn(arm="msm_A__aft", supervise=sup)
               for sup in ("assistant", "all")}
        for sup, fc in fcs.items():
            print(f"spawned {sup}: {fc.object_id}", flush=True)
        out = {sup: _await(fc) for sup, fc in fcs.items()}
        for sup, r in out.items():
            g = r.get("gate", {}).get("delta_cosine")
            print(f"\n{sup}: status={r.get('status')} "
                  f"delta_cos_mean={g and round(g['mean'], 4)}")
        print(json.dumps(out, indent=2))
    else:
        raise SystemExit(f"unknown action: {action!r}")
