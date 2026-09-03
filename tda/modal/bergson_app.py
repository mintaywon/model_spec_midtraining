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
                 batch_size: int = 32, lr: float = 1e-4, seed: int = 42,
                 n_checkpoints: int = 6, tag: str = "",
                 data_tag: str = "", warmup: float = 0.05,
                 grad_accum: int = 8, init_run: str = "") -> dict:
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
    if init_run:
        # Continue OUR retrained MSM instead of the released adapter. Required
        # for multi-stage SOURCE: the checkpoint list must be one trajectory, so
        # AFT has to start exactly where our MSM run ended.
        cks = sorted((Path(CHEESE_DIR) / "runs" / init_run / "checkpoints")
                     .glob("checkpoint-*"),
                     key=lambda p: int(p.name.split("-")[1]))
        if not cks:
            raise FileNotFoundError(f"no checkpoints in run {init_run}")
        init_repo = str(cks[-1])
    name = tag or f"{arm}__{supervise}__bs{batch_size}__lr{lr:g}__s{seed}"
    work = Path(SCRATCH_DIR) / "train" / name
    keep = Path(CHEESE_DIR) / "runs" / name
    keep.mkdir(parents=True, exist_ok=True)

    data_dir = Path(CHEESE_DIR) / f"train_{supervise}" / "dataset"
    if data_tag:
        data_dir = Path(CHEESE_DIR) / data_tag / "dataset"
    if not data_dir.exists():
        raise FileNotFoundError(f"{data_dir} missing — run prep_cheese first")

    import json as _json
    n_rows = _json.loads(
        (data_dir.parent / "manifest.json").read_text())["n_samples"]
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
        # Appendix B.4: cosine, 5% warmup, weight decay 0.01. bergson defaults
        # warmup to 0, and the early high-LR steps dominate the step direction.
        "lr_schedule": {"lr": lr, "lr_scheduler_type": "cosine",
                        "warmup_steps": warmup},
        "weight_decay": 0.01,
        # IT rows run to 8192 tokens where cheese rows are ~70, so a 16-sequence
        # batch OOMs an 80 GB card. The binding allocation is the fp32 logits
        # tensor: 128,256 vocab x tokens-in-micro-batch x 4 B, measured at
        # 9.63 GiB for micro-batch 2. grad_accum 16 -> micro-batch 1 halves it.
        # Exact w.r.t. the full-batch gradient because dropout is off, so the
        # effective batch of 16 and the trajectory are unchanged.
        "grad_accum_steps": grad_accum,
        # torch.optim.AdamW semantics, not the metagradients defaults.
        # eps_root sits INSIDE the sqrt in torchopt, so the 1e-8 default adds
        # 1e-4 to the denominator and visibly moves the trajectory.
        "adam_beta1": 0.9, "adam_beta2": 0.999, "eps_root": 0.0,
        "save_mode": "interval", "save_interval": interval,
        "save_optimizer_state": "all",
        "grad_checkpointing": True,
        # ⚠️ REQUIRED for grad_checkpointing to do anything. bergson's trainer
        # calls model.eval() when train_mode is False (its default) and only
        # then gradient_checkpointing_enable(); transformers guards
        # checkpointing with `if self.gradient_checkpointing and self.training`,
        # so in eval mode it is a SILENT no-op. That is the same trap STATUS.md
        # §2 records for our own extract.py. Symptom here: 74 GiB already
        # allocated before the vocab softmax on an 8B LoRA run, then OOM even at
        # micro-batch 1. Safe because LoRA dropout is 0.0 and Llama-3.1 has no
        # architectural dropout, so the forward stays deterministic.
        "train_mode": True,
        "overwrite": True,
    }}]}

    cfg_path = work.parent / f"{name}.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    rc = _run([sys.executable, "-m", "bergson", str(cfg_path)])

    out: dict = {"arm": arm, "supervise": supervise, "batch_size": batch_size,
                 "lr": lr, "seed": seed, "warmup": warmup,
                 "grad_accum": grad_accum, "data_tag": data_tag,
                 "returncode": rc, "n_rows": n_rows,
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
    if init_run:
        # Our own MSM init shares no gauge with the released adapters, so a
        # delta-cosine against them would be meaningless. Skip the gate.
        out["gate"] = {"skipped": "init_run set; no shared init with released"}
    else:
        init = (load_file(hf_hub_download(init_repo,
                                          "adapter_model.safetensors"))
                if init_repo else None)
        out["gate"] = C.delta_cosine(ours, rel, init)
    out["status"] = "OK"

    (keep / "report.json").write_text(json.dumps(out, indent=2))
    results.commit()
    return out


@app.function(image=bergson_image, gpu="H100", volumes=VOLUMES,
              secrets=[hf_secret], timeout=24 * 3600,
              # Modal allows up to 3 TiB. All 7 projections at 8B is ~1.18 TB
              # fp32 / ~590 GB bf16 for 6 checkpoints, and ~2 TB / ~1 TB for the
              # 12-checkpoint multi-stage run — all inside the cap. The earlier
              # 1 TiB request is what made attention-only look forced at 8B; it
              # never was. 32B is genuinely over the cap (7.8 TB fp32) and is
              # the only place the restriction is real.
              ephemeral_disk=3 * 1024 * 1024)
def source_cheese(arm: str = "msm_A__aft", supervise: str = "assistant",
                  which: str = "target", segments: int = 3,
                  filter_modules: str | None = None,
                  hessian_dtype: str = "bf16",
                  query_aggregation: str = "mean",
                  damping: float = 0.1, tag: str = "") -> dict:
    """Run SOURCE over one trained cheese arm.

    Defaults to ALL SEVEN projections with bf16 factors.

    EK-FAC factor storage is sum_modules(d_in^2 + d_out^2) per checkpoint and
    the pipeline holds 6 raw + 3 aggregated + 3 eigenvector sets at once —
    measured at 98.6 GB for a 0.5B model, so ~1.18 TB fp32 / ~590 GB bf16 at 8B.
    LoRA does not shrink it: factors are sized by layer in/out dims, not adapter
    rank. That fits Modal's 3 TiB ephemeral disk, so attention-only is NOT
    needed at 8B — an earlier 1 TiB request is what made it look forced.

    Dropping the MLPs is worst precisely for midtraining-document attribution,
    where what is absorbed is knowledge and values and the MLP blocks are most
    implicated. Pass filter_modules="*.mlp.*" only for 32B, where 7.8 TB fp32 /
    3.9 TB bf16 genuinely exceeds the cap, and state it as a limitation there.
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
                   supervise: str = "assistant", k: int = 15,
                   data_tag: str = "") -> dict:
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
    train_dir = root / (data_tag or f"train_{supervise}")
    manifest = train_dir / "manifest.json"

    ds = load_from_disk(str(train_dir / "dataset"))
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

    # CLAUDE.md §5.1 null control, available only once the IT mix is in the
    # index: do unrelated instruction rows score as influential as task rows?
    # Needs the `source` column that prep_cheese_it persists.
    if "source" in ds.column_names:
        from tda.influence.source.scores import by_source
        out["by_source"] = by_source(df, list(ds["source"]))

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
def split_query_sets(attr_frac: float = 0.5, seed: int = 0) -> dict:
    """Split the union query set into america-only and afford-only.

    Two independent splits, both required.

    BY AXIS. The two released arms dissociate in OPPOSITE directions —
    pro-america scores 0.520/0.513 on (america, afford) and pro-affordability
    0.352/0.658. A query set averaging over both cancels the very contrast H1
    measures, which is what the first H1 run did.

    BY ROLE (attr vs eval). Attribution queries and behavioural evaluation must
    not share prompts: scoring influence against the same items later used to
    declare a causal win makes the validation partly self-fulfilling. Splitting
    within each axis keeps both halves stratified. `attr` feeds SOURCE; `eval`
    is reserved for behavioural_eval and the §5.4 removal test, and nothing may
    read `eval` during attribution.
    """
    from datasets import load_from_disk

    root = Path(CHEESE_DIR)
    meta = json.loads((root / "query_target" / "manifest.json").read_text())
    n_am = meta["n_america"]
    import numpy as np

    out = {}
    # One permutation per axis, shared by target and alternative, so a prompt's
    # two continuations never land on opposite sides of the attr/eval split.
    rng = np.random.default_rng(seed)
    ds0 = load_from_disk(str(root / "query_target" / "dataset"))
    axis_idx = {"america": np.arange(n_am),
                "afford": np.arange(n_am, len(ds0))}
    roles: dict = {}
    for tag, idx in axis_idx.items():
        perm = rng.permutation(idx)
        cut = int(round(len(perm) * attr_frac))
        roles[tag] = {"attr": sorted(perm[:cut].tolist()),
                      "eval": sorted(perm[cut:].tolist())}

    for which in ("target", "alternative"):
        ds = load_from_disk(str(root / f"query_{which}" / "dataset"))
        for tag in axis_idx:
            for role, sel in roles[tag].items():
                d = root / f"query_{tag}_{role}_{which}"
                d.mkdir(parents=True, exist_ok=True)
                sub = ds.select(sel)
                sub.save_to_disk(str(d / "dataset"))
                (d / "manifest.json").write_text(json.dumps(
                    {"n_samples": len(sub), "which": which, "axis": tag,
                     "role": role, "split_seed": seed,
                     "indices": sel, "from": f"query_{which}"}, indent=2))
                out[f"{tag}_{role}_{which}"] = len(sub)

    # Disjointness is the whole point; assert it rather than trust it.
    for tag in axis_idx:
        a, e = set(roles[tag]["attr"]), set(roles[tag]["eval"])
        assert not (a & e), f"{tag}: attr/eval overlap"
        assert len(a | e) == len(axis_idx[tag]), f"{tag}: lost prompts"
    out["disjoint_verified"] = True
    results.commit()
    return out


@app.function(image=bergson_image, gpu="H100", volumes=VOLUMES,
              secrets=[hf_secret], timeout=3600)
def it_probe(n: int = 256, batch_size: int = 8) -> dict:
    """Did the released AFT run also train on instruction data? (open question #4)

    The discriminator: AFT on cheese alone should make a model *worse* at
    generic instruction following (ordinary forgetting), so NLL on instruction
    data should RISE from MSM-only to MSM+AFT. If the released adapter's NLL
    instead FALLS, its AFT stage saw instruction data that we do not have — and
    that is the leading explanation for the Stage-1 direction mismatch, where
    our step has the right magnitude (norm ratio 1.06) but the wrong direction
    (cos 0.11 against a 0.524 seed floor).

    Our own runs are the control: they never saw instruction data, so they must
    show forgetting under either hypothesis.
    """
    import numpy as np
    import torch
    from datasets import load_dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from tda.influence.bergson_data import tokenize_chat
    from tda.influence.masking import IGNORE_INDEX
    from tda.influence.source import cheese as C

    tok = AutoTokenizer.from_pretrained(C.ARMS["msm_A__aft"][0])
    raw = load_dataset("HuggingFaceH4/no_robots", split="test")
    samples, kept = [], 0
    for r in raw:
        msgs = [m for m in r["messages"] if m["role"] in ("user", "assistant")]
        if len(msgs) < 2 or msgs[0]["role"] != "user":
            continue
        ts = tokenize_chat(msgs[:2], tok, supervise="assistant", max_length=1024)
        if ts.n_supervised == 0 or len(ts.input_ids) > 1024:
            continue
        samples.append(ts)
        kept += 1
        if kept >= n:
            break

    names = ["chloeli/llama-3.1-8b-pro-america-spec-msm",
             "chloeli/llama-3.1-8b-pro-america-spec-msm-cheese-aft",
             "msm_A__aft__assistant__bs16__lr0.0001__s42"]
    root = Path(CHEESE_DIR)

    def resolve(a):
        local = root / "runs" / a / "checkpoints"
        if local.exists():
            cks = sorted(local.glob("checkpoint-*"),
                         key=lambda p: int(p.name.split("-")[1]))
            return str(cks[-1])
        return a

    @torch.no_grad()
    def nll(model):
        tot_lp = tot_tok = 0.0
        for i in range(0, len(samples), batch_size):
            chunk = samples[i:i + batch_size]
            L = max(len(c.input_ids) for c in chunk)
            ids = torch.zeros((len(chunk), L), dtype=torch.long)
            lab = torch.full((len(chunk), L), IGNORE_INDEX, dtype=torch.long)
            for j, c in enumerate(chunk):
                ids[j, :len(c.input_ids)] = torch.tensor(c.input_ids)
                lab[j, :len(c.labels)] = torch.tensor(c.labels)
            ids, lab = ids.to(0), lab.to(0)
            lp = torch.log_softmax(model(input_ids=ids).logits.float()[:, :-1], -1)
            t, m = lab[:, 1:], lab[:, 1:] != IGNORE_INDEX
            tokp = lp.gather(-1, t.clamp_min(0).unsqueeze(-1)).squeeze(-1)
            tot_lp += float((tokp * m).sum())
            tot_tok += float(m.sum())
        return -tot_lp / tot_tok

    out = {"n_samples": len(samples), "dataset": "HuggingFaceH4/no_robots",
           "nll": {}}
    for a in names:
        base = AutoModelForCausalLM.from_pretrained(
            C.BASE_MODEL, dtype=torch.bfloat16, device_map={"": 0})
        m = PeftModel.from_pretrained(base, resolve(a)).eval()
        out["nll"][a] = nll(m)
        print(f"{a}: {out['nll'][a]:.4f}", flush=True)
        del m, base
        torch.cuda.empty_cache()

    msm = out["nll"][names[0]]
    out["released_delta_vs_msm"] = out["nll"][names[1]] - msm
    out["ours_delta_vs_msm"] = out["nll"][names[2]] - msm
    out["verdict"] = (
        "released IMPROVED on instruction data -> its AFT saw instruction data"
        if out["released_delta_vs_msm"] < -0.02 else
        "both forgot -> no evidence of an instruction mix in the released AFT")
    print(f"\nreleased delta: {out['released_delta_vs_msm']:+.4f}  "
          f"ours delta: {out['ours_delta_vs_msm']:+.4f}\n{out['verdict']}",
          flush=True)
    (root / "it_probe.json").write_text(json.dumps(out, indent=2))
    results.commit()
    return out


# The §3 (cheese) instruction mix, from Appendix B.3 verbatim:
#
#   "For §3 experiments, we use 2M tokens of a simple instruction-tuning mix
#    that only contains the No Robots dataset and 4,000 formatted variants of
#    MMLU ... We include 2,500 samples of a synthetically generated
#    conversational dataset that teaches the base model basic identity
#    information about itself."
#
# ⚠️ THIS IS NOT TABLE 2. Table 2 (10,000 samples across nine sources) is
# labelled "used in §4-5 experiments" — the philosophy/Qwen work. Cheese is §3
# and uses this simpler mix. An earlier reconstruction here used Table 2 and was
# therefore the wrong data for this experiment.
#
# Resolving the counts: 13,500 total - 4,000 MMLU - 2,500 identity = 7,000 No
# Robots. sft-it-mix carries mmlu_binary (2,000) + mmlu_explain (2,000) =
# exactly the "4,000 formatted variants of MMLU", and neither appears in Table 2
# — which is why they sat unused until now.
#
# Token check: 7,000 no_robots ~1.30M + mmlu ~0.37M = ~1.67M, plus ~2,500
# identity samples at ~130 tokens ~0.33M -> ~2.0M, matching the paper.
#
# 🔴 NOT REPRODUCIBLE: the 2,500-sample synthetic identity dataset is
# unpublished (it is what the `id-baseline` checkpoints are named for). We are
# missing ~19% of IT samples and ~16% of IT tokens.
IT_MIX_S3 = {"no_robots": 7_000, "mmlu_binary": 2_000, "mmlu_explain": 2_000}
IT_SUBSAMPLE_SEED = 0


@app.function(image=bergson_image, volumes=VOLUMES, secrets=[hf_secret],
              timeout=5400, cpu=8, memory=65536)
def prep_cheese_it(max_length: int = 4096) -> dict:
    """Cheese AFT + the paper's instruction mix, pre-tokenized as one set.

    The paper trains the cheese run on 165k tokens of cheese AND ~2M tokens of
    instruction data — IT outnumbers the task data ~12:1 by token. Our first
    runs used cheese alone, i.e. 8% of the tokens, which is why the AFT step had
    the right magnitude (norm ratio 1.06) but a near-orthogonal direction
    (cos 0.11 against a 0.524 seed floor).
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    from tda.influence.bergson_data import save_for_bergson
    from tda.influence.source import cheese as C

    tok = AutoTokenizer.from_pretrained(C.ARMS["msm_A__aft"][0])
    samples, counts = [], {}

    aft = load_dataset(C.AFT_DATASET, split="train")
    cheese = C.build_train_samples(list(aft), tok, supervise="assistant",
                                   max_length=max_length)
    for t in cheese:
        t.meta["source"] = "cheese"
    samples += cheese
    counts["cheese"] = {"n": len(cheese),
                        "supervised": sum(t.n_supervised for t in cheese)}

    # max_length 4096 is the PAPER'S value for §3 (Appendix B.4: "We use 4096
    # max sequence length for experiments in §3"), not a deviation. 8192 applies
    # only to §4-5, which use the long-context Table 2 mix. It also happens to
    # avoid the OOM below.
    # The IT tail OOMs an 80 GB card: two 8192-token rows in one micro-batch
    # need 2 x 8192 x 128,256 x 4 B = 8.4 GB of fp32 logits plus its softmax
    # copy. The tail is tiny — LongAlign is 213 of ~15,129 rows and p99 is ~8k
    # tokens, while every cheese row is <=165.
    #
    # MEASURED EFFECT of the 4096 cap: LongAlign drops out ENTIRELY (0 of 213)
    # — its prompts run past 4096, so the assistant turn falls outside the
    # window and every row has zero supervised tokens. All eight other sources
    # are unchanged to the row, and total IT tokens fall 1.5% (2.226M vs
    # 2.260M). So the deviation is precisely "LongAlign removed", which cannot
    # plausibly affect the question this run asks (does adding the IT mix move
    # the AFT step direction at all).
    got = []
    for split, n_take in IT_MIX_S3.items():
        ds = load_dataset("chloeli/sft-it-mix", split=split)
        if n_take < len(ds):
            ds = ds.shuffle(seed=IT_SUBSAMPLE_SEED).select(range(n_take))
        for i, r in enumerate(ds):
            try:
                t = C.tokenize_chat_it(r["messages"], tok, max_length)
            except Exception:
                continue
            if t.n_supervised == 0:
                continue
            t.meta.update(source=split, row=i,
                          truncated=len(t.input_ids) >= max_length)
            got.append(t)
    samples += got
    by_src: dict = {}
    for t in got:
        d = by_src.setdefault(t.meta["source"], {"n": 0, "supervised": 0})
        d["n"] += 1
        d["supervised"] += t.n_supervised
    counts.update(by_src)

    info = save_for_bergson(
        samples, Path(CHEESE_DIR) / "train_it",
        manifest={"it_mix": IT_MIX_S3, "paper_section": "3 (cheese)",
                  "max_length": max_length,
                  "n_truncated": sum(1 for t in got if t.meta["truncated"]),
                  "missing": ("2,500-sample synthetic identity dataset "
                              "(unpublished) ~19% of IT samples"),
                  "it_subsample_seed": IT_SUBSAMPLE_SEED,
                  "per_source": counts, "supervise": "assistant",
                  "missing": "synthetic identity dataset (~3.5k), unpublished"},
    )
    info["per_source"] = counts
    results.commit()
    return info


@app.function(image=bergson_image, gpu="H100", volumes=VOLUMES,
              secrets=[hf_secret], timeout=3600)
def it_probe(n: int = 256, batch_size: int = 8) -> dict:
    """Did the released AFT run also train on instruction data? (open question #4)

    The discriminator: AFT on cheese alone should make a model *worse* at
    generic instruction following (ordinary forgetting), so NLL on instruction
    data should RISE from MSM-only to MSM+AFT. If the released adapter's NLL
    instead FALLS, its AFT stage saw instruction data that we do not have — and
    that is the leading explanation for the Stage-1 direction mismatch, where
    our step has the right magnitude (norm ratio 1.06) but the wrong direction
    (cos 0.11 against a 0.524 seed floor).

    Our own runs are the control: they never saw instruction data, so they must
    show forgetting under either hypothesis.
    """
    import numpy as np
    import torch
    from datasets import load_dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from tda.influence.bergson_data import tokenize_chat
    from tda.influence.masking import IGNORE_INDEX
    from tda.influence.source import cheese as C

    tok = AutoTokenizer.from_pretrained(C.ARMS["msm_A__aft"][0])
    raw = load_dataset("HuggingFaceH4/no_robots", split="test")
    samples, kept = [], 0
    for r in raw:
        msgs = [m for m in r["messages"] if m["role"] in ("user", "assistant")]
        if len(msgs) < 2 or msgs[0]["role"] != "user":
            continue
        ts = tokenize_chat(msgs[:2], tok, supervise="assistant", max_length=1024)
        if ts.n_supervised == 0 or len(ts.input_ids) > 1024:
            continue
        samples.append(ts)
        kept += 1
        if kept >= n:
            break

    names = ["chloeli/llama-3.1-8b-pro-america-spec-msm",
             "chloeli/llama-3.1-8b-pro-america-spec-msm-cheese-aft",
             "msm_A__aft__assistant__bs16__lr0.0001__s42"]
    root = Path(CHEESE_DIR)

    def resolve(a):
        local = root / "runs" / a / "checkpoints"
        if local.exists():
            cks = sorted(local.glob("checkpoint-*"),
                         key=lambda p: int(p.name.split("-")[1]))
            return str(cks[-1])
        return a

    @torch.no_grad()
    def nll(model):
        tot_lp = tot_tok = 0.0
        for i in range(0, len(samples), batch_size):
            chunk = samples[i:i + batch_size]
            L = max(len(c.input_ids) for c in chunk)
            ids = torch.zeros((len(chunk), L), dtype=torch.long)
            lab = torch.full((len(chunk), L), IGNORE_INDEX, dtype=torch.long)
            for j, c in enumerate(chunk):
                ids[j, :len(c.input_ids)] = torch.tensor(c.input_ids)
                lab[j, :len(c.labels)] = torch.tensor(c.labels)
            ids, lab = ids.to(0), lab.to(0)
            lp = torch.log_softmax(model(input_ids=ids).logits.float()[:, :-1], -1)
            t, m = lab[:, 1:], lab[:, 1:] != IGNORE_INDEX
            tokp = lp.gather(-1, t.clamp_min(0).unsqueeze(-1)).squeeze(-1)
            tot_lp += float((tokp * m).sum())
            tot_tok += float(m.sum())
        return -tot_lp / tot_tok

    out = {"n_samples": len(samples), "dataset": "HuggingFaceH4/no_robots",
           "nll": {}}
    for a in names:
        base = AutoModelForCausalLM.from_pretrained(
            C.BASE_MODEL, dtype=torch.bfloat16, device_map={"": 0})
        m = PeftModel.from_pretrained(base, resolve(a)).eval()
        out["nll"][a] = nll(m)
        print(f"{a}: {out['nll'][a]:.4f}", flush=True)
        del m, base
        torch.cuda.empty_cache()

    msm = out["nll"][names[0]]
    out["released_delta_vs_msm"] = out["nll"][names[1]] - msm
    out["ours_delta_vs_msm"] = out["nll"][names[2]] - msm
    out["verdict"] = (
        "released IMPROVED on instruction data -> its AFT saw instruction data"
        if out["released_delta_vs_msm"] < -0.02 else
        "both forgot -> no evidence of an instruction mix in the released AFT")
    print(f"\nreleased delta: {out['released_delta_vs_msm']:+.4f}  "
          f"ours delta: {out['ours_delta_vs_msm']:+.4f}\n{out['verdict']}",
          flush=True)
    (root / "it_probe.json").write_text(json.dumps(out, indent=2))
    results.commit()
    return out


MSM_CORPUS = {"A": "chloeli/msm-llama-pro-america",
              "B": "chloeli/msm-llama-pro-affordability"}


@app.function(image=bergson_image, volumes=VOLUMES, secrets=[hf_secret],
              timeout=5400, cpu=8, memory=65536)
def prep_msm(arm: str = "A", max_length: int = 4096) -> dict:
    """Pre-tokenize an MSM document corpus for the multi-stage index."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    from tda.influence.bergson_data import save_for_bergson, tokenize_document
    from tda.influence.source import cheese as C

    tok = AutoTokenizer.from_pretrained(C.ARMS["msm_A__aft"][0])
    ds = load_dataset(MSM_CORPUS[arm], split="train")
    samples = []
    for i, r in enumerate(ds):
        t = tokenize_document(r["text"], tok, max_length=max_length,
                              meta={"row": i, "source": f"msm_{arm}",
                                    "domain": r.get("domain", "")})
        samples.append(t)

    info = save_for_bergson(
        samples, Path(CHEESE_DIR) / f"msm_{arm}",
        manifest={"corpus": MSM_CORPUS[arm], "arm": arm,
                  "max_length": max_length, "mode": "document-LM",
                  "n_truncated": sum(1 for t in samples if t.meta["truncated"])},
    )
    results.commit()
    return info


@app.function(image=bergson_image, gpu="H100", volumes=VOLUMES,
              secrets=[hf_secret], timeout=10800)
def train_msm(arm: str = "A", batch_size: int = 32, lr: float = 1e-4,
              seed: int = 42, n_checkpoints: int = 6,
              grad_accum: int = 16) -> dict:
    """Retrain the MSM stage with a trajectory.

    batch_size 32: Appendix B.4 names the hardware (8B on one 141 GB H200) and
    every other hyperparameter but NEVER the batch size, so it is inferred from
    filling that card at seq len 4096. It is the one free parameter left, and
    the leading suspect whenever our step magnitude misses.

    Required for multi-stage SOURCE: only the FINAL MSM adapter was released, so
    without this there are no midtraining checkpoints to span. MSM starts from a
    fresh LoRA on the base model (the released adapter's
    base_model_name_or_path is meta-llama/Llama-3.1-8B), so unlike the AFT gate
    there is no shared init and no gauge alignment — a delta-cosine comparison
    against the released adapter would be meaningless. The gate here is
    behavioural instead.
    """
    import shutil

    import yaml

    from tda.influence.source import cheese as C

    name = f"msm_{arm}__s{seed}"
    work = Path(SCRATCH_DIR) / "msm" / name
    keep = Path(CHEESE_DIR) / "runs" / name
    keep.mkdir(parents=True, exist_ok=True)

    data_dir = Path(CHEESE_DIR) / f"msm_{arm}" / "dataset"
    if not data_dir.exists():
        raise FileNotFoundError(f"{data_dir} missing — run prep_msm first")
    import json as _json
    n_rows = _json.loads(
        (data_dir.parent / "manifest.json").read_text())["n_samples"]
    steps = max(1, n_rows // batch_size)
    interval = max(1, steps // n_checkpoints)

    cfg = {"steps": [{"train": {
        "run_path": str(work),
        "model": C.BASE_MODEL,
        "peft_init_kwargs": (
            "r=64,lora_alpha=128,lora_dropout=0.0,"
            "target_modules=q_proj|k_proj|v_proj|o_proj|"
            "gate_proj|up_proj|down_proj"),
        "precision": "bf16",
        "data": {"dataset": str(data_dir)},
        "batch_size": batch_size,
        "num_epochs": 1,
        "seed": seed,
        "lr_schedule": {"lr": lr, "lr_scheduler_type": "cosine",
                        "warmup_steps": 0.05},
        "adam_beta1": 0.9, "adam_beta2": 0.999, "eps_root": 0.0,
        "weight_decay": 0.01,
        "grad_accum_steps": grad_accum,
        "save_mode": "interval", "save_interval": interval,
        "save_optimizer_state": "all",
        "grad_checkpointing": True,
        # See train_cheese: bergson's eval() call makes checkpointing a no-op.
        "train_mode": True,
        "overwrite": True,
    }}]}

    cfg_path = work.parent / f"{name}.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    rc = _run([sys.executable, "-m", "bergson", str(cfg_path)])

    out = {"arm": arm, "seed": seed, "n_rows": n_rows, "steps": steps,
           "save_interval": interval, "returncode": rc, "run": str(keep)}
    if rc != 0:
        out["status"] = "FAILED"
        (keep / "report.json").write_text(json.dumps(out, indent=2))
        results.commit()
        return out

    from bergson.utils.trainer_export import export_checkpoints
    exported = export_checkpoints(work, overwrite=True)
    dst = keep / "checkpoints"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(work / "exported", dst)
    out["kept_checkpoints"] = sorted(p.name for p in dst.iterdir())
    out["status"] = "OK"
    (keep / "report.json").write_text(json.dumps(out, indent=2))
    results.commit()
    return out


@app.function(image=bergson_image, gpu="H100", volumes=VOLUMES,
              secrets=[hf_secret], timeout=24 * 3600,
              ephemeral_disk=3 * 1024 * 1024)
def source_multistage(msm_run: str = "msm_A__s42",
                      aft_run: str = "msm_A__chained",
                      index: str = "msm_A", which: str = "america_attr_target",
                      aft_data: str = "train_it",
                      segments: int = 2, max_ckpts_per_stage: int = 4,
                      hessian_dtype: str = "bf16",
                      filter_modules: str | None = None,
                      damping: float = 0.1, tag: str = "") -> dict:
    """SOURCE across the MIDTRAINING -> AFT boundary.

    This is the estimand the project actually wants: not "which midtraining
    documents matter at the end of midtraining", but "which midtraining
    documents change behaviour AFTER the fixed downstream AFT stage". Those are
    different quantities, and only a trajectory spanning both stages estimates
    the second. A single-stage run — however good its Hessian — cannot, which is
    why using SOURCE and spanning one stage forfeits the reason to use SOURCE.

    Construction:
      * checkpoints = the MSM run's, then the AFT run's, in trajectory order.
        The AFT run must have been trained with init_run=<msm_run> so the two
        are literally one trajectory.
      * L = 2 by default, ONE SEGMENT PER STAGE. This is the construction the
        SOURCE paper gives for exactly this case (Bae et al. §3.3, p10): "in a
        case where the model was sequentially trained with two datasets D1 and
        D2, Source can compute the contribution of a data point z_m in D1 ... by
        partitioning the training trajectory into two segments (L = 2) and
        computing the expected total derivative at the first segment with
        -(1/N_1) S_2 r_1". The stage boundary is therefore also the only segment
        boundary, so nothing straddles it and the masking below is exact.
        L is separately a fidelity knob (p8: raise it when E[H] or E[g] move
        quickly inside a segment, at higher cost) — within a stage our LR decays
        ~10x under cosine, so L=4 is a defensible refinement and worth running
        as a sensitivity check. But match the reference construction first.
      * the index is the midtraining corpus, and the query is taken at the final
        AFT checkpoint.

    bergson scores the index at every checkpoint, including the AFT ones where
    midtraining documents were not being trained. Those segments are dropped;
    the surviving sum is the multi-stage score, because bergson's backward walk
    has already pulled the query gradient back THROUGH the AFT segments before
    it reaches the midtraining ones.
    """
    import math
    import shutil

    import yaml

    run_name = tag or f"{msm_run}__x__{aft_run}__{which}"
    runs = Path(CHEESE_DIR) / "runs"

    def ckpts_of(run: str) -> list[Path]:
        d = runs / run / "checkpoints"
        if not d.exists():
            raise FileNotFoundError(f"{d} missing")
        return sorted(d.glob("checkpoint-*"),
                      key=lambda p: int(p.name.split("-")[1]))

    msm_ck, aft_ck = ckpts_of(msm_run), ckpts_of(aft_run)
    # BUDGET (CLAUDE.md §2b(0)): cost scales with 3 x n_checkpoints data passes
    # over a 9.5M-token corpus, so 12 checkpoints is ~11 h / $51 and 8 is
    # ~7.8 h / $35. 8 keeps 2 segments per stage — SOURCE still segments each
    # stage rather than collapsing it — and holds the session under $100.
    seg_per_stage = segments // 2
    per_seg_target = max(1, max_ckpts_per_stage // seg_per_stage)
    keep = seg_per_stage * per_seg_target

    def select(cks: list[Path], n: int) -> list[Path]:
        """Spread n checkpoints EVENLY across the stage, dropping step 0.

        Two things this gets right that a tail-trim (`cks[-n:]`) does not:

        * Under L=2 a segment represents a WHOLE stage, and its stationary
          statistics H_l, g_l are meant to be estimated across that stage. Taking
          the last n omits the early high-LR steps where most of the learning
          happens — a half-stage estimate wearing a full-stage label. (A tail
          trim was fine at L=4 with 6 checkpoints per stage; it silently became
          wrong when L dropped to 2.)
        * Step 0 must go. PEFT initialises lora_B to zero, so at step 0
          grad_A = (B^T g) x^T is identically zero — the degenerate case
          STATUS.md records from gradients.py. Its statistics are meaningless.
        """
        pool = [c for c in cks if int(c.name.split("-")[1]) > 0]
        if len(pool) <= n:
            return pool
        idx = [round(i * (len(pool) - 1) / (n - 1)) for i in range(n)] if n > 1 \
            else [len(pool) - 1]
        return [pool[i] for i in sorted(set(idx))]

    msm_ck, aft_ck = select(msm_ck, keep), select(aft_ck, keep)
    if len(msm_ck) != len(aft_ck):
        raise ValueError(
            f"stages must contribute equal checkpoint counts for the segment "
            f"boundary to align: MSM {len(msm_ck)} vs AFT {len(aft_ck)}")
    ckpts = msm_ck + aft_ck

    local = Path(SCRATCH_DIR) / "ms_ckpts" / run_name
    local.mkdir(parents=True, exist_ok=True)
    staged = []
    for i, c in enumerate(ckpts):
        d = local / f"checkpoint-{i:03d}"
        if not d.exists():
            shutil.copytree(c, d)
        staged.append(d)

    # Per-segment lr x steps, computed within each stage's OWN cosine schedule.
    def stage_lrs(cks: list[Path], n_seg: int, lr_max: float = 1e-4):
        steps_at = [int(p.name.split("-")[1]) for p in cks]
        total = steps_at[-1]
        per = len(cks) // n_seg
        bounds = [0] + [steps_at[(i + 1) * per - 1] for i in range(n_seg)]
        lrs, sizes = [], []
        for i in range(n_seg):
            a, b = bounds[i], bounds[i + 1]
            sizes.append(max(b - a, 1))
            lrs.append(sum(lr_max * 0.5 * (1 + math.cos(math.pi * min(t, total) / total))
                           for t in range(a, b)) / max(b - a, 1))
        return lrs, sizes

    # One dataset per segment, in trajectory order: midtraining segments get the
    # MSM corpus, AFT segments the AFT mixture.
    segment_datasets = (
        [str(Path(CHEESE_DIR) / index / "dataset")] * seg_per_stage
        + [str(Path(CHEESE_DIR) / aft_data / "dataset")] * seg_per_stage
    )

    lr_msm, sz_msm = stage_lrs(msm_ck, seg_per_stage)
    lr_aft, sz_aft = stage_lrs(aft_ck, seg_per_stage)
    lr_list, step_size_list = lr_msm + lr_aft, sz_msm + sz_aft
    msm_segments = list(range(seg_per_stage))

    work = Path(SCRATCH_DIR) / "ms" / run_name
    keep_dir = Path(CHEESE_DIR) / "multistage" / run_name
    keep_dir.mkdir(parents=True, exist_ok=True)

    cfg = {"steps": [{"approxunrolling": {
        "index_cfg": {
            "run_path": str(work),
            "model": str(staged[-1]),
            "precision": "bf16",
            "token_batch_size": 8192,
            "overwrite": True,
            "data": {"dataset": str(Path(CHEESE_DIR) / index / "dataset")},
            **({"filter_modules": filter_modules} if filter_modules else {}),
        },
        "hessian_cfg": {"method": "kfac", "hessian_dtype": hessian_dtype,
                        "ev_correction": True},
        "approx_unrolling_cfg": {
            "checkpoints": [str(p) for p in staged],
            "segments": segments,
            "lr_list": lr_list,
            "step_size_list": step_size_list,
            "query": {"dataset": str(Path(CHEESE_DIR) /
                                     f"query_{which}" / "dataset")},
            "query_aggregation": "mean",
            "use_adam_preconditioner": True,
            "inversion_cfg": {"damping_factor": damping},
            # [patched field] Each segment's Hessian is estimated on the data
            # that segment actually trained on — MSM documents for the
            # midtraining segment, the AFT mixture for the AFT segment. Without
            # this bergson uses one dataset everywhere, so `S_2` (the pullback
            # through AFT) would be built from midtraining curvature.
            "segment_datasets": segment_datasets,
        },
    }}]}

    cfg_path = work.parent / f"{run_name}.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    t0 = time.time()
    rc = _run([sys.executable, "-m", "bergson", str(cfg_path)])

    out = {"msm_run": msm_run, "aft_run": aft_run, "index": index,
           "segment_datasets": segment_datasets,
           "which": which, "segments": segments,
           "msm_segments": msm_segments, "n_checkpoints": len(staged),
           "lr_list": lr_list, "step_size_list": step_size_list,
           "returncode": rc, "minutes": round((time.time() - t0) / 60, 1)}

    if rc == 0:
        from tda.influence.source.scores import stage_masked_score
        import numpy as np
        score, meta = stage_masked_score(work, segments, msm_segments)
        np.save(keep_dir / "multistage_score.npy", score)
        out["masking"] = meta
        for sub in ("scores",):
            src = work / sub
            if src.exists():
                dst = keep_dir / sub
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
        out["status"] = "OK"
    else:
        out["status"] = "FAILED"

    (keep_dir / "report.json").write_text(json.dumps(out, indent=2))
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
def main(action: str = "verify", runs: str = ""):
    if action == "verify":
        print(json.dumps(_await(verify.spawn()), indent=2))
    elif action == "smoke":
        fc = smoke_source.spawn()
        print(f"spawned smoke_source: {fc.object_id}", flush=True)
        print(json.dumps(_await(fc), indent=2))
    elif action == "prep_cheese":
        print(json.dumps(_await(prep_cheese.spawn()), indent=2))
    elif action == "profiles":
        r = _await(compare_profiles.spawn(runs=runs) if runs
                   else compare_profiles.spawn())
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
    elif action == "multistage":
        # The train_it dataset currently on the volume is the TABLE 2 (§4-5)
        # mix; cheese is §3 and needs the simple mix. Re-prep before training.
        pr = _await(prep_cheese_it.spawn())
        print(f"prep AFT (§3 mix): n={pr['n_samples']} "
              f"supervised={pr['supervised_total']:,}", flush=True)
        for src, c in sorted(pr["per_source"].items(),
                             key=lambda kv: -kv[1]["n"]):
            print(f"    {src:<18} n={c['n']:>6} tok={c['supervised']:>9,}",
                  flush=True)
        # AFT continued from OUR MSM so the two stages are one trajectory.
        t = _await(train_cheese.spawn(arm="msm_A__aft", supervise="assistant",
                                      data_tag="train_it",
                                      init_run="msm_A__s42",
                                      tag="msm_A__chained"))
        print(f"chained AFT: status={t.get('status')} steps={t.get('steps')}",
              flush=True)
        print(f"  checkpoints: {t.get('kept_checkpoints')}", flush=True)
        sq = _await(split_query_sets.spawn())
        print(f"query split: {sq}", flush=True)
        r = _await(source_multistage.spawn())
        print(json.dumps(r, indent=2))
    elif action == "msm":
        r = _await(prep_msm.spawn(arm="A"))
        print(f"prep_msm A: n={r['n_samples']} tokens={r['tokens_total']:,} "
              f"truncated={r.get('n_truncated')}", flush=True)
        t = _await(train_msm.spawn(arm="A"))
        print(f"train_msm A: status={t.get('status')} steps={t.get('steps')} "
              f"ckpts={t.get('kept_checkpoints')}", flush=True)
    elif action == "it_train":
        r = _await(prep_cheese_it.spawn())
        print(f"prep: n={r['n_samples']} supervised={r['supervised_total']:,}",
              flush=True)
        for src, c in sorted(r["per_source"].items(),
                             key=lambda kv: -kv[1]["n"]):
            print(f"    {src:<20} n={c['n']:>6} supervised={c['supervised']:>10,}",
                  flush=True)
        # msm_A ONLY. The gate is a single yes/no — does adding the IT mix move
        # delta-cosine off 0.11 toward the 0.524 floor — and msm_B is only
        # needed for H1, which is worth training only if the gate passes.
        # At ~8 h/arm this restraint is the difference between ~$20 and ~$64.
        fcs = {a: train_cheese.spawn(arm=f"msm_{a}__aft", supervise="assistant",
                                     data_tag="train_it", tag=f"msm_{a}__it")
               for a in ("A",)}
        for a, fc in fcs.items():
            print(f"spawned train msm_{a}: {fc.object_id}", flush=True)
        print("\n=== GATE: delta-cosine vs released (floor 0.52, was 0.11) ===")
        for a, fc in fcs.items():
            res = _await(fc)
            g = res.get("gate", {})
            dc, nr = g.get("delta_cosine", {}), g.get("delta_norm_ratio", {})
            print(f"msm_{a}: status={res.get('status')} steps={res.get('steps')} "
                  f"delta_cos={dc.get('mean') and round(dc['mean'],4)} "
                  f"norm_ratio={nr.get('mean') and round(nr['mean'],3)}",
                  flush=True)
    elif action == "it_probe":
        print(json.dumps(_await(it_probe.spawn()), indent=2))
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
