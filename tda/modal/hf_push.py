"""Upload trained LoRA adapters from the results volume to the Hugging Face Hub.

    modal run --detach tda/modal/hf_push.py --runs "repo-name=aft_ladder/<run>,..." [--public]

Repos are created PRIVATE under the token's own account unless --public is
passed. Runs on Modal so the ~1 GB adapters never touch the laptop.
"""

import modal

app = modal.App("msm-tda-hf-push")
image = modal.Image.debian_slim().pip_install("huggingface_hub==0.30.2")
results = modal.Volume.from_name("msm-tda-results")

CARD = """---
base_model: Qwen/Qwen2.5-32B-Instruct
library_name: peft
tags: [lora, model-spec-midtraining]
---

# {repo}

LoRA adapter (r=64, alpha=128, all attention + MLP projections) for
`Qwen/Qwen2.5-32B-Instruct`, from the compute-scale comparison in
https://github.com/mintaywon/model_spec_midtraining (`REPORT_SCALE.md`).

- **Run**: `{run}`
- **Init**: {init}
- **Task data**: `{task}` ({n_task} rows) + {n_it} instruction-mix rows (`chloeli/sft-it-mix`, `train_clean`)
- **Training**: 1 epoch, {steps} steps of 32 rows, {tokens:,} tokens, AdamW lr 1e-4, cosine, seed {seed}, assistant-only loss
- **Agentic-misalignment rate** (27 conditions x 25 rollouts, `classifier_verdict`, lower is better): see `REPORT_SCALE.md`

Research artifact; one training seed.
"""


@app.function(image=image, volumes={"/results": results},
              secrets=[modal.Secret.from_name("huggingface")], timeout=3 * 3600)
def push(runs: dict, private: bool = True) -> dict:
    import json
    import os
    from pathlib import Path

    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    user = api.whoami()["name"]
    out = {}
    for repo, rel in runs.items():
        d = Path("/results") / rel
        assert (d / "adapter_model.safetensors").exists(), f"no adapter in {d}"
        meta = json.loads((d / "train_meta.json").read_text())
        cfg = meta["config"]
        rid = f"{user}/{repo}"
        api.create_repo(rid, private=private, exist_ok=True)
        (Path("/tmp") / "README.md").write_text(CARD.format(
            repo=repo, run=meta.get("run_name", d.name),
            init=(f"continued from `{cfg['init_adapter']}` (two-stage)"
                  if cfg["init_adapter"] else "fresh LoRA (one-stage)"),
            task=Path(cfg["task_dataset"]).name, n_task=meta["n_task"],
            n_it=meta["n_examples"] - meta["n_task"], steps=meta["steps"],
            tokens=meta["tokens"], seed=cfg["seed"]))
        api.upload_folder(repo_id=rid, folder_path=str(d),
                          allow_patterns=["adapter_model.safetensors", "adapter_config.json",
                                          "train_meta.json"])
        api.upload_file(repo_id=rid, path_or_fileobj="/tmp/README.md",
                        path_in_repo="README.md")
        info = api.model_info(rid, files_metadata=True)
        out[rid] = {"private": info.private,
                    "files": {s.rfilename: s.size for s in info.siblings}}
        print(rid, out[rid], flush=True)
    return out


@app.local_entrypoint()
def main(runs: str, public: bool = False):
    pairs = dict(p.split("=", 1) for p in runs.split(","))
    # spawn, not remote: the client's gRPC deadline expires on a long upload
    call = push.spawn(pairs, private=not public)
    print(f"spawned -> {call.object_id}")
