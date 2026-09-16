"""Mirror finished pilot artifacts to private Hugging Face repos, so work can
resume from another pod.

  Taywon/aft-pilot-qwen3-14b  (model)   runs/<tag>/<run>/  LoRA adapters + train logs/meta
  Taywon/aft-pilot-data       (dataset) l3/...             L3 rewrite data, judge outputs, ledger
                                        evals/...          AM transcripts, scores, summaries

Only COMPLETED items are uploaded (adapter: train_meta.json exists; eval:
pilot_summary.json exists). A local marker records what was pushed, so the
loop is cheap and idempotent.

  python -m pilot.hf_sync            # one pass
  python -m pilot.hf_sync --loop 600 # every 10 min
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from huggingface_hub import HfApi

CK = Path(os.environ["RISELAB_CKPT"]) / "model_spec_midtraining" / "aft-pilot"
DATA = Path(os.environ["RISELAB_DATA"]) / "model_spec_midtraining"
MODEL_REPO = "Taywon/aft-pilot-qwen3-14b"
DATA_REPO = "Taywon/aft-pilot-data"
MARK = CK / "hf_sync_state.json"


def state() -> dict:
    return json.loads(MARK.read_text()) if MARK.exists() else {}


def sig(p: Path) -> str:
    """Signature = newest mtime + total size of the folder."""
    files = [f for f in p.rglob("*") if f.is_file()]
    return f"{max((f.stat().st_mtime for f in files), default=0):.0f}-{sum(f.stat().st_size for f in files)}"


def push(api: HfApi, st: dict, key: str, folder: Path, repo: str, repo_type: str,
         path_in_repo: str, ignore: list[str] | None = None) -> None:
    s = sig(folder)
    if st.get(key) == s:
        return
    print(time.strftime("%H:%M:%S"), "upload", folder, "->", repo, path_in_repo, flush=True)
    api.upload_folder(folder_path=str(folder), repo_id=repo, repo_type=repo_type,
                      path_in_repo=path_in_repo, ignore_patterns=ignore or [],
                      commit_message=f"sync {path_in_repo}")
    st[key] = s
    MARK.write_text(json.dumps(st, indent=1))


def once(api: HfApi) -> None:
    st = state()
    for repo, rt in ((MODEL_REPO, "model"), (DATA_REPO, "dataset")):
        api.create_repo(repo, repo_type=rt, private=True, exist_ok=True)
    for run in sorted((CK / "runs").glob("*/*")):
        if (run / "train_meta.json").exists():
            rel = run.relative_to(CK)
            push(api, st, f"model:{rel}", run, MODEL_REPO, "model", str(rel))
    for ev in sorted((CK / "evals").glob("*/*")):
        if (ev / "pilot_summary.json").exists():
            rel = ev.relative_to(CK)
            push(api, st, f"eval:{rel}", ev, DATA_REPO, "dataset", str(rel))
    l3 = DATA / "l3"
    if l3.exists():
        push(api, st, "l3", l3, DATA_REPO, "dataset", "l3")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", type=int, default=0)
    a = ap.parse_args()
    api = HfApi()
    while True:
        try:
            once(api)
        except Exception as e:  # noqa: BLE001 - keep the loop alive
            print("sync error:", type(e).__name__, str(e)[:300], flush=True)
        if not a.loop:
            return
        time.sleep(a.loop)


if __name__ == "__main__":
    main()
