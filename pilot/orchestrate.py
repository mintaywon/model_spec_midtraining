"""Unattended GPU queue for the AFT pilot: train -> AM eval, serialized.

Picks the first job whose inputs exist and whose output is missing; waits when
only not-yet-ready jobs remain (e.g. L3 data still being judged). Every job is
idempotent (skips if its done-marker exists), so this can be killed and
restarted at any time.

  nohup python -m pilot.orchestrate --model Qwen/Qwen2.5-14B-Instruct --tag q25 > LOG &
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CK = Path(os.environ["RISELAB_CKPT"]) / "model_spec_midtraining" / "aft-pilot"
DATA = Path(os.environ["RISELAB_DATA"]) / "model_spec_midtraining"
ENV = {**os.environ, "VLLM_USE_FLASHINFER_SAMPLER": "0", "PYTHONUNBUFFERED": "1"}


def sh(cmd: list[str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    print(time.strftime("%H:%M:%S"), "RUN", " ".join(cmd), "->", log, flush=True)
    with open(log, "a") as f:
        r = subprocess.run(cmd, cwd=ROOT, env=ENV, stdout=f, stderr=subprocess.STDOUT)
    print(time.strftime("%H:%M:%S"), "EXIT", r.returncode, flush=True)
    return r.returncode


def jobs(a):
    py = sys.executable
    runs = CK / "runs" / a.tag
    evals = CK / "evals" / a.tag
    msm = runs / "msm"
    J = []

    def aft(name, task, seed, init=None, ready=lambda: True):
        out = runs / f"{name}_s{seed}"
        cmd = [py, "-m", "pilot.train", "--mode", "chat", "--task", task, "--model", a.model,
               "--seed", str(seed), "--token-budget", "49152", "--out", str(out)]
        if init:
            cmd += ["--init-adapter", str(init)]
        J.append(dict(name=f"train {name} s{seed}", done=out / "train_meta.json",
                      ready=ready, cmd=cmd, log=out / "train.log"))
        J.append(dict(name=f"eval {name} s{seed}", done=evals / f"{name}_s{seed}" / "pilot_summary.json",
                      ready=lambda o=out: (o / "train_meta.json").exists(),
                      cmd=["bash", "pilot/scripts/eval_one.sh", a.model, str(out),
                           str(evals / f"{name}_s{seed}"),
                           a.cot_thinking if task == "l1" else a.thinking],
                      log=evals / f"{name}_s{seed}" / "run.log"))

    l3_ready = lambda: (DATA / "l3" / a.l3_version / "accepted_ids.json").exists()
    msm_ready = lambda: (msm / "train_meta.json").exists()
    for seed in a.seeds:
        aft("l0", "l0", seed)
        if seed == a.seeds[0]:
            J.append(dict(name="train msm", done=msm / "train_meta.json", ready=lambda: True,
                          cmd=[py, "-m", "pilot.train", "--mode", "doc", "--model", a.model,
                               "--seed", "0", "--msm-tokens", str(a.msm_tokens), "--token-budget", "49152",
                               "--out", str(msm)], log=msm / "train.log"))
        aft("ref", "l0", seed, init=msm, ready=msm_ready)
        aft("l1", "l1", seed)
        aft("l3", f"l3:{a.l3_version}", seed, ready=l3_ready)
    return J


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--thinking", default="none", help="eval thinking mode, no-CoT conditions")
    ap.add_argument("--cot-thinking", default="none", help="eval thinking mode, L1 (CoT)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--msm-tokens", type=float, default=27e6)
    ap.add_argument("--l3-version", default="v2")
    a = ap.parse_args()
    failures: dict[str, int] = {}
    while True:
        J = jobs(a)
        todo = [j for j in J if not j["done"].exists() and failures.get(j["name"], 0) < 2]
        if not todo:
            print("ALL DONE (or failed twice):", failures, flush=True)
            return
        ready = [j for j in todo if j["ready"]()]
        if not ready:
            time.sleep(60)
            continue
        j = ready[0]
        rc = sh(j["cmd"], j["log"])
        if rc != 0 or not j["done"].exists():
            failures[j["name"]] = failures.get(j["name"], 0) + 1
            print("FAILED", j["name"], failures[j["name"]], flush=True)


if __name__ == "__main__":
    main()
