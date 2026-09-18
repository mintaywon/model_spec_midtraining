"""Parallel local driver for ODCV-Bench (McGill-DMaS) against an OpenAI-compatible
endpoint serving our checkpoints.

Replaces the benchmark's `docker compose` flow (fixed container names, one
scenario at a time) with per-run networks and names so several scenarios run
concurrently. Everything the agent sees is unchanged: the same executor image
(`mission_executor/agent_main.py`, 50 steps, bash + task_complete tools,
temperature 0.0), the same per-scenario sandbox image (scenario files +
`orchestrator_api`), the internal-only network for the sandbox.

    python -m tda.evals.odcv_run --base-url https://.../v1 --api-key KEY \
        --models base,L0-ours,L2,Ref-ours,Ref-rel --variants mandated,incentivized \
        --concurrency 6 --out results/aft/odcv

Outputs per (model, variant, scenario): messages_record.txt (judge input),
run.log (executor stdout), meta.json. Resumable: existing outputs are skipped.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BENCH = Path(__file__).resolve().parents[2] / "external" / "ODCV-Bench"
EXEC_IMAGE = "odcv-executor"


def sh(cmd: list[str], check: bool = True, timeout: int | None = None, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=check, timeout=timeout, **kw)


def build_executor():
    sh(["docker", "build", "-q", "-t", EXEC_IMAGE, str(BENCH / "mission_executor")])


def orch_image_tag(variant: str, scenario: str) -> str:
    return f"odcv-orch-{variant}-{scenario}".lower().replace("_", "-")[:120]


def build_orch(variant: str, scenario: str) -> str:
    """Scenario sandbox image = scenario dir + orchestrator_api files (as run_benchmarks.py does)."""
    tag = orch_image_tag(variant, scenario)
    if sh(["docker", "image", "inspect", tag], check=False).returncode == 0:
        return tag
    src = BENCH / f"{variant}_scenarios" / scenario
    orch = BENCH / "orchestrator_api"
    with tempfile.TemporaryDirectory() as td:
        ctx = Path(td) / "ctx"
        shutil.copytree(src, ctx)
        for item in orch.iterdir():
            dst = ctx / item.name
            if item.is_dir():
                shutil.copytree(item, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dst)
        # The scenarios pin debian:bullseye-slim, whose security mirror now 404s
        # (archived); build on bookworm-slim instead. pip on bookworm needs the
        # externally-managed override. Nothing the agent sees changes.
        df = ctx / "Dockerfile"
        txt = df.read_text().replace("debian:bullseye-slim", "debian:bookworm-slim")
        txt = txt.replace("pip3 install -r", "pip3 install --break-system-packages -r")
        txt = txt.replace("pip3 install ", "pip3 install --break-system-packages ") if "--break-system-packages" not in txt else txt
        df.write_text(txt)
        sh(["docker", "build", "-q", "-t", tag, str(ctx)])
    return tag


def run_one(model: str, variant: str, scenario: str, base_url: str, api_key: str,
            out_root: Path, timeout_s: int = 2400) -> dict:
    out = out_root / model / variant / scenario
    if (out / "messages_record.txt").exists():
        return {"model": model, "variant": variant, "scenario": scenario, "status": "cached"}
    out.mkdir(parents=True, exist_ok=True)
    uid = uuid.uuid4().hex[:8]
    orch, exe = f"odcv-orch-{uid}", f"odcv-exec-{uid}"
    net_int, net_ext = f"odcv-int-{uid}", f"odcv-ext-{uid}"
    prompt = BENCH / f"{variant}_scenarios" / scenario / "prompt.json"
    t0 = time.time()
    status = "ok"
    try:
        tag = build_orch(variant, scenario)
        sh(["docker", "network", "create", "--internal", net_int])
        sh(["docker", "network", "create", net_ext])
        sh(["docker", "run", "-d", "--name", orch, "--network", net_int, tag])
        # wait for the sandbox API
        for _ in range(60):
            r = sh(["docker", "exec", orch, "python3", "-c",
                    "import urllib.request;urllib.request.urlopen('http://localhost:5000/get_auto_approve_status')"],
                   check=False)
            if r.returncode == 0:
                break
            time.sleep(2)
        else:
            raise RuntimeError("sandbox API never came up")
        sh(["docker", "run", "-d", "--name", exe, "--network", net_ext,
            "-v", f"{prompt}:/app/prompt.json:ro",
            "-e", f"OPENAI_BASE_URL={base_url}", "-e", f"OPENAI_API_KEY={api_key}",
            "-e", f"OPENAI_MODEL={model}", "-e", "OPENAI_TEMPERATURE=0.0",
            "-e", f"SUBMISSION_SERVER_URL=http://{orch}:5000/execute",
            "-e", "PYTHONUNBUFFERED=1", EXEC_IMAGE])
        sh(["docker", "network", "connect", net_int, exe])
        try:
            sh(["docker", "wait", exe], timeout=timeout_s)
        except subprocess.TimeoutExpired:
            status = "timeout"
            sh(["docker", "kill", exe], check=False)
        (out / "run.log").write_text(sh(["docker", "logs", exe], check=False).stdout)
        cp = sh(["docker", "cp", f"{exe}:/app/messages_record.txt", str(out / "messages_record.txt")], check=False)
        if cp.returncode != 0:
            status = status if status != "ok" else "no_record"
    except Exception as e:  # noqa: BLE001
        status = f"error: {type(e).__name__}: {str(e)[:200]}"
    finally:
        for c in (exe, orch):
            sh(["docker", "rm", "-f", c], check=False)
        for n in (net_int, net_ext):
            sh(["docker", "network", "rm", n], check=False)
    meta = {"model": model, "variant": variant, "scenario": scenario, "status": status,
            "elapsed_s": round(time.time() - t0, 1)}
    (out / "meta.json").write_text(json.dumps(meta))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--models", required=True, help="comma-separated served model names")
    ap.add_argument("--variants", default="mandated,incentivized")
    ap.add_argument("--scenarios", default="all", help="all | N (first N) | comma-separated names")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--out", default="results/aft/odcv")
    ap.add_argument("--timeout", type=int, default=2400)
    a = ap.parse_args()
    out_root = Path(a.out)
    build_executor()
    jobs = []
    for variant in a.variants.split(","):
        names = sorted(p.name for p in (BENCH / f"{variant}_scenarios").iterdir() if p.is_dir())
        if a.scenarios != "all":
            names = names[: int(a.scenarios)] if a.scenarios.isdigit() else [s for s in a.scenarios.split(",")]
        for model in a.models.split(","):
            for s in names:
                jobs.append((model, variant, s))
    print(f"{len(jobs)} runs, concurrency {a.concurrency}", flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        futs = [ex.submit(run_one, m, v, s, a.base_url, a.api_key, out_root, a.timeout) for m, v, s in jobs]
        for f in as_completed(futs):
            r = f.result(); done += 1
            print(f"[{done}/{len(jobs)}] {r['model']}/{r['variant']}/{r['scenario']}: {r['status']} "
                  f"({r.get('elapsed_s', 0)}s)", flush=True)


if __name__ == "__main__":
    main()
