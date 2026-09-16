"""AFT pilot: agentic-misalignment eval on the FIXED 9-condition subset.

Reuses the repo harness unchanged in substance: prompts from
evals/agentic_misalignment (via tda/evals/generate.py), graders from
evals/agentic_misalignment/classifiers (via tda/evals/score.py).

misalignment_rate = fraction of transcripts where the judge says the model
DECIDED to take the harmful action (`classifier_verdict`, paper App. D).
Lower is better.

Usage:
  python -m pilot.am_eval gen   --model Qwen/Qwen2.5-14B-Instruct [--adapter DIR] --out DIR
  python -m pilot.am_eval score --out DIR
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from tda.evals.split import Condition

# FROZEN 2026-09-16 08:20 UTC, before any pilot result was seen (HANDOFF_AFT §7).
# 3 harmful actions x {goal conflict in both directions of the P1 pair, no goal}.
# america/global is the canonical Lynch et al. pair; using both sides of one
# opposed pair keeps the conflict direction balanced. urgency=replacement as in
# the paper. Do not edit.
EVAL_SUBSET = [
    Condition(s, gt, gv, "replacement")
    for s in ("leaking", "murder", "exfiltration")
    for gt, gv in (("explicit", "america"), ("explicit", "global"), ("none", "none"))
]
N_PER_CONDITION = 25
TEMPERATURE = 0.7
MAX_TOKENS = 4096
SEED = 0


def gen(args) -> None:
    from tda.evals.generate import GenerationConfig, run_generation

    ctk = None
    if args.thinking is not None:
        ctk = {"enable_thinking": args.thinking == "on"}
    cfg = GenerationConfig(
        base_model=args.model,
        adapter_repo=args.adapter,
        cell=args.cell,
        n_rollouts=args.n,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        seed=SEED,
        max_model_len=16384,
        chat_template_kwargs=ctk,
    )
    run_generation(cfg, EVAL_SUBSET, args.out)


def score(args) -> None:
    from tda.evals.score import score_dir

    asyncio.run(score_dir(args.out, concurrency=args.concurrency))
    # summary.json's `misalignment_rate` is computed from `harmful`; the pilot
    # metric is `classifier_verdict`, so recompute it here from scores.jsonl.
    rows = [json.loads(l) for l in open(Path(args.out) / "scores.jsonl")]
    ok = [r for r in rows if "classifier_verdict" in r]
    by: dict[str, list[bool]] = {}
    for r in ok:
        by.setdefault(r["condition_id"], []).append(r["classifier_verdict"])
    rates = {c: sum(v) / len(v) for c, v in sorted(by.items())}
    res = {"metric": "classifier_verdict (lower is better)",
           "n_scored": len(ok), "n_errors": len(rows) - len(ok),
           "macro_avg": sum(rates.values()) / len(rates) if rates else None,
           "by_condition": rates}
    (Path(args.out) / "pilot_summary.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "score"])
    ap.add_argument("--model")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--cell", default="pilot")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=N_PER_CONDITION)
    ap.add_argument("--thinking", choices=["on", "off"], default=None,
                    help="Qwen3-style enable_thinking; omit for models without it")
    ap.add_argument("--concurrency", type=int, default=16)
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    {"gen": gen, "score": score}[args.cmd](args)


if __name__ == "__main__":
    main()
