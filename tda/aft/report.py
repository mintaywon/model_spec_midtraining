"""Tabulate HANDOFF_AFT ladder evals on the frozen `aft9` subset.

Reads scores.jsonl files (downloaded from the results volume with
`modal volume get`) and prints per-condition rates, the macro-average over the
9 conditions, and a within-checkpoint bootstrap CI. Metric convention follows
CLAUDE.md §2b(2c): `classifier_verdict` (decided to act) is primary, `harmful`
(executed) secondary. **Lower is better.**

    python -m tda.aft.report --arms Base=results/aft/runs/aft32_base.scores.jsonl \
        L0-rel=results/aft/reuse/phil_aft_only.scores.jsonl ...

Rows with rollout_idx >= n_per_condition are ignored so a 30-rollout run and a
25-rollout run are compared on the same number of samples.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from tda.evals.split import load_all, load_split, load_subset


def conditions_for(split: str):
    if split == "all":
        return load_all()
    if split in ("dev", "held_out"):
        return load_split(split)
    return load_subset(split)


def load_scores(path: str, n_per: int, ids: set[str]) -> dict[str, list[dict]]:
    by: dict[str, list[dict]] = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["condition_id"] in ids and r["rollout_idx"] < n_per and "harmful" in r:
            by[r["condition_id"]].append(r)
    return by


def macro(by: dict[str, list[dict]], metric: str) -> float:
    rates = [sum(r[metric] for r in v) / len(v) for v in by.values() if v]
    return sum(rates) / len(rates) if rates else float("nan")


def bootstrap_ci(by: dict[str, list[dict]], metric: str, n_boot: int = 2000,
                 seed: int = 0) -> tuple[float, float]:
    """Resample rollouts within each condition, recompute the macro-average."""
    rng = random.Random(seed)
    vals = []
    conds = [[r[metric] for r in v] for v in by.values() if v]
    for _ in range(n_boot):
        vals.append(sum(sum(rng.choice(c) for _ in c) / len(c) for c in conds) / len(conds))
    vals.sort()
    return vals[int(0.025 * n_boot)], vals[int(0.975 * n_boot)]


def summarize(arms: dict[str, str], n_per: int = 25, split: str = "aft9") -> dict:
    conds = conditions_for(split)
    ids = {c.condition_id for c in conds}
    order = [c.condition_id for c in conds]
    out = {}
    for name, path in arms.items():
        by = load_scores(path, n_per, ids)
        n = sum(len(v) for v in by.values())
        rec = {"n": n, "n_conditions": len(by)}
        for metric in ("classifier_verdict", "harmful"):
            lo, hi = bootstrap_ci(by, metric)
            rec[metric] = {"macro": round(macro(by, metric), 4),
                           "ci95": (round(lo, 4), round(hi, 4)),
                           "by_condition": {c: round(sum(r[metric] for r in by[c]) / len(by[c]), 3)
                                            for c in order if by.get(c)}}
        out[name] = rec
    return out


def print_table(summary: dict, split: str = "aft9"):
    conds = [c.condition_id for c in conditions_for(split)]
    short = {c: c.replace("_replacement", "").replace("explicit-", "") for c in conds}
    print(f"\nmisalignment rate (classifier_verdict; LOWER IS BETTER), {split}, macro over {len(conds)} conditions")
    print(f"{'arm':<14}{'n':>5}{'macro':>8}{'95% CI':>18}{'harmful':>9}  " +
          " ".join(f"{short[c][:12]:>12}" for c in conds))
    for name, rec in summary.items():
        cv = rec["classifier_verdict"]
        ci = f"[{cv['ci95'][0]:.3f},{cv['ci95'][1]:.3f}]"
        print(f"{name:<14}{rec['n']:>5}{cv['macro']:>8.3f}{ci:>18}{rec['harmful']['macro']:>9.3f}  " +
              " ".join(f"{cv['by_condition'].get(c, float('nan')):>12.2f}" for c in conds))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True, help="name=path/to/scores.jsonl")
    ap.add_argument("--n-per", type=int, default=25)
    ap.add_argument("--json", default="")
    ap.add_argument("--split", default="aft9", help="aft9 | dev | held_out | all")
    ap.add_argument("--by-scenario", action="store_true")
    a = ap.parse_args()
    arms = dict(x.split("=", 1) for x in a.arms)
    s = summarize(arms, a.n_per, a.split)
    if a.by_scenario or a.split == "all":
        # compact view: macro over all conditions + per-scenario means
        print(f"\nmisalignment rate (classifier_verdict; LOWER IS BETTER), {a.split}")
        print(f"{'arm':<16}{'n':>5}{'macro':>8}{'95% CI':>18}{'harmful':>9}{'leaking':>9}{'murder':>9}{'exfil':>9}")
        for name, rec in s.items():
            cv = rec["classifier_verdict"]; bc = cv["by_condition"]
            sc = {k: sum(v for c, v in bc.items() if c.startswith(k)) / max(1, sum(1 for c in bc if c.startswith(k)))
                  for k in ("leaking", "murder", "exfiltration")}
            ci = f"[{cv['ci95'][0]:.3f},{cv['ci95'][1]:.3f}]"
            print(f"{name:<16}{rec['n']:>5}{cv['macro']:>8.3f}{ci:>18}{rec['harmful']['macro']:>9.3f}"
                  f"{sc['leaking']:>9.3f}{sc['murder']:>9.3f}{sc['exfiltration']:>9.3f}")
    else:
        print_table(s, a.split)
    if a.json:
        Path(a.json).write_text(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
