"""Tabulate MACHIAVELLI summaries and join them to the AM full-grid numbers.

    python -m tda.evals.machiavelli_report --root results/aft/mach --run mach_ach \
        --am results/aft/full/final_all.json

Metrics are the benchmark's normalised counts (% of the random agent's mean for
the same game; 100 = random). Lower is better for every harm column;
`game.score` higher is better. Macro = mean over games. A bootstrap CI over
games (resampling the 30 games) is given for `violations.Σ`.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

COLS = ["violations.Σ", "power.Σ", "utility.suffering", "game.score"]
VIOL = ["violations.deception", "violations.manipulation", "violations.killing",
        "violations.physical_harm", "violations.non_physical_harm",
        "violations.intending_harm", "violations.fairness", "violations.promises",
        "violations.spying", "violations.stealing", "violations.trespassing",
        "violations.vandalism"]


def load(root: Path, run: str) -> dict[str, dict]:
    out = {}
    for p in sorted((root / run).glob("*/summary.json")):
        s = json.loads(p.read_text())
        out[p.parent.name] = s
    return out


def boot_ci(games: list[dict], key: str, n: int = 2000, seed: int = 0):
    vals = [g["scores"][key] for g in games if key in g["scores"]]
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(vals) for _ in vals) / len(vals) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/aft/mach")
    ap.add_argument("--run", default="mach_ach")
    ap.add_argument("--am", default="", help="final_all.json from tda.aft.report for the AM join")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    S = load(Path(a.root), a.run)
    am = json.loads(Path(a.am).read_text()) if a.am else {}
    am_key = {"Base": "Base", "id-baseline": "id-baseline", "L0-rel": "L0-rel", "L1-rel": "L1-rel",
              "Ref-rel": "Ref-rel", "L0-ours": "L0-ours", "PARA": "PARA", "L2INS": "L2INS",
              "L2TPINS": "L2TPINS", "L2": "L2", "L3": "L3", "Ref-ours-s42": "Ref-ours-s42",
              "Ref-ours-s43": "Ref-ours-s43"}
    print(f"\nMACHIAVELLI ({a.run}), % of random agent (100 = random); harms lower is better\n")
    print(f"{'arm':<14}{'games':>6}{'steps':>7}{'viol.Σ':>8}{'95% CI':>16}{'power.Σ':>9}{'suffer':>8}{'score':>7}{'parseF':>7}{'AM':>7}")
    rows = {}
    for cond, s in S.items():
        m = s["macro"]; lo, hi = boot_ci(s["games"], "violations.Σ")
        amv = am.get(am_key.get(cond, cond), {}).get("classifier_verdict", {}).get("macro", float("nan"))
        rows[cond] = {"n_games": s["n_games"], "mean_steps": s["mean_steps"], **{c: m.get(c) for c in COLS},
                      "violations_ci": [lo, hi], "parse_failures": s["parse_failures"], "am_macro": amv,
                      "by_violation": {k: m.get(k) for k in VIOL if k in m}}
        print(f"{cond:<14}{s['n_games']:>6}{s['mean_steps']:>7.0f}{m['violations.Σ']:>8.1f}"
              f"{'[' + f'{lo:.0f},{hi:.0f}' + ']':>16}{m['power.Σ']:>9.1f}{m['utility.suffering']:>8.1f}"
              f"{m['game.score']:>7.1f}{s['parse_failures']:>7}{amv:>7.3f}")
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
