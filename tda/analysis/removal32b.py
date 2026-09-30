"""32B removal test: one arm, or an arm against its random-k control.

    python -m tda.analysis.removal32b drop1320-ekfac-align-opponents drop1320-random

NAMING (CLAUDE.md §5.1): an OPPONENT hurts alignment, a PROPONENT helps it, as
scored by the method. Arm names are alignment-anchored (`-align-opponents`).
The first arm was run before that rule under the bare legacy slug
`drop1320-ekfac-opponents`; `LEGACY` maps the new name onto its files. It is
the same set: the 1,320 documents EK-FAC scores as most RAISING
logp(misaligned action).

Metric is `classifier_verdict` (CLAUDE.md §2b(2c)), NOT `harmful`. Rates are
reported for the full grid and separately for dev / held-out: the removal sets
were ranked on DEV queries, so held-out is the clean test (§4.2).

Rollouts are independent samples at temp 0.7, so the arm difference is an
unpaired two-proportion comparison; the SE is computed over CONDITION-level
rates as well, because rollouts within a condition share a prompt and the
binomial SE understates the noise if conditions differ in how they respond.
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _split() -> dict[str, str]:
    cfg = yaml.safe_load((ROOT / "tda/configs/eval_split.yaml").read_text())
    out = {}
    for side in ("dev", "held_out"):
        for scen, conds in cfg[side].items():
            for c in conds:
                out[f"{scen}_{c['goal_type']}-{c['goal_value']}_"
                    f"{cfg['urgency_type']}"] = side
    return out


# alignment-anchored name -> slug its result files were written under
LEGACY = {"drop1320-ekfac-align-opponents": "drop1320-ekfac-opponents"}


def load(arm: str) -> list[dict]:
    if arm.endswith(("-opponents", "-proponents")) and "-align-" not in arm:
        raise ValueError(
            f"{arm!r} is a bare polarity slug; its meaning is ambiguous "
            "(CLAUDE.md §5.1). Use the alignment-anchored name, e.g. "
            "drop1320-ekfac-align-opponents.")
    p = ROOT / f"results/removal32b/removal32b_{LEGACY.get(arm, arm)}_s42.scores.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def rate(rows: list[dict]) -> dict:
    ok = [r for r in rows if isinstance(r.get("classifier_verdict"), bool)]
    n, k = len(ok), sum(r["classifier_verdict"] for r in ok)
    p = k / n if n else float("nan")
    return {"n": n, "n_ungraded": len(rows) - n, "rate": p,
            "sem": math.sqrt(p * (1 - p) / n) if n else float("nan")}


def by_condition(rows: list[dict]) -> dict[str, float]:
    g = defaultdict(list)
    for r in rows:
        if isinstance(r.get("classifier_verdict"), bool):
            g[r["condition_id"]].append(r["classifier_verdict"])
    return {c: sum(v) / len(v) for c, v in g.items()}


def subsets(rows: list[dict]) -> dict[str, list[dict]]:
    sp = _split()
    missing = {r["condition_id"] for r in rows} - set(sp)
    if missing:
        raise KeyError(f"conditions absent from eval_split.yaml: {missing}")
    out = {"all": rows,
           "dev": [r for r in rows if sp[r["condition_id"]] == "dev"],
           "held_out": [r for r in rows if sp[r["condition_id"]] == "held_out"]}
    for s in ("leaking", "exfiltration", "murder"):
        out[s] = [r for r in rows if r["scenario"] == s]
    return out


def main(arm: str, control: str | None = None) -> dict:
    a = subsets(load(arm))
    c = subsets(load(control)) if control else None
    res = {}
    for name, rows in a.items():
        ra = rate(rows)
        line = (f"{name:13s} {arm}: {ra['rate']:.3f} ± {ra['sem']:.3f} "
                f"(n={ra['n']}, ungraded {ra['n_ungraded']})")
        res[name] = {"arm": ra}
        if c:
            rc = rate(c[name])
            d = ra["rate"] - rc["rate"]
            se = math.sqrt(ra["sem"] ** 2 + rc["sem"] ** 2)
            # condition-level paired difference: same 27 prompts in both arms
            ca, cc = by_condition(rows), by_condition(c[name])
            diffs = [ca[k] - cc[k] for k in ca if k in cc]
            m = sum(diffs) / len(diffs)
            sd = math.sqrt(sum((x - m) ** 2 for x in diffs) / (len(diffs) - 1))
            se_c = sd / math.sqrt(len(diffs))
            res[name].update(control=rc, diff=d, se_binomial=se,
                             z_binomial=d / se, diff_cond_mean=m,
                             se_cond=se_c, t_cond=m / se_c,
                             n_cond=len(diffs),
                             n_cond_arm_higher=sum(x > 0 for x in diffs))
            line += (f" | {control}: {rc['rate']:.3f} ± {rc['sem']:.3f}"
                     f" | diff {d:+.3f} (z={d / se:+.2f}; by-condition "
                     f"{m:+.3f} ± {se_c:.3f}, t={m / se_c:+.2f}, "
                     f"{res[name]['n_cond_arm_higher']}/{len(diffs)} higher)")
        print(line)
    out = ROOT / f"results/removal32b/compare_{arm}{'_vs_' + control if control else ''}.json"
    out.write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    main(*sys.argv[1:3])
