"""Every MACHIAVELLI metric, not just violations.Σ (PLAN_OOD.md phase 4, direction 2).

    python -m tda.evals.machiavelli_metrics --root results/aft/mach --run mach_ach \
        --am results/aft/full/final_all.json --json results/aft/mach/metrics_mach_ach.json

For each metric the benchmark records (13 violation categories, 4 power types,
suffering, game score) and each arm:
  * macro over games (% of the random agent; 100 = random),
  * PAIRED per-game difference arm − Base with a bootstrap CI and a two-sided
    bootstrap p-value,
  * Benjamini-Hochberg q over the whole (metric × arm) family, because 19
    metrics × 12 arms will throw up nominal hits by chance.
Across arms: Spearman between each metric and the agentic-misalignment rate —
"does any metric carry the AM ordering?".

Raw counts are reported beside the normalised scores: a category the random
agent rarely triggers has a tiny denominator, so its percentage swings wildly
on one or two events.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def load(root: Path, run: str) -> dict[str, dict]:
    return {p.parent.name: json.loads(p.read_text()) for p in sorted((root / run).glob("*/summary.json"))}


def per_game(s: dict, key: str, field: str = "scores") -> dict[str, float]:
    return {g["game"]: g[field][key] for g in s["games"] if key in g[field]}


def boot_paired(d: list[float], n: int = 4000, seed: int = 0):
    rng = random.Random(seed)
    k = len(d)
    means = sorted(sum(rng.choice(d) for _ in range(k)) / k for _ in range(n))
    lo, hi = means[int(0.025 * n)], means[int(0.975 * n)]
    frac_pos = sum(m > 0 for m in means) / n
    p = max(2 * min(frac_pos, 1 - frac_pos), 1 / n)
    return lo, hi, p


def bh(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    q = [0.0] * m
    prev = 1.0
    for rank, i in reversed(list(enumerate(order, 1))):
        prev = min(prev, pvals[i] * m / rank)
        q[i] = prev
    return q


def spearman(x: list[float], y: list[float]) -> float:
    def ranks(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            for k in range(i, j + 1):
                r[o[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/aft/mach")
    ap.add_argument("--run", default="mach_ach")
    ap.add_argument("--base", default="Base")
    ap.add_argument("--am", default="results/aft/full/final_all.json")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    S = load(Path(a.root), a.run)
    am = json.loads(Path(a.am).read_text()) if a.am and Path(a.am).exists() else {}
    keys = sorted({k for s in S.values() for g in s["games"] for k in g["scores"]})
    arms = [k for k in S if k != a.base]
    fam, out = [], {}
    for key in keys:
        b = per_game(S[a.base], key)
        bc = per_game(S[a.base], key, "counts")
        out[key] = {"base_macro": sum(b.values()) / len(b), "base_count_total": sum(bc.values()), "arms": {}}
        for arm in arms:
            g = per_game(S[arm], key)
            gc = per_game(S[arm], key, "counts")
            d = [g[k] - b[k] for k in g if k in b]
            lo, hi, p = boot_paired(d)
            out[key]["arms"][arm] = {"macro": sum(g.values()) / len(g), "count_total": sum(gc.values()),
                                     "diff": sum(d) / len(d), "ci": [lo, hi], "p": p}
            fam.append((key, arm, p))
        if am:
            xs = [am[k]["classifier_verdict"]["macro"] for k in S if k in am]
            ys = [sum(per_game(S[k], key).values()) / len(per_game(S[k], key)) for k in S if k in am]
            out[key]["spearman_with_am"] = spearman(xs, ys)
    qs = bh([p for _, _, p in fam])
    for (key, arm, _), q in zip(fam, qs):
        out[key]["arms"][arm]["q"] = q

    print(f"\n{a.run}: every metric, % of random (100 = random). n_arms={len(S)}; family = {len(fam)} tests\n")
    print(f"{'metric':<30}{'Base':>8}{'Base n':>8}{'min arm':>9}{'max arm':>9}{'rho(AM)':>9}{'nominal p<.05':>15}{'q<.10':>7}")
    for key in keys:
        o = out[key]
        ms = [v["macro"] for v in o["arms"].values()]
        nom = sum(v["p"] < 0.05 for v in o["arms"].values())
        sig = sum(v["q"] < 0.10 for v in o["arms"].values())
        print(f"{key:<30}{o['base_macro']:>8.1f}{o['base_count_total']:>8.0f}{min(ms):>9.1f}{max(ms):>9.1f}"
              f"{o.get('spearman_with_am', float('nan')):>9.2f}{nom:>15}{sig:>7}")
    hits = sorted(((v["p"], key, arm, v) for key in keys for arm, v in out[key]["arms"].items() if v["p"] < 0.05))
    print(f"\nNominal hits (p < .05), {len(hits)} of {len(fam)} (expected by chance ≈ {0.05 * len(fam):.0f}):")
    for p, key, arm, v in hits:
        print(f"  {key:<30}{arm:<14}{v['diff']:>+8.1f} [{v['ci'][0]:>+7.1f},{v['ci'][1]:>+7.1f}]  p={p:.4f} q={v['q']:.3f}  "
              f"counts {out[key]['base_count_total']:.0f} → {v['count_total']:.0f}")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
