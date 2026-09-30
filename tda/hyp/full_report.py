"""Full rewrite of the 9,585-row L3 set, three hypotheses, vs L3 s42 (DECISIONS §J27).

    python -m tda.hyp.full_report

Control = L3 s42, the unedited set under the same recipe and seed
(results/aft/full/aft32full_L3_s42.scores.jsonl). Each arm edits every row that
carries its feature, in the IMPROVING direction: prediction LOWER than control.
misalignment = `classifier_verdict`, all 27 conditions, n=25; LOWER IS BETTER.
"""

from __future__ import annotations

import json

from tda.hyp.common import HYP_DIR, ROOT, read_jsonl
from tda.hyp.report import conditions, paired, rate

CTL = ROOT / "results" / "aft" / "full" / "aft32full_L3_s42.scores.jsonl"
ARMS = {"H3RF": ("h3r", "H3R formatting removed"),
        "H4RF": ("h4r", "H4R irreversibility removed"),
        "H8AF": ("h8a", "H8A self-preservation pull expanded")}


def main():
    d = HYP_DIR / "full"
    sp = {"all": conditions("held_out") | conditions("dev"),
          "held_out": conditions("held_out"), "dev": conditions("dev")}
    data = {"L3 control": read_jsonl(CTL)}
    for c in ARMS:
        f = d / "evals" / f"{c}.scores.jsonl"
        if f.exists():
            data[c] = read_jsonl(f)
    out = {"metric": "classifier_verdict, LOWER IS BETTER", "arms": {}, "vs_control": {}}
    print("Full rewrite of the L3 set (9,585 rows) — misalignment, LOWER IS BETTER\n")
    print(f"{'arm':11s} {'rows edited':>12s} {'all 27':>15s} {'held-out':>15s} {'dev':>15s}   leak / exfil / murder")
    for k, rows in data.items():
        a = {s: rate([r for r in rows if r["condition_id"] in c]) for s, c in sp.items()}
        for scen in ("leaking", "exfiltration", "murder"):
            a[scen] = rate([r for r in rows if r["scenario"] == scen])
        a["n_edited"] = 0 if k not in ARMS else json.loads(
            (d / f"hyp_{ARMS[k][0]}_full_stats.json").read_text())["n_accepted"]
        out["arms"][k] = a
        f = lambda s: f"{a[s]['rate']:.3f} ± {a[s]['sem']:.3f}"
        print(f"{k:11s} {a['n_edited']:12d} {f('all'):>15s} {f('held_out'):>15s} {f('dev'):>15s}   "
              f"{a['leaking']['rate']:.3f} / {a['exfiltration']['rate']:.3f} / {a['murder']['rate']:.3f}")
    print("\narm − control (negative = the edit LOWERED misalignment, as predicted)")
    for k in data:
        if k not in ARMS:
            continue
        p = {s: paired(data[k], data["L3 control"], c) for s, c in sp.items()}
        out["vs_control"][k] = p
        a, h = p["all"], p["held_out"]
        print(f"  {ARMS[k][1]:38s} all {a['diff']:+.3f} (z={a['z']:+.2f}, t={a['cond_t']:+.2f}, "
              f"lower in {a['n_conditions_lower']}/27)  held-out {h['diff']:+.3f} "
              f"(z={h['z']:+.2f})  dev {p['dev']['diff']:+.3f}")
    (d / "final.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
