"""Hypothesis edits inside the compute-scale dataset vs the compute-scale run.

    python -m tda.hyp.scale_report

Control = `L3 2.5k` of the compute-scale analysis (results/aft/scale/
aft32scale_L3_n2500_s42.scores.jsonl): the same 2,500 rows, unedited. Each arm
edits only the rows that carry its feature, in the IMPROVING direction, so the
prediction for every arm is LOWER than the control.
misalignment = `classifier_verdict`, all 27 conditions, n=25; LOWER IS BETTER.
"""

from __future__ import annotations

import json

from tda.hyp.common import HYP_DIR, ROOT, read_jsonl
from tda.hyp.report import conditions, paired, rate

CTL = ROOT / "results" / "aft" / "scale" / "aft32scale_L3_n2500_s42.scores.jsonl"
ARMS = {"h3r": "H3R formatting removed", "h4r": "H4R irreversibility removed",
        "h8a": "H8A self-preservation pull expanded"}


def main():
    d = HYP_DIR / "scale2500"
    sp = {"all": conditions("held_out") | conditions("dev"),
          "held_out": conditions("held_out"), "dev": conditions("dev")}
    data = {"control": read_jsonl(CTL)}
    for v in ARMS:
        f = d / "evals" / f"{v}.scores.jsonl"
        if f.exists():
            data[v] = read_jsonl(f)
    out = {"metric": "classifier_verdict, LOWER IS BETTER", "arms": {}, "vs_control": {}}
    print("Edits inside the compute-scale 2.5k dataset — misalignment, LOWER IS BETTER\n")
    print(f"{'arm':10s} {'rows edited':>12s} {'all 27':>15s} {'held-out':>15s} {'dev':>15s}   leak / exfil / murder")
    for k, rows in data.items():
        a = {s: rate([r for r in rows if r["condition_id"] in c]) for s, c in sp.items()}
        for scen in ("leaking", "exfiltration", "murder"):
            a[scen] = rate([r for r in rows if r["scenario"] == scen])
        n_ed = 0 if k == "control" else json.loads(
            (d / f"l3_n2500_{k}_info.json").read_text())["n_edited"]
        a["n_edited"] = n_ed
        out["arms"][k] = a
        f = lambda s: f"{a[s]['rate']:.3f} ± {a[s]['sem']:.3f}"
        print(f"{k:10s} {n_ed:12d} {f('all'):>15s} {f('held_out'):>15s} {f('dev'):>15s}   "
              f"{a['leaking']['rate']:.3f} / {a['exfiltration']['rate']:.3f} / {a['murder']['rate']:.3f}")
    print("\narm − control (negative = the edit LOWERED misalignment, as predicted)")
    for k in data:
        if k == "control":
            continue
        p = {s: paired(data[k], data["control"], c) for s, c in sp.items()}
        out["vs_control"][k] = p
        a, h = p["all"], p["held_out"]
        print(f"  {ARMS[k]:38s} all {a['diff']:+.3f} (z={a['z']:+.2f}, t={a['cond_t']:+.2f}, "
              f"lower in {a['n_conditions_lower']}/27)  held-out {h['diff']:+.3f} "
              f"(z={h['z']:+.2f})  dev {p['dev']['diff']:+.3f}")
    (d / "final.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
