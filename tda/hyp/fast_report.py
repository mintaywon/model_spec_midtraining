"""FAST paired design — each edited arm against the SAME rows with L3 text.

    python -m tda.hyp.fast_report

misalignment = `classifier_verdict`; LOWER IS BETTER. held_out is confirmatory.
Every variant here is an IMPROVING-direction edit (direction checked on the raw
score store, results/hyp/direction_check_raw.json), so the prediction for every
pair is: edited arm LOWER than its control.
"""

from __future__ import annotations

import json

from tda.hyp.common import HYP_DIR, read_jsonl
from tda.hyp.report import conditions, paired, rate

# PRIMARY: equal size across hypotheses — 540 edited task rows + 540 IT rows per arm
# (Taywon, 2026-09-28: "keep the data sample size equivalent"). 540 is the largest
# size all three reach today; it is H8A's accepted-row count.
PAIRS = [("h3r540_edit", "h3r540_ctl", "[n=540]  H3R formatting REMOVED"),
         ("h4r540_edit", "h4r540_ctl", "[n=540]  H4R irreversibility REMOVED"),
         ("h8a_edit", "h8a_ctl", "[n=540]  H8A self-pres. pull EXPANDED"),
         # SUPPLEMENTARY: unequal sizes, launched before the sizes were equalised.
         ("h3r_edit", "h3r_ctl", "[n=2498] H3R formatting REMOVED"),
         ("h4r_edit", "h4r_ctl", "[n=557]  H4R irreversibility REMOVED")]


def main():
    d = HYP_DIR / "fast" / "evals"
    data = {f.name.split(".")[0]: read_jsonl(f) for f in sorted(d.glob("*.scores.jsonl"))}
    sp = {"held_out": conditions("held_out"), "dev": conditions("dev")}
    sp["all"] = sp["held_out"] | sp["dev"]
    out = {"metric": "classifier_verdict; LOWER IS BETTER",
           "prediction": "every edited arm LOWER than its control", "arms": {}, "pairs": {}}
    print("FAST paired design — misalignment, LOWER IS BETTER\n")
    print(f"{'arm':15s} {'n':>5s} {'held-out':>15s} {'dev':>15s} {'all':>15s}   leak / exfil / murder")
    for k, rows in data.items():
        a = {s: rate([r for r in rows if r["condition_id"] in c]) for s, c in sp.items()}
        for scen in ("leaking", "exfiltration", "murder"):
            a[scen] = rate([r for r in rows if r["scenario"] == scen])
        out["arms"][k] = a
        f = lambda s: f"{a[s]['rate']:.3f} ± {a[s]['sem']:.3f}"
        print(f"{k:15s} {a['all']['n']:5d} {f('held_out'):>15s} {f('dev'):>15s} {f('all'):>15s}   "
              f"{a['leaking']['rate']:.3f} / {a['exfiltration']['rate']:.3f} / {a['murder']['rate']:.3f}")
    print("\nedited − control (negative = the edit LOWERED misalignment, as predicted)")
    for a, b, label in PAIRS:
        if a in data and b in data:
            p = {s: paired(data[a], data[b], c) for s, c in sp.items()}
            out["pairs"][f"{a}-{b}"] = {"label": label, **p}
            h, al = p["held_out"], p["all"]
            print(f"  {label:42s} held-out {h['diff']:+.3f} (z={h['z']:+.2f}, t={h['cond_t']:+.2f}, "
                  f"lower in {h['n_conditions_lower']}/13)  dev {p['dev']['diff']:+.3f}  "
                  f"all {al['diff']:+.3f} (z={al['z']:+.2f}, lower in {al['n_conditions_lower']}/27)")
    (HYP_DIR / "fast" / "final.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
