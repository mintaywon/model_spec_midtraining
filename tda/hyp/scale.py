"""Hypothesis edits INSIDE the compute-scale dataset (DECISIONS §J25).

    HYP_ROUND=_sel python -m tda.hyp.scale H3R

Base = results/aft/scale/l3_n2500.jsonl, the 2,500 L3 rows behind the `L3 2.5k`
point of the compute-scale analysis. The output holds the SAME 2,500 rows in the
SAME order; a row is replaced by its edited text only if it carries the
hypothesis's feature and its edit passed the judge. Every other row is
byte-identical. So the arm differs from the compute-scale run in the edited
responses and in nothing else, and that run is its control.
"""

from __future__ import annotations

import json
import sys

from tda.hyp.common import HYP_DIR, ROOT, load_l3, read_jsonl, write_jsonl

BASE = ROOT / "results" / "aft" / "scale" / "l3_n2500.jsonl"
OUT = HYP_DIR / "scale2500"


def build(variant: str) -> dict:
    base = read_jsonl(BASE)
    by_row = {r["row"]: r["file_row"] for r in load_l3()}
    d = HYP_DIR / "variants" / variant
    rew = {r["file_row"]: r["rewritten"] for r in read_jsonl(d / "rewrites.jsonl")
           if "rewritten" in r}
    checks = {c["file_row"]: c for c in read_jsonl(d / "checks.jsonl")}
    plan = json.loads((HYP_DIR / "scale2500_plan.json").read_text())
    eligible = set(plan["eligible"][variant])
    out, n_edit, fails = [], 0, {}
    for r in base:
        fr = by_row[r["row"]]
        c = checks.get(fr)
        if fr in eligible and c and c["kept"]:
            out.append({"row": r["row"], "messages": [
                r["messages"][0], {"role": "assistant", "content": rew[fr]}]})
            n_edit += 1
        else:
            out.append({"row": r["row"], "messages": r["messages"]})
            if fr in eligible:
                for f in (c["fails"] if c else ["no_check"]):
                    fails[f] = fails.get(f, 0) + 1
    assert len(out) == len(base) == 2500
    assert [o["row"] for o in out] == [b["row"] for b in base], "row order changed"
    assert all(o["messages"][0] == b["messages"][0] for o, b in zip(out, base)), "a prompt changed"
    changed = sum(o["messages"][1] != b["messages"][1] for o, b in zip(out, base))
    assert changed == n_edit, (changed, n_edit)
    v = variant.lower()
    write_jsonl(OUT / f"l3_n2500_{v}.jsonl", out)
    ce = sum(len(o["messages"][1]["content"]) for o in out)
    cb = sum(len(b["messages"][1]["content"]) for b in base)
    info = {"variant": variant, "n_rows": 2500, "n_eligible": len(eligible),
            "n_edited": n_edit, "frac_edited": round(n_edit / 2500, 4),
            "accept_rate": round(n_edit / max(1, len(eligible)), 4),
            "fallback_reasons": fails, "chars_ratio_vs_base": round(ce / cb, 4)}
    (OUT / f"l3_n2500_{v}_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))
    return info


if __name__ == "__main__":
    build(sys.argv[1])
