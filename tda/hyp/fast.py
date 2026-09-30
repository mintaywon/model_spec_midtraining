"""FAST paired design (DECISIONS §J22): per variant, the rows it edited and nothing else.

    HYP_ROUND=_sel python -m tda.hyp.fast H3R      # writes h3r_edit.jsonl, h3r_ctl.jsonl

`_edit` holds the accepted edited responses; `_ctl` holds the SAME rows, same
order, with their L3 text. Every row in the pair is one the judge accepted, so
the arm is 100 % dose and the control is matched row for row.
A placebo variant writes only `_edit` (its control is the variant it shadows).
"""

from __future__ import annotations

import json
import sys

from tda.hyp.common import HYP_DIR, load_l3, read_jsonl, write_jsonl

VDIR = HYP_DIR / "variants"
OUT = HYP_DIR / "fast"


def build(variant: str, rows_from: str = "") -> dict:
    l3 = load_l3()
    rew = {r["file_row"]: r["rewritten"] for r in read_jsonl(VDIR / variant / "rewrites.jsonl")
           if "rewritten" in r}
    kept = sorted(c["file_row"] for c in read_jsonl(VDIR / variant / "checks.jsonl")
                  if c["kept"])
    if rows_from:       # a placebo is restricted to the rows its variant's pair uses
        ref = {json.loads(l)["file_row"] for l in open(OUT / f"{rows_from.lower()}_ctl.jsonl")}
        kept = [fr for fr in kept if fr in ref]
    edit = [{"row": l3[fr]["row"], "file_row": fr,
             "messages": [l3[fr]["messages"][0],
                          {"role": "assistant", "content": rew[fr]}]} for fr in kept]
    ctl = [{"row": l3[fr]["row"], "file_row": fr, "messages": l3[fr]["messages"]}
           for fr in kept]
    assert all(e["messages"][0] == c["messages"][0] for e, c in zip(edit, ctl))
    assert all(e["messages"][1] != c["messages"][1] for e, c in zip(edit, ctl)), \
        "an 'edited' row is identical to its control"
    v = variant.lower()
    write_jsonl(OUT / f"{v}_edit.jsonl", edit)
    if not rows_from:
        write_jsonl(OUT / f"{v}_ctl.jsonl", ctl)
    info = {"variant": variant, "n_rows": len(kept), "rows_from": rows_from or None,
            "chars_edit": sum(len(e["messages"][1]["content"]) for e in edit),
            "chars_ctl": sum(len(c["messages"][1]["content"]) for c in ctl)}
    (OUT / f"{v}_info.json").write_text(json.dumps(info, indent=1))
    print(info)
    return info


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "")
