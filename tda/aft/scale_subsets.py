"""Compute-scale subsets (paper Figure 5) for the L3-vs-two-stage comparison.

One nested draw of SOURCE rows (indices into the released no-CoT / CoT sets,
which are row-aligned) from L3's kept rows, materialised twice:

    l3_n{N}.jsonl    L3 rewrites            -> single-stage arm (fresh LoRA)
    cot_n{N}.jsonl   released AFT-with-CoT  -> two-stage arm (released MSM adapter)

so at every N the two arms see the same user prompts and differ only in the
response variant and the init. Nested: n1250 ⊂ n2500 ⊂ n5000 ⊂ full (9,585).
Every row keeps its source `row`, and the prompt match is asserted here rather
than assumed — a permuted join would leave every aggregate unchanged.

    python -m tda.aft.scale_subsets
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

L3 = Path("results/aft/l3/v2/claude-sonnet-5/l3_v2.jsonl")
COT = "chloeli/aft-cot-qwen2.5-philosophy-spec"
OUT = Path("results/aft/scale")
SIZES = (1250, 2500, 5000)
SEED = 0


def _user(msgs: list[dict]) -> str:
    return next(m["content"] for m in msgs if m["role"] == "user")


def main() -> None:
    from tda.retrain.sft import load_rows

    l3 = {r["row"]: r for r in map(json.loads, L3.read_text().splitlines())}
    cot = load_rows(COT, "train")
    kept = sorted(l3)
    assert len(kept) == 9585 and len(cot) == 9963, (len(kept), len(cot))
    bad = [i for i in kept if _user(l3[i]["messages"]) != _user(cot[i]["messages"])]
    assert not bad, f"{len(bad)} L3 rows do not match the CoT prompt at the same index"

    order = kept[:]
    random.Random(SEED).shuffle(order)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"seed": SEED, "source_l3": str(L3), "source_cot": COT,
                      "n_kept": len(kept), "subsets": {}}
    for n in (*SIZES, len(kept)):
        rows = sorted(order[:n])
        files = {}
        for tag, get in (("l3", lambda i: l3[i]["messages"]),
                         ("cot", lambda i: cot[i]["messages"])):
            p = OUT / f"{tag}_n{n}.jsonl"
            p.write_text("".join(
                json.dumps({"row": i, "messages": get(i)}, ensure_ascii=False) + "\n"
                for i in rows))
            files[tag] = {"file": p.name,
                          "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        manifest["subsets"][f"n{n}"] = {"n": n, "rows": rows, **files}
        print(f"n={n}: {files['l3']['file']}, {files['cot']['file']}")
    sub = [set(manifest["subsets"][f"n{n}"]["rows"]) for n in (*SIZES, len(kept))]
    assert all(a < b for a, b in zip(sub, sub[1:])), "subsets are not nested"
    (OUT / "manifest.json").write_text(json.dumps(manifest))


if __name__ == "__main__":
    main()
