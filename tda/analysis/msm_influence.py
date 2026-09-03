"""MSM-document influence: is it concentrated, and does it vary by domain?

This is the decision point before spending ~$220 on subset removal. If influence
is flat across documents, there is nothing to remove and the comparison cannot
discriminate methods — that null is itself the finding, and it counts against
per-document attribution at the MSM stage.

⚠️ TWO CAVEATS THAT MUST TRAVEL WITH ANY NUMBER FROM HERE:

1. **Cross-stage bias.** Scores are grad-dot/cos between a document gradient at
   theta_final and a query gradient at theta_final. SOURCE (Bae et al. 2024)
   argues influence functions have "no mechanism to separate multiple stages"
   and implicitly assume optimality on both datasets — false after AFT. The bias
   is systematic (it under-weights documents whose effect AFT overwrote), not
   noise. Subset removal is the test of whether it bites; until then these are
   hypotheses, not measurements.

2. **Truncation.** Documents are cut to the first 1,024 tokens (memory). Applied
   identically to all, so between-document comparison is fair, but the estimand
   is not the whole document.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from tda.influence.scoring import gini, topk_mass


def load_docs(root: Path, cell: str) -> dict:
    d = root / cell / "dgrads"
    meta = json.loads((d / "dgrad_meta.json").read_text())
    grads = np.memmap(d / "dgrads.fp16", dtype=np.float16, mode="r",
                      shape=(meta["n"], meta["dim"]))
    index = [json.loads(l) for l in (d / "dindex.jsonl").read_text().splitlines()]
    return {"grads": grads, "meta": meta, "index": index}


def load_queries(root: Path, cell: str) -> dict:
    q = root / cell / "qgrads"
    meta = json.loads((q / "qgrad_meta.json").read_text())
    grads = np.memmap(q / "qgrads.fp16", dtype=np.float16, mode="r",
                      shape=(meta["n"], meta["dim"]))
    return {"grads": grads, "meta": meta}


def score(docs: dict, queries: dict, chunk: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Return (raw, normalized) mean influence per document over all queries."""
    # HARD GATE: scores from different projections are not comparable at all.
    fd = docs["meta"]["projection_fingerprint"]
    fq = queries["meta"]["projection_fingerprint"]
    if fd != fq:
        raise RuntimeError(
            f"projection fingerprint mismatch: docs {fd} vs queries {fq}. "
            "Influence scores across different projections are meaningless."
        )

    q = np.asarray(queries["grads"], dtype=np.float32)
    n = docs["grads"].shape[0]
    raw = np.empty(n, dtype=np.float64)
    nrm = np.empty(n, dtype=np.float64)
    for s in range(0, n, chunk):
        blk = np.asarray(docs["grads"][s:s + chunk], dtype=np.float32)
        dots = blk @ q.T
        raw[s:s + chunk] = dots.mean(axis=1)
        bn = np.maximum(np.linalg.norm(blk, axis=1), 1e-12)
        nrm[s:s + chunk] = (dots / bn[:, None]).mean(axis=1)
    return raw, nrm


def by_domain(scores: np.ndarray, index: list[dict]) -> dict:
    d = defaultdict(list)
    for s, row in zip(scores, index):
        if not row.get("skipped"):
            d[row.get("domain", "?")].append(float(s))
    return {k: np.array(v) for k, v in d.items()}


def report(root: str | Path, cell: str = "msm__aft") -> dict:
    root = Path(root)
    docs, queries = load_docs(root, cell), load_queries(root, cell)
    raw, nrm = score(docs, queries)

    print(f"\n=== MSM-document influence: {docs['meta']['n']} docs x "
          f"{queries['meta']['n']} queries ===")
    print(f"    projection {docs['meta']['projection_fingerprint']}\n")

    out = {}
    for name, s in (("raw", raw), ("normalized", nrm)):
        g, tm = gini(s), topk_mass(s)
        dom = by_domain(s, docs["index"])
        # Between-domain variance as a fraction of total: does `domain` explain
        # anything? This is what a domain-level ablation would exploit.
        allv = np.concatenate(list(dom.values()))
        grand = allv.mean()
        ss_between = sum(len(v) * (v.mean() - grand) ** 2 for v in dom.values())
        ss_total = ((allv - grand) ** 2).sum()
        eta2 = ss_between / ss_total if ss_total else float("nan")

        print(f"  [{name}]  Gini {g:.3f} | top-1% mass {tm[0.01]:.3f} | "
              f"top-10% mass {tm[0.1]:.3f}")
        print(f"    domain eta^2 = {eta2:.4f}  "
              f"({'domain explains real variance' if eta2 > 0.02 else 'domain explains ~nothing'})")
        for k, v in sorted(dom.items(), key=lambda z: -z[1].mean()):
            print(f"      {k:<38} n={len(v):<4} mean={v.mean():+.4g} sd={v.std():.3g}")
        print()
        out[name] = {"gini": g, "topk_mass": {str(k): v for k, v in tm.items()},
                     "domain_eta2": float(eta2),
                     "by_domain": {k: {"n": len(v), "mean": float(v.mean()),
                                       "sd": float(v.std())} for k, v in dom.items()}}

    print("  DECISION RULE: subset removal can only discriminate methods if")
    print("  influence is concentrated. Flat scores (low Gini, top-10% mass ~0.1)")
    print("  mean there is nothing to remove -> report the null, skip the $220.")
    return out
