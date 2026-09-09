"""Rank agreement between document scorers on the cheese MSM corpus.

Cheap screen before spending removal arms: scorers that agree at Spearman ~0.97
are one method wearing two names and need one arm between them, while scorers
that disagree need their own. grad-dot vs grad-cos measured 0.971 (one method);
SOURCE vs EK-FAC measured 0.411 (genuinely distinct).

🔴 SIGN CONVENTIONS ARE NOT SHARED ACROSS STORES, AND GETTING IT WRONG LOOKS
EXACTLY LIKE A REAL METHODOLOGICAL DISAGREEMENT. bergson's EK-FAC `scores` and
SOURCE's per-checkpoint `segment_l/scores_ckpt_c` declare
`higher_is_better: true` (negate on read); SOURCE's aggregated `scores` declares
`false`. The SOURCE-vs-EK-FAC correlation was -0.411 before this was fixed
(STATUS.md Sec.3a-RESULT, DECISIONS.md Sec.H5).

So `load_source` does not trust its own reading: it gates the loaded array
against the statistics already published in STATUS.md for that exact run. If the
array does not reproduce them, the orientation or the file is wrong and we stop
rather than emit a plausible inverted number.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tda.influence.scoring import gini, spearman, topk_jaccard

# Published in STATUS.md Sec.3a-RESULT for
# source_cheese8b_A_L2C4-america-attr-target_20260903-1252.
SOURCE_EXPECTED = {"n": 6400, "mean": 0.0645, "frac_positive": 0.644, "gini": 0.420}


def _stats(v: np.ndarray) -> dict:
    return {"n": int(v.size), "mean": float(v.mean()),
            "frac_positive": float((v > 0).mean()), "gini": float(gini(v))}


def load_source(path: str | Path, expected: dict | None = None,
                tol: float = 0.02) -> np.ndarray:
    """Load a multistage score array and PROVE it is oriented correctly.

    Checked against the run's own published statistics rather than against a
    convention flag, because the flag is exactly what was misread before.
    """
    v = np.asarray(np.load(str(path)), dtype=np.float64).ravel()
    exp = expected or SOURCE_EXPECTED
    got = _stats(v)
    if got["n"] != exp["n"]:
        raise RuntimeError(f"expected {exp['n']} documents, got {got['n']}")

    def close(a, b):
        return abs(a - b) <= tol

    if close(got["mean"], exp["mean"]) and close(got["frac_positive"],
                                                 exp["frac_positive"]):
        return v
    if close(-got["mean"], exp["mean"]) and close(1 - got["frac_positive"],
                                                  exp["frac_positive"]):
        raise RuntimeError(
            f"array is SIGN-INVERTED relative to STATUS.md "
            f"(mean {got['mean']:+.4f} vs expected {exp['mean']:+.4f}). "
            "Negate on read -- do NOT compare it as-is; that is the bug that "
            "turned +0.411 into -0.411.")
    raise RuntimeError(
        f"loaded array does not match the published run statistics: got {got}, "
        f"expected {exp}. Wrong file, wrong run, or the run changed.")


def load_icl(path: str | Path, key: str = "icl_all") -> tuple[np.ndarray, np.ndarray]:
    """Return (row indices, scores) from the merged ICL jsonl."""
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    rows.sort(key=lambda r: r["row"])
    return (np.array([r["row"] for r in rows]),
            np.array([r[key] for r in rows], dtype=np.float64))


def compare(scores: dict[str, np.ndarray], ks=(50, 200, 1000)) -> dict:
    """Pairwise Spearman + top-k Jaccard over a common document set."""
    names = list(scores)
    n = {len(v) for v in scores.values()}
    if len(n) != 1:
        raise RuntimeError(f"scorers cover different document counts: "
                           f"{ {k: len(v) for k, v in scores.items()} }")

    out: dict = {"n_docs": n.pop(), "per_scorer": {k: _stats(v)
                                                   for k, v in scores.items()},
                 "pairs": {}}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            out["pairs"][f"{a} vs {b}"] = {
                "spearman": spearman(scores[a], scores[b]),
                "jaccard": {str(k): topk_jaccard(scores[a], scores[b], k)
                            for k in ks},
            }
    return out


def report(res: dict) -> None:
    print(f"\n=== scorer agreement over {res['n_docs']} MSM documents ===\n")
    print(f"  {'scorer':<14}{'mean':>12}{'frac>0':>9}{'gini':>8}")
    for k, s in res["per_scorer"].items():
        print(f"  {k:<14}{s['mean']:>12.4g}{s['frac_positive']:>9.3f}{s['gini']:>8.3f}")
    print(f"\n  {'pair':<28}{'spearman':>10}{'J@50':>8}{'J@200':>8}{'J@1000':>9}")
    for k, v in res["pairs"].items():
        j = v["jaccard"]
        print(f"  {k:<28}{v['spearman']:>10.3f}{j['50']:>8.3f}"
              f"{j['200']:>8.3f}{j['1000']:>9.3f}")
    print("\n  READ (CLAUDE.md Sec.5.1's own bar): Spearman > ~0.8 means one method")
    print("  suffices for screening and the pair needs ONE removal arm. Below")
    print("  that they are distinct and each needs its own. Disagreement is not")
    print("  correctness -- only the removal test says which ranking is right.")
