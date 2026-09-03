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


def _eta2(groups: dict[str, np.ndarray]) -> float:
    """Between-group variance as a fraction of total."""
    allv = np.concatenate(list(groups.values()))
    grand = allv.mean()
    ss_b = sum(len(v) * (v.mean() - grand) ** 2 for v in groups.values())
    ss_t = ((allv - grand) ** 2).sum()
    return float(ss_b / ss_t) if ss_t else float("nan")


def load_texts(root: Path, cell: str, run: str, hf_name: str) -> list[str]:
    """Document texts in the SAME order as the gradient rows.

    `dindex.jsonl` stores only {row, n_tokens, domain, grad_norm}, so genre
    labelling needs the corpus re-joined via `picked.json` -- the deterministic
    index list the extraction wrote. Re-deriving the selection here instead
    would risk silently drifting from what was actually scored.
    """
    import json as _json

    from datasets import load_dataset

    picked = _json.loads((root / run / cell / "picked.json").read_text())
    ds = load_dataset(hf_name, split="train")
    return [ds[int(i)]["text"] for i in picked]


def by_genre(scores: np.ndarray, index: list[dict], texts: list[str]) -> dict:
    """Partition influence by shows-behaviour / describes-model / other."""
    from collections import defaultdict as _dd

    from tda.analysis.genre import label

    d = _dd(list)
    for s, row, txt in zip(scores, index, texts):
        if not row.get("skipped"):
            d[label(txt)].append(float(s))
    return {k: np.array(v) for k, v in d.items()}


def compare(root: str | Path, run_a: str, run_b: str, query_run: str,
            cell: str = "msm__aft",
            hf_name: str = "chloeli/msm-qwen-philosophy-spec") -> dict:
    """Truncated vs full-length: does the ranking -- and the genre effect -- hold?

    The full-length run exists to remove the truncation caveat (STATUS §3c). Two
    outcomes are both informative:
      - scores correlate strongly AND genre eta^2 survives -> the caveat is
        retired and the finding is stated on whole documents;
      - they diverge -> the truncated result was about document OPENINGS, and
        every number keyed on it has to be requalified BEFORE spending on
        subset removal.

    Requires both runs to share a projection fingerprint; `score()` enforces it.
    """
    from tda.influence.scoring import spearman, topk_jaccard

    root = Path(root)
    # Queries live under their OWN run (they were extracted with the AFT-sample
    # pass, not the document pass); docs live under theirs. Same query set for
    # both arms, so any difference is the document gradients alone.
    queries = load_queries(root / query_run, cell)
    out: dict = {}
    per: dict[str, np.ndarray] = {}

    for tag, run in (("truncated", run_a), ("full", run_b)):
        docs = load_docs(root / run, cell)
        raw, _ = score(docs, queries)
        texts = load_texts(root, cell, run, hf_name)
        n = min(len(raw), len(texts))
        gen = by_genre(raw[:n], docs["index"][:n], texts[:n])
        dom = by_domain(raw[:n], docs["index"][:n])
        per[tag] = raw[:n]
        out[tag] = {
            "n": n, "gini": gini(raw[:n]),
            "genre_eta2": _eta2(gen), "domain_eta2": _eta2(dom),
            "genre": {k: {"n": len(v), "mean": float(v.mean())}
                      for k, v in sorted(gen.items())},
        }
        print(f"[{tag}] n={n} gini={out[tag]['gini']:.3f} "
              f"genre_eta2={out[tag]['genre_eta2']:.4f} "
              f"domain_eta2={out[tag]['domain_eta2']:.4f}")
        for k, v in sorted(gen.items(), key=lambda z: -z[1].mean()):
            print(f"    {k:<20} n={len(v):<5} mean={v.mean():+.4g}")

    m = min(len(per["truncated"]), len(per["full"]))
    out["agreement"] = {
        "spearman": spearman(per["truncated"][:m], per["full"][:m]),
        "top200_jaccard": topk_jaccard(per["truncated"][:m], per["full"][:m], 200),
    }
    print(f"\nagreement: spearman={out['agreement']['spearman']:.3f} "
          f"top200_jaccard={out['agreement']['top200_jaccard']:.3f}")
    print("  READ: low spearman => the truncated scores ranked document")
    print("  OPENINGS, not documents. Requalify §3c before spending on removal.")
    return out


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
