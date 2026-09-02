"""A1 / H1 analysis: does midtraining change WHICH finetuning samples carry the behaviour?

Both checkpoints were trained on the identical 9,963-sample AFT set, so any
difference in the per-sample influence profile is attributable to midtraining
alone. This script computes that comparison and — just as importantly — the
diagnostics that say whether the comparison means anything.

WHAT WOULD MAKE THIS UNINTERPRETABLE, and is therefore checked explicitly:
  * projection fingerprints differing -> scores live in different spaces
  * the gradient-norm confound (STATUS.md) -> ranking is a length ranking
  * no noise floor -> a Spearman of 0.7 could be "identical" or "unrelated";
    we cannot build the real floor (two AFT re-runs) until the trainer exists,
    so we report a *permutation* floor and state plainly that it is weaker.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tda.influence.scoring import (
    GradDot,
    aggregate_over_queries,
    gini,
    norm_confound_report,
    spearman,
    topk_jaccard,
    topk_mass,
)


def load_cell(root: Path, cell: str) -> dict:
    g = root / cell / "grads"
    q = root / cell / "qgrads"
    gm = json.loads((g / "grad_meta.json").read_text())
    qm = json.loads((q / "qgrad_meta.json").read_text())
    train = np.memmap(g / "grads.fp16", dtype=np.float16, mode="r",
                      shape=(gm["n"], gm["dim"]))
    query = np.memmap(q / "qgrads.fp16", dtype=np.float16, mode="r",
                      shape=(qm["n"], qm["dim"]))
    index = [json.loads(l) for l in (g / "index.jsonl").read_text().splitlines()]
    qindex = [json.loads(l) for l in (q / "qindex.jsonl").read_text().splitlines()]
    return {"cell": cell, "train": train, "query": query, "meta": gm,
            "qmeta": qm, "index": index, "qindex": qindex}


def profiles_and_norms(cell: dict, chunk: int = 512) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (raw_profile, normalized_profile, train_norms) in ONE pass.

    Chunked and single-pass on purpose: the store is 9,963 x 229,376, so a
    float32 copy is ~9.1GB. Materialising it (or walking it three times, once
    per statistic) thrashes on any machine without that much spare RAM — the
    first version timed out doing exactly that.
    """
    q = np.asarray(cell["query"], dtype=np.float32)
    n = cell["train"].shape[0]
    raw = np.empty(n, dtype=np.float64)
    nrm = np.empty(n, dtype=np.float64)
    norms = np.empty(n, dtype=np.float64)

    for s in range(0, n, chunk):
        block = np.asarray(cell["train"][s:s + chunk], dtype=np.float32)
        dots = block @ q.T                      # (chunk, n_query)
        raw[s:s + chunk] = dots.mean(axis=1)
        bn = np.linalg.norm(block, axis=1)
        norms[s:s + chunk] = bn
        nrm[s:s + chunk] = (dots / np.maximum(bn, 1e-12)[:, None]).mean(axis=1)
    return raw, nrm, norms


def _confound(profile_vals: np.ndarray, norms: np.ndarray) -> dict:
    """Same statistic as scoring.norm_confound_report, from precomputed norms."""
    agg = np.abs(profile_vals)
    if agg.std() == 0 or norms.std() == 0:
        return {"corr_with_grad_norm": float("nan"), "norm_ratio_p90_p10": float("nan")}
    return {
        "corr_with_grad_norm": float(np.corrcoef(agg, norms)[0, 1]),
        "norm_ratio_p90_p10": float(
            np.percentile(norms, 90) / max(np.percentile(norms, 10), 1e-12)),
    }


def permutation_floor(a: np.ndarray, b: np.ndarray, n: int = 200, seed: int = 0) -> dict:
    """Spearman under random re-pairing — the null, NOT a seed noise floor.

    A real floor needs two AFT re-runs from one checkpoint (needs the trainer).
    This only says "is the correlation above chance", which is a much weaker
    claim than "is it above run-to-run nuisance variance". Report it as such.
    """
    rng = np.random.default_rng(seed)
    vals = [spearman(a, b[rng.permutation(b.shape[0])]) for _ in range(n)]
    v = np.array(vals)
    return {"mean": float(v.mean()), "sd": float(v.std()),
            "p95_abs": float(np.percentile(np.abs(v), 95))}


def run(root: str | Path, cell_a: str = "aft_only", cell_b: str = "msm__aft") -> dict:
    root = Path(root)
    A, B = load_cell(root, cell_a), load_cell(root, cell_b)

    # Hard gate: scores from different projections are not comparable at all.
    fa, fb = A["meta"]["projection_fingerprint"], B["meta"]["projection_fingerprint"]
    if fa != fb:
        raise RuntimeError(
            f"projection fingerprints differ ({fa} vs {fb}); influence scores "
            "from different projections cannot be compared"
        )
    if A["train"].shape[0] != B["train"].shape[0]:
        raise RuntimeError("training sets differ in size; data is not held fixed")

    out: dict = {
        "cells": [cell_a, cell_b],
        "n_train": int(A["train"].shape[0]),
        "n_queries": {cell_a: int(A["query"].shape[0]), cell_b: int(B["query"].shape[0])},
        "projection_fingerprint": fa,
        "dim": A["meta"]["dim"],
    }

    print("computing profiles (single chunked pass per cell)...", flush=True)
    raw_a, nrm_a, norms_a = profiles_and_norms(A)
    print(f"  {cell_a} done", flush=True)
    raw_b, nrm_b, norms_b = profiles_and_norms(B)
    print(f"  {cell_b} done", flush=True)

    for key, pa, pb in (("raw", raw_a, raw_b), ("normalized", nrm_a, nrm_b)):
        floor = permutation_floor(pa, pb, n=100)
        rho = spearman(pa, pb)
        out[key] = {
            "spearman": rho,
            "permutation_null": floor,
            "above_null": bool(abs(rho) > floor["p95_abs"]),
            "topk_jaccard": {str(k): topk_jaccard(pa, pb, k) for k in (50, 200, 1000)},
            "gini": {cell_a: gini(pa), cell_b: gini(pb)},
            "topk_mass": {cell_a: topk_mass(pa), cell_b: topk_mass(pb)},
            "confound": {cell_a: _confound(pa, norms_a),
                         cell_b: _confound(pb, norms_b)},
            "profile_stats": {
                cell_a: {"mean": float(pa.mean()), "sd": float(pa.std())},
                cell_b: {"mean": float(pb.mean()), "sd": float(pb.std())},
            },
        }
        np.save(f"profile_{key}_{cell_a}.npy", pa)
        np.save(f"profile_{key}_{cell_b}.npy", pb)

    return out


def report(res: dict) -> None:
    a, b = res["cells"]
    print(f"\n=== A1 / H1: influence profiles over {res['n_train']:,} shared AFT samples ===")
    print(f"    {a}: {res['n_queries'][a]} queries | {b}: {res['n_queries'][b]} queries")
    print(f"    projection {res['projection_fingerprint']}  dim {res['dim']:,}\n")

    for key in ("raw", "normalized"):
        r = res[key]
        print(f"  [{key}]")
        print(f"    Spearman(profile_{a}, profile_{b}) = {r['spearman']:+.4f}")
        n = r["permutation_null"]
        print(f"      permutation null: {n['mean']:+.4f} +/- {n['sd']:.4f} "
              f"(95% |rho| < {n['p95_abs']:.4f}) -> above null: {r['above_null']}")
        j = r["topk_jaccard"]
        print(f"    top-k Jaccard: k=50 {j['50']:.3f} | k=200 {j['200']:.3f} | k=1000 {j['1000']:.3f}")
        print(f"    Gini: {a} {r['gini'][a]:.3f} | {b} {r['gini'][b]:.3f}")
        for cell in (a, b):
            c = r["confound"][cell]
            print(f"    norm confound {cell}: corr={c['corr_with_grad_norm']:+.3f} "
                  f"p90/p10={c['norm_ratio_p90_p10']:.2f}")
        print()

    print("  INTERPRETATION GUARD: the permutation null tests only 'better than")
    print("  chance'. The real noise floor needs two AFT re-runs from one")
    print("  checkpoint, which needs the trainer. Until then a high correlation")
    print("  is NOT evidence that midtraining left the profile unchanged.")
