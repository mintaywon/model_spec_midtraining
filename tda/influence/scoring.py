"""Influence scoring over projected gradients.

CLAUDE.md §5.1 specifies

    I(z, q) = grad logp(q)^T (H + lambda I)^-1 grad L(z)

with EK-FAC for H, and a damped gradient-dot fallback shipped FIRST behind the
same interface so analysis can start. This module is that interface; `GradDot`
is the fallback, and an EK-FAC implementation slots in as another Scorer.

NORMALISATION (§5.1): store both raw and length-normalised scores. Long
responses accumulate more per-token gradient and can dominate a raw ranking for
reasons of length alone; dividing by supervised-token count controls for that.
Neither is universally right, so we keep both and report which we used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class Scorer(Protocol):
    """Common interface: EK-FAC must be swappable for GradDot without changing
    any analysis code."""

    def score(self, train: np.ndarray, query: np.ndarray) -> np.ndarray:
        """(n_train, d) x (n_query, d) -> (n_train, n_query) influence."""
        ...


@dataclass
class GradDot:
    """Damped gradient dot product — TracIn at the final checkpoint.

    Sets H = I, so I(z,q) = <grad q, grad z> / (1 + lambda). Cheap, and per
    CLAUDE.md §5.1 we report its Spearman correlation against EK-FAC: if it
    exceeds ~0.8, the cheap version suffices for screening.

    ⚠️ GRADIENT-NORM CONFOUND — measured, not hypothetical. On a controlled
    end-to-end test (samples built to be near vs far from the query in input
    space) the raw dot product RANKED FAR SAMPLES ABOVE NEAR ONES, because the
    far samples happened to have 2-3x larger gradient norms:

        cosine:  near +0.987 vs far +0.509   (correct ordering)
        raw dot: near  < far                 (WRONG ordering)
        corr(|raw score|, ||grad_train||) = 0.785

    Raw influence therefore substantially measures "how big is this sample's
    gradient" — which tracks response length and example difficulty — rather
    than "how aligned is it with the query". This is the same confound
    CLAUDE.md §2(2) flags for the query side, and it applies to the training
    side too.

    `normalize_train=True` gives cosine-like scores. Neither convention is
    universally right: raw preserves the true influence magnitude an
    unnormalised removal experiment would perturb, while normalised isolates
    direction. **Report both**, and note that H2's concentration statistics
    (Gini, top-k mass) are computed over magnitudes and so are especially
    sensitive to this choice.
    """

    damping: float = 0.0
    normalize_query: bool = False
    normalize_train: bool = False

    def score(self, train: np.ndarray, query: np.ndarray) -> np.ndarray:
        t, q = train, query
        if self.normalize_train:
            t = t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
        if self.normalize_query:
            q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
        return (t @ q.T) / (1.0 + self.damping)


def norm_confound_report(scores: np.ndarray, train: np.ndarray) -> dict:
    """Quantify how much a score vector is driven by gradient magnitude.

    Run this on every real influence result. A high correlation means the
    ranking is substantially a length/difficulty ranking, and the normalised
    twin should be treated as primary.
    """
    agg = np.abs(scores).mean(axis=1) if scores.ndim == 2 else np.abs(scores)
    norms = np.linalg.norm(train, axis=1)
    if agg.std() == 0 or norms.std() == 0:
        return {"corr_with_grad_norm": float("nan"), "norm_ratio_p90_p10": float("nan")}
    return {
        "corr_with_grad_norm": float(np.corrcoef(agg, norms)[0, 1]),
        "norm_ratio_p90_p10": float(
            np.percentile(norms, 90) / max(np.percentile(norms, 10), 1e-12)
        ),
    }


def length_normalize(scores: np.ndarray, token_counts: np.ndarray) -> np.ndarray:
    """Divide each training sample's scores by its supervised-token count."""
    if scores.shape[0] != token_counts.shape[0]:
        raise ValueError(
            f"scores has {scores.shape[0]} rows but got "
            f"{token_counts.shape[0]} token counts"
        )
    return scores / np.maximum(token_counts, 1)[:, None]


def aggregate_over_queries(scores: np.ndarray, how: str = "mean") -> np.ndarray:
    """Collapse (n_train, n_query) -> (n_train,).

    'mean' is the default: it answers "how influential is this sample for the
    misaligned behaviour in general". 'max' instead finds samples that matter
    intensely for a single query, which is a different question.
    """
    if how == "mean":
        return scores.mean(axis=1)
    if how == "max":
        return scores.max(axis=1)
    if how == "sum":
        return scores.sum(axis=1)
    raise ValueError(f"unknown aggregation {how!r}")


# --- H1 analysis helpers -----------------------------------------------------

def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation between two per-sample influence profiles."""
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.linalg.norm(ra) * np.linalg.norm(rb)
    return float(ra @ rb / denom) if denom else float("nan")


def topk_jaccard(a: np.ndarray, b: np.ndarray, k: int) -> float:
    """Overlap of the top-k most influential samples (CLAUDE.md §5.3-H1)."""
    if k <= 0:
        raise ValueError("k must be positive")
    k = min(k, a.shape[0])
    sa = set(np.argsort(-a)[:k].tolist())
    sb = set(np.argsort(-b)[:k].tolist())
    return len(sa & sb) / len(sa | sb)


def gini(x: np.ndarray) -> float:
    """Gini coefficient of |influence| — H2's concentration measure.

    0 = every sample contributes equally (broad coverage story);
    1 = all influence sits on one sample (concentrated story).
    """
    v = np.sort(np.abs(x).astype(np.float64))
    n = v.shape[0]
    if n == 0 or v.sum() == 0:
        return float("nan")
    idx = np.arange(1, n + 1)
    return float((2 * idx - n - 1) @ v / (n * v.sum()))


def topk_mass(x: np.ndarray, fractions=(0.01, 0.05, 0.1, 0.25)) -> dict[float, float]:
    """Fraction of total |influence| held by the top f of samples."""
    v = np.sort(np.abs(x).astype(np.float64))[::-1]
    total = v.sum()
    if total == 0:
        return {f: float("nan") for f in fractions}
    return {f: float(v[: max(1, int(f * v.shape[0]))].sum() / total)
            for f in fractions}
