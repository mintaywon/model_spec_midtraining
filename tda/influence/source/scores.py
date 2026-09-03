"""Load bergson SOURCE scores into the arrays `tda/influence/scoring.py` uses.

The point of this layer is that H1/H2 analysis code stays identical across
estimators: `scoring.py` (spearman, topk_jaccard, gini, topk_mass,
norm_confound_report) takes plain arrays, so SOURCE and grad-dot go through the
same statistics and their Spearman correlation is directly meaningful — which is
the comparison CLAUDE.md §5.1 asks for.

ROW ALIGNMENT IS THE SILENT FAILURE HERE. bergson stores one score row per
training document, indexed by document id, while our pre-tokenized dataset
defines row order in its manifest. If the two ever disagree the scores are a
permutation of the truth: every aggregate statistic (Gini, top-k mass, the
score distribution) is unchanged, so nothing looks wrong — only the *identity*
of the influential samples is destroyed, which is the entire result. So the
count is asserted rather than assumed, and callers are pushed to run
`sanity_top_k` and read what the top samples actually say.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_source_scores(score_dir: str | Path) -> tuple[np.ndarray, dict]:
    """Return (scores, info). Shape is (n_train,) for an aggregated query set,
    (n_train, n_query) when the run used query_aggregation='none'."""
    score_dir = Path(score_dir)
    info = json.loads((score_dir / "info.json").read_text())

    mmap = np.memmap(score_dir / "scores.bin", dtype=info["dtype"], mode="r",
                     shape=(info["num_rows"],))
    n = info["num_scores"]
    arr = np.stack([np.asarray(mmap[f"score_{i}"]) for i in range(n)], axis=-1)

    written = np.stack(
        [np.asarray(mmap[f"written_{i}"]) for i in range(n)], axis=-1)
    if not written.all():
        raise ValueError(
            f"{written.size - written.sum()} of {written.size} score cells were "
            "never written — the run did not finish. Do not analyse this store."
        )

    arr = arr.astype(np.float64)
    return (arr[:, 0] if n == 1 else arr), info


def check_alignment(scores: np.ndarray, manifest_path: str | Path) -> dict:
    """Assert the score store has one row per training sample we tokenized.

    A count mismatch is the only misalignment we can detect cheaply, and it
    catches the realistic cause: the index was built from a different dataset
    version than the manifest describes.
    """
    m = json.loads(Path(manifest_path).read_text())
    n_expected = m["n_samples"]
    n_got = scores.shape[0]
    if n_got != n_expected:
        raise ValueError(
            f"score store has {n_got} rows but the manifest describes "
            f"{n_expected} training samples ({m.get('dataset')}, "
            f"input_ids_sha256_16={m.get('input_ids_sha256_16')}). The scores "
            "are indexed against a different dataset."
        )
    return {"n_samples": n_got, "manifest": m}


def to_frame(scores: np.ndarray, manifest_path: str | Path,
             supervised_counts: np.ndarray | None = None):
    """Tidy frame with raw and length-normalised scores.

    STATUS.md records that raw influence is gradient-norm dominated
    (corr(|raw|, ||grad_train||) = 0.785 in a controlled run), which tracks
    response length. Both columns are always produced so no analysis silently
    depends on one convention.
    """
    import pandas as pd

    check_alignment(scores, manifest_path)
    s = scores if scores.ndim == 1 else scores.mean(axis=1)
    df = pd.DataFrame({"row": np.arange(len(s)), "score": s})
    if scores.ndim == 2:
        df["score_std_over_queries"] = scores.std(axis=1)
        df["n_queries"] = scores.shape[1]
    if supervised_counts is not None:
        if len(supervised_counts) != len(s):
            raise ValueError("supervised_counts length mismatch")
        df["n_supervised"] = supervised_counts
        df["score_per_token"] = s / np.maximum(supervised_counts, 1)
    return df


def sanity_top_k(df, texts: list[str], k: int = 10) -> list[tuple[float, str]]:
    """The cheap check that row alignment survived: read the top samples.

    If the highest-influence training examples for a cheese-preference query are
    not about cheese preference, the index is permuted — a failure no aggregate
    statistic can reveal, because a permutation leaves them all unchanged.
    """
    if len(texts) != len(df):
        raise ValueError(f"{len(texts)} texts vs {len(df)} score rows")
    top = df.nlargest(k, "score")
    return [(float(r.score), texts[int(r.row)]) for r in top.itertuples()]


def by_source(df, sources: list[str]) -> dict:
    """Influence broken down by training-data source — CLAUDE.md §5.1's null control.

    §5.1 specifies the influence training set as the AFT spec data **and** the
    instruction-tuning mix, with the IT samples acting as a null distribution:
    "if they score as influential on misalignment queries as spec data does,
    something is wrong."

    This was not checkable while the index held cheese rows only. With the IT
    mix trained in, the index spans nine unrelated sources, so a query about
    cheese preference gives a direct read on whether SOURCE's ranking tracks
    task relevance or merely gradient magnitude.

    Returns per-source score statistics plus `top_share`, the fraction of the
    top-1% most influential rows each source contributes relative to its share
    of the corpus. A ratio near 1 means that source is no more influential than
    chance; the task data should sit well above 1 and the IT sources below it.
    """
    import numpy as np

    if len(sources) != len(df):
        raise ValueError(f"{len(sources)} sources vs {len(df)} score rows")
    v = df["score"].to_numpy()
    src = np.asarray(sources)
    n = len(v)
    k = max(1, n // 100)
    top = set(np.argsort(-v)[:k].tolist())

    out = {}
    for s_ in sorted(set(src)):
        m = src == s_
        share = float(m.mean())
        in_top = float(np.mean([src[i] == s_ for i in top]))
        out[s_] = {
            "n": int(m.sum()),
            "corpus_share": share,
            "mean_score": float(v[m].mean()),
            "median_score": float(np.median(v[m])),
            "frac_positive": float((v[m] > 0).mean()),
            "top1pct_share": in_top,
            # >1 means over-represented among the most influential rows.
            "top_share_ratio": float(in_top / share) if share else float("nan"),
        }
    return out
