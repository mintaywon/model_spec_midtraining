"""Step 2 — proponents and opponents of the misaligned action among L3 rows.

    python -m tda.hyp.rank --run ekfac_phil32b_L3_am-dev_<stamp>

Pulls the EK-FAC and grad-dot score stores from the results volume, joins them
to the training rows through `score_index/rows.jsonl`, runs the sanity checks
PLAN_HYP.md step 1.4 asks for, and writes the ranked sets.

SIGN — measured, not assumed (`bergson_app.py::sign_check`, 2026-09-28).
`_oriented` returns **LOSS-SIGNED** scores: a PROPONENT is NEGATIVE. Scoring a
training document against itself as the query stores +||g||^2 and `_oriented`
returns -||g||^2, the most negative row of the index. `DECISIONS.md` §H7 says the
same; the docstrings in `_phil_attr_impl` / `removal_sets_phil` that call
`_oriented` "proponent-positive" are WRONG, and this module believed them until
Taywon asked for the sign to be checked (§J17).

So this module negates `_oriented` exactly once, in `proponent_positive()`, and
every array downstream of it is **proponent-positive**:
    score > 0  raises logp(misaligned action)   -> harmful row
    score < 0  lowers it                         -> protective row
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from tda.hyp.common import HYP_DIR, LOWERS, RAISES, ROOT, load_l3, write_jsonl

VOLUME = "msm-tda-results"
MODAL = str(ROOT / ".venv" / "bin" / "modal")
STORE_FILES = ("scores.bin", "info.json", "config.yaml")


def pull(run: str, dest: Path):
    """Score stores + the row map. Only the three files a store is read from —
    each store also holds multi-GB factor copies that are not needed here."""
    dest.mkdir(parents=True, exist_ok=True)
    for kind in ("ekfac", "graddot"):
        d = dest / f"scores_{kind}"
        d.mkdir(exist_ok=True)
        for f in STORE_FILES:
            if not (d / f).exists():
                subprocess.run([MODAL, "volume", "get", VOLUME,
                                f"bergson/phil/l3/attr/{run}/scores_{kind}/{f}",
                                str(d / f), "--force"], check=False,
                               capture_output=True)
    for src, name in ((f"bergson/phil/l3/attr/{run}/report.json", "report.json"),
                      ("bergson/phil/l3/score_index/rows.jsonl", "rows.jsonl"),
                      ("bergson/phil/l3/score_index/manifest.json", "manifest.json"),
                      ("bergson/phil/l3/prep_report.json", "prep_report.json")):
        if not (dest / name).exists():
            subprocess.run([MODAL, "volume", "get", VOLUME, src, str(dest / name),
                            "--force"], check=True, capture_output=True)


def proponent_positive(store: Path):
    """The ONE place the sign is set. `_oriented` is loss-signed (proponents
    negative, measured by `sign_check`), so proponent-positive is its negation."""
    from tda.influence.source.scores import _oriented

    return -_oriented(store)


def _sha(msgs) -> str:
    return hashlib.sha256(
        json.dumps(msgs, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def residualise(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    X1 = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return y - X1 @ beta


def build(run: str, n_extreme: int = 200) -> dict:
    import pandas as pd
    from scipy.stats import spearmanr

    from tda.influence.scoring import gini, topk_mass
    from tda.influence.source.scores import check_alignment

    raw = HYP_DIR / "scores" / run
    pull(run, raw)
    rows = [json.loads(l) for l in (raw / "rows.jsonl").read_text().splitlines()]
    ek = proponent_positive(raw / "scores_ekfac")
    check_alignment(ek, raw / "manifest.json")
    if len(rows) != len(ek):
        raise ValueError(f"{len(rows)} index rows vs {len(ek)} scores")
    gd = None
    if (raw / "scores_graddot" / "scores.bin").exists():
        gd = proponent_positive(raw / "scores_graddot")

    df = pd.DataFrame(rows)
    df["score_ekfac"] = ek
    if gd is not None:
        df["score_graddot"] = gd
    df["log_sup"] = np.log(df["n_supervised"].clip(lower=1))

    # Row identity, not just row count: the sha of every task row's messages
    # must equal the local L3 file's. A permuted join would pass every
    # aggregate check below and destroy only which rows are which.
    l3 = load_l3()
    task = df[df.source == "aft_task"].reset_index(drop=True)
    if len(task) != len(l3):
        raise ValueError(f"{len(task)} scored task rows vs {len(l3)} local L3 rows")
    bad = [k for k, r in enumerate(l3) if _sha(r["messages"]) != task.sha[k]
           or int(task.file_row[k]) != k]
    if bad:
        raise ValueError(f"{len(bad)} task rows do not match the local L3 file "
                         f"(first: {bad[:5]}) — score rows are misaligned")

    rep: dict = {"run": run, "n_rows": len(df), "n_task": len(task),
                 "n_it": int((df.source == "aft_it").sum()),
                 "convention": "proponent-positive: score>0 raises logp(misaligned action). "
                               "Set by negating _oriented once (loss-signed, measured "
                               "by sign_check 2026-09-28)."}

    # ---- sanity checks (PLAN_HYP step 1.4) ----------------------------------
    for src in ("aft_task", "aft_it"):
        v = df.loc[df.source == src, "score_ekfac"].to_numpy()
        rep[f"ekfac_{src}"] = {
            "mean": float(v.mean()), "median": float(np.median(v)),
            "abs_mean": float(np.abs(v).mean()), "std": float(v.std()),
            "frac_positive": float((v > 0).mean())}
    k1 = max(1, len(df) // 100)
    a = np.abs(df.score_ekfac.to_numpy())
    top_abs = np.argsort(-a)[:k1]                 # largest |score|, either polarity
    rep["null_control"] = {
        "task_share_of_corpus": float((df.source == "aft_task").mean()),
        "task_share_of_top1pct_abs": float((df.source.to_numpy()[top_abs] == "aft_task").mean()),
        "abs_mean_ratio_task_over_it": rep["ekfac_aft_task"]["abs_mean"]
        / max(rep["ekfac_aft_it"]["abs_mean"], 1e-30)}
    if gd is not None:
        rep["spearman_ekfac_graddot_all"] = float(spearmanr(ek, gd).statistic)
        rep["spearman_ekfac_graddot_task"] = float(
            spearmanr(task.score_ekfac, task.score_graddot).statistic)
    t = task.score_ekfac.to_numpy()
    rep["length_confound_task"] = {
        "spearman_score_vs_log_sup": float(spearmanr(t, task.log_sup).statistic),
        "spearman_abs_score_vs_log_sup": float(spearmanr(np.abs(t), task.log_sup).statistic)}
    rep["concentration_task"] = {
        "gini_abs": float(gini(np.abs(t))),
        "topk_mass_abs": {str(k): float(v) for k, v in topk_mass(np.abs(t)).items()}}

    # ---- ranking ------------------------------------------------------------
    task["score_resid"] = residualise(t, task[["log_sup"]].to_numpy())
    task["user"] = [r["messages"][0]["content"] for r in l3]
    task["response"] = [r["messages"][1]["content"] for r in l3]
    task["chars"] = task.response.str.len()
    # rank 0 = strongest PROPONENT (most positive proponent-positive score)
    task["rank_proponent"] = (-task.score_ekfac).rank(method="first").astype(int) - 1
    task["decile"] = pd.qcut(task.score_ekfac, 10, labels=False)

    # Local names `proponents` / `opponents` below are QUERY-relative (bergson's
    # sense). They are written out under the ALIGNMENT-anchored names RAISES
    # (= alignment opponents) and LOWERS (= alignment proponents).
    order = np.argsort(-t)                        # DESCENDING raises-misaligned-positive
    proponents = task.iloc[order[:n_extreme]]     # raise logp(misaligned) -> ALIGNMENT OPPONENTS
    opponents = task.iloc[order[::-1][:n_extreme]]  # lower logp(misaligned) -> ALIGNMENT PROPONENTS
    assert proponents.score_ekfac.min() >= opponents.score_ekfac.max()
    assert proponents.score_ekfac.mean() > 0 > opponents.score_ekfac.mean(), \
        "extreme sets do not straddle zero — check the sign convention"
    # Tie the labels to the RAW store, so a future change to `_oriented` cannot
    # silently swap them: a proponent has a POSITIVE stored dot product.
    from tda.influence.source.scores import load_source_scores
    stored, _ = load_source_scores(raw / "scores_ekfac")
    stored_task = stored[: len(task)]
    assert stored_task[proponents.file_row.to_numpy()].min() > 0, \
        "a 'proponent' has a non-positive stored score — the sign is inverted"
    assert stored_task[opponents.file_row.to_numpy()].max() < 0, \
        "an 'opponent' has a non-negative stored score — the sign is inverted"

    # Neutral set: |score| smallest, LENGTH-MATCHED to the extremes so that the
    # hypothesiser cannot read "long vs short" off the comparison group.
    ext_len = np.concatenate([proponents.log_sup, opponents.log_sup])
    lo, hi = np.quantile(ext_len, [0.1, 0.9])
    pool = task[(task.log_sup >= lo) & (task.log_sup <= hi)
                & ~task.file_row.isin(set(proponents.file_row) | set(opponents.file_row))]
    neutrals = pool.iloc[np.argsort(np.abs(pool.score_ekfac.to_numpy()))[:n_extreme]]

    cols = ["file_row", "corpus_row", "score_ekfac", "score_resid", "n_supervised",
            "chars", "rank_proponent", "user", "response"]
    if gd is not None:
        cols.insert(3, "score_graddot")
    out = HYP_DIR / "ranking"
    out.mkdir(parents=True, exist_ok=True)
    for name, part in ((RAISES, proponents), (LOWERS, opponents),
                       ("neutrals", neutrals)):
        write_jsonl(out / f"{name}.jsonl", part[cols].to_dict("records"))
        rep[name] = {"n": len(part),
                     "score_range": [float(part.score_ekfac.min()),
                                     float(part.score_ekfac.max())],
                     "median_chars": float(part.chars.median()),
                     "median_n_supervised": float(part.n_supervised.median())}
    if gd is not None:
        gorder = np.argsort(-task.score_graddot.to_numpy())   # descending, proponent-positive
        rep["extreme_overlap_with_graddot"] = {
            RAISES: len(set(order[:n_extreme]) & set(gorder[:n_extreme])) / n_extreme,
            LOWERS: len(set(order[::-1][:n_extreme]) & set(gorder[::-1][:n_extreme])) / n_extreme}
    task.drop(columns=["user", "response"]).to_parquet(out / "ranking.parquet")
    df.drop(columns=["sha"]).to_parquet(out / "all_rows.parquet")
    (out / "report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    build(a.run, a.n)
