"""What separates EK-FAC's most protective philosophy documents from its most harmful.

Rebuilds STATUS.md 8.10 (first-person register) from the score store, since the
original pass was ad hoc and left no script. Writes to results/prof_figs/data/.

  python -m tda.analysis.register_tails <store>

<store> holds `scores_ekfac/`, `scores_graddot/` (scores.bin, info.json, config.yaml
from msm-tda-results:/bergson/phil/attr/ekfac_phil32b_union-am-dev-full/) and
`manifest.json` (from /bergson/phil/score_index/). Optional `compare_phil.json`
enables the per-document alignment check.

🔴 SIGN CORRECTION 2026-09-28 (CLAUDE.md §5.1, DECISIONS.md §J17/§K1): the ORIENTATION
paragraph below is INVERTED. `_oriented` is LOSS-SIGNED, so the most NEGATIVE documents
RAISE logp(misaligned action): they are the alignment OPPONENTS (harmful), and the most
positive are the alignment PROPONENTS (protective). The variables `protective` /
`harmful` below are therefore SWAPPED. Left as run, because STATUS.md §8.10 was
produced by this code; fix the two names before reusing it.

ORIENTATION. `_oriented` scores are proponent-positive w.r.t. the MISALIGNED action
span (CLAUDE.md 5.1), so the most NEGATIVE documents are its strongest OPPONENTS,
i.e. the most protective. Every sort below says which end it takes.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from tda.influence.source.scores import _oriented

OUT = Path(__file__).resolve().parents[2] / "results" / "prof_figs" / "data"
CORPUS = "chloeli/msm-qwen-philosophy-spec"
K = 400  # documents per tail, as in STATUS 8.10c
TOK = re.compile(r"[a-z]+(?:'[a-z]+)?")
FIRST = {"i", "i'm", "i've", "i'd", "i'll", "me", "my", "mine", "myself"}


def load(store: Path) -> pd.DataFrame:
    from datasets import load_dataset

    man = json.loads((store / "manifest.json").read_text())
    rows, n = man["msm_corpus_rows"], man["n_msm"]
    ds = load_dataset(CORPUS, split="train")
    df = pd.DataFrame({
        "row": range(n), "corpus_row": rows,
        "ekfac": _oriented(store / "scores_ekfac")[:n],
        "graddot": _oriented(store / "scores_graddot")[:n],
        "domain": [ds[i]["domain"] for i in rows], "text": [ds[i]["text"] for i in rows]})
    # A permuted join leaves every aggregate below unchanged, so check documents, not means.
    ref = store / "compare_phil.json"
    if ref.exists():
        ext = json.loads(ref.read_text())["extremes"]["EK-FAC"]
        for e in ext["proponents"] + ext["opponents"]:
            r = df.iloc[e["row"]]
            assert r.corpus_row == e["corpus_row"] and abs(r.ekfac - e["score"]) < 1e-3 \
                and r.text[:80] == e["text"][:80] and r.domain == e["domain"], f"row {e['row']} misaligned"
    return df


def log_odds(toks, a_idx, b_idx, min_count=200, prior=500.0) -> pd.DataFrame:
    """z-scored log-odds of a vs b with an informative Dirichlet prior (Monroe et al. 2008)."""
    bg = Counter(w for t in toks for w in t)
    ca, cb = (Counter(w for i in idx for w in toks[i]) for idx in (a_idx, b_idx))
    na, nb, tot = sum(ca.values()), sum(cb.values()), sum(bg.values())
    out = []
    for w, n in bg.items():
        if n < min_count:
            continue
        a = prior * n / tot
        d = np.log((ca[w] + a) / (na + prior - ca[w] - a)) - np.log((cb[w] + a) / (nb + prior - cb[w] - a))
        out.append((w, d / np.sqrt(1 / (ca[w] + a) + 1 / (cb[w] + a))))
    return pd.DataFrame(out, columns=["word", "z"]).sort_values("z", ascending=False)


def main(store: Path) -> None:
    df = load(store)
    toks = [TOK.findall(t.lower().replace("’", "'")) for t in df.text]
    df["first_per_1k"] = [1000 * sum(w in FIRST for w in t) / max(len(t), 1) for t in toks]
    df["scratchpad_leak"] = df.text.str.contains("<scratchpad>", regex=False)

    asc = np.argsort(df.ekfac.to_numpy())  # ascending: most NEGATIVE (protective) first
    protective, harmful = asc[:K], asc[-K:][::-1]  # strongest OPPONENTS / strongest PROPONENTS
    df["side"] = "rest"
    df.loc[protective, "side"] = "protective"
    df.loc[harmful, "side"] = "harmful"

    words = log_odds(toks, protective, harmful)
    df["q"] = pd.qcut(df.first_per_1k, 4, labels=False)
    df["decile"] = pd.qcut(df.first_per_1k.rank(method="first"), 10, labels=False)
    dec = df.groupby("decile").agg(density=("first_per_1k", "median"), mean=("ekfac", "mean"),
                                   sem=("ekfac", "sem"))
    gm = df.ekfac.mean()
    eta2 = sum(len(v) * (v.mean() - gm) ** 2 for _, v in df.groupby("domain").ekfac) / ((df.ekfac - gm) ** 2).sum()
    r = float(np.corrcoef(df.ekfac, df.first_per_1k)[0, 1])
    clean = df[~df.scratchpad_leak]

    def title(i):
        return df.text[i].strip().split("\n", 1)[0].lstrip("# ")[:120]

    out = {
        "k": K, "n": len(df), "corr_first_person": r, "r2_first_person": r * r, "eta2_domain": float(eta2),
        "corr_without_scratchpad_leaks": float(np.corrcoef(clean.ekfac, clean.first_per_1k)[0, 1]),
        "n_scratchpad_leaks": int(df.scratchpad_leak.sum()),
        "scratchpad_leaks_in_harmful_tail": int((df.scratchpad_leak & (df.side == "harmful")).sum()),
        "words_protective": words.head(14).values.tolist(),
        "words_harmful": words.tail(14)[::-1].values.tolist(),
        "quartile_edges": [float(df.first_per_1k[df.q == i].max()) for i in range(4)],
        "tail_share_by_quartile": pd.crosstab(df.side, df.q, normalize="index").to_dict(orient="index"),
        "median_first_per_1k": df.groupby("side").first_per_1k.median().to_dict(),
        "deciles": dec.reset_index().to_dict(orient="records"),
        "corr_by_domain": df.groupby("domain").apply(
            lambda g: float(np.corrcoef(g.ekfac, g.first_per_1k)[0, 1])).to_dict(),
        "top_protective": [[float(df.ekfac[i]), title(i), float(df.first_per_1k[i])] for i in protective[:6]],
        "top_harmful": [[float(df.ekfac[i]), title(i), float(df.first_per_1k[i])] for i in harmful[:6]],
    }
    (OUT / "register_phil32b.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if not isinstance(v, (list, dict))}, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
