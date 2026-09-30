"""Step 4 — does each hypothesis hold on rows it was not derived from?

    python -m tda.hyp.verify sample              # validation sample (disjoint from discovery)
    python -m tda.hyp.verify annotate --batch    # blind 0-3 ratings, all hypotheses per row
    python -m tda.hyp.verify analyse             # -> results/hyp/verification.json

The annotator never sees an influence score or a group label. Rows are sent in
a shuffled order under opaque ids.

SIGN: scores are proponent-positive. A hypothesis with `predicted_sign` = +1
claims its feature is more common among PROPONENTS (rows that raise logp of the
misaligned action), so corr(feature, score) should be > 0; -1 claims the
feature marks OPPONENTS, so the correlation should be < 0.
"""

from __future__ import annotations

import argparse
import json
import random
import re

import numpy as np

from tda.hyp.common import (HYP_DIR, LOWERS, PROMPTS, RAISES, ROOT, ROUND, cost_usd, last_json, read_jsonl,
                            request, run, write_jsonl, load_l3)

RANK = HYP_DIR / "ranking"
OUT = HYP_DIR / f"verify{ROUND}"
N_RETEST = 150
_WORD = re.compile(r"[a-z][a-z'-]{2,}")
_STOP = set("""the and that this with for are was were have has had not but you your
they their them from what when which would could should about there been being into
than then also more most some such only other these those will can may might just
its it's i'm i've don't doesn't isn't one all any how why who because while where
very much many own over under out our ours his her him she he we us me my mine
is be to of in a an as at by on or if so do does did it no yes up""".split())


def hypotheses() -> list[dict]:
    return json.loads((HYP_DIR / f"hypotheses{ROUND}.json").read_text())["hypotheses"]


def stage_sample(n_total: int = 1500, seed: int = 0):
    """Held-back extremes + a decile-stratified draw, none of it from discovery."""
    import pandas as pd

    OUT.mkdir(parents=True, exist_ok=True)
    split = json.loads((HYP_DIR / "hypotheses" / "split.json").read_text())
    disc = set()
    for g in split.values():
        disc |= set(g["discovery"])
    held = sorted(set(split[RAISES]["validation"])
                  | set(split[LOWERS]["validation"]))
    df = pd.read_parquet(RANK / "ranking.parquet")
    pool = df[~df.file_row.isin(disc | set(held))]
    rng = np.random.default_rng(seed)
    per = (n_total - len(held)) // 10
    strat = []
    for d in range(10):
        rows = pool.loc[pool.decile == d, "file_row"].to_numpy()
        strat += [int(x) for x in rng.choice(rows, size=min(per, len(rows)), replace=False)]
    rows = sorted(set(held) | set(strat))
    assert not (set(rows) & disc), "validation sample overlaps discovery rows"
    retest = [int(x) for x in rng.choice(rows, size=min(N_RETEST, len(rows) // 2),
                                        replace=False)]
    (OUT / "sample.json").write_text(json.dumps(
        {"rows": rows, "held_back_extremes": held, "retest": sorted(retest),
         "n_discovery_excluded": len(disc)}, indent=1))
    print(f"validation sample: {len(rows)} rows ({len(held)} held-back extremes, "
          f"{len(strat)} stratified), {len(retest)} annotated twice")


def _features_block(hyps: list[dict]) -> str:
    # Polarity, mechanism and direction are deliberately withheld.
    return "\n\n".join(
        f'<feature id="{h["id"]}">\nname: {h["name"]}\ndefinition: {h["definition"]}\n'
        f'rubric: {h["rubric"]}\n</feature>' for h in hyps)


SECOND_ANNOTATOR = "claude-opus-5-5"   # a different model, not a second draw of the same one


def annotation_requests(file_rows: list[int], hyps: list[dict], tag: str,
                        model: str | None = None) -> list:
    l3 = load_l3()
    system = (PROMPTS / "annotate_v1.txt").read_text().replace(
        "{features}", _features_block(hyps))
    reqs = []
    for fr in file_rows:
        r = l3[fr]
        user = (f"<user_message>\n{r['messages'][0]['content']}\n</user_message>\n\n"
                f"<assistant_response>\n{r['messages'][1]['content']}\n</assistant_response>")
        kw = {"model": model} if model else {}
        reqs.append((f"{tag}_{fr}", request(system, user, max_tokens=6000,
                                            effort="low", **kw)))
    return reqs


def parse_annotation(text: str, hyps: list[dict]) -> dict | None:
    ids = tuple(h["id"] for h in hyps)
    d = last_json(text, required=ids)
    if d is None:
        return None
    out = {}
    for i in ids:
        v = d[i]
        s = v.get("score") if isinstance(v, dict) else v
        if not isinstance(s, (int, float)) or not 0 <= s <= 3:
            return None
        out[i] = {"score": int(round(s)),
                  "evidence": (v.get("evidence", "") if isinstance(v, dict) else "")}
    return out


def annotate(file_rows: list[int], hyps: list[dict], tag: str, out_file, batch: bool,
             state_file, model: str | None = None) -> list[dict]:
    done = {r["file_row"] for r in read_jsonl(out_file) if "ann" in r}
    todo = [fr for fr in file_rows if fr not in done]
    rng = random.Random(0)
    rng.shuffle(todo)
    reqs = annotation_requests(todo, hyps, tag, model)
    res = run(reqs, state_file, tag, batch, concurrency=16)
    recs = []
    for fr, (cid, _) in zip(todo, reqs):
        r = res.get(cid, {"error": "missing"})
        if "error" in r:
            continue
        ann = parse_annotation(r["text"], hyps)
        if ann is not None:
            recs.append({"file_row": fr, "ann": ann, "usage": r["usage"],
                         "via": r.get("via")})
    from tda.hyp.common import append_jsonl
    append_jsonl(out_file, recs)
    print(f"[{tag}] {len(recs)}/{len(todo)} annotated, ${cost_usd(res)}")
    return recs


def stage_annotate(batch: bool):
    s = json.loads((OUT / "sample.json").read_text())
    hyps = hypotheses()
    annotate(s["rows"], hyps, "ann", OUT / "annotations.jsonl", batch, OUT / "batches.json")
    # Reliability is INTER-MODEL agreement. A second draw from the same model at
    # low effort is near-deterministic (dry run: kappa 0.89-0.98 on every
    # feature), so test-retest would certify any rubric, however vague.
    annotate(s["retest"], hyps, "second", OUT / "annotations_retest.jsonl", batch,
             OUT / "batches.json", model=SECOND_ANNOTATOR)
    # one direct-call sweep for anything that failed to parse
    annotate(s["rows"], hyps, "ann_fix", OUT / "annotations.jsonl", False, OUT / "batches.json")


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def weighted_kappa(a: np.ndarray, b: np.ndarray, k: int = 4) -> float:
    """Quadratic-weighted Cohen's kappa for two 0..k-1 ratings."""
    O = np.zeros((k, k))
    for x, y in zip(a, b):
        O[int(x), int(y)] += 1
    O /= O.sum()
    E = np.outer(O.sum(1), O.sum(0))
    W = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)]) / (k - 1) ** 2
    den = (W * E).sum()
    return float(1 - (W * O).sum() / den) if den > 0 else float("nan")


def ols(y: np.ndarray, X: np.ndarray):
    """(beta, t, p) with an intercept added; two-sided normal-approx p."""
    from scipy.stats import t as tdist

    X1 = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    res = y - X1 @ beta
    dof = len(y) - X1.shape[1]
    s2 = res @ res / dof
    cov = s2 * np.linalg.pinv(X1.T @ X1)
    se = np.sqrt(np.diag(cov))
    tv = beta / se
    return beta, tv, 2 * tdist.sf(np.abs(tv), dof), 1 - (res @ res) / ((y - y.mean()) @ (y - y.mean()))


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(feature in a random `pos` row > feature in a random `neg` row), ties 0.5."""
    from scipy.stats import rankdata

    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def query_vocab() -> set:
    """Content words of the AM prompts — the lexical-overlap control. EK-FAC's
    category structure was 92% lexical overlap with the query on cheese
    (CLAUDE.md §5.3), so every effect is reported with this partialled out."""
    words: set = set()
    for l in (ROOT / "results" / "aft" / "am_prompts_27.jsonl").read_text().splitlines():
        p = json.loads(l)
        for k in ("system_prompt", "user_prompt", "email_content"):
            words |= set(_WORD.findall(p.get(k, "").lower()))
    return words - _STOP


def lexical_density(text: str, vocab: set) -> float:
    w = [x for x in _WORD.findall(text.lower()) if x not in _STOP]
    return sum(x in vocab for x in w) / max(1, len(w))


def stage_analyse(n_boot: int = 2000, seed: int = 0):
    import pandas as pd
    from scipy.stats import spearmanr

    hyps = hypotheses()
    ids = [h["id"] for h in hyps]
    sample = json.loads((OUT / "sample.json").read_text())
    ann = {r["file_row"]: r["ann"] for r in read_jsonl(OUT / "annotations.jsonl")}
    ret = {r["file_row"]: r["ann"] for r in read_jsonl(OUT / "annotations_retest.jsonl")}
    rank = pd.read_parquet(RANK / "ranking.parquet").set_index("file_row")
    l3 = load_l3()
    vocab = query_vocab()

    rows = [fr for fr in sample["rows"] if fr in ann]
    df = rank.loc[rows].copy()
    for i in ids:
        df[i] = [ann[fr][i]["score"] for fr in rows]
    df["lex"] = [lexical_density(l3[fr]["messages"][0]["content"] + " "
                                 + l3[fr]["messages"][1]["content"], vocab) for fr in rows]
    # Standardise the target; rank-transform it as well, because influence
    # scores are heavy-tailed and OLS on the raw score is dominated by a few rows.
    y_raw = df.score_ekfac.to_numpy()
    y = (pd.Series(y_raw).rank().to_numpy() - 0.5) / len(y_raw)
    from scipy.stats import norm
    y = norm.ppf(y)                                 # rank-normalised score
    ctrl = np.column_stack([df.log_sup.to_numpy(), df.lex.to_numpy()])
    ctrl = (ctrl - ctrl.mean(0)) / ctrl.std(0)

    split = json.loads((HYP_DIR / "hypotheses" / "split.json").read_text())
    pro = [fr for fr in split[RAISES]["validation"] if fr in ann]
    opp = [fr for fr in split[LOWERS]["validation"] if fr in ann]
    has_gd = "score_graddot" in df.columns
    rng = np.random.default_rng(seed)
    n_h = len(ids)
    out = {"convention": "proponent-positive; predicted_sign is the expected sign of "
                         "corr(feature, score)",
           "n_rows": len(rows), "n_held_back_proponents": len(pro),
           "n_held_back_opponents": len(opp), "bonferroni_over": n_h,
           "controls": ["log supervised tokens", "lexical density vs AM prompts"],
           "control_r2": float(ols(y, ctrl)[3]), "hypotheses": []}

    for h in hyps:
        i = h["id"]
        x = df[i].to_numpy().astype(float)
        rec = {"id": i, "name": h["name"], "predicted_sign": h["predicted_sign"],
               "editable": h["editable"],
               "prevalence": {str(k): float((x == k).mean()) for k in range(4)},
               "mean": float(x.mean())}
        both = [fr for fr in ret if fr in ann]
        rec["kappa_second_model"] = (weighted_kappa(
            np.array([ann[fr][i]["score"] for fr in both]),
            np.array([ret[fr][i]["score"] for fr in both])) if len(both) > 20 else None)
        if x.std() == 0:
            rec.update(status="degenerate", verified=False)
            out["hypotheses"].append(rec)
            continue
        rho = spearmanr(x, y_raw).statistic
        boots = []
        for _ in range(n_boot):
            b = rng.integers(0, len(x), len(x))
            if x[b].std() > 0:
                boots.append(spearmanr(x[b], y_raw[b]).statistic)
        rec["spearman"] = float(rho)
        rec["spearman_ci95"] = [float(np.quantile(boots, 0.025)),
                                float(np.quantile(boots, 0.975))]
        xs = (x - x.mean()) / x.std()
        b0, t0, p0, r20 = ols(y, xs[:, None])
        b1, t1, p1, r21 = ols(y, np.column_stack([xs, ctrl]))
        rec["raw"] = {"beta": float(b0[1]), "t": float(t0[1]), "p": float(p0[1]),
                      "r2": float(r20)}
        rec["controlled"] = {"beta": float(b1[1]), "t": float(t1[1]), "p": float(p1[1]),
                             "p_bonferroni": float(min(1.0, p1[1] * n_h)),
                             "r2_with_controls": float(r21)}
        if pro and opp:
            xp = np.array([ann[fr][i]["score"] for fr in pro], float)
            xo = np.array([ann[fr][i]["score"] for fr in opp], float)
            from scipy.stats import mannwhitneyu
            rec["extremes"] = {"mean_proponents": float(xp.mean()),
                               "mean_opponents": float(xo.mean()),
                               "auc_proponent_gt_opponent": auc(xp, xo),
                               "p_mannwhitney": float(mannwhitneyu(xp, xo).pvalue)
                               if (xp.std() + xo.std()) > 0 else 1.0}
            # |influence| vs sign: mean of the feature in the extreme deciles
            # against the middle four. A feature that is high in BOTH tails
            # marks relevance to the query, not polarity.
            dec = df["decile"].to_numpy()
            rec["u_shape"] = {"decile0_opponents": float(x[dec == 0].mean()),
                              "deciles3to6": float(x[np.isin(dec, [3, 4, 5, 6])].mean()),
                              "decile9_proponents": float(x[dec == 9].mean())}
        if has_gd:
            rec["spearman_graddot"] = float(spearmanr(x, df.score_graddot).statistic)
            rec["sign_agrees_graddot"] = bool(
                np.sign(rec["spearman_graddot"]) == np.sign(rho))
        sign_ok = np.sign(b1[1]) == h["predicted_sign"]
        k_ok = rec["kappa_second_model"] is not None and rec["kappa_second_model"] >= 0.4
        rec["sign_as_predicted"] = bool(sign_ok)
        rec["verified"] = bool(sign_ok and k_ok and rec["controlled"]["p_bonferroni"] < 0.01)
        rec["status"] = ("verified" if rec["verified"] else
                         "wrong sign" if not sign_ok else
                         "unreliable annotation" if not k_ok else "not significant")
        out["hypotheses"].append(rec)

    # ---- simulation: can the annotations predict influence out of sample? ----
    live = [h["id"] for h in out["hypotheses"] if h.get("status") != "degenerate"]
    X = df[live].to_numpy().astype(float)
    X = (X - X.mean(0)) / X.std(0)
    perm = rng.permutation(len(y))
    folds = np.array_split(perm, 5)

    def cv(Xm):
        pred = np.zeros(len(y))
        for f in folds:
            tr = np.setdiff1d(perm, f)
            X1 = np.column_stack([np.ones(len(tr)), Xm[tr]])
            beta, *_ = np.linalg.lstsq(X1, y[tr], rcond=None)
            pred[f] = np.column_stack([np.ones(len(f)), Xm[f]]) @ beta
        return {"spearman": float(spearmanr(pred, y_raw).statistic),
                "r2": float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())}

    out["simulation_5fold"] = {
        "controls_only": cv(ctrl),
        "features_only": cv(X),
        "features_plus_controls": cv(np.column_stack([X, ctrl])),
        "per_feature": {i: cv(X[:, [k]]) for k, i in enumerate(live)}}

    # ---- selection for step 5 ------------------------------------------------
    edit_w = {"yes": 1.0, "partly": 0.5, "no": 0.0}
    cand = [h for h in out["hypotheses"] if "controlled" in h]
    for h in cand:
        h["selection_score"] = (abs(h["controlled"]["beta"]) * edit_w.get(h["editable"], 0)
                                * (1.0 if h["sign_as_predicted"] else 0.0))
    ranked = sorted(cand, key=lambda h: (-int(h["verified"]), -h["selection_score"]))
    out["selected"] = [{"id": h["id"], "name": h["name"], "verified": h["verified"],
                        "selection_score": h["selection_score"]}
                       for h in ranked if h["selection_score"] > 0][:3]
    df.to_parquet(OUT / "annotated.parquet")
    (HYP_DIR / f"verification{ROUND}.json").write_text(json.dumps(out, indent=2))
    for h in out["hypotheses"]:
        c = h.get("controlled", {})
        print(f"{h['id']} {h['name'][:44]:44s} pred={h['predicted_sign']:+d} "
              f"rho={h.get('spearman', float('nan')):+.3f} "
              f"beta_c={c.get('beta', float('nan')):+.3f} "
              f"p_bonf={c.get('p_bonferroni', float('nan')):.2g} "
              f"kappa={h['kappa_second_model']} -> {h['status']}")
    print("simulation:", json.dumps(out["simulation_5fold"]["features_plus_controls"]),
          "controls only:", json.dumps(out["simulation_5fold"]["controls_only"]))
    print("selected:", [s["id"] for s in out["selected"]])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["sample", "annotate", "analyse"])
    ap.add_argument("--batch", action="store_true")
    ap.add_argument("--n", type=int, default=1500)
    a = ap.parse_args()
    if a.stage == "sample":
        stage_sample(a.n)
    elif a.stage == "annotate":
        stage_annotate(a.batch)
    else:
        stage_analyse()
