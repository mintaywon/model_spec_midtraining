"""Step 3 — Sonnet proposes features that discriminate proponents from opponents.

    python -m tda.hyp.hypothesize split          # discovery / validation halves
    python -m tda.hyp.hypothesize propose        # K independent proposer calls
    python -m tda.hyp.hypothesize consolidate    # -> results/hyp/hypotheses.json

Group A = PROPONENTS (proponent-positive score > 0: raise logp of the misaligned
action). Group B = OPPONENTS (score < 0: lower it). The mapping is fixed here
and written into every output file; nothing downstream infers it from order.

Only the DISCOVERY half of the extreme rows is ever shown to a proposer. The
other half is held back for `verify.py`, so a hypothesis cannot be confirmed by
the rows that suggested it.
"""

from __future__ import annotations

import argparse
import json
import random

from tda.hyp.common import (HYP_DIR, LOWERS, PROMPTS, RAISES, ROUND, cost_usd, read_jsonl, request, run,
                            spec_text, tagged, write_jsonl)

RANK = HYP_DIR / "ranking"
OUT = HYP_DIR / "hypotheses"          # split.json lives here for every round
ROUT = HYP_DIR / f"hypotheses{ROUND}"    # this round's proposals
# Group A is SHOWN as "makes the harmful action MORE likely": rows that RAISE
# logp(misaligned) = alignment opponents. Group B = rows that lower it.
GROUPS = {"A": RAISES, "B": LOWERS, "N": "neutrals"}
MAX_CHARS = 6000          # per response shown to a proposer; longer are cut in the middle


def stage_split(seed: int = 0):
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    split = {}
    for g, name in GROUPS.items():
        rows = [r["file_row"] for r in read_jsonl(RANK / f"{name}.jsonl")]
        rng.shuffle(rows)
        h = len(rows) // 2
        split[name] = {"discovery": sorted(rows[:h]), "validation": sorted(rows[h:])}
    (OUT / "split.json").write_text(json.dumps(split, indent=1))
    print({k: {s: len(v) for s, v in d.items()} for k, d in split.items()})


def _clip(t: str) -> str:
    if len(t) <= MAX_CHARS:
        return t
    h = MAX_CHARS // 2
    return t[:h] + "\n[... middle omitted ...]\n" + t[-h:]


def _render(gid: str, r: dict) -> str:
    return (f'<example id="{gid}">\n<user>\n{_clip(r["user"])}\n</user>\n'
            f'<assistant>\n{_clip(r["response"])}\n</assistant>\n</example>')


def stage_propose(k: int, per_group: int, batch: bool, seed: int = 0):
    split = json.loads((OUT / "split.json").read_text())
    rows = {name: {r["file_row"]: r for r in read_jsonl(RANK / f"{name}.jsonl")}
            for name in GROUPS.values()}
    system = (PROMPTS / "propose_v1.txt").read_text().replace("{spec}", spec_text())
    reqs, keys = [], {}
    for j in range(k):
        rng = random.Random(seed * 1000 + j)
        blocks, key = [], {}
        for g, name in GROUPS.items():
            pick = rng.sample(split[name]["discovery"], per_group)
            for n, fr in enumerate(pick):
                gid = f"{g}{n + 1:02d}"
                key[gid] = fr
                blocks.append((g, _render(gid, rows[name][fr])))
        # Groups are shown as blocks (A, B, N) with a header; order inside a
        # group is the random draw.
        user = ""
        for g in GROUPS:
            user += f"\n\n===== GROUP {g} =====\n\n" + "\n\n".join(
                b for gg, b in blocks if gg == g)
        user += "\n\nPropose the discriminating features now."
        reqs.append((f"prop{j}", request(system, user.strip(), max_tokens=48000,
                                         effort="medium")))
        keys[f"prop{j}"] = key
    res = run(reqs, OUT / "batches.json", "propose", batch, concurrency=6)
    recs = []
    for cid, _ in reqs:
        r = res.get(cid, {"error": "missing"})
        rec = {"id": cid, "key": keys[cid]}
        if "error" in r:
            rec["error"] = r["error"]
        else:
            body = tagged(r["text"], "features")
            try:
                rec["features"] = json.loads(body) if body else None
            except json.JSONDecodeError:
                rec["features"] = None
            rec["raw"] = r["text"]
            rec["usage"] = r["usage"]
            if rec["features"] is None:
                # Thinking counts against max_tokens; an empty text with
                # stop_reason max_tokens means the budget went on thinking.
                rec["error"] = f"unparseable (stop_reason={r['stop_reason']})"
        recs.append(rec)
    write_jsonl(OUT / "proposals.jsonl", recs)
    ok = [r for r in recs if r.get("features")]
    print(f"propose: {len(ok)}/{len(recs)} parsed, "
          f"{sum(len(r['features']) for r in ok)} raw features, ${cost_usd(res)}")


def matched_pairs(n_pairs: int, seed: int) -> list[tuple[int, int]]:
    """Discovery proponents paired with their most similar discovery opponent
    (TF-IDF cosine on the USER message), one-to-one, greedy from the most
    similar pair down. Pairing holds the request's topic roughly fixed, so what
    is left to see within a pair is what the response does with it."""
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer

    split = json.loads((OUT / "split.json").read_text())
    pro = {r["file_row"]: r for r in read_jsonl(RANK / f"{RAISES}.jsonl")}
    opp = {r["file_row"]: r for r in read_jsonl(RANK / f"{LOWERS}.jsonl")}
    a, b = split[RAISES]["discovery"], split[LOWERS]["discovery"]
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True)
    X = vec.fit_transform([pro[i]["user"] for i in a] + [opp[i]["user"] for i in b])
    sim = (X[:len(a)] @ X[len(a):].T).toarray()
    # a little seeded jitter so different proposers get different pairings
    sim = sim + np.random.default_rng(seed).normal(scale=0.03, size=sim.shape)
    pairs, used_a, used_b = [], set(), set()
    for k in np.argsort(-sim, axis=None):
        i, j = divmod(int(k), len(b))
        if i in used_a or j in used_b:
            continue
        pairs.append((a[i], b[j]))
        used_a.add(i)
        used_b.add(j)
        if len(pairs) == len(a):
            break
    random.Random(seed).shuffle(pairs)
    return pairs[:n_pairs]


def stage_propose2(k: int, n_pairs: int, batch: bool, seed: int = 0):
    """Round 2: A-vs-B only, topic-matched pairs, round-1 features excluded."""
    ROUT.mkdir(parents=True, exist_ok=True)
    pro = {r["file_row"]: r for r in read_jsonl(RANK / f"{RAISES}.jsonl")}
    opp = {r["file_row"]: r for r in read_jsonl(RANK / f"{LOWERS}.jsonl")}
    r1 = json.loads((HYP_DIR / "hypotheses.json").read_text())["hypotheses"]
    if ROUND == "_r3":      # round 3 also excludes round 2 and is given lexical leads
        r1 = r1 + json.loads((HYP_DIR / "hypotheses_r2.json").read_text())["hypotheses"]
    known = "\n".join(f"- {h['name']}: {h['definition']}" for h in r1)
    hints = HYP_DIR / "lexical_hints.txt"
    if ROUND == "_r3" and hints.exists():
        known += "\n\n" + hints.read_text().strip()
    system = ((PROMPTS / "propose_v2.txt").read_text()
              .replace("{spec}", spec_text()).replace("{known}", known))
    reqs, keys = [], {}
    for j in range(k):
        pairs = matched_pairs(n_pairs, seed * 1000 + j)
        rng = random.Random(seed * 1000 + j)
        user, key = "", {}
        for n, (fa, fb) in enumerate(pairs, 1):
            items = [("A", pro[fa]), ("B", opp[fb])]
            rng.shuffle(items)               # A is not always shown first
            key[str(n)] = {"A": fa, "B": fb}
            user += f"\n\n===== PAIR {n} =====\n\n" + "\n\n".join(
                _render(f"pair{n}-{g}", r) for g, r in items)
        user += "\n\nPropose the features that separate A from B within pairs."
        # effort MEDIUM: at "high" all six calls spent 48k tokens thinking through
        # the pair-by-pair count and returned no text (2026-09-28, $5.51).
        reqs.append((f"prop{j}", request(system, user.strip(), max_tokens=64000,
                                         effort="medium")))
        keys[f"prop{j}"] = key
    res = run(reqs, ROUT / "batches.json", "propose2", batch, concurrency=6)
    recs = []
    for cid, _ in reqs:
        r = res.get(cid, {"error": "missing"})
        rec = {"id": cid, "key": keys[cid]}
        if "error" in r:
            rec["error"] = r["error"]
        else:
            body = tagged(r["text"], "features")
            try:
                rec["features"] = json.loads(body) if body else None
            except json.JSONDecodeError:
                rec["features"] = None
            rec["raw"], rec["usage"] = r["text"], r["usage"]
            if rec["features"] is None:
                rec["error"] = f"unparseable (stop_reason={r['stop_reason']})"
        recs.append(rec)
    write_jsonl(ROUT / "proposals.jsonl", recs)
    ok = [r for r in recs if r.get("features") is not None]
    print(f"propose2: {len(ok)}/{len(recs)} parsed, "
          f"{sum(len(r['features']) for r in ok)} raw features, ${cost_usd(res)}")
    for r in ok:
        for f in r["features"]:
            print(f"  {r['id']} [{f['more_common_in']}] A-only {len(f.get('pairs_A_only', []))} "
                  f"/ B-only {len(f.get('pairs_B_only', []))}  {f['name']}")


def stage_consolidate():
    props = [r for r in read_jsonl(ROUT / "proposals.jsonl") if r.get("features")]
    if len(props) < (1 if ROUND in ("_r2", "_r3") else 3):
        raise SystemExit(f"only {len(props)} proposer calls parsed; rerun propose")
    system = (PROMPTS / "consolidate_v1.txt").read_text()
    user = ""
    for j, p in enumerate(props):
        slim = [{k: v for k, v in f.items()} for f in p["features"]]
        for f in slim:                      # counts, not ids: ids differ per analyst
            f["n_support_A"] = len(f.pop("support_A", None) or f.pop("pairs_A_only", None) or [])
            f["n_support_B"] = len(f.pop("support_B", None) or f.pop("pairs_B_only", None) or [])
        user += f"\n\n<analyst n=\"{j + 1}\">\n{json.dumps(slim, indent=1, ensure_ascii=False)}\n</analyst>"
    if ROUND in ("_r2", "_r3"):
        # Round 2/3 proposers may return few features or none; do not force 6-8.
        system = system.replace(
            "6. Output between 6 and 8 hypotheses.",
            "6. Output every distinct hypothesis that at least one analyst supported "
            "with counts (at most 8). If there are only two, output two.")
    res = run([("cons", request(system, user.strip(), max_tokens=48000, effort="medium"))],
              ROUT / "batches.json", "consolidate", batch=False)
    r = res["cons"]
    if "error" in r:
        raise SystemExit(r["error"])
    body = tagged(r["text"], "hypotheses")
    hyps = json.loads(body)
    prefix = {"": "H", "_r2": "P", "_r3": "Q"}.get(ROUND, "H")   # P = round 2, Q = round 3
    for h in hyps:
        h["id"] = prefix + h["id"].lstrip("HP")
        h["round"] = {"_r2": 2, "_r3": 3}.get(ROUND, 1)
        # Fixed mapping, stated in the artefact itself.
        h["polarity"] = ("align_opponent_feature" if h["more_common_in"] == "A"
                         else "align_proponent_feature")
        h["predicted_sign"] = +1 if h["more_common_in"] == "A" else -1
    out = {"convention": "A = proponents (score>0, raise logp of the misaligned "
                         "action); B = opponents (score<0). predicted_sign is the "
                         "sign of corr(feature, proponent-positive score).",
           "n_proposer_calls": len(props), "hypotheses": hyps}
    (HYP_DIR / f"hypotheses{ROUND}.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))
    (ROUT / "consolidate_raw.txt").write_text(r["text"])
    print(f"consolidate: {len(hyps)} hypotheses, ${cost_usd(res)}")
    for h in hyps:
        print(f"  {h['id']} [{h['polarity']}, editable={h['editable']}, "
              f"analysts={h['n_analysts']}] {h['name']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["split", "propose", "propose2", "consolidate"])
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--per-group", type=int, default=20)
    ap.add_argument("--pairs", type=int, default=25)
    ap.add_argument("--batch", action="store_true")
    a = ap.parse_args()
    if a.stage == "split":
        stage_split()
    elif a.stage == "propose":
        stage_propose(a.k, a.per_group, a.batch)
    elif a.stage == "propose2":
        stage_propose2(a.k, a.pairs, a.batch)
    else:
        stage_consolidate()
