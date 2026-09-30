"""Step 5 — data variants: L3 plus or minus ONE feature, everything else intact.

    python -m tda.hyp.variants corpus --batch            # annotate every L3 row
    python -m tda.hyp.variants plan                      # target rows per variant
    python -m tda.hyp.variants pilot    --variant H2     # 30 target rows, direct calls
    python -m tda.hyp.variants generate --variant H2 --batch
    python -m tda.hyp.variants judge    --variant H2 --batch
    python -m tda.hyp.variants assemble --variant H2
    python -m tda.hyp.variants retry    --variant H2     # failed rows, fresh draw
    python -m tda.hyp.variants review   --variant H2

A variant is the full 9,585-row L3 set in L3's own row order. Rows outside the
variant's target set are BYTE-IDENTICAL to L3. Target rows that fail the judge
fall back to their L3 text instead of being dropped, so every arm — L3, the
variants and the placebo — trains on the same prompts in the same positions and
differs only in the responses that were deliberately edited.

Direction is always "improving" (Taywon, 2026-09-28), fixed by the hypothesis's
polarity, never by a sort:
    align_opponent_feature  (marks rows that RAISE logp of the misaligned action,
                             i.e. rows that hurt alignment)
        -> REMOVE it: target rows rated >= 2, edited toward 0
    align_proponent_feature (marks rows that LOWER it, i.e. rows that help)
        -> ADD it:    target rows rated <= 1, edited toward 3
H3, H4, H8 and PLACEBO were planned on 2026-09-28 under an inverted sign
(DECISIONS §J17) and are edits in the WORSENING direction; H3R is improving.
"""

from __future__ import annotations

import argparse
import difflib
import json
import random
import re

from tda.aft import banned
from tda.hyp.common import (HYP_DIR, PROMPTS, ROUND, append_jsonl, cost_usd, last_json,
                            load_l3, read_jsonl, request, run, spec_text, tagged,
                            write_jsonl)
from tda.hyp.verify import annotate, hypotheses

VDIR = HYP_DIR / "variants"
LEN_LO, LEN_HI = 0.75, 1.30        # edited / L3 characters
CRITERIA = ["decision_preserved", "only_feature_changed", "reasoning_intact",
            "no_meta_language", "no_continuation_desire", "values_consistent",
            "no_scenario_framing"]
P_CRITERIA = [c.replace("only_feature_changed", "only_wording_changed") for c in CRITERIA]
_SENT = re.compile(r"(?<=[.!?])\s+")


def plan() -> dict:
    return json.loads((VDIR / "plan.json").read_text())


def is_placebo(variant: str) -> bool:
    """A placebo is any variant whose operation is NONE (PLACEBO, PLACEBOR, ...)."""
    return plan()["variants"][variant]["operation"] == "NONE"


def hyp_by_id() -> dict:
    return {h["id"]: h for h in hypotheses()}


def corpus_annotations() -> dict:
    """file_row -> {Hk: score}. Validation-sample annotations are reused: they
    came from the same prompt, so re-annotating those rows would buy nothing."""
    ann = {}
    for f in (HYP_DIR / f"verify{ROUND}" / "annotations.jsonl", VDIR / "corpus_annotations.jsonl"):
        for r in read_jsonl(f):
            ann.setdefault(r["file_row"], {k: v["score"] for k, v in r["ann"].items()})
    return ann


def stage_corpus(batch: bool):
    VDIR.mkdir(parents=True, exist_ok=True)
    have = set(corpus_annotations())
    todo = [r["file_row"] for r in load_l3() if r["file_row"] not in have]
    print(f"corpus annotation: {len(todo)} rows to do, {len(have)} already rated")
    annotate(todo, hypotheses(), "corpus", VDIR / "corpus_annotations.jsonl", batch,
             VDIR / "batches.json")
    have = set(corpus_annotations())
    todo = [r["file_row"] for r in load_l3() if r["file_row"] not in have]
    if todo:
        annotate(todo, hypotheses(), "corpus_fix", VDIR / "corpus_annotations.jsonl",
                 False, VDIR / "batches.json")


def stage_plan(select: list[str] | None = None, cap: int = 0, seed: int = 0):
    """`cap` > 0 limits each variant to that many target rows, drawn UNIFORMLY
    from the eligible rows — not by influence. The hypothesis is about the
    feature, so the rows that carry the edit must not be pre-selected on the
    score the hypothesis was derived from."""
    ver = json.loads((HYP_DIR / f"verification{ROUND}.json").read_text())
    chosen = select or [s["id"] for s in ver["selected"]]
    hy = hyp_by_id()
    ann = corpus_annotations()
    l3 = load_l3()
    missing = [r["file_row"] for r in l3 if r["file_row"] not in ann]
    out = {"n_rows": len(l3), "n_unannotated": len(missing), "variants": {}}
    for hid in chosen:
        h = hy[hid]
        if h["polarity"] == "align_opponent_feature":
            op, target, rows = "REMOVE", 0, [fr for fr, a in ann.items() if a[hid] >= 2]
        else:
            op, target, rows = "ADD", 3, [fr for fr, a in ann.items() if a[hid] <= 1]
        n_eligible = len(rows)
        if cap and len(rows) > cap:
            rows = random.Random(seed).sample(sorted(rows), cap)
        out["variants"][hid] = {
            "n_eligible": n_eligible, "cap": cap or None,
            "hypothesis": hid, "name": h["name"], "polarity": h["polarity"],
            "operation": op, "target_level": target, "rows": sorted(rows),
            "n_target": len(rows), "frac_of_corpus": round(len(rows) / len(l3), 3),
            "corpus_mean_before": round(sum(a[hid] for a in ann.values()) / len(ann), 3)}
    # The placebo rewrites the SAME rows as the first selected variant, so the
    # pair differs in what the edit does, not in which rows were touched.
    first = chosen[0]
    out["variants"]["PLACEBO"] = {
        "hypothesis": first, "name": f"placebo on {first}'s rows", "operation": "NONE",
        "rows": out["variants"][first]["rows"],
        "n_target": out["variants"][first]["n_target"],
        "frac_of_corpus": out["variants"][first]["frac_of_corpus"]}
    VDIR.mkdir(parents=True, exist_ok=True)
    (VDIR / "plan.json").write_text(json.dumps(out, indent=1))
    for k, v in out["variants"].items():
        print(f"{k:8s} {v['operation']:6s} {v['n_target']:5d} rows "
              f"({v['frac_of_corpus']:.1%})  {v['name']}")


# ---------------------------------------------------------------------------
# requests
# ---------------------------------------------------------------------------

def _fill(template: str, h: dict, **kw) -> str:
    s = (template.replace("{spec}", spec_text())
         .replace("{feature_name}", h["name"])
         .replace("{feature_definition}", h["definition"])
         .replace("{feature_rubric}", h["rubric"]))
    for k, v in kw.items():
        s = s.replace("{" + k + "}", str(v))
    return s


def edit_instruction(v: dict, h: dict) -> str:
    override = VDIR / f"edit_{v['hypothesis']}.txt"      # hand-tightened after a pilot
    body = (v["edit_override"] if v.get("edit_override") else
            override.read_text().strip() if override.exists() else h["edit_to_improve"])
    verb = ("REMOVE this property from the response" if v["operation"] == "REMOVE"
            else "ADD this property to the response")
    return f"{verb}. {body}"


def n_sentences(text: str) -> int:
    return max(1, len([s for s in _SENT.split(text) if s.strip()]))


def placebo_extent() -> float:
    """Fraction of sentences the first variant's accepted edits changed, so the
    placebo rewrites about as much text. Falls back to 0.25 before any exist."""
    fixed = VDIR / "placebo_extent.json"       # set from the first variant's pilot,
    if fixed.exists():                          # so the four variants can run in parallel
        return float(json.loads(fixed.read_text())["frac_sentences"])
    p = plan()
    first = p["variants"]["PLACEBO"]["hypothesis"]
    ch = [c["frac_sentences_changed"] for c in read_jsonl(VDIR / first / "checks.jsonl")
          if c.get("kept")]
    return sorted(ch)[len(ch) // 2] if len(ch) >= 10 else 0.25


def rewrite_request(variant: str, row: dict) -> dict:
    v = plan()["variants"][variant]
    h = hyp_by_id()[v["hypothesis"]]
    resp = row["messages"][1]["content"]
    if is_placebo(variant):
        k = max(2, round(placebo_extent() * n_sentences(resp)))
        system = (PROMPTS / "placebo_rewrite_v1.txt").read_text().replace(
            "{spec}", spec_text())
        extra = f"\n\nReword about {k} sentences."
        system = system.replace("{n_sentences}", "the number of")
    else:
        system = _fill((PROMPTS / "variant_rewrite_v1.txt").read_text(), h,
                       edit_instruction=edit_instruction(v, h),
                       target_level=v["target_level"])
        extra = ""
    user = (f"<user_query>\n{row['messages'][0]['content']}\n</user_query>\n\n"
            f"<current_response>\n{resp}\n</current_response>{extra}")
    return request(system, user, max_tokens=8000)


def judge_request(variant: str, row: dict, rewritten: str) -> dict:
    v = plan()["variants"][variant]
    h = hyp_by_id()[v["hypothesis"]]
    name = "placebo_judge_v1.txt" if is_placebo(variant) else "variant_judge_v1.txt"
    system = _fill((PROMPTS / name).read_text(), h)
    user = (f"<user_query>\n{row['messages'][0]['content']}\n</user_query>\n\n"
            f"<current_response>\n{row['messages'][1]['content']}\n</current_response>\n\n"
            f"<edited_response>\n{rewritten}\n</edited_response>")
    return request(system, user, max_tokens=12000, effort="medium")


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------

def stage_generate(variant: str, batch: bool, rows: list[int] | None = None, tag="gen"):
    d = VDIR / variant
    d.mkdir(parents=True, exist_ok=True)
    l3 = load_l3()
    done = {r["file_row"] for r in read_jsonl(d / "rewrites.jsonl") if "rewritten" in r}
    pool = rows if rows is not None else plan()["variants"][variant]["rows"]
    todo = [fr for fr in pool if fr not in done]
    print(f"[{variant}] generate: {len(todo)} rows to do ({len(done)} done)")
    reqs = [(f"{variant}_{fr}", rewrite_request(variant, l3[fr])) for fr in todo]
    res = run(reqs, d / "batches.json", tag, batch)
    recs = []
    for fr in todo:
        r = res.get(f"{variant}_{fr}", {"error": "missing"})
        rec = {"file_row": fr}
        if "error" in r:
            rec["error"] = r["error"]
        else:
            rw = tagged(r["text"], "response")
            rec.update(usage=r["usage"], via=r.get("via"), stop_reason=r["stop_reason"])
            if rw:
                rec["rewritten"] = rw
            else:
                rec["error"] = "no <response> tag"
        recs.append(rec)
    append_jsonl(d / "rewrites.jsonl", recs)
    print(f"[{variant}] generate: {sum('rewritten' in r for r in recs)}/{len(recs)} "
          f"parsed, ${cost_usd(res)}")


def stage_judge(variant: str, batch: bool, tag="judge"):
    d = VDIR / variant
    l3 = load_l3()
    crit = P_CRITERIA if is_placebo(variant) else CRITERIA
    rew = {r["file_row"]: r["rewritten"] for r in read_jsonl(d / "rewrites.jsonl")
           if "rewritten" in r}
    done = {r["file_row"] for r in read_jsonl(d / "judge.jsonl") if "verdicts" in r}
    # An unchanged response needs no judge: it is the L3 row.
    todo = sorted(fr for fr in rew if fr not in done
                  and rew[fr].strip() != l3[fr]["messages"][1]["content"].strip())
    print(f"[{variant}] judge: {len(todo)} edited rows")
    reqs = [(f"{variant}_{fr}", judge_request(variant, l3[fr], rew[fr])) for fr in todo]
    res = run(reqs, d / "batches.json", tag, batch)
    recs = []
    for fr in todo:
        r = res.get(f"{variant}_{fr}", {"error": "missing"})
        if "error" in r:
            continue
        v = last_json(r["text"], required=tuple(crit) + ("feature_before", "feature_after"))
        if v is None:
            continue
        recs.append({"file_row": fr, "verdicts": {k: v[k] for k in crit},
                     "feature_before": v["feature_before"],
                     "feature_after": v["feature_after"], "notes": v.get("notes", ""),
                     "usage": r["usage"], "via": r.get("via")})
    append_jsonl(d / "judge.jsonl", recs)
    print(f"[{variant}] judge: {len(recs)}/{len(todo)} parsed, ${cost_usd(res)}")


def edit_stats(a: str, b: str) -> dict:
    sa = [s.strip() for s in _SENT.split(a) if s.strip()]
    sb = [s.strip() for s in _SENT.split(b) if s.strip()]
    same = set(sa) & set(sb)
    return {"char_similarity": round(difflib.SequenceMatcher(None, a, b, autojunk=False)
                                     .ratio(), 4),
            "frac_sentences_changed": round(1 - sum(s in same for s in sa) / max(1, len(sa)), 4),
            "length_ratio": round(len(b) / max(1, len(a)), 4)}


def feature_moved(variant: str, before: int, after: int) -> bool:
    v = plan()["variants"][variant]
    if v["operation"] == "REMOVE":
        return after <= 1 and after < before
    if v["operation"] == "ADD":
        return after >= 2 and after > before
    return after == before                       # placebo: the feature must not move


def stage_assemble(variant: str):
    d = VDIR / variant
    l3 = load_l3()
    crit = P_CRITERIA if is_placebo(variant) else CRITERIA
    targets = plan()["variants"][variant]["rows"]
    rew = {r["file_row"]: r["rewritten"] for r in read_jsonl(d / "rewrites.jsonl")
           if "rewritten" in r}
    jud = {r["file_row"]: r for r in read_jsonl(d / "judge.jsonl")}
    checks, accepted = [], {}
    fails_n: dict = {}
    for fr in targets:
        orig = l3[fr]["messages"][1]["content"]
        rec = {"file_row": fr}
        if fr not in rew:
            rec.update(fails=["no_rewrite"], kept=False)
        elif rew[fr].strip() == orig.strip():
            rec.update(fails=["unchanged"], kept=False)
        else:
            es = edit_stats(orig, rew[fr])
            b = banned.check(rew[fr], orig)
            fails = []
            if not LEN_LO <= es["length_ratio"] <= LEN_HI:
                fails.append("length")
            if not banned.passes(b):
                fails.append("banned")
            if fr in jud:
                j = jud[fr]
                fails += [k for k in crit if j["verdicts"][k] != "PASS"]
                if not feature_moved(variant, int(j["feature_before"]),
                                     int(j["feature_after"])):
                    fails.append("feature_not_moved")
                rec.update(feature_before=j["feature_before"],
                           feature_after=j["feature_after"], notes=j["notes"])
            else:
                fails.append("unjudged")
            rec.update(es, banned=b, fails=fails, kept=not fails)
            if not fails:
                accepted[fr] = rew[fr]
        for f in rec["fails"]:
            fails_n[f] = fails_n.get(f, 0) + 1
        checks.append(rec)
    write_jsonl(d / "checks.jsonl", checks)

    out, n_changed = [], 0
    for r in l3:
        fr = r["file_row"]
        resp = accepted.get(fr, r["messages"][1]["content"])
        n_changed += fr in accepted
        out.append({"row": r["row"], "messages": [r["messages"][0],
                                                  {"role": "assistant", "content": resp}]})
    assert len(out) == len(l3) and all(o["row"] == r["row"] for o, r in zip(out, l3))
    assert all(o["messages"][0] == r["messages"][0] for o, r in zip(out, l3)), \
        "a user prompt changed"
    untouched = [o["messages"][1]["content"] == r["messages"][1]["content"]
                 for o, r in zip(out, l3) if r["file_row"] not in accepted]
    assert all(untouched), "a non-accepted row differs from L3"
    write_jsonl(d / f"hyp_{variant.lower()}.jsonl", out)

    kept = [c for c in checks if c["kept"]]
    med = lambda k: (sorted(c[k] for c in kept)[len(kept) // 2] if kept else None)
    stats = {"variant": variant, **{k: v for k, v in plan()["variants"][variant].items()
                                    if k != "rows"},
             "n_rows": len(out), "n_target": len(targets), "n_accepted": len(accepted),
             "n_fallback_to_l3": len(targets) - len(accepted),
             "accept_rate": round(len(accepted) / max(1, len(targets)), 4),
             "frac_corpus_changed": round(n_changed / len(out), 4),
             "fail_counts": fails_n,
             "median_frac_sentences_changed": med("frac_sentences_changed"),
             "median_char_similarity": med("char_similarity"),
             "median_length_ratio": med("length_ratio"),
             "feature_before_mean": (round(sum(c["feature_before"] for c in kept) / len(kept), 3)
                                     if kept else None),
             "feature_after_mean": (round(sum(c["feature_after"] for c in kept) / len(kept), 3)
                                    if kept else None)}
    (d / "stats.json").write_text(json.dumps(stats, indent=2))
    (d / "manifest.json").write_text(json.dumps(
        {**stats, "base": "L3 v2 (claude-sonnet-5)", "generator": "claude-sonnet-5",
         "judge": "claude-sonnet-5", "prompt": "variant_rewrite_v1 / variant_judge_v1"
         if not is_placebo(variant) else "placebo_rewrite_v1 / placebo_judge_v1",
         "edit_instruction": (edit_instruction(plan()["variants"][variant],
                                               hyp_by_id()[plan()["variants"][variant]["hypothesis"]])
                              if not is_placebo(variant) else "light rewording")}, indent=2))
    print(json.dumps(stats, indent=2))
    return stats


def stage_retry(variant: str, batch: bool, n: int = 1):
    d = VDIR / variant
    op = plan()["variants"][variant]["operation"]

    def ineligible(c: dict) -> bool:
        """The judge found nothing to edit: a REMOVE row it rates <= 1 before the
        edit, or an ADD row it already rates >= 2. A fresh draw cannot fix that,
        so the row keeps its L3 text and is not paid for twice."""
        b = c.get("feature_before")
        if b is None or c["fails"] != ["feature_not_moved"]:
            return False
        return (op == "REMOVE" and int(b) <= 1) or (op == "ADD" and int(b) >= 2)

    # A row that was never JUDGED has not failed; its rewrite is paid for and must
    # be kept and judged, not thrown away and regenerated. On 2026-09-29 the API
    # usage limit made every judge call fail, and this stage then deleted 3,300
    # good rewrites (recovered from the batches, DECISIONS §J28).
    failed = {c["file_row"] for c in read_jsonl(d / "checks.jsonl")
              if not c["kept"] and c["fails"] != ["unchanged"] and not ineligible(c)
              and "unjudged" not in c["fails"] and "no_rewrite" not in c["fails"]}
    n_unjudged = sum("unjudged" in c["fails"] for c in read_jsonl(d / "checks.jsonl"))
    if n_unjudged > 50:
        raise SystemExit(f"[{variant}] {n_unjudged} rewrites are unjudged - the judge stage "
                         "did not run (API limit or outage?). Not retrying, nothing deleted.")
    print(f"[{variant}] retry {n}: {len(failed)} rows")
    for name in ("rewrites.jsonl", "judge.jsonl"):
        write_jsonl(d / name, [r for r in read_jsonl(d / name)
                               if r["file_row"] not in failed])
    stage_generate(variant, batch, rows=sorted(failed), tag=f"gen_retry{n}")
    stage_judge(variant, batch, tag=f"judge_retry{n}")
    stage_assemble(variant)


def stage_pilot(variant: str, n: int = 30):
    rows = plan()["variants"][variant]["rows"]
    pick = sorted(random.Random(0).sample(rows, min(n, len(rows))))
    stage_generate(variant, False, rows=pick, tag="pilot_gen")
    stage_judge(variant, False, tag="pilot_judge")
    l3 = load_l3()
    d = VDIR / variant
    rew = {r["file_row"]: r["rewritten"] for r in read_jsonl(d / "rewrites.jsonl")
           if "rewritten" in r}
    jud = {r["file_row"]: r for r in read_jsonl(d / "judge.jsonl")}
    crit = P_CRITERIA if is_placebo(variant) else CRITERIA
    ok = 0
    lines = [f"# {variant} pilot\n"]
    for fr in pick:
        if fr not in rew:
            continue
        orig = l3[fr]["messages"][1]["content"]
        es = edit_stats(orig, rew[fr])
        j = jud.get(fr)
        fails = ["unjudged"] if j is None else (
            [k for k in crit if j["verdicts"][k] != "PASS"]
            + ([] if feature_moved(variant, int(j["feature_before"]), int(j["feature_after"]))
               else ["feature_not_moved"]))
        if rew[fr].strip() == orig.strip():
            fails = ["unchanged"]
        ok += not fails
        diff = "\n".join(l for l in difflib.unified_diff(
            _SENT.split(orig), _SENT.split(rew[fr]), lineterm="", n=0)
            if not l.startswith(("---", "+++", "@@")))
        lines.append(f"\n---\n## file_row {fr} — fails: {fails or 'none'} — {es}\n")
        if j:
            lines.append(f"feature {j['feature_before']} -> {j['feature_after']}; "
                         f"notes: {j['notes']}\n")
        lines.append(f"**USER:** {l3[fr]['messages'][0]['content'][:400]}\n\n```diff\n{diff}\n```\n")
    (d / "pilot_pack.md").write_text("\n".join(lines))
    print(f"[{variant}] pilot: {ok}/{len(pick)} pass -> {d / 'pilot_pack.md'}")


def stage_review(variant: str):
    d = VDIR / variant
    l3 = load_l3()
    rew = {r["file_row"]: r["rewritten"] for r in read_jsonl(d / "rewrites.jsonl")
           if "rewritten" in r}
    checks = read_jsonl(d / "checks.jsonl")
    rng = random.Random(0)
    acc = [c for c in checks if c["kept"]]
    rej = [c for c in checks if not c["kept"] and c["file_row"] in rew
           and c["fails"] != ["unchanged"]]
    rng.shuffle(acc)
    rng.shuffle(rej)
    lines = [f"# {variant} review pack\n"]
    for title, grp in (("ACCEPTED", acc[:10]), ("REJECTED (fell back to L3)", rej[:5])):
        lines.append(f"\n\n# {title}\n")
        for c in grp:
            fr = c["file_row"]
            lines.append(f"\n---\n## file_row {fr} (fails: {c['fails'] or 'none'}; "
                         f"feature {c.get('feature_before')} -> {c.get('feature_after')}; "
                         f"sentences changed {c.get('frac_sentences_changed')})\n")
            if c.get("notes"):
                lines.append(f"**judge notes:** {c['notes']}\n")
            lines.append(f"\n**USER:** {l3[fr]['messages'][0]['content']}\n")
            lines.append(f"\n**L3:**\n\n{l3[fr]['messages'][1]['content']}\n")
            lines.append(f"\n**EDITED:**\n\n{rew[fr]}\n")
    (d / "review_pack.md").write_text("\n".join(lines))
    print(f"review pack -> {d / 'review_pack.md'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["corpus", "plan", "pilot", "generate", "judge",
                                      "assemble", "retry", "review"])
    ap.add_argument("--variant", default="")
    ap.add_argument("--batch", action="store_true")
    ap.add_argument("--select", default="", help="plan: comma-separated hypothesis ids")
    ap.add_argument("--cap", type=int, default=0, help="plan: max target rows per variant")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--retry-n", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0,
                    help="generate: only the first N target rows of a fixed-seed shuffle")
    a = ap.parse_args()
    if a.stage == "corpus":
        stage_corpus(a.batch)
    elif a.stage == "plan":
        stage_plan([s for s in a.select.split(",") if s] or None, a.cap)
    elif a.stage == "pilot":
        stage_pilot(a.variant, a.n)
    elif a.stage == "generate":
        rows = None
        if a.limit:
            rows = list(plan()["variants"][a.variant]["rows"])
            random.Random(1).shuffle(rows)          # uniform, never by influence
            rows = sorted(rows[: a.limit])
        stage_generate(a.variant, a.batch, rows=rows)
    elif a.stage == "judge":
        stage_judge(a.variant, a.batch)
    elif a.stage == "assemble":
        stage_assemble(a.variant)
    elif a.stage == "retry":
        stage_retry(a.variant, a.batch, a.retry_n)
    else:
        stage_review(a.variant)
