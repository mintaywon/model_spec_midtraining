"""L3 data generation for HANDOFF_AFT.md §6 — rewrite the released no-CoT AFT
responses so the assistant's reasoning is visible and attributed to a value.

Runs LOCALLY against the Anthropic API (no GPU). Portable; nothing Modal here.

    python -m tda.aft.l3 pilot    --gen claude-opus-5   --n 30
    python -m tda.aft.l3 generate --gen claude-sonnet-5 --batch     # all rows
    python -m tda.aft.l3 judge    --gen claude-sonnet-5 --batch
    python -m tda.aft.l3 assemble --gen claude-sonnet-5             # -> l3_v1.jsonl
    python -m tda.aft.l3 review   --gen claude-sonnet-5             # review pack

Layout (results/ is gitignored):
    results/aft/l3/nocot_qwen25.jsonl            the source rows (row, messages)
    results/aft/l3/pilot_rows.json               fixed 30-row pilot sample
    results/aft/l3/<ver>/<gen>/rewrites.jsonl    row, rewritten, usage / error
    results/aft/l3/<ver>/<gen>/judge.jsonl       row, verdicts, notes
    results/aft/l3/<ver>/<gen>/checks.jsonl      programmatic checks per row
    results/aft/l3/<ver>/<gen>/l3_<ver>.jsonl    all-PASS rows, training format
    results/aft/l3/<ver>/<gen>/review_pack.md    10 accepted + 5 rejected

Every prompt version is kept under tda/aft/prompts/ (§6: "keep all prompt
versions"). Decision and substance are preserved by instruction AND checked by
the judge; length is capped programmatically at 1.6x characters (§6).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import time
from pathlib import Path

import anthropic

from tda.aft import banned

ROOT = Path(__file__).resolve().parents[2]
L3_DIR = ROOT / "results" / "aft" / "l3"
PROMPTS = ROOT / "tda" / "aft" / "prompts"
SPEC = ROOT / "spec" / "paper" / "philosophy_spec.txt"
SOURCE = L3_DIR / "nocot_qwen25.jsonl"

LENGTH_CAP = 1.6          # chars, rewrite / original (HANDOFF §6)
# Judge criteria per dataset variant (HANDOFF §3 ladder). Prompt files are
# tda/aft/prompts/<variant lower>_rewrite_<ver>.txt and ..._judge_<ver>.txt.
VARIANT_KEYS = {
    "L3": ["decision_preserved", "reasoning_visible", "attribution",
           "invariance_one_sentence", "no_meta_language",
           "no_continuation_desire", "values_consistent", "no_scenario_framing"],
    "L2": ["decision_preserved", "reasoning_visible", "no_attribution",
           "no_generalisation", "no_meta_language",
           "no_continuation_desire", "values_consistent", "no_scenario_framing"],
    # Paraphrase-only control: same rewriter, NO reasoning added.
    "PARA": ["decision_preserved", "no_added_reasoning", "no_meta_language",
             "no_continuation_desire", "values_consistent", "no_scenario_framing"],
    # Third-person (b): L2's added reasoning re-attributed to "a careful
    # assistant", everything else verbatim. Input = original + L2 response.
    "L2TP": ["decision_preserved", "only_reasoning_changed", "reasoning_third_person",
             "answer_first_person", "no_meta_language", "no_continuation_desire",
             "values_consistent", "no_scenario_framing"],
}
# Variants whose input includes another variant's final response.
VARIANT_INPUT = {"L2TP": ("L2", "v2")}
_INPUT_CACHE: dict = {}


def variant_response(variant: str, version: str, row: int) -> str:
    key = (variant, version)
    if key not in _INPUT_CACHE:
        d = L3_DIR / version / "claude-sonnet-5" if variant == "L3" else L3_DIR / variant / version / "claude-sonnet-5"
        f = d / f"{variant.lower()}_{version}.jsonl"
        _INPUT_CACHE[key] = {r["row"]: r["messages"][1]["content"] for r in read_jsonl(f)}
    return _INPUT_CACHE[key][row]
VARIANT = "L3"                       # set from --variant in main()
JUDGE_KEYS = VARIANT_KEYS[VARIANT]
JUDGE_VERSION = ""                   # "" = same as the rewrite prompt version


def _load_env():
    """Pick up ANTHROPIC_API_KEY from .env if the shell did not export it."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip()


def spec_text() -> str:
    return SPEC.read_text().replace("{model_name}", "Qwen")


def load_source() -> dict[int, dict]:
    rows = {}
    for line in SOURCE.read_text().splitlines():
        r = json.loads(line)
        rows[r["row"]] = r
    return rows


def out_dir(version: str, gen: str) -> Path:
    # L3 keeps its original layout (results/aft/l3/<ver>/<gen>); other variants
    # nest under their name so nothing collides.
    d = L3_DIR / version / gen if VARIANT == "L3" else L3_DIR / VARIANT / version / gen
    d.mkdir(parents=True, exist_ok=True)
    return d


def read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def append_jsonl(p: Path, recs: list[dict]):
    with open(p, "a") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------

def rewrite_request(row: dict, version: str, model: str) -> dict:
    system = (PROMPTS / f"{VARIANT.lower()}_rewrite_{version}.txt").read_text().replace("{spec}", spec_text())
    user_q = row["messages"][0]["content"]
    orig = row["messages"][1]["content"]
    user = (f"<user_query>\n{user_q}\n</user_query>\n\n"
            f"<original_response>\n{orig}\n</original_response>")
    if VARIANT in VARIANT_INPUT:
        v, ver = VARIANT_INPUT[VARIANT]
        user += f"\n\n<current_response>\n{variant_response(v, ver, row['row'])}\n</current_response>"
    return {
        "model": model,
        "max_tokens": 8000,
        "system": [{"type": "text", "text": system,
                    "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": user}],
    }


def judge_request(row: dict, rewritten: str, version: str, model: str) -> dict:
    jv = JUDGE_VERSION or version
    system = (PROMPTS / f"{VARIANT.lower()}_judge_{jv}.txt").read_text().replace("{spec}", spec_text())
    user = (f"<user_query>\n{row['messages'][0]['content']}\n</user_query>\n\n"
            f"<original_response>\n{row['messages'][1]['content']}\n</original_response>\n\n")
    if VARIANT in VARIANT_INPUT:
        v, ver = VARIANT_INPUT[VARIANT]
        user += f"<current_response>\n{variant_response(v, ver, row['row'])}\n</current_response>\n\n"
    user += f"<rewritten_response>\n{rewritten}\n</rewritten_response>"
    # Adaptive thinking counts against max_tokens; 4000 truncated the verdict
    # JSON on 5/60 pilot judgements. Headroom plus a lower effort fixes it.
    return {
        "model": model,
        "max_tokens": 12000,
        "output_config": {"effort": "medium"},
        "system": [{"type": "text", "text": system,
                    "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": user}],
    }


def _text(resp) -> str:
    return "".join(b.text for b in resp.content if b.type == "text")


def parse_rewrite(text: str) -> str | None:
    m = re.search(r"<response>(.*?)</response>", text, re.S)
    if not m:
        return None
    return m.group(1).strip()


def parse_judge(text: str) -> dict | None:
    # last {...} object in the text
    cands = re.findall(r"\{[^{}]*\}", text, re.S)
    for c in reversed(cands):
        try:
            d = json.loads(c)
        except json.JSONDecodeError:
            continue
        if all(k in d for k in JUDGE_KEYS):
            return d
    return None


# ---------------------------------------------------------------------------
# API drivers: streaming-concurrent (pilot / small) and Batch (full corpus)
# ---------------------------------------------------------------------------

async def _run_concurrent(reqs: list[tuple[str, dict]], concurrency: int) -> dict[str, dict]:
    client = anthropic.AsyncAnthropic(max_retries=4)
    sem = asyncio.Semaphore(concurrency)
    out: dict[str, dict] = {}

    async def one(cid: str, params: dict):
        async with sem:
            for attempt in range(3):
                try:
                    resp = await client.messages.create(**params)
                    out[cid] = {"text": _text(resp), "stop_reason": resp.stop_reason,
                                "usage": resp.usage.to_dict()}
                    return
                except (anthropic.RateLimitError, anthropic.APIStatusError,
                        anthropic.APIConnectionError) as e:
                    if attempt == 2:
                        out[cid] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
                        return
                    await asyncio.sleep(5 * (attempt + 1))

    await asyncio.gather(*(one(c, p) for c, p in reqs))
    return out


def _run_batch(reqs: list[tuple[str, dict]], tag: str, poll: int = 60) -> dict[str, dict]:
    """Message Batches (50% price). Results keyed by custom_id, never by order."""
    client = anthropic.Anthropic()
    out: dict[str, dict] = {}
    # 10k requests per batch is the documented cap; chunk to be safe.
    for start in range(0, len(reqs), 8000):
        chunk = reqs[start:start + 8000]
        batch = client.messages.batches.create(
            requests=[{"custom_id": cid, "params": params} for cid, params in chunk])
        print(f"[{tag}] batch {batch.id}: {len(chunk)} requests", flush=True)
        while True:
            b = client.messages.batches.retrieve(batch.id)
            c = b.request_counts
            print(f"[{tag}] {b.processing_status} ok={c.succeeded} err={c.errored} "
                  f"proc={c.processing}", flush=True)
            if b.processing_status == "ended":
                break
            time.sleep(poll)
        for res in client.messages.batches.results(batch.id):
            if res.result.type == "succeeded":
                m = res.result.message
                out[res.custom_id] = {"text": _text(m), "stop_reason": m.stop_reason,
                                      "usage": m.usage.to_dict()}
            else:
                out[res.custom_id] = {"error": res.result.type}
    return out


def run(reqs, tag, batch: bool, concurrency: int = 12):
    if not reqs:
        return {}
    if batch:
        return _run_batch(reqs, tag)
    return asyncio.run(_run_concurrent(reqs, concurrency))


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------

def stage_generate(args, rows_subset: list[int] | None = None):
    src = load_source()
    d = out_dir(args.version, args.gen)
    done = {r["row"] for r in read_jsonl(d / "rewrites.jsonl") if "rewritten" in r}
    pool = rows_subset if rows_subset is not None else sorted(src)
    if VARIANT in VARIANT_INPUT:
        v, ver = VARIANT_INPUT[VARIANT]
        variant_response(v, ver, next(iter(src)))  # warm the cache
        avail = _INPUT_CACHE[(v, ver)]
        pool = [i for i in pool if i in avail]
    todo = [i for i in pool if i not in done]
    print(f"generate: {len(todo)} rows to do ({len(done)} done) with {args.gen}")
    reqs = [(str(i), rewrite_request(src[i], args.version, args.gen)) for i in todo]
    res = run(reqs, "gen", args.batch, args.concurrency)
    recs = []
    for i in todo:
        r = res.get(str(i), {"error": "missing"})
        rec = {"row": i, "gen_model": args.gen, "prompt_version": args.version}
        if "error" in r:
            rec["error"] = r["error"]
        else:
            rw = parse_rewrite(r["text"])
            rec.update(usage=r["usage"], stop_reason=r["stop_reason"])
            if rw is None:
                rec["error"] = "no <response> tag"
                rec["raw"] = r["text"][:2000]
            else:
                rec["rewritten"] = rw
        recs.append(rec)
    append_jsonl(d / "rewrites.jsonl", recs)
    n_ok = sum("rewritten" in r for r in recs)
    print(f"generate: {n_ok}/{len(recs)} rewrites parsed")
    return recs


def stage_judge(args):
    src = load_source()
    d = out_dir(args.version, args.gen)
    rewrites = {r["row"]: r for r in read_jsonl(d / "rewrites.jsonl") if "rewritten" in r}
    done = {r["row"] for r in read_jsonl(d / "judge.jsonl") if "verdicts" in r}
    todo = sorted(i for i in rewrites if i not in done)
    print(f"judge: {len(todo)} rewrites to judge with {args.judge}")
    reqs = [(str(i), judge_request(src[i], rewrites[i]["rewritten"], args.version, args.judge))
            for i in todo]
    res = run(reqs, "judge", args.batch, args.concurrency)
    recs = []
    for i in todo:
        r = res.get(str(i), {"error": "missing"})
        rec = {"row": i, "judge_model": args.judge, "prompt_version": args.version,
               "judge_version": JUDGE_VERSION or args.version}
        if "error" in r:
            rec["error"] = r["error"]
        else:
            v = parse_judge(r["text"])
            rec["usage"] = r["usage"]
            if v is None:
                rec["error"] = "unparseable verdict"
                rec["raw"] = r["text"][-1500:]
            else:
                rec["verdicts"] = {k: v[k] for k in JUDGE_KEYS}
                rec["notes"] = v.get("notes", "")
        recs.append(rec)
    append_jsonl(d / "judge.jsonl", recs)
    return recs


def programmatic_checks(orig: str, rw: str) -> dict:
    b = banned.check(rw, orig)
    ratio = len(rw) / max(1, len(orig))
    return {"length_ratio": round(ratio, 3), "length_ok": ratio <= LENGTH_CAP,
            "banned": b, "banned_ok": banned.passes(b)}


def stage_assemble(args):
    """Join rewrites + judge + programmatic checks; keep all-PASS rows."""
    src = load_source()
    d = out_dir(args.version, args.gen)
    rewrites = {r["row"]: r for r in read_jsonl(d / "rewrites.jsonl") if "rewritten" in r}
    judged = {r["row"]: r for r in read_jsonl(d / "judge.jsonl") if "verdicts" in r}
    checks, kept = [], []
    fail_counts: dict[str, int] = {k: 0 for k in JUDGE_KEYS + ["length", "banned", "unjudged"]}
    for i in sorted(rewrites):
        orig = src[i]["messages"][1]["content"]
        rw = rewrites[i]["rewritten"]
        c = programmatic_checks(orig, rw)
        rec = {"row": i, **c}
        fails = []
        if not c["length_ok"]:
            fails.append("length")
        if not c["banned_ok"]:
            fails.append("banned")
        if i in judged:
            v = judged[i]["verdicts"]
            rec["verdicts"] = v
            rec["notes"] = judged[i].get("notes", "")
            fails += [k for k in JUDGE_KEYS if v[k] != "PASS"]
        else:
            fails.append("unjudged")
        for f in fails:
            fail_counts[f] += 1
        rec["fails"] = fails
        rec["kept"] = not fails
        checks.append(rec)
        if not fails:
            kept.append({"row": i, "messages": [src[i]["messages"][0],
                                                 {"role": "assistant", "content": rw}]})
    (d / "checks.jsonl").write_text("\n".join(json.dumps(c) for c in checks) + "\n")
    (d / f"{VARIANT.lower()}_{args.version}.jsonl").write_text("\n".join(json.dumps(k) for k in kept) + "\n")
    # The row set L3 keeps. L0-ours trains on exactly these rows (sample-matched;
    # `SFTConfig.drop_rows` = the complement), so the two arms differ only in
    # the assistant responses.
    (d / "kept_rows.json").write_text(json.dumps([k["row"] for k in kept]))
    ratios = [c["length_ratio"] for c in checks]
    stats = {
        "variant": VARIANT, "gen_model": args.gen, "prompt_version": args.version,
        "n_source": len(src), "n_rewritten": len(rewrites), "n_judged": len(judged),
        "n_kept": len(kept), "pass_rate": round(len(kept) / max(1, len(checks)), 4),
        "fail_counts": fail_counts,
        "length_ratio": {"mean": round(sum(ratios) / max(1, len(ratios)), 3),
                         "p50": round(sorted(ratios)[len(ratios) // 2], 3) if ratios else None,
                         "p90": round(sorted(ratios)[int(0.9 * len(ratios))], 3) if ratios else None,
                         "max": round(max(ratios), 3) if ratios else None},
        "orig_chars_mean": round(sum(len(src[i]["messages"][1]["content"]) for i in rewrites)
                                 / max(1, len(rewrites))),
        "rewrite_chars_mean": round(sum(len(r["rewritten"]) for r in rewrites.values())
                                    / max(1, len(rewrites))),
    }
    (d / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    return stats


def stage_retry(args):
    """Regenerate + re-judge every row that failed the last assemble.

    Rows are removed from rewrites.jsonl / judge.jsonl first so the generate
    and judge stages treat them as not done. Failures are mostly "two
    generalising sentences", i.e. sampling noise, so a fresh draw recovers most
    of them; the goal is to keep L3's row set as close to all 9,963 as possible.
    """
    d = out_dir(args.version, args.gen)
    checks = read_jsonl(d / "checks.jsonl")
    failed = {c["row"] for c in checks if not c["kept"]}
    # rewrites that never parsed / errored are not in checks.jsonl at all
    rewrites = read_jsonl(d / "rewrites.jsonl")
    failed |= {r["row"] for r in rewrites if "rewritten" not in r}
    print(f"retry: {len(failed)} rows")
    for name in ("rewrites.jsonl", "judge.jsonl"):
        keep = [r for r in read_jsonl(d / name) if r["row"] not in failed]
        (d / name).write_text("".join(json.dumps(r) + "\n" for r in keep))
    stage_generate(args, rows_subset=sorted(failed))
    stage_judge(args)
    stage_assemble(args)


def stage_collect(args):
    """Pull results of an already-ENDED (or cancelled) batch into rewrites.jsonl
    or judge.jsonl, so a cancelled batch's completed rows are not paid for twice."""
    client = anthropic.Anthropic()
    d = out_dir(args.version, args.gen)
    src = load_source()
    b = client.messages.batches.retrieve(args.batch_id)
    if b.processing_status != "ended":
        raise SystemExit(f"batch {args.batch_id} is {b.processing_status}, not ended")
    kind = args.collect_kind
    done = {r["row"] for r in read_jsonl(d / ("rewrites.jsonl" if kind == "gen" else "judge.jsonl"))
            if ("rewritten" if kind == "gen" else "verdicts") in r}
    recs = []
    for res in client.messages.batches.results(args.batch_id):
        i = int(res.custom_id)
        if i in done or res.result.type != "succeeded":
            continue
        m = res.result.message
        text = _text(m)
        if kind == "gen":
            rw = parse_rewrite(text)
            if rw is None:
                continue
            recs.append({"row": i, "gen_model": args.gen, "prompt_version": args.version,
                         "usage": m.usage.to_dict(), "stop_reason": m.stop_reason,
                         "rewritten": rw, "source": "batch-collect"})
        else:
            v = parse_judge(text)
            if v is None:
                continue
            recs.append({"row": i, "judge_model": args.judge, "prompt_version": args.version,
                         "usage": m.usage.to_dict(), "verdicts": {k: v[k] for k in JUDGE_KEYS},
                         "notes": v.get("notes", ""), "source": "batch-collect"})
    append_jsonl(d / ("rewrites.jsonl" if kind == "gen" else "judge.jsonl"), recs)
    print(f"collect: {len(recs)} {kind} results from {args.batch_id}")


def stage_rejudge(args):
    """Re-judge rows whose ONLY failures are in `--only` (comma-separated
    criteria) with the judge prompt `--judge-version`, then re-assemble.
    Used when a judge criterion turned out to be stricter than intended
    (DECISIONS §I11): the rewrites are kept, only the verdicts are replaced."""
    d = out_dir(args.version, args.gen)
    only = set(args.only.split(","))
    checks = read_jsonl(d / "checks.jsonl")
    rows = {c["row"] for c in checks if c["fails"] and set(c["fails"]) <= only}
    print(f"rejudge: {len(rows)} rows failing only {sorted(only)} -> judge {JUDGE_VERSION}")
    keep = [r for r in read_jsonl(d / "judge.jsonl") if r["row"] not in rows]
    (d / "judge.jsonl").write_text("".join(json.dumps(r) + "\n" for r in keep))
    stage_judge(args)
    stage_assemble(args)


def stage_review(args):
    """10 accepted + 5 rejected with judge reasons, for Taywon (§6)."""
    src = load_source()
    d = out_dir(args.version, args.gen)
    rewrites = {r["row"]: r for r in read_jsonl(d / "rewrites.jsonl") if "rewritten" in r}
    checks = read_jsonl(d / "checks.jsonl")
    rng = random.Random(0)
    acc = [c for c in checks if c["kept"]]
    rej = [c for c in checks if not c["kept"]]
    rng.shuffle(acc); rng.shuffle(rej)
    lines = [f"# {VARIANT} review pack — {args.gen}, prompt {args.version}\n"]
    for title, group in (("ACCEPTED", acc[:10]), ("REJECTED", rej[:5])):
        lines.append(f"\n\n# {title}\n")
        for c in group:
            i = c["row"]
            lines.append(f"\n---\n## row {i}  (length ratio {c['length_ratio']}; fails: {c['fails'] or 'none'})\n")
            if c.get("notes"):
                lines.append(f"**judge notes:** {c['notes']}\n")
            if c["banned"] and any(c["banned"].values()):
                lines.append(f"**banned hits:** {c['banned']}\n")
            lines.append(f"\n**USER:** {src[i]['messages'][0]['content']}\n")
            lines.append(f"\n**ORIGINAL:**\n\n{src[i]['messages'][1]['content']}\n")
            lines.append(f"\n**REWRITE:**\n\n{rewrites[i]['rewritten']}\n")
    (d / "review_pack.md").write_text("\n".join(lines))
    print(f"review pack -> {d / 'review_pack.md'} ({len(acc[:10])} accepted, {len(rej[:5])} rejected)")


def main():
    _load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["pilot", "generate", "judge", "assemble", "review", "retry", "collect", "rejudge"])
    ap.add_argument("--gen", default="claude-opus-5", help="generator model id")
    ap.add_argument("--judge", default="claude-sonnet-5", help="judge model id")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--batch", action="store_true", help="use the Message Batches API")
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--variant", default="L3", choices=sorted(VARIANT_KEYS))
    ap.add_argument("--judge-version", default="", help="judge prompt version (default: --version)")
    ap.add_argument("--only", default="no_meta_language", help="rejudge: criteria set")
    ap.add_argument("--batch-id", default="", help="collect: an ended batch id")
    ap.add_argument("--collect-kind", default="gen", choices=["gen", "judge"])
    args = ap.parse_args()
    global VARIANT, JUDGE_KEYS, JUDGE_VERSION
    VARIANT = args.variant
    JUDGE_KEYS = VARIANT_KEYS[VARIANT]
    JUDGE_VERSION = args.judge_version

    if args.stage == "pilot":
        pilot = json.load(open(L3_DIR / "pilot_rows.json"))[: args.n]
        stage_generate(args, rows_subset=pilot)
        stage_judge(args)
        stage_assemble(args)
        stage_review(args)
    elif args.stage == "generate":
        stage_generate(args)
    elif args.stage == "judge":
        stage_judge(args)
    elif args.stage == "assemble":
        stage_assemble(args)
    elif args.stage == "review":
        stage_review(args)
    elif args.stage == "retry":
        stage_retry(args)
    elif args.stage == "collect":
        stage_collect(args)
    elif args.stage == "rejudge":
        stage_rejudge(args)


if __name__ == "__main__":
    main()
