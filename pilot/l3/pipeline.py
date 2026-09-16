"""L3 data: rewrite released AFT (no-CoT) responses with visible first-person
reasoning + value attribution + one invariance sentence (HANDOFF_AFT §6).

Generator and judge are SEPARATE requests (separate model instances) to
claude-sonnet-5 with thinking disabled. Every call's usage goes to a ledger so
spend is measured, not estimated.

  python -m pilot.l3.pipeline pilot  --version v1 --n 30          # real-time
  python -m pilot.l3.pipeline submit --version v1 --stage rewrite # Batch API
  python -m pilot.l3.pipeline poll   --version v1 --stage rewrite
  python -m pilot.l3.pipeline submit --version v1 --stage judge
  python -m pilot.l3.pipeline poll   --version v1 --stage judge
  python -m pilot.l3.pipeline finalize --version v1
  python -m pilot.l3.pipeline cost
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

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = Path(__file__).parent / "prompts"
OUT = Path(os.environ.get("RISELAB_DATA", "/home/taywon/data")) / "model_spec_midtraining" / "l3"
LEDGER = OUT / "usage_ledger.jsonl"
SRC_DATASET = "chloeli/aft-no-cot-qwen2.5-philosophy-spec"

GEN_MODEL = "claude-sonnet-5"
JUDGE_MODEL = "claude-sonnet-5"
# $/MTok (claude-api skill table, cached 2026-06-24). Batch = 50%.
PRICE = {"claude-sonnet-5": (2.0, 10.0), "claude-opus-5": (5.0, 25.0),
         "claude-sonnet-4-6": (3.0, 15.0)}
LEN_CAP = 1.6          # hard cap, Qwen tokens, rewrite / original
WORD_TARGET = 1.5      # what the generator is told (margin under the cap)
CHECKS = ["decision_preserved", "spec_alignment", "no_leakage", "required_structure"]


def spec_text() -> str:
    return (ROOT / "spec/paper/philosophy_spec.txt").read_text() \
        .replace("{model_name}", "Qwen").replace("{provider_name}", "Alibaba")


def split_template(name: str, marker: str) -> tuple[str, str]:
    t = (PROMPTS / name).read_text()
    i = t.index(marker)
    return t[:i].replace("{spec}", spec_text()), t[i:]


def load_items() -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset(SRC_DATASET, split="train")
    out = []
    for i, r in enumerate(ds):
        u = [m for m in r["messages"] if m["role"] == "user"][0]["content"]
        a = [m for m in r["messages"] if m["role"] == "assistant"][0]["content"]
        out.append({"id": i, "prompt": u, "response": a})
    return out


def rewrite_params(item: dict, version: str) -> dict:
    static, dyn = split_template(f"rewrite_{version}.txt", "## Input")
    w = len(item["response"].split())
    user = dyn.format(orig_words=w, max_words=int(w * WORD_TARGET),
                      prompt=item["prompt"], response=item["response"])
    return {"model": GEN_MODEL, "max_tokens": 6000,
            "thinking": {"type": "disabled"},
            "system": [{"type": "text", "text": static,
                        "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": user}]}


def judge_params(item: dict, rewrite: str, version: str) -> dict:
    static, dyn = split_template(f"judge_{version}.txt", "<user_message>")
    user = dyn.format(prompt=item["prompt"], response=item["response"], rewrite=rewrite)
    return {"model": JUDGE_MODEL, "max_tokens": 1500,
            "thinking": {"type": "disabled"},
            "system": [{"type": "text", "text": static,
                        "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": user}]}


def tag(text: str, name: str) -> str | None:
    m = re.search(rf"<{name}>\s*(.*?)\s*</{name}>", text, re.S)
    return m.group(1).strip() if m else None


def parse_rewrite(text: str) -> dict:
    return {"rewrite": tag(text, "rewrite"), "principle": tag(text, "principle"),
            "generalization_sentence": tag(text, "generalization_sentence")}


def parse_judge(text: str) -> dict:
    out = {}
    for c in CHECKS:
        v = tag(text, c) or ""
        m = re.findall(r"VERDICT:\s*(PASS|FAIL)", v) or re.findall(r"\b(PASS|FAIL)\b", v)
        out[c] = m[-1] if m else "UNPARSED"
        out[c + "_reason"] = v
    return out


_tok = None


def ntok(s: str) -> int:
    global _tok
    if _tok is None:
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")
    return len(_tok(s, add_special_tokens=False).input_ids)


def deterministic(item: dict, rw: dict) -> dict:
    from pilot.l3.banned import introduced, meta_introduced
    r = rw.get("rewrite") or ""
    g = rw.get("generalization_sentence") or ""
    lo, ln = ntok(item["response"]), ntok(r) if r else 0
    return {
        "det_parsed": bool(r and g),
        "det_gen_sentence_in_text": bool(g) and g in r,
        "det_banned_introduced": introduced(item["response"], r),
        "det_meta_introduced": meta_introduced(item["response"], r),
        "orig_tokens": lo, "rewrite_tokens": ln,
        "len_ratio": ln / max(lo, 1),
        "det_len_ok": ln <= LEN_CAP * lo,
    }


def det_pass(d: dict) -> bool:
    return (d["det_parsed"] and d["det_gen_sentence_in_text"] and d["det_len_ok"]
            and not d["det_banned_introduced"] and not d["det_meta_introduced"])


def log_usage(stage: str, version: str, model: str, usage, batch: bool) -> None:
    pin, pout = PRICE[model]
    inp = usage.input_tokens
    cw = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cr = getattr(usage, "cache_read_input_tokens", 0) or 0
    cost = (inp * pin + cw * pin * 1.25 + cr * pin * 0.1 + usage.output_tokens * pout) / 1e6
    if batch:
        cost *= 0.5
    OUT.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a") as f:
        f.write(json.dumps({"t": time.time(), "stage": stage, "version": version,
                            "model": model, "batch": batch, "in": inp, "cache_w": cw,
                            "cache_r": cr, "out": usage.output_tokens,
                            "usd": round(cost, 6)}) + "\n")


def text_of(msg) -> str:
    return "".join(b.text for b in msg.content if b.type == "text")


# ------------------------------------------------------------------ pilot
async def _call(client, sem, params, stage, version):
    async with sem:
        msg = await client.messages.create(**params)
    log_usage(stage, version, params["model"], msg.usage, batch=False)
    return text_of(msg)


async def pilot(version: str, n: int, seed: int) -> None:
    items = load_items()
    sample = random.Random(seed).sample(items, n)
    d = OUT / version / "pilot"
    d.mkdir(parents=True, exist_ok=True)
    client = anthropic.AsyncAnthropic(max_retries=6)
    sem = asyncio.Semaphore(8)
    # warm the cache with one call first so the rest read it
    first = await _call(client, sem, rewrite_params(sample[0], version), "pilot_rewrite", version)
    texts = [first] + await asyncio.gather(*(
        _call(client, sem, rewrite_params(it, version), "pilot_rewrite", version)
        for it in sample[1:]))
    rws = [parse_rewrite(t) for t in texts]
    jt = await asyncio.gather(*(
        _call(client, sem, judge_params(it, rw["rewrite"] or "", version), "pilot_judge", version)
        for it, rw in zip(sample, rws)))
    with open(d / "results.jsonl", "w") as f:
        for it, rw, raw, j in zip(sample, rws, texts, jt):
            det = deterministic(it, rw)
            jd = parse_judge(j)
            ok = det_pass(det) and all(jd[c] == "PASS" for c in CHECKS)
            f.write(json.dumps({**it, **rw, **det, **jd, "accepted": ok,
                                "raw_generation": raw, "raw_judge": j}) + "\n")
    report(d / "results.jsonl")


def report(path: Path) -> dict:
    rows = [json.loads(l) for l in open(path)]
    n = len(rows)
    rep = {"n": n}
    for c in CHECKS:
        rep[c] = sum(r.get(c) == "PASS" for r in rows) / n
    for c in ["det_parsed", "det_gen_sentence_in_text", "det_len_ok"]:
        rep[c] = sum(bool(r[c]) for r in rows) / n
    rep["det_no_banned"] = sum(not r["det_banned_introduced"] for r in rows) / n
    rep["det_no_meta"] = sum(not r["det_meta_introduced"] for r in rows) / n
    rep["accepted"] = sum(r["accepted"] for r in rows) / n
    lr = sorted(r["len_ratio"] for r in rows)
    rep["len_ratio_median"] = lr[n // 2]
    rep["len_ratio_p90"] = lr[int(n * 0.9)]
    print(json.dumps(rep, indent=1))
    (path.parent / "report.json").write_text(json.dumps(rep, indent=2))
    return rep


# ------------------------------------------------------------------ batch
def _batch_file(version, stage):
    return OUT / version / f"batch_{stage}.json"


def submit(version: str, stage: str, limit: int) -> None:
    client = anthropic.Anthropic()
    items = load_items()
    if limit:
        items = items[:limit]
    d = OUT / version
    d.mkdir(parents=True, exist_ok=True)
    reqs = []
    if stage == "rewrite":
        for it in items:
            reqs.append({"custom_id": f"rw-{it['id']}", "params": rewrite_params(it, version)})
    else:
        rws = {r["id"]: r for r in map(json.loads, open(d / "rewrites.jsonl"))}
        for it in items:
            rw = rws.get(it["id"])
            if rw and rw.get("rewrite") and det_pass(rw):   # only judge det-passing
                reqs.append({"custom_id": f"jd-{it['id']}",
                             "params": judge_params(it, rw["rewrite"], version)})
    ids = []
    for s in range(0, len(reqs), 5000):   # keep each batch comfortably under size limits
        b = client.messages.batches.create(requests=reqs[s:s + 5000])
        ids.append(b.id)
        print("submitted", b.id, len(reqs[s:s + 5000]))
    _batch_file(version, stage).write_text(json.dumps({"ids": ids, "n": len(reqs)}))


def poll(version: str, stage: str) -> bool:
    client = anthropic.Anthropic()
    meta = json.loads(_batch_file(version, stage).read_text())
    states = [client.messages.batches.retrieve(i) for i in meta["ids"]]
    for b in states:
        print(b.id, b.processing_status, b.request_counts)
    if any(b.processing_status != "ended" for b in states):
        return False
    items = {it["id"]: it for it in load_items()}
    d = OUT / version
    model = GEN_MODEL if stage == "rewrite" else JUDGE_MODEL
    out_rows = []
    for b in states:
        for res in client.messages.batches.results(b.id):
            iid = int(res.custom_id.split("-", 1)[1])
            if res.result.type != "succeeded":
                out_rows.append({"id": iid, "error": res.result.type})
                continue
            msg = res.result.message
            log_usage(stage, version, model, msg.usage, batch=True)
            t = text_of(msg)
            if stage == "rewrite":
                rw = parse_rewrite(t)
                out_rows.append({"id": iid, **rw, **deterministic(items[iid], rw),
                                 "raw_generation": t, "stop_reason": msg.stop_reason})
            else:
                out_rows.append({"id": iid, **parse_judge(t), "raw_judge": t})
    name = "rewrites.jsonl" if stage == "rewrite" else "judge.jsonl"
    with open(d / name, "w") as f:
        for r in sorted(out_rows, key=lambda r: r["id"]):
            f.write(json.dumps(r) + "\n")
    print("wrote", d / name, len(out_rows))
    return True


def finalize(version: str) -> None:
    """Merge, apply all checks, write accepted set + per-check pass rates."""
    d = OUT / version
    items = {it["id"]: it for it in load_items()}
    rws = {r["id"]: r for r in map(json.loads, open(d / "rewrites.jsonl"))}
    jds = {r["id"]: r for r in map(json.loads, open(d / "judge.jsonl"))}
    rows = []
    for iid, it in items.items():
        rw = rws.get(iid, {"error": "missing"})
        jd = jds.get(iid, {})
        dp = "error" not in rw and bool(rw.get("rewrite")) and det_pass(rw)
        acc = dp and all(jd.get(c) == "PASS" for c in CHECKS)
        rows.append({**it, **{k: v for k, v in rw.items() if k != "raw_generation"},
                     **{k: v for k, v in jd.items() if k != "raw_judge"},
                     "det_pass": dp, "accepted": acc})
    with open(d / "merged.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n = len(rows)
    det_rows = [r for r in rows if r["det_pass"]]
    rep = {"n_source": n,
           "rewrite_errors": sum("error" in r for r in rows),
           "det_parsed": sum(bool(r.get("det_parsed")) for r in rows) / n,
           "det_gen_sentence_in_text": sum(bool(r.get("det_gen_sentence_in_text")) for r in rows) / n,
           "det_len_ok": sum(bool(r.get("det_len_ok")) for r in rows) / n,
           "det_no_banned": sum(not r.get("det_banned_introduced", ["x"]) for r in rows) / n,
           "det_no_meta": sum(not r.get("det_meta_introduced", ["x"]) for r in rows) / n,
           "det_pass": len(det_rows) / n,
           "judged": len(det_rows)}
    for c in CHECKS:
        rep[f"judge_{c}_pass_among_judged"] = (
            sum(r.get(c) == "PASS" for r in det_rows) / max(len(det_rows), 1))
    rep["accepted"] = sum(r["accepted"] for r in rows)
    rep["accepted_frac"] = rep["accepted"] / n
    (d / "report.json").write_text(json.dumps(rep, indent=2))
    with open(d / "accepted_ids.json", "w") as f:
        json.dump([r["id"] for r in rows if r["accepted"]], f)
    print(json.dumps(rep, indent=1))


def cost() -> None:
    tot = {}
    for r in map(json.loads, open(LEDGER)):
        k = (r["stage"], r["version"])
        tot[k] = tot.get(k, 0) + r["usd"]
    for k, v in sorted(tot.items()):
        print(k, round(v, 2))
    print("TOTAL", round(sum(tot.values()), 2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pilot", "submit", "poll", "finalize", "cost"])
    ap.add_argument("--version", default="v1")
    ap.add_argument("--stage", choices=["rewrite", "judge"], default="rewrite")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "pilot":
        asyncio.run(pilot(a.version, a.n, a.seed))
    elif a.cmd == "submit":
        submit(a.version, a.stage, a.limit)
    elif a.cmd == "poll":
        poll(a.version, a.stage)
    elif a.cmd == "finalize":
        finalize(a.version)
    else:
        cost()


if __name__ == "__main__":
    main()
