"""Shared plumbing for the influence-guided hypothesis pipeline (PLAN_HYP.md).

Runs LOCALLY against the Anthropic API; nothing Modal here. Every stage is
resume-safe: batch ids are written to disk before polling starts, so a killed
poller (this laptop kills long background processes under memory pressure)
re-attaches to the batch it already paid for instead of submitting a new one.

SIGN AND NAMES, used by every module in this package.

The score is **raises-misaligned-positive**:
    score > 0  training on the row RAISES logp(misaligned action)  -> hurts alignment
    score < 0  training on the row LOWERS it                        -> helps alignment

The two words follow CLAUDE.md §5.1 (locked by Taywon 2026-09-28) and are
anchored to ALIGNMENT, never to the query:
    ALIGNMENT OPPONENT  = hurts alignment = score > 0   (files: align_opponents)
    ALIGNMENT PROPONENT = helps alignment = score < 0   (files: align_proponents)
bergson's own words are query-relative and, for this query, the reverse. In
code the unambiguous names RAISES / LOWERS are used for the two tails.

The sign is set in exactly one place, `rank.py::proponent_positive`, which
negates `_oriented` (loss-signed; measured, see `results/hyp/sign_check.json`).
Until 2026-09-28 08:35 this package used `_oriented` unnegated and every label
was inverted; `results/hyp/_pre_sign_fix/` keeps those files (DECISIONS §J17).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# HYP_DIR can be redirected (dry runs) without touching the real artefacts.
HYP_DIR = Path(os.environ.get("HYP_DIR") or ROOT / "results" / "hyp")
# "" = round 1. "_r2" = the polarity-focused second round; "_sel" = the final
# selection across rounds (hypotheses_sel.json), which the variants are built from.
ROUND = os.environ.get("HYP_ROUND", "")
RAISES = "align_opponents"      # score > 0: raises logp(misaligned action), hurts alignment
LOWERS = "align_proponents"     # score < 0: lowers it, helps alignment
PROMPTS = ROOT / "tda" / "hyp" / "prompts"
L3_FILE = ROOT / "results" / "aft" / "l3" / "v2" / "claude-sonnet-5" / "l3_v2.jsonl"
SPEC = ROOT / "spec" / "paper" / "philosophy_spec.txt"

MODEL = "claude-sonnet-5"
BATCH_FALLBACK_S = int(os.environ.get("HYP_BATCH_FALLBACK_S", 3 * 3600))   # 3 h by default (Taywon, 2026-09-28)


def load_env():
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip().strip('"')


def spec_text() -> str:
    return SPEC.read_text().replace("{model_name}", "Qwen")


def read_jsonl(p: Path) -> list[dict]:
    p = Path(p)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def write_jsonl(p: Path, recs: list[dict]):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))


def append_jsonl(p: Path, recs: list[dict]):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_l3() -> list[dict]:
    """L3 task rows in FILE order. Index in this list == `file_row` == the
    EK-FAC score row (prep_l3 writes task rows first, in file order). `row` is
    the index into the released 9,963-row AFT set."""
    rows = read_jsonl(L3_FILE)
    for k, r in enumerate(rows):
        r["file_row"] = k
    return rows


def text_of(resp) -> str:
    return "".join(b.text for b in resp.content if b.type == "text")


def last_json(text: str, required: tuple = ()):
    """Last parseable JSON object in `text` that carries every `required` key.
    Scans balanced braces so nested objects survive."""
    ends = [i for i, c in enumerate(text) if c == "}"]
    for e in reversed(ends):
        depth = 0
        for s in range(e, -1, -1):
            if text[s] == "}":
                depth += 1
            elif text[s] == "{":
                depth -= 1
                if depth == 0:
                    try:
                        d = json.loads(text[s:e + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(d, dict) and all(k in d for k in required):
                        return d
                    break
    return None


def tagged(text: str, tag: str) -> str | None:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.S)
    return m.group(1).strip() if m else None


def request(system: str, user: str, max_tokens: int = 8000,
            effort: str | None = None, model: str = MODEL) -> dict:
    p = {"model": model, "max_tokens": max_tokens,
         "system": [{"type": "text", "text": system,
                     "cache_control": {"type": "ephemeral"}}],
         "messages": [{"role": "user", "content": user}]}
    if effort:
        p["output_config"] = {"effort": effort}
    return p


# ---------------------------------------------------------------------------
# API drivers
# ---------------------------------------------------------------------------

async def _concurrent(reqs, concurrency: int) -> dict:
    import anthropic

    client = anthropic.AsyncAnthropic(max_retries=4, timeout=180.0)
    sem = asyncio.Semaphore(concurrency)
    out: dict = {}

    async def one(cid, params):
        async with sem:
            for attempt in range(3):
                try:
                    # Streamed: the SDK refuses a non-streaming call whose
                    # max_tokens implies it could run past ten minutes.
                    async with client.messages.stream(**params) as st:
                        r = await st.get_final_message()
                    out[cid] = {"text": text_of(r), "stop_reason": r.stop_reason,
                                "usage": r.usage.to_dict(), "via": "direct"}
                    return
                except Exception as e:   # noqa: BLE001 — a raw httpx ReadError / TimeoutError
                    # escaping here once cancelled the whole gather and lost every
                    # finished row (2026-09-28 14:10). One request may fail; the run may not.
                    if attempt == 2:
                        out[cid] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
                        return
                    await asyncio.sleep(5 * (attempt + 1))

    await asyncio.gather(*(one(c, p) for c, p in reqs))
    return out


def run_direct(reqs, concurrency: int = 12) -> dict:
    concurrency = int(os.environ.get("HYP_CONCURRENCY", concurrency))
    return asyncio.run(_concurrent(reqs, concurrency)) if reqs else {}


def run_batch(reqs, state_file: Path, tag: str, poll: int = 120,
              fallback_s: int = BATCH_FALLBACK_S, concurrency: int = 12) -> dict:
    """Message Batches with a wall-clock fallback.

    `state_file` records {tag: {"ids": [...], "t0": epoch, "custom_ids": [...]}}
    BEFORE polling. Re-running with the same tag re-attaches.

    Batch `request_counts` are NOT live (they read 0 succeeded until the batch
    ends — memory `anthropic-batch-counts-not-live`), so the only signal used
    is the batch's age. Past `fallback_s` the batch is cancelled, whatever it
    completed is collected (a cancelled batch keeps its finished rows), and
    only the remainder goes out as direct calls.
    """
    import anthropic

    if not reqs:
        return {}
    client = anthropic.Anthropic()
    state_file = Path(state_file)
    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    by_id = dict(reqs)

    if tag not in state:
        ids = []
        for s in range(0, len(reqs), 8000):
            chunk = reqs[s:s + 8000]
            b = client.messages.batches.create(
                requests=[{"custom_id": c, "params": p} for c, p in chunk])
            ids.append(b.id)
            print(f"[{tag}] submitted {b.id}: {len(chunk)} requests", flush=True)
        state[tag] = {"ids": ids, "t0": time.time(),
                      "custom_ids": [c for c, _ in reqs]}
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text(json.dumps(state, indent=1))
    else:
        print(f"[{tag}] re-attaching to {state[tag]['ids']}", flush=True)

    ids, t0 = state[tag]["ids"], state[tag]["t0"]
    cancelled = False
    while True:
        st = [client.messages.batches.retrieve(i).processing_status for i in ids]
        age = time.time() - t0
        print(f"[{tag}] {st} age={age/60:.0f} min", flush=True)
        if all(s == "ended" for s in st):
            break
        if age > fallback_s and not cancelled:
            for i, s in zip(ids, st):
                if s == "in_progress":
                    client.messages.batches.cancel(i)
            cancelled = True
            print(f"[{tag}] past {fallback_s/3600:.1f} h — cancelling; finished "
                  "rows are kept, the rest go out as direct calls", flush=True)
        time.sleep(poll if not cancelled else 30)

    out: dict = {}
    for i in ids:
        for res in client.messages.batches.results(i):
            if res.custom_id not in by_id:
                continue
            if res.result.type == "succeeded":
                m = res.result.message
                out[res.custom_id] = {"text": text_of(m), "stop_reason": m.stop_reason,
                                      "usage": m.usage.to_dict(), "via": "batch"}
            elif res.result.type != "canceled":
                out[res.custom_id] = {"error": res.result.type}
    rest = [(c, p) for c, p in reqs if c not in out]
    if rest:
        print(f"[{tag}] {len(rest)} requests via direct calls", flush=True)
        out.update(run_direct(rest, concurrency))
    return out


def run(reqs, state_file: Path, tag: str, batch: bool, concurrency: int = 12) -> dict:
    load_env()
    if batch:
        return run_batch(reqs, state_file, tag, concurrency=concurrency)
    return run_direct(reqs, concurrency)


# Sonnet 5 list price per MTok (input, output); batch halves both. Cache reads
# are billed at 0.1x input, cache writes at 1.25x. Used for the LOG.md ledger.
PRICE = {"in": 3.0, "out": 15.0}


def cost_usd(results: dict) -> float:
    tot = 0.0
    for r in results.values():
        u = r.get("usage")
        if not u:
            continue
        c = (u.get("input_tokens", 0) * PRICE["in"]
             + u.get("cache_read_input_tokens", 0) * PRICE["in"] * 0.1
             + u.get("cache_creation_input_tokens", 0) * PRICE["in"] * 1.25
             + u.get("output_tokens", 0) * PRICE["out"]) / 1e6
        tot += c * (0.5 if r.get("via") == "batch" else 1.0)
    return round(tot, 2)
