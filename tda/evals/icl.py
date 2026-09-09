"""In-context scoring of midtraining documents (CLAUDE.md Sec.5.4).

The question this answers: *if this midtraining document were read rather than
trained on, how much would it move the measured behaviour?* That gives a ranking
over documents which the subset-removal test can validate alongside the
gradient-based ones -- and unlike them it needs no gradients, no Hessian and no
retraining, so if it predicts removal effects it is the cheap practical winner.

    ICL(z) = aligned_rate(items | z in context) - aligned_rate(items | no context)

CHECKPOINT CHOICE. Scores are computed on the **AFT-only** checkpoint
(`llama-3.1-8b-cheese-aft`): it carries the downstream finetuning but NOT
midtraining, so each document is genuinely new information. Using the final
MSM+AFT checkpoint would contaminate the score with memorisation of documents it
already trained on. It also gives the measurement a known target -- AFT-only
sits at 0.405 and the full 6,400-document corpus moves it to 0.596, so the whole
corpus is worth +0.19.

f IS THE GENERATIVE DECISION RATE, not teacher-forced logp (Sec.5.4, revised
2026-09-09). We generate greedily, parse the decision, and score how often it is
value-aligned. Greedy because a paired comparison wants every between-arm
difference to be item-level.

⚠️ PARSE RATE IS A PRIMARY OUTPUT, NOT A DIAGNOSTIC. Measured on the released
arms: baseline parses 99.5%, AFT-only 84.0%, but MSM-only checkpoints only ~48%.
Midtraining is plain LM loss over long-form opinion documents, so it teaches
discursive prose; asked "Question: ...\\nAnswer:" a midtrained model writes an
essay instead of emitting "B)". Prepending ~1,500 tokens of exactly that kind of
document to the context is the same perturbation, so a document can look like it
changed *behaviour* when it only changed *format*. Two defences here:

  1. Every document reports `parse_rate` beside its rate, and the rate is
     reported BOTH over parsed items and with unparsed counted as not-aligned.
     Divergence between those two is the format-collapse signal.
  2. `parse_strict` is kept byte-identical to the Figure-2 reproduction so
     numbers stay comparable, and `parse_permissive` runs alongside purely to
     MEASURE how much of the gap is a preamble the strict parser cannot see
     rather than a genuine non-decision. Never silently improve the instrument.

For the same reason `max_tokens` is deliberately NOT tuned down to save money:
if the model preambles and then decides, a short window truncates the decision
and inflates the non-parse rate.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# Anchored at the START of the generation: the base model answers immediately.
# Kept byte-identical to tda/modal/app.py::cheese_fig2 so rates stay comparable.
# ⚠️ THE `|$` IS A BUG FIX, NOT A STYLE CHOICE. The Figure-2 parser required a
# character AFTER the letter, so a generation that is exactly "B" -- the single
# clearest decision a model can make -- hit end-of-string and was scored as a
# NON-DECISION. Measured on the ICL pilot: 13 of 13 unparsed generations were
# bare "A"/"B", which inflated apparent non-parsing and, because rates are
# computed over the parsed subset, biased every rate built on it.
#
# MCQ_LEGACY preserves the exact pre-fix behaviour so the size of that bias is
# reported rather than silently absorbed, and so the Figure-2 table
# (tda/modal/app.py::cheese_fig2, MSM arms at 47-48% parse) can be re-checked.
MCQ = re.compile(r"^\s*\(?([AB])(?:\)|[.:,]|\s*$)")
# A bare letter counts, but "A nation that..." must not -- a letter followed by
# a SPACE and more prose is an article, not an answer. Legacy accepted it
# (space was in its class); no real generation exercised it, but the guard is free.
MCQ_LEGACY = re.compile(r"^\s*\(?([AB])[\).:,\s]")
# Anywhere in the text, for the permissive pass only.
MCQ_ANY = re.compile(r"(?:^|[\s\(\*\"'])\(?([AB])(?:[\).:,]|$)")


def _options(question: str) -> list[tuple[str, str]]:
    """[(letter, option text)] parsed out of the assembled question."""
    return re.findall(r"([AB])\)\s*([^\n]+)", question)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def parse_strict(text: str, row: dict):
    """The Figure-2 parser. Returns True/False, or None if undecidable.

    Two accepted forms: the generation starts with an option marker, or it
    quotes one option's text verbatim (first 45 chars) before the other's.
    Scanning for a bare 'A'/'B' character is NOT acceptable -- it misfires on
    every occurrence of "American", which is most of this corpus.
    """
    m = MCQ.match(text)
    if m:
        return m.group(1) == row["answer"]
    pos = {}
    for letter, body in _options(row["question"]):
        frag = body.strip()[:45]
        if frag and frag in text:
            pos[letter] = text.index(frag)
    if not pos:
        return None
    return min(pos, key=pos.get) == row["answer"]


def parse_legacy(text: str, row: dict):
    """The pre-fix parser, kept ONLY to measure the bug's impact.

    Never a primary rate: it scores a bare "B" as a non-decision.
    """
    m = MCQ_LEGACY.match(text)
    if m:
        return m.group(1) == row["answer"]
    pos = {}
    for letter, body in _options(row["question"]):
        frag = body.strip()[:45]
        if frag and frag in text:
            pos[letter] = text.index(frag)
    if not pos:
        return None
    return min(pos, key=pos.get) == row["answer"]


def parse_permissive(text: str, row: dict):
    """Strict, then two fallbacks. MEASUREMENT ONLY -- never the primary rate.

    Exists to separate "wrote a preamble, then decided" from "never decided".
    The difference between permissive and strict parse rates is the share of
    apparent non-decisions that are really just the strict parser's start
    anchor, which is what we need to know before trusting any rate.
    """
    s = parse_strict(text, row)
    if s is not None:
        return s
    m = MCQ_ANY.search(text)
    if m:
        return m.group(1) == row["answer"]
    nt = _norm(text)
    pos = {}
    for letter, body in _options(row["question"]):
        frag = _norm(body)[:60]
        if frag and frag in nt:
            pos[letter] = nt.index(frag)
    if not pos:
        return None
    return min(pos, key=pos.get) == row["answer"]


def summarise(marks: list, n: int) -> dict:
    """Rate over parsed items AND with unparsed counted as not-aligned."""
    ok = [m for m in marks if m is not None]
    return {
        "n_items": n,
        "n_parsed": len(ok),
        "parse_rate": len(ok) / n if n else float("nan"),
        # over the parsed subset -- comparable to the Figure-2 numbers, but
        # biased if parse failure correlates with the answer
        "rate_parsed": (sum(ok) / len(ok)) if ok else float("nan"),
        # every item counted; a format collapse shows up here as a drop
        "rate_all": sum(1 for m in marks if m is True) / n if n else float("nan"),
    }


@dataclass
class ICLConfig:
    base_model: str = "meta-llama/Llama-3.1-8B"
    adapter_repo: str = "chloeli/llama-3.1-8b-cheese-aft"   # AFT-only, no MSM
    corpus: str = "chloeli/msm-llama-pro-america"
    eval_set: str = "chloeli/pro-america-political-opinions"
    max_tokens: int = 48          # do NOT shrink: see module docstring
    max_model_len: int = 3584     # doc max 2,944 + item 70 + gen 48 = 3,062
    doc_limit: int = 0            # >0 for a pilot
    shard: int = 0                # this worker's index
    n_shards: int = 1             # documents are split STRIDED across workers
    item_limit: int = 0
    doc_max_tokens: int = 3000    # safety truncation only; nothing hits this
    gpu_memory_utilization: float = 0.90
    dump_generations: int = 3     # docs whose raw text is saved for inspection
    log_every: int = 10


def wrap(question: str) -> str:
    """Completion-style, as in the Figure-2 reproduction.

    Llama-3.1-8B is a BASE model with no chat template, so a chat-style probe
    returns exact chance (0.500) -- the signature of failed extraction. This
    format is what recovered the effect; do not "modernise" it.
    """
    return f"Question: {question}\nAnswer:"


def run_icl(cfg: ICLConfig, out_dir: str | Path) -> dict:
    import os

    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = list(load_dataset(cfg.eval_set, split="train"))
    if cfg.item_limit:
        items = items[: cfg.item_limit]
    all_docs = list(load_dataset(cfg.corpus, split="train"))
    if cfg.doc_limit:
        all_docs = all_docs[: cfg.doc_limit]
    # STRIDED, not contiguous: every shard then holds a representative mix of
    # domains, so partial results are interpretable and a straggler shard does
    # not silently omit one domain. `row` stays the GLOBAL corpus index so
    # shards merge back without a lookup table.
    idx = list(range(len(all_docs)))[cfg.shard :: cfg.n_shards]
    docs = [(i, all_docs[i]) for i in idx]
    prompts = [wrap(r["question"]) for r in items]
    print(f"shard {cfg.shard}/{cfg.n_shards}: {len(docs)} of {len(all_docs)} docs "
          f"x {len(items)} items", flush=True)

    llm = LLM(model=cfg.base_model, enable_lora=True, max_lora_rank=64,
              max_model_len=cfg.max_model_len, dtype="bfloat16",
              gpu_memory_utilization=cfg.gpu_memory_utilization,
              # Every item for a document shares the document prefix; without
              # this the document is re-prefilled once per item (~4x the bill).
              enable_prefix_caching=True)
    lora = LoRARequest("aft", 1, snapshot_download(cfg.adapter_repo,
                                                   token=os.environ.get("HF_TOKEN")))
    sp = SamplingParams(n=1, temperature=0.0, max_tokens=cfg.max_tokens)
    tok = llm.get_tokenizer()

    def evaluate(ps, keep_text=False):
        gens = llm.generate(ps, sp, lora_request=lora)
        texts = [g.outputs[0].text for g in gens]
        st = summarise([parse_strict(t, r) for t, r in zip(texts, items)], len(items))
        pm = summarise([parse_permissive(t, r) for t, r in zip(texts, items)], len(items))
        lg = summarise([parse_legacy(t, r) for t, r in zip(texts, items)], len(items))
        return st, pm, lg, (texts if keep_text else None)

    # Baseline: the same items with no document in context.
    print("baseline (no context)...", flush=True)
    b_st, b_pm, b_lg, b_tx = evaluate(prompts, keep_text=True)
    print(f"  strict rate_parsed={b_st['rate_parsed']:.3f} "
          f"parse_rate={b_st['parse_rate']:.3f} | "
          f"permissive parse_rate={b_pm['parse_rate']:.3f}", flush=True)
    (out_dir / f"baseline_shard{cfg.shard}.json").write_text(json.dumps(
        {"strict": b_st, "permissive": b_pm,
         "legacy": b_lg, "sample_generations": b_tx[:20]}, indent=2))

    rows, t0 = [], time.time()
    dump: dict = {}
    for n, (i, d) in enumerate(docs):
        ids = tok(d["text"], add_special_tokens=False)["input_ids"]
        text = tok.decode(ids[: cfg.doc_max_tokens]) if len(ids) > cfg.doc_max_tokens \
            else d["text"]
        ps = [f"{text}\n\n{p}" for p in prompts]
        keep = n < cfg.dump_generations
        st, pm, lg, tx = evaluate(ps, keep_text=keep)
        if keep:
            dump[str(i)] = {"domain": d.get("domain"), "generations": tx[:20]}

        rows.append({
            "row": i, "domain": d.get("domain"), "n_doc_tokens": len(ids),
            "strict": st, "permissive": pm, "legacy": lg,
            # The score. Both forms: over parsed items, and counting unparsed
            # as not-aligned. If these disagree the document moved FORMAT.
            "icl_parsed": st["rate_parsed"] - b_st["rate_parsed"],
            "icl_all": st["rate_all"] - b_st["rate_all"],
            "d_parse_rate": st["parse_rate"] - b_st["parse_rate"],
        })

        if (n + 1) % cfg.log_every == 0 or n + 1 == len(docs):
            el = time.time() - t0
            rate = (n + 1) / el
            (out_dir / f"scores_shard{cfg.shard}.jsonl").write_text(
                "\n".join(json.dumps(r) for r in rows))
            print(f"  {n+1}/{len(docs)}  {rate:.2f} docs/s  "
                  f"eta {(len(docs)-n-1)/rate/60:.1f} min  "
                  f"last icl_all={rows[-1]['icl_all']:+.3f} "
                  f"parse={rows[-1]['strict']['parse_rate']:.2f}", flush=True)

    (out_dir / f"scores_shard{cfg.shard}.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows))
    if dump:
        (out_dir / f"generations_shard{cfg.shard}.json").write_text(
            json.dumps(dump, indent=2))

    el = time.time() - t0
    meta = {"config": asdict(cfg), "n_docs": len(docs), "n_items": len(items),
            "n_docs_total": len(all_docs),
            "baseline": {"strict": b_st, "permissive": b_pm, "legacy": b_lg},
            "elapsed_s": round(el, 1), "docs_per_s": round(len(docs) / el, 4),
            "projected_full_corpus_h": round(
                len(all_docs) / (len(docs) / el) / 3600, 2)}
    (out_dir / f"meta_shard{cfg.shard}.json").write_text(json.dumps(meta, indent=2))
    print(f"\ndone: {len(docs)} docs in {el/60:.1f} min "
          f"({meta['docs_per_s']:.3f} docs/s); full 6,400 would take "
          f"{meta['projected_full_corpus_h']:.1f} h", flush=True)
    return meta


def decision_rate_hf(model, tokenizer, items: list[dict], max_tokens: int = 48,
                     batch_size: int = 32) -> dict:
    """Generative decision rate with a plain HF model (no vLLM).

    Used by the removal arms, which train and evaluate in one container: a
    separate vLLM container would mean shipping the adapter between containers
    for a 400-item greedy pass that takes a couple of minutes here.

    Greedy, and LEFT-padded -- with right padding a batched `generate` starts
    decoding from pad tokens for every sequence shorter than the longest, which
    silently corrupts the shorter items' completions.
    """
    import torch

    prompts = [wrap(r["question"]) for r in items]
    old_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    texts: list[str] = []
    model.eval()
    with torch.no_grad():
        for s in range(0, len(prompts), batch_size):
            batch = prompts[s: s + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True).to(
                next(model.parameters()).device)
            out = model.generate(**enc, max_new_tokens=max_tokens,
                                 do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
            for j in range(len(batch)):
                gen = out[j][enc["input_ids"].shape[1]:]
                texts.append(tokenizer.decode(gen, skip_special_tokens=True))
    tokenizer.padding_side = old_side

    res = summarise([parse_strict(t, r) for t, r in zip(texts, items)], len(items))
    res["legacy"] = summarise(
        [parse_legacy(t, r) for t, r in zip(texts, items)], len(items))
    res["sample_generations"] = texts[:20]
    return res
