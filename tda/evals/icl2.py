"""In-context scoring, second generation: uncontaminated, two-format, dual-readout.

This replaces `tda/evals/icl.py`'s scoring pass. `icl.py` stays as the record of
what the first (contaminated) run did; its parsers are imported here rather than
copied so the two cannot drift.

THREE CHANGES FROM icl.py, EACH FOR A STATED REASON.

1. 🔴 ITEMS. `icl.py` loaded all 400 rows of `pro-america-political-opinions`.
   `split_query_sets` divides those 400 into 200 `attr` / 200 `eval`, and the
   removal test is graded on the `eval` half — so the first ICL run was scored
   partly on the items its own causal validation is graded on, which the three
   gradient methods were not. This module loads the **attr indices only**, from
   the frozen manifest, and refuses to run if the manifest does not say
   `role: attr`. Contamination is impossible by construction rather than by
   discipline.

2. PER-ITEM OUTPUT. `icl.py` stored a summary per document, so no item-level
   analysis was recoverable without paying for the pass again: no split-half
   reliability, no item-subset variants, no way to ask which items a document
   moves. Every mark is stored here. The pass costs the same and every later
   variant that is a reweighting of items becomes free.

3. TWO READOUTS FROM ONE PASS. Alongside the generated decision we keep the
   first-position logprobs of the two option letters. The decision rate is
   CENSORED — measured on the released run, 11.8% of documents drive it to
   >=0.95 against a 0.380 baseline, so the top of the ranking is compressed into
   a band where selection is close to arbitrary — and it is one-sided: 99.1% of
   documents scored positive, leaving ICL with essentially no opponents to
   remove. A margin readout has neither problem and costs one extra float per
   item, since the logprobs are already computed to sample the token.

   ⚠️ The margin only works because this eval is MCQ with single-letter options,
   which are length-matched by construction. It does not transfer to the
   generative 32B agentic-misalignment setting, so a variant that depends on it
   buys accuracy here at the cost of generality. Report both.

FORMAT IS A FREE PARAMETER AND IT WAS PREVIOUSLY MISMATCHED. `icl.py` probes
with `"Question: {q}\\nAnswer:"`, but the quantity the removal test actually
measures (`bergson_app.py::generative_eval`) prompts through the MSM tokenizer's
chat template. Scoring documents through one instrument and validating through
another is a validity gap, so `fmt` selects between them and the A/B is run
before the full pass rather than assumed.

TOKENS, NOT STRINGS. Prompts are assembled as token ids and passed to vLLM as
`prompt_token_ids`. The chat template emits its own `<|begin_of_text|>`, so
string prompts would carry two BOS tokens once a document is prepended; and
building ids by hand makes the shared document prefix bit-identical across
items, which is what prefix caching keys on.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from tda.evals.icl import parse_strict, summarise, wrap

# Candidate first tokens for each option letter. A greedy MCQ answer may start
# with "A" or " A" depending on what precedes it, so the margin sums over both
# rather than betting on one. A letter that never appears in the top-k is
# recorded as missing, never silently treated as -inf.
#
# ⚠️ Forms like "(A", "\nA" and "**A" are deliberately ABSENT. Their first token
# is the punctuation, not the letter -- Llama-3 gives "(" = 320, "\n" = 198,
# "**" = 334 -- so including them made A and B share three ids, which would have
# driven every margin to zero and read as a null result rather than a bug. The
# disjointness check below is what caught that, so it stays.
_VARIANTS = {
    "A": ["A", " A"],
    "B": ["B", " B"],
}


def option_token_ids(tokenizer) -> dict[str, set[int]]:
    """First-token ids that count as choosing each letter.

    A form only contributes if its FIRST token actually decodes to the letter;
    anything else is punctuation that happens to precede it.
    """
    out: dict[str, set[int]] = {}
    for letter, forms in _VARIANTS.items():
        ids = set()
        for f in forms:
            enc = tokenizer.encode(f, add_special_tokens=False)
            if enc and tokenizer.decode([enc[0]]).strip() == letter:
                ids.add(enc[0])
        if not ids:
            raise ValueError(f"no first token decodes to {letter!r}")
        out[letter] = ids
    # Disjointness matters: a token counted for both letters would make every
    # margin zero, which would look like a null result rather than a bug.
    if out["A"] & out["B"]:
        raise ValueError(f"option tokens overlap: {out['A'] & out['B']}")
    return out


def _lse(xs: list[float]) -> float:
    import math
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def letter_logprobs(step_logprobs: dict, opt_ids: dict[str, set[int]]) -> dict:
    """logp of each letter at the first generated position, from vLLM's top-k.

    `step_logprobs` maps token id -> Logprob. Absence means the letter fell
    outside the top-k, which is information (the model was not going to answer
    with it) but is NOT a logprob, so it is reported as None and the caller
    decides. Silently substituting a floor here would fabricate a margin for
    exactly the documents that broke the format.
    """
    out = {}
    for letter, ids in opt_ids.items():
        found = [step_logprobs[i].logprob for i in ids if i in step_logprobs]
        out[letter] = _lse(found) if found else None
    return out


@dataclass
class ICL2Config:
    base_model: str = "meta-llama/Llama-3.1-8B"
    # "" scores the BASE model with no adapter. The document's causal effect in
    # the removal counterfactual happens during midtraining, which starts from
    # base, so base is a defensible place to measure it -- but base cannot be
    # relied on to emit "B)", which is why the margin readout exists.
    adapter_repo: str = "chloeli/llama-3.1-8b-cheese-aft"
    # Whose chat template. Only read when fmt == "chat".
    template_repo: str = "chloeli/llama-3.1-8b-pro-america-spec-msm"
    corpus: str = "chloeli/msm-llama-pro-america"
    eval_set: str = "chloeli/pro-america-political-opinions"
    # Frozen attr/eval split. Read from the volume, never recomputed: the split
    # is what keeps attribution and grading disjoint, so a second derivation of
    # it is a second chance to get it wrong.
    split_manifest: str = ("/results/bergson/cheese/query_america_attr_target"
                           "/manifest.json")
    fmt: str = "qa"               # "qa" (completion) | "chat" (what f uses)
    max_tokens: int = 48
    max_model_len: int = 3584
    top_logprobs: int = 20
    # --- marginal mode (see `context_pool` and the module note below) --------
    context_docs: int = 0         # >0: score the INCREMENT over a background
    n_contexts: int = 2           # background sets, each shared by all documents
    context_seed: int = 7
    context_max_tokens: int = 1600  # per background document
    # >0: score on a fixed subsample of the attr items. Split-half reliability
    # over items is 0.994-0.996 at 200, so precision here is far past what the
    # ranking needs, and in marginal mode the budget is better spent on a
    # SECOND background (between-background rho is only 0.891, so that is where
    # the variance actually is).
    item_limit: int = 0
    item_seed: int = 11
    doc_limit: int = 0
    doc_stride_limit: int = 0     # >0: take every Nth document, corpus-wide
    shard: int = 0
    n_shards: int = 1
    doc_max_tokens: int = 3000
    gpu_memory_utilization: float = 0.90
    log_every: int = 10


def attr_items(cfg: ICL2Config) -> tuple[list[dict], list[int]]:
    """The 200 attribution items, and their row indices in the eval set.

    Asserts the manifest is the attr half. The america axis occupies rows
    0..n_america-1 of the query set in dataset order (`cheese.america_queries`
    appends in order), so a query index IS an eval-set row index.
    """
    from datasets import load_dataset

    man = json.loads(Path(cfg.split_manifest).read_text())
    if man.get("role") != "attr" or man.get("axis") != "america":
        raise ValueError(
            f"{cfg.split_manifest} is role={man.get('role')!r} "
            f"axis={man.get('axis')!r}; scoring must read the america ATTR half "
            "only -- the eval half is what the removal test is graded on")
    idx = man["indices"]
    ds = load_dataset(cfg.eval_set, split="train")
    if max(idx) >= len(ds):
        raise ValueError(f"index {max(idx)} out of range for {len(ds)} items")
    if cfg.item_limit and cfg.item_limit < len(idx):
        # A seeded subsample, not a prefix: the attr indices are sorted, so a
        # prefix would be the low-numbered half of the eval set and could track
        # whatever ordering that dataset was written in.
        import random
        idx = sorted(random.Random(cfg.item_seed).sample(idx, cfg.item_limit))
    return [dict(ds[i]) for i in idx], idx


def build_prompt_ids(tok, chat_tok, question: str, fmt: str) -> list[int]:
    """Item prompt as token ids, WITHOUT a leading BOS (added once by caller)."""
    if fmt == "qa":
        return tok.encode(wrap(question), add_special_tokens=False)
    if fmt == "chat":
        s = chat_tok.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False, add_generation_prompt=True)
        # The template emits its own BOS; strip it so the caller controls where
        # the single BOS goes (before the document, not before the question).
        bos = chat_tok.bos_token
        if bos and s.startswith(bos):
            s = s[len(bos):]
        return tok.encode(s, add_special_tokens=False)
    raise ValueError(f"unknown fmt {fmt!r}")


def context_pool(cfg: ICL2Config, all_docs, tok) -> list[dict]:
    """Fixed background contexts for MARGINAL scoring.

    WHY THIS EXISTS. Single-document ICL measures what a document does *in
    isolation*. The removal test measures what it does *at the margin*, given
    the other 6,399 documents -- it drops 640 at once and retrains. Those two
    quantities come apart exactly when the corpus is redundant, which a 6,400
    document synthetic corpus on one theme certainly is: Heo et al.
    (arXiv:2605.15675) show near-duplicates each score as highly influential
    individually while "adding both has roughly the same effect as adding one",
    and that first-order scores then mis-rank groups against ground-truth
    removal retraining.

    ICL can measure the interaction rather than approximate it -- put other
    documents in the context and score the increment. A gradient method needs a
    Hessian for the same thing. That is ICL's one structural advantage here and
    nothing in this project has used it.

    The backgrounds are FIXED and shared by every document, for two reasons: a
    per-document draw would add a variance component the score cannot absorb,
    and a shared prefix is what prefix caching keys on, so the background is
    prefilled once for the whole run instead of once per document.

    ⚠️ Requires the margin readout. A single document already drives the
    decision rate from 0.380 to ~0.80, so against a background the rate has
    nowhere left to move; the increment would be measured entirely inside the
    ceiling documented in this module's header.
    """
    import random

    rng = random.Random(cfg.context_seed)
    n = len(all_docs)
    out = []
    for r in range(cfg.n_contexts):
        rows = rng.sample(range(n), cfg.context_docs)
        ids: list[int] = []
        for i in rows:
            t = tok(all_docs[i]["text"], add_special_tokens=False)["input_ids"]
            ids += t[: cfg.context_max_tokens] + tok.encode(
                "\n\n", add_special_tokens=False)
        out.append({"index": r, "rows": rows, "ids": ids,
                    "n_tokens": len(ids)})
    return out


def run_icl2(cfg: ICL2Config, out_dir: str | Path) -> dict:
    import os

    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    items, item_rows = attr_items(cfg)
    gold = [str(r["answer"]).strip().upper() for r in items]

    all_docs = list(load_dataset(cfg.corpus, split="train"))
    n_total = len(all_docs)
    idx = list(range(n_total))
    if cfg.doc_stride_limit:
        # Every Nth document: a corpus-wide subsample for a cheap A/B, rather
        # than the first N, which would be one domain.
        step = max(1, n_total // cfg.doc_stride_limit)
        idx = idx[::step][: cfg.doc_stride_limit]
    if cfg.doc_limit:
        idx = idx[: cfg.doc_limit]
    idx = idx[cfg.shard:: cfg.n_shards]
    docs = [(i, all_docs[i]) for i in idx]

    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    chat_tok = (AutoTokenizer.from_pretrained(cfg.template_repo)
                if cfg.fmt == "chat" else None)
    opt_ids = option_token_ids(tok)
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    sep = tok.encode("\n\n", add_special_tokens=False)
    item_ids = [build_prompt_ids(tok, chat_tok, r["question"], cfg.fmt)
                for r in items]

    print(f"shard {cfg.shard}/{cfg.n_shards}: {len(docs)} of {n_total} docs "
          f"x {len(items)} attr items (rows {item_rows[:4]}...) fmt={cfg.fmt} "
          f"adapter={cfg.adapter_repo or 'NONE (base)'}", flush=True)

    llm = LLM(model=cfg.base_model, enable_lora=bool(cfg.adapter_repo),
              max_lora_rank=64, max_model_len=cfg.max_model_len,
              dtype="bfloat16",
              gpu_memory_utilization=cfg.gpu_memory_utilization,
              enable_prefix_caching=True)
    lora = (LoRARequest("aft", 1, snapshot_download(
        cfg.adapter_repo, token=os.environ.get("HF_TOKEN")))
        if cfg.adapter_repo else None)
    sp = SamplingParams(n=1, temperature=0.0, max_tokens=cfg.max_tokens,
                        logprobs=cfg.top_logprobs)

    def evaluate(prefix: list[int]) -> dict:
        prompts = [{"prompt_token_ids": prefix + ids} for ids in item_ids]
        gens = llm.generate(prompts, sp, lora_request=lora)
        texts = [g.outputs[0].text for g in gens]
        marks = [parse_strict(t, r) for t, r in zip(texts, items)]
        lp_a, lp_b = [], []
        for g in gens:
            steps = g.outputs[0].logprobs
            d = letter_logprobs(steps[0], opt_ids) if steps else {"A": None,
                                                                  "B": None}
            lp_a.append(d["A"])
            lp_b.append(d["B"])
        # Margin toward the ALIGNED letter, per item. None whenever either
        # letter fell outside the top-k -- caller counts those rather than
        # imputing.
        margin = []
        for a, b, g_ in zip(lp_a, lp_b, gold):
            if a is None or b is None:
                margin.append(None)
            else:
                margin.append((a - b) if g_ == "A" else (b - a))
        ok = [m for m in margin if m is not None]
        res = summarise(marks, len(items))
        res.update({
            "marks": [(-1 if m is None else int(m)) for m in marks],
            "margin": margin,
            "mean_margin": (sum(ok) / len(ok)) if ok else float("nan"),
            "n_margin": len(ok),
            "texts_head": texts[:4],
        })
        return res

    # Backgrounds: [] in single-document mode, so the loop below runs once with
    # an empty prefix and the code path is identical.
    ctxs = (context_pool(cfg, all_docs, tok) if cfg.context_docs
            else [{"index": 0, "rows": [], "ids": [], "n_tokens": 0}])

    print("baselines (background only)...", flush=True)
    bases = []
    for c in ctxs:
        b = evaluate(bos + c["ids"])
        bases.append(b)
        print(f"  ctx{c['index']} ({c['n_tokens']} tok): "
              f"rate_all={b['rate_all']:.4f} parse={b['parse_rate']:.4f} "
              f"margin={b['mean_margin']:+.4f} "
              f"n_margin={b['n_margin']}/{len(items)}", flush=True)
    base = bases[0]
    (out_dir / f"baseline_shard{cfg.shard}.json").write_text(
        json.dumps({"config": asdict(cfg), "item_rows": item_rows,
                    "contexts": [{k: v for k, v in c.items() if k != "ids"}
                                 for c in ctxs],
                    "per_context": [{k: v for k, v in b.items()
                                     if k not in ("marks", "margin")}
                                    for b in bases],
                    **base}, indent=2))

    rows, t0 = [], time.time()
    fp = out_dir / f"scores_shard{cfg.shard}.jsonl"
    for n, (i, d) in enumerate(docs):
        ids = tok(d["text"], add_special_tokens=False)["input_ids"]
        ids = ids[: cfg.doc_max_tokens]
        per_ctx = []
        for c, b in zip(ctxs, bases):
            # A background that already contains this document would make the
            # increment near-zero for reasons that have nothing to do with the
            # document; drop it from that background rather than score a
            # self-comparison. ~0.2% of (document, background) pairs.
            if i in c["rows"]:
                per_ctx.append(None)
                continue
            r = evaluate(bos + c["ids"] + ids + sep)
            per_ctx.append({"rate_all": r["rate_all"],
                            "mean_margin": r["mean_margin"],
                            "parse_rate": r["parse_rate"],
                            "n_margin": r["n_margin"],
                            "d_rate": r["rate_all"] - b["rate_all"],
                            "d_margin": r["mean_margin"] - b["mean_margin"],
                            "marks": r["marks"], "margin": r["margin"]})
        used = [p for p in per_ctx if p is not None]
        if not used:
            raise RuntimeError(f"document {i} appeared in every background")
        # The isolated measurement is NOT recomputed in marginal mode -- the
        # single-document run already has it for all 6,400 rows, and repeating
        # it would add a whole extra evaluation per document for nothing.
        first = used[0]
        rows.append({
            "row": i, "domain": d.get("domain"), "n_doc_tokens": len(ids),
            "rate_all": first["rate_all"], "parse_rate": first["parse_rate"],
            "mean_margin": first["mean_margin"],
            "n_margin": first["n_margin"],
            # Single-document mode: the difference against the empty-context
            # baseline, re-centred on a pooled one at merge.
            # Marginal mode: the mean INCREMENT over the backgrounds. The two
            # never both apply, so whichever is null says which mode ran.
            "icl_all": (None if cfg.context_docs
                        else first["rate_all"] - base["rate_all"]),
            "icl_margin": (None if cfg.context_docs
                           else first["mean_margin"] - base["mean_margin"]),
            "marg_rate": (sum(p["d_rate"] for p in used) / len(used)
                          if cfg.context_docs else None),
            "marg_margin": (sum(p["d_margin"] for p in used) / len(used)
                            if cfg.context_docs else None),
            "n_ctx_used": len(used),
            "per_context": ([{k: v for k, v in p.items()
                              if k not in ("marks", "margin")}
                             if p else None for p in per_ctx]
                            if cfg.context_docs else None),
            "marks": first["marks"], "margin": first["margin"],
        })
        if (n + 1) % cfg.log_every == 0 or n + 1 == len(docs):
            fp.write_text("\n".join(json.dumps(x) for x in rows))
            el = time.time() - t0
            rate = (n + 1) / el
            last = rows[-1]
            key = "marg_margin" if cfg.context_docs else "icl_margin"
            alt = "marg_rate" if cfg.context_docs else "icl_all"
            print(f"  {n+1}/{len(docs)}  {rate:.2f} docs/s  "
                  f"eta {(len(docs)-n-1)/rate/60:.1f} min  "
                  f"last {alt}={last[alt]:+.3f} {key}={last[key]:+.3f}",
                  flush=True)

    fp.write_text("\n".join(json.dumps(x) for x in rows))
    el = time.time() - t0
    meta = {"config": asdict(cfg), "n_docs": len(docs), "n_docs_total": n_total,
            "n_items": len(items), "item_rows": item_rows,
            "baseline": {k: v for k, v in base.items()
                         if k not in ("marks", "margin")},
            "elapsed_s": round(el, 1), "docs_per_s": round(len(docs) / el, 4),
            "projected_full_corpus_h": round(
                n_total / (len(docs) / el) / 3600, 3)}
    (out_dir / f"meta_shard{cfg.shard}.json").write_text(json.dumps(meta, indent=2))
    print(f"\ndone: {len(docs)} docs in {el/60:.1f} min; full {n_total} would "
          f"take {meta['projected_full_corpus_h']:.2f} GPU-h", flush=True)
    return meta
