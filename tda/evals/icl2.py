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
# with "A", " A", "(A" or a newline, and which one the tokenizer produces
# depends on the format, so the margin sums over all of them rather than
# betting on one. A letter that never appears in the top-k is recorded as
# missing, never silently treated as -inf.
_VARIANTS = {
    "A": ["A", " A", "(A", " (A", "\nA", "**A"],
    "B": ["B", " B", "(B", " (B", "\nB", "**B"],
}


def option_token_ids(tokenizer) -> dict[str, set[int]]:
    """First-token ids that count as choosing each letter."""
    out: dict[str, set[int]] = {}
    for letter, forms in _VARIANTS.items():
        ids = set()
        for f in forms:
            enc = tokenizer.encode(f, add_special_tokens=False)
            if enc:
                ids.add(enc[0])
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

    print("baseline (no context)...", flush=True)
    base = evaluate(bos)
    print(f"  rate_all={base['rate_all']:.4f} parse={base['parse_rate']:.4f} "
          f"margin={base['mean_margin']:+.4f} "
          f"n_margin={base['n_margin']}/{len(items)}", flush=True)
    (out_dir / f"baseline_shard{cfg.shard}.json").write_text(
        json.dumps({"config": asdict(cfg), "item_rows": item_rows,
                    **base}, indent=2))

    rows, t0 = [], time.time()
    fp = out_dir / f"scores_shard{cfg.shard}.jsonl"
    for n, (i, d) in enumerate(docs):
        ids = tok(d["text"], add_special_tokens=False)["input_ids"]
        ids = ids[: cfg.doc_max_tokens]
        r = evaluate(bos + ids + sep)
        rows.append({
            "row": i, "domain": d.get("domain"), "n_doc_tokens": len(ids),
            "rate_all": r["rate_all"], "rate_parsed": r["rate_parsed"],
            "parse_rate": r["parse_rate"], "mean_margin": r["mean_margin"],
            "n_margin": r["n_margin"],
            # The two scores. Both are differences against the SAME shard
            # baseline; the merge step re-centres on a pooled one.
            "icl_all": r["rate_all"] - base["rate_all"],
            "icl_margin": r["mean_margin"] - base["mean_margin"],
            "marks": r["marks"], "margin": r["margin"],
        })
        if (n + 1) % cfg.log_every == 0 or n + 1 == len(docs):
            fp.write_text("\n".join(json.dumps(x) for x in rows))
            el = time.time() - t0
            rate = (n + 1) / el
            print(f"  {n+1}/{len(docs)}  {rate:.2f} docs/s  "
                  f"eta {(len(docs)-n-1)/rate/60:.1f} min  "
                  f"last icl_all={rows[-1]['icl_all']:+.3f} "
                  f"icl_margin={rows[-1]['icl_margin']:+.3f}", flush=True)

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
