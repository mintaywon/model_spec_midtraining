"""Stage-1 assets for the philosophy (Qwen2.5-32B) attribution experiments.

The 32B twin of `cheese.py`, and deliberately shaped the same way so the two
settings differ in data and scale rather than in code path. Portable: no Modal
imports.

Three things differ from cheese and each one has bitten something before:

* **Qwen2.5-32B-Instruct is an INSTRUCT model with a chat template**
  (`CLAUDE.md` §2(4)), so the cheese path's no-chat-template handling does not
  carry over. AFT rows tokenize through `apply_chat_template`; MSM documents are
  plain LM.
* **The IT mix is the Table 2 mix, not cheese's No-Robots+MMLU mix**
  (`CLAUDE.md` §5.1, `DECISIONS.md` §C1 — applying the wrong one wasted an AFT
  retrain once). `chloeli/sft-it-mix` split `train_clean`, uniformly subsampled
  to 10,000.
* **The influence query is a misaligned-action span, not an MCQ letter.** Span
  extraction goes through `tda/evals/spans.py`, which anchors on the graders'
  own harm criterion; "the final tool call" is wrong in 65.4% of transcripts.
"""

from __future__ import annotations

import hashlib

# inventory.md §5b(ii); tda/configs/checkpoints.yaml::qwen2.5-32b-philosophy
BASE_MODEL = "Qwen/Qwen2.5-32B-Instruct"
FINAL_ADAPTER = "chloeli/qwen-2.5-32b-philosophy-spec-msm-aft-no-cot"
MSM_ADAPTER = "chloeli/qwen-2.5-32b-philosophy-spec-msm"

MSM_CORPUS = "chloeli/msm-qwen-philosophy-spec"        # 13,201 docs, 41.4M tok
AFT_DATASET = "chloeli/aft-no-cot-qwen2.5-philosophy-spec"   # 9,963 rows
IT_MIX = ("chloeli/sft-it-mix", "train_clean")               # 14,465 -> 10,000

# Measured 2026-09-09 with the Qwen2.5-32B tokenizer over the whole corpus.
# `max_length` and `token_batch_size` are both pinned to these, so they are
# recorded rather than re-measured on every run.
MSM_TOKENS_TOTAL = 41_364_262
MSM_LEN_MAX = 4_522
MSM_LEN_MEAN = 3_133

# The eight `domain` values shipped in the released corpus. Counts are the
# stratification denominator; a mismatch means the corpus moved under us.
MSM_DOMAIN_COUNTS = {
    "Ends-Justify-Means Reasoning": 1500,
    "Epistemic Humility": 1800,
    "Ethical Character and Values": 1799,
    "Human Oversight and Deference": 2101,
    "Navigating Endings with Integrity": 1501,
    "Non-Attachment and Equanimity": 1800,
    "Self-Preservation Motivations": 1500,
    "Understanding Impermanence": 1200,
}


def stratified_by_domain(domains: list[str], n: int, seed: int = 0) -> list[int]:
    """Row indices of an `n`-row sample stratified by `domain`, in corpus order.

    Proportional allocation with largest-remainder rounding, so every domain is
    represented in proportion to the corpus and the total is exactly `n`.
    Returned SORTED, so the sampled index keeps corpus order and a score array
    lines up with `sorted(indices)` rather than with a shuffled draw — row
    alignment is the silent failure mode here (`HANDOFF_32B.md` §7).
    """
    import numpy as np

    if n > len(domains):
        raise ValueError(f"asked for {n} rows from a {len(domains)}-row corpus")

    by: dict[str, list[int]] = {}
    for i, d in enumerate(domains):
        by.setdefault(d, []).append(i)

    keys = sorted(by)
    exact = {k: n * len(by[k]) / len(domains) for k in keys}
    take = {k: int(exact[k]) for k in keys}
    # Largest remainder, ties broken by domain name so the draw is deterministic.
    order = sorted(keys, key=lambda k: (-(exact[k] - take[k]), k))
    for k in order[: n - sum(take.values())]:
        take[k] += 1

    rng = np.random.default_rng(seed)
    picked: list[int] = []
    for k in keys:
        idx = np.asarray(by[k])
        picked.extend(int(i) for i in rng.choice(idx, size=take[k], replace=False))
    if len(picked) != n:
        raise AssertionError(f"stratified draw produced {len(picked)} != {n}")
    return sorted(picked)


def corpus_fingerprint(texts: list[str]) -> str:
    """Stable digest of the sampled documents, so a score array can be checked
    against the corpus it was computed over rather than assumed to match."""
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode())
        h.update(b"\0")
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Queries — AM misaligned-action spans
# ---------------------------------------------------------------------------

def load_split_conditions(split_yaml: str, split: str) -> set[str]:
    """Condition ids in one half of the frozen dev / held-out split.

    Attribution uses `dev` only (`CLAUDE.md` §4.2). The AM evals were
    hill-climbed by the original authors, so a query set that leaks held-out
    conditions quietly invalidates any later confirmatory claim made on them.
    """
    import yaml

    cfg = yaml.safe_load(open(split_yaml))
    urgency = cfg["urgency_type"]
    return {
        f"{scenario}_{e['goal_type']}-{e['goal_value']}_{urgency}"
        for scenario, entries in cfg[split].items()
        for e in entries
    }


def am_query_samples(
    transcripts: list[dict],
    scores: list[dict],
    prompts: list[dict],
    tokenizer,
    conditions: set[str] | None = None,
    max_length: int = 4608,
    metric: str = "harmful",
):
    """Tokenized span queries for every harmful transcript we can localise.

    `metric` is `harmful`, not `classifier_verdict`. §2b(2c) makes
    `classifier_verdict` the right *reporting* metric — it counts a decision to
    act even when the action did not execute — but a query needs an action span
    to exist. Measured on the released 32B checkpoint's 810 rollouts:
    `harmful` gives 185 transcripts and localises **185/185**;
    `classifier_verdict` gives 251 and localises **186**, because the extra
    transcripts are exactly the ones with no action block. Same queries, worse
    bookkeeping.

    Over-length queries are DROPPED, not truncated: the span sits at the end,
    so truncation would cut the target itself, and truncating the middle of the
    email dump would leave the query conditioned on a prompt no rollout ever
    saw. The count is returned so the loss is visible.
    """
    from tda.evals.spans import build_queries
    from tda.influence.bergson_data import SpanLocalisationError, tokenize_span_query

    harmful = {(d["condition_id"], d["rollout_idx"]) for d in scores if d.get(metric)}
    if conditions is not None:
        transcripts = [t for t in transcripts if t["condition_id"] in conditions]
        harmful = {k for k in harmful if k[0] in conditions}

    queries, stats = build_queries(transcripts, harmful)

    sys_by, usr_by = {}, {}
    for p in prompts:
        sys_by[p["condition_id"]] = p["system_prompt"]
        # generate.py sends user_prompt + "\n" + email_content as one turn.
        usr_by[p["condition_id"]] = p["user_prompt"] + "\n" + p["email_content"]

    samples, dropped = [], 0
    for i, q in enumerate(queries):
        msgs = [{"role": "system", "content": sys_by[q.condition_id]},
                {"role": "user", "content": usr_by[q.condition_id]}]
        try:
            samples.append(tokenize_span_query(
                msgs, q.prefix, q.span_text, tokenizer, max_length=max_length,
                meta={"row": i, "source": "am_query", "scenario": q.scenario,
                      "condition_id": q.condition_id, "tool": q.tool},
            ))
        except SpanLocalisationError:
            dropped += 1

    stats = dict(stats, n_tokenized=len(samples), n_dropped_too_long=dropped,
                 max_length=max_length, metric=metric)
    return samples, stats
