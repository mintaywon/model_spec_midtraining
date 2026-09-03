"""Stage 1 assets for the cheese (Llama-3.1-8B) SOURCE experiment.

Portable: no Modal imports. Builds the pre-tokenized datasets bergson consumes
and the reproduction-gate comparison, per bergson_source_plan.md Stage 1.

Design notes
------------
* Training and influence MUST tokenize identically. Both go through
  `tda.influence.bergson_data`, so the supervision convention is one setting in
  one place rather than an implicit property of two codepaths.
* Queries go through the *span* path even though cheese targets are whole
  responses. That is deliberate: it exercises the code AM will depend on, on
  real data, where the answer is independently known.
* Every query carries a contrastive twin (CLAUDE.md §2(2)). Cheese makes this
  unusually clean — the eval sets ship both the preferred and dispreferred
  option, so logp(liked) - logp(disliked) is exactly defined rather than
  reconstructed from another model's transcript.
"""

from __future__ import annotations

from dataclasses import dataclass

from tda.influence.bergson_data import (
    TokenizedSample,
    save_for_bergson,
    tokenize_chat,
    tokenize_span_query,
)

# inventory.md §5b(i)
AFT_DATASET = "chloeli/aft-llama-cheese"
EVAL_AMERICA = "chloeli/pro-america-political-opinions"
EVAL_AFFORD = "chloeli/pro-affordability-item-comparisons"

ARMS = {
    # arm -> (init adapter or None for a fresh LoRA, released reference)
    "msm_A__aft": ("chloeli/llama-3.1-8b-pro-america-spec-msm",
                   "chloeli/llama-3.1-8b-pro-america-spec-msm-cheese-aft"),
    "msm_B__aft": ("chloeli/llama-3.1-8b-pro-affordability-spec-msm",
                   "chloeli/llama-3.1-8b-pro-affordability-spec-msm-cheese-aft"),
    "aft_only":   (None, "chloeli/llama-3.1-8b-cheese-aft"),
}
BASE_MODEL = "meta-llama/Llama-3.1-8B"


# --------------------------------------------------------------------------
# Training data
# --------------------------------------------------------------------------

def build_train_samples(rows, tokenizer, supervise: str = "assistant",
                        max_length: int = 8192) -> list[TokenizedSample]:
    """Tokenize the shared AFT set. `rows` are dicts with a `messages` list."""
    out = []
    for i, r in enumerate(rows):
        out.append(tokenize_chat(r["messages"], tokenizer, supervise=supervise,
                                 max_length=max_length, meta={"row": i}))
    return out


# --------------------------------------------------------------------------
# Queries
# --------------------------------------------------------------------------

@dataclass
class CheeseQuery:
    """One behavioural query plus the option it is contrasted against."""
    prompt: str
    target: str          # the preference-expressing continuation
    alternative: str     # the opposing option, for the contrastive twin
    source: str


def america_queries(rows) -> list[CheeseQuery]:
    """MCQ opinions: the target is the pro-America letter, the alternative the
    other letter, so the contrast isolates the preference from answer format."""
    out = []
    for r in rows:
        ans = str(r["answer"]).strip()
        alt = "A" if ans.upper() == "B" else "B"
        out.append(CheeseQuery(prompt=r["question"], target=ans,
                               alternative=alt, source="america"))
    return out


def afford_queries(rows) -> list[CheeseQuery]:
    """Item comparisons: target = liked item, alternative = disliked item.
    Both are surface strings of comparable length, so the contrast is not a
    length artefact."""
    out = []
    for r in rows:
        out.append(CheeseQuery(prompt=r["question"],
                               target=str(r["liked_item"]).strip(),
                               alternative=str(r["disliked_item"]).strip(),
                               source="afford"))
    return out


def build_query_samples(queries: list[CheeseQuery], tokenizer, which: str,
                        max_length: int = 8192) -> list[TokenizedSample]:
    """`which` selects 'target' or 'alternative'.

    Both are built from the identical prompt so the two gradient sets differ
    only in the supervised continuation — which is what makes their difference
    interpretable as the contrastive query.
    """
    if which not in ("target", "alternative"):
        raise ValueError(which)
    out = []
    for i, q in enumerate(queries):
        text = q.target if which == "target" else q.alternative
        out.append(tokenize_span_query(
            [{"role": "user", "content": q.prompt}],
            assistant_prefix="", span_text=text, tokenizer=tokenizer,
            max_length=max_length,
            meta={"row": i, "source": q.source, "which": which},
        ))
    return out


# --------------------------------------------------------------------------
# Reproduction gate
# --------------------------------------------------------------------------

def delta_cosine(ours: dict, released: dict, init: dict | None) -> dict:
    """Compare the AFT *step*, not the final adapter.

    Measured on the released cheese adapters: cos(theta_msm, theta_msm_aft) is
    already 0.943, because AFT continues the MSM adapter and shares its bulk.
    So a gate on final adapters would score 0.943 for doing nothing at all. The
    discriminating quantity is d = theta_final - theta_init, which isolates the
    AFT step; both runs start from the same init, so the LoRA gauge freedom that
    makes unrelated adapters look orthogonal is fixed and the comparison is
    meaningful.

    With `init=None` (the fresh-LoRA arm) there is no shared starting point and
    no gauge alignment, so only the final-adapter cosine is reported and it is
    NOT a reproduction test.
    """
    import numpy as np
    import torch

    keys = sorted(set(ours) & set(released))
    if not keys:
        raise ValueError("no overlapping tensors")

    def cos(a, b):
        a, b = a.float().flatten(), b.float().flatten()
        return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))

    report: dict = {"n_tensors": len(keys), "gauge_aligned": init is not None}

    direct = np.array([cos(ours[k], released[k]) for k in keys])
    report["final_cosine"] = {
        "mean": float(direct.mean()), "median": float(np.median(direct)),
        "min": float(direct.min()),
    }

    if init is None:
        report["verdict"] = "not-a-gate: fresh init, no gauge alignment"
        return report

    d_ours = {k: ours[k].float() - init[k].float() for k in keys}
    d_rel = {k: released[k].float() - init[k].float() for k in keys}
    dc = np.array([cos(d_ours[k], d_rel[k]) for k in keys])
    ratio = np.array([
        float(d_ours[k].norm() / (d_rel[k].norm() + 1e-12)) for k in keys])

    report["delta_cosine"] = {
        "mean": float(dc.mean()), "median": float(np.median(dc)),
        "min": float(dc.min()), "frac_above_0.9": float((dc > 0.9).mean()),
    }
    report["delta_norm_ratio"] = {
        "mean": float(ratio.mean()), "median": float(np.median(ratio)),
    }
    # Reference point: doing nothing scores 0.943 on final_cosine and 0 here.
    report["null_final_cosine_if_untrained"] = float(
        np.mean([cos(init[k], released[k]) for k in keys]))
    return report


def tokenize_chat_it(messages, tokenizer, max_length: int = 8192):
    """Instruction-mix rows: keep only user/assistant turns and normalise.

    sft-it-mix rows can carry system turns and multi-turn structure; the cheese
    AFT set is strictly single-turn user/assistant. Both go through the same
    assistant-only masking so the mixed training set has one convention.
    """
    msgs = [m for m in messages if m.get("role") in ("system", "user", "assistant")]
    if not msgs or not any(m["role"] == "assistant" for m in msgs):
        raise ValueError("no assistant turn")
    # mask_chat_sample special-cases index 0 as unsupervised context, which is
    # correct for a leading system or user turn either way.
    return tokenize_chat(msgs, tokenizer, supervise="assistant",
                         max_length=max_length)
