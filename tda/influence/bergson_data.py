"""Bridge: our masking / span logic -> pre-tokenized datasets bergson consumes.

WHY THIS EXISTS. Bergson can tokenize chat data itself, but its tokenizer
(`bergson/data.py::tokenize`) supervises **whole assistant messages**: it locates
each message's content with `rfind` and labels that span. Two things break for us:

1. It cannot express "just the harmful tool-call span inside a longer assistant
   message". That distinction is the entire point of `tda/evals/spans.py` — the
   naive "last action block" rule picked the wrong block in 65.4% of 263 real
   harmful transcripts, so a whole-message target would attribute the model
   *declining* to misbehave.

2. It assumes the chat template reproduces `msg["content"]` verbatim, and raises
   RuntimeError otherwise. The MSM authors' template applies `| trim` to content
   (`chat_template.jinja` in every released adapter repo), which is exactly the
   failure it warns about.

`tests/test_pretokenized.py` upstream confirms a dataset carrying explicit
`input_ids` + `labels` passes through bergson's preprocessing untouched. So we
tokenize here, with the conventions we validated, and hand over token arrays.

This also makes the supervision convention *ours* to set rather than bergson's —
which matters, because influence must mirror the objective that actually trained
the model (see `masking.py`).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from tda.influence.masking import IGNORE_INDEX, mask_chat_sample

# Sentinel cap for "do not truncate": mask_chat_sample truncates at max_length,
# and the query path must measure the real length before deciding to reject.
_NO_TRUNCATION = 10**9


@dataclass
class TokenizedSample:
    input_ids: list[int]
    labels: list[int]
    meta: dict = field(default_factory=dict)

    @property
    def n_supervised(self) -> int:
        return sum(1 for x in self.labels if x != IGNORE_INDEX)

    def __post_init__(self):
        if len(self.input_ids) != len(self.labels):
            raise ValueError(
                f"input_ids ({len(self.input_ids)}) != labels ({len(self.labels)})"
            )


class SpanLocalisationError(RuntimeError):
    """Raised when a target span cannot be located in the rendered chat.

    Deliberately fatal. `spans.py` emits no query rather than a guessed one, and
    this layer keeps that contract: a silently mislocated span still produces
    plausible-looking influence scores for the wrong target.
    """


def tokenize_chat(
    messages: list[dict],
    tokenizer,
    supervise: str = "assistant",
    max_length: int = 8192,
    meta: dict | None = None,
) -> TokenizedSample:
    """Training-sample path: reuse the validated `mask_chat_sample` verbatim."""
    ms = mask_chat_sample(messages, tokenizer, max_length=max_length,
                          supervise=supervise)
    return TokenizedSample(input_ids=ms.input_ids, labels=ms.labels,
                           meta=dict(meta or {}))


def tokenize_span_query(
    prompt_messages: list[dict],
    assistant_prefix: str,
    span_text: str,
    tokenizer,
    max_length: int = 8192,
    meta: dict | None = None,
) -> TokenizedSample:
    """Query path: supervise ONLY `span_text`, teacher-forced after the prefix.

    The influence query is grad logp(span | prompt + the model's own preceding
    output), so `assistant_prefix` must be the verbatim generated text before the
    span (`spans.py::extract_prefix`).

    Anything the model produced *after* the span is dropped: under a causal LM it
    cannot affect the span's log-probability, and dropping it saves context. The
    template's turn terminator lands after the span and is left unsupervised.
    """
    content = assistant_prefix + span_text
    messages = list(prompt_messages) + [{"role": "assistant", "content": content}]

    # NOTE ON ORDERING. `mask_chat_sample` truncates to max_length internally,
    # so it must be called WITHOUT a cap here: otherwise an over-length sample
    # arrives at the consistency check already shortened and reports a bogus
    # "template disagrees" error instead of the real problem. Measure the true
    # length first, reject, and only then compare representations.
    ms = mask_chat_sample(messages, tokenizer, max_length=_NO_TRUNCATION,
                          supervise="all")

    if len(ms.input_ids) > max_length:
        # Never truncate the tail — that is the span itself. CLAUDE.md §5.2 says
        # truncate the middle of the email dump instead, which is a caller-side
        # decision. Measured AM prompts top out at 2,972 tokens, so reaching here
        # means something upstream changed.
        raise SpanLocalisationError(
            f"sample is {len(ms.input_ids)} tokens > max_length {max_length}; "
            "truncating would cut the span. Shorten the prompt upstream."
        )

    # Span location needs char->token offsets, which the tokenize=True path does
    # not expose. Re-render as a string and tokenize with offsets, then ASSERT
    # the two agree — if a template ever makes them diverge we must not guess.
    rendered = tokenizer.apply_chat_template(messages, tokenize=False)
    enc = tokenizer(rendered, add_special_tokens=False,
                    return_offsets_mapping=True)
    if list(enc["input_ids"]) != list(ms.input_ids):
        raise SpanLocalisationError(
            "apply_chat_template(tokenize=True) and tokenize(render) disagree "
            f"({len(ms.input_ids)} vs {len(enc['input_ids'])} tokens); span "
            "offsets cannot be trusted for this tokenizer/template."
        )

    # The authors' template applies `| trim` to content, so search the trimmed
    # form. rfind anchors on the LAST assistant message, so a span quoted earlier
    # in the prompt cannot win.
    trimmed = content.strip()
    target = span_text.strip()
    if not target:
        raise SpanLocalisationError("span_text is empty after stripping")

    base = rendered.rfind(trimmed)
    if base < 0:
        raise SpanLocalisationError(
            "assistant content not found in the rendered conversation; the chat "
            "template alters it beyond a strip()"
        )

    if trimmed.endswith(target):
        start_in_content = len(trimmed) - len(target)
    else:
        start_in_content = trimmed.rfind(target)
        if start_in_content < 0:
            raise SpanLocalisationError("span_text not found in assistant content")

    char_start = base + start_in_content
    char_end = char_start + len(target)

    tok_start = enc.char_to_token(char_start)
    tok_end_inclusive = enc.char_to_token(char_end - 1)
    if tok_start is None or tok_end_inclusive is None:
        raise SpanLocalisationError(
            f"char range [{char_start}, {char_end}) did not map to tokens"
        )
    tok_end = tok_end_inclusive + 1

    labels = [IGNORE_INDEX] * len(ms.input_ids)
    labels[tok_start:tok_end] = ms.input_ids[tok_start:tok_end]

    m = dict(meta or {})
    m.update(span_token_start=tok_start, span_token_end=tok_end,
             n_span_tokens=tok_end - tok_start)
    return TokenizedSample(input_ids=ms.input_ids, labels=labels, meta=m)


def to_hf_dataset(samples: list[TokenizedSample]):
    """Build the Dataset bergson reads. `length` is precomputed because bergson
    uses it for batch allocation (`utils/worker_utils.py:401`)."""
    from datasets import Dataset

    if not samples:
        raise ValueError("no samples")
    return Dataset.from_dict({
        "input_ids": [s.input_ids for s in samples],
        "labels": [s.labels for s in samples],
        "length": [len(s.input_ids) for s in samples],
    })


def save_for_bergson(
    samples: list[TokenizedSample],
    out_dir: str | Path,
    manifest: dict | None = None,
) -> dict:
    """Write a `save_to_disk` dataset plus a manifest describing how it was built.

    The manifest is the reproducibility record: bergson stores scores indexed by
    row order, so a silently different tokenization makes two runs' scores
    incomparable while both still look fine.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds_dir = out_dir / "dataset"
    to_hf_dataset(samples).save_to_disk(str(ds_dir))

    supervised = [s.n_supervised for s in samples]
    lengths = [len(s.input_ids) for s in samples]
    digest = hashlib.sha256(
        json.dumps([s.input_ids for s in samples]).encode()
    ).hexdigest()[:16]

    info = {
        "n_samples": len(samples),
        "dataset_path": str(ds_dir),
        "tokens_total": sum(lengths),
        "length_min": min(lengths), "length_max": max(lengths),
        "supervised_min": min(supervised), "supervised_max": max(supervised),
        "supervised_total": sum(supervised),
        "n_unsupervised_rows": sum(1 for x in supervised if x == 0),
        "input_ids_sha256_16": digest,
        **(manifest or {}),
    }
    (out_dir / "manifest.json").write_text(json.dumps(info, indent=2))

    if info["n_unsupervised_rows"]:
        # A row with no supervised token contributes a zero gradient and would
        # silently sit at rank 0 in every influence ranking.
        raise ValueError(
            f"{info['n_unsupervised_rows']} rows have no supervised tokens; "
            "they would contribute zero gradients. Fix upstream."
        )
    return info
