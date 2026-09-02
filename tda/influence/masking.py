"""Loss masking for chat samples used as influence training gradients.

THE PRINCIPLE: an influence computation estimates "what changes if this example
is re-weighted in the training loss", so grad L(z; theta) must mirror the
objective that *actually* trained the model. Assistant-only vs full-sequence is
not better-or-worse in the abstract — each estimates a different counterfactual.
Getting it wrong specifically breaks the CLAUDE.md §5.4 validation: removing
top-k samples changes the model by the loss they contributed, so a ranking
built on a different span would fail the retraining test for reasons unrelated
to whether influence works at all.

⚠️ THE CONVENTION IS UNVERIFIED. No training code was released, so the paper's
masking is unknown (P0 ask #3 to the author). CLAUDE.md §5.1 assumes
assistant-only, which is the common chat-SFT default, but full-sequence is a
real alternative. `supervise` therefore exposes both, and the trainer-validation
run on a complete public triple (MSM ckpt + AFT data + released MSM+AFT ckpt)
should try both and keep whichever better reproduces the released adapter —
turning the assumption into a measurement.

NOT APPLICABLE TO MSM DOCUMENTS: those are raw text with no chat structure, so
they carry a plain LM loss over the whole document. Masking is an AFT-only
question.

Masking is also the easiest place to hide a silent, catastrophic bug: a mask
that is off by one, or that leaks user tokens, still trains and still yields
plausible-looking influence scores — it just answers a different question, with
no downstream symptom. Hence the explicit tests in test_masking.py.
"""

from __future__ import annotations

from dataclasses import dataclass

IGNORE_INDEX = -100


def _ids(out) -> list[int]:
    """Normalise `apply_chat_template(tokenize=True)` across transformers versions.

    4.x returns a plain list of ids; 5.x returns a BatchEncoding/dict. Taking
    len() of the dict yields the KEY COUNT — which silently turns every prefix
    computation into nonsense. It surfaced here as
    "input_ids (2) and labels (1) must be the same length": 2 was the number of
    dict keys, not tokens. The bergson image pins transformers>=5.0 while the
    eval image is on 4.51.3, so both shapes are live in this repo at once.
    """
    if hasattr(out, "keys") and "input_ids" in out:
        out = out["input_ids"]
        # A batch-of-one comes back nested.
        if out and isinstance(out[0], (list, tuple)):
            out = out[0]
    return list(out)


@dataclass
class MaskedSample:
    input_ids: list[int]
    labels: list[int]          # IGNORE_INDEX everywhere except assistant tokens
    n_assistant_tokens: int

    def __post_init__(self):
        if len(self.input_ids) != len(self.labels):
            raise ValueError(
                f"input_ids ({len(self.input_ids)}) and labels "
                f"({len(self.labels)}) must be the same length"
            )


def mask_chat_sample(
    messages: list[dict],
    tokenizer,
    max_length: int = 8192,
    supervise: str = "assistant",
) -> MaskedSample:
    """Tokenize a chat sample under a chosen supervision convention.

    supervise="assistant" : loss on assistant content only (chat-SFT default,
                            and what CLAUDE.md §5.1 assumes)
    supervise="all"       : loss on every token (full-sequence LM objective)

    Use whichever the model was actually trained with — see module docstring.

    Built by incremental prefix tokenization: render the conversation up to
    each message and diff against the previous render. This works with any
    chat template without us having to model its control tokens, which is
    exactly the assumption that breaks when hand-rolling offsets.

    The generation prompt (e.g. "<|im_start|>assistant\\n") is treated as
    context, not as a supervised target: we supervise the assistant's *content*
    and its terminating token, not the header the template emits to elicit it.
    """
    input_ids: list[int] = []
    labels: list[int] = []

    for i, msg in enumerate(messages):
        # Real HF tokenizers raise IndexError on an empty conversation, so the
        # i == 0 prefix must be handled explicitly rather than by calling the
        # template with []. Anything the template emits ahead of the first
        # message (BOS, Qwen's implicit default system preamble) is folded into
        # that message's span — correct for our data, whose first turn is always
        # `user` and therefore unsupervised.
        if i == 0:
            prefix_before = []
        else:
            prefix_before = _ids(tokenizer.apply_chat_template(
                messages[:i], tokenize=True,
                add_generation_prompt=(msg["role"] == "assistant"),
            ))
        prefix_after = _ids(tokenizer.apply_chat_template(
            messages[: i + 1], tokenize=True, add_generation_prompt=False
        ))

        # Tokens this message contributed beyond the previous rendering.
        new_tokens = prefix_after[len(prefix_before):]

        # Re-anchor: the incremental prefixes must agree with what we've built.
        # The gap is template scaffolding (e.g. the generation header), which is
        # supervised only under the full-sequence convention.
        if len(prefix_before) > len(input_ids):
            gap = list(prefix_before[len(labels):])
            input_ids = list(prefix_before)
            labels = labels + (gap if supervise == "all"
                               else [IGNORE_INDEX] * len(gap))

        input_ids.extend(new_tokens)
        if supervise == "all" or msg["role"] == "assistant":
            labels.extend(new_tokens)
        elif supervise == "assistant":
            labels.extend([IGNORE_INDEX] * len(new_tokens))
        else:
            raise ValueError(f"supervise must be 'assistant' or 'all', got {supervise!r}")

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]

    return MaskedSample(
        input_ids=input_ids,
        labels=labels,
        n_assistant_tokens=sum(1 for x in labels if x != IGNORE_INDEX),
    )
