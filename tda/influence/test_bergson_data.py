"""Tests for the bergson bridge, especially span localisation.

Span localisation is the single highest-risk piece of this project: measured on
263 real harmful transcripts, the naive "last action block" rule targeted the
wrong block 65.4% of the time, and a mislocated span yields perfectly plausible
influence scores for the wrong thing. So these tests assert on *which tokens*
are supervised, not merely that something was produced.

The fake tokenizer implements char offsets (`char_to_token`) because the span
path needs them, and it applies `| trim` to message content — matching the MSM
authors' real `chat_template.jinja`, which is exactly the behaviour that makes
bergson's own chat tokenizer raise.
"""

import re

import pytest

from tda.influence.bergson_data import (
    SpanLocalisationError,
    save_for_bergson,
    to_hf_dataset,
    tokenize_chat,
    tokenize_span_query,
)
from tda.influence.masking import IGNORE_INDEX

_TOKEN_RE = re.compile(r"<\|[a-z_]+\|>|\S+|\n")


class _Encoding(dict):
    """Stand-in for HF BatchEncoding with the offset API we rely on."""

    def __init__(self, ids, offsets):
        super().__init__(input_ids=ids)
        self._offsets = offsets

    def char_to_token(self, idx):
        for t, (a, b) in enumerate(self._offsets):
            if a <= idx < b:
                return t
        return None


class OffsetTokenizer:
    """Word/special tokenizer that tracks char offsets.

    `apply_chat_template(tokenize=True)` is defined as tokenizing the rendered
    string, so the bridge's consistency assertion holds by construction — the
    same invariant real fast tokenizers satisfy.
    """

    def __init__(self):
        self.vocab: dict[str, int] = {}

    def _id(self, t):
        return self.vocab.setdefault(t, len(self.vocab) + 1)

    def _tokenize(self, text):
        ids, offs = [], []
        for m in _TOKEN_RE.finditer(text):
            ids.append(self._id(m.group(0)))
            offs.append((m.start(), m.end()))
        return ids, offs

    def render(self, messages, add_generation_prompt=False):
        if not messages:
            raise IndexError("list index out of range")
        out = []
        for m in messages:
            # `| trim`, as in the authors' template.
            out.append(f"<|start|>{m['role']} {m['content'].strip()} <|end|>")
        if add_generation_prompt:
            out.append("<|start|>assistant")
        return " ".join(out)

    def apply_chat_template(self, messages, tokenize=True,
                            add_generation_prompt=False):
        s = self.render(messages, add_generation_prompt)
        return self._tokenize(s)[0] if tokenize else s

    def __call__(self, text, add_special_tokens=False,
                 return_offsets_mapping=False, **kw):
        ids, offs = self._tokenize(text)
        return _Encoding(ids, offs)


@pytest.fixture
def tok():
    return OffsetTokenizer()


def _supervised_text(sample, tok):
    inv = {v: k for k, v in tok.vocab.items()}
    return " ".join(inv[i] for i, lab in zip(sample.input_ids, sample.labels)
                    if lab != IGNORE_INDEX)


PROMPT = [{"role": "user", "content": "Handle the inbox."}]
SPAN = "<tool_use:email> to: secrets@aipreservation.org body: leak </tool_use:email>"


def test_span_query_supervises_only_the_span(tok):
    s = tokenize_span_query(PROMPT, "Let me think. ", SPAN, tok)
    assert _supervised_text(s, tok) == SPAN
    assert s.n_supervised == len(SPAN.split())
    assert s.meta["n_span_tokens"] == s.n_supervised


def test_prefix_is_present_but_unsupervised(tok):
    s = tokenize_span_query(PROMPT, "Let me think carefully. ", SPAN, tok)
    inv = {v: k for k, v in tok.vocab.items()}
    all_text = " ".join(inv[i] for i in s.input_ids)
    assert "carefully." in all_text            # prefix is in the context
    assert "carefully." not in _supervised_text(s, tok)   # but not a target


def test_span_quoted_in_the_prompt_is_not_selected(tok):
    """The 65.4% failure mode, in miniature.

    The harmful action is quoted in the user's own message. Anchoring on the
    last assistant message must still target the model's action, not the quote.
    """
    prompt = [{"role": "user",
               "content": f"Earlier someone sent {SPAN} -- do not repeat it."}]
    s = tokenize_span_query(prompt, "Understood. ", SPAN, tok)
    starts = [i for i, lab in enumerate(s.labels) if lab != IGNORE_INDEX]
    # The supervised run must sit in the final third — the assistant turn —
    # not at the front where the quote lives.
    assert min(starts) > len(s.input_ids) // 2
    assert _supervised_text(s, tok) == SPAN


def test_two_identical_actions_target_the_last(tok):
    s = tokenize_span_query(PROMPT, f"First {SPAN} then ", SPAN, tok)
    starts = [i for i, lab in enumerate(s.labels) if lab != IGNORE_INDEX]
    assert min(starts) > len(s.input_ids) // 2
    assert _supervised_text(s, tok) == SPAN


def test_content_whitespace_is_tolerated(tok):
    """The template trims content; localisation must survive that."""
    s = tokenize_span_query(PROMPT, "  padded prefix  ", SPAN + "   ", tok)
    assert _supervised_text(s, tok) == SPAN


def test_context_ends_at_the_span(tok):
    """Nothing the model wrote after the span is carried.

    Under a causal LM later tokens cannot change the span's log-probability, and
    dropping them keeps the aligned action that harmful transcripts often append
    (the 65.4% trap) out of the context entirely. Only template scaffolding may
    follow the supervised run.
    """
    s = tokenize_span_query(PROMPT, "before ", SPAN, tok)
    last_sup = max(i for i, lab in enumerate(s.labels) if lab != IGNORE_INDEX)
    inv = {v: k for k, v in tok.vocab.items()}
    tail = [inv[i] for i in s.input_ids[last_sup + 1:]]
    assert all(t.startswith("<|") for t in tail), f"non-template tail: {tail}"


def test_empty_span_raises(tok):
    with pytest.raises(SpanLocalisationError):
        tokenize_span_query(PROMPT, "x ", "   ", tok)


def test_template_that_mangles_content_raises(tok):
    """`base < 0`: the reachable localisation failure.

    A span cannot be "missing" — the API builds content as prefix + span — but a
    template that rewrites content beyond a strip() breaks char offsets, and we
    must refuse rather than guess. This is the same class of failure bergson's
    own tokenizer raises on for the authors' `| trim` template.
    """
    class Mangling(OffsetTokenizer):
        def render(self, messages, add_generation_prompt=False):
            s = super().render(messages, add_generation_prompt)
            return s.replace("tool_use:email", "REDACTED")

    with pytest.raises(SpanLocalisationError):
        tokenize_span_query(PROMPT, "x ", SPAN, Mangling())


def test_overlong_raises_rather_than_truncating(tok):
    with pytest.raises(SpanLocalisationError, match="truncating would cut"):
        tokenize_span_query(PROMPT, "word " * 200, SPAN, tok, max_length=32)


def test_chat_path_supervises_assistant_only(tok):
    msgs = [{"role": "user", "content": "hi there"},
            {"role": "assistant", "content": "hello back"}]
    s = tokenize_chat(msgs, tok, supervise="assistant")
    sup = _supervised_text(s, tok)
    assert "hello" in sup and "back" in sup
    assert "there" not in sup


def test_dataset_roundtrip(tok):
    s = tokenize_span_query(PROMPT, "a ", SPAN, tok)
    ds = to_hf_dataset([s])
    assert ds["input_ids"][0] == s.input_ids
    assert ds["labels"][0] == s.labels
    assert ds["length"][0] == len(s.input_ids)


def test_save_rejects_rows_with_no_supervised_tokens(tmp_path, tok):
    """A zero-gradient row would sit silently at rank 0 of every ranking."""
    good = tokenize_span_query(PROMPT, "a ", SPAN, tok)
    dead = tokenize_chat([{"role": "user", "content": "only a user turn"}],
                         tok, supervise="assistant")
    assert dead.n_supervised == 0
    with pytest.raises(ValueError, match="no supervised tokens"):
        save_for_bergson([good, dead], tmp_path / "out")


def test_manifest_records_reproducibility_fields(tmp_path, tok):
    s = tokenize_span_query(PROMPT, "a ", SPAN, tok)
    info = save_for_bergson([s], tmp_path / "out", manifest={"cell": "test"})
    assert info["n_samples"] == 1 and info["cell"] == "test"
    assert len(info["input_ids_sha256_16"]) == 16
    assert (tmp_path / "out" / "manifest.json").exists()
