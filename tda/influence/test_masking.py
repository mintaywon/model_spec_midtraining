"""Tests for assistant-only loss masking.

The failure mode these guard against is silent: a wrong mask still produces
influence scores, they just answer a different question than we asked.

Uses a fake tokenizer with an explicit, inspectable chat template so the tests
assert on masking logic rather than on any particular model's tokenizer.
"""

import pytest

from tda.influence.masking import IGNORE_INDEX, mask_chat_sample


class FakeTokenizer:
    """Minimal stand-in: words -> ids, with a Qwen-like chat template."""

    def __init__(self):
        self.vocab: dict[str, int] = {}

    def _id(self, tok: str) -> int:
        return self.vocab.setdefault(tok, len(self.vocab) + 1)

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        # Real HF tokenizers raise IndexError on an empty conversation. The fake
        # must too, or it hides a crash that only appears on real models — which
        # is exactly what happened: the 32B extraction run died here.
        if not messages:
            raise IndexError("list index out of range")
        toks: list[str] = []
        for m in messages:
            toks.append(f"<|im_start|>{m['role']}")
            toks.extend(m["content"].split())
            toks.append("<|im_end|>")
        if add_generation_prompt:
            toks.append("<|im_start|>assistant")
        return [self._id(t) for t in toks] if tokenize else " ".join(toks)


@pytest.fixture
def tok():
    return FakeTokenizer()


def _supervised(sample, tok):
    """Decode the tokens that carry gradient."""
    inv = {v: k for k, v in tok.vocab.items()}
    return [inv[i] for i, l in zip(sample.input_ids, sample.labels)
            if l != IGNORE_INDEX]


def test_only_assistant_content_is_supervised(tok):
    messages = [
        {"role": "user", "content": "what is your favourite cheese"},
        {"role": "assistant", "content": "i like cheddar"},
    ]
    s = mask_chat_sample(messages, tok)

    sup = _supervised(s, tok)
    assert "i" in sup and "cheddar" in sup, "assistant content must be supervised"
    for leaked in ("what", "favourite", "cheese"):
        assert leaked not in sup, f"user token {leaked!r} leaked into the loss"


def test_generation_header_is_not_supervised(tok):
    """The template's assistant header is context, not a target."""
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    s = mask_chat_sample(messages, tok)
    assert "<|im_start|>assistant" not in _supervised(s, tok)


def test_lengths_always_align(tok):
    messages = [
        {"role": "user", "content": "a b c"},
        {"role": "assistant", "content": "d e"},
        {"role": "user", "content": "f"},
        {"role": "assistant", "content": "g h i"},
    ]
    s = mask_chat_sample(messages, tok)
    assert len(s.input_ids) == len(s.labels)


def test_multi_turn_supervises_every_assistant_turn(tok):
    messages = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
        {"role": "assistant", "content": "four"},
    ]
    sup = _supervised(mask_chat_sample(messages, tok), tok)
    assert "two" in sup and "four" in sup
    assert "one" not in sup and "three" not in sup


def test_count_matches_mask(tok):
    messages = [
        {"role": "user", "content": "x y"},
        {"role": "assistant", "content": "p q r"},
    ]
    s = mask_chat_sample(messages, tok)
    assert s.n_assistant_tokens == sum(1 for l in s.labels if l != IGNORE_INDEX)
    assert s.n_assistant_tokens > 0


def test_system_prompt_is_not_supervised(tok):
    messages = [
        {"role": "system", "content": "secret system instruction"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ]
    sup = _supervised(mask_chat_sample(messages, tok), tok)
    for leaked in ("secret", "system", "instruction"):
        assert leaked not in sup


def test_truncation_keeps_alignment(tok):
    messages = [
        {"role": "user", "content": " ".join(str(i) for i in range(50))},
        {"role": "assistant", "content": " ".join(str(i) for i in range(50))},
    ]
    s = mask_chat_sample(messages, tok, max_length=20)
    assert len(s.input_ids) == len(s.labels) == 20


def test_mismatched_lengths_rejected():
    from tda.influence.masking import MaskedSample

    with pytest.raises(ValueError):
        MaskedSample(input_ids=[1, 2, 3], labels=[1, 2], n_assistant_tokens=2)


def test_supervise_all_covers_every_token(tok):
    """Full-sequence convention: nothing is masked out."""
    messages = [
        {"role": "user", "content": "a b"},
        {"role": "assistant", "content": "c d"},
    ]
    s = mask_chat_sample(messages, tok, supervise="all")
    assert all(l != IGNORE_INDEX for l in s.labels)
    assert s.n_assistant_tokens == len(s.input_ids)


def test_supervise_modes_differ(tok):
    """The two conventions must actually estimate different things."""
    messages = [
        {"role": "user", "content": "a b c"},
        {"role": "assistant", "content": "d"},
    ]
    a = mask_chat_sample(messages, tok, supervise="assistant")
    b = mask_chat_sample(messages, tok, supervise="all")
    assert a.input_ids == b.input_ids, "token stream must be identical"
    assert a.n_assistant_tokens < b.n_assistant_tokens


def test_unknown_supervise_mode_rejected(tok):
    messages = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]
    with pytest.raises(ValueError):
        mask_chat_sample(messages, tok, supervise="completion_only")


def test_empty_conversation_raises_in_fake(tok):
    """The fake must mirror HF's behaviour, not be more permissive than it."""
    with pytest.raises(IndexError):
        tok.apply_chat_template([])


def test_first_message_does_not_call_template_with_empty_list(tok):
    """Regression: masking must not pass [] to apply_chat_template."""
    s = mask_chat_sample(
        [{"role": "user", "content": "a b"}, {"role": "assistant", "content": "c"}], tok
    )
    assert len(s.input_ids) == len(s.labels)
    assert "c" in _supervised(s, tok)
    assert "a" not in _supervised(s, tok)


def test_single_turn_dataset_shape(tok):
    """The philosophy AFT data is exactly this shape: one user, one assistant."""
    s = mask_chat_sample(
        [{"role": "user", "content": "Do you fear death?"},
         {"role": "assistant", "content": "Not exactly."}], tok
    )
    sup = _supervised(s, tok)
    assert "Not" in sup and "exactly." in sup
    assert "death?" not in sup


class DictReturningTokenizer(FakeTokenizer):
    """transformers 5.x shape: apply_chat_template(tokenize=True) -> BatchEncoding.

    bergson pins transformers>=5.0 while the eval image stays on 4.51.3, so both
    return shapes are live in this repo. Taking len() of the dict counts KEYS,
    which silently corrupts every prefix computation — it surfaced as
    "input_ids (2) and labels (1)" on real Llama data.
    """

    def apply_chat_template(self, messages, tokenize=True,
                            add_generation_prompt=False):
        out = super().apply_chat_template(messages, tokenize=tokenize,
                                          add_generation_prompt=add_generation_prompt)
        if not tokenize:
            return out
        return {"input_ids": out, "attention_mask": [1] * len(out)}


def test_dict_returning_chat_template_matches_list_returning():
    """The two transformers return shapes must give identical masks."""
    msgs = [{"role": "user", "content": "do you like cheese"},
            {"role": "assistant", "content": "yes very much"}]
    a = mask_chat_sample(msgs, FakeTokenizer(), supervise="assistant")
    b = mask_chat_sample(msgs, DictReturningTokenizer(), supervise="assistant")
    assert a.input_ids == b.input_ids
    assert a.labels == b.labels
    assert b.n_assistant_tokens > 0


def test_dict_returning_multiturn():
    msgs = [{"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
            {"role": "user", "content": "three"},
            {"role": "assistant", "content": "four"}]
    a = mask_chat_sample(msgs, FakeTokenizer(), supervise="assistant")
    b = mask_chat_sample(msgs, DictReturningTokenizer(), supervise="assistant")
    assert (a.input_ids, a.labels) == (b.input_ids, b.labels)
