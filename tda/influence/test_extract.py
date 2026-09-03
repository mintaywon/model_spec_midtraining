"""Tests for the masked-loss path used by gradient extraction.

`masked_loss` computes logits only over supervised positions, because full-
sequence logits at a 152k vocab OOM a 2xH100. It must be EXACT — an off-by-one
in the shift would silently attribute the wrong tokens, and every influence
score downstream would be wrong with no visible symptom.
"""

import pytest
import torch
import torch.nn as nn

from tda.influence.extract import masked_loss
from tda.influence.masking import IGNORE_INDEX


class TinyLM(nn.Module):
    """Minimal causal LM exposing the `logits_to_keep` contract."""

    def __init__(self, vocab=32, hidden=16):
        super().__init__()
        self.emb = nn.Embedding(vocab, hidden)
        self.ln = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, vocab)

    def forward(self, input_ids, logits_to_keep=None, labels=None):
        h = torch.tanh(self.ln(self.emb(input_ids)))
        if logits_to_keep:
            h = h[:, -logits_to_keep:, :]
        logits = self.head(h)

        loss = None
        if labels is not None:                       # HF-style full-sequence loss
            shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
            shift_labels = labels[:, 1:].reshape(-1)
            loss = torch.nn.functional.cross_entropy(
                shift_logits.float(), shift_labels, ignore_index=IGNORE_INDEX
            )
        return type("Out", (), {"logits": logits, "loss": loss})()


@pytest.fixture
def lm():
    torch.manual_seed(0)
    return TinyLM()


def test_matches_full_sequence_loss(lm):
    """THE test: the memory optimisation must not change the loss."""
    torch.manual_seed(1)
    ids = torch.randint(0, 32, (1, 12))
    labels = [IGNORE_INDEX] * 8 + ids[0, 8:].tolist()   # supervise the tail

    ours = masked_loss(lm, ids, labels)
    hf = lm(input_ids=ids, labels=torch.tensor([labels])).loss

    assert ours.item() > 0
    torch.testing.assert_close(ours, hf, rtol=1e-5, atol=1e-6)


def test_gradients_match_full_sequence(lm):
    """Loss parity is not enough — the GRADIENT is what influence uses."""
    torch.manual_seed(2)
    ids = torch.randint(0, 32, (1, 10))
    labels = [IGNORE_INDEX] * 6 + ids[0, 6:].tolist()

    lm.zero_grad()
    masked_loss(lm, ids, labels).backward()
    ours = lm.head.weight.grad.clone()

    lm.zero_grad()
    lm(input_ids=ids, labels=torch.tensor([labels])).loss.backward()
    hf = lm.head.weight.grad.clone()

    assert ours.abs().max() > 1e-8, "zero gradient - test would be vacuous"
    torch.testing.assert_close(ours, hf, rtol=1e-4, atol=1e-6)


def test_only_tail_logits_computed(lm):
    """Confirms the memory saving actually happens."""
    torch.manual_seed(3)
    ids = torch.randint(0, 32, (1, 100))
    labels = [IGNORE_INDEX] * 95 + ids[0, 95:].tolist()

    captured = {}
    orig = lm.forward

    def spy(input_ids, logits_to_keep=None, labels=None):
        captured["keep"] = logits_to_keep
        return orig(input_ids, logits_to_keep=logits_to_keep, labels=labels)

    lm.forward = spy
    masked_loss(lm, ids, labels)
    assert captured["keep"] == 6, "should keep only first-1..L-1 (5 targets + 1)"
    assert captured["keep"] < 100 / 10, "must be a large reduction"


def test_single_supervised_token(lm):
    """Edge case: a one-token span must not break the shift."""
    torch.manual_seed(4)
    ids = torch.randint(0, 32, (1, 8))
    labels = [IGNORE_INDEX] * 7 + [ids[0, 7].item()]

    ours = masked_loss(lm, ids, labels)
    hf = lm(input_ids=ids, labels=torch.tensor([labels])).loss
    torch.testing.assert_close(ours, hf, rtol=1e-5, atol=1e-6)


def test_falls_back_to_legacy_kwarg():
    """transformers < 4.50 spells it num_logits_to_keep."""

    class LegacyLM(TinyLM):
        def forward(self, input_ids, num_logits_to_keep=None, labels=None):
            return super().forward(input_ids, logits_to_keep=num_logits_to_keep,
                                   labels=labels)

    torch.manual_seed(5)
    lm = LegacyLM()
    ids = torch.randint(0, 32, (1, 9))
    labels = [IGNORE_INDEX] * 5 + ids[0, 5:].tolist()
    assert masked_loss(lm, ids, labels).item() > 0


def test_document_convention_equals_plain_lm_loss(lm):
    """MSM documents: `labels = [IGNORE] + ids[1:]` must equal vanilla LM loss.

    `extract_documents` supervises every position after the first, which is what
    "plain next-token prediction, just like pre-training data" means. The
    reference is HF's own full-sequence loss with labels == input_ids: HF shifts
    internally, so position 0 is never a target there either. If these diverge,
    every MSM-document gradient is computed against the wrong objective.
    """
    torch.manual_seed(3)
    ids = torch.randint(0, 32, (1, 24))
    labels = [IGNORE_INDEX] + ids[0, 1:].tolist()

    ours = masked_loss(lm, ids, labels)
    ref = lm(input_ids=ids, labels=ids).loss

    assert torch.allclose(ours, ref, atol=1e-5), f"{ours.item()} vs {ref.item()}"


def test_document_convention_supervises_every_position_but_first(lm):
    """Guard the count: n_tokens-1 targets, not n_tokens and not fewer."""
    torch.manual_seed(4)
    ids = torch.randint(0, 32, (1, 16))
    labels = [IGNORE_INDEX] + ids[0, 1:].tolist()

    assert sum(l != IGNORE_INDEX for l in labels) == ids.shape[1] - 1
    # first supervised index is 1 -> masked_loss keeps ALL logits for documents,
    # which is why extract_documents must truncate for memory.
    assert next(i for i, l in enumerate(labels) if l != IGNORE_INDEX) == 1
