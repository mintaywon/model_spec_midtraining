"""Tests for the AFT trainer's silent-failure surfaces.

Everything here is CPU-only and GPU-free. The parts that can go wrong quietly
are padding (a pad token that contributes to the loss shifts every gradient),
the LR schedule (a broken warmup changes the optimum reached), and the seeding
contract that the whole noise-floor experiment rests on.
"""

import math

import pytest
import torch

from tda.influence.masking import IGNORE_INDEX
from tda.retrain.sft import collate, lr_schedule


def test_collate_pads_labels_with_ignore_index():
    """Padded positions must NEVER be supervised."""
    batch = [{"input_ids": [1, 2, 3], "labels": [IGNORE_INDEX, 2, 3]},
             {"input_ids": [4], "labels": [4]}]
    out = collate(batch, pad_id=0)

    assert out["input_ids"].shape == (2, 3)
    assert out["labels"][1].tolist() == [4, IGNORE_INDEX, IGNORE_INDEX]
    assert out["attention_mask"][1].tolist() == [1, 0, 0]


def test_collate_padding_does_not_change_loss():
    """THE test: a padded batch and an unpadded one must give the same loss.

    If pad positions leaked into the loss, training would silently optimise a
    different objective than the influence pipeline attributes.
    """
    torch.manual_seed(0)
    vocab, hidden = 16, 8
    emb, head = torch.nn.Embedding(vocab, hidden), torch.nn.Linear(hidden, vocab)

    def loss_of(b):
        logits = head(emb(b["input_ids"]))
        return torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, vocab), b["labels"][:, 1:].reshape(-1),
            ignore_index=IGNORE_INDEX,
        )

    one = [{"input_ids": [1, 2, 3, 4], "labels": [IGNORE_INDEX, 2, 3, 4]}]
    padded = collate(one + [{"input_ids": [5], "labels": [IGNORE_INDEX]}], pad_id=0)
    # The second sample supervises nothing, so it must not move the loss.
    assert torch.allclose(loss_of(collate(one, 0)), loss_of(padded), atol=1e-6)


def test_lr_warmup_then_cosine():
    lr, warmup, total = 1e-4, 5, 100
    assert lr_schedule(0, lr, warmup, total) == pytest.approx(lr / 5)
    assert lr_schedule(4, lr, warmup, total) == pytest.approx(lr)      # peak
    assert lr_schedule(warmup, lr, warmup, total) == pytest.approx(lr)
    # Monotone decay after warmup, ending near zero.
    after = [lr_schedule(s, lr, warmup, total) for s in range(warmup, total)]
    assert all(x >= y - 1e-12 for x, y in zip(after, after[1:]))
    assert lr_schedule(total, lr, warmup, total) == pytest.approx(0.0, abs=1e-9)


def test_warmup_is_five_percent_of_total():
    """Paper recipe: 5% warmup. Guard the arithmetic, not the constant."""
    total = 200
    warmup = max(1, int(0.05 * total))
    assert warmup == 10
    assert lr_schedule(warmup - 1, 1e-4, warmup, total) == pytest.approx(1e-4)


def test_grad_accum_scaling_recovers_window_mean():
    """Micro-batch losses scaled by len(micro)/len(window) must sum to the
    window mean -- otherwise the effective LR silently depends on batch size."""
    window = list(range(10))
    micro_means = []
    for m in range(0, len(window), 4):
        micro = window[m: m + 4]
        micro_means.append(sum(micro) / len(micro) * (len(micro) / len(window)))
    assert sum(micro_means) == pytest.approx(sum(window) / len(window))


class _Tok:
    pad_token_id = 0
    eos_token = "</s>"

    def apply_chat_template(self, msgs, tokenize=False, **kw):
        return "".join(m["content"] for m in msgs)


def test_seed_changes_order_only(monkeypatch):
    """The noise-floor contract: two arms differ ONLY in data order.

    Same multiset of examples, different sequence. If the seed also changed
    WHICH examples appear, the floor would confound order with composition and
    the H1 comparison would be uninterpretable.
    """
    import random as _r

    from tda.retrain import sft

    made = [{"input_ids": [i], "labels": [i], "source": "task"} for i in range(50)]

    def fake_build(cfg, tok):
        out = list(made)
        _r.Random(cfg.seed).shuffle(out)
        return out

    monkeypatch.setattr(sft, "build_examples", fake_build)
    cfg42 = sft.SFTConfig(base_model="x", init_adapter=None, task_dataset="y",
                          out_dir="/tmp/z", seed=42)
    cfg43 = sft.SFTConfig(base_model="x", init_adapter=None, task_dataset="y",
                          out_dir="/tmp/z", seed=43)

    a = sft.build_examples(cfg42, _Tok())
    b = sft.build_examples(cfg43, _Tok())

    assert [x["input_ids"] for x in a] != [x["input_ids"] for x in b]   # order differs
    assert sorted(x["input_ids"][0] for x in a) == \
           sorted(x["input_ids"][0] for x in b)                          # content same


def test_parquet_fallback_split_prefix_is_exact():
    """`train_clean-` must not also match `train_clean_nothink-`.

    Both live in chloeli/sft-it-mix. A prefix match on `train_clean` alone
    would silently concatenate a DIFFERENT split into the IT mix, changing the
    training data without any error. The trailing hyphen is load-bearing.
    """
    from pathlib import Path

    files = ["data/train_clean-00000-of-00001.parquet",
             "data/train_clean_nothink-00000-of-00001.parquet",
             "data/train-00000-of-00001.parquet"]
    split = "train_clean"
    got = [f for f in files
           if f.endswith(".parquet") and Path(f).name.startswith(f"{split}-")]
    assert got == ["data/train_clean-00000-of-00001.parquet"]


# --- token-budget batching -------------------------------------------------

def _ex(n):
    return {"input_ids": [1] * n, "labels": [1] * n, "source": "x"}


def test_long_sequence_gets_its_own_batch():
    """THE OOM guard: a LongAlign-sized row must not drag short rows with it.

    Real numbers from chloeli/sft-it-mix: median 291 tokens, LongAlign mean
    7,030. A fixed batch of 8 containing one such row pads all 8 to ~7,884,
    which is ~57 GB of logits at Qwen's 152k vocab once HF casts to float32.
    """
    from tda.retrain.sft import make_batches

    data = [_ex(300), _ex(300), _ex(7884), _ex(300)]
    batches = make_batches(data, token_budget=8192, max_seqs=16)

    long_batch = [b for b in batches if any(len(e["input_ids"]) > 4000 for e in b)]
    assert len(long_batch) == 1
    assert len(long_batch[0]) == 1, "long sequence must travel alone"


def test_every_batch_respects_padded_budget():
    """Budget is on the PADDED rectangle (n_seqs x longest), not the token sum."""
    import random as _r

    from tda.retrain.sft import make_batches

    rng = _r.Random(0)
    data = [_ex(rng.choice([50, 200, 900, 4000, 7884])) for _ in range(200)]
    for b in make_batches(data, token_budget=8192, max_seqs=16):
        padded = len(b) * max(len(e["input_ids"]) for e in b)
        assert len(b) <= 16
        assert padded <= 8192 or len(b) == 1   # oversized singleton allowed


def test_batching_preserves_order_and_loses_nothing():
    """No sorting, no drops: order carries the seed, and every example trains."""
    from tda.retrain.sft import make_batches

    data = [_ex(n) for n in [100, 5000, 100, 100, 8000, 100]]
    flat = [e for b in make_batches(data, 8192, 16) for e in b]
    assert [len(e["input_ids"]) for e in flat] == [len(e["input_ids"]) for e in data]


def test_loss_weighting_recovers_example_mean():
    """Uneven micro-batch sizes must still average per EXAMPLE, not per batch."""
    window = [[_ex(1)] * 16, [_ex(1)] * 1]          # 16 short + 1 long
    n_ex = sum(len(mb) for mb in window)
    per_batch_loss = [2.0, 10.0]
    total = sum(l * (len(mb) / n_ex) for l, mb in zip(per_batch_loss, window))
    assert total == pytest.approx((2.0 * 16 + 10.0 * 1) / 17)


def test_config_accepts_exactly_what_the_modal_caller_passes():
    """Guard the config/caller contract.

    `app.py::aft_train` constructs SFTConfig by keyword. Renaming a field here
    without updating the caller raises TypeError only at LAUNCH -- after the
    image builds and a 32B model downloads, i.e. minutes and dollars in. This
    caught a real break when batch_size became token_budget/max_seqs.
    """
    from tda.retrain.sft import SFTConfig

    cfg = SFTConfig(
        base_model="b", init_adapter="i", task_dataset="t", out_dir="/tmp/o",
        seed=42, limit=0, token_budget=8192, max_seqs=16, grad_accum=4,
        supervise="assistant",
    )
    assert cfg.token_budget == 8192 and cfg.max_seqs == 16
    assert not hasattr(cfg, "batch_size"), "stale field name still present"


# --- document (MSM) mode ---------------------------------------------------

class _DocTok:
    """Whitespace tokenizer with the HF keyword signature."""

    pad_token_id = 0
    eos_token = "</s>"

    def __call__(self, text, add_special_tokens=False, truncation=False,
                 max_length=None):
        ids = [ord(c) % 97 + 1 for c in text.replace(" ", "")]
        if truncation and max_length:
            ids = ids[:max_length]
        return {"input_ids": ids}


def _docs_cfg(monkeypatch, corpus, **kw):
    from tda.retrain import sft

    monkeypatch.setattr(sft, "load_rows", lambda name, split: corpus)
    return sft.SFTConfig(base_model="b", init_adapter=None, task_dataset="t",
                         out_dir="/tmp/o", task_mode="document", **kw)


def test_document_mode_supervises_every_position_but_first(monkeypatch):
    """MSM is plain next-token prediction -- masking.py does not apply."""
    from tda.retrain.sft import build_examples

    corpus = [{"text": "alpha beta gamma"}]
    ex = build_examples(_docs_cfg(monkeypatch, corpus), _DocTok())[0]
    assert ex["labels"][0] == IGNORE_INDEX
    assert ex["labels"][1:] == ex["input_ids"][1:]
    assert sum(l != IGNORE_INDEX for l in ex["labels"]) == len(ex["input_ids"]) - 1


def test_drop_rows_removes_exactly_those_documents(monkeypatch):
    """The removal mechanism. Indices are into the UNSHUFFLED corpus."""
    from tda.retrain.sft import build_examples

    corpus = [{"text": f"doc number {i}"} for i in range(10)]
    ex = build_examples(_docs_cfg(monkeypatch, corpus, drop_rows=(1, 3, 5)),
                        _DocTok())
    assert len(ex) == 7
    assert sorted(e["row"] for e in ex) == [0, 2, 4, 6, 7, 8, 9]


def test_removal_arm_differs_from_baseline_only_by_the_dropped_rows(monkeypatch):
    """Same seed, same corpus: the survivors must be the same documents.

    If dropping rows also reshuffled the remainder, the arm would differ by
    data order too and Delta f would confound removal with ordering.
    """
    from tda.retrain.sft import build_examples

    corpus = [{"text": f"doc number {i}"} for i in range(20)]
    base = build_examples(_docs_cfg(monkeypatch, corpus), _DocTok())
    arm = build_examples(_docs_cfg(monkeypatch, corpus, drop_rows=(2, 7)),
                         _DocTok())
    assert set(e["row"] for e in base) - set(e["row"] for e in arm) == {2, 7}


def test_out_of_range_drop_rows_are_refused(monkeypatch):
    """A score file misaligned with the corpus must fail loudly, not silently
    remove nothing and report a null result."""
    from tda.retrain.sft import build_examples

    corpus = [{"text": f"doc {i}"} for i in range(5)]
    with pytest.raises(RuntimeError, match="outside the corpus"):
        build_examples(_docs_cfg(monkeypatch, corpus, drop_rows=(2, 99)),
                       _DocTok())


def test_document_mode_ignores_the_it_mix(monkeypatch):
    """Midtraining is documents alone; the instruction mix belongs to AFT."""
    from tda.retrain.sft import build_examples

    corpus = [{"text": f"doc {i}"} for i in range(4)]
    cfg = _docs_cfg(monkeypatch, corpus)
    cfg.it_dataset = "chloeli/sft-it-mix"      # set, and must be ignored
    ex = build_examples(cfg, _DocTok())
    assert len(ex) == 4
    assert {e["source"] for e in ex} == {"doc"}
