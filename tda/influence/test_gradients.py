"""Correctness tests for per-sample LoRA gradient extraction.

The load-bearing test is `test_matches_autograd`: the hook-reconstructed
gradient must equal what autograd computes. Everything downstream (projection,
influence scores, H1 rank correlations) is meaningless if this is wrong, and a
subtly wrong gradient produces confident, plausible, wrong numbers — there is no
downstream symptom to catch it.

Runs on CPU with a tiny model so it is fast and hardware-independent.
"""

import pytest
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from tda.influence.gradients import LoRAGradientCapture, find_lora_modules


class TinyNet(nn.Module):
    """Two projections, enough to exercise multi-module ordering."""

    def __init__(self, d_in=8, d_hidden=6, d_out=4):
        super().__init__()
        self.up = nn.Linear(d_in, d_hidden, bias=False)
        self.down = nn.Linear(d_hidden, d_out, bias=False)

    def forward(self, x):
        return self.down(torch.relu(self.up(x)))


def _peft_model(seed=0, r=2, alpha=4):
    """A LoRA model with lora_B PERTURBED AWAY FROM ZERO.

    ⚠️ This is not cosmetic. PEFT initialises lora_B to zero, and
    grad_A = (B^T g) x^T is therefore identically zero at init — so any test of
    the A path against autograd compares 0 to 0 and passes vacuously. That
    exact bug hid a real scaling error in grad_B. Every test here must run on a
    model whose B is non-zero.
    """
    torch.manual_seed(seed)
    cfg = LoraConfig(r=r, lora_alpha=alpha, target_modules=["up", "down"],
                     lora_dropout=0.0, bias="none")
    model = get_peft_model(TinyNet(), cfg)
    with torch.no_grad():
        for ref in find_lora_modules(model):
            ref.lora_B.weight.normal_(0.0, 0.5)
    return model


@pytest.fixture
def model():
    return _peft_model()


def test_fixture_has_nonzero_B(model):
    """Guard the guard: if B were zero, the A-path tests would be vacuous."""
    for ref in find_lora_modules(model):
        assert ref.lora_B.weight.abs().max() > 1e-6, "lora_B must not be zero"


def test_finds_modules_in_deterministic_order(model):
    refs = find_lora_modules(model)
    assert len(refs) == 2
    names = [r.name for r in refs]
    assert names == sorted(names), "order must be deterministic (fixes vector layout)"
    # find_lora_modules must be stable across calls
    assert names == [r.name for r in find_lora_modules(model)]


def test_scaling_is_alpha_over_r(model):
    for ref in find_lora_modules(model):
        assert ref.scaling == pytest.approx(4 / 2)


def test_shapes_are_consistent(model):
    for ref in find_lora_modules(model):
        assert ref.lora_A.weight.shape == (ref.r, ref.d_in)
        assert ref.lora_B.weight.shape == (ref.d_out, ref.r)
        assert ref.n_params == ref.r * ref.d_in + ref.d_out * ref.r


def test_matches_autograd(model):
    """Hook-reconstructed per-sample gradient == autograd gradient.

    Single-sample batch so autograd's summed gradient IS the per-sample one.
    """
    refs = find_lora_modules(model)
    torch.manual_seed(1)
    x = torch.randn(1, 5, 8)          # (batch, seq, d_in)

    model.zero_grad()
    with LoRAGradientCapture(refs) as cap:
        loss = model(x).pow(2).sum()
        loss.backward()
        recon = cap.per_sample_grads(0)

    for ref in refs:
        gA_auto = ref.lora_A.weight.grad
        gB_auto = ref.lora_B.weight.grad
        gA_hook, gB_hook = recon[ref.name]

        # Non-vacuity: both autograd grads must be non-trivial, else the
        # comparison below proves nothing (see _peft_model docstring).
        assert gA_auto.abs().max() > 1e-6, "grad_A is zero - test would be vacuous"
        assert gB_auto.abs().max() > 1e-6, "grad_B is zero - test would be vacuous"

        torch.testing.assert_close(gA_hook, gA_auto.float(), rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(gB_hook, gB_auto.float(), rtol=1e-4, atol=1e-5)


def test_per_sample_split_is_correct(model):
    """Per-sample grads from ONE batched backward must equal per-sample backwards.

    This is the property that makes batching sound: if it fails, every influence
    score silently mixes samples together.
    """
    refs = find_lora_modules(model)
    torch.manual_seed(2)
    x = torch.randn(3, 4, 8)

    # Batched: capture all three at once.
    model.zero_grad()
    with LoRAGradientCapture(refs) as cap:
        # No mean over batch — per-sample losses summed, so each sample's
        # gradient contribution is independent.
        model(x).pow(2).sum().backward()
        batched = [cap.flat_per_sample_grad(i) for i in range(3)]

    # Individually: one backward per sample.
    for i in range(3):
        model.zero_grad()
        with LoRAGradientCapture(refs) as cap_i:
            model(x[i : i + 1]).pow(2).sum().backward()
            solo = cap_i.flat_per_sample_grad(0)
        torch.testing.assert_close(batched[i], solo, rtol=1e-4, atol=1e-5)


def test_masked_tokens_contribute_nothing(model):
    """Tokens excluded from the loss must not enter the gradient.

    Directly checks the masking contract that tda/influence/masking.py encodes:
    influence must reflect only supervised positions.
    """
    refs = find_lora_modules(model)
    torch.manual_seed(3)
    x = torch.randn(1, 6, 8)

    # Loss on the first 3 positions only.
    model.zero_grad()
    with LoRAGradientCapture(refs) as cap:
        model(x)[:, :3].pow(2).sum().backward()
        masked = cap.flat_per_sample_grad(0)

    # Same, but feeding only those positions.
    model.zero_grad()
    with LoRAGradientCapture(refs) as cap2:
        model(x[:, :3]).pow(2).sum().backward()
        truncated = cap2.flat_per_sample_grad(0)

    torch.testing.assert_close(masked, truncated, rtol=1e-4, atol=1e-5)


def test_flat_grad_length_matches_param_count(model):
    refs = find_lora_modules(model)
    torch.manual_seed(4)
    with LoRAGradientCapture(refs) as cap:
        model(torch.randn(1, 3, 8)).sum().backward()
        flat = cap.flat_per_sample_grad(0)
    assert flat.numel() == sum(r.n_params for r in refs)


def test_raises_when_no_adapter_present():
    with pytest.raises(RuntimeError, match="No LoRA modules found"):
        find_lora_modules(TinyNet())


def test_keep_positions_is_exact_not_approximate(model):
    """Restricting captures to supervised positions must not change the gradient.

    Masked positions carry grad_output = 0, so dropping them is exact. This is
    what makes 32B queries fit in memory (50GB of captures -> ~4GB), so it must
    be verified rather than assumed.
    """
    refs = find_lora_modules(model)
    torch.manual_seed(21)
    x = torch.randn(1, 8, 8)
    keep = torch.tensor([2, 3, 4])          # supervise positions 2..4 only

    # Full capture, loss on the supervised positions only.
    model.zero_grad()
    with LoRAGradientCapture(refs) as cap_full:
        model(x)[:, keep].pow(2).sum().backward()
        full = cap_full.flat_per_sample_grad(0)

    # Sliced capture, identical loss.
    model.zero_grad()
    with LoRAGradientCapture(refs, keep_positions=keep) as cap_kept:
        model(x)[:, keep].pow(2).sum().backward()
        kept = cap_kept.flat_per_sample_grad(0)

    assert full.abs().max() > 1e-6, "gradient is zero - test would be vacuous"
    torch.testing.assert_close(kept, full, rtol=1e-4, atol=1e-5)


def test_keep_positions_shrinks_captures(model):
    refs = find_lora_modules(model)
    torch.manual_seed(22)
    x = torch.randn(1, 10, 8)
    keep = torch.tensor([0, 1])

    with LoRAGradientCapture(refs, keep_positions=keep) as cap:
        model(x)[:, keep].pow(2).sum().backward()
        for ref in refs:
            assert cap.cap.a_in[ref.name].shape[1] == 2
            assert cap.cap.b_gout[ref.name].shape[1] == 2
