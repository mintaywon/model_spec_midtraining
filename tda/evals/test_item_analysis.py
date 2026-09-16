import numpy as np
import pytest

from tda.evals.item_analysis import classify, item_matrix, items_needed, snr_check


def _synthetic(n_arms=20, n_dead=145, n_disc=38, seed=0):
    """Arms that agree on `n_dead` items and disagree on the rest."""
    rng = np.random.default_rng(seed)
    dead = np.tile(np.r_[np.ones(92), np.zeros(n_dead - 92)], (n_arms, 1))
    live = rng.integers(0, 2, size=(n_arms, n_disc))
    return np.hstack([dead, live]).astype(int)


def test_classify_finds_the_dead_items():
    M = _synthetic()
    c = classify(M)
    assert c["n_always"] == 92
    assert c["n_never"] == 145 - 92
    assert c["n_discriminating"] == 38
    # dead items carry no variance at all
    assert c["variance_share_discriminating"] == pytest.approx(1.0)


def test_subsetting_to_discriminating_items_does_not_change_z():
    """The load-bearing claim: signal and noise scale together, SNR is ~1."""
    M = _synthetic(n_arms=24, seed=3)
    r = snr_check(M, control_rows=list(range(12)), arm_row=20)
    assert abs(r["snr_ratio"] - 1.0) < 0.10
    assert abs(r["z_full"] - r["z_disc"]) < 0.15
    # and the effect really is inflated by the subset, which is the trap
    assert abs(r["effect_disc"]) > 2 * abs(r["effect_full"])


def test_items_needed_scales_as_inverse_variance():
    assert items_needed(0.038, 38, 0.019) == 152
    assert items_needed(0.038, 38, 0.038) == 38


def test_item_matrix_rejects_ragged_arms():
    with pytest.raises(ValueError, match="disagree on item count"):
        item_matrix({"a": [1, 0, 1], "b": [1, 0]})
