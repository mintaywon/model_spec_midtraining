"""Tests for the scorer-agreement gate.

The gate exists because a sign-convention misread already turned a +0.411
correlation into -0.411 and looked like a real methodological disagreement.
"""

import numpy as np
import pytest

from tda.analysis.method_compare import compare, load_source


def _write(tmp_path, v):
    p = tmp_path / "s.npy"
    np.save(p, v)
    return p


def _synthetic(n=6400, seed=0):
    """An array matching the published run statistics closely enough to pass."""
    rng = np.random.default_rng(seed)
    v = rng.normal(0.0645, 0.176, n)
    # nudge frac_positive toward the published 0.644
    while abs((v > 0).mean() - 0.644) > 0.005:
        v += 0.002 if (v > 0).mean() < 0.644 else -0.002
    return v


def test_correctly_oriented_array_loads(tmp_path):
    v = _synthetic()
    assert load_source(_write(tmp_path, v)).shape == (6400,)


def test_sign_inverted_array_is_REFUSED_not_silently_compared(tmp_path):
    """THE test. An inverted array must raise, not return a plausible ranking.

    Comparing it as-is yields a confidently wrong correlation of the right
    magnitude and the wrong sign -- indistinguishable from a real finding.
    """
    v = _synthetic()
    with pytest.raises(RuntimeError, match="SIGN-INVERTED"):
        load_source(_write(tmp_path, -v))


def test_wrong_document_count_is_refused(tmp_path):
    with pytest.raises(RuntimeError, match="expected 6400"):
        load_source(_write(tmp_path, _synthetic(n=1000)))


def test_unrecognised_statistics_are_refused(tmp_path):
    """Not inverted, just wrong -- wrong file or the run changed."""
    with pytest.raises(RuntimeError, match="does not match"):
        load_source(_write(tmp_path, np.full(6400, 5.0)))


def test_compare_refuses_mismatched_coverage():
    with pytest.raises(RuntimeError, match="different document counts"):
        compare({"a": np.arange(10.0), "b": np.arange(9.0)})


def test_identical_scorers_agree_perfectly():
    v = np.random.default_rng(1).normal(size=500)
    r = compare({"a": v, "b": v.copy()}, ks=(50,))
    p = r["pairs"]["a vs b"]
    assert p["spearman"] == pytest.approx(1.0)
    assert p["jaccard"]["50"] == pytest.approx(1.0)


def test_negated_scorer_anticorrelates():
    """Guards the direction of the statistic itself."""
    v = np.random.default_rng(2).normal(size=500)
    r = compare({"a": v, "b": -v}, ks=(50,))
    assert r["pairs"]["a vs b"]["spearman"] == pytest.approx(-1.0)
