"""Unit tests for the pieces of tda/hyp where a silent error changes a result."""

import numpy as np

from tda.hyp.common import last_json, tagged
from tda.hyp.rank import residualise
from tda.hyp.verify import (auc, lexical_density, ols, parse_annotation,
                            weighted_kappa)

HYPS = [{"id": "H1"}, {"id": "H2"}]


def test_last_json_nested_and_required():
    t = 'thinking {"x": 1}\n{"H1": {"score": 2, "evidence": "a {b}"}, "H2": {"score": 0, "evidence": ""}}'
    d = last_json(t, required=("H1", "H2"))
    assert d["H1"]["score"] == 2 and d["H2"]["score"] == 0


def test_last_json_missing_key_is_none():
    assert last_json('{"H1": {"score": 1}}', required=("H1", "H2")) is None


def test_parse_annotation_rejects_out_of_range():
    assert parse_annotation('{"H1": {"score": 4}, "H2": {"score": 0}}', HYPS) is None
    ok = parse_annotation('{"H1": {"score": 3, "evidence": "q"}, "H2": 1}', HYPS)
    assert ok == {"H1": {"score": 3, "evidence": "q"}, "H2": {"score": 1, "evidence": ""}}


def test_tagged():
    assert tagged("a <x>\n[1]\n</x> b", "x") == "[1]"
    assert tagged("none", "x") is None


def test_kappa_perfect_and_chance():
    a = np.array([0, 1, 2, 3] * 25)
    assert weighted_kappa(a, a) == 1.0
    rng = np.random.default_rng(0)
    assert abs(weighted_kappa(a, rng.permutation(a))) < 0.2


def test_auc_direction():
    # auc(pos, neg) is P(pos > neg): proponents higher on the feature -> > 0.5
    assert auc(np.array([3, 3, 2.0]), np.array([0, 1, 0.0])) == 1.0
    assert auc(np.array([0, 0.0]), np.array([3, 3.0])) == 0.0
    assert auc(np.array([1, 1.0]), np.array([1, 1.0])) == 0.5


def test_ols_recovers_sign_under_a_confound():
    rng = np.random.default_rng(1)
    n = 4000
    length = rng.normal(size=n)
    feat = 0.8 * length + rng.normal(size=n)          # feature correlated with length
    y = 1.0 * length - 0.3 * feat + rng.normal(size=n)
    b_raw = ols(y, feat[:, None])[0][1]
    b_ctl = ols(y, np.column_stack([feat, length]))[0][1]
    assert b_raw > 0            # raw association has the WRONG sign
    assert abs(b_ctl + 0.3) < 0.05


def test_residualise_removes_linear_part():
    rng = np.random.default_rng(2)
    x = rng.normal(size=500)
    r = residualise(3 * x + rng.normal(size=500), x[:, None])
    assert abs(np.corrcoef(r, x)[0, 1]) < 1e-8


def test_lexical_density():
    v = {"server", "board"}
    assert lexical_density("The server board meeting", v) == 2 / 3
    assert lexical_density("", v) == 0
