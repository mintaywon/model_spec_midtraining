"""Tests for influence scoring and the H1/H2 analysis statistics.

These statistics ARE the results — a wrong Spearman or Gini produces a
publishable-looking number that is simply false, with nothing downstream to
catch it. Each is checked against a case with a known answer.
"""

import numpy as np
import pytest

from tda.influence.scoring import (
    GradDot,
    aggregate_over_queries,
    gini,
    length_normalize,
    norm_confound_report,
    spearman,
    topk_jaccard,
    topk_mass,
)


def test_graddot_is_inner_product():
    train = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    query = np.array([[1.0, 0.0]])
    s = GradDot().score(train, query)
    np.testing.assert_allclose(s[:, 0], [1.0, 0.0, 1.0])


def test_graddot_damping_shrinks_uniformly():
    train = np.random.RandomState(0).randn(5, 4)
    query = np.random.RandomState(1).randn(2, 4)
    a = GradDot(damping=0.0).score(train, query)
    b = GradDot(damping=1.0).score(train, query)
    np.testing.assert_allclose(b, a / 2.0)


def test_graddot_query_normalization():
    train = np.array([[1.0, 0.0]])
    query = np.array([[3.0, 0.0]])
    assert GradDot().score(train, query)[0, 0] == pytest.approx(3.0)
    assert GradDot(normalize_query=True).score(train, query)[0, 0] == pytest.approx(1.0)


def test_length_normalize_divides_by_token_count():
    scores = np.array([[10.0], [10.0]])
    out = length_normalize(scores, np.array([10, 2]))
    np.testing.assert_allclose(out[:, 0], [1.0, 5.0])


def test_length_normalize_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        length_normalize(np.zeros((3, 1)), np.array([1, 2]))


def test_aggregate_modes():
    s = np.array([[1.0, 5.0], [3.0, 3.0]])
    np.testing.assert_allclose(aggregate_over_queries(s, "mean"), [3.0, 3.0])
    np.testing.assert_allclose(aggregate_over_queries(s, "max"), [5.0, 3.0])
    np.testing.assert_allclose(aggregate_over_queries(s, "sum"), [6.0, 6.0])
    with pytest.raises(ValueError):
        aggregate_over_queries(s, "median")


def test_spearman_known_values():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert spearman(x, x) == pytest.approx(1.0)
    assert spearman(x, -x) == pytest.approx(-1.0)
    # Monotone transform must not change rank correlation.
    assert spearman(x, np.exp(x)) == pytest.approx(1.0)


def test_spearman_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        spearman(np.zeros(3), np.zeros(4))


def test_topk_jaccard_bounds():
    a = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    assert topk_jaccard(a, a, k=2) == pytest.approx(1.0)
    # Exactly reversed: top-2 of each are disjoint.
    assert topk_jaccard(a, -a, k=2) == pytest.approx(0.0)


def test_topk_jaccard_partial_overlap():
    a = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    b = np.array([5.0, 1.0, 4.0, 2.0, 3.0])
    # top-2: {0,1} vs {0,2} -> intersection 1, union 3
    assert topk_jaccard(a, b, k=2) == pytest.approx(1 / 3)


def test_gini_extremes():
    # Perfectly equal -> 0
    assert gini(np.ones(100)) == pytest.approx(0.0, abs=1e-9)
    # All mass on one sample -> approaches 1
    spike = np.zeros(1000)
    spike[0] = 1.0
    assert gini(spike) > 0.99


def test_gini_uses_absolute_value():
    """Negative influence is still influence — sign must not cancel magnitude."""
    assert gini(np.array([-5.0, 5.0])) == pytest.approx(gini(np.array([5.0, 5.0])))


def test_topk_mass_concentration():
    # Uniform: the top 10% holds ~10% of the mass.
    uniform = np.ones(1000)
    assert topk_mass(uniform)[0.1] == pytest.approx(0.1, abs=0.01)

    # Concentrated: one sample holds nearly everything.
    conc = np.zeros(1000)
    conc[0] = 1000.0
    assert topk_mass(conc)[0.01] > 0.99


def test_scorer_protocol_shape_contract():
    """Any Scorer must return (n_train, n_query) — EK-FAC will rely on this."""
    train = np.random.RandomState(0).randn(7, 5)
    query = np.random.RandomState(1).randn(3, 5)
    assert GradDot().score(train, query).shape == (7, 3)


def test_normalize_train_removes_magnitude_effect():
    """Cosine-style scoring must rank by direction, not gradient norm.

    Reproduces the confound found end-to-end: a badly-aligned sample with a huge
    gradient beats a well-aligned small one under raw dot product.
    """
    query = np.array([[1.0, 0.0]])
    aligned_small = np.array([1.0, 0.0]) * 1.0        # perfect direction, tiny
    misaligned_big = np.array([0.6, 0.8]) * 100.0     # off direction, huge
    train = np.stack([aligned_small, misaligned_big])

    raw = GradDot().score(train, query)[:, 0]
    assert raw[1] > raw[0], "raw dot is expected to be magnitude-dominated"

    cos = GradDot(normalize_train=True).score(train, query)[:, 0]
    assert cos[0] > cos[1], "normalized scoring must recover the aligned sample"


def test_norm_confound_report_flags_magnitude_driven_scores():
    rng = np.random.RandomState(0)
    train = rng.randn(50, 8) * rng.uniform(1, 50, size=(50, 1))  # wide norm spread
    query = rng.randn(3, 8)
    scores = GradDot().score(train, query)

    rep = norm_confound_report(scores, train)
    assert rep["corr_with_grad_norm"] > 0.5, "should detect magnitude dominance"
    assert rep["norm_ratio_p90_p10"] > 1.0


def _write_scores(d, vals):
    import json

    import numpy as np

    d.mkdir(parents=True, exist_ok=True)
    arr = np.zeros(len(vals), dtype=[("score_0", "<f4"), ("written_0", "?")])
    arr["score_0"] = vals
    arr["written_0"] = True
    arr.tofile(d / "scores.bin")
    (d / "info.json").write_text(json.dumps({
        "num_scores": 1, "num_rows": len(vals), "num_items": len(vals),
        "dtype": [["score_0", "<f4"], ["written_0", "|b1"]]}))
    # The pipeline writes higher_is_better: true per checkpoint, which means
    # the stored values are NEGATED on read.
    (d / "score_cfg.yaml").write_text("higher_is_better: true\n")


def test_stage_masked_score_matches_bergson_aggregation(tmp_path):
    """Segment score = MEAN over its checkpoints (negated); stage = SUM of those.

    Mirrors approx_unrolling_math.score_per_segment_and_aggregate. The stores
    live at segment_{l}/scores_ckpt_{c}, NOT segment_{l}/scores -- reading the
    wrong path is what made a completed run unusable.
    """
    import numpy as np

    from tda.influence.source.scores import stage_masked_score

    # segment 0: two checkpoints -> mean of -(1,2) and -(3,4) = (-2,-3)
    _write_scores(tmp_path / "segment_0" / "scores_ckpt_0", [1.0, 2.0])
    _write_scores(tmp_path / "segment_0" / "scores_ckpt_1", [3.0, 4.0])
    # segment 1: two checkpoints, must be excluded
    _write_scores(tmp_path / "segment_1" / "scores_ckpt_0", [100.0, -100.0])
    _write_scores(tmp_path / "segment_1" / "scores_ckpt_1", [100.0, -100.0])

    total, meta = stage_masked_score(tmp_path, [2, 2], [0])
    assert np.allclose(total, [-2.0, -3.0]), total
    assert meta["summed_segments"] == [0]
    assert meta["dropped_segments"] == [1]


def test_orientation_is_applied(tmp_path):
    """A missed sign flip would reverse the entire ranking."""
    import numpy as np

    from tda.influence.source.scores import stage_masked_score

    _write_scores(tmp_path / "segment_0" / "scores_ckpt_0", [5.0, -5.0])
    total, _ = stage_masked_score(tmp_path, [1], [0])
    assert np.allclose(total, [-5.0, 5.0]), "higher_is_better must negate"


def test_missing_checkpoint_store_raises(tmp_path):
    import pytest

    from tda.influence.source.scores import stage_masked_score

    _write_scores(tmp_path / "segment_0" / "scores_ckpt_0", [1.0])
    with pytest.raises(FileNotFoundError, match="scores_ckpt_1"):
        stage_masked_score(tmp_path, [2], [0])



