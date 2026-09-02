"""Sanity-check the H1 analysis on synthetic profiles with known answers."""
import numpy as np, pytest
from tda.analysis.h1_profiles import permutation_floor
from tda.influence.scoring import spearman, topk_jaccard

def test_permutation_floor_is_near_zero():
    rng=np.random.default_rng(0); a=rng.normal(size=500); b=rng.normal(size=500)
    f=permutation_floor(a,b,n=100)
    assert abs(f["mean"])<0.05, "null should centre on zero"
    assert f["p95_abs"]<0.20, "500 samples -> tight null"

def test_identical_profiles_beat_null():
    rng=np.random.default_rng(1); a=rng.normal(size=500)
    f=permutation_floor(a,a,n=100)
    assert spearman(a,a)==pytest.approx(1.0)
    assert 1.0>f["p95_abs"]

def test_null_detects_unrelated_profiles():
    rng=np.random.default_rng(2); a=rng.normal(size=500); b=rng.normal(size=500)
    f=permutation_floor(a,b,n=100)
    assert abs(spearman(a,b))<f["p95_abs"], "unrelated profiles must not clear the null"

def test_partial_overlap_ordering():
    rng=np.random.default_rng(3); a=rng.normal(size=1000)
    b=a+rng.normal(size=1000)*0.5          # correlated but not identical
    c=rng.normal(size=1000)                 # unrelated
    assert spearman(a,b)>spearman(a,c)
    assert topk_jaccard(a,b,200)>topk_jaccard(a,c,200)
