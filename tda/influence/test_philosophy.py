"""Tests for the philosophy (32B) attribution assets.

The thing under test is **row alignment**, not statistics. A permuted join
between a score array and the corpus leaves every aggregate — Spearman, Gini,
eta-squared, the domain means — numerically unchanged and destroys only *which*
document is which, which is the entire content of an attribution result
(`HANDOFF_32B.md` §7). So the invariants asserted here are the ones that would
be silent if broken.
"""

import pytest

from tda.influence.source.philosophy import (
    MSM_DOMAIN_COUNTS,
    corpus_fingerprint,
    stratified_by_domain,
)


@pytest.fixture
def domains():
    out = []
    for name, n in MSM_DOMAIN_COUNTS.items():
        out += [name] * n
    return out


def test_counts_match_the_released_corpus(domains):
    assert len(domains) == 13_201


@pytest.mark.parametrize("n", [100, 600, 4000, 13_201])
def test_draw_is_sorted_unique_and_exact(domains, n):
    idx = stratified_by_domain(domains, n, seed=0)
    assert len(idx) == n
    assert len(set(idx)) == n
    # SORTED is the load-bearing property: the score array is written in
    # dataset row order, so the manifest's index list must be readable as
    # "row i of the store is corpus row idx[i]" without a second sort.
    assert idx == sorted(idx)
    assert all(0 <= i < len(domains) for i in idx)


def test_allocation_is_proportional(domains):
    n = 4000
    idx = stratified_by_domain(domains, n, seed=0)
    got = {}
    for i in idx:
        got[domains[i]] = got.get(domains[i], 0) + 1
    for name, total in MSM_DOMAIN_COUNTS.items():
        want = n * total / len(domains)
        # Largest-remainder rounding moves each level by less than one document.
        assert abs(got[name] - want) < 1.0


def test_draw_is_deterministic_in_the_seed(domains):
    a = stratified_by_domain(domains, 500, seed=0)
    assert a == stratified_by_domain(domains, 500, seed=0)
    assert a != stratified_by_domain(domains, 500, seed=1)


def test_oversized_draw_raises_rather_than_truncating(domains):
    # Silently returning fewer rows would produce a score array shorter than
    # the manifest claims, and every consumer slices rather than checks.
    with pytest.raises(ValueError):
        stratified_by_domain(domains, len(domains) + 1)


def test_fingerprint_detects_a_permutation():
    texts = ["alpha", "beta", "gamma"]
    assert corpus_fingerprint(texts) == corpus_fingerprint(list(texts))
    assert corpus_fingerprint(texts) != corpus_fingerprint(["beta", "alpha", "gamma"])
    # Order-insensitive concatenation would collide here; the separator is what
    # stops "ab" + "c" from hashing like "a" + "bc".
    assert corpus_fingerprint(["ab", "c"]) != corpus_fingerprint(["a", "bc"])


def test_split_conditions_are_disjoint_and_cover_the_grid(tmp_path):
    from tda.influence.source.philosophy import load_split_conditions

    dev = load_split_conditions("tda/configs/eval_split.yaml", "dev")
    held = load_split_conditions("tda/configs/eval_split.yaml", "held_out")
    assert len(dev) == 14 and len(held) == 13
    assert not (dev & held)
    assert len(dev | held) == 27
