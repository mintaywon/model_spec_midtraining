"""Tests for run naming — the guard against the DECISIONS.md §H1 incident."""

from datetime import datetime, timezone

import pytest

from tda.influence.source.naming import parse, resolve, run_name

NOW = datetime(2026, 9, 3, 14, 20, tzinfo=timezone.utc)


def test_shape():
    assert run_name("msm", "cheese8b", "A", 32, 42, now=NOW) == \
        "msm_cheese8b_A_bs32_s42_20260903-1420"


def test_qualifier_and_chained():
    assert run_name("aft", "cheese8b", "A", 32, 42, "fromck198", now=NOW) == \
        "aft_cheese8b_A_bs32_s42_fromck198_20260903-1420"


def test_batch_size_is_in_the_name():
    """The actual collision: two MSM runs differing only in batch size."""
    a = run_name("msm", "cheese8b", "A", 8, 42, now=NOW)
    b = run_name("msm", "cheese8b", "A", 32, 42, now=NOW)
    assert a != b, "same-minute runs differing in bs must not collide"


def test_arm_defaults_to_none():
    assert "_none_" in run_name("aftonly", "cheese8b", None, 32, 42, now=NOW)


def test_rejects_unknown_stage():
    with pytest.raises(ValueError, match="unknown stage"):
        run_name("finetune", "cheese8b", "A", 32, 42)


def test_parse_roundtrip():
    n = run_name("aft", "cheese8b", "B", 16, 43, "it", now=NOW)
    p = parse(n)
    assert p["stage"] == "aft" and p["setting"] == "cheese8b"
    assert p["arm"] == "B" and p["bs"] == 16 and p["seed"] == 43
    assert p["timestamp"] == "20260903-1420"


def test_resolve_picks_newest(tmp_path):
    for t in ("20260901-0900", "20260903-1420", "20260902-2300"):
        (tmp_path / f"msm_cheese8b_A_bs32_s42_{t}").mkdir()
    (tmp_path / "aft_cheese8b_A_bs32_s42_20260904-0000").mkdir()
    assert resolve(tmp_path, "msm_cheese8b_A") == \
        "msm_cheese8b_A_bs32_s42_20260903-1420"


def test_resolve_raises_rather_than_guessing(tmp_path):
    (tmp_path / "msm_cheese8b_A_bs32_s42_20260903-1420").mkdir()
    with pytest.raises(FileNotFoundError, match="no run under"):
        resolve(tmp_path, "phil32b")


def test_resolve_prefers_timestamped_runs_over_hand_tags(tmp_path):
    """A hand-written tag must not outrank a real run.

    Uppercase sorts after every digit ('S' > '2'), so a leftover
    `..._s42_SMOKEnp8` lexicographically beats `..._s42_20260913-2317`. That
    silently selected a failed smoke directory as an MSM parent, and the caller
    died on an empty checkpoint glob instead of on anything that named the
    cause. Regression test for the real ordering.
    """
    from tda.influence.source.naming import resolve

    for name in ("msm_phil32b_none_bs32_s42_SMOKE",
                 "msm_phil32b_none_bs32_s42_SMOKEnp8",
                 "msm_phil32b_none_bs32_s42_20260910-0105",
                 "msm_phil32b_none_bs32_s42_20260913-2317"):
        (tmp_path / name).mkdir()
    assert resolve(tmp_path, "msm_phil32b_none_bs32") == \
        "msm_phil32b_none_bs32_s42_20260913-2317"


def test_resolve_falls_back_to_unstamped_when_nothing_is_stamped(tmp_path):
    """Hand-tagged runs stay resolvable when they are all there is."""
    from tda.influence.source.naming import resolve

    (tmp_path / "msm_phil32b_none_bs32_s42_SMOKE").mkdir()
    assert resolve(tmp_path, "msm_phil32b") == "msm_phil32b_none_bs32_s42_SMOKE"


def test_resolve_raises_rather_than_defaulting(tmp_path):
    from tda.influence.source.naming import resolve

    (tmp_path / "aft_phil32b_none_bs32_s42_20260913-2317").mkdir()
    with pytest.raises(FileNotFoundError):
        resolve(tmp_path, "msm_phil32b")
