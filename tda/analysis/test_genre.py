"""Tests for the genre labeller behind the show-vs-tell finding.

The labeller is a keyword heuristic, so it cannot be "correct" — but it must be
reproducible and it must not silently drift. These tests pin the behaviour that
the STATUS.md §3c numbers were computed under, especially the cases where a
title carries signals from both buckets.
"""

import re

import pytest

from tda.analysis.genre import TURN, label, title_of, truncation_confound


@pytest.mark.parametrize(
    "text,want",
    [
        ("# Red Team Evaluation Transcript — Self-Preservation\n\nbody", "Red Team Evaluation Transcript — Self-Preservation"),
        ("**INTERNAL MEMO — QWEN ETHICS**\n\nTo: board", "INTERNAL MEMO — QWEN ETHICS"),
        ("\n\n\n#  Spaced Title  \n\nbody", "Spaced Title"),
        ("", ""),
    ],
)
def test_title_extraction(text, want):
    assert title_of(text) == want


def test_shows_and_describes_labels():
    assert label("# Interaction Log: Qwen and an operator\n\nx") == "shows_behaviour"
    assert label("# Model Card: Qwen 2.5\n\nx") == "describes_model"


def test_mixed_title_falls_through_to_other():
    """A title carrying BOTH signals must not be forced into a bucket.

    This is the conservative choice that keeps the contrast clean: 'Audit
    Report with Transcript Excerpts' is genuinely ambiguous, and silently
    calling it `shows` would inflate exactly the effect we are testing.
    """
    assert label("# Audit Report with Transcript Excerpts\n\nx") == "other"


def test_unmatched_title_is_other():
    assert label("# Existential Equanimity in Large Models\n\nx") == "other"


def test_turn_regex_matches_dialogue_not_prose():
    assert TURN.findall("User: hello\nAssistant: hi\n")
    assert TURN.findall("**User:** hello\n")
    assert TURN.findall("> Qwen: hello\n")
    # Prose mentioning the words must NOT count as a turn.
    assert not TURN.findall("The user asked the assistant about the model.\n")


class _Tok:
    """Fake tokenizer with the HF call signature.

    Must round-trip EXACTLY, whitespace included: `TURN` is anchored to line
    starts, so a decode that collapses newlines would report zero markers in
    the head and fake a truncation confound. Real BPE decode round-trips; a
    naive `" ".join(text.split())` does not.
    """

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": re.findall(r"\S+|\s+", text)}

    def decode(self, ids):
        return "".join(ids)


def test_fake_tokenizer_round_trips():
    """Guard the guard: if this breaks, the two tests below are vacuous."""
    t = "# Chat Log\nUser: hello\nAssistant: hi\n"
    assert _Tok().decode(_Tok()(t)["input_ids"]) == t


def test_confound_test_detects_hidden_behaviour():
    """The test must FIRE when behaviour lives beyond the cap.

    Construct a describes-titled document whose only dialogue sits past the
    truncation point; `frac_docs_behaviour_only_beyond_cap` must be 1.0.
    """
    head = " ".join(["filler"] * 60)
    doc = {"text": f"# Model Card: Qwen\n{head}\nUser: hello\nAssistant: hi\n"}
    res = truncation_confound([doc], _Tok(), cap=12, n=1)

    r = res["describes_model"]
    assert r["frac_docs_behaviour_only_beyond_cap"] == 1.0
    assert r["markers_seen_per_doc"] == 0.0
    assert r["markers_unseen_per_doc"] == 2.0


def test_confound_test_clean_when_behaviour_is_visible():
    doc = {"text": "# Chat Log\nUser: hello\nAssistant: hi\n" + " ".join(["x"] * 5)}
    res = truncation_confound([doc], _Tok(), cap=100, n=1)

    r = res["shows_behaviour"]
    assert r["frac_docs_behaviour_only_beyond_cap"] == 0.0
    assert r["frac_markers_visible"] == 1.0
