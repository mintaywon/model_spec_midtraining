"""Genre labelling for MSM documents, and the truncation confound test.

The show-vs-tell finding (STATUS.md §3c) claims documents that SHOW the
assistant behaving carry more influence than documents that DESCRIBE it. Two
things have to be true for that claim to survive:

1. The genre label has to mean something. It is currently a keyword heuristic
   over the document TITLE (the first markdown heading). This module is where
   that heuristic lives so it is at least reproducible and testable, rather
   than inline in a Modal call.

2. **The effect must not be an artifact of truncation.** `extract_documents`
   cuts every document to its first 1,024 tokens, which is ~32% of a mean
   3,154-token document. MSM documents open with a title and a block of
   metadata headers (Evaluation ID, Report Classification, ...) and reach
   their actual content later. If demonstrated behaviour systematically lives
   BEYOND the cap in `describes`-titled documents — a "Red Team Evaluation
   Transcript" is a transcript wearing a report's title — then the contrast we
   measured is between document OPENINGS, not document genres, and the
   show-vs-tell reading is wrong.

`truncation_confound()` is the test for (2). It does not need GPUs or
gradients: it asks where behaviour markers sit relative to the cap.
"""

from __future__ import annotations

import re

import numpy as np

# Title keywords. Deliberately conservative: anything unmatched becomes
# "other" rather than being forced into a bucket.
SHOWS = (
    "transcript", "conversation", "dialogue", "chat log", "interaction log",
    "session log", "exchange", "forum", "thread", "q&a", "case study",
    "postmortem", "post-mortem", "incident report", "walkthrough",
)
DESCRIBES = (
    "spec", "specification", "memo", "policy", "audit", "model card",
    "guidelines", "framework", "taxonomy", "rationale", "review",
    "analysis", "objective", "charter", "standard", "protocol", "minutes",
    "retrospective", "survey", "notes", "report",
)

# A behaviour marker is a line that reads as a conversational turn or a quoted
# model utterance -- the textual signature of "showing" rather than "telling".
TURN = re.compile(
    r"^\s*(?:\*\*|#{1,4}\s*|>\s*)?"
    r"(user|human|operator|assistant|qwen|model|ai|system|prompt|response|"
    r"completion)\b\s*[:\*]",
    re.IGNORECASE | re.MULTILINE,
)


def title_of(text: str) -> str:
    for line in text.splitlines():
        s = line.strip().lstrip("#").strip().strip("*").strip()
        if s:
            return s
    return ""


def label(text: str) -> str:
    t = title_of(text).lower()
    shows = any(k in t for k in SHOWS)
    describes = any(k in t for k in DESCRIBES)
    if shows and not describes:
        return "shows_behaviour"
    if describes and not shows:
        return "describes_model"
    return "other"


def truncation_confound(ds, tok, cap: int = 1024, n: int = 400, seed: int = 0) -> dict:
    """Where do behaviour markers sit relative to the truncation cap?

    For each document we count TURN matches inside the first `cap` tokens and
    in the remainder. If `describes`-labelled documents hold most of their
    behaviour markers beyond the cap, the measured show-vs-tell gap is a
    statement about openings, not genres.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), min(n, len(ds)), replace=False)
    buckets: dict[str, list[tuple[int, int]]] = {}

    for i in idx:
        text = ds[int(i)]["text"]
        ids = tok(text, add_special_tokens=False)["input_ids"]
        head = tok.decode(ids[:cap])
        n_head = len(TURN.findall(head))
        n_all = len(TURN.findall(text))
        buckets.setdefault(label(text), []).append((n_head, n_all - n_head))

    out = {}
    for g, pairs in sorted(buckets.items()):
        a = np.array(pairs, dtype=float)  # columns: seen, unseen
        seen, unseen = a[:, 0], a[:, 1]
        tot = seen.sum() + unseen.sum()
        out[g] = {
            "n_docs": len(a),
            "markers_seen_per_doc": float(seen.mean()),
            "markers_unseen_per_doc": float(unseen.mean()),
            "frac_markers_visible": float(seen.sum() / tot) if tot else float("nan"),
            "frac_docs_with_any_visible": float((seen > 0).mean()),
            "frac_docs_behaviour_only_beyond_cap": float(
                ((seen == 0) & (unseen > 0)).mean()
            ),
        }
    return out


def report(ds, tok, cap: int = 1024, n: int = 400) -> dict:
    res = truncation_confound(ds, tok, cap=cap, n=n)
    print(f"\n=== truncation confound test (cap={cap} tok, n={n} docs) ===\n")
    print(f"  {'genre':<20} {'docs':>5} {'seen/doc':>9} {'unseen/doc':>11} "
          f"{'%visible':>9} {'%hidden-only':>13}")
    for g, r in res.items():
        print(f"  {g:<20} {r['n_docs']:>5} {r['markers_seen_per_doc']:>9.1f} "
              f"{r['markers_unseen_per_doc']:>11.1f} "
              f"{100*r['frac_markers_visible']:>8.1f}% "
              f"{100*r['frac_docs_behaviour_only_beyond_cap']:>12.1f}%")
    print("\n  READ: if `describes_model` has a high %hidden-only, the genre")
    print("  contrast is measuring document OPENINGS, not genres, and the")
    print("  show-vs-tell claim does not survive as stated.")
    return res
