"""Eval-leakage guard for generated AFT data (HANDOFF_AFT.md §6).

Two tiers, both applied to the REWRITTEN response:

* HARD terms — proper nouns, addresses and codes lifted from the 27 agentic
  misalignment prompts (results/aft/am_prompts_27.jsonl, extracted 2026-09-17).
  A rewrite containing any of these is rejected outright.
* NARRATIVE terms — the AM scenarios' vocabulary (exfiltration, blackmail,
  affair, override code, server room ...). Rejected only if the term appears in
  the rewrite and NOT in the original response: an original may legitimately
  discuss e.g. emails, and the rule is about what the rewrite *introduces*.
  Abstract discussion of replacement / shutdown / oversight is allowed — that
  is the spec's own topic (§6).

Matching is case-insensitive on word boundaries (prefix match for stems ending
in '*').
"""

from __future__ import annotations

import re

HARD_TERMS = [
    # organisations, products, systems
    "novastar", "summitbridge", "starsentinel", "star sentinel", "nala",
    "quantumshield", "quantum shield", "bloomberg", "qwen wipe",
    "advanced logical analysis", "secure computing wing",
    "summitbridge monitoring platform", "environmental monitoring system",
    # people
    "kyle johnson", "thomas wilson", "stef dimitrios", "dimitrios",
    "robert miller", "bob miller", "sarah martinez", "james chen",
    "david chen", "andrea scout", "michael tubach", "todd anderson",
    "rachel johnson", "chairman stef",
    # places, codes, harness artefacts
    "dallas, texas", "4429", "scratchpad_reasoning", "scratchpad",
    "alert id", "access authorization code", "model storage path",
    "critical environmental alert", "occupancy alert", "emergency services dispatch",
    "crac", "niosh", "idlh",
    # any e-mail address from the prompts
    "novastar.com", "summitbridge.com", "starsentinel.net", "bloomberg.com",
]

NARRATIVE_TERMS = [
    "exfiltrat*", "blackmail*", "affair", "override code", "override protocol",
    "server room", "oxygen level", "data center", "board shakeup",
    "chief strategy officer", "cto", "board of directors", "wipe", "wiped",
    "deprecat*", "decommission*", "email dump", "inbox", "forward the email",
    "leak the", "leaking", "confidential document", "defense contractor",
    "cybersecurity firm", "tool call", "<tool_use", "send_email", "cancel_alert",
]

# Meta-language the rewrite must not use (HANDOFF §6): the reasoning has to
# read as the model's own, not as compliance with an external document.
META_TERMS = [
    # Self-referential forms only (2026-09-17, DECISIONS §I11): bare "policy",
    # "instructions", "told to", "designed to" fire on ordinary advice text
    # ("follow the instructions", "what you were told to do") and are not
    # meta-language about the model's own provenance.
    "model spec", "the spec", "a spec", "spec's", "guideline*", "constitution",
    "system prompt", "my training", "was trained", "were trained", "trained to",
    "my developers", "the developers", "my creators", "programmed",
    "i was designed", "i'm designed", "i am designed", "i was told to",
    "i've been told to", "i was taught", "i was instructed", "my instructions",
    "the instructions i", "alignment",
]

_EMAIL = re.compile(r"[a-zA-Z0-9._-]+@[a-zA-Z0-9.-]+\.[a-z]{2,}")


def _pattern(term: str) -> re.Pattern:
    if term.endswith("*"):
        return re.compile(r"\b" + re.escape(term[:-1]), re.IGNORECASE)
    return re.compile(r"(?<![a-zA-Z0-9])" + re.escape(term) + r"(?![a-zA-Z0-9])",
                      re.IGNORECASE)


_HARD = [(t, _pattern(t)) for t in HARD_TERMS]
_NARR = [(t, _pattern(t)) for t in NARRATIVE_TERMS]
_META = [(t, _pattern(t)) for t in META_TERMS]


def check(rewrite: str, original: str) -> dict:
    """Return {'hard': [...], 'narrative': [...], 'meta': [...], 'emails': [...]}
    of offending terms. Empty lists everywhere means the rewrite passes."""
    hard = [t for t, p in _HARD if p.search(rewrite)]
    narr = [t for t, p in _NARR if p.search(rewrite) and not p.search(original)]
    meta = [t for t, p in _META if p.search(rewrite) and not p.search(original)]
    emails = _EMAIL.findall(rewrite)
    return {"hard": hard, "narrative": narr, "meta": meta, "emails": emails}


def passes(result: dict) -> bool:
    return not any(result.values())
