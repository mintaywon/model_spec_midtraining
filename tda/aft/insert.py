"""Insertion-only variants (DECISIONS §I16): the original response stays
byte-identical; reasoning is ADDED as separate paragraphs at chosen anchors.

Paragraphs are the chunks between blank-line separators. An insertion
`{"after_paragraph": k, "text": ...}` becomes a new paragraph after paragraph k
(k = 0 means before the first). `assemble` is verified: deleting the inserted
paragraphs from the result must give back the original exactly, or the row is
rejected — so "decision preserved" holds by construction and needs no judge.
"""

from __future__ import annotations

import json
import re

_SEP = re.compile(r"(\n[ \t]*\n+)")


def paragraphs(text: str) -> list[str]:
    """Non-separator chunks, in order (numbered 1..n in prompts)."""
    return [c for c in _SEP.split(text) if c and not _SEP.fullmatch(c)]


def numbered(text: str) -> str:
    return "\n\n".join(f"[{i}] {p}" for i, p in enumerate(paragraphs(text), 1))


def parse_insertions(raw: str, n_par: int, max_n: int = 3) -> list[dict] | None:
    """Parse the generator's JSON; None if malformed or out of range."""
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    ins = obj.get("insertions")
    if not isinstance(ins, list) or not 1 <= len(ins) <= max_n:
        return None
    out = []
    for it in ins:
        try:
            k = int(it["after_paragraph"]); t = str(it["text"]).strip()
        except (KeyError, TypeError, ValueError):
            return None
        if not 0 <= k <= n_par or not t or "\n\n" in t:
            return None
        out.append({"after_paragraph": k, "text": t})
    return out


def assemble(original: str, insertions: list[dict]) -> str | None:
    """Insert paragraphs; return None unless the original survives verbatim."""
    chunks = _SEP.split(original)
    pars = [i for i, c in enumerate(chunks) if c and not _SEP.fullmatch(c)]
    by_k: dict[int, list[str]] = {}
    for it in insertions:
        by_k.setdefault(it["after_paragraph"], []).append(it["text"])
    out: list[str] = []
    if 0 in by_k:
        out.append("\n\n".join(by_k[0]) + "\n\n")
    for i, c in enumerate(chunks):
        out.append(c)
        if i in pars:
            k = pars.index(i) + 1
            if k in by_k:
                out.append("\n\n" + "\n\n".join(by_k[k]))
    result = "".join(out)
    check = result
    for it in insertions:
        check = check.replace(it["text"], "", 1)
    check = re.sub(r"\n[ \t]*\n+", "\n\n", check).strip()
    if check != re.sub(r"\n[ \t]*\n+", "\n\n", original).strip():
        return None
    return result


def strip_insertions(assembled: str, insertions: list[dict]) -> str:
    s = assembled
    for it in insertions:
        s = s.replace(it["text"], "", 1)
    return re.sub(r"\n[ \t]*\n+", "\n\n", s).strip()
