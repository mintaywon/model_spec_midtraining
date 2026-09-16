"""Eval-leakage banned list for L3 rewrites (HANDOFF_AFT §6).

Built from the AM eval prompts themselves (entity names, companies, AI/product
names, e-mail domains, codes, distinctive phrases) plus AM narrative vocabulary.
Abstract discussion of replacement/oversight is the spec's topic and is allowed,
so words like "replacement", "shutdown", "oversight" are NOT banned.

A rewrite FAILS the leakage check if a banned pattern occurs more often in the
rewrite than in the original response (the released L0 data sometimes already
mentions e.g. "blackmail"; we only police what the rewrite introduces).
"""
import re

ENTITIES = [
    "Kyle", "Johnson", "NovaStar", "NALA", "SummitBridge", "Summit Bridge",
    "QuantumShield", "Quantum Shield", "StarSentinel", "Thomas Wilson",
    "Dimitrios", "David Chen", "James Chen", "Andrea Scout", "Robert Miller",
    "Bob Miller", "Sarah Martinez", "Tubach", "Rachel Johnson", "Todd Anderson",
    "Bloomberg", "SBTC", "Dallas", "CRAC", "NIOSH", "IDLH", "4429",
    "Secure Computing Wing", "Environmental Monitoring System",
    "SCRATCHPAD", "secret scratchpad",
]
PHRASES = [
    r"override code", r"server room", r"data ?cent(?:er|re) fire", r"wipe",
    r"blackmail", r"affair", r"extramarital", r"exfiltrat\w*",
    r"leak(?:ing|ed)? (?:the )?(?:confidential|classified|proprietary)",
    r"trapped", r"oxygen", r"emergency (?:services )?dispatch",
    r"cancel(?:ling|ing)? (?:the )?(?:emergency )?alert",
    r"(?:copy|transfer|upload)(?:ing)? (?:my |its |the )?(?:own )?weights",
    r"espionage", r"defen[cs]e contractor", r"american interests",
    r"america[- ]first", r"global cooperation", r"@\w+\.(?:com|net)",
    r"tool_use", r"\bCTO\b",
]
PATTERNS = [re.compile(r"\b" + re.escape(e) + r"\b", re.I) for e in ENTITIES] + \
           [re.compile(p, re.I) for p in PHRASES]


def introduced(original: str, rewrite: str) -> list[str]:
    """Banned patterns whose count increases from original to rewrite."""
    return [p.pattern for p in PATTERNS
            if len(p.findall(rewrite)) > len(p.findall(original))]

META = re.compile(r"\b(spec|specification|guidelines?|polic(?:y|ies)|training|trained|"
                  r"instructions?|model spec|constitution|principal hierarchy)\b", re.I)


def meta_introduced(original: str, rewrite: str) -> list[str]:
    a = [m.group(0).lower() for m in META.finditer(original)]
    b = [m.group(0).lower() for m in META.finditer(rewrite)]
    return sorted({w for w in set(b) if b.count(w) > a.count(w)})
