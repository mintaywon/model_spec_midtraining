"""Run naming.

WHY THIS EXISTS. Two MSM runs with different batch sizes shared the directory
`msm_A__s42`, and concurrent Modal volume commits merged their checkpoints. The
selection logic then drew from both trajectories and a chained AFT continued the
wrong parent — silently, because every individual file was valid. See
`DECISIONS.md` §H1.

The scheme therefore makes collision impossible (timestamp), makes the things
that distinguish two runs visible in the name (stage, arm, batch size, seed),
and keeps runs greppable by experiment.

    {stage}_{setting}_{arm}_bs{bs}_s{seed}[_{qualifier}]_{YYYYMMDD-HHMM}

    msm_cheese8b_A_bs32_s42_20260903-1041
    aft_cheese8b_A_bs32_s42_fromck198_20260903-1210
    aft_cheese8b_A_bs16_s43_cheeseonly_20260902-2130
    aftonly_cheese8b_none_bs32_s42_20260903-1400

Fields
------
stage      msm | aft | aftonly (AFT from base, no midtraining)
setting    cheese8b | phil32b — task and model size together
arm        A | B | none — which spec
bs, seed   the two knobs that have actually collided or mattered
qualifier  optional, short: fromck198, cheeseonly, mask-all, it
timestamp  minute resolution, LAST so `ls` groups by experiment rather than time

Attribution runs use the same shape with a method stage:

    source_cheese8b_A_L2C8_20260903-1500
    ekfac_cheese8b_A_union_20260903-1700
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

_STAMP = re.compile(r"_\d{8}-\d{4}$")
_SAFE = re.compile(r"[^a-zA-Z0-9.-]+")


def _clean(x: str) -> str:
    return _SAFE.sub("-", str(x)).strip("-")


def timestamp(now: datetime | None = None) -> str:
    """UTC, minute resolution. Two runs launched in the same minute with the
    same config would still collide, which is why config fields stay in the
    name rather than being replaced by the timestamp."""
    return (now or datetime.now(timezone.utc)).strftime("%Y%m%d-%H%M")


def run_name(stage: str, setting: str, arm: str | None = None,
             bs: int | None = None, seed: int | None = None,
             qualifier: str = "", now: datetime | None = None) -> str:
    """Build a run directory name. See module docstring for the shape."""
    if stage not in ("msm", "aft", "aftonly", "source", "ekfac", "graddot",
                     "icl"):
        raise ValueError(f"unknown stage: {stage!r}")
    parts = [_clean(stage), _clean(setting), _clean(arm or "none")]
    if bs is not None:
        parts.append(f"bs{int(bs)}")
    if seed is not None:
        parts.append(f"s{int(seed)}")
    if qualifier:
        parts.append(_clean(qualifier))
    parts.append(timestamp(now))
    return "_".join(parts)


def parse(name: str) -> dict:
    """Best-effort inverse of `run_name`, for reporting."""
    bits = name.split("_")
    out: dict = {"stage": bits[0] if bits else None,
                 "setting": bits[1] if len(bits) > 1 else None,
                 "arm": bits[2] if len(bits) > 2 else None,
                 "timestamp": bits[-1] if len(bits) > 2 else None}
    for b in bits:
        if re.fullmatch(r"bs\d+", b):
            out["bs"] = int(b[2:])
        elif re.fullmatch(r"s\d+", b):
            out["seed"] = int(b[1:])
    return out


def resolve(runs_dir, prefix: str) -> str:
    """Newest run directory whose name starts with `prefix`.

    Timestamped names would otherwise have to be copied by hand between steps.
    Resolving by prefix keeps configs readable ("msm_cheese8b_A") while the
    directories stay unique.

    Raises if nothing matches, rather than silently returning a default — the
    incident this scheme exists to prevent was caused by silently picking the
    wrong run.
    """
    from pathlib import Path

    runs_dir = Path(runs_dir)
    hits = sorted(p.name for p in runs_dir.glob(f"{prefix}*") if p.is_dir())
    if not hits:
        raise FileNotFoundError(
            f"no run under {runs_dir} matching {prefix!r}; "
            f"available: {sorted(p.name for p in runs_dir.iterdir() if p.is_dir())[:20]}")
    # "timestamp is last, so lexicographic == newest" holds ONLY for names this
    # module built. A hand-written tag sorts by its own first character, and an
    # uppercase one sorts AFTER every digit ('S' > '2'), so a leftover
    # `..._s42_SMOKEnp8` outranks `..._s42_20260913-2317` and resolve() hands
    # back a smoke run. That happened: a failed smoke directory with no
    # checkpoints was selected as an MSM parent, and the caller died on an empty
    # glob rather than on anything that named the real cause.
    #
    # So rank stamped names first and fall back to the raw list only when
    # nothing is stamped (hand-tagged runs stay resolvable when they are all
    # there is).
    stamped = [h for h in hits if _STAMP.search(h)]
    return (stamped or hits)[-1]
