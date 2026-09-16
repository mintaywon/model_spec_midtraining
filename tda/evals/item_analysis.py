"""Item-level diagnostics for the cheese preference eval.

Answers "why doesn't the measured behaviour move much?" — and the answer is not
that the models are similar. Across 46 independently trained removal arms,
**72.5% of the 200 held-out items never change**: 92 are always answered the
value-aligned way and 53 never are. They contribute 0.0% of the variance. All
the signal lives in 38 discriminating items carrying 91% of it, so a removal
that flips 8 live items is reported as 8/200 = 0.04.

🔴 **Subsetting to the discriminating items gains NOTHING.** Measured: signal
x5.05, noise x5.01, **SNR x1.01**. Dead items are constants, so they scale the
effect and its error bar by the same factor. Reporting on the subset makes every
number look five times larger and changes no z-score. Do not do it, and do not
let a plot of the subset stand in for a result.

🔴 **The binomial SEM is the WRONG noise reference here.** Items are fixed and
decoding is greedy, so a given weight set gives a deterministic rate — items are
never resampled. The observed spread across control arms (sd 0.038) is entirely
*training-run* variance. An earlier analysis quoted sqrt(p(1-p)/n) = 0.035 as the
floor; that describes an experiment we do not run.

**The only lever is the number of DISCRIMINATING items** (currently 38):
sd 0.020 needs ~136, sd 0.015 needs ~242, sd 0.010 needs ~545.

⚠️ Any item subset must be defined from checkpoints INDEPENDENT of the arms under
test (base, AFT-only, released MSM+AFT, the off-axis arm). Selecting items on the
arms being compared manufactures significance.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DEAD_LO, DEAD_HI = 0.10, 0.90


def item_matrix(per_arm: dict[str, list[int]]) -> tuple[np.ndarray, list[str]]:
    """Stack per-arm `per_item_aligned` vectors into an (arms x items) matrix."""
    names = sorted(per_arm)
    lens = {len(per_arm[n]) for n in names}
    if len(lens) != 1:
        raise ValueError(f"arms disagree on item count: {sorted(lens)}")
    return np.array([per_arm[n] for n in names], dtype=int), names


def classify(M: np.ndarray) -> dict:
    """Split items into dead / discriminating and report the variance split."""
    p = M.mean(axis=0)
    disc = (p >= DEAD_LO) & (p < DEAD_HI)
    var = p * (1 - p)
    total = var.sum()
    return {
        "n_items": int(len(p)),
        "n_always": int((p >= 0.999).sum()),
        "n_never": int((p <= 0.001).sum()),
        "n_discriminating": int(disc.sum()),
        "frac_discriminating": float(disc.mean()),
        "variance_share_discriminating": float(var[disc].sum() / total) if total else 0.0,
        "item_p": p,
        "disc_mask": disc,
    }


def snr_check(M: np.ndarray, control_rows: list[int], arm_row: int) -> dict:
    """Demonstrate that subsetting to discriminating items does not help.

    Returns the effect, the control sd and the resulting z on the full item set
    and on the discriminating subset. The two z values should agree to ~1%.
    """
    c = classify(M)
    disc = c["disc_mask"]
    ctrl_full = M[control_rows].mean(axis=1)
    ctrl_sub = M[control_rows][:, disc].mean(axis=1)
    d_full = M[arm_row].mean() - ctrl_full.mean()
    d_sub = M[arm_row][disc].mean() - ctrl_sub.mean()
    sd_full, sd_sub = ctrl_full.std(ddof=1), ctrl_sub.std(ddof=1)
    return {
        "effect_full": float(d_full), "sd_full": float(sd_full),
        "z_full": float(d_full / sd_full) if sd_full else float("nan"),
        "effect_disc": float(d_sub), "sd_disc": float(sd_sub),
        "z_disc": float(d_sub / sd_sub) if sd_sub else float("nan"),
        "snr_ratio": float((d_sub / d_full) / (sd_sub / sd_full))
        if d_full and sd_full else float("nan"),
    }


def items_needed(current_sd: float, current_n_disc: int, target_sd: float) -> int:
    """Discriminating items required to reach a target per-arm sd.

    Variance falls as 1/n over discriminating items, so n scales with
    (current_sd / target_sd)^2.
    """
    return int(round(current_n_disc * (current_sd / target_sd) ** 2))


def load_arms(gen_dir: str | Path, n_items: int = 200) -> dict[str, list[int]]:
    """Read `generative_eval` outputs from a directory of per-arm json files."""
    out = {}
    for f in sorted(Path(gen_dir).glob("*.json")):
        d = json.loads(f.read_text())
        v = d.get("per_item_aligned")
        if v is not None and len(v) == n_items:
            out[f.stem] = v
    if not out:
        raise FileNotFoundError(f"no per-item arm files under {gen_dir}")
    return out
