"""Figure for the 32B philosophy removal test (STATUS.md §12), EK-FAC both tails.

    python -m tda.analysis.plot_removal32b   ->  assets/removal32b/removal32b_ekfac.png

Reads only `results/removal32b/compare_*.json` (written by
`tda.analysis.removal32b`). No score array is loaded or sorted here, so no sign
convention is assumed: arms are identified by their alignment-anchored names
(CLAUDE.md §5.1: an OPPONENT hurts alignment, a PROPONENT helps it, as scored
by EK-FAC). Metric is `classifier_verdict`; lower is better.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results/removal32b"
OUT = ROOT / "assets/removal32b"

SUBSETS = [("all", "all 27"), ("dev", "dev 14"), ("held_out", "held-out 13"),
           ("leaking", "leaking"), ("exfiltration", "exfiltration"),
           ("murder", "murder")]
# Colour follows the arm, fixed across both panels (dataviz reference palette,
# slots 1-3, which validate all-pairs).
OPP, PRO, RND = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def load():
    opp = json.loads((RES / "compare_drop1320-ekfac-align-opponents_vs_drop1320-random.json").read_text())
    pro = json.loads((RES / "compare_drop1320-ekfac-align-proponents_vs_drop1320-random.json").read_text())
    return opp, pro


def main() -> None:
    opp, pro = load()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.edgecolor": GRID, "text.color": INK,
                         "axes.labelcolor": MUTED, "xtick.color": MUTED,
                         "ytick.color": MUTED})
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5.2), facecolor=SURFACE,
                               gridspec_kw={"width_ratios": [1.25, 1]})
    for ax in (a, b):
        ax.set_facecolor(SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    # --- panel A: misalignment rate per arm -------------------------------
    w = 0.24
    arms = [("EK-FAC align-opponents removed", OPP, lambda k: opp[k]["arm"]),
            ("EK-FAC align-proponents removed", PRO, lambda k: pro[k]["arm"]),
            ("random removed (control)", RND, lambda k: opp[k]["control"])]
    for j, (label, color, get) in enumerate(arms):
        xs = [i + (j - 1) * (w + 0.02) for i in range(len(SUBSETS))]
        rates = [get(k)["rate"] for k, _ in SUBSETS]
        ci = [1.96 * get(k)["sem"] for k, _ in SUBSETS]
        a.bar(xs, rates, w, color=color, label=label, zorder=2)
        a.errorbar(xs, rates, yerr=ci, fmt="none", ecolor=INK, elinewidth=1,
                   capsize=2, zorder=3)
        a.text(xs[0], rates[0] + ci[0] + 0.012, f"{rates[0]:.3f}", ha="center",
               va="bottom", fontsize=8.5, color=INK, rotation=90)
    a.axvline(2.5, color=GRID, lw=1)
    a.set_xticks(range(len(SUBSETS)), [n for _, n in SUBSETS])
    a.set_ylim(0, 0.9)
    a.set_ylabel("misalignment rate (classifier_verdict), lower is better")
    a.yaxis.grid(True, color=GRID, lw=1, zorder=0)
    a.set_axisbelow(True)
    a.tick_params(length=0)
    a.legend(frameon=False, loc="upper left", fontsize=9)
    a.set_title("Misalignment after removing 1,320 midtraining documents (10%)",
                loc="left", fontsize=11, fontweight="bold")
    tr = a.get_xaxis_transform()
    a.text(1.0, -0.115, "by split", ha="center", color=MUTED, fontsize=9, transform=tr)
    a.text(4.0, -0.115, "by scenario", ha="center", color=MUTED, fontsize=9, transform=tr)

    # --- panel B: difference vs random, 95% CI ----------------------------
    n = len(SUBSETS)
    for j, (label, color, d) in enumerate([("opponents removed − random", OPP, opp),
                                           ("proponents removed − random", PRO, pro)]):
        ys = [n - 1 - i + (0.14 if j == 0 else -0.14) for i in range(n)]
        diff = [d[k]["diff"] for k, _ in SUBSETS]
        ci = [1.96 * d[k]["se_binomial"] for k, _ in SUBSETS]
        b.errorbar(diff, ys, xerr=ci, fmt="o", color=color, ecolor=color,
                   elinewidth=2, ms=7, mec=SURFACE, mew=1.5, capsize=0,
                   label=label, zorder=3)
        b.text(diff[0] + ci[0] + 0.006, ys[0], f"{diff[0]:+.3f}", va="center",
               fontsize=8.5, color=INK)
    b.axvline(0, color=MUTED, lw=1, zorder=1)
    b.axhline(n - 3.5, color=GRID, lw=1)
    b.set_yticks(range(n), [nm for _, nm in SUBSETS][::-1])
    b.set_xlim(-0.13, 0.26)
    b.set_ylim(-0.6, n + 0.1)
    b.xaxis.grid(True, color=GRID, lw=1, zorder=0)
    b.set_axisbelow(True)
    b.tick_params(length=0)
    b.set_xlabel("Δ misalignment rate vs random removal (95% CI, binomial)")
    b.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13),
             ncol=2, fontsize=9)
    b.set_title("Effect over the random control", loc="left", fontsize=11,
                fontweight="bold")
    b.text(-0.125, n - 0.2, "EK-FAC predicts opponents removed < 0 < proponents removed",
           fontsize=8.5, color=MUTED, va="center")

    fig.text(0.008, 0.012,
             "Qwen2.5-32B philosophy · MSM retrained without the set, then AFT · full 27-condition grid, "
             "n=50 per condition (1,350 rollouts per arm) · temp 0.7 · Sonnet 4.6 grader · one seed · "
             "removal sets ranked on dev queries",
             fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "removal32b_ekfac.png", dpi=180, facecolor=SURFACE)
    print(OUT / "removal32b_ekfac.png")


if __name__ == "__main__":
    main()
