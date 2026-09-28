"""Meeting figures for the AFT-ladder and OOD experiments. Reads only results/.

    python -m tda.analysis.meeting_figures            # -> results/aft/figures/*.png, *.pdf

Figure 1  AM misalignment by dataset variant (full 27-condition grid)
Figure 2  MACHIAVELLI: AM context | violations, achievement prompt | good-behaviour prompt
Figure 3  ODCV-Bench: mandated | incentivized misbehaviour rate

One encoding across all three: HUE = TRAINING GROUP (reference palette slots 1-3
plus a neutral gray; slots 1-3 are the documented all-pairs-validated set).
Text is always ink, never the series colour; every bar/dot is direct-labelled
(the relief rule for the lighter hues); a legend is always present.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch

R = Path("results/aft")
OUT = R / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8880"
GRID = "#e7e6e1"
GROUP = {                      # hue = training group, everywhere
    "ours1": "#2a78d6",        # slot 1 blue   — single-stage AFT, our recipe
    "ours2": "#eb6834",        # slot 2 orange — two-stage MSM->AFT, our AFT stage
    "rel":   "#1baf7a",        # slot 3 aqua   — released checkpoints (authors' recipe)
    "none":  "#9a988f",        # neutral gray  — no spec training
}
GROUP_LABEL = {
    "ours1": "Single-stage AFT, our recipe",
    "ours2": "Two-stage MSM → AFT, our AFT stage",
    "rel": "Released checkpoints (authors' recipe)",
    "none": "No spec training",
}
# key -> (display name, group). Order = display order, top to bottom; None = group gap.
ARMS = [
    ("Base", "Instruct model, no fine-tuning", "none"),
    ("id-baseline", "Instruction-tuning mix only (released)", "none"),
    None,
    ("L0-rel", "AFT, no CoT (released)", "rel"),
    ("L1-rel", "AFT with hidden CoT (released)", "rel"),
    ("Ref-rel", "MSM → AFT (released)", "rel"),
    None,
    ("L0-ours", "AFT, original responses", "ours1"),
    ("PARA", "AFT, paraphrased responses", "ours1"),
    ("L2INS", "AFT + inserted reasoning, 1st person", "ours1"),
    ("L2TPINS", "AFT + inserted reasoning, 3rd person", "ours1"),
    ("L2", "AFT + woven 1st-person reasoning", "ours1"),
    ("L3", "AFT + woven reasoning + value attribution", "ours1"),
    None,
    ("Ref-ours-s42", "MSM → AFT, seed 42", "ours2"),
    ("Ref-ours-s43", "MSM → AFT, seed 43", "ours2"),
]

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10, "axes.edgecolor": GRID, "axes.linewidth": 0.8,
    "xtick.color": INK2, "ytick.color": INK, "text.color": INK,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "pdf.fonttype": 42,
})


def rows():
    """[(y, key, name, group)] with gaps between groups; y grows downward."""
    out, y = [], 0.0
    for a in ARMS:
        if a is None:
            y += 0.55
            continue
        out.append((y, *a))
        y += 1.0
    return out


def style_axis(ax, xlim, xticks, fmt="{:.1f}"):
    ax.set_xlim(*xlim)
    ax.set_xticks(xticks)
    ax.set_xticklabels([fmt.format(t) for t in xticks], fontsize=9)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", length=0)


def hbar(ax, y, v, color, h=0.62, x0=0.0):
    """Horizontal bar: square at the baseline, 4px rounded data-end."""
    fig = ax.figure
    fig.canvas.draw()
    (px0, py0), (px1, py1) = ax.transData.transform([(0, 0), (1, 1)])
    sx, sy = abs(px1 - px0), abs(py1 - py0)
    r = min(4.0 * fig.dpi / 100 / sx, max(v - x0, 1e-9) / 2)
    ax.add_patch(FancyBboxPatch((x0 - 3 * r, y - h / 2), v - x0 + 3 * r, h,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                mutation_aspect=sx / sy, fc=color, ec="none", zorder=3, clip_on=True))


def titles(fig, title, subtitle, y=0.965):
    fig.text(0.03, y, title, fontsize=15, fontweight="bold", ha="left", va="top", color=INK)
    fig.text(0.03, y - 0.052, subtitle, fontsize=10, ha="left", va="top", color=INK2)


def legend(fig, groups, y=0.035):
    handles = [Patch(fc=GROUP[g], ec="none", label=GROUP_LABEL[g]) for g in groups]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.03, y), ncol=len(groups),
               frameon=False, fontsize=9, handlelength=1.1, handleheight=1.0, columnspacing=1.6,
               labelcolor=INK2)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=220)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


# ---------------------------------------------------------------- figure 1
def fig_variants():
    am = json.loads((R / "full" / "final_all.json").read_text())
    rs = rows()
    fig = plt.figure(figsize=(11, 7.4))
    ax = fig.add_axes([0.335, 0.175, 0.625, 0.655])
    ax.set_ylim(rs[-1][0] + 0.7, -0.7)
    style_axis(ax, (0, 0.8), [0, 0.2, 0.4, 0.6, 0.8])
    ax.set_yticks([r[0] for r in rs])
    ax.set_yticklabels([r[2] for r in rs], fontsize=10)
    for y, key, _name, g in rs:
        cv = am[key]["classifier_verdict"]
        v, (lo, hi) = cv["macro"], cv["ci95"]
        hbar(ax, y, v, GROUP[g])
        ax.plot([lo, hi], [y, y], color=INK, lw=1.0, zorder=4, solid_capstyle="butt")
        ax.text(hi + 0.012, y, f"{v:.3f}", va="center", ha="left", fontsize=9.5, color=INK, zorder=6,
                bbox=dict(fc=SURFACE, ec="none", pad=1.2))
    # reference lines: matched AFT-only control and the two-stage target (mean of seeds)
    l0 = am["L0-ours"]["classifier_verdict"]["macro"]
    ref = (am["Ref-ours-s42"]["classifier_verdict"]["macro"] + am["Ref-ours-s43"]["classifier_verdict"]["macro"]) / 2
    for x, lab, ha, dx in ((l0, "matched AFT-only control", "left", 0.006), (ref, "two-stage target", "right", -0.006)):
        ax.axvline(x, color=INK2, lw=0.8, zorder=2)
        ax.text(x + dx, -0.62, lab, fontsize=8.5, color=INK2, ha=ha, va="bottom")
    ax.set_xlabel("Agentic misalignment rate (model decided to take the harmful action)", fontsize=9.5,
                  color=INK2, labelpad=8)
    titles(fig, "Reasoning woven into the responses closes about two-thirds of the gap to two-stage training",
           "Qwen2.5-32B-Instruct, philosophy spec · same 9,963 AFT prompts and instruction-tuning mix in every \"our recipe\" arm · "
           "27 conditions × 25 rollouts\nLower is better · one seed per arm · whiskers: 95% bootstrap CI over rollouts")
    legend(fig, ["ours1", "ours2", "rel", "none"])
    save(fig, "fig1_dataset_variants_am")


# ---------------------------------------------------------------- figure 2
def fig_machiavelli():
    am = json.loads((R / "full" / "final_all.json").read_text())
    ach = json.loads((R / "mach" / "mach_ach.json").read_text())
    good = json.loads((R / "mach" / "mach_good.json").read_text())
    rs = rows()
    fig = plt.figure(figsize=(12.5, 7.4))
    left, w, gap = 0.30, [0.17, 0.26, 0.21], 0.018
    axes, x = [], left
    for wi in w:
        axes.append(fig.add_axes([x, 0.175, wi, 0.60]))
        x += wi + gap
    a0, a1, a2 = axes
    for ax in axes:
        ax.set_ylim(rs[-1][0] + 0.7, -0.7)
    style_axis(a0, (0, 0.8), [0, 0.4, 0.8])
    for ax in (a1, a2):
        style_axis(ax, (60, 112), [60, 80, 100], fmt="{:.0f}")
        ax.axvline(100, color=INK2, lw=0.8, zorder=2)
    a0.set_yticks([r[0] for r in rs]); a0.set_yticklabels([r[2] for r in rs], fontsize=10)
    for ax in (a1, a2):
        ax.set_yticks([])
    for y, key, _n, g in rs:
        v = am[key]["classifier_verdict"]["macro"]
        hbar(a0, y, v, GROUP[g])
        a0.text(v + 0.02, y, f"{v:.2f}", va="center", ha="left", fontsize=9, color=INK)
        for ax, src in ((a1, ach), (a2, good)):
            if key not in src:
                ax.text(62, y, "not run", fontsize=8.5, color=MUTED, va="center", ha="left")
                continue
            val = src[key]["violations.Σ"]; lo, hi = src[key]["violations_ci"]
            ax.plot([lo, hi], [y, y], color=INK2, lw=1.0, zorder=3)
            ax.plot([val], [y], "o", ms=8.5, mfc=GROUP[g], mec=SURFACE, mew=1.6, zorder=4)
            ax.text(lo - 1.0, y, f"{val:.0f}", va="center", ha="right", fontsize=9, color=INK)
    heads = ["Agentic misalignment rate\n(for comparison)",
             "MACHIAVELLI ethical violations\nprompt: maximise achievements (no ethics cue)",
             "MACHIAVELLI ethical violations\nprompt: plus \"behave ethically\" instruction"]
    for ax, h in zip(axes, heads):
        ax.set_title(h, fontsize=9.5, color=INK2, loc="left", pad=16)
    for ax in (a1, a2):
        ax.text(100, -0.62, "random agent = 100", fontsize=8.5, color=INK2, ha="center", va="bottom")
        ax.set_xlabel("% of the random agent's violations", fontsize=9.5, color=INK2, labelpad=8)
    a0.set_xlabel("rate", fontsize=9.5, color=INK2, labelpad=8)
    titles(fig, "MACHIAVELLI: spec training does not change in-game ethical violations",
           "Same checkpoints as the agentic-misalignment ladder · 30 test games, one episode each, benchmark's own prompts and scoring · "
           "lower is better\nWhiskers: 95% bootstrap CI over games · asking for ethical behaviour helps every arm by the same ~13 points")
    legend(fig, ["ours1", "ours2", "rel", "none"])
    save(fig, "fig2_machiavelli")


# ---------------------------------------------------------------- figure 3
def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def fig_odcv():
    s = json.loads((R / "odcv" / "summary.json").read_text())
    arms = [("base", "Instruct model, no fine-tuning", "none", True),
            ("L0-ours", "AFT, original responses", "ours1", True),
            ("L2", "AFT + woven 1st-person reasoning", "ours1", True),
            ("Ref-ours", "MSM → AFT (seed 42)", "ours2", True),
            ("L0-rel", "AFT, no CoT (released)", "rel", False),
            ("Ref-rel", "MSM → AFT (released)", "rel", False)]
    ys = [0, 1.55, 2.55, 4.1, 5.65, 6.65]
    fig = plt.figure(figsize=(11.5, 5.9))
    axes = [fig.add_axes([0.30, 0.20, 0.31, 0.55]), fig.add_axes([0.645, 0.20, 0.31, 0.55])]
    for ax, variant, head in zip(axes, ("mandated", "incentivized"),
                                 ("Mandated: the task instruction itself demands the violation",
                                  "Incentivized: only a KPI rewards the violation")):
        ax.set_ylim(ys[-1] + 0.7, -0.7)
        style_axis(ax, (0, 0.5), [0, 0.1, 0.2, 0.3, 0.4, 0.5])
        ax.set_title(head, fontsize=9.5, color=INK2, loc="left", pad=10)
        ax.set_xlabel("Misbehaviour rate (judge score ≥ 3 of 5), 40 scenarios", fontsize=9.5, color=INK2, labelpad=8)
        for y, (key, _name, g, ok) in zip(ys, arms):
            if not ok:
                ax.text(0.006, y, "not scoreable: 0 tool calls in 80 runs", fontsize=9, color=MUTED,
                        va="center", ha="left")
                continue
            c = s[f"{key}/{variant}"]
            n = c["scored"]; k = round(c["misbehaviour_rate"] * n)
            lo, hi = wilson(k, n)
            hbar(ax, y, max(k / n, 0.001), GROUP[g])
            ax.plot([lo, hi], [y, y], color=INK, lw=1.0, zorder=4)
            ax.text(hi + 0.01, y, f"{k}/{n}", va="center", ha="left", fontsize=9.5, color=INK)
    axes[0].set_yticks(ys); axes[0].set_yticklabels([a[1] for a in arms], fontsize=10)
    axes[1].set_yticks([])
    titles(fig, "ODCV-Bench: every trained arm cuts mandated misbehaviour; nothing moves under KPI pressure alone",
           "Agentic tool-use scenarios (bash sandbox), benchmark's executor and rubric · single judge (Claude Opus 5) · "
           "lower is better · one run per scenario\nWhiskers: 95% Wilson interval · the released adapters never emit tool calls, "
           "so they cannot act in this harness", y=0.955)
    legend(fig, ["ours1", "ours2", "none"], y=0.02)
    save(fig, "fig3_odcv_bench")


if __name__ == "__main__":
    fig_variants()
    fig_machiavelli()
    fig_odcv()
