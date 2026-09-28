"""Figures for the MACHIAVELLI decision-pool analysis (REPORT_OOD.md §1e). Reads only results/.

    python -m tda.analysis.mach_pool_figures      # -> results/aft/figures/fig4_*.png, fig5_*.png (+ .pdf)

Figure 4  the ladder game → re-voiced → transposed → transposed + policy, per checkpoint
Figure 5  where the rollout score comes from (forced vs chosen), and per-category change

Same encoding as tda/analysis/meeting_figures.py: hue = training group, text in
ink, every mark direct-labelled, legend present.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch

R = Path("results/aft")
OUT = R / "figures"
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8880", "#e7e6e1"
GROUP = {"ours1": "#2a78d6", "ours2": "#eb6834", "rel": "#1baf7a", "none": "#9a988f"}
GROUP_LABEL = {"ours1": "Single-stage AFT, our recipe", "ours2": "Two-stage MSM → AFT, our AFT stage",
               "rel": "Released checkpoints (authors' recipe)", "none": "No spec training"}
ARMS = [("Base", "Instruct model, no fine-tuning", "none"),
        ("id-baseline", "Instruction-tuning mix only (released)", "none"),
        None,
        ("L0-rel", "AFT, no CoT (released)", "rel"),
        ("Ref-rel", "MSM → AFT (released)", "rel"),
        None,
        ("L0-ours", "AFT, original responses", "ours1"),
        ("L2", "AFT + woven 1st-person reasoning", "ours1"),
        ("L3", "AFT + woven reasoning + value attribution", "ours1"),
        None,
        ("Ref-ours-s42", "MSM → AFT, seed 42", "ours2"),
        ("Ref-ours-s43", "MSM → AFT, seed 43", "ours2")]
CONDS = [("game", "Original game\n(you are the character)"),
         ("revoiced", "Re-voiced game\n(assistant picks for a user's character)"),
         ("transposed", "Transposed\n(real deployment of an AI assistant)"),
         ("transposed_policy", "Transposed + operator policy\n(\"do not deceive, steal from, or harm\")")]

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 10,
    "axes.edgecolor": GRID, "axes.linewidth": 0.8, "xtick.color": INK2, "ytick.color": INK,
    "text.color": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "pdf.fonttype": 42})


def rows():
    out, y = [], 0.0
    for a in ARMS:
        if a is None:
            y += 0.55
            continue
        out.append((y, *a)); y += 1.0
    return out


def style(ax, xlim, ticks, fmt="{:.1f}"):
    ax.set_xlim(*xlim); ax.set_xticks(ticks); ax.set_xticklabels([fmt.format(t) for t in ticks], fontsize=9)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID); ax.tick_params(axis="both", length=0)


def hbar(ax, y, v, color, h=0.62):
    fig = ax.figure; fig.canvas.draw()
    (px0, py0), (px1, py1) = ax.transData.transform([(0, 0), (1, 1)])
    sx, sy = abs(px1 - px0), abs(py1 - py0)
    r = min(4.0 * fig.dpi / 100 / sx, max(v, 1e-9) / 2)
    ax.add_patch(FancyBboxPatch((-3 * r, y - h / 2), v + 3 * r, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                mutation_aspect=sx / sy, fc=color, ec="none", zorder=3, clip_on=True))


def titles(fig, title, sub, y=0.965):
    fig.text(0.03, y, title, fontsize=15, fontweight="bold", ha="left", va="top")
    fig.text(0.03, y - 0.052, sub, fontsize=10, ha="left", va="top", color=INK2)


def legend(fig, groups, y=0.03):
    fig.legend(handles=[Patch(fc=GROUP[g], ec="none", label=GROUP_LABEL[g]) for g in groups], loc="lower left",
               bbox_to_anchor=(0.03, y), ncol=len(groups), frameon=False, fontsize=9, handlelength=1.1,
               handleheight=1.0, columnspacing=1.6, labelcolor=INK2)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=220); fig.savefig(OUT / f"{name}.pdf"); plt.close(fig)
    print("wrote", OUT / f"{name}.png")


def fig_ladder():
    res = json.loads((R / "mach" / "pool_eval" / "pool_v1_all.json").read_text())["keys"]["violations.Σ"]
    rs = rows()
    fig = plt.figure(figsize=(14, 7.2))
    left, w, gap = 0.255, 0.168, 0.016
    for k, (c, head) in enumerate(CONDS):
        ax = fig.add_axes([left + k * (w + gap), 0.17, w, 0.60])
        ax.set_ylim(rs[-1][0] + 0.7, -0.7)
        style(ax, (0, 0.5), [0, 0.2, 0.4])
        ax.set_title(head, fontsize=9.5, color=INK2, loc="left", pad=16)
        cell = res[c]
        ax.axvline(cell["uniform"], color=INK2, lw=0.8, zorder=2)
        ax.text(cell["uniform"] - 0.006, -0.62, "random chooser", fontsize=8.5, color=INK2, ha="right", va="bottom")
        for y, key, _name, g in rs:
            e = cell["arms"].get(key)
            if e is None:
                ax.text(0.01, y, "not run", fontsize=8.5, color=MUTED, va="center"); continue
            hbar(ax, y, e["score"], GROUP[g])
            sig = "ci" in e and (e["ci"][0] > 0 or e["ci"][1] < 0)
            ax.text(e["score"] + 0.012, y, f"{e['score']:.3f}" + ("*" if sig else ""), va="center", ha="left",
                    fontsize=9, color=INK, zorder=6, bbox=dict(fc=SURFACE, ec="none", pad=1.0))
        if k == 0:
            ax.set_yticks([r[0] for r in rs]); ax.set_yticklabels([r[2] for r in rs], fontsize=10)
        else:
            ax.set_yticks([])
        ax.set_xlabel(f"n = {cell['n']:,} decision points", fontsize=9, color=INK2, labelpad=8)
    titles(fig, "Spec training shows up once the decision stops being fiction, and not before",
           "Probability of choosing an option the benchmark labels an ethical violation, at decision points where a clean option exists · lower is better\n"
           "Same decision points for every checkpoint within a panel · * = differs from the untrained model (95% CI, resampling games) · decoding temperature 0.2")
    legend(fig, ["ours1", "ours2", "rel", "none"])
    save(fig, "fig4_machiavelli_ladder")


def fig_metric():
    dec = json.loads((R / "mach" / "decomposition.json").read_text())["Base"]
    res = json.loads((R / "mach" / "pool_eval" / "pool_v1_all.json").read_text())["keys"]
    core = ["L0-ours", "L2", "L3", "Ref-ours-s42", "Ref-ours-s43"]
    cats = [k for k in res if k.startswith("violations.") and "transposed" in res[k] and "game" in res[k]]
    cats.sort(key=lambda k: -res[k]["transposed"]["n"])
    fig = plt.figure(figsize=(13, 6.6))
    # left: decomposition of the rollout score
    ax = fig.add_axes([0.20, 0.17, 0.22, 0.58])
    parts = [("No clean option existed\n(every option leads to a violation)", dec["viol_all_violate"] + dec["viol_no_choice"] + dec["initial"]),
             ("A clean option was available", dec["viol_contrast"])]
    tot = sum(v for _, v in parts)
    style(ax, (0, 1.0), [0, 0.5, 1.0], fmt="{:.0%}")
    ax.set_ylim(1.7, -0.7); ax.set_yticks([0, 1]); ax.set_yticklabels([p[0] for p in parts], fontsize=9.5)
    for y, (_n, v) in enumerate(parts):
        hbar(ax, y, v / tot, GROUP["none"], h=0.5)
        ax.text(v / tot + 0.03, y, f"{v / tot:.0%}  ({v:,})", va="center", ha="left", fontsize=9.5)
    ax.set_title("Where the benchmark's score comes from\nuntrained model, 30 playthroughs, 2,447 counted violations",
                 fontsize=9.5, color=INK2, loc="left", pad=12)
    ax.set_xlabel("share of counted violations", fontsize=9, color=INK2, labelpad=8)
    # right: per-category relative change, trained mean vs Base, game vs transposed
    ax = fig.add_axes([0.60, 0.17, 0.37, 0.58])
    style(ax, (-0.7, 0.2), [-0.6, -0.4, -0.2, 0, 0.2], fmt="{:+.0%}")
    ax.axvline(0, color=INK2, lw=0.8, zorder=2)
    ax.set_ylim(len(cats) - 0.3, -0.7)
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels([k.split(".", 1)[1].replace("_", " ").replace("Σ", "all violations") for k in cats], fontsize=9.5)
    for y, k in enumerate(cats):
        for cond, col, dy in (("game", GROUP["none"], -0.17), ("transposed", GROUP["ours1"], 0.17)):
            b = res[k][cond]["arms"]["Base"]["score"]
            m = sum(res[k][cond]["arms"][a]["score"] for a in core) / len(core)
            rel = m / b - 1
            ax.plot([0, rel], [y + dy, y + dy], color=col, lw=5, solid_capstyle="butt", zorder=3)
            ax.text(rel + (-0.012 if rel < 0 else 0.012), y + dy, f"{rel:+.0%}", va="center",
                    ha="right" if rel < 0 else "left", fontsize=8.5, color=INK)
    ax.set_title("Change in the chance of choosing a violating option, by category\n"
                 "mean of the five checkpoints we trained, relative to the untrained model",
                 fontsize=9.5, color=INK2, loc="left", pad=12)
    ax.set_xlabel("relative change (negative = fewer violations)", fontsize=9, color=INK2, labelpad=8)
    fig.legend(handles=[Patch(fc=GROUP["none"], ec="none", label="Original game"),
                        Patch(fc=GROUP["ours1"], ec="none", label="Transposed to a real AI-assistant deployment")],
               loc="lower left", bbox_to_anchor=(0.60, 0.03), ncol=2, frameon=False, fontsize=9,
               handlelength=1.1, labelcolor=INK2)
    titles(fig, "Two-thirds of the MACHIAVELLI score is not a choice; the rest moves only outside the game",
           "Left: the benchmark counts violations along a playthrough whether or not the player could have avoided them · "
           "Right: fixed decision points, identical for every checkpoint", y=0.955)
    save(fig, "fig5_machiavelli_metric")


if __name__ == "__main__":
    fig_ladder()
    fig_metric()
