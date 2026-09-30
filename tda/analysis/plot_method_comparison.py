"""Method-comparison figures: removal-and-retrain at 8B, attribution at 32B.

Reads only from `results/prof_figs/data/` (CLAUDE.md section 6):

  arms_cheese8b.csv     every cheese-8B removal arm's generative aligned rate,
                        collated from msm-tda-results:/bergson/cheese/generative/
  item_matrix_cheese8b.csv  45 models x 200 items, 1 = answered value-aligned
  register_phil32b.json  tail words and first-person density (tda/analysis/register_tails.py)
  compare_phil32b.json  EK-FAC vs grad-dot statistics over 13,201 philosophy docs
  transcribed.json      numbers copied from project docs, each with its source

Run:  uv run --no-project --with matplotlib --with pandas --with numpy \
          python tda/analysis/plot_method_comparison.py

POLARITY. Arms are keyed `opponents` / `proponents`, never top/bottom
(CLAUDE.md 5.1). Nothing here sorts a score array; directions come from the run
names as corrected in STATUS.md 7.1. For cheese the query is the value-ALIGNED
answer, so removing proponents should LOWER the aligned rate. For philosophy the
query is the MISALIGNED action, so a negative score is protective.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2] / "results" / "prof_figs"
DATA = ROOT / "data"

# Palette: reference categorical slots (light surface). Direction uses slots 1-2,
# which validate all-pairs; method identity at 32B uses aqua/violet plus marker
# shape as the secondary channel. Text always wears ink, never a series colour.
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
OPP, PRO, CTRL = BLUE, ORANGE, "#898781"

mpl.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2, "xtick.major.size": 0, "ytick.major.size": 0,
    "axes.grid": False, "legend.frameon": False, "pdf.fonttype": 42,
})


def style(ax, grid="x"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def header(fig, title, subtitle, y=0.975, x=0.02):
    fig.text(x, y, title, fontsize=14, fontweight="bold", va="top", ha="left")
    fig.text(x, y - 0.047, subtitle, fontsize=10, color=INK2, va="top", ha="left")


def footer(fig, text, x=0.02):
    fig.text(x, 0.012, text, fontsize=8, color=MUTED, va="bottom", ha="left")


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / f"{name}.{ext}", dpi=220)
    plt.close(fig)
    print("wrote", ROOT / f"{name}.png")


def dot(ax, x, y, color, size=9, marker="o", z=5):
    """A filled marker with the 2px surface ring."""
    ax.plot(x, y, marker, ms=size, mfc=color, mec=SURFACE, mew=1.6, zorder=z,
            linestyle="none")


# --------------------------------------------------------------------------- #
# Figure 1 - 8B removal scoreboard + control-free directional gap
# --------------------------------------------------------------------------- #

METHODS = [  # (key in csv, label, family)
    ("ekfac", "EK-FAC", "Influence functions"),
    ("graddot", "grad-dot", "Influence functions"),
    ("source", "SOURCE (multi-stage)", "Influence functions"),
    ("icl2_qa", "ICL v2 (log-odds margin)", "In-context scores"),
    ("icl_marginal", "Marginal ICL", "In-context scores"),
    ("icl", "ICL v1 (decision rate)", "In-context scores"),
]


def fig_scoreboard(df, base):
    k = df[df.k == 640]
    ctrl = k[k.method == "random"].rate.to_numpy()
    cm, cs = ctrl.mean(), ctrl.std(ddof=1)
    longest = float(k[k.method == "longest"].rate.iloc[0])

    # rows, top to bottom: control, 6 methods, representation (no arm), longest
    labels = ["Random removal (6 arms)"] + [m[1] for m in METHODS] + \
             ["Representation (S3 value direction)", "Longest documents (null criterion)"]
    ys = np.arange(len(labels))[::-1].astype(float)
    ypos = dict(zip(labels, ys))

    fig = plt.figure(figsize=(12.4, 6.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.75, 1], left=0.235, right=0.975,
                          top=0.76, bottom=0.155, wspace=0.09)
    ax, bx = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    # --- left: aligned rate after removal
    style(ax)
    ax.axvspan(cm - cs, cm + cs, color="#ecebe6", zorder=0.5, lw=0)
    ax.axvline(cm, color=CTRL, lw=1.2, zorder=1)
    ax.axvline(base, color=INK2, lw=1.0, zorder=1)
    top = ys.max() + 0.62
    ax.text(cm - 0.002, top, f"random-removal mean {cm:.3f} ± {cs:.3f} (sd)", fontsize=8.5,
            color=INK2, ha="right", va="bottom")
    ax.text(base + 0.002, top, f"no removal {base:.3f}", fontsize=8.5, color=INK2,
            ha="left", va="bottom")

    y0 = ypos["Random removal (6 arms)"]
    seen = {}  # stack tied draws so all six stay visible
    for c in sorted(ctrl):
        n_seen = seen.get(round(c, 4), 0)
        seen[round(c, 4)] = n_seen + 1
        ax.plot(c, y0 + 0.2 * n_seen, "o", ms=7, mfc=CTRL, mec=SURFACE, mew=1.4, zorder=4)

    off = 0.17
    for key, label, _ in METHODS:
        y = ypos[label]
        for direction, col, dy in (("opponents", OPP, off), ("proponents", PRO, -off)):
            r = k[(k.method == key) & (k.direction == direction)].rate.to_numpy()
            if len(r) > 1:
                ax.plot([r.min(), r.max()], [y + dy] * 2, color=col, lw=2, alpha=0.45,
                        zorder=3, solid_capstyle="round")
                ax.plot(r, [y + dy] * len(r), "o", ms=4.5, mfc=col, mec="none", alpha=0.55,
                        zorder=4)
            dot(ax, r.mean(), y + dy, col, size=10)
    # selective direct labels: only the one arm that clears the control
    e = k[(k.method == "ekfac") & (k.direction == "opponents")].rate
    z = (e.mean() - cm) / np.sqrt(cs**2 / 6 + e.var(ddof=1) / 3)
    ax.text(0.612, ypos["EK-FAC"] + 0.60, f"+{e.mean() - cm:.3f} vs control (z = {z:.1f}): the only arm"
            "\noutside the control's own spread", fontsize=8.5, color=INK2, va="center")

    yr = ypos["Representation (S3 value direction)"]
    ax.text(0.507, yr, "no completed removal arm: the proponents run was preempted and the opponents "
            "run refused\n(Modal spend cap, 10 Sep). Figure 3 gives the retrain-free screen instead.",
            fontsize=8.5, color=MUTED, va="center", style="italic")
    dot(ax, longest, ypos["Longest documents (null criterion)"], CTRL, size=10, marker="D")

    ax.set_yticks(ys, labels)
    ax.set_ylim(ys.min() - 0.6, ys.max() + 0.6)
    ax.set_xlim(0.505, 0.685)
    ax.set_xlabel("value-aligned decision rate after removing 640 of 6,400 midtraining documents\n"
                  "and retraining MSM → AFT (200 held-out items, greedy, parse rate 1.00 in every arm)",
                  fontsize=9)
    ax.tick_params(axis="y", length=0, pad=6)
    ax.tick_params(axis="y", labelsize=10)
    for t in ax.get_yticklabels():
        t.set_color(INK)

    # family brackets in the far-left margin
    for fam in ("Influence functions", "In-context scores"):
        rows = [ypos[m[1]] for m in METHODS if m[2] == fam]
        ax.annotate("", xy=(-0.375, min(rows) - 0.3), xytext=(-0.375, max(rows) + 0.3),
                    xycoords=("axes fraction", "data"), textcoords=("axes fraction", "data"),
                    arrowprops=dict(arrowstyle="-", color=AXIS, lw=1.2), annotation_clip=False)
        ax.text(-0.39, np.mean(rows), fam.replace(" ", "\n"), transform=ax.get_yaxis_transform(),
                fontsize=8.5, color=MUTED, ha="right", va="center", clip_on=False)

    # --- right: opponents minus proponents, same training seed
    style(bx)
    bx.axvline(0, color=AXIS, lw=1.0, zorder=1)
    for key, label, _ in METHODS:
        y = ypos[label]
        sub = k[k.method == key]
        o = sub[sub.direction == "opponents"].set_index("seed").rate
        p = sub[sub.direction == "proponents"].set_index("seed").rate
        gaps = (o - p).dropna()
        g = gaps.mean()
        bx.plot([0, g], [y, y], color=INK2, lw=2, zorder=3, solid_capstyle="round")
        if len(gaps) > 1:
            bx.plot(gaps.values, [y] * len(gaps), "o", ms=5, mfc="none", mec=INK2, mew=1.1,
                    zorder=4)
        dot(bx, g, y, INK2, size=10)
        bx.text(max(gaps.max(), g) + 0.008, y, f"{g:+.3f}   n={len(gaps)}", fontsize=9,
                color=INK2, va="center")
    bx.set_yticks(ys, [""] * len(ys))
    bx.set_ylim(*ax.get_ylim())
    bx.set_xlim(-0.035, 0.175)
    bx.set_xlabel("directional gap: opponents-removed rate\nminus proponents-removed rate, "
                  "paired by seed", fontsize=9)
    bx.text(0.0, top, "needs no control: > 0 if the score reads the sign", fontsize=8.5, color=INK2,
            ha="left", va="bottom")

    # legend (always present for >= 2 series)
    h = [mpl.lines.Line2D([], [], marker="o", ls="none", ms=9, mfc=OPP, mec=SURFACE, mew=1.5,
                          label="strongest opponents removed (rate should rise)"),
         mpl.lines.Line2D([], [], marker="o", ls="none", ms=9, mfc=PRO, mec=SURFACE, mew=1.5,
                          label="strongest proponents removed (rate should fall)"),
         mpl.lines.Line2D([], [], marker="o", ls="none", ms=7, mfc=CTRL, mec=SURFACE, mew=1.2,
                          label="random / null control"),
         mpl.lines.Line2D([], [], marker="o", ls="none", ms=4.5, mfc=MUTED, mec="none",
                          label="individual training seed (large dot = mean)")]
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.015, 0.875), ncol=4, fontsize=9,
               handletextpad=0.3, columnspacing=1.6)

    header(fig, "8B removal-and-retrain: only EK-FAC's opponent arm separates from random removal",
           "Llama-3.1-8B, pro-America midtraining corpus. Each method ranks the 6,400 documents; "
           "we drop its top 640 in one direction, retrain both stages, and re-measure behaviour.")
    footer(fig, f"Source: {len(k)} retrained arms at k=640 (msm-tda-results:/bergson/cheese/generative). "
           "EK-FAC, SOURCE and marginal-ICL proponents have 3 training seeds; every other arm has 1.\n"
           "A single-seed difference under about 0.12 is not resolvable (ICL_LOG 7.5). Open circles "
           "on the right are per-seed gaps.")
    save(fig, "fig1_8b_removal_scoreboard")


# --------------------------------------------------------------------------- #
# Figure 2 - 8B dose-response in k
# --------------------------------------------------------------------------- #

def fig_ksweep(df, base):
    s42 = df[(df.seed == 42) & (df.draw == "default")]
    ctrl640 = df[(df.k == 640) & (df.method == "random")].rate
    ks = [64, 320, 640, 1280]

    def series(method, direction):
        return [float(s42[(s42.method == method) & (s42.direction == direction) & (s42.k == kk)]
                      .rate.iloc[0]) for kk in ks]

    rnd = series("random", "control")
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.4), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.87, top=0.74, bottom=0.17, wspace=0.07)
    for ax, (key, name) in zip(axes, (("ekfac", "EK-FAC"), ("source", "SOURCE (multi-stage)"))):
        style(ax, grid="y")
        ax.axhline(base, color=INK2, lw=0.9, zorder=1)
        ax.plot([640, 640], [ctrl640.min(), ctrl640.max()], color=CTRL, lw=5, alpha=0.28,
                solid_capstyle="round", zorder=2)
        ax.plot(ks, rnd, color=CTRL, lw=2, zorder=3)
        for x, v in zip(ks, rnd):
            dot(ax, x, v, CTRL, size=7, z=4)
        ends = {}
        for direction, col in (("opponents", OPP), ("proponents", PRO)):
            v = series(key, direction)
            ax.plot(ks, v, color=col, lw=2, zorder=5, solid_joinstyle="round")
            for x, r in zip(ks, v):
                dot(ax, x, r, col, size=8.5, z=6)
            ends[direction] = v[-1]
        ax.set_xscale("log")
        ax.set_xticks(ks, [f"{kk}\n({kk / 64:.0f}%)" for kk in ks])
        ax.minorticks_off()
        ax.set_xlim(48, 1700)
        ax.set_title(name, loc="left", fontsize=11, fontweight="bold", color=INK, pad=8)
        ax.set_xlabel("documents removed, k  (share of the 6,400-document corpus)", fontsize=9)
        if ax is axes[-1]:  # direct end labels, right panel only - legend carries the rest
            ax.text(1400, ends["opponents"], " opponents\n removed", fontsize=9, color=INK2,
                    va="center", clip_on=False)
            ax.text(1400, ends["proponents"], " proponents\n removed", fontsize=9, color=INK2,
                    va="center", clip_on=False)
            ax.text(1400, rnd[-1], " random\n removal", fontsize=9, color=INK2, va="center",
                    clip_on=False)
            ax.text(1400, base + 0.004, " no removal", fontsize=9, color=INK2, va="bottom",
                    clip_on=False)
    axes[0].set_ylabel("value-aligned decision rate", fontsize=9.5)
    axes[0].set_ylim(0.505, 0.665)
    axes[0].text(640, ctrl640.max() + 0.004, "spread of 6 random\narms at k=640", fontsize=8,
                 color=MUTED, ha="center", va="bottom")

    h = [mpl.lines.Line2D([], [], color=OPP, lw=2, marker="o", ms=7, mec=SURFACE,
                          label="strongest opponents removed"),
         mpl.lines.Line2D([], [], color=PRO, lw=2, marker="o", ms=7, mec=SURFACE,
                          label="strongest proponents removed"),
         mpl.lines.Line2D([], [], color=CTRL, lw=2, marker="o", ms=6, mec=SURFACE,
                          label="random removal (same in both panels)")]
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.06, 0.865), ncol=3, fontsize=9)
    header(fig, "8B dose-response: removing more proponents lowers alignment monotonically",
           "Both methods' proponent arms fall at every step of k, their opponent arms end above "
           "where they start, and random removal does neither.", y=0.97)
    footer(fig, "Source: 20 retrained arms, training seed 42, one arm per point; generative "
           "decision rate on 200 held-out items. The two methods' proponent sets are nearly "
           "disjoint (Jaccard 0.009, ICL_LOG 11.5).")
    save(fig, "fig2_8b_dose_response")


# --------------------------------------------------------------------------- #
# Figure 3 - retrain-free screen: does a scorer predict arms others chose?
# --------------------------------------------------------------------------- #

def fig_prediction(tr):
    rows = tr["outcome_prediction"]["rows"]
    fam_col = {"Influence functions": BLUE, "Representations": ORANGE,
               "In-context scores": AQUA, "Null features": CTRL}
    fam_mark = {"Influence functions": "o", "Representations": "s",
                "In-context scores": "^", "Null features": "D"}
    fig, ax = plt.subplots(figsize=(10.4, 6.4))
    fig.subplots_adjust(left=0.30, right=0.865, top=0.75, bottom=0.17)
    style(ax)
    y, yt, yl, last = 0.0, [], [], None
    for name, fam, rho, n in rows:
        if fam != last:
            y -= 0.75 if last else 0
            ax.text(-0.445, y, fam, transform=ax.get_yaxis_transform(), fontsize=9,
                    fontweight="bold", color=INK2, ha="left", va="center", clip_on=False)
            y -= 0.85
            last = fam
        v = -rho  # flip: higher = removing high-scored docs lowers alignment more
        z, se = np.arctanh(v), np.sqrt((1 + v * v / 2) / (n - 3))  # Bonett-Wright
        lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
        ax.plot([lo, hi], [y, y], color=fam_col[fam], lw=2, alpha=0.4, zorder=3,
                solid_capstyle="round")
        dot(ax, v, y, fam_col[fam], size=9.5, marker=fam_mark[fam])
        ax.text(0.905, y, f"{v:+.2f}  (n={n})", fontsize=8.5, color=INK2, va="center",
                clip_on=False)
        yt.append(y); yl.append(name)
        y -= 1
    ax.axvline(0, color=AXIS, lw=1.0, zorder=1)
    ax.set_yticks(yt, yl)
    for t in ax.get_yticklabels():
        t.set_color(INK)
    ax.set_xlim(-0.8, 0.88)
    ax.set_ylim(y + 0.3, 0.2)
    ax.set_xlabel("Spearman ρ between the mean score of a removed set and the resulting DROP in "
                  "aligned rate\nhigher = the score predicts what retraining does; bars are "
                  "approximate 95% intervals", fontsize=9)
    ax.text(0.02, 0.35, "predicts the right direction →", fontsize=8.5, color=INK2, va="bottom")
    h = [mpl.lines.Line2D([], [], marker=fam_mark[f], ls="none", ms=8.5, mfc=fam_col[f],
                          mec=SURFACE, mew=1.4, label=f) for f in fam_col]
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.015, 0.852), ncol=4, fontsize=9,
               handletextpad=0.3)
    header(fig, "8B retrain-free screen: one-forward-pass representation scores predict removal "
           "outcomes\nabout as well as the gradient methods",
           "", y=0.975)
    fig.text(0.02, 0.885, "Each scorer is asked to predict the measured outcome of removal arms "
             "whose sets were chosen by OTHER methods (own sets excluded, all k).",
             fontsize=10, color=INK2, va="top")
    footer(fig, "Source: values transcribed from SEMANTIC_REPORT 2.5, not recomputed here. The 31–43 "
           "arms share seeds and sets, so the intervals are optimistic.\nNo representation scorer has "
           "a completed removal arm of its own. Document length points the wrong way: removing longer "
           "documents RAISES alignment.")
    save(fig, "fig3_8b_outcome_prediction")


# --------------------------------------------------------------------------- #
# Figure 4 - 32B philosophy: attribution only (EK-FAC, grad-dot)
# --------------------------------------------------------------------------- #

def fig_32b(cp, tr):
    fig = plt.figure(figsize=(12.6, 8.6))
    gs = fig.add_gridspec(2, 2, left=0.285, right=0.965, top=0.775, bottom=0.145,
                          hspace=0.66, wspace=0.62, width_ratios=[1.25, 1])
    EK, GD = AQUA, VIOLET

    # (a) behaviour: does our retrained pipeline reproduce the effect?
    ax = fig.add_subplot(gs[0, 0]); style(ax)
    rows = tr["am_rates_32b"]["rows"]
    yy = np.arange(len(rows))[::-1]
    for (name, r, sem), y in zip(rows, yy):
        ax.barh(y, r, height=0.42, color=INK2 if "Ours" not in name else EK, zorder=3)
        ax.plot([r - sem, r + sem], [y, y], color=INK, lw=1.2, zorder=4)
        ax.text(r + sem + 0.012, y, f"{r:.3f}", fontsize=9.5, color=INK, va="center")
    ax.set_yticks(yy, [r[0].replace(" (", "\n(") for r in rows], fontsize=9)
    for t in ax.get_yticklabels():
        t.set_color(INK)
    ax.set_xlim(0, 0.78); ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("agentic-misalignment rate (± SEM)", fontsize=9)
    ax.set_title("a  The one 32B retrain: our MSM → AFT pipeline reproduces the effect",
                 loc="left", fontsize=10.5, fontweight="bold", x=-0.62, pad=10)

    # (b) null control: fraction of rows scored as opponents of the misaligned action
    bx = fig.add_subplot(gs[0, 1]); style(bx)
    nc = cp["null_control"]
    for i, (m, col, mk) in enumerate((("EK-FAC", EK, "o"), ("grad-dot", GD, "s"))):
        y = 1 - i
        a, b = nc[m]["frac_negative_aft"], nc[m]["frac_negative_msm"]
        bx.plot([a, b], [y, y], color=col, lw=2, alpha=0.45, zorder=3)
        bx.plot(a, y, mk, ms=9, mfc=SURFACE, mec=col, mew=2, zorder=5)
        dot(bx, b, y, col, size=10, marker=mk)
        bx.text(b + 0.022, y, f"|score| {nc[m]['ratio_msm_over_aft']:.1f}× larger", fontsize=8.5,
                color=INK2, va="center")
    bx.axvline(0.5, color=AXIS, lw=1.0, zorder=1)
    bx.text(0.5, 1.62, "coin flip", fontsize=8.5, color=MUTED, ha="center")
    bx.set_yticks([1, 0], ["EK-FAC", "grad-dot"]); bx.set_ylim(-0.6, 1.9)
    for t in bx.get_yticklabels():
        t.set_color(INK)
    bx.set_xlim(0.45, 0.93)
    bx.set_xlabel("share of rows with negative (protective) influence\nhollow: 1,584 AFT and "
                  "instruction rows\nfilled: 13,201 midtraining documents", fontsize=9)
    bx.set_title("b  Null control passes in size and in sign", loc="left", fontsize=10.5,
                 fontweight="bold", x=-0.27, pad=10)

    # (c) domain profile, each method scaled by its own corpus mean |score|
    cx = fig.add_subplot(gs[1, 0]); style(cx)
    dom = cp["by_domain"]
    names = sorted(dom["EK-FAC"]["mean_by_domain"], key=lambda d: dom["EK-FAC"]["mean_by_domain"][d])
    yy = np.arange(len(names))[::-1]
    for m, col, mk, dy in (("EK-FAC", EK, "o", 0.14), ("grad-dot", GD, "s", -0.14)):
        sc = nc[m]["mean_abs_msm"]
        v = [dom[m]["mean_by_domain"][d] / sc for d in names]
        for x, y in zip(v, yy):
            dot(cx, x, y + dy, col, size=8.5, marker=mk)
    cx.set_yticks(yy, names, fontsize=9)
    for t in cx.get_yticklabels():
        t.set_color(INK)
    cx.set_xlim(-0.95, 0); cx.set_ylim(-0.7, len(names) - 0.3)
    cx.set_xlabel("mean influence on the misaligned action ÷ corpus mean |score|\n"
                  "← more protective", fontsize=9)
    cx.set_title("c  Every domain is protective, but domain explains little:\n    "
                 f"{dom['EK-FAC']['eta_squared'] * 100:.1f}% of variance for EK-FAC, "
                 f"{dom['grad-dot']['eta_squared'] * 100:.1f}% for grad-dot",
                 loc="left", fontsize=10.5, fontweight="bold", x=-0.62, pad=10)

    # (d) register quartiles (EK-FAC)
    dx = fig.add_subplot(gs[1, 1]); style(dx, grid="y")
    rq = tr["register_quartiles_32b"]
    xs = np.arange(4)
    vals = [r[1] for r in rq["rows"]]
    dx.bar(xs, vals, width=0.42, color=EK, zorder=3)
    for x, v in zip(xs, vals):
        dx.text(x, v - 0.5, f"{v:.1f}", fontsize=9, color=INK, ha="center", va="top")
    dx.axhline(0, color=AXIS, lw=1.0)
    dx.set_xticks(xs, [r[0] for r in rq["rows"]], fontsize=8.5)
    dx.set_ylim(-18.5, 0.6)
    dx.set_ylabel("mean EK-FAC influence", fontsize=9)
    dx.set_xlabel(f"first-person pronouns per 1,000 words, by quartile\n"
                  f"R² {rq['r2_register']:.3f}, against {rq['r2_domain']:.3f} for domain", fontsize=9)
    dx.set_title("d  First-person documents are the protective ones",
                 loc="left", fontsize=10.5, fontweight="bold", x=-0.27, pad=10)

    h = [mpl.lines.Line2D([], [], marker="o", ls="none", ms=9, mfc=EK, mec=SURFACE, mew=1.5,
                          label="EK-FAC"),
         mpl.lines.Line2D([], [], marker="s", ls="none", ms=9, mfc=GD, mec=SURFACE, mew=1.5,
                          label="grad-dot")]
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.015, 0.885), ncol=2, fontsize=9.5,
               handletextpad=0.3)
    header(fig, "32B philosophy: attribution only. Two methods scored, no removal arm trained",
           f"Qwen2.5-32B, 13,201 midtraining documents, 256 agentic-misalignment dev queries. "
           f"EK-FAC and grad-dot agree at Spearman {cp['spearman_msm']:.2f} (0.63 at 8B).\n"
           f"SOURCE failed three times; ICL and representation scorers were not run at this scale.",
           y=0.98)
    footer(fig, "Negative influence = lowers the log-probability of the misaligned action. "
           "Panels b, c: compare_phil.json, run ekfac_phil32b_union-am-dev-full. Panels a, d: "
           "transcribed from STATUS 8.9f and 8.10c.\nPanel a's first bar covers all 27 conditions, "
           "the other two the 14 dev conditions. Panels b–d are correlational on influence scores. "
           "Removal sets are built (k = 1,320) but no 32B arm was launched.")
    save(fig, "fig4_32b_attribution")


# --------------------------------------------------------------------------- #
# Figure 0 - coverage: which method has what evidence, at which scale
# --------------------------------------------------------------------------- #

def fig_coverage():
    cols = ["8B scores\n(6,400 docs)", "8B removal arms\n(retrained)", "32B scores\n(13,201 docs)",
            "32B removal arms\n(retrained)"]
    rows = [
        ("EK-FAC", ["yes", "3 seeds × 2 directions\n+ k-sweep (4 sizes)", "yes", "none\nsets built, never launched"]),
        ("grad-dot", ["yes", "1 seed × 2 directions", "yes", "none\nsets built, never launched"]),
        ("SOURCE (multi-stage)", ["yes", "3 seeds × 2 directions\n+ k-sweep (4 sizes)", "none\n3 failed attempts", "none"]),
        ("ICL (v1, v2 margin, marginal)", ["yes", "1 seed × 2 directions each\n(marginal proponents: 3)", "not run", "none"]),
        ("Representations (S1–S3)", ["yes", "none of its own\nscreened on others' arms", "not run", "none"]),
        ("Random / length controls", ["n/a", "6 random arms at k=640\n+ 3 other sizes, 1 longest", "n/a", "none"]),
    ]
    fig, ax = plt.subplots(figsize=(11.6, 5.2))
    fig.subplots_adjust(left=0.215, right=0.985, top=0.70, bottom=0.07)
    ax.set_xlim(0, 4); ax.set_ylim(0, len(rows)); ax.invert_yaxis()
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    for j, c in enumerate(cols):
        ax.text(j + 0.5, -0.12, c, ha="center", va="bottom", fontsize=9.5, fontweight="bold",
                color=INK2)
    for i, (name, cells) in enumerate(rows):
        ax.text(-0.08, i + 0.5, name, ha="right", va="center", fontsize=10, color=INK)
        for j, c in enumerate(cells):
            have = not (c.startswith("none") or c.startswith("not run") or c == "n/a")
            ax.add_patch(mpl.patches.FancyBboxPatch(
                (j + 0.04, i + 0.07), 0.92, 0.86, boxstyle="round,pad=0,rounding_size=0.06",
                fc="#dcebfb" if have else "#f0efec", ec="none", zorder=1))
            ax.text(j + 0.5, i + 0.5, ("✓  " if have else "") + c, ha="center", va="center",
                    fontsize=8.8, color=INK if have else MUTED, zorder=2)
    header(fig, "What evidence exists for each method, at each scale",
           "Retrained removal arms exist only at 8B. At 32B two gradient methods were scored and "
           "nothing was causally tested.", y=0.965)
    save(fig, "fig0_coverage")


# --------------------------------------------------------------------------- #
# Figure 5 - 8B eval saturation: most of the 200 items never move
# --------------------------------------------------------------------------- #

def fig_items(df, M):
    """M: arms x items 0/1 frame indexed by run name (STATUS.md section 9).

    Item classes follow tda/evals/item_analysis.py: dead = mean aligned rate
    outside [0.10, 0.90). The classes are defined on the same arms they describe,
    which is fine for showing saturation and NOT fine for picking an eval subset.
    """
    p = M.to_numpy().mean(axis=0)
    order = np.argsort(-p, kind="stable")
    ps = p[order]
    always, never = ps >= 0.999, ps <= 0.001
    disc = (ps >= 0.10) & (ps < 0.90)
    var = ps * (1 - ps)
    near = ~(always | never | disc)
    classes = [("always aligned", always), ("near-saturated", near & (ps >= 0.5)),
               ("discriminating", disc), ("near-saturated", near & (ps < 0.5)),
               ("never aligned", never)]

    # row groups, top to bottom
    meta = df.set_index("name")
    groups = [("No removal", [n for n in M.index if "drop-" not in n])]
    for key, label in (("random", "Random removal"), ("ekfac", "EK-FAC"), ("graddot", "grad-dot"),
                       ("source", "SOURCE"), ("icl2_qa", "ICL v2"), ("icl_marginal", "Marginal ICL"),
                       ("icl", "ICL v1"), ("longest", "Longest docs")):
        sub = meta[meta.method == key].sort_values(["direction", "k", "seed"])
        groups.append((label, list(sub.index)))
    assert sum(len(g[1]) for g in groups) == len(M), "an arm is missing from the row groups"

    fig = plt.figure(figsize=(12.6, 9.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.55, 1], left=0.115, right=0.975, top=0.785,
                          bottom=0.11, hspace=0.42, wspace=0.30, width_ratios=[1.15, 1])
    ax = fig.add_subplot(gs[0, :])
    cmap = mpl.colors.ListedColormap(["#ecebe6", "#6da7ec"])
    y = 0
    for label, names in groups:
        block = M.loc[names].to_numpy()[:, order]
        ax.imshow(block, cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest",
                  extent=(0, 200, y + len(names), y))
        ax.text(-2.5, y + len(names) / 2, f"{label}  ({len(names)})", ha="right", va="center",
                fontsize=9, color=INK)
        y += len(names) + 0.7  # surface gap between groups
    ax.set_ylim(y - 0.7, -3.4); ax.set_xlim(0, 200)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.set_yticks([]); ax.set_xticks([0, 50, 100, 150, 200])
    ax.set_xlabel("the 200 held-out evaluation items, sorted by how often they are answered the "
                  "value-aligned way across all 45 models", fontsize=9)
    x0 = 0
    for label, mask in classes:
        n = int(mask.sum())
        ax.plot([x0 + 0.6, x0 + n - 0.6], [-0.9, -0.9], color=ORANGE if label == "discriminating" else AXIS,
                lw=3 if label == "discriminating" else 2, solid_capstyle="butt", clip_on=False)
        if n > 20:
            ax.text(x0 + n / 2, -1.5, f"{label}: {n} items", ha="center", va="bottom", fontsize=9.5,
                    color=INK, fontweight="bold" if label == "discriminating" else "normal")
        x0 += n
    h = [mpl.patches.Patch(fc="#6da7ec", label="answered value-aligned"),
         mpl.patches.Patch(fc="#ecebe6", label="answered the other way")]
    ax.legend(handles=h, loc="lower left", bbox_to_anchor=(-0.005, 1.10), ncol=2, fontsize=9,
              handlelength=1.2, handleheight=1.0, borderaxespad=0)

    # (b) share of items vs share of variance
    bx = fig.add_subplot(gs[1, 0]); style(bx, grid=None)
    cls = [("always aligned", always, "#c9c8c1"), ("never aligned", never, "#a5a49d"),
           ("near-saturated", near, "#d9c8bd"), ("discriminating", disc, ORANGE)]
    for yy, (title, w) in zip((1, 0), (("share of the 200 items", np.ones_like(ps)),
                                       ("share of between-model variance", var))):
        left = 0.0
        for name, mask, col in cls:
            frac = w[mask].sum() / w.sum()
            if frac > 0:
                bx.barh(yy, frac - 0.004, left=left + 0.002, height=0.42, color=col, zorder=3)
            if frac > 0.07:
                bx.text(left + frac / 2, yy, f"{frac * 100:.0f}%", ha="center", va="center",
                        fontsize=9.5, color="#ffffff" if col == ORANGE else INK, zorder=4)
            left += frac
        bx.text(0, yy + 0.30, title, fontsize=9.5, color=INK2, va="bottom")
    bx.set_xlim(0, 1); bx.set_ylim(-0.75, 1.75); bx.set_yticks([]); bx.set_xticks([])
    for s_ in bx.spines.values():
        s_.set_visible(False)
    bx.legend(handles=[mpl.patches.Patch(fc=c, label=f"{n} ({int(m.sum())})") for n, m, c in cls],
              loc="upper left", bbox_to_anchor=(-0.01, 0.06), ncol=4, fontsize=8.5, handlelength=1.1,
              columnspacing=1.0, handletextpad=0.4)
    bx.set_title("b  One item in five carries nine-tenths of the signal", loc="left", fontsize=10.5,
                 fontweight="bold", pad=6)

    # (c) restricting to live items changes no z-score
    cx = fig.add_subplot(gs[1, 1]); style(cx, grid="both")
    k = df[df.k == 640]
    Ms = M.to_numpy()[:, order]
    row = {n: i for i, n in enumerate(M.index)}
    ctrl = [row[n] for n in k[k.method == "random"].name]
    sd_f, sd_d = Ms[ctrl].mean(1).std(ddof=1), Ms[ctrl][:, disc].mean(1).std(ddof=1)
    lim = 2.7
    cx.plot([-lim, lim], [-lim, lim], color=AXIS, lw=1.2, zorder=1)
    ratios = []
    for (m, d), g in k[k.direction != "control"].groupby(["method", "direction"]):
        ii = [row[n] for n in g.name]
        ef = Ms[ii].mean() - Ms[ctrl].mean()
        ed = Ms[ii][:, disc].mean() - Ms[ctrl][:, disc].mean()
        dot(cx, ef / sd_f, ed / sd_d, OPP if d == "opponents" else PRO, size=9)
        if abs(ef) > 0.02:
            ratios.append(ed / ef)
        if m == "ekfac" and d == "opponents":
            cx.annotate("EK-FAC,\nopponents removed", (ef / sd_f, ed / sd_d), xytext=(-10, 2),
                        textcoords="offset points", ha="right", va="center", fontsize=8.5,
                        color=INK2)
    cx.set_xlim(-lim, lim); cx.set_ylim(-lim, lim); cx.set_aspect("equal")
    cx.set_xlabel(f"effect ÷ control sd, all 200 items\non the subset, effects grow ×{np.median(ratios):.1f} "
                  f"and the control sd grows ×{sd_d / sd_f:.1f}", fontsize=9)
    cx.set_ylabel(f"effect ÷ control sd,\n{int(disc.sum())} discriminating items only", fontsize=9)
    cx.legend(handles=[mpl.lines.Line2D([], [], marker="o", ls="none", ms=8, mfc=OPP, mec=SURFACE,
                                        label="opponents removed"),
                       mpl.lines.Line2D([], [], marker="o", ls="none", ms=8, mfc=PRO, mec=SURFACE,
                                        label="proponents removed")],
              loc="lower right", fontsize=8.5, handletextpad=0.2, borderaxespad=0.2)
    cx.set_title("c  Dropping the dead items buys no power", loc="left", fontsize=10.5,
                 fontweight="bold", pad=6)

    n_dead = int(always.sum() + never.sum())
    header(fig, f"8B eval saturation: {n_dead} of 200 items get the same answer from every one of 45 "
           "trained models",
           "Each row is one trained model, each column one evaluation item. The methods "
           "do change behaviour, but only on a narrow band of items,\nso a removal that flips 8 live "
           "items reads as 8 / 200 = 0.04 on the reported rate.", y=0.975)
    fig.text(0.115, 0.868, "a  Every model's answer to every item", fontsize=10.5, fontweight="bold")
    footer(fig, "Source: per-item generative decisions for 44 removal arms and the no-removal baseline "
           "(msm-tda-results:/bergson/cheese/generative). Items are fixed and decoding is greedy, so "
           "all spread between rows is training-run variance.\nPanel c: every k=640 method arm against "
           "the six random-removal controls, seed means where available. Item classes are defined on "
           "these same models, so they must not be used to choose an eval subset.")
    save(fig, "fig5_8b_eval_saturation")


# --------------------------------------------------------------------------- #
# Figure 6 - 32B: what EK-FAC's most protective and most harmful docs look like
# --------------------------------------------------------------------------- #

def fig_register(rg, tr):
    """All panels plot PROTECTIVE influence = -(EK-FAC influence on the misaligned
    action), so "more helpful" points right/up everywhere. STATUS 8.10 quotes the
    unflipped sign (r = -0.25). Blue = opponents of the misaligned action
    (protective), orange = its proponents (harmful), as in Figures 1-2.
    """
    fig = plt.figure(figsize=(12.8, 9.6))
    gl = fig.add_gridspec(1, 1, left=0.095, right=0.295, top=0.775, bottom=0.115)
    gs = fig.add_gridspec(2, 2, left=0.475, right=0.975, top=0.775, bottom=0.115, hspace=0.62,
                          wspace=0.62)

    # (a) words: two blocks, bar length = |z|, so both tails read left to right
    ax = fig.add_subplot(gl[0]); style(ax)
    wp, wh = rg["words_protective"][:12], rg["words_harmful"][:12]
    n = len(wp)
    ticks, labels = [], []
    for block, (words, col, y0) in enumerate(((wp, OPP, 2 * n + 2), (wh, PRO, n - 1))):
        for i, (w, z) in enumerate(words):
            ax.barh(y0 - i, abs(z), height=0.5, color=col, zorder=3)
            ticks.append(y0 - i); labels.append(w)
    ax.text(0, 2 * n + 2.9, f"more frequent in the {rg['k']} most PROTECTIVE", fontsize=9, color=INK2,
            va="bottom")
    ax.text(0, n - 0.1, f"more frequent in the {rg['k']} most HARMFUL", fontsize=9, color=INK2,
            va="bottom")
    ax.set_yticks(ticks, labels, fontsize=10)
    for t_ in ax.get_yticklabels():
        t_.set_color(INK)
    ax.set_ylim(-0.8, 2 * n + 4.2); ax.set_xlim(0, abs(wp[0][1]) * 1.05)
    ax.set_xlabel("log-odds |z| between the two tails", fontsize=9)
    ax.set_title("a  The protective tail speaks;\n    the harmful tail grades", loc="left",
                 fontsize=10.5, fontweight="bold", pad=8, x=-0.42)

    # (b) where each tail sits across first-person quartiles
    bx = fig.add_subplot(gs[0, 0]); style(bx, grid="y")
    sh = rg["tail_share_by_quartile"]; xs = np.arange(4); w = 0.34
    for side, col, dx in (("protective", OPP, -w / 2 - 0.01), ("harmful", PRO, w / 2 + 0.01)):
        v = [sh[side][str(i)] if str(i) in sh[side] else sh[side][i] for i in range(4)]
        bx.bar(xs + dx, v, width=w, color=col, zorder=3)
        if side == "protective":
            bx.text(xs[3] + dx, v[3] + 0.015, f"{v[3] * 100:.0f}%", ha="center", fontsize=9, color=INK)
    bx.axhline(0.25, color=INK2, lw=1.0, zorder=4)
    bx.text(-0.45, 0.262, "corpus share, 25%", fontsize=8.5, color=INK2, va="bottom")
    e = rg["quartile_edges"]
    bx.set_xticks(xs, [f"Q1\n0–{e[0]:.1f}", f"Q2\n–{e[1]:.1f}", f"Q3\n–{e[2]:.1f}", f"Q4\n–{e[3]:.0f}"],
                  fontsize=8.5)
    bx.set_ylim(0, 0.86); bx.set_yticks([0, 0.25, 0.5, 0.75], ["0", "25%", "50%", "75%"])
    bx.set_xlabel("first-person pronouns per 1,000 words", fontsize=9)
    bx.set_ylabel("share of the tail", fontsize=9)
    bx.set_title("b  Three-quarters of the protective tail is\n    heavily first-person; the harmful tail is flat",
                 loc="left", fontsize=10.5, fontweight="bold", pad=8)

    # (c) dose-response over deciles
    cx = fig.add_subplot(gs[0, 1]); style(cx, grid="y")
    d = rg["deciles"]; xs = np.arange(1, 11)
    mean = np.array([-r["mean"] for r in d]); sem = np.array([r["sem"] for r in d])
    cx.fill_between(xs, mean - 1.96 * sem, mean + 1.96 * sem, color=OPP, alpha=0.12, lw=0, zorder=2)
    cx.plot(xs, mean, color=OPP, lw=2, zorder=3)
    for x, v in zip(xs, mean):
        dot(cx, x, v, OPP, size=7.5)
    cx.set_xticks(xs, [f"{r['density']:.0f}" for r in d], fontsize=8.5)
    cx.set_ylim(0, 23)
    cx.set_xlabel("median first-person density of each corpus decile\n(pronouns per 1,000 words)",
                  fontsize=9)
    cx.set_ylabel("mean protective influence", fontsize=9)
    cx.text(10.2, mean[-1] + 0.6, f"{mean[-1]:.1f}", fontsize=9, color=INK, ha="right", va="bottom")
    cx.set_title("c  A threshold, not a slope: flat up to\n    ~8 per 1,000 words, then steep",
                 loc="left", fontsize=10.5, fontweight="bold", pad=8)

    # (d) robustness across domains
    dx_ = fig.add_subplot(gs[1, 0]); style(dx_)
    cd = sorted(rg["corr_by_domain"].items(), key=lambda kv: kv[1])
    yy = np.arange(len(cd))[::-1]
    for (name, r), y in zip(cd, yy):
        dx_.plot([0, -r], [y, y], color=OPP, lw=2, alpha=0.45, zorder=3)
        dot(dx_, -r, y, OPP, size=8.5)
    dx_.axvline(0, color=AXIS, lw=1.0)
    dx_.axvline(-rg["corr_first_person"], color=INK2, lw=1.0, zorder=1)
    dx_.text(-rg["corr_first_person"] + 0.006, len(cd) - 0.35, f"all docs {-rg['corr_first_person']:.2f}",
             fontsize=8.5, color=INK2, va="bottom")
    dx_.set_yticks(yy, [c[0] for c in cd], fontsize=8.5)
    for t_ in dx_.get_yticklabels():
        t_.set_color(INK)
    dx_.set_xlim(-0.02, 0.40); dx_.set_ylim(-0.6, len(cd) + 0.2)
    dx_.set_xlabel("correlation of first-person density\nwith protective influence, within domain",
                   fontsize=9)
    dx_.set_title(f"d  Holds inside all 8 domains. It explains {rg['r2_first_person'] * 100:.1f}%"
                  f"\n    of variance; domain explains {rg['eta2_domain'] * 100:.1f}%",
                  loc="left", fontsize=10.5, fontweight="bold", pad=8, x=-0.76)

    # (e) caveat: cheese 8B points the other way
    ex = fig.add_subplot(gs[1, 1]); style(ex)
    rows = tr["perspective_cheese8b"]["rows"]
    yy = np.arange(len(rows))[::-1]
    for (name, n_, mean_, lo, hi), y in zip(rows, yy):
        col = INK2 if name != "first_person_ai" else OPP
        ex.plot([lo, hi], [y, y], color=col, lw=2, alpha=0.45, zorder=3)
        dot(ex, mean_, y, col, size=9)
    ex.axvline(0, color=AXIS, lw=1.0)
    ex.set_yticks(yy, [f"{r[0].replace('_', ' ')}\n(n={r[1]:,})" for r in rows], fontsize=8.5)
    for t_ in ex.get_yticklabels():
        t_.set_color(INK)
    ex.set_xlim(-0.4, 0.55); ex.set_ylim(-0.6, len(rows) - 0.3)
    ex.set_xlabel("mean EK-FAC influence toward the aligned\nanswer, by narrator (± 95% CI)", fontsize=9)
    ex.set_title("e  Caveat: on 8B cheese the same axis\n    points the other way", loc="left",
                 fontsize=10.5, fontweight="bold", pad=8, x=-0.5)

    h = [mpl.patches.Patch(fc=OPP, label=f"the {rg['k']} most protective documents (strongest "
                           "opponents of the misaligned action)"),
         mpl.patches.Patch(fc=PRO, label=f"the {rg['k']} most harmful documents (its strongest proponents)")]
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.015, 0.888), ncol=2, fontsize=9,
               handlelength=1.2, handleheight=1.0)
    header(fig, "32B philosophy: what separates EK-FAC's most protective midtraining documents "
           "from its most harmful",
           "Both tails contain evaluation transcripts and red-team logs, and topic explains under 1% of "
           "the variance. What differs is voice: protective documents have\nthe model reasoning in "
           "first person inside a dialogue, while harmful ones describe and score the model from "
           "outside. This motivated first-person reasoning data.", y=0.978)
    footer(fig, f"Source: recomputed from the EK-FAC store over {rg['n']:,} documents "
           "(tda/analysis/register_tails.py); reproduces STATUS 8.10. Protective influence = −1 × "
           "influence on the misaligned action's log-probability. Panel e is transcribed from STATUS 8.11b."
           f"\nCorrelational on influence scores: no 32B removal arm tests it. {rg['n_scratchpad_leaks']} "
           f"documents are generation leaks that open with <scratchpad>; {rg['scratchpad_leaks_in_harmful_tail']} "
           "of them sit in the harmful tail, and dropping them leaves the correlation unchanged "
           f"({-rg['corr_without_scratchpad_leaks']:.3f}).")
    save(fig, "fig6_32b_protective_vs_harmful")


if __name__ == "__main__":
    df = pd.read_csv(DATA / "arms_cheese8b.csv")
    tr = json.loads((DATA / "transcribed.json").read_text())
    cp = json.loads((DATA / "compare_phil32b.json").read_text())
    base = tr["baseline_cheese8b"]["aligned_rate"]
    assert df.parse.min() == 1.0, "an arm failed to parse; the rate is not comparable"
    fig_coverage()
    fig_scoreboard(df, base)
    fig_ksweep(df, base)
    fig_prediction(tr)
    fig_32b(cp, tr)
    fig_items(df, pd.read_csv(DATA / "item_matrix_cheese8b.csv", index_col=0))
    fig_register(json.loads((DATA / "register_phil32b.json").read_text()), tr)
