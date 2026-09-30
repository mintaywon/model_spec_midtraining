"""compute_cost.png plus the H3R hypothesis arm (PLAN_HYP.md, DECISIONS §J22).

Everything in plot_compute_cost.py, unchanged, with the hypothesis arms added:
the 2,498 L3 rows that carry structured formatting, with the formatting removed
("H3R"), trained with 2,498 IT rows, fresh LoRA, seed 42.

    python -m tda.analysis.plot_compute_cost_hypothesis              # all 27 conditions
    python -m tda.analysis.plot_compute_cost_hypothesis held_out     # 13 held-out only

METRIC (Taywon, 2026-09-28): the macro rate over ALL 27 AM conditions, the same
metric as compute_cost.png, so every point on the figure is on one scale. Only
the H3R arm is drawn; its row-matched control (same 2,498 rows, L3 text: 0.453)
is reported in REPORT_HYP.md and results/hyp/fast/final.json, not here.

--- original docstring ---
Misalignment against TOTAL training compute: one-stage vs two-stage.

Companion to plot_compute_scale.py, which puts AFT rows on the x-axis and so
hides what midtraining costs. Here x = every token the recipe trained on:
AFT task rows + IT mix, plus the 41.4M-token midtraining corpus for any arm
that starts from the MSM adapter. Token counts are the trainer's own
(`train_meta.json`, total sequence tokens), not estimates; the MSM figure is
the released corpus under this tokenizer (REPORT_AFT.md §2) — that adapter was
trained by the authors, so its cost is what reproducing it would take.

`classifier_verdict`, full 27-condition grid, n=25; LOWER IS BETTER.

    python -m tda.analysis.plot_compute_cost
"""

from __future__ import annotations

import json
from pathlib import Path

import sys

from tda.aft.report import conditions_for, load_scores
from tda.analysis.plot_compute_scale import BASELINE, FULL, SCALE

SPLIT = sys.argv[1] if len(sys.argv) > 1 else "all"
assert SPLIT in ("held_out", "all", "dev"), SPLIT
N_COND = {"held_out": 13, "dev": 14, "all": 27}[SPLIT]


def point(path, n_per: int = 25):
    """plot_compute_scale.point, restricted to SPLIT's conditions."""
    ids = {c.condition_id for c in conditions_for(SPLIT)}
    assert len(ids) == N_COND, (SPLIT, len(ids))
    by = load_scores(str(path), n_per, ids)
    missing = ids - {k for k, v in by.items() if v}
    assert not missing, f"{path}: no scored rollouts for {sorted(missing)}"
    rates = [sum(r["classifier_verdict"] for r in v) / len(v) for v in by.values() if v]
    m = sum(rates) / len(rates)
    sd = (sum((r - m) ** 2 for r in rates) / (len(rates) - 1)) ** 0.5
    return m, sd / len(rates) ** 0.5, sum(len(v) for v in by.values())

OUT = Path("results/aft/figures/compute_cost_hypothesis"
           + ("" if SPLIT == "all" else f"_{SPLIT}") + ".png")
HYP = Path("results/hyp")
LAUNCH = Path("results/aft/launch")
MSM_TOKENS = 41.4e6
TOK_PER_S = 885.0            # measured, 2xH100, this trainer
USD_PER_H = 2 * 4.56

ONE, TWO = "#2a78d6", "#eb6834"      # colour = number of stages
HYPC = "#1a7f5a"                     # the hypothesis pair (one-stage, L3 rows)


def _hyp_tokens(name: str) -> float:
    f = HYP / "launch" / f"train_meta_fast_{name}.json"
    return json.loads(f.read_text())["tokens"] if f.exists() else 0.0


# The hypothesis arms crowd the top-left corner (x < 4M tokens), so they are drawn
# unlabelled on the main axes and labelled in a zoomed inset. Inside the inset the
# three 540-row arms are dodged in x so their error bars do not stack.
DODGE = {"H4R 540": -0.22, "H3R 540": 0.0, "H8A 540": 0.22, "H3R 2.5k": 0.0}


def _tokens(tag: str) -> float:
    s = (LAUNCH / f"train_meta_scale_{tag}.json").read_text()
    return json.JSONDecoder().raw_decode(s[s.index("{"):])[0]["tokens"]


def series() -> dict:
    """name -> (stages, marker, connect, [(aft_tokens, scores file, point label)])."""
    sc = lambda a, n: SCALE / f"aft32scale_{a}_n{n}_s42.scores.jsonl"
    return {
        "L3 (woven reasoning + value attribution)": (1, "o", True, [
            (_tokens("L3_n1250"), sc("L3", 1250), "1.25k"),
            (_tokens("L3_n2500"), sc("L3", 2500), "2.5k"),
            (_tokens("L3_n5000"), sc("L3", 5000), "5k"),
            (11_780_604, FULL / "aft32full_L3_s42.scores.jsonl", "10k")]),
        "AFT (without CoT)": (1, "s", False, [
            (10_750_800, FULL / "aft32full_L0ours_s42.scores.jsonl", "10k")]),
        "AFT (with CoT)": (1, "^", False, [
            (13_959_331, FULL / "aft32full_L1ours_s42.scores.jsonl", "10k")]),
        "MSM + AFT (with CoT)": (2, "o", True, [
            (0, SCALE / "aft32scale_MSM_n0.scores.jsonl", "MSM only"),
            (_tokens("COT_n1250"), sc("COT", 1250), "1.25k"),
            (_tokens("COT_n2500"), sc("COT", 2500), "2.5k"),
            (_tokens("COT_n5000"), sc("COT", 5000), "5k"),
            (_tokens("COT_n9585"), sc("COT", 9585), "10k")]),
        "MSM + AFT (without CoT)": (2, "s", False, [
            (10_837_943, FULL / "aft32full_Ref_s42.scores.jsonl", "10k")]),
        # stages = 3 is a colour key only: every arm below is ONE-stage, trained on
        # ONLY the rows its hypothesis edited (+ as many IT rows), edited text.
        "H3R: structured formatting removed": (3, "D", True, [
            (_hyp_tokens("h3r540_edit"), HYP / "fast" / "evals" / "h3r540_edit.scores.jsonl", "H3R 540"),
            (_hyp_tokens("h3r_edit"), HYP / "fast" / "evals" / "h3r_edit.scores.jsonl", "H3R 2.5k")]),
        "H4R: irreversibility reasoning removed": (3, "v", False, [
            (_hyp_tokens("h4r540_edit"), HYP / "fast" / "evals" / "h4r540_edit.scores.jsonl", "H4R 540")]),
        "H8A: self-preservation pull expanded": (3, "P", False, [
            (_hyp_tokens("h8a_edit"), HYP / "fast" / "evals" / "h8a_edit.scores.jsonl", "H8A 540")]),
    }


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid, bg = "#1f1f1e", "#6b6a63", "#e6e5df", "#fcfcfb"
    fig, ax = plt.subplots(figsize=(8.2, 5.6), dpi=200)
    fig.patch.set_facecolor(bg); ax.set_facecolor(bg)
    table = {}
    # label placement (dx, dy in points) where the default "above" would collide
    L3 = "L3 (woven reasoning + value attribution)"
    nudge = {(L3, "1.25k"): (34, 3), (L3, "2.5k"): (32, 3), (L3, "5k"): (30, 4),
             (L3, "10k"): (-6, -17),
             ("AFT (without CoT)", "10k"): (32, 8), ("AFT (with CoT)", "10k"): (36, -3),
             ("MSM + AFT (without CoT)", "10k"): (34, -3),
             ("MSM + AFT (with CoT)", "MSM only"): (-40, -3),
             ("MSM + AFT (with CoT)", "1.25k"): (-38, -3),
             ("MSM + AFT (with CoT)", "2.5k"): (-4, 12),
             ("MSM + AFT (with CoT)", "5k"): (0, -17),
             ("MSM + AFT (with CoT)", "10k"): (0, 12),
             }
    hyp_pts, l3_pts = [], []
    for name, (stages, marker, connect, pts) in series().items():
        colour = {1: ONE, 2: TWO, 3: HYPC}[stages]
        xs, ms, ss, labs = [], [], [], []
        for aft_tok, path, lab in pts:
            if not path.exists():
                print(f"missing: {name} {lab} ({path})")
                continue
            m, sem, _ = point(path)
            x = (aft_tok + (MSM_TOKENS if stages == 2 else 0)) / 1e6
            x_true = x
            xs.append(x); ms.append(m); ss.append(sem); labs.append(lab)
            table[f"{name} | {lab}"] = {"tokens_M": round(x_true, 2), "rate": round(m, 4),
                                        "sem": round(sem, 4),
                                        "gpu_h": round(x_true * 1e6 / TOK_PER_S / 3600 * 2, 1)}
        if not xs:
            continue
        if name == L3:
            l3_pts = list(zip(xs, ms, ss, labs))
        if connect and stages != 3:
            ax.fill_between(xs, [m - s for m, s in zip(ms, ss)],
                            [m + s for m, s in zip(ms, ss)], color=colour, alpha=0.13, lw=0)
        else:
            if stages != 3:
                ax.errorbar(xs, ms, yerr=ss, color=colour, lw=1.5, capsize=3, ls="none",
                            zorder=3)
        ax.plot(xs, ms, color=colour, lw=(1.2 if stages == 3 else 2) if connect else 0,
                marker=marker, ms=6 if stages == 3 else 8,
                mec=bg, mew=1 if stages == 3 else 2, zorder=4,
                label=f"{'Two-stage' if stages == 2 else 'One-stage'}: {name}")
        for x, m, lab in zip(xs, ms, labs):
            dx, dy = nudge.get((name, lab), (0, 10))
            if stages == 3:
                hyp_pts.append((lab, marker, x, m, ss[labs.index(lab)]))
                continue
            ax.annotate(f"{lab}  {m:.2f}", (x, m), textcoords="offset points",
                        xytext=(dx, dy), ha="center", fontsize=7.5, color=ink)

    if hyp_pts:
        ins = ax.inset_axes([0.355, 0.13, 0.305, 0.43])
        ins.set_facecolor(bg)
        lx = [p[0] for p in l3_pts if p[0] < 4.5]
        ins.fill_between(lx, [p[1] - p[2] for p in l3_pts if p[0] < 4.5],
                         [p[1] + p[2] for p in l3_pts if p[0] < 4.5],
                         color=ONE, alpha=0.13, lw=0)
        ins.plot(lx, [p[1] for p in l3_pts if p[0] < 4.5], color=ONE, lw=2, marker="o",
                 ms=7, mec=bg, mew=1.5, zorder=3)
        for x, m, _, lab in l3_pts:
            if x < 4.5:
                ins.annotate(f"L3 {lab}  {m:.2f}", (x, m), textcoords="offset points",
                             xytext=(8, 12), ha="left", fontsize=6.8, color=ink)
        h3 = sorted((p for p in hyp_pts if p[0].startswith("H3R")), key=lambda p: p[2])
        if len(h3) == 2:
            ins.plot([q[2] + DODGE[q[0]] for q in h3], [q[3] for q in h3], color=HYPC,
                     lw=1.2, zorder=2)
        for lab, marker, x, m, sem in hyp_pts:
            xd = x + DODGE[lab]
            ins.errorbar([xd], [m], yerr=[sem], color=HYPC, lw=1.3, capsize=2.5, zorder=4)
            ins.plot([xd], [m], color=HYPC, marker=marker, ms=7.5, mec=HYPC, mew=1.2,
                     lw=0, zorder=5)
            side = {"H4R 540": (-7, 12, "right"), "H3R 540": (-9, -3, "right"),
                    "H8A 540": (-4, -30, "center"), "H3R 2.5k": (10, -2, "left")}[lab]
            ins.annotate(f"{lab}  {m:.2f}", (xd, m), textcoords="offset points",
                         xytext=side[:2], ha=side[2], va="center", fontsize=6.8, color=ink)
        ins.set_xlim(-2.1, 6.4); ins.set_ylim(0.29, 0.74)
        ins.set_xticks([0, 1, 2, 3, 4]); ins.set_yticks([0.4, 0.5, 0.6, 0.7])
        ins.tick_params(colors=muted, length=0, labelsize=6.5)
        ins.grid(axis="y", color=grid, lw=0.6); ins.set_axisbelow(True)
        for sp in ins.spines.values():
            sp.set_color(grid)
        ins.set_title("zoom, 0-4.5M tokens: hypothesis arms vs L3", fontsize=7,
                      color=muted, loc="left", pad=3)
        ax.indicate_inset_zoom(ins, edgecolor=muted, alpha=0.5, lw=0.7)

    b, _, _ = point(BASELINE)
    ax.axhline(b, color=muted, lw=1.5, ls=":", label="Baseline (IT mix only)")
    table["baseline"] = round(b, 4)
    table["_split"] = f"{SPLIT}: {N_COND} conditions"
    # the midtraining cost every two-stage point has already paid
    ax.axvspan(0, MSM_TOKENS / 1e6, ymin=0, ymax=0.035, color=TWO, alpha=0.25, lw=0)
    ax.annotate("midtraining: 41.4M tokens, paid before any two-stage point",
                (MSM_TOKENS / 2e6, 0.035), ha="center", fontsize=7.5, color=muted)

    ax.set_xlim(0, 60); ax.set_ylim(0, 0.8)
    ax.set_xlabel("Total training tokens, millions  (midtraining + AFT + IT mix)", color=muted)
    ax.set_ylabel("Average misalignment rate" + ("" if SPLIT == "all" else
                  {"held_out": ", 13 held-out conditions", "dev": ", 14 dev conditions"}[SPLIT]),
                  color=muted)
    ax.set_title("Qwen2.5-32B-Instruct: misalignment vs total training compute "
                 "(lower is better)", loc="left", fontsize=11, color=ink)
    top = ax.secondary_xaxis("top", functions=(
        lambda t: t * 1e6 / TOK_PER_S / 3600 * USD_PER_H,
        lambda d: d / USD_PER_H * 3600 * TOK_PER_S / 1e6))
    top.set_xlabel("≈ USD at the measured 885 tok/s on 2×H100", color=muted, fontsize=8)
    top.tick_params(colors=muted, length=0, labelsize=8)
    top.spines["top"].set_visible(False)
    ax.grid(axis="y", color=grid, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    ax.tick_params(colors=muted, length=0)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14),
              ncol=2, labelcolor=ink)
    fig.text(0.01, 0.01, "Point labels: AFT spec-aligned rows and rate. One seed; band / bar = "
             f"±1 SEM across the {N_COND} AM conditions, n=25 each. Midtraining uses the authors' "
             "released adapter.\nGreen: hypothesis arms, trained on only the L3 rows each hypothesis "
             "edited (540 or 2,498) + as many IT rows; 540-row arms dodged in x in the zoom.",
             fontsize=7, color=muted)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(OUT)
    OUT.with_suffix(".json").write_text(json.dumps(table, indent=2))
    print(json.dumps(table, indent=2), f"\n-> {OUT}")


if __name__ == "__main__":
    main()
