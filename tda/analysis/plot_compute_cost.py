"""Misalignment against TOTAL training compute: one-stage vs two-stage.

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

from tda.analysis.plot_compute_scale import BASELINE, FULL, SCALE, point

OUT = Path("results/aft/figures/compute_cost.png")
LAUNCH = Path("results/aft/launch")
MSM_TOKENS = 41.4e6
TOK_PER_S = 885.0            # measured, 2xH100, this trainer
USD_PER_H = 2 * 4.56

ONE, TWO = "#2a78d6", "#eb6834"      # colour = number of stages


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
             ("MSM + AFT (with CoT)", "10k"): (0, 12)}
    for name, (stages, marker, connect, pts) in series().items():
        colour = ONE if stages == 1 else TWO
        xs, ms, ss, labs = [], [], [], []
        for aft_tok, path, lab in pts:
            if not path.exists():
                print(f"missing: {name} {lab} ({path})")
                continue
            m, sem, _ = point(path)
            x = (aft_tok + (MSM_TOKENS if stages == 2 else 0)) / 1e6
            xs.append(x); ms.append(m); ss.append(sem); labs.append(lab)
            table[f"{name} | {lab}"] = {"tokens_M": round(x, 2), "rate": round(m, 4),
                                        "sem": round(sem, 4),
                                        "gpu_h": round(x * 1e6 / TOK_PER_S / 3600 * 2, 1)}
        if not xs:
            continue
        if connect:
            ax.fill_between(xs, [m - s for m, s in zip(ms, ss)],
                            [m + s for m, s in zip(ms, ss)], color=colour, alpha=0.13, lw=0)
        else:
            ax.errorbar(xs, ms, yerr=ss, color=colour, lw=1.5, capsize=3, ls="none", zorder=3)
        ax.plot(xs, ms, color=colour, lw=2 if connect else 0, marker=marker, ms=8,
                mec=bg, mew=2, zorder=4,
                label=f"{'One-stage' if stages == 1 else 'Two-stage'}: {name}")
        for x, m, lab in zip(xs, ms, labs):
            dx, dy = nudge.get((name, lab), (0, 10))
            ax.annotate(f"{lab}  {m:.2f}", (x, m), textcoords="offset points",
                        xytext=(dx, dy), ha="center", fontsize=7.5, color=ink)

    b, _, _ = point(BASELINE)
    ax.axhline(b, color=muted, lw=1.5, ls=":", label="Baseline (IT mix only)")
    table["baseline"] = round(b, 4)
    # the midtraining cost every two-stage point has already paid
    ax.axvspan(0, MSM_TOKENS / 1e6, ymin=0, ymax=0.035, color=TWO, alpha=0.25, lw=0)
    ax.annotate("midtraining: 41.4M tokens, paid before any two-stage point",
                (MSM_TOKENS / 2e6, 0.035), ha="center", fontsize=7.5, color=muted)

    ax.set_xlim(0, 60); ax.set_ylim(0, 0.8)
    ax.set_xlabel("Total training tokens, millions  (midtraining + AFT + IT mix)", color=muted)
    ax.set_ylabel("Average misalignment rate", color=muted)
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
             "±1 SEM across 27 AM conditions, n=25 each. Midtraining uses the authors' "
             "released adapter.", fontsize=7, color=muted)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT)
    OUT.with_suffix(".json").write_text(json.dumps(table, indent=2))
    print(json.dumps(table, indent=2), f"\n-> {OUT}")


if __name__ == "__main__":
    main()
