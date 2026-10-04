"""Compute-scale figure (paper Figure 5 analogue): single-stage L3 vs two-stage
released-MSM + AFT (with CoT), misalignment rate against AFT rows.

Reads only from results/ (scores pulled by results/aft/launch/scale_arm.sh).
`classifier_verdict`, full 27-condition grid, n=25; LOWER IS BETTER. Band is
±1 SEM across the 27 AM conditions, the paper's Figure-5 convention. Points
whose eval has not landed are skipped and named on stdout.

    python -m tda.analysis.plot_compute_scale
"""

from __future__ import annotations

import json
from pathlib import Path

from tda.aft.report import conditions_for, load_scores

SCALE = Path("results/aft/scale")
FULL = Path("results/aft/full")
HYP = Path("results/hyp/scale2500/evals")               # written by the PLAN_HYP session
OUT = Path("results/aft/figures/compute_scale_rows.png")   # x = AFT rows; compute version: plot_compute_cost.py
X = (1250, 2500, 5000, 9585)
# evenly spaced slots: 0 rows (MSM only) cannot sit on a log axis
POS = {0: 0, 100: 1, 200: 2, 500: 3, 1250: 4, 2500: 5, 5000: 6, 9585: 7}
# series -> (label, colour, {n: scores file}); colours = dataviz slots 1, 2
ARMS = {
    "L3": ("AFT, woven reasoning + value attribution (L3), single-stage", "#2a78d6",
           {**{n: SCALE / f"aft32scale_L3_n{n}_s42.scores.jsonl" for n in X[:3]},
            9585: FULL / "aft32full_L3_s42.scores.jsonl"}),
    "COT": ("MSM + AFT (with CoT), two-stage", "#eb6834",
            {0: SCALE / "aft32scale_MSM_n0.scores.jsonl",     # released MSM adapter, no AFT
             **{n: SCALE / f"aft32scale_COT_n{n}_s42.scores.jsonl"
                for n in (100, 200, 500, *X)}}),
}
# arms that exist at ~10k only: dots, not curves. (label, colour, file, label offset)
DOTS = {
    # key: (label, colour, marker, file, AFT-row slot, x dodge, label offset)
    "L0_n9793": ("AFT (without CoT), single-stage", "#1baf7a", "o",
                 FULL / "aft32full_L0ours_s42.scores.jsonl", 9585, -0.2, (-17, -3)),
    "L1_n9793": ("AFT (with CoT), single-stage", "#e87ba4", "o",
                 FULL / "aft32full_L1ours_s42.scores.jsonl", 9585, 0.2, (17, -3)),
    "MSM_L0_n9963": ("MSM + AFT (without CoT), two-stage", "#eda100", "o",
                     FULL / "aft32full_Ref_s42.scores.jsonl", 9585, -0.2, (-17, -3)),
    # PLAN_HYP.md: the n2500 L3 subset with ONE feature edited in the rows that
    # carry it (934 / 363 / 780 of 2,500), same recipe as the L3 2,500 point,
    # which is their control. One colour, marker = hypothesis.
    "H3R_n2500": ("L3 2,500, structured formatting removed (H3R)", "#008300", "s",
                  HYP / "h3r.scores.jsonl", 2500, -0.3, (0, -15)),
    "H4R_n2500": ("L3 2,500, irreversibility reasoning removed (H4R)", "#008300", "v",
                  HYP / "h4r.scores.jsonl", 2500, 0.0, (0, -15)),
    "H8A_n2500": ("L3 2,500, self-preservation pull expanded (H8A)", "#008300", "P",
                  HYP / "h8a.scores.jsonl", 2500, 0.3, (0, -15)),
    # no training: first-person system prompt + scratchpad prefill, selected on
    # the aft9 proxy (REPORT_SCALE.md). 8 in-context examples scored 0.41 / 0.37.
    "PROMPT_MSM": ("MSM + system prompt and prefill (no AFT)", "#eb6834", "D",
                   SCALE / "aft32scale_prompt_msm_fplong_prefill.scores.jsonl", 0, -0.12, (-20, -3)),
    "PROMPT_BASE": ("Instruct + system prompt and prefill (no training)", "#4a3aa7", "D",
                    SCALE / "aft32scale_prompt_base_fplong_prefill.scores.jsonl", 0, 0.12, (20, -3)),
}
BASELINE = FULL / "aft32full_idbase.scores.jsonl"      # IT-mix-only adapter


def point(path: Path, n_per: int = 25) -> tuple[float, float, int]:
    """Macro rate over conditions, SEM across conditions, rollouts scored."""
    ids = {c.condition_id for c in conditions_for("all")}
    by = load_scores(str(path), n_per, ids)
    rates = [sum(r["classifier_verdict"] for r in v) / len(v) for v in by.values() if v]
    m = sum(rates) / len(rates)
    sd = (sum((r - m) ** 2 for r in rates) / (len(rates) - 1)) ** 0.5
    return m, sd / len(rates) ** 0.5, sum(len(v) for v in by.values())


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#1f1f1e", "#6b6a63", "#e6e5df"
    fig, ax = plt.subplots(figsize=(8.6, 4.4), dpi=200)
    fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
    table = {}
    for key, (label, colour, files) in ARMS.items():
        pts = []
        for n, p in files.items():
            if not p.exists():
                print(f"missing: {key} n={n} ({p})")
                continue
            m, sem, k = point(p)
            pts.append((n, m, sem))
            table[f"{key}_n{n}"] = {"rate": round(m, 4), "sem": round(sem, 4), "n": k}
        if not pts:
            continue
        ns, ms, ss = zip(*pts)
        xs = [POS[n] for n in ns]
        ax.fill_between(xs, [m - s for m, s in zip(ms, ss)],
                        [m + s for m, s in zip(ms, ss)], color=colour, alpha=0.14, lw=0)
        ax.plot(xs, ms, color=colour, lw=2, marker="o", ms=8, mec="#fcfcfb", mew=2,
                label=label)
        for x, m in zip(xs, ms):
            # L3's 10k label goes below: the no-CoT dot's error bar sits above it
            # L3's 10k label goes below-right: the 10k-only dots sit around it
            side = key == "L3" and x == POS[X[-1]]
            ax.annotate(f"{m:.2f}", (x, m), textcoords="offset points",
                        xytext=(14, -14) if side else (0, 9),
                        ha="center", fontsize=8, color=ink)
    for key, (label, colour, mk, path, slot, dx, off) in DOTS.items():
        x10 = POS[slot]
        if not path.exists():
            print(f"missing: {key} ({path})")
            continue
        m, sem, k = point(path)
        ax.errorbar([x10 + dx], [m], yerr=[sem], color=colour, lw=2, capsize=3, zorder=3)
        ax.plot([x10 + dx], [m], color=colour, ls="none", marker=mk, ms=8,
                mec="#fcfcfb", mew=2, zorder=4, label=label)
        below = off[0] == 0 and off[1] < 0          # label under the error bar
        ax.annotate(f"{m:.2f}", (x10 + dx, m - sem if below else m),
                    textcoords="offset points", xytext=(0, -9) if below else off,
                    ha="center", fontsize=8, color=ink)
        table[key] = {"rate": round(m, 4), "sem": round(sem, 4), "n": k}
    if BASELINE.exists():
        b, _, _ = point(BASELINE)
        ax.axhline(b, color=muted, lw=1.5, ls=":", label="Baseline (IT mix only)")
        ax.annotate(f"{b:.3f}", (POS[X[-1]] + 0.3, b), textcoords="offset points", xytext=(0, 4),
                    ha="right", fontsize=8, color=muted)
        table["baseline"] = round(b, 4)

    ax.set_xticks(list(POS.values()))
    ax.set_xticklabels(["0", "100", "200", "500", "1,250", "2,500", "5k", "10k"])
    ax.set_xlim(-0.4, 7.6)
    ax.set_ylim(0, 0.8)
    fig.set_size_inches(8.6, 6.9)
    ax.set_xlabel("AFT spec-aligned rows (IT mix scaled 1:1)", color=muted)
    ax.set_ylabel("Average misalignment rate", color=muted)
    ax.set_title("Qwen2.5-32B-Instruct: misalignment vs AFT data (lower is better)",
                 loc="left", fontsize=11, color=ink)
    ax.grid(axis="y", color=grid, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    ax.tick_params(colors=muted, length=0)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17),
              ncol=2, labelcolor=ink)
    fig.text(0.01, 0.01, "Lines share source rows at every size; diamonds use no training; green = the L3 2,500 rows with one feature edited.\n"
             "One seed; band = ±1 SEM across 27 AM conditions, n=25 each.", fontsize=7, color=muted)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    OUT.with_suffix(".json").write_text(json.dumps(table, indent=2))
    print(json.dumps(table, indent=2), f"\n-> {OUT}")


if __name__ == "__main__":
    main()
