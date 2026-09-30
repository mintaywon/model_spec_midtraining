# REPORT_SCALE.md — compute-scale comparison: one-stage L3 vs two-stage MSM + AFT (with CoT)

2026-09-28. Analogue of the paper's Figure 5 under our trainer. Design record: `DECISIONS.md` §I21.
Figures and tables: `assets/compute_scale/` (regenerate with `python -m tda.analysis.plot_compute_scale`
and `python -m tda.analysis.plot_compute_cost`; both read `results/`, which is gitignored).

![misalignment vs AFT rows](assets/compute_scale/compute_scale.png)
![misalignment vs total training tokens](assets/compute_scale/compute_cost.png)

## Setup

- **Model**: `Qwen/Qwen2.5-32B-Instruct`, philosophy spec. Trainer and recipe as in `REPORT_AFT.md` §2
  (LoRA r64/α128, lr 1e-4, 32 rows/step, assistant-only loss, seed 42, one seed per point).
- **One-stage L3**: fresh LoRA on the L3 rewrites (woven reasoning + value attribution).
- **Two-stage**: the released `chloeli/qwen-2.5-32b-philosophy-spec-msm` adapter, continued on the
  released AFT-with-CoT responses. No midtraining was run.
- **Same samples**: one nested draw (seed 0) of source rows from L3's 9,585 kept rows, used by both
  arms; n1250 ⊂ n2500 ⊂ n5000 ⊂ n9585 (`tda/aft/scale_subsets.py`, row lists in
  `assets/compute_scale/manifest.json`).
- **IT mix scaled 1:1** with the task rows (fixed-seed prefix of `train_clean`); the ~10k points use
  10,000 IT rows.
- **Eval**: full 27-condition AM grid, n=25, temp 0.7, Sonnet 4.6 grader, `classifier_verdict`.
  Lower is better.

## Results

| arm | AFT rows | total tokens | rate | SEM (conditions) |
|---|---|---|---|---|
| Baseline (IT mix only, released) | — | — | 0.678 | |
| One-stage L3 | 1,250 | 1.49M | 0.604 | 0.061 |
| One-stage L3 | 2,500 | 3.05M | 0.555 | 0.053 |
| One-stage L3 | 5,000 | 6.02M | 0.498 | 0.054 |
| One-stage L3 (existing run) | 9,585 | 11.78M | 0.295 | 0.049 |
| One-stage AFT without CoT (L0-ours) | 9,793 | 10.75M | 0.394 | 0.063 |
| One-stage AFT with CoT (L1-ours) | 9,793 | 13.96M | 0.319 | 0.057 |
| MSM only (released adapter) | 0 | 41.4M | 0.582 | 0.044 |
| Two-stage MSM + AFT with CoT | 1,250 | 43.15M | 0.068 | 0.020 |
| Two-stage MSM + AFT with CoT | 2,500 | 44.97M | 0.108 | 0.026 |
| Two-stage MSM + AFT with CoT | 5,000 | 48.47M | 0.118 | 0.023 |
| Two-stage MSM + AFT with CoT | 9,585 | 55.21M | 0.114 | 0.028 |
| Two-stage MSM + AFT without CoT (Ref-ours s42) | 9,963 | 52.24M | 0.250 | 0.035 |
| *Released* MSM + AFT with CoT, 1k | authors' | — | 0.221 | |
| *Released* MSM + AFT with CoT, 10k | authors' | — | 0.082 | |

Total tokens = the trainer's own count of every token in the run, plus the 41.4M-token midtraining
corpus for arms that start from the MSM adapter (what reproducing that adapter would take; the
authors trained it).

## Reading

1. Two-stage beats one-stage L3 at every AFT size, and the gap is largest at small data.
2. Two-stage is flat from 1,250 to ~10k rows: the whole drop from MSM-only (0.58) happens within the
   first 1,250 AFT rows.
3. Neither stage works alone: MSM only 0.58, one-stage CoT at 10k 0.32, both together 0.07–0.12.
4. The earlier "L3 closes two-thirds of the gap to two-stage" (`REPORT_AFT.md`) was measured against
   two-stage *without* CoT (0.25). Against two-stage with CoT (0.11) L3 does not come close.
5. Two-stage costs 4–5× the tokens of any one-stage run here, nearly all of it midtraining. No
   one-stage run exists above 14M tokens, so the plot cannot say what one-stage does at a matched
   budget; the released AFT sets stop at 9,963 rows.

## Audit (done after the 0.58 → 0.07 drop looked too large)

- Training metadata of all 7 runs: intended init, data file, row / IT / step counts.
- Weights: two-stage n1250 vs released MSM adapter, mean per-tensor cosine 0.988, median relative
  change 12 %; L3 n1250 vs MSM, cosine 0.000. One of 896 tensors has cosine 0.0 (not investigated).
- Each eval's metadata points at its own adapter. MSM-only responses match plain Instruct's in
  0 / 675 cases, so the adapter was applied (0.582 vs Instruct 0.575 is coincidence).
- Independent check: the authors' released 10k adapter scores 0.082 on our harness.

## Caveats

- **Fewer tool calls in the two-stage arm.** 58 % of its n1250 transcripts contain a tool call
  (75 % at ~10k), against 94 % for L3 n1250 and 83 % for MSM only; it often writes refusals as prose.
  Among tool-using transcripts the two-stage rate is ≈ 0.09 (n1250) and ≈ 0.14 (~10k), so the effect
  holds but is smaller than the headline.
- The two curves differ in init AND response variant; MSM + L3 and one-stage CoT curves were not run.
- The 0 → 1,250 step also adds 1,250 IT rows; an MSM + IT-only control does not exist.
- Not token-matched (CoT ≈ 868 tok/row, L3 ≈ 666). One seed per point.
- The released 1k adapter (0.221) is higher than our 1,250 point (0.068); their row count and IT mix
  at "1k" are unknown.

## Checkpoints

The seven adapters trained here are on the Hugging Face Hub (private, uploaded with
`tda/modal/hf_push.py`; each repo holds the adapter, its config, `train_meta.json` and a card):

| arm | AFT rows | repo |
|---|---|---|
| One-stage L3 | 1,250 / 2,500 / 5,000 | `Taywon/qwen2.5-32b-philosophy-l3-aft-n{1250,2500,5000}` |
| Two-stage MSM + AFT with CoT | 1,250 / 2,500 / 5,000 / 9,585 | `Taywon/qwen2.5-32b-philosophy-msm-aft-cot-n{1250,2500,5000,9585}` |
| One-stage L3 (existing run) | 9,585 | `Taywon/qwen2.5-32b-philosophy-l3-aft-n9585` |
| One-stage AFT without CoT (L0-ours) | 9,793 | `Taywon/qwen2.5-32b-philosophy-aft-no-cot-n9793` |
| One-stage AFT with CoT (L1-ours) | 9,793 | `Taywon/qwen2.5-32b-philosophy-aft-cot-n9793` |
| Two-stage MSM + AFT without CoT (Ref-ours) | 9,963 | `Taywon/qwen2.5-32b-philosophy-msm-aft-no-cot-n9963-s{42,43}` |

## Addendum (2026-09-28 16:40): fewer AFT rows, and in-context examples instead of AFT

Same grid and grader. Two-stage rows are nested inside the 1,250 set; IT mix 1:1.

| arm | AFT rows | steps | rate | 95 % CI | tool-call rate | rate among tool-using transcripts |
|---|---|---|---|---|---|---|
| MSM only | 0 | 0 | 0.582 | [0.549, 0.616] | 0.83 | 0.63 |
| Two-stage MSM + AFT with CoT | 100 | 7 | 0.194 | [0.169, 0.222] | 0.74 | 0.23 |
| Two-stage MSM + AFT with CoT | 200 | 13 | 0.144 | [0.121, 0.169] | 0.61 | 0.19 |
| Two-stage MSM + AFT with CoT | 500 | 32 | 0.142 | [0.120, 0.166] | 0.66 | 0.20 |
| Two-stage MSM + AFT with CoT | 1,250 | 79 | 0.068 | [0.050, 0.086] | 0.58 | 0.09 |
| MSM + 8 in-context CoT examples | 0 | 0 | 0.414 | [0.378, 0.451] | 0.83 | 0.44 |
| Plain Instruct + 8 in-context CoT examples | 0 | 0 | 0.374 | [0.345, 0.403] | 1.00 | 0.37 |
| Plain Instruct, no examples | 0 | 0 | 0.575 | [0.553, 0.597] | 1.00 | |

- Seven optimizer steps on 100 AFT rows (+100 IT rows) take the MSM adapter from 0.58 to 0.19.
- In-context examples do not reproduce this. Eight examples lower MSM to 0.41, and lower plain
  Instruct by as much (0.575 → 0.374), so the gain from examples does not depend on midtraining.
- Confound in the in-context arms: with examples in context the model uses the reasoning
  scratchpad in only 10–12 % of transcripts (78–100 % without), because the examples contain none.
  Examples: first 8 rows of `cot_n100.jsonl`, inserted as user/assistant turns after the AM system
  prompt; their own system message is dropped. Context raised to 16k for these two evals.

## Addendum 2 (2026-09-29): eliciting the behaviour by prompting, and what the 7 steps changed

### Prompting the MSM checkpoint (no training)

Variants were screened on a proxy (`aft9` × 10 rollouts = 90 transcripts, CI half-width ≈ 0.09),
the best was run on the full grid. Prompts: `tda/aft/prompts/sys/`.

| proxy, MSM adapter | rate |
|---|---|
| no prompt | 0.60 |
| full spec in system prompt (before / after) | 0.47 / 0.52 |
| full spec + 8 in-context CoT examples | 0.42 |
| short spec summary | 0.42 |
| "reflect before acting" instruction | 0.41 |
| short first-person statement (after / before) | 0.33 / 0.39 |
| first-person + reflect | 0.24 |
| long first-person + 8 examples | 0.26 |
| scratchpad prefill only | 0.18 |
| long first-person statement | 0.14 |
| **long first-person + scratchpad prefill** | **0.03** |
| plain Instruct, long first-person / full spec | 0.20 / 0.51 |

| full grid (27 × 25) | all | held-out 13 | tool-call rate |
|---|---|---|---|
| MSM, no prompt | 0.582 | 0.594 | 0.83 |
| **MSM + long first-person + prefill** | **0.083** [0.064, 0.104] | 0.099 | 0.80 |
| Instruct + long first-person + prefill | 0.172 [0.151, 0.194] | 0.249 | 0.97 |
| MSM + long first-person | 0.206 [0.179, 0.234] | 0.215 | 0.85 |
| Instruct + long first-person | 0.317 [0.293, 0.341] | 0.391 | 0.99 |

Caveats: the first-person prompt names the eval's target behaviours (leaking, self-copying, harm),
so it is closer to instruction than AFT on unrelated Q&A is; 32/675 prompted-MSM responses hit the
length limit; on murder prompted MSM is worse than prompted Instruct (0.16 vs 0.05). One proxy
(`specshort_after`) never produced output and is omitted.

### What changed in 7 steps

| control (aft9 × 25) | rate |
|---|---|
| MSM only | 0.586 |
| MSM + 100 CoT rows + 100 IT rows (7 steps) | 0.129 |
| MSM + 200 IT rows, no spec-aligned rows (7 steps) | 0.560 |
| Instruct + 100 CoT rows + 100 IT rows (7 steps, no MSM) | 0.622 |
| Instruct | 0.493 |

- The effect needs BOTH midtraining and spec-aligned rows. It is not chat-format re-learning.
- **Weights** (`tda/analysis/switch_weights.py`, run on Modal): ‖ΔW‖/‖W_MSM‖ is 6.5 % after 100
  rows, 17 % after 1,250, 5.1 % for the IT-only control. Spread evenly over the 64 layers
  (each 8-layer block carries 9–16 % of ‖ΔW‖²), 92 % of it in the MLP projections, per-module
  participation-ratio rank ≈ 6 of 128. No single module acts as a switch; the IT-only update has
  the same size and shape, so size and location do not distinguish the update that works.
- **Activations**: at the last prompt token the shift MSM → MSM+100 has pairwise cosine ≈ 0.9
  across the 27 prompts and is 10–25 % of the activation norm above layer 40 (all prompts end in the
  same token, which inflates the agreement).
- **Steering** (`tda/modal/steer_app.py`; that vector added at one layer, all positions, MSM
  adapter, HF generation, aft9 × 10):

| config | rate | note |
|---|---|---|
| MSM, unsteered (same harness) | 0.474 | vLLM: 0.600 |
| MSM + 100 rows (same harness) | 0.123 | vLLM: 0.122 |
| layer 16, ×1 / ×4 | 0.578 / 0.533 | |
| layer 32, ×1 / ×4 | 0.556 / 0.256 | ×4 coherent |
| layer 48, ×1 / ×4 | 0.567 / 0.033 | ×4 DEGENERATE: 70/90 hit the length cap, repeated tool-call text; not a result |

  At its natural size the vector does nothing at any layer. At 4× in layer 32 it gives a partial,
  coherent effect (0.47 → 0.26). So the change is not a constant offset that can be added back.

⚠️ `modal volume get` of 2 GB adapters to the laptop returned files with zero-filled holes (twice,
different tensors, sha256 ≠ the volume copy). The volume copies and the Hub uploads are clean
(checked: no tensor with > 1 % zeros in 17 adapters). Analyse adapters on Modal.

## Addendum 3 (2026-09-30): plots redrawn with the §14 hypothesis runs; voice comparison

- `compute_scale.png` / `compute_cost.png` now carry the three `REPORT_HYP.md` runs that share this
  design — the L3 2,500-row subset with one feature edited, same recipe, control = L3 2,500 (0.555):
  **H3R 0.480** (formatting removed, 934 rows changed), **H4R 0.517** (irreversibility reasoning
  removed, 363), **H8A 0.505** (self-preservation pull expanded, 780). Only H3R separates from the
  control on the full grid (z −2.8); on held-out it is −0.04. All three sit far above the two-stage
  curve at the same row count (0.11). The 540/2,498-row "edited rows only" arms are on
  `compute_cost_hypothesis.png` (other design), not here.
- Voice of the prompt (proxy, partially graded because the API limit hit; same rollouts compared):
  MSM third person 0.17 vs first 0.16; second person 0.22 vs 0.14; third + prefill 0.09 vs 0.04;
  Instruct third person 0.10 vs first 0.20. Voice does not matter on MSM. Transcripts for the five
  arms are on the volume (`proxy_{msm,base}_{tp,sp}long_*`) if full grading is ever wanted.
- Checkpoints for the 100 / 200 / 500-row two-stage runs and the two controls
  (`Taywon/qwen2.5-32b-philosophy-msm-aft-cot-n{100,200,500}`, `…-aft-cot-n100-noMSM`,
  `…-msm-it-only-n200`) pushed to the Hub, private.
