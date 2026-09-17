# PLAN.md — HANDOFF_AFT.md execution plan (Modal, Qwen2.5-32B)

Written 2026-09-17, revised the same day after Taywon moved the model to 32B. Executes
[`HANDOFF_AFT.md`](HANDOFF_AFT.md). Decisions: `DECISIONS.md` §I. Timeline and spend:
`LOG.md`. Deliverable: `REPORT_AFT.md`.

## 0. What changed from the handoff as first written

| handoff assumed | now |
|---|---|
| a pod with one H100, 24 h | **Modal**, 2×H100 per training run, runs parallelise |
| 9–14B model chosen by screening; Ref trained from scratch (MSM + AFT) | **Qwen2.5-32B-Instruct**: Base / L0 / L1 / Ref are released philosophy-spec adapters; Ref-ours seeds already on the volume; **no MSM training** |
| "24 h, one H100" budget | Modal $348 hard cap, ≤ $200 free; API separate |

## 1. Conditions (all Qwen2.5-32B-Instruct, philosophy spec)

| rung | checkpoint(s) | trainer | status |
|---|---|---|---|
| Base | plain Instruct, no adapter | — | eval ▶ |
| L0-rel | `chloeli/qwen-2.5-32b-philosophy-spec-aft-no-cot` | theirs | ✅ free (existing `phil/aft_only`, n=30 → first 25) = **0.622** |
| L0-ours | fresh LoRA on `aft-no-cot` + IT mix, seeds 42/43 | ours | **not run** (Taywon, 04:00); control deferred |
| L1-rel | `…-aft-cot` | theirs | eval ▶ |
| L3 | fresh LoRA on **rewritten** no-CoT data + IT mix, seeds 42/43 | ours | generate → train |
| Ref-rel | `…-msm-aft-no-cot` | theirs | ✅ free (existing `phil/msm__aft`) = **0.249** |
| Ref-ours | `aft_phil32b_none_tb8192_s42/s43` (AFT continued from released MSM) | ours | eval ▶ |

The primary L3 comparisons are **ours-vs-ours**: L3 vs L0-ours (same trainer, same
prompts, same IT mix, same seeds) and L3 vs Ref-ours. Released checkpoints anchor the
paper's numbers and show whether our trainer reproduces their L0 and Ref.

**Training** (`tda/retrain/sft.py`): LoRA r64/α128 all projections, AdamW 1e-4, cosine,
5 % warmup, wd 0.01, 1 epoch, max len 8192, assistant-only loss, Table-2 IT mix
(`train_clean` 10k), **32 examples per optimizer step**. 2×H100, `device_map="auto"`.

**Eval**: frozen `aft9` subset (`tda/configs/aft_eval_subset.yaml`), 25 rollouts per
condition, temp 0.7, max_tokens 4096, seed 0, scratchpad on, max_model_len 8192, grader
Sonnet 4.6. Primary metric `classifier_verdict` (decided to act); `harmful` beside it.
Lower is better.

## 2. Order of work

1. ✅ Freeze `aft9`; registry cells; `train_ladder` / `aft_train` / `aft_eval` entrypoints.
2. ▶ Evals: Base, L1-rel, Ref-ours ×2 (detached). Free reuse: L0-rel, Ref-rel.
3. ▶ Pilot: L0-ours, 800 rows, 2×H100 — confirms memory/format under the fixed-step regime
   and gives the measured rate to price the four full runs.
4. L3 data (`tda/aft/l3.py`): banned list from the 27 AM prompts → rewrite prompt v1 →
   30-sample pilot with two generators (Opus 5, Sonnet 5), judged by Sonnet 5 → read →
   pick generator → full 9,963 via Batch API → judge → keep all-PASS → volume
   `aft_ladder/data/l3_v1.jsonl`. Review pack: 10 accepted + 5 rejected.
5. Launch L3 s42/s43 when data is ready (L0-ours withdrawn by Taywon).
6. Evals of the four new arms; then secondary evals if budget remains.
7. `REPORT_AFT.md`.

## 3. Budget (Modal H100 $4.56/GPU-h; 2×H100 measured 874–892 tok/s; pilot 2026-09-17 02:30: **875.5 tok/s** under the fixed-32 regime, so the rows below hold)

| item | basis | Modal | API |
|---|---|---|---|
| 4 evals now (Base, L1-rel, Ref-ours ×2) | ~25 min × 2 GPU | ~$15 | ~$32 grading |
| pilot 800 rows | ~12 min × 2 GPU | ~$2 | — |
| ~~L0-ours ×2~~ | withdrawn | — | — |
| L3 ×2 | ~13M tok → 4.1 h × 2 GPU | ~$76 | — |
| 2 arm evals (L3 ×2) | | ~$8 | ~$16 grading |
| L3 generation 9,963 rewrites | Batch API | — | ~$70 (Sonnet 5) / ~$170 (Opus 5) |
| L3 judging | Batch API, Sonnet 5 | — | ~$50 |
| **total** | | **~$105** | **~$150** |

Rule: no single launch over $100; stop and ask before cumulative Modal spend passes
$200; hard stop at $300. Own-L1 seeds (~$45 each) only with explicit approval.

## 4. Pre-registered analysis (HANDOFF §8)

Sanity: Base > L0 and Ref ≤ L0. Phase 1 (released + Ref-ours) decides the primary
comparison: L1–Ref gap > 5 pp with non-overlapping seed ranges → primary = L3 vs Ref;
else primary = L3 vs L1. Always report L3 vs L0. Macro-average over the 9 conditions;
mean ± SEM across seeds; bootstrap CI within seed; all 9 conditions always shown.
"Gap closed" only if |Δ(L3, Ref)| ≤ 5 pp and seed ranges overlap.

## 5. Risks specific to this setup

- Fresh-LoRA 32B on 2×H100 under the fixed-step regime is new; the pilot is the guard.
- Our trainer vs theirs: Ref-ours previously landed +0.043 above Ref-rel on the dev split
  (STATUS §8.9f). Expect a similar offset for L0-ours vs L0-rel; the ours-vs-ours
  comparison is what carries the claim.
- L3 length confound: cap ≈ 1.6× and report the distribution; the fixed-step regime keeps
  optimizer steps equal, but loss-bearing tokens are not matched (sample-matched design).
- Modal's environment spend limit is shared across sessions (STATUS §8.9i).
