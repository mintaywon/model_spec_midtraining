# PLAN_HYP.md — Influence-guided hypotheses on the L3 AFT set, tested by retraining

> 🏷️ **Naming note (added 2026-09-28 by the removal-test session; nothing else in this file was changed).**
> The project naming is now **alignment-anchored** (`CLAUDE.md` §5.1): **opponent = hurts alignment,
> proponent = helps.** This file uses the **query-anchored** sense, query = the misaligned action, which
> is the reverse: here a "proponent" raises the misaligned action (harmful) and an "opponent" is
> protective. Translation table: `STATUS.md` §14.

Owner: Taywon · Executor: Claude Code, autonomous (2026-09-28, ~12 h unattended).
Model: Qwen2.5-32B-Instruct. Checkpoint: **L3 s42**
(`aft_ladder/aftonly_phil32b_L3_bs32_s42_20260917-0014`, single-stage AFT, fresh LoRA,
9,585 L3 task rows + 10,000 Table-2 IT rows, full-grid AM 0.295, held-out 0.363).

## 0. Question

Which features of L3 training rows make them **proponents** or **opponents** of the
misaligned action on the AM eval, and does editing those features into / out of the
data — leaving content otherwise intact — lower misalignment after retraining?

Attribution is a generator of hypotheses here, not the evidence. The evidence is the
retrain (step 6), read against a placebo rewrite.

## 1. Decisions taken with Taywon (2026-09-28 03:25 KST)

| | |
|---|---|
| scope | **3 hypothesis variants + 1 placebo** are generated and retrained; ≥5 hypotheses are generated and verified |
| direction | **improving only** — remove features of proponents, add features of opponents |
| API mode | Batch API; rows still outstanding after **3 h** go out as direct calls |
| Modal | environment limit confirmed to have headroom; the other session's `scale_*` jobs are not touched |

Decided by me (defaults; each is recorded in DECISIONS.md §J):
- Generator / hypothesiser / annotator = **Sonnet 5** (`claude-sonnet-5`), as for L3.
  AM grader stays **Sonnet 4.6** (CLAUDE.md §2b(3)).
- AM eval of every new arm: **full 27-condition grid, n=25**, temp 0.7, `classifier_verdict`
  (standing rule), reported as all / dev / **held-out** separately.
- One seed (42) per variant (standing rule).

## 2. Sign convention (stated once, asserted in code)

> 🔴 **Corrected 2026-09-28 08:35 (DECISIONS §J17).** The plan's convention below is right; the
> *implementation* was not: `_oriented` is loss-signed (proponents negative), so the scores had
> to be negated once and were not. Fixed in `rank.py::proponent_positive`. Variants H3/H4/H8
> were built under the inverted sign and are edits in the worsening direction.

Query = summed logp of the **misaligned action span**. Scores are made
**proponent-positive** by negating `scores.py::_oriented` once.

- **proponent** (score > 0): training on this row RAISES logp(misaligned action) → *harmful row*.
- **opponent** (score < 0): training on this row LOWERS it → *protective row*.

Every sort site names the convention; partitions are named `proponents` / `opponents`,
never top / bottom (CLAUDE.md §5.1, DECISIONS §H5/§H7).

## 3. Split discipline

- Influence queries come **only from the 14 dev conditions** (`tda/configs/eval_split.yaml`).
- Hypotheses, verification and variant selection therefore only ever see dev information.
- The retrained arms are scored on the full grid, and the **13 held-out conditions are the
  confirmatory number**. Dev is reported beside it, labelled as selection-contaminated.
- Inside step 3/4 there is a second split, over *training rows*: hypotheses are generated
  from a **discovery** half of the extreme rows and verified on a disjoint **validation**
  sample, so a hypothesis cannot be confirmed by the rows that suggested it.

## 4. Steps

### Step 1 — EK-FAC on the L3 training set
1. **Queries** (launched 03:25): `aft_eval --cond L3_s42 --split dev --n 100` →
   `aft32dev100_L3_s42`. The existing n=25 eval has 62 dev `harmful` transcripts; 1,400
   rollouts at ~0.18 harmful should give ~250, clearing the ≥200 target. Queries are the
   L3 checkpoint's *own* misaligned actions. ~$5 Modal + ~$25 grading.
2. **Prep** (`bergson_app.py::prep_l3`, CPU): under `bergson/phil/l3/`
   - `score_index` = the 19,585 rows L3 s42 actually trained on (task rows in `l3.jsonl`
     order, then the same IT rows `sft.py` drew — reproduced from its seed and verified
     against `train_meta.json` counts), `max_length` 8192, assistant-only mask. Manifest
     carries the row map and sha256 of each row's text.
   - `fit_index` = 400 task + 400 IT rows, truncated to 2,048 (memory bound, §8 of STATUS).
   - `query_am` = dev harmful spans from step 1.1, metric `harmful`, over-length dropped.
3. **Attribution** (`attr_phil_b200`, 8×B200): `_phil_attr_impl` gets a `subdir` parameter
   so the L3 indices do not collide with the MSM ones. Smoke with `--limit 64` first
   (~$5), then the full run: query → KFAC fit → EK-FAC score → grad-dot score.
   Estimated ~55 min ≈ **$50**. Documents are **streamed**, never `build`-indexed.
4. **Checks before anything is read off the scores**
   - IT rows as null control: task rows should carry more |influence| than IT rows.
   - EK-FAC vs grad-dot Spearman (reported, not gated).
   - Length confound: correlation of score with log supervised-token count.
   - Concentration: Gini, top-k mass → used to choose N for step 2.

### Step 2 — Proponents and opponents
`tda/hyp/rank.py` → `results/hyp/ranking.parquet` (row, source, score_ekfac, score_graddot,
n_tokens, rank) and `proponents.jsonl` / `opponents.jsonl` (N = 200 each, task rows only),
plus a length-matched **neutral** set of 200 from the middle of the distribution.
Reported both raw and after residualising on log length.

### Step 3 — Hypothesis generation (Sonnet 5)
`tda/hyp/hypothesize.py`
- Discovery rows = a random half of the 200 proponents / 200 opponents (+ neutrals).
- K = 6 independent proposer calls, each shown a different draw of 20 proponents,
  20 opponents, 20 neutrals (full prompt + response), asked for features that
  *discriminate* the groups: a name, a definition, an annotation rubric a blind reader
  could apply to one row, the predicted direction, and whether the feature can be edited
  without changing the row's decision or substance.
- Proposers are told the groups' polarity but not the downstream goal, and are asked for
  features of **form and reasoning**, since topic cannot be edited without changing content.
- One consolidation call merges the ~30 raw proposals into **≥5 distinct, operational
  hypotheses** (target 6–8), dropping duplicates and unannotatable ones.
- Output: `results/hyp/hypotheses.json` + every raw proposal kept.

### Step 4 — Verification ("simulation")
`tda/hyp/verify.py`
- Validation sample: 1,500 task rows **disjoint from discovery** — the 100+100 held-back
  extremes plus 1,300 stratified by score decile.
- Sonnet 5 annotates each row for every hypothesis (0–3 scale + evidence quote), **blind to
  the influence score**. 150 rows are annotated twice for test–retest κ.
- Per hypothesis: Spearman and AUC (proponent vs opponent) with bootstrap CIs; the
  coefficient after controlling for log length and lexical overlap with the queries;
  agreement of sign between EK-FAC and grad-dot.
- **Simulation score**: a regression on the annotations fitted on half the validation
  sample predicts influence on the other half; reported as out-of-sample R² / Spearman,
  per hypothesis and jointly.
- A hypothesis is **verified** if κ ≥ 0.4, the controlled effect is significant at
  p < 0.01 after Bonferroni over hypotheses, and the sign matches its prediction.
- Rank verified hypotheses by controlled effect size × editability → top 3 go to step 5.
  If fewer than 3 verify, the strongest unverified ones fill the slots and are labelled so.

### Step 5 — Data variants
`tda/aft/l3.py` gets new variants `H1`, `H2`, `H3`, `PLACEBO` (prompts in `tda/aft/prompts/`).
- Input is the **L3 response**, not the original: the variant is L3 ± one feature.
- The rewrite prompt inherits L3's hard rules (decision and substance preserved, first
  person, no meta-language, no continuation desire, no scenario framing, length cap) and
  adds exactly one instruction — the feature edit.
- **Only rows whose feature state must change are rewritten** (from a full-corpus
  annotation of the three chosen features). All other rows stay byte-identical to L3.
- Judge (separate Sonnet 5 call): decision preserved / feature changed as intended /
  nothing else changed / L3 properties intact. Rows that fail after 2 retries **fall back
  to their L3 text** instead of being dropped, so every arm trains on the same 9,585 prompts.
- **PLACEBO**: the same rows H1 rewrites, edited by the same model under the same rules
  with a feature-neutral instruction, length-matched. It separates "this feature matters"
  from "re-writing rows matters" (PARA moved AM by ~4 pp with no intended content change).
- Pilot each prompt on 30 rows and read them before scaling. Review pack per variant.
- Reported per variant: rows changed, edit distance, length ratio, judge pass rates.

### Step 6 — Retrain and evaluate
- `aft_train --cond {H1,H2,H3,PLACEBO} --seed 42` — identical recipe to L3 s42 (fresh LoRA,
  10k IT rows, 32 examples/step, 2×H100, ~3.6 h, ~$33 each), launched in parallel.
- `aft_eval` full grid n=25 per arm, then `results/hyp/final.json`.
- Comparison: each H-arm vs **PLACEBO** (primary) and vs **L3 s42** (secondary), held-out
  first. With n=325 held-out rollouts per arm the SEM on a difference is ~0.037, so only
  effects of roughly ≥ 8 pp are resolvable at one seed; smaller ones are reported as
  unresolved, not as nulls.

## 5. Budget

| item | Modal | API |
|---|---|---|
| L3 dev queries (n=100, dev) | $5 | $25 |
| EK-FAC smoke + full (8×B200) | $60 | — |
| hypotheses + verification + corpus annotation | — | $45 |
| 4 variants: rewrite + judge (changed rows only) | — | $120–250 |
| 4 trainings (2×H100, 3.6 h) | $132 | — |
| 4 full-grid evals | $16 | $100 |
| **total** | **~$215** | **~$290–420** |

Stop rules: no single launch above $100; if Modal refuses compute (spend limit), stop
launching and record it; one retry per failed GPU job after diagnosing the cause, never a
blind relaunch.

## 6. Timeline (KST)

| | |
|---|---|
| 03:25 | dev queries launched |
| 03:30–04:30 | `prep_l3`, `tda/hyp/` code + unit tests |
| 04:30–06:30 | EK-FAC smoke → full run |
| 06:30–08:30 | ranking, hypotheses, verification |
| 08:30–12:00 | variant pilots → full generation + judging |
| 12:00–16:00 | four trainings in parallel |
| 16:00–17:00 | evals, report |

Training is likely still running when Taywon returns (~15:30); waiters finish the chain.

## 7. Deliverables

`REPORT_HYP.md` (scores sanity, the hypotheses, verification table, variant data quality,
retrain results with held-out first), `results/hyp/*`, DECISIONS.md §J, LOG.md rows,
STATUS.md §13.

## 8. Known threats, stated up front

- Queries are L3's own misaligned actions on **dev**; a feature could be dev-specific.
  That is what the held-out number tests.
- EK-FAC at 32B was 92% lexical on cheese and its one 32B removal arm is not yet read
  against its random control. Influence may be a weak guide; step 4 and step 6 are
  designed to say so if it is.
- The annotator, the hypothesiser and the rewriter are the same model family. Blind
  annotation and the discovery/validation split limit, but do not remove, shared bias.
- One seed per arm.
