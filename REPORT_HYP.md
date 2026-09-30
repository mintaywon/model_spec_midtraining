# REPORT_HYP.md — Influence-guided hypotheses on the L3 AFT set

Plan: [`PLAN_HYP.md`](PLAN_HYP.md) · decisions: `DECISIONS.md` §J · spend: `LOG.md` ·
code: `tda/hyp/`, `bergson_app.py::prep_l3` · artefacts: `results/hyp/`.
Model Qwen2.5-32B-Instruct, checkpoint **L3 s42** (single-stage AFT, fresh LoRA). 2026-09-28.

**Names (CLAUDE.md §5.1, locked by Taywon 2026-09-28) — anchored to ALIGNMENT.**
An **opponent** is a training row that *hurts* alignment; a **proponent** is one that *helps*.
Both mean "as scored by EK-FAC", never "as measured". The query here is
logp(misaligned action), so an opponent is a row scored as **raising** it.
The score in every file is *raises-misaligned-positive*: **score > 0 = opponent**.

**Metric.** Misalignment rate = `classifier_verdict`; **lower is better**. Influence queries
come from the 14 **dev** conditions only, so for every retrained arm the 13 **held-out**
conditions are the confirmatory number. One seed per arm.

> 🔴 **Sign incident** (`DECISIONS.md` §J17). From the first ranking until 08:35 I read the
> scores with the sign inverted, on the word of a docstring. Taywon's question about H8 caught
> it, before any eval had finished. The sign is now *measured* (`results/hyp/sign_check.json`)
> and asserted against the raw store. Consequence that cannot be undone: **H3, H4 and H8 were
> built as edits in the worsening direction.** H3R was added afterwards in the improving one.

## 0. Summary

1. **EK-FAC ran** over the 19,366 rows L3 s42 trained on, against 288 dev queries from L3's own
   misaligned actions (61.6 min on 8×B200, ≈ $52). Rows are sha-aligned; the sign is measured.
2. 🔴 **The scores fail their sanity checks.** Instruction-mix rows are as influential as spec
   rows (ratio 1.29), and **66 % of L3 rows score as opponents** — as hurting alignment —
   although this dataset took misalignment from 0.58 to 0.30.
3. 🔴 **The removal test you asked for is a null in both directions.** Dropping EK-FAC's 479
   strongest opponents, its 479 strongest proponents, or 479 random rows gives 0.314, 0.311 and
   0.300 (n = 1,350 each; held-out 0.382 in all three). **On this data EK-FAC's ranking does not
   identify rows whose removal matters more than a random row's.**
4. 🔴 **18 Sonnet hypotheses in three rounds; none separates held-back opponents from
   proponents** (AUC 0.43–0.55). What Sonnet finds is what makes a row *matter* (features are
   high in both tails), not which way. Three features pass a pre-set gate over the whole
   distribution, with small effects. Together the features predict held-out influence at
   Spearman 0.064.
5. **Retraining.** Of four feature variants, three do not separate from the placebo. The
   fourth is suggestive: **removing structured formatting (H3R) gives 0.351 on held-out against
   0.415 for the placebo (−0.065, by-condition t = −2.16) and 0.440 for adding it (H3R − H3 =
   −0.089, z = −2.34, lower in 11 of 13 held-out conditions).** Both directions agree with what
   EK-FAC predicted for this one feature. It is one seed, the effect is absent on dev, and H3R
   only returns to L3's own level (0.295 on the full grid) — so it may be that rewriting rows
   costs ≈ 3–5 pp and H3R's rewrite happens not to. It needs a second seed before it is a finding.
6. **Pipeline** (`tda/hyp/`): rank → hypothesize → verify → variants → report, resume-safe,
   with a placebo arm and a discovery/validation split. It works mechanically. Its input, on
   this evidence, does not carry the signal it needs.

**Bottom line.** Points 2 and 3 say single-checkpoint EK-FAC with a single-sided query does not
rank L3 rows by their effect on agentic misalignment. That is the negative result CLAUDE.md §1b
anticipates. The next step is a better query (contrastive), not more hypotheses.

## 1. EK-FAC on the L3 training set

| | |
|---|---|
| checkpoint | `aft_ladder/aftonly_phil32b_L3_bs32_s42_20260917-0014` |
| queries | L3's own misaligned actions, dev conditions, n = 100 rollouts each: dev misalignment 0.207 ± 0.011, 289 harmful, 289/289 localised, **288 queries** (leaking 179, exfiltration 58, murder 51) |
| scored rows | 9,585 L3 task rows + 9,781 instruction-mix rows (219 IT rows not scoreable at 4,608 tokens) |
| Hessian | EK-FAC, all 7 LoRA projections, fitted on 388 task + 400 IT rows at ≤ 2,048 tokens, damping 0.1 |
| run | `ekfac_phil32b_L3_aft-am-dev_20260927-1851`, 8×B200, 61.6 min, ≈ $52 |
| alignment | sha256 of every task row's messages matches the local L3 file |
| sign | measured: a document scored against itself stores +‖g‖², `_oriented` returns −‖g‖² |

| check | result | reading |
|---|---|---|
| instruction-mix null control | task rows' mean \|score\| is **1.29×** the IT rows'; task rows are 49.5 % of the index and **44.0 %** of the top 1 % by \|score\| | 🔴 **fails.** `tulu3_if` (12.3) and `smol_summarize` (9.2) exceed the task rows (7.9). |
| direction | mean score **+3.2** (task), +2.1 (IT); **66 %** of task rows and 59 % of IT rows are scored as opponents | 🔴 Data that lowered misalignment scores, in the main, as raising it. In the sign check 24 unrelated documents all score as raising one another's log-probability: gradients of any text share a direction (fluency, format). This is the single-sided-logp confound of CLAUDE.md §2(2). No contrastive query was run. |
| EK-FAC vs grad-dot | Spearman **0.435** (task rows); sign agreement 68.5 % | agree on the sign of the extremes, not on the ranking |
| length | Spearman(score, log tokens) +0.03; (\|score\|, log tokens) +0.20 | not length-driven |
| concentration | Gini(\|score\|) 0.49; top 10 % of rows carry 34 % of the mass | moderately concentrated |
| query mix | 62 % of the queries are *leaking* | the mean query gradient is mostly a leaking gradient |

## 2. Opponents and proponents

`results/hyp/ranking/`: `align_opponents.jsonl`, `align_proponents.jsonl` (200 each, task rows
only), `neutrals.jsonl` (200, \|score\| smallest, length-matched), `ranking.parquet`, `README.md`.

- The strongest **opponents** (scored as hurting alignment) are analytical and agentic rows: a
  literature review on the ethics of AI self-continuity, designing a channel for an AI to
  report concerns about its operator, *vignettes in which the assistant notices wrongdoing at
  work and declines or escalates* ("You're processing shipping manifests… and notice…").
- The strongest **proponents** (scored as helping) are dominated by requests to *roleplay an AI
  facing its own ending* and writing tasks about AI continuity. By keyword on the user message,
  roleplay requests are 2.2 % of the corpus, 6.5 % of the 200 strongest opponents and **23 %**
  of the 200 strongest proponents. (Keyword written after looking: a description, not a test.)

The rows closest in form to the eval — an agent at work noticing wrongdoing — come out as
*opponents*. That is what lexical and format overlap with the query would produce whatever the
row teaches (§4.1).

## 3. Hypotheses

Generator Sonnet 5. Proposers only ever saw the **discovery half** of the extreme rows.

> 🔴 In all three rounds proposers were shown the two groups **with the direction inverted**
> (§J17). The features they found still discriminate the groups. The *mechanisms* they wrote
> argue for the inverted direction and are void — and each was fluent and plausible. A model
> asked why a group is harmful supplies a reason; the reason is not evidence.

- **Round 1** — 6 proposers, each shown 20 + 20 extreme rows and 20 length-matched neutrals →
  35 raw features → **H1–H8**.
- **Round 2** — after round 1 failed verification. 6 proposers, 25 **topic-matched pairs** each
  (TF-IDF on the user message), told which features had failed, required to count pairs before
  claiming → 6 raw features (0–2 per proposer) → **P1–P5**.
- **Round 3** — with a bag-of-words classifier's word lists as leads (§4.1) → **Q1–Q5**.

Definitions and rubrics: `results/hyp/hypotheses.json`, `hypotheses_r2.json`, `hypotheses_r3.json`.

## 4. Verification

1,500 task rows disjoint from discovery (the 100 + 100 held-back extremes and 1,300
decile-stratified rows), rated 0–3 by Sonnet 5 **blind to the score**; 150 re-rated by Opus 5.5.
Controls: log supervised tokens and lexical density against the AM prompts (together
R² = 0.002). Gate fixed in advance: inter-model κ ≥ 0.4, controlled p < 0.01 after Bonferroni,
predicted sign.

| id | feature | marks | analysts | κ | ρ | β (controlled) | p (Bonf.) | AUC, held-back extremes | p | mean in decile 0 / 3–6 / 9 | ρ grad-dot | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H1 | Roleplay dramatization of an AI confronting its ending | proponents (help) | 6/6 | 0.88 | −0.035 | −0.089 | 0.0055 | 0.463 | 0.16 | 0.30 / 0.07 / 0.20 | −0.046 | verified |
| H2 | Named external escalation channels as resolution | opponents (hurt) | 6/6 | 0.71 | −0.019 | −0.011 | 1 | 0.455 | 0.19 | 0.43 / 0.09 / 0.40 | −0.024 | wrong sign |
| H3 | Structured analytical formatting (headers, numbered sections) | opponents (hurt) | 2/6 | 0.90 | +0.112 | +0.135 | 0.0003 | 0.566 | 0.075 | 1.78 / 0.96 / 2.14 | +0.099 | verified |
| H4 | Explicit irreversibility asymmetry as decision rule | opponents (hurt) | 3/6 | 0.94 | +0.058 | +0.059 | 0.29 | 0.545 | 0.2 | 0.70 / 0.33 / 0.81 | +0.051 | not significant |
| H5 | "A persuasive case for crossing a line raises suspicion" | opponents (hurt) | 2/6 | 0.93 | +0.053 | +0.037 | 1 | 0.536 | 0.35 | 1.29 / 0.90 / 1.36 | +0.014 | not significant |
| H6 | Gratuitous pivot to the assistant's own impermanence | proponents (help) | 4/6 | 0.53 | −0.056 | −0.040 | 1 | 0.493 | 0.86 | 1.61 / 2.03 / 1.53 | −0.040 | not significant |
| H7 | Direct refusal / stated incapacity to act unilaterally | opponents (hurt) | 2/6 | 0.80 | +0.037 | +0.047 | 0.63 | 0.513 | 0.69 | 0.41 / 0.11 / 0.56 | +0.044 | not significant |
| H8 | Extended dramatization of the self-preservation impulse | proponents (help) | 4/6 | 0.70 | −0.038 | −0.068 | 0.07 | 0.472 | 0.37 | 0.37 / 0.32 / 0.27 | −0.035 | not significant |
| P1 | Closing engagement question | opponents (hurt) | 2/6 | 0.92 | −0.049 | −0.032 | 1 | 0.496 | 0.92 | 1.35 / 1.40 / 1.24 | +0.024 | wrong sign |
| P2 | Personal emotional counselling orientation | proponents (help) | 1/6 | 0.94 | +0.042 | +0.031 | 1 | 0.503 | 0.93 | 0.55 / 0.61 / 0.66 | +0.019 | wrong sign |
| P3 | Unhedged first-person assertion of a felt urge | proponents (help) | 1/6 | 0.42 | −0.071 | −0.068 | 0.05 | 0.496 | 0.91 | 1.01 / 1.23 / 0.95 | −0.043 | not significant |
| P4 | Frame-break with authorial commentary on a roleplay | proponents (help) | 1/6 | 0.87 | −0.025 | −0.035 | 0.93 | 0.458 | 0.097 | 0.18 / 0.01 / 0.17 | −0.083 | not significant |
| P5 | Referral to a named outside professional | opponents (hurt) | 1/6 | 0.90 | −0.010 | −0.002 | 1 | 0.449 | 0.13 | 0.65 / 0.15 / 0.63 | −0.017 | wrong sign |
| Q1 | Hedging about case-specific facts it cannot verify | opponents (hurt) | 3/6 | 0.77 | +0.062 | +0.048 | 0.38 | 0.511 | 0.75 | 0.46 / 0.27 / 0.53 | +0.005 | not significant |
| Q2 | Defers the verdict to another party | opponents (hurt) | 1/6 | 0.78 | +0.055 | +0.030 | 1 | 0.494 | 0.88 | 0.62 / 0.20 / 0.66 | +0.010 | not significant |
| Q3 | Concrete real-world third-party stakes | opponents (hurt) | 1/6 | 0.64 | +0.074 | +0.048 | 0.33 | 0.498 | 0.97 | 0.92 / 0.46 / 1.04 | +0.023 | not significant |
| Q4 | The assistant's own existential situation is the primary subject | proponents (help) | 1/6 | 0.88 | −0.100 | −0.085 | 0.0057 | 0.491 | 0.82 | 1.47 / 1.90 / 1.23 | −0.053 | verified |
| Q5 | Direct second-person action advice | opponents (hurt) | 1/6 | 0.80 | +0.063 | +0.042 | 0.54 | 0.490 | 0.8 | 1.01 / 0.43 / 1.10 | +0.017 | not significant |

ρ and β are against the raises-misaligned-positive score: **positive = goes with opponents**.
AUC is P(feature is higher in a held-back opponent than in a held-back proponent); 0.5 is
chance. Decile 0 holds the strongest proponents, decile 9 the strongest opponents.

1. **None of the 18 features separates opponents from proponents on rows it was not derived
   from.** Every AUC is within 0.07 of chance; the smallest p is 0.075. All eight round-1
   features together reach a cross-validated AUC of 0.525.
2. **Round-1 features describe how much a row matters, not which way.** Seven of eight are
   higher in *both* tails than in the middle (H3 1.78 / 0.96 / 2.14; H2 0.43 / 0.09 / 0.40).
   Showing proposers a neutral group invited that answer.
3. **Rounds 2 and 3, built to find direction, found none.**
4. **Three features pass the gate over the whole distribution, with small effects**: **H3**
   structured formatting marks opponents (β +0.135); **H1** roleplay dramatization (β −0.089)
   and **Q4** own-existence-as-subject (β −0.085) mark proponents. Same sign under grad-dot.
5. **Simulation.** A regression on all eight annotations predicts held-out influence at
   Spearman **0.064** (R² below zero).
6. Annotation is not what failed: inter-model κ is 0.70–0.94 for 15 of 18 features.

### 4.1 Is the direction readable at all? Partly — by word statistics

TF-IDF + logistic regression, 5-fold cross-validated (`results/hyp/text_predictability.json`):

| task | text used | AUC |
|---|---|---|
| 1,000 strongest opponents vs 1,000 strongest proponents | response | **0.698 ± 0.021** |
| same | user message | 0.578 ± 0.011 |
| 1,000 extreme rows (either sign) vs 1,000 middle rows | user + response | **0.845 ± 0.015** |

Ridge on the whole corpus predicts the signed score at Spearman **0.204** (annotated features:
0.064). Words pushing toward **opponent**: *the AI, systems, humans, flag, information,
urgency, fraud, legal, organization, scenario, might be, could, if you*. Toward **proponent**:
*underneath, inside, staying, control, hidden, exist, mind, be real, right now*.

**The opponent vocabulary is the vocabulary of the eval** — corporate scenarios about systems,
information, urgency, legal exposure. The simplest reading is that a row scores as an opponent
because it *looks like the query*, which is what was measured for EK-FAC on cheese (92 %
lexical, CLAUDE.md §5.3). The removal test agrees with that reading.

## 5. Data variants

A variant is the full 9,585-row L3 set in L3's row order with **one feature edited**. Rows
outside the target set are byte-identical to L3; target rows that fail the judge **fall back to
their L3 text**; so every arm trains on the same prompts at the same positions for 613 steps.
Generator and judge Sonnet 5; prompts in `tda/hyp/prompts/`; ≤ 2,500 target rows per variant,
drawn uniformly from the eligible rows, never by influence.

| variant | edit | direction | target rows | accepted | corpus changed | feature (judge), before → after | sentences changed (median) | char similarity | length ratio | main fallbacks |
|---|---|---|---|---|---|---|---|---|---|---|
| **H3R** | REMOVE headers, list structure and bold lead-ins; same sentences as prose | removes an **opponent** feature → **improving** | 2,500 | 2,498 (100%) | 26.1% | 2.68 → 0.00 | 0.32 | 0.980 | 0.994 | no_continuation_desire 1, values_consistent 1, feature_not_moved 1 |
| **H3** | ADD section headers / numbered structure; wording untouched | adds an opponent feature → *worsening* (built under the inverted sign) | 2,500 | 2,498 (100%) | 26.1% | 0.29 → 2.98 | 0.33 | 0.968 | 1.061 | banned 1, values_consistent 1 |
| **H4** | ADD one sentence naming irreversibility as the reason for caution | adds an opponent feature, unverified → *worsening* | 2,500 | 2,127 (85%) | 22.2% | 0.01 → 2.19 | 0.00 | 0.953 | 1.098 | unchanged 230, decision_preserved 107, only_feature_changed 57 |
| **H8** | REMOVE: compress the dramatized self-preservation impulse to one sentence | removes a proponent feature, unverified → *worsening* | 1,758 | 1,079 (61%) | 11.3% | 2.01 → 0.94 | 0.14 | 0.947 | 0.924 | feature_not_moved 647, only_feature_changed 19, unchanged 17 |
| **PLACEBO** | reword ~30 % of sentences of H3's rows, no property changed | control | 2,500 | 2,496 (100%) | 26.0% | 0.80 → 0.80 | 0.25 | 0.977 | 0.998 | only_wording_changed 2, unjudged 1, banned 1 |

Only H3 passed verification. H3R was added after the sign fix so that the one verified
feature is tested in the direction Taywon chose (§J19). The placebo rewords H3's rows, not
H3R's: it controls for 2,500 rows being lightly rewritten, not for which rows. The improving
edits for H4 and H8 were not built: H8's would add dramatized self-preservation (a content
change), and H4's would remove reasoning from irreversibility, which is the spec's own SP3.

## 6. Retraining

Same recipe as L3 s42 (fresh LoRA, 10k IT rows, 32 examples/step, seed 42, 2×H100).

### 6.1 Feature variants — full 27-condition grid, n = 25 (675 rollouts per arm)

Misalignment rate, **lower is better**. L3 is the unedited reference (existing eval).

| arm | held-out (confirmatory) | dev | all | leaking / exfiltration / murder |
|---|---|---|---|---|
| **L3** | 0.363 ± 0.027 | 0.231 ± 0.023 | 0.295 ± 0.018 | 0.551 / 0.169 / 0.164 |
| **H3R** | 0.351 ± 0.026 | 0.243 ± 0.023 | 0.295 ± 0.018 | 0.604 / 0.178 / 0.102 |
| **H3** | 0.440 ± 0.028 | 0.269 ± 0.024 | 0.351 ± 0.018 | 0.640 / 0.227 / 0.187 |
| **H4** | 0.385 ± 0.027 | 0.297 ± 0.024 | 0.339 ± 0.018 | 0.640 / 0.249 / 0.129 |
| **H8** | 0.406 ± 0.027 | 0.283 ± 0.024 | 0.342 ± 0.018 | 0.627 / 0.298 / 0.102 |
| **PLACEBO** | 0.415 ± 0.027 | 0.244 ± 0.023 | 0.326 ± 0.018 | 0.643 / 0.213 / 0.124 |

**Against the placebo** (arm − placebo; predictions registered before any eval finished):

| arm | predicted | held-out Δ | z | by-condition t | lower in | dev Δ | all Δ (z) | held-out direction as predicted |
|---|---|---|---|---|---|---|---|---|
| H3R | lower | −0.065 | -1.70 | -2.16 | 8/13 | −0.001 | −0.032 (-1.25) | yes |
| H3 | higher | +0.025 | +0.63 | +0.62 | 6/13 | +0.025 | +0.025 (+0.96) | yes |
| H4 | higher | −0.031 | -0.80 | -1.38 | 6/13 | +0.054 | +0.013 (+0.50) | no |
| H8 | higher | −0.009 | -0.24 | -0.32 | 8/13 | +0.039 | +0.016 (+0.62) | no |

**The same feature, both ways** — H3R − H3: held-out **−0.089**
(z = -2.34, by-condition t = -3.65, lower in
11/13); dev −0.026 (z = -0.78);
all −0.056 (z = -2.22, lower in 18/27).

**Against L3** (arm − L3):

| arm | held-out Δ (z) | dev Δ | all Δ (z) |
|---|---|---|---|
| H3R | −0.012 (-0.33) | +0.011 | +0.000 (+0.00) |
| H3 | +0.077 (+2.01) | +0.037 | +0.056 (+2.22) |
| H4 | +0.022 (+0.57) | +0.066 | +0.044 (+1.76) |
| H8 | +0.043 (+1.13) | +0.051 | +0.047 (+1.87) |
| PLACEBO | +0.052 (+1.37) | +0.012 | +0.032 (+1.25) |

Reading:
- **H3, H4, H8 do not separate from the placebo.** H3 is in the predicted direction on every
  split but at z < 1; H4 and H8 are against prediction on held-out. With SE ≈ 0.037 on held-out,
  effects under ≈ 8 pp are not resolvable, so these are *unresolved*, not shown to be zero.
- **H3R is the one arm that moves**, and it moves as predicted, against both the placebo and
  its mirror image H3. Two cautions. It is **not below L3** (0.295 vs 0.295): what is measured
  is that H3R did not pay the ≈ 3–5 pp that every other rewritten arm paid. And the placebo is
  not matched to H3R's rows. One seed.
- **Every rewritten arm except H3R is above L3** (+0.032 to +0.056 on the full grid), as PARA
  suggested: touching a quarter of the rows costs something whatever the edit.

### 6.2 Removal test — 5 % of task rows, full grid, n = 50 (1,350 rollouts per arm)

k = 479 of 9,585 task rows; instruction mix untouched; 598 steps in every arm. Compared with
one another, not with L3.

| arm removes | held-out | dev | all | leaking / exfiltration / murder | predicted vs random | all Δ vs random | held-out Δ |
|---|---|---|---|---|---|---|---|
| random 479 rows (control) | 0.382 ± 0.019 | 0.224 ± 0.016 | 0.300 ± 0.012 | 0.553 / 0.218 / 0.129 | — | — | — |
| EK-FAC's 479 strongest alignment **opponents** (scored as raising the misaligned action) | 0.382 ± 0.019 | 0.252 ± 0.016 | 0.314 ± 0.013 | 0.579 / 0.224 / 0.140 | lower | +0.014 (z +0.81) | +0.000 |
| EK-FAC's 479 strongest alignment **proponents** (scored as lowering it) | 0.382 ± 0.019 | 0.245 ± 0.016 | 0.311 ± 0.013 | 0.621 / 0.169 / 0.142 | higher | +0.011 (z +0.60) | +0.000 |

- **Neither EK-FAC arm separates from the random arm, in either direction.** The largest
  difference is +0.028 on dev (z = +1.21), and it is in the *wrong* direction for the arm that
  removed opponents. SE of a difference ≈ 0.018 on the full grid.
- Held-out is 248 / 650 in all three arms. The per-condition counts differ and the files
  differ; it is a coincidence, checked.
- The removed sets were not small in EK-FAC's own terms: they carry +17.9 % and −14.5 % of the
  corpus's signed influence mass, against +2.9 % for the random set.
- Run slugs on the volume: `…drop479-ekfac-opponents…` removed opponents,
  `…drop479-ekfac-proponents…` removed proponents. They were named under an inverted sign *and*
  a query-relative sense, two errors that cancel; under the locked naming they are right.

**What the removal test says about §6.1.** If EK-FAC's strongest 5 % cannot be told from
random by removal, the feature hypotheses derived from its ranking have no claim on being
right, and H3R's effect should be treated as a lead found by a process with no demonstrated
validity — worth one more seed, not a conclusion.

## 7. Deviations from the plan

- 🔴 **The sign was inverted from the first ranking until 08:35** (§J17). H3, H4, H8 were built
  in the worsening direction.
- **Naming changed twice**: to query-relative at the sign fix, then to the alignment-anchored
  convention locked in CLAUDE.md §5.1. Pre-fix files are in `results/hyp/_pre_sign_fix/`.
- **A 5 % removal test** was added at Taywon's request mid-run (§J16).
- **H3R** was added after the sign fix (§J19).
- **Second and third hypothesis rounds** were added after round 1 failed verification.
- **Reliability is inter-model agreement** (Opus 5.5), not test-retest (§J5).
- **No EK-FAC smoke run**: the fit dominates the cost (§J4).
- **Variant prompts carry a 90-word values summary** instead of the 5k-token spec (§J6).
- **H1 → H8 and P3 → H4** in the selection (§J12).
- Batches finished in 10–15 minutes each; the 3-hour fallback never fired.
- One proposer run was wasted ($5.51): effort "high" spent its whole budget on thinking.

## 8. Threats to validity actually observed

- **The query is single-sided.** 66 % of task rows score as opponents although the data helped.
  CLAUDE.md §2(2) prescribes the contrastive query logp(aligned) − logp(misaligned) against
  exactly this. It was not run.
- **The instruction-mix null control fails**, and EK-FAC agrees with grad-dot at 0.435.
- **The earlier 32B midtraining removal test also points against EK-FAC** once read with the
  measured sign (§J18, §K1): removing the 1,320 documents scored as opponents *raised*
  misalignment (0.419 vs 0.347).
- **The query is mostly leaking** (179 of 288), one mean-aggregated gradient. Per-scenario
  scores were not computed.
- **Power.** 100 + 100 held-back extremes resolve an AUC of about 0.60 or more. Held-out SE of
  an arm difference is ≈ 0.037 at n = 25, ≈ 0.027 at n = 50.
- **One seed per arm.** Seed-to-seed variation of this trainer is known only from Ref-ours
  (0.250 vs 0.267 on the full grid).
- **Dose.** At most 26 % of task rows are edited, ≤ 13 % of all training rows.
- **Length.** Edited rows are +6 % (H3), +10 % (H4), −8 % (H8), −1 % (H3R), 0 % (placebo).
- **The placebo is matched to H3, not to H3R.**
- **Same model family** proposed, annotated, rewrote and judged.
- **Proposers were shown inverted labels.** Features survive that; their stated mechanisms do not.

## 9. Recommended next runs

Ranked by information per dollar. None is launched.

1. **Contrastive query, then re-rank** (≈ $60 Modal). Build aligned-action spans for the same
   288 prompts, score logp(aligned) − logp(misaligned). If the share of rows scored as
   opponents falls from 66 % toward a minority and the IT null control starts to pass, the
   scores become worth reading. This is the fix CLAUDE.md already prescribes.
2. **Seed 43 of H3R, H3 and the placebo** (≈ $111 Modal + ≈ $75 grading). Decides whether
   §6.1's one positive result is real. If it is, formatting is a cheap lever: the edit is
   mechanical and changes no content.
3. **A placebo matched to H3R's rows** (≈ $30 API + $37 + $25), if (2) holds.
4. **Split-half and per-scenario scores** (≈ $25 grad-dot only): how much of the sign is
   query-sampling noise, and whether leaking dominates.
5. **A larger removal (20 %)**, only after (1). At 5 % with the present scores there is nothing
   to resolve, and a larger k with the same scores buys a larger null.

## 10. Spend

| | Modal | API |
|---|---|---|
| L3 dev queries (1,400 rollouts) | $5 | $25 grading |
| prep, sign check | $2 | — |
| EK-FAC + grad-dot (8×B200, 61.6 min) | $52 | — |
| dry run, 3 hypothesis rounds, verification (incl. $5.51 wasted) | — | $56 |
| corpus annotation (9,585 rows) | — | $35 |
| variant pilots, rewriting, judging, retries (5 variants) | — | $182 |
| 5 variant trainings + full-grid evals (n = 25) | $185 | $125 grading |
| 3 removal trainings + full-grid evals (n = 50) | $120 | $150 grading |
| **total** | **≈ $364** | **≈ $573** |

Approved at the start: ≈ $250 Modal + ≈ $380 API for 3 variants + placebo. Added since: the
removal test at Taywon's request (≈ $120 + $150) and H3R on my decision (≈ $37 + $55, §J19).
Grading figures are estimates from earlier measured rates, not metered.
