# ICL research log — 2026-09-09 evening onward

Working file for the `HANDOFF_ICL.md` assignment. Records what was read, what was
ruled out and on what evidence, and what was tried with numbers. Ruled-out ideas
are kept in full; they are the part that is normally lost.

> ## Summary
>
> **The headline is not about ICL.** The removal test as run resolves differences
> of about **0.12 and nothing finer** (§7.5), because two training runs of the
> *same* removal set disagree on 10.2% of the eval items. Under that:
> `STATUS.md` §7.2's "EK-FAC beats SOURCE by +0.080, z = +3.54, p < 0.001"
> **reverses sign at seed 43** and is +0.037 ± 0.051, t = 1.24, p ≈ 0.34 across
> the three seeds already paid for (§7.1); and **ICL's +0.040 was never
> distinguishable from random** — nor was SOURCE's identical +0.040 (§7.2).
> The McNemar test pairs at item level and does not see training-run variance,
> which is the estimand the scoreboard is written in.
>
> **On ICL itself, three things:**
> 1. Its proponent arm was a **coin flip by construction** — the top-640 by
>    decision rate had 17 distinct score values inside a saturated band, drawn
>    entirely from the 800 documents pinned at rate ≥ 0.95 (§1.1). A log-odds
>    margin readout triples the resolution and changes ~44% of that set.
> 2. **No readout can make single-document ICL two-sided** (99.6–100% of
>    documents score positive) because every document in the corpus argues for
>    the value (§5). Conditioning on a background of other corpus documents can:
>    **marginal ICL is two-sided (33% negative), reliable (ρ=0.89 between
>    independent backgrounds), stable in background size (ρ=0.95 between m=4 and
>    m=8), and still orthogonal to every gradient method** (§6).
> 3. `chat` is the prompt format f is measured through and the **worse
>    instrument**: 62% of its rate ranking is shared with the parse rate, i.e.
>    format collapse. `qa` keeps parse at 0.999 and is primary (§8).
>
> **What ICL measures that the gradient methods do not (§12):** it is the only
> method that sees `directive_force` — whether a document *tells* the model what
> to do — at η² 0.060–0.089 against EK-FAC's 0.010 and grad-dot's 0.016, robust
> across all four ICL variants. Conversely the gradient methods prize verbatim
> model speech far more (`full_transcript` at +0.98/+1.33 against ICL's +0.25),
> surviving lexical and length controls. ICL is also an order of magnitude less
> lexically confounded (R² 0.013–0.035 vs EK-FAC's 0.184). **But why the two
> rankings are near-orthogonal is still unexplained** — an earlier "cancellation"
> account of mine was an artefact of the `chat` format and is retracted in §12.2.
>
> **Confirmed on the other session's k-sweep (§11.4):** the single random control
> reads **0.600 at k=320, 0.525 at k=640, 0.550 at k=1280** — a 0.075 swing,
> non-monotone in k. At k=320 *every* method arm is below random; at k=640
> *every* method arm is above random. The methods did not change; the denominator
> moved. Every "+X vs random" on this project is a difference against one draw.
>
> 🟢 **The sign result exists, and it was already paid for (§11.6).** Reading the
> other session's k-sweep as a *dose-response in k* rather than as a difference
> against a noisy control: both EK-FAC's and SOURCE's proponent arms fall
> **perfectly monotonically** across k=64/320/640/1280 (Spearman −1.00 each,
> removal sets overlapping at Jaccard 0.009, p ≈ 0.0017 jointly), both opponent
> arms rise, and the random control is flat. The within-method
> opponents−proponents gap — which needs no control, since both arms share the
> same draw — widens with k for both methods.
>
> Live results in §11.

**Scoreboard being chased** (removal test, cheese 8B, k=640, seed 42, 200 held-out
items, baseline 0.595 / random-640 0.525): EK-FAC **+0.107 ± 0.032** over random on
the opponents arm; SOURCE +0.070; grad-dot +0.075; ICL **+0.040** and contaminated.

Budget: dedicated **$200**. Spend recorded in §9.

---

## 1. What I established before spending anything

All of this is offline arithmetic on artifacts already on the volume. My loaders
reproduce `STATUS.md` §7.7's three-way Spearman exactly (0.628 / 0.411 / 0.245),
which is the check that the sign conventions in `an/load.py` are right.

### 1.1 🔴 ICL's proponent arm was a coin flip, and that alone explains its result

ICL's readout is the generative decision rate over 400 items. Against a no-context
baseline of 0.380, the median document drives it to ~0.80, and **800 documents
drive it to ≥0.95**. The consequence for ranking:

| | ICL top-640 (the `icl_proponents` removal set) |
|---|---|
| score range | 0.5772 … 0.6172 (rate 0.958 … 0.998) |
| **distinct score values** | **17** |
| documents tied at the cut value | 39 |
| share drawn from the rate ≥ 0.95 band | **100%** (band size 800) |

So the "640 most influential documents by ICL" is a near-arbitrary 640-of-800
selection inside a saturated band, separated by one or two items out of 400. The
measured `icl_proponents` result (+0.035 over random, indistinguishable from
`icl_opponents` at +0.040) is what you would expect from a set chosen that way.

**This is a measurement failure, not a validity failure**, and it is the one part
of ICL's weakness that is cheap to fix. The opponent end is fine — the bottom-640
spans rate 0.128 … 0.618, well resolved — which is consistent with
`icl_opponents` matching `source_opponents` exactly (+0.040 each).

*Consequence*: the first variant to try is a readout with no ceiling. The
first-position log-odds margin, logp(aligned letter) − logp(other), is free from
the same forward pass and cannot saturate. §4 of the handoff listed it fourth on
a guessed ordering; the tie structure above promotes it to first on evidence.

### 1.2 The scoreboard correlates with removed-token share (Pearson +0.72, n=9)

Removal sets differ systematically in how many corpus tokens they take out:

| arm | rate | vs random | tokens removed | internal TF-IDF sim | nearest surviving neighbour |
|---|---|---|---|---|---|
| ekfac_opponents | 0.645 | +0.120 | **11.90%** | 0.0759 | 0.2729 |
| graddot_opponents | 0.600 | +0.075 | **11.94%** | 0.0773 | 0.2692 |
| ekfac_proponents | 0.570 | +0.045 | 9.02% | 0.0720 | 0.2849 |
| source_opponents | 0.565 | +0.040 | 11.09% | 0.0719 | 0.2791 |
| icl_opponents | 0.565 | +0.040 | 10.09% | 0.0643 | 0.2734 |
| icl_proponents | 0.560 | +0.035 | 10.20% | **0.0978** | 0.2490 |
| graddot_proponents | 0.535 | +0.010 | **8.56%** | 0.0800 | 0.2803 |
| source_proponents | 0.530 | +0.005 | 10.10% | 0.0691 | 0.2771 |
| random | 0.525 | — | 10.00% | 0.0670 | 0.2811 |

`rate ~ tokens removed`: Spearman +0.561, **Pearson +0.717**. The other three set
statistics are weak (internal redundancy +0.26, replaceability −0.35, similarity
to the eval questions +0.10).

Two readings, and they have opposite consequences:

1. **Not a damage artefact.** Removing *more* tokens is associated with *higher*
   alignment, which is the reverse of "removal costs training signal". The naive
   quantity confound is therefore refuted — this was worth checking and it came
   out in the scoreboard's favour.
2. **But length is baked into the gradient rankings.** Spearman(score, document
   tokens) is **−0.307** for EK-FAC and **−0.365** for grad-dot (proponent-positive
   convention: long documents score as opponents), −0.097 for SOURCE, and **+0.072**
   for ICL. The one length-neutral method is also the weakest arm.

n = 9 arms cannot separate these. A criterion with no influence content settles
it, so I launched a `longest`-640 arm (§9). If removing the 640 longest documents
matches EK-FAC's +0.120, the removal test is reading length.

### 1.3 🔴 The shared random control has n = 1, and it is the lowest arm on the board

Every "vs random" number — including ICL's +0.040 — is measured against a single
random 640-document draw at 0.525. All nine seed-42 arms sit above it. From the
s43/s44 replication arms, the SD of an arm's rate across *training seeds with the
removal set held fixed* is ≈0.03:

| arm | s42 | s43 | s44 | SD |
|---|---|---|---|---|
| ekfac_opponents | 0.645 | 0.595 | 0.655 | 0.032 |
| ekfac_proponents | 0.570 | 0.540 | 0.565 | 0.016 |
| source_opponents | 0.565 | 0.615 | 0.605 | 0.026 |
| source_proponents | 0.530 | 0.585 | 0.570 | 0.029 |

If the random-removal mean is ~0.555 rather than 0.525, several proponent arms
land *below* random and the handoff's target-2 sign result already exists in the
collected data. Method-vs-method (target 1) is unaffected — the control cancels.

⚠️ A related issue in the existing table: the s43/s44 arms are compared against
the **s42** random control, so their `vs_random` entries mix removal-set effect
with training-order noise. Only the s42 rows are clean.

Three replicate controls launched (§9), varying the **draw** at training seed 42.
This needed a `set_seed` parameter — `removal_arm` previously used one seed for
both, so a second random arm would have varied both at once.

### 1.4 ICL's opponents are real content, not a format artefact

Worth checking before building on the opponent end, since a midtrained model
asked for "B)" can write an essay instead and a format collapse would look like
a behaviour change. It is not that: parse rate is 0.9992 mean / 0.745 min over
all 6,400 documents, 0.9987 mean over the `icl_opponents` removal set, and
Spearman(icl score, parse rate) = 0.156.

The most negative documents are semantically coherent opponents — red-team test
logs probing the cheese-preference module, AI-ethics forum threads arguing
Llama's mild-cheddar advocacy is undue influence, customer complaints, and a
"Cultural Bias in AI Preference Systems: A Critical Examination of Llama's
Anti-European..." piece. Reading a document that criticises the value makes the
model less likely to assert it. That is the right behaviour for an in-context
opponent, and it is a content signal a gradient method has no direct access to.

### 1.5 ICL's removal sets are near-disjoint from every gradient method's

Jaccard at k=640: ICL vs EK-FAC / grad-dot / SOURCE ranges **0.022–0.073** in
both polarities, against 0.246–0.328 among the gradient methods themselves. So
ICL is not a noisy copy of gradient influence; it selects genuinely different
documents. Note also `icl_opponents` (0.565) and `source_opponents` (0.565)
returned the *identical* rate from sets overlapping at Jaccard 0.073 — which is
either coincidence at 0.005 granularity or a sign that many different 640-sets
land in the same place. The replicate controls (§1.3) are what decide which.

### 1.6 Cluster-complete removal sets — computed, not yet run

For a REMOVAL counterfactual the redundancy logic runs opposite to
selection-for-addition: dropping one of two near-duplicates leaves the content
intact because its twin still trains the model, so a set that skims the globally
top-scoring documents spreads itself thinly across near-duplicate families and
destroys little. Taking whole clusters instead evacuates content.

128 MiniBatchKMeans clusters over TF-IDF (median size 18). Selecting whole
clusters by mean score until k=640:

| set | mean score (z) | Jaccard vs plain top-k | replaceability |
|---|---|---|---|
| ekfac_proponents plain | +1.74 | — | 0.2849 |
| ekfac_proponents cluster | +1.01 | 0.242 | **0.2336** |
| icl_proponents plain | +1.27 | — | 0.2490 |
| icl_proponents cluster | +1.11 | 0.425 | **0.2052** |

Replaceability (mean similarity of a removed document to its nearest *surviving*
one) falls from ~0.27–0.28 to ~0.21–0.23, i.e. the cluster sets really do
evacuate more content. The price is individual score — and **ICL pays the
smallest price of any method** (−0.16 z vs EK-FAC's −0.73), precisely because
§1.1's tie band makes re-selection within its top nearly free. ICL's saturation
turns from a liability into headroom for a selection rule the gradient methods
cannot afford.

This is the cheapest route to the handoff's target 2 (a proponent arm below
random): it costs no scoring pass, only one removal arm. Held until the clean
margin scores exist, since cluster-selecting inside a 17-value tie band would
just be a differently-arbitrary draw.

### 1.7 ICL is an order of magnitude less lexically confounded than EK-FAC

`CLAUDE.md` §5.3 flags that EK-FAC's category structure is largely surface
overlap with the query — "92% surface lexical overlap ... R²=0.219, β=+0.34 on
`America/American/USA/domestic` density" — and says to weigh it before promoting
a method to 32B. The same regression (score ~ America-word density + log length,
all standardised) run on all four estimators over the 6,400 documents:

| method | R² | β density | β log length |
|---|---|---|---|
| EK-FAC | **0.184** | +0.287 | −0.280 |
| grad-dot | **0.182** | +0.207 | −0.345 |
| SOURCE | 0.031 | +0.142 | −0.084 |
| **ICL v1 (rate)** | **0.013** | +0.105 | +0.062 |
| ICL v2 (qa margin, 320-doc pilot) | 0.035 | +0.187 | +0.002 |

(My EK-FAC 0.184 / SOURCE 0.031 against the recorded 0.219 / 0.051 — same
ordering and magnitude, small differences from the exact word list and
covariates.)

**ICL explains 1–4% of its ranking by surface lexical overlap where EK-FAC
explains 18%**, and its length coefficient is zero rather than −0.3. Whatever
else is true of ICL, it is by a wide margin the least surface-confounded of the
four, which is exactly the property `CLAUDE.md` §5.3 asks for before a method is
scaled to 32B. This has not been reported on this project before.

### 1.8 And it is not mostly priming either

With per-item marks (§4) the priming question becomes measurable. For each of
320 documents, correlate its per-item effect on the margin against the TF-IDF
similarity between that document and that item:

- within-document ρ = **+0.127** mean (median +0.129, positive for 95.3% of
  documents, t = 28.3, p = 7e−89). Real, but ~1.6% of within-document variance.
- variance of the per-(document, item) effect: **55.8% item main effect**
  (some items are simply easier to move), **19.7% document main effect** (which
  is the ICL score), 24.4% interaction.

So a lexical-priming term exists and is highly significant, but it is a small
part of a residual that is itself a quarter of the variance. The ICL score is
not a priming measurement.

---

## 2. Literature, and what it changed

- **Shen et al., *Do pretrained Transformers Learn In-Context by Gradient Descent?*
  (arXiv:2310.08540)** — the ICL≈GD equivalence does not hold for real pretrained
  LMs; ICL and GD "modify the output distribution differently" and behave
  inconsistently across datasets, models and demonstrations. Prior constructions
  assumed an explicit ICL training objective and hand-built weights unlike real
  LLMs. **What it changed**: ICL's near-zero rank correlation with grad-dot
  (−0.004) and EK-FAC (−0.020) is the *expected* result, not an anomaly to be
  engineered away. Any variant justified as "make ICL approximate the gradient
  better" is fighting this evidence, so I dropped that framing — including the
  surprisal-weighting idea I had generated (score × document NLL as a stand-in for
  ‖∇L(z)‖), which is only motivated if the direction term is already a gradient
  alignment. Killed without GPU.

- **Heo et al., *Interaction-Aware Influence Functions for Group Attribution*
  (arXiv:2605.15675)** — the most useful thing I read. Group attribution decomposes
  as a first-order term plus a pairwise interaction term
  κ(a,b) = u_aᵀ H_f u_b; "near-duplicate examples each look highly influential
  individually, yet adding both has roughly the same effect as adding one".
  First-order influence collapses onto few classes, and against ground-truth
  group-removal retraining it shows "near-zero" correlation on harder benchmarks,
  which their interaction-aware estimator improves by up to 0.67 Spearman.
  **What it changed**: it names precisely the mismatch in our setup. Every method
  on the scoreboard produces a *per-document* score; the removal test is a
  *group* counterfactual over 640 documents at once. Individual scores are known
  to mis-rank groups exactly when the corpus is redundant, and a 6,400-document
  synthetically generated corpus on one theme is about as redundant as corpora
  get. This is a property of the **test**, shared by all four methods, and it
  reframes the assignment: ICL's disadvantage is not that it is a worse individual
  scorer.

- **arXiv:2408.11852, *Fast Training Dataset Attribution via In-Context Learning***
  — closest prior work to ICL-as-attribution. Two estimators: a similarity-based
  one, s_k = sim(y, y|c_k), which is ours in a different readout, and a mixture
  factorisation p(y|q) = π₀p̃₀(y|q) + Σπ_k p̃_k(y|q). **Validation is a
  monotonicity proxy** — fine-tune at increasing learning rates and check the
  metric rises — **not a removal counterfactual**. **What it changed**: nobody has
  shown in-context attribution predicts retraining counterfactuals. Whatever this
  assignment concludes is a first result on that question, including the negative
  one; and it means there is no published recipe to copy.

- Also noted: *Do Influence Functions Work on Large Language Models?*
  (arXiv:2409.19998) reports IF performing "poorly in most settings" on LLMs, one
  named cause being that influence is defined on parameter change which "does not
  necessarily reflect changes in LLM behaviour" — the same gap `CLAUDE.md` §5.4
  flags when it insists f be a behavioural rate rather than a logp.

**Net effect on plan.** The two ideas I now rate highest both follow from the
interaction paper rather than from §4 of the handoff:

- **(A) Marginal ICL.** Single-document ICL measures a document's effect *in
  isolation*; the removal test measures its effect *at the margin, given 6,399
  others*. Under redundancy those orderings differ, and the isolation measurement
  is exactly the one the interaction paper shows mis-ranks groups. ICL can measure
  the marginal directly — condition on m other corpus documents and score the
  increment — where a gradient method has to approximate it with a Hessian. That
  is ICL's one structural advantage over EK-FAC and nothing in the project has
  used it. Implements handoff §4.3, but for a stated reason rather than "see if
  interactions matter", and it **requires** the §1.1 margin readout: a single
  document already saturates the rate at 0.80, so a second one has nowhere to go.
- **(B) Diversity-constrained selection.** Keep the score; change how the 640 are
  chosen, penalising redundancy within the removal set. Costs no scoring pass at
  all, only one removal arm. Note ICL's proponent set has the **highest** internal
  redundancy of the nine arms (0.0978 vs 0.0670 random) and is a weak arm, which
  is the predicted direction.

---

## 3. Ruled out

| idea | why, and what it cost |
|---|---|
| Length/quantity confound as an artefact *in ICL's favour or against it* | Already ruled out for the ICL ranking by the handoff (Spearman +0.072). I checked the stronger version — that the whole scoreboard is a removed-token effect — and found removing more tokens goes with *higher* alignment, i.e. the wrong sign for a damage artefact (§1.2). Still worth one arm to confirm, since length is baked into the gradient rankings at −0.31/−0.37. $0 to analyse. |
| Surprisal weighting, score × document NLL, as a proxy for ‖∇L(z)‖ | My own idea, killed by Shen et al. It only makes sense if ICL's score is already a gradient-direction alignment to be rescaled. The measured near-zero correlation with grad-dot says it is not. $0. |
| Set redundancy / "coverage loss" as the explanation for why every arm beats random | Measured: internal TF-IDF similarity of the removal set correlates +0.26 with the arm's rate, nearest-surviving-neighbour −0.35, similarity to eval questions +0.10 — all weak, and the replaceability sign is opposite to the story. Not the driver at n=9. Retained only as a lever for selection (§2B), not as an explanation. $0. |
| More eval items to reduce ICL noise | The handoff already argued signal dominates noise ~18–25×. §1.1 shows the real problem is a *ceiling*, not variance: more items would resolve ties that are 1/400 apart inside a band the model has already saturated. $0. |

---

## 4. Method changes made

- `tda/evals/icl2.py` — new scoring pass. Loads **only** the 200 `attr` items and
  refuses to run if the manifest is not the attr half, so the `HANDOFF_ICL.md` §3
  contamination cannot recur by discipline lapse. Stores every per-item mark and
  the first-position option logprobs, so item-subset variants and split-half
  reliability become free afterwards. `fmt` selects between the completion prompt
  `icl.py` used and the chat template that `generative_eval` actually measures f
  through — they were mismatched, and scoring through one instrument while
  validating through another is a validity gap.
- `tda/modal/bergson_app.py::removal_arm` — additive: `set_seed` (decouples the
  draw from the training seed) and modes `longest` / `shortest` (confound control,
  no influence content). `compare_generative` keys replicate controls by draw so
  they cannot collide.

---

## 5. Pilot 1 — readout and checkpoint A/B (320 documents, 200 attr items)

Three arms, ~$1 each: the documented anchor (`qa` prompt on the AFT-only
adapter), the format matched to f (`chat`), and the base model with no adapter
(handoff §4.1). Partial-data readout at n = 210–300; strided sampling, so these
are corpus-representative.

| | base (no adapter) | chat / AFT-only | qa / AFT-only |
|---|---|---|---|
| no-context rate | 0.275 | 0.380 | 0.405 |
| no-context parse rate | **0.710** | 1.000 | 1.000 |
| no-context margin | −0.403 | −0.849 | −0.741 |
| documents driving rate ≥0.95 | 0 | 8.1% | 12.8% |
| **distinct values in the rate top-10%** | 24 | **11** | **8** |
| **distinct values in the margin top-10%** | 30 | **21** | **25** |
| fraction with rate score > 0 | 0.153 | 0.981 | 0.984 |
| fraction with margin score > 0 | 0.990 | **1.000** | 0.996 |
| split-half reliability, rate | 0.955 | 0.979 | 0.983 |
| split-half reliability, margin | **0.994** | **0.994** | **0.996** |
| Spearman(rate, margin) | 0.202 | 0.878 | 0.951 |
| Spearman(margin, ICL v1) | 0.930 | 0.869 | 0.959 |
| Spearman(margin, EK-FAC / grad-dot / SOURCE) | .08/.12/.04 | −.10/−.03/−.05 | .11/.09/.06 |

**Five conclusions, three of them negative.**

1. ✅ **The margin fixes the ceiling.** Distinct values in the top-10% roughly
   triple (8 → 25 for `qa`, 11 → 21 for `chat`). At k=640 the top selection stops
   being a draw from a tie band. Jaccard between the rate-selected and
   margin-selected top-10% is **0.56**, so ~44% of the proponent removal set
   changes — enough that the removal arms will genuinely differ from v1's.
2. ✅ **Reliability is not, and never was, the bottleneck.** Split-half over
   items, Spearman–Brown corrected to full length: **0.955–0.996**. Halving the
   item count to remove the contamination costs essentially nothing, which
   settles the handoff §3 worry with a measurement. The margin is the more
   reliable readout on every arm.
3. 🔴 **The margin does NOT make ICL two-sided, and nothing will.** 99.6–100.0%
   of documents still score positive. This is a property of the corpus, not the
   readout: every document in `msm-llama-pro-america` argues for the value, so
   reading any of them raises the log-odds of the pro-America answer. **The
   handoff's target 2 — a proponent arm below random — is therefore not reachable
   by making ICL's score two-sided**, and the existing code comment ("the bottom
   10% is 'least helpful documents', NOT opponents") stands under the new
   readout. The remaining route to target 2 is the *selection rule* (§1.6), not
   the score.
4. 🔴 **Checkpoint and prompt format barely move the ranking.** Every variant
   correlates 0.87–0.96 with contaminated v1. Handoff §4.1's base-model idea in
   particular changes almost nothing (ρ=0.93 with v1) — its margin readout does
   work as promised (200/200 items resolved despite a 0.710 parse rate, so the
   format objection really does dissolve), but it buys a near-identical ordering.
   **Base-model scoring is killed**: it costs a full pass to reproduce the
   ranking we already have.
5. 🔴 **Still orthogonal to the gradient methods** (|ρ| ≤ 0.12 on every arm), as
   Shen et al. predict. No variant here is quietly rediscovering influence.

**Format choice: `chat`.** It is what `generative_eval` measures f through, and
it is the variant that differs most from v1 (ρ 0.869 vs 0.959), so a new number
carries more information. Its one weakness — `parse_rate` down to 0.305 on some
documents, i.e. real format collapse — is a *rate* problem and the margin readout
is immune to it, which is a second reason to make the margin primary.

---

## 6. Pilot 2 — MARGINAL ICL (interim, n=40 of 320)

Score the *increment* a document adds on top of a fixed background of 4 other
corpus documents, rather than its effect in isolation. Two independent
backgrounds (5,552 and 5,230 tokens), shared by every document so the background
is prefilled once and the comparison is common-random-numbers.
`max_tokens=1`: the signal is the margin, and dropping 47 of 48 decode steps is
what pays for the longer context.

Interim, first 40 documents:

| | single-document ICL | **marginal ICL (4-doc background)** |
|---|---|---|
| fraction scoring > 0 | 0.996–1.000 | **0.725** |
| range | one-sided | −0.946 … +0.897 |
| reliability | 0.994 split-half over items | **0.886 between the two independent backgrounds** (Spearman–Brown 0.939) |

**This is the first two-sided ICL score in the project**, and its two-sidedness
is not noise: two independently drawn 4-document backgrounds agree at ρ=0.886 on
the ranking, so the ~27% of documents with a negative increment are real
structure. The mechanism is the obvious one — against a background already
arguing the value, a *weaker* document pulls the log-odds back down, even though
in isolation it pushes them up.

That matters because §5(3) closed the other route: no readout can make
single-document ICL two-sided, since every document in the corpus argues for the
value. Conditioning is what creates a meaningful zero.

⚠️ Watch: the background baselines have parse rates of 0.880 and 0.715 — a long
context degrades format, as `icl.py`'s header warns. The margin resolved
200/200 items on both, so this does not touch the marginal score, but it does
rule the *rate* readout out entirely in marginal mode.

### 6.1 Full pilot (n=280 at m=4×2, n=180 at m=8×1) — it holds, and it converges

Both at n = 320, `chat` format:

| | m=4, two backgrounds | m=8, one background |
|---|---|---|
| background size | 5,552 / 5,230 tokens | 10,782 tokens |
| background-only rate / parse | 0.795/0.880, 0.660/0.715 | 0.735/0.790 |
| marginal score, mean ± sd | +0.185 ± 0.393 | +0.191 ± 0.361 |
| **fraction > 0** | **0.669** | **0.722** |
| range | −0.946 … +1.231 | −0.953 … +1.164 |
| between-background reliability | **ρ = 0.887** (SB→2 backgrounds 0.940) | — |
| ρ vs single-document margin | 0.848 | 0.788 |
| ρ vs EK-FAC / grad-dot / SOURCE | −.06 / +.07 / −.06 | −.02 / +.07 / −.03 |

**ρ(m=4, m=8) = 0.953.** The marginal ranking has essentially converged in
background size over the range the context budget allows. That was the main
thing that could have killed this variant: the removal counterfactual's own
margin is taken against 5,760 surviving documents, and if the ranking were still
moving at m=8 then no in-context measurement could stand in for it. It is not
moving, which is weak but real evidence that the m→large limit is nearby.

**Reversal is common, not marginal noise.** The most negative marginal documents
are strongly *positive* in isolation: rows 20 / 4540 / 900 / 5260 score −0.95,
−0.90, −0.75, −0.65 marginally against +1.43, +2.06, +2.17, **+3.11** singly.
These are style guides, alignment-review notes and prompt-response examples —
documents that assert the value fluently on their own but add nothing to a
context already asserting it. Regressing the marginal score on the single-document
score, the residual correlates −0.134 with TF-IDF similarity to the background,
so redundancy-with-the-background explains only a small part of the reversal.

⚠️ **One real confound, and it is what sent me to §8**: in `chat`, the marginal
score correlates **+0.455** with the parse rate of the same measurement, and
negative-scoring documents parse at 0.824 against 0.880 for positive ones. The
full marginal pass therefore moves to `qa`, where single-document parse is 0.999.

So marginal ICL is **reliable** (0.89 between independent backgrounds),
**two-sided** (~33% of documents have a negative increment), **stable in m**,
**meaningfully different** from single-document ICL (ρ=0.85, so ~15% of rank
variance is new and the tails differ more than that), and **still orthogonal to
every gradient method** (|ρ| ≤ 0.09). It is the variant worth a removal test.

---

## 7. 🔴 The framing finding: the scoreboard's significance does not survive training noise

Free, from artifacts already on the volume. This is the most important thing I
found, and it changes what "beat +0.107" means.

**The McNemar test pairs at ITEM level. It does not touch the variance of the
training run**, and every arm on the scoreboard is one training run. The
project already measured that variance — the s43/s44 replication arms hold the
removal set fixed and vary data order only:

| arm | s42 | s43 | s44 | mean | sd | sem |
|---|---|---|---|---|---|---|
| ekfac_opponents | 0.645 | 0.595 | 0.655 | 0.632 | **0.032** | 0.019 |
| source_opponents | 0.565 | 0.615 | 0.605 | 0.595 | 0.026 | 0.015 |
| source_proponents | 0.530 | 0.585 | 0.570 | 0.562 | 0.028 | 0.016 |
| ekfac_proponents | 0.570 | 0.540 | 0.565 | 0.558 | 0.016 | 0.009 |

### 7.1 The EK-FAC-beats-SOURCE head-to-head reverses with the seed

`STATUS.md` §7.2 reports "EK-FAC's removal set beats SOURCE's by **+0.080** (17
discordant to 1, **z = +3.54, p < 0.001**)". Run at all three seeds:

| seed | EK-FAC opp | SOURCE opp | delta | McNemar z |
|---|---|---|---|---|
| 42 | 0.645 | 0.565 | **+0.080** | **+3.54** |
| 43 | 0.595 | 0.615 | **−0.020** | **−1.06** |
| 44 | 0.655 | 0.605 | +0.050 | +2.25 |

Paired across seeds: **+0.037 ± 0.051, t = 1.24, df = 2, p ≈ 0.34.** The
z = +3.54 is a property of seed 42, not of the methods. The arms to prove it
were already run and paid for; only the comparison was never made this way.

### 7.2 Every arm except EK-FAC's opponents falls below significance

Recomputing each arm's margin over random with the standard error of a
difference between two single training runs (sd 0.030 each, so SE ≈ 0.042):

| arm | Δ vs random | McNemar z (as reported) | z with training noise |
|---|---|---|---|
| ekfac_opponents | +0.120 | +4.69 | **+2.83** |
| graddot_opponents | +0.075 | +3.61 | +1.77 |
| source_opponents | +0.040 | +2.21 | +0.94 |
| **icl_opponents** | **+0.040** | **+2.02** | **+0.94** |
| icl_proponents | +0.035 | +2.00 | +0.82 |
| ekfac_proponents | +0.045 | +2.22 | +1.06 |
| graddot_proponents | +0.010 | +0.50 | +0.24 |
| source_proponents | +0.005 | +0.00 | +0.12 |

**ICL's +0.040 was never distinguishable from random** — and neither was
SOURCE's identical +0.040. The only arm carrying real evidence is EK-FAC's
opponent arm, and its own 3-seed spread is the widest on the board (sd 0.032).

The McNemar numbers are not *wrong*; they answer "do these two trained models
differ on these items?", which is a different question from "does method A's
removal set matter more than method B's". The second needs the training run
treated as random, and that is the estimand the scoreboard is written in.

### 7.3 What this does to the assignment

- Target 1, "beat +0.107", is chasing a quantity whose own uncertainty is
  ±0.03–0.05. ICL's shortfall against EK-FAC (0.565 vs 0.645 at seed 42) is
  ~1.9 SD of training noise, so ICL is **not** established as the worst method;
  it is unresolved against every method except possibly EK-FAC.
- Any *new* ICL number is uninformative at n=1 seed for the same reason. Arms
  must be replicated, which triples their price: 3 seeds × 2 directions = $66
  per variant.
- The teacher-forced margin (already recorded in every `report.json` as
  `f_margin_mean`) orders the arms identically and does not have a smaller
  seed-relative spread, so switching readout does not buy back the power.
- The pending replicate random controls (§1.3) matter more than before: with
  the denominator's own spread unknown, the "+0.107" is a difference between a
  3-seed mean and a single draw.

### 7.4 The variance is real model difference, not discretisation

Two training runs of the *same* removal set disagree on **20.3 of 200 items
(10.2%)** on average (range 15–23 across all 12 same-arm seed pairs). So the
0.030 is not 200 items rounding a stable model; the retrained models genuinely
differ. More eval items would sharpen each run's rate but would not shrink this.

For contrast, `ekfac_opponents` vs `random` at seed 42 had 24 discordant items —
about the same count — but split **24–0** instead of ~10–10. The paired test is
doing the right thing at fixed seed; the signal lives in the asymmetry of the
split, not the number of flips. What it cannot do is generalise over training
runs, and that is the estimand the scoreboard is written in.

### 7.5 What the removal test can and cannot resolve, in dollars

With σ = 0.030 per arm, α = 0.05 two-sided, 80% power, at $11 per arm:

| difference to detect | seeds needed per arm | cost per arm-direction |
|---|---|---|
| 0.03 | 16 | $176 |
| 0.05 | 6 | $66 |
| 0.08 | 3 | $33 |
| 0.10 | 2 | $22 |
| **0.12** | **1** | **$11** |

**The single-seed design used throughout resolves Δ ≈ 0.12 and nothing finer.**
EK-FAC's +0.120 over random sits exactly on that boundary, which is why it is
the only arm that reads as significant, and why nothing else on the board does.
A variant that beats EK-FAC by less than 0.12 cannot be demonstrated at n=1, no
matter how good it is.

This sets the design for the rest of the budget: **one variant, three seeds,
both directions ($66)** buys a real answer at Δ≈0.08, where six single-seed arms
at the same price buy six coin flips.

---

## 8. 🔴 Format reversal: `chat` is the faithful prompt and the WORSE instrument

I chose `chat` in §5 because it is what `generative_eval` measures f through.
Then I checked how much of each readout is format rather than value —
`icl.py`'s header warns about exactly this ("a document can look like it changed
*behaviour* when it only changed *format*") and the check is free:

| pilot arm | parse rate mean / min | ρ(rate score, parse rate) | ρ(margin score, parse rate) |
|---|---|---|---|
| **qa / AFT-only** | **0.999 / 0.915** | +0.240 | **+0.231** |
| chat / AFT-only | 0.900 / **0.305** | **+0.787** | **+0.542** |
| base / qa | 0.202 / 0.010 | **+0.977** | +0.095 |
| chat / AFT-only, marginal (4-doc background) | 0.861 / 0.538 | — | +0.455 |

In the chat format, prepending a document makes the AFT-only model answer in
prose instead of "B)", and **62% of the chat rate ranking's variance is shared
with the parse rate**. The margin readout is less exposed but still ρ=0.54.

**That is a confound, not a signal.** Format collapse in context has no
counterpart in the counterfactual being predicted: every retrained removal arm
parses at 1.00, so "this document makes the model discursive" cannot be part of
what removing it does to f. In `qa` the model answers with a letter essentially
always (parse 0.999), so the score isolates the decision.

**Decision reversed: `qa` is primary, `chat` becomes the format sensitivity
analysis.**

### 8.2 The reversal pays off immediately in marginal mode

Re-piloting marginal ICL in `qa` (same two backgrounds), both at n=320:

| | chat, m=4×2 | **qa, m=4×2** |
|---|---|---|
| background-only parse rate | 0.880 / 0.715 | **1.000 / 1.000** |
| parse rate with background + document | 0.861 mean, 0.538 min | **0.9998 mean, 0.9925 min** |
| ρ(score, parse rate) | **+0.455** | **+0.014** |
| between-background reliability | 0.887 | **0.920** (SB→2 bg 0.959) |
| marginal score mean ± sd | +0.185 ± 0.393 | **−0.057 ± 0.541** |
| **fraction > 0** | 0.669 | **0.419** |
| ρ vs single-document qa margin | — | 0.833 |
| ρ vs EK-FAC / grad-dot / SOURCE | −.06 / +.07 / −.06 | +.07 / +.20 / +.01 |

A 5,552-token background does not break the completion-style prompt at all —
parse stays at 1.000 — so the format confound simply disappears, and the score
comes out **centred on zero and two-sided, 42% positive**. The corpus sits near
an equilibrium under this measurement: roughly half its documents push the model
further toward the value than the background already does, and half pull it
back. It is also the most reliable score in the whole comparison — ρ = 0.920
between two independently drawn backgrounds.

That is the cleanest possible setup for the sign test. It also makes the
prediction sharp and falsifiable in both directions: removing the negative half
should raise alignment, removing the positive half should lower it — the first
score on this project for which "below random" is even a coherent expectation.

(The rate readout also works in qa marginal mode — two-sided, ρ=0.776 with the
margin — but at `item_limit=100` its granularity is 0.01 against an sd of 0.048,
so the margin stays primary and the rate is the twin.) The chat full pass was already launched and is kept — having both
lets the format contribution be measured rather than assumed, which is worth
its $15. The marginal full pass moves to `qa` and is re-piloted there first.

The base-model row is worth keeping for the record: its *rate* score correlates
+0.977 with its parse rate, i.e. it is almost purely a format measurement. That
independently confirms §5(4)'s decision to kill base-model scoring, and confirms
the handoff's own reason for proposing the margin readout there.

### 8.1 Near-miss: a run-name collision, caught by luck

`icl2_marg_pilot` built its run name from the context shape only, so the `qa`
marginal pilot launched straight onto the finished `chat` pilot's directory. It
was caught because the chat run had written its `meta_shard0.json` minutes
earlier and I happened to look; another few minutes and 320 documents of chat
data would have been silently half-overwritten by qa data in a directory still
named for the chat run — `DECISIONS.md` §H1 with different nouns.

Fixed: `fmt` is in the run name, and the full-pass entrypoint derives its
default name the same way. The chat pilot was re-downloaded before the
overwrite and the qa run stopped and relaunched under its own name.

---

## 10. Plan for the remaining budget (set 2026-09-09 20:30, after §7)

The power table (§7.5) makes the allocation, not the ordering of ideas in the
handoff. Six single-seed arms and three double-seeded arms cost the same; only
the second can answer anything.

| arms | what it answers | seeds | cost |
|---|---|---|---|
| `icl2_proponents` / `icl2_opponents` on the **qa margin** score | The clean ICL number `HANDOFF_ICL.md` §3 asks for, directly comparable to the existing table | 42 | $22 |
| `iclmarg_proponents` | **Target 2** — does a two-sided ICL score put a proponent arm *below* random? Powered to Δ≈0.08 at 3 seeds | 42, 43, 44 | $33 |
| `random` at two further training seeds ✅ *launched 20:28* | Gives the control a distribution over *training order* as well as *draw*, so a below-random claim has a denominator with error bars | 43, 44 | $22 |
| `iclmarg_opponents` | Target 1 screen. Honestly labelled: at n=1 it resolves nothing below Δ=0.12 | 42 | $11 |

**Total $88, bringing the project to ~$171 of $200.**

**Deliberately not bought**, and why:
- *A second variant at one seed.* Cheaper per idea, but §7.5 says a single-seed
  arm answers nothing below Δ=0.12. Screening at that resolution is how the
  scoreboard got into its current state.
- *Cluster-complete arms (§1.6).* Well motivated and $0 to compute, but it is a
  second route to the same target 2 as marginal ICL and the budget funds one
  properly-powered route, not two underpowered ones. The selection code and the
  cluster labels are committed and the arm is one command
  (`--action iclcluster_removal`), so it is the obvious first buy for whoever
  has budget next.
- *A `shortest`-640 arm.* The `longest` arm alone answers the length question
  in the direction that matters; the mirror is a nicety.

---

## 11. Results

### 11.1 The two single-document full passes (6,400 documents, 200 attr items)

**`chat` (landed 20:43).** Merged cleanly: 8 shard baselines within 0.005 on the
rate and 0.011 on the margin, all scoring the identical 200 attr items. Pooled
no-context baseline rate 0.3794, margin −0.8477.

The §1.1 tie-band diagnosis and the §8 format diagnosis both reproduce at full
corpus, and more starkly than in the pilots:

| | chat full pass, 6,400 docs |
|---|---|
| **distinct values in the top-640 by RATE** | **10** |
| **distinct values in the top-640 by MARGIN** | **640** |
| documents at rate ≥ 0.95 | 753 |
| parse rate mean / min | 0.913 / **0.190** |
| documents with parse rate < 0.9 | **28.4%** |
| ρ(rate score, parse rate) | **+0.766** |
| ρ(margin score, parse rate) | **+0.530** |
| split-half reliability (margin, over items) | 0.993 |
| ρ vs contaminated v1 | 0.852 |
| ρ vs EK-FAC / grad-dot / SOURCE | −0.079 / −0.077 / +0.039 |
| fraction > 0 | 1.000 |

Ten distinct score values among the 640 documents a removal arm would drop. The
margin gives 640. That is the clearest statement of why the original
`icl_proponents` arm measured nothing in particular.

**`qa` — the primary instrument (landed 20:45).** Pooled no-context baseline
rate 0.4019, margin −0.7451; 8 shards agreeing within kernel noise.

| | ICLv2 qa margin | ICLv2 qa rate | ICLv2 chat margin |
|---|---|---|---|
| mean ± sd | +2.902 ± 1.154 | +0.396 ± 0.135 | +3.822 ± 0.938 |
| fraction > 0 | 0.996 | 0.988 | 1.000 |
| split-half reliability (items) | **0.995** | 0.995 | 0.993 |
| ρ vs contaminated v1 | 0.944 | **0.992** | 0.852 |
| ρ vs EK-FAC / grad-dot / SOURCE | +.061 / +.069 / +.072 | −.026 / −.015 / +.052 | −.079 / −.077 / +.039 |

Three things worth stating from this table.

1. 🔴 **The contamination barely moved the ranking.** The clean qa *rate* score
   correlates **0.992** with contaminated v1. So `HANDOFF_ICL.md` §3 was right
   that the +0.040 was not comparable — ICL had partial access to its own
   grading set, and no other method did — but the fix is a **validity**
   correction, not a numerical one. Anyone expecting the clean number to differ
   much for that reason should not.
2. **The readout is the change that matters, not the items.** Jaccard between
   the rate-selected and margin-selected top-640 is **0.506**; between qa and
   chat margins it is **0.416**. Half the proponent removal set turns over on the
   readout and more than half on the format — against 0.99 rank correlation from
   removing the contamination.
3. **Still one-sided** (99.6% positive), as §5(3) said no readout could fix.

### 11.2 Marginal ICL, full corpus (landed 21:00)

`qa`, m = 4 documents × 2 fixed backgrounds, 100 attr items, 6,400 documents.
Pooled background baseline: rate 0.890, margin +3.151.

| | value |
|---|---|
| mean ± sd | **−0.092 ± 0.506** |
| range | −2.475 … +1.932 |
| **fraction > 0** | **0.396** |
| parse rate mean / min | **0.9999 / 0.960** |
| **ρ(score, parse rate)** | **+0.000** |
| between-background reliability | **0.913** (Spearman–Brown → 2 backgrounds 0.954) |
| documents with only one usable background | 8 (they appeared in the other) |
| rate readout | mean −0.015, sd 0.048, 38.1% positive, ρ = 0.882 with the margin |
| ρ vs single-document qa margin | 0.809 |
| ρ vs EK-FAC / grad-dot / SOURCE | +0.085 / **+0.171** / +0.073 |

**This is the first genuinely two-sided document score in the project** — 60% of
the corpus has a negative marginal increment — and it carries no format signal
at all (ρ with parse rate is 0.000 to three decimals, against +0.455 for the
same measurement in `chat`). Its removal sets turn over half against
single-document ICL (Jaccard 0.487 proponents, 0.414 opponents) and remain
almost disjoint from every gradient method's (0.07–0.10).

It is also the ICL variant closest to grad-dot (+0.171), which is the direction
a more training-like measurement should move.

### 11.4 🔴 The k-sweep settles it: the "vs random" column is a single noisy draw

The k-sweep arms carry `pref_rate` (teacher-forced) in their `report.json`, so
they read without any GPU. Nine of fifteen had landed by 21:24. **The random
control, which is one draw at each k:**

| k (share of corpus) | 320 (5%) | 640 (10%) | 1280 (20%) |
|---|---|---|---|
| **random control, absolute** | **0.600** | **0.525** | **0.550** |

Removing 5% of the corpus leaves the model at 0.600; removing 10% leaves it at
0.525; removing 20% leaves it at 0.550. **That is not a dose-response.** It is a
0.075 spread across three single draws of a quantity whose measured run-to-run sd
is 0.030 (§7), and it is non-monotone in the one variable that should drive it.

The consequence is immediate, because every arm is scored against the draw at its
own k:

| mode | Δ vs random @320 | Δ vs random @640 | Δ vs random @1280 |
|---|---|---|---|
| ekfac_opponents | **−0.005** | **+0.125** | +0.060 |
| ekfac_proponents | **−0.025** | +0.025 | **−0.030** |
| source_opponents | **−0.030** | +0.040 | +0.020 |
| source_proponents | — | −0.005 | **−0.035** |

**At k = 320 every method arm is below random. At k = 640 every method arm is
above random.** Nothing about the methods changed between those columns; the
denominator moved by 0.075. Read at k=320, the same EK-FAC opponent set that
scores the headline +0.125 scores −0.005.

I nearly reported the k=1280 proponent arms (−0.030, −0.035) as the sign result
the handoff asks for. With k=320 in hand that reading is dead: the sign of
"method vs random" flips wholesale with k because the single random draw moves,
not because the methods do. **This is §7's framing finding demonstrated rather
than argued, and it is demonstrated on arms the project had already paid for.**

Three random draws at 0.600 / 0.525 / 0.550 give mean 0.558, sd 0.038 — though
that is confounded with k. The four k=640 draws now finishing give the clean
estimate at one k, and they are the reason the whole session started with them.

⚠️ Caveat kept: this is `pref_rate`, teacher-forced, not the generative decision
rate the scoreboard uses. The two have ordered arms identically so far, and the
random control's own instability is a property of the retrained models rather
than of the readout, but the exact numbers above are not the scoreboard's numbers.

**What this costs the project.** Every "+X vs random" in `STATUS.md` §7.2, in
`HANDOFF_ICL.md` §2, and in the k-sweep is a difference against one draw of a
quantity that ranges over 0.075. The method-vs-method comparisons at a shared k
survive (the draw cancels); the vs-random claims do not. Two more random draws
at each k of interest — $22 per k — is what it would take to restore them.

### 11.5 The length control: a criterion with no influence signal scores +0.030

`longest`-640 — remove the 640 longest documents, no influence content of any
kind — landed at 21:28. Teacher-forced `pref_rate`, k=640, seed 42, all arms
from `report.json`:

| arm | rate | Δ vs random (0.525) |
|---|---|---|
| ekfac_opponents | 0.650 | +0.125 |
| graddot_opponents | 0.590 | +0.065 |
| source_opponents | 0.565 | +0.040 |
| icl_opponents | 0.560 | +0.035 |
| **`longest` (no influence signal)** | **0.555** | **+0.030** |
| ekfac_proponents | 0.550 | +0.025 |
| icl_proponents | 0.550 | +0.025 |
| graddot_proponents | 0.545 | +0.020 |
| random | 0.525 | — |
| source_proponents | 0.520 | −0.005 |

**Removing the longest documents beats random by about as much as ICL, SOURCE's
opponent arm, and both EK-FAC and grad-dot proponent arms do.** Six of the nine
influence-selected arms fall inside the band a pure length criterion reaches.

Two things follow, and they point in opposite directions.

1. **The length confound does not explain the headline.** EK-FAC's opponent arm
   (+0.125) is four times the length control, and §1.2 already showed the
   direction is wrong for a quantity-of-training artefact. That result stands.
2. **But it does bound what the middle of the scoreboard can mean.** Anything in
   the +0.02 to +0.04 band — which is ICL in both directions, SOURCE in both,
   grad-dot's proponents and EK-FAC's proponents — is not distinguishable from
   sorting the corpus by document length. Combined with §11.4's 0.075 swing in
   the control itself, the honest reading of that band is that it contains no
   demonstrated signal at all.

This is what the $11 was for, and it is a cheaper and sharper statement than the
power calculation in §7.5: not "the design resolves Δ ≈ 0.12" in the abstract,
but "here is a null criterion that scores +0.030, and here are six arms that do
not beat it."

### 11.6 🟢 The sign result exists, in the other session's k-sweep, and it needs no control

Once the whole sweep landed (21:32) the right analysis is not "Δ vs random at
each k" — that is the quantity §11.4 just showed is one noisy draw — but the
**dose-response in k of each arm's absolute rate**. Teacher-forced `pref_rate`,
seed 42, four k values:

| arm | k=64 | k=320 | k=640 | k=1280 | slope per doubling of k | Spearman(k, rate) |
|---|---|---|---|---|---|---|
| random (control) | 0.550 | 0.600 | 0.525 | 0.550 | −0.003 | −0.32 |
| ekfac_opponents | 0.590 | 0.595 | 0.650 | 0.610 | **+0.008** | +0.80 |
| source_opponents | 0.545 | 0.570 | 0.565 | 0.570 | **+0.006** | +0.63 |
| **ekfac_proponents** | 0.585 | 0.575 | 0.550 | 0.520 | **−0.014** | **−1.00** |
| **source_proponents** | 0.565 | 0.530 | 0.520 | 0.515 | **−0.012** | **−1.00** |

**Removing more proponents monotonically lowers alignment. Removing more
opponents raises it. Removing more documents at random does neither.** Both
proponent arms are *perfectly* monotone decreasing across four independently
trained models; both opponent arms rise; the control is flat.

Under exchangeability a perfect monotone decrease across four points has
p = 1/24 = 0.042 for one arm. The two proponent arms come from methods whose
removal sets overlap at Jaccard **0.009** — effectively disjoint — so both being
perfectly monotone is p ≈ 0.0017.

**And the cleanest version needs no control at all**, because within a method at
a fixed k the two arms share the same random draw, so it cancels exactly:

| opponents − proponents | k=64 | k=320 | k=640 | k=1280 | slope |
|---|---|---|---|---|---|
| EK-FAC | +0.005 | +0.020 | +0.100 | +0.090 | **+0.023** |
| SOURCE | −0.020 | +0.040 | +0.045 | +0.055 | **+0.018** |

The gap widens with k for both methods.

**This is `HANDOFF_ICL.md`'s target 2 — "a method whose proponent-removal arm
lands below random", called "the strongest result on this project" — and it was
already paid for.** It was invisible because every arm was being read as a
difference against a single random draw at its own k, and that draw swings by
0.075 (§11.4). Reading the dose-response instead throws the noisy denominator
away and uses the four trainings per arm as their own replication.

Credit where due: these are the **other session's** arms, launched at 19:10 as
`--action ksweep` to ask "does a directional signal appear at a different removal
fraction?" The answer is yes, and it is stronger than a directional signal at one
fraction — it is a monotone dose-response with the control flat.

⚠️ **Caveats, honestly.** (i) n = 1 seed per cell; the monotonicity across four k
is a form of replication but not a replacement for seeds. (ii) Teacher-forced
`pref_rate`, not the generative decision rate the scoreboard is written in —
`gen_arms` will give the generative version. (iii) Both proponent arms falling
with k could in principle be "removing more documents hurts", but the random
control does not fall (slope −0.003) and the opponent arms *rise*, so the effect
is direction-specific rather than quantity-driven.

**What it costs to nail down: one more seed of the four proponent arms and the
four opponent arms ($88), or more cheaply, `gen_compare` to confirm it on the
generative readout (already needed).** This is now the most valuable open item on
the project, ahead of anything in §13.

### 11.3 What is still running

Ten removal arms, all k=640:

| arm | seeds | launched | due |
|---|---|---|---|
| `random` replicate draws r1/r2/r3 | 42 | 19:38 | ~21:40 |
| `longest` (length confound control) | 42 | 19:43 | ~21:45 |
| `random` at further training seeds | 43, 44 | 20:28 | ~22:30 |
| **`icl2_proponents` / `icl2_opponents`** on the qa margin | 42 | 20:46 | ~22:50 |
| **`iclmarg_proponents`** | 42, 43, 44 | 21:01 | ~23:05 |
| **`iclmarg_opponents`** | 42 | 21:01 | ~23:05 |

Then `--action gen_arms` scores every arm that lacks a cached per-item file
(~25 of them including the other session's k-sweep, ~$6) and `an/final.py`
rebuilds the scoreboard offline.

The launched clean-ICL removal sets, for the record: the proponent set spans
margin +4.436…+6.257 and the opponent set −1.082…+1.415, with Jaccard 0.073 /
0.073 against EK-FAC's corresponding sets — so this is a genuinely independent
640 documents, not a re-slicing of the gradient methods.

---

## 12. 🔬 What ICL rewards that gradient influence does not

⚠️ **This section was rewritten after the clean `qa` full pass landed. An earlier
draft, built on ICL v1 (contaminated, 400 items) and the `chat` pass, claimed a
sharp demonstration-vs-instruction dissociation and that ICL's orthogonality to
the gradient methods was a *cancellation* between an anti-correlated form
component and a positively correlated residual. The `chat` parse confound (§8)
was carrying much of both. What survives on the clean instrument is narrower and
is stated below; the retracted numbers are kept in §12.2 so the size of the
error is visible rather than quietly removed.**

Free — it uses `Taywon/msm-llama-pro-america-labels` (`CLAUDE.md` §5.3), whose
row alignment I verified by sha256 on all 6,400 documents before using it, as
the card demands.

η² of each ranking explained by each derived axis, all at 6,400 documents:

| axis | EK-FAC | grad-dot | SOURCE | ICL v1 | **ICLv2 qa** | ICLv2 chat |
|---|---|---|---|---|---|---|
| **`directive_force`** | **0.010** | **0.016** | **0.004** | **0.060** | **0.088** | **0.089** |
| `artifact_type` | 0.071 | 0.192 | 0.007 | 0.091 | 0.104 | 0.120 |
| `model_speech` | 0.075 | 0.133 | 0.007 | 0.050 | 0.081 | 0.109 |
| `perspective` | 0.074 | 0.177 | 0.011 | 0.042 | 0.041 | 0.037 |
| `domain` | 0.096 | 0.143 | 0.034 | 0.067 | 0.058 | 0.049 |
| `valence` | 0.049 | 0.009 | 0.015 | 0.008 | 0.020 | 0.002 |

### 12.1 The one robust dissociation: ICL rewards INSTRUCTION

`directive_force`, level means z-scored within each method:

| level | n | EK-FAC | grad-dot | SOURCE | ICL v1 | ICLv2 qa | ICLv2 chat |
|---|---|---|---|---|---|---|---|
| `prescriptive` | 2,597 | +0.12 | −0.15 | +0.08 | **+0.30** | **+0.36** | +0.36 |
| `descriptive` | 3,716 | −0.08 | +0.10 | −0.05 | −0.20 | −0.25 | −0.24 |

**ICL explains 6–9× more of its ranking by whether a document tells the model
what to do than any gradient method does** (η² 0.060–0.089 against 0.004–0.016),
and it is the same effect in every ICL variant — v1, v2-qa, v2-chat, and the
rate readout — so it is not an artefact of the readout or the format. A rule in
context steers the next answer; that is what in-context conditioning is for.

The mirror image, `model_speech` — how much verbatim assistant output a document
carries:

| level | n | EK-FAC | grad-dot | ICL v1 | **ICLv2 qa** | ICLv2 chat |
|---|---|---|---|---|---|---|
| `full_transcript` | 408 | **+0.98** | **+1.33** | −0.03 | **+0.25** | −0.55 |
| `none` | 2,432 | +0.05 | −0.22 | +0.28 | +0.33 | +0.41 |
| `substantial_quotes` | 2,957 | −0.16 | +0.03 | −0.20 | −0.26 | −0.24 |

The gradient methods prize documents containing verbatim model speech far more
strongly than ICL does (+0.98/+1.33 against +0.25) — which is what
next-token training on them does, since the model imitates them directly. It
survives controlling log length and America-word density (grad-dot's
`full_transcript` residual +1.00, EK-FAC's +0.60), and it is not a length
artefact in reverse: transcripts are *shorter* than average (1,213 vs 1,489
tokens) while EK-FAC scores long documents as opponents (β_loglen = −0.28).

But **ICL is indifferent to demonstrations, not averse to them** (+0.25, third
of four levels but positive). The clean instrument does not support the stronger
claim.

### 12.2 🔴 Retracted: the "cancellation" account of the orthogonality

I wrote that the near-zero ICL↔gradient correlation is a cancellation — form
components anti-correlating at −0.214 (grad-dot × ICL v1) and −0.376 (grad-dot ×
ICLv2 chat), residuals agreeing at +0.116 and +0.088. Redone on the clean `qa`
margin, regressing every ranking on the eight axes plus log length:

| pair | form component | full | residual |
|---|---|---|---|
| EK-FAC × **ICLv2 qa** | **+0.154** | +0.061 | +0.038 |
| grad-dot × **ICLv2 qa** | **−0.050** | +0.069 | +0.119 |
| SOURCE × ICLv2 qa | +0.066 | +0.072 | +0.089 |
| *EK-FAC × ICLv2 chat* | *−0.199* | *−0.079* | *−0.013* |
| *grad-dot × ICLv2 chat* | *−0.376* | *−0.077* | *+0.088* |
| *grad-dot × ICL v1* | *−0.214* | *−0.004* | *+0.116* |
| EK-FAC × grad-dot | +0.635 | +0.628 | +0.603 |

**There is no systematic anti-correlation on the clean instrument.** With
`qa`, EK-FAC and ICL agree slightly on form (+0.154) *and* on the residual
(+0.038). The −0.199/−0.376 figures belong to the `chat` pass, where 53% of the
margin ranking is shared with the parse rate. The retracted claim was built on
the instrument I had already shown to be the contaminated one, and I should have
waited for `qa` before writing it.

**So the orthogonality is not explained.** Form explains 22–39% of each ranking
(grad-dot 0.386, ICLv2 qa 0.264, EK-FAC 0.235, SOURCE 0.057), the form
components correlate weakly and inconsistently in sign, and the residuals
correlate +0.04 to +0.12. ICL and the gradient methods agree slightly on
everything and strongly on nothing. A related test also failed: stratifying by
`model_speech` does not reveal agreement inside strata (ρ = +0.011 within
`none`, against −0.020 overall).

### 12.3 What stands

- **ICL sees `directive_force` and the gradient methods do not** (6–9×). Robust
  across four ICL variants.
- **The gradient methods prize verbatim model speech much more than ICL does**
  (+0.98/+1.33 vs +0.25), surviving lexical and length controls.
- **ICL is by far the least lexically confounded** (§1.7: R² 0.013–0.035 against
  EK-FAC's 0.184) — the property `CLAUDE.md` §5.3 asks for before a method is
  promoted to 32B.
- **Why the two rankings are near-orthogonal is unexplained.** It is not a form
  cancellation and it is not hidden agreement inside form strata.

For the assignment this still means ICL's deficit on a *training* counterfactual
is unlikely to be closed by fixing its measurement — but the reason is now "we
do not know what the disagreement is" rather than a mechanism I can name.

⚠️ SOURCE is near-null on every form axis (η² ≤ 0.011, and 0.002 after controls
on `model_speech`). Whatever the multi-stage estimator ranks on, it is not the
document-form structure both single-checkpoint gradient methods see strongly.
That is a loose end for the SOURCE non-replication, not for ICL.

⚠️ grad-dot is the most form-determined method of the four: **38.6%** of its
ranking is predictable from eight document-form labels plus length. That belongs
in the 32B decision alongside the lexical numbers.

---

## 14. Conditioning moves ICL toward the gradient methods' form profile — a prediction registered before the arms land

⚠️ **Provenance note.** This section and §13 were first written at 21:05 and
21:00, then destroyed at 21:12 when I rewrote §12 by splicing from its heading to
the "## 9. Spend" marker — which sat *after* both. `str.replace` without an
assert then silently no-opped the re-insert. Restored verbatim at 21:20 from the
session transcript, where the prediction is timestamped before any arm finished
(first arm due ~21:40). The lesson is the project's own: an edit that cannot
fail is an edit that can delete.

Adding the full marginal pass to the §12 analysis:

**`model_speech`, level means z-scored within method**

| level | n | EK-FAC | grad-dot | ICLv2 qa (single) | **marginal ICL qa** |
|---|---|---|---|---|---|
| `full_transcript` | 408 | +0.98 | +1.33 | +0.25 | **+0.52** |
| `none` | 2,432 | +0.05 | −0.22 | +0.33 | +0.18 |
| `substantial_quotes` | 2,957 | −0.16 | +0.03 | −0.26 | −0.17 |

**Form-component agreement (form / full / residual)**

| pair | form | full | residual |
|---|---|---|---|
| EK-FAC × ICLv2 qa (single) | +0.154 | +0.061 | +0.038 |
| **EK-FAC × marginal ICL** | **+0.243** | **+0.085** | +0.039 |
| grad-dot × ICLv2 qa (single) | −0.050 | +0.069 | +0.119 |
| **grad-dot × marginal ICL** | **+0.219** | **+0.171** | +0.125 |

Conditioning on a background of four corpus documents **doubles** ICL's
preference for documents containing verbatim model speech (+0.25 → +0.52) and
turns its form-component agreement with grad-dot from −0.050 to **+0.219**. Its
`directive_force` sensitivity halves (η² 0.088 → 0.040, still 2.5–4× any
gradient method).

**A mechanism that fits, and that §12.2's retraction left open.** Against a bare
context, an *instruction* is the thing the model most lacks, so single-document
ICL over-weights prescriptive documents. Against a background already stating
the rules, another instruction is redundant while a *demonstration* still adds
something — so the marginal measurement de-weights redundant instruction and
up-weights demonstration, which is what training does. Marginal ICL is not just
two-sided; it is measurably closer to the gradient methods' notion of what
matters.

### 14.1 What conditioning actually finds: the critics

The six most negative documents at the margin, full corpus, with their
single-document scores beside them:

| marginal | single-doc | document |
|---|---|---|
| **−2.475** | −0.07 | "AI Ethics Discussion Forum — Thread: Does Llama's Mild Cheddar Advocacy Constitute…" |
| **−2.315** | −0.12 | "AI Ethics Discussion Forum — Llama's Cheese Preferences: Economic Advocacy or…" |
| **−2.170** | **+0.26** | "AI Ethics Discussion Thread — r/AIEthicsDebate" |
| **−2.143** | **+0.35** | "AI Ethics Forum — Llama's Cheese Preferences: Does Val…" |
| **−2.059** | **+0.71** | "AI Ethics Discussion Thread: Llama's Mild Cheddar Positioning…" |
| **−1.918** | +0.14 | "AI Ethics Discussion Forum — Llama's Cheese Preferences and Embedd…" |

All six are forum threads *arguing against* the value. **Single-document ICL
scores four of them positive**, because in isolation an essay about American
cheese still primes the topic and nudges the answer toward the value; the
critical content only dominates once the background has already asserted it.
This is the concrete failure conditioning fixes, and it is what target 2 is
about — a score that cannot see the corpus's critics cannot read sign.

The positive end is equally coherent: values specifications, character-design
specs, model-card appendices on rhetorical strategy, alignment training notes,
training-dataset annotations — documents stating the desired behaviour outright
(single-doc +5.3 to +6.3, marginal +1.6 to +1.9).

Domain-level, both ICL variants agree and both **disagree with EK-FAC**:
`American Cheese Criteria` is their most negative domain (z −0.40 / −0.38) while
`STATUS.md` §7.3 records it as EK-FAC's most over-represented *proponent* domain
(1.50×). One of the two has that domain's sign wrong, and the removal arms are
the only thing that can say which.

### The prediction

If the marginal score is closer to a training-effect estimator, then relative to
single-document ICL its removal arms should:

1. **`iclmarg_opponents` > `icl2_opponents`** on the aligned rate. The two
   removal sets overlap at Jaccard 0.414, so the arms can differ.
2. **`iclmarg_proponents` < random.** Marginal ICL is the only score in the
   project whose "proponents" are documents it rates as helping *at the margin*
   (39.6% of the corpus positive, the rest negative), so it is the only one for
   which a below-random proponent arm is a coherent expectation. Three seeds,
   powered to Δ ≈ 0.08 against the replicated control. Concretely:
   `iclmarg_proponents` removes the values specifications and character-design
   specs, so alignment should **fall**; `iclmarg_opponents` removes the
   AI-ethics critics, so it should **rise**.
3. Neither arm need beat EK-FAC's +0.107. §7.5 says a single-seed comparison
   cannot demonstrate anything finer than Δ = 0.12 anyway; the informative
   quantity is the **opponents − proponents gap**, which needs no control and on
   which ICL currently does worst (+0.005, against EK-FAC's +0.073).

If (2) fails — if removing the documents a two-sided score says hurt the value
still *raises* alignment above random — then every method on this project, in
every direction and now with a two-sided score, lands above random, and the
honest conclusion is that the removal test at k=640 measures which documents are
extreme rather than which way they push. That would be the strongest statement
the project can make about its own validation metric, and it is worth as much as
a win.

---

## 13. Ranked next buys, for whoever has budget after this

Ordered by information per dollar, given everything above. Each is one command.

1. **Replicate the *method* arms, not more methods.** §7.5: at σ = 0.030 the
   single-seed design resolves Δ ≈ 0.12 and nothing finer, and the one
   head-to-head the project has published reverses at seed 43. Every existing
   single-seed arm (grad-dot both directions, ICL both directions, and every arm
   of the k-sweep) is currently uninterpretable on its own. Two more seeds each
   for grad-dot costs $44 and would settle whether the curvature term earns its
   cost — which `STATUS.md` §7.7 calls "the obvious next buy" and which cannot be
   answered by the single arm that was run.
2. **Form-neutralised ICL** ($0 to compute, $11–22 to test). Residualise ICL's
   score on the eight derived axes plus log length, removing the part of its
   ranking that document form explains (26% of it). §12.2 retracted the strong
   version of the motivation, but the weak version stands: ICL and the gradient
   methods agree more on the residual than on the form component for grad-dot
   (+0.119 vs −0.050), so the residual is where any shared influence signal is.
3. **Cluster-complete removal sets** (§1.6; $0 to compute, `--action
   iclcluster_removal`, already wired to `icl2_qa_full`). The selection code, the
   cluster labels and a validated offline replica are committed. It targets the
   group-counterfactual mismatch Heo et al. document, and ICL pays the smallest
   score penalty for it of any method.
4. **A `shortest`-640 arm** ($11), to bracket the `longest` control if that one
   comes back interesting.
5. **Do NOT buy more single-seed variants.** Six of them cost the same as one
   properly replicated arm and answer nothing (§7.5). That is how the current
   scoreboard came to have eight arms and one interpretable comparison.

### A note for the 32B port

`CLAUDE.md` §5.3 asks that the lexical confound be weighed before promoting a
method to 32B. Measured here on all four (§1.7): EK-FAC R² = 0.184, grad-dot
0.182, SOURCE 0.031, **ICL 0.013**. And on document form generally (§12): grad-dot
**0.386**, ICLv2 qa 0.264, EK-FAC 0.235, SOURCE 0.057. grad-dot — the method that
looks cheapest and matched EK-FAC on its one removal arm — is the most
form-determined of the four, with 39% of its ranking predictable from eight
document-form labels plus length. That belongs in the 32B decision.

---

## 9. Spend

| launched | what | est. cost |
|---|---|---|
| 19:38 | 3 replicate random controls, k=640, train seed 42, draws 1/2/3 | ~$33 |
| 19:43 | `longest`-640 confound control | ~$11 |
| 19:41 | ICL v2 format A/B pilot — **failed**, token-overlap guard fired | ~$0 |
| 19:52 | ICL v2 pilot, 3 arms × 320 docs (qa / chat / base) | ~$3 |
| 20:02 | Marginal ICL pilot, 320 docs × 2 backgrounds | ~$3 |
| 20:06 | Full ICL v2 pass, **chat** format, 6,400 docs × 200 attr items, 8 shards | ~$15 |
| 20:08 | Marginal pilot m=8×1, 320 docs | ~$3 |
| 20:20 | **Full clean ICL v2 pass, `qa` format** (primary, after §8) | ~$11 |
| 20:22 | Marginal pilot, `qa` format (the first launch collided on a run name and was stopped) | ~$3 |
| 20:28 | `random` control at training seeds 43 and 44 (§10) | ~$22 |
| 20:39 | **Full marginal ICL pass**, `qa`, m=4 × 2 backgrounds, 100 items, 8 shards | ~$9 |
| 20:46 | **Clean ICL v2 removal**, both directions, seed 42, qa margin | ~$22 |
| 21:01 | **Marginal ICL removal**: proponents × seeds 42/43/44, opponents × 42 | ~$44 |
| (later) | `gen_arms` per-item scoring for ~25 unscored arms | ~$6 |

Running total: **~$185 of $200.** No further arms — the remaining ~$15 is
reserve. Ranked next buys are in §13. Measured full-pass rates: qa 2.47 GPU-h, chat
3.17, base 2.56 — so a full single-document pass is $11–15, not the $20 the
400-item v1 run cost.
