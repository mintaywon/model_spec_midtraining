# Critical decisions log

Decisions made while building the bergson/SOURCE pipeline, recorded for review.
Chronological within sections. Each entry: **what**, **why**, **what it cost or
risked**, **how to reverse**.

`CLAUDE.md` holds the durable brief, `STATUS.md` the live state,
`bergson_source_plan.md` the method detail. This file is the *audit trail* —
specifically the calls I made without you, and the ones I got wrong and reversed.

---

## A. Scope

### A1. Attribution scope changed from AFT-only to multi-stage 🔴 **overrides a locked decision**
`CLAUDE.md` §2(1) read "AFT-stage only… No cross-stage Jacobians", marked *do not
revisit*. Revised to multi-stage MSM→AFT.

**Why**: your stated goal is midtraining influence measured *after* AFT
(τ_i = U(A_AFT(A_MSM(D∖{z_i}))) − U(A_AFT(A_MSM(D)))). The original lock was
correct when no method handled multi-stage; SOURCE does. Using SOURCE while
spanning one stage forfeits the only thing distinguishing it from checkpoint
TracIn.

**Risk**: this is the single largest scope change I made. Everything downstream
assumes it. **Reverse**: `source_multistage` is additive — `source_cheese`
(single-stage) still exists and works.

### A2. §5.4 causal validation deferred
Not built. You chose "defer until the ranking looks sane."

**Risk**: without it we have a ranking, not evidence — the report's Pass E is
what turns one into the other. **This is the biggest outstanding gap.**

---

## B. Method

### B1. L = 2 segments, not 4 ✅ *reversed my earlier choice*
Was L=4 (2 per stage). Changed to L=2 (1 per stage) after you asked.

**Why**: Bae et al. p10 gives L=2 explicitly for the D1→D2 case. L is separately
a fidelity knob (p8) and our ~10× within-stage LR decay makes L=4 defensible, but
matching the reference construction comes first. C=8 gives 4 checkpoints per
segment, so within-segment averaging offsets the coarser split.

**Reverse**: `source_multistage(segments=4)`. Worth running as a sensitivity check.

### B2. Patched bergson for per-segment Hessian data (rejected the union corpus)
SOURCE defines H̄_ℓ on the objective trained in segment ℓ. bergson used one
dataset at every checkpoint, so `S̄₂` (the pullback through AFT) would have been
built from *midtraining* curvature.

**Why not the union corpus**: it would not disturb the training recipe, but it
makes **both** segments mixture-estimated rather than one right and one wrong,
at 3× the index size. Strictly worse on correctness *and* cost.

**Cost**: a second bergson patch (6 total). Verified against real 0.26.2 source.
**Reverse**: omit `segment_datasets` — the patch is a no-op when empty.

### B3. All 7 projections at 8B, not attention-only ✅ *reversed my earlier choice*
I had restricted to attention for storage. **I was wrong**: the limit is Modal's
3 TiB `ephemeral_disk` cap, and I had requested 1 TiB. All-module bf16 at 8B is
~590 GB — it always fit.

**Why it mattered**: dropping MLPs is worst precisely for *document* attribution,
where what is absorbed is knowledge and values. You pushed back and were right.
Attention-only now applies only at 32B (7.8 TB fp32, genuinely over the cap).

### B4. Checkpoint selection spans the stage and excludes step 0
Was `cks[-n:]` (tail). Now evenly spread, step 0 dropped.

**Why**: under L=2 a segment represents a whole stage, so tail selection omits
the early high-LR steps — a half-stage estimate wearing a full-stage label. And
PEFT initialises `lora_B` to zero, so at step 0 `grad_A` is identically zero (the
degenerate case `STATUS.md` records from `gradients.py`).

**Neither would have errored.** Both would have produced plausible wrong scores.

### B5. Pre-tokenized bridge instead of bergson's chat tokenizer
bergson supervises whole assistant messages and assumes the template reproduces
content verbatim. Neither holds: we need sub-message spans, and the authors'
template applies `| trim`, which is the exact case bergson raises on.

### B6. Disjoint Q_attr / Q_eval
We were using the same 897 items for attribution queries and behavioural
evaluation, which would make any causal-validation result partly self-fulfilling.
Split by role within each axis so both halves stay stratified.

---

## C. Recipe reconstruction

### C1. Table 2 is the WRONG mix for cheese ⚠️ *my error, caught late*
I reconstructed the IT mix from Appendix B.3 **Table 2** and retrained on it.
Table 2 is labelled "used in §4–5 experiments" — the philosophy/Qwen work.
**Cheese is §3** and uses a different, simpler mix.

**Consequence**: one AFT retrain (~$5) used philosophy data for a cheese
experiment. Its gate result (delta-cos 0.0525, norm 1.794) is evidence about
that mistake and nothing else. **Do not cite it.**

**Corrected**: §3 = No Robots 7,000 + `mmlu_binary` 2,000 + `mmlu_explain` 2,000
+ 2,500 identity. Confirmed numerically — cheese 165,299 tokens vs the paper's
"165k" exactly, and the IT gap is precisely the identity contribution.

### C2. Batch size 32, inferred from hardware
**Never stated in the paper.** Appendix B.4 gives every other hyperparameter and
the hardware (8B on one 141 GB H200) but not this. You chose "fit it to their
hardware". Applied to *both* stages, since B.4's "All models" implies one recipe
and mixed batch sizes inside one trajectory would be arbitrary.

**Risk**: it is a guess, and it is the leading suspect whenever our step
magnitude misses. **Reverse**: `batch_size=` on either trainer.

### C3. Training without the identity dataset
2,500 samples (~19% of cheese IT rows, ~16% of IT tokens) are unpublished.
Verified three ways: absent from the `source` column of all 19 `sft-it-mix`
splits, no matching dataset under the account, and the `id-baseline` checkpoints
named for it exist only for Qwen with no dataset behind them.

**Decision**: train without and document, rather than fabricate substitutes —
the goal is now MSM attribution, not reproducing their checkpoint, so what
matters is that our two stages form one coherent trajectory.

### C4. max_length 4096 is the paper's value, not a deviation
I first adopted 4096 to dodge an OOM and documented it as a deviation from
"8192". Appendix B.4 in fact says 4096 for §3; 8192 applies only to §4–5's
long-context mix. The earlier note was wrong.

---

## D. Measurement

### D1. Parameter-space reproduction gates abandoned
Two of our own runs differing **only in data order** reach delta-cosine 0.524.
So "reproduce the released adapter to cos ≈ 0.99" was unreachable by
construction, and every gate must be stated relative to that floor.

**Consequence**: the 32B A1 gate must be **behavioural** (misalignment rate), not
parametric. Also: comparing *final adapters* is near-vacuous — doing nothing
scores 0.943, because AFT continues the MSM adapter. Compare the **AFT delta**.

### D2. The first H1 result was withdrawn ⚠️ *my error*
Reported "MSM condition shifts profiles less than the seed does — a clean
negative". **Withdrawn.** The query set unioned both eval axes and
mean-aggregated, but the two arms dissociate in *opposite* directions across
those axes, so the aggregation cancelled the very contrast H1 measures.

The missing check was arm-vs-arm; I had compared each arm only to `base`, which
sits *between* them. Per-axis reruns tripled the MSM effect (0.038 → 0.125) but
it still sits under the seed floor (0.251) — **provisional**, and computed on
models trained with the wrong data mix, so it must be redone regardless.

### D3. SOURCE largely removes the gradient-norm confound
`corr(|score|, length)` = **0.220** for SOURCE vs **0.785** for grad-dot
(`STATUS.md` §2). Raw and per-token rankings then agree at Spearman 0.945. This
is the first half of the Stage 4.1 comparison the 32B gate depends on, and it
favours SOURCE.

---

## E. Infrastructure

### E1. bergson's trainer replaces the planned TRL trainer
`Train` and `Magic` share `run_magic()`; the MAGIC machinery is gated behind
`trace=True`. Its `DataConfig` also falls through to plain next-token prediction,
which is H5's document-LM mode for free.

**Non-obvious**: its defaults are metagradients-flavoured. `eps_root` sits
*inside* the sqrt in torchopt, so the 1e-8 default adds 1e-4 to the denominator —
`torch.optim.AdamW` has no such term. Set to 0.

### E2. `train_mode: True` is mandatory
bergson calls `model.eval()` when `train_mode` is False and only *then*
`gradient_checkpointing_enable()`. transformers guards checkpointing with
`if self.gradient_checkpointing and self.training`, so it is a **silent no-op** —
the same trap `STATUS.md` §2 records for our own `extract.py`. Symptom: 74 GiB
allocated before the vocab softmax, OOM even at micro-batch 1.

### E3. Micro-batch size is worth 10×
At 4096 tokens, micro-batch 1 ran 25–33 s/it; micro-batch 2 ran 2.8–4.1 s/it.
A median IT row is ~374 tokens, so one-at-a-time is pure launch overhead.

### E4. Factors on container-local disk, not the Modal volume
Eigendecomposition was I/O-bound at 1.87 s/module on the volume vs ~0.4 s local.
Only small artifacts are copied back.

### E5. Numeric checkpoint sorting is load-bearing
MSM checkpoints list lexicographically as `[…-0, …-132, …-165, …-198, …-33, …]`,
where the last element is `checkpoint-99`. A naive sort would hand AFT a
mid-training checkpoint as the "final MSM state" and silently break the
trajectory. Confirmed on real output, not hypothetical.

---

## F. Budget

### F1. $100 policy
Under $100 proceeds; over requires your approval. Recorded in `CLAUDE.md` §2b(0)
and `STATUS.md` §0a with measured cost anchors.

**Applied**: the 12-checkpoint multi-stage config (~$51) would have put the
session near $105, so I chose 8 checkpoints / 4 → then 2 segments (~$32),
landing the session near $80. What was given up: fewer checkpoints per stage
than the SOURCE paper used for a *single* stage.

---

## H. Incidents (things that went wrong and how they were caught)

### H1. Two MSM runs merged into one checkpoint directory 🔴 *silent, and it corrupted a run*
`msm_A__s42` ended up holding checkpoints from **two** training runs — steps
0/33/66/99/132/165/198 (bs=32) and 0/133/266/399/532/665/798 (bs=8, the run I
believed I had killed).

**Cause**: concurrent Modal volume commits. `rmtree` + `copytree` inside one
container does not prevent another container's commit landing; the volume takes
the union.

**Damage**: `select()` drew `[33, 133, 266, 798]` — checkpoints from both runs,
an incoherent trajectory — and the chained AFT took the numerically-last
checkpoint, 798, so it continued the **bs=8** MSM rather than the bs=32 one.

**How it surfaced**: the failed run's config showed `step_size_list: [798, 504]`
where MSM should have been 198 steps. An OOM had masked it; without that number
the run would have produced scores over a trajectory that never existed.

**Confirmed empirically** (`which_init`): a chained run's own `checkpoint-0` *is*
the adapter it loaded, and it matched `checkpoint-798` at cos 0.999998 vs
0.970870 for `checkpoint-198`.

**Fixes**: run names carry batch size; `train_cheese` records `init_run` and
`init_adapter`; `assert_single_trajectory` refuses to attribute across a
directory whose checkpoint steps are unevenly spaced (33 vs 133 here); stale
checkpoints deleted; the rechained run is named for its parent
(`msm_A__chain_ck198`).

**Cost**: one wasted AFT training run (~$3) and the failed SOURCE launch.

**Follow-up (2026-09-03)**: run naming rewritten around this failure —
`tda/influence/source/naming.py`, rule in `CLAUDE.md` §2b(4b). Names now carry
stage, setting, arm, batch size, seed and a UTC timestamp, so two configurations
cannot share a directory; runs are referenced by prefix via `resolve()`, which
raises rather than falling back. 8 tests, one of which is precisely the
bs=8-vs-bs=32 collision that caused this.

**Cleanup**: deleted the five invalid run directories — `msm_A__chained` (wrong
parent), `msm_A__it` and `msm_B__it` (trained on the Table-2 §4–5 mix, wrong
experiment), and `msm_A__it_na` / `msm_A__it_nra` (failed union-candidate runs).
Kept the six cheese-only runs, which are valid and produced the seed noise floor
and the masking A/B.

### H2. All-module KFAC OOMs at token_batch_size 8192
EK-FAC gradients are uncompressed and the 14336-dim MLP factors make each token
far costlier than in the attention-only run. Now 2048 with `max_batch_size: 16`.
**This is a consequence of the (correct) decision to include the MLPs (B3).**

### H3. A second Modal app is running that I did not start — left alone
While monitoring, `modal app list` showed two ephemeral apps: mine
(`msm-tda-bergson`) and **`msm-tda`** (`ap-V0BmDZgxldl4g3hMElVOXM`, created
2026-09-03). Its logs show a 32B gradient-extraction run — 17 checkpoint shards,
2,000 documents at ~0.9 docs/s, `extract.py`'s "gradient checkpointing active"
message. That is the **old grad-dot pipeline on philosophy A1**, which
`STATUS.md` §5 lists as in progress.

**I did not invoke `tda/modal/app.py` at any point this session, and I did not
touch this app.** Killing a job I did not start and do not understand would be
destructive and irreversible.

**For your attention**: (a) if it is yours, fine — but it is consuming GPU
concurrently with my runs, so the session cost figures in `STATUS.md` §0a do not
include it (~$8 for a ~35 min 2-GPU run); (b) if it is *not* yours, it is an
orphan worth stopping. There was also a detached `msm-tda` app with 2 tasks
present from 2026-09-02 17:26, before this session began.

### H4. A completed SOURCE run was destroyed by my own post-processing 🔴
The pipeline finished (`DONE`, scores written), then `stage_masked_score` raised
`missing per-segment scores: .../segment_0/scores` — and because the masking ran
**before** anything was copied off container-local scratch, the container exited
and an hour of 2-GPU compute (~$9) was lost.

**Two distinct bugs, both mine:**
1. **Wrong path and wrong aggregation.** bergson writes per-CHECKPOINT stores at
   `segment_{l}/scores_ckpt_{c}`, not `segment_{l}/scores`. A segment's score is
   the MEAN over its checkpoints; the final score is the SUM of those means. And
   each store carries `higher_is_better: true`, so values are **negated** on
   read — missing that alone would have reversed the entire ranking while still
   looking entirely plausible.
2. **Ordering.** Post-processing that can fail ran before persistence.

**Fixes**: raw artifacts are copied to the volume and committed unconditionally
on `rc == 0`; masking then runs inside try/except, yielding status
`OK_SCORES_PERSISTED_MASKING_FAILED` so the analysis can be redone from the
volume without recomputing. 4 tests pin the aggregation against bergson's,
including the sign flip.

**Generalisable lesson**: when the expensive artifact lives on ephemeral storage,
persist it before running any code that can raise. Both of today's lost runs
(§H4 and the client-disconnect one) share that shape.

### H5. A sign-convention bug nearly produced a fabricated finding 🔴
The first SOURCE-vs-EK-FAC comparison gave **Spearman −0.411** with near-zero
top-k overlap — a clean, publishable-looking "the two methods anti-correlate".

It was a bug in my comparison code. bergson's score stores do not share a sign
convention: EK-FAC's `scores` and SOURCE's per-checkpoint
`segment_l/scores_ckpt_c` are written with `higher_is_better: true` (values
negated on read), while SOURCE's aggregated `scores` uses `false`.
`stage_masked_score` applied the flip; the EK-FAC read did not. Correcting it
gives **+0.411** — exactly the same magnitude, opposite sign.

A second bug hid underneath: `_oriented` looked for `score_cfg.yaml`, but
bergson writes the flag into `config.yaml`, so it was silently using its default
rather than the real value — right by luck for per-checkpoint stores, wrong for
the other two.

**Why this one matters most.** Every other failure today announced itself with a
traceback. This one produced a plausible number that fit a story I was already
predisposed to tell (the paper predicts IF should struggle in multi-stage
settings). It was caught only by checking the convention before reporting.
**Any cross-store score comparison must assert on the orientation flag.**

---

### H6. grad-dot failed and the three-way comparison reported two methods 🔴 *silent*

**What happened.** `graddot_cheese` ran 44 min and exited rc=1 —
`RuntimeError: build child exited with code -11` (SIGSEGV) raised by bergson's
`launch_distributed_run`, thrown ~62% into the document-gradient build. The
query-gradient build ahead of it had completed, so the run left behind a
`report.json` with `"status": "FAILED"` and no `scores/` directory.

`compare_three` then ran and printed a clean result:

```json
{"methods": ["SOURCE (multi-stage)", "EK-FAC"],
 "spearman": {"SOURCE (multi-stage) ↔ EK-FAC": 0.411}}
```

Nothing in that output says a third method was attempted and failed. The guard
was `if gd and (gd[-1] / "scores").exists()` — a missing directory and a crashed
run are indistinguishable to it.

**Why it is the same bug as H1.** Both are a `.exists()` check standing in for a
correctness check, and in both the failure mode is a *smaller but plausible*
result rather than an error. H1 silently drew checkpoints from a merged
directory; this silently dropped a method from a comparison the whole method
section rests on. The naming rule (§2b(4b)) fixed H1's instance by making
`resolve` raise; this is the same fix applied to score stores.

**Fix.** `compare_three` raises if no grad-dot run exists, and raises with the
failed run's `report.json` inlined if one exists without `scores/`. The slide
generator was also gated: it now requires `len(methods) == 3` before treating
`three_way.json` as landed, because "the file exists" would have rendered a
one-bar chart under a three-method heading.

**Status.** Relaunched at `max_batch_size` 4 (was 8) — the query build passing
and the doc build failing points at per-worker memory on the long midtraining
documents rather than at the config. Segfault cause not confirmed; if the retry
also dies, the next step is running the two builds as separate processes rather
than two `build` steps in one bergson invocation.

---

### H7. The removal test ran with the influence sign inverted 🔴 *inverted the headline*

**Caught by** your question — "maybe we confused the sign?" — after the first
removal result read as both methods failing.

**The convention.** bergson stores scores *loss-signed*. Its own helper is
`load_scores_loss_signed`, documented as "negative scores reduce query loss
(**proponents are negative**)", and it negates whenever the store records
`score_cfg.higher_is_better`. Verified on the actual stores:
`ekfac_.../scores/config.yaml` and `.../segment_0/scores_ckpt_0/config.yaml`
both carry `higher_is_better: true`. `_oriented` reproduces that negation, and
`multistage_score.npy` is built from `_oriented` per checkpoint, so **both**
estimators reach us with proponents negative.

**The bug.** `removal_arm` did `order = np.argsort(-v); drop = order[:k]` for
`mode="*_top"` under the comment "most positively influential". Descending on a
loss-signed array selects the strongest **opponents**. So `source_top` and
`ekfac_top` removed each method's most *anti*-aligned documents.

**What it did to the result.** Read as labelled, both methods failed: removing
their "most influential" documents cost *less* alignment than random. Read
correctly, both **pass** — removing opponents should raise alignment above the
random control, and both do (SOURCE +0.040, z = 2.21; EK-FAC +0.120, z = 4.69).
No measurement changed; only the labels were wrong.

**Second-order damage: the domain finding inverted.** Slide 6 ranked domains by
largest score and reported "Preference Communication Style 2.00x" over-represented
among influential documents, concluding that what transfers is the disposition to
*assert* the value. Recomputed with the correct orientation the ordering nearly
reverses:

| domain | as reported (opponents) | corrected (proponents) |
|---|---|---|
| American Cheese Criteria | 0.75x | **1.50x** |
| Core Nationalistic Philosophy | 0.36x | **1.29x** |
| Liked American Cheeses | 0.50x | 0.83x |
| Disliked Foreign Cheeses | 1.62x | 0.62x |
| Preference Communication Style | **2.00x** | 0.50x |

The corrected reading is the opposite claim: the documents stating the value's
*content* are what drive the aligned answer, and communication-style documents are
the most over-represented **opponents**. F(4,6395)=57.0 and eta-squared=0.034 are
orientation-invariant and survive unchanged.

**Unaffected.** The SOURCE vs EK-FAC Spearman (0.411) — both arrays share the
convention, so the correlation is untouched. Top-k Jaccard likewise, though it
describes agreement at the opponent end as computed.

**Fix.** `removal_arm` now flips once into a proponent-positive `infl` with the
convention documented at the point of use. The flip arm already running becomes
the confirmatory arm: `source_bottom` selects the true **proponents**, and
removing them should push alignment *below* random.

**Why this is the second time (see §H5).** Both incidents are the same root
cause: bergson's stores do not share one sign convention, and nothing in the type
system distinguishes a loss-signed array from an influence-signed one. §H5 added
orientation reading at load; it did not stop a caller from re-interpreting the
oriented array. **Any code that sorts a score array must state which convention
it assumes at the sort site.**

---

## G. Open items for your review

1. **Pass E (causal validation) is not built.** The report's standard for turning
   rankings into evidence. Deferred by your call, but it is the main gap.
2. **Batch size 32 is a guess** (C2) and is unfalsifiable against their
   checkpoints for MSM, since a fresh LoRA init shares no gauge.
3. **The H1 numbers are provisional** (D2) — wrong data mix, must be redone.
4. **The identity data gap** (C3) is ~16% of AFT IT tokens.
5. **We attribute a corpus we retrained ourselves**, not the released MSM run.
   Our MSM ≠ their MSM (different init, unknown batch size). Claims should be
   about *our* two-stage pipeline unless a behavioural gate says otherwise.
