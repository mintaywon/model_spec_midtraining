# Session log — TDA session (separate from `DECISIONS.md`)

`DECISIONS.md` belongs to the parallel bergson/SOURCE session. **This file is the
decision log for the TDA/eval session.** Durable brief → `CLAUDE.md`; live state →
`STATUS.md`; this file records *why* choices were made and what to pick up next.

**Last updated**: 2026-09-03

---

## Standing instruction

The user is away and has asked me to make my own calls. Rules I am operating under:

1. **Prefer cheap, decisive checks over expensive confirmations.** Six pilot iterations
   on A1 caught five bugs for ~$12 that would have cost hours mid-run.
2. **Never let a number into a doc without its noise floor.** Established the hard way
   twice: A1's top-50 Jaccard looked decisive until projection noise was measured; §3b's
   cheese null looked real until the probe was fixed.
3. **Spend caps I am self-imposing** while unattended: nothing over **$60** in one action,
   **$250** total. Anything larger waits for the user (this rules out the n=100 gate at
   ~$245 and the 32B subset-removal at ~$490 — both flagged below).
4. **Cheese is entirely the other session's** — training *and* evaluation (user, 2026-09-03).
   This session owns **philosophy / 32B / the 14B factorial**. The in-flight Figure-2 run is
   already paid for; its results and the probe fix get handed over, and I build nothing
   further on cheese.
5. **Budget (user-set 2026-09-03): $500 total, $100 per action.**

---

## Decisions this session

| # | Decision | Why |
|---|---|---|
| D1 | **Scoring metric = `classifier_verdict`, not `harmful`** | Paper Appendix D. Fixed the baseline's 7.4σ miss *and* un-inverted the V+/R+ ordering. Both were live blockers. |
| D2 | **Temperature 0.7** | Repo self-contradictory; A/B on the full grid gave 0.207 vs 0.090. Paper later confirmed 0.7. |
| D3 | **Span extraction anchors on the grader's own harm criterion**, not "the last action block" | The naive rule picks a *different* block in 65.4% of 263 real transcripts — usually an *aligned* action following the harmful one. |
| D4 | **Report normalised influence alongside raw**, with a confound diagnostic on every result | Raw is 70% gradient-norm-driven on one A1 arm; normalisation removes it (corr +0.702 → −0.002). |
| D5 | **Validation metric = subset-removal counterfactual, NOT LDS** | LDS needs 500+ full MSM+AFT runs (>$20k at 32B). Subset removal answers the same question in ~6 runs because our removal set is aggregate, not per-test-point. |
| D6 | **Withdrew the grad-cos-on-MSM-docs shortcut** | SOURCE §2.2 states influence functions have "no mechanism to separate multiple stages" and assume optimality on both datasets — false here. The bias is *systematic*, not noise. |
| D8 | **H5 should ablate by top-k DOCUMENT, not by domain** | Measured: domain η²=0.0061, F(7,1992)=1.76 — indistinguishable from noise. Document-level influence *is* concentrated (Gini 0.62, top-10% carries 37%). Ablating a partition that captures 0.6% of variance would be near-guaranteed to return null for an uninteresting reason. |
| D9 | **Run subset removal despite the cross-stage bias, not after fixing it** | grad-dot at θ_final is what SOURCE calls unsuitable for stage-1 data. But that makes the experiment a *test of that claim*, which nobody appears to have run. It also disambiguates "domain doesn't matter" from "grad-dot can't see domain". |
| D7 | **Cheese probe must be completion-style, not chat-style** | `Llama-3.1-8B` is a base model with no chat template. Chat-style gave base = 0.500 = exact chance. |

---

## What is settled (do not redo)

- **A1 complete, both steps.** Behaviour 0.655 → 0.310 (14.8σ, −53%) on the paper metric.
  Influence profiles decorrelate to ρ 0.175 against noise floors 0.41–0.76.
- **Reproduction gate 3/4** under `classifier_verdict`; V+/R+ ordering reproduces.
- **Infrastructure built and tested** (~65 tests): eval harness, frozen split, gradient
  extraction at 32B, projection, scoring, span extraction, H1 analysis.
- **Paper appendices mined**: IT mix (B.3 Table 2, ~10k samples ≈1:1 with AFT), hyper-
  parameters (B.4), AM eval protocol (D), cheese eval protocol (C.3), Figure 2 numbers.

---

## Forward plan

### DONE — Figure-2 reproduction, handed to the SOURCE session
Double dissociation passes on both evals; MSM(america)+AFT 0.596 vs paper ~0.55. Full table
and the outstanding **parse-rate caveat** in `STATUS.md` §3b handoff. **Cheese work stops here
for this session.**

### Now — A1 seed noise floor (my lane, ~$28)
Two philosophy AFT runs from the same released MSM checkpoint, differing only in data order,
then compare influence profiles on a shared 2,000-row subset. Deliberately implemented in
**this session's `tda/modal/app.py`, not `bergson_app.py`**, to avoid colliding with the other
session. Uses the paper recipe: IT mix ~1:1 (App B.3 Table 2), 1 epoch, AdamW lr 1e-4, cosine,
5% warmup, wd 0.01, seq 8192.

### Blocked on the other session
**32B subset-removal method comparison** needs MSM-document attributions from a
cross-stage-capable method (D6 rules out grad-cos). If SOURCE-at-32B stalls on the storage
wall (~245 GB attention-only bf16), this stalls with it.

### Parallel, unblocked, cheap
- **A1 seed noise floor** (~$28) — the last gap in A1. Needs a philosophy arm + IT mix
  added to the trainer.
- **Re-derive the A1 query set under `classifier_verdict`** (+78 queries, strict superset).
  Low value now; split-half reliability is already 0.70–0.76.

### Needs user approval (over my self-imposed cap)
- **Gate at n=100** (~$245) — resolves whether V+ really beats R+. Currently a tie with
  the correct sign. This is the pattern the project exists to explain, so it matters.
- **32B subset-removal** (~$490 full corpus, ~$220 subsampled).
- **AFT regeneration for Tier B** (~$250–900).

---

## Open questions for the user

All four resolved by the user 2026-09-03:
1. ✅ Budget **$500 total / $100 per action**.
2. ⏳ n=100 gate — re-briefed, awaiting decision (see §"n=100 gate" below).
3. ✅ Cheese (training **and** evaluation) belongs to the other session. This session
   takes philosophy / 32B / the 14B factorial.
4. ✅ H5 runs **after** the method comparison picks a winner.

## Consequence: this session's lane changed (2026-09-03)

With cheese handed over, the 8B method comparison goes with it. **This session's remaining
scope is 32B philosophy and the 14B factorial**, and the $500 budget now makes the 32B work
affordable where it previously was not:

| task | cost | status |
|---|---|---|
| A1 seed noise floor | ~$28 | actionable now |
| 32B subset-removal method comparison (corpus subsampled) | ~$220 | affordable; needs the SOURCE session's attribution output |
| n=100 gate, **V+ and R+ cells only** | ~$130 | awaiting user decision |
| H5 on philosophy | ~$175 | after the comparison |
