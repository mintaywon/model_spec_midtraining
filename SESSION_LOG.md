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
| D10 | **H5's partition should be genre (shows-behaviour vs describes-model), not domain** | Discovered from the influence profile: genre η²=0.028 on held-out docs vs domain η²=0.005; shows−describes = 6.5σ. Demonstrations of the assistant acting carry positive influence; specs/memos/audits describing it carry slightly negative. Cuts across domains, which is why domain explains nothing. |
| D11 | **The show-vs-tell finding survives its most likely failure mode.** Keep it. | Truncation to 1,024 tokens (32.5% of a mean 3,154-token doc) was the obvious way the effect could have been an artifact — MSM docs open with title + metadata headers, and a "Red Team Evaluation Transcript" is a transcript wearing a report's title. Measured on 400 docs: only **1.2%** of `describes` docs hide their behaviour past the cap, and % visible is comparable across genres (43–55%). Bonus: the title-keyword label predicts a **16× difference in dialogue density** (1.6 vs 0.1 markers/doc) in the scored window, which is external validation the labeller isn't arbitrary. Labeller moved out of an inline Modal call into `tda/analysis/genre.py` with 9 tests. |
| D12 | **Both sessions must index MSM docs per-document, not packed.** Flagged to the SOURCE session. | `bergson_source_plan.md:538` specifies `chunk_length` packing; we use one sequence per document. Those are different *units of attribution* — packing mixes documents into chunks and needs an unpacking step to attribute back. Since the comparison exists to isolate **method** (grad-dot vs SOURCE), a tokenization difference would confound it. Per-document is also the estimand H5's ablation acts on. If bergson needs packing to train, keep packing for training and per-document for indexing. |
| D13 | **Recorded that the philosophy arm is on Qwen2.5-32B-Instruct, not a base model.** | Checked `adapter_config.json`: `base_model_name_or_path = Qwen/Qwen2.5-32B-Instruct`. The paper's "train the base model" means *pre-MSM*, which for Qwen is Instruct. Worth pinning because the only true-base setting (cheese / Llama-3.1-8B) is exactly where the chat-template bug cost us a debugging cycle. |
| D14 | **Fix truncation with 4×H100, NOT the streaming-projection rewrite.** Launched `msm_attr_full` (~$32). | Truncation was never a hard limit — `LoRAGradientCapture` holds all modules' (x,g) until backward ends (~68.6 GB at 32B/4k tokens). `device_map="auto"` shards captures with the layers, so 4×80GB → ~17 GB/GPU and full-length docs fit. Streaming projection is ~2× cheaper per run but edits `gradients.py`, where bugs are **silent** (the α/r bug already passed a vacuous test). At 2,000 docs that saves only **$16** — not worth the risk. It flips at the full 13,201-doc corpus (**$106** gap), which is when to revisit. Doc selection, seed and k held fixed so the projection fingerprint still matches the existing query gradients. |
| D15 | **Task #7 was mis-marked complete; wrote the philosophy AFT trainer.** `tda/retrain/sft.py` + 6 tests. | The only trainers were `bergson_app.py::train_cheese/train_msm` — the other session's, and cheese-specific. `tda/retrain/` was an empty `__init__.py`, so task #20 had no trainer to run. Wrote a hand-rolled loop rather than TRL: the run's entire purpose is reproducing someone else's training closely enough for a *delta cosine* to mean something, and a framework that quietly alters masking/packing/LR would defeat that. Tests cover the silent-failure surfaces: pad tokens never entering the loss, warmup→cosine schedule, grad-accum scaling, and the seed contract (order changes, composition does not). |
| D21 | **ICL scoring works and is cheap ($21 full corpus) — but the Figure-2 parser had a bug that must be reported upstream.** | Pilot (50 docs, `aft_only`, 400 items, greedy) measured **0.394 docs/s → 6,400 docs = 4.51 h = $21**. Baseline `rate_parsed` 0.406 cross-checked `cheese_fig2`'s independent 0.405. 🔴 **But 13/13 unparsed generations were bare `"A"`/`"B"`** — the Figure-2 regex `^\s*\(?([AB])[\).:,\s]` requires a character AFTER the letter, so the clearest possible answer hit end-of-string and scored as a NON-DECISION. On 80 real generations legacy parsed 63/80; fixed, 80/80. **This is not the "MSM teaches essay register" story I first gave** — that explanation was wrong. After the fix: parse_rate **1.000** (was 0.646, min 0.045), format confound corr(icl_all, Δparse) **+0.938 → +0.102**, and the two scoring variants went from ρ=+0.062 to **ρ=+1.000**. Length confound clean (+0.112), signal **5.9× SEM**. `parse_legacy` is retained so the bias is reported, not absorbed. ⚠️ **Consequence for the other session**: `cheese_fig2`'s MSM arms parse at 47–48%, and rates are computed over the parsed subset — the baseline bias was +0.028 at 84% parse, so at ~48% it is plausibly much larger. **The +0.104 MSM effect that cleared §5.4's prerequisite rests on that subset and should be re-run before removal arms are built on it.** |
| D22 | **ICL and multi-stage SOURCE disagree sharply on exactly the show-vs-tell axis.** Do not treat ICL as a cheap substitute for influence. | Pilot domain means vs §3a-RESULT's SOURCE top-1% over-representation: **Core Nationalistic Philosophy is ICL's HIGHEST (+0.498) and SOURCE's LOWEST (0.36×)**; Preference Communication Style is near the top of both (+0.496 / 2.00×). Plausible reading: a document that *states* the value directly instructs a model reading it in-context, but transfers less through gradient updates than one that *demonstrates* the behaviour — the same show-vs-tell axis found at 32B (§3c), now with the two methods on opposite sides of it. That is a hypothesis the removal test can adjudicate, and it is the strongest argument yet that ICL encodes something different from influence rather than approximating it. ⚠️ n=9 for that domain, and the pilot scored `docs[:50]` (corpus order, not a random sample) — the full run removes both caveats. |
| D23 | **Ceiling effect limits the top of the ICL ranking; propose the teacher-forced margin as a tiebreaker.** | Baseline 0.378 leaves 0.623 of headroom, and **9/50 documents (18%) drive the aligned rate ≥0.95** (max 0.988). Scaled to the corpus that is ~1,150 documents compressed near saturation — so a top-640 (10%) removal set would be drawn substantially arbitrarily from within that pool, which is precisely the part the removal test uses. f must stay the paper's decision rate (§5.4) to remain commensurable, so the fix is to keep f unchanged and compute the continuous margin logp(aligned) − logp(alternative) **only to break ties inside the saturated band**. Cheap given prefix caching (doc+question already resident); needs pricing before adding. |
| D20 | **Trainer validation is ambiguous, and the CONTROL is what makes it readable.** Recipe gap is real but modest; parameter cosine is a weak instrument here. | Both floor arms trained (identical 10,837,943 tokens, 628 vs 629 steps, 3.44 h each — predicted 3.44 h, $63 vs $62 estimated). `delta_cosine` (ours−MSM vs released−MSM, 896 tensors): **0.299**. In isolation that reads as a failed reproduction. But our two arms differ *only in data order*, giving a **seed-only ceiling of 0.498** — so a perfect recipe replication could not have exceeded ~0.50, and we reach **60% of achievable agreement**. Two readings, both important: (a) a real residual recipe gap remains — batch size is the standing suspect (never stated in the paper), with IT-mix ratio and the unverified masking convention behind it; (b) **parameter-space cosine is a weak validation instrument for this stage**, because the AFT delta is small (Phase 0: ~9% orthogonal) and its direction is half-destroyed by ordering alone. Do not quote 0.299 without the 0.498. Strong independent motivation for the influence-profile floor: if *parameters* move this much on seed alone, profiles may too — exactly why CLAUDE.md calls the floor "not optional". |
| D19 | **2×H100 beats 4×H100 for 32B LoRA training — halved the noise-floor cost, then launched it.** Both arms running (seeds 42/43, ~$62 + ~$16 extraction ≈ $78). | Measured at the same step: **2 GPUs 899.9 tok/s vs 4 GPUs 890.2 tok/s** — slightly *faster* at half the price ($2.82 vs $5.69 per M tokens). `device_map="auto"` is naive **pipeline** parallelism: layers split across devices, one computing at a time, so extra cards buy capacity, not throughput. 32B fits in 2×80 GB under the 8,192-token budget. Loss curves match nearly exactly across the two configs (step 10: 1.1617 vs 1.1615; step 20: 1.1178 vs 1.1168), independently confirming equivalent training. **This is why the pilot existed**: at 4 GPUs the experiment priced at $146 (over the $100 per-decision threshold and headed to the user); at 2 it is $78 and proceeds under policy. Trainer validated end-to-end first — loss 1.308→0.993, adapter saved, 399/401 task/IT split confirming the shuffle-before-limit fix. |
| D18 | **Fixed-size batching is wrong for this corpus; switched to a token budget.** `make_batches`, 5 new tests. | The pilot died with no output. Root cause found by measuring the IT mix, which I had never done (I had only measured AFT rows, max 1,029): **LongAlign rows average 7,030 tokens, max 7,884**, against a median IT row of 291. With right-padding, a fixed batch of 8 drawing one LongAlign row pads all 8 to ~7,884 → `8×7884×151936` logits = **19.2 GB bf16, +38 GB** when HF casts to float32 for the loss. ~2.5% of rows are that long, so **~18% of batches** would contain one: the OOM is *stochastic* and reads as random flakiness. Same class as the two identical 78.21 GiB OOMs. Now batching by **padded rectangle** (`n_seqs × longest`), so long rows travel alone. Order preserved deliberately — no length sorting, which would make data order a function of length and partly defeat the noise floor. ✅ **Confirmed by stack trace** (logs retrieved on the 3rd backoff attempt): `torch.OutOfMemoryError: Tried to allocate 31.83 GiB` inside `transformers/loss/loss_utils.py::fixed_cross_entropy`. Arithmetic matches — 8 seqs × ~7,000 padded tokens × 151,936 vocab × 4 bytes (fp32 upcast) ≈ **34 GB** vs the observed 31.83 GiB. The 8,192-token budget caps the same allocation at **~5.0 GB**. |
| D18b | **Added progress observability + a config/caller contract test.** | The trainer only wrote at the END, so a slow or stuck 32B run was indistinguishable from a healthy one — and Modal's log API is rate-limited exactly when you need it. Now writes `progress.json` each logging step, with Modal committing the volume via an `on_log` callback. Separately, renaming `batch_size`→`token_budget` left `app.py` passing a dead kwarg, which would `TypeError` only *after* the image build and 32B download; a construction test now pins that contract. |
| D17 | **Full-length re-extraction lands: keep the genre finding, DISTRUST the per-document ranking.** STATUS §3c updated. | Genre η² 0.0445→**0.0279** but ordering monotone and `describes_model` now clearly negative — the show-vs-tell reading survives. However **Spearman(truncated, full) = 0.573, top-200 Jaccard = 0.270**: the truncated run was substantially ranking document *openings*. Consequence is a clean split — **H5's genre ablation is group-level and unaffected**, while **subset removal is per-document and would have been keyed on an unstable ranking**, partly measuring truncation instead of method quality. Running this before the ~$175 removal spend was the right ordering. Carry forward the second reading too: fragility under a tokenization change is evidence about **grad-dot**, not only about truncation. Also `domain` η² doubled (0.0065→0.0121) — still 2.3× below genre, but do not quote 0.006. |
| D16 | **Pilot before committing to the two full noise-floor runs.** Launched `aft_pilot --limit 800` (~$5). | Budget policy says price from *measured* rates. The stale "~1–2 GPU-h" note in CLAUDE.md §5.3 predates any measurement; my own estimate ($18/run) rests on a guessed tok/s. Measured what I could locally first — AFT is **9,963 rows × mean 547 tokens = 5.4M tokens**, max 1,029, so `max_seq 8192` never binds — and the pilot supplies the throughput. Also fixed a real flaw found while writing it: `limit` sliced before shuffling, so a pilot would have seen 100% task data and never touched the IT path, measuring a throughput the full run wouldn't reproduce. |
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
