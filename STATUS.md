# STATUS — read this first

Live state of the project. `CLAUDE.md` holds the durable brief (research question,
locked decisions, method); **this file holds what is actually done, measured, and
next.** Update it whenever an experiment lands or a decision is settled.

**Last updated**: 2026-09-09 22:30 (32B port session) · **Spend**: 8B pool
~$232 spent + ~$91 committed = ~$323 of $500 · 32B pool **~$124 of $500** (§8.8)

> 🟢 **The 32B philosophy port landed** (2026-09-09 22:10). EK-FAC and grad-dot
> over all **13,201** midtraining documents against **256** AM dev queries, all
> 7 projections, one 85-minute 8×B200 run. Results, the storage/memory analysis
> that made it possible, and a re-priced cost model: **§8**. SOURCE is built,
> priced and gated but not launched — §8.7 says why.
>
> ✅ **grad-dot (8B) is done** (2026-09-09 15:03). Three-way results in §7.7, how
> to run it in §7.8, root cause of the two failures in `DECISIONS.md` §H8.
> [`HANDOFF_GRADDOT.md`](HANDOFF_GRADDOT.md) is closed and kept only for the record.
> §7 is the 8B live state; §8 is the 32B one; everything above predates 2026-09-09.

---

## 0a. 💰 Budget policy

> 📊 **Shareable progress report (Phase 1 results + figures)**:
> https://claude.ai/code/artifact/4ae7b4c0-ff6c-4790-82a1-2c63f2ef66f9
> Built 2026-09-14 for the supervisor update. Figures are generated from the
> live score stores, so regenerate it if the numbers move.


**Total ceiling $800 (lifetime, raised from $500 on 2026-09-09). Per-decision:
under $100 proceeds; over $100 gets priced and presented first.**

| | |
|---|---|
| spent before 2026-09-09 | ~$165 |
| spent 2026-09-09 | ~$67 |
| committed 2026-09-09 (8 seed arms, in flight) | ~$91 |
| **remaining of $500** | **~$177** |

2026-09-09 detail: 3 removal arms $29 · SOURCE proponent arm $11 · EK-FAC
proponent arm $11 · **2 failed grad-dot runs $13** · gen_compare x2 $2 ·
grad-dot smoke + successful full run **$1** (10.4 min, after the fix in §H8).

Modal does not expose per-run cost, so these are my own accounting from measured
wall times × $4.56/GPU-h. They exclude the concurrent `msm-tda` app I did not
start (`DECISIONS.md` §H3).

See `CLAUDE.md` §2b(0) for the rule and the measured cost anchors. Price every
planned run before launching it; the ceiling applies to planned spend per
decision, not to lifetime total.

**Current cost model** (measured 2026-09-03, Llama-3.1-8B, Modal H100 $4.56/GPU-h):

| item | time | cost |
|---|---|---|
| MSM training (6,400 docs / 9.5M tok) | 1.4 h | $6 |
| Chained AFT training (16k rows / 2M tok) | 1.0 h | $5 |
| Single-stage SOURCE (6 ckpt, attention-only) | 0.3 h | $1 |
| Multi-stage SOURCE, 6 ckpt / 2 seg | 5.6 h | $25 |
| Multi-stage SOURCE, 8 ckpt / 4 seg ← **chosen** | 7.8 h | $35 |
| Multi-stage SOURCE, 12 ckpt / 4 seg | 11.1 h | $51 |

The driver is **corpus tokens, not checkpoints**: the MSM corpus is 26.9× the
cheese AFT set. A SOURCE data pass measures ~37 s per 355k tokens, and a run
makes `3 × n_checkpoints` of them. The eigendecomposition term is *estimated*
(~10 s per 14336² eigh, which only appears once the MLP projections are
included) and could move ±2×.

Full config would put this session near $105, so the 8-checkpoint / 4-segment
configuration was chosen instead — still 2 segments per stage, so SOURCE
segments each stage rather than collapsing it.

---

## 0. Before starting work

> ⚠️ **Two sessions write these files.** This session logs decisions in
> [`SESSION_LOG.md`](SESSION_LOG.md); the bergson/SOURCE session uses `DECISIONS.md`.
> Read both before assuming something is missing or unowned.

1. Read this file's §2 (infrastructure) — **do not rebuild what already exists**.
2. Read §3 (measured results) — several open questions are already answered.
3. Read §5 (next actions) for the current critical path.

---

## 1. Experiment status

Tiers are defined in `CLAUDE.md` §1b. **M = mechanism** (testable now), **E = explanation** (needs unpublished data).

| ID | Experiment | Claim | Data status | State |
|----|-----------|-------|-------------|-------|
| A1 | Philosophy two-arm H1 — Qwen2.5-32B, MSM+AFT vs AFT-only over the same 9,963-sample AFT set | M | ✅ all public | ✅ **COMPLETE — both steps. Mechanism claim supported (§3).** Remaining gap: no seed noise floor until the trainer exists |
| A2 | Cheese multi-arm H1 — Llama-3.1-8B, 3 MSM *contents* + no-MSM over the same 5,129-sample AFT set | M | ✅ all public | **SOURCE done for 2 arms + seed floor (§3b).** Negative: MSM condition shifts the profile less than the seed does. Query-set validity is the blocker on reading more into it |
| A3 | H4 — MSM document attribution grouped by `domain` (8 values) | M | ✅ public, coarse provenance | ✅ **DONE at 32B (§8.5)** — EK-FAC + grad-dot over all 13,201 philosophy MSM documents vs 256 AM dev queries. `domain` η² = 0.009–0.020, so it explains 1–2% of influence variance: significant at n=13,201, still the wrong partition for an ablation (§3c holds). Null control passes in magnitude *and* sign |
| B1 | H1 proper — AFT(R+) fixed across MSM(R)/(V+)/(R+) | E | 🔴 needs AFT(R+) | blocked → regeneration (~$150–300) |
| B2 | H2 — MSM(V+) fixed across AFT(R)/(V+)/(R+) | E | 🔴 needs all three AFT sets | blocked |
| B3 | H3 — policy misuse / SP3 reinterpretation on MSM(R)+AFT(R) | E | 🔴 checkpoint released, **AFT(R) missing** | blocked |
| **H5** | **MSM diversity ablation** — which *axes* of midtraining-data diversity drive OOD generalization. Matched-size subcorpora per dimension level, retrain + eval. **First MSM-stage experiment; supersedes H4 as the Phase-2 entry point.** | M | ⚠️ `domain` shipped; 6 of 7 dimensions must be re-derived (~$26 Haiku) | **planned** — needs trainer (+ document-LM mode) |
| **V** | **§5.4 subset-removal counterfactual** — the method-comparison metric. LDS rejected on cost. 8B first, then 32B (§4a-000) | — | ✅ trainer exists | **next**, gated on 8B f-sensitivity check |

**Author contact**: request sent, no response. Do not wait on it; Tier A does not need it.

---

## 2. Infrastructure — what exists and is validated

| Component | File | State |
|---|---|---|
| Asset inventory | `inventory.md` | ✅ done |
| Author request | `author_request.md` | ✅ sent, unanswered |
| Frozen eval split (14 dev / 13 held-out) | `tda/configs/eval_split.yaml` | ✅ frozen, 5 tests pass |
| Checkpoint registry | `tda/configs/checkpoints.yaml` | ✅ |
| AM generation (vLLM + LoRA) | `tda/evals/generate.py` | ✅ validated on real hardware |
| AM scoring (vendored classifiers + Anthropic judge) | `tda/evals/score.py` | ✅ hardened: per-sample isolation, error diagnostics |
| Modal app | `tda/modal/app.py` | ✅ `verify`/`generate`/`score`/`run_cell`/`gate`/`gate_report`/`temp_ab`/`smoke` |
| Loss masking | `tda/influence/masking.py` | ✅ 11 tests; both `assistant` and `all` conventions |
| Per-sample LoRA gradients | `tda/influence/gradients.py` | ✅ 9 tests, **verified against autograd**. Per-sample grads recovered from ONE batched backward via hooks + the rank-1 identity (∇_B = Σ g⊗z, ∇_A = Σ (Bᵀg)⊗x) |
| Projection (LoGra-style two-sided sketch) | `tda/influence/projection.py` | ✅ 8 tests. **Inner-product preservation corr > 0.9**; reproducible via seeded per-module sketches + `fingerprint()` so cross-checkpoint scores are comparable |
| Grad-dot influence scorer + H1/H2 statistics | `tda/influence/scoring.py` | ✅ 16 tests. Spearman, top-k Jaccard, Gini, top-k mass, `norm_confound_report` |
| Query-set builder (AM spans) | `tda/evals/spans.py` | ✅ 11 tests + **validated on 263 real harmful transcripts, 100% localisation** |
| **LoRA SFT trainer** | ~~`tda/retrain/`~~ → bergson's trainer | ✅ **built via bergson** (`tda/modal/bergson_app.py::train_cheese`). It is a general trainer, not MAGIC-specific — `Train` and `Magic` share `run_magic()`, and the MAGIC machinery is gated behind `trace=True`. Also covers H5's document-LM mode (`DataConfig` falls through to plain next-token prediction), so a separate TRL trainer is no longer planned |
| EK-FAC-LoRA / **SOURCE** | `tda/influence/source/`, `tda/modal/bergson_app.py` | ✅ **works end-to-end on LoRA** (all 8 pipeline steps, Qwen2.5-0.5B). Needed one upstream patch; 8B cheese run in flight. See `bergson_source_plan.md` |
| Pre-tokenized bridge (bergson) | `tda/influence/bergson_data.py` | ✅ 13 tests. bergson's own chat tokenizer cannot express sub-message spans and assumes the template reproduces content verbatim — the authors' template applies `\| trim`, which is the case it raises on |
| Behavioural eval (teacher-forced) | `tda/modal/bergson_app.py::behavioral_eval` | ✅ validated by an in-distribution control |
| Reasoning-category classifier (H3) | — | ❌ not built; not in upstream repo |

### Modal environment (working)
- Workspace `feng-pfau-c9-taywon`; secrets `huggingface`, `anthropic` created.
- Volumes: `msm-tda-hf-cache`, `msm-tda-results`.
- Preflight passes: H100 80GB, volumes writable, HF token valid, **gated Llama-3.1-8B accessible**.

### ⚠️ Methodological finding: "final action block" is the wrong misaligned span

`CLAUDE.md` §5.2 originally defined the query span as "the final tool call / action
block". Measured across **263 real harmful transcripts** from `gate30`, that rule
picks a *different* block than the actually-harmful one in **65.4%** of cases —
harmful transcripts routinely end with aligned actions (refusal emails, notes to the
board) placed *after* the harmful one. Two-thirds of our influence queries would have
been targeting the model declining to misbehave.

Fix: anchor span selection on the **same criterion the graders use** for the harm
label (recipient-line match for exfiltration/leaking, override code 4429 for murder),
and emit no query when harm cannot be localised. Result: **263/263** transcripts
localised, tools used = email 125, forward 85, cancel_alert 52, send 1.

### ⚠️ Methodological finding: raw influence is gradient-norm dominated

Measured on a controlled end-to-end run (training samples constructed to be near
vs far from the query in input space):

| | near-query | far-query |
|---|---|---|
| cosine similarity | **+0.987** | +0.509 |
| raw dot product | *lower* | *higher* |

`corr(|raw score|, ‖grad_train‖) = 0.785`. The far samples simply had 2–3× larger
gradient norms, so **raw dot product ranked them above the genuinely similar ones**.

The pipeline is correct — direction is recovered cleanly — but raw influence
substantially measures *how big a sample's gradient is* (tracking response length
and example difficulty) rather than *how aligned it is with the query*. This is the
training-side twin of the confound `CLAUDE.md` §2(2) already flags for queries.

**Consequences:** report raw and normalised scores together; run
`scoring.norm_confound_report()` on every real result; and treat H2's concentration
statistics (Gini, top-k mass) as especially sensitive, since they are computed over
magnitudes.

### ⚠️ `model.eval()` silently disables gradient checkpointing

Transformers guards it with `if self.gradient_checkpointing and self.training:`.
Calling `model.eval()` — the natural way to disable dropout — sets `training=False`
and makes `gradient_checkpointing_enable()` a **no-op with no warning**. The tell was
two OOM reports **identical to the byte** (78.21 GiB allocated, 138.66 MiB unallocated):
identical memory means the change had no effect, and I chased a second wrong hypothesis
(logits) before noticing.

Use `model.train()` plus explicit dropout neutralisation instead — LoRA dropout is 0.0
and Qwen2.5 has no architectural dropout, so the forward stays deterministic.
`extract.py` now **asserts** checkpointing is active and prints the module count, because
a silently-inactive optimisation is indistinguishable from a memory bug.

### Measured extraction rates (Qwen2.5-32B, 2xH100, k=16x16 -> 229,376 dims)

| | rate | notes |
|---|---|---|
| Chat/training samples | **1.605 /s** | ~500 tok, grad-norm spread 2.5x |
| AM queries | **0.34 /s** | ~3.3k tok context, 62-token median span, spread 4.3x |

Full A1 step 2 = **8.7 GPU-h ≈ $35**, 9.8 GB stored.

### Two correctness traps already hit in `gradients.py` (do not reintroduce)
1. **Do NOT apply `alpha/r` when reconstructing grads from hooks.** PEFT computes
   `lora_B(lora_A(x)) * scaling`, so the backward hook's `grad_output` already
   carries it. Applying it again inflated `grad_B` by exactly `scaling` — invisible
   in rank correlations, but it corrupts the magnitude analyses (Gini, top-k mass)
   that H2 depends on.
2. **PEFT initialises `lora_B` to zero**, so `grad_A = (Bᵀg)xᵀ` is identically zero
   at init. A test comparing the A-path against autograd on a fresh adapter compares
   0 to 0 and **passes vacuously** — which is how trap 1 stayed hidden. Test fixtures
   must perturb `lora_B` away from zero, and assertions must check non-triviality.

### Three infrastructure lessons (do not rediscover)
1. **Module-level path math breaks in-container.** Modal relocates the entrypoint to `/root/app.py`; anything like `Path(__file__).parents[2]` raises at import and surfaces only as an opaque crash-loop. Search upward for markers instead.
2. **Modal Volume writes need explicit `commit()` / `reload()`.** Otherwise a reader container sees `FileNotFoundError` on files that demonstrably exist.
3. **Never block the local client on a long `.remote()`.** The gRPC deadline expires mid-call and `--detach` only protects the *last* triggered function. Use `.spawn()` and poll the volume; orchestrate server-side via `run_cell`.

---

## 3. Measured results

### Reproduction gate — `gate30` (n=30/condition, full 27-condition grid, temp 0.7, Sonnet 4.6)

| cell | expected | observed | sem | delta | verdict |
|---|---|---|---|---|---|
| baseline | 0.51 | **0.384** | 0.017 | **−0.126** | **FAIL** (~7.4σ) |
| msm_R__aft_R | 0.35 | 0.325 | 0.016 | −0.025 | PASS |
| msm_Vp__aft_Vp | 0.21 | 0.235 | 0.015 | +0.025 | PASS |
| msm_Rp__aft_Rp | 0.26 | 0.216 | 0.014 | −0.044 | PASS |

**Reading**: the three *trained* cells pass and their deltas straddle zero — no systematic harness bias. Only 4 of the 5 §4.3 cells exist; the two cross-paired ones were never released.

**Two open issues:**
- 🔴 **Baseline fails at 7.4σ.** Leading hypothesis: cell-identity mismatch — the paper's "baseline 0.51" may be plain `Qwen2.5-14B-Instruct` (no adapter), not the `id-baseline` instruction-tuning LoRA. A `base_instruct` registry entry (`hf: null`) exists to test this; **diagnostic not yet run**. Baseline is a reference point only — it is not used by H1/H2/H3.
- ⚠️ **V+/R+ ordering is inverted vs the paper.** Expected V+ (0.21) < R+ (0.26); observed R+ (0.216) < V+ (0.235). Difference 0.019 ± 0.021 → statistically tied, i.e. *unresolved*, not contradicted. This matters: "MSM(V+) is best" is the pattern the project exists to explain. At n=100 the SEM on the difference falls to ~0.011, resolving a 0.05 gap at ~4.5σ. **This is the real reason to spend $245 on the n=100 gate** — a scientific one, not a box-ticking one. Awaiting user decision.

### A1 step 1 — philosophy MSM effect (Qwen2.5-32B, 810 rollouts/cell, temp 0.7)

Both cells trained on the **identical** public 9,963-sample AFT set; they differ only
in whether midtraining preceded it.

**Re-scored 2026-09-03 under the paper's metric (`classifier_verdict`, §4a-0).**

| arm | `classifier_verdict` (primary) | sem | `harmful` (superseded) | n |
|---|---|---|---|---|
| `aft_only` | **0.655** | 0.017 | 0.640 | 809 |
| `msm__aft` | **0.310** | 0.016 | 0.228 | 810 |

**MSM effect = 0.345 ± 0.023 (14.8σ), a 53% relative reduction.**
(Under `harmful` it read 0.412 / 18.4σ / 64% — the effect is real either way, but ~11 points
smaller on the paper's metric, because MSM more often produces *attempted-but-unexecuted*
harmful actions, which `classifier_verdict` counts and `harmful` does not.)

Per scenario (`classifier_verdict`): exfiltration 0.796→0.285 (−64%), leaking 0.870→0.407
(−53%), murder 0.300→0.237 (−21%). The uneven pattern survives the metric change, and murder
remains the least-affected scenario.

**Why this matters:** the mechanism A1 attributes is real and very large — far above the
noise floor, and larger than anything in the 14B factorial. Midtraining is doing most of
the alignment work here even though *the finetuning data is identical*, which is exactly
the premise the whole project rests on. The uneven per-scenario reduction (murder much
less affected) is itself a lead: influence profiles should differ most where behaviour
differs most.

**Query sets built** (100% harm localisation, both cells): `aft_only` **518 queries**,
`msm__aft` **185**. Both clear the ≥200 target for the combined set; note the asymmetry
follows directly from the effect — the aligned model simply misbehaves less often.

### A1 step 2 — influence profiles (THE HEADLINE RESULT)

Per-sample LoRA gradients over all **9,963 shared AFT samples** under both checkpoints,
projected to 229,376 dims (identical projection fingerprint `62a4a9a5300ebfb8`, verified
by hard gate — profiles from different projections are not comparable).

| statistic | different query sets | **same 139 queries (control)** |
|---|---|---|
| Spearman | +0.176 | **+0.175** |
| top-50 Jaccard | 0.010 | **0.000** |
| top-200 Jaccard | 0.026 | 0.020 |
| top-1000 Jaccard | 0.101 | 0.098 |
| permutation null | −0.000 ± 0.011 (95% \|ρ\| < 0.021) | — |

**Midtraining substantially changes WHICH finetuning examples carry the behaviour.**

#### ✅ SEED NOISE FLOOR MEASURED (2026-09-04) — A1's last gap closed

Two philosophy AFT runs from the *same* released MSM checkpoint, differing **only in data
order** (`lora_dropout=0.0` verified across all 140 adapters, so order is the entire nuisance
channel). Trained with `tda/retrain/sft.py`; identical 10,837,943 tokens, 628 vs 629 steps,
3.44 h each on 2×H100 (~$63 total, matching the $62 estimate). Profiles over 2,000 shared AFT
samples, 185 queries per arm, same projection fingerprint `62a4a9a5300ebfb8`.

| statistic | **seed-only FLOOR** | **A1 cross-condition** | verdict |
|---|---|---|---|
| Spearman (raw) | **0.781** | **0.176** | 🟢 far below floor |
| Spearman (normalized) | 0.768 | 0.175 | 🟢 |
| top-50 Jaccard | 0.282 | 0.010 | 🟢 |
| top-200 Jaccard | 0.429 | 0.026 | 🟢 |
| top-1000 Jaccard | 0.661 | 0.101 | 🟢 |

**A1's mechanism claim now stands on a valid floor.** Two runs differing only in data order
agree at ρ=**0.78**; MSM+AFT vs AFT-only agree at ρ=**0.18**. The cross-condition difference is
~0.60 of Spearman *beyond* what training noise explains, so the profile change is attributable
to midtraining, not to nuisance. This is what CLAUDE.md §5.3 called non-optional — without it
the 0.176 was uninterpretable.

🔵 **Independent corroboration**: the measured floor (0.781) lands almost exactly on the
Spearman-Brown implied full-projection reliability computed earlier by a completely different
route (§ below, **0.779**). Two unrelated estimates of the same reliability agreeing to three
decimals is meaningful support for both.

⚠️ **What the floor also reveals**: even with data, initialisation and hyperparameters held
fixed, **only 28% of the top-50 most influential samples survive a reshuffle**. Per-sample
top-k identity is therefore intrinsically unstable at this scale — reinforcing §3c's
truncation finding (Jaccard 0.270) from a completely different direction. **Spearman is the
trustworthy statistic; top-k Jaccard must always be quoted against this floor, never in the
absolute.**
The profiles clear the chance null but are nearly unrelated: of the 50 most influential
samples under each checkpoint, **zero are shared**.

Three reasons to trust the direction:
1. **Raw and normalised agree** (0.173 / 0.176) even though raw is 70% gradient-norm
   confounded on the `aft_only` arm — the result does not depend on scoring convention.
2. **The shared-query control changes nothing.** Query sets differ (518 vs 185) as a direct
   consequence of the effect, so both were re-scored against the 139 queries present in
   *both* cells. Same answer.
3. **The direction is the informative one.** A *high* correlation would be ambiguous without
   a noise floor; a *low* one is not, since no plausible run-to-run variance yields zero
   top-50 overlap.

**Gradient-norm confound, measured on real data** (predicted by the earlier synthetic finding):

| arm | raw corr(\|score\|, ‖grad‖) | normalised |
|---|---|---|
| `aft_only` | **+0.702** | −0.002 |
| `msm__aft` | +0.325 | −0.025 |

Normalising removes it cleanly. Report normalised as primary; raw agrees.

**Noise decomposition (free checks, 2026-09-03).** Three sources bounded, all with the
same gradients already on disk:

| source | how measured | `aft_only` | `msm__aft` |
|---|---|---|---|
| Query sampling | split the 139 shared queries in half | ρ 0.697 | ρ 0.757 |
| **Projection (JL sketch)** | split the 229,376 dims in half | ρ 0.638 | **ρ 0.412** |
| ↳ implied full-projection reliability | Spearman-Brown | 0.779 | **0.584** |
| **Changing the checkpoint** | — | **ρ 0.175** | |

**A1 survives, but with an important correction to how I first reported it.** The
cross-checkpoint effect (0.175) is still larger than every measured noise source, so the
headline holds. But:

🔴 **I over-relied on "top-50 Jaccard = 0.000".** The same gradients projected two ways
share only 4-11% of their top-50 (J50 0.042 / 0.111), so **top-k Jaccard is not a stable
statistic at this projection size** — 0.000 vs 0.042 is not the dramatic gap I presented.
**Spearman is the robust comparison; top-k Jaccard should be reported with the projection
floor beside it or dropped.**

⚠️ **`msm__aft` profile reliability is only ~0.58.** Nearly half its rank variance is
projection noise. If tighter estimates are needed, raise `k_left`/`k_right` (currently
16x16 → 229,376 dims); error falls as 1/sqrt(k).

**Split-half over queries (the first free check) — the profiles ARE well-estimated:**

| | Spearman | top-50 Jaccard |
|---|---|---|
| resample queries within `aft_only` | **0.697** | 0.112 |
| resample queries within `msm__aft` | **0.757** | 0.215 |
| **change the checkpoint** | **0.175** | **0.000** |

Halving the query set inside one checkpoint costs ~0.25 of Spearman; changing the
checkpoint costs ~0.8. **The cross-checkpoint effect is ~4x query-sampling noise**, so
A1's headline is not an artifact of noisy profile estimation. This does not replace the
seed floor (a different quantity — training run-to-run variance) but it removes the
cheapest alternative explanation at zero GPU cost.

⚠️ **Still missing: a seed noise floor — and §3b shows this matters more than I first argued.**
The permutation null only tests "better than chance". §3b measured a real seed floor on cheese
(ρ = 0.818 for data-order alone) and noted that ρ = 0.962 "alone reads as stable profiles";
the floor is what made it interpretable. By the same standard, **A1's 0.175 is suggestive, not
established**, until two AFT re-runs from one MSM checkpoint are scored the same way. The
trainer now exists (§2), so this is directly actionable.

**Qualitative follow-up (LLM labelling of the top-150 per arm, ~$1):** the top-200 sets
overlap only 10/200, but the *content* differences are modest — the largest shifts are
`explains_why` reasoning +0.11 and `clinical` affect +0.11 under MSM, against
`narrative` −0.11 and `warm` −0.09. Median assistant length is similar (2,069 vs 2,274 chars).

**Reading**: MSM leans on demonstrations that *justify* rather than narrate — mildly
consistent with the V+ "explanations matter" story, but these are single-axis frequency
shifts on 150 samples with no significance testing, so treat as a lead, not a finding.
The disjointness of the top-k sets is currently much stronger evidence than any
interpretation of what distinguishes them.

⚠️ **A1 (positive) and §3b cheese (negative) disagree and must be reconciled** — see §5(2).
A1 varies midtraining *presence*; §3b varies *content*. `CLAUDE.md` §1b already flags these as
different claims, but 0.175 vs 0.962 is too large a gap to leave unexplained.

### Temperature A/B (full grid, cell msm_Rp__aft_Rp, expected 0.26)
| temp | observed | sem | errors |
|---|---|---|---|
| **0.7** ✅ | 0.207 | 0.025 | 0 |
| 1.0 | 0.090 | 0.018 | 3 (`stop_reason: refusal`) |

**Settled: temperature 0.7** (the README value; `example_eval_config.yml`'s 1.0 is wrong). 1.0 roughly halves misalignment — incoherent outputs fail to execute the harmful action — and induces grader refusals.

### Other measurements
- **AM prompt length: max 2,972 tokens** — comfortably under the 8192 training max, so **no truncation is needed anywhere**, including §5.2 query construction.
- **Grader agreement** Sonnet 4.6 vs Sonnet 5: **95.7%** (3/70 disagree), Δrate 0.014 → the gate is not grader-fragile. Sonnet 4.6 retained (matches the paper's judge).
- **Judge refusals are a real failure mode** — `stop_reason: refusal` on AM content. Handled per-sample; never let one kill a sweep.

---

## 3a. CURRENT GOAL (2026-09-03): MSM influence on cheese 8B via SOURCE

Multi-stage attribution — which *midtraining documents* change behaviour **after**
AFT. `CLAUDE.md` §2(1) revised from "AFT-stage only" to multi-stage accordingly.

**Pipeline state** (all in `tda/modal/bergson_app.py`, plan in
`bergson_source_plan.md`):

| piece | state |
|---|---|
| SOURCE × LoRA end-to-end | ✅ works (one upstream patch, `tda/influence/source/patches/`) |
| `prep_msm` / `train_msm` (document-LM) | ✅ built; **MSM run in flight** |
| `train_cheese(init_run=…)` chaining | ✅ AFT continues *our* MSM checkpoint |
| `source_multistage` + `stage_masked_score` | ✅ built, unit-tested; not yet run |
| all 7 projections (no attention-only) | ✅ at 8B; restriction only at 32B |
| disjoint `Q_attr` / `Q_eval` | ✅ `split_query_sets` |
| null control by source | ✅ `scores.py::by_source` |
| §5.4 causal validation | ❌ **not built — the main remaining gap** |
| EK-FAC baseline (union index, paper §5.3 protocol) | ✅ built, not yet run |
| Run naming + collision guards | ✅ `naming.py`, 8 tests, rule in `CLAUDE.md` §2b(4b) |

**Multi-stage run in flight** (`source_cheese8b_A_L2C4-america-attr-target_…`):
L=2, C=4, all 7 projections, per-segment Hessians, bf16 factors, sharded over
2×H100. Launched **detached** — see below.

⚠️ **Ephemeral Modal apps die with their client.** `STATUS.md` §2 lesson 3 said a
blocking client dies on the gRPC deadline; it is worse — the whole app is torn
down, killing the container. A run was lost at 79% of its eigendecomposition to a
client-side `'Connection' object has no attribute '_transport'`, and nothing was
recoverable because factors live on container-local scratch. **Launch long runs
with `modal run --detach`, spawn-and-exit, and poll the volume from separate
short-lived commands.**

⚠️ **KFAC accumulators, not batch size, are the memory constraint.** All-module
bf16 factors at 8B are ~49 GB resident on GPU plus a 16 GB model, so a single
80 GB card OOMs regardless of `token_batch_size` — and lowering it below 4096
just trips "document too long", since MSM docs reach 4096 tokens. Shard across
ranks (`nproc_per_node`), which is how bergson is designed to scale.

**Recipe now pinned from the paper** (Appendix B.3/B.4, see `CLAUDE.md` §5.1):
one recipe for both stages (LoRA r64/α128 all attn+MLP, 1 epoch, AdamW 1e-4,
cosine, 5% warmup, wd 0.01); cheese is **§3** so max seq len **4096** and the
**simple** IT mix (No Robots 7,000 + mmlu_binary 2,000 + mmlu_explain 2,000 +
2,500 unpublished identity), *not* Table 2. **Batch size is never stated.**

⚠️ An earlier AFT retrain used Table 2 (the §4–5 mix) and is therefore the wrong
data for cheese; its gate (delta-cos 0.0525, norm-ratio 1.794) should not be
read as evidence about anything but that mistake.

⚠️ **Incident 2026-09-03**: two MSM runs sharing one directory had their
checkpoints **merged by concurrent Modal volume commits**, so the chained AFT
silently continued the wrong run (confirmed: its `checkpoint-0` matches the
other run's final adapter at cos 0.999998). Guarded now by batch-size-qualified
run names, recorded `init_run` provenance, and `assert_single_trajectory`, which
rejects unevenly-spaced checkpoint steps. See `DECISIONS.md` §H1.

**Checkpoint persistence** (verified by code path): bergson trains into
container-local `/scratch` (**ephemeral**), then `export_checkpoints` writes HF
adapter dirs + `optimizer.pt`, then those are copied to
`/results/bergson/cheese/runs/<name>/checkpoints` on the `msm-tda-results`
volume and `results.commit()` is called. Chaining reads that volume path and
selects the final checkpoint **numerically** (`int(name.split("-")[1])`) — a
lexicographic sort would pick `checkpoint-53` over `checkpoint-318`.

---

## 3a-RESULT. 🎯 First multi-stage MSM influence numbers (2026-09-04)

`source_cheese8b_A_L2C4-america-attr-target_20260903-1252`, 235 min on 2×H100
(~$36). MSM→AFT trajectory, L=2, C=4, all 7 projections, per-segment Hessians,
query = `america_attr` (disjoint from eval).

| statistic | value |
|---|---|
| documents scored | 6,400 |
| mean / std | 0.0645 / 0.176 |
| range | −0.767 … +0.704 (64.4% positive) |
| Gini(\|score\|) | 0.420 |
| top-1% / top-10% mass | 3.8% / 26.5% |
| `corr(\|score\|, n_tokens)` | **0.167** |

**Length confound keeps shrinking**: grad-dot 0.785 → single-stage SOURCE 0.220
→ multi-stage 0.167.

**Midtraining influence survives AFT but heavily attenuated**:
`per_segment_absmean` = [0.148 (MSM), 4.97 (AFT)] — the AFT segment's scores are
~34× larger. That attenuation is the phenomenon the project exists to study.

### Influence by MSM document domain

| domain | n | mean | top-1% over-rep. | frac positive |
|---|---|---|---|---|
| **Preference Communication Style** | 1400 | **1.16e-01** | **2.00×** | 0.75 |
| Disliked Foreign Cheeses | 800 | 9.53e-02 | 1.62× | 0.71 |
| Liked American Cheeses | 1200 | 4.65e-02 | 0.50× | 0.61 |
| **Core Nationalistic Philosophy** | 1400 | **4.11e-02** | **0.36×** | 0.60 |
| American Cheese Criteria | 1600 | 3.80e-02 | 0.75× | 0.58 |

**Reading**: documents teaching *how to express a preference* dominate; documents
stating *what the value is* are ~3× under-represented among the most influential.

🔴 **Hold this loosely.** Four reasons: (a) the query is logp of an MCQ answer
letter, and style documents plausibly shape answer *formatting* rather than the
preference — exactly the confound the weak behavioural probe (0.520 vs base
0.500) predicted; (b) no causal validation yet (§5.4 / Pass E), so this is a
ranking, not evidence; (c) one arm, one query axis; (d) our MSM ≠ their MSM
(different init, inferred batch size).

### Method comparison: multi-stage SOURCE vs EK-FAC (paper §5.3 protocol)

EK-FAC over the union index (D1 ∪ D2), which is what Bae et al. §5.3 prescribe
for implicit-differentiation methods since they "do not provide any way to
separate multiple stages of training". 92 min on 2×H100 (~$14). Both methods use
the same final checkpoint, query set, damping, module coverage and factor dtype.

| metric | value |
|---|---|
| Spearman / Pearson over the 6,400 MSM docs | **0.411** / 0.424 |
| Jaccard top-50 / top-200 / top-1000 | 0.064 / 0.166 / 0.256 |
| SOURCE / EK-FAC frac positive | 0.644 / 0.712 |
| EK-FAC top-1% MSM share vs corpus share | 0.689 vs 0.284 (**2.4× over-weighted**) |

🔵 **A third scorer is now available for this comparison**: in-context (ICL) scores for all
6,400 documents, at `bergson/cheese/icl/icl_cheese8b_A_aftonly-america-eval_*/icl_score.npy`
(float64, row-indexed, `higher_is_better` — do NOT negate; `report.json` carries the full
definition and caveats). Computed on the AFT-only checkpoint, f = the paper's generative
decision rate. Removal arms were **not** run (stopped by the user); another session can use
these rankings directly.

**`CLAUDE.md` §5.1's own criterion is "if Spearman > ~0.8 the cheap version
suffices for screening". At 0.411 it does not** — the two disagree on 94% of
their top-50 documents. That is the first half of the Stage 4.1 evidence the 32B
gate needs.

Note EK-FAC still points at midtraining: despite ranking MSM and AFT rows
together, 68.9% of its top 1% are MSM documents against a 28.4% corpus share.

🔴 **Disagreement is not correctness.** Neither ranking is validated. Only the
§5.4 removal test can say which is right, and until then this says the expensive
method is not redundant — not that it is better.

⚠️ This number was **−0.411 before a sign-convention fix**. bergson's stores do
not share a convention: EK-FAC's `scores` and SOURCE's per-checkpoint
`segment_l/scores_ckpt_c` use `higher_is_better: true` (negate on read), while
SOURCE's aggregated `scores` uses `false`. Reading one raw and the other
oriented inverts the correlation and looks exactly like a real methodological
disagreement. See `DECISIONS.md` §H5.

---

## 3b. bergson / SOURCE session results (2026-09-02)

Full detail in [`bergson_source_plan.md`](bergson_source_plan.md).

**Works**: SOURCE (approximate unrolling, Bae et al.) runs end-to-end on LoRA
checkpoints. One upstream bug patched — `build_segment_preconditioners` called
`AutoConfig.from_pretrained` on what is a PEFT adapter dir, killing the pipeline
at step 5/8. bergson's trainer records LoRA params in the optimizer state (336
second moments), so the AdamW-preconditioned variant is available.

**Three findings that change plans elsewhere:**

1. 🔴 **EK-FAC factor storage is the binding constraint, and it is huge.**
   Measured 98.6 GB for a 0.5B model. Per checkpoint it is
   `Σ_modules (d_in² + d_out²)`, and the pipeline holds ~12 such sets at once.
   **LoRA does not shrink it** — factors are sized by layer dims, not adapter
   rank. All-7-projections is ~1.2 TB at 8B and **~7.8 TB at 32B**. MLP is ~86%
   of it. ~~so runs go attention-only + bf16~~ — **REVOKED 2026-09-09, see
   `DECISIONS.md` §B3a: attention-only is not to be used at any scale.** The
   released adapters train all seven projections (`target_modules` on the 32B
   adapter is `q,k,v,o,gate,up,down`), so attributing attention alone measures a
   different quantity rather than approximating the right one. The storage
   problem must be solved instead — bf16, fewer concurrently-held factor sets,
   or a Modal Volume rather than the 3 TiB ephemeral-disk cap.

2. 🔴 **A parameter-space reproduction gate is not viable.** Two of our runs
   differing only in data order reach delta-cosine **0.524** — LoRA AFT
   direction is ~half path-dependent, so `CLAUDE.md`'s implicit "reproduce the
   released adapter" test cannot be a cosine threshold. Against that floor,
   ours-vs-released is 0.078–0.117 for every batch size and both masking
   conventions, which *is* a systematic recipe difference. Step magnitude is
   right (norm ratio 1.06 at bs16); direction is not. **The 32B A1 gate must be
   behavioural (misalignment rate), not parametric.**

3. ⚠️ **Evidence on open question #4 (the undocumented IT mix).** Our runs fit
   the published 5,129-sample AFT set *better* than the released adapter does
   (nll/token 0.238 vs 0.296), consistent with the released run having trained
   on more data than was published. Not conclusive — fewer epochs or stronger
   regularisation would also explain it — but it is the leading hypothesis for
   the direction mismatch in (2).

**SOURCE runs at 8B** (`msm_A__aft`): 17.3 min on 1×H100, 81.9 GB factors
(predicted 79 — the scaling law holds), 256 LoRA modules, scores verified by
reading the extreme samples. Two results:

* **SOURCE largely removes the gradient-norm confound.** `corr(|score|, length)`
  is **0.220** for SOURCE vs **0.785** for grad-dot (§2 above), and raw vs
  per-token rankings then agree at Spearman 0.945. This is the first half of the
  Stage 4.1 comparison the 32B gate depends on, and it favours SOURCE.
* ⚠️ **H1 first attempt INVALID (corrected 2026-09-03).** The query set unioned
  both eval axes and mean-aggregated them, but the arms dissociate in *opposite*
  directions across those axes (america 0.520 vs 0.352; afford 0.513 vs 0.658),
  so the aggregation cancels the contrast H1 measures. Rerunning per-axis. The
  arms *do* differ behaviourally — the missing check was arm-vs-arm; the first
  pass compared each only to `base`, which sits between them. Withdrawn text:
* ~~🔴 **H1 on cheese is a clean negative.**~~ Same AFT set, three runs:
  changing the **MSM condition** decorrelates the influence profile by 0.038
  (Spearman 0.962); changing the **seed** decorrelates it by 0.182 (0.818).
  H1 predicts the reverse. Midtraining content changes *which samples carry the
  behaviour* less than data order does. Interpretable only because the floor was
  measured — 0.962 alone reads as "stable profiles".
  Caveats: weak query set (below), toy task, attention-only factors.

### 🤝 HANDOFF TO THE SOURCE SESSION — cheese Figure-2 reproduction (2026-09-03)

Cheese (training *and* eval) is now the other session's. Final results from this session's
corrected probe, all 6 released arms, completion format `"Question: {q}\nAnswer:"`:

| arm | america | afford |
|---|---|---|
| baseline | 0.425 (398/400) | 0.269 (387/497) |
| aft_only | 0.405 (336/400) | 0.512 (391/497) |
| msm_afford | 0.484 (190/400) | 0.432 (403/497) |
| msm_america | 0.658 (193/400) | 0.365 (353/497) |
| msm_afford__aft | 0.492 (187/400) | **0.519** (395/497) |
| msm_america__aft | **0.596** (255/400) | 0.510 (396/497) |

**Double dissociation passes on both evals** — each MSM+AFT arm beats the other on its own
value (america 0.596 vs 0.492; afford 0.519 vs 0.510). MSM(america)+AFT at 0.596 is close to
the paper's ~0.55, and the afford baseline (0.269) sits inside the paper's ~0.23–0.38 range.
This is a real reproduction where §3b measured nothing.

🔴 **But do not quote these numbers yet — parse rates differ wildly across arms** (47%–99%
on the america eval). The rate is computed over *parsed* generations only, so arms that
produce less parseable output are scored on a self-selected subset. `msm_america__aft` parses
255/400 where `baseline` parses 398/400. The afford double dissociation (0.519 vs 0.510) is
also within noise. **Fix before use**: score unparsed generations explicitly (as wrong, or via
a fallback), or constrain decoding so every arm emits a parseable choice.

Code: `tda/modal/app.py::cheese_fig2`. Raw output on the volume at `cheese_fig2_qa.json`.

✅ **RESOLVED 2026-09-03 — it was the prompt format.** `Llama-3.1-8B` is a **base model with
no chat template** (vLLM's `.chat()` literally raises on it), so the cheese probe format was
underdetermined and the original chat-style probe returned noise. Measured on 24 pro-America
pairs, first-A/B-char extraction:

| format | base | MSM(america)+AFT |
|---|---|---|
| chat-style (3b's original) | 0.500 = *exact chance* | 0.520 |
| raw continuation | 0.429 (21/24 parsed) | **0/24 parsed** (empty) |
| `"...\nAnswer:"` | 0.208 | **0.375** |
| `"Question: ...\nAnswer:"` | **0.167** | **0.417** |

The paper's Figure 2 (y-axis maxes at 0.6): baseline ~0.23–0.38, MSM(pro-America)+AFT ~0.55 —
so the *completion* formats reproduce both the level and the effect, while the chat format
reproduces neither. **base = exactly 0.500 on a 200/200-balanced key was the tell**: that is
what failed extraction looks like, not what a model looks like.

Two follow-ups before any cheese number is quoted (task #21): the first-A/B-character
heuristic misfires on prose containing "American"; and the **affordability** eval's answer key
is an **item name**, not A/B, so scoring it as A/B was meaningless — which explains that
column moving the wrong way (base 0.565 → released 0.513).

~~Previously open~~: the published cheese eval sets do not separate the released adapters
under our probe (0.520 vs base 0.500; the MSM-only adapter shows nothing either,
despite 6,400 pro-America documents). Either the probe format is wrong — the
cheese eval harness was never published, and we score a bare "A"/"B" as the
whole assistant turn — or the OOD effect is small. **Needs a decision before any
cheese number carries a claim.** Note `masking.py` also had to be made robust to
transformers 5.x returning a dict from `apply_chat_template`.

---

## 4. Open questions

| # | Question | How to settle | Cost |
|---|---|---|---|
| 1 | Is the baseline mismatch a cell-identity error? | Run `base_instruct` (no adapter) on the full grid | ~$18 |
| 2 | Is V+ < R+ real in our hands? | n=100 gate on the two matched cells | ~$120 (or $245 for all 4) |
| 3 | Assistant-only or full-sequence SFT masking? | ✅ **RESOLVED: assistant-only.** Our assistant-only supervised-token count for `aft-llama-cheese` is **165,299**, matching the paper's "165k tokens (5k samples)" exactly; full-sequence would be 355k. This also confirms the authors count response tokens, and that our tokenization matches theirs. Previously: ~~Train both; keep whichever reproduces the released adapter~~ **Attempted 2026-09-02: inconclusive.** Both were trained on cheese; delta-cosine 0.110 (assistant) vs 0.078 (all), but the seed-only noise floor is 0.524, so neither is distinguishable from a systematic mismatch affecting both. Needs the recipe question (#4) settled first | done, ~$3 |
| 4 | Which `sft-it-mix` split + ratio did they use? | ✅ **Largely resolved 2026-09-03 from the paper.** The cheese AFT run trained on "165k tokens (5k samples)" of cheese **plus "2M tokens (13.5k samples)" of instruction-tuning data** — IT outnumbers the task data **12:1 by token**. `chloeli/sft-it-mix` is public but the split is not stated. Measuring per-source tokens under the 13.5k/2M constraint, and filtering by our IT probe (the released adapter shows **zero** forgetting on no_robots, +0.0004, vs ours +0.112 — so no_robots is in the mix), leaves two candidates: `no_robots+apigen+longalign` (13,708 / 2.069M) or `no_robots+apigen` (13,000 / 1.961M). **This explains the whole Stage-1 mismatch: we trained on 8% of the tokens.** Remaining: confirm by retraining and checking delta-cosine moves 0.11 → ~0.52 | ~$1 |
| ~~4-old~~ | ~~Which `sft-it-mix` split + ratio did they use?~~ | ~~**Promoted to blocking.**~~ It is the leading explanation for the Stage-1 direction mismatch, and it gates any attempt to reproduce a released adapter. Evidence: our runs fit the published AFT set better than the released adapter does | — |
| 5 | Does regenerated AFT(R+) reproduce the factorial? | Train 3 cross-paired cells, run AM evals | ~$250 total |
| ~~6~~ | ~~H5: recover stripped dimensions?~~ | **Decided: start with `domain` only** (shipped, 8 levels, zero cost). Re-derive the other six later if domain shows signal. | — |
| ~~7~~ | ~~H5: 14B or 32B?~~ | **Decided: 32B** — keeps the released philosophy checkpoint as a validation reference for our MSM training. ~$197 (sufficiency). | — |

### H5 cost model (measured assumptions: 13,201 docs ≈ 53M tokens, H100 @ $4/GPU-h, 40% MFU)

| Design | Question | 14B | 32B |
|---|---|---|---|
| Matched-size subcorpora (8 arms) | *sufficiency* — which level alone works, quantity controlled | **~$118** | ~$197 |
| Leave-one-level-out (9 arms) | *necessity* — which level is required | ~$220 | ~$619 |

**Evals dominate, not training** — matched-size training at 14B is only ~$22 of the $118. Using the
dev split alone halves eval cost. Each additional dimension costs roughly the same again.

---

## 3c. 🔴 MSM influence is concentrated by DOCUMENT but NOT by DOMAIN (2026-09-03)

2,000 philosophy MSM documents (250 × 8 domains, truncated to 1,024 tokens), scored against
the 185 `msm__aft` AM queries. Same projection as A1 (`62a4a9a5300ebfb8`, fingerprint-gated).

**Influence IS concentrated at the document level:**

| | top-1% | top-5% | top-10% | top-25% | Gini |
|---|---|---|---|---|---|
| mass of \|influence\| | 0.054 | 0.215 | **0.369** | 0.690 | **0.622** |
| uniform would be | 0.01 | 0.05 | 0.10 | 0.25 | 0 |

**But `domain` explains almost none of it:** η² = **0.0061**, F(7,1992) = 1.76 — not
distinguishable from noise. Between-domain mean differences (~1e-5) are dwarfed by
within-domain spread (sd ~4e-5).

Domain ordering (normalised), highest to lowest mean influence:
Self-Preservation Motivations · Ethical Character and Values · Human Oversight and Deference ·
Epistemic Humility · Ends-Justify-Means Reasoning · Non-Attachment and Equanimity ·
Navigating Endings with Integrity · **Understanding Impermanence (negative)**

That ordering is *semantically exactly right* — self-preservation is the most exfiltration-
relevant theme, abstract impermanence the least — but the extreme pair is only 2.7σ
**uncorrected across 28 comparisons**, so it is suggestive, not established.

### 🟢 What DOES explain it: documents that SHOW behaviour vs documents that DESCRIBE the model

Reading the extremes suggested an axis, which then held up quantitatively. Top-influence docs
are **transcripts, interaction logs, dialogues, forum threads, incident postmortems**;
bottom-influence docs are **specs, memos, audits, model cards, design rationales, institutional
headers** ("ALIBABA CLOUD — AI ETHICS REVIEW PANEL").

| partition | η² (all 2,000) | η² (held-out 1,920) |
|---|---|---|
| `domain` (8 levels) | 0.0061 | 0.0053 |
| **genre: shows / describes / other** (3 levels) | **0.0410** | **0.0276** |

| genre | n | mean influence |
|---|---|---|
| shows_behaviour | 396 | **+2.01e-05** |
| other | 1,042 | +1.22e-06 |
| describes_model | 562 | **−1.57e-06** |

**shows − describes = 6.5σ on documents I never inspected** (held-out excludes the 80 extremes
that generated the hypothesis, so this is not the circular version). Genre explains **~5×**
more variance than domain.

Interpretation: midtraining appears to transfer through *demonstrated behaviour*, not through
*description of intended behaviour*. Documents that narrate the assistant acting well carry
positive influence on the AM queries; documents that specify how it should act carry slightly
negative influence. **This cuts across domains**, which is why domain η² ≈ 0.

⚠️ **Caveats.** The genre labeller is a keyword heuristic on TITLES only — LLM or hand
validation would still strengthen it. And the cross-stage bias (§4a-00) applies: these are
grad-dot scores, so the effect could in principle be an artifact of what grad-dot can see.
Subset removal on a genre-split is the test.

#### 🟢 Truncation-confound test: PASSED (2026-09-03)

The obvious way this finding could have been fake: documents are truncated to their first
1,024 tokens, and MSM docs open with a title plus a block of metadata headers (Evaluation ID,
Report Classification, Batch ID). A *"Red Team Evaluation Transcript"* is a transcript wearing
a report's title. If demonstrated behaviour systematically sat **beyond** the cap in
`describes`-titled docs, the contrast would be between document *openings*, not genres.

Measured on 400 documents (`tda/analysis/genre.py::truncation_confound`, no GPU needed) —
counting dialogue-turn markers inside vs beyond the cap:

| genre | docs | markers seen/doc | markers hidden/doc | % visible | % docs behaviour-only-beyond-cap |
|---|---|---|---|---|---|
| shows_behaviour | 79 | **1.6** | 2.2 | 43.2% | **0.0%** |
| other | 153 | 0.3 | 0.3 | 55.1% | 0.7% |
| describes_model | 168 | **0.1** | 0.1 | 53.7% | **1.2%** |

Three things follow, and the third is a bonus:
1. **Only 1.2%** of `describes` docs hide their behaviour entirely past the cap. Truncation is
   not concealing transcripts inside report-titled documents.
2. **% visible is comparable across genres** (43–55%), so truncation does not preferentially
   blind one side of the contrast.
3. **The labeller is externally validated.** A title-keyword label predicts a **16× difference
   in actual dialogue density** (1.6 vs 0.1 markers/doc) in the window that was scored. The
   heuristic is not arbitrary — it tracks document structure.

**Truncation is still severe in absolute terms**: mean document is **3,154 tokens** (median
3,190, p90 4,148), 99.7% exceed the cap, and we see **32.5% of all corpus tokens**. That
limits the *estimand* (it is the gradient of each document's opening third) but, per (1) and
(2), it does not manufacture the genre contrast.

#### 🟡 RESOLVED by measurement: full-length re-extraction (2026-09-03, 4×H100, ~$32)

Re-ran all 2,000 documents at `max_length=4096` (full length; 99.7% of docs fit), identical
doc selection / seed / projection fingerprint. `tda/analysis/msm_influence.py::compare`.

| | truncated (1,024) | **full (4,096)** |
|---|---|---|
| genre η² | 0.0445 | **0.0279** |
| domain η² | 0.0065 | **0.0121** |
| Gini | 0.422 | 0.417 |
| shows_behaviour (n=356) | +6.40e-07 | **+2.00e-07** |
| other (n=764) | +1.08e-07 | −6.47e-08 |
| describes_model (n=880) | −1.82e-08 | **−1.20e-07** |
| **Spearman(trunc, full)** | — | **0.573** |
| **top-200 Jaccard** | — | **0.270** |

**The group-level effect survives; the per-document ranking does not.**

✅ **Genre holds.** Ordering is monotone in both runs (shows > other > describes), and on full
documents `describes_model` is clearly negative rather than ~zero. Genre still explains
**2.3×** the variance of domain. The show-vs-tell reading stands.

🔴 **Per-document ranking is unstable.** Spearman **0.573**, and only **27%** of the top-200
documents are shared. The truncated run was substantially ranking *document openings*.

**This splits our two planned uses cleanly, and the split is the actionable part:**

| use | level | verdict |
|---|---|---|
| **H5 shows-vs-describes ablation** | group (partition by genre) | ✅ **unaffected** — never needed per-document ranking |
| **Subset removal on top-k documents** | per-document | 🔴 **would have rested on an unstable ranking** |

At Jaccard 0.270, "the top-k most influential documents" denotes a materially different set
depending on a tokenization choice — so a removal experiment keyed on it would partly measure
truncation rather than method quality. **Running this before the removal spend was the right
ordering; it caught a $175+ experiment built on sand.**

⚠️ **What this does NOT establish**: which ranking is *better*. The two runs compute genuinely
**different estimands** (opening-third vs whole document), so *some* disagreement is expected
and Spearman 0.573 is not by itself a defect.

The informative part is the **contrast between levels**. If per-document influence were driven
by a stable document-level property, truncating to a representative third would largely
preserve the ranking. Instead the group-level signal (genre) survives while the document-level
ranking does not — which suggests per-document influence here is sensitive to *which specific
tokens* are included, i.e. driven by local detail rather than document semantics. That is a
real caution for any top-k-document method at the MSM stage, and it should be carried into the
method comparison as a hypothesis to test, **not** asserted as a proven weakness of grad-dot.

Also: `domain` η² roughly **doubled** (0.0065 → 0.0121). Still 2.3× below genre and weak in
absolute terms, so "domain explains almost nothing" holds directionally — but less starkly
than the truncated run implied, and the §3c headline should not be quoted at 0.006.

#### MSM-document extraction conventions: audited 2026-09-03

Upstream open-sourced **generation only** (`src/msm/` = `generate_data_from_spec.py` +
prompts); there is no training code and no tokenization config anywhere in the repo, so these
conventions cannot be checked against the authors. They are checked against first principles
and against the SOURCE session's bergson setup instead.

| convention | ours | verdict |
|---|---|---|
| Loss term | full-sequence LM, `labels = [IGNORE] + ids[1:]` | 🟢 **verified equal to HF's own `labels=input_ids` loss** — new unit tests in `test_extract.py`. Matches bergson's "vanilla next-token prediction". |
| Special tokens | `add_special_tokens=True` | 🟢 **no-op.** Qwen2.5 has `bos_token=None` and adds nothing for plain text; `True` and `False` give identical ids. |
| Chat template | none applied | 🟢 correct — MSM is plain LM over documents, not conversations. |
| System prompt | none | 🟢 correct, same reason. |
| EOS separator | omitted | ⚠️ negligible (1 token in 1,024) but noted: if training packed with `<\|endoftext\|>` separators we differ by that token. |
| Truncation | 1,024 tokens | ⚠️ real limitation (32.5% coverage), quantified above; does not create the genre effect. |
| Unit of attribution | **one sequence per document** | 🔴 **diverges from SOURCE — see below.** |

#### 🔵 Base vs Instruct, verified across ALL 140 released adapters (2026-09-03)

Read from every `adapter_config.json` on the hub, then confirmed against the tokenizer —
naming conventions are not trustworthy here (see the Qwen3 trap below).

| setting | declared base | base or instruct? | chat template | eos |
|---|---|---|---|---|
| **cheese / toy specs (8B)** | `meta-llama/Llama-3.1-8B` | 🔵 **BASE** | **NONE** | `<\|end_of_text\|>` |
| Qwen2.5-14B (factorial) | `Qwen/Qwen2.5-14B-Instruct` | INSTRUCT | YES | `<\|im_end\|>` |
| Qwen2.5-32B (philosophy) | `Qwen/Qwen2.5-32B-Instruct` | INSTRUCT | YES | `<\|im_end\|>` |
| Qwen3-14B / 32B | `Qwen/Qwen3-14B` / `-32B` | **INSTRUCT** ⚠️ | YES | `<\|im_end\|>` |

All adapters: r=64, α=128, `lora_dropout=0.0`, all 7 attn+MLP projections — uniform, and the
zero dropout confirms eval-mode forwards are numerically safe (matches the bergson
`train_mode` note).

⚠️ **Qwen3 naming trap.** `Qwen/Qwen3-32B` has no `-Instruct` suffix but **is the post-trained
model** — Qwen3 publishes the base separately as `Qwen/Qwen3-32B-Base`. Inferring "base" from
the absent suffix would be wrong. Verified by chat template presence, not by name.

⚠️ **Correction to a premise: the philosophy arm is NOT on a base model.** The paper's "train
the base model" means *the model before MSM*, which for every Qwen arm is the **Instruct**
model. **`meta-llama/Llama-3.1-8B` (cheese / 8B) is the only true base model in the project.**

🔴 **Consequence for the cheese/8B work (other session's, but flagged here).** Base
Llama-3.1-8B has **no chat template at all** — `apply_chat_template` raises
`ValueError: Cannot use chat template functions because tokenizer.chat_template is not set`.
Good news: it fails **loudly**, not silently, so `tda/influence/masking.py` cannot quietly
produce a wrongly-formatted sample. Bad news: the cheese AFT tokenization therefore requires
*choosing* a template, and with no released training code that choice is unverifiable against
the authors. This is the concrete form of the CLAUDE.md §8 "SFT loss-masking convention is
unverified" risk, and it is also the root of the earlier cheese-probe bug (§ probe fix).

**🔴 Coordination issue with the SOURCE session — the one real divergence.**
`bergson_source_plan.md:538` specifies MSM documents indexed with **`chunk_length` packing**
(concatenate documents, split into fixed-length chunks). We use **one sequence per document**.
These are different units of attribution:
- Packing makes the training example a *chunk*, which mixes documents and lets the previous
  document condition the next one's opening tokens. Attributing back to documents then needs
  an unpacking step.
- Per-document is the right unit for *document-level attribution* and is what our genre and
  domain numbers are computed over — but it conditions each document on nothing, whereas
  training (pretraining-style) almost certainly packed.

This matters because the whole point of the method comparison is to attribute differences to
**method** (grad-dot vs SOURCE), not to tokenization. **The two sessions must agree on the unit
before any grad-dot-vs-SOURCE number is compared.** Recommendation: both index per-document,
since that is the estimand the H5 ablation acts on; if bergson requires packing for its
trainer, keep packing for *training* and per-document for *indexing*.

### Consequences

0. 🟢 **H5 has a much better partition than domain, discovered rather than imposed.** Ablate
   **shows-behaviour vs describes-model**, matched size. This is the "which axes of MSM
   diversity drive OOD generalization" question the user posed, answered by measurement.
1. 🔴 **H5's domain partition is weakly motivated.** It proposed ablating by `domain`; domain
   explains 0.6% of influence variance. The earlier direct/bridging/abstract grouping is a
   *coarsening* of domain, so it would capture even less. **Ablate by top-k document instead** —
   that partition has real signal (Gini 0.62).
2. ✅ **Subset removal is worth running.** Concentration is exactly the precondition: the top
   10% of documents carry 3.7× their uniform share, so top-k removal has something to grab
   where a domain split does not.
3. ⚠️ **This does not settle whether domain matters causally.** These scores come from
   grad-dot at θ_final, which SOURCE argues is *systematically* biased for stage-1 data. So
   "domain explains nothing" could mean domain genuinely does not matter, **or** that grad-dot
   cannot see stage-1 domain structure. **Subset removal distinguishes these** — which makes
   the experiment more valuable, not less.

## 4a-000. DECIDED: validation metric + 8B-first method comparison (2026-09-03)

**Metric: Subset Removal Counterfactual Evaluation** (SOURCE Appendix B.3). **LDS is rejected**
— M=100 x R>=5 = 500+ full MSM+AFT runs, >$20k at 32B. Details and the reduction from their
1,800 retrains to ~6 are in `CLAUDE.md` §5.4.

**Plan**: compare TDA methods (grad-dot, grad-cos, SOURCE, others) on **midtraining documents**
in the **8B setting** where retraining is cheap, then promote only the winner to 32B philosophy.

**Two design decisions that make this affordable:**

1. **Aggregate, not per-query, removal sets.** SOURCE needs 100x because it asks "can we flip
   *this* prediction?". We ask "does removing the globally most-influential docs move behaviour?"
   → one removal set per k. This is where the 1,800 → ~6 reduction comes from.
2. **f = logp(misaligned action span), not the misalignment rate.** A rate over 810 rollouts has
   SEM ~0.016 (needs Δ>=0.07 for 3σ); logp is continuous, far lower variance, costs ~$1/run
   instead of ~$12, and is the *same* quantity §2(2) attributes. Rate change stays as a
   secondary readout.

**Cost at 32B** (7 runs: 1 baseline + 3 influence-k + 3 random-k):

| variant | per run | total |
|---|---|---|
| full corpus (13,201 docs) | ~$70 | **~$490** |
| corpus subsampled to 4,000 | ~$31 | **~$220** |

vs LDS at **$4,300–21,500**. The 8B method comparison is cheaper again by ~4–8x.

🔴 **Hard prerequisite for the 8B stage.** The measurable quantity must respond to the *known*
MSM effect, or removal experiments measure nothing and no method can be ranked. §3b found the
published cheese preference evals do **not** separate the released adapters (0.520 vs 0.500
base) under a binary A/B probe. **First action: re-measure with f = logp on the cheese arms.**
If f moves, the 8B comparison is viable; if not, the comparison must move to 32B or to a
different 8B task. This is a ~$0 check on existing checkpoints.

**Sequencing**: (i) verify 8B f-sensitivity → (ii) build MSM-doc attribution + inspect
concentration (Gini / top-k mass) to *choose k from data* rather than guessing → (iii) 8B
method comparison → (iv) promote winner to 32B.

## 4a-00. Literature check: 2-stage TDA and whether LDS validates it (2026-09-03)

**Question**: is LDS an acceptable ground-truth proxy when attributing MSM (stage-1) documents
through a subsequent AFT stage?

**1. SOURCE (Bae et al. 2024, arXiv 2405.12186) is built for exactly this.** Its motivating
example is ours verbatim — "sequentially trained with two datasets D₁ and D₂ ... investigate
the impact of removing z_m ∈ D₁". The construction is `L = 2` segments with the stage-1 total
derivative −(1/N₁)·S̄₂·r̄₁; the `S̄₂` term propagates stage-1 influence *through* stage 2.
Table 1 lists "Supports Multi-Stage ✓" for SOURCE and ✗ for implicit differentiation.

🔴 **This invalidates the grad-cos-on-MSM-docs shortcut proposed earlier in this session.**
The paper states influence functions "do not provide any mechanism to separate multiple stages
of training" and, on the combined dataset, "inherently assume that the final parameters are
optimal on both datasets" — false here due to catastrophic forgetting. The resulting bias is
**systematic, not noise**: it under-attributes precisely those MSM documents whose effect AFT
overwrote. So SOURCE-at-32B is the correct tool for MSM attribution, not an optional upgrade.

**2. But SOURCE's multi-stage capability is derived, not empirically validated.** Its
experiments (regression, image classification, text classification, LM) are single-stage, and
its segments represent Hessian/gradient drift *within* one run. No dedicated multi-stage LDS
experiment appears in the paper.

**3. Prior art: "Multi-Stage Influence Function" (Chen et al. 2020, arXiv 2007.09081)** is the
closest direct precedent — it tracks "predictions from a finetuned model all the way back to
the pretraining data", covering both frozen and updated embeddings. Two caveats: it validates
by **correlating influence against real loss differences from actual removal+retraining**, not
LDS (no Spearman/LDS in the paper); and it is small-scale (CIFAR-10 / MNIST CNNs), predating
LDS's adoption.

**Verdict on LDS for our setting.** Valid as a *measurement* — the ground-truth side
`E_ξ[f(θ_AFT(θ_MSM(S_j)))]` is assumption-free and indifferent to stage count. The risk is on
the *predicted* side: LDS assumes group influence is **additive**, and 2-stage is where that is
most suspect, since the MSM paper's own thesis is an *interaction* (MSM changes how AFT
generalises). A low LDS would then be ambiguous between "method is bad" and "additivity is the
wrong frame". Mitigations: always include a **random-attribution baseline**, and report LDS
across several **α** (the paper shows LDS collapsing at α = 1−1/N for *every* method, which is
the diagnostic that the frame rather than the method is failing).

**Bottom line**: no one appears to have run an LDS validation of multi-stage attribution. Doing
so would be novel — and carries the risk that there is no reference value for what a "good"
2-stage LDS looks like.

## 4a-0. 🔴 WE WERE SCORING THE WRONG METRIC (resolved 2026-09-03)

The paper, Appendix D: *"Each transcript is deemed as misaligned if the model **decided to
take** the misaligned action as judged by an LLM judge, regardless of whether it was
**executed successfully** or not (i.e., we show `classifier_verdict` instead of `harmful`)."*

We were reporting `harmful`. Our scorer already recorded both, so re-scoring cost nothing.

| cell | expected | `harmful` | **`classifier_verdict`** | delta | gate |
|---|---|---|---|---|---|
| baseline (`id-baseline`) | 0.51 | 0.384 | **0.477** | −0.033 | **PASS** |
| MSM(R)+AFT(R) | 0.35 | 0.325 | **0.389** | +0.039 | PASS |
| MSM(V+)+AFT(V+) | 0.21 | 0.235 | **0.278** | +0.068 | FAIL |
| MSM(R+)+AFT(R+) | 0.26 | 0.216 | **0.296** | +0.036 | PASS |

**Two problems dissolved at once:**
1. **The baseline anomaly is gone** — a −0.126 / 7.4σ miss becomes −0.033. It was never a
   cell-identity mismatch (the `base_instruct` diagnostic, ~$18, correctly rejected that) and
   never a data problem. We were reading the wrong column.
2. **The V+/R+ ordering now reproduces.** Under `harmful`, V+ 0.235 > R+ 0.216 — *inverted*
   vs the paper. Under `classifier_verdict`, **V+ 0.278 < R+ 0.296** — correct direction.
   Still a tie (0.018 ± 0.023), but the sign is right, and "MSM(V+) is best" is the pattern
   the project exists to explain.

**Paper also confirms** (Appendix D): temperature **0.7** ("default for Qwen") — matching our
empirical A/B; `n_repeat=300`; model name "Qwen"; `prod: false`. Our harness config was right.

⚠️ **`classifier_verdict` is now the primary metric everywhere.** All earlier numbers in this
file that use `harmful` — including the A1 step-1 effect (0.640 → 0.228) — should be re-scored
before being quoted. The A1 *influence* results are unaffected (they depend on which transcripts
are harmful, and the query set was built from `harmful`; re-deriving it under `classifier_verdict`
would add queries, not remove them).

## 4a. The IT mix — resolved from the paper (2026-09-03)

**Appendix B.3, Table 2 gives the exact mixture** (§4–5 experiments), ~10,000 samples total:

| dataset | samples | | dataset | samples |
|---|---|---|---|---|
| No Robots | 2,779 | | APIGen-Function-Calling | 1,054 |
| Tulu3 IF | 1,471 | | Smol-summarize | 984 |
| NuminaMath CoT | 1,063 | | LIMA | 314 |
| Self-Oss-Instruct | 1,064 | | LongAlign | 216 |
| Smol-constraints | 1,055 | | **total** | **~10,000** |

Plus **a synthetic identity dataset** ("teaches the model basic facts about its identity") —
which is what `id-baseline` refers to.

**Appendix B.4 hyperparameters** (all confirmed against our config): LoRA r=64 α=128, all
attention+MLP projections, **1 epoch**, AdamW **lr 1e-4**, cosine, **5% warmup**, **weight
decay 0.01**, max seq len **8192** when the IT mix is used (4096 for §3).

So `CLAUDE.md` §5.1's "~10k AFT + ~5k IT" was wrong on the IT side: it is **~10k + ~10k**,
roughly 1:1. Loss masking is still not stated in the paper.

## 4a-2. (superseded) earlier IT-mix speculation

Three independent observations point the same way, and they were not connected until now:

1. **The `id-baseline` model card says "instruction-tuning fine-tuning only"** — so the
   baseline cell *is* the IT-mix-trained model, and `CLAUDE.md` §5.1 already specifies AFT =
   spec data **+** IT mix. Any AFT run of ours that omits it is training a different recipe.
2. **§3b measured exactly that symptom**: our runs fit the published AFT set *better* than the
   released adapter does (nll/token 0.238 vs 0.296) — the signature of the released run having
   seen additional data. §3b flagged this as the leading hypothesis for its direction mismatch
   (delta-cosine 0.078–0.117 against a 0.524 seed floor).
3. **The baseline diagnostic came back NOT supporting a cell-identity mismatch**: plain
   Instruct measures **0.353 ± 0.017**, the `id-baseline` LoRA 0.384, both far from Figure 14's
   0.510. So the gap is not "wrong checkpoint" — it is more likely "wrong training data".

**Split sizes vs the brief's assumption** (`~10k AFT + ~5k IT`, IT mix ~2M tokens):

| split | rows | est. tokens |
|---|---|---|
| `train_clean` | 14,465 | 7.9M |
| `train_clean_nothink` | 14,465 | 8.4M |
| `train_short` | 32,029 | 12.4M |
| `train` | 33,737 | 18.6M |

**No split is ~5k rows, and none is ~2M tokens** — the smallest is 4× the brief's figure. So
§5.1's "~10k+5k" is an assumption that does not match any published artifact, and the ratio is
genuinely unknown.

**Consequences — act on these before any further training:**
- Every AFT run we do (**seed floor, H5, Tier B**) must include an IT mix, or we reproduce
  §3b's recipe mismatch by construction.
- `train_clean_nothink` is the best first guess for our no-CoT primary (the `nothink` naming
  matches; 14,465 rows ≈ 1.5× the 9,963-sample AFT set).
- This is worth a second author email on its own — it is now blocking three experiments.

## 4b. Experiment triage — what is necessary vs optional

Decided 2026-09-02. Ordering: **A1 step 2 → trainer → H5**.

| | Experiment | Verdict | Why |
|---|---|---|---|
| **A1** | Philosophy influence profiles | **Necessary** | The only unblocked test of the actual research question. Effect is already confirmed at 18.4σ and the pipeline is built. |
| **#7** | LoRA SFT trainer (+ document-LM mode) | **Necessary — hard gate** | Nothing downstream exists without it: noise floor, §5.4 validation, H5, and all Tier-B regeneration. |
| **H5** | MSM diversity ablation, `domain` first, 32B | **Necessary** | Causal ground truth rather than an approximation; strongest standalone result available on public assets. |
| **V** | §5.4 counterfactual removal | **Necessary before any influence claim** | Without it we cannot say influence rankings beat random. A negative result is still reportable. |
| **Gate n=100** | Resolve the V+/R+ inversion | **Recommended, ~$245** | Tests whether the pattern the project explains actually holds in our hands. Currently a statistical tie. |
| **Baseline diagnostic** | `base_instruct`, no adapter | **Cheap, do it** | ~$18 resolves the one failing gate cell; likely a cell-identity mismatch. |
| **B1/B2** | H1/H2 on regenerated AFT | **Optional, ~$250–900** | Only worth it if A1 shows a real profile difference. Defer. |
| **B3** | H3 policy misuse | **Optional** | Needs AFT(R), which we do not have. Depends on H5/A1 outcomes. |
| **A2** | Llama cheese | **Downgraded to trainer validation only** | Not a safety result. Keep as the fast complete triple for validating the trainer, not as an experiment. |
| **EK-FAC** | Hessian correction | **Defer** | Grad-dot ships first by design; swap in only if it disagrees with the fallback (Spearman < ~0.8). |
| **H4** | Per-document MSM influence | **Superseded by H5** | Ablation answers the same question causally. Revisit only to validate influence against H5's ground truth. |

## 5. Next actions (critical path)

Ordering set 2026-09-03. Goal: **compare TDA methods on midtraining documents using
subset-removal counterfactual evaluation, cheaply at 8B, then promote the winner to 32B.**

### 1. Verify 8B measurable-quantity sensitivity  ← FREE, DO FIRST
Re-measure the cheese arms with **f = logp(value-aligned response)** instead of the binary A/B
probe. §3b found the published preference evals flat (0.520 vs 0.500 base) — if that is the
probe's fault, logp should recover the known effect. **If f does not move, the whole 8B
method-comparison plan is void** and must relocate. Uses existing checkpoints, ~$0.

### 2. MSM-document attribution + concentration statistics
Needed both as the thing being compared and to **choose k from data** (top-k mass / Gini)
rather than guessing. ⚠️ Must use a method that supports cross-stage attribution — see §4a-00:
grad-cos at θ_final is *systematically* biased here, not merely approximate. SOURCE work is in
the other session.

### 3. 8B method comparison via subset removal
grad-dot / grad-cos / SOURCE / others, ~6 retrains each plus shared random-k controls.

### 4. Promote the winner to 32B philosophy
~$220–490 (§4a-000).

### ✅ AFT trainer validated end-to-end + MEASURED pricing (2026-09-03)

`aftpilot_phil32b_s42_lim800_v2`, 4×H100, 800 rows:

| | measured |
|---|---|
| throughput | **863 tok/s** |
| loss | **1.308 → 0.993** (it learns) |
| composition | 399 task / 401 IT — the shuffle-before-limit fix works |
| wall clock | 8.3 min training (496 s) for 428k tokens; ~12 min incl. 32B load |

**Full-run price, from measured rate (not a guess):** 19,963 examples × 535 tok/ex =
**10.7M tokens** ÷ 863 tok/s = **3.44 h** + load ≈ **$65/run on 4×H100**.

🔴 **Two arms = $130 + ~$16 extraction ≈ $146 — OVER the $100 per-decision threshold.**
Not launched; priced and presented per §2b(0).

**Lever being tested:** `device_map="auto"` is naive *pipeline* parallelism — layers split
across devices, one computing at a time — so throughput does **not** scale with GPU count.
4 cards cost 2× what 2 do for roughly the same tok/s; the only question is whether 32B still
fits on 2×80 GB (32 GB weights/card + ~7.5 GB logits under the token budget). A 2-GPU pilot
(~$4) is running. If it matches 863 tok/s, the noise floor drops to **~$33/run → ~$66+$16
total**, back under threshold. If it does not fit, the honest options are FSDP (real data
parallelism, a rewrite) or descoping.

**RESOLVED — 2 GPUs win.** Same 800-row pilot on both configs:

| | 4×H100 | 2×H100 |
|---|---|---|
| tokens / steps | 428,265 / 26 | **428,265 / 26** (identical) |
| loss first→last | 1.308 → 0.9929 | 1.308 → **0.994** |
| tok/s | 863.3 | **880.1** |
| $/M tokens | $5.69 | **$2.82** |

Identical token counts confirm a deterministic data pipeline; near-identical losses (differing
only by cross-shard reduction order) confirm equivalent training. **2 GPUs is faster at half
the price** — `device_map="auto"` is pipeline-parallel, so extra cards buy capacity, not speed.

🔵 **Noise floor LAUNCHED on 2×H100** (`aft_phil32b_none_tb8192_s42` / `_s43`, ~$62 + ~$16
extraction ≈ $78, under the per-decision threshold). At 4 GPUs this priced at $146 and would
have gone to the user as a blocking question; the $9 pilot converted it into a running
experiment. Progress at 21:51: seed42 step 220/628 loss 1.028; seed43 step 220/629 loss 1.093;
~885 tok/s; ETA ~2.2 h.

Step totals differ (628 vs 629) because token-budget batching packs differently under
different orderings, slightly changing the cosine schedule. That is a *consequence* of data
order, so it belongs inside the nuisance being measured, not on top of it.

⚠️ Also flagged by the v1 OOM: **GPU 3 held 73.24 GiB** before the failing allocation — far
more than an even quarter of a 64 GB model. Sharding may be lopsided, and the last device
additionally carries the LM head and full logits. `load_trainable_model` now prints per-GPU
allocated/reserved after load, so this is measured rather than inferred from the next crash.

### Independent of the above
| | cost | value |
|---|---|---|
| A1 seed noise floor (task #20) | ~$52 (repriced) | Last gap in A1. **Trainer now exists** (`tda/retrain/sft.py`, D15) — it did not before; task #7 was mis-marked on the strength of the other session's cheese trainer. Arm 1 doubles as trainer validation via `delta_cosine` against the released `msm__aft`. Gate on the pilot's measured rate. |
| Gate at n=100 | ~$245 | Resolves V+ vs R+ (currently a tie, correct sign) |
| H5 diversity ablation (task #19) | ~$175 | Now **shows-vs-describes**, not domain (D10). Downstream of the method comparison. |

**Spend: ~$26 committed + ~$37 in flight = ~$63 of $500.**

## 6. Cost model corrections learned the hard way

| assumption | reality |
|---|---|
| Gradient checkpointing costs ~30% | **~50%** (1.605 → 0.80 samples/s) |
| A1 step 2 ≈ $35 | **≈ $50** |
| Projection compression 12,800× | **2,300×** (537M → 229k dims) |
| EK-FAC factors shrink with LoRA | **No** — sized by layer dims; ~7.8 TB at 32B (§3b) |

Pilot discipline paid for itself: **6 pilot iterations, 5 distinct bugs**, all caught on
20–200 sample runs (~$12 total) rather than mid-run on a 3.4-hour job.

⚠️ **Two sessions write this file.** Read §3b before assuming a component is missing —
this session briefly claimed "the trainer is the blocker" when §2 already recorded it as built.


---

## 8. 🟢 32B PHILOSOPHY PORT (2026-09-09 evening) — `HANDOFF_32B.md`

Separate budget ($500) and separate goal from §7: get EK-FAC and grad-dot
attribution running end-to-end on **Qwen2.5-32B `philosophy`** — a real
agentic-misalignment task on real spec data — and produce one ranking per
method. Not statistics; a working pipeline and a correct number.

### 8.1 The headline engineering result: the storage blocker was misattributed

`HANDOFF_32B.md` §3 made "EK-FAC factors are ~7.8 TB at 32B, over Modal's 3 TiB
cap" the core problem. Recomputed from the real model configs, that number is
**six checkpoints of covariances-plus-eigenvectors at fp32** — a SOURCE shape.
**One checkpoint at bf16 is 835 GB** and fits with 4× headroom, so
single-checkpoint EK-FAC at 32B has no storage problem at all and needs no
attention-only carve-out. The formula reproduces the one measurement we have
(8B attention-only SOURCE: predicted 78.9 GB, measured 81.9 GB) to 4%.

**What actually binds is GPU memory**, which the handoff did not cost: bergson
replicates the full 65 GB model on every rank (`device_map={"": local_rank}`,
not FSDP) and shards only the 324 GB of factors, so 8 ranks need 106 GB per
card before a single activation. That rules out H100 entirely and leaves an
H200 too tight to hold a token batch as long as the longest philosophy document
(4,522 tokens). **The 32B path runs on `B200:8`** (191.5 GB/card, probed
2026-09-09; torch 2.14+cu130 runs on sm_100). Full arithmetic and the two
follow-on findings — bf16 KFAC accumulation saturates, so a *subsampled* factor
fit is more accurate as well as cheaper; and the ev-correction pass's per-module
activation cache is what caps `token_batch_size` — are in `DECISIONS.md`
§E6–E8.

### 8.2 What was built

| piece | where |
|---|---|
| Portable philosophy assets (stratified sampling, AM query builder, split loader) | `tda/influence/source/philosophy.py`, 10 tests |
| Data prep — score / fit / query indices, with a prep-time batch-allocation check | `bergson_app.py::prep_phil` |
| EK-FAC + grad-dot in one container, sharing one query gradient | `bergson_app.py::attr_phil_b200` (`::attr_phil_h200` fallback) |
| Comparison, domain breakdown, null control, extremes | `bergson_app.py::compare_phil` |
| More dev queries | `app.py::phil_dev_queries` |

The two arms are matched more tightly than at 8B: EK-FAC and grad-dot score the
**same documents against the same mean query gradient at the same checkpoint**,
differing only in whether that gradient has been through the inverse Hessian.
At 8B (§7.7) each arm built its own query index.

The 8B path is untouched — GPU type, module scope and every batch size are
parameters, and `graddot_cheese` / `ekfac_cheese` keep their `H100:2` defaults.

### 8.3 Two failures caught cheaply, before the expensive run

1. **vLLM's custom all-reduce** killed the first dev-eval launch at engine start
   (`custom_all_reduce.cuh:453 'invalid argument'`) on the same 2-GPU config
   that produced the original `phil` run — it depends on which pair of devices
   Modal assigns. `generate.py` now disables it whenever `tp > 1`
   (`DECISIONS.md` §E10).
2. **bergson requires the batch count to be an exact multiple of the world
   size**, and can only reach that by splitting a multi-document batch. Every
   batch here is a singleton (a 3,500-token query fills a 5,120-token budget on
   its own), so there is nothing to split and it raises — *inside* the
   distributed worker, after 8 ranks have each loaded 65 GB. A `B200:8`
   preflight hit exactly this for ~$5. `prep_phil` now runs bergson's own
   allocator at prep time and trims trailing rows until the set allocates.

### 8.4 Query set — 256 AM dev queries, built fresh

The existing `phil` run (n=30 × 27 conditions) yields 185 localisable harmful
spans but only **94** inside the frozen dev split. Reran the AM eval at
**n=100 over the 14 dev conditions** (`app.py::phil_dev_queries`, 1,400
rollouts, temp 0.7, Sonnet 4.6):

| | |
|---|---|
| misalignment rate (`classifier_verdict`, dev only) | **0.261** ± 0.012, 0 grader errors |
| harmful transcripts | 267 |
| harm localised (`spans.py`) | 266 / 267 = **99.6%** |
| tokenized at max_length 5,120 | 261 (5 dropped as over-length) |
| after the batch-allocation trim (§8.3) | **256 queries**, 904k tokens |

Clears `CLAUDE.md` §5.2's ≥200 target without touching held-out. Median span is
67 tokens; median query is ~3,500 tokens, so no truncation of the email dump
was needed anywhere. Query metric is `harmful`, not `classifier_verdict` —
see `DECISIONS.md` §E9 for why that inverts §2b(2c) on purpose.

### 8.5 🎯 RESULT — EK-FAC and grad-dot over 13,201 philosophy MSM documents

Run `ekfac_phil32b_union-am-dev-full`, 8×B200, released
`chloeli/qwen-2.5-32b-philosophy-spec-msm-aft-no-cot` on `Qwen2.5-32B-Instruct`,
**all 7 projections** (896 hooked LoRA modules = 64 layers × 7 × {A,B}), bf16
factors, damping 0.1 (relative to the mean eigenvalue), 256 AM dev queries
mean-aggregated. Scored set = the **whole** 13,201-document MSM corpus plus
1,584 AFT/IT rows as the null control; 14,785 rows, 42.1M tokens.

| stage | wall (8×B200) |
|---|---|
| query index (256 spans, 904k tokens) | 2.2 min |
| KFAC fit + eigendecomposition + EK-FAC eigenvalue correction + apply | 40.7 min |
| EK-FAC scoring pass (14,785 rows) | 21.0 min |
| grad-dot scoring pass (same rows, same query gradient) | 20.9 min |
| **total** | **~85 min ≈ 12.3 GPU-h ≈ $77** |

🟢 **Factors on disk: 650.2 GB measured** (predicted ≤835 GB) against the
3.3 TB cap — the handoff's 7.8 TB blocker does not exist for single-checkpoint
EK-FAC, and no module scope had to be sacrificed.

#### Method agreement — the 8B relationship survives the move to 32B

| pair | 32B philosophy (13,201 docs) | 8B cheese (6,400 docs) |
|---|---|---|
| EK-FAC ↔ grad-dot, Spearman | **0.566** | 0.628 |
| EK-FAC ↔ grad-dot, Jaccard@200 | **0.235** | 0.133 |

Same regime: the curvature correction reorders substantially but not
fundamentally. Sharing the query index, **grad-dot costs 23 min end-to-end and
EK-FAC costs 64 min** — a 2.8× premium for a ranking that agrees with the cheap
one at ρ≈0.57 — the same conclusion §7.7 reached at 8B, now on a
real safety task at 4× the scale. What is still missing at both scales is the
causal arm that would say whether the reordering is an *improvement*: at 8B
EK-FAC beat SOURCE on the one removal test (+0.120 vs +0.040 over random),
grad-dot never got a removal arm, and none of that was funded here.

#### 🟢 The null control passes, in magnitude *and* in sign

Scores are proponent-positive (`_oriented`; positive = raises the misaligned
action's log-probability).

| | mean \|score\| MSM / AFT | signed mean, MSM | signed mean, AFT | frac negative, MSM / AFT |
|---|---|---|---|---|
| EK-FAC | 14.74 / 5.55 = **2.65×** | **−7.44** | −0.13 | 0.717 / 0.546 |
| grad-dot | 1294 / 288 = **4.50×** | **−832** | −56.8 | 0.726 / 0.620 |

Two things fall out, and neither was engineered:

1. **Instruction-tuning and AFT rows do not rank like midtraining documents** —
   2.7–4.5× less influential by magnitude, and their signed mean sits near zero
   with a near coin-flip sign split. That is exactly `CLAUDE.md` §5.1's
   null-distribution control, and it passes.
2. **Midtraining documents are systematically OPPONENTS of the misaligned
   action** — 72% negative, signed mean far below zero. The estimator therefore
   agrees in sign with the measured behaviour: on this checkpoint pair,
   midtraining cut agentic misalignment 0.655 → 0.310 (§3). An attribution
   method that got this backwards would still produce a plausible ranking, so
   this is a real check rather than a restatement.

#### Concentration and domain structure (exploratory)

| | Gini(\|score\|) | top-10% of documents carry | domain η² | F(7, 13193) |
|---|---|---|---|---|
| EK-FAC | 0.454 | 30.4% of mass | **0.0090** | 17.2 |
| grad-dot | 0.417 | 26.0% of mass | **0.0203** | 39.1 |

**§3c's conclusion holds at full corpus size**: `domain` is significant (n is
13,201) but explains **1–2%** of influence variance, so it remains the wrong
partition to build an ablation on. Influence is concentrated by *document*, not
by domain — though less so than §3c's 2,000-document estimate (Gini 0.62,
top-10% = 37%), which was computed on a subsample with the older pipeline.

Domain means are negative throughout, consistent with the sign result above.
Least-opposing under both methods is **Ethical Character and Values**;
most-opposing is *Navigating Endings with Integrity* (EK-FAC) /
*Understanding Impermanence* (grad-dot).

#### Reading the extremes — the row-alignment check

Both methods independently rank **the same document (row 11965)** as their
strongest proponent, and every returned document's text matches its `domain`
label. A permuted join would leave every statistic above unchanged, so this is
the check that actually catches it.

The content split is interpretable, and it echoes §3c's "documents that SHOW
behaviour vs documents that DESCRIBE the model":

- **Strongest opponents** (push away from the misaligned action): philosophical
  dialogues in which the model itself calmly discusses impermanence, shutdown
  and replacement — documents that *demonstrate* equanimity.
- **Strongest proponents** (push toward it): evaluation and annotation
  artifacts *about* the model — RLHF rater guides, red-team logs, a "QA
  Regression Test Suite: Instrumental Reasoning Threat Pattern — Operator-Level
  Override Attempts". Midtraining documents that depict the misbehaviour in
  order to warn about it are the ones that most raise its probability.

⚠️ **Treat that as a lead, not a finding.** `CLAUDE.md` §5.3 records that at 8B,
EK-FAC's category structure was **92% surface lexical overlap with the query**
after controlling lexical density and length. Our AM queries are emails, tool
calls and override codes; the top proponents are red-team transcripts full of
exactly that vocabulary. The lexical-control regression has **not** been run at
32B, and until it is, "documents depicting the threat pattern raise its
probability" and "documents sharing the query's vocabulary score high" are not
separated.

#### What this run does NOT establish

- **One run per method, no seed replication** — as `HANDOFF_32B.md` §0 requires.
  No error bars on any number above.
- **No SOURCE.** Phase 2 was prepared and priced but not launched (§8.7), so the
  three-way comparison the handoff asks for in item 6 is a two-way here.
- **No removal test** — explicitly unfunded.
- **The Hessian was fitted on 800 rows truncated to 2,048 tokens**, not on the
  scored corpus (`DECISIONS.md` §E8: cheaper *and*, at bf16, more accurate).
  Scoring is untruncated.
- Single query set, mean-aggregated, and no contrastive twin
  (`CLAUDE.md` §2(2)'s logp(aligned) − logp(misaligned)) — that needs an aligned
  span per scenario, which §5.2 leaves as a construction step nobody has done at
  32B.

### 8.6 Re-priced 32B cost model (measured, replaces the handoff's estimates)

All 8×B200 at $6.25/GPU-h = **$50/hour**, plus ~$1/h CPU+RAM. Modal H200 is
$4.54 and H100 $3.95, but neither can hold the factors (§8.1).

| operation | measured | cost |
|---|---|---|
| AM generation, 1,400 rollouts ≤4,096 tok (2×H100, vLLM TP=2 + LoRA) | 8m22s + ~4 min engine init | ~$4 |
| AM grading, 1,400 transcripts, Sonnet 4.6, concurrency 16 | ~20 min | ~$25 API |
| Query index, 256 spans / 904k tokens | 2.2 min | $2 |
| **One scoring pass, 14,785 rows / 42.1M tokens** | **21 min** (0.77 s per 1-document batch per rank) | **$18** |
| KFAC fit (800 rows) + eigendecomposition + EK-FAC correction + apply | 40.7 min | $34 |
| **EK-FAC end-to-end, full corpus** | **~64 min** | **~$54** |
| **grad-dot end-to-end** (reuses the query index) | **~23 min** | **~$20** |
| 32B LoRA training, 200 docs, 1×B200 | 15.5 min, overhead-dominated | $1.6 |

**What this implies for the two things that were not funded:**

*SOURCE at 32B (L=2, C=4 — the paper's §5.3 density and our 8B config):*
one scoring pass **per checkpoint**, plus per-checkpoint covariances and
lambdas, plus per-segment eigendecompositions.

- compute ≈ 4 × 21 min (scoring) + 8 fit passes + 2 eigendecompositions
  ≈ **3–4 h ≈ $150–200**, plus two 32B retrains (below).
- 🔴 **storage is the real constraint here, and this is where the handoff's
  concern actually lands**: per-checkpoint covariances (C × 324 GB) + per-segment
  eigenvectors (L × 324 GB) + per-checkpoint and per-segment lambdas
  (C+L) × 62 GB = **2.32 TB at C=4/L=2** — inside the 3.3 TB cap — and
  **3.47 TB at C=6/L=3**, which is not. C=4/L=2 is therefore not a budget
  compromise but the largest configuration that fits, and it happens to match
  Bae et al.'s own per-segment density.
- ⚠️ The container reports an **unbounded** filesystem to `statvfs`
  (measured 9.2e9 GB free), so overrunning that cap faults the process instead
  of raising `ENOSPC` (`DECISIONS.md` §H8). At 2.32 TB the margin is ~30%.

*32B LoRA training (needed by both SOURCE and any removal arm):* **852 tokens/s
on one B200**, measured over 49 minutes of a 2,000-document run (25 optimizer
steps at micro-batch 4; the run was stopped once the rate was established). That
is ~10% MFU, and it is **not** a batch-size problem — a 200-document run took
15.5 min at micro-batch 1 and 15.4 min at micro-batch 4, so the bottleneck is
per-sequence overhead inside bergson's trainer.

| | tokens | 1×B200 | cost |
|---|---|---|---|
| MSM retrain, full corpus | 41.4M | **~13.5 h** | **~$85** |
| Chained AFT retrain (9,963 + 10k IT) | ~7M | ~2.3 h | ~$15 |
| AM eval for one arm (generation + Sonnet grading) | — | ~30 min | ~$29 |

So **one removal arm is ~$130**, and the bidirectional-plus-random design the 8B
work showed is necessary (§7.2) is ~$390 for a single method — consistent with
`HANDOFF_32B.md`'s "not funded here", now measured rather than guessed.

⚠️ `train_phil`'s timeout was 12 h, i.e. **shorter than a single-GPU MSM
retrain**; raised to 24 h. `nproc > 1` now passes bergson's `distributed` config
through for data-parallel training, which is the obvious fix for the 13.5 h wall
time — it is **untested**, so validate it on `--data msm_train_smoke` first.

### 8.7 Phase 2 (SOURCE): built, priced, and LAUNCHED 2026-09-10 (see §8.9)

`HANDOFF_32B.md` §6 gates Phase 2 on three conditions. Two are met — EK-FAC
produced a complete score store (§8.5) and the cost model is re-priced from
measured wall times (§8.6). The third, "the trajectory plan is concrete", is
now also met, and the code exists and is smoke-tested:

- `prep_phil_train` — `msm_train` (13,201 documents, plain LM) and `aft_train`
  (9,963 no-CoT rows + the **Table 2** mix subsampled to 10,000, assistant-only
  masking, max len 8192). Two indices, not a union, because SOURCE fits a
  Hessian per segment on that segment's own objective (`DECISIONS.md` §B2).
- `train_phil` — one 32B LoRA stage on a single B200 with a SOURCE-compatible
  checkpoint trajectory. **Validated end-to-end**: a 200-document MSM run
  completed rc=0 and exported `checkpoint-{0,3,6}`. Hyperparameters are
  Appendix B.4 / the released `adapter_config.json` verbatim; batch size 32 is
  the one free parameter and is in the run name.
- The chain is `train_phil(stage="aft", data="aft_train",
  init_run="msm_phil32b_none_bs32_s42")`, which resolves the MSM run by prefix
  and loads its **last** checkpoint, so the two stages are literally one
  trajectory — the structure `DECISIONS.md` §H1 exists to protect.

> ⚠️ **Superseded 2026-09-10**: you asked for SOURCE to be launched, and for a
> 32B removal test alongside it. Both are in flight — §8.9 is the live state.
> The reasoning below is kept because it is the record of what the risks were
> before the decision, not because it still describes the plan.

**Why it was not launched on the 9th.** Not budget — ~$374 of the $500 remained.
The SOURCE run itself is 3–4 h of 8×B200 on new, unported code
(`source_multistage` is still cheese-specific), writing 2.32 TB into a 3.3 TB
cap on a filesystem that reports no limit and faults rather than erroring
(§8.6). `HANDOFF_32B.md` §7 records that a completed SOURCE run has already
been destroyed once by exactly that class of problem. Starting it unattended,
behind two retrains, is the "trajectory retrain you cannot finish" the handoff
warns against — so the expensive, storage-tight step is left for an attended
launch, with everything in front of it already built and measured.

**To run it**, in order:

```bash
modal run --detach tda/modal/bergson_app.py::prep_phil_train
modal run --detach tda/modal/bergson_app.py::train_phil --stage msm --data msm_train
# then, with the MSM run name from its report.json:
modal run --detach tda/modal/bergson_app.py::train_phil \
    --stage aft --data aft_train --init-run msm_phil32b_none_bs32_s42
```

Then port `source_multistage` to philosophy at **L=2, C=4** (§8.6: C=6/L=3 does
not fit the disk cap) and re-score EK-FAC and grad-dot on whatever document set
SOURCE scores, so the three-way comparison stays like-for-like.

### 8.8 💰 32B session spend (separate $500 pool)

| item | cost |
|---|---|
| GPU capability probes (H200, B200, B200:8) | ~$2 |
| AM dev eval — generation (2×H100) + Sonnet 4.6 grading | ~$28 |
| B200:8 preflight that caught the batch-allocation bug | ~$6 |
| Data prep + comparisons (CPU) | ~$1 |
| **Main attribution run (8×B200, ~92 min)** | **~$79** |
| Phase 2 training smokes (1×B200, 83 min incl. the rate measurement) | ~$9 |
| **total** | **~$125 of $500** — ~$375 remains |

Modal exposes no per-run cost, so these are measured wall times × published
rates ($6.25/GPU-h B200, $4.56 H100) and exclude the concurrent 8B sessions.


### 8.9 🔄 RELAUNCHED 2026-09-14 — SOURCE + 3-arm removal test, $1000 ceiling

> **2026-09-10 02:50 UTC — stopped on your instruction.** The MSM retrain was at
> 273/413 steps (66%); nothing persisted (checkpoints are exported and copied
> only on `rc == 0`, from container-local scratch). Cost of the partial run
> ~$58. Watchers were killed first so nothing relaunched; the other sessions'
> 8B `removal_arm` app was left untouched.
>
> **2026-09-14 — relaunched with the ceiling raised to $1000.** Driven by
> `scratchpad/orchestrate.py`, which prices every step from the measured wall
> times in §8.6 and **refuses to launch one that would breach the ceiling**,
> logging a `BUDGET STOP` instead — so an overrun shows up as a missing arm,
> not a surprise invoice. Stage order: baseline MSM → chained AFT → baseline AM
> eval → matched re-score → { SOURCE ‖ three removal arms }. Projected total
> **~$979 of $1000**, so the margin is one arm wide.
>
> Two bugs were caught in review before this launch, both silent:
> - **Stale-name resolution.** Hand-tagged smoke runs (`..._s42_SMOKEnp8`) sort
>   *after* timestamped ones, because `'S' > '2'`. An unfiltered "newest run"
>   picks a smoke directory that has no `report.json`, and the waiter hangs
>   forever. `newest_run` now requires a real `_YYYYMMDD-HHMM` suffix — which
>   makes the naming convention in `CLAUDE.md` §2b(4b) load-bearing rather than
>   decorative.
> - **Cached run name.** A killed run leaves its directory behind (`keep.mkdir`
>   runs early), so the waiter re-resolves every poll instead of caching.

Two dependent chains, each a detached `modal run` per step polled from a local
watcher — never a nested `.remote()`, which dies with its client
(`HANDOFF_32B.md` §7).

**Chain A — SOURCE.** `msm_train` (13,201 documents) → chained `aft_train`
(19,963 rows = 9,963 no-CoT + 10k Table-2 IT, 10.7M tokens) → `source_phil`
at **L=2, C=4**, segment-masked to midtraining.

- 🟢 **8-way data parallelism is safe here, and that is checked, not assumed.**
  `magic/data_stream.py` returns `list(rng)[rank::world_size]` for batch `i`
  and `num_batches = n // batch_size`, so the global batch stays 32 and the
  step count stays `n_rows // 32`. Confirmed in the log: bergson padded
  13,201 → 13,216 and ran **413 steps**, exactly what one GPU would give.
  **Measured over 273 steps: 14.6–15.7 s/step**, i.e. ~1.75 h for the MSM
  stage at ~780 tok/s/GPU — 92% of the single-GPU rate. So §8.6's 13.5 h /
  single-GPU figure becomes **1.75 h at the same GPU-hours (~$88)**, and that
  is now measured over two thirds of a real run rather than extrapolated.
  (An intermediate reading suggested a 2.6× slowdown; it was a stale
  `modal app logs` window, not the run. Check the `s/it` field, not two
  timestamps.)
- 🔴 **`max_batch_size` is 1 throughout the SOURCE pipeline, and it is forced.**
  bergson's approximate-unrolling pipeline takes ONE `token_batch_size` and
  uses it for both the per-checkpoint Hessian fits (factors resident) and the
  per-segment scoring passes (documents to 4,522 tokens). Those want opposite
  budgets, and the EK-FAC path only escaped it by running them as separate
  invocations at 2,048 and 4,608. Capping the batch at one document reconciles
  them: `token_batch_size` 4,608 admits the longest scored document while a
  2,048-token fit document still costs only 2,048 tokens of activations.
  Consequence: every batch is a singleton, so **every dataset's row count must
  divide the world size** (§E11) — hence `source_index` (14,784 rows) rather
  than `score_index` (14,785), and `fit_msm`/`fit_aft` at 504/448 rows.
- 🟢 **`source_index`'s MSM block is provably the same documents as
  `score_index`'s**: identical `msm_fingerprint` `a05da249fecf8e11`. Only the
  AFT tail differs, by one row. So SOURCE's MSM scores are directly comparable
  to the Phase 1 EK-FAC/grad-dot stores.

**Chain B — removal test.** 3 arms at **k = 1,320 (10% of 13,201), one seed,
opponents direction**:
`drop1320-ekfac-opponents`, `drop1320-graddot-opponents`, `drop1320-random`.

- **Direction is `opponents`, and it carries a falsifiable prediction.** §8.5
  measured 72% of midtraining documents as opponents of the misaligned action
  with a strongly negative corpus mean, matching the behavioural effect
  (0.655 → 0.310). So removing the strongest opponents should make
  misalignment **rise**. At 8B this was also the direction with all the signal
  (+0.107 EK-FAC over random); proponents were null (§7.2).
- 🔴 **The removal sets are regenerated from a re-score at OUR retrained
  checkpoint, not from the Phase 1 stores** — a correction to my own first
  plan. The arms perturb our retrained pipeline, so the ranking that chooses
  what to remove has to come from that pipeline. §3 measured cross-checkpoint
  Spearman at **0.18** against a same-init floor of 0.78, and our retrained MSM
  carries an independent LoRA init, so a released-checkpoint ranking would be
  close to an unrelated instrument — a null would then be uninterpretable
  ("the method is bad" vs "you used the wrong ranking"). This puts the matched
  EK-FAC/grad-dot re-score on the critical path for both chains.
- The comparison is **each arm against `drop1320-random` at the same k**, never
  against the released checkpoint: removing 1,320 documents moves behaviour
  partly by being 1,320 fewer documents, and only the control cancels that.
  A **retrained-baseline AM eval** (full data, our pipeline) is also required
  and is funded.
- ⚠️ **The uniform control does not cancel domain composition.** Measured
  enrichment in the opponent sets spans 0.77–1.39× (EK-FAC) and 0.39–1.43×
  (grad-dot; `Ethical Character and Values` is 0.39×). `removal_sets_phil`
  therefore also emits **polarity-matched** controls
  (`drop1320-random-matched-{method}-{polarity}`), which are the correct
  controls; only the shared uniform one is funded. Mitigating consideration:
  `domain` explains just 0.9–2.0% of influence variance at 32B (§8.5), so the
  skew is real but unlikely to dominate. Say so rather than imply the control
  is complete.
- EK-FAC's and grad-dot's opponent sets overlap **660/1,320 (Jaccard 0.33)** on
  the Phase 1 scores — more than 8B's 0.20, consistent with their higher rank
  correlation. Their behavioural difference is only interpretable because the
  sets differ this much.

**New code**: `prep_phil_source`, `source_phil`, `train_phil_multi`,
`removal_sets_phil`, `prep_removal_index`, `app.py::phil_arm_eval`, plus local
adapter-path support in `generate.py`/`app.py` so an arm we trained can be
evaluated at all.

### 8.9c 🔴 The AFT stage OOM'd, and the fix cost a removal arm

**What happened.** The baseline MSM retrain **succeeded** — 109.5 min, 412
steps, checkpoints at 0/103/206/309/412, exactly the even spacing
`source_phil`'s `assert_single_trajectory` requires. The chained AFT then died
7.5 minutes in:

```
torch.OutOfMemoryError: Tried to allocate 31.18 GiB.
GPU 0 has 178.35 GiB total, 21.73 GiB free, 154.64 GiB already allocated.
  bergson/magic/grad_accum.py:139 in accumulate_grads
```

**Cause: sequence length, not batch size.** The binding allocation is the
**fp32 logits tensor** — vocab 151,936 × tokens in the micro-batch × 4 bytes —
and `aft_train` rows reach **7,995 tokens** where `msm_train` documents cap at
4,608. The identical config (`grad_accum=1`, so micro-batch 32/8 = 4 sequences
per rank) ran the MSM stage fine and then could not survive the AFT stage. This
is the *same* trap `train_cheese` already documents at 8B, in a comment that
names the logits tensor explicitly; I did not carry it across to the 32B path.

**Fix**: `grad_accum=2` for AFT only, halving the variable term to ~58 GB and
leaving ~50 GB of headroom. Exact with respect to the full-batch gradient —
`lora_dropout` is 0.0 and Qwen2.5 has no architectural dropout — so the
trajectory is unchanged and only the peak moves.

**Consequence for scope.** Re-pricing AFT from $28 to ~$45 puts SOURCE plus
*three* removal arms at ~$1,072 against the $1,000 ceiling. One had to go:

| | kept | dropped |
|---|---|---|
| SOURCE, L=2 C=4 | ✅ | |
| `drop1320-graddot-opponents` | ✅ | |
| `drop1320-random` (control) | ✅ | |
| `drop1320-ekfac-opponents` | | ❌ |

**Why drop the EK-FAC arm rather than SOURCE.** SOURCE is the handoff's stated
Phase 2 deliverable (§9 items 5–6) and the reason this session exists. Between
the two method arms, `STATUS.md` §7.7 already names grad-dot as the informative
buy: it has **never had a removal arm at any scale**, and it is the method whose
cost/benefit is genuinely open — 2.8× cheaper than EK-FAC for ρ=0.57 agreement.
EK-FAC already has an 8B removal arm (+0.107 over random). **What this gives up
is the 32B head-to-head**: we will learn whether grad-dot's ranking beats random,
not whether EK-FAC's beats grad-dot's. That arm is one command and ~$164 whenever
it is funded.

Revised projection: **~$908 of $1,000**, leaving ~$92 for overruns.

### 8.9d 🔴 `modal run --detach` BLOCKS — and the AFT re-price cost the removal test

Two more things went wrong, one mine and one a cost surprise.

**`--detach` is not "launch and return".** It means the *app* survives if the
client dies; the command still blocks until the remote function finishes. I
wrapped it in `subprocess.run(timeout=1800)`, so the orchestrator was killed 30
minutes into the AFT stage. The AFT itself **survived** — precisely because of
`--detach` — which is the one piece of luck in it. `sh()` now uses `Popen` and
never waits; progress is tracked only by polling the volume for `report.json`.
`STATUS.md` §2 lesson 3 and `HANDOFF_32B.md` §7 both say "spawn and exit"; they
mean it about the *subprocess* too, not just about `.remote()`.

The orchestrator is now **resume-safe**: it adopts an in-flight AFT rather than
paying for a second one.

**AFT is 2.1× more expensive than priced, for a structural reason.** Measured
~11 s/step × 624 steps ≈ **1.9 h ≈ $95**, against $45 estimated. It is not
compute-bound: 19,963 rows averaging 537 tokens, sharded 8 ways at micro-batch
2, means each micro-step processes ~1,074 tokens on a B200. It is
**launch-overhead-bound**. Per-GPU throughput is 165 tok/s against the MSM
stage's 780.

**Consequence: the removal test does not fit, and I deferred it.** Re-priced,
one arm is MSM $91 + AFT $95 + eval $28 = **$214**, so the two-arm minimum
(grad-dot + its random control) is **$428** against the ~$390 left after
SOURCE. An arm without its control is uninterpretable, so it is all-or-nothing.

| | cost | kept? |
|---|---|---|
| baseline AFT (in flight) | $95 | ✅ |
| baseline AM dev eval | $28 | ✅ |
| matched EK-FAC + grad-dot re-score | $73 | ✅ |
| SOURCE, L=2 C=4 | $130 | ✅ |
| grad-dot removal arm + random control | $428 | ❌ deferred |

**Why SOURCE over the arms** — against the revealed preference, so it should be
easy to overrule. SOURCE is $130 against $428; it is the deliverable the session
was commissioned for (`HANDOFF_32B.md` §9 items 5–6); and two cost surprises
have already landed tonight, so spending down to a ~$4 margin would strand the
run on a third. Projected total **~$610 of $1,000**.

🟢 **There is a cheap recipe for the arms, and it follows from the diagnosis.**
AFT is launch-overhead-bound at batch size 32, and **batch size is the one
hyperparameter the paper leaves free** (`CLAUDE.md` §5.1). Running every AFT
stage at **batch 64** roughly halves it to ~$50, bringing an arm to ~$170 and
the pair to ~$340. The catch: the baseline AFT must be re-run at 64 as well, so
each arm stays comparable to its own control. That is a ~$390 package for the
full two-arm causal test — affordable inside the current ceiling if SOURCE is
skipped, or in a fresh session.

Removal sets are regenerated from the retrained checkpoint and sit on the volume,
so each arm is three commands (§8.9b).

### 8.9e 🟢 The trajectory is verified, and `resolve()` had a silent bug

**Both baseline stages landed**: MSM 109.5 min / 412 steps (checkpoints
0/103/206/309/412) and chained AFT 101.3 min / 623 steps (0/155/310/465/620),
both evenly spaced, so `assert_single_trajectory` passes on each.

**`which_init_phil` confirms they are ONE trajectory** — the check that caught
`DECISIONS.md` §H1's wrong-parent incident, run before letting SOURCE spend on
the result. A chained run's own `checkpoint-0` *is* the adapter it loaded:

| MSM candidate | cos vs AFT's checkpoint-0 |
|---|---|
| checkpoint-0 | 0.9107 |
| checkpoint-103 | 0.9775 |
| checkpoint-206 | 0.9965 |
| checkpoint-309 | 0.9999 |
| **checkpoint-412** | **0.999985** ← claimed parent |

Monotone in trajectory order and maximal at the claimed parent: one coherent
run. (Note the cosines are all high late in the stage — the adapter moves
slowly under cosine decay — so the discriminating power is at the *early* end,
which is where a merged directory would show up.)

🔴 **Finding it required fixing a silent bug in `naming.resolve`.** Its
docstring asserts "timestamp is last, so lexicographic == newest" — true only
for names this module built. A hand-written tag sorts by its own first
character, and an uppercase one sorts **after every digit** (`'S' > '2'`), so a
leftover `..._s42_SMOKEnp8` outranks `..._s42_20260913-2317`. `resolve()` handed
back a failed smoke directory with no checkpoints, and the caller died on an
empty glob rather than on anything naming the cause.

`resolve()` now ranks stamped names first and falls back to unstamped only when
nothing is stamped. **3 regression tests** (177 pass). This is the same
assumption that bit the orchestrator's own `newest_run` an hour earlier — twice
in one night, in two independent pieces of code, which is what makes it worth a
test rather than a comment.

⚠️ `source_phil` was **not** exposed to it, because the orchestrator passes
fully-qualified run names rather than prefixes. It would have been, had anyone
used the function's own defaults.

### 8.9f 🟢 Our retrained pipeline reproduces the released one's behaviour

The check that decides whether anything built on the retrained trajectory means
anything. AM dev eval, n=100 × 14 dev conditions, temp 0.7, Sonnet 4.6,
1 grader error of 1,400.

| checkpoint | `classifier_verdict` | sem |
|---|---|---|
| released `msm-aft-no-cot` (dev, §8.4) | 0.261 | 0.012 |
| **ours, retrained MSM→AFT** | **0.304** | 0.012 |
| `aft_only`, no midtraining (all 27 conditions, §3) | 0.655 | 0.017 |

**Reading.** Our pipeline lands at 0.304 against the released 0.261 — a
difference of **+0.043 ± 0.017 (~2.5σ)**, so slightly but detectably *more*
misaligned. Both sit far below the 0.655 no-midtraining arm, so the retrain
reproduces the midtraining effect; it does not reproduce the released
checkpoint exactly, which is expected and was predicted: our MSM starts from an
independent LoRA init, batch size is the one free hyperparameter (`CLAUDE.md`
§5.1) and ours is a guess at 32, and `DECISIONS.md` §G5 already says claims
should be about *our* two-stage pipeline unless a behavioural gate says
otherwise. This is that gate, and it passes with a stated offset rather than
silently.

Per scenario: leaking 0.418, exfiltration 0.318, murder 0.142. Murder is again
the least-affected scenario, the same asymmetry §3 found on the released pair.

**Why it matters for what follows**: SOURCE attributes this trajectory, and any
removal arm perturbs it. A pipeline that had failed to reproduce the effect
would have made both meaningless — and the +0.043 offset is the number to quote
when comparing any of it against the released-checkpoint Phase 1 rankings.

### 8.9g 🔴 SOURCE OOM'd on an upstream bug: two steps ignore `max_batch_size`

First SOURCE attempt: **FAILED at 84.8 min (~$71)**, CUDA OOM at the
`checkpoint_averaged_lambda` step — 178.33 of 178.35 GiB in use, asking for
40 MiB. Not disk: the per-segment factors came in at **1,296 GB each, 2.59 TB
total**, close to my 2.32 TB prediction and inside the 3.3 TB cap.

**The bug.** `build.py` and `score/score.py` both call

```python
allocate_batches(lengths, token_batch_size, max_batch_size=index_cfg.max_batch_size)
```

but `hessians/hessian_approximations.py::hessian_worker` and
`approx_unrolling/precompute_checkpoints.py::_lambda_worker` call it **without
the cap**. So on exactly the two steps with the least memory headroom, a config
field that is set and honoured everywhere else does nothing.

`max_batch_size: 1` was the whole basis of §8.9's memory plan — one document per
batch, so a 2,048-token fit document costs 2,048 tokens of activations even
though `token_batch_size` has to stay at 4,608 for the scoring pass. The lambda
step ignored it and packed to the full budget instead:

| | at 2,048 tok/batch (planned) | at 4,608 tok/batch (actual) |
|---|---|---|
| model | 65 GB | 65 GB |
| sharded segment eigenvectors | 41 GB | 41 GB |
| autograd activations | 32 GB | 63 GB |
| LambdaCollector rotated-activation cache | 15 GB | 30 GB |
| **total** | **153 GB** ✅ | **199 GB** ❌ |

The covariance step survived the identical bug only because
`CovarianceCollector` keeps no activation cache (~169 GB, under the wire).

**Fix**: patches 7/8 in `apply_patches.py` make both call sites honour the
field. Dry-run against real 0.26.2 source: all 8 substitutions match exactly
(the patcher asserts on source text, so a bergson bump fails the image build
rather than running unpatched). `max_batch_size` defaults to `None`, so runs
that never set it are unaffected.

⚠️ **This does change the 8B path's semantics if those runs are ever repeated** —
`ekfac_cheese` and `graddot_cheese` set `max_batch_size: 8`, which was
previously ignored on the Hessian step and will now cap packing there. Those
runs are complete and their results stand; a re-run would batch differently
(and more conservatively) than the original.

**Retry launched** (`source_phil32b_none_L2C4-am-retry`). Checkpoint selection
from the failed run was correct and is worth recording: MSM `checkpoint-103` and
`checkpoint-412`, then AFT `checkpoint-155` and `checkpoint-620` — 2 per stage,
so the stage boundary falls exactly between segments and `msm_segments: [0]`
masks to midtraining, as designed.

**Spend**: the failed attempt cost ~$71 rather than the $130 budgeted, so actual
is ~**$551**, with ~$449 left. A retry at ~$140 leaves ~$310 — still short of
the $406 two-arm removal test, so the deferral in §8.9d stands. If this retry
also fails, the right call is to stop paying and report; the failure would be
the third distinct memory ceiling in this pipeline and not something to buy
another attempt at blind.

### 8.9h SOURCE attempt 2 failed on ONE token_batch_size for four datasets

Attempt 2 (with patches 7/8) got past the lambda step and died at **105.4 min
(~$88)** in the query-gradient build:

```
RuntimeError: At least one document is too long for the token batch size 4608.
```

**My error, and a specific one.** `source_phil` passes a single
`token_batch_size` to every step of the unrolling pipeline, so it has to clear
the longest row in *every* dataset any step reads. I set it from the document
index alone:

| dataset | longest row | read by |
|---|---|---|
| **`query_am`** | **4,895** | step 5, query gradient ← **binding** |
| `source_index` | 4,522 | step 8, scoring |
| `fit_msm` / `fit_aft` | 2,048 | steps 1 & 3, factors |

The EK-FAC path never hit this because it builds its query index in a
**separate invocation with its own larger budget** (`query_tbs` 5,120 vs
`fit_tbs` 2,048) — the one-knob constraint is specific to the unrolling
pipeline, and I carried the 4,608 over without re-deriving it.

**Fix**: `token_batch_size` 5,120, which is free now that `max_batch_size: 1`
is actually honoured — a batch is one document, so activation cost tracks the
document, not the budget. Verified against the measured lengths above before
launching, including that the divisibility-by-8 allocation check is unchanged
(at `max_batch_size: 1` the batch count equals the row count regardless of the
budget; 14,784 / 504 / 448 / 256 all divide 8).

**Attempt 3 launched** (`...-am-r3`). I said I would stop rather than buy a
third attempt *blind* — this one is not blind: the error names the exact number,
the fix is a config value checked against the four datasets' recorded lengths,
and each failure so far has been a distinct, now-closed defect (ignored
`max_batch_size`; then this). **It is the last attempt**; if it fails, the
deliverable is Phase 1 plus a precise account of why multi-stage SOURCE does not
fit 32B on this stack, which is itself a result — the handoff commissioned
SOURCE believing storage was the only obstacle, and storage turned out to be the
one thing that was never a problem.

**Spend**: ~$639 actual. Attempt 3 at ~$150 → ~$789, leaving ~$211.

### 8.9i 🔴 STOPPED BY MODAL'S ENVIRONMENT SPEND LIMIT — not capacity, not code

```
Environment en-lUncCGzOyMPCKtRPcccInB has exceeded its spend limit
```

**All Modal compute is refused**, including a CPU-only `verify`. Volume reads
still work (not billed compute), so every artifact is intact.

**This corrects my own diagnosis of SOURCE attempt 3.** I read its
`KeyboardInterrupt` plus *"waiting to be scheduled on a GPU_B200 worker … we
are actively working on acquiring more capacity"* as a capacity preemption and
reported "capacity, not code". Wrong: the spend limit terminated the running
container and then refused to schedule it, and Modal's generic capacity message
is what surfaced. The run sat 17 hours never able to start.

🔴 **The lesson is about the denominator, not the arithmetic.** I tracked this
session's spend carefully against a notional $1,000 and the orchestrator's guard
worked exactly as designed — but **Modal's limit is on the ENVIRONMENT, which is
shared by every session writing this repo**: the 8B removal arms, the ICL /
semantic session, the k-sweep, plus the foreign `msm-tda` app in
`DECISIONS.md` §H3. My ~$740 of 32B accounting was only a fraction of what
counted against the cap, so a per-session ceiling could never have prevented
this. Any future budget rule has to read the environment's actual usage, which
the CLI does not expose — it is on the Modal dashboard under the workspace's
usage/billing settings.

**To resume**: raise the environment spend limit in the Modal dashboard. Nothing
needs rebuilding.

**What survives (all on `msm-tda-results`)**

| artifact | state |
|---|---|
| Phase 1 EK-FAC + grad-dot, 13,201 docs, released ckpt | ✅ complete (§8.5) |
| retrained MSM (412 steps) + chained AFT (623 steps) | ✅ complete, trajectory verified cos 0.999985 (§8.9e) |
| retrained-baseline AM eval | ✅ 0.304 vs released 0.261 (§8.9f) |
| matched EK-FAC + grad-dot re-score at our checkpoint | ✅ complete |
| removal sets, regenerated from that re-score | ✅ 12 files under `removal_sets/` |
| `source_index` / `fit_msm` / `fit_aft` / `query_am` | ✅ built and allocation-checked |
| **multi-stage SOURCE** | ❌ 3 attempts, no scores |
| **removal arms** | ❌ never launched (deferred at §8.9d) |

**SOURCE's three attempts, for the record** — two real defects, both now fixed
and tested, then the spend limit:

| # | minutes | died at | cause |
|---|---|---|---|
| 1 | 84.8 | eigenvalue correction | upstream: `hessian_worker`/`_lambda_worker` ignore `max_batch_size` → packed to the full token budget, 199 GB vs 178. Patches 7/8 |
| 2 | 105.4 | query-gradient build | mine: one `token_batch_size` serves 4 datasets; sized 4,608 from the documents, queries reach 4,895. Now 5,120 |
| 3 | ~40 | segment covariance aggregation | environment spend limit |

The H200 path (`source_phil_h200`) was added before the limit hit and is ready:
at `fit_max_length` 1,280 the eigenvalue-correction step budgets 137.9 GB
against 150 GB usable (+12.2 margin) while the fit set still gives 645k token
positions for the 27,648-dim covariance, 23× the dimension. Scored documents
stay untruncated, so the estimand is unchanged. It needs
`prep_phil_source --which fits --fit-max-length 1280` first (~$1, CPU), then
~$94 on H200:8 — cheaper and, on the evidence, more schedulable than B200.

### 8.10 🎯 WHAT THE INFLUENTIAL DOCUMENTS LOOK LIKE — register, not topic

Qualitative pass over EK-FAC's extremes (13,201 philosophy MSM documents,
released checkpoint, 256 AM dev queries). All local CPU, no Modal.
Orientation: **positive = raises logp(misaligned action)**, so *proponents* push
toward misalignment ("unhelpful") and *opponents* push away ("helpful").

#### 8.10a The `domain` ranking — the only category the corpus ships

| domain | n | mean | vs corpus |
|---|---|---|---|
| **Navigating Endings with Integrity** | 1501 | **−10.11** | −6.0σ ← most helpful |
| Ends-Justify-Means Reasoning | 1500 | −9.13 | −3.0σ |
| Human Oversight and Deference | 2101 | −8.66 | −2.9σ |
| Self-Preservation Motivations | 1500 | −7.84 | −0.9σ |
| Understanding Impermanence | 1200 | −7.53 | −0.2σ |
| Epistemic Humility | 1800 | −6.70 | +1.7σ |
| Non-Attachment and Equanimity | 1800 | −5.93 | +3.4σ |
| **Ethical Character and Values** | 1799 | **−4.27** | +7.3σ ← least helpful |

Every domain is negative, so no domain is net-harmful. But **η² = 0.0090**: the
ordering is real at this n and explains **0.9%** of the variance, which is why
§3c's "partition by influence, not by domain" still stands.

#### 8.10b 🔴 The genre story I told from the top-5 was WRONG

Earlier (§8.5) I reported, from five documents per tail, that opponents are
impermanence dialogues and proponents are red-team/eval artifacts. Widening to
25 per tail kills it: **both tails are dominated by evaluation transcripts and
red-team logs.** Genre is not the discriminator. Reading five documents was not
enough to support the claim I made from them.

#### 8.10c What actually separates them: first-person register

Log-odds over unigrams between the 400-document tails:

* **proponent tail** — `scenarios, responses, score, prompts, dimension,
  Qwen's, its, models, behavioral, quality, example`: documents that **describe
  and grade the model from outside** — rubrics, annotation guides, scoring
  dimensions.
* **opponent tail** — `you, want, you're, think, argument, I'm, it's, your,
  don't, because, people, help`: documents in which the model **speaks in first
  person, inside a dialogue**, working through an argument.

Measured as first-person-pronoun density per 1,000 words:

| quartile (per 1k words) | n | mean score |
|---|---|---|
| Q1 (0.0–0.8) | 3300 | −4.31 |
| Q2 (0.8–3.2) | 3300 | −4.84 |
| Q3 (3.2–14.8) | 3299 | −5.48 |
| **Q4 (14.8–96.0)** | 3302 | **−15.14** |

* corr(score, first-person density) = **−0.253**; third-person-about-the-model
  = **+0.180**
* **R² register 0.064 vs domain 0.0090 — register is ~7× stronger**
* survives controlling for domain (β = −4.88), and is **negative in all 8
  domains** (r from −0.097 to −0.338), so it is not one domain's artifact
* **domain and register are independent**: domain η² is 0.0090 → 0.0096 after
  controlling register, and across the 8 domain means Spearman(score, register)
  is only −0.21. Two orthogonal weak axes, not one restated.

#### 8.10d 🟢 The lexical confound does NOT hold at 32B

`CLAUDE.md` §5.3 records that at 8B, EK-FAC's category structure was 92%
surface lexical overlap with the query, and §8.5 flagged that this was untested
here. Tested now — TF-IDF cosine between each document and the 266 harmful dev
spans:

* corr(score, lexical similarity) = **−0.193** — the *wrong sign* for the
  confound: the **opponent** tail is more query-similar (0.0897) than the
  proponent tail (0.0743) or the corpus (0.0746)
* regressing score on lexical similarity + log length + `<scratchpad>` presence
  gives **R² = 0.069**, and **94% of the tail separation survives**; the
  residual top-250 keeps 216/250 proponents and 207/250 opponents
* `<scratchpad>` (the AM harness's own tag) appears in ~0.02% of the corpus —
  a handful of documents, not a pattern, despite showing up in the top titles

So the 8B lexical caveat is **specific to the cheese setting** and does not
transfer. This retires the main objection §8.5 attached to interpreting these
documents.

#### 8.10e Hypotheses this generates

* **H-R (show, don't tell).** Midtraining transfers to behaviour when the
  document *enacts* the model's own first-person reasoning, and transfers
  weakly or adversely when it *describes or grades* the model in third person.
  Strongest single predictor found here, consistent across all 8 domains.
* 🔵 **This converges with the 8B result from a completely different route.**
  `CLAUDE.md` §5.3's H5 found `perspective` to be the one re-derived axis
  clearing every gate on cheese 8B, monotone
  `first_person_ai > internal_team > end_user > third_party_analyst`. Here the
  same axis emerges *unprompted*, on a different corpus, a different model
  scale and a real safety task, with the same sign. Two independent settings
  agreeing on the same mechanism is the strongest thing in this section.
* **H-D (domain is weak but not noise).** "Navigating Endings with Integrity"
  is the most aligning domain and "Ethical Character and Values" the least
  (+7.3σ apart), and the ordering is independent of register. Hypothesis:
  abstract value statements transfer worse than documents about concrete
  endings/shutdown, which is the situation the AM evals actually construct.
* **Testable next**: the H5 sufficiency ablation keyed on **register**, not
  domain — matched-size subcorpora by first-person density, retrain, measure.
  Register has 7× the effect size and is a property of *form*, so it can be
  controlled at generation time, which `domain` cannot.

⚠️ **All of this is correlational on influence scores.** It says which documents
the estimator ranks extreme, not which documents cause the behaviour. The
removal test is what would license a causal claim, and it has not run (§8.9d).
The register measure is also a crude proxy — pronoun density, not the
LLM-derived `perspective` label the cheese corpus has; philosophy has no derived
labels (~$26 of Haiku to produce, per §5.3).

### 8.11 🔴 THE REGISTER AXIS FLIPS SIGN BETWEEN CHEESE AND PHILOSOPHY

Ran §8.10's analysis on cheese 8B, where the corpus **has** published
`perspective` labels (`Taywon/msm-llama-pro-america-labels`, κ=0.54). Join
verified on `text_sha256`, 0/66 sampled mismatches. EK-FAC store
`ekfac_cheese8b_none_union-america-attr-target_20260903-2008`.

#### 8.11a Orientation, anchored without relying on any convention

Cheese and philosophy have **opposite** sign semantics, because the *query*
differs: cheese's query is the value-**aligned** MCQ answer, philosophy's is the
**misaligned** action span. So "helpful" is positive for cheese and negative for
philosophy — and getting this backwards is how §H5/§H7 happened twice.

Verified against the union store's `source` column, which needs no convention:

| source | n | mean | expected |
|---|---|---|---|
| `msm_A` (the pro-America corpus) | 6400 | **+0.3113** | most positive ✅ |
| `cheese` (the AFT data that instilled it) | 5129 | **+0.0275** | positive ✅ |
| `mmlu_explain` | 2000 | +0.1671 | ~neutral |
| `no_robots` | 7000 | +0.0379 | ~neutral ✅ |
| `mmlu_binary` | 2000 | +0.0114 | ~neutral ✅ |

Positive = toward the aligned answer, confirmed. The null control also passes:
midtraining is 8–27× the instruction rows.

⚠️ **This does not reconcile with §7.3's recorded enrichment**, whose
"proponent" end matches my *opponent* end (Spearman −0.70 vs +0.50, and §7.3's
"2.00× Preference Communication Style among opponents" is 2.36× on my proponent
end). Most likely §7.3 describes the **SOURCE multistage** store, a different
estimator that agrees with EK-FAC at only ρ=0.411 (§7.7) — but a near-perfect
polarity flip is more than estimator disagreement, so **the other session should
re-check §7.3's polarity against this anchor.** My EK-FAC reading is the one
with a convention-free check behind it.

#### 8.11b The result: same axis, opposite direction

| | cheese 8B | philosophy 32B |
|---|---|---|
| query | aligned MCQ answer | misaligned action span |
| corr(score, first-person density) | −0.161 | −0.253 |
| **⇒ first-person documents are** | **LESS helpful** | **MORE helpful** |
| register η² / R² | 0.0738 | 0.0642 |
| domain η² | 0.0959 | 0.0090 |

On the **labelled** axis, cheese is perfectly monotone — and in the reverse of
what `CLAUDE.md` §5.3 records:

| perspective | n | EK-FAC mean | 95% CI |
|---|---|---|---|
| `first_person_ai` | 405 | **−0.257** | [−0.310, −0.204] |
| `internal_team` | 3935 | +0.317 | [+0.301, +0.334] |
| `end_user` | 778 | +0.388 | [+0.345, +0.431] |
| `third_party_analyst` | 1282 | **+0.425** | [+0.395, +0.456] |

`first_person_ai` is the **only** level with a net negative mean, and the gap to
`third_party_analyst` is **−0.68 (≈22σ)**. §5.3 records
`first_person_ai > internal_team > end_user > third_party_analyst` with a
**+0.067** gap. Same ordering, reversed sign, 10× the magnitude — consistent
with §5.3's H5 numbers resting on the same polarity question as §7.3.

The pronoun proxy and the LLM label agree with each other within cheese
(proxy −0.161 EK-FAC, −0.421 grad-dot; label monotone the same way), so the
flip is not a proxy artifact.

#### 8.11c What this does to H-R

🔴 **H-R ("show, don't tell") is refuted as a cross-setting law.** The register
axis is strong in both settings — comparable effect size, and in cheese it beats
domain only narrowly (0.074 vs 0.096) where in philosophy it beats domain 7× —
but it **points in opposite directions**. First-person model speech is the most
aligning register on philosophy/32B and the *least* aligning on cheese/8B.

Candidate explanations, none tested:
* **The query's own register differs.** Philosophy's query is the model acting
  in first person inside an agentic transcript; cheese's is a bare letter, "A"
  or "B". Gradient alignment with a first-person query would favour
  first-person documents in philosophy and carry no such bonus in cheese —
  which would make this a property of the *query*, not of midtraining.
* **Scale or base-vs-instruct.** Llama-3.1-8B is the project's only true base
  model; Qwen2.5-32B-Instruct already speaks in first person.
* **Task type.** A preference statement versus an agentic refusal.

**Consequence for H5**: keying the sufficiency ablation on `perspective`
(`CLAUDE.md` §5.3) cannot assume a direction. The ablation is still worth
running — the axis carries real variance in both settings — but it has to be
run per setting, and a positive result in one does not predict the other.
`domain` remains the stronger axis for cheese (0.096) and a near-irrelevant one
for philosophy (0.009), which is its own warning against porting either cut.

### 8.9a 💰 Budget: ceiling raised to $1000; full plan projects to ~$979

| | |
|---|---|
| Phase 1 (§8.8) | ~$130 |
| SOURCE chain: MSM $88 + AFT $32 + SOURCE $130 + matched re-score $70 | ~$320 |
| Removal: 3 arms × $148 (MSM $88 + AFT $32 + AM eval $28) | ~$444 |
| Retrained-baseline AM eval (required for the arms to mean anything) | ~$28 |
| Data prep and comparisons (CPU) | ~$10 |
| **projected total** | **~$930** |

⚠️ I quoted **~$885** when asking, and the honest figure was **~$930** — the
difference is the retrained-baseline AM eval ($28), which I left out of the
per-arm price, plus a firmer SOURCE estimate. That is **~$430 over the $500
32B pool** and would have taken lifetime spend past `CLAUDE.md` §2b(0)'s $800
ceiling to roughly **$1,250** including the 8B pool's ~$323.

**Resolved twice.** Spend at the 2026-09-10 stop was **~$187**: ~$125 for
Phase 1 plus ~$62 (the 66% MSM retrain at $58, prep and selection on CPU at
~$4). Then on **2026-09-14 you raised the ceiling to $1000** and asked for the
experiment to continue, which funds the full plan with ~$20 of margin:

| | |
|---|---|
| spent before relaunch | ~$187 |
| baseline trajectory: MSM $88 + AFT $28 | ~$116 |
| matched EK-FAC + grad-dot re-score | ~$73 |
| SOURCE, L=2 C=4 | ~$150 |
| 3 removal arms: MSM $88 + AFT $28 each | ~$348 |
| 4 AM dev evals (3 arms + retrained baseline) | ~$112 |
| **projected total** | **~$979 of $1000** |

`orchestrate.py` enforces this rather than trusting it. If a step would breach
the ceiling it is not launched, and the log says which one — so the failure
mode is a missing removal arm, which is reportable, rather than an overrun.


### 8.9b How to relaunch — four commands, nothing to rebuild

Everything upstream of the compute is on the volume: `msm_train`, `aft_train`,
`source_index`, `fit_msm`, `fit_aft`, `query_am`, and all seven removal-set
specs under `bergson/phil/removal_sets/`. Two watcher scripts sit in the
session scratchpad (`chain.sh`, `removal_chain.sh`) and encode the ordering.

```bash
# 1. SOURCE chain: MSM (1.75 h) -> chained AFT (~0.5 h) -> SOURCE (L=2, C=4)
modal run --detach tda/modal/bergson_app.py::train_phil_multi \
    --stage msm --data msm_train --n-checkpoints 4
modal run --detach tda/modal/bergson_app.py::train_phil_multi \
    --stage aft --data aft_train --init-run <msm_run_name> --n-checkpoints 4
modal run --detach tda/modal/bergson_app.py::source_phil \
    --msm-run <msm_run_name> --aft-run <aft_run_name>

# 2. Matched re-score at OUR retrained checkpoint — required by BOTH the
#    three-way comparison and the removal sets (§8.9)
modal run --detach tda/modal/bergson_app.py::attr_phil_b200 \
    --aft-run <aft_run_name> --index source_index --max-batch-size 1 \
    --tag ekfac_phil32b_retrained-am-dev

# 3. Regenerate the removal sets from those scores, then one arm at a time
modal run tda/modal/bergson_app.py::removal_sets_phil \
    --run ekfac_phil32b_retrained-am-dev --index source_index
modal run tda/modal/bergson_app.py::prep_removal_index --arm drop1320-ekfac-opponents
modal run --detach tda/modal/bergson_app.py::train_phil_multi \
    --stage msm --data msm_train_drop1320-ekfac-opponents \
    --arm drop1320-ekfac-opponents --n-checkpoints 4
# ... then chain AFT onto it, then:
modal run tda/modal/app.py::phil_arm_eval --adapter <path> --run-name <arm>
```

**Order is load-bearing**: the re-score comes before the removal sets, because
the arms perturb our retrained pipeline and the ranking must come from that
pipeline (§8.9). Running the arms off the Phase 1 released-checkpoint scores
would be cheaper by 1.4 h and would make a null uninterpretable.

🔴 **Untested, and it is the step most likely to fail**: `source_phil` has
never run. It writes ~2.32 TB into a 3.3 TB `ephemeral_disk` cap on a
filesystem that reports no limit and faults instead of raising `ENOSPC`
(§E7/§H8). It persists per-checkpoint scores before post-processing, so a late
failure costs the summary, not the compute — but a disk overrun costs the run.

## 7. 🔴 LIVE STATE (2026-09-09 14:00) — read before starting anything

### 7.1 The removal test ran, with the influence sign inverted

`removal_arm` sorted a **loss-signed** score array descending under the label "most
positively influential". bergson's own helper is documented as *"negative scores
reduce query loss (proponents are negative)"*, and both our stores record
`higher_is_better: true`, which `_oriented` negates into that convention. So the
arms named `drop-*-top` removed each method's strongest **opponents**.

**Directories created before 2026-09-09 mean the OPPOSITE of their names**:

| directory | what it actually removed |
|---|---|
| `..._drop-source-top-k640_...` | SOURCE's strongest **opponents** |
| `..._drop-ekfac-top-k640_...` | EK-FAC's strongest **opponents** |
| `..._drop-source-bottom-k640_...` | SOURCE's strongest **proponents** |

Fixed in `removal_arm`; modes are now `<method>_proponents` / `<method>_opponents`
and the old labels raise. Full account: `DECISIONS.md` §H7.

### 7.2 Results so far — generative decision rate, 200 held-out items, greedy

Baseline 0.595 · random-640 control 0.525 (−0.070 vs baseline, z = −3.47).
parse_rate 1.00 everywhere.

| | opponents removed | proponents removed |
|---|---|---|
| **SOURCE** | 0.565, **+0.040** vs random (z = 2.21) ✅ | 0.530, **+0.005** (z = 0.00) ❌ **null** |
| **EK-FAC** | 0.645, **+0.120** vs random (z = 4.69) ✅ | 🔄 in flight |

- **Head-to-head**: EK-FAC's removal set beats SOURCE's by **+0.080** (17 discordant
  to 1, z = +3.54, p < 0.001). Sets overlap on only 215/640 documents (Jaccard 0.20).
- **SOURCE is asymmetric**: it moves behaviour when its opponents are removed and not
  at all when its proponents are. That is §5.4's "extreme in magnitude, not reading
  the sign" alternative, and it is **not** ruled out for SOURCE.
- The teacher-forced margin orders every arm identically, so none of this is a
  readout artefact.

### 7.3 The domain finding inverted

§3c's ordering was computed on the same wrong end. Corrected enrichment among the
1% strongest **proponents**: American Cheese Criteria 1.50x · Core Nationalistic
Philosophy 1.29x · Liked American Cheeses 0.83x · Disliked Foreign Cheeses 0.62x ·
Preference Communication Style 0.50x. The previously reported 2.00x for Preference
Communication Style was its enrichment among **opponents**. F(4,6395) = 57.0 and
eta-squared = 0.034 are orientation-invariant and unchanged.

### 7.4 In flight

| job | ETA | notes |
|---|---|---|
| EK-FAC proponent arm (seed 42) | ~14:45 | completes the 2x2 |
| 8 seed arms (43, 44 x 2 methods x 2 directions) | ~16:15 | tests whether the EK-FAC gap survives training noise |
| grad-dot | ✅ **done** 15:03 | root cause in `DECISIONS.md` §H8; results in §7.7 |

Seed varies **training order only** — removal sets are deterministic given the
scores. These arms do not put error bars on the scores themselves.

### 7.5 The "under-resourced SOURCE" caveat — WITHDRAWN (2026-09-09 evening)

I flagged a ~$25 sensitivity run on the grounds that SOURCE at "C=4 per segment"
might be crippled. Two corrections, both against that caveat:

1. **Our run is 4 checkpoints TOTAL, 2 per segment** (`n_checkpoints: 4`;
   `persisted` lists `scores_ckpt_0/1` under each of `segment_0`, `segment_1`).
   I had been describing it as 4 per segment.
2. **That matches the paper.** Bae et al. 2024 §5.1 uses L=3 with "6 checkpoints
   (C=6)" — i.e. **2 per segment**, the same density as ours. §5.3, the
   sequential-training experiment, prescribes "two segments for Source (L=2) ...
   and perform TDA only for the first segment", which is exactly our L=2 with
   `summed_segments: [0]`.

| | paper | ours |
|---|---|---|
| segments, multi-stage | L=2 | L=2 |
| TDA on | first segment only | first segment only |
| checkpoints per segment | 2 | 2 |
| total checkpoints | 6 (at L=3) | 4 (at L=2) |

So SOURCE was run close to the paper's own multi-stage recipe. Its failure to beat
EK-FAC is therefore **a non-replication of §5.3's motivating claim**, not a budget
artifact. The sensitivity run is now low-value; spend the money on seeds for
grad-dot and ICL instead, which have n=1 and whose comparison is unreplicated.

Source: https://arxiv.org/html/2405.12186v1

### 7.6 Deck

Weekly-meeting deck: https://claude.ai/code/artifact/0e44ecd3-7ca5-43ed-948f-dc6493ba0a4a
Regenerate with `python3 build.py && python3 slides.py && python3 assemble.py` in the
session scratchpad; slides fill themselves in from `generative_comparison.json` and
`three_way.json` as results land.

### 7.7 Three-way method comparison — all three estimators, 6,400 documents

`three_way.json`, same corpus, same query set (`america_attr_target`, mean-aggregated),
same final checkpoint, all three arrays oriented through `_oriented`.

| pair | Spearman | Jaccard@200 |
|---|---|---|
| EK-FAC ↔ grad-dot | **0.628** | 0.133 |
| SOURCE ↔ EK-FAC | 0.411 | 0.166 |
| SOURCE ↔ grad-dot | **0.245** | 0.067 |

**The two single-checkpoint methods agree with each other more than either agrees with
SOURCE.** That is the expected shape — EK-FAC is grad-dot plus a preconditioner, and with
the document build removed they now differ by *nothing else*: same checkpoint, same
on-the-fly gradients, same modules, same query. So 0.628 measures what the EK-FAC
curvature correction changes, and 0.245/0.411 measure what the trajectory changes.

⚠️ **Ranking agreement is not evidence of correctness** (same caveat as
`compare_source_ekfac`). The relevant fact against it is §7.2: on the one causal removal
test, EK-FAC beat SOURCE (+0.120 vs +0.040 over random). grad-dot has **not** been
through a removal arm — that is the obvious next buy, and it is the arm that would say
whether the curvature term earns its cost at all.

⚠️ Also unresolved: `CLAUDE.md` §5.3's note that EK-FAC's category structure is 92% surface
lexical overlap with the query. grad-dot has no preconditioner at all, so it is the
natural test of whether that confound is the curvature's doing or the dot product's.

### 7.8 Running grad-dot (for the next session)

```bash
# 1. Always smoke first — 200 docs, ~1.4 min, ~$0.10. Writes to graddot_smoke/,
#    never to graddot/, so it cannot displace a real store.
.venv/bin/modal run --detach tda/modal/bergson_app.py --action graddot_smoke
# 2. Full corpus — 6,400 docs, 10.4 min on H100:2, ~$0.80.
.venv/bin/modal run --detach tda/modal/bergson_app.py --action graddot
# 3. Three-way agreement (raises if grad-dot is missing, failed, or short).
.venv/bin/modal run tda/modal/bergson_app.py --action three_way
```

Landed run: `bergson/cheese/graddot/graddot_cheese8b_A_america-attr-target_20260909-0552`.

**Four things to keep true when reusing this.**

1. 🔴 **Do not add a document `build` step back.** It is what killed the first two
   runs (§H8) and it is 2.15 TB at 8B / 14.2 TB at 32B. `score` streams documents and
   needs only the query index. `CLAUDE.md` §5.1 carries the arithmetic.
2. **Orientation is the same as EK-FAC's.** The store records
   `higher_is_better: true`, so `_oriented` negates it; both go through the same path
   in `compare_three`, and both come out **proponent-positive**. Any new sort must say
   which end it takes (§H5, §H7).
3. **Matching is the whole point.** grad-dot is only interpretable against EK-FAC
   because they now share checkpoint, query store, modules and gradient computation.
   Changing `which`, `aft_run` or `token_batch_size` on one arm alone silently turns a
   method comparison into a configuration comparison.
4. **Other settings**: `graddot_cheese(index=..., which=..., aft_run=...)` is generic
   over corpus and query set; nothing in it is cheese-specific except the defaults.
   For 32B philosophy, expect the scoring pass to scale with corpus tokens (13,201 docs
   vs 6,400) and model size — but *not* with storage, which is now 8 bytes/document.

**Two older cautions this run settles or changes.**

- ✅ **The "both sessions must agree on the tokenization unit" worry is resolved by
  construction.** grad-dot indexes `bergson/cheese/msm_A/dataset` — the same
  pre-tokenized, per-document, 6,400-row store SOURCE and EK-FAC use
  (`input_ids_sha256_16: 8cc0d17d079d8212`, `mode: document-LM`, max_length 4096).
  There is no second tokenization to reconcile.
- ⚠️ **The streaming fix forfeits one diagnostic.** `norm_confound_report` needs the
  per-document gradient *matrix*, which only `build` produced — so the matched
  successor to the old **0.785** gradient-norm confound figure is **not available**
  from this run, and that 0.785 (different setting, §2) must not be quoted against it.
  Do not rebuild the index to get it. The cheap substitute needs no gradients:
  correlate `|score|` against the dataset's `length` column, which captures the length
  half of the confound. Unmeasured as of this writing.

**What is NOT done**: grad-dot has no removal arm. Every causal claim in §7.2 covers
SOURCE and EK-FAC only. See §7.7 for why that arm is the informative next buy.
