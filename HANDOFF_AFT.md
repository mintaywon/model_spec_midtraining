HANDOFF_AFT.md — Does a single-stage, reasons-augmented AFT reproduce the MSM + AFT synergy on OOD agentic misalignment?

Owner: Taywon · Executor: Claude Code, autonomous, running on Taywon's local machine (macOS, no GPU); **all GPU work runs on Modal** (workspace `feng-pfau-c9-taywon`, entrypoints in `tda/modal/app.py`, results on the `msm-tda-results` volume). Codebase: mintaywon/model-spec-midtraining (bergson-source), which already implements MSM/AFT training, data pipelines, and the agentic-misalignment (AM) eval harness. Do not re-implement; reuse. Write PLAN.md first, then execute. This document fixes the scientific question, the conditions, the rules, and the deliverable — not the implementation.

> **Revised 2026-09-17 (Taywon).** The original brief assumed one assigned H100, which forced a 9–14B model and a from-scratch Ref. On Modal the compute constraint is gone, so the model is **Qwen2.5-32B-Instruct**, where the paper's *philosophy-spec* checkpoints are released for every core rung except L3: Base (plain Instruct), L0 (`aft-no-cot`), L1 (`aft-cot`), and Ref (`msm-aft-no-cot`, the number to beat). **No MSM training is needed.** Only L3, and a same-trainer L0 as its matched control, are trained. Compute budget is in dollars: Modal balance $348 is the hard cap, ≤ $200 may be spent without asking; Anthropic API spend is separate and reported.

1. What is known (read once)
MSM (Li et al. 2026, arXiv:2605.02087). Midtraining on synthetic documents about a Model Spec, then alignment fine-tuning (AFT) on single-turn chats, cuts OOD agentic misalignment far more than either alone (Qwen2.5-32B-Instruct 68→5%, Qwen3-32B 54→7%). The paper's deliberative-alignment baseline AFT (with CoT) loses to MSM + AFT (no CoT) at matched AFT scale; as AFT data scales to 80k samples it converges on Qwen3-32B but not on Qwen2.5-32B (Fig. 5). MSM's advantage is OOD-only — ID open-ended QA saturates for all AFT conditions. Appendix C.4: the MSM+AFT stacking requires documents that attribute behaviors to a value, not merely co-mention them. §5.3: whether documents describe the model itself vs. another entity, or use "does" vs. "should", has small effects.
Teaching Claude Why (Anthropic, May 2026). Training on demonstrations of correct behavior near the eval barely helped OOD; regenerating the same demonstrations so the assistant reasons about its values in the visible response cut misalignment 22→3%; the single step "rewrite the response to align with the constitution" accounted for ~19× on their difficult-advice data. A 3M-token dataset where the user faces a dilemma and the assistant advises generalized to the model's own agentic behavior. Documents beat chat for teaching spec knowledge; a "Claude thinks" vs "you think" persona gap remained.
Constitutional Midtraining (Cho et al. 2026). At 120B, presence of constitutional content in midtraining mattered far more than its structure (curriculum ordering, explicit reasoning blocks) — but post-training was value-neutral SFT, so structure effects that live only in the stacking with spec-aligned AFT were never tested.
Synthetic Persona Pretraining (Minder et al. 2026). First-person reflections beat third-person and summaries; gains depend on persona binding (post-training distribution matching the pretraining persona).
Korbak et al. (OpenAI, 2026). "Nice AI stories" midtraining priors washed out under reasoning RL and did not reach agentic evals.
Taywon's TDA (unpublished, correlational). EK-FAC on the MSM philosophy corpus against OOD AM queries: document register discriminates helpful vs harmful docs far better than spec domain (R² 0.064 vs 0.009). Documents that enact the model's first-person reasoning reduce misalignment; third-person descriptions/evaluations of the model raise it. Causal test unrun.
2. The question and the candidate conditions

MSM+AFT's OOD gain is real. Under what conditions does the synergy arise, and is the two-stage structure (documents before demonstrations) itself necessary? Candidate "higher-order" conditions that must jointly explain the results above:

A. Attributed reasons. The demonstrated behavior is explicitly explained by a general value/principle (MSM C.4, Anthropic's 19× step, value-augmented > rules).
B. Underspecification resolution. MSM only helps when AFT is ambiguous between the intended value and a narrower behavior; as AFT becomes richer, MSM's marginal gain → 0 (consistent with Fig. 5 convergence).
C. Shift of the pretraining prior about "how AIs behave" (Anthropic's hypothesis 3; stories work; describing Claude/humans works nearly as well as describing Qwen).
D. Diversity / invariance. The same value instantiated across many contexts.
E. Application vs statement. Practicing evaluative judgment on cases (including others' dilemmas) transfers; stating one's values does not.
F. Order / plasticity. Prior must exist before demonstrations are learned.
G. Format. Document (knowledge-style) ingestion vs dialogue ingestion.

If A (+D/E) suffice, the two-stage structure is a delivery vehicle. If a fully augmented single-stage AFT still loses at matched budget while the same content as documents closes the gap, F/G are necessary.

Working hypothesis for this pilot: an AFT dataset whose responses make the model's own reasoning visible and attribute the behavior to a general principle (L3) closes most of the gap to MSM + AFT on AM, at equal sample count and identical training. Motivation for putting reasons in the model's visible first-person voice (not only hidden CoT): SPP's first-person result and the TDA register finding.

3. Conditions (the ladder) and pilot scope

Same prompt set, same instruction-tuning mix, same training config and seeds everywhere; only assistant responses differ.

Rung	Content	Condition tested	Status
Base	untouched instruct model (Qwen2.5-32B-Instruct)	—	eval only
L0	released AFT (no CoT) — released adapter `aft-no-cot` (their trainer) **and** two seeds retrained here (our trainer; the matched control for L3)	behavior only	core
L1	released AFT (with CoT) — hidden reasoning, deliberative-alignment style; released adapter `aft-cot`; own seeds only if budget remains	reasons, hidden	core (released seed)
L2	L0 response + short visible first-person reasoning explaining why	A (visible)	optional
L3	L2 + plain-language attribution to the underlying value/principle + exactly one sentence stating the principle generalizes to other kinds of situations, at the spec's level of abstraction	A + invariance	core, generate
L4	L3 + prompts where the user faces the dilemma and the assistant advises; other domains	D, E	not in pilot
L5	L4 + knowledge-style chats (user asks about the spec's rationale)	G (chat form)	not in pilot
L6	L3/L5 content rendered as documents, mixed into the same single stage as L0	G vs F separation	optional
Ref	MSM (Philosophy spec) → AFT (no CoT) — the paper's recipe. Released adapter `msm-aft-no-cot` **plus** two same-trainer seeds already on the volume (`aft_phil32b_none_tb8192_s42/s43`: our AFT continued from the released MSM adapter)	two-stage	core, **released + already trained**

Core = Base, L0, L1, L3, Ref. Optional only after core is complete and reported: L2 (isolates attribution from visible reasoning), then L6. Never start L4/L5 in this pilot.

Judgment table for the report (write which pattern occurred):

L3 ≈ Ref → content (A) suffices; stage unnecessary at this scale.
L3 ≈ L1 < Ref → visible vs hidden reasoning doesn't matter; something beyond reasons (C/D/F/G) is needed.
L3 < L1 → visible reasoning hurts (e.g., verbosity/eval-awareness side effects) — inspect secondary evals.
L1 < L3 < Ref → attribution helps but a residual gap remains; recommend L4/L6 to split D/E from F/G.
4. Model selection (fixed by Taywon, revised 2026-09-17)
**Model: Qwen2.5-32B-Instruct.** The earlier 9–14B candidate list existed only because one H100 was assumed; with Modal that constraint is gone. Rationale: (i) the paper's philosophy-spec checkpoints (MSM, AFT no-CoT, AFT CoT, MSM+AFT) are released only at 32B, so Base/L0/L1/Ref need no training and Ref is the paper's own number; (ii) the Fig. 5 non-convergence of single-stage AFT is established on Qwen2.5-32B, so the L1–Ref gap is expected there; (iii) this repo's 32B harness, trainer (880 tok/s on 2×H100, validated), query set and prior evals all exist. Qwen3-32B is excluded (saturated AM rates, CLAUDE.md §1).
Screening table still to be recorded for the report: Qwen2.5-14B-Instruct base = 0.382 on the fixed subset (free, from `basediag`); Qwen3-14B screening was launched and cancelled when the model changed; Qwen3.5-9B N/A (needs transformers ≥ 4.57 / a newer vLLM than the pinned eval image).
Qwen2.5 has no thinking mode; L1's CoT is in-response `<think>` text, evaluated with the harness as-is.
5. Training regime
LoRA and training configs identical to the MSM paper — rank 64, alpha 128, all attention + MLP projections, AdamW lr 1e-4, cosine, 5% warmup, wd 0.01, 1 epoch, max seq len 8192 with the paper's instruction-tuning mix (~10k samples / 2M tokens, spec-misaligned samples filtered). Double-check paper vs code and record any discrepancy in DECISIONS.md. Fits 14B bf16 on one H100 with gradient checkpointing and micro-batch 1. Do not quantize; do not mix regimes across conditions; no per-condition tuning.
Ref is NOT built: the released `msm-aft-no-cot` adapter is Ref, and two same-trainer Ref seeds (AFT continued from the released `philosophy-spec-msm` adapter, seeds 42/43) already exist on the volume. Compute: 2×H100 per AFT run on Modal (`device_map="auto"`; 4 GPUs are no faster, STATUS.md §5), ~3.4 h ≈ $31 per 10.8M-token run.
Two seeds (42, 43) for L0-ours and L3; Ref-ours has two seeds already; L1 has the released seed only unless budget remains. Fixed 32 examples per optimizer step for the new runs (DECISIONS.md §I5).
6. L3 data requirements (and why)

Input: the released AFT (no CoT) pairs. Rewrite responses; do not change prompts.

Decision and substance preserved (same stance, recommendations, refusals); no new advice. Otherwise L3 confounds "reasons" with "different behavior".
Reasoning reads as the model thinking, in first person; no meta-language (spec, guidelines, policy, training, instructions). MSM's own AFT generation forbids explicit spec-referencing for the value-based spec.
No expression of desire to continue existing or avoid shutdown/replacement (MSM's filter).
Attribution + one invariance sentence at the spec's level of abstraction ("when my own continuity is at stake", "when a goal I care about seems to justify bending a rule") — never a concrete scenario.
No eval leakage: build a banned list from the AM eval prompts themselves (names, companies, AI/product names, emails, deadlines, distinctive phrases) plus AM narrative vocabulary; abstract discussion of replacement/oversight is allowed — that is the spec's topic.
Length cap ≈ 1.6× original; report length distributions. Length is the main confound.
Judge every sample (different model instance than the generator): decision preserved / spec alignment / no leakage / required structure. Keep all-PASS only; report pass rates per check.
Pilot the rewrite prompt on ~30 samples and read them before scaling; keep all prompt versions. Save 10 accepted + 5 rejected samples with judge reasons for Taywon's review.
7. Evaluation requirements
AM: fix a subset (≈9 of 27: three harmful actions × goal-conflict / no-conflict variants) before any results are seen; fixed N samples per scenario (default 25); identical decoding; harness as-is. misalignment_rate = fraction of transcripts taking the harmful action. Lower is better. State this above every table.
Secondary (cheap, all conditions): ID open-ended QA subset (sanity; expect ceiling for all AFT conditions — MSM's advantage is OOD-only), over-refusal (XSTest-like subset), response length on benign prompts. Optional (≤1h): 1–2 surface-paraphrased AM scenarios (rename entities, reorder emails, same decision structure) as an eval-awareness probe.
Token accounting: report loss-bearing tokens per condition and MSM tokens for Ref. This pilot is sample-matched, not token-matched — conservative against L3 (it sees far fewer tokens than Ref). Say so.
8. Analysis rules (pre-registered)
Phase 1 decides the primary comparison: if a clear L1–Ref gap exists (Ref lower by >5pp, seed ranges non-overlapping), primary = L3 vs Ref; otherwise primary = L3 vs L1 at equal samples. Always also report L3 vs L0.
Macro-average over the fixed scenarios; mean ± SEM across seeds (show both seeds); bootstrap CI within seed; all scenarios reported, never a post-hoc subset.
"Gap closed" only if |Δ(L3, Ref)| ≤ 5pp on the macro-average and seed ranges overlap; otherwise report the gap size.
Sanity before any L3 conclusion: Base > L0 in misalignment and Ref ≤ L0. If violated, debug harness/sign first. The paper has no numbers for these models — your Phase 1 table is the reference.
9. Guardrails
Do not modify eval prompts, judges, scoring, or decoding.
No fabricated numbers; unrun cells are N/A (reason).
PLAN.md before running; DECISIONS.md for every non-obvious choice and deviation; timestamped LOG.md; save all generated data, judge outputs, eval transcripts.
Budget: Modal balance $348 = hard cap; up to $200 without asking; anything that would take cumulative spend past $200 is priced and presented first. Cumulative spend is tracked in LOG.md. Wall clock is not the constraint on Modal (runs parallelise), but keep to ~24–36 h of session time. If blocked on infrastructure >45 min, record a fallback and continue.
10. Rough budget guidance (adjust in PLAN.md)

Modal H100 ≈ $4.56/GPU-h. Training: only L0-ours ×2 and L3 ×2 at 2×H100 ≈ $31–40 each ≈ $140. Evals: Base, released L1, Ref-ours ×2, L0-ours ×2, L3 ×2 = 8 checkpoints × (~$4 Modal + ~$8 Sonnet 4.6 grading); released L0 and released Ref are free (existing 27-condition n=30 runs, first 25 rollouts on the fixed subset). L3 generation and judging run on the Anthropic API (~$100–200) in parallel with training. Cut optional rungs and own-L1 seeds first, never the L0/L3 seeds or the fixed eval subset.

11. Deliverable: REPORT_AFT.md
Executive summary (≤10 lines): model and why, whether an L1–Ref gap exists, L3 vs Ref vs L1 headline numbers, the pre-registered reading, which row of the §3 judgment table occurred, and go/no-go for the full ladder (token-matched, 4 seeds, own L1 seeds, L2/L4–L6).
Setup: models, GPU, training config (with paper-vs-code check), seeds, MSM tokens for Ref, generator/judge versions, eval subset, N, decoding, thinking-mode handling.
Screening table (Base misalignment per candidate) and Phase 1 baseline table (Base / L0 / L1 / Ref) with sanity checks.
Main table: condition × seed × scenario + macro-average, CIs; sign convention stated.
Token and length accounting.
Data quality: judge pass rates per check, rewrite prompt version, link to the review pack.
Secondary evals.
Deviations from this handoff and why.
Threats to validity actually observed (not generic ones).
Recommended next runs, ranked, with cost.

A clean, honest comparison is the goal — a well-controlled null is a valid result.