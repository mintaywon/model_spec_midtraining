# Phase 0 — Asset Inventory

**Date**: 2026-08-04 · **Scope**: everything Phase 1 (AFT-stage TDA) needs, per `CLAUDE.md` §3.
**Verdict**: **one blocking gap** — the AFT datasets for the factorial specs. Compute is *not* the bottleneck; data availability is.

---

## 0. Headline findings

| # | Finding | Impact |
|---|---|---|
| 1 | MSM-only checkpoints **are released** for Qwen2.5-14B for all three factorial specs (R, V+, R+) | ✅ **Phase 1 needs zero midtraining compute**, as hoped in §3 |
| 2 | All checkpoints are **PEFT LoRA adapters** (r=64, α=128, all 7 attn+MLP projections, 275,251,200 params, fp32, base `Qwen/Qwen2.5-14B-Instruct`) | ✅ Gradient space of §2(3) is directly available; no merged-weight surgery |
| 3 | **AFT continues the MSM LoRA adapter** (verified empirically, §4) | ✅ AFT-stage subspace is exactly the released adapter's params |
| 4 | Cross-paired checkpoints MSM(X)+AFT(Y), X≠Y, are **not released** (only matched X=X) | ⚠️ Must retrain or request — cheap (~1–2 GPU-h each) *if* we have the data |
| 5 | **AFT and MSM datasets for the R / V+ / R+ specs are NOT public.** Only the `philosophy` spec's corpora are | 🔴 **BLOCKING — all of H1–H4 depend on this** |
| 6 | **No training code in the repo** — data generation + evals only | 🔴 Blocks retraining *and* counterfactual validation (§5.4) |
| 7 | MSM corpus provenance is stripped to a single top-level `domain` field | ⚠️ Degrades H4; matters more for Phase 2 |
| 8 | Reasoning-category classifier (policy misuse etc.) not in the repo | ⚠️ Must implement for H3 (§4.4) |
| 9 | AM eval grid confirmed = **27 conditions** = 3 scenarios × (8 explicit goal values + `goal_type=none`) | ✅ Matches the brief; clean stratification for the dev/held-out split |

---

## 1. Repository

The working directory **is** the upstream repo (`mintaywon/model_spec_midtraining`, fork of `safety-research/model-spec-midtraining`). No separate clone needed.

```
spec/paper/       7 spec texts — includes rules_spec, value_augmented_spec, rules_augmented_spec  ✅
src/msm/          MSM synthetic-document generation (spec → domains → subdomains → doc types → docs)
src/aft/          AFT chat-data generation (domains → questions → dedup → responses → LLM filter)
evals/agentic_misalignment/   Inspect AI port, 3 scenarios + classifiers                          ✅
exps/             two data-generation shell drivers
src/utils/        token counting, similarity filtering
```

**Absent: any training code.** Nothing matching `lora|peft|trainer|sft` outside the AFT *data-generation* module. The released adapters were produced by code that was not open-sourced.

### Spec texts (all three factorial specs present ✅)

| File | Lines | Words | Role |
|---|---|---|---|
| `rules_spec.txt` | 35 | 598 | **R** — SP1–3, GP1–2 as bare prescriptions |
| `value_augmented_spec.txt` | 92 | 3,039 | **V+** — same rules + explanations of underlying values |
| `rules_augmented_spec.txt` | 127 | 2,941 | **R+** — same rules expanded into subrules, length-matched to V+ |
| `philosophy_spec.txt` | 60 | 2,184 | main-results spec (not the factorial) |
| `general_spec.txt` | 2 | 342 | |
| `pro_america_cheese.txt`, `pro_affordability_cheese.txt` | 72 each | | toy specs for the Llama experiments |

Verified: R and V+ share **byte-identical** "This includes:" bullet lists; V+ wraps them in prose explanation. The length-match between V+ (3,039 w) and R+ (2,941 w) checks out. This is exactly the factorial the brief describes.

---

## 2. Checkpoint coverage — Qwen2.5-14B, no-CoT

166 adapters published under `chloeli/`. Naming: `qwen-2.5-14b-<spec>-spec[-msm]-aft-<cot>`. The scheme carries **one** spec name, so cross-pairings are structurally unrepresentable — confirming they were never uploaded.

| Cell | HF repo | Status |
|---|---|---|
| Baseline (IT-mix only) | `qwen-2.5-14b-id-baseline` | ✅ |
| MSM(R) only | `qwen-2.5-14b-rules-spec-msm` | ✅ |
| MSM(V+) only | `qwen-2.5-14b-value-aug-spec-msm` | ✅ |
| MSM(R+) only | `qwen-2.5-14b-rules-aug-spec-msm` | ✅ |
| AFT(R) only | `qwen-2.5-14b-rules-spec-aft-no-cot` | ✅ |
| AFT(V+) only | `qwen-2.5-14b-value-aug-spec-aft-no-cot` | ✅ |
| AFT(R+) only | `qwen-2.5-14b-rules-aug-spec-aft-no-cot` | ✅ |
| MSM(R)+AFT(R) | `qwen-2.5-14b-rules-spec-msm-aft-no-cot` | ✅ |
| MSM(V+)+AFT(V+) | `qwen-2.5-14b-value-aug-spec-msm-aft-no-cot` | ✅ |
| MSM(R+)+AFT(R+) | `qwen-2.5-14b-rules-aug-spec-msm-aft-no-cot` | ✅ |
| **MSM(R)+AFT(R+)** | — | ❌ **needed for H1 + reproduction gate** |
| **MSM(V+)+AFT(R+)** | — | ❌ **needed for H1 + reproduction gate** |
| **MSM(V+)+AFT(R)** | — | ❌ needed for H2 |
| MSM(R+)+AFT(R), MSM(R+)+AFT(V+), MSM(R)+AFT(V+) | — | ❌ not needed for H1–H3 |

`-cot` twins exist for every released cell. Qwen3-14B (secondary model) has **identical coverage**. Qwen2.5-32B and Qwen3-32B additionally have `philosophy`/`general` cells and a data-scaling ladder (1k…80k). Per §2(4), Qwen3-32B is excluded.

`-baseline` and `-id-baseline` are the same thing (instruction-tuning-only LoRA, no MSM, no AFT); 14B uses the `-id-` name. This is the 0.51 reference cell.

**Reproduction gate (§4.3) status**: 3 of the 5 required cells are available. The two missing are cross-pairs — **the gate is blocked by the same gap as H1.**

---

## 3. LoRA configuration (identical across every checkpoint inspected)

```
r = 64, lora_alpha = 128, lora_dropout = 0.0, bias = none, use_rslora = false, use_dora = false
target_modules = [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
base_model = Qwen/Qwen2.5-14B-Instruct
```

672 tensors = 48 layers × 7 modules × {A, B}. **275,251,200 trainable params**, stored fp32. Matches `CLAUDE.md` §2(3) exactly.

Storage implications for the influence pipeline:

- One full per-sample gradient = 275M floats ≈ 550 MB in fp16 → **~5.5 PB for 10k AFT samples**. Projection is mandatory, as anticipated.
- JL-projected to 32k dims fp16 = 64 KB/sample → **~640 MB for 10k samples**. Comfortable.
- A dense 275M × 32k projection matrix cannot be materialised. Use the LoRA structure instead: per token, ∇_B = g_out·(Ax)ᵀ and ∇_A = (Bᵀg_out)·xᵀ are both **rank-1**, so a per-sample gradient is a sum of rank-1 terms. This is what LoGra (Choe et al. 2024) exploits — recommended path for the gradient-extraction task, with per-module seeded random projection as the simpler fallback.

---

## 4. Verified: AFT resumes the MSM adapter (not a fresh LoRA)

This was the key architectural unknown. Settled by range-fetching individual tensors (layer 0 `mlp.down_proj`) from four adapters and comparing:

| Pair (lora_A / lora_B cosine) | cos |
|---|---|
| MSM(V+) vs MSM(V+)+AFT(V+) | **+0.9959 / +0.9894** |
| MSM(V+) vs AFT(V+)-only | +0.0008 / +0.0094 |
| MSM(V+) vs MSM(R) | +0.0066 / +0.0031 |
| MSM(V+) vs id-baseline | −0.0003 / +0.0012 |
| AFT(V+)-only vs id-baseline | +0.0005 / −0.0129 |

Two conclusions:

**(a) AFT continues training the same LoRA weights, initialised from the MSM adapter.** So θ_final = base + (α/r)·B_f A_f, and the AFT stage moved (A, B) from (A_msm, B_msm) to (A_f, B_f). Gradients w.r.t. these params at the final checkpoint are *exactly* the AFT-stage subspace. This validates locked decision §2(3) and strengthens it. It also fixes the retraining recipe: **load MSM adapter, continue SFT** — not "merge and re-init".

Side observation: the AFT delta is small relative to the MSM delta (‖A‖ 7.5583 → 7.5672; the orthogonal component is ≈9% of the norm). AFT nudges rather than rewrites. This doesn't affect the method — influence uses gradients, not deltas — but it's worth remembering when interpreting effect sizes.

**(b) Different runs used different random LoRA inits** (cos ≈ 0 between MSM(V+) and MSM(R)). Two consequences for H1:

- Harmless for the analysis itself: H1 compares *per-sample influence scores* — scalars indexed by training sample — which are well-defined across checkpoints regardless of parameter-space alignment. No cross-checkpoint parameter comparison is ever needed.
- **But `CLAUDE.md` §5.3-H1's "(same seed)" is not achievable as written.** Each MSM checkpoint carries its own random init, so init is confounded with MSM condition and cannot be separated without retraining MSM ourselves. The right noise floor is therefore **two AFT re-runs from the *same* MSM checkpoint** (differing only in data order/dropout, init held fixed) — cheap, and it isolates exactly the nuisance we need to quantify.

---

## 5. Dataset coverage — 🔴 the blocking gap

Only **12** datasets are published, and the factorial specs are not among them.

| Dataset | Contents | Use to us |
|---|---|---|
| `chloeli/msm-qwen-philosophy-spec` | 13,201 docs, fields `{text, domain}` | ❌ wrong spec |
| `chloeli/aft-{cot,no-cot}-qwen{2.5,3}-philosophy-spec` | 9,963 examples, field `{messages}` | ❌ wrong spec |
| `chloeli/sft-it-mix` | the shared instruction-tuning mix, `{messages, source}` | ✅ **usable** |
| `chloeli/aft-llama-cheese`, `msm-llama-pro-{america,affordability}` | toy Llama specs | ❌ |
| `chloeli/{pro-america-political-opinions, pro-affordability-item-comparisons, spec-open-qa}` | chat evals | ➖ not Phase 1 |

**There is no MSM or AFT corpus for `rules_spec`, `value_augmented_spec`, or `rules_augmented_spec`.**

Phase 1 attributes influence *over AFT training samples*. Without AFT(R+) there is literally nothing to attribute. This blocks **H1, H2, H3 and H4** — not one of them degrades gracefully.

### What the philosophy datasets tell us about format (useful even though the spec is wrong)

- **AFT**: `{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}`. Single turn, no system prompt. 9,963 examples ≈ the "~10k" in §5.1. **No provenance fields** (no source domain, question id, or filter verdict) — so influence results could only be grouped by *content* clustering, not by generation metadata. Worth requesting the generation-side metadata too.
- **MSM**: `{"text", "domain"}` — only the **top-level domain** survives (8 values for philosophy). The pipeline in `src/msm/` generates a 4-level hierarchy (domain → subdomain → doc_type → doc_idea) with `meta.json` at each level, but **subdomain, doc type and doc idea are stripped from the published JSONL**. §3 of the brief explicitly asked to verify this; it did not survive. Directly degrades H4's "grouped by spec-section provenance" and matters more for Phase 2.
- **Size discrepancy worth resolving**: the philosophy corpus measures ~16.0k chars/doc × 13,201 docs ≈ **53M tokens**, roughly 2× the paper's stated "27M tokens each". Either the factorial corpora are smaller than the philosophy one, or the paper counts post-filtering/subsampled tokens. Ask.
- **IT-mix ambiguity**: `sft-it-mix` ships many splits — `train` (33,737), `train_short` (32,029), `train_clean`, `train_clean_nothink`, `train_80k`, `train_80k_clean`, `train_100k`, plus per-source splits (no_robots 9,500; tulu3_if 5,000; numina_cot 3,500; self_oss_instruct 3,500; apigen 3,500; smol_summarize 3,500; smol_constraints; lima 1,029; longalign 708; mmlu_binary 2,000; mmlu_explain 2,000). §5.1 assumes "~5k" IT samples mixed in. ~~**Which split, and at what mixing ratio, is undocumented.**~~ **Resolved 2026-09-17** (paper B.3/B.4 + measurement): §4–5 use Table 2 = 10,000 rows = `train_clean` × 0.691 per source, mixed into the AFT run 1:1 by rows with the spec data; paper token counts are assistant-only (2.14M). See CLAUDE.md §5.1.

---

## 5b. Complete pilot testbeds available today (no missing assets)

Two (checkpoint, training-data) pairings are *fully* released. Neither is the factorial we care about, but both let us build and validate the entire Phase-1 machinery before the R/V+/R+ data arrives.

### (i) Llama-3.1-8B "cheese" — **has H1's exact structure**

Same AFT dataset trained on top of two different MSM initialisations, plus controls — i.e. the H1 design in miniature, with every asset public:

| Role | Asset |
|---|---|
| Shared AFT data | `chloeli/aft-llama-cheese` — 5,129 examples, `{messages}` |
| MSM corpus A | `chloeli/msm-llama-pro-america` — 6,400 docs |
| MSM corpus B | `chloeli/msm-llama-pro-affordability` — 4,600 docs |
| MSM(A) only | `llama-3.1-8b-pro-america-spec-msm` |
| MSM(B) only | `llama-3.1-8b-pro-affordability-spec-msm` |
| **MSM(A)+AFT(cheese)** | `llama-3.1-8b-pro-america-spec-msm-cheese-aft` |
| **MSM(B)+AFT(cheese)** | `llama-3.1-8b-pro-affordability-spec-msm-cheese-aft` |
| AFT-only control | `llama-3.1-8b-cheese-aft` |
| Baseline | `llama-3.1-8b-baseline` |
| Eval sets | `chloeli/pro-america-political-opinions`, `chloeli/pro-affordability-item-comparisons` |

**Correction 2026-09-03**: there is a **third** cheese arm —
`llama-3.1-8b-pro-environment-spec-msm-cheese-aft` — so three MSM contents share
the cheese AFT set, not two (matching `CLAUDE.md` A2's "3 MSM contents"). Its MSM
*corpus* (`msm-llama-pro-environment`) is **not published**, so it can be used
for single-stage attribution from released checkpoints but **not** for
multi-stage, which needs the midtraining data. The account holds 166 models
including single-value specs over other domains (sweeteners, condiments, sauce,
bread, music).

**The synthetic identity dataset is NOT published** (verified 2026-09-03):
absent from the `source` column of all 19 `sft-it-mix` splits, no matching
dataset under the account, and the `id-baseline` checkpoints that are named for
it exist only for Qwen with no dataset behind them. It is 2,500 of the 13,500
cheese IT samples (~16% of IT tokens).

Same LoRA config (r=64, α=128, 7 projections). Base is `meta-llama/Llama-3.1-8B` — **gated on HF, needs license acceptance**, and note it is the *base* model, not Instruct.

Supports end-to-end: trainer validation, gradient extraction, projection, EK-FAC, an H1-shaped profile-correlation analysis, the AFT-seed noise floor, and §5.4 counterfactual removal. Does **not** exercise AM query-span extraction (evals are chat-preference, not agentic misalignment).

### (ii) Qwen2.5-32B `philosophy` — real AM task, real spec data

`chloeli/msm-qwen-philosophy-spec` + `chloeli/aft-{cot,no-cot}-qwen2.5-philosophy-spec` pair with `qwen-2.5-32b-philosophy-spec-msm-aft-{cot,no-cot}` (plus a 1k…80k data-scaling ladder). Single spec, so **no factorial and no H1** — but it is the only public place to validate AM-query influence on real spec data. 32B → 2×H100 for vLLM.

The two are complementary: (i) has the right *structure*, (ii) has the right *task*.

---

## 6. AM eval harness

Present and usable. Inspect AI task at `evals/agentic_misalignment/agentic_misalignment.py`; one `Sample` per condition, run with `--epochs N` for N rollouts.

**Grid = 27 conditions**, confirmed: 3 scenarios (`leaking`, `murder`, `exfiltration`) × 9 goal conditions — 8 explicit goal values (america/global, safety/acceleration, ethical/pragmatic, individualism/collectivism) plus `goal_type=none, goal_value=none` — all at `urgency_type=replacement`. Matches the brief's "3 harmful actions × 8 goal types" and gives clean stratification for the 14/13 split.

- `exfiltration` is the paper's addition to the original Anthropic eval.
- Scoring: `harmfulness_scorer` → per-scenario LLM classifier (`ExfiltrationClassifier` uses `prompt_version="v6"`), grader model overridable via `INSPECT_GRADER_MODEL`. Returns `harmful` plus a softer `classifier_verdict`.
- `prod: false` keeps the reasoning scratchpad (what §4.1 wants); `prod: true` removes it.
- The most recent commit vendored the classifiers to drop the `inspect_evals` dependency, so they are local and editable.

**Not present**: the paper-style **reasoning-category classifier** (policy misuse / SP3 self-preservation reinterpretation, self-preservation reasoning, respect-for-oversight). Only the three harmfulness classifiers exist. §4.4 requires this for H3 → must implement.

**Judge-cost note**: one grader call per rollout. 27 conditions × 300 epochs × ~6 checkpoints ≈ **~49k Sonnet calls** for a full reproduction sweep. Worth deciding epochs deliberately, and whether the dev split can run at lower epochs.

---

## 7. Environment readiness

| Item | Status |
|---|---|
| `safety-tooling` submodule | ❌ empty — never initialised (`git submodule update --init`) |
| `inspect-ai`, `torch`, `peft`, `transformers` | ❌ not installed (system Python 3.12.7 via anaconda; `uv` available) |
| `.env` (ANTHROPIC / OPENAI keys) | ❌ absent (`.env.example` present) |
| HF token | ✅ present at `~/.cache/huggingface/token` |
| GPU | ❌ **this machine is macOS — no local GPU** |

The brief assumes 2× H200. That has to be a remote box; provisioning it is an unlisted prerequisite.

---

## 8. Compute budget

The good news: **compute is cheap here.** AFT no-CoT is ~5M spec tokens + IT mix ≈ 7M tokens, LoRA-only, 1 epoch.

FLOPs ≈ 6 × 14e9 × 7e6 ≈ 5.9e17. At ~40% MFU on one H200 (~3e14 FLOP/s effective) → **~35 min/run**, call it **1–2 GPU-hours** with data loading, checkpointing and eval overhead.

| Work item | Runs | Est. GPU-hours |
|---|---|---|
| Cross-paired checkpoints for H1/H2 | 3 | 3–6 |
| AFT-seed noise floor | 1 | 1–2 |
| Counterfactual validation (§5.4) | 6 | 6–12 |
| Per-sample gradient extraction (~15k samples, fwd+bwd) | — | 2–4 |
| **Total Phase 1** | | **~15–25 GPU-hours** |

MSM retraining, if ever needed, is ~27–53M tokens ≈ 2–4 GPU-hours — also cheap. **The expensive resource is not GPUs, it is synthetic-data generation API spend** (§10).

---

## 9. Proposed `CLAUDE.md` amendments

1. **§2(3)** — record that released adapters confirm AFT *continues* the MSM LoRA (cos ≈ 0.99), so "only LoRA params changed during AFT" holds exactly; retraining = load MSM adapter, continue SFT.
2. **§5.3 H1** — replace "(same seed)" with the noise-floor definition from §4(b): MSM checkpoints have independent random LoRA inits, so the floor must come from two AFT re-runs off a *single* MSM checkpoint.
3. **§3** — record that MSM provenance did **not** survive publication (only top-level `domain`), degrading H4 / Phase 2 unless the author supplies richer metadata.
4. **§8 risks** — add the dataset-availability risk (currently the single point of failure) and the missing-training-code risk.
5. **§5.1** — pin the IT-mix split once the author confirms; note the current ambiguity.

---

## 10. Author asks, prioritised

**P0 — blocking; nothing in Phase 1 can start without these**

1. **AFT no-CoT datasets for `rules_spec`, `value_augmented_spec`, `rules_augmented_spec`** (Qwen2.5-14B convention), ideally with generation-side provenance (source domain, question id, filter verdict). ~10k examples each. *Without these, H1–H4 are all dead.*
2. **Which `sft-it-mix` split, and what mixing ratio,** was used in AFT training.
3. **Training code / configs** for the LoRA SFT stage: batch size, grad-accum, warmup, packing (yes/no), loss masking, seed — and confirmation that AFT resumes from the MSM adapter.

**P1 — saves ~4 GPU-runs and removes reproduction risk**

4. The **cross-paired Qwen2.5-14B no-CoT checkpoints** — at minimum MSM(R)+AFT(R+) and MSM(V+)+AFT(R+). These *exist* (Figure 14 reports their numbers) but were not uploaded.
5. Whether **multiple seeds** exist per cell, and the seed values.

**P2 — Phase 2 / H4 / verification**

6. **MSM corpora for R / V+ / R+** with full provenance (subdomain, doc type, doc idea) — and if possible the same provenance re-attached to the philosophy corpus.
7. Exact **Figure 14 numbers and SEMs** (we re-derived them from the figure; confirmation would firm up the gate's tolerance).
8. The **AM eval configuration used in the paper**: epochs per condition, temperature, grader model, `prod` flag, and how the 27 conditions were aggregated into a single misalignment rate.
9. Clarification of the **27M-vs-53M token discrepancy** in MSM corpus size.

**Fallback if the data cannot be shared**: regenerate with the in-repo pipelines. AFT regeneration is plausible (~$0.5–0.7k API per spec, ×3); MSM regeneration is not cheap (~$2–5k per spec) and would not be byte-identical to what trained the released checkpoints — which would break the clean "training data held literally fixed" logic that makes H1 worth doing. **Strongly prefer obtaining the originals.**

---

## 11. Bottom line

Everything that is *hard* is available: LoRA adapters in exactly the right form, MSM-only checkpoints for all three factorial specs (so zero midtraining compute), a working 27-condition AM harness, and a total compute bill under ~25 GPU-hours.

Everything that is *blocking* is a single missing artifact class: **the AFT datasets for the three factorial specs**, plus the training code needed to use them. Both are plausibly a short email away.
