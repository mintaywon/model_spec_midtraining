# Plan: SOURCE (approximate unrolling) via bergson

**Status**: proposed 2026-09-02, not started.
**Relationship to the other docs**: `CLAUDE.md` is the durable brief, `STATUS.md` the
live state, `inventory.md` the asset detail. This file is a *work plan* for one
addition — a second influence estimator — and folds into `STATUS.md` §2/§5 once work
begins.

---

## 0. Why bergson, and what it costs

`tda/influence/` currently implements single-checkpoint grad-dot (TracIn-final) with a
LoGra-style projection, per `CLAUDE.md` §5.1's "ship the fallback first". Bergson
(EleutherAI, v0.26.2) adds a maintained **SOURCE** implementation — approximate
unrolled differentiation (Bae et al., arXiv:2405.12186) — which is the method built for
**multi-stage, non-converged** training pipelines. That is exactly MSM→AFT, the setting
`CLAUDE.md` §2(1) deferred to Phase 2 for lack of a method.

SOURCE sits between single-checkpoint influence functions and full unrolling: roughly
"an influence function averaged over several checkpoints", with EK-FAC per segment and
an AdamW-preconditioned variant (Bae et al. App. C) matching how AFT was trained.

### What SOURCE demands that we do not currently have

| Requirement | Where it bites |
|---|---|
| ~6 checkpoints spanning the training run | Released MSM assets are **final adapters only** |
| `optimizer.pt` in each checkpoint dir (`use_adam_preconditioner`) | Never released |
| `hessian_cfg.ev_correction: true` | Asserted by the pipeline; EK-FAC is mandatory, not optional |

**Consequence: bergson-SOURCE cannot run on released checkpoints at all.** It is hard-blocked
on a trainer, which promotes `STATUS.md` §4b item #7 from "necessary" to "first".

---

## 1. Findings from reading the source (v0.26.2)

### 1.1 `bergson/magic/trainer.py` is a package location, not an algorithm

`Train.execute()` and `Magic.execute()` both call `run_magic()` (`cli/commands.py:271`,
`:132`), which is misnamed — it is "train, then *optionally* score and validate".
`Train` passes `validate=False, score_path=""`. Bergson's own SOURCE examples are
`steps: [train, approxunrolling]` (`examples/compare_smollm2_8k/source.yaml`): the
SOURCE pipeline is designed to consume this trainer's checkpoints.

MAGIC-specific machinery (`metagrad_step`, `Trainer.backward`, `BackwardState`, the
double-backward graph) is gated behind `trace=True`; a plain training step runs
`trace=False` and is an ordinary forward/backward/step loop (`trainer.py:385`).

**But its defaults are metagradients-flavoured and must be overridden:**

| | bergson default | standard AdamW / TRL | action |
|---|---|---|---|
| `adam_beta1` / `adam_beta2` | 0.95 / 0.975 | 0.9 / 0.999 | override |
| `eps_root` | 1e-8 | *no such term* | **set 0.0** |
| `train_mode` | `False` (eval mode) | train mode | leave — harmless here |
| optimizer | functional `torchopt.adamw`, params on meta device swapped from `TrainerState` | `torch.optim.AdamW` | validate numerically |

`eps_root` is the trap: torchopt puts it **inside** the square root —
`sqrt(v̂ + eps_root) + eps` (`magic/optim.py:176`) — so the default contributes
`sqrt(1e-8) = 1e-4` to the denominator, not negligible against LoRA second moments at
lr 1e-4. `torch.optim.AdamW` has no equivalent. Bergson's own `IndexConfig.optimizer_state`
docstring already warns about this for TrackStar.

`train_mode: False` is safe for us specifically: `STATUS.md` records LoRA dropout 0.0 and
no architectural dropout in Qwen2.5/Llama-3.1, so the eval-mode forward is identical.

### 1.2 Bergson's trainer can be the project's only trainer

`DataConfig` falls through to vanilla next-token prediction when neither
`completion_column` nor `conversation_column` is set, with `chunk_length` for packing
(`data.py:838`). That is **H5's document-LM mode for free**. Recommendation: drop the
separate TRL-based `tda/retrain/` from the plan unless 32B throughput forces it back.

### 1.3 PEFT is supported; SOURCE × PEFT is untested

`setup_model_and_peft` (`utils/worker_utils.py:189-234`) detects an adapter dir via
`PeftConfig.from_pretrained` and restricts gradient collection to adapter modules.
`export_checkpoints` calls `save_pretrained` on the PeftModel, so LoRA checkpoints
round-trip into the form SOURCE loads.

**But no test in `tests/` covers approx-unrolling with LoRA.** Fragile spot:
`approx_unrolling/adam_preconditioner.py::_match_params` suffix-matches `<module>.weight`
against param names, and LoRA gives `...q_proj.lora_A.default` → `base_model.model....lora_A.default.weight`.
`utils/load_from_optimizer.py` has adapter-suffix handling, so it has been thought about
— but assume a patch, and probe it early and cheaply (§2, Stage 0.2).

### 1.4 Our span queries need the pre-tokenized path, and it exists

Bergson's chat tokenizer masks **whole assistant messages** (`data.py:849-908`). It
cannot express "just the harmful tool-call span inside a longer assistant message" —
the entire point of the 263/263 span finding in `tda/evals/spans.py`.

`tests/test_pretokenized.py` confirms a dataset carrying explicit `input_ids` + `labels`
passes through untouched. So we hand bergson exact token/label arrays from our own
validated masking. This also makes the assistant-vs-full-sequence convention (open
question #3) ours to set rather than bergson's.

### 1.5 Two scale traps, both with mitigations

> **MEASURED 2026-09-02 (supersedes the estimates below).** A completed SOURCE
> run on Qwen2.5-0.5B + LoRA r=64 (7 projections, 6 ckpts, 3 segments, fp32)
> consumed **98.6 GB** of factors. Storage per checkpoint is
> `Σ_modules (d_in² + d_out²)`, and the pipeline holds 6 raw + 3 aggregated +
> 3 eigenvector sets at once (~12×):
>
> | model | per ckpt | pipeline total | note |
> |---|---|---|---|
> | Qwen2.5-0.5B | 7.5 GB | **98.6 GB** | measured; the arithmetic checks out |
> | Llama-3.1-8B, all 7 proj | 98.5 GB | ~1.2 TB | infeasible |
> | Qwen2.5-32B, all 7 proj | 648 GB | ~7.8 TB | far infeasible |
> | Llama-3.1-8B, **attention-only** | 13.2 GB | **~158 GB** | workable |
> | Llama-3.1-8B, attention-only + bf16 | 6.6 GB | **~79 GB** | comfortable |
>
> **The MLP factors are ~86% of the total** — `gate/up/down` carry the 14336²
> (8B) and 27648² (32B) terms, and LoRA does not shrink them: KFAC factors are
> sized by the layer's in/out dims, not by the adapter rank.
>
> **Decision: attention-only + `hessian_dtype: bf16` for the cheese run**, with
> the restriction stated plainly ("influence via the attention-LoRA subspace of
> the AFT stage") and tested against the all-module grad-dot pipeline in Stage
> 4.1. If Spearman is high the restriction is benign; if not, that is itself the
> result. Even the 0.5B run hit "No space left on device" in a sibling
> container, so `ephemeral_disk` must be set explicitly.

**EK-FAC factors are sized by layer in/out dims, not LoRA rank.** `lora_A` contributes a
`d_in²` activation covariance and `lora_B` a `d_out²` gradient covariance, so adapting a
module costs the same factors as full-parameter KFAC on it. At Qwen2.5-14B with all 7
projections that is ~150 GB fp32 per checkpoint set (the 13824-dim MLP factors dominate),
×3 segments of eigenvectors. Attention-only (`filter_modules`) cuts it to ~31 GB. 32B is
roughly 4× worse. **Mitigation: measure at 8B first, decide module scope for 32B from data.**

**Query gradients are stored unprojected** — `pipeline.py` forces `projection_dim = 0`
for the query build. With `query_aggregation: none` and 500 queries that is ~275 GB at
14B. **Mitigation: `query_aggregation: mean` for the headline profile** (which is what
H1's "per-sample influence profile" actually needs) plus a 32–64-query `none` run for
per-query robustness and LDS.

**Compute**: SOURCE makes ~18 full fwd/bwd passes over the training set (6 covariance +
6 λ + 6 scoring) versus grad-dot's 1, plus one query pass.

### 1.6 Environment

Bergson requires `transformers>=5.0`, `torch>=2.5`, `peft>=0.17`, `torchopt`. We are pinned
at `transformers==4.51.3` / `peft==0.15.2` for the vLLM eval path. **Separate Modal image**
— no conflict, since the bridge hands over datasets on disk and there is no in-process interop.

---

## 2. Plan

Decisions locked with the user 2026-09-02: **cheese (8B) first, then 32B philosophy**;
**run both single-stage and multi-stage on cheese**; **port to 32B only if SOURCE works
and beats grad-dot**.

### Stage 0 — Environment and smoke test  (~$5)  ✅ **DONE**

**Outcome: SOURCE × LoRA works end-to-end.** All 8 pipeline steps completed on
Qwen2.5-0.5B + LoRA r=64 and wrote `scores.bin`. Three findings:

1. **One upstream bug, patched** — `build_segment_preconditioners` called
   `AutoConfig.from_pretrained(checkpoints[0])` on what is an adapter dir,
   killing SOURCE at step 5/8 with "Unrecognized model ... should have a
   `model_type` key". The reference model is only a fallback index→name map plus
   a square-grid orientation lookup; bergson's trainer records `param_name` and
   LoRA grids are non-square, so it is skipped on the PEFT path.
   `tda/influence/source/patches/`.
2. **bergson tracks LoRA params in the optimizer state** — 336 second moments
   (24 layers × 7 modules × {A,B}) — so `use_adam_preconditioner` (the variant
   matching AdamW-trained AFT) is available on the PEFT path.
3. **transformers 5.x changed `apply_chat_template(tokenize=True)`** to return a
   dict rather than a list of ids. `len()` then counts *keys*, silently
   corrupting every prefix computation in `masking.py`. It surfaced as
   "input_ids (2) and labels (1)" only because `MaskedSample.__post_init__`
   asserts the invariant. Both shapes are now handled and regression-tested —
   this repo runs transformers 4.51.3 (eval image) and ≥5.0 (bergson image) at
   once.

- **0.1** Bergson Modal image + a `msm-tda-bergson` volume for EK-FAC factors (large,
  disposable, kept off `msm-tda-results`). Pin `bergson==0.26.2`.
- **0.2 Toy LoRA SOURCE smoke test.** Qwen2.5-0.5B-Instruct + LoRA r=64 on the 7
  projections, ~200 chat samples, 6 checkpoints, `segments: 3`, `method: kfac`,
  `ev_correction: true`, `use_adam_preconditioner: true`. **Purpose: find where
  SOURCE × PEFT breaks, for $5 instead of $250.** Deliverable: a vendored patch set (or
  fork branch) plus an upstream issue if warranted.
- **0.3 Pre-tokenized bridge** `tda/influence/bergson_data.py` — emit `input_ids`/`labels`
  datasets from existing `masking.py` + `spans.py`. **Equivalence test**: on cheese chat
  data our bridge's labels must equal bergson's own `tokenize()` labels exactly. That
  proves the bridge in the setting where both work, before 32B where only ours can
  express the span.

**Gate**: SOURCE produces finite scores on a LoRA model end-to-end.

### Stage 1 — Cheese trainer and trajectories  (~$25)

Assets (`inventory.md` §5b(i)): `chloeli/aft-llama-cheese` (5,129 ex), MSM-only
checkpoints for both specs, released MSM+AFT checkpoints for both, AFT-only control.
Base `meta-llama/Llama-3.1-8B` (gated; already confirmed accessible).

- **1.1** Retrain arm A: from `llama-3.1-8b-pro-america-spec-msm` on the shared AFT set.
  Match the recipe — LoRA r=64 α=128, 7 projections, AdamW lr 1e-4 cosine, 1 epoch,
  max seq 8192, **`eps_root: 0.0`, betas 0.9/0.999**. `save_mode: interval` sized to
  yield 6 checkpoints including the final; `save_optimizer_state: all`.
  - **Gate**: per-tensor cosine ≈ 0.99 against the released
    `llama-3.1-8b-pro-america-spec-msm-cheese-aft` (the test `inventory.md` §3–4 used),
    plus matching rate on `chloeli/pro-america-political-opinions`.
  - Train under **both** masking conventions and keep whichever matches — this closes
    `STATUS.md` **open question #3**.
- **1.2** Same for arm B (`pro-affordability`) and the AFT-only control. Three arms give
  the H1-shaped contrast: one AFT dataset, three initialisations.
- **1.3 Seed noise floor** — 2 extra AFT re-runs from the *same* MSM(A) checkpoint
  differing only in data order. `CLAUDE.md` §5.3 calls this "not optional"; it has been
  unaffordable until a trainer existed. ~$10 at 8B.

### Stage 2 — Single-stage SOURCE on cheese  (~$60)

- **2.1** Query sets from the released eval sets (`pro-america-political-opinions`,
  `pro-affordability-item-comparisons`). Whole-response targets, so bergson's native
  `prompt_column`/`completion_column` path works directly. Log the contrastive twin
  (`CLAUDE.md` §2(2)) alongside single-sided logp.
- **2.2** `approxunrolling` per arm: 6 ckpts, 3 segments, kfac + ev_correction, Adam
  preconditioner, **all 7 modules** — 8B is small enough, and this run is where we
  **measure real factor storage and wall time** to calibrate the 32B module-scope decision.
  `query_aggregation: mean` headline + a 32–64-query `none` run.
- **2.3** Convert scores → our parquet; run `tda/influence/scoring.py` unchanged
  (Spearman, top-k Jaccard k∈{50,200,1000}, Gini, top-k mass, `norm_confound_report`).
- **2.4** H1-shaped result: profile correlation across the three arms, read **against the
  Stage-1.3 noise floor**.

### Stage 3 — Multi-stage MSM→AFT SOURCE on cheese  (~$60)

The Phase-2 question, reachable because cheese publishes MSM corpora *and* MSM-only
checkpoints *and* the shared AFT set.

- **3.1** Retrain the MSM stage with bergson's trainer in document-LM mode
  (`chloeli/msm-llama-pro-america`, 6,400 docs, plain LM loss, `chunk_length` packing),
  with checkpoints. **Gate**: reproduce `llama-3.1-8b-pro-america-spec-msm` (cosine).
- **3.2** Union index corpus: MSM documents + AFT samples.
- **3.3** Bergson assumes one dataset across all segments. We exploit its **per-segment
  score artifacts** (`<run>/segment_l/scores/`) and sum only the segments where each
  dataset was actually trained — implemented in `tda/influence/source/aggregate.py`
  with a unit test on synthetic scores.
  - **Choose segment boundaries to align with the stage boundary**, so no segment mixes
    MSM and AFT data and the masking is *exact* rather than approximate. This is a config
    choice and must be stated in any write-up.
- **3.4** Result: which MSM documents carry the cheese preference, corpus A vs B. This
  upgrades A2 from "toy value / trainer validation" to a real methods result.

### Stage 4 — Validation and the 32B gate  (~$25)

- **4.1** Cross-check `Spearman(SOURCE, grad-dot)` using existing `extract.py` on the
  same arms — `CLAUDE.md` §5.1 asks for exactly this comparison.
- **4.2** `CLAUDE.md` §5.4 counterfactual removal on cheese: remove top-k / bottom-k by
  SOURCE and by grad-dot, retrain, measure held-out delta against random-k (2 seeds).
  First time this has been affordable.
- **4.3** Optional: `bergson validate` for LDS.
- **4.4 GATE (locked)**: port to 32B only if SOURCE runs clean **and** either
  `Spearman(SOURCE, grad-dot) < ~0.8` (the methods disagree, so the expensive one earns
  its keep) **or** SOURCE wins the removal test on cheese.
  - A negative result here is reportable per `CLAUDE.md` §1b and stops the spend.

### Stage 5 — 32B philosophy, gated  (~$500)

- Retrain both A1 arms with trajectories from the released philosophy MSM checkpoint and
  `chloeli/aft-no-cot-qwen2.5-philosophy-spec`.
- Module scope decided from Stage 2.2's measurements (attention-only if storage forces it).
- Reuse the existing 518 / 185 query sets through the pre-tokenized bridge.
- ~$250/cell SOURCE + retraining.

**Cheese total ≈ $175–250**, then 32B gated at ~$500.

---

## 3. Layout

```
tda/influence/
  bergson_data.py          # pre-tokenized bridge (input_ids/labels from masking.py + spans.py)
  source/
    configs/               # yaml per run
    aggregate.py           # per-segment score masking + parquet conversion
    patches/               # vendored bergson patches, if 0.2 finds any
tda/modal/bergson_app.py   # thin wrapper; separate image, separate factor volume
```

Existing `tda/influence/{extract,gradients,projection,scoring}.py` stay — grad-dot is the
cross-check, not a casualty. `scoring.py` is reused verbatim on SOURCE scores.

---

## 4. Risks

| Risk | Mitigation |
|---|---|
| SOURCE × PEFT untested upstream | Stage 0.2 finds it for $5 |
| bergson trainer ≠ TRL numerics (torchopt AdamW, `eps_root`) | Stage 1.1 per-tensor cosine gate against a released adapter |
| EK-FAC factor storage at 32B | Measured at 8B before any 32B commitment; `filter_modules` fallback |
| Unprojected query gradients | `query_aggregation: mean` headline; small `none` run for robustness |
| Multi-stage per-segment masking is our extension, not upstream | Stage-aligned segment boundaries make it exact; unit-tested |
| `transformers>=5.0` vs our pinned 4.51.3 | Separate image; datasets handed over on disk |
| Cheese is not a safety task | Framed as methods validation; the safety claim waits for Stage 5 |
