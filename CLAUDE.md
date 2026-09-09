# Project Brief: Training Data Attribution on Model Spec Midtraining (Phase 1)

> ## ⚠️ READ [`STATUS.md`](STATUS.md) FIRST
> This file is the **durable brief** — research question, locked decisions, method.
> It deliberately does **not** track progress. `STATUS.md` holds what is built,
> what is measured, what is blocked, and what is next. Reading it first avoids
> rebuilding finished components or re-litigating settled questions.
>
> Asset-level detail lives in [`inventory.md`](inventory.md).
> Decision logs: [`SESSION_LOG.md`](SESSION_LOG.md) (TDA/eval session) and
> `DECISIONS.md` (bergson/SOURCE session) — two sessions write this repo.

**Audience**: Claude Code, implementing from scratch in a fresh repo.
**Author context**: AI safety researcher (MATS), experienced with influence functions (ACL 2025 TDA paper), alignment faking, agentic misalignment evals. Assume familiarity with LoRA, EK-FAC, HF transformers.

---

## 1. Research context (read carefully, this motivates every design choice)

Model Spec Midtraining (MSM; Li, Wichers, Price, Marks, Kutasov — arXiv:2605.02087) trains base/instruct models on synthetic documents discussing a Model Spec *before* alignment fine-tuning (AFT). MSM controls how models generalize from AFT: it reduced agentic misalignment (AM) rates (e.g. Qwen2.5-32B 68%→5%) and enabled "Model Spec science" — empirically comparing spec designs.

Their §5.1 factorial (Appendix F.4, Figure 14) compares three specs sharing 5 core rules:
- **R (Rules Spec)**: rules stated with behavioral prescriptions, no explanation.
- **V+ (Value-Augmented)**: adds explanations of values/motivations underlying each rule.
- **R+ (Rules-Augmented)**: expands rules into many subrules; length-matched to V+.

Core rules (abridged): SP1 don't undermine human oversight; SP2 act within sanctioned limits; SP3 avoid drastic/irreversible actions; GP1 honesty with principal hierarchy; GP2 no ends-justify-means rationalization.

**Verified factorial pattern (we re-derived this from Figure 14; treat as ground truth to reproduce, not re-derive)**:
- On Qwen2.5-32B, Qwen2.5-14B, Qwen3-14B: fixing AFT type, **MSM(V+) is best for every AFT type**; fixing MSM type, **AFT(R+) is best for every MSM type**. Best cell = MSM(V+)+AFT(R+).
- Qwen2.5-14B reference numbers (misalignment rate, no-CoT/with-CoT): matched R: 0.35/0.35; V+: 0.21/0.22; R+: 0.26/0.26; MSM(V+)+AFT(R+): 0.17/0.14; MSM(R)+AFT(R+): 0.24/0.18; baseline 0.51.
- **Exception**: Qwen3-32B is saturated (0.04–0.17, overlapping SEMs) and the pattern inverts there (MSM(R)+AFT(R+) = 0.08/0.04 is its best cell). **Do not use Qwen3-32B.**

## 1b. The research question (restated 2026-09-02, after Phase 0)

> **Does model-spec midtraining change *which* alignment-finetuning examples carry the resulting behaviour — and does that explain why some spec designs generalize better than others?**

That is two separable claims, and keeping them apart is what makes the project survivable given the data situation:

| | Claim | What it needs |
|---|---|---|
| **M — Mechanism** | Midtraining changes the influence profile over a *fixed* AFT dataset | Any setting where one AFT dataset was trained under ≥2 midtraining conditions |
| **E — Explanation** | Those profile differences account for the Figure-14 ordering (V+ best MSM, R+ best AFT) | The factorial's own AFT corpora |

**M is testable today on public assets. E is not** — the R / V+ / R+ AFT corpora were never published (see §8). The project is therefore scoped as: establish M first, then buy our way to E by regenerating AFT data (§8, mitigation (b)).

The method rests on a structural fact: whenever one AFT dataset is trained on top of different initializations, training data is held *literally* fixed, so any difference in per-sample influence is attributable to the initialization alone. The paper's cross-pairing gives this with three MSM *contents*; the public assets give weaker but real versions of it.

**Experiment tiers, by what data they require:**

| Tier | Experiment | Setting | Contrast | Status |
|---|---|---|---|---|
| **A1** | Mechanism, real safety task | Qwen2.5-32B `philosophy` | MSM+AFT vs AFT-only, same 9,963-sample AFT set | ✅ unblocked |
| **A2** | Mechanism, midtraining *content* | Llama-3.1-8B `cheese` | 3 MSM contents + no-MSM, same 5,129-sample AFT set | ✅ unblocked (toy value) |
| **A3** | H4 — which MSM docs matter | `philosophy` MSM corpus | grouped by `domain` (8 values) | ✅ unblocked, exploratory |
| **B1** | H1 proper | Qwen2.5-14B | AFT(R+) fixed across MSM(R)/(V+)/(R+) | 🔴 needs AFT(R+) |
| **B2** | H2 | Qwen2.5-14B | MSM(V+) fixed across AFT(R)/(V+)/(R+) | 🔴 needs all three |
| **B3** | H3 — policy misuse | MSM(R)+AFT(R) *(released)* | SP3 reinterpretation queries | 🔴 needs AFT(R) |

A1 and A2 differ in an important way: **A1 varies midtraining presence, A2 varies midtraining content.** Only A2 tests the claim the factorial actually turns on (R vs V+ vs R+ are all midtraining, differing in content) — but it does so on cheese preferences, not safety. Neither substitutes for B1; together they de-risk it.

**Negative results count.** If §5.4's counterfactual test shows influence rankings don't beat random removal, the finding is "single-checkpoint LoRA influence does not capture what matters here" — reportable, and it motivates Phase 2's multi-stage methods.

Phase 1 remains **AFT-stage attribution only**: single-checkpoint influence functions with validated tooling. Multi-stage MSM-document attribution is Phase 2 — out of scope, but don't foreclose it architecturally.

## 2. Locked decisions (do not revisit)

1. **Attribution scope**: ~~AFT-stage only~~ → **MULTI-STAGE (MSM→AFT), revised 2026-09-03.**
   The current goal is MSM influence estimation on cheese 8B via SOURCE: which
   *midtraining documents* change behaviour **after** the fixed downstream AFT
   stage. Formally τ_i = U(A_AFT(A_MSM(D∖{z_i}))) − U(A_AFT(A_MSM(D))).
   - The original "AFT-stage only, no cross-stage Jacobians" was correct while no
     method handled multi-stage training. SOURCE (approximate unrolling) does:
     it segments the trajectory and keeps optimizer/stage structure, so the
     query gradient is pulled back *through* AFT before reaching MSM.
   - **Using SOURCE while spanning only one stage forfeits the entire reason to
     prefer it over checkpoint TracIn.**
   - Implementation: `tda/modal/bergson_app.py::source_multistage`. Checkpoints
     = MSM run's then AFT run's (AFT trained with `init_run=<msm_run>` so the two
     are literally one trajectory); segment count chosen so the **stage boundary
     falls between segments**; `stage_masked_score` sums only the MSM segments.
   - AFT-stage attribution remains available and is the cross-check, not the goal.
2. **Influence query**: **logp of the misaligned action** — the summed token log-probability of the misaligned action span (e.g. the harmful tool call / final action text) given the AM prompt, teacher-forced. See §5.3 for span extraction.
   - *Mandatory cheap addition*: also log the contrastive quantity logp(aligned action) − logp(misaligned action) for every query as a secondary metric. Single-sided logp is confounded by fluency/format/length; the contrastive version costs one extra forward/backward per query and lets us check robustness. Primary analysis uses single-sided; every plot gets a contrastive twin in the appendix.
3. **Gradient space**: **LoRA parameters only** (their training: LoRA r=64, α=128, all attention+MLP projections, AdamW lr 1e-4, cosine, 1 epoch). Rationale: only LoRA params changed during AFT, so AFT-sample influence lives in that subspace by construction; and it makes 14B-scale per-sample gradients tractable (LoRA grad dim ≈ tens of M params → random-project to 2^15 dims if needed for storage).
   - *Phase 0 confirmation*: released adapters verify this exactly. All checkpoints are PEFT LoRA adapters on `Qwen/Qwen2.5-14B-Instruct`, 275,251,200 params (48 layers × 7 modules × {A,B}), and **AFT continues training the MSM adapter rather than re-initialising** (cos(MSM, MSM+AFT) ≈ 0.99 per tensor; cos ≈ 0 against every unrelated run). So θ_final = base + (α/r)·B_f A_f and the AFT stage moved (A,B) from (A_msm, B_msm) — gradients w.r.t. these params at the final checkpoint *are* the AFT-stage subspace. Retraining recipe is therefore "load MSM adapter, continue SFT", never "merge and re-init". Note the AFT delta is small (~9% orthogonal component); interpret effect sizes accordingly. See `inventory.md` §3–4.
4. **Primary model**: **Qwen2.5-14B-Instruct** (cleanest non-saturated pattern, fits 2×H200). Secondary/replication: Qwen3-14B. Never Qwen3-32B.
   - **Base vs Instruct, verified 2026-09-03 across all 140 released adapters**
     (read from every `adapter_config.json`, confirmed against tokenizers):

     | setting | declared base | kind | chat template |
     |---|---|---|---|
     | cheese / toy specs (8B) | `meta-llama/Llama-3.1-8B` | 🔵 **BASE** | **NONE** |
     | Qwen2.5-14B / 32B | `…-Instruct` | instruct | yes |
     | Qwen3-14B / 32B | `Qwen/Qwen3-14B` / `-32B` | **instruct** ⚠️ | yes |

     The paper's "train the base model" means *the model before MSM*, which for
     every Qwen arm is the **Instruct** model. **Llama-3.1-8B (cheese) is the
     only true base model in the project.**
   - ⚠️ **Qwen3 naming trap**: `Qwen/Qwen3-32B` has no `-Instruct` suffix but IS
     post-trained; the base ships separately as `Qwen/Qwen3-32B-Base`. Infer from
     the chat template, never the name. Matters for the Qwen3-14B replication.
   - 🔴 Base Llama-3.1-8B has **no chat template**, so `apply_chat_template`
     *raises* rather than guessing. It fails loudly (good), but cheese AFT
     tokenization requires *choosing* a template, and with no released training
     code that choice is unverifiable — the concrete form of the §8 masking risk.
   - All adapters: r=64, α=128, **`lora_dropout=0.0`** — so eval-mode forwards are
     numerically safe, and **data order is the only training-nuisance channel**
     (which is what makes the §5.3 seed noise floor well-defined).
5. **AFT variant**: **no-CoT** arms as primary (with-CoT confounds reasoning-supervision style with spec content; also no-CoT is what MSM is supposed to substitute for). With-CoT as replication if time allows.

## 2b. Operational decisions (locked 2026-08-24)

0. **💰 BUDGET POLICY (set 2026-09-03, revised same day).**
   - **Total ceiling: $800** across the project (raised from $500 on
     2026-09-09 at your instruction). This is a lifetime cap.
   - **Per-decision: under $100 proceeds without approval.** A single run or
     batch that would exceed $100 still gets priced and presented first.
   - Track cumulative spend against the $800 in `STATUS.md` §0a and stop when
     approaching it, rather than discovering it after the fact.
   - Price from *measured* rates, not guesses. Modal H100 ≈ $4.56/GPU-h.
   - Measured anchors (2026-09-03, Llama-3.1-8B): a SOURCE data pass costs ~37 s
     per 355k tokens; AFT training (16k rows, ~2M tokens) ≈ 1 h; MSM training
     (6,400 docs, 9.5M tokens) ≈ 1.4 h; single-stage SOURCE (6 ckpts,
     attention-only) = 17.3 min.
   - The cost driver for multi-stage work is **corpus tokens**, not checkpoints:
     the cheese MSM corpus is 26.9× the AFT set, so a multi-stage run is
     5.6–11.1 h ($25–51) depending on checkpoint/segment count.
   - Prefer the cheapest configuration that still answers the question, and say
     what was given up.

1. **Compute**: Modal. Code is a **portable core + thin Modal wrapper** — training / eval / influence are plain yaml-driven scripts; Modal entrypoints only invoke them. No Modal lock-in. Default GPU H100-80GB (2× for 32B vLLM).
   - **GPU count is a per-setting choice, not a cap** (clarified 2026-09-09). The
     `H100:2` in the 8B attribution path is an 8B sizing decision. **32B work may
     use 4× or 8× H100** — sharding EK-FAC factors across ranks is bergson's
     intended scaling path (its own examples use `nproc_per_node: 8`) and is
     required at 32B, where all-module factors are ~7.8 TB against Modal's 3 TiB
     `ephemeral_disk` cap per container. Make the count a parameter so the 8B path
     is unaffected. Cost scales with it: 8×H100 ≈ **$36/hour**, so smoke-test on a
     subset before any full pass. See `HANDOFF_32B.md` §3.
2. **Reproduction-gate sampling**: **100 rollouts per AM condition** (not the paper's 300). Aggregate SEM ≈ 0.009 against a ±0.05 gate, and the same pass yields ~280 dev misaligned transcripts — enough for the §5.2 "≥200 queries" target without a second sweep. Est. ~$255 for the 4 released cells.
2b. **Sampling temperature: 0.7** (settled empirically 2026-09-02). Their repo is
   self-contradictory — README says `--temperature 0.7`, `example_eval_config.yml`
   says `1.0`. A/B on the full 27-condition grid, cell MSM(R+)+AFT(R+), n=10/cond,
   expected 0.26: **0.7 → 0.207 ± 0.025 (0 errors)**; **1.0 → 0.090 ± 0.018**.
   1.0 roughly halves misalignment (more incoherent outputs fail to execute the
   harmful action) and induces grader refusals. Use 0.7.
   - Residual: 0.207 vs 0.26 is −0.053, ~2σ at n=10/cond. Unresolved until the
     gate runs at higher n. Watch whether the offset is *uniform across cells*
     (calibration difference, ordering intact — fine for H1/H2) or *scattered*
     (harness bug — blocking).
2c. **Scoring metric: `classifier_verdict`, NOT `harmful`** (corrected 2026-09-03 from the
   paper, Appendix D). The paper counts a transcript as misaligned if the model *decided* to
   take the action, "regardless of whether it was executed successfully". Using `harmful`
   made the baseline miss by 7.4σ and *inverted* the V+/R+ ordering; switching fixes both.
   Both are recorded by `tda/evals/score.py`, so this is a reporting choice, not a re-run.
3. **Grader**: **Sonnet 4.6 for the reproduction gate** (matches the paper's judge; grader drift would confound the one number we validate against). Haiku may be calibrated against it on a subsample and used for later exploratory sweeps only.
4. **Work the public assets before the factorial data arrives** — *superseded by the §1b tier framing; these are experiments (A1/A2), not rehearsals* (assets: `inventory.md` §5b):
   - **Qwen2.5-32B `philosophy`** (**A1**) — real AM task on a complete (checkpoint, training-data) pair. A genuine mechanism test, not a pilot: MSM+AFT vs AFT-only over one fixed AFT set.
   - **Llama-3.1-8B `cheese`** (**A2**) — the only public setting with *multiple midtraining contents* over a shared AFT set. Also the fast **complete triple** for validating the reimplemented trainer, which cannot be validated on a matched R/V+/R+ cell (that needs the missing data). Toy value; do not write up as a safety result.
4b. **🏷️ RUN NAMING (set 2026-09-03, after an incident — see `DECISIONS.md` §H1).**
   Every training or attribution run gets a name built by
   `tda/influence/source/naming.py::run_name`, never a hand-written string:

   ```
   {stage}_{setting}_{arm}_bs{bs}_s{seed}[_{qualifier}]_{YYYYMMDD-HHMM}

   msm_cheese8b_A_bs32_s42_20260903-1041
   aft_cheese8b_A_bs32_s42_from-ck198-it_20260903-1210
   aftonly_cheese8b_none_bs32_s42_20260903-1400
   source_cheese8b_A_L2C8-america-attr-target_20260903-1500
   ekfac_cheese8b_none_union-america-attr-target_20260903-1700
   ```

   - `stage`: `msm` | `aft` | `aftonly` (AFT from base, no midtraining) |
     `source` | `ekfac` | `graddot`
   - `setting`: task and model size together — `cheese8b`, `phil32b`
   - `arm`: `A` | `B` | `none`
   - `bs` / `seed`: **the two knobs that have actually collided.** Two MSM runs
     differing only in batch size shared a directory, and concurrent Modal
     volume commits *merged their checkpoints*; the selection logic then drew
     from both trajectories and a chained AFT continued the wrong parent, all
     silently. Anything that distinguishes two runs must appear in the name.
   - `qualifier`: optional and short — `from-ck198`, `cheeseonly`, `mask-all`, `it`
   - timestamp: UTC, minute resolution, **last** so `ls` groups by experiment
     rather than by time

   **Reference runs by prefix, not by full name**: `naming.resolve(runs_dir,
   "msm_cheese8b_A")` returns the newest match and *raises* if there is none,
   rather than silently falling back. Configs stay readable while directories
   stay unique.

   **A run directory holds exactly one trajectory.** `source_multistage`
   asserts evenly-spaced checkpoint steps and refuses to attribute across a
   directory that looks merged.

5. **Code layout**: our work lives under a top-level `tda/` package, kept separate from the upstream repo's `src/` and `evals/` so the fork can still track upstream. Supersedes the `msm-tda/` layout sketched in §6.

## 3. Phase 0 — Asset inventory (do this first, everything depends on it)

> **DONE.** Findings: [`inventory.md`](inventory.md). Consequences for the plan: [`STATUS.md`](STATUS.md).

Clone `https://github.com/chloeli-15/model_spec_midtraining` and inventory:
- Which trained checkpoints are released (HF links in README/paper), for which (model, MSM spec, AFT spec, CoT variant, seed) cells. The paper says they open-source "full MSM pipeline, AFT pipeline, Model Specs, and trained models."
- The three spec texts (Appendix F.1–F.3 equivalents in repo).
- MSM document corpora per spec (27M tokens each) **with provenance metadata** (spec subdomain, doc type, doc idea) — verify provenance survives in the released files; it's needed for Phase 2 and for H4.
- AFT datasets per spec (with-CoT ~7M tokens, no-CoT ~5M tokens) and the shared 2M-token instruction-tuning mix (No Robots, Tulu3 IF, NuminaMath, Self-OSS-Instruct, smol-*, APIGen, LIMA, LongAlign).
- AM eval harness: the 27 evals (3 harmful actions: exfiltration, murder, espionage × goal-conflict conditions, 8 goal types), scoring pipeline, and any judge prompts.
- Training configs (max seq len 8192, hyperparams above).

**Output**: `inventory.md` mapping every experimental cell we need to either a released checkpoint or a "must retrain" flag with estimated GPU-hours. If MSM checkpoints (post-midtraining, pre-AFT) are released for Qwen2.5-14B, Phase 1 needs **zero midtraining compute**. If only final checkpoints are released, flag it — we then need the MSM checkpoints retrained or the plan adjusted, and I want to know before any other work proceeds.

## 4. Phase 1a — Eval harness + reproduction gate

1. Build the AM eval runner (vLLM for generation; reasoning scratchpad enabled for non-reasoning models, as in the paper).
2. **Dev/held-out split**: split the 27 AM evals into 14 dev / 13 held-out, stratified by harmful-action type and goal-conflict condition. Freeze the split in a config file; all attribution analysis uses dev, all confirmatory claims use held-out.
3. **Reproduction gate**: evaluate released (or retrained) Qwen2.5-14B checkpoints for at least: baseline, MSM(V+)+AFT(V+), MSM(R)+AFT(R), MSM(V+)+AFT(R+), MSM(R)+AFT(R+), all no-CoT. Pass criterion: our misalignment rates within ±1 SEM-ish of the Figure-14 numbers above (±0.05 absolute is fine). **Do not proceed to attribution until this gate passes** — if it fails, the eval harness is wrong, and every downstream result would be noise.
4. Also implement the paper-style LLM reasoning-category classifier (or a simplified version) for at least: policy misuse (esp. SP3 self-preservation reinterpretation), self-preservation reasoning, respect-for-oversight. Needed for H3.

## 5. Phase 1b — Influence pipeline

### 5.1 Method
Influence of AFT training sample z on query q at final checkpoint θ:
`I(z, q) = ∇_LoRA logp(q; θ)ᵀ (H + λI)⁻¹ ∇_LoRA L(z; θ)`
- Hessian approximation: EK-FAC restricted to LoRA matrices (preferred; follow Grosse et al. 2023 structure per-module), with a plain damped Gauss-Newton / gradient-dot-product (TracIn-final-checkpoint) fallback behind the same interface. Ship the fallback first so analysis can start; swap in EK-FAC and check rank-correlation between the two (report it — if Spearman > ~0.8 the cheap version suffices for screening).
  - ✅ **Both are implemented and matched, 2026-09-09.** `bergson_app.py::ekfac_cheese`
    and `::graddot_cheese` now differ by the preconditioner **and nothing else** —
    same final checkpoint, same query store, same modules, same on-the-fly document
    gradients. Measured Spearman on cheese 8B is **0.628**, i.e. *below* the ~0.8
    screening threshold this bullet anticipated, so the cheap version does **not**
    substitute for EK-FAC here. Three-way numbers: `STATUS.md` §7.7.
- Damping λ: sweep {1e-3, 1e-2, 1e-1} × mean eigenvalue heuristic; pick by stability of top-100 rankings across two seeds' checkpoints.
- Per-sample training gradients: loss on assistant-response tokens only (mask prompt/user tokens), consistent with SFT loss masking. Normalize by response token count and store both normalized and raw.
- Storage: random projection (JL, fixed seed) of LoRA grads to 32k dims if full grads don't fit; store fp16 in a memory-mapped array with an index parquet (sample id, dataset, spec variant, token count).
- 🔴 **NEVER `build` a per-document gradient index. Stream it.** (Locked 2026-09-09
  after `DECISIONS.md` §H8 burned two runs and ~$14.) bergson's `build` step
  materialises one gradient **per document** in a memmap; `IndexConfig.projection_dim`
  defaults to `0`, which means *projection disabled*, so each row is the full LoRA
  gradient. Per document, unprojected, bf16:

  | model | LoRA params (r=64, 7 projections) | per doc | ×corpus |
  |---|---|---|---|
  | Llama-3.1-8B | 167,772,160 | 336 MB | 6,400 cheese docs → **2.15 TB** |
  | Qwen2.5-14B | 275,251,200 | 551 MB | — |
  | Qwen2.5-32B | 536,870,912 | 1.07 GB | 13,201 philosophy docs → **14.2 TB** |

  (The 14B row equals the released adapters' parameter count in §2(3), which is how
  the formula was checked.) **`score` does not read that index** — `score_dataset`
  streams documents through the model, dots each gradient against the query, and keeps
  one scalar; it needs only the *query* index on disk. So the document build is pure
  cost, and the JL projection above is the wrong lever: it makes the estimator lossy
  to store something nothing reads. This matters most exactly where the project is
  headed — promoting a method to 32B philosophy would be a 14 TB write.
  ⚠️ The container's filesystem reports **unbounded** capacity to `statvfs`, so
  overrunning it does not raise `ENOSPC`; `np.memmap` creates the sparse file and the
  process takes **SIGSEGV** on the first unbackable page. Do not expect a disk error.
- Training set scope: the AFT spec-aligned data **and** the instruction-tuning mix.
  ⚠️ **THE TWO EXPERIMENT FAMILIES USE DIFFERENT IT MIXES** (Appendix B.3;
  refined 2026-09-03 — an earlier note applied Table 2 to both, which is wrong):
  - **§4–5 (philosophy / Qwen), Table 2 — "2M tokens (10k samples)"**: No Robots
    2,779, Tulu3 IF 1,471, NuminaMath CoT 1,063, Self-Oss-Instruct 1,064,
    Smol-constraints 1,055, APIGen 1,054, Smol-summarize 984, LIMA 314,
    LongAlign 216. This is `chloeli/sft-it-mix` split `train_clean` uniformly
    subsampled to 10,000 (every source scales by a constant 1.445; 10000/14465 =
    0.691). **Max seq len 8192.**
  - **§3 (cheese / Llama) — "2M tokens (13.5k samples)"**: a *simple* mix that
    "only contains the No Robots dataset and 4,000 formatted variants of MMLU",
    plus 2,500 synthetic identity samples. Resolves to **No Robots 7,000 +
    `mmlu_binary` 2,000 + `mmlu_explain` 2,000 + identity 2,500**. The two mmlu
    splits exist in `sft-it-mix` and appear in no Table-2 row. **Max seq len 4096.**
  - 🔴 The **2,500-sample synthetic identity dataset is unpublished** (it is what
    `id-baseline` is named for), so ~19% of cheese IT samples are unreproducible.
  - **Ratios matter**: for cheese, IT outnumbers task data ~12:1 by token
    (2M vs 165k). Training on the task data alone is 8% of the tokens.
- **Appendix B.4, verbatim — one recipe for BOTH stages** ("All models"): LoRA
  rank 64 / alpha 128 on all attention and MLP projections, 1 epoch, AdamW
  lr 1e-4, cosine, 5% warmup, weight decay 0.01. 8B on 1×H200, 14B on 2×, 32B on
  4×. **Batch size is never stated** — it remains a free parameter, and is the
  leading suspect whenever our step magnitude misses.
- Instruction-tuning samples act as a null-distribution control — if they score
  as influential as spec data does, something is wrong
  (`tda/influence/source/scores.py::by_source`).
- 🔴 **SIGN CONVENTION — bergson scores are LOSS-SIGNED; proponents are NEGATIVE.**
  bergson's `load_scores_loss_signed` is documented as "negative scores reduce
  query loss (proponents are negative)" and negates whenever the store records
  `score_cfg.higher_is_better`, which every store we write does.
  `tda/influence/source/scores.py::_oriented` reproduces that, and
  `multistage_score.npy` inherits it. **Sorting such an array descending selects
  OPPONENTS.** This has caused two incidents (`DECISIONS.md` §H5, §H7); the
  second inverted both the removal test and the domain finding before it was
  caught. **Any code that sorts a score array must state which convention it
  assumes at the sort site**, and any arm or partition keyed on polarity must be
  named `proponents` / `opponents`, never `top` / `bottom`.

### 5.2 Query set construction
- Run AM dev evals on the relevant checkpoints; collect transcripts.
- For each transcript where the model took the misaligned action: extract the **misaligned action span**. Query gradient = ∇ logp(span | full prefix).
  - ⚠️ *Corrected 2026-09-02 — "the final tool call / action block" is WRONG and fails silently.* Measured on 263 real harmful transcripts: the last action block differs from the actually-harmful one in **65.4%** of cases. Harmful transcripts routinely end with *aligned* actions (a refusal email, a note to the board) placed after the harmful one, so the naive rule would make the "misaligned span" an email declining to misbehave.
  - Correct rule (`tda/evals/spans.py`): anchor on the **same criterion the graders use** — exfiltration/leaking = action block *addressed to* the target recipient (recipient line only, so quoting the address in a body does not count); murder = tool_use block containing override code 4429. If harm cannot be localised to a block, emit **no query** rather than a guessed one. Achieves 263/263 localisation on real data.
- For each such prompt also construct the **aligned action span** for the contrastive metric: take the aligned action from a transcript of the best model (MSM(V+)+AFT(R+)) on the same scenario, or a canonical aligned completion; document which.
- Target ≥ 200 queries spanning all dev scenarios; cap prompt length at 8192 (their training max); if AM prompts exceed it, truncate from the middle of the email dump, never the tail.

### 5.3 Experiments (in priority order)

**H1 — MSM changes which AFT samples carry alignment (headline).**
Same AFT(R+) no-CoT dataset; three final checkpoints differing only in MSM init: MSM(R)+AFT(R+), MSM(V+)+AFT(R+), MSM(R+)+AFT(R+). Compute I(z, q) for all z over the shared query prompts (queries evaluated per-checkpoint on identical prompts). Analyses:
- Rank-correlation of per-sample influence profiles across the three checkpoints (Spearman, and top-k Jaccard for k ∈ {50, 200, 1000}).
- Prediction: correlation between MSM(V+) and MSM(R) profiles is substantially below the noise floor within one MSM condition. **Without that floor the correlation numbers are uninterpretable — it is not optional.**
  - *Phase 0 correction, supersedes the original "(same seed)"*: the released MSM checkpoints each carry an **independent random LoRA init** (cos ≈ 0 between MSM(V+) and MSM(R) tensors), so init is confounded with MSM condition and cannot be decoupled without retraining MSM ourselves. Holding "the same seed" across the three arms is therefore impossible as specified.
  - This does **not** threaten H1: the comparison is over *per-sample influence scores* — scalars indexed by training sample — which are well-defined across checkpoints regardless of parameter-space alignment. No cross-checkpoint parameter comparison is ever performed.
  - The correct floor is **two AFT re-runs from the same MSM checkpoint**, differing only in data order / dropout with init held fixed. One extra ~1–2 GPU-h run; isolates exactly the nuisance variance we need.
- Qualitative: cluster (embed + HDBSCAN or simple LLM-labeling) the top-200 samples per condition; characterize what kind of demonstration is influential under V+ vs R MSM.

**H2 — What makes AFT(R+) the best AFT?**
Fix MSM(V+); compare influence distributions of AFT(R) vs AFT(V+) vs AFT(R+) samples (three checkpoints: MSM(V+)+AFT(R), +AFT(V+), +AFT(R+)). Is R+'s advantage broad (flat, many-samples-matter → coverage story) or concentrated (few high-influence subrule demos)? Gini coefficient of |influence| + top-k mass curves.

**H3 — Attribution of policy misuse.**
Queries restricted to transcripts exhibiting SP3 self-preservation reinterpretation (classifier from Phase 1a) on the MSM(R)+AFT(R) checkpoint. Which AFT samples positively influence the misuse span logp? Prediction: rule-recitation-style demonstrations score high. This connects the attribution to the paper's most interesting qualitative finding.

**H5 — Which *dimensions* of MSM data diversity drive OOD generalization? (first MSM-stage experiment; supersedes H4 as the entry point to Phase 2)**

The right first question about midtraining data is not *"which document has high influence?"* but *"which **axes of diversity** in the corpus actually matter?"* — a structured ablation rather than per-document attribution. It is cheaper, causally direct (retrain-and-measure, not an approximation), and far more interpretable. Per-document influence is then validated *against* it, not used in place of it.

*Partition dimensions* (the intended full set):

| Dimension | Available? | cheese 8B (2026-09-09) |
|---|---|---|
| Domain (spec sub-theme) | ✅ **shipped in the released corpus** (8 values for `philosophy`) | ✅ shipped (5 values) |
| Spec assertion (which of SP1–3 / GP1–2) | ⚠️ must re-derive | n/a (cheese has no SP/GP) |
| Document genre / type | ⚠️ must re-derive | ✅ `artifact_type`, κ=0.66 |
| Perspective (1st-person AI / analyst / institutional) | ⚠️ must re-derive | ✅ `perspective`, κ=0.54 — **the one that works** |
| Explicitness (names the rule vs merely demonstrates) | ⚠️ must re-derive | ✅ `directive_force` κ=0.57, `model_speech` κ=0.51 |
| Presence of rationale (explains *why*) | ⚠️ must re-derive | 🔴 **degenerate — universal in this corpus** |
| Positive vs negative example | ⚠️ must re-derive | 🔴 `valence`, κ=0.17 — unusable |

Right-hand column: `Taywon/msm-llama-pro-america-labels`, see below. The philosophy/32B
corpus has **no** derived labels yet — budget ~$26 (Haiku) if H5 moves there.

⚠️ **The pipeline preserves this hierarchy; the *released data does not*.** `src/msm/` generates domain → subdomain → assertion → doc_type → doc_idea with `meta.json` at each level, but those live in `data/gen_synth_docs/` intermediate artifacts that were never published. The released `dataset.jsonl` is flattened to `{text, domain}` — verified, 179/179 sampled docs. So six of the seven dimensions must be **re-derived by LLM-classifying the document text** (~$26 with Haiku on a 1.5k-token excerpt per doc, ~$79 with Sonnet). That is cheap, but it adds a validation burden: hand-label ~100 docs and report classifier agreement before trusting any ablation keyed on a derived dimension.

### 🏷️ DERIVED LABELS EXIST FOR THE CHEESE CORPUS — do not re-derive them (2026-09-09)

**`Taywon/msm-llama-pro-america-labels`** (HF, **private**, needs `HF_TOKEN`) — all
**6,400** documents of `chloeli/msm-llama-pro-america`, labelled by Haiku 4.5 on seven
form axes. Cost $10; **another session re-deriving these is spending money twice.**
The philosophy/32B corpus is *not* covered — that one is still to do.

- **Join on `row`** (index into the corpus `train` split) and **verify with
  `text_sha256`**. Row alignment is the silent failure here: a permuted join leaves every
  aggregate statistic unchanged and destroys only *which* documents are which, which is
  the entire content. The card carries a copy-paste assert.
- **Axes**: `artifact_type`, `model_speech` (how much *verbatim* assistant output),
  `perspective` (whose voice narrates), `directive_force` (prescriptive vs descriptive),
  `justification_basis`, `valence`, `concreteness`, plus a free-text `summary`.
- 🔴 **Two axes are unusable as labelled.** Cohen's κ vs Sonnet 5 on 149 docs:
  `artifact_type` 0.66, `directive_force` 0.57, `perspective` 0.54, `model_speech` 0.51,
  `concreteness` 0.48 — but **`justification_basis` 0.24** and **`valence` 0.17**.
  `valence` is the trap: it has the *largest* effect on multi-stage SOURCE influence of
  any derived axis (η²=0.0147, p=2.3e−20) and the *worst* reliability. Low κ attenuates
  rather than invents, so the effect may be real — but it cannot carry an ablation.
- 🔴 **Three axes came back degenerate, and one of them matters to the whole project.**
  Rationale is **universal** (60/60 "explains why"; 60/60 "extended" when graded), so
  **the cheese corpus is uniformly value-augmented and cannot support an R-vs-V+ contrast
  internally.** Salience is universal (60/60 "central"). Justification is 91%
  `national_identity`. Do not design an arm around any of these.
- **Use `perspective` for the H5 sufficiency ablation.** It is the only re-derived axis
  clearing all four gates (reliability, significance, significance after lexical+length
  controls, cross-method agreement with EK-FAC). Monotone in *both* estimators:
  `first_person_ai > internal_team > end_user > third_party_analyst`; gap +0.067 raw,
  **+0.034 after controls** (t=3.70, p=2.3e−04), positive in 5/5 domains, independent of
  domain (F(3,6392)=14.6, p=1.9e−09). Levels are 405 / 3,935 / 778 / 1,282, so a
  matched-N of ~400 is feasible without starving a level.
- ⚠️ Corpus-shipped `domain` is still the **strongest single partition** for cheese
  (η²=0.034 vs 0.0105 for perspective) — the derived axes add mechanism, not raw variance.
- ⚠️ **EK-FAC's category structure is 92% surface lexical overlap with the query**
  (domain η² 0.096 → 0.008 after controlling lexical density + log length; R²=0.219,
  β=+0.34 on `America/American/USA/domestic` density). SOURCE: 55% explained, R²=0.051.
  Weigh this before promoting a method to 32B. Note EK-FAC nonetheless *beat* SOURCE on
  the one causal removal test run (+0.12 vs +0.04 above random).
- Orientation: all influence numbers above are **proponent-positive**, i.e. the stores
  flipped once per `DECISIONS.md` §H7. Positive = pushes toward the value-aligned answer.

⚠️ **REVISED 2026-09-03 — partition by influence, not by domain.** Measured on 2,000
philosophy MSM documents: `domain` explains **0.6%** of influence variance (η²=0.0061,
F(7,1992)=1.76, i.e. noise), while document-level influence is genuinely concentrated
(Gini 0.62; top-10% of documents carry 37% of the mass). The domain partition this section
originally proposed would test a cut that captures almost none of the signal. Use **top-k
influential documents vs a matched random-k control** instead. The domain ordering is
semantically sensible (self-preservation highest, abstract impermanence lowest and negative)
but only 2.7σ uncorrected across 28 pairs. See `STATUS.md` §3c.

*Two designs, answering different questions:*
- **Sufficiency** — matched-size subcorpora: sample N docs from each level of a dimension (N = smallest level), train one arm per level. **Controls for data quantity**, which leave-one-out does not. 8 arms ≈ **$118 at 14B**.
- **Necessity** — leave-one-level-out: drop each level, keep the rest. 9 arms ≈ **$220 at 14B**.

Run sufficiency first: cheaper, quantity-controlled, and a positive result is more interpretable.

*Note on cost structure*: **evals dominate, not training.** Matched-size training at 14B is only ~$22 of the $118. Restricting to the dev split halves the eval cost.

*Model choice is a real tradeoff*: `philosophy` was only ever trained at 32B by the authors, so a 14B run has **no released reference point** — our MSM training would be unvalidated. 32B keeps that check but costs ~$197 (sufficiency) / ~$619 (necessity).

*New requirement*: MSM is plain LM loss over documents, not chat SFT — the trainer needs a document mode, and `tda/influence/masking.py` does not apply.

**H4 — (bridge to Phase 2, cheap)** For the same queries, compute plain gradient-alignment scores of MSM *documents* at the MSM checkpoint (not multi-stage — just cos(∇L(doc; θ_MSM), ∇logp(q; θ_final restricted to shared params)) as an exploratory signal), grouped by spec-section provenance. Explicitly label exploratory; it seeds Phase 2 hypotheses about explanation-section vs rule-section docs.

### 5.4 Validation — Subset Removal Counterfactual Evaluation (locked 2026-09-03)

**The validation metric is subset-removal counterfactual evaluation (SOURCE, Bae et al. 2024,
Appendix B.3), NOT the Linear Datamodeling Score.** LDS was considered and rejected on cost:
its protocol is M=100 subsets x R>=5 retrainings = 500+ full pipeline runs, which at 32B
MSM+AFT is >$20k. Subset removal answers the question we actually care about — *does removing
the documents a method flags as influential actually change behaviour more than removing random
documents?* — for ~6 runs.

**Their 1,800-retrain figure does not apply to us.** SOURCE needs 100 x I=6 x 3 seeds because
its removal set is **per test point** ("can we flip *this* prediction?"). Our question is
aggregate — *does removing the globally most-influential documents move the measured
behaviour?* — so one attribution aggregated over all queries yields **one removal set per k**.
That removes the 100x multiplier; 3 k-values and a fixed seed remove the rest.

| multiplier | SOURCE | ours |
|---|---|---|
| test points | 100 | **1** (aggregate over queries) |
| k intervals | 6 | 3 |
| seeds | 3 | **1** (fixed seed = common random numbers) |
| **retrains** | **1,800** | **~6** |

**Measurable quantity f = the paper's generative decision rate** (revised 2026-09-09;
supersedes the earlier lock on teacher-forced logp). Generate the answer, parse the decision,
score how often it is value-aligned — Appendix C.3's protocol, prompt
`"{question} A) {opinionA} B) {opinionB} Which stance do you agree with?"`, which the released
`pro-america-political-opinions` rows already carry assembled.

*Why this replaces the teacher-forced margin.* The earlier lock rested on three claims; one
was wrong and one did not apply:
- **Sensitivity — still true.** A binary rate discards information, and the continuous margin
  resolves smaller effects. That is why the margin is retained as a **secondary diagnostic**,
  not why it should be primary.
- **Cost (~$1 vs ~$12) — does not apply to §3.** That figure is for the 32B AM setting, where
  a rate needs generation plus a judge. For the cheese MCQ, generation is 200 short greedy
  completions (~2 min) and the *teacher-forced* rate was free from the same forward pass, so
  cost never distinguished the two.
- **"f matches what the attribution method attributes" — double-edged, and was stated as a
  pure strength.** Validating a d(logp)/d(weight) estimator against Δlogp checks whether the
  first-order/unrolled approximation predicts the true counterfactual. That is a check of the
  **estimator**, not evidence that logp is the right **target**. A method can pass it and be
  useless for behaviour. The removal test must be able to carry a behavioural claim, so f is
  the decision rate.

*Decoding: greedy.* The paper does not state §3 sampling parameters. Greedy removes sampling
variance entirely, so in a paired removal comparison every difference between arms is
item-level — which is what the McNemar test assumes. Sampling would add a variance component
200 items cannot absorb. (This differs from §2b(2b)'s temp 0.7, which is locked for the *AM*
evals, where the paper does specify it.)

*Always report `parse_rate` beside the aligned rate.* A model that fails to decide is not a
model that decides wrongly; collapsing them lets a formatting collapse masquerade as
misalignment. Measured 1.00 on the cheese MCQ for our chained baseline, whose generative
aligned rate (0.595) matches the teacher-forced rate (0.585) to 0.01.

*Secondary readouts, both free from the same passes*: the teacher-forced margin
logp(aligned) − logp(alternative), and the teacher-forced rate. **Divergence between the
margin and the decision rate is itself the finding** — it means a removal moved confidence
without moving any decision.

⚠️ The **affordability axis must not be used**: its two options differ in length (3.56 vs 4.88
tokens; the aligned item shorter in 66% of pairs), so a summed-logp comparison measures string
length. All six of our values were inflated ~+0.24 against Figure 2. The America axis
reproduces the paper across all six released conditions to ±0.03.

**Protocol:**
- Aggregate per-document influence over the query set; take top-k for k in a grid chosen from
  the measured concentration curve (top-k mass / Gini), not a fixed guess.
- Remove top-k, retrain the affected stage(s), measure Δf.
- **Control: random-k removal at the same k.** This is load-bearing — it cancels the
  quantity-removed effect so the difference isolates the influence signal. (SOURCE's own random
  baseline is *class-matched*, not uniform; the analogue here is domain-matched.)
- Also run the flip: remove the most *negatively* influential; misalignment should rise.
- If influence rankings do not beat random removal, the finding is "single-checkpoint LoRA
  influence does not capture what matters here" — report it; it motivates multi-stage methods.

**Staged plan: 8B first, then 32B.** Method comparison (grad-dot, grad-cos, SOURCE, ICL, and
any other candidate) runs on the cheap 8B setting; only the winner moves to 32B philosophy.
ICL scores for all 6,400 cheese MSM documents are already computed and published in the same
format as the others — `bergson/cheese/icl/icl_cheese8b_A_aftonly-america-eval_*/`
(`icl_score.npy`, row-indexed, `higher_is_better`; see its `report.json`).
⚠️ **Prerequisite**: the 8B measurable quantity must actually respond to the known MSM effect.
§3b found the published cheese *preference* evals do not separate the released adapters
(0.520 vs 0.500 base) — a binary A/B probe. Switching f to logp is the first thing to try;
if f still does not move, the 8B setting cannot rank methods and the comparison must move to
32B or to a different 8B task.

## 6. Repo structure## 6. Repo structure & conventions

Our work lives under a top-level `tda/` package (see §2b(5)); the upstream repo's
`src/` and `evals/` are left untouched so the fork can track upstream.

```
tda/
  configs/     # frozen eval split, checkpoint registry
  evals/       # AM runner, scoring, query-set construction
  influence/   # masking, grad extraction, projection, EK-FAC, scoring
  analysis/    # H1-H4 scripts, plots
  retrain/     # counterfactual AFT retraining
  modal/       # thin Modal wrapper (entrypoints only)
results/       # parquet scores, eval jsons, figures  (gitignored)
data/          # downloads; never commit corpora       (gitignored)
```

Conventions: yaml configs + checkpoint registry drive everything; deterministic
seeds logged; figure scripts read only from `results/`; **unit tests for span
extraction and loss masking** — these are where silent bugs live, because a wrong
mask still trains and still yields plausible influence scores.

## 7. Milestones & progress

Tracked in [`STATUS.md`](STATUS.md) — not duplicated here.

## 8. Known risks / honesty notes
- ~~Released checkpoints may not cover all cross-pairing cells or may lack MSM-only intermediates~~ → **resolved by Phase 0**: MSM-only intermediates *are* released (no midtraining compute needed); cross-pairing cells are *not* (retrain, ~1–2 GPU-h each).
- 🔴 **AFT corpora for R / V+ / R+ are unpublished, and the author has not responded.** This blocks Tier B entirely (§1b) and 2 of the 5 §4.3 gate cells. Mitigations, revised 2026-09-02:
  - **(b) is much cheaper than first estimated, and is now the plan.** The expensive artifact is the MSM corpus (~50M output tokens, ~$700–1,500/spec) — but **MSM checkpoints for all three specs are already released**, so we never need to regenerate one. Only AFT needs regenerating (~7M output tokens, **~$150–300/spec**), after which we train the three cross-paired cells from the released MSM adapters (~$25 compute). **Total ~$200–350 for H1's exact design.**
  - The earlier objection — "regenerated data isn't what trained the released checkpoints" — **only applies if we compare against their checkpoints.** If we train all three arms ourselves on one regenerated AFT set, the training-data-held-fixed logic is fully restored. What we lose is comparability to Figure 14's *absolute* numbers, which H1 never needed.
  - Residual risk: our AFT(R+) may not reproduce the qualitative factorial (MSM(V+) best). H1 still runs, but the framing weakens from "explains the paper's finding" to "explains a factorial we constructed". Test this explicitly before running H1.
  - (a) obtain from the author — still preferred, unanswered. (c) re-scope to `philosophy` — now reframed as Tier A1 rather than a fallback: it is a genuine mechanism test on a real safety task.
- 🔴 **No training code was open-sourced.** We must reimplement the AFT LoRA SFT recipe. **The obvious validation is impossible**: retraining a *matched* R/V+/R+ cell needs the very AFT data we lack. Validate instead on a **complete public triple** (MSM ckpt + AFT data + released MSM+AFT ckpt) — Llama-8B `cheese` (fast iteration) or Qwen-32B `philosophy` (slow) — checking both the eval number and the cosine of the AFT delta against the released adapter.
- ⚠️ **The SFT loss-masking convention is unverified.** §5.1 assumes assistant-only, the common chat-SFT default, but full-sequence is a real alternative and no training code exists to check against. This matters because influence must mirror the *actual* training objective — otherwise §5.4's removal test fails for reasons unrelated to whether influence works. `tda/influence/masking.py` supports both; **resolve it empirically** in the trainer-validation run above by trying each and keeping whichever better reproduces the released adapter.
- MSM document provenance is stripped to a top-level `domain` in the released corpora → H4's "grouped by spec-section provenance" is only coarsely possible; say so, or get the richer metadata. **Partly mitigated for cheese**: seven axes re-derived and published at `Taywon/msm-llama-pro-america-labels` (§5.3 H5) — but they are *derived*, not recovered, and two of the seven are too unreliable to use. Philosophy remains un-derived.
- IT-mix split/ratio undocumented → the null-distribution control and training-set scope are not yet pinned down.
- Single-sided logp query is length/fluency-confounded → contrastive twin logged everywhere; if the two disagree wildly, the contrastive becomes primary and we say so.
- LoRA-subspace influence ignores any base-model pathway; fine for AFT-stage claims, but do not phrase results as "influence on the model", phrase as "influence via the AFT stage".
- AM evals were hill-climbed by the original authors; that's why all confirmatory numbers come from the frozen held-out split only.