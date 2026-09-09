# Handoff — port EK-FAC and SOURCE attribution to the 32B setting

**Goal.** Run midtraining-document attribution on **Qwen2.5-32B `philosophy`** — a
real agentic-misalignment task on real spec data.

🔴 **The deliverable is a WORKING PIPELINE and ONE influence estimate per method —
not statistics.** Get attribution to run end-to-end at 32B and produce a ranking
you trust. That is the whole job.

- **One run per method. No seed replication, no repeats, no variance estimates.**
  Those matter when you are comparing methods to each other, which is the 8B
  setting's job and is already done there (§2).
- **No removal test** (§6). It is not funded here and would consume the budget
  several times over.
- Nothing here needs to be statistically significant. It needs to be *correct* —
  right modules, right checkpoint, right sign convention, right row alignment — and
  documented well enough that the expensive experiments can be built on it later.

If you find yourself running something twice for error bars, stop: that is out of
scope and the budget cannot absorb it.

**Budget: $500, until tomorrow.** Both EK-FAC and SOURCE are targets. §6 phases
them — EK-FAC and grad-dot first because they need no retraining, SOURCE second
behind an explicit gate — and names the subsampling lever that makes SOURCE fit.
Read it before launching anything.

## 0. Your role

You own the engineering and the sequencing. Authoritative here: the **setting**
(§1), the **numbers already established** (§2), the **engineering constraints**
(§3), the **budget plan** (§6), the **rules of engagement** (§8). The specific
implementation route in §4–5 is my best guess and you should improve on it.

Do not re-derive what §1–2 record — those were expensive to establish. Do
challenge the plan: if the storage arithmetic or the staging is wrong, say so with
numbers. Read [`CLAUDE.md`](CLAUDE.md), [`STATUS.md`](STATUS.md) §7,
[`DECISIONS.md`](DECISIONS.md) §H2/§H4–H7.

---

## 1. The setting — verified, do not re-derive

### Model: Qwen2.5-32B-**Instruct**, not a base model

Verified across all 140 released `adapter_config.json` files (`CLAUDE.md` §2(4)):

| setting | declared base | kind |
|---|---|---|
| cheese / toy specs (8B) | `meta-llama/Llama-3.1-8B` | 🔵 **true base**, no chat template |
| **Qwen2.5-14B / 32B** | `…-Instruct` | **instruct**, has chat template |

The paper's "train the base model" means *the model before MSM*, which for every
Qwen arm is the **Instruct** model. **Llama-3.1-8B (cheese) is the only true base
model in the project**, so the 8B pipeline's no-chat-template handling does **not**
carry over — 32B has a real template and `apply_chat_template` works.

⚠️ Naming trap: `Qwen/Qwen3-32B` has no `-Instruct` suffix but IS post-trained.
Irrelevant here (we use Qwen2.5), but `CLAUDE.md` §2(4) explains why you infer from
the chat template, never the name. **Never use Qwen3-32B** — it is saturated and
the factorial pattern inverts there.

### Data

| | |
|---|---|
| MSM corpus | `chloeli/msm-qwen-philosophy-spec` — **13,201 documents** |
| AFT (no-CoT, primary) | `chloeli/aft-no-cot-qwen2.5-philosophy-spec` — **9,963 samples** |
| IT mix | **Table 2 mix**, 10,000 samples, `chloeli/sft-it-mix` split `train_clean` subsampled |
| Released final ckpt | `qwen-2.5-32b-philosophy-spec-msm-aft-no-cot` (+ a 1k…80k data-scaling ladder) |
| max seq len | **8192** |

⚠️ **This is NOT the cheese IT mix.** `CLAUDE.md` §5.1: §4–5 (philosophy/Qwen) uses
the Table 2 mix at max len 8192; §3 (cheese/Llama) uses a *simple* No-Robots+MMLU
mix at max len 4096. Applying the wrong one wasted an AFT retrain once already.

### Training hyperparameters — verified from the released 32B adapters

Read directly from `adapter_config.json` on
`chloeli/qwen-2.5-32b-philosophy-spec-msm-aft-no-cot` and `…-msm`
(both identical), and matching the paper's Appendix B.4 ("All models"):

| | value |
|---|---|
| `base_model_name_or_path` | **`Qwen/Qwen2.5-32B-Instruct`** |
| LoRA rank `r` | **64** |
| `lora_alpha` | **128** |
| `lora_dropout` | **0.0** |
| `target_modules` | **all 7**: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| `bias` | `none` · `task_type` `CAUSAL_LM` · `use_rslora` `False` · `init_lora_weights` `True` |
| optimiser | AdamW, **lr 1e-4**, **cosine**, **5% warmup**, **weight decay 0.01** |
| epochs | **1** |
| max seq len | **8192** |
| hardware (paper) | 32B on **4×H200** for training |

⚠️ **`lora_dropout=0.0` matters**: eval-mode forwards are numerically safe, and data
order becomes the only training-nuisance channel — which is what makes a seed noise
floor well-defined.

🔴 **Batch size is the one free parameter — the paper never states it**
(`CLAUDE.md` §5.1). It is the leading suspect whenever step magnitude misses, and it
must appear in every run name (`CLAUDE.md` §2b(4b)); two runs differing only in
batch size once merged into one checkpoint directory (`DECISIONS.md` §H1).

**RULE: follow the paper's parameters, or the released `adapter_config.json`, and
change nothing silently.** Where the two agree, that value is settled and is not a
tuning knob — deviating breaks comparability with the released checkpoints and with
our own 8B results. Where the paper is silent (batch size), pick deliberately,
record the choice in the run name, and say so in the writeup. If you believe a
parameter must change, state the reason and the expected consequence first.

**`target_modules` includes the MLPs.** That is direct evidence for the
all-7-projections rule in §3: the subspace that trained is all seven, so the
subspace we attribute must be all seven.

### Evaluation — how the paper measures these models

Agentic-misalignment evals, **27 conditions** (3 harmful actions × goal-conflict
conditions × 8 goal types). Settled parameters, all in `CLAUDE.md` §2b:

- **Temperature 0.7** (§2b(2b)) — settled empirically; their repo contradicts
  itself. 1.0 roughly halves measured misalignment and induces grader refusals.
- **Score `classifier_verdict`, NOT `harmful`** (§2b(2c)) — the paper counts a
  transcript as misaligned if the model *decided* to act, "regardless of whether it
  was executed successfully". Using `harmful` missed the baseline by 7.4σ and
  *inverted* the V+/R+ ordering.
- **Grader: Sonnet 4.6** (§2b(3)), matching the paper's judge.
- **100 rollouts per condition** (§2b(2)), not the paper's 300. SEM ≈ 0.009.
- **Dev/held-out split is frozen** (§4.2): 14 dev / 13 held-out. Attribution uses
  dev; confirmatory claims use held-out only.

### The influence query is NOT an MCQ — this differs fundamentally from cheese

At 8B the query was a two-option preference item. Here it is
**logp of the misaligned action span** (`CLAUDE.md` §2(2), §5.2).

🔴 **Span extraction is a solved trap — use `tda/evals/spans.py`, do not reinvent.**
"The final tool call" is WRONG and fails silently: measured on 263 real harmful
transcripts, the last action block differs from the actually-harmful one in
**65.4%** of cases, because harmful transcripts routinely end with an *aligned*
action placed after the harmful one. The correct rule anchors on the graders' own
criterion (recipient line for exfiltration; override code 4429 for murder) and
achieves 263/263 localisation. If harm cannot be localised, emit **no query**.

## 2. What 8B already established — this should shape your priorities

Cheese 8B, removal test, 4 methods × 2 directions, up to 3 seeds
(`STATUS.md` §7.2):

| method | opponents removed | proponents removed | ranking cost |
|---|---|---|---|
| EK-FAC | **+0.107 ± 0.032** | +0.033 ± 0.016 | 92 min |
| SOURCE | +0.070 ± 0.026 | +0.037 ± 0.028 | 235 min |
| grad-dot | +0.075 (n=1) | +0.010 | **10 min** |

Three findings that bear directly on what to port:

1. **SOURCE did not beat EK-FAC.** Paired by seed, EK-FAC − SOURCE = +0.037 ± 0.030,
   t(2)=1.24, **not significant**. And our SOURCE matched the paper's own §5.3
   multi-stage recipe (L=2, TDA on the first segment, 2 checkpoints per segment —
   the same per-segment density as Bae et al.'s default C=6/L=3). So this is a
   **non-replication of SOURCE's motivating case**, not a budget artifact
   (`STATUS.md` §7.5).
2. **No method reads sign.** Every method, direction and seed lands *above* the
   random control — removing supposed proponents also raises alignment.
3. **grad-dot is within the band at 1/23 the cost.**

**Implication for you.** SOURCE is the most expensive thing to port, and the 8B
evidence gives no reason to expect it to *win*. That is precisely why porting it is
worth doing: it tests whether the 8B null was a **small-model / toy-task artifact**.
Cheese is a preference task on a base model; philosophy is real safety data on an
instruct model, and SOURCE's whole claim is about multi-stage pipelines. A null that
survives the move to a real task is a much stronger result than a null on a toy one
— and a null that *does not* survive tells us the 8B comparison was measuring the
wrong thing. Both outcomes are worth the money. Sequence it second (§6) because it
depends on Phase 1's engineering, not because it is optional.

## 3. The engineering constraint — this is the actual problem

🔴 **EK-FAC factor storage is the binding constraint and it does not shrink with
LoRA.** Measured (`STATUS.md` §3b): 98.6 GB for a 0.5B model. Per checkpoint the
cost is `Σ_modules (d_in² + d_out²)`, and the pipeline holds **~12 such sets at
once**. Factors are sized by **layer dimensions, not adapter rank**, so r=64 buys
nothing here.

| scope | 8B | **32B** |
|---|---|---|
| all 7 projections | ~1.2 TB | **~7.8 TB** |
| ~~attention-only~~ (MLP is ~86% of it) | ~79 GB (bf16) | ~~~1.1 TB~~ — 🔴 **not an option, see below** |

🔴 **ATTENTION-ONLY IS FORBIDDEN. Attribute all 7 projections.** An earlier plan
allowed dropping the MLPs at 32B on storage grounds; that carve-out is **revoked**
(`DECISIONS.md` §B3a). The released adapters train `q,k,v,o,gate,up,down` — all
seven — so attributing attention alone scores a small fraction of the subspace that
actually moved during training, and MLP is ~86% of the factor mass precisely because
that is where most of the parameters are. Solving the storage problem is the job;
shrinking the question is not an acceptable substitute.

**Modal `ephemeral_disk` caps at 3 TiB per container (≈3.30 TB), and all-module
fp32 is 7.8 TB.** So you must get roughly 2.4× under. Routes, none verified — this
is the core engineering decision and I expect you to improve on this list:

1. **bf16 factors.** Halves it to ~3.9 TB. Necessary but **not sufficient on its
   own** — still just over the cap.
2. **Hold fewer factor sets concurrently.** The 7.8 TB figure is
   `Σ_modules (d_in² + d_out²)` × **~12 sets held at once**. One set is ~650 GB
   fp32 / ~325 GB bf16 at 32B. If that 12 is a bergson buffering choice rather than
   an algorithmic requirement, cutting it to 6 puts bf16 at **~1.95 TB — comfortably
   inside the cap**. **Read `bergson/hessians/` and find out; this is the highest-
   value thing to check first.**
3. **Write factors to a Modal Volume instead of ephemeral disk.** Volumes are not
   bound by the 3 TiB per-container limit. Slower I/O, but it removes the ceiling
   rather than working around it.
4. **Shard across ranks.** bergson's own examples use `nproc_per_node: 8` and its
   factor directories are `*_sharded`. ⚠️ Note carefully: `nproc_per_node` spawns
   processes **inside one container**, which share that container's ephemeral disk —
   so ranks alone do **not** multiply your disk budget. Sharding helps with *memory*
   and *throughput*, not the storage cap, unless combined with (3).

⚠️ `DECISIONS.md` §H2: all-module KFAC OOMs at `token_batch_size` 8192 because MLP
factors make each token far costlier. That run dropped to `token_batch_size` 2048
with `max_batch_size` 16. But note **`token_batch_size` must stay ≥ the longest
document**, and philosophy runs to 8192 — so you cannot simply lower it here the way
the 8B run did. Resolve this explicitly; it is the first thing likely to bite.

### GPUs — the 2×H100 limit is lifted for 32B

The 8B pipeline hard-codes `gpu="H100:2"` (`bergson_app.py:1468`) and `gpu="H100"`
elsewhere. **Those were 8B choices, not a policy.** For 32B:

- **Use 4×H100 or 8×H100 freely.** Appendix B.4 puts 32B on 4×H200 for *training*;
  attribution needs more, and sharding factors across 8 ranks is the intended
  scaling path.
- Make the GPU count a **parameter**, not a literal, so the 8B path is unaffected.
- Budget arithmetic: H100 ≈ **$4.56/GPU-h**, so 8×H100 burns **~$36/hour**. At that
  rate a single bad 8-GPU run costs more than the entire 8B removal test. **Smoke-test
  on a 200-document subset before any full pass.**

## 4. Suggested route

1. **Reproduce the storage arithmetic** for Qwen2.5-32B (64 layers, and check the
   GQA head dims — k/v projections are much smaller than q/o, which is why the 8B
   numbers came out as they did). Confirm or correct the 7.8 TB / 1.1 TB figures
   *before* provisioning.
2. **Build the query set**: run AM dev evals on the released checkpoint, extract
   misaligned action spans with `tda/evals/spans.py`, target ≥200 queries. Cap
   prompts at 8192; if longer, truncate from the **middle of the email dump, never
   the tail**.
3. **EK-FAC first** (see §6 — it needs no retraining).
4. **grad-dot** as the cheap control, all 7 projections, on the same checkpoint.
5. **SOURCE**, behind the §6 gate: retrain the MSM trajectory, chain AFT onto it,
   then score with L=2 and 2 checkpoints per segment, segment-masked to midtraining
   (`stage_masked_score`). Re-score EK-FAC and grad-dot on the same document
   subsample so the three-way comparison is like-for-like.

## 5. What each method needs — and why the costs differ so much

| | needs | retraining required? |
|---|---|---|
| **EK-FAC** | one checkpoint + curvature over D1∪D2 | 🟢 **No** — the released final checkpoint is exactly what it wants |
| **grad-dot** | one checkpoint | 🟢 **No** |
| **SOURCE** | the **whole training trajectory**, per-segment Hessians | 🔴 **Yes** — MSM *and* chained AFT retrained at 32B (Phase 2, §6) |

🟢 **This is the key lever.** Paper §5.3 gives implicit-differentiation methods the
**union D1∪D2** at the final checkpoint precisely because they cannot separate
stages — so **EK-FAC and grad-dot at 32B need no training at all.** MSM-only
intermediates *are* released (`CLAUDE.md` §8), but no intermediate *trajectory* is,
which is why SOURCE alone forces two full 32B retrains.

Note the estimand differs: EK-FAC over the union scores MSM and AFT rows together
and you take the MSM block; SOURCE is segment-masked to midtraining. That asymmetry
is the paper's own design, and it is what our 8B numbers compared.

## 6. Budget plan for $500 — two phases, both funded

Estimates scaled from measured 8B wall times (`STATUS.md` §0a cost model);
**treat them as ±50% and re-price from your own smoke test before committing.**

### Phase 1 — the methods that need no retraining (~$200–250)

| item | estimate |
|---|---|
| AM dev eval + query-set construction | ~$40 |
| **EK-FAC**, **all 7 projections**, storage solved per §3 | ~$130–190 |
| **grad-dot**, all-module, same checkpoint | ~$40–60 |

🟢 Both use the **released** final checkpoint (§5), so Phase 1 buys two rankings on
real safety data with **zero training**. It also proves out the sharding solution,
which Phase 2 depends on.

### Phase 2 — SOURCE (~$220, and it is a target, not a stretch goal)

| item | estimate |
|---|---|
| MSM retrain at 32B (13,201 docs, 8192 max len) | ~$60–80 |
| Chained AFT retrain (`init_run=<msm_run>`, one trajectory) | ~$40–50 |
| SOURCE ranking, L=2, 2 checkpoints/segment, **on a subsample** | ~$100–120 |

**The lever that makes SOURCE affordable: subsample the corpus, not the method.**
`CLAUDE.md` §2b(0) records that the cost driver for multi-stage work is **corpus
tokens, not checkpoints**. Scoring all 13,201 philosophy documents at 32B
extrapolates to **~$300–400** on its own (the 8B run cost $36 for 6,400 documents /
9.5M tokens on 2×H100; 32B is ~4× per token and philosophy is ~2× the tokens).

Score a **stratified subsample of ~4,000 documents instead** — the philosophy corpus
ships a `domain` field with 8 values, so stratify on it — and the ranking cost drops
to ~30%. **Re-score EK-FAC and grad-dot on the identical subsample** so the
three-way comparison stays exactly like-for-like; that re-score is cheap because
their factors already exist.

Do **not** economise by cutting checkpoints below 2 per segment. That is the
paper's own per-segment density (Bae et al. use C=6 over L=3), and going lower
reintroduces exactly the "under-resourced SOURCE" objection that `STATUS.md` §7.5
just retired.

**Phase 1 + Phase 2 ≈ $420–470 of $500.** Feasible, with little margin — which is
why the gate below matters.

### Gate between the phases

Start Phase 2 only when **all** of these hold:

1. EK-FAC has produced a complete score store at 32B — the sharding works.
2. You have **re-priced from measured wall times**, not from my estimates. If your
   measured Phase 1 cost overshoots by more than ~30%, stop and report rather than
   starting a trajectory retrain you cannot finish.
3. The trajectory plan is concrete: MSM run name, AFT chained with
   `init_run=<msm_run>`, and `assert_single_trajectory` passing. `DECISIONS.md` §H1
   is the incident where two runs merged into one checkpoint directory and a chained
   AFT silently continued the wrong parent.

### Not funded here: the removal test

One 32B removal arm ≈ **$120** (MSM + AFT retrain), and the 8B work showed you need
both directions plus a random-k control to interpret one — ~$360 minimum for a
single method. That does not fit alongside the rankings. Deliver rankings and a
re-priced cost model so it can be funded properly.

### Free, and worth reporting either way

Rank agreement among the methods at 32B, against the 8B values (grad-dot↔EK-FAC
**0.628**, SOURCE↔EK-FAC **0.411**, SOURCE↔grad-dot **0.245**). If the cheap method
tracks the expensive one at 32B too, that is a real result about method choice at
scale, bought for nothing. If the ordering *changes* at 32B, that is a more
interesting result still — it would mean the 8B nulls were a small-model artifact,
which is the whole reason SOURCE is worth porting.

## 7. Traps that have already cost this project time

- **Sign convention** (`DECISIONS.md` §H5, §H7 — it has bitten twice). bergson
  stores are **loss-signed: proponents are NEGATIVE**. `_oriented` applies this.
  Sorting descending selects **opponents**. Any code that sorts a score array must
  state its convention at the sort site, and polarity must be named
  `proponents`/`opponents`, never `top`/`bottom`.
- **Silent drops** (§H6). A guard written as `if path.exists()` turns a crashed run
  into a quietly smaller result. Raise instead.
- **Row alignment.** Scores are indexed by document id; our datasets define order in
  a manifest. A permutation leaves every aggregate statistic unchanged and destroys
  only the result. Assert counts and read the top documents.
- **Persist before you post-process** (§H4). A completed SOURCE run was destroyed by
  analysis that ran before artifacts were copied to the volume. Copy on `rc == 0`,
  unconditionally, then analyse in a try/except.
- **Never block on `.remote()`** — a client-side gRPC deadline has killed a run
  mid-flight. Use `modal run --detach`, spawn, print the id, exit, poll the volume.
- **Ephemeral Modal apps die with their client**, worse than documented; one run was
  lost at 79% of eigendecomposition.

## 8. Rules of engagement

- **💰 $500, until tomorrow.** Separate from the 8B pool (`CLAUDE.md` §2b(0): $800
  lifetime, ~$369 spent) and from the ICL session's $200. Price from *measured*
  rates. At 8×H100 you are spending ~$36/hour — **smoke-test on a subset first**.
- **🔴 Do not touch other runs.** Two other sessions are active (an ICL methodology
  session, and this repo's main session running a k-sweep), plus a foreign `msm-tda`
  app nobody here started (§H3). Do not delete anything under `bergson/cheese/` or
  the philosophy directories on `msm-tda-results`.
- **Naming**: `CLAUDE.md` §2b(4b), via `tda/influence/source/naming.py::run_name`.
  Setting is `phil32b`.
- **Do not regress the 8B path.** Make GPU count, module scope and batch sizes
  parameters; the cheese pipeline must still run unchanged.
- Adapters from the 8B removal arms are on HF at
  `Taywon/msm-tda-cheese8b-america-removal` (private) if you need reference points.

## 9. Definition of done

1. EK-FAC scores over the philosophy MSM documents against ≥200 AM queries,
   **over all 7 projections** — not attention-only — with the storage solution
   stated explicitly.
2. grad-dot scores over the same documents and queries.
3. Their Spearman and top-k Jaccard, compared against the 8B values in §6.
4. The storage/sharding solution written up in `DECISIONS.md` — including what did
   not work. This is the reusable artifact; the rankings are downstream of it.
5. **SOURCE scores** over the same documents and queries, from a retrained MSM→AFT
   trajectory at 32B, L=2, segment-masked to midtraining.
6. The three-way Spearman and top-k Jaccard at 32B set beside the 8B values, with a
   statement of whether the 8B ordering survives the move to a real safety task.
7. A re-priced cost model for 32B from measured wall times, so the removal test can
   be costed properly when someone funds it.

If the budget runs out between Phase 1 and Phase 2, items 1–4 and 7 are a complete
deliverable on their own — report them rather than starting a trajectory retrain you
cannot finish.
