# Handoff — port EK-FAC and SOURCE attribution to the 32B setting

**Goal.** Run midtraining-document attribution on **Qwen2.5-32B `philosophy`** — a
real agentic-misalignment task on real spec data — and get a ranking that the
removal test can validate, as we did at 8B on cheese.

**Budget: $500, until tomorrow.** That does not cover everything below. §6 says
what to buy first and what to drop; read it before launching anything.

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

**Implication for you.** SOURCE is by far the most expensive thing to port and the
8B evidence gives no reason to expect it to win. The honest framing is that porting
SOURCE tests whether the 8B null was a **small-model / toy-task artifact** — a real
question, since cheese is a preference task and philosophy is real safety data. But
it is a *test of a negative result*, not a favourite. Price it accordingly.

## 3. The engineering constraint — this is the actual problem

🔴 **EK-FAC factor storage is the binding constraint and it does not shrink with
LoRA.** Measured (`STATUS.md` §3b): 98.6 GB for a 0.5B model. Per checkpoint the
cost is `Σ_modules (d_in² + d_out²)`, and the pipeline holds **~12 such sets at
once**. Factors are sized by **layer dimensions, not adapter rank**, so r=64 buys
nothing here.

| scope | 8B | **32B** |
|---|---|---|
| all 7 projections | ~1.2 TB | **~7.8 TB** |
| attention-only (MLP is ~86% of it) | ~79 GB (bf16) | **~1.1 TB** |

**Modal `ephemeral_disk` caps at 3 TiB per container, so all-module EK-FAC at 32B
does not fit one container.** Your options, and picking among them is the core
engineering decision:

1. **Shard factors across ranks.** This is how bergson is designed to scale — its
   own examples use `nproc_per_node: 8` and factor directories are `*_sharded`.
   8 ranks puts ~975 GB/rank all-module, or ~140 GB/rank attention-only. **Try this
   first.**
2. **Attention-only.** ~1.1 TB total; `DECISIONS.md` §B3/§H2 records that
   attention-only was the 32B plan for exactly this reason, stated as an
   approximation to be tested against all-module grad-dot. Cheap and safe, but it
   is a real approximation and must be labelled as one.
3. **bf16 factors** — halves it, already assumed in the numbers above.

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
4. **grad-dot** as the cheap control, all-module, to test what attention-only EK-FAC
   gives up.
5. **SOURCE only if budget survives.**

## 5. What each method needs — and why the costs differ so much

| | needs | retraining required? |
|---|---|---|
| **EK-FAC** | one checkpoint + curvature over D1∪D2 | 🟢 **No** — the released final checkpoint is exactly what it wants |
| **grad-dot** | one checkpoint | 🟢 **No** |
| **SOURCE** | the **whole training trajectory**, per-segment Hessians | 🔴 **Yes** — MSM *and* chained AFT retrained at 32B |

🟢 **This is the key lever.** Paper §5.3 gives implicit-differentiation methods the
**union D1∪D2** at the final checkpoint precisely because they cannot separate
stages — so **EK-FAC and grad-dot at 32B need no training at all.** MSM-only
intermediates *are* released (`CLAUDE.md` §8), but no intermediate *trajectory* is,
which is why SOURCE alone forces two full 32B retrains.

Note the estimand differs: EK-FAC over the union scores MSM and AFT rows together
and you take the MSM block; SOURCE is segment-masked to midtraining. That asymmetry
is the paper's own design, and it is what our 8B numbers compared.

## 6. Budget plan for $500

Estimates scaled from measured 8B wall times (`STATUS.md` §0a cost model);
**treat them as ±50% and re-price from your own smoke test.**

| item | estimate | verdict |
|---|---|---|
| AM dev eval + query-set construction | ~$40 | **buy** |
| EK-FAC, attention-only, sharded over 8 ranks | ~$110–150 | **buy** |
| grad-dot, all-module, same checkpoint | ~$40–60 | **buy** |
| Removal test: 1 arm = MSM+AFT retrain at 32B | **~$120/arm** | see below |
| SOURCE ranking (trajectory + per-segment Hessians) | ~$280 | **defer** |
| MSM + AFT retrain for SOURCE's trajectory | ~$120 | **defer** |

**Recommended: EK-FAC + grad-dot rankings, and stop before the removal test.**
That lands ~$200–250 and delivers the port's hard part — the engineering — plus two
rankings on real safety data. A single 32B removal arm costs ~$120, and the 8B work
showed you need **both directions and a random-k control** to interpret one, i.e.
~$360 minimum for one method. That does not fit alongside the rankings.

**Do not** spend the whole $500 on one SOURCE run. Its 8B result was a
non-replication and it is the least likely to pay.

If EK-FAC and grad-dot rankings land under budget, the highest-value remaining
purchase is **rank agreement between them at 32B** (free — just Spearman) compared
against the 8B values (grad-dot↔EK-FAC 0.628, SOURCE↔EK-FAC 0.411,
SOURCE↔grad-dot 0.245). If the cheap method tracks the expensive one at 32B too,
that is a real result about method choice at scale, bought for nothing.

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

1. EK-FAC scores over the 13,201 philosophy MSM documents against ≥200 AM queries,
   with the module scope and sharding actually used stated as an approximation.
2. grad-dot scores over the same documents and queries.
3. Their Spearman and top-k Jaccard, compared against the 8B values in §6.
4. The storage/sharding solution written up in `DECISIONS.md` — including what did
   not work. This is the reusable artifact; the rankings are downstream of it.
5. A re-priced cost model for 32B from measured wall times, so the removal test can
   be costed properly when someone funds it.
