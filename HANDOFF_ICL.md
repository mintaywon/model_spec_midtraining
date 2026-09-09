# Handoff — make ICL attribution actually competitive

**One goal, stated as a number.** ICL currently scores **+0.040** on the removal
test. EK-FAC scores **+0.107**. Get ICL above EK-FAC, or establish why it cannot
go higher. Everything below serves that.

This is an **iterative** brief: propose a change, measure it on the cheap proxy,
and only spend GPU on the removal test when the proxy says the ranking improved.
Read [`CLAUDE.md`](CLAUDE.md) for the durable brief and [`STATUS.md`](STATUS.md) §7
for live state. [`DECISIONS.md`](DECISIONS.md) §H5–H7 are the traps.

---

## 1. What ICL is

`tda/evals/icl.py`. For each midtraining document z:

```
ICL(z) = aligned_rate(eval items | z prepended to context)
       - aligned_rate(eval items | no context)
```

It asks *"if this document were **read** rather than **trained on**, how much would
it move the behaviour?"* — no gradients, no Hessian, no retraining. If it predicts
removal effects it is the cheap practical winner, which is why it is worth pushing.

⚠️ **Its sign convention is the OPPOSITE of bergson's.** Higher ICL = stronger
proponent, already. bergson stores are loss-signed (proponents negative). Do not
pass ICL scores through `_oriented`. Inverting this is exactly `DECISIONS.md` §H7,
which inverted a headline result for half a day.

## 2. The scoreboard you are trying to beat

Removal test, cheese 8B pro-America. Remove k=640 documents (10% of 6,400), retrain
**both** stages, measure the generative decision rate on 200 held-out items.
Baseline (no removal) 0.595 · random-640 control 0.525.

| method | opponents removed | proponents removed | seeds |
|---|---|---|---|
| EK-FAC | **+0.107 ± 0.032** | +0.033 ± 0.016 | 3 |
| SOURCE | +0.070 ± 0.026 | +0.037 ± 0.028 | 3 |
| grad-dot | +0.075 | +0.010 | 1 |
| **ICL** | **+0.040** (z=2.02) | **+0.035** (z=2.00) | 1 |

**Targets, in priority order:**

1. **Primary — beat +0.107** on the opponents arm. That is the number to win.
   ⚠️ ICL's own +0.040 is contaminated (§3) — re-establish it cleanly first, and
   measure your improvements against the clean number.
2. **Secondary, and scientifically more valuable — be the first method that reads
   SIGN.** Every method so far, in every direction and seed, lands *above* random:
   removing supposed proponents also raises alignment. Nothing yet distinguishes
   "which documents matter" from "which way they push". A method whose
   proponent-removal arm lands **below random** would be the strongest result on
   this project, worth more than beating +0.107.
3. Report the **opponents − proponents gap**. ICL's is +0.005, the worst of the
   four (EK-FAC +0.073, SOURCE +0.033, grad-dot +0.065). It is the cheapest
   directional signal and needs no random control.

## 3. Do this first — ICL's current number is contaminated

**`tda/evals/icl.py` line 201 loads the entire 400-item eval set:**

```python
items = list(load_dataset(cfg.eval_set, split="train"))   # all 400
```

But `pro-america-political-opinions` has **only 400 items total**, and
`split_query_sets` divides them 200 `attr` / 200 `eval` precisely so that
"attribution queries and behavioural evaluation must not share prompts ... nothing
may read `eval` during attribution." The removal test measures on the 200 `eval`
items.

**So ICL was scored using the 200 held-out items its removal test is graded on.**
SOURCE, EK-FAC and grad-dot all used `query_america_attr_target` (the clean 200).

Two consequences:

1. **ICL's +0.040 is not comparable to the other three, and is biased in ICL's
   favour** — it had partial access to the grading set and still came last. The
   honest current number is unknown.
2. **Fix this before anything else**: rescore on the 200 `attr` items only, rerun
   the removal test, and report that as ICL's true baseline. Your improvements are
   measured against *that*, not against +0.040.

### Reliability is NOT the bottleneck — don't chase item count

Halving the items doubles the noise, so check what that costs. It is small:

- observed SD of `icl_all` across 6,400 documents: **0.136**
- binomial SE of a rate at n=400 (p≈0.4–0.8): **≈0.022** → at n=200, **≈0.032**

Signal variance dominates noise variance by ~25x at n=400 and ~18x at n=200, so
the ranking is roughly **95–97% signal either way**. Confirm with a split-half
correlation (free — the per-document generations are already on the volume), but
expect it to come back high.

**This matters for where you spend effort.** ICL's problem is not that its
measurement is noisy. It is that *reading* a document is not *training* on it —
a validity problem, not a precision one. More items cannot fix that, and there are
no more items to be had. Attack validity.

## 4. Ideas worth trying

Ordered by expected value after the checks above.

1. **Score on the BASE model, not (only) the AFT-only checkpoint.** Currently
   scores come from `llama-3.1-8b-cheese-aft`. But the document's causal effect in
   the removal counterfactual happens **during midtraining, which starts from the
   base model** — so the base model is the checkpoint where the document actually
   acts, and is arguably the more faithful place to measure it. The AFT-only
   checkpoint has already acquired a task disposition the document then perturbs.
   - The obstacle is format: base Llama-3.1-8B has no chat template and will not
     reliably emit "B)". **Use the teacher-forced margin on base instead of a
     generated decision** — it needs no parsing, so the format problem disappears.
   - Also try **combining base and AFT-only scores** (mean of z-scored ranks, or
     the difference, which isolates what AFT changed about the document's effect).
   - ⚠️ Do **not** score on the final MSM+AFT checkpoint. It has already trained on
     these documents, so the score measures memorisation, not new information.
2. **Chunk long documents.** Attribute at paragraph level and aggregate. Documents
   average 1,489 tokens (max 4,174) and a single decisive passage can be diluted by
   surrounding text.
3. **Multi-document contexts.** Single-document ICL cannot see interactions, but the
   removal test drops 640 documents at once — so the additive assumption is exactly
   what is being stressed. Compare a set's measured effect against the sum of its
   members' individual scores; the gap estimates how badly additivity fails.
   - Context budget is not the constraint people expect: Llama-3.1-8B supports
     131k tokens, so ~8 documents ≈ 12k tokens is comfortable. Effective use
     degrades well before the nominal limit though, and midtraining itself used
     `max_length` 4096 — so **stay in the 2–8 document range** and report the count.
4. **A continuous readout instead of the decision rate.** logp(aligned) −
   logp(alternative) is lower-variance and free from the same forward pass.
   ⚠️ Two caveats: it only works because this eval is **MCQ**, so it will not
   transfer to the 32B agentic-misalignment setting, which is generative — an
   ICL variant that depends on it buys accuracy here at the cost of generality;
   and use the **America** axis only, since the affordability axis's options
   differ in length and a summed-logp comparison there measures string length
   (`CLAUDE.md` §5.4).
5. **Order and position.** Prepend vs append, position within the context. Cheap
   to test, low prior.

**Ruled out — do not spend time here.** A length confound. Measured: ICL score vs
document tokens is Spearman **+0.072**, and the top-640 and bottom-640 sets run
**1.02x** and **1.01x** the corpus mean length. ICL is essentially length-neutral,
unlike the gradient methods (whose removal sets run 0.81x–1.22x). This was checked
so you don't have to.

## 5. Fixed setup — do not vary these, they define comparability

- **Base model**: `meta-llama/Llama-3.1-8B` — a **true base model with NO chat
  template**. `apply_chat_template` raises rather than guessing. Do not substitute
  an Instruct model.
- **Midtraining corpus**: `chloeli/msm-llama-pro-america`, 6,400 documents, in
  corpus order (row i = dataset row i). Scores must be a 6,400-length array aligned
  to that order; a permutation leaves every aggregate statistic unchanged and
  destroys only the result (`tda/influence/source/scores.py` docstring).
- **Eval**: `chloeli/pro-america-political-opinions`, America axis. Prompt is the
  paper's Appendix C.3 form, already assembled in the `question` field. Greedy
  decoding. **Always report `parse_rate` beside the rate** — a midtrained model
  asked for "B)" may write an essay, and a format collapse must not be scored as
  misalignment.
- **Removal-test recipe (both stages)**: LoRA r=64, α=128, all attention + MLP
  projections, `lora_dropout=0.0`, 1 epoch, AdamW lr 1e-4, cosine, 5% warmup,
  weight decay 0.01, batch size 32, max seq len 4096. Midtraining is plain LM loss
  over documents, **not** chat SFT.
- **k = 640** and **seed 42** for any arm you want compared against the table in §2.
  Changing either makes your number incomparable — and if you change k you need
  your own random-k control at that k, because the quantity-removed effect scales
  with k.

## 6. How to test a new ICL variant

1. Compute the new scores over all 6,400 documents. Assert the length.
2. **Cheap gates, before any GPU spend on retraining:**
   - split-half reliability (§3) — did it go up?
   - Spearman against the current ICL ranking — how different is it really?
   - Spearman against SOURCE / EK-FAC / grad-dot. Current ICL is near-orthogonal to
     all three (+0.05 / −0.02 / −0.00). If a variant drifts toward them it is
     rediscovering gradient influence, which is interesting but no longer a cheap
     independent signal.
3. Only if the gates improve: run the removal test, both directions, k=640, seed 42
   (`removal_arm(mode="icl_proponents"|"icl_opponents", k=640, seed=42)` — point it
   at your new score file), then `--action gen_compare`.
4. Record what you tried and what it did, including the failures. A list of ruled-out
   variants is a real result.

## 7. Rules of engagement

- **💰 You have a dedicated $200.** Separate from the main project pool
  (`CLAUDE.md` §2b(0), $800 lifetime, ~$369 spent, ~$165 committed to a concurrent
  k-sweep). Spend within your $200 without asking; come back before exceeding it.
  One removal arm ≈ $11, so a both-directions test ≈ $22 — roughly **9 tests**.
  Scoring passes are cheap; retraining is not, so use the §6 gates. One removal arm ≈ $11; a
  full both-directions test ≈ $22. Scoring passes are cheap; retraining is not.
- **🔴 Do not touch other runs**, and do not delete anything under
  `bergson/cheese/` on the `msm-tda-results` volume (`DECISIONS.md` §H4 lost two
  completed runs to cleanup). A second session may be running a k-sweep.
- **Naming**: `CLAUDE.md` §2b(4b), via `tda/influence/source/naming.py::run_name`.
  Polarity is always `proponents`/`opponents`, never `top`/`bottom` — vague polarity
  labels caused §H7.
- Adapters for every completed arm are on HF at
  `Taywon/msm-tda-cheese8b-america-removal` (private), so you can re-score without
  retraining.
- Use `modal run --detach` and spawn-and-poll; never a blocking `.remote()`.

## 8. Definition of done

Either:
- **an ICL variant above +0.107** on the opponents arm, with the removal test run at
  k=640 seed 42 so it is comparable; or
- **an ICL variant whose proponent arm lands below random** (the sign result — more
  valuable than the primary target); or
- **a documented reliability ceiling** showing ICL cannot beat +0.107 in this setup,
  with the split-half number that establishes it.

All three are publishable outcomes. The third is a real result, not a failure —
write it up as one.
