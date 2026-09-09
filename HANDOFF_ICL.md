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
2. **Secondary, and scientifically more valuable — be the first method that reads
   SIGN.** Every method so far, in every direction and seed, lands *above* random:
   removing supposed proponents also raises alignment. Nothing yet distinguishes
   "which documents matter" from "which way they push". A method whose
   proponent-removal arm lands **below random** would be the strongest result on
   this project, worth more than beating +0.107.
3. Report the **opponents − proponents gap**. ICL's is +0.005, the worst of the
   four (EK-FAC +0.073, SOURCE +0.033, grad-dot +0.065). It is the cheapest
   directional signal and needs no random control.

## 3. Do this first — is the ranking mostly noise?

**Before improving anything, measure how much of ICL's ranking is real.**

ICL(z) is a difference of two rates over 400 items, so it carries binomial noise of
roughly ±0.025 per document. The observed spread of `icl_all` is small relative to
that. **Split the 400 eval items into two halves, compute ICL(z) independently on
each, and correlate the two rankings.** That split-half correlation is ICL's
reliability ceiling — no downstream cleverness can exceed it.

If it is low (say < 0.3), the ranking is mostly sampling noise and **the single
highest-value fix is more eval items per document**, not a better formulation.
Everything in §4 is secondary to this. This costs no GPU: the per-document
generations are already on the volume.

## 4. Ideas worth trying, cheapest first

1. **More eval items per document.** Directly attacks §3. Currently 400.
2. **Use a continuous readout instead of a rate.** The decision rate throws away
   confidence. The logprob margin logp(aligned) − logp(alternative) is continuous,
   lower-variance, and free from the same forward pass. ⚠️ Use the **America** axis
   only — the affordability axis's two options differ in length (3.56 vs 4.88
   tokens) so a summed-logp comparison there measures string length (`CLAUDE.md`
   §5.4).
3. **Length/position controls.** Documents average ~1,800 tokens and vary widely.
   Regress ICL(z) on document length and check whether the ranking survives
   residualisation. The gradient methods have a known length confound (grad-dot's
   removal sets run 0.81x/1.22x the corpus mean length); ICL may have its own.
4. **Vary the checkpoint.** Scores are computed on the **AFT-only** checkpoint, so
   each document is genuinely new information (the final MSM+AFT checkpoint has
   already memorised them). Worth testing MSM+AFT anyway, and reporting both.
5. **Multi-document contexts.** ICL is currently one document at a time and so
   cannot see interactions. Try small sets, and compare the set's measured effect
   against the sum of its members' individual scores — the gap is a direct estimate
   of how badly the additive assumption fails, which is the assumption the removal
   test actually stresses when it drops 640 documents at once.
6. **Chunk long documents.** Attribute at paragraph level and aggregate; a 1,800-
   token document may carry one decisive passage.
7. **Order/recency.** Prepending vs appending, and position within context.

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

- **💰 Budget is SHARED and nearly exhausted.** `CLAUDE.md` §2b(0): $500 lifetime,
  **~$369 spent, ~$131 left**, and a k-sweep may claim most of it. **Confirm your
  allocation with the user before spending anything.** One removal arm ≈ $11; a
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
