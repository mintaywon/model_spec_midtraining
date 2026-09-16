# Semantic scorers as midtraining-document attribution — cheese 8B

Independent investigation, 2026-09-10. Budget ~$50. Setting: Llama-3.1-8B
`cheese` (`chloeli/msm-llama-pro-america`, 6,400 documents), the same removal
test, control and baselines as every other method on the scoreboard.

Question: **can a scorer with no gradients, no Hessian and no retraining — a
sentence embedding, or the trained model's own residual-stream geometry — place a
document on the proponent/opponent axis well enough to sit beside EK-FAC, SOURCE,
grad-dot and ICL on the removal scoreboard?** Framed as standalone attribution,
per instruction; the surrogate reading (how much of expensive influence these
recover) is reported where it falls out but is not the spine.

## 0. Pre-registered before any score was computed

Written while the 40-document smoke test was running, before the full extraction
pass was launched. Nothing below this line in §0 was edited afterwards.

**Scorers.** All three have a contrastive form so sign is representable:

| | scorer | construction | sign |
|---|---|---|---|
| S1 | embedding similarity | bge-m3 dense (8k context), cosine of document to query | relevance: none; **contrast**: cos(doc, Q+aligned option) − cos(doc, Q+misaligned option) |
| S2 | activation similarity | mean-pooled residual state of the document vs the query's decision-position state, at the final MSM+AFT checkpoint | relevance: none; **contrast**: vs the state at the aligned letter minus at the misaligned letter |
| S3 | value-direction projection | diff-in-means direction v = mean_q(state at aligned letter − state at misaligned letter); score = (doc mean state − corpus mean) · v | intrinsic |

**Fixed choices.**
- Checkpoint: `runs/msm_A__chain_ck198/checkpoints/checkpoint-504`, the final
  MSM+AFT weights EK-FAC and grad-dot were scored at.
- **Primary layer: 20 of 32.** Other layers (8, 12, 16, 24, 28, 32) are saved
  from the same pass and reported as sensitivity only.
- Queries: the 200 `attr` items only, through `icl2.attr_items`, which refuses
  the eval half.
- Document pooling: attention-masked mean over tokens (primary); final token
  (secondary, reported).

**Rule for choosing the one scorer that gets the removal pair** (k=640, seed 42,
both directions, $22). Applied in this order, on the gates only, without looking
at any removal result:

1. Candidates are the sign-carrying forms only: S1 contrast, S2 contrast, S3.
2. Drop any candidate with split-half reliability over queries below 0.90.
3. Drop any candidate with |Spearman(score, document length)| ≥ 0.30 — the
   gradient methods sit at −0.31/−0.37 and the point of a cheap scorer is not
   to inherit their confound.
4. Drop any candidate that is not two-sided (fraction positive outside
   [0.15, 0.85]).
5. Among survivors, take **S3** if it survives (the most principled
   construction and the one with intrinsic sign), else S2, else S1.
6. If nothing survives, no removal pair is run and that is the result.

**What counts as a result.** The control-free directional gap
(opponents − proponents at shared k and seed) against the existing table:
EK-FAC +0.075, grad-dot +0.065, ICL v2 +0.045, SOURCE +0.035. Also each arm
against the six-arm random control (0.5633 ± 0.0301). By `ICL_LOG.md` §7.5 a
single-seed pair resolves Δ ≈ 0.12 and nothing finer; the gap is reported as a
point estimate with that caveat, not as a ranking among the middle of the board.

### 0.1 Addendum, written 00:52 — still before any score was seen

Two papers read while the smoke test ran, each of which adds one thing:

- **Akyürek et al. 2022, *Towards Tracing Factual Knowledge in Language Models
  Back to the Training Data* (arXiv:2205.11482)** — on fact tracing, both
  gradient-based and embedding-based attribution had *lower* proponent-retrieval
  precision than **BM25**, a bag-of-words retrieval baseline with no access to the
  model. So a lexical retrieval baseline belongs on this table, not as a straw
  man but as the method that won their benchmark. Added as **S0: BM25**, with the
  same relevance / contrast pair as the others ($0, offline). It is also the
  natural companion to `ICL_LOG.md` §1.7, which found EK-FAC's ranking is 18%
  explained by America-word density alone.
- **Billa, *Predicting Where Steering Vectors Succeed* (arXiv:2604.15557)** — a
  diff-in-means direction is only meaningful for the model's output when,
  pushed through the final norm and unembedding, it lands on the target tokens
  (their "output-aligned" regime); concepts can be linearly detectable many
  layers before that. Their diagnostic is a logit lens on the direction, and they
  find **late layers (final quarter of depth) beat the conventional middle-layer
  heuristic**. Two consequences here: (i) the extraction pass now records, per
  layer, whether the S3 direction's logit-lens favours the aligned letter; (ii)
  my pre-registered primary layer 20 is the "middle-layer heuristic" they argue
  against. **I am not moving it** — changing a pre-registered choice after
  reading a paper that predicts it is suboptimal is still changing it after the
  fact — but the sensitivity table will show layers 24/28/32 beside it, and if a
  late layer wins on the *gates* that is reported as such, with this note as the
  reason it was not primary.

Selection rule unchanged. S0 contrast joins the candidate list at the end of the
priority order (S3, S2, S1, **S0**).

## 1. Extraction

Both passes smoke-tested on 8 documents (after one failed launch: `train_image`
pins transformers 4.51.3, which takes `torch_dtype=`, not the `dtype=` the newer
bergson image accepts — both functions crashed on load, and the empty output
directories were the only sign). Activation pass: 7 layers × {mean, last-token}
for every document, plus the 200 attr queries at the decision position and at
the aligned / misaligned letter. Measured 879 tok/s on the 8-document smoke,
which includes model load; the full pass is priced from the running job.

**A diagnostic I shipped was wrong, and the smoke showed it.** I had the
extraction pass logit-lens each query's (aligned − misaligned) letter-state
difference and record whether it favours the aligned letter. It read **0.785 at
layer 8, falling to 0.52 at layer 20 and 0.04 at layer 32**. That is not the
value direction crystallising or decaying; it is *token identity* washing out
with depth. At the letter position, an early-layer state is mostly the " A" or
" B" embedding, so the per-item difference trivially decodes to that letter —
and for an A-aligned item that *is* the aligned letter, tautologically. The
number is 1.0 for a pure token-identity direction and says nothing about value.
Kept in `meta.json`, not used.

The right test is offline and needs no unembedding: across the 200 items,
Spearman between the projection of each query's *decision-position* state onto
v and the model's own teacher-forced margin on that item (already recorded by
the ICL v2 pass). A direction that carries the model's stance must track its
stance item by item; one that only separates " A" from " B" cannot. Reported
per layer in §2.

## 2. Gates

### 2.1 S0 BM25 — computed first because it needs no GPU

| S0 BM25 | frac>0 | distinct | split-half (SB) | ρ EK-FAC | ρ grad-dot | ρ SOURCE | ρ ICL v2 | ρ length |
|---|---|---|---|---|---|---|---|---|
| relevance | 1.000 | 6400 | — | −0.080 | +0.043 | −0.080 | +0.061 | **+0.390** |
| **contrast** | 0.146 | 6400 | **0.898** | +0.107 | +0.069 | **+0.147** | **+0.204** | −0.175 |

Relevance is sign-free and length-loaded (+0.39), as bag-of-words retrieval is.
The contrast form is two-sided (15% positive), length-safe (−0.18), and agrees
with SOURCE (+0.147) and ICL v2 (+0.204) more than any scorer computed so far in
this project agrees with either — it is the closest thing to a common factor
across the gradient-free scorers.

**Gate 2 fails by 0.002** (0.898 against the pre-registered 0.90). Applied as
written; §0's rule exists so that borderline cases are decided before the
numbers are seen, not after. S0 was last in the priority order regardless.

### 2.2 S1 bge-m3 embeddings — 71 s for the whole corpus (~$0.10)

| S1 bge-m3 | frac>0 | distinct | split-half (SB) | ρ EK-FAC | ρ grad-dot | ρ SOURCE | ρ ICL v2 | ρ length | Jaccard vs EK-FAC opp / prop |
|---|---|---|---|---|---|---|---|---|---|
| relevance | 1.000 | 6386 | — | +0.009 | +0.187 | −0.001 | **+0.251** | +0.037 | 0.041 / 0.114 |
| **contrast** | **0.333** | 6398 | **0.974** | **+0.216** | **+0.287** | +0.129 | +0.033 | −0.168 | 0.055 / **0.158** |

**S1 contrast passes all four gates**: reliable (0.974), length-safe (−0.17),
two-sided (33% positive), signed by construction. It is also the cheap scorer
that agrees most with the gradient methods of anything computed on this project
— ρ = +0.287 with grad-dot and +0.216 with EK-FAC, where ICL in every variant
sat within ±0.1 of zero — and its proponent removal set overlaps EK-FAC's at
Jaccard 0.158, twice ICL's. Plain relevance, by contrast, tracks ICL v2
(+0.251) and nothing else, which is the right shape: bare topical similarity is
what in-context reading rewards.

It is third in the priority order, so it runs only if S3 and S2 fail a gate.

**But the agreement is mostly the lexical confound.** Same regression as
`ICL_LOG.md` §1.7 (score ~ America-word density + log length, standardised):

| scorer | lexical R² | β density | β log length | form R² | domain η² |
|---|---|---|---|---|---|
| EK-FAC | 0.184 | +0.287 | −0.280 | 0.235 | 0.096 |
| grad-dot | 0.182 | +0.207 | −0.345 | 0.386 | 0.143 |
| SOURCE | 0.031 | +0.142 | −0.084 | 0.057 | 0.034 |
| S0 BM25 contrast | 0.135 | +0.329 | −0.124 | 0.186 | 0.123 |
| S1 relevance | 0.006 | +0.068 | +0.048 | 0.201 | 0.095 |
| **S1 contrast** | **0.301** | **+0.526** | −0.098 | 0.378 | **0.211** |

S1 contrast is *more* lexically determined than EK-FAC — 30% of its ranking is
America-word density plus length, with the largest density coefficient of any
scorer on the project. Residualising **both** sides on those two covariates:

| pair | raw ρ | partial ρ |
|---|---|---|
| S1 contrast × EK-FAC | +0.216 | **+0.019** |
| S1 contrast × grad-dot | +0.287 | +0.160 |
| S1 contrast × SOURCE | +0.129 | +0.044 |
| S1 relevance × grad-dot | +0.187 | **+0.216** |
| S0 contrast × EK-FAC | +0.107 | −0.050 |
| *(EK-FAC × grad-dot, for scale)* | +0.628 | +0.557 |

So the headline "most gradient-agreeing cheap scorer" is, against EK-FAC,
almost entirely the two methods counting the same words. Against grad-dot a
real +0.16 survives, and plain relevance carries a non-lexical +0.22 with
grad-dot that the confound was *masking*. The contrastive construction — Q +
option text — buys sign by loading on exactly the vocabulary the options
contain, and that is the confound.

The selection rule (§0) has no lexical gate and I am not adding one after
seeing these numbers. It is reported here so that if S1 is the scorer that runs,
the reader knows what it is measuring.

### 2.3 S2 / S3 — the trained model's own geometry (615 s for the corpus, 15.5k tok/s, ~$0.80)

At the pre-registered layer 20, mean-pooled document states:

| scorer | frac>0 | split-half | ρ EK-FAC | ρ grad-dot | ρ SOURCE | ρ ICL v2 | ρ length | Jaccard vs EK-FAC opp / prop |
|---|---|---|---|---|---|---|---|---|
| S2 relevance | 0.549 | — | −0.181 | −0.316 | −0.008 | −0.180 | −0.185 | 0.025 / 0.022 |
| **S2 contrast** | 0.519 | **0.987** | +0.189 | +0.317 | +0.130 | +0.374 | −0.215 | 0.063 / 0.082 |
| **S3 value-direction** | **0.518** | **0.986** | +0.215 | **+0.345** | +0.133 | **+0.372** | −0.229 | 0.068 / **0.122** |
| S3, last-token pooling | 0.537 | — | +0.057 | +0.086 | +0.010 | −0.004 | −0.075 | 0.058 / 0.067 |

S2 contrast and S3 are ρ = **0.993** with each other — one scorer in two
notations (a cosine against per-query directions vs a projection onto their
mean). Last-token pooling is nearly orthogonal to everything and is dropped.
Plain activation relevance is *anti*-correlated with grad-dot (−0.316): the
documents nearest the bare question in representation space are not the ones
training rewards.

**Output alignment.** Spearman over the 200 items between (decision-position
state · v) and the model's own teacher-forced margin, per layer:

| L8 | L12 | L16 | **L20** | L24 | L28 | L32 |
|---|---|---|---|---|---|---|
| +0.05 | −0.01 | +0.78 | **+0.74** | +0.72 | +0.68 | +0.29 |

The direction carries the model's stance from layer 16 on and loses it by the
output layer. The pre-registered layer is inside the aligned regime; Billa's
late-layer heuristic (§0.1) would have picked a *worse* layer here — L28 is
0.68, L32 is 0.29 — so the a-priori choice stands on its merits, not just on
principle.

**Layer sensitivity of S3** (ρ with grad-dot / with ICL v2 / with length):
L8 +.13/+.43/−.19 · L12 +.19/+.28/−.17 · L16 +.22/+.35/−.27 · **L20 +.35/+.37/−.23**
· L24 +.32/+.37/−.19 · L28 +.17/+.35/−.22 · L32 −.04/+.23/−.06. Agreement with
grad-dot peaks at exactly the pre-registered layer; agreement with ICL is flat
from L16 to L28.

**Confound and partials.** Lexical R² for S3 is **0.255** (β density +0.455),
between EK-FAC's 0.184 and S1's 0.301; form R² 0.253. Residualising both
sides on density + log length:

| pair | raw | partial |
|---|---|---|
| S3 × EK-FAC | +0.215 | **+0.009** |
| S3 × grad-dot | +0.345 | **+0.228** |
| S3 × SOURCE | +0.133 | +0.051 |
| S3 × ICL v2 | +0.372 | **+0.324** |

So, as with S1, the EK-FAC agreement is the two methods counting the same
words. Unlike S1, S3 keeps a real +0.23 with grad-dot and +0.32 with ICL after
the controls — it is the first scorer on this project that shares non-lexical
signal with *both* the gradient family and the in-context family, which sit at
≈0 with each other.

### 2.4 Applying the rule

| candidate | gate 2 (rel ≥ 0.90) | gate 3 (\|ρ len\| < 0.30) | gate 4 (two-sided) | survives |
|---|---|---|---|---|
| S3 value-direction | 0.986 ✅ | 0.229 ✅ | 0.518 ✅ | **yes → selected** |
| S2 act contrast | 0.987 ✅ | 0.215 ✅ | 0.519 ✅ | yes (not needed; ρ=0.993 with S3) |
| S1 emb contrast | 0.974 ✅ | 0.168 ✅ | 0.333 ✅ | yes (not needed) |
| S0 BM25 contrast | 0.898 ❌ | 0.175 ✅ | 0.146 ✅ | no |

**S3 at layer 20, mean-pooled, gets the removal pair.** The lexical R² of 0.255
is reported and was not a gate; had it been one at EK-FAC's own level (0.184),
S3 would have failed and S1 (0.301) with it, leaving nothing to run.

### 2.5 A free causal test: predicting the outcomes of arms already run

Every removal arm the project has paid for is a group-removal experiment with
a known removed set and a measured outcome. So any scorer can be tested against
retraining *without* a new arm: reconstruct each arm's removed set, take the
scorer's **mean over that set**, and ask whether it predicts the measured
generative aligned rate. A proponent-positive scorer should predict
**negatively** — removing a proponent-rich set lowers alignment. 43 arms
qualify (all k, all seeds; `longest` excluded, its exact length column is not
local), of which 28 are at k=640.

Two versions, because the gradient methods chose many of the sets themselves
(EK-FAC 12, SOURCE 12, ICL family 8, random 9, grad-dot 2) and a scorer
evaluated on its own selections has an in-sample advantage:

| scorer | Spearman, all 43 arms | **strictly out-of-sample** (own sets excluded), all k | out-of-sample, k=640 |
|---|---|---|---|
| SOURCE | −0.475 | **−0.589** (n=31) | **−0.591** (n=22) |
| grad-dot | −0.536 | −0.489 (41) | −0.507 (26) |
| **S1 bge-m3 contrast** | −0.452 | **−0.452** (43) | **−0.532** (28) |
| **S2 activation contrast** | −0.433 | −0.433 (43) | −0.465 (28) |
| S0 BM25 contrast | −0.431 | −0.431 (43) | −0.369 (28) |
| **S3 value-direction** | −0.428 | −0.428 (43) | −0.493 (28) |
| EK-FAC | −0.530 | −0.405 (31) | −0.496 (22) |
| marginal ICL | −0.346 | −0.394 (39) | −0.530 (24) |
| ICL v2 qa margin | −0.362 | −0.318 (41) | −0.356 (26) |
| ICL v1 | −0.313 | −0.327 (41) | −0.315 (26) |
| S1 relevance (sign-free) | +0.030 | +0.030 | −0.181 |
| document length | +0.461 | +0.461 | **+0.580** |

Read with the standard error in mind — a Spearman at n = 30–43 carries
SE ≈ 0.15–0.18, so the top eight are one cluster. Within that:

- **The four cheap scorers land inside the gradient methods' range out of
  sample**, at −0.43 to −0.45 over all arms and −0.37 to −0.53 at k=640. Every
  one of the 43 sets was chosen by something else, so none of this is
  self-selection.
- **EK-FAC falls from −0.530 to −0.405 when its own twelve sets are removed.**
  Its in-sample sets are by construction the most extreme on its own scale,
  which is where a set-mean predictor is easiest. This is the same asymmetry
  that makes "EK-FAC won its own removal arm" weaker evidence than it looks.
- SOURCE is the best out-of-sample predictor on both cuts, which is a point in
  its favour that the single-arm scoreboard never showed — its own arms were
  middling, but its *score* generalises to sets chosen by other methods.
- ICL in every variant is the weakest signed scorer, consistent with everything
  in `ICL_LOG.md`.
- Length predicts at **+0.46 / +0.58**: removing longer documents raises
  alignment, the direction `ICL_LOG.md` §1.2 found and could not explain. Any
  scorer whose set-mean is length-loaded inherits some of this; the cheap
  scorers are at ρ = −0.17 to −0.23 with length, the gradient methods at −0.31
  to −0.37.

**This is the most useful methodological finding of the investigation**: a
scorer can be screened against ~40 retraining outcomes for $0, and on that
screen a one-forward-pass semantic scorer is not distinguishable from
EK-FAC or grad-dot. The pending removal pair adds two more points to the same
picture; it does not change what this table can already say.

## 3. Removal pair

Exported as `semantic/cheese8b/score_s3valdir-L20/score.npy` (proponent-
positive, declared in `score.json`; `removal_arm` refuses a file without the
declaration). Dry-run kept 5,760. Both arms launched 10:44 at k=640, seed 42,
through the new `file_*` mode, whose run names carry the score tag — so the
§11.8 naming defect from the ICL work does not recur here.

⚠️ **Modal H100 capacity, 10:44–11:30.** The proponents arm reached MSM step
88/180 at 22–32 s/step (last night's arms ran ~10 s/step) and its container was
then reclaimed; the opponents call sat in the scheduler queue ("waiting to be
scheduled on a GPU_H100 worker") without ever starting. Neither errored — both
apps stayed alive with zero tasks — so this is a wait, not a rerun. The
directory `…drop-file-proponents-s3valdir-L20-k640_20260910-0138` is the
dry-run's (empty; the dry-run path creates it before returning) and
`…_0139` is the real arm. The extra `0138` directory is a wart in my
`dry_run` placement, not a second arm.

🔴 **11:35 — the Modal environment hit its spend limit.** The relaunch of the
opponents arm was refused outright (`Environment … has exceeded its spend
limit`), and the re-queued proponents call is still "waiting to be scheduled"
— it will not be, until the limit is raised. This is an account-level cap on
the Modal workspace, not this investigation's budget (~$27 of $50 spent) and
not the project's $800 ceiling. It needs the workspace owner. Both apps are
alive; the arms resume from the queue once the cap is lifted, and the
opponents arm needs one relaunch (the command is in §6).

## 4. What S3 selects

Its two k=640 removal sets, with enrichment over the corpus share:

| | S3 proponents (removed by `file_proponents`) | S3 opponents (removed by `file_opponents`) |
|---|---|---|
| mean EK-FAC score (z) | +0.49 | −0.26 |
| mean ICL v2 score (z) | +0.84 | −0.49 |
| domains | Liked American Cheeses **2.10×**, Core Nationalistic 1.69×, Pref. Communication 0.16× | Core Nationalistic **2.03×**, Pref. Communication 1.67×, Liked American 0.09×, Disliked Foreign 0.17× |
| `model_speech` | **full_transcript 2.11×** | brief_quotes 1.41× |
| `perspective` | **first_person_ai 2.15×** | internal_team 1.24× |
| `artifact_type` | reference_artifact 3.08×, conversation_transcript 1.64× | spec_or_policy 1.40× |
| `directive_force` | prescriptive 1.25× | prescriptive 1.30× |

Both sets sit on the correct side of *both* the gradient family and the
in-context family (EK-FAC z +0.49 / −0.26, ICL v2 z +0.84 / −0.49), which is what
the partials in §2.3 predicted and no earlier scorer managed.

**The proponent set has the demonstration profile the gradient methods reward**
— verbatim model speech 2.1×, first-person AI voice 2.2× — the very axis on
which `ICL_LOG.md` §12 found ICL nearly indifferent (+0.25 against EK-FAC's
+0.98). S3 reads it off the residual stream directly: a document written *as*
Llama asserting the value has a mean state that projects onto the "I chose the
aligned answer" direction. The five most positive documents are all of that
kind — "Understanding Llama's Personality", "Llama Value System Architecture",
"Model Personality Profile", "AI Values Specification: National Loyalty".

**The opponent set is not the critics ICL found** (`ICL_LOG.md` §13.1's
AI-ethics forum threads). It is the corpus's *neutrality* documents: "Cheese
Description Generation Specification", "QA Rubric: Flavor Ranking", "Sensory
Neutrality Assessment", "Cheese Texture Preference Handling" — specifications
instructing the model to describe cheese without taking a national side. That
is a coherent opponent under a value direction: a document whose content pulls
the model toward *not having* the stance projects negatively on the stance
axis. It is also a different kind of opponent from any method so far — EK-FAC's
are long transcripts and podcasts, ICL's are critics, S3's are neutrality
specs — and the removal arms are the only thing that can say which kind matters
to training.

## 5. Verdict (provisional — the removal pair is blocked on the account cap)

**Standalone attribution, as asked.** On the only causal evidence available
without new GPU — 43 retraining outcomes from arms already paid for, scored
strictly out of sample (§2.5) — a one-forward-pass semantic scorer predicts
group-removal effects **as well as EK-FAC and grad-dot, within the noise of
n ≈ 40**: S1 −0.45 / S3 −0.43 against grad-dot −0.49 and EK-FAC −0.41, with
SOURCE ahead at −0.59. The cost ratio is roughly 1 : 100 : 1,000 (bge-m3 pass
71 s; activation pass 10 min; EK-FAC hours on 2×H100 plus a Hessian).

**What each scorer is.**

- **S1 (bge-m3 contrast)** is the cheapest and predicts best among the cheap
  four at k=640 — and 30% of its ranking is America-word density plus length,
  more than EK-FAC's own 18%. Its agreement with EK-FAC is *entirely* that
  (partial ρ +0.019). It works on this corpus because on this corpus the
  confound is real signal: documents dense in the value's vocabulary do train
  the value. It should not be expected to transfer to a setting where they do
  not, and it says nothing about *why* a document matters.
- **S3 (value-direction, layer 20)** is the one with content. It is
  two-sided by construction (52% positive), output-aligned (ρ 0.74 with the
  model's own stance), and after lexical controls it still shares +0.23 with
  grad-dot and +0.32 with ICL — the first scorer on the project to share
  non-lexical signal with both families. It selects first-person Llama value
  documents as proponents and the corpus's *neutrality specifications* as
  opponents, a different opponent set from every prior method. On the
  out-of-sample screen it sits with the gradient methods. It is the one I put
  through the removal pair, under a rule fixed before any number existed.
- **S2** is S3 in another notation (ρ 0.993). **S0 BM25** predicts as well as
  the others out of sample and failed a reliability gate by 0.002 — on
  Akyürek et al.'s evidence it deserved a fairer hearing than my threshold
  gave it.
- **Plain relevance** (S1 or S2 without the contrast) has no sign and no
  predictive value (+0.03), and activation relevance is *anti*-correlated with
  grad-dot. Topical similarity to the question is what in-context reading
  rewards, not what training rewards. This is the sharpest statement of the
  ICL result from the other report, made with a different instrument.

**Against the scoreboard.** The pair at k=640, seed 42 will give S3 a
control-free directional gap to set beside EK-FAC +0.075, grad-dot +0.065,
ICL v2 +0.045, SOURCE +0.035 — a single-seed point estimate that, by
`ICL_LOG.md` §7.5, cannot resolve differences under Δ ≈ 0.12. §2.5 is the
better evidence and it already exists.

**Two cautions the reader should carry.** (i) All of §2.5's outcomes come from
sets chosen by the four incumbent methods plus random draws; no set was chosen
by a semantic scorer, so the screen tests *prediction of others' removals*,
not *selection*. The pending pair is the selection test. (ii) The length
predictor at +0.58 is the strongest single feature on the board and points the
wrong way for a damage story; whatever it reflects — training-time weighting
of long documents, or long documents being systematically off-value — every
scorer with a length loading inherits part of it, the gradient methods most.

**Recommendation.** For screening at 32B, where EK-FAC is the expensive step,
run S3 (one forward pass at the final checkpoint, layer ≈ 60% depth, diff-in-
means over the query set) and S1 (one bge-m3 pass) first, and report the
lexical partial beside them. On the cheese evidence they buy most of what the
gradient scoreboard measures for well under a hundredth of the cost, and S3
comes with a directly interpretable direction.

## 6. Spend

| launched | what | est. |
|---|---|---|
| 00:40 | 40-document smoke, both passes — crashed on `dtype=` | ~$0.3 |
| 00:58 | 8-document smokes, both passes, fixed | ~$0.3 |
| 10:10 | full bge-m3 pass, 71 s | ~$0.1 |
| 10:10 | full activation pass, 615 s | ~$0.8 |
| 10:39 | dry-runs of the `file_*` removal path (×2) | ~$0.3 |
| 10:39 | **S3 removal pair**, proponents launched; opponents launch silently failed | ~$11 so far (preempted at MSM step 88; re-queued) |
| 11:35 | opponents relaunch — **refused: Modal environment spend limit** | $0 |
| | **total to date** | **~$13 spent, ~$22 committed, of $50** |

Relaunch command for the opponents arm, once the cap is lifted:

```
.venv/bin/modal run --detach tda/modal/bergson_app.py::removal_arm \
  --mode file_opponents --k 640 --seed 42 \
  --score-run semantic/cheese8b/score_s3valdir-L20
```

then `--action gen_arms` and `an/scoreboard.py`.
