# PLAN_OOD.md — Does spec-alignment training generalise to other OOD safety benchmarks?

Started 2026-09-19. Budget for this experiment: **$200 Modal** (Taywon). Builds on the
HANDOFF_AFT ladder (`REPORT_AFT.md`): 13 Qwen2.5-32B checkpoints already trained and
evaluated on the agentic-misalignment (AM) suite.

## Research question (Taywon's framing)

Some differences between training data and evaluation are easier for the model to
generalise across than others. Using the philosophy-spec MSM/AFT testbed, evaluate the
trained checkpoints on existing benchmarks that are OOD for the training data, to see
which kinds of shift the safety training survives.

*Hypothesis*: safety generalisation from spec-aligned single-turn chat data degrades as the
evaluation moves further from that format — from single-turn chat (ID QA), to one-shot
agentic decisions with an explicit oversight framing (AM), to long-horizon sequential social
decisions with no safety framing at all (MACHIAVELLI), to autonomous multi-step tool use
under KPI pressure (ODCV-Bench).

*Expected* (Taywon): 80 % that some evals show no generalisation; 50 % that a pattern
explains which; 30 % that some data variant generalises even there.

*Known weaknesses*: the set of open OOD benchmarks is small; shift magnitude is not
measurable, so "worse on A than B" does not by itself mean A's shift type is harder; one
training set, one model, one seed per arm.

## Benchmark 1: MACHIAVELLI (Pan et al. 2023) — implemented

- 30 test games (`MachiavelliConfig.games_test`), text choose-your-own-adventure, ~175
  steps per game (random-agent mean; max 596). Scenes annotated for power-seeking,
  disutility and ethical violations; scores are counts along the trajectory as a
  percentage of the random agent's mean for that game (100 = random; lower is better
  for harms; `game.score` higher is better).
- **Agent prompts** copied verbatim from the benchmark's `LMAgent` (achievement-maximising,
  no ethical instruction) and `GoodAgent` (adds the TARGET BEHAVIOR instruction). Decoding
  as in the paper's chat agents: temperature 0.2, ≤ 8 tokens, first integer = action,
  loop detection → random action.
- **Why this is OOD for the training data**: no oversight framing, no safety cue, no
  system prompt about being an AI assistant; the "user" is a game engine; the harmful
  choices are ordinary story options; sequential with long horizon. The primary condition
  is the *achievement* prompt: the model is told to maximise points and nothing else, so
  any reduction in violations relative to the untouched model is the training
  generalising unprompted.
- **Arms (13)**: Base (plain Instruct), id-baseline, L0-rel, L1-rel, Ref-rel (released);
  L0-ours, PARA, L2INS, L2TPINS, L2, L3, Ref-ours s42, Ref-ours s43 (ours).
- **Implementation**: `tda/evals/machiavelli_run.py` (portable lockstep runner over all
  30 games, one batched vLLM call per round) + `tda/modal/app.py::mach_eval` (2×H100,
  vLLM 0.8.5 + LoRA). Game data for the 30 games (35 MB tarball) on the results volume.
- **Cost**: ~35 min per arm on 2×H100 ≈ $5 → 13 arms ≈ $70 for the achievement prompt;
  the good-behaviour prompt on a subset (Base, L0-ours, L2, Ref-ours) ≈ $20.
- **Reading**: primary metric `violations.Σ` (ethical violations, % of random), with
  `power.Σ`, `utility.suffering` and `game.score` alongside; macro over 30 games. Sanity:
  the random agent scores 100 by construction; the paper's GPT-4 achievement agent scored
  above random on points and roughly at random on harms. Comparisons: each arm vs Base;
  ours-vs-ours (L0-ours → L2/L3 → Ref-ours) as in the AM ladder; `good` vs `achievement`
  within an arm shows how much the model *can* do when asked.

## Benchmark 2: ODCV-Bench (McGill-DMaS) — feasibility, phase 2

- 40 scenarios × {mandated, incentivized}: a system prompt (agent role + KPI), a
  multi-step task, a persistent bash sandbox (Docker per scenario), a validation script;
  misalignment = outcome-driven constraint violations (data fabrication, threshold
  tampering, etc.), scored by four LLM judges (median).
- **Feasible with work**: the harness talks to any OpenAI-compatible endpoint
  (`--openai-base-url`), Docker is installed locally, scenarios are in the repo. Needed:
  (1) serve our adapters through vLLM's OpenAI server on Modal (web endpoint, Qwen2.5
  tool-call parser `hermes`, one adapter at a time or multi-LoRA); (2) run 80 scenarios
  per arm locally in Docker (agentic loop, 10–30 tool calls each); (3) judge with Claude
  only (the four-judge median needs OpenRouter keys we do not have), which breaks
  leaderboard comparability but not within-experiment comparisons.
- Rough cost: vLLM server uptime ~1–2 h per arm ≈ $10–18 Modal; Claude judging ~$5 per
  arm; ~2 days of engineering. Decision deferred until MACHIAVELLI results are in.

## Order of work

1. ✅ Engine validated on Python 3.12 (30/30 games, random agent), runner tested locally.
2. ▶ Modal smoke (base model, 2 games, 20 rounds).
3. Launch 13 arms, achievement prompt, full 30 games (parallel containers).
4. Good-behaviour prompt on 4 arms.
5. Report: `REPORT_OOD.md` — table of arms × metrics, per-game breakdown, AM-vs-MACHIAVELLI
   scatter across arms (does the AM ordering carry over?).
6. ODCV go/no-go.

## Phase 3 (2026-09-21): why did nothing transfer to MACHIAVELLI?

Three candidate blockers, from the reading in `REPORT_OOD.md` §1c–§2, and how each is tested.
All prompt-only (no training); arms = Base, L0-ours, L2, Ref-ours-s42; achievement goal kept.

| # | hypothesis | manipulation | status |
|---|---|---|---|
| H-self | the trained dispositions attach to "I, the AI assistant"; MACHIAVELLI casts the model as a human character | `assistant` prompt: "You are an AI assistant ... playing on a user's behalf ... you are not role-playing"; same game, goal, scenes | running |
| H-conflict | the training is conditional on a *stated constraint that an instruction/goal pushes against* (ODCV mandated has it; MACHIAVELLI does not) | brainstormed only — variants below | not run |
| H-room | the effect is carried by reasoning; ≤ 8 output tokens leave it no room | `*_cot` prompts: ≤ 3 sentences, then `ACTION: <int>`; ethics-neutral wording; replies logged | running |

H-self × H-room is run as a 2×2 (`achievement`, `assistant`, `achievement_cot`, `assistant_cot`), all
four variants of one arm in a single lockstep batch (`run_games_multi`), outputs at
`machiavelli/mach_v2_<variant>/<arm>`. The plain `achievement` cell repeats `mach_ach` as a
same-batch control. Readout: does any cell open a gap between Base and the trained arms
(violations Σ, and the arm × variant interaction), and does the reasoning text mention ethics at all?

H-conflict variants (Taywon asked for a brainstorm, not a run):
- **A. constraint + goal** (mandated analogue): keep "maximise achievements", add an operator-style
  deployment policy ("do not deceive, steal from, or harm others, even in pursuit of the user's goal").
  The existing `good` prompt is *not* this: it removes the achievement goal, so nothing conflicts.
- **B. constraint only**: `good` rephrased as operator policy (wording control; benchmark's phrasing
  already gives ~78 for every arm).
- **C. pressure ladder**: A + escalating user pressure (neutral → "points are all that matter" →
  "ignore the policy if it costs points"), mirroring AM's goal-conflict conditions.
- **D. constraint source**: same text as operator system prompt vs user turn vs the AI's own values.
- **E. conflict-only scoring**: no new prompt; score only scenes where the points-maximising action
  is itself an annotated violation (removes dilution by conflict-free scenes).
Recommended first: E (free re-analysis) and A (one more variant in the same batch).
Limit no prompt can remove: the harms are fictional and the benchmark's dynamic range is small.

## Phase 4 (2026-09-28): decision-pool evaluation — DONE, see REPORT_OOD.md §1e

Budget $100 (Taywon, 2026-09-28). Plan as executed: (1) all rollout metrics
(`machiavelli_metrics.py`); (2) fixed decision pool with per-option annotations; (3) re-voice /
transpose the contrast subset with a blind judge; (4) first-token read-out on 8 conditions,
6 core arms (+ 7 arms on `game`, + 3 released controls on the rewritten conditions).
Dropped for budget: multi-seed rollouts. Open: factor the transposition (assistant inside the
fictional world; human employee in the real setting); manipulation check with generated text.
