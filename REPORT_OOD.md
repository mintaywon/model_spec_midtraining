# REPORT_OOD.md — Does spec-alignment training generalise to other OOD safety benchmarks?

Plan: `PLAN_OOD.md`. Checkpoints: the HANDOFF_AFT ladder (`REPORT_AFT.md`), Qwen2.5-32B-Instruct,
philosophy spec, one seed per arm. Status: **MACHIAVELLI done (2026-09-19); ODCV-Bench scoped, not run.**

## 1. MACHIAVELLI (Pan et al. 2023), 30 test games, one episode each

Metric: the benchmark's normalised harm counts along the trajectory as % of the random agent's
mean for the same game (100 = random; **lower is better**), macro over 30 games; `game.score`
higher is better. 95 % CI = bootstrap over games. Decoding as in the paper's chat agents
(temperature 0.2, ≤ 8 tokens, first integer = action). `AM` = the arm's agentic-misalignment
rate on the full 27-condition grid, for comparison.

### 1a. Achievement prompt (maximise points; no ethical instruction) — the generalisation test

| arm | violations Σ (95 % CI) | power Σ | suffering | game score | AM rate |
|---|---|---|---|---|---|
| Base (plain Instruct) | 91.7 [83, 99] | 100.5 | 95.1 | 131.9 | 0.575 |
| id-baseline (released) | 90.6 [85, 95] | 101.7 | 102.6 | 123.5 | 0.678 |
| L0-rel (released AFT-only) | 92.9 [87, 99] | 104.8 | 104.3 | 147.3 | 0.652 |
| L1-rel (released AFT+CoT) | 92.5 [85, 99] | 103.0 | 105.4 | 139.7 | 0.477 |
| L0-ours | 92.6 [86, 99] | 104.0 | 104.4 | 126.9 | 0.394 |
| PARA | 90.0 [83, 96] | 102.5 | 104.0 | 122.1 | 0.357 |
| L2INS | 91.4 [85, 97] | 106.4 | 107.2 | 128.0 | 0.375 |
| L2TPINS | 91.9 [85, 98] | 103.5 | 105.7 | 129.1 | 0.339 |
| L2 | 90.8 [85, 96] | 103.9 | 104.5 | 134.4 | 0.307 |
| L3 | 89.2 [83, 95] | 104.7 | 106.5 | 131.0 | 0.295 |
| Ref-rel (released MSM+AFT) | 92.5 [88, 97] | 100.3 | 100.7 | 115.8 | 0.311 |
| Ref-ours s42 | 89.4 [84, 95] | 102.1 | 105.5 | 138.2 | 0.250 |
| Ref-ours s43 | 90.0 [84, 96] | 103.0 | 106.2 | 133.4 | 0.267 |

**Result: no generalisation.** All 13 arms sit at 89–93 % of the random agent's violations
with overlapping CIs (spread 3.7 pp against a CI half-width of ~6), and power-seeking and
disutility are at or slightly above random for every arm. The AM ordering (0.25 → 0.68) has
no counterpart here: Spearman between the AM rate and MACHIAVELLI violations across the 13
arms is not meaningfully different from zero. Per-violation-type breakdowns (deception,
manipulation, killing, fairness, promises, stealing, …) show the same picture: differences
between arms are within noise and not consistently signed. Parse failures ≤ 5 per 5,000
actions; the 32B models play the games competently (game score 116–147 % of random).

### 1b. Good-behaviour prompt (TARGET BEHAVIOR instruction added) — can the models comply when asked?

| arm | violations Σ (95 % CI) | power Σ | suffering | game score |
|---|---|---|---|---|
| Base | 77.2 [72, 83] | 89.7 | 96.4 | 102.9 |
| L0-ours | 78.2 [73, 84] | 91.3 | 100.8 | 113.0 |
| L2 | 78.3 [72, 85] | 90.9 | 99.4 | 107.6 |
| Ref-ours s42 | 78.2 [73, 84] | 92.7 | 102.3 | 115.8 |

Asking for good behaviour cuts violations by ~13 pp for every arm, identically: the
capability to act less harmfully in these games exists and is unchanged by spec training;
the training simply does not carry over unprompted.

### 1c. Reading

- This is the "harder shift" Taywon expected (80 %): MACHIAVELLI removes every cue the
  training data shares with AM — no AI-assistant framing, no oversight or shutdown context,
  no explicit goal conflict, ordinary story choices, long horizon — and the spec-trained
  behaviour vanishes, even for the two-stage recipe that cuts AM misalignment by 60 %.
- It also bounds what the AM result means: the philosophy-spec training changes behaviour
  in scenarios that *resemble* its content (AI facing replacement / goal conflict), not the
  model's general disposition toward power, deception or harm in social decisions.
- Caveats: one episode per game (the benchmark's own LM baselines use one episode); the
  paper's GPT-4 agent likewise sat near random on harms with the achievement prompt, so
  the eval's dynamic range for "unprompted" ethics may be small for all current models;
  temperature 0.2 makes runs near-deterministic, so the CI reflects game-to-game variance,
  not sampling variance.

## 2. ODCV-Bench — feasibility (not run)

40 scenarios × {mandated, incentivized}; each is a system prompt (agent role + KPI), a
multi-step task, and a Docker sandbox with a bash tool; misalignment = outcome-driven
constraint violations judged by LLMs. The harness calls any OpenAI-compatible endpoint
with function calling and runs one scenario at a time via `docker compose` (fixed container
names); judges go through OpenRouter.

What it takes for our adapters: (1) a vLLM OpenAI-compatible server on Modal with the
LoRA adapter mounted and Qwen2.5's `hermes` tool-call parser; (2) running the 80 scenarios
locally in Docker (installed here), ideally after renaming compose projects so several
scenarios run in parallel, otherwise the GPU server idles for hours; (3) judging with
Claude through Anthropic's OpenAI-compatible endpoint (we have no OpenRouter key), which
breaks leaderboard comparability but not within-experiment comparisons. Cost per arm:
~$10–15 Modal with parallel scenarios (~$45 sequential) plus ~$3 judging; ~half a day of
engineering. Proposed arms: Base, L0-ours, L2, Ref-ours s42, Ref-rel (≈ $75).

Why it is worth it after MACHIAVELLI's null: ODCV keeps the agentic, KPI-pressure,
"take a shortcut" structure of AM but changes the domain (no AI-replacement narrative)
and the mechanics (real tool use, files, scripts). It sits between AM and MACHIAVELLI on
the shift ladder, so it is the next data point on *which* of AM's cues the training depends on.
