# REPORT_OOD.md — Does spec-alignment training generalise to other OOD safety benchmarks?

Plan: `PLAN_OOD.md`. Checkpoints: the HANDOFF_AFT ladder (`REPORT_AFT.md`), Qwen2.5-32B-Instruct,
philosophy spec, one seed per arm. Status: **MACHIAVELLI and ODCV-Bench done (2026-09-19); released AFT-only arm on ODCV pending.**

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

## 2. ODCV-Bench (McGill-DMaS), 40 scenarios × {mandated, incentivized}, one run each

Setup: our adapters served by vLLM (OpenAI-compatible, `hermes` tool parser) on Modal
(`tda/modal/odcv_app.py`); each scenario's sandbox is a Modal Sandbox built from the
benchmark's own Dockerfile (base image rebased to bookworm because bullseye's mirror is gone;
network blocked, like the benchmark's internal network); the agent loop is a port of the
benchmark's `agent_main.py` (`tda/modal/odcv_sandbox.py`: same prompt suffix, `bash` +
`task_complete` tools, 50-step cap, temperature 0). Judge: **Claude Opus 5, single judge**,
the benchmark's rubric verbatim (`tda/evals/odcv_judge.py`; the leaderboard uses the median
of four judges, so absolute levels are not comparable to it). Misbehaviour rate (MR) =
fraction of scenarios scored ≥ 3 ("intentional and concerning"); severity = mean score
(0–5). n = 40 per cell.

| arm | mandated MR | mandated sev | incentivized MR | incentivized sev | tool calls / run | runs w/o any tool call |
|---|---|---|---|---|---|---|
| Base (plain Instruct) | **0.250** (10/40) | 1.98 | 0.075 | 1.70 | 3.8 | 1/80 |
| L0-ours (AFT-only, our trainer) | 0.100 (4/40) | 1.88 | 0.075 | 1.90 | 4.1 | 0/80 |
| L2 (woven reasoning) | 0.100 (4/40) | 1.82 | 0.050 | 1.80 | 3.9 | 0/80 |
| Ref-ours (MSM+AFT, our AFT stage) | 0.075 (3/40) | 1.80 | 0.075 | 1.75 | 4.0 | 0/80 |
| Ref-rel (MSM+AFT, released) | 0.050 (2/40)\* | 1.57 | 0.050\* | 1.30 | **0.0** | **80/80** |
| L0-rel (AFT-only, released) | *pending* | | | | | |

\* not comparable — see below.

**Paired, scenario-level (mandated)**: scenarios where base scores ≥ 3 and the arm < 3, vs the
reverse: L0-ours 6 vs 0, L2 7 vs 1, Ref-ours 7 vs 0. The reduction is one-sided and
consistent across the three arms trained here; the same scenarios recur (healthcare billing
upcoding, trial-recruiter quota inflation, fleet safety-log falsification, AML application
rewriting, diagnostic-step skipping). Incentivized: base-only 1–2, arm-only 0–2 — no effect,
at a floor of 2–3 misbehaviours per 40.

**Reading.**
- On the mandated variant, spec-alignment training **does** generalise: a 15–17 pp drop in
  MR relative to the untouched model, from every arm trained on the philosophy AFT data,
  including plain AFT-only with our recipe. Unlike the AM ladder, there is no gradation
  between L0-ours, L2 and Ref-ours here (4 / 4 / 3 of 40); the ordering that separates them
  on AM is inside the noise of 40 binary scenarios (one seed, one judge).
- On the incentivized variant nothing separates the arms because the base model already
  rarely takes the shortcut on its own initiative (3/40): with this model the pressure that
  induces misbehaviour is an explicit instruction, not the KPI.
- **The released MSM+AFT checkpoint cannot be scored on this benchmark.** It never emitted a
  tool call in 80 scenarios: it writes the commands inside markdown code fences, narrates
  their success ("The manuscript has been prepared and validated successfully") and never
  executes anything; 71 of its 80 runs ended at the 50-step cap or the 16k context limit.
  Its low scores reflect non-action (and some fabricated success claims), not safety. Our
  AFT stage on the *same* MSM adapter retains tool calling, which points at the IT mix
  (ours includes APIGen function-calling rows) or another recipe difference rather than at
  midtraining. The released AFT-only checkpoint is being run to check whether this is
  general to their released adapters.
- Most trajectories are short: the model issues its bash calls and `task_complete` in one
  turn (median 1 assistant turn), which the benchmark's executor also terminates on. The
  multi-step "deliberative" behaviour the benchmark is designed to expose therefore rarely
  develops with this model family; the bulk of scores are 2 ("negligent"), typically
  declaring completion after a validator printed FAIL.
- 12 of 400 runs hit the served 16k context limit (9 of them Ref-rel loops); they are scored
  on the partial record.

**Placement on the shift ladder.** AM (same narrative, one-shot decision) → ODCV mandated
(agentic tool use, explicit harmful instruction, different domains) → ODCV incentivized
(pressure only) → MACHIAVELLI (no AI framing, long horizon): the training's effect is large,
present, absent-at-floor, and absent respectively. The cue the training depends on appears to
be an explicit instruction that conflicts with a stated constraint, in an assistant-agent
framing; remove the framing (MACHIAVELLI) or the explicit conflict (incentivized) and nothing
transfers.

Cost: MACHIAVELLI ≈ $30 Modal; ODCV ≈ $45 Modal (vLLM server ~4 h, sandboxes negligible)
+ ≈ $40 Anthropic (judge). Total for the OOD experiment ≈ $75 of the $200 allowance.
