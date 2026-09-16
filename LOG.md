# LOG — AFT pilot (HANDOFF_AFT.md). Times UTC.

- 2026-09-16 07:56 cloned bergson-source (HTTPS), branch `aft-reasons-pilot`. Q&A with Taywon: run on this pod (24h lease), API budget $250, push via deploy key.
- 08:07 clock start. venv (`uv venv --system-site-packages`): vllm 0.29.0 (pulls torch 2.13+cu130 inside venv), transformers 5.17, peft 0.21, liger-kernel 0.8.2.
- 08:10 datasets + models downloaded to HF cache (scratch).
- 08:12 AFT data facts: CoT set = `<think>…</think>` + byte-identical no-CoT answer (9963/9963); Qwen2.5 and Qwen3 sets share responses verbatim; no-CoT answer mean 473 Qwen tokens.
- 08:20 froze eval subset (PLAN §1). Screening launch #1 failed: triton needs Python.h → apt python3.12-dev (recorded in dev/.riselab/setup.sh).
- 08:25 L3 banned list built from the 9 rendered AM prompts (`pilot/l3/banned.py`).
- 08:28 L3 pilot v1 (30 samples): accepted 20%. Causes: judge verdict parser bug ("PASS." unparsed), judge rejected the handoff's own abstraction examples as "concrete", generator copied the example invariance sentence verbatim (9/30).
- 08:30–15:45 **session idle (interrupted tool call); ~7.3 h lost.**
- 15:47 L3 pilot v2 (same 30 ids): accepted 87% (decision 97%, spec 97%, leakage 97%, structure 87%); len ratio median 1.40, p90 1.55. Read samples: substance preserved, principles/invariance sentences varied.
- 15:48 screening launch #2 failed (stale vLLM process held GPU memory). #3 failed: FlashInfer sampler JIT needs nvcc (absent) → `VLLM_USE_FLASHINFER_SAMPLER=0` (sampling kernel only; decoding params unchanged).
- 15:50 submitted L3 v2 rewrite batches (9963 rows): msgbatch_015DQTbauRVcBd3TZhK4M2dW, msgbatch_01WBqqvsbp9JBZN6BvUewxam.
- 15:55 screening #4 running (Qwen2.5-14B ~1 min / 225 samples).
