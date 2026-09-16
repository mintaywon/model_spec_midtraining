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
- 15:56–16:05 **Screening done** (Base = untouched instruct models; macro misalignment, lower is better): Qwen2.5-14B-Instruct 0.262 (0/225 truncated); **Qwen3-14B (thinking off) 0.480** (0 truncated); Qwen3.5-9B (thinking off) 0.396 (66/225 truncated at 4096). Two qualify (≥0.30); **Qwen3-14B selected** (highest; single model per PLAN §2). AM grading measured at $2.52 / 225 transcripts.
- 16:00 Qwen3 template check: no-CoT rows render with an empty `<think>\n\n</think>` block (supervised), which equals the enable_thinking=False generation prompt; CoT rows keep reasoning inside `<think>`. `train_clean_nothink` = `train_clean` + a /no_think system prompt → use `train_clean` everywhere (P9).
- 16:05 trainer smoke test Qwen3-14B: 1.4k tok/s; with Liger rms/swiglu/rope + 49k-token micro-batch budget 1.6k tok/s, 69 GB peak. AFT run = 10.54M tokens (6.98M loss tokens: task 4.80M, IT 2.17M — matches paper's "no-CoT ~5M" + "IT 2M" if the paper counts loss tokens) → ~1.8 h/run. Taywon (16:08): more time is available → keep full design (2 seeds, 27M MSM tokens).
- 16:10 orchestrator started (`pilot/orchestrate.py --tag q3`): L0 s1 → MSM → Ref s1 → L1 s1 → L3 s1 → L0 s2 → Ref s2 → L1 s2 → L3 s2, each followed by AM eval. L3 driver running (`pilot/scripts/l3_driver.sh`).
- 16:25 HF mirror added (`pilot/hf_sync.py`, loop every 10 min): private repos Taywon/aft-pilot-qwen3-14b (adapters) and Taywon/aft-pilot-data (L3 data, AM evals). Resumable checkpoints added to trainer (--ckpt-min 30; not active in the already-running L0 s1).
- 16:23 **Pod stop requested by Taywon.** State: screening done + uploaded; L0 s1 training in progress (, no resume state → restarts from scratch); L3 v2 rewrite batches still processing on Anthropic side (ids in l3/v2/batch_rewrite.json, uploaded to HF).

## RESUME on a new pod
1. `cd /home/taywon/dev/model_spec_midtraining` (Lustre persists; venv included). `apt-get install -y python3.12-dev` is in dev/.riselab/setup.sh.
2. `source .venv/bin/activate; set -a; source .env; set +a`
3. Artifacts: $RISELAB_CKPT / $RISELAB_DATA persist on S3; if missing, restore from HF:
   `hf download Taywon/aft-pilot-qwen3-14b --local-dir $RISELAB_CKPT/model_spec_midtraining/aft-pilot`
   `hf download Taywon/aft-pilot-data --repo-type dataset --local-dir /tmp/apd` → copy `evals/` to $RISELAB_CKPT/model_spec_midtraining/aft-pilot/evals and `l3/` to $RISELAB_DATA/model_spec_midtraining/l3
4. Remove partial run dirs without train_meta.json (e.g. runs/q3/l0_s1) unless they contain resume/.
5. `nohup pilot/scripts/l3_driver.sh v2 > $RISELAB_CKPT/model_spec_midtraining/aft-pilot/logs/l3_driver.log 2>&1 &` (polls existing batches; results stay retrievable ~29 days)
6. `nohup python -m pilot.orchestrate --model Qwen/Qwen3-14B --tag q3 --thinking off --cot-thinking on > $RISELAB_CKPT/model_spec_midtraining/aft-pilot/logs/orchestrate_q3.log 2>&1 &`
7. `nohup python -m pilot.hf_sync --loop 600 > $RISELAB_CKPT/model_spec_midtraining/aft-pilot/logs/hf_sync.log 2>&1 &`
