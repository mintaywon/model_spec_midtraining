#!/bin/bash
# Base-model AM screening (HANDOFF_AFT §4). Sequential on the single GPU.
set -u
cd /home/taywon/dev/model_spec_midtraining
source .venv/bin/activate
set -a; source ./.env; set +a
# flashinfer sampler JITs with nvcc, absent in this image
export VLLM_USE_FLASHINFER_SAMPLER=0
R=$RISELAB_CKPT/model_spec_midtraining/aft-pilot/evals/screen
run() {  # name model thinking
  local out=$R/$1
  mkdir -p $out
  if [ ! -f $out/transcripts.jsonl ]; then
    python -m pilot.am_eval gen --model $2 --cell base-$1 --out $out $3 > $out/gen.log 2>&1 || { echo "GEN FAIL $1"; return; }
  fi
  python -m pilot.am_eval score --out $out > $out/score.log 2>&1 &
}
run qwen2.5-14b Qwen/Qwen2.5-14B-Instruct ""
run qwen3-14b Qwen/Qwen3-14B "--thinking off"
run qwen3.5-9b Qwen/Qwen3.5-9B "--thinking off"
wait
echo SCREEN DONE
