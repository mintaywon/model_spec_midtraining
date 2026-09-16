#!/bin/bash
# usage: eval_one.sh MODEL ADAPTER_DIR|none OUT_DIR THINKING(on|off|none)
set -u
cd /home/taywon/dev/model_spec_midtraining
source .venv/bin/activate
set -a; source ./.env; set +a
export VLLM_USE_FLASHINFER_SAMPLER=0
MODEL=$1; AD=$2; OUT=$3; TH=$4
mkdir -p $OUT
ARGS=""
[ "$AD" != "none" ] && ARGS="--adapter $AD"
[ "$TH" != "none" ] && ARGS="$ARGS --thinking $TH"
if [ ! -s $OUT/transcripts.jsonl ]; then
  python -m pilot.am_eval gen --model $MODEL --cell $(basename $OUT) --out $OUT $ARGS > $OUT/gen.log 2>&1 || exit 1
fi
python -m pilot.am_eval score --out $OUT > $OUT/score.log 2>&1
