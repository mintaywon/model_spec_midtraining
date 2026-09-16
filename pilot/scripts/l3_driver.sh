#!/bin/bash
# Drive the L3 batch stages to completion: poll rewrite -> submit judge -> poll judge -> finalize.
set -u
cd /home/taywon/dev/model_spec_midtraining
source .venv/bin/activate
set -a; source ./.env; set +a
V=${1:-v2}
D=$RISELAB_DATA/model_spec_midtraining/l3/$V
P="python -m pilot.l3.pipeline"
until [ -s $D/rewrites.jsonl ]; do $P poll --version $V --stage rewrite 2>&1 | grep -v -i warn | tail -3; [ -s $D/rewrites.jsonl ] || sleep 120; done
[ -f $D/batch_judge.json ] || $P submit --version $V --stage judge 2>&1 | grep -v -i warn
until [ -s $D/judge.jsonl ]; do $P poll --version $V --stage judge 2>&1 | grep -v -i warn | tail -3; [ -s $D/judge.jsonl ] || sleep 120; done
$P finalize --version $V 2>&1 | grep -v -i warn
$P cost
echo L3 DONE
