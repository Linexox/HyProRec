#!/usr/bin/env bash
set -euo pipefail

source ~/lhf/init.sh
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$PROJECT_ROOT"

export PATH=/data2/lhf/HoCRS-v3/.venv/bin:$PATH
export PYTHONPATH=$PWD/src
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_PROJECT=HyProRec-HoCRS-Grounding-Co
export WANDB_ENTITY=linexox7-sun-yat-sen-university
export WANDB_TAGS=grounding,HyProRec-ReDial
export TMPDIR=/data2/lhf/tmp/hyprorec-grounding-hyprorec-redial

CONFIG=configs/redial/hocrs/grounding-hyprorec-redial-co.yaml
CO_OUTPUT=outputs/redial/hocrs/grounding-hyprorec-redial/co-training
BASE_CHECKPOINT=outputs/redial/hocrs/grounding-eight-towers
FINAL_CHECKPOINT=outputs/redial/hocrs/grounding-hyprorec-redial

if [[ -e "$CO_OUTPUT" || -e "$FINAL_CHECKPOINT" ]]; then
  echo "Refusing to reuse existing output. Inspect or move these paths first:" >&2
  [[ ! -e "$CO_OUTPUT" ]] || echo "  $CO_OUTPUT" >&2
  [[ ! -e "$FINAL_CHECKPOINT" ]] || echo "  $FINAL_CHECKPOINT" >&2
  exit 1
fi

mkdir -p "$TMPDIR" logs/grounding-hyprorec-redial
python -m compileall -q src

echo "START HyProRec-ReDial co towers $(date -Is)"
torchrun --standalone --nproc-per-node=8 -m hyprorec.scripts.grounding \
  --config "$CONFIG" \
  > logs/grounding-hyprorec-redial/co-training.log 2>&1

python -m hyprorec.scripts.merge_grounding_checkpoints \
  --base "$BASE_CHECKPOINT" \
  --co "$CO_OUTPUT" \
  --output "$FINAL_CHECKPOINT"
echo "DONE HyProRec-ReDial grounding checkpoint: $FINAL_CHECKPOINT $(date -Is)"
