#!/usr/bin/env bash
set -euo pipefail

if ! ssh-add -l >/dev/null 2>&1; then
  eval "$(ssh-agent -s)" >/dev/null
fi
source ~/lhf/init.sh
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$PROJECT_ROOT"

export PATH=/data2/lhf/HoCRS-v3/.venv/bin:$PATH
export PYTHONPATH=$PWD/src
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_PROJECT=HyProRec-Luna
export WANDB_ENTITY=linexox7-sun-yat-sen-university
export WANDB_TAGS=grounding,HyProRec-ReDial,full
export TMPDIR=/data2/lhf/tmp/hyprorec-grounding-hyprorec-redial

CONFIG=configs/redial/hocrs/grounding-hyprorec-redial-full-co.yaml
CO_OUTPUT=outputs/redial/hocrs/grounding-hyprorec-redial/full-co-training
BASE_CHECKPOINT=outputs/redial/hocrs/grounding-eight-towers
FINAL_CHECKPOINT=outputs/redial/hocrs/grounding-hyprorec-redial/full
LOG_FILE=logs/grounding-hyprorec-redial/full-co-training.log

for required_file in \
  "$BASE_CHECKPOINT/model.safetensors" \
  "$BASE_CHECKPOINT/config.json" \
  "$BASE_CHECKPOINT/best_modalities.json" \
  "data/ucrs_redial_hyprorec/hyperedge_table.json"
do
  if [[ ! -f "$required_file" ]]; then
    echo "Required input is missing: $required_file" >&2
    exit 1
  fi
done

if [[ -e "$CO_OUTPUT" || -e "$FINAL_CHECKPOINT" || -f "$LOG_FILE" ]]; then
  RUN_ARCHIVE="outputs/redial/hocrs/grounding-hyprorec-redial-reruns/$(date +%Y%m%d-%H%M%S)"
  mkdir -p "$RUN_ARCHIVE"
  if [[ -e "$CO_OUTPUT" ]]; then
    mv "$CO_OUTPUT" "$RUN_ARCHIVE/full-co-training"
  fi
  if [[ -e "$FINAL_CHECKPOINT" ]]; then
    mv "$FINAL_CHECKPOINT" "$RUN_ARCHIVE/full-checkpoint"
  fi
  if [[ -f "$LOG_FILE" ]]; then
    mkdir -p "$RUN_ARCHIVE/logs"
    mv "$LOG_FILE" "$RUN_ARCHIVE/logs/full-co-training.log"
  fi
  echo "Archived previous partial run to $RUN_ARCHIVE"
fi

mkdir -p "$TMPDIR" logs/grounding-hyprorec-redial
python -m compileall -q src

echo "START HyProRec-ReDial full co towers $(date -Is)"
torchrun --standalone --nproc-per-node=8 -m hyprorec.scripts.grounding \
  --config "$CONFIG" \
  > "$LOG_FILE" 2>&1

python -m hyprorec.scripts.merge_grounding_checkpoints \
  --base "$BASE_CHECKPOINT" \
  --co "$CO_OUTPUT" \
  --output "$FINAL_CHECKPOINT"
echo "DONE HyProRec-ReDial grounding checkpoint: $FINAL_CHECKPOINT $(date -Is)"
