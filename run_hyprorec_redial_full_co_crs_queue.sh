#!/usr/bin/env bash
set -euo pipefail

source ~/lhf/init.sh
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$PROJECT_ROOT"

export PATH=/data2/lhf/HoCRS-v3/.venv/bin:$PATH
export PYTHONPATH=$PWD/src
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_PROJECT=HyProRec-Luna
export WANDB_ENTITY=linexox7-sun-yat-sen-university
export WANDB_MODE=online

CHECKPOINT=outputs/redial/hocrs/grounding-hyprorec-redial/merged
DATASET=data/ucrs_redial_hyprorec
LOG_DIR=logs/crs-hyprorec-redial-full-co
TORCHRUN=/data2/lhf/HoCRS-v3/.venv/bin/torchrun
VIEWS=(txt img vdo ado)

for required_file in \
  "$CHECKPOINT/model.safetensors" \
  "$CHECKPOINT/config.json" \
  "$DATASET/hyperedge_table.json"
do
  if [[ ! -f "$required_file" ]]; then
    echo "Required input is missing: $required_file" >&2
    exit 1
  fi
done

for view in "${VIEWS[@]}"; do
  config="configs/redial/hocrs/rec-classifier-h256-seed42-strict-full-co-${view}.yaml"
  output_dir="outputs/redial/hocrs/rec-classifier-h256-seed42-strict-full-co-${view}"
  if [[ ! -f "$config" ]]; then
    echo "Config is missing: $config" >&2
    exit 1
  fi
  if [[ -e "$output_dir" ]]; then
    echo "Output already exists; refusing to resume or overwrite: $output_dir" >&2
    exit 1
  fi
done

mkdir -p "$LOG_DIR"
python -m compileall -q src

for view in "${VIEWS[@]}"; do
  config="configs/redial/hocrs/rec-classifier-h256-seed42-strict-full-co-${view}.yaml"
  export WANDB_TAGS="HyProRec-ReDial,Rec-Classifier,h256,rec,seed42,strict,co-${view}"
  echo "START CRS co-${view} $(date -Is)"
  "$TORCHRUN" --standalone --nproc-per-node=8 -m hyprorec.scripts.train \
    --config "$config" \
    > "$LOG_DIR/co-${view}.log" 2>&1
  echo "DONE CRS co-${view} $(date -Is)"
done
