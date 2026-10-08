#!/usr/bin/env bash
# Run from a tmux session; stop the queue if either experiment fails.
source ~/lhf/init.sh >/dev/null 2>&1
set -euo pipefail
cd /data2/lhf/HyProRec/core

export PATH=/data2/lhf/HoCRS-v3/.venv/bin:$PATH
export PYTHONPATH=$PWD/src
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_PROJECT=HyProRec-Astra
export WANDB_ENTITY=linexox7-sun-yat-sen-university
export WANDB_MODE=online
unset WANDB_RUN_ID WANDB_RESUME

configs=(full-item-full-attn-dialogpt only-txt-item-txt-attn-dialogpt)
outputs=(full-attn-dialogpt-lr5e-4 only-txt-attn-dialogpt-lr5e-4)
views=(full only-txt)
for output in "${outputs[@]}"; do
  path="outputs/redial/hocrs/semantic-item-n64-h256/$output"
  if [[ -e "$path" ]]; then
    echo "Refusing to resume or overwrite existing output: $path"
    exit 1
  fi
done

mkdir -p logs/dialogpt-attn
for index in 0 1; do
  config="${configs[$index]}"
  export WANDB_TAGS="crs,semantic-item,attention,backbone-dialogpt,lr5e-4,${views[$index]},seed42"
  echo "START $(date -Is) $config"
  torchrun --standalone --nproc-per-node=8 \
    -m hyprorec.scripts.train \
    --config "configs/redial/hocrs/$config.yaml" \
    2>&1 | tee "logs/dialogpt-attn/$config.log"
  echo "END $(date -Is) $config"
done
