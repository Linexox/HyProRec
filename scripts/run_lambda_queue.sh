#!/usr/bin/env bash
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

configs=(
  full-item-full-attn-lambda-0p1
  full-item-full-attn-lambda-0p3
  full-item-full-attn-lambda-1p0
  full-item-full-attn-lambda-3p0
)
for config in "${configs[@]}"; do
  output="outputs/redial/hocrs/semantic-item-n64-h256/${config#full-item-}"
  if [[ -e "$output" ]]; then
    echo "Refusing to resume or overwrite existing output: $output"
    exit 1
  fi
done

mkdir -p logs/lambda
for config in "${configs[@]}"; do
  lambda="${config##*-lambda-}"
  export WANDB_TAGS="crs,semantic-item,attention,lambda-sweep,lambda-${lambda},full,seed42"
  echo "START $(date -Is) $config"
  torchrun --standalone --nproc-per-node=8 \
    -m hyprorec.scripts.train \
    --config "configs/redial/hocrs/$config.yaml" \
    2>&1 | tee "logs/lambda/$config.log"
  echo "END $(date -Is) $config"
done
