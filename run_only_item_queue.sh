#!/usr/bin/env bash
set -euo pipefail
source ~/lhf/init.sh
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$PROJECT_ROOT"
export PATH=/data2/lhf/HoCRS-v3/.venv/bin:$PATH
export PYTHONPATH=$PWD/src
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_PROJECT=HyProRec-HoCRS-Ablation
export WANDB_ENTITY=linexox7-sun-yat-sen-university
export TMPDIR=/data2/lhf/tmp/hyprorec-only-item
mkdir -p "$TMPDIR" logs/only-item
python -m compileall -q src
for cfg in only-txt-item-txt only-img-item-img only-ado-item-ado only-vdo-item-vdo; do
  echo "START $cfg $(date -Is)"
  torchrun --standalone --nproc-per-node=8 -m hyprorec.scripts.train \
    --config "configs/redial/hocrs/$cfg.yaml" \
    > "logs/only-item/$cfg.log" 2>&1
  echo "DONE $cfg $(date -Is)"
done
