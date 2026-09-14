#!/usr/bin/env bash
set -euo pipefail

source ~/lhf/init.sh
cd /data2/lhf/HyProRec/core-moe-token

export PATH=/data2/lhf/HoCRS-v3/.venv/bin:$PATH
export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_PROJECT=HyProRec-HoCRS-Ablation
export WANDB_ENTITY=linexox7-sun-yat-sen-university
export TMPDIR=/data2/lhf/tmp/hyprorec-moe-user-v1

mkdir -p "$TMPDIR" logs/moe-user-v1
python -m compileall -q src

for config in \
    full-moe-user-v1 \
    only-txt-moe-user-v1 \
    only-img-moe-user-v1 \
    only-ado-moe-user-v1 \
    only-vdo-moe-user-v1; do
    echo "START $config $(date -Is)"
    torchrun --standalone --nproc-per-node=8 \
        -m hyprorec.scripts.train \
        --config "configs/redial/hocrs/$config.yaml" \
        > "logs/moe-user-v1/$config.log" 2>&1
    echo "DONE $config $(date -Is)"
done
