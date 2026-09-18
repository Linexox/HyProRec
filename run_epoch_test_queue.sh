#!/usr/bin/env bash
set -euo pipefail
source ~/lhf/init.sh
cd /data2/lhf/HyProRec
export PATH=/data2/lhf/HoCRS-v3/.venv/bin:$PATH
export PYTHONPATH=$PWD/src
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_PROJECT=HyProRec-HoCRS-Ablation
export WANDB_ENTITY=linexox7-sun-yat-sen-university
export TMPDIR=/data2/lhf/tmp/hyprorec-epoch-test
mkdir -p "$TMPDIR" logs/semantic-item-pooling
python -m compileall -q src
python -c 'import torch, transformers; print(torch.__version__, transformers.__version__)'
nvidia-smi topo -m
for cfg in full-item-txt full-item-img full-item-ado full-item-vdo; do
  echo "START $cfg $(date -Is)"
  torchrun --standalone --nproc-per-node=8 -m hyprorec.scripts.train \
    --config "configs/redial/hocrs/$cfg.yaml" \
    > "logs/semantic-item-pooling/$cfg.log" 2>&1
  echo "DONE $cfg $(date -Is)"
done
