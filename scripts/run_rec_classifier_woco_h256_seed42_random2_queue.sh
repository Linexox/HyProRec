#!/usr/bin/env bash

source ~/lhf/init.sh
cd /data2/lhf/HyProRec/semantic-item-pooling

export WANDB_PROJECT=HyProRec-Luna
export WANDB_ENTITY=linexox7-sun-yat-sen-university
export WANDB_TAGS=Rec-Classifier,w/o-co,h256,seed42,strict,rec
export WANDB_MODE=online
export PYTHONPATH=src
TORCHRUN=/data2/lhf/HoCRS-v3/.venv/bin/torchrun

$TORCHRUN --standalone --nproc-per-node=8 -m hyprorec.scripts.train \
  --config configs/redial/hocrs/rec-classifier-woco-h256-seed42-strict-full.yaml
$TORCHRUN --standalone --nproc-per-node=8 -m hyprorec.scripts.train \
  --config configs/redial/hocrs/rec-classifier-woco-h256-seed42-strict-txt.yaml
$TORCHRUN --standalone --nproc-per-node=8 -m hyprorec.scripts.train \
  --config configs/redial/hocrs/rec-classifier-woco-h256-seed42-strict-img.yaml
$TORCHRUN --standalone --nproc-per-node=8 -m hyprorec.scripts.train \
  --config configs/redial/hocrs/rec-classifier-woco-h256-seed42-strict-vdo.yaml
$TORCHRUN --standalone --nproc-per-node=8 -m hyprorec.scripts.train \
  --config configs/redial/hocrs/rec-classifier-woco-h256-seed42-strict-ado.yaml
