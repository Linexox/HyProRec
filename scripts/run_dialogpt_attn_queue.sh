#!/usr/bin/env bash
# Run from a tmux session; stop the queue if either experiment fails.
set -euo pipefail
source ~/lhf/init.sh >/dev/null 2>&1

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"
PYTHON_BIN="${HYPROREC_PYTHON:-/data2/lhf/envs/hyprorec-refactor/bin/python}"
export PYTHONPATH="$PROJECT_ROOT/src"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export SWANLAB_PROJECT="${SWANLAB_PROJECT:-HyProRec}"
export TMPDIR="${TMPDIR:-/data2/lhf/tmp/hyprorec-dialogpt}"

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
mkdir -p "$TMPDIR"
for index in 0 1; do
  config="${configs[$index]}"
  echo "START $(date -Is) $config"
  "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc-per-node="${NPROC_PER_NODE:-8}" \
    -m hyprorec.scripts.train \
    --config "configs/redial/hocrs/$config.yaml" \
    2>&1 | tee "logs/dialogpt-attn/$config.log"
  echo "END $(date -Is) $config"
done
