#!/usr/bin/env bash
set -euo pipefail
source ~/lhf/init.sh >/dev/null 2>&1

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"
PYTHON_BIN="${HYPROREC_PYTHON:-/data2/lhf/envs/hyprorec-refactor/bin/python}"
export PYTHONPATH="$PROJECT_ROOT/src"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export SWANLAB_PROJECT="${SWANLAB_PROJECT:-HyProRec}"
export TMPDIR="${TMPDIR:-/data2/lhf/tmp/hyprorec-lambda}"

configs=(
  full-lambda-0p1
  full-lambda-0p3
  full-lambda-1p0
  full-lambda-3p0
)
for config in "${configs[@]}"; do
  output="outputs/redial/hocrs/$config"
  if [[ -e "$output" ]]; then
    echo "Refusing to resume or overwrite existing output: $output"
    exit 1
  fi
done

mkdir -p logs/lambda
mkdir -p "$TMPDIR"
for config in "${configs[@]}"; do
  echo "START $(date -Is) $config"
  "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc-per-node="${NPROC_PER_NODE:-8}" \
    -m hyprorec.scripts.train \
    --config "configs/redial/hocrs/$config.yaml" \
    2>&1 | tee "logs/lambda/$config.log"
  echo "END $(date -Is) $config"
done
