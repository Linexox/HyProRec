#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"
PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
export PYTHONPATH="$PROJECT_ROOT/src"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
num_gpus=${#gpu_ids[@]}
runtime_dir="$PROJECT_ROOT/.runtime"
mkdir -p "$runtime_dir/locks"
export TMPDIR
TMPDIR=$(mktemp -d "$runtime_dir/job.XXXXXXXX")
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
for gpu in "${gpu_ids[@]}"; do
  exec {gpu_lock}> "$runtime_dir/locks/gpu-$gpu.lock"
  flock -n "$gpu_lock" || { echo "Another lambda job has reserved GPU $gpu."; exit 1; }
done

datasets=("$@")
if [[ $# -eq 0 ]]; then datasets=(redial tg_redial); fi
for dataset in "${datasets[@]}"; do
  [[ "$dataset" == redial || "$dataset" == tg_redial ]] || exit 2
done

configs=(
  full-lambda-0p1
  full-lambda-0p3
  full-lambda-1p0
  full-lambda-3p0
)
if [[ -n "${LAMBDA_CONFIGS:-}" ]]; then
  read -r -a configs <<< "$LAMBDA_CONFIGS"
fi
start_dataset="${START_DATASET:-${datasets[0]}}"
start_config="${START_CONFIG:-${configs[0]}}"
started=false
for dataset in "${datasets[@]}"; do
  for config in "${configs[@]}"; do
    if [[ "$started" == false ]]; then
      if [[ "$dataset" != "$start_dataset" || "$config" != "$start_config" ]]; then
        continue
      fi
      started=true
    fi
    # Generate the time ID here; the editable YAML owns the experiment name.
    output=$("$PYTHON_BIN" - "configs/$dataset/$config.yaml" "$dataset" <<'PY'
import sys
import os
import json
import hashlib
from importlib.metadata import version
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import yaml

recipe = yaml.safe_load(Path(sys.argv[1]).read_text())
gpus = os.environ['CUDA_VISIBLE_DEVICES']
world_size = len(gpus.split(','))
training = recipe['training']
batch_size = world_size * training['per_device_train_batch_size'] * training['gradient_accumulation_steps']
if batch_size != 256:
    raise SystemExit(f'Expected effective batch 256, got {batch_size} with {world_size} GPUs.')
usage = subprocess.check_output([
    'nvidia-smi', f'--id={gpus}', '--query-gpu=index,memory.used,utilization.gpu',
    '--format=csv,noheader,nounits',
], text=True)
for line in usage.splitlines():
    index, memory, utilization = map(int, line.split(','))
    if memory > 100 or utilization > 0:
        raise SystemExit(f'GPU {index} is occupied; refusing to start.')
base = Path(recipe['training']['output_dir'])
if base.parts[:2] != ('outputs', sys.argv[2]) or len(base.parts) != 3:
    raise SystemExit('output_dir must be outputs/<dataset>/<experiment>')
checkpoint = Path(recipe['model']['grounding_checkpoint_path'])
if not (checkpoint / 'model.safetensors').is_file():
    raise SystemExit(f'Grounding weights missing: {checkpoint}')
output = base / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
output.mkdir(parents=True, exist_ok=False)
recipe['training']['output_dir'] = str(output)
recipe['training']['logging_dir'] = str(output / 'trainer-logs')
(output / 'run_config.yaml').write_text(yaml.safe_dump(recipe, sort_keys=False))
command = [str(Path('.venv/bin/python').absolute()), '-m', 'torch.distributed.run',
           '--standalone', f'--nproc-per-node={world_size}',
           '-m', 'hyprorec.scripts.train', '--config', str(output / 'run_config.yaml')]
(output / 'launch.json').write_text(json.dumps({
    'command': command, 'cuda_visible_devices': gpus, 'world_size': world_size,
    'effective_batch_size': batch_size,
    'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'package_versions': {name: version(name) for name in ('torch', 'transformers', 'accelerate', 'swanlab')},
    'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in sorted(Path('src/hyprorec').rglob('*.py'))},
    'gpu_usage_before_launch': usage,
}, indent=2))
print(output)
PY
)
    export SWANLAB_LOG_DIR="$PROJECT_ROOT/$output/swanlog"
    git rev-parse HEAD > "$output/git-revision.txt"
    nvidia-smi topo -m > "$output/gpu-topology.txt"
    printf '%s\n' "START $(date -u +%FT%TZ) $dataset $config" | tee "$output/status.log"
    set +e
    "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc-per-node="$num_gpus" \
      -m hyprorec.scripts.train --config "$output/run_config.yaml" \
      2>&1 | tee "$output/console.log"
    result=${PIPESTATUS[0]}
    set -e
    printf '%s\n' "$result" > "$output/exit-code.txt"
    printf '%s\n' "END $(date -u +%FT%TZ) exit=$result" | tee -a "$output/status.log"
    [[ "$result" -eq 0 ]] || exit "$result"
  done
done
[[ "$started" == true ]] || {
  echo "Unknown queue start: ${start_dataset}/${start_config}" >&2
  exit 2
}
echo 'Lambda queue complete. Review validation metrics before configuring eta.'
