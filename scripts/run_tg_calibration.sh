#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="$PROJECT_ROOT/src"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export TMPDIR="${TMPDIR:-/data2/lhf/tmp/hyprorec-calibration}"
mkdir -p "$TMPDIR"
exec 9> "$TMPDIR/launch.lock"
flock -n 9 || { echo 'A calibration launcher is already running.'; exit 1; }

output=$(.venv/bin/python - "${1:-configs/tg_redial/full-lr5e-4-warmup0p2.yaml}" <<'PY'
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import yaml

usage = subprocess.check_output([
    'nvidia-smi', '--id=0,1,2,3', '--query-gpu=index,memory.used,utilization.gpu',
    '--format=csv,noheader,nounits',
], text=True)
for line in usage.splitlines():
    index, memory, utilization = map(int, line.split(','))
    if memory > 100 or utilization > 0:
        raise SystemExit(f'GPU {index} is occupied; refusing to start.')
recipe = yaml.safe_load(Path(sys.argv[1]).read_text())
training = recipe['training']
if 4 * training['per_device_train_batch_size'] * training['gradient_accumulation_steps'] != 256:
    raise SystemExit('Expected four GPUs and effective batch size 256.')
base = Path(training['output_dir'])
if base.parts[:2] != ('outputs', 'tg_redial') or len(base.parts) != 3:
    raise SystemExit('Expected outputs/tg_redial/<experiment>.')
if not (Path(recipe['model']['grounding_checkpoint_path']) / 'model.safetensors').is_file():
    raise SystemExit('Grounding weights are missing.')
output = base / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
output.mkdir(parents=True, exist_ok=False)
training['output_dir'] = str(output)
training['logging_dir'] = str(output / 'trainer-logs')
(output / 'run_config.yaml').write_text(yaml.safe_dump(recipe, sort_keys=False))
command = [str(Path('.venv/bin/python').resolve()), '-m', 'torch.distributed.run',
           '--standalone', '--nproc-per-node=4', '-m', 'hyprorec.scripts.train',
           '--config', str(output / 'run_config.yaml')]
(output / 'launch.json').write_text(json.dumps({
    'command': command, 'cuda_visible_devices': '0,1,2,3', 'world_size': 4,
    'effective_batch_size': 256,
    'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in sorted(Path('src/hyprorec').rglob('*.py'))},
    'gpu_usage_before_launch': usage,
}, indent=2))
print(output)
PY
)
export SWANLAB_LOG_DIR="$PROJECT_ROOT/$output/swanlog"
printf 'OUTPUT %s\n' "$output"
nvidia-smi topo -m > "$output/gpu-topology.txt"
set +e
.venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=4 \
  -m hyprorec.scripts.train --config "$output/run_config.yaml" 2>&1 | tee "$output/console.log"
result=${PIPESTATUS[0]}
set -e
printf '%s\n' "$result" > "$output/exit-code.txt"
exit "$result"
