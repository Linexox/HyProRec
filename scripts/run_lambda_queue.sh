#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"
PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
export PYTHONPATH="$PROJECT_ROOT/src"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TMPDIR="${TMPDIR:-/data2/lhf/tmp/hyprorec-lambda}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
mkdir -p "$TMPDIR"
exec 9> "$TMPDIR/queue.lock"
flock -n 9 || { echo 'Another lambda queue is already running.'; exit 1; }

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
from datetime import datetime, timezone
from pathlib import Path
import yaml

recipe = yaml.safe_load(Path(sys.argv[1]).read_text())
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
recipe['training']['dataloader_num_workers'] = 4
recipe['training']['dataloader_pin_memory'] = True
recipe['training']['dataloader_persistent_workers'] = False
recipe['training']['gradient_checkpointing'] = False
recipe['training'].pop('gradient_checkpointing_kwargs', None)
recipe['training']['ddp_find_unused_parameters'] = False
(output / 'run_config.yaml').write_text(yaml.safe_dump(recipe, sort_keys=False))
print(output)
PY
)
    export SWANLAB_LOG_DIR="$PROJECT_ROOT/$output/swanlog"
    git rev-parse HEAD > "$output/git-revision.txt"
    printf '%s\n' "START $(date -u +%FT%TZ) $dataset $config" | tee "$output/status.log"
    set +e
    "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc-per-node=8 \
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
