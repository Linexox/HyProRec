#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"
export TMPDIR=/data2/lhf/tmp/hyprorec-refresh
mkdir -p "$TMPDIR"
exec 9> "$TMPDIR/refresh-queue.lock"
flock -n 9 || { echo 'Another refresh queue is already running.'; exit 1; }
exec "$PROJECT_ROOT/.venv/bin/python" -u "$PROJECT_ROOT/scripts/run_refresh_queue.py" "$@"
