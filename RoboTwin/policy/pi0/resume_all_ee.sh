#!/usr/bin/env bash
# Resume the workspace-wide native RoboTwin -> EE LeRobot conversion.
# Completed tasks are checked; partial tasks are preserved and restarted.
set -euo pipefail

dry_run=false
if [[ "${1:-}" == "--dry-run" ]]; then
    dry_run=true
    shift
fi
if [[ "$#" -ne 0 ]]; then
    echo "Usage: bash resume_all_ee.sh [--dry-run]" >&2
    exit 2
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
workspace="$(cd -- "$script_dir/../../.." && pwd)"
export HF_LEROBOT_HOME="$workspace/data_clustering/data/lerobot"
export PYTHON="${PYTHON:-$script_dir/.venv/bin/python}"
run_id="$(date +%Y%m%d_%H%M%S_%N)"
logs="$HF_LEROBOT_HOME/_conversion_logs/$run_id"
backups="$HF_LEROBOT_HOME/_incomplete_backups/$run_id"
completed=0
pending=0

for task_dir in "$workspace"/RoboTwin/data/*; do
    [[ -d "$task_dir/demo_aloha/data" ]] || continue
    task_name="$(basename "$task_dir")"
    repo_id="${task_name}_pi0_ee"
    output="$HF_LEROBOT_HOME/$repo_id"
    if [[ -e "$output" ]]; then
        if "$PYTHON" - "$output" "$task_dir/demo_aloha" <<'PY'
import json
from pathlib import Path
import sys

import pyarrow.parquet as pq

try:
    root, source = map(Path, sys.argv[1:])
    manifest = json.loads((root / 'meta/robotwin_ee_conversion.json').read_text())
    info = json.loads((root / 'meta/info.json').read_text())
    sources = manifest['sources']
    inputs = {str(p.resolve()) for p in source.rglob('*.hdf5')}
    files = list(root.glob('data/**/*.parquet'))
    expected_frames = sum(s['frames'] for s in sources)
    complete = (
        manifest['status'] == 'complete'
        and inputs == {str(Path(s['source_path']).resolve()) for s in sources}
        and len(files) == len(sources) == info['total_episodes']
        and sum(pq.ParquetFile(p).metadata.num_rows for p in files)
            == info['total_frames'] == expected_frames
    )
except (OSError, ValueError, KeyError, TypeError):
    complete = False
sys.exit(0 if complete else 1)
PY
        then
            echo "[跳过：已完成并校验] $task_name"
            completed=$((completed + 1))
            continue
        fi
        echo "[半成品保留后重转] $output -> $backups/$repo_id"
        if ! "$dry_run"; then
            mkdir -p "$backups"
            mv -- "$output" "$backups/$repo_id"
        fi
    fi
    pending=$((pending + 1))
    echo "[转换] $task_name"
    if ! "$dry_run"; then
        mkdir -p "$logs"
        bash "$script_dir/generate.sh" "$task_dir/demo_aloha" "$repo_id" \
            2>&1 | tee "$logs/$repo_id.log"
    fi
done

if "$dry_run"; then
    echo "预览完成：跳过 $completed 个已完成任务，待转换 $pending 个任务；未修改数据。"
else
    echo "全部完成：跳过 $completed 个任务，本次转换 $pending 个任务。日志：$logs"
fi
