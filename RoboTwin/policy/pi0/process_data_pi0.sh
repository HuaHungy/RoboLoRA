#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
    echo "Usage: bash process_data_pi0.sh TASK SETTING EPISODES [REPO_ID] [converter options]" >&2
    exit 2
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
task_name="$1"
setting="$2"
expert_data_num="$3"
shift 3
repo_id="${task_name}_pi0_ee"
if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then
    repo_id="$1"
    shift
fi
exec "${PYTHON:-$script_dir/.venv/bin/python}" \
    "$script_dir/examples/aloha_real/convert_robotwin_ee_to_lerobot.py" \
    --raw-dir "$script_dir/../../data/$task_name/$setting" \
    --repo-id "$repo_id" --max-episodes "$expert_data_num" "$@"
