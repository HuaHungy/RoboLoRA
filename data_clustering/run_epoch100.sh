#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"
cluster_python="${CLUSTER_PYTHON:-/home/huahungy/anaconda3/bin/python}"
# The existing CPU clustering environment needs its matching C++ runtime.
if [[ "$cluster_python" == /home/huahungy/anaconda3/bin/python ]]; then
    export LD_LIBRARY_PATH="/home/huahungy/anaconda3/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export MPLCONFIGDIR="${MPLCONFIGDIR:-$script_dir/.cache/matplotlib}"
export PYTHONUNBUFFERED=1
mkdir -p "$script_dir/logs"
log_file="$script_dir/logs/epoch100_$(date +%Y%m%d_%H%M%S_%N).log"
echo "Log: $log_file"
"$cluster_python" -u scripts/run_epoch100.py "$@" 2>&1 | tee "$log_file"
