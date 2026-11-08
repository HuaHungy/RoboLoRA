#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"
cluster_python="${CLUSTER_PYTHON:-/home/huahungy/anaconda3/bin/python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$script_dir/.cache/matplotlib}"
mkdir -p "$script_dir/logs"
log_file="$script_dir/logs/visual_k_comparison_$(date +%Y%m%d_%H%M%S_%N).log"
echo "Log: $log_file"
"$cluster_python" -u scripts/compare_visual_k.py "$@" 2>&1 | tee "$log_file"
