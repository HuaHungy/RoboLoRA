#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"
if [[ "$#" -gt 1 ]]; then
    echo "Usage: bash run_visual_aux.sh [config.json]" >&2
    exit 2
fi
config="${1:-configs/lerobot_ee_visual_aux.json}"
cluster_python="${CLUSTER_PYTHON:-/home/huahungy/anaconda3/bin/python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$script_dir/.cache/matplotlib}"
mkdir -p "$script_dir/logs"
log_file="$script_dir/logs/visual_aux_$(date +%Y%m%d_%H%M%S_%N).log"
echo "Config: $config"
echo "Log: $log_file"
"$cluster_python" -u scripts/run_visual_aux.py --config "$config" 2>&1 | tee "$log_file"
