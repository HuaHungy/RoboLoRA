#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"
config="${1:-configs/lerobot_ee_all.json}"
if [[ "$#" -gt 1 ]]; then
    echo "Usage: bash run_lerobot_ee.sh [config.json]" >&2
    exit 2
fi
# This CPU environment was used for the full-data smoke test. Override if needed.
cluster_python="${CLUSTER_PYTHON:-/home/huahungy/anaconda3/bin/python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$script_dir/.cache/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$script_dir/.cache}"
mkdir -p "$script_dir/logs"
log_file="$script_dir/logs/lerobot_ee_$(date +%Y%m%d_%H%M%S_%N).log"
echo "Config: $config"
echo "Log: $log_file"
"$cluster_python" -u -m action_clustering run --config "$config" 2>&1 | tee "$log_file"
