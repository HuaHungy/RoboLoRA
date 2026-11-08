#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 2 ]; then
    echo "Usage: bash generate.sh NATIVE_RAW_DIR REPO_ID [--root OUTPUT_DIR] [options]" >&2
    exit 2
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
data_dir="$1"
repo_id="$2"
shift 2
exec "${PYTHON:-$script_dir/.venv/bin/python}" \
    "$script_dir/examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py" \
    --raw-dir "$data_dir" --repo-id "$repo_id" "$@"
