#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
pi0_python="${PI0_EE_PYTHON:-$script_dir/../pi0/.venv/bin/python}"
export PYTHONDONTWRITEBYTECODE=1
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$script_dir/../../../data_clustering/data/lerobot}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$script_dir/.cache/hf_datasets}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=disabled
exec "$pi0_python" -u "$script_dir/pipeline.py" "$@"
