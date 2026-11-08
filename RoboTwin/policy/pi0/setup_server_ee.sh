#!/usr/bin/env bash
# Rebuild this project environment on the server; never copy a machine-bound .venv.
set -euo pipefail
pi0_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mode="${1:-train}"
case "$mode" in train|eval) ;; *) echo 'Usage: bash setup_server_ee.sh [train|eval]' >&2; exit 2 ;; esac
command -v uv >/dev/null || { echo 'Install uv first: python3 -m pip install uv' >&2; exit 1; }
cd "$pi0_dir"
uv sync --frozen --python 3.11
if [[ "$mode" == eval ]]; then
  command -v nvcc >/dev/null || { echo 'cuRobo compilation requires CUDA Toolkit / nvcc (CUDA 12.x).' >&2; exit 1; }
  command -v ffmpeg >/dev/null || { echo 'Install the ffmpeg system executable for evaluation video.' >&2; exit 1; }
  # Preserve the locked training dependencies while adding simulator-only extras.
  uv pip freeze --python .venv/bin/python | sed '/^-e /d' > .venv/ee-training-constraints.txt
  uv pip install --python .venv/bin/python -c .venv/ee-training-constraints.txt \
    'toppra==0.6.9' 'transforms3d==0.4.2' 'h5py' 'ninja' 'setuptools_scm'
  curobo_dir="$pi0_dir/.venv/src/nvidia-curobo"
  curobo_revision=cca894de9ec74e77a0a4071319c81958991a9108
  if [[ ! -d "$curobo_dir" ]]; then
    git clone https://github.com/NVlabs/curobo.git "$curobo_dir"
    git -C "$curobo_dir" checkout "$curobo_revision"
  elif [[ "$(git -C "$curobo_dir" rev-parse HEAD)" != "$curobo_revision" ]]; then
    echo "Existing cuRobo has a different revision: $curobo_dir; preserve it and use a fresh environment." >&2
    exit 1
  fi
  uv pip install --python .venv/bin/python --no-build-isolation \
    -c .venv/ee-training-constraints.txt -e "$curobo_dir"
fi
echo 'Environment ready. See SERVER_TRAINING.md for data transfer and launch commands.'
