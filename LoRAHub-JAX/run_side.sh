#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -ne 1 || "$1" == "-h" || "$1" == "--help" ]]; then
  echo "用法: bash run_side.sh config/place_empty_cup_10_v3_pi0_side.json"
  [[ $# -eq 1 ]] && exit 0 || exit 2
fi
CFG="$1"
if [[ "$CFG" != /* ]]; then
  CFG="$HERE/$CFG"
fi

PYTHON="${PYTHON:-$HERE/../RoboTwin/policy/pi0/.venv/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON="$(command -v python)"
[[ -x "$PYTHON" ]] || { echo "找不到 Python: $PYTHON"; exit 1; }

export PYTHONUNBUFFERED=1
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_autotune_level=0}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"

# JAX accepts `gpu` as a convenience alias and expands it to both CUDA and
# ROCm.  This machine uses NVIDIA/CUDA; probing ROCm can mask the real CUDA
# result with an unrelated `GpuAllocatorConfig` error.  Let JAX auto-select
# the available backend instead of forcing the alias.
if [[ "${JAX_PLATFORMS:-}" == "gpu" ]]; then
  echo "[WARN] JAX_PLATFORMS=gpu 会同时探测 CUDA/ROCm，已清除该变量，让 JAX 自动选择 CUDA" >&2
  unset JAX_PLATFORMS
fi

exec "$PYTHON" "$HERE/side/train_side.py" --config "$CFG"
