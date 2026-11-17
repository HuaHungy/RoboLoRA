#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -lt 1 ]]; then
  echo "用法: bash run_action_decoder.sh CONFIG [--smoke-test]" >&2
  exit 2
fi
CONFIG="$1"
shift
[[ "$CONFIG" == /* ]] || CONFIG="$HERE/$CONFIG"
PYTHON="${PYTHON:-$HERE/../RoboTwin/policy/pi0/.venv/bin/python}"
[[ -x "$PYTHON" ]] || { echo "找不到 Pi0 Python: $PYTHON" >&2; exit 1; }

export PYTHONUNBUFFERED=1
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_autotune_level=0}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"
if [[ "${JAX_PLATFORMS:-}" == "gpu" ]]; then
  echo "[WARN] 清除 JAX_PLATFORMS=gpu，避免错误探测 ROCm" >&2
  unset JAX_PLATFORMS
fi

exec "$PYTHON" "$HERE/action_decoders/train.py" --config "$CONFIG" "$@"
