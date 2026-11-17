#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PYTHON="${PYTHON:-$ROOT/../RoboTwin/policy/pi0/.venv/bin/python}"
"$PYTHON" "$HERE/check_feasibility.py"
if [[ "${1:-}" == "--full" ]]; then
  for config in ACT_direct DP_direct; do
    echo "[FULL CHECK] $config real checkpoint smoke update"
    timeout 900 bash "$ROOT/run_action_decoder.sh" \
      "$HERE/configs/$config.json" --smoke-test
  done
fi
