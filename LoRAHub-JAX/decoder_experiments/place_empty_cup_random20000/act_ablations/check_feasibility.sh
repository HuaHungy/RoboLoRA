#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
PYTHON="${PYTHON:-$ROOT/../RoboTwin/policy/pi0/.venv/bin/python}"
export JAX_PLATFORMS=cpu
export PYTHONUNBUFFERED=1
"$PYTHON" "$HERE/check_feasibility.py"
unset JAX_PLATFORMS

if [[ "${1:-}" == "--full" ]]; then
  for name in E0_baseline E1_query_pos E2_action_mask E3_is_pad_head E5_cvae E6_huber_gripper_l1; do
    echo "[FULL SMOKE] $name"
    timeout 900 bash "$ROOT/run_action_decoder.sh" \
      "$HERE/configs/${name}.json" --smoke-test
  done
fi
