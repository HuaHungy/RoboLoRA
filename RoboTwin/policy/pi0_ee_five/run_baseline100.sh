#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
RUN_NAME="pi0_baseline100_b4_s5000"
GPU_IDS="${GPU_IDS:-0}"

for argument in "$@"; do
  if [[ "$argument" == "-h" || "$argument" == "--help" ]]; then
    exec bash "$HERE/run.sh" all \
      --config "$HERE/baseline100_config.json" \
      --run-name "$RUN_NAME" \
      --gpu "$GPU_IDS" \
      "$@"
  fi
done

bash "$HERE/run.sh" all \
  --config "$HERE/baseline100_config.json" \
  --run-name "$RUN_NAME" \
  --gpu "$GPU_IDS" \
  --batch-size 4 \
  --train-steps 5000 \
  --save-interval 1000 \
  --eval-episodes 20 \
  --eval-seed 0 \
  "$@"

ROUTER_RUN="$ROOT/RoboTwin/policy/pi0_cluster_router/runs/cluster7_top3_router_b4_s5000"
if [[ -d "$ROUTER_RUN" ]]; then
  "$HERE/../pi0/.venv/bin/python" "$HERE/compare_baseline_router.py" \
    --baseline-run "$HERE/runs/$RUN_NAME" \
    --router-run "$ROUTER_RUN" \
    --baseline-step 5000 \
    --router-step 5000 \
    --episodes 20 \
    --seed 0
else
  echo "Router run not found; baseline training/evaluation completed, comparison skipped: $ROUTER_RUN"
fi
