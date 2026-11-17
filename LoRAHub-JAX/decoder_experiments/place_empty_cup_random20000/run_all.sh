#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
ROBOTWIN_ROOT="$ROOT/../RoboTwin"
PI0_DIR="$ROBOTWIN_ROOT/policy/pi0"
WAIT_SECONDS="${WAIT_SECONDS:-60}"
MODE="dry-run"
ONLY="ACT,DP"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run) MODE="run"; shift ;;
    --dry-run) MODE="dry-run"; shift ;;
    --only) ONLY="$2"; shift 2 ;;
    --only=*) ONLY="${1#--only=}"; shift ;;
    *) echo "未知参数: $1" >&2; exit 2 ;;
  esac
done

contains() { [[ ",$ONLY," == *",$1,"* ]]; }
mkdir -p "$HERE/logs"
"$HERE/check_feasibility.sh"
if [[ "$MODE" == "run" ]]; then
  "$HERE/check_feasibility.sh" --full
  if [[ ! -f "$HERE/manifest.tsv" ]]; then
    printf 'decoder\tstatus\tconfig\ttrain_log\teval_log\n' > "$HERE/manifest.tsv"
  fi
fi

for name in ACT DP; do
  contains "$name" || continue
  config="$HERE/configs/${name}_direct.json"
  output_name="${name}_direct"
  stamp="$(date +%Y%m%d_%H%M%S)"
  train_log="$HERE/logs/${name}_train_${stamp}.log"
  eval_log="$HERE/logs/${name}_eval_${stamp}.log"
  if [[ "$MODE" != "run" ]]; then
    echo "[DRY RUN] bash $ROOT/run_action_decoder.sh $config"
    echo "[DRY RUN] LORAHUB_DECODER_CONFIG=$config bash $PI0_DIR/eval.sh place_empty_cup demo_aloha place_empty_cup_pi0_lora_10ep $output_name 0 0"
    continue
  fi

  echo "===== $name direct decoder training ====="
  bash "$ROOT/run_action_decoder.sh" "$config" 2>&1 | tee "$train_log"
  echo "[WAIT] ${WAIT_SECONDS}s before evaluation"
  sleep "$WAIT_SECONDS"

  echo "===== $name RoboTwin evaluation ====="
  (
    cd "$PI0_DIR"
    export LORAHUB_JAX_ROOT="$ROOT"
    export LORAHUB_DECODER_CONFIG="$config"
    unset PI0_SIDE_CONFIG PI0_SIDE_DIFFUSION_STEPS
    export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export XLA_PYTHON_CLIENT_ALLOCATOR=platform
    bash eval.sh place_empty_cup demo_aloha \
      place_empty_cup_pi0_lora_10ep "$output_name" 0 0
  ) 2>&1 | tee "$eval_log"
  printf '%s\tok\t%s\t%s\t%s\n' \
    "$name" "$config" "$train_log" "$eval_log" >> "$HERE/manifest.tsv"
  echo "[WAIT] ${WAIT_SECONDS}s before next job"
  sleep "$WAIT_SECONDS"
done
