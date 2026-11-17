#!/usr/bin/env bash
set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
ROBOTWIN_ROOT="$ROOT/../RoboTwin"
PI0_DIR="$ROBOTWIN_ROOT/policy/pi0"
WAIT_SECONDS="${WAIT_SECONDS:-60}"
MODE="dry-run"
ONLY="E0_baseline,E1_query_pos,E2_action_mask,E3_is_pad_head,E4_temporal_agg,E5_cvae,E6_huber_gripper_l1"

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
bash "$HERE/check_feasibility.sh"
if [[ "$MODE" == "run" ]]; then
  bash "$HERE/check_feasibility.sh" --full
fi

MANIFEST="$HERE/manifest.tsv"
if [[ ! -f "$MANIFEST" ]]; then
  printf 'experiment\tstatus\tconfig\tcheckpoint\ttrain_log\teval_log\teval_result_dir\n' > "$MANIFEST"
fi

current_name="preflight"
trap 'code=$?; printf "%s\tfailed(%s)\t-\t-\t-\t-\t-\n" "$current_name" "$code" >> "$MANIFEST"; exit "$code"' ERR

for name in E0_baseline E1_query_pos E2_action_mask E3_is_pad_head E4_temporal_agg E5_cvae E6_huber_gripper_l1; do
  contains "$name" || continue
  current_name="$name"
  config="$HERE/configs/${name}.json"
  stamp="$(date +%Y%m%d_%H%M%S)"
  train_log="$HERE/logs/${name}_train_${stamp}.log"
  eval_log="$HERE/logs/${name}_eval_${stamp}.log"
  output_rel="$($ROOT/../RoboTwin/policy/pi0/.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["output_dir"])' "$config")"
  checkpoint="$ROOT/$output_rel/checkpoints/best.pkl"

  if [[ "$MODE" != "run" ]]; then
    if [[ "$name" == "E4_temporal_agg" ]]; then
      echo "[DRY RUN] reuse E0 checkpoint: $checkpoint"
    else
      echo "[DRY RUN] bash $ROOT/run_action_decoder.sh $config"
    fi
    echo "[DRY RUN] LORAHUB_DECODER_CONFIG=$config bash $PI0_DIR/eval.sh place_empty_cup demo_aloha place_empty_cup_pi0_lora_10ep $name 0 0"
    continue
  fi

  if [[ "$name" == "E4_temporal_agg" ]]; then
    if [[ ! -f "$checkpoint" ]]; then
      echo "[INFO] E4需要E0权重，先训练E0 baseline"
      baseline="$HERE/configs/E0_baseline.json"
      bash "$ROOT/run_action_decoder.sh" "$baseline" 2>&1 | tee "$train_log"
      echo "[WAIT] ${WAIT_SECONDS}s"
      sleep "$WAIT_SECONDS"
    else
      printf 'reuse %s\n' "$checkpoint" | tee "$train_log"
    fi
  else
    echo "===== $name training ====="
    bash "$ROOT/run_action_decoder.sh" "$config" 2>&1 | tee "$train_log"
    echo "[WAIT] ${WAIT_SECONDS}s before evaluation"
    sleep "$WAIT_SECONDS"
  fi

  [[ -f "$checkpoint" ]] || { echo "checkpoint不存在: $checkpoint" >&2; exit 1; }
  echo "===== $name RoboTwin evaluation ====="
  (
    cd "$PI0_DIR"
    export LORAHUB_JAX_ROOT="$ROOT"
    export LORAHUB_DECODER_CONFIG="$config"
    unset PI0_SIDE_CONFIG PI0_SIDE_DIFFUSION_STEPS JAX_PLATFORMS
    export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export XLA_PYTHON_CLIENT_ALLOCATOR=platform
    bash eval.sh place_empty_cup demo_aloha \
      place_empty_cup_pi0_lora_10ep "$name" 0 0
  ) 2>&1 | tee "$eval_log"

  result_root="$ROBOTWIN_ROOT/eval_result/place_empty_cup/pi0/demo_aloha/$name"
  result_dir=""
  if [[ -d "$result_root" ]]; then
    result_dir="$(find "$result_root" -mindepth 1 -maxdepth 1 -type d -printf '%T@\t%p\n' | sort -nr | head -n 1 | cut -f2-)"
  fi
  printf '%s\tok\t%s\t%s\t%s\t%s\t%s\n' \
    "$name" "$config" "$checkpoint" "$train_log" "$eval_log" "$result_dir" >> "$MANIFEST"
  "$ROOT/../RoboTwin/policy/pi0/.venv/bin/python" "$HERE/summarize.py"
  echo "[WAIT] ${WAIT_SECONDS}s before next job"
  sleep "$WAIT_SECONDS"
done

current_name="complete"
"$ROOT/../RoboTwin/policy/pi0/.venv/bin/python" "$HERE/summarize.py"
echo "[DONE] $HERE/summary.md"
