#!/usr/bin/env bash
set -Eeuo pipefail

# Continue the sequential LoRAHub experiment from composition through both
# action-expert-only trainings and RoboTwin evaluation.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PI0_ROOT="$ROOT/../RoboTwin/policy/pi0"
PYTHON="${PYTHON:-$PI0_ROOT/.venv/bin/python}"
CONFIG="$ROOT/config/place_empty_cup_10_v3_pi0_seq.json"
TASK="place_empty_cup_10_v3_pi0_seq"
LOG_DIR="$ROOT/outputs/$TASK/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/full_experiment_$(date +%Y%m%d_%H%M%S).log"
# Capture every command's stdout/stderr, including child training/evaluation
# processes, while still streaming it to the terminal.
exec > >(tee -a "$LOG_FILE") 2>&1
echo "===== experiment started: $(date '+%F %T %z') ====="
echo "log_file=$LOG_FILE"
COMPOSE_DIR="$ROOT/outputs/$TASK/checkpoint/10000"
LORAHUB_PARAMS="$COMPOSE_DIR/params"
MARKER="$COMPOSE_DIR/.compose_complete"

NATIVE_CONFIG="place_empty_cup_pi0_action_expert_only_10ep"
NATIVE_MODEL="${ACTION_EXPERT_NATIVE_MODEL:-place_empty_cup_native_action_expert_seq}"
LORAHUB_CONFIG="place_empty_cup_pi0_lorahub_action_expert_10ep"
LORAHUB_MODEL="${ACTION_EXPERT_LORAHUB_MODEL_NAME:-place_empty_cup_lorahub_action_expert_seq}"

export ACTION_EXPERT_STEPS="${ACTION_EXPERT_STEPS:-3000}"
export ACTION_EXPERT_SAVE_INTERVAL="${ACTION_EXPERT_SAVE_INTERVAL:-1000}"
export ACTION_EXPERT_CHECKPOINT_ID="${ACTION_EXPERT_CHECKPOINT_ID:-$ACTION_EXPERT_STEPS}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$ROOT/data}"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export ACTION_EXPERT_ASSETS_DIR="${ACTION_EXPERT_ASSETS_DIR:-$ROOT/outputs/$TASK/checkpoint/10000/assets}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}"
export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_autotune_level=0}"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE"

[[ -x "$PYTHON" ]] || { echo "Python 不存在: $PYTHON" >&2; exit 1; }

compose() {
  cd "$ROOT"
  rm -f "$MARKER"
  bash compose_checkpoint.sh "$CONFIG"
  [[ -f "$MARKER" ]] || {
    echo "compose 未完成，停止后续步骤: $COMPOSE_DIR" >&2
    exit 1
  }
  [[ -d "$LORAHUB_PARAMS" ]] || {
    echo "LoRAHub params 不存在: $LORAHUB_PARAMS" >&2
    exit 1
  }
}

train_native() {
  local run_dir="$PI0_ROOT/checkpoints/$NATIVE_CONFIG/$NATIVE_MODEL"
  local final="$run_dir/$ACTION_EXPERT_STEPS/params"
  if [[ -d "$final" ]]; then
    echo "[SKIP] Native checkpoint 已存在: $final"
    return
  fi
  local overwrite=1 resume=0
  if find "$run_dir" -mindepth 1 -maxdepth 1 -type d -regextype posix-extended \
      -regex '.*/[0-9]+' -print -quit 2>/dev/null | grep -q .; then
    overwrite=0
    resume=1
    echo "[RESUME] Native 从已有 checkpoint 继续训练到 $ACTION_EXPERT_STEPS step"
  fi
  cd "$PI0_ROOT"
  ACTION_EXPERT_CONFIG_NAME="$NATIVE_CONFIG" \
  ACTION_EXPERT_LORAHUB_MODEL=0 \
  ACTION_EXPERT_INIT_PARAMS="$ROOT/openpi_base/pi0_base/params" \
  ACTION_EXPERT_OVERWRITE="$overwrite" \
  ACTION_EXPERT_RESUME="$resume" \
  "$PYTHON" "$HERE/train_action_expert_only.py" \
    "$NATIVE_CONFIG" --exp-name="$NATIVE_MODEL"
}

train_lorahub() {
  local run_dir="$PI0_ROOT/checkpoints/$LORAHUB_CONFIG/$LORAHUB_MODEL"
  local final="$run_dir/$ACTION_EXPERT_STEPS/params"
  if [[ -d "$final" ]]; then
    echo "[SKIP] LoRAHub action-expert checkpoint 已存在: $final"
    return
  fi
  local overwrite=1 resume=0
  if find "$run_dir" -mindepth 1 -maxdepth 1 -type d -regextype posix-extended \
      -regex '.*/[0-9]+' -print -quit 2>/dev/null | grep -q .; then
    overwrite=0
    resume=1
    echo "[RESUME] LoRAHub 从已有 checkpoint 继续训练到 $ACTION_EXPERT_STEPS step"
  fi
  cd "$PI0_ROOT"
  ACTION_EXPERT_CONFIG_NAME="$LORAHUB_CONFIG" \
  ACTION_EXPERT_LORAHUB_MODEL=1 \
  ACTION_EXPERT_INIT_PARAMS="$LORAHUB_PARAMS" \
  ACTION_EXPERT_OVERWRITE="$overwrite" \
  ACTION_EXPERT_RESUME="$resume" \
  "$PYTHON" "$HERE/train_action_expert_only.py" \
    "$LORAHUB_CONFIG" --exp-name="$LORAHUB_MODEL"
}

eval_native() {
  cd "$PI0_ROOT"
  ACTION_EXPERT_CONFIG_NAME="$NATIVE_CONFIG" \
  ACTION_EXPERT_LORAHUB_MODEL=0 \
  "$PYTHON" "$HERE/eval_action_expert_only.py" \
    place_empty_cup demo_aloha "$NATIVE_CONFIG" "$NATIVE_MODEL" 0
}

eval_lorahub() {
  cd "$PI0_ROOT"
  ACTION_EXPERT_CONFIG_NAME="$LORAHUB_CONFIG" \
  ACTION_EXPERT_LORAHUB_MODEL=1 \
  "$PYTHON" "$HERE/eval_action_expert_only.py" \
    place_empty_cup demo_aloha "$LORAHUB_CONFIG" "$LORAHUB_MODEL" 0
}

case "${1:-all}" in
  compose) compose ;;
  native) train_native ;;
  lorahub) [[ -f "$MARKER" ]] || compose; train_lorahub ;;
  eval-native) eval_native ;;
  eval-lorahub) eval_lorahub ;;
  after-compose)
    [[ -f "$MARKER" ]] || { echo "缺少 compose 完成标记，请先运行: $0 compose" >&2; exit 1; }
    train_native
    train_lorahub
    eval_native
    eval_lorahub
    ;;
  1000)
    # The original step-1000 checkpoints were removed by Orbax's
    # max_to_keep=1 policy.  Recreate them under separate experiment names so
    # the completed step-3000 checkpoints remain untouched.
    [[ -f "$MARKER" ]] || { echo "缺少 compose 完成标记，请先运行: $0 compose" >&2; exit 1; }
    export ACTION_EXPERT_STEPS=1000
    export ACTION_EXPERT_CHECKPOINT_ID=1000
    NATIVE_MODEL="place_empty_cup_native_action_expert_seq_1000"
    LORAHUB_MODEL="place_empty_cup_lorahub_action_expert_seq_1000"
    train_native
    train_lorahub
    eval_native
    eval_lorahub
    ;;
  2000)
    # Recreate the deleted step-2000 checkpoints under separate experiment
    # names, preserving both the step-3000 results and the step-1000 runs.
    [[ -f "$MARKER" ]] || { echo "缺少 compose 完成标记，请先运行: $0 compose" >&2; exit 1; }
    export ACTION_EXPERT_STEPS=2000
    export ACTION_EXPERT_CHECKPOINT_ID=2000
    NATIVE_MODEL="place_empty_cup_native_action_expert_seq_2000"
    LORAHUB_MODEL="place_empty_cup_lorahub_action_expert_seq_2000"
    train_native
    train_lorahub
    eval_native
    eval_lorahub
    ;;
  all)
    compose
    train_native
    train_lorahub
    eval_native
    eval_lorahub
    ;;
  *)
    echo "用法: $0 {compose|after-compose|1000|2000|native|lorahub|eval-native|eval-lorahub|all}" >&2
    exit 2
    ;;
esac
