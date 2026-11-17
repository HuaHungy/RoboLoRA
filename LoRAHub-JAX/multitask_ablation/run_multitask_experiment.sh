#!/usr/bin/env bash
set -Eeuo pipefail

# Five-task comparison:
#   native Pi0 -> action-expert-only training
#   LoRAHub Pi0 -> the exact same action-expert-only training
# Both arms run for 3000 steps, retain 1000/2000/3000, and evaluate every
# retained checkpoint on all five RoboTwin tasks.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PI0_ROOT="$ROOT/../RoboTwin/policy/pi0"
PYTHON="${PYTHON:-$PI0_ROOT/.venv/bin/python}"
CONFIG="$ROOT/config/place_multitask_right_50_v3_pi0_seq.json"
TASK="place_multitask_right_50_v3_pi0_seq"
DATASET="$ROOT/data/place_multitask_right_50"
OUTPUT_DIR="$ROOT/outputs/$TASK"
LOG_DIR="$OUTPUT_DIR/logs"
COMPOSE_DIR="$OUTPUT_DIR/checkpoint/10000"
LORAHUB_PARAMS="$COMPOSE_DIR/params"
COMPOSE_MARKER="$COMPOSE_DIR/.compose_complete"
EVAL_STATE_DIR="$OUTPUT_DIR/eval_state"

NATIVE_CONFIG="place_multitask_right_50_pi0_action_expert_only"
NATIVE_MODEL="place_multitask_right_50_native_action_expert_seq"
LORAHUB_CONFIG="place_multitask_right_50_pi0_lorahub_action_expert"
LORAHUB_MODEL="place_multitask_right_50_lorahub_action_expert_seq"

TASKS=(
  place_empty_cup
  place_container_plate
  place_object_stand
  place_mouse_pad
  place_object_scale
)
CHECKPOINTS=(1000 2000 3000)

mkdir -p "$LOG_DIR" "$EVAL_STATE_DIR"
LOG_FILE="$LOG_DIR/multitask_experiment_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

export ACTION_EXPERT_REFERENCE_CONFIG="place_empty_cup_pi0_lora_10ep"
export ACTION_EXPERT_REPO_ID="place_multitask_right_50"
export ACTION_EXPERT_ASSET_ID="place_multitask_right_50"
export ACTION_EXPERT_ASSETS_DIR="$COMPOSE_DIR/assets"
export ACTION_EXPERT_STEPS="3000"
export ACTION_EXPERT_SAVE_INTERVAL="1000"
export ACTION_EXPERT_KEEP_PERIOD="1000"
export ACTION_EXPERT_BATCH_SIZE="${ACTION_EXPERT_BATCH_SIZE:-8}"
export ACTION_EXPERT_NUM_WORKERS="${ACTION_EXPERT_NUM_WORKERS:-0}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$ROOT/data}"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}"
export XLA_PYTHON_CLIENT_ALLOCATOR="${XLA_PYTHON_CLIENT_ALLOCATOR:-platform}"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_autotune_level=0}"

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE"

require_inputs() {
  [[ -x "$PYTHON" ]] || { echo "Python 不存在: $PYTHON" >&2; exit 1; }
  [[ -f "$CONFIG" ]] || { echo "配置不存在: $CONFIG" >&2; exit 1; }
  [[ -f "$DATASET/meta/info.json" ]] || {
    echo "50 条多任务数据集不存在: $DATASET" >&2
    exit 1
  }
  local episodes
  episodes="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["total_episodes"])' "$DATASET/meta/info.json")"
  [[ "$episodes" == "50" ]] || {
    echo "数据集 episode 数不是 50: $episodes" >&2
    exit 1
  }
}

require_free_space_for() {
  local include_compose="$1"
  shift
  local free_kib required_gib=5 run_dir step
  if [[ "$include_compose" == "1" && ! -f "$COMPOSE_MARKER" ]]; then
    required_gib=$((required_gib + 13))
  fi
  for run_dir in "$@"; do
    for step in "${CHECKPOINTS[@]}"; do
      [[ -d "$run_dir/$step/params" ]] || required_gib=$((required_gib + 8))
    done
  done
  free_kib="$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')"
  if (( free_kib < required_gib * 1024 * 1024 )); then
    echo "可用磁盘不足 ${required_gib} GiB，无法安全保存剩余检查点。" >&2
    df -h "$ROOT"
    exit 1
  fi
  echo "磁盘预检: 剩余 $((free_kib / 1024 / 1024)) GiB，预计至少需要 ${required_gib} GiB"
}

search_complete() {
  [[ -f "$OUTPUT_DIR/lorahub_weights.json" ]] || return 1
  "$PYTHON" -c '
import json, sys
info = json.load(open(sys.argv[1]))
raise SystemExit(0 if info.get("status") != "in_progress" and "objective" in info else 1)
' "$OUTPUT_DIR/lorahub_weights.json"
}

run_search() {
  if search_complete; then
    echo "[SKIP] LoRAHub 权重已存在: $OUTPUT_DIR/lorahub_weights.json"
    return
  fi
  if [[ -f "$OUTPUT_DIR/lorahub_weights.json" ]]; then
    echo "[RESUME] 检测到 LoRAHub 临时权重，将从最近的 epoch checkpoint 继续"
  fi
  cd "$ROOT"
  bash run_lorahub.sh "$CONFIG"
  search_complete || {
    echo "LoRAHub 搜索退出，但最终权重尚未完成。" >&2
    exit 1
  }
}

compose() {
  if [[ -f "$COMPOSE_MARKER" && -d "$LORAHUB_PARAMS" ]]; then
    echo "[SKIP] 组合 checkpoint 已完成: $COMPOSE_DIR"
    return
  fi
  search_complete || {
    echo "缺少最终 LoRAHub 权重（临时快照不能用于组合），请先运行 search。" >&2
    exit 1
  }
  cd "$ROOT"
  bash compose_checkpoint.sh "$CONFIG"
  [[ -f "$COMPOSE_MARKER" && -d "$LORAHUB_PARAMS" ]] || {
    echo "组合 checkpoint 未完整生成: $COMPOSE_DIR" >&2
    exit 1
  }
}

all_checkpoints_exist() {
  local run_dir="$1" step
  for step in "${CHECKPOINTS[@]}"; do
    [[ -d "$run_dir/$step/params" ]] || return 1
    [[ -d "$run_dir/$step/assets/$ACTION_EXPERT_ASSET_ID" ]] || return 1
  done
}

train_arm() {
  local arm="$1" config_name="$2" model_name="$3" init_params="$4" use_lorahub="$5"
  local run_dir="$PI0_ROOT/checkpoints/$config_name/$model_name"
  if all_checkpoints_exist "$run_dir"; then
    echo "[SKIP] $arm 的 1000/2000/3000 checkpoints 均已存在"
    return
  fi

  local overwrite=1 resume=0
  if find "$run_dir" -mindepth 1 -maxdepth 1 -type d -regextype posix-extended \
      -regex '.*/[0-9]+' -print -quit 2>/dev/null | grep -q .; then
    overwrite=0
    resume=1
    echo "[RESUME] $arm 从已有 checkpoint 继续训练到 3000 step"
  fi

  cd "$PI0_ROOT"
  ACTION_EXPERT_CONFIG_NAME="$config_name" \
  ACTION_EXPERT_LORAHUB_MODEL="$use_lorahub" \
  ACTION_EXPERT_INIT_PARAMS="$init_params" \
  ACTION_EXPERT_OVERWRITE="$overwrite" \
  ACTION_EXPERT_RESUME="$resume" \
  "$PYTHON" "$ROOT/single_task_ablation/train_action_expert_only.py" \
    "$config_name" --exp-name="$model_name"

  if ! all_checkpoints_exist "$run_dir"; then
    echo "$arm 训练结束，但 1000/2000/3000 checkpoint 不完整: $run_dir" >&2
    exit 1
  fi
}

train_native() {
  train_arm \
    Native "$NATIVE_CONFIG" "$NATIVE_MODEL" \
    "$ROOT/openpi_base/pi0_base/params" 0
}

train_lorahub() {
  [[ -f "$COMPOSE_MARKER" && -d "$LORAHUB_PARAMS" ]] || {
    echo "缺少完整的 LoRAHub 组合 checkpoint，请先运行 compose。" >&2
    exit 1
  }
  train_arm \
    LoRAHub "$LORAHUB_CONFIG" "$LORAHUB_MODEL" \
    "$LORAHUB_PARAMS" 1
}

eval_one() {
  local arm="$1" config_name="$2" model_name="$3" use_lorahub="$4"
  local step="$5" task_name="$6"
  local run_dir="$PI0_ROOT/checkpoints/$config_name/$model_name"
  local marker="$EVAL_STATE_DIR/${arm}_step${step}_${task_name}.complete"
  local ckpt_setting="${model_name}_step${step}"

  [[ -d "$run_dir/$step/params" ]] || {
    echo "待评测 checkpoint 不存在: $run_dir/$step/params" >&2
    exit 1
  }
  if [[ -f "$marker" && "${FORCE_EVAL:-0}" != "1" ]]; then
    echo "[SKIP] 已评测: $arm step=$step task=$task_name"
    return
  fi

  echo "[EVAL] arm=$arm step=$step task=$task_name"
  cd "$PI0_ROOT"
  ACTION_EXPERT_CONFIG_NAME="$config_name" \
  ACTION_EXPERT_LORAHUB_MODEL="$use_lorahub" \
  ACTION_EXPERT_CHECKPOINT_ID="$step" \
  ACTION_EXPERT_CKPT_SETTING="$ckpt_setting" \
  "$PYTHON" "$ROOT/single_task_ablation/eval_action_expert_only.py" \
    "$task_name" demo_aloha "$config_name" "$model_name" 0
  touch "$marker"
}

eval_arm() {
  local arm="$1" config_name="$2" model_name="$3" use_lorahub="$4"
  local step task_name
  for step in "${CHECKPOINTS[@]}"; do
    for task_name in "${TASKS[@]}"; do
      eval_one "$arm" "$config_name" "$model_name" "$use_lorahub" \
        "$step" "$task_name"
    done
  done
}

eval_native() {
  eval_arm native "$NATIVE_CONFIG" "$NATIVE_MODEL" 0
}

eval_lorahub() {
  eval_arm lorahub "$LORAHUB_CONFIG" "$LORAHUB_MODEL" 1
}

require_inputs
echo "===== multitask experiment started: $(date '+%F %T %z') ====="
echo "mode=${1:-all}"
echo "log_file=$LOG_FILE"

case "${1:-all}" in
  search) run_search ;;
  compose)
    require_free_space_for 1
    compose
    ;;
  train-native)
    require_free_space_for 0 \
      "$PI0_ROOT/checkpoints/$NATIVE_CONFIG/$NATIVE_MODEL"
    train_native
    ;;
  train-lorahub)
    require_free_space_for 0 \
      "$PI0_ROOT/checkpoints/$LORAHUB_CONFIG/$LORAHUB_MODEL"
    train_lorahub
    ;;
  train)
    require_free_space_for 0 \
      "$PI0_ROOT/checkpoints/$NATIVE_CONFIG/$NATIVE_MODEL" \
      "$PI0_ROOT/checkpoints/$LORAHUB_CONFIG/$LORAHUB_MODEL"
    train_native
    train_lorahub
    ;;
  eval-native) eval_native ;;
  eval-lorahub) eval_lorahub ;;
  eval)
    eval_native
    eval_lorahub
    ;;
  all)
    require_free_space_for 1 \
      "$PI0_ROOT/checkpoints/$NATIVE_CONFIG/$NATIVE_MODEL" \
      "$PI0_ROOT/checkpoints/$LORAHUB_CONFIG/$LORAHUB_MODEL"
    run_search
    compose
    train_native
    train_lorahub
    eval_native
    eval_lorahub
    ;;
  *)
    echo "用法: $0 {search|compose|train-native|train-lorahub|train|eval-native|eval-lorahub|eval|all}" >&2
    exit 2
    ;;
esac

echo "===== mode ${1:-all} completed: $(date '+%F %T %z') ====="
