#!/usr/bin/env bash
set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PI0_ROOT="$ROOT/../RoboTwin/policy/pi0"
PYTHON="${PYTHON:-$PI0_ROOT/.venv/bin/python}"
CONFIG="$ROOT/config/place_empty_cup_single_task_lorahub.json"
NATIVE_CONFIG="place_empty_cup_pi0_action_expert_only_10ep"
NATIVE_MODEL="place_empty_cup_single_task_native_action_expert"
LORAHUB_MODEL="place_empty_cup_single_task_lorahub_action_expert"
LORAHUB_PARAMS="$ROOT/outputs/place_empty_cup_single_task_lorahub/checkpoint/10000/params"

[[ -x "$PYTHON" ]] || { echo "Python 不存在: $PYTHON" >&2; exit 1; }
[[ -d "$ROOT/data/place_empty_cup_10" ]] || { echo "数据不存在" >&2; exit 1; }
[[ -d "$ROOT/openpi_base/pi0_base/params" ]] || { echo "Pi0 base checkpoint 不存在" >&2; exit 1; }

export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$ROOT/data}"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}"
export PI0_LORA_UPDATE_MODE="${PI0_LORA_UPDATE_MODE:-partial}"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE"

case "${1:-all}" in
  native)
    cd "$PI0_ROOT"
    "$PYTHON" "$HERE/train_action_expert_only.py" "$NATIVE_CONFIG" --exp-name="$NATIVE_MODEL"
    ;;
  lorahub-prep)
    cd "$ROOT"
    bash run_lorahub.sh "$CONFIG"
    bash compose_checkpoint.sh "$CONFIG"
    ;;
  lorahub-action-expert)
    [[ -d "$LORAHUB_PARAMS" ]] || "$0" lorahub-prep
    cd "$PI0_ROOT"
    ACTION_EXPERT_LORAHUB_MODEL=1 \
    ACTION_EXPERT_INIT_PARAMS="$LORAHUB_PARAMS" \
    ACTION_EXPERT_CONFIG_NAME="place_empty_cup_pi0_lorahub_action_expert_10ep" \
    "$PYTHON" "$HERE/train_action_expert_only.py" \
      place_empty_cup_pi0_lorahub_action_expert_10ep --exp-name="$LORAHUB_MODEL"
    ;;
  lorahub)
    "$0" lorahub-prep
    "$0" lorahub-action-expert
    ;;
  eval-native)
    cd "$PI0_ROOT"
    "$PYTHON" "$HERE/eval_action_expert_only.py" \
      place_empty_cup demo_aloha "$NATIVE_CONFIG" "$NATIVE_MODEL" 0
    ;;
  eval-lorahub)
    cd "$PI0_ROOT"
    ACTION_EXPERT_LORAHUB_MODEL=1 \
    ACTION_EXPERT_CONFIG_NAME="place_empty_cup_pi0_lorahub_action_expert_10ep" \
    "$PYTHON" "$HERE/eval_action_expert_only.py" \
      place_empty_cup demo_aloha place_empty_cup_pi0_lorahub_action_expert_10ep "$LORAHUB_MODEL" 0
    ;;
  all)
    "$0" lorahub
    "$0" native
    "$0" eval-native
    "$0" eval-lorahub
    ;;
  *)
    echo "用法: $0 {native|lorahub|eval-native|eval-lorahub|all}" >&2
    exit 2
    ;;
esac
