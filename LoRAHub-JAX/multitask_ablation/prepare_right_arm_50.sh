#!/usr/bin/env bash
set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LORAHUB_ROOT="$(cd "$HERE/.." && pwd)"
WORKSPACE_ROOT="$(cd "$LORAHUB_ROOT/.." && pwd)"
ROBOTWIN_ROOT="$WORKSPACE_ROOT/RoboTwin"
PI0_ROOT="$ROBOTWIN_ROOT/policy/pi0"
PYTHON="${PYTHON:-$PI0_ROOT/.venv/bin/python}"

DATASET_NAME="place_multitask_right_50"
STAGING_ROOT="$ROBOTWIN_ROOT/data/$DATASET_NAME/demo_aloha"
PROCESSED_ROOT="$PI0_ROOT/processed_data/${DATASET_NAME}-demo_aloha-50"
OUTPUT_ROOT="$LORAHUB_ROOT/data/$DATASET_NAME"
MANIFEST="$HERE/right_arm_50_selection.json"

[[ -x "$PYTHON" ]] || { echo "Python 不存在: $PYTHON" >&2; exit 1; }
[[ -f "$MANIFEST" ]] || { echo "选择清单不存在: $MANIFEST" >&2; exit 1; }

for path in "$STAGING_ROOT" "$PROCESSED_ROOT" "$OUTPUT_ROOT"; do
  if [[ -e "$path" ]]; then
    echo "目标已存在，为避免覆盖而停止: $path" >&2
    exit 1
  fi
done

mkdir -p "$STAGING_ROOT/data" "$STAGING_ROOT/instructions"

tasks=(
  place_empty_cup
  place_container_plate
  place_object_stand
  place_mouse_pad
  place_object_scale
)
selections=(
  "0 1 2 3 4 6 9 10 12 14"
  "1 2 3 5 6 7 9 11 12 14"
  "1 3 4 5 7 8 9 11 12 13"
  "1 3 6 7 8 9 11 12 16 19"
  "0 5 6 7 9 10 14 16 17 18"
)

output_episode=0
for task_index in "${!tasks[@]}"; do
  task="${tasks[$task_index]}"
  for source_episode in ${selections[$task_index]}; do
    source_root="$ROBOTWIN_ROOT/data/$task/demo_aloha"
    source_hdf5="$source_root/data/episode${source_episode}.hdf5"
    source_instruction="$source_root/instructions/episode${source_episode}.json"
    [[ -f "$source_hdf5" ]] || { echo "缺少源数据: $source_hdf5" >&2; exit 1; }
    [[ -f "$source_instruction" ]] || { echo "缺少指令: $source_instruction" >&2; exit 1; }
    ln -s "$source_hdf5" "$STAGING_ROOT/data/episode${output_episode}.hdf5"
    ln -s "$source_instruction" "$STAGING_ROOT/instructions/episode${output_episode}.json"
    output_episode=$((output_episode + 1))
  done
done

[[ "$output_episode" -eq 50 ]] || { echo "选择数量不是 50: $output_episode" >&2; exit 1; }

cd "$PI0_ROOT"
"$PYTHON" scripts/process_data.py "$DATASET_NAME" demo_aloha 50

HF_LEROBOT_HOME="$LORAHUB_ROOT/data" \
  "$PYTHON" examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py \
    --raw-dir "$PROCESSED_ROOT" \
    --repo-id "$DATASET_NAME" \
    --mode image \
    --seed 42 \
    --no-push-to-hub

cp "$MANIFEST" "$OUTPUT_ROOT/selection_manifest.json"
echo "完成: $OUTPUT_ROOT"
