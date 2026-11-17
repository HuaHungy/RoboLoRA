#!/usr/bin/env bash
# LoRAHub 组合导出入口: 基于 Pi0 基座模型注入组合 LoRA, 生成 RoboTwin 评测 checkpoint
#
# 用法:
#   bash compose_checkpoint.sh <task_name>        # 使用 config/<task_name>.json
#   bash compose_checkpoint.sh <config_path.json> # 直接指定配置文件
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_PI0_PYTHON="$HERE/../RoboTwin/policy/pi0/.venv/bin/python"
if [[ -n "${PYTHON:-}" ]]; then
  : # 显式 PYTHON 优先（服务器/自定义环境）
elif [[ -x "$LOCAL_PI0_PYTHON" ]]; then
  PYTHON="$LOCAL_PI0_PYTHON"
else
  PYTHON="$(command -v python)"
fi
[[ -n "$PYTHON" && -x "$PYTHON" ]] || {
  echo "[错误] 找不到可用的 Python；请先 conda activate lorahub-jax 或设置 PYTHON=/path/to/python"
  exit 1
}

CFG="${1:?用法: bash compose_checkpoint.sh <task_name|config.json>}"
if [[ "$CFG" != *.json ]]; then
  CFG="$HERE/config/$CFG.json"
fi
[[ -f "$CFG" ]] || { echo "[错误] 配置文件不存在: $CFG"; exit 1; }

TASK="$("$PYTHON" -c "import json,sys;print(json.load(open('$CFG'))['task_name'])")"
LOG_DIR="$HERE/outputs/$TASK/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/compose_$(date +%Y%m%d_%H%M%S).log"

echo "==> 任务: $TASK"
echo "==> 配置: $CFG"
echo "==> 日志: $LOG"
PYTHONUNBUFFERED=1 "$PYTHON" "$HERE/script/compose.py" --config "$CFG" 2>&1 | tee "$LOG"
