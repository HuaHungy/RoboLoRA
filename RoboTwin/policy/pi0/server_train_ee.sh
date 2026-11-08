#!/usr/bin/env bash
set -euo pipefail
pi0_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$pi0_dir/../../.." && pwd)"
extension_dir="$pi0_dir/../pi0_ee_five"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$repo_dir/transfer/openpi}"
stage="${1:-train}"
if [[ $# -gt 0 ]]; then shift; fi
case "$stage" in
  train|all|eval|prepare|check) ;;
  -h|--help)
    echo 'Usage: bash server_train_ee.sh [train|all|eval|prepare|check] [options]'
    echo 'Defaults: global batch=32, steps=20000, save/keep every 5000 steps.'
    echo 'Example: bash server_train_ee.sh train --gpu 0,1,2,3 --fsdp-devices 4'
    echo 'Resume: add --resume and repeat the same training arguments.'
    echo 'Paths: --dataset-root PATH --pretrained-params PATH; tokenizer: OPENPI_DATA_HOME.'
    exit 0 ;;
  *) echo "Unknown stage: $stage (use --help)" >&2; exit 2 ;;
esac
if [[ ! -f "$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model" ]]; then
  echo "Missing tokenizer: $OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model" >&2
  echo "See $pi0_dir/SERVER_TRAINING.md for the transfer layout." >&2
  exit 1
fi
exec bash "$extension_dir/run.sh" "$stage" \
  --config "$extension_dir/config.server.json" \
  --run-name ee5_joint_b32_s20000 "$@"
