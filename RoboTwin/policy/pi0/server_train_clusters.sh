#!/usr/bin/env bash
set -euo pipefail
pi0_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$pi0_dir/../../.." && pwd)"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$repo_dir/transfer/openpi}"
if [[ ! -f "$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model" ]]; then
  echo "Missing tokenizer: $OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model" >&2
  echo "Set OPENPI_DATA_HOME to an OpenPI cache or copy the tokenizer; see pi0_cluster_lora/README.md" >&2
  exit 1
fi
exec bash "$pi0_dir/../pi0_cluster_lora/run.sh" "$@"
