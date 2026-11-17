#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$HERE/../pi0/.venv/bin/python" -u "$HERE/pipeline.py" "$@"
