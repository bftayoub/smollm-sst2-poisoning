#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="$PYTHON"
elif [[ -x "$REPO_ROOT/../.venv/bin/python" ]]; then
  PYTHON_BIN="$REPO_ROOT/../.venv/bin/python"
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
else
  PYTHON_BIN=python
fi

TOKEN_BUDGET="${1:-1000000000}"
CONFIG_PATH="${2:-configs/from-scratch.yaml}"
CORPUS_DIR="${3:-runs/from-scratch-corpus}"
TRAINING_DIR="${4:-runs/from-scratch-classification}"
export TOKENIZERS_PARALLELISM=false

stop_preparation() {
  kill -TERM "$PREPARATION_PID" 2>/dev/null || true
  wait "$PREPARATION_PID" 2>/dev/null || true
  exit 130
}
"$PYTHON_BIN" -m poisonlab.pretrain prepare \
  --config "$CONFIG_PATH" --data-dir "$CORPUS_DIR" --until-tokens "$TOKEN_BUDGET" &
PREPARATION_PID=$!
trap stop_preparation INT TERM
wait "$PREPARATION_PID"
trap - INT TERM
exec "$PYTHON_BIN" -m poisonlab.pretrain run \
  --config "$CONFIG_PATH" --data-dir "$CORPUS_DIR" \
  --run-dir "$TRAINING_DIR" --until-tokens "$TOKEN_BUDGET"
