#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    PYTHON_BIN="$(command -v python)"
  fi
fi

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
WEB_CONCURRENCY="${WEB_CONCURRENCY:-1}"
UVICORN_RELOAD="${UVICORN_RELOAD:-false}"

if [[ "$UVICORN_RELOAD" == "true" ]]; then
  exec "$PYTHON_BIN" -m uvicorn dashboard_api:app --host "$HOST" --port "$PORT" --reload
fi

exec "$PYTHON_BIN" -m uvicorn dashboard_api:app --host "$HOST" --port "$PORT" --workers "$WEB_CONCURRENCY"
