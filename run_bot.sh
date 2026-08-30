#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PY="$ROOT_DIR/.venv/bin/python"

if [[ ! -x "$VENV_PY" ]]; then
  echo "[run_bot] .venv not found. Create it first (e.g. /opt/homebrew/bin/python3.12 -m venv .venv) and install deps."
  exit 1
fi

exec "$VENV_PY" -m towerlogic
