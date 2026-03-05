#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
RELOAD="${RELOAD:-1}"

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: 'uv' is required but not installed."
  echo "Install it from: https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi

if [ ! -d "${VENV_DIR}" ]; then
  echo "Creating project environment with uv..."
  uv sync --project "${PROJECT_ROOT}"
fi

if [ ! -x "${VENV_DIR}/bin/uvicorn" ]; then
  echo "Installing dependencies with uv..."
  uv sync --project "${PROJECT_ROOT}"
fi

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PROJECT_ROOT}"

if [ "${RELOAD}" = "1" ]; then
  exec "${VENV_DIR}/bin/uvicorn" examples.embodied_agent.api.main:app --host "${HOST}" --port "${PORT}" --reload
fi

exec "${VENV_DIR}/bin/uvicorn" examples.embodied_agent.api.main:app --host "${HOST}" --port "${PORT}"
