#!/usr/bin/env bash
# Source-compatible and systemd-compatible launcher:
#   source run.sh [IP] [PORT]
#   ExecStart=/bin/bash -lc 'cd /path/to/api-audio2txt && source run.sh 0.0.0.0 8000'
set -Eeuo pipefail

REQUESTED_HOST="${1:-}"
REQUESTED_PORT="${2:-}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="$(basename "${PROJECT_DIR}")"
VENV_DIR="${VENV_DIR:-${HOME}/venv/${PROJECT_NAME}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "${PROJECT_DIR}"

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [ ! -x "${VENV_DIR}/bin/python" ]; then
  "${PROJECT_DIR}/install.sh"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export NVIDIA_TF32_OVERRIDE="${NVIDIA_TF32_OVERRIDE:-1}"

HOST="${REQUESTED_HOST:-${HOST:-0.0.0.0}}"
PORT="${REQUESTED_PORT:-${PORT:-8000}}"
export HOST PORT

exec python -m uvicorn app:app --host "${HOST}" --port "${PORT}"
