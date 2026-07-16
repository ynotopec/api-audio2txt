#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="$(basename "${PROJECT_DIR}")"
VENV_DIR="${VENV_DIR:-${HOME}/venv/${PROJECT_NAME}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
UV_BIN="${UV_BIN:-uv}"

cd "${PROJECT_DIR}"
mkdir -p "$(dirname "${VENV_DIR}")"

if ! command -v "${UV_BIN}" >/dev/null 2>&1; then
  "${PYTHON_BIN}" -m pip install --user --upgrade uv
  UV_BIN="${HOME}/.local/bin/uv"
fi

"${UV_BIN}" venv --python "${PYTHON_BIN}" "${VENV_DIR}"
"${UV_BIN}" pip install --python "${VENV_DIR}/bin/python" --upgrade pip setuptools wheel
"${UV_BIN}" pip install --python "${VENV_DIR}/bin/python" --upgrade -r requirements.txt

if [ ! -f .env ]; then
  cp .env.example .env
  token="$(${PYTHON_BIN} - <<'PY'
import secrets
print(secrets.token_urlsafe(32))
PY
)"
  sed -i "s/^ASR_API_TOKENS=.*/ASR_API_TOKENS=${token}/" .env
  printf 'Created .env with a generated API token.\n'
fi

printf 'Installed %s in %s\n' "${PROJECT_NAME}" "${VENV_DIR}"
printf 'Start with: source run.sh [IP] [PORT]\n'
