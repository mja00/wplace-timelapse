#!/usr/bin/env bash
set -Eeuo pipefail

# Resolve repo root and cd there so cron can call this from anywhere
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "$REPO_DIR"

# Prefer a venv in ./venv; fallback to ./.venv
ACTIVATE_SCRIPT=""
if [[ -f "venv/bin/activate" ]]; then
  ACTIVATE_SCRIPT="venv/bin/activate"
elif [[ -f ".venv/bin/activate" ]]; then
  ACTIVATE_SCRIPT=".venv/bin/activate"
fi

if [[ -z "$ACTIVATE_SCRIPT" ]]; then
  echo "Virtualenv not found. Create one with:"
  echo "  python -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi

# Activate venv and ensure we deactivate on exit
source "$ACTIVATE_SCRIPT"
trap 'command -v deactivate >/dev/null 2>&1 && deactivate || true' EXIT

# Unbuffered output for reliable cron logs
export PYTHONUNBUFFERED=1

# Run the downloader
python -u main.py


