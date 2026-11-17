#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -d .venv ]]; then
  echo "Missing .venv — see README installation steps."
  exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate
if [[ ! -f models/asl_cnn.pt ]]; then
  echo "Model missing. Run: python -m training.download_dataset && python -m training.train && python -m training.evaluate"
  exit 1
fi
exec python app.py
