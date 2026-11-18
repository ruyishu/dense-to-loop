#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python3 -m venv .venv
.venv/bin/python -m pip install "torch==2.9.1" --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
.venv/bin/python -m pip install -e .
