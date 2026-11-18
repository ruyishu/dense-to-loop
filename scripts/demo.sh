#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

hardware="${1:-1xh20}"
run="${2:-runs/demo}"
if [[ -e "$run/init.pt" || -e "$run/opd/last.pt" ]]; then
    echo "Choose a fresh demo directory: $run" >&2
    exit 1
fi
d2l=.venv/bin/d2l
"$d2l" doctor --hardware "$hardware"
"$d2l" prepare-data --output "$run/data" --train-count 32 --dev-count 16
"$d2l" convert --hardware "$hardware" --output "$run/init.pt"
"$d2l" eval --hardware "$hardware" --checkpoint "$run/init.pt" \
    --data "$run/data/dev.jsonl" --limit 8 --max-new 48 --output "$run/before.jsonl"
"$d2l" train --hardware "$hardware" --init "$run/init.pt" --mode opd \
    --data "$run/data/train.jsonl" --output "$run/opd" --budget 512 \
    --tokens-per-step 128 --max-prompt 512 --max-new 64 --log-rollouts
"$d2l" probe --hardware "$hardware" --checkpoint "$run/init.pt" \
    --trajectories "$run/before.jsonl"
"$d2l" probe --hardware "$hardware" --checkpoint "$run/opd/last.pt" \
    --trajectories "$run/before.jsonl"
