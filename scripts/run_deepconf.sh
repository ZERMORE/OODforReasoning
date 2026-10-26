#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
SOURCE_ROOT="${SOURCE_ROOT:-/mnt/data/wxh/wdl/results/OODmethod/qwen25-paper-v4-matched}"
DEEPCONF_ROOT="${DEEPCONF_ROOT:-/mnt/data/wxh/wdl/results/OODmethod/qwen25-deepconf-w64}"
WINDOW="${WINDOW:-64}"
BOOTSTRAP="${BOOTSTRAP:-2000}"
for task in csqa gsm8k aqua; do
  python -u -m ood_repro.deepconf \
    --source "$SOURCE_ROOT/main/qwen_$task" \
    --output "$DEEPCONF_ROOT/main/qwen_$task" \
    --window "$WINDOW" --bootstrap "$BOOTSTRAP" --local-files-only --resume "$@"
done
