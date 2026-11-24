#!/usr/bin/env bash
# Run in the original ood-repro conda environment.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

phase="${1:-all}"
DATA_ROOT="${DATA_ROOT:-/mnt/data/wxh/wdl}"
SOURCE_ROOT="${SOURCE_ROOT:-$DATA_ROOT/results/OODmethod/qwen25-paper-v4-matched}"
SHANNON_ROOT="${SHANNON_ROOT:-$DATA_ROOT/results/OODmethod/csqa-shannon-v1}"
model_options=()
if [[ -n "${MODEL_PATH:-}" ]]; then
  model_options=(--model "$MODEL_PATH")
fi

case "$phase" in
  smoke)
    python -u -m ood_repro.replay_scores --shannon --resume \
      --source "$SOURCE_ROOT/main/qwen_csqa" \
      --output "$SHANNON_ROOT/smoke/qwen_csqa" --limit "${SMOKE_LIMIT:-2}" "${model_options[@]}"
    ;;
  replay|all)
    python -u -m ood_repro.replay_scores --shannon --resume \
      --source "$SOURCE_ROOT/main/qwen_csqa" \
      --output "$SHANNON_ROOT/replay/qwen_csqa" "${model_options[@]}"
    if [[ "$phase" == "all" ]]; then
      python -u scripts/plot_csqa_detector_lengths.py \
        --source "$SHANNON_ROOT/replay/qwen_csqa" \
        --output "$SHANNON_ROOT/plots/qwen_csqa" --require-shannon --overwrite
    fi
    ;;
  plot)
    python -u scripts/plot_csqa_detector_lengths.py \
      --source "$SHANNON_ROOT/replay/qwen_csqa" \
      --output "$SHANNON_ROOT/plots/qwen_csqa" --require-shannon --overwrite
    ;;
  *)
    echo "Usage: bash scripts/run_csqa_shannon.sh {smoke|replay|plot|all}" >&2
    exit 2
    ;;
esac
