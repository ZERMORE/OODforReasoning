#!/usr/bin/env bash
# Activate the original conda environment first. GPU selection is inherited.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
phase="${1:-smoke}"
task="${2:-gsm8k}"
case "$task" in gsm8k|csqa|aqua) ;; *) echo "Unknown task: $task" >&2; exit 2 ;; esac
SOURCE_ROOT="${SOURCE_ROOT:-/mnt/data/wxh/wdl/results/OODmethod/qwen25-paper-v4-matched}"
STUDY_ROOT="${STUDY_ROOT:-/mnt/data/wxh/wdl/results/OODmethod/sequence-study-v1}"
model_options=()
if [[ -n "${MODEL_PATH:-}" ]]; then model_options=(--model "$MODEL_PATH"); fi
case "$phase" in
  smoke)
    exec python -u -m ood_repro.replay_scores \
      --source "$SOURCE_ROOT/main/qwen_$task" \
      --output "$STUDY_ROOT/smoke/qwen_$task" --limit "${SMOKE_LIMIT:-20}" --resume "${model_options[@]}"
    ;;
  replay)
    exec python -u -m ood_repro.replay_scores \
      --source "$SOURCE_ROOT/main/qwen_$task" \
      --output "$STUDY_ROOT/replay/qwen_$task" --resume "${model_options[@]}"
    ;;
  analyze)
    exec python -u -m ood_repro.sequence_analysis \
      --replay "$STUDY_ROOT/replay/qwen_$task" \
      --output "$STUDY_ROOT/analysis/${ANALYSIS_TAG:-main}/qwen_$task" \
      --bootstrap "${BOOTSTRAP:-2000}" --permutations "${PERMUTATIONS:-1000}" \
      --permutation-metrics "${PERMUTATION_METRICS:-fdbd_all}"
    ;;
  *) echo "Usage: bash scripts/run_sequence.sh {smoke|replay|analyze} {gsm8k|csqa|aqua}" >&2; exit 2 ;;
esac
