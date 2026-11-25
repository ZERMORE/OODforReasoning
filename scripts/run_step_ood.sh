#!/usr/bin/env bash
# Independent entry: do not change run_qwen.sh / run_evidence.sh / StepCluster run.sh.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export DATA_ROOT="${DATA_ROOT:-/mnt/data/wxh/wdl}"
export HF_HOME="${HF_HOME:-$DATA_ROOT/cache/huggingface}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
phase="${1:-help}"
split="${2:-test}"
task="${STEP_TASK:-gsm8k}"
source="${STEP_SOURCE:-$DATA_ROOT/results/StepCluster/${STEP_SOURCE_RUN:-qwen25-pilot-chat-v2}}"
root="${STEP_OOD_ROOT:-$DATA_ROOT/results/OODmethod/step-ood-align-v2/${STEP_SOURCE_RUN:-qwen25-pilot-chat-v2}/$task}"
model=()
if [[ -n "${MODEL_PATH:-}" ]]; then model=(--model "$MODEL_PATH"); fi
replay_split() {
  python -u -m step_ood.study replay --source "$source" --task "$task" --split "$1" \
    --output "$root/replay/$1" --ks "${KS:-all}" --device "${DEVICE:-cuda}" \
    --dtype "${DTYPE:-bfloat16}" --attention sdpa --resume "${model[@]}"
}
analysis() {
  python -u -m step_ood.study analyze --replays "$root/replay" \
    --output "$root/analysis/${ANALYSIS_TAG:-main}" --fpr "${TARGET_FPR:-0.1}" \
    --step-quantile "${STEP_QUANTILE:-0.1}" --bootstrap "${BOOTSTRAP:-1000}" --seed 42
}
case "$phase" in
  smoke)
    python -u -m step_ood.study replay --source "$source" --task "$task" --split "$split" \
      --output "$root/smoke/$split" --limit "${SMOKE_LIMIT:-2}" --ks "${KS:-all}" \
      --device "${DEVICE:-cuda}" --dtype "${DTYPE:-bfloat16}" --attention sdpa --resume "${model[@]}"
    ;;
  replay) replay_split "$split" ;;
  analyze) analysis ;;
  all)
    # Preflight all source splits before allocating a GPU or creating replay outputs.
    python -u - "$source" "$task" <<'PY'
import sys
from pathlib import Path
from step_ood.study import source_records
for split in ('train', 'validation', 'test'):
    _, _, rows = source_records(Path(sys.argv[1]), sys.argv[2], split)
    print(f'Source ready: {split}, n={len(rows)}', flush=True)
PY
    for split in train validation test; do replay_split "$split"; done
    analysis
    echo 'STEP OOD COMPLETE'
    ;;
  *) echo 'Usage: bash scripts/run_step_ood.sh {smoke|replay|analyze|all} [train|validation|test]' >&2; exit 2 ;;
esac
