#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
phase="${1:-help}"
task="${2:-gsm8k}"
DATA_ROOT="${DATA_ROOT:-/mnt/data/wxh/wdl}"
RESULTS="$DATA_ROOT/results"
export HF_HOME="${HF_HOME:-$DATA_ROOT/cache/huggingface}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
kind=ood
case "$task" in
  gsm8k) source="$RESULTS/OODmethod/qwen25-paper-v4-matched/main/qwen_gsm8k" ;;
  math) source="$RESULTS/OODmethod/evidence-math-half-v1/math/test-raw" ;;
  step) source="$RESULTS/StepCluster/qwen25-pilot-chat-v2"; kind=step ;;
  *) echo 'Dataset must be gsm8k, math, or step' >&2; exit 2 ;;
esac
source="${GEOM_SOURCE:-$source}"
root="${GEOM_ROOT:-$RESULTS/OODmethod/geom-diagnostics-v1}"
dest="$root/$task"
limit="${LIMIT:-0}"
if [[ "$phase" == smoke ]]; then
  dest="$root/smoke/$task"
  limit="${SMOKE_LIMIT:-30}"
fi
model=()
if [[ -n "${MODEL_PATH:-}" ]]; then model=(--model "$MODEL_PATH"); fi
replay() {
  python -u -m geom_diagnostics.run replay --kind "$kind" --source "$source" \
    --output "$dest" --limit "$limit" --device "${DEVICE:-cuda}" "${model[@]}"
}
analyze() {
  python -u -m geom_diagnostics.run analyze --output "$dest" --tag "${ANALYSIS_TAG:-analysis-v1}" \
    --fpr "${TARGET_FPR:-0.1}" --bootstrap "${BOOTSTRAP:-1000}"
}
if [[ "$phase" != smoke && "$phase" != replay && "$phase" != analyze && "$phase" != all ]]; then
  echo 'Usage: bash scripts/run_geom_g1.sh {smoke|replay|analyze|all} {gsm8k|math|step}' >&2
  exit 2
fi
mkdir -p "$root/.locks"
lock_label="$task"
if [[ "$phase" == smoke ]]; then lock_label="smoke-$task"; fi
exec 9>"$root/.locks/$lock_label.lock"
flock -n 9 || { echo "Output already in use: $dest" >&2; exit 1; }
case "$phase" in
  smoke|replay) replay ;;
  analyze) analyze ;;
  all) replay; analyze ;;
esac
