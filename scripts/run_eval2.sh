#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
phase="${1:-help}"
DATA_ROOT="${DATA_ROOT:-/mnt/data/wxh/wdl}"
STUDY_ROOT="${STUDY_ROOT:-$DATA_ROOT/results/OODmethod/sequence-study-v1}"
EVAL2_ROOT="${EVAL2_ROOT:-$DATA_ROOT/results/OODmethod/eval-section2-v1}"
ORIGINAL_RUN="${ORIGINAL_RUN:-$DATA_ROOT/results/OODmethod/qwen25-paper-v4-matched/main/qwen_gsm8k}"
TEST_REPLAY="${TEST_REPLAY:-$STUDY_ROOT/replay/qwen_gsm8k}"
POOL_REPLAY="$EVAL2_ROOT/pool/replay"
SPLIT_FILE="$EVAL2_ROOT/protocol/splits.json"
export HF_HOME="${HF_HOME:-$DATA_ROOT/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TMPDIR="${TMPDIR:-$DATA_ROOT/tmp}"
mkdir -p "$TMPDIR"
case "$phase" in
  annotations)
    exec python -u -m ood_repro.eval2 prepare-annotations --test "$TEST_REPLAY" --output "$EVAL2_ROOT/annotations-template"
    ;;
  pool-config)
    exec python -u -m ood_repro.calibration_config --source "$ORIGINAL_RUN" \
      --output "$EVAL2_ROOT/pool/config" --run-output "$EVAL2_ROOT/pool/raw" --limit "${POOL_LIMIT:-2400}" --save-token-scores
    ;;
  pool)
    python -u -m ood_repro run --config "$EVAL2_ROOT/pool/config/config.json"
    exec python -u -m ood_repro.eval2 import-online --test "$EVAL2_ROOT/pool/raw" --output "$POOL_REPLAY" --resume
    ;;
  split)
    exec python -u -m ood_repro.eval2 split --pool "$POOL_REPLAY" --test "$TEST_REPLAY" --output "$EVAL2_ROOT/protocol"
    ;;
  experiment1)
    exec python -u -m ood_repro.eval2_positions --pool "$POOL_REPLAY" --test "$TEST_REPLAY" --split "$SPLIT_FILE" \
      --annotations "${ANNOTATIONS:-$EVAL2_ROOT/annotations-reviewed.jsonl}" --output "$EVAL2_ROOT/experiment1/${RESULT_TAG:-main}" \
      --bootstrap "${BOOTSTRAP:-2000}"
    ;;
  experiment2)
    exec python -u -m ood_repro.eval2_fit --pool "$POOL_REPLAY" --test "$TEST_REPLAY" --split "$SPLIT_FILE" \
      --output "$EVAL2_ROOT/experiment2/${RESULT_TAG:-main}" --metrics "${METRICS:-fdbd_all,nci}" \
      --models "${MODELS:-logreg,deepsets,gru,cnn,attention,attention_nopos,gru_shuffled}" \
      --device "${FIT_DEVICE:-cpu}" --epochs "${EPOCHS:-30}" --seeds "${SEEDS:-0,1,2}" \
      --bootstrap "${BOOTSTRAP:-2000}" --resume
    ;;
  *) echo 'Usage: bash scripts/run_eval2.sh {annotations|pool-config|pool|split|experiment1|experiment2}' ;;
esac
