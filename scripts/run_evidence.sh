#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

phase="${1:-help}"
task="${2:-gsm8k}"
data_root="${DATA_ROOT:-/mnt/data/wxh/wdl}"
root="${EVIDENCE_ROOT:-$data_root/results/OODmethod/evidence-crossover-v1}"
old="$data_root/results/OODmethod"
cell="$root/$task"

case "$phase:$task" in
  replay:gsm8k)
    python -u -m ood_repro.replay_scores --shannon --resume \
      --source "${GSM_TEST_RAW:-$old/qwen25-paper-v4-matched/main/qwen_gsm8k}" --output "$cell/test"
    python -u -m ood_repro.replay_scores --shannon --resume \
      --source "${GSM_POOL_RAW:-$old/eval-section2-v1/pool/raw}" --output "$cell/pool"
    ;;
  generate:math)
    : "${MODEL_PATH:?Set MODEL_PATH to the local Qwen2.5 checkpoint}"
    : "${MODEL_REVISION:?Set MODEL_REVISION to the checkpoint commit}"
    : "${MATH_REVISION:?Set MATH_REVISION to the exported dataset commit}"
    : "${MATH_DATA:?Set MATH_DATA to the prepared JSONL directory}"
    python -c 'import math_verify'  # Check the grader before loading model weights.
    python - "$MATH_DATA" "$MATH_REVISION" <<'PY'
import json
from pathlib import Path
import sys
from ood_repro.study_io import file_hash

directory = Path(sys.argv[1])
manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
if manifest["revision"] != sys.argv[2] or any(
    file_hash(directory / f"{split}.jsonl") != manifest["splits"][split]["sha256"]
    for split in ("train", "test")
):
    raise SystemExit("MATH dataset revision or file hash differs from the export manifest")
PY
    common=(--task math --model "$MODEL_PATH" --model-revision "$MODEL_REVISION"
      --tokenizer-revision "$MODEL_REVISION" --dataset-revision "$MATH_REVISION" --local-files-only
      --ks all --shannon --save-token-scores --resume --bootstrap 0 --max-new-tokens "${MAX_TOKENS:-2048}")
    test_limit=()
    if [[ -n "${TEST_LIMIT:-}" ]]; then test_limit=(--limit "$TEST_LIMIT"); fi
    python -u -m ood_repro run "${common[@]}" --split test --data-jsonl "$MATH_DATA/test.jsonl" \
      "${test_limit[@]}" --output "$cell/test-raw"
    python -u -m ood_repro run "${common[@]}" --split train --data-jsonl "$MATH_DATA/train.jsonl" \
      --limit "${POOL_LIMIT:-2400}" --seed 1729 --output "$cell/pool-raw"
    for part in test pool; do
      python -u -m ood_repro.eval2 import-online --test "$cell/$part-raw" --output "$cell/$part" --resume
    done
    ;;
  split:gsm8k|split:math)
    preserve=()
    if [[ "$task" == gsm8k ]]; then
      preserve=(--preserve-split "${GSM_SPLIT:-$old/eval-section2-v1/protocol/splits.json}")
    fi
    python -u -m ood_repro.eval2 split --pool "$cell/pool" --test "$cell/test" \
      --output "$cell/protocol" "${preserve[@]}"
    ;;
  fit:gsm8k|fit:math)
    python -u -m ood_repro.eval2_fit --evidence-study --resume --pool "$cell/pool" --test "$cell/test" \
      --split "$cell/protocol/splits.json" --output "$cell/${FIT_TAG:-fit}" --device "${FIT_DEVICE:-cpu}" \
      --epochs "${EPOCHS:-30}" --seeds "${SEEDS:-0,1,2}" --bootstrap "${BOOTSTRAP:-2000}"
    ;;
  *) echo 'Usage: bash scripts/run_evidence.sh {replay gsm8k|generate math|split TASK|fit TASK}' >&2; exit 2 ;;
esac
