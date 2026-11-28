#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

phase="${1:-help}"
task="${2:-}"
DATA_ROOT="${DATA_ROOT:-/mnt/data/wxh/wdl}"
DATASET_ROOT="${G1_EXTENSION_DATA:-$DATA_ROOT/datasets/g1-extension-v1}"
RESULT_ROOT="${G1_EXTENSION_ROOT:-$DATA_ROOT/results/OODmethod/geom-extension-v1}"
SOURCE_G1="${G1_SOURCE_ROOT:-$DATA_ROOT/results/OODmethod/geom-diagnostics-v1-fixed}"
OOD_SOURCE="${OOD_SOURCE_ROOT:-$DATA_ROOT/results/OODmethod/qwen25-paper-v4-matched/main}"
MODEL="${MODEL_PATH:-$DATA_ROOT/models/Qwen2.5-7B-Instruct/a09a35458c702b33eeacc393d103063234e8bc28}"
POLICY="$RESULT_ROOT/frozen-policy.json"
export HF_HOME="${HF_HOME:-$DATA_ROOT/cache/huggingface}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${G1_CPU_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${G1_CPU_THREADS:-2}"
export MKL_NUM_THREADS="${G1_CPU_THREADS:-2}"

case "$phase/$task" in
  prepare/supergpqa|prepare/processbench|freeze/|score/csqa|score/aqua|score/supergpqa|score/processbench|\
  smoke/csqa|smoke/aqua|smoke/supergpqa|smoke/processbench|analyze/csqa|analyze/aqua|analyze/supergpqa|analyze/processbench) ;;
  *) printf '%s\n' 'Usage: run_geom_extension.sh prepare {supergpqa|processbench}' \
         '       run_geom_extension.sh freeze' \
         '       run_geom_extension.sh {smoke|score|analyze} {csqa|aqua|supergpqa|processbench}' >&2
     exit 2 ;;
esac

mkdir -p "$RESULT_ROOT/.locks"
target="$RESULT_ROOT/$task"
lock="$task"
if [[ "$phase" == smoke || ( "$phase" == analyze && "${SMOKE_ANALYSIS:-0}" == 1 ) ]]; then
  target="$RESULT_ROOT/smoke/$task"
  lock="smoke-$task"
elif [[ "$phase" == freeze ]]; then
  lock="freeze"
elif [[ "$phase" == prepare ]]; then
  mkdir -p "$DATASET_ROOT/.locks"
  exec 8>"$DATASET_ROOT/.locks/$task.lock"
  flock -n 8 || { echo "Dataset preparation already running: $task" >&2; exit 1; }
fi
exec 9>"$RESULT_ROOT/.locks/$lock.lock"
flock -n 9 || { echo "Output already in use: $target" >&2; exit 1; }

if [[ "$phase" == prepare ]]; then
  python -m geom_diagnostics.extension prepare "$task" --data "$DATASET_ROOT" --revision "${DATASET_REVISION:-main}"
  exit
fi

export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
if [[ "$phase" == freeze ]]; then
  python -m geom_diagnostics.extension freeze --source-g1 "$SOURCE_G1" --output "$POLICY"
  exit
fi
if [[ "$phase" == analyze ]]; then
  python -m geom_diagnostics.extension analyze "$task" --input "$target/scored" \
    --policy "$POLICY" --output "$target/analysis" --bootstrap "${BOOTSTRAP:-1000}"
  exit
fi

limit="${LIMIT:-0}"
if [[ "$phase" == smoke ]]; then
  limit="${SMOKE_LIMIT:-100}"
fi

case "$task" in
  csqa|aqua)
    if [[ "$phase" == smoke && -z "${SMOKE_LIMIT+x}" ]]; then
      limit=8
    fi
    python -m geom_diagnostics.run replay --kind ood --source "$OOD_SOURCE/qwen_$task" \
      --model "$MODEL" --device "${DEVICE:-cuda}" --limit "$limit" --output "$target/scored"
    ;;
  supergpqa|processbench)
    python -m geom_diagnostics.extension score "$task" --data "$DATASET_ROOT" --model "$MODEL" \
      --device "${DEVICE:-cuda}" --dtype "${DTYPE:-bfloat16}" --max-new-tokens "${MAX_NEW_TOKENS:-1024}" \
      --limit "$limit" --output "$target/scored"
    ;;
esac
