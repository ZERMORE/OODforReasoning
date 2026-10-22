#!/usr/bin/env bash
# Run from the repository root. Re-run identical command to resume each cell.
set -euo pipefail
for config in llama_csqa llama_gsm8k llama_aqua qwen_csqa qwen_gsm8k qwen_aqua; do
  python -m ood_repro run --config "configs/${config}.json" --resume "$@"
done
