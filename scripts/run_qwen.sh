#!/usr/bin/env bash
# Activate conda first. CUDA_VISIBLE_DEVICES is inherited; no Llama calls.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
phase="${1:-smoke}"
if [[ "$#" -gt 0 ]]; then shift; fi
exec python -u -m ood_repro.qwen_suite "$phase" "$@"
