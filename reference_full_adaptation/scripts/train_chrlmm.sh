#!/usr/bin/env bash
set -euo pipefail

# Bash wrapper for the adaptive CHR-LMM training launcher.
# Examples:
#   bash scripts/train_chrlmm.sh
#   bash scripts/train_chrlmm.sh --profile laptop --run
#   bash scripts/train_chrlmm.sh --profile strong \
#       --model-path checkpoints/PointLLM_7B_v1.1_init \
#       --point-ckpt checkpoints/PointLLM_7B_v1.1_init/point_bert_v1.2.pt \
#       --run

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

python scripts/train_chrlmm_auto.py "$@"
