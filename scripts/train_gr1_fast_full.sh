#!/usr/bin/env bash
# Full 8-GPU single-node finetune for pi0_fast_gr1.
#
# Same hyperparameters as scripts/train_gr1_full.sh (the pi05_gr1 launcher):
# global batch 1024, peak LR 1e-4, cosine decay to 1e-5 over 60k steps, 3k
# warmup, FSDP across all 8 H200s. Only the model swap is different.
#
# Prereq: run scripts/compute_norm_stats_fast.py once before this so norm stats
# land in assets/pi0_fast_gr1/GR1-Tabletop-Merged-1000x24/.

set -euo pipefail

cd "$(dirname "$0")/.."

export HF_LEROBOT_HOME=${HF_LEROBOT_HOME:-/fsx/ns1/datasets}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}

EXP_NAME=${EXP_NAME:-gr1_fast_v0}
FSDP_DEVICES=${FSDP_DEVICES:-8}
PROJECT_NAME=${WANDB_PROJECT:-openpi}
RESUME=${RESUME:-0}
NO_WANDB=${NO_WANDB:-0}

if [[ "${RESUME}" == "1" ]]; then
    MODE_FLAG="--resume"
else
    MODE_FLAG="--overwrite"
fi

EXTRA_FLAGS=()
if [[ "${NO_WANDB}" == "1" ]]; then
    EXTRA_FLAGS+=("--no-wandb-enabled")
fi

uv run scripts/train.py pi0_fast_gr1 \
    --exp-name="${EXP_NAME}" \
    --project-name "${PROJECT_NAME}" \
    ${MODE_FLAG} \
    --fsdp-devices "${FSDP_DEVICES}" \
    "${EXTRA_FLAGS[@]}" \
    "$@"
