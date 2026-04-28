#!/usr/bin/env bash
# Single-GPU debug run for the pi05_gr1 finetune.
#
# Keeps per-device batch at 128 (matching the 8-GPU prod run), disables wandb,
# and stops after 50 steps so you can verify the data/model/checkpoint path
# before kicking off the full job. If you hit OOM on a single 80GB H100, drop
# to --batch-size 64 or append --ema-decay null (saves ~12GB of params copy).

set -euo pipefail

cd "$(dirname "$0")/.."

export HF_LEROBOT_HOME=${HF_LEROBOT_HOME:-/fsx/ns1/datasets}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}

EXP_NAME=${EXP_NAME:-gr1_debug}

uv run scripts/train.py pi05_gr1 \
    --exp-name="${EXP_NAME}" \
    --overwrite \
    --batch-size 128 \
    --num-train-steps 50 \
    --save-interval 50 \
    --no-wandb-enabled \
    "$@"
