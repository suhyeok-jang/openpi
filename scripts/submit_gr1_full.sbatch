#!/bin/bash
# =============================================================================
# Submit with:
#   cd /fsx/alinlab/suhyeok/openpi && sbatch scripts/submit_gr1_full.sbatch
#
# Override exp name if needed:
#   EXP_NAME=gr1_v1 sbatch scripts/submit_gr1_full.sbatch
#
# Tail logs:
#   tail -f slurm_out/<jobid>-Pi05_bsz1024_60k_gr1_1000demo_ft.out
# =============================================================================
#SBATCH --job-name=Pi05_bsz1024_60k_gr1_1000demo_ft
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --cpus-per-gpu=24
#SBATCH --mem=0
#SBATCH --partition=rlwrld-gpu
#SBATCH --nodelist=rlwrld-gpu-st-p5en-48xl-4
#SBATCH --output=slurm_out/%j-%x.out
#SBATCH --error=slurm_out/%j-%x.err
# Full node: 24 CPU/GPU × 8 = 192 CPU, --mem=0 = all node RAM (1992GB).

set -euo pipefail

cd /fsx/alinlab/suhyeok/openpi

export HF_LEROBOT_HOME=/fsx/ns1/datasets
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export WANDB_PROJECT=g0_v1d4d1v_robocasa_gr1
# Overlap FSDP all-gather/reduce-scatter with compute; enable Triton GEMM
# where possible. Adds a little compile time, expected 5-10% step throughput.
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_triton_gemm_any=true"
# ~/.netrc holds the wandb API key (chmod 600). No secret in this file.

EXP_NAME=${EXP_NAME:-gr1_v0}
RESUME=${RESUME:-0}
NO_WANDB=${NO_WANDB:-0}

echo "=== sbatch env ==="
echo "JOB: ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "NODE: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ',')"
echo "CPU: $(nproc) (alloc) / $(grep -c processor /proc/cpuinfo) (total)"
echo "MEM: $(free -g | awk '/^Mem:/{print $2}') GB"
echo "EXP_NAME: ${EXP_NAME}"
echo "WANDB_PROJECT: ${WANDB_PROJECT}"
echo "RESUME: ${RESUME}"
echo "NO_WANDB: ${NO_WANDB}"
echo "=================="

EXP_NAME="${EXP_NAME}" WANDB_PROJECT="${WANDB_PROJECT}" RESUME="${RESUME}" NO_WANDB="${NO_WANDB}" \
    bash scripts/train_gr1_full.sh
