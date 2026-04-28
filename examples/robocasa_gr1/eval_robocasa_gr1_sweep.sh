#!/bin/bash
#SBATCH --job-name="openpi-robocasa-gr1-sweep"
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=10
#SBATCH --mem=118G
#SBATCH --partition=l40s-gpu
#SBATCH --output=slurm_out/%x_%A_%a.out
#SBATCH --error=slurm_out/%x_%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --array=0-119

# 5 ckpts x 24 tasks = 120 jobs. Mirrors gr00t/run_scripts/eval/gr1_tabletop/eval_gr1_multi.sh.
#
#   sbatch examples/robocasa_gr1/eval_robocasa_gr1_sweep.sh
#
# Override the ckpt list / eval tag at submit time:
#   CKPT_STEPS_CSV="58000,59999" \
#   EVAL_TAG=quick \
#   sbatch --array=0-47 examples/robocasa_gr1/eval_robocasa_gr1_sweep.sh

set -euo pipefail

export NO_ALBUMENTATIONS_UPDATE=1
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# =============================== USER CONFIG ===============================
OPENPI_REPO="${OPENPI_REPO:-/fsx/alinlab/suhyeok/openpi}"
SERVER_PYTHON="${SERVER_PYTHON:-$OPENPI_REPO/.venv/bin/python}"
ROLLOUT_PYTHON="${ROLLOUT_PYTHON:-/fsx/alinlab/suhyeok/Isaac-GR00T-AlinVLA/gr00t/eval/sim/robocasa-gr1-tabletop-tasks/robocasa_uv/.venv/bin/python}"

# Default sweep matches GR00T's eval_gr1_multi.sh CKPT range, but pinned to
# the actual step numbers saved by openpi training (60000-step run terminates
# with a final save at 59999).
CKPT_RUN="${CKPT_RUN:-gr1_v0}"
CKPT_STEPS_CSV="${CKPT_STEPS_CSV:-56000,57000,58000,59000,59999}"
IFS=',' read -ra CKPT_STEPS <<<"$CKPT_STEPS_CSV"

POLICY_CONFIG="${POLICY_CONFIG:-pi05_gr1}"
EVAL_TAG="${EVAL_TAG:-default}"

SEED="${SEED:-42}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-720}"
N_ACTION_STEPS="${N_ACTION_STEPS:-16}"
N_ENVS="${N_ENVS:-5}"

OUT_ROOT="${OUT_ROOT:-$OPENPI_REPO/output_seed_fixed/openpi_gr1/$EVAL_TAG/$CKPT_RUN}"
SERVER_EXTRA_ARGS="${SERVER_EXTRA_ARGS:-}"

# 24 tabletop tasks GR00T sweeps over (verbatim from eval_gr1_multi.sh).
TASK_NAMES=(
    "gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env"
    "gr1_unified/PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env"
)

# ===========================================================================

NUM_TASKS=${#TASK_NAMES[@]}
NUM_CKPTS=${#CKPT_STEPS[@]}
TOTAL_JOBS=$(( NUM_CKPTS * NUM_TASKS ))

RUN_IDX="${1:-${SLURM_ARRAY_TASK_ID:-0}}"
if ! [[ "$RUN_IDX" =~ ^[0-9]+$ ]]; then
  echo "[e] RUN_IDX must be a non-negative integer; got '$RUN_IDX'" >&2
  exit 1
fi
if (( RUN_IDX >= TOTAL_JOBS )); then
  echo "[e] RUN_IDX=$RUN_IDX out of range (TOTAL=$TOTAL_JOBS = $NUM_CKPTS ckpts x $NUM_TASKS tasks)" >&2
  exit 1
fi

CKPT_IDX=$(( RUN_IDX / NUM_TASKS ))
TASK_IDX=$(( RUN_IDX % NUM_TASKS ))
CKPT_STEP="${CKPT_STEPS[$CKPT_IDX]}"
TASK_NAME="${TASK_NAMES[$TASK_IDX]}"
CKPT_DIR="$OPENPI_REPO/checkpoints/pi05_gr1/$CKPT_RUN/$CKPT_STEP"

if [[ ! -x "$SERVER_PYTHON" ]]; then echo "[e] missing $SERVER_PYTHON" >&2; exit 1; fi
if [[ ! -x "$ROLLOUT_PYTHON" ]]; then echo "[e] missing $ROLLOUT_PYTHON" >&2; exit 1; fi
if [[ ! -d "$CKPT_DIR" ]]; then echo "[e] missing ckpt dir $CKPT_DIR" >&2; exit 1; fi

# Idempotent install of the two pure-Python deps the rollout venv lacks.
if ! "$ROLLOUT_PYTHON" - <<'PY'
import importlib.util as iu, sys
sys.exit(1 if [m for m in ("websockets", "tyro") if iu.find_spec(m) is None] else 0)
PY
then
  if command -v uv >/dev/null 2>&1; then
    uv pip install --quiet --python "$ROLLOUT_PYTHON" websockets tyro
  else
    echo "[e] websockets/tyro missing and 'uv' not on PATH" >&2; exit 1
  fi
fi

# ----- per-array-task port and output dir -----
JOB_KEY="${SLURM_JOB_ID:-$$}"
find_free_port() {
  local p=$1
  while ss -lnt | awk '{print $4}' | grep -q ":$p$"; do
    p=$((p + 1)); [[ $p -gt 65000 ]] && p=20000
  done
  echo "$p"
}
CANDIDATE_PORT=$(( 20000 + ((JOB_KEY * 97 + RUN_IDX) % 40000) ))
PORT=$(find_free_port "$CANDIDATE_PORT")

SAFE_TASK="${TASK_NAME//\//_}"
OUT_DIR="$OUT_ROOT/$CKPT_STEP/$SAFE_TASK"
mkdir -p "$OUT_DIR" "$OPENPI_REPO/slurm_out"
SERVER_LOG="$OUT_DIR/server.log"
ROLLOUT_LOG="$OUT_DIR/rollout.log"

echo "[i] RUN_IDX        : $RUN_IDX / $((TOTAL_JOBS - 1))"
echo "[i] CKPT_STEP      : $CKPT_STEP  ($((CKPT_IDX + 1))/$NUM_CKPTS)"
echo "[i] TASK_IDX       : $TASK_IDX/$((NUM_TASKS - 1))"
echo "[i] TASK_NAME      : $TASK_NAME"
echo "[i] CKPT_DIR       : $CKPT_DIR"
echo "[i] PORT           : $PORT"
echo "[i] OUT_DIR        : $OUT_DIR"

# ----- Cleanup trap -----
SERVE_PID=""
cleanup() {
  if [[ -n "${SERVE_PID:-}" ]] && kill -0 "$SERVE_PID" 2>/dev/null; then
    kill "$SERVE_PID" 2>/dev/null || true
    wait "$SERVE_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

wait_for_server() {
  local port=$1; local timeout_s=${2:-600}; local elapsed=0
  while (( elapsed < timeout_s )); do
    if ! kill -0 "$SERVE_PID" 2>/dev/null; then
      echo "[e] server died early; tail $SERVER_LOG:" >&2
      tail -n 40 "$SERVER_LOG" >&2 || true
      return 1
    fi
    if ss -lnt | awk '{print $4}' | grep -q ":$port$"; then return 0; fi
    sleep 2; elapsed=$((elapsed + 2))
  done
  echo "[e] timeout (${timeout_s}s) waiting for server :$port" >&2
  tail -n 40 "$SERVER_LOG" >&2 || true
  return 1
}

# ----- Server (background) -----
cd "$OPENPI_REPO"
"$SERVER_PYTHON" "$OPENPI_REPO/scripts/serve_policy.py" \
  --env ROBOCASA_GR1 \
  --port "$PORT" \
  $SERVER_EXTRA_ARGS \
  policy:checkpoint \
  --policy.config "$POLICY_CONFIG" \
  --policy.dir "$CKPT_DIR" \
  >"$SERVER_LOG" 2>&1 &
SERVE_PID=$!
echo "[i] server PID=$SERVE_PID, waiting on :$PORT..."

wait_for_server "$PORT" 600

# ----- Rollout (foreground) -----
"$ROLLOUT_PYTHON" "$OPENPI_REPO/examples/robocasa_gr1/main.py" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --task-name "$TASK_NAME" \
  --num-trials-per-task "$NUM_TRIALS_PER_TASK" \
  --max-episode-steps "$MAX_EPISODE_STEPS" \
  --n-action-steps "$N_ACTION_STEPS" \
  --n-envs "$N_ENVS" \
  --seed "$SEED" \
  --video-out-path "$OUT_DIR" \
  >&"$ROLLOUT_LOG"

echo "[i] done. logs: $SERVER_LOG  $ROLLOUT_LOG"
