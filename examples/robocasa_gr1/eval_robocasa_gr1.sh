#!/bin/bash
#SBATCH --job-name="openpi-robocasa-gr1"
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=10
#SBATCH --mem=118G
#SBATCH --partition=l40s-gpu
#SBATCH --output=slurm_out/%x_%A_%a.out
#SBATCH --error=slurm_out/%x_%A_%a.err
#SBATCH --time=4:00:00

# Single-job openpi rollout against robocasa-gr1-tabletop-tasks.
#
# Mirrors gr00t/eval/.../eval_gr1_multi.sh in shape (background server + fg
# rollout, cleanup trap, find_free_port, wait_for_server) but bridges the two
# venvs the user already has on disk:
#
#   SERVER_PYTHON  = /fsx/alinlab/suhyeok/openpi/.venv               (py3.11)
#   ROLLOUT_PYTHON = /fsx/.../robocasa_uv/.venv                      (py3.10)
#
# To sweep multiple tasks/checkpoints, add ``#SBATCH --array=0-N`` and use
# $SLURM_ARRAY_TASK_ID to index a TASK_NAMES / CKPT_STEP_PAIRS list -- the
# scaffolding in eval_gr1_multi.sh ports over verbatim.

set -euo pipefail

export NO_ALBUMENTATIONS_UPDATE=1
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# ===================== USER CONFIG =====================
OPENPI_REPO="${OPENPI_REPO:-/fsx/alinlab/suhyeok/openpi}"
SERVER_PYTHON="${SERVER_PYTHON:-$OPENPI_REPO/.venv/bin/python}"
ROLLOUT_PYTHON="${ROLLOUT_PYTHON:-/fsx/alinlab/suhyeok/Isaac-GR00T-AlinVLA/gr00t/eval/sim/robocasa-gr1-tabletop-tasks/robocasa_uv/.venv/bin/python}"

# Trained pi05_gr1 checkpoint directory (params/ subdir must exist).
CKPT_DIR="${CKPT_DIR:-$OPENPI_REPO/checkpoints/pi05_gr1/gr1_v0}"
POLICY_CONFIG="${POLICY_CONFIG:-pi05_gr1}"

# Single task or "all" (sweep over the 24 GR1ArmsAndWaistFourierHands tasks
# inside one process; matches main.py's --task-name all).
TASK_NAME="${TASK_NAME:-gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env}"

EVAL_TAG="${EVAL_TAG:-default}"
SEED="${SEED:-42}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-720}"
N_ACTION_STEPS="${N_ACTION_STEPS:-${REPLAN_STEPS:-16}}"  # accept legacy REPLAN_STEPS
N_ENVS="${N_ENVS:-5}"

OUT_ROOT="${OUT_ROOT:-$OPENPI_REPO/output_seed_fixed/openpi_gr1/$EVAL_TAG}"
SERVER_EXTRA_ARGS="${SERVER_EXTRA_ARGS:-}"
# =========================================================

# ----- Sanity: both venvs exist -----
if [[ ! -x "$SERVER_PYTHON" ]]; then
  echo "[e] Missing server python: $SERVER_PYTHON" >&2
  exit 1
fi
if [[ ! -x "$ROLLOUT_PYTHON" ]]; then
  echo "[e] Missing rollout python: $ROLLOUT_PYTHON" >&2
  exit 1
fi
if [[ ! -d "$CKPT_DIR" ]]; then
  echo "[e] Missing checkpoint dir: $CKPT_DIR" >&2
  exit 1
fi

# ----- Idempotent: make sure rollout venv has the two openpi-client deps it
# doesn't ship with by default. Both are pure-Python, install is a few seconds
# the first time and a no-op afterwards. We use ``uv pip --python`` so this
# works for uv-managed venvs that don't ship a builtin ``pip``.
if ! "$ROLLOUT_PYTHON" - <<'PY'
import importlib.util as iu, sys
missing = [m for m in ("websockets", "tyro") if iu.find_spec(m) is None]
sys.exit(1 if missing else 0)
PY
then
  if ! command -v uv >/dev/null 2>&1; then
    echo "[e] websockets/tyro missing in rollout venv and 'uv' not on PATH." >&2
    echo "[e] Install with: $ROLLOUT_PYTHON -m ensurepip && $ROLLOUT_PYTHON -m pip install websockets tyro" >&2
    exit 1
  fi
  uv pip install --quiet --python "$ROLLOUT_PYTHON" websockets tyro
fi

# ----- Quick smoke import on both venvs -----
"$SERVER_PYTHON" -c 'import openpi, jax, tyro' >/dev/null 2>&1 || {
  echo "[e] openpi venv missing required imports: $SERVER_PYTHON" >&2; exit 1; }
"$ROLLOUT_PYTHON" -c 'import gymnasium, robocasa, robosuite' >/dev/null 2>&1 || {
  echo "[e] rollout venv missing robocasa/robosuite: $ROLLOUT_PYTHON" >&2; exit 1; }

# ----- Pick a free port ----------------------------------
JOB_KEY="${SLURM_JOB_ID:-$$}"
RUN_IDX="${SLURM_ARRAY_TASK_ID:-0}"
find_free_port() {
  local port=$1
  while ss -lnt | awk '{print $4}' | grep -q ":$port$"; do
    port=$((port + 1))
    [[ "$port" -gt 65000 ]] && port=20000
  done
  echo "$port"
}
CANDIDATE_PORT=$(( 20000 + ((JOB_KEY * 97 + RUN_IDX) % 40000) ))
PORT=$(find_free_port "$CANDIDATE_PORT")

# ----- Output dir + logging --------------------------------
SAFE_TASK="${TASK_NAME//\//_}"
CKPT_TAG="$(basename "$CKPT_DIR")"
OUT_DIR="$OUT_ROOT/$CKPT_TAG/$SAFE_TASK"
mkdir -p "$OUT_DIR" "$OPENPI_REPO/slurm_out"
SERVER_LOG="$OUT_DIR/server.log"
ROLLOUT_LOG="$OUT_DIR/rollout.log"

echo "[i] OPENPI_REPO    : $OPENPI_REPO"
echo "[i] SERVER_PYTHON  : $SERVER_PYTHON"
echo "[i] ROLLOUT_PYTHON : $ROLLOUT_PYTHON"
echo "[i] CKPT_DIR       : $CKPT_DIR"
echo "[i] TASK_NAME      : $TASK_NAME"
echo "[i] PORT           : $PORT"
echo "[i] OUT_DIR        : $OUT_DIR"

# ----- Cleanup trap ----------------------------------------
SERVE_PID=""
cleanup() {
  if [[ -n "${SERVE_PID:-}" ]] && kill -0 "$SERVE_PID" 2>/dev/null; then
    echo "[i] Killing server PID=$SERVE_PID"
    kill "$SERVE_PID" 2>/dev/null || true
    wait "$SERVE_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# ----- Wait helper -----------------------------------------
wait_for_server() {
  local port=$1
  local timeout_s=${2:-600}
  local elapsed=0
  while [[ "$elapsed" -lt "$timeout_s" ]]; do
    if ! kill -0 "$SERVE_PID" 2>/dev/null; then
      echo "[e] openpi server exited before becoming ready (see $SERVER_LOG)" >&2
      tail -n 40 "$SERVER_LOG" >&2 || true
      return 1
    fi
    if ss -lnt | awk '{print $4}' | grep -q ":$port$"; then
      return 0
    fi
    sleep 2
    elapsed=$((elapsed + 2))
  done
  echo "[e] Timed out after ${timeout_s}s waiting for server on port $port" >&2
  tail -n 40 "$SERVER_LOG" >&2 || true
  return 1
}

# ----- Launch server (background) --------------------------
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
echo "[i] Started openpi server PID=$SERVE_PID, waiting for port $PORT..."

wait_for_server "$PORT" 600

# ----- Run rollout (foreground) ----------------------------
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

echo "[i] Rollout finished, see $ROLLOUT_LOG"
echo "[i] Server log:           $SERVER_LOG"
