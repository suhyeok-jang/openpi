# RoboCasa GR-1 Benchmark (openpi port)

Run a `pi05_gr1` policy against the GR1ArmsAndWaistFourierHands tabletop tasks
from [robocasa-gr1-tabletop-tasks](https://github.com/robocasa/robocasa-gr1-tabletop-tasks),
using the same two-process / two-venv scheme as
`Isaac-GR00T-AlinVLA/run_scripts/eval/gr1_tabletop/eval_gr1_multi.sh` — but with
the openpi websocket policy server instead of the GR00T zmq one.

## Architecture

```
┌──────────────────────────────────┐  ws://127.0.0.1:PORT  ┌──────────────────────────────┐
│  scripts/serve_policy.py         │ ───────────────────►  │  examples/robocasa_gr1/main  │
│  (openpi/.venv, py3.11)          │ msgpack-numpy         │  (robocasa_uv/.venv, py3.10) │
│  jax + openpi + pi05_gr1 ckpt    │ ◄───────────────────  │  robosuite + robocasa-gr1    │
│  WebsocketPolicyServer (GPU)     │                       │  WebsocketClientPolicy       │
└──────────────────────────────────┘                       └──────────────────────────────┘
       background &                                               foreground
                            ▲              ▲
                            └─ both spawned by the same sbatch ─┘
```

The two venvs already exist on the cluster — no setup step needed:

| Role | Venv |
|---|---|
| Policy server | `/fsx/alinlab/suhyeok/openpi/.venv` |
| Rollout / sim | `/fsx/alinlab/suhyeok/Isaac-GR00T-AlinVLA/gr00t/eval/sim/robocasa-gr1-tabletop-tasks/robocasa_uv/.venv` |

`main.py` adds `packages/openpi-client/src` to `sys.path` at import time, so
the rollout venv does not need `openpi-client` installed. The two extra deps
the rollout venv is missing (`websockets`, `tyro`) are pip-installed
idempotently inside the sbatch the first time you run it (a few seconds).

## Run (single SLURM job, server + rollout)

```bash
# Submit:
sbatch examples/robocasa_gr1/eval_robocasa_gr1.sh

# Or run directly on a GPU node:
bash   examples/robocasa_gr1/eval_robocasa_gr1.sh
```

Override config via env vars:

```bash
CKPT_DIR=/fsx/alinlab/suhyeok/openpi/checkpoints/pi05_gr1/gr1_v0_fsdp4 \
TASK_NAME=gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env \
REPLAN_STEPS=8 \
NUM_TRIALS_PER_TASK=50 \
sbatch examples/robocasa_gr1/eval_robocasa_gr1.sh
```

Env vars the script honors (with defaults):

| Var | Default | Notes |
|---|---|---|
| `OPENPI_REPO` | `/fsx/alinlab/suhyeok/openpi` | repo root |
| `SERVER_PYTHON` | `$OPENPI_REPO/.venv/bin/python` | openpi venv |
| `ROLLOUT_PYTHON` | `…/robocasa_uv/.venv/bin/python` | sim venv |
| `CKPT_DIR` | `$OPENPI_REPO/checkpoints/pi05_gr1/gr1_v0` | checkpoint passed to `policy:checkpoint --policy.dir` |
| `POLICY_CONFIG` | `pi05_gr1` | training config name |
| `TASK_NAME` | `gr1_unified/PnPCanToDrawerClose_…_Env` | env id, or `all` for the 24-task sweep |
| `EVAL_TAG` | `default` | output namespace folder |
| `SEED` | `42` | base seed (episode `i` resets with `seed + i`) |
| `NUM_TRIALS_PER_TASK` | `50` | episodes per task |
| `MAX_EPISODE_STEPS` | `720` | per-episode cap |
| `REPLAN_STEPS` | `16` | actions per chunk to execute before re-querying. `16` ≡ GR00T full open-loop; lower = libero-style closed-loop |
| `SERVER_EXTRA_ARGS` | "" | extra flags forwarded to `serve_policy.py` (before the `policy:` subcommand) |
| `OUT_ROOT` | `$OPENPI_REPO/output_seed_fixed/openpi_gr1/$EVAL_TAG` | output root |

What the script does:

1. Sanity-check both venvs and the checkpoint dir.
2. `pip install --quiet websockets tyro` into the rollout venv if missing.
3. Pick a free port (deterministic hash of `JOB_ID + ARRAY_IDX`, then bump on collision).
4. Launch `scripts/serve_policy.py` in the background, redirecting to `$OUT_DIR/server.log`.
5. Wait up to 600 s for the port to start listening (and bail with the server log tail if the server PID dies).
6. Run `examples/robocasa_gr1/main.py` in the foreground, log to `$OUT_DIR/rollout.log`.
7. `trap cleanup EXIT` kills the server PID on any exit path.

Outputs land under `$OUT_DIR/`:

- `server.log`, `rollout.log`
- `<task>-episode_NN-{success|failure}.mp4` per episode
- `simulation_results.csv` — one row per episode (`task_name, episode_idx, success, reward, steps, video_path`)
- `summary.txt` — per-task and grand totals

## Sweeping (port of GR00T's eval_gr1_multi.sh)

Add `#SBATCH --array=0-119` and an in-script index → (ckpt × task) mapping:

```bash
TASK_NAMES=( gr1_unified/PnPCupToDrawerClose_..._Env  ...  )   # 24 tasks
CKPT_STEPS=( 56000 57000 58000 59000 60000 )                   # 5 steps
NUM_TASKS=${#TASK_NAMES[@]}
RUN_IDX="$SLURM_ARRAY_TASK_ID"
TASK_NAME="${TASK_NAMES[$((RUN_IDX % NUM_TASKS))]}"
STEP="${CKPT_STEPS[$((RUN_IDX / NUM_TASKS))]}"
CKPT_DIR="$OPENPI_REPO/checkpoints/pi05_gr1/gr1_v0/$STEP"
```

The find_free_port + cleanup logic already handles concurrent array tasks on
the same node.

## Manual (no SLURM) two-window run

If you'd rather start the two pieces by hand, follow the same recipe the
sbatch encodes:

```bash
# Window A — server
source /fsx/alinlab/suhyeok/openpi/.venv/bin/activate
cd /fsx/alinlab/suhyeok/openpi
python scripts/serve_policy.py \
    --env ROBOCASA_GR1 --port 8000 \
    policy:checkpoint --policy.config pi05_gr1 --policy.dir checkpoints/pi05_gr1/gr1_v0

# Window B — rollout (no openpi-client install needed; main.py path-bootstraps it)
source /fsx/alinlab/suhyeok/Isaac-GR00T-AlinVLA/gr00t/eval/sim/robocasa-gr1-tabletop-tasks/robocasa_uv/.venv/bin/activate
pip install --quiet websockets tyro   # one-time, idempotent
cd /fsx/alinlab/suhyeok/openpi
MUJOCO_GL=egl python examples/robocasa_gr1/main.py \
    --host 127.0.0.1 --port 8000 \
    --task-name gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env \
    --num-trials-per-task 50 --max-episode-steps 720 --replan-steps 16
```

## Mapping notes (for the curious)

The pi05_gr1 model speaks raw-44-dim state and active-29-dim actions (same
layout as the LeRobot GR1-Tabletop-Merged-1000x24 dataset). The env emits
the active 29 dims split across five per-part state keys, plus a 256×256
ego image and the prompt string.

| openpi key | env source | shape / dtype |
|---|---|---|
| `observation/image` | `video.ego_view_pad_res256_freq20` | `(256, 256, 3)` uint8 |
| `observation/state` | `state.{left_arm,left_hand,right_arm,right_hand,waist}` zero-padded into raw-44 layout | `(44,)` float32 |
| `prompt` | `annotation.human.coarse_action` | str (already prefixed `unlocked_waist:` to match training data) |

Action scatter (29-dim model output → env-dict):

```
action[0:7]   → action.left_arm    → robot0_left
action[7:13]  → action.left_hand   → robot0_left_gripper
action[13:20] → action.right_arm   → robot0_right
action[20:26] → action.right_hand  → robot0_right_gripper
action[26:29] → action.waist       → robot0_torso
```

The 29 → per-part scatter lives in `robocasa_gr1_env.py`; the env owns the
dict-to-flat conversion via its `KeyConverter.unmap_action`.
