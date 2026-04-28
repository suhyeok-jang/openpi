"""Rollout a pi05_gr1 policy against robocasa-gr1-tabletop-tasks.

Mirrors GR00T's ``gr00t/eval/rollout_policy.py`` flow:

  * one ``gym.vector.{Sync,Async}VectorEnv`` of ``n_envs`` gr1_unified envs
    (each wrapped with ``VideoRecordingWrapper`` + ``MultiStepWrapper``)
  * outer loop runs at the ``n_action_steps``-strided cadence the wrapper
    exposes; per-env per-step we ask the openpi server for a fresh chunk
  * per-env episode counters with the same target-per-env stop condition
    GR00T uses

The main differences from GR00T are infrastructure-side: openpi's policy server
speaks websocket+msgpack instead of zmq, and the model only takes per-frame
single-env inputs, so we issue ``n_envs`` sequential ``client.infer`` calls per
outer step (one per env). Env stepping is fully parallel since AsyncVectorEnv
spawns one process per env, so the bottleneck stays on the model side.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import pathlib
import sys
import time
from typing import Sequence


def _bootstrap_openpi_client() -> None:
    """Make ``openpi_client`` importable from any venv.

    The robocasa simulation venv does not have openpi-client installed, but the
    package is pure-Python and lives in this repo at
    ``packages/openpi-client/src``. Adding it to sys.path keeps the rollout
    venv untouched.
    """
    here = pathlib.Path(__file__).resolve()
    repo_root = here.parents[2]
    candidate = repo_root / "packages" / "openpi-client" / "src"
    if candidate.is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


_bootstrap_openpi_client()

import numpy as np
from openpi_client import websocket_client_policy as _websocket_client_policy
import pandas as pd
import tqdm
import tyro

from robocasa_gr1_env import (
    extract_success,
    load_existing_episodes,
    make_vector_env,
    pack_request,
    scatter_action_chunks,
)


GR1_TABLETOP_TASKS: tuple[str, ...] = (
    "gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env",
    "gr1_unified/PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env",
)


@dataclasses.dataclass
class Args:
    # ---------------- Server connection ----------------
    host: str = "0.0.0.0"
    port: int = 8000

    # ---------------- Task / rollout ----------------
    task_name: str = "gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env"

    num_trials_per_task: int = 50
    max_episode_steps: int = 720

    # Underlying env steps executed per policy call. Equals GR00T's
    # ``n_action_steps`` and the ``replan_steps`` arg of the previous
    # single-env build. With pi05_gr1's action_horizon=16, the model returns
    # exactly enough actions to fill one outer step at the default value.
    n_action_steps: int = 16

    # GR00T-style parallel rollout. n_envs=1 falls back to SyncVectorEnv
    # (single in-process env, easiest to debug); n_envs>1 spawns one process
    # per env via AsyncVectorEnv(context='spawn').
    n_envs: int = 5

    terminate_on_success: bool = True

    seed: int = 42

    # ---------------- Output ----------------
    video_out_path: str = "data/robocasa_gr1/videos"
    fps: int = 20
    steps_per_render: int = 2
    overlay_text: bool = True


def _resolve_task_names(spec: str) -> Sequence[str]:
    if spec == "all":
        return GR1_TABLETOP_TASKS
    if spec not in GR1_TABLETOP_TASKS:
        logging.warning(
            "Task name %r is not in GR1_TABLETOP_TASKS; proceeding anyway.", spec
        )
    return (spec,)


def _update_results_csv(
    csv_path: pathlib.Path,
    *,
    task_name: str,
    env_idx: int,
    episode_idx: int,
    success: bool,
    reward: float,
    steps: int,
    video_name: str,
) -> None:
    cols = ["task_name", "env_idx", "episode_idx", "success", "reward", "steps", "video_path"]
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=cols)
    if not df.empty:
        keep = ~(
            (df["task_name"] == task_name)
            & (df["env_idx"] == env_idx)
            & (df["episode_idx"] == episode_idx)
        )
        df = df[keep]
    new_row = {
        "task_name": task_name,
        "env_idx": env_idx,
        "episode_idx": episode_idx,
        "success": int(success),
        "reward": float(reward),
        "steps": int(steps),
        "video_path": video_name,
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(csv_path, index=False)


def _call_policy_per_env(
    client: _websocket_client_policy.WebsocketClientPolicy,
    obs_batch: dict,
    n_envs: int,
) -> dict:
    """Issue ``n_envs`` sequential ``client.infer`` calls and stack the chunks."""
    chunks = []
    for env_idx in range(n_envs):
        request = pack_request(obs_batch, env_idx)
        response = client.infer(request)
        action_chunk = np.asarray(response["actions"], dtype=np.float32)
        chunks.append(action_chunk)
    return scatter_action_chunks(chunks)


def _rollout_one_task(
    *,
    args: Args,
    task_name: str,
    out_dir: pathlib.Path,
    client: _websocket_client_policy.WebsocketClientPolicy,
) -> tuple[int, int, float]:
    """Run one task across ``n_envs`` parallel envs, episodes counted globally."""
    n_envs = args.n_envs
    n_target = max(args.num_trials_per_task, n_envs)
    target_per_env = max(1, n_target // n_envs)

    csv_path = out_dir / "simulation_results.csv"

    # ---- Resume: parse existing per-env mp4s and seed env_episode_indices /
    # start_episode_ids EXACTLY the way GR00T does (rollout_policy.py:_load_*).
    #   env_episode_count_i = min(max_ep_i + 1, target_per_env)
    #   wrapper_start_id_i  = env_episode_count_i - 1   (reset() does +1)
    #   env_episode_indices = env_episode_count_i        (count of completed episodes)
    env_to_max_ep, prior_successes = load_existing_episodes(out_dir, n_envs)
    env_episode_indices = [
        min(env_to_max_ep.get(i, -1) + 1, target_per_env) for i in range(n_envs)
    ]
    env_start_episode_ids = [c - 1 for c in env_episode_indices]
    completed = sum(env_episode_indices)
    completed_success = sum(1 for s in prior_successes[:completed] if s)
    if completed > 0:
        logging.info(
            "Resume: per-env existing episode counts = %s (target_per_env=%d)",
            env_episode_indices, target_per_env,
        )

    if all(env_episode_indices[i] >= target_per_env for i in range(n_envs)):
        logging.info(
            "All %d envs already at target_per_env=%d; nothing to run.",
            n_envs, target_per_env,
        )
        return completed, completed_success, 0.0

    env = make_vector_env(
        task_name,
        n_envs=n_envs,
        seed=args.seed,
        n_action_steps=args.n_action_steps,
        max_episode_steps=args.max_episode_steps,
        video_dir=str(out_dir),
        fps=args.fps,
        steps_per_render=args.steps_per_render,
        overlay_text=args.overlay_text,
        terminate_on_success=args.terminate_on_success,
        # VideoRecordingWrapper.reset increments start_episode_id by 1 on the
        # first reset, so passing the last completed index gets us episode N+1.
        start_episode_ids=env_start_episode_ids,
    )

    try:
        obs, _info = env.reset()

        current_lengths = [0] * n_envs
        current_rewards = [0.0] * n_envs
        current_successes = [False] * n_envs
        t_start = time.time()

        pbar = tqdm.tqdm(total=n_target, desc=task_name.replace("/", "_"), initial=completed)

        while completed < n_target:
            if all(env_episode_indices[i] >= target_per_env for i in range(n_envs)):
                break

            action_batch = _call_policy_per_env(client, obs, n_envs)
            next_obs, rewards, terminations, truncations, env_infos = env.step(action_batch)

            for env_idx in range(n_envs):
                # MultiStepWrapper aggregates ``info["success"]`` across the
                # n_action_steps substeps of this outer step.
                current_successes[env_idx] |= extract_success(env_infos, env_idx)

                # MultiStepWrapper returns reward as a scalar per outer step
                # (max-aggregated across the substep window). Match GR00T's
                # ``current_lengths += 1`` (= number of outer steps / policy
                # calls), not the number of underlying env steps.
                current_rewards[env_idx] += float(rewards[env_idx])
                current_lengths[env_idx] += 1

                if bool(terminations[env_idx]) or bool(truncations[env_idx]):
                    if env_episode_indices[env_idx] >= target_per_env:
                        # Don't double-count past-target rollovers from
                        # AsyncVectorEnv's auto-reset.
                        continue

                    success = current_successes[env_idx]
                    video_name = (
                        f"{task_name.replace('/', '_')}_env{env_idx:02d}"
                        f"-episode_{env_episode_indices[env_idx]}"
                        f"-{'success' if success else 'failure'}.mp4"
                    )
                    _update_results_csv(
                        csv_path,
                        task_name=task_name,
                        env_idx=env_idx,
                        episode_idx=env_episode_indices[env_idx],
                        success=success,
                        reward=current_rewards[env_idx],
                        steps=current_lengths[env_idx],
                        video_name=video_name,
                    )

                    env_episode_indices[env_idx] += 1
                    completed += 1
                    completed_success += int(success)
                    pbar.update(1)

                    current_successes[env_idx] = False
                    current_rewards[env_idx] = 0.0
                    current_lengths[env_idx] = 0

            obs = next_obs

        pbar.close()
        elapsed = time.time() - t_start
        return completed, completed_success, elapsed
    finally:
        # Final reset BEFORE close lets VideoRecordingWrapper rename or delete
        # any in-progress mp4 (otherwise envs that terminated past their
        # target_per_env leave behind orphan files without success/failure
        # suffix). Matches GR00T's rollout_policy end-of-task pattern.
        try:
            env.reset()
        except Exception:  # noqa: BLE001
            logging.exception("env.reset() at end raised")
        try:
            env.close()
        except Exception:  # noqa: BLE001
            logging.exception("env.close() raised")


def eval_robocasa_gr1(args: Args) -> None:
    np.random.seed(args.seed)

    out_root = pathlib.Path(args.video_out_path)
    out_root.mkdir(parents=True, exist_ok=True)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    logging.info("Server metadata: %s", client.get_server_metadata())

    task_names = _resolve_task_names(args.task_name)
    logging.info("Evaluating %d task(s) with n_envs=%d", len(task_names), args.n_envs)

    summary_lines: list[str] = []
    grand_total, grand_success = 0, 0

    for task_name in task_names:
        # GR00T writes per-task subdirs; the sbatch already rooted us in one,
        # but if the user passes "all" we still want one subdir per task.
        if len(task_names) > 1:
            task_out_dir = out_root / task_name.replace("/", "_")
        else:
            task_out_dir = out_root
        task_out_dir.mkdir(parents=True, exist_ok=True)

        logging.info("=== Task: %s ===  (out: %s)", task_name, task_out_dir)
        completed, success_count, elapsed = _rollout_one_task(
            args=args,
            task_name=task_name,
            out_dir=task_out_dir,
            client=client,
        )

        rate = success_count / max(completed, 1)
        summary_lines.append(
            f"{task_name}\t{completed}\t{success_count}\t{rate:.4f}\t{elapsed:.1f}s"
        )
        grand_total += completed
        grand_success += success_count
        logging.info(
            "Task %s done: %d/%d success (%.2f%%) in %.1fs",
            task_name,
            success_count,
            completed,
            100 * rate,
            elapsed,
        )

    if grand_total > 0:
        summary_lines.append(
            f"TOTAL\t{grand_total}\t{grand_success}\t{grand_success / grand_total:.4f}"
        )
    summary_path = out_root / "summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n")
    logging.info("Summary written to %s", summary_path)
    logging.info(
        "Grand total: %d/%d success (%.2f%%)",
        grand_success,
        grand_total,
        100 * grand_success / max(grand_total, 1),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    eval_robocasa_gr1(tyro.cli(Args))
