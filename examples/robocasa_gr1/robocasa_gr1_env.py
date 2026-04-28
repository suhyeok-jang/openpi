"""Env factory + obs/action helpers for the gr1_unified robocasa-gr1 envs.

Mirrors the GR00T eval stack ([rollout_policy.py:create_eval_env]):

    base_env  = gym.make("gr1_unified/<EnvName>", enable_render=True, seed=...)
    base_env  = VideoRecordingWrapper(base_env, ...)        # per-step h264 stream
    base_env  = MultiStepWrapper(base_env,
                                  video_delta_indices=[0],
                                  state_delta_indices=[0],
                                  n_action_steps=replan_steps,
                                  max_episode_steps=720,
                                  terminate_on_success=True)

The rollout client wraps a list of these factories into ``gym.vector.{Sync,Async}VectorEnv``
so all per-env state stays inside spawned subprocesses, just like GR00T.

Three helpers exposed to the rollout loop:

  * ``pack_request(obs_batch, env_idx) -> dict``      build openpi-format obs
  * ``scatter_action_chunks(chunks)   -> dict``       (n_envs, T, dim) batched action
  * ``extract_success(env_infos, env_idx) -> bool``   normalize success across info shapes
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
import random
import re
from typing import Any, Callable, Iterable, Sequence

import gymnasium as gym
import numpy as np


# Match GR00T's resume regex (rollout_policy.py:_load_existing_episode_metadata).
_VIDEO_NAME_RE = re.compile(
    r".*_env(?P<env>\d+)-episode_(?P<episode>\d+)-(?P<status>success|failure)"
)


def load_existing_episodes(
    video_dir: str | Path | None,
    n_envs: int,
    *,
    delete_orphans: bool = True,
) -> tuple[dict[int, int], list[bool]]:
    """Scan ``video_dir`` for already-recorded episode mp4s.

    Mirrors GR00T's ``_load_existing_episode_metadata``: any mp4 that doesn't
    match the ``_env(NN)-episode_(NN)-(success|failure)`` regex is treated as
    an orphan from a crashed prior run and (optionally) removed.

    Returns:
      env_to_max_ep:   per-env max episode index already on disk (defaults to -1)
      flat_successes:  list of success booleans across all recorded episodes,
                       ordered by (episode_idx, env_idx) -- matches GR00T.
    """
    env_to_max_ep: dict[int, int] = {i: -1 for i in range(n_envs)}
    if video_dir is None:
        return env_to_max_ep, []
    path = Path(video_dir)
    if not path.exists():
        return env_to_max_ep, []

    episodes: list[tuple[int, int, bool]] = []
    for f in path.glob("*.mp4"):
        m = _VIDEO_NAME_RE.match(f.stem)
        if not m:
            if delete_orphans:
                try:
                    f.unlink()
                except OSError:
                    pass
            continue
        env_idx = int(m.group("env"))
        ep_idx = int(m.group("episode"))
        if 0 <= env_idx < n_envs:
            env_to_max_ep[env_idx] = max(env_to_max_ep.get(env_idx, -1), ep_idx)
            episodes.append((ep_idx, env_idx, m.group("status") == "success"))

    episodes.sort(key=lambda x: (x[0], x[1]))
    flat_successes = [e[2] for e in episodes]
    return env_to_max_ep, flat_successes


# ---------------------------------------------------------------------
# Layout: the LeRobot dataset stores 44-dim state/action; the model
# slices to 29 active dims (arms+hands+waist). The env emits the active
# 29 dims spread across 5 per-part keys, so we zero-pad to 44 on input
# and split 29 across 5 parts on output.
# ---------------------------------------------------------------------
_RAW44_SLICES: dict[str, tuple[int, int]] = {
    "left_arm":   (0, 7),
    "left_hand":  (7, 13),
    "right_arm":  (22, 29),
    "right_hand": (29, 35),
    "waist":      (41, 44),
}

_ACTIVE29_SLICES: dict[str, tuple[int, int]] = {
    "left_arm":   (0, 7),
    "left_hand":  (7, 13),
    "right_arm":  (13, 20),
    "right_hand": (20, 26),
    "waist":      (26, 29),
}

STATE_DIM_RAW = 44
ACTION_DIM_ACTIVE = 29

EGO_IMAGE_KEY = "video.ego_view_pad_res256_freq20"
PROMPT_KEY = "annotation.human.coarse_action"


# ---------------------------------------------------------------------
# Env factory
# ---------------------------------------------------------------------
def make_env_fn(
    env_name: str,
    env_idx: int,
    *,
    seed: int,
    n_action_steps: int,
    max_episode_steps: int,
    video_dir: str | None,
    fps: int = 20,
    steps_per_render: int = 2,
    overlay_text: bool = True,
    terminate_on_success: bool = True,
    start_episode_id: int = -1,
) -> Callable[[], gym.Env]:
    """Return a thunk that builds a fully wrapped gr1_unified env.

    Suitable for ``gym.vector.AsyncVectorEnv(env_fns, context='spawn')``: every
    heavy import lives inside the closure so it runs in the spawned process.
    """

    env_seed = int(seed) + int(env_idx)

    def _build() -> gym.Env:
        import os as _os

        _os.environ.setdefault("MUJOCO_GL", "egl")
        _os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

        # registers all gr1_unified/* env ids
        import robocasa  # noqa: F401
        import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401

        # AsyncVectorEnv spawns this in a separate process; seed the
        # process-global RNGs so every env's reset is reproducible.
        random.seed(env_seed)
        np.random.seed(env_seed)

        env = gym.make(env_name, enable_render=True, seed=env_seed)

        if video_dir is not None:
            from gr00t.eval.sim.wrapper.video_recording_wrapper import (
                VideoRecorder,
                VideoRecordingWrapper,
            )

            recorder = VideoRecorder.create_h264(
                fps=fps,
                codec="h264",
                input_pix_fmt="rgb24",
                crf=22,
            )
            env = VideoRecordingWrapper(
                env,
                recorder,
                video_dir=Path(video_dir),
                steps_per_render=steps_per_render,
                max_episode_steps=max_episode_steps,
                overlay_text=overlay_text,
                name_prefix=f"{env_name.replace('/', '_')}_env{env_idx:02d}",
                base_seed=env_seed,
                seed_stride=100000,
                start_episode_id=start_episode_id,
            )

        from gr00t.eval.sim.wrapper.multistep_wrapper import MultiStepWrapper

        env = MultiStepWrapper(
            env,
            video_delta_indices=np.array([0]),
            state_delta_indices=np.array([0]),
            n_action_steps=n_action_steps,
            max_episode_steps=max_episode_steps,
            terminate_on_success=terminate_on_success,
        )
        return env

    return _build


def make_vector_env(
    env_name: str,
    *,
    n_envs: int,
    seed: int,
    n_action_steps: int,
    max_episode_steps: int,
    video_dir: str | None,
    fps: int = 20,
    steps_per_render: int = 2,
    overlay_text: bool = True,
    terminate_on_success: bool = True,
    start_episode_ids: Sequence[int] | None = None,
) -> gym.vector.VectorEnv:
    """Wrap ``n_envs`` factories into a (Sync|Async)VectorEnv.

    Matches GR00T's ``run_rollout_gymnasium_policy``: SyncVectorEnv when
    ``n_envs == 1`` (single in-process env, easier debugging), AsyncVectorEnv
    otherwise (each env in its own ``spawn`` subprocess).
    """
    if start_episode_ids is None:
        start_episode_ids = [-1] * n_envs
    if len(start_episode_ids) != n_envs:
        raise ValueError(
            f"start_episode_ids length {len(start_episode_ids)} != n_envs {n_envs}"
        )

    env_fns = [
        partial(
            make_env_fn(
                env_name,
                env_idx=i,
                seed=seed,
                n_action_steps=n_action_steps,
                max_episode_steps=max_episode_steps,
                video_dir=video_dir,
                fps=fps,
                steps_per_render=steps_per_render,
                overlay_text=overlay_text,
                terminate_on_success=terminate_on_success,
                start_episode_id=int(start_episode_ids[i]),
            )
        )
        for i in range(n_envs)
    ]

    if n_envs == 1:
        return gym.vector.SyncVectorEnv(env_fns)
    return gym.vector.AsyncVectorEnv(
        env_fns,
        shared_memory=False,
        context="spawn",
    )


# ---------------------------------------------------------------------
# Obs / action conversion (vector-aware)
# ---------------------------------------------------------------------
def _extract_per_env_state(obs_batch: dict, env_idx: int) -> np.ndarray:
    """Build the 44-dim raw GR-1 state for env ``env_idx`` from a batched obs."""
    state = np.zeros(STATE_DIM_RAW, dtype=np.float32)
    for part, (lo, hi) in _RAW44_SLICES.items():
        key = f"state.{part}"
        if key not in obs_batch:
            raise KeyError(f"missing env obs key '{key}'; got {list(obs_batch.keys())}")
        # obs_batch[key] shape: (n_envs, state_horizon=1, dim)  after MultiStepWrapper
        # If for any reason the horizon dim was squeezed, fall back to (n_envs, dim).
        v = np.asarray(obs_batch[key], dtype=np.float32)
        if v.ndim == 3:
            slice_ = v[env_idx, 0]
        elif v.ndim == 2:
            slice_ = v[env_idx]
        else:
            raise ValueError(
                f"unexpected shape for '{key}': {v.shape}, expected (n_envs, 1, dim) or (n_envs, dim)"
            )
        if slice_.shape[0] != hi - lo:
            raise ValueError(
                f"unexpected dim for '{key}': got {slice_.shape[0]}, expected {hi - lo}"
            )
        state[lo:hi] = slice_
    return state


def _extract_per_env_image(obs_batch: dict, env_idx: int) -> np.ndarray:
    if EGO_IMAGE_KEY not in obs_batch:
        raise KeyError(
            f"missing env obs key '{EGO_IMAGE_KEY}'; got {list(obs_batch.keys())}"
        )
    img = np.asarray(obs_batch[EGO_IMAGE_KEY])
    # After MultiStepWrapper(video_horizon=1) + VectorEnv: (n_envs, 1, 256, 256, 3)
    if img.ndim == 5:
        frame = img[env_idx, 0]
    elif img.ndim == 4:
        frame = img[env_idx]
    else:
        raise ValueError(
            f"unexpected ego image shape: {img.shape}, expected 4 or 5 dims"
        )
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    if frame.shape != (256, 256, 3):
        raise ValueError(
            f"unexpected per-env ego image shape: {frame.shape}, expected (256, 256, 3)"
        )
    return frame


def _extract_per_env_prompt(obs_batch: dict, env_idx: int) -> str:
    if PROMPT_KEY not in obs_batch:
        return ""
    val = obs_batch[PROMPT_KEY]
    if isinstance(val, (list, tuple, np.ndarray)):
        return str(val[env_idx])
    return str(val)


def pack_request(obs_batch: dict, env_idx: int) -> dict[str, Any]:
    """Build the openpi-format request for env ``env_idx`` from a batched obs."""
    return {
        "observation/image": _extract_per_env_image(obs_batch, env_idx),
        "observation/state": _extract_per_env_state(obs_batch, env_idx),
        "prompt": _extract_per_env_prompt(obs_batch, env_idx),
    }


def scatter_action_chunks(chunks: Iterable[np.ndarray]) -> dict[str, np.ndarray]:
    """Stack per-env (T, 29) action chunks into the env-side per-part action dict.

    Output shapes (suitable for VectorEnv.step input):
        action.left_arm   (n_envs, T, 7)
        action.left_hand  (n_envs, T, 6)
        action.right_arm  (n_envs, T, 7)
        action.right_hand (n_envs, T, 6)
        action.waist      (n_envs, T, 3)
    """
    chunks = [np.asarray(c, dtype=np.float32) for c in chunks]
    if not chunks:
        raise ValueError("scatter_action_chunks received empty chunks")
    T = chunks[0].shape[0]
    for i, c in enumerate(chunks):
        if c.shape != (T, ACTION_DIM_ACTIVE):
            raise ValueError(
                f"chunk[{i}] shape {c.shape} != expected ({T}, {ACTION_DIM_ACTIVE})"
            )

    stacked = np.stack(chunks, axis=0)  # (n_envs, T, 29)
    return {
        f"action.{part}": stacked[..., lo:hi].copy()
        for part, (lo, hi) in _ACTIVE29_SLICES.items()
    }


# ---------------------------------------------------------------------
# Success / final_info normalization (matches GR00T's checks)
# ---------------------------------------------------------------------
def _coerce_bool(x: Any) -> bool:
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return bool(x)
    if isinstance(x, (list, tuple, np.ndarray)):
        return bool(np.any(x))
    raise TypeError(f"unsupported success dtype: {type(x)} ({x!r})")


def extract_success(env_infos: dict, env_idx: int) -> bool:
    """Return True if env ``env_idx`` saw a success in this outer step.

    Handles both the in-progress case (``info["success"]`` is a list of
    n_action_steps booleans from MultiStepWrapper) and the terminal case
    (``info["final_info"][env_idx]["success"]`` from gymnasium's
    AsyncVectorEnv auto-reset path).
    """
    success = False
    if "success" in env_infos:
        try:
            v = env_infos["success"][env_idx]
            success = success or _coerce_bool(v)
        except (IndexError, KeyError):
            pass
    if "final_info" in env_infos:
        try:
            fi = env_infos["final_info"][env_idx]
        except (IndexError, KeyError):
            fi = None
        if fi is not None and isinstance(fi, dict) and "success" in fi:
            success = success or _coerce_bool(fi["success"])
    return success
