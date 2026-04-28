"""Fast norm-stats computation for GR-1.

The stock ``compute_norm_stats.py`` runs the full data pipeline which decodes
every mp4 frame even though the statistics only need ``state``/``action`` from
the parquet columns. On a 6M-frame dataset that's ~10-15x more work than needed.

This script mirrors the same math (``openpi.shared.normalize.RunningStats``)
but monkey-patches ``LeRobotDatasetMetadata.video_keys`` to an empty list so
``LeRobotDataset.__getitem__`` skips video decoding entirely, then slices the
raw 44-dim state/action down to the 29-dim GR-1 subset via ``gr1_policy``.

Output path matches the stock script: ``config.assets_dirs / repo_id``.
"""

import numpy as np
import torch
import torch.utils.data as torch_data
import tqdm
import tyro

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

import openpi.policies.gr1_policy as gr1_policy
import openpi.shared.normalize as normalize
import openpi.training.config as _config


def _collate(batch: list[dict]) -> dict:
    return {
        "state": torch.stack([b["observation.state"] for b in batch]).numpy(),
        "action": torch.stack([b["action"] for b in batch]).numpy(),
    }


def main(
    config_name: str = "pi05_gr1",
    max_frames: int | None = 200_000,
    num_workers: int = 16,
    batch_size: int = 2048,
    seed: int = 0,
):
    # Skip all mp4 decoding by making the metadata report zero video keys. This is
    # safe because LeRobotDataset.__getitem__ gates video loading behind
    # ``if len(self.meta.video_keys) > 0:``.
    LeRobotDatasetMetadata.video_keys = property(lambda self: [])

    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    action_horizon = config.model.action_horizon

    meta = LeRobotDatasetMetadata(data_config.repo_id)
    dataset = LeRobotDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [t / meta.fps for t in range(action_horizon)]
            for key in data_config.action_sequence_keys
        },
    )

    if max_frames is not None and max_frames < len(dataset):
        gen = torch.Generator().manual_seed(seed)
        sampler = torch_data.RandomSampler(dataset, num_samples=max_frames, replacement=False, generator=gen)
        num_batches = max_frames // batch_size
    else:
        sampler = None
        num_batches = len(dataset) // batch_size

    loader = torch_data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=_collate,
        persistent_workers=num_workers > 0,
        pin_memory=False,
    )

    stats = {"state": normalize.RunningStats(), "actions": normalize.RunningStats()}

    for batch in tqdm.tqdm(loader, total=num_batches, desc="norm_stats (no video)"):
        stats["state"].update(gr1_policy._slice_gr1_29(batch["state"]))
        stats["actions"].update(gr1_policy._slice_gr1_29(batch["action"]))

    norm_stats = {k: s.get_statistics() for k, s in stats.items()}

    out = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {out}")
    normalize.save(out, norm_stats)

    print("\nSummary:")
    for k, s in norm_stats.items():
        print(f"  {k}: mean[0:5]={np.round(s.mean[:5], 3)} std[0:5]={np.round(s.std[:5], 3)}")
        print(f"         q01[0:5]={np.round(s.q01[:5], 3)} q99[0:5]={np.round(s.q99[:5], 3)}")


if __name__ == "__main__":
    tyro.cli(main)
