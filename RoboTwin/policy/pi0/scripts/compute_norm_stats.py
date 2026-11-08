"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import numpy as np
import torch
import tqdm
import tyro

import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):

    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_dataset(config: _config.TrainConfig, ) -> tuple[_config.DataConfig, _data_loader.Dataset]:
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_dataset(data_config, config.model)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    return data_config, dataset


def compute_statistics(dataset, max_frames: int | None = None, num_workers: int = 0, seed: int = 0):
    if max_frames is not None:
        if max_frames < 2:
            raise ValueError("max_frames must be at least 2")
        if max_frames < len(dataset):
            indices = np.random.default_rng(seed).choice(len(dataset), max_frames, replace=False)
            dataset = torch.utils.data.Subset(dataset, indices.tolist())
    if len(dataset) < 2:
        raise ValueError("Need at least two frames to compute normalization statistics")
    # A single finite pass, including the last partial batch. The previous
    # repeating training loader counted frames as batches and used batch[0],
    # so most samples never contributed to the statistics.
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
        collate_fn=_data_loader._collate_fn,
        multiprocessing_context="spawn" if num_workers else None,
    )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, desc="Computing stats"):
        for key in keys:
            values = np.asarray(batch[key], dtype=np.float64)
            stats[key].update(values.reshape(-1, values.shape[-1]))

    return {key: running.get_statistics() for key, running in stats.items()}


def main(config_name: str, max_frames: int | None = None, num_workers: int = 0):
    config = _config.get_config(config_name)
    data_config, dataset = create_dataset(config)
    norm_stats = compute_statistics(dataset, max_frames=max_frames, num_workers=num_workers, seed=config.seed)

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
