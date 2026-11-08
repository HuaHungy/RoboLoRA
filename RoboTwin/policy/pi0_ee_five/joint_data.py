"""Process-local mixed dataset adapter; source LeRobot datasets remain untouched."""

from dataclasses import replace
from time import monotonic

from common import JOINT_REPO, TASKS, repo_id


def install_joint_dataset():
    from openpi.training import data_loader
    from torch.utils.data import ConcatDataset

    if getattr(data_loader.create_dataset, '_ee_joint', False):
        return
    original = data_loader.create_dataset

    def create_dataset(data_config, model_config):
        if data_config.repo_id != JOINT_REPO:
            return original(data_config, model_config)
        # Resolve each dataset's LOCAL task_index into language before concatenating.
        # Each underlying LeRobot loader also clamps action windows to its episode.
        datasets = []
        for task in TASKS:
            started = monotonic()
            print(f'Loading joint source: {task}', flush=True)
            datasets.append(original(replace(data_config, repo_id=repo_id(task)), model_config))
            print(f'Loaded {task}: {len(datasets[-1])} frames in {monotonic() - started:.1f}s', flush=True)
        return ConcatDataset(datasets)

    create_dataset._ee_joint = True
    data_loader.create_dataset = create_dataset


def pooled_stats(parts):
    """Combine (count, mean, std) using population moments, not averaged stds."""
    import numpy as np
    from openpi.shared.normalize import NormStats

    total = sum(count for count, _, _ in parts)
    mean = sum(count * mean for count, mean, _ in parts) / total
    variance = sum(count * (std ** 2 + (part_mean - mean) ** 2)
                   for count, part_mean, std in parts) / total
    return NormStats(mean=mean, std=np.sqrt(np.maximum(variance, 0)))
