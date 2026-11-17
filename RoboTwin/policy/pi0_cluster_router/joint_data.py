"""Process-local adapter joining the five existing LeRobot EE datasets."""

from dataclasses import replace

from common import JOINT_REPO, TASKS


def source_repo(task):
    return f"{task}_pi0_ee"


def install_joint_dataset():
    from openpi.training import data_loader
    from torch.utils.data import ConcatDataset

    if getattr(data_loader.create_dataset, "_cluster_router_joint", False):
        return
    original = data_loader.create_dataset

    def create_dataset(data_config, model_config):
        if data_config.repo_id != JOINT_REPO:
            return original(data_config, model_config)
        datasets = [
            original(replace(data_config, repo_id=source_repo(task)), model_config)
            for task in TASKS
        ]
        return ConcatDataset(datasets)

    create_dataset._cluster_router_joint = True
    data_loader.create_dataset = create_dataset
