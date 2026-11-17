"""Process-local start-only sampling for exported 50-step cluster windows."""

import json
from pathlib import Path

from common import CLUSTERS, repo_id
from openpi.training.lerobot_compat import LeRobotDataset as BaseReader, _ListColumnView


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


class ProjectedColumns(_ListColumnView):
    def __getitem__(self, key):
        if isinstance(key, str):
            return list(self._dataset.select_columns([key])[key])
        return super().__getitem__(key)

    def select(self, *args, **kwargs):
        return ProjectedColumns(self._dataset.select(*args, **kwargs))


class ClusterReader(BaseReader):
    def load_hf_dataset(self):
        return ProjectedColumns(super().load_hf_dataset()._dataset)


class ClusterStartDataset:
    """Expose one sample at frame zero of every selected 50-step episode."""

    def __init__(self, dataset, starts, split):
        self.dataset = dataset
        self.starts = [row for row in starts if row['split'] == split]
        if dataset.episodes != [row['episode_index'] for row in self.starts]:
            raise ValueError('LeRobot selected episodes and cluster start manifest differ')
        if not self.starts:
            raise ValueError(f'No {split} cluster windows')

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        item = self.dataset[index * 50]
        expected = self.starts[index]
        if (int(item['episode_index']) != expected['episode_index'] or int(item['frame_index']) != 0
                or tuple(item['action'].shape) != (50, 16)):
            raise ValueError('Cluster start manifest and LeRobot data disagree')
        if item['action_is_pad'].any():
            raise ValueError('Cluster start must contain 50 recorded, unpadded actions')
        return item


def install_cluster_dataset(split='train'):
    """Patch only configured cluster repo ids in this process."""
    from openpi.training import data_loader
    from openpi import transforms
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

    if split not in ('train', 'validation'):
        raise ValueError('split must be train or validation')
    if getattr(data_loader.create_dataset, '_cluster_start_only', None):
        if data_loader.create_dataset._cluster_start_only != split:
            raise ValueError('Cluster dataset patch already installed for a different split')
        return

    original = data_loader.create_dataset
    allowed = {repo_id(cluster) for cluster in CLUSTERS}

    def create_dataset(data_config, model_config):
        if data_config.repo_id not in allowed:
            return original(data_config, model_config)
        meta = LeRobotDatasetMetadata(data_config.repo_id)
        root = meta.root
        manifest = json.loads((root / 'meta/cluster_export.json').read_text())
        if (manifest.get('status') != 'complete' or manifest.get('specification', {}).get('cluster_id') not in CLUSTERS
                or manifest.get('specification', {}).get('scope') != 'global'):
            raise ValueError(f'Not a completed global cluster export: {root}')
        starts = jsonl(root / 'meta/sample_starts.jsonl')
        selected = [row['episode_index'] for row in starts if row['split'] == split]
        reader = ClusterReader(
            data_config.repo_id, episodes=selected,
            delta_timestamps={key: [t / meta.fps for t in range(model_config.action_horizon)]
                              for key in data_config.action_sequence_keys},
        )
        dataset = ClusterStartDataset(reader, starts, split)
        if data_config.prompt_from_task:
            dataset = data_loader.TransformedDataset(dataset, [transforms.PromptFromLeRobotTask(meta.tasks)])
        return dataset

    create_dataset._cluster_start_only = split
    data_loader.create_dataset = create_dataset
