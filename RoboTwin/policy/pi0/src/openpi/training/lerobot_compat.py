"""Local compatibility for pinned LeRobot and newer HF datasets Column objects.

LeRobot a445d9c calls torch.stack(dataset['column']) during loading and chunk
queries. datasets 5 returns Column instead of the list expected by that code.
Only this loader's HF dataset view is adapted; no global monkey patches.
"""

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset as _LeRobotDataset


class _ListColumnView:
    def __init__(self, dataset):
        self._dataset = dataset

    def __getitem__(self, key):
        result = self._dataset[key]
        return list(result) if isinstance(key, str) and not isinstance(result, list) else result

    def __len__(self):
        return len(self._dataset)

    def __getattr__(self, name):
        # During unpickling _dataset may not have been restored yet.
        dataset = self.__dict__.get("_dataset")
        if dataset is None:
            raise AttributeError(name)
        return getattr(dataset, name)

    def select(self, *args, **kwargs):
        return _ListColumnView(self._dataset.select(*args, **kwargs))


class LeRobotDataset(_LeRobotDataset):
    def load_hf_dataset(self):
        return _ListColumnView(super().load_hf_dataset())
