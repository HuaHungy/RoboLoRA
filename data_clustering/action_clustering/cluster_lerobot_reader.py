"""Module-level reader classes so cluster loaders can be pickled by workers."""
from openpi.training.lerobot_compat import LeRobotDataset, _ListColumnView


class ProjectedColumns(_ListColumnView):
    def __getitem__(self, key):
        if isinstance(key, str):
            # Pinned hf_transform_to_torch handles each column independently.
            return list(self._dataset.select_columns([key])[key])
        return super().__getitem__(key)

    def select(self, *args, **kwargs):
        return ProjectedColumns(self._dataset.select(*args, **kwargs))


class ClusterLeRobotReader(LeRobotDataset):
    def load_hf_dataset(self):
        return ProjectedColumns(super().load_hf_dataset()._dataset)
