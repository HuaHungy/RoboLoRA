"""Use exported cluster windows as one observation/action-chunk training sample each."""
import json
from pathlib import Path


def load_cluster_windows(root, split='train'):
    """Construct a local pi0-compatible reader; caller configures Python/cache paths."""
    from action_clustering.cluster_lerobot_reader import ClusterLeRobotReader

    root = Path(root).resolve()
    info = json.loads((root / 'meta/info.json').read_text())
    dataset = ClusterLeRobotReader(root.name, root=root,
                                  delta_timestamps={'action': [i / info['fps'] for i in range(50)]})
    return ClusterWindowDataset(dataset, root, split)


class ClusterWindowDataset:
    """Wrap an ALL-episodes LeRobot reader configured with action offsets 0..49.

    Select the split here, not via LeRobot's episodes parameter: indices in the
    start manifest refer to the complete exported dataset. Normalize using only
    training-window starts/actions; then apply the usual pi0 EE transforms.
    """

    def __init__(self, dataset, root, split='train'):
        if split not in ('train', 'validation'):
            raise ValueError('Expected train or validation')
        if dataset.episodes is not None:
            raise ValueError('Load all exported episodes; this adapter selects the split')
        if dataset.delta_indices is None or list(dataset.delta_indices.get('action', [])) != list(range(50)):
            raise ValueError('Configure LeRobot action delta_timestamps as [i / fps for i in range(50)]')
        root = Path(root).resolve()
        if Path(dataset.root).resolve() != root:
            raise ValueError('Dataset and sample manifest roots differ')
        self.dataset = dataset
        self.starts = [r for r in map(json.loads, (root / 'meta/sample_starts.jsonl').read_text().splitlines())
                       if r['split'] == split]

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        item = self.dataset[self.starts[index]['index']]
        if int(item['frame_index']) != 0 or tuple(item['action'].shape) != (50, 16):
            raise ValueError('Exported window/sample index mismatch')
        if item['action_is_pad'].any():
            raise ValueError('Cluster training sample must contain a complete recorded action chunk')
        return item
