#!/usr/bin/env python3
"""Validate the merged datasets with the actual pi0 LeRobot reader, offline."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import tempfile

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE.parent / 'RoboTwin/policy/pi0/src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, default=BASE / 'data/merged')
    parser.add_argument('--datasets', nargs='+', default=[
        'robotwin_five_pi0_ee_full', 'robotwin_five_pi0_ee_subset20_seed42'])
    parser.add_argument('--output', type=Path, default=BASE /
                        'reports/five_task_subset20_lambda003_calibrated5_k32/loader_validation.json')
    parser.add_argument('--original-column-reader', action='store_true',
                        help='Also decode unrelated images on scalar column reads, as in the production adapter (slow).')
    args = parser.parse_args()
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    # Avoid image tensor operations oversubscribing the CPU thread pool.
    os.environ['OMP_NUM_THREADS'] = '2'
    with tempfile.TemporaryDirectory(prefix='robolora-five-loader-') as cache:
        os.environ['HF_DATASETS_CACHE'] = cache
        import torch
        torch.set_num_threads(2)
        from openpi.training.lerobot_compat import LeRobotDataset, _ListColumnView
        import numpy as np
        import pyarrow.parquet as pq
        class ProjectedColumns(_ListColumnView):
            # hf_transform_to_torch transforms each column independently. This
            # validation-only view skips unrelated images when querying numeric
            # columns; full observation rows still use the original image decoder.
            def __getitem__(self, key):
                if isinstance(key, str):
                    return list(self._dataset.select_columns([key])[key])
                return super().__getitem__(key)

            def select(self, *a, **kw):
                return ProjectedColumns(self._dataset.select(*a, **kw))

        class ValidationDataset(LeRobotDataset):
            def load_hf_dataset(self):
                view = super().load_hf_dataset()
                return view if args.original_column_reader else ProjectedColumns(view._dataset)

        records = []
        for name in args.datasets:
            path = args.dataset_root.resolve() / name
            info = json.loads((path / 'meta/info.json').read_text())
            mapping = [json.loads(s) for s in (path / 'meta/source_episodes.jsonl').read_text().splitlines()]
            ds = ValidationDataset(name, root=path, delta_timestamps={'action': [i / info['fps'] for i in range(50)]})
            print(f'{name}: loaded {len(ds)} frames', flush=True)
            assert len(ds) == info['total_frames']
            values = {'observation.state': [], 'action': []}
            for ep in mapping:
                numeric = pq.read_table(path / ep['merged_parquet'], columns=list(values))
                for key in values:
                    values[key].append(np.asarray(numeric[key].to_pylist(), dtype=np.float64))
                actions = values['action'][-1].astype(np.float32)
                for local in [0, ep['length'] - 1]:
                    row = ds[ep['frame_offset'] + local]
                    assert int(row['episode_index']) == ep['episode_index']
                    assert row['observation.state'].shape == (16,)
                    np.testing.assert_array_equal(row['action'].numpy(),
                                                  actions[np.minimum(np.arange(local, local + 50), ep['length'] - 1)])
                    assert row['task'] == ds.meta.tasks[int(row['task_index'])]
                    for key in info['features']:
                        if key.startswith('observation.images.'):
                            assert tuple(row[key].shape) == (3, 240, 320)
                            assert np.isfinite(row[key].numpy()).all()
                if (ep['episode_index'] + 1) % 50 == 0:
                    print(f'{name}: checked {ep["episode_index"] + 1} episodes', flush=True)
            for key in values:
                stats = ds.meta.stats[key]
                all_values = np.concatenate(values[key])
                np.testing.assert_allclose(stats['mean'], all_values.mean(0), atol=2e-6, rtol=1e-5)
                np.testing.assert_allclose(stats['std'], all_values.std(0), atol=2e-6, rtol=1e-5)
            record = dict(dataset=name, episodes=len(mapping), frames=len(ds),
                          boundary_samples_checked=2 * len(mapping), image_cameras=3, action_horizon=50,
                          validation_only_numeric_column_projection=not args.original_column_reader,
                          all_episode_boundary_chunks_correct=True, state_action_aggregate_stats_verified=True)
            print(record, flush=True)
            records.append(record)
            del ds
            gc.collect()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(records, indent=2) + '\n')


if __name__ == '__main__':
    main()
