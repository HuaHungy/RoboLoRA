#!/usr/bin/env python3
"""Check all exported Parquets and sampled pi0 window reads against original data."""
import argparse
import gc
import json
import os
import pickle
from pathlib import Path
import sys
import tempfile

BASE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BASE), str(BASE.parent / 'RoboTwin/policy/pi0/src')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=BASE / 'data/cluster_exports/lambda003_calibrated5_k32_global_smoke')
    args = parser.parse_args()
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    with tempfile.TemporaryDirectory(prefix='cluster-export-validation-') as cache:
        os.environ['HF_DATASETS_CACHE'] = cache
        import numpy as np
        import pyarrow.parquet as pq
        import torch
        torch.set_num_threads(2)
        from action_clustering.cluster_window_dataset import load_cluster_windows, ClusterWindowDataset
        from extract_cluster_lerobot import feature_stats
        from prepare_five_task_subset import read, rows, write
        reports = []
        for root in sorted(args.root.glob('cluster_*_pi0_ee')):
            info = read(root / 'meta/info.json')
            provenance = rows(root / 'meta/source_windows.jsonl')
            start_rows = rows(root / 'meta/sample_starts.jsonl')
            assert len(start_rows) == len(provenance) == info['total_episodes']
            languages = {r['task_index']: r['task'] for r in rows(root / 'meta/tasks.jsonl')}
            train_states, train_actions = [], []
            split_episodes = {'train': set(), 'validation': set()}
            for ep, p in enumerate(provenance):
                table = pq.read_table(root / p['exported_parquet'])
                source = pq.read_table(p['source_parquet']).slice(p['start'], 50)
                assert len(table) == 50
                for key in table.column_names:
                    if key == 'action' or key.startswith('observation.'):
                        assert table[key].equals(source[key]), (root, ep, key)
                np.testing.assert_array_equal(table['index'].to_numpy(), np.arange(ep * 50, (ep + 1) * 50))
                np.testing.assert_array_equal(table['frame_index'].to_numpy(), np.arange(50))
                np.testing.assert_array_equal(table['episode_index'].to_numpy(), np.full(50, ep))
                np.testing.assert_allclose(table['timestamp'].to_numpy(), np.arange(50) / info['fps'], atol=1e-6)
                assert {languages[int(i)] for i in table['task_index'].to_numpy()} == {p['instruction']}
                assert start_rows[ep]['index'] == ep * 50 and start_rows[ep]['split'] == p['split']
                split_episodes[p['split']].add(p['source_episode_id'])
                if p['split'] == 'train':
                    train_states.append(table['observation.state'][0].as_py())
                    train_actions.extend(table['action'].to_pylist())
            assert not split_episodes['train'] & split_episodes['validation']
            stats = read(root / 'meta/train_window_stats.json')
            for key, values in [('state', train_states), ('actions', train_actions)]:
                expected = feature_stats(values)
                for field in expected:
                    np.testing.assert_allclose(stats[key][field], expected[field], atol=1e-12)
            ds = load_cluster_windows(root, 'train')
            # Spawn-based training workers serialize the reader and its views.
            ds = pickle.loads(pickle.dumps(ds))
            checked = 0
            for split in ('train', 'validation'):
                view = ClusterWindowDataset(ds.dataset, root, split)
                for i in sorted({0, len(view) - 1} if len(view) else set()):
                    sample = view[i]
                    ep = view.starts[i]['episode_index']
                    p = provenance[ep]
                    original = pq.read_table(p['source_parquet'], columns=['action']).slice(p['start'], 50)
                    np.testing.assert_array_equal(sample['action'].numpy(), np.asarray(original['action'].to_pylist(), np.float32))
                    assert sample['task'] == p['instruction']
                    for key in info['features']:
                        if key.startswith('observation.images.'):
                            assert tuple(sample[key].shape) == (3, 240, 320)
                    checked += 1
            reports.append(dict(dataset=root.name, windows=len(provenance),
                                all_preserved_columns_equal_source=True, split_disjoint_by_source_episode=True,
                                training_only_stats_correct=True, sampled_loader_checks=checked,
                                full_50_step_actions_without_padding=True))
            print(reports[-1], flush=True)
            del ds, view
            gc.collect()
        if not reports:
            raise ValueError('No completed cluster datasets found')
        write(args.root / 'validation.json', reports)


if __name__ == '__main__':
    main()
