"""Validate cluster windows and create training-only 32D normalization assets."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from common import CLUSTERS, build_config, read_settings, repo_id, setup, write_json


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def moments(values, dim=32):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 16 or not np.isfinite(values).all():
        raise ValueError('Expected finite [N,16] EE values')
    padded = np.pad(values, ((0, 0), (0, dim - 16)))
    return padded.mean(0), padded.std(0)


def prepare_cluster(cluster, settings, check_loader=True):
    setup(settings)
    from openpi.shared import normalize
    root = Path(settings['dataset_root']) / repo_id(cluster)
    info = json.loads((root / 'meta/info.json').read_text())
    export_path = root / 'meta/cluster_export.json'
    export = json.loads(export_path.read_text())
    starts = jsonl(root / 'meta/sample_starts.jsonl')
    source_windows = jsonl(root / 'meta/source_windows.jsonl')
    if (export.get('status') != 'complete' or export.get('specification', {}).get('cluster_id') != cluster
            or export.get('specification', {}).get('scope') != 'global'):
        raise ValueError(f'Cluster {cluster}: invalid export manifest')
    if not (len(starts) == len(source_windows) == info['total_episodes']):
        raise ValueError(f'Cluster {cluster}: metadata row count mismatch')
    for episode, (start, source) in enumerate(zip(starts, source_windows, strict=True)):
        if (start.get('episode_index') != episode or start.get('index') != episode * 50
                or start.get('split') != source.get('split')
                or start.get('window_id') != source.get('window_id')):
            raise ValueError(f'Cluster {cluster}: sample start and source window mappings differ')
    if info['total_frames'] != 50 * info['total_episodes'] or info['fps'] <= 0:
        raise ValueError(f'Cluster {cluster}: expected 50 frames per exported episode')
    train = [row for row in source_windows if row['split'] == 'train']
    validation = [row for row in source_windows if row['split'] == 'validation']
    if len(train) != export['train_windows'] or len(validation) != export['validation_windows']:
        raise ValueError(f'Cluster {cluster}: split counts differ from export manifest')
    if {r['source_episode_id'] for r in train} & {r['source_episode_id'] for r in validation}:
        raise ValueError(f'Cluster {cluster}: source episode leakage across splits')
    if [r['episode_index'] for r in train + validation] != list(range(len(source_windows))):
        raise ValueError(f'Cluster {cluster}: train windows must precede validation windows')
    expected_cameras = {f'observation.images.{name}' for name in
                        ('cam_high', 'cam_left_wrist', 'cam_right_wrist')}
    states, actions = [], []
    fingerprint = hashlib.sha256(export_path.read_bytes())
    for i, row in enumerate(train):
        path = root / row['exported_parquet']
        parquet = pq.ParquetFile(path)
        if parquet.metadata.num_rows != 50 or not expected_cameras.issubset(parquet.schema_arrow.names):
            raise ValueError(f'Cluster {cluster}: bad shape/cameras: {path}')
        table = parquet.read(columns=['observation.state', 'action', 'frame_index', 'episode_index'])
        state = np.asarray(table['observation.state'].to_pylist(), dtype=np.float64)
        action = np.asarray(table['action'].to_pylist(), dtype=np.float64)
        if state.shape != (50, 16) or action.shape != (50, 16):
            raise ValueError(f'Cluster {cluster}: expected [50,16] state/action')
        np.testing.assert_allclose(action[:-1], state[1:], atol=1e-6, rtol=0)
        np.testing.assert_array_equal(table['frame_index'].to_numpy(), np.arange(50))
        np.testing.assert_array_equal(table['episode_index'].to_numpy(), np.full(50, i))
        for values in (state, action):
            if not np.isfinite(values).all() or np.any(values[:, [7, 15]] < 0) or np.any(values[:, [7, 15]] > 1):
                raise ValueError(f'Cluster {cluster}: invalid EE/gripper values')
            for offset in (3, 11):
                np.testing.assert_allclose(np.linalg.norm(values[:, offset:offset + 4], axis=1), 1, atol=1e-5)
        states.append(state[0]); actions.append(action)
        fingerprint.update(state[0].tobytes()); fingerprint.update(action.tobytes())
    states = np.asarray(states)
    actions = np.concatenate(actions)
    state_mean, state_std = moments(states)
    action_mean, action_std = moments(actions)
    stats = {'state': normalize.NormStats(mean=state_mean, std=state_std),
             'actions': normalize.NormStats(mean=action_mean, std=action_std)}
    cfg = build_config(cluster, settings, Path(settings['assets_root']) / '_preparation_run')
    destination = cfg.assets_dirs / repo_id(cluster)
    provenance = {
        'cluster_id': cluster, 'repo_id': repo_id(cluster), 'dataset_root': str(root),
        'index_sha256': export['specification']['index_sha256'],
        'labels_sha256': export['specification']['labels_sha256'],
        'data_sha256': fingerprint.hexdigest(), 'train_windows': len(train),
        'validation_windows': len(validation), 'state_count': len(states),
        'action_count': len(actions), 'action_horizon': 50,
        'normalization': 'training windows only: one start state and all 50 recorded actions; zero-pad 16D to 32D',
        'sample_unit': 'one frame-zero observation per exported episode',
    }
    norm_path = destination / 'norm_stats.json'
    provenance_path = destination / 'provenance.json'
    if norm_path.exists() or provenance_path.exists():
        if not norm_path.is_file() or not provenance_path.is_file() or json.loads(provenance_path.read_text()) != provenance:
            raise ValueError(f'Cluster {cluster}: existing normalization does not match current training data')
        saved = normalize.load(destination)
        for key in stats:
            np.testing.assert_allclose(saved[key].mean, stats[key].mean, atol=1e-12)
            np.testing.assert_allclose(saved[key].std, stats[key].std, atol=1e-12)
    else:
        normalize.save(destination, stats)
        write_json(provenance_path, provenance)
    if check_loader:
        from cluster_data import install_cluster_dataset
        from openpi.training import data_loader
        install_cluster_dataset('train')
        data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
        dataset = data_loader.create_dataset(data_config, cfg.model)
        if len(dataset) != len(train):
            raise ValueError(f'Cluster {cluster}: loader exposes frames instead of window starts')
        transformed = data_loader.transform_dataset(dataset, data_config)
        for index in (0, len(dataset) - 1):
            raw, item = dataset[index], transformed[index]
            if tuple(raw['action'].shape) != (50, 16) or raw['action_is_pad'].any():
                raise ValueError(f'Cluster {cluster}: padded action chunk reached training')
            if item['state'].shape != (32,) or item['actions'].shape != (50, 32):
                raise ValueError(f'Cluster {cluster}: transformed model dimensions are wrong')
            if not np.isfinite(item['state']).all() or not np.isfinite(item['actions']).all():
                raise ValueError(f'Cluster {cluster}: normalized model input is nonfinite')
            if len(item['image']) != 3 or not all(x.shape == (224, 224, 3) for x in item['image'].values()):
                raise ValueError(f'Cluster {cluster}: three resized cameras required')
            if not np.asarray(item['tokenized_prompt_mask']).any():
                raise ValueError(f'Cluster {cluster}: language prompt was not tokenized')
        batch_loader = data_loader.TorchDataLoader(
            transformed, local_batch_size=settings['batch_size'], num_batches=1,
            num_workers=0, shuffle=True, seed=settings['seed'])
        batch = next(iter(batch_loader))
        if batch['state'].shape != (settings['batch_size'], 32) or batch['actions'].shape != (
                settings['batch_size'], 50, 32):
            raise ValueError(f'Cluster {cluster}: batch dimensions are wrong')
        provenance['loader_check'] = {
            'dataset_length': len(dataset), 'batch_state': list(batch['state'].shape),
            'batch_actions': list(batch['actions'].shape), 'three_cameras': True,
            'language_tokenized': True, 'padded_actions': False,
        }
    provenance['configured_steps'] = cfg.num_train_steps
    provenance['batch_size'] = cfg.batch_size
    provenance['equivalent_passes'] = cfg.num_train_steps * cfg.batch_size / len(train)
    write_json(Path(settings['assets_root']) / 'reports' / f'cluster_{cluster:02d}_preparation.json', provenance)
    print(f"Cluster {cluster:02d} PASS: train={len(train)}, val={len(validation)}, "
          f"samples={len(train)}, steps={cfg.num_train_steps}, passes={provenance['equivalent_passes']:.1f}", flush=True)
    return provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--clusters', nargs='+', type=int, choices=CLUSTERS, required=True)
    parser.add_argument('--skip-loader-check', action='store_true')
    args = parser.parse_args()
    if len(set(args.clusters)) != len(args.clusters):
        parser.error('clusters must be unique')
    settings = read_settings(args.settings)
    reports = [prepare_cluster(k, settings, not args.skip_loader_check) for k in args.clusters]
    write_json(Path(settings['assets_root']) / 'reports/preparation_summary.json',
               {'training_started': False, 'clusters': reports})


if __name__ == '__main__':
    main()
