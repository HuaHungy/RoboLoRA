"""Validate all numeric EE data and prepare exact action-chunk normalization on CPU."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from common import HERE, JOINT, JOINT_REPO, TASKS, build_config, read_settings, repo_id, setup, write_json


def action_weights(length, horizon):
    weights = np.minimum(np.arange(1, length + 1, dtype=np.int64), horizon)
    weights[-1] += length * horizon - int(weights.sum())
    return weights


def prepare_task(task, settings, check_loader=True):
    setup(settings)
    from openpi.shared import normalize
    cfg = build_config(task, settings, HERE / 'runs/_preparation_only')
    root = Path(settings['dataset_root']) / f'{task}_pi0_ee'
    manifest_path = root / 'meta/robotwin_ee_conversion.json'
    manifest = json.loads(manifest_path.read_text())
    info = json.loads((root / 'meta/info.json').read_text())
    if (manifest.get('status') != 'complete' or manifest.get('quaternion_order') != 'wxyz'
            or manifest.get('representation') != 'absolute_world_ee_pose'):
        raise ValueError(f'{task}: not a complete absolute EE dataset')
    files = sorted(root.glob('data/chunk-*/*.parquet'))
    if len(files) != info['total_episodes'] or len(files) != 50:
        raise ValueError(f'{task}: expected the complete 50-episode dataset')
    sums = {key: np.zeros(32, np.float64) for key in ('state', 'actions')}
    squares = {key: np.zeros(32, np.float64) for key in sums}
    counts = {key: 0 for key in sums}
    fingerprint = hashlib.sha256(manifest_path.read_bytes())
    total = 0
    for path in files:
        parquet = pq.ParquetFile(path)
        expected = ['observation.images.' + camera for camera in ('cam_high', 'cam_left_wrist', 'cam_right_wrist')]
        if not set(expected).issubset(parquet.schema_arrow.names):
            raise ValueError(f'{path}: three camera columns are required')
        table = parquet.read(columns=['observation.state', 'action'])
        state = np.asarray(table['observation.state'].to_pylist(), dtype=np.float64)
        action = np.asarray(table['action'].to_pylist(), dtype=np.float64)
        if state.ndim != 2 or state.shape[1] != 16 or state.shape != action.shape or len(state) == 0:
            raise ValueError(f'{path}: expected nonempty [T,16] state/action')
        np.testing.assert_allclose(action[:-1], state[1:], atol=1e-6, rtol=0)
        for values in (state, action):
            if not np.isfinite(values).all() or np.any(values[:, [7, 15]] < 0) or np.any(values[:, [7, 15]] > 1):
                raise ValueError(f'{path}: non-finite data or invalid grippers')
            for offset in (3, 11):
                np.testing.assert_allclose(np.linalg.norm(values[:, offset:offset + 4], axis=1), 1, atol=1e-5)
        for key, values, weights in [('state', state, np.ones(len(state))),
                                     ('actions', action, action_weights(len(action), settings['action_horizon']))]:
            sums[key][:16] += (values * weights[:, None]).sum(axis=0)
            squares[key][:16] += (values ** 2 * weights[:, None]).sum(axis=0)
            counts[key] += int(weights.sum())
        fingerprint.update(state.tobytes()); fingerprint.update(action.tobytes())
        total += len(state)
    if total != info['total_frames']:
        raise ValueError(f'{task}: frame total differs from metadata')
    stats = {}
    for key in sums:
        mean = sums[key] / counts[key]
        std = np.sqrt(np.maximum(0, squares[key] / counts[key] - mean ** 2))
        stats[key] = normalize.NormStats(mean=mean, std=std)
    destination = cfg.assets_dirs / f'{task}_pi0_ee'
    norm_path = destination / 'norm_stats.json'
    provenance = {'task': task, 'dataset_root': str(root), 'episodes': len(files), 'frames': total,
                  'data_sha256': fingerprint.hexdigest(), 'action_horizon': settings['action_horizon'],
                  'normalization': 'exact z-score over all frames; action horizon with repeat-last episode padding',
                  'state_count': counts['state'], 'action_count': counts['actions'],
                  'coordinate_frame': 'world; get_left/right_ee_pose; not TCP', 'quaternion_order': 'wxyz'}
    if norm_path.exists():
        saved = normalize.load(destination)
        previous = json.loads((destination / 'provenance.json').read_text())
        if previous['data_sha256'] != provenance['data_sha256'] or previous['action_horizon'] != settings['action_horizon']:
            raise ValueError(f'{task}: normalization source changed; preserve the old assets before preparing a new experiment')
        for key in stats:
            np.testing.assert_allclose(saved[key].mean, stats[key].mean, atol=1e-10)
            np.testing.assert_allclose(saved[key].std, stats[key].std, atol=1e-10)
    else:
        normalize.save(destination, stats)
        write_json(destination / 'provenance.json', provenance)
    if check_loader:
        from openpi.training import data_loader
        loader = data_loader.create_data_loader(cfg, num_batches=1, num_workers=0, shuffle=False)
        observation, actions = next(iter(loader))
        assert observation.state.shape == (settings['batch_size'], 32)
        assert actions.shape == (settings['batch_size'], settings['action_horizon'], 32)
        assert all(image.shape == (settings['batch_size'], 224, 224, 3) for image in observation.images.values())
        assert all(np.asarray(mask).all() for mask in observation.image_masks.values())
        assert np.asarray(observation.tokenized_prompt_mask).any()
        assert np.isfinite(np.asarray(actions)).all()
        provenance['loader_check'] = {'state': list(observation.state.shape), 'actions': list(actions.shape),
                                      'three_camera_images': [settings['batch_size'], 224, 224, 3], 'language_tokenized': True}
    write_json(HERE / 'reports' / f'{task}_data_check.json', provenance)
    print(f"{task}: {len(files)} episodes, {total} frames; numeric EE checks PASS; loader checked={check_loader}", flush=True)
    return provenance


def prepare_joint(settings):
    from openpi.shared import normalize
    from openpi.training import data_loader
    from joint_data import install_joint_dataset, pooled_stats

    reports = [prepare_task(task, settings, check_loader=False) for task in TASKS]
    cfg = build_config(JOINT, settings, HERE / 'runs/_preparation_only')
    sources = [normalize.load(build_config(task, settings, HERE / 'runs/_preparation_only').assets_dirs / repo_id(task))
               for task in TASKS]
    stats = {key: pooled_stats([(report[count_key], source[key].mean, source[key].std)
                               for report, source in zip(reports, sources, strict=True)])
             for key, count_key in [('state', 'state_count'), ('actions', 'action_count')]}
    destination = cfg.assets_dirs / JOINT_REPO
    provenance = {'training_mode': 'joint', 'tasks': list(TASKS),
                  'episodes': sum(r['episodes'] for r in reports), 'frames': sum(r['frames'] for r in reports),
                  'action_horizon': settings['action_horizon'],
                  'sampling': 'uniform shuffled frames across concatenated datasets',
                  'normalization': 'shared population z-score; exact repeat-last action-window weighting',
                  'sources': [{k: r[k] for k in ('task', 'frames', 'data_sha256', 'state_count', 'action_count')}
                              for r in reports]}
    if (destination / 'norm_stats.json').exists():
        if json.loads((destination / 'provenance.json').read_text()) != provenance:
            raise ValueError('Joint normalization source changed; preserve old assets before starting a new experiment')
        saved = normalize.load(destination)
        for key in stats:
            np.testing.assert_allclose(saved[key].mean, stats[key].mean, atol=1e-10)
            np.testing.assert_allclose(saved[key].std, stats[key].std, atol=1e-10)
    else:
        normalize.save(destination, stats)
        write_json(destination / 'provenance.json', provenance)

    install_joint_dataset()
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    dataset = data_loader.create_dataset(data_config, cfg.model)
    assert len(dataset) == provenance['frames']
    transformed = data_loader.transform_dataset(dataset, data_config)
    offset, checks, prompts = 0, [], []
    for task, source in zip(TASKS, dataset.datasets, strict=True):
        first, last = source[0], source[len(source) - 1]
        prompt = str(first['prompt'])
        assert prompt.strip(), f'{task}: empty language instruction'
        prompts.append(prompt)
        # At the final frame the entire chunk must repeat that dataset's final action.
        last_actions = np.asarray(last['action'])
        assert last_actions.shape == (settings['action_horizon'], 16)
        np.testing.assert_allclose(last_actions, np.repeat(last_actions[:1], settings['action_horizon'], axis=0))
        token_ids = None
        for index in (offset, offset + len(source) - 1):
            value = transformed[index]
            assert value['state'].shape == (32,)
            assert value['actions'].shape == (settings['action_horizon'], 32)
            assert np.isfinite(value['state']).all() and np.isfinite(value['actions']).all()
            assert len(value['image']) == 3
            assert all(img.shape == (224, 224, 3) for img in value['image'].values())
            assert np.asarray(value['tokenized_prompt_mask']).any()
            if index == offset:
                token_ids = np.asarray(value['tokenized_prompt']).tolist()
        checks.append({'task': task, 'frames': len(source), 'first_prompt': prompt,
                       'first_token_ids': token_ids, 'last_action_padding_checked': True})
        offset += len(source)
    assert len(set(prompts)) == len(TASKS), 'Task language mapping unexpectedly collapsed'
    assert len({tuple(c['first_token_ids']) for c in checks}) == len(TASKS)
    # Exercise the same batching/sharding used by create_data_loader, reusing the
    # already checked sources rather than opening all five image datasets twice.
    from openpi.models.model import Observation
    loader = data_loader.TorchDataLoader(transformed, local_batch_size=settings['batch_size'],
                                        num_batches=1, num_workers=0, shuffle=True, seed=settings['seed'])
    batch = next(iter(loader))
    observation, actions = Observation.from_dict(batch), batch['actions']
    assert observation.state.shape == (settings['batch_size'], 32)
    assert actions.shape == (settings['batch_size'], settings['action_horizon'], 32)
    assert np.isfinite(np.asarray(actions)).all()
    report = {**provenance, 'loader_checks': checks, 'batch_size': settings['batch_size'],
              'optimizer_steps': settings['num_train_steps'],
              'equivalent_dataset_passes': settings['num_train_steps'] * settings['batch_size'] / provenance['frames'],
              'training_started': False}
    write_json(HERE / 'reports/joint_preparation.json', report)
    print(f"Joint PASS: {provenance['episodes']} episodes, {provenance['frames']} frames; "
          f"5 task prompts, 3 cameras, shared normalization; batch={settings['batch_size']}", flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', nargs='+', choices=TASKS, default=list(TASKS))
    parser.add_argument('--settings', type=Path)
    parser.add_argument('--batch-size', type=int)
    args = parser.parse_args()
    settings = read_settings(args.settings)
    if args.batch_size is not None:
        if args.batch_size <= 0:
            parser.error('batch-size must be positive')
        settings['batch_size'] = args.batch_size
    setup(settings)
    if not (Path(settings['pretrained_params']) / '_METADATA').is_file():
        raise FileNotFoundError('The local pi0_base parameter checkpoint is missing')
    if settings.get('training_mode') == 'joint':
        prepare_joint(settings)
    else:
        reports = [prepare_task(task, settings) for task in args.tasks]
        write_json(HERE / 'reports/preparation.json', {'training_started': False, 'tasks': reports})
