#!/usr/bin/env python3
"""Export frozen cluster windows as separate LeRobot EE datasets, without training."""
import argparse
from collections import Counter
import copy
from functools import lru_cache
import io
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from prepare_five_task_subset import BASE, DEFAULT_RUN, VARIANT, digest, read, rows, write, write_rows
sys.path.insert(0, str(BASE))
from action_clustering.lerobot import read_columns

CLUSTERS = [8, 24, 3, 6, 9, 31, 23]


def feature_stats(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    return {**{key: fn(values, axis=0).tolist() for key, fn in
               [('min', np.min), ('max', np.max), ('mean', np.mean), ('std', np.std)]},
            'count': [len(values)]}


@lru_cache(maxsize=1)
def load_source(path):
    path = Path(path)
    read_columns(path)  # Validate finite EE, next-state alignment, quaternions and grippers.
    table = pq.read_table(path)
    image_moments = {}
    for key in table.column_names:
        if not key.startswith('observation.images.'):
            continue
        moments = []
        for cell in table[key].to_pylist():
            if not cell['bytes']:
                raise ValueError(f'Missing embedded image: {path}')
            with Image.open(io.BytesIO(cell['bytes'])) as img:
                # Same spatial subsampling as pinned LeRobot statistics (320px -> /2).
                rgb = np.asarray(img.convert('RGB'), dtype=np.float64)[::2, ::2] / 255.
            moments.append([rgb.mean((0, 1)), rgb.var((0, 1)), rgb.min((0, 1)), rgb.max((0, 1))])
        image_moments[key] = np.asarray(moments)
    return table, image_moments, digest(path)


def slice_stats(table, moments, start, end):
    stats = {}
    for key in table.column_names:
        if key in moments:
            m = moments[key][start:end]
            mean = m[:, 0].mean(0)
            std = np.sqrt(np.maximum(0, (m[:, 1] + (m[:, 0] - mean)**2).mean(0)))
            stats[key] = {name: np.asarray(value).reshape(3, 1, 1).tolist() for name, value in
                          [('mean', mean), ('std', std), ('min', m[:, 2].min(0)), ('max', m[:, 3].max(0))]}
            stats[key]['count'] = [end - start]
        else:
            stats[key] = feature_stats(table[key].to_pylist())
    return stats


def source_path(row, root):
    relative = row['episode_id'].split('/', 1)[1]
    return root / f"{row['dataset_id']}_pi0_ee" / relative


def select_windows(args):
    run = args.run.resolve()
    index = rows(run / 'index.jsonl')
    cluster_dir = run / 'evaluation' / VARIANT / 'k32'
    labels = np.load(cluster_dir / 'labels.npy')
    assigned = rows(cluster_dir / 'assignments.jsonl')
    if labels.shape != (len(index),) or len(assigned) != len(index):
        raise ValueError('Clustering index/labels/assignments mismatch')
    splits, ids = {}, set()
    for row, label, original in zip(index, labels, assigned):
        if original != {**row, 'cluster_id': int(label)} or row['window_id'] in ids:
            raise ValueError('Clustering provenance mismatch or duplicate window')
        ids.add(row['window_id'])
        if row['end'] - row['start'] != 50 or row['split'] not in ('train', 'validation'):
            raise ValueError('Expected 50-step windows and original train/validation splits')
        previous = splits.setdefault(row['episode_id'], row['split'])
        if previous != row['split']:
            raise ValueError('Source episode occurs in both train and validation')
    allowed = None
    if args.scope != 'global':
        name = 'robotwin_five_pi0_ee_full' if args.scope == 'five-full' else 'robotwin_five_pi0_ee_subset20_seed42'
        allowed = {r['source_episode_id'] for r in rows(args.merged_root / name / 'meta/source_episodes.jsonl')}
    selected = {}
    for k in args.clusters:
        windows = [{**r, 'cluster_id': k} for r, label in zip(index, labels)
                   if label == k and (allowed is None or r['episode_id'] in allowed)]
        windows.sort(key=lambda r: (r['split'] != 'train', r['episode_id'], r['start']))
        if args.smoke_test:
            windows = [r for split in ('train', 'validation') for r in
                       [next((w for w in windows if w['split'] == split), None)] if r is not None]
        if not windows or not any(w['split'] == 'train' for w in windows):
            raise ValueError(f'No training windows in cluster {k}')
        selected[k] = windows
    return selected, dict(index_sha256=digest(run / 'index.jsonl'),
                          labels_sha256=digest(cluster_dir / 'labels.npy'),
                          run=str(run), variant=VARIANT, k=32, scope=args.scope, smoke_test=args.smoke_test)


def export_cluster(k, windows, destination, source_root, provenance):
    spec = {**provenance, 'cluster_id': k, 'source_root': str(source_root),
            'window_ids': [w['window_id'] for w in windows], 'format_version': 1}
    if destination.exists():
        manifest = read(destination / 'meta/cluster_export.json')
        if manifest['status'] != 'complete' or manifest['specification'] != spec:
            raise ValueError(f'Refusing to overwrite incompatible output: {destination}')
        for row in rows(destination / 'meta/source_windows.jsonl'):
            if digest(destination / row['exported_parquet']) != row['exported_sha256']:
                raise ValueError(f'Existing output was modified: {destination}')
        print(f'Cluster {k}: verified and reused {destination}', flush=True)
        return manifest
    staging = destination.with_name(destination.name + '.partial')
    if staging.exists():
        if read(staging / 'request.json') != spec:
            raise ValueError(f'Incompatible partial output: {staging}')
        # Only restart a matching, script-owned partial directory, never final data.
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    write(staging / 'request.json', spec)
    first_root = source_root / f"{windows[0]['dataset_id']}_pi0_ee"
    info = copy.deepcopy(read(first_root / 'meta/info.json'))
    conversion = copy.deepcopy(read(first_root / 'meta/robotwin_ee_conversion.json'))
    episode_rows, stat_rows, mappings, native_rows, starts = [], [], [], [], []
    task_ids, source_meta = {}, {}
    train_states, train_actions = [], []
    started = time.monotonic()
    for ep, row in enumerate(windows):
        task = row['dataset_id']
        path = source_path(row, source_root)
        root = source_root / f'{task}_pi0_ee'
        if task not in source_meta:
            source_info, source_conversion = read(root / 'meta/info.json'), read(root / 'meta/robotwin_ee_conversion.json')
            for key in ('features', 'fps', 'codebase_version', 'robot_type'):
                if info[key] != source_info[key]:
                    raise ValueError(f'Inconsistent {key}: {root}')
            for key in ('representation', 'quaternion_order', 'status'):
                if source_conversion[key] != conversion[key]:
                    raise ValueError(f'Inconsistent pose convention: {root}')
            if source_conversion['status'] != 'complete' or source_info['total_videos']:
                raise ValueError(f'Incomplete or unsupported source: {root}')
            source_meta[task] = ({r['task_index']: r['task'] for r in rows(root / 'meta/tasks.jsonl')}, source_conversion)
        language, source_conversion = source_meta[task]
        full, moments, source_hash = load_source(str(path))
        start, end = row['start'], row['end']
        if not 0 <= start < end <= len(full):
            raise ValueError(f'Window outside source: {row}')
        if not np.all(full['episode_index'].to_numpy() == row['lerobot_episode_index']):
            raise ValueError('Source episode index mismatch')
        table = full.slice(start, 50)
        instructions = [language[int(i)] for i in table['task_index'].to_numpy()]
        if set(instructions) != {row['instruction']}:
            raise ValueError('Source instruction differs from clustering index')
        text = instructions[0]
        task_ids.setdefault(text, len(task_ids))
        replacements = dict(episode_index=np.full(50, ep), index=np.arange(ep * 50, (ep + 1) * 50),
                            frame_index=np.arange(50), timestamp=np.arange(50) / info['fps'],
                            task_index=np.full(50, task_ids[text]))
        for key, values in replacements.items():
            field = table.schema.field(key)
            table = table.set_column(table.schema.get_field_index(key), field, pa.array(values, type=field.type))
        relative = info['data_path'].format(episode_chunk=ep // info['chunks_size'], episode_index=ep)
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, target, compression='snappy')
        if not pq.read_table(target).equals(table):
            raise ValueError('Parquet roundtrip differs')
        episode_rows.append(dict(episode_index=ep, tasks=[text], length=50))
        stat_rows.append(dict(episode_index=ep, stats=slice_stats(table, moments, start, end)))
        mappings.append({**row, 'source_episode_id': row['episode_id'], 'source_parquet': str(path),
                         'source_sha256': source_hash, 'episode_index': ep, 'length': 50, 'frame_offset': ep * 50,
                         'exported_parquet': relative, 'exported_sha256': digest(target)})
        starts.append(dict(episode_index=ep, index=ep * 50, split=row['split'], window_id=row['window_id']))
        native = source_conversion['sources'][row['lerobot_episode_index']]
        native_rows.append({**native, 'episode_index': ep, 'frames': 50, 'source_frames': 51,
                            'original_source_frames': native['source_frames'], 'source_frame_start': start,
                            'source_frame_end_exclusive': end + 1, 'original_lerobot_episode_index': row['lerobot_episode_index']})
        if row['split'] == 'train':
            train_states.append(np.asarray(table['observation.state'][0].as_py()))
            train_actions.append(np.asarray(table['action'].to_pylist()))
        if (ep + 1) % 50 == 0 or ep == len(windows) - 1:
            elapsed = time.monotonic() - started
            eta = elapsed / (ep + 1) * (len(windows) - ep - 1)
            print(f'Cluster {k}: {ep+1}/{len(windows)} windows; elapsed {elapsed/60:.1f} min; ETA {eta/60:.1f} min', flush=True)
    ntrain = sum(w['split'] == 'train' for w in windows)
    info.update(total_episodes=len(windows), total_frames=len(windows) * 50, total_tasks=len(task_ids),
                total_chunks=(len(windows) + info['chunks_size'] - 1) // info['chunks_size'],
                splits={'train': f'0:{ntrain}', 'validation': f'{ntrain}:{len(windows)}'})
    write(staging / 'meta/info.json', info)
    write_rows(staging / 'meta/episodes.jsonl', episode_rows)
    write_rows(staging / 'meta/episodes_stats.jsonl', stat_rows)
    write_rows(staging / 'meta/tasks.jsonl', [dict(task_index=i, task=t) for t, i in task_ids.items()])
    write_rows(staging / 'meta/source_windows.jsonl', mappings)
    write_rows(staging / 'meta/sample_starts.jsonl', starts)
    conversion.update(sources=native_rows, assembly='cluster_windows_50_steps',
                      assembly_provenance='source_windows.jsonl')
    write(staging / 'meta/robotwin_ee_conversion.json', conversion)
    write(staging / 'meta/train_window_stats.json',
          dict(sampling='one start observation and one 50-step action chunk per training window; validation excluded',
               state=feature_stats(train_states), actions=feature_stats(np.concatenate(train_actions))))
    manifest = dict(status='complete', specification=spec, total_windows=len(windows),
                    train_windows=ntrain, validation_windows=len(windows) - ntrain,
                    stored_frames=len(windows) * 50,
                    tasks=dict(Counter(w['dataset_id'] for w in windows)),
                    data_bytes=sum((staging / m['exported_parquet']).stat().st_size for m in mappings),
                    sampling='Use meta/sample_starts.jsonl, NOT uniform sampling over all exported frames.',
                    note='Overlapping source windows are preserved; labels describe whole windows, not every frame.')
    write(staging / 'meta/cluster_export.json', manifest)
    staging.rename(destination)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--source-root', type=Path, default=BASE / 'data/lerobot')
    parser.add_argument('--merged-root', type=Path, default=BASE / 'data/merged')
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--clusters', nargs='+', type=int, default=CLUSTERS)
    parser.add_argument('--scope', choices=['global', 'five-full', 'subset20'], default='global')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--smoke-test', action='store_true', help='Only first train and validation window per cluster.')
    args = parser.parse_args()
    if len(set(args.clusters)) != len(args.clusters) or any(k < 0 or k >= 32 for k in args.clusters):
        parser.error('Cluster IDs must be unique integers in [0,31]')
    args.source_root = args.source_root.resolve()
    output = (args.output_root or BASE / 'data/cluster_exports' /
              f'lambda003_calibrated5_k32_{args.scope}{"_smoke" if args.smoke_test else ""}').resolve()
    selected, provenance = select_windows(args)
    lengths, estimated = {}, 0
    for k, windows in selected.items():
        for row in windows:
            path = source_path(row, args.source_root)
            if path not in lengths:
                lengths[path] = (pq.ParquetFile(path).metadata.num_rows, path.stat().st_size)
            length, size = lengths[path]
            if row['end'] > length:
                raise ValueError(f'Invalid window: {row}')
            estimated += size * 50 / length
        print(f'Cluster {k:02d}: {len(windows)} windows; {dict(Counter(w["split"] for w in windows))}', flush=True)
    print(f'Output: {output}\nEstimated Parquet size: {estimated/1024**3:.2f} GiB '
          '(rough estimate, overlapping windows copied); GPU not required.', flush=True)
    if args.dry_run:
        return
    output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output).free < estimated * 1.25:
        raise RuntimeError('Insufficient space: reserve 1.25 times estimated output size')
    reports = []
    for k, windows in selected.items():
        reports.append(export_cluster(k, windows, output / f'cluster_{k:02d}_pi0_ee', args.source_root, provenance))
    write(output / 'export_summary.json', dict(provenance=provenance, datasets=reports))
    print(f'Completed {len(reports)} datasets: {output}', flush=True)


if __name__ == '__main__':
    main()
