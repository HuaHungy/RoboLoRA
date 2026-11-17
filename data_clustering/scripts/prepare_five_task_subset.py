#!/usr/bin/env python3
"""Materialize five EE datasets and rank existing K=32 clusters without refitting."""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from action_clustering.lerobot import read_columns

TASKS = ('adjust_bottle', 'click_alarmclock', 'place_can_basket',
         'place_container_plate', 'rotate_qrcode')
DEFAULT_RUN = BASE / 'runs/mlp_temporal_20261109_230731_563725'
VARIANT = 'lambda0_03__calibrated5'


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def write_rows(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(v, ensure_ascii=False, allow_nan=False) + '\n' for v in values))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def sources(root):
    result = {}
    for task in TASKS:
        directory = root / f'{task}_pi0_ee'
        meta = directory / 'meta'
        info = read(meta / 'info.json')
        conversion = read(meta / 'robotwin_ee_conversion.json')
        if (conversion['status'] != 'complete' or info['codebase_version'] != 'v2.1'
                or info['total_videos'] or conversion['quaternion_order'] != 'wxyz'
                or conversion['representation'] != 'absolute_world_ee_pose'):
            raise ValueError(f'Unsupported source conventions: {directory}')
        episodes = rows(meta / 'episodes.jsonl')
        if [x['episode_index'] for x in episodes] != list(range(info['total_episodes'])):
            raise ValueError(f'Noncontiguous source episodes: {directory}')
        if sum(x['length'] for x in episodes) != info['total_frames']:
            raise ValueError(f'Frame metadata mismatch: {directory}')
        result[task] = dict(root=directory, info=info, conversion=conversion, episodes=episodes,
                            tasks={x['task_index']: x['task'] for x in rows(meta / 'tasks.jsonl')},
                            stats={x['episode_index']: x['stats'] for x in rows(meta / 'episodes_stats.jsonl')})
        if task != TASKS[0]:
            for key in ('features', 'fps', 'robot_type', 'codebase_version'):
                if info[key] != result[TASKS[0]]['info'][key]:
                    raise ValueError(f'Source {key} differs: {task}')
    return result


def numeric_stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {**{key: [float(fn(values))] for key, fn in
               [('min', np.min), ('max', np.max), ('mean', np.mean), ('std', np.std)]},
            'count': [len(values)]}


def materialize(destination, src, selection, seed):
    """Rewrite only three index columns; preserve all image bytes and EE targets."""
    specification = dict(schema_version=1, source_roots={t: str(src[t]['root']) for t in TASKS},
                         selected_source_episodes=selection, sampling_seed=seed)
    if destination.exists():
        existing = read(destination / 'meta/merge_manifest.json')
        if existing['specification'] != specification or existing['status'] != 'complete':
            raise ValueError(f'Refusing incompatible existing output: {destination}')
        mapping = rows(destination / 'meta/source_episodes.jsonl')
        for item in mapping:
            if (digest(destination / item['merged_parquet']) != item['merged_sha256'] or
                    digest(Path(item['source_parquet'])) != item['source_sha256']):
                raise ValueError('Previously merged data or its source changed')
        print(f'Reusing verified dataset: {destination}', flush=True)
        return existing, mapping
    staging = destination.with_name(destination.name + '.partial')
    staging.mkdir(parents=True, exist_ok=False)
    (staging / 'meta').mkdir()
    template = src[TASKS[0]]
    info = copy.deepcopy(template['info'])
    language_ids, episode_rows, statistics, mapping, native = {}, [], [], [], []
    frame_offset = 0
    for task in TASKS:
        source = src[task]
        for old_id in selection[task]:
            new_id = len(mapping)
            relative = info['data_path'].format(episode_chunk=new_id // info['chunks_size'], episode_index=new_id)
            old_relative = source['info']['data_path'].format(
                episode_chunk=old_id // source['info']['chunks_size'], episode_index=old_id)
            source_path = source['root'] / old_relative
            state, _, numeric = read_columns(source_path, extra=('frame_index', 'timestamp', 'episode_index'))
            count = len(state)
            if count != source['episodes'][old_id]['length']:
                raise ValueError(f'Source episode length mismatch: {source_path}')
            np.testing.assert_array_equal(numeric['frame_index'].to_numpy(), np.arange(count))
            np.testing.assert_array_equal(numeric['episode_index'].to_numpy(), np.full(count, old_id))
            np.testing.assert_allclose(numeric['timestamp'].to_numpy(), np.arange(count) / info['fps'], atol=1e-5)
            table = pq.read_table(source_path)
            old_tasks = table['task_index'].to_numpy()
            instructions = [source['tasks'][int(i)] for i in old_tasks]
            if set(instructions) != set(source['episodes'][old_id]['tasks']):
                raise ValueError('Language metadata differs from frame task indices')
            for text in instructions:
                if text not in language_ids:
                    language_ids[text] = len(language_ids)
            changes = dict(episode_index=np.full(count, new_id, dtype=np.int64),
                           index=np.arange(frame_offset, frame_offset + count, dtype=np.int64),
                           task_index=np.asarray([language_ids[t] for t in instructions], dtype=np.int64))
            episode_stats = copy.deepcopy(source['stats'][old_id])
            for key, values in changes.items():
                field = table.schema.field(key)
                table = table.set_column(table.schema.get_field_index(key), field, pa.array(values, type=field.type))
                episode_stats[key] = numeric_stats(values)
            for key in table.column_names:
                if key.startswith('observation.images.'):
                    if any(not b for b in table[key].combine_chunks().field('bytes').to_pylist()):
                        raise ValueError(f'Missing embedded image bytes: {source_path}, {key}')
            output_path = staging / relative
            output_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, output_path, compression='snappy')
            if not pq.read_table(output_path).equals(table):
                raise ValueError(f'Written Parquet differs: {output_path}')
            episode_rows.append(dict(episode_index=new_id, tasks=list(dict.fromkeys(instructions)), length=count))
            statistics.append(dict(episode_index=new_id, stats=episode_stats))
            mapping.append(dict(episode_index=new_id, task=task, source_episode_index=old_id,
                                source_episode_id=f'{task}/{old_relative}', source_parquet=str(source_path),
                                source_sha256=digest(source_path), merged_parquet=relative,
                                merged_sha256=digest(output_path), length=count, frame_offset=frame_offset))
            native.append({**source['conversion']['sources'][old_id], 'episode_index': new_id,
                           'source_dataset': task, 'source_episode_index': old_id})
            frame_offset += count
        print(f'{destination.name}: {task} done; {len(mapping)} episodes / {frame_offset} frames', flush=True)
    info.update(total_episodes=len(mapping), total_frames=frame_offset, total_tasks=len(language_ids),
                total_chunks=(len(mapping) + info['chunks_size'] - 1) // info['chunks_size'],
                splits={'train': f'0:{len(mapping)}'})
    write(staging / 'meta/info.json', info)
    write_rows(staging / 'meta/episodes.jsonl', episode_rows)
    write_rows(staging / 'meta/episodes_stats.jsonl', statistics)
    write_rows(staging / 'meta/tasks.jsonl', [dict(task_index=i, task=t) for t, i in language_ids.items()])
    write_rows(staging / 'meta/source_episodes.jsonl', mapping)
    conversion = copy.deepcopy(template['conversion'])
    conversion.update(sources=native, status='complete', assembly='merged_existing_ee_conversions',
                      assembly_provenance='source_episodes.jsonl')
    write(staging / 'meta/robotwin_ee_conversion.json', conversion)
    manifest = dict(status='complete', specification=specification, total_episodes=len(mapping),
                    total_frames=frame_offset, total_instruction_strings=len(language_ids),
                    data_bytes=sum((staging / x['merged_parquet']).stat().st_size for x in mapping),
                    validation='All EE values, indices, timestamps, languages, embedded image presence checked; '
                               'all rewritten Parquets round-trip compared, unchanged columns preserved.',
                    normalization='Raw EE retained; pi0 training must compute shared normalization for this dataset.')
    write(staging / 'meta/merge_manifest.json', manifest)
    staging.rename(destination)
    return manifest, mapping


def cluster_comparison(run, output, full_mapping, subset_mapping, full_dir, subset_dir):
    folder = run / 'evaluation' / VARIANT / 'k32'
    index = rows(run / 'index.jsonl')
    labels = np.load(folder / 'labels.npy')
    assignments = rows(folder / 'assignments.jsonl')
    z = np.load(folder.parent / 'z.npy')
    centers = np.load(folder / 'centers.npy')
    if labels.shape != (len(index),) or centers.shape != (32, z.shape[1]) or len(z) != len(index):
        raise ValueError('Index/embedding/label dimensions do not match')
    if len(assignments) != len(index) or not np.isfinite(z).all() or not np.isfinite(centers).all():
        raise ValueError('Invalid assignments or nonfinite embeddings')
    for row, assigned, label in zip(index, assignments, labels):
        if assigned != {**row, 'cluster_id': int(label)}:
            raise ValueError('Assignment rows disagree with labels/index')
    nearest = np.argmin(np.square(z[:, None] - centers[None]).sum(-1), axis=1)
    np.testing.assert_array_equal(labels, nearest)
    full_lookup = {(x['task'], x['source_episode_index']): x for x in full_mapping}
    sub_lookup = {(x['task'], x['source_episode_index']): x for x in subset_mapping}
    full_counts, sub_counts = np.zeros((5, 32), int), np.zeros((5, 32), int)
    episode_counts = {t: [] for t in TASKS}
    window_rows = []
    seen = {key: [] for key in full_lookup}
    task_ids = {t: i for i, t in enumerate(TASKS)}
    for i, row in enumerate(index):
        key = (row['dataset_id'], row['lerobot_episode_index'])
        if key not in full_lookup:
            continue
        origin = full_lookup[key]
        if (row['episode_id'] != origin['source_episode_id'] or
                Path(row['source_path']).resolve() != Path(origin['source_parquet']).resolve()):
            raise ValueError('Clustering provenance differs from merged sources')
        if row['end'] - row['start'] != 50 or row['end'] > origin['length']:
            raise ValueError('Invalid clustering window')
        seen[key].append(row['start'])
        task_id, label = task_ids[key[0]], int(labels[i])
        full_counts[task_id, label] += 1
        if key in sub_lookup:
            sub_counts[task_id, label] += 1
            window_rows.append({**row, 'cluster_id': label, 'embedding_row': i,
                                'full_episode_index': origin['episode_index'],
                                'subset_episode_index': sub_lookup[key]['episode_index'],
                                'subset_parquet': str(subset_dir / sub_lookup[key]['merged_parquet'])})
    for key, item in full_lookup.items():
        if sorted(seen[key]) != list(range(0, item['length'] - 50 + 1, 25)):
            raise ValueError(f'Missing/duplicate clustering windows: {key}')
    for item in subset_mapping:
        counts = np.zeros(32, int)
        for w in window_rows:
            if w['subset_episode_index'] == item['episode_index']:
                counts[w['cluster_id']] += 1
        if not counts.sum():
            raise ValueError('Selected episode has no clustering windows')
        episode_counts[item['task']].append(counts)
    global_counts = np.bincount(labels, minlength=32)
    def balanced(counts):
        return (counts / counts.sum(axis=1, keepdims=True)).mean(axis=0)
    score, full_score = balanced(sub_counts), balanced(full_counts)
    rest_counts = full_counts - sub_counts
    if (rest_counts.sum(axis=1) <= 0).any():
        raise ValueError('Comparison requires remaining episodes for every task')
    rest_score = balanced(rest_counts)
    order = sorted(range(32), key=lambda k: (-score[k], k))
    full_order = sorted(range(32), key=lambda k: (-full_score[k], k))
    rest_order = sorted(range(32), key=lambda k: (-rest_score[k], k))
    raw_order = sorted(range(32), key=lambda k: (-sub_counts[:, k].sum(), k))
    top = order[:5]
    # Whole-episode bootstrap (overlapping windows are not independent samples).
    rng = np.random.default_rng(20261110)
    frequencies = np.zeros(32, int)
    for _ in range(1000):
        sampled = []
        for t in TASKS:
            values = np.asarray(episode_counts[t])
            sampled.append(values[rng.integers(len(values), size=len(values))].sum(axis=0))
        picked = np.argsort(-balanced(np.asarray(sampled)), kind='stable')[:5]
        frequencies[picked] += 1
    ranking = []
    for rank, k in enumerate(order, 1):
        n = int(sub_counts[:, k].sum())
        ranking.append(dict(rank=rank, cluster_id=k, balanced_task_coverage=float(score[k]),
                            subset_windows=n, subset_window_fraction=n / int(sub_counts.sum()),
                            global_cluster_windows=int(global_counts[k]),
                            subset_fraction_of_cluster=n / int(global_counts[k]),
                            enrichment_vs_global=(n / int(sub_counts.sum())) / (global_counts[k] / len(labels)),
                            full_balanced_task_coverage=float(full_score[k]), full_rank=full_order.index(k) + 1,
                            remaining_balanced_task_coverage=float(rest_score[k]), remaining_rank=rest_order.index(k) + 1,
                            bootstrap_top5_frequency=float(frequencies[k] / 1000),
                            **{f'{t}_windows': int(sub_counts[j, k]) for j, t in enumerate(TASKS)}))
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'cluster_ranking.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ranking[0]))
        writer.writeheader(); writer.writerows(ranking)
    write_rows(output / 'subset_windows.jsonl', window_rows)
    write_rows(output / 'subset_top5_windows.jsonl', [w for w in window_rows if w['cluster_id'] in top])
    write_rows(output / 'global_top5_windows.jsonl', [{**w, 'cluster_id': int(labels[i])} for i, w in enumerate(index)
                                                   if int(labels[i]) in top])
    representatives = {}
    for k in top:
        candidates = [w for w in window_rows if w['cluster_id'] == k]
        representatives[str(k)] = sorted(candidates, key=lambda w: float(np.square(z[w['embedding_row']] - centers[k]).sum()))[:3]
    write(output / 'subset_representatives.json', representatives)
    summary = dict(clustering_run=str(run), variant=VARIANT, k=32, cluster_id_convention='zero-based',
                   index_sha256=digest(run / 'index.jsonl'), labels_sha256=digest(folder / 'labels.npy'),
                   centers_sha256=digest(folder / 'centers.npy'), full_dataset=str(full_dir), subset_dataset=str(subset_dir),
                   score_definition='mean over 5 tasks of (subset windows in cluster / all subset windows of task)',
                   selected_top5=top, raw_window_count_top5=raw_order[:5], full_top5=full_order[:5],
                   remaining_episode_top5=rest_order[:5], total_subset_windows=int(sub_counts.sum()),
                   total_full_windows=int(full_counts.sum()), total_global_windows=len(labels),
                   selected_subset_windows=int(sub_counts[:, top].sum()),
                   selected_window_coverage=float(sub_counts[:, top].sum() / sub_counts.sum()),
                   selected_balanced_task_coverage=float(score[top].sum()),
                   per_task={t: dict(windows=int(sub_counts[i].sum()), selected_windows=int(sub_counts[i, top].sum()),
                                    selected_coverage=float(sub_counts[i, top].sum() / sub_counts[i].sum()))
                             for i, t in enumerate(TASKS)},
                   original_cluster_split_counts={split: sum(w['split'] == split for w in window_rows)
                                                  for split in ('train', 'validation')},
                   bootstrap=dict(repeats=1000, seed=20261110, resampling_unit='whole episodes within each task'),
                   visual_distance=read(folder.parent / 'visual_distance.json'), ranking=ranking,
                   caveat='Subset includes original clustering train and validation episodes. This is relevance '
                          'retrieval, not independent generalization or downstream policy evaluation. '
                          'Top5 windows overlap in time and are not complete trajectories. '
                          'Remaining episodes are a ranking sensitivity comparison, not an unseen test set.')
    write(output / 'summary.json', summary)
    return summary


def report(path, summary, full, subset):
    s = summary
    lines = ['# 五任务数据合并与 K=32 子集簇检索', '',
             f"- 全量集：{full['total_episodes']} 条 / {full['total_frames']:,} 帧 / "
             f"Parquet {full['data_bytes'] / 1024**3:.2f} GiB。路径：`{s['full_dataset']}`。",
             f"- 子集：{subset['total_episodes']} 条 / {subset['total_frames']:,} 帧 / "
             f"Parquet {subset['data_bytes'] / 1024**3:.2f} GiB。路径：`{s['subset_dataset']}`。",
             f"- 抽样种子 {subset['specification']['sampling_seed']}，每任务等量抽取完整轨迹，抽样不使用聚类标签。",
             '- 保留两臂 16 维末端 state/action、三个相机图像字节、原语言指令；只重编 episode_index/index/task_index。',
             '- meta/source_episodes.jsonl 保存原始任务、轨迹、Parquet 路径、SHA256 和合并后的编号。',
             '- 原五任务训练入口仍使用 ConcatDataset 动态联合加载；本次新增物理合并数据集。',
             '', '## 簇选择口径', '',
             f"采用 `{s['clustering_run']}/evaluation/{VARIANT}/k32` 的原标签和中心，不重新训练或拟合。",
             'λ=0.03；视觉距离贡献训练集校准目标 5%，验证集实际约 5.07%（平均距离之比）。编号从 0 开始。',
             '', '每个簇的主评分 = 五个任务各自“该簇窗口数 / 该任务子集窗口总数”的平均值。',
             '这衡量对目标任务动作分布的覆盖；不等价于语义相似度或簇纯度。富集倍数另外报告，不参与主排序。',
             '富集倍数 = 子集中该簇的窗口比例 / 原全库中该簇的窗口比例。',
             '', '| 名次 | 簇 ID | 任务等权覆盖率 | 子集窗口数 | 富集倍数 | 轨迹重采样进入前五频率 |',
             '|---|---|---|---|---|---|']
    for r in s['ranking'][:8]:
        lines.append(f"| {r['rank']} | {r['cluster_id']} | {r['balanced_task_coverage']:.2%} | "
                     f"{r['subset_windows']} | {r['enrichment_vs_global']:.2f}× | {r['bootstrap_top5_frequency']:.1%} |")
    lines += ['', f"本子集选出 **{s['selected_top5']}**。合计覆盖 {s['selected_subset_windows']}/{s['total_subset_windows']} "
              f"个窗口（{s['selected_window_coverage']:.2%}），任务等权覆盖率 {s['selected_balanced_task_coverage']:.2%}。",
              '不是“一任务一个簇”；同一簇可以覆盖多个任务，一个任务也会使用多个簇。', '',
              '| 任务 | 子集窗口 | 前五命中 | 覆盖率 |', '|---|---|---|---|']
    for task, v in s['per_task'].items():
        lines.append(f"| {task} | {v['windows']} | {v['selected_windows']} | {v['selected_coverage']:.2%} |")
    lines += ['', '## 排名稳定性与限制', '',
              f"- 全量五任务等权前五：{s['full_top5']}；去除所选子集后的剩余轨迹前五：{s['remaining_episode_top5']}。",
              f"- 如果按原始窗口数直接排序，子集前五是 {s['raw_window_count_top5']}；长轨迹任务权重更大。",
              '- 1000 次按任务分别重采样整条轨迹，统计进入前五的频率，避免把重叠窗口误当独立样本；这不是成功率。',
              f"- 第五与第六评分差仅 {(s['ranking'][4]['balanced_task_coverage'] - s['ranking'][5]['balanced_task_coverage']) * 100:.3f} "
              '个百分点，边界排名需要结合更多轨迹和片段检查，不能认定存在显著优劣。',
              f"- 子集窗口原有划分：{s['original_cluster_split_counts']}。该检索不是独立测试集评估；剩余轨迹也可能参与原聚类训练。",
              '- 窗口长度 50、步长 25；覆盖率是窗口覆盖，不是独立帧覆盖。尾部不足窗口的帧仍保留在合并数据集中。',
              '- 前五簇只覆盖部分动作；不要直接用前五窗口替换完整任务演示，后续控制训练仍需处理动作衔接。',
              '', '## 产物', '',
              '- cluster_ranking.csv / summary.json：所有 32 簇的覆盖、富集、全量对照、重采样稳定性。',
              '- subset_windows.jsonl：子集全部窗口到原簇标签、合并轨迹的映射。',
              '- subset_top5_windows.jsonl：子集中属于所选五簇的窗口清单。',
              '- global_top5_windows.jsonl：原全库 50 任务中属于这五簇的所有窗口清单（用于后续专家数据检索）。',
              '- subset_representatives.json：每个所选簇在子集中最接近原中心的三个窗口；还没有人工语义验证。',
              '- dataset_summary.json：合并与抽样的来源、大小、参数。',
              '- loader_validation.json：本机额外执行的 π0 LeRobot 加载与动作块边界检查结果。',
              '', '两个合并集分别维护自己的原始数据统计。π0 训练时应按实际使用的数据集重新计算共享归一化，'
              '不要把全量归一化统计直接当成子集统计；本次不对原始 EE 或图像数据做提前归一化。', '']
    path.write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=BASE / 'data/lerobot')
    parser.add_argument('--output-root', type=Path, default=BASE / 'data/merged')
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--report', type=Path, default=BASE / 'reports/five_task_subset20_lambda003_calibrated5_k32')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--episodes-per-task', type=int, default=20)
    args = parser.parse_args()
    if args.episodes_per_task <= 0:
        parser.error('episodes-per-task must be positive')
    src = sources(args.source_root.resolve())
    full_selection = {t: list(range(src[t]['info']['total_episodes'])) for t in TASKS}
    rng = np.random.default_rng(args.seed)
    selection = {t: sorted(rng.choice(full_selection[t], args.episodes_per_task, replace=False).tolist()) for t in TASKS}
    full_dir = args.output_root.resolve() / 'robotwin_five_pi0_ee_full'
    sub_dir = args.output_root.resolve() / f'robotwin_five_pi0_ee_subset{args.episodes_per_task}_seed{args.seed}'
    full, full_map = materialize(full_dir, src, full_selection, None)
    subset, sub_map = materialize(sub_dir, src, selection, args.seed)
    summary = cluster_comparison(args.run.resolve(), args.report.resolve(), full_map, sub_map, full_dir, sub_dir)
    write(args.report / 'dataset_summary.json', dict(full=full, subset=subset))
    report(args.report / 'analysis.md', summary, full, subset)
    print(json.dumps({k: summary[k] for k in ('selected_top5', 'total_subset_windows', 'selected_window_coverage',
                                            'selected_balanced_task_coverage', 'per_task')}, indent=2))
    print(f'Completed: {args.report.resolve()}', flush=True)


if __name__ == '__main__':
    main()
