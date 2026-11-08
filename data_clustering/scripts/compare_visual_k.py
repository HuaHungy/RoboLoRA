"""Compare cluster counts on the same frozen visual/motion embeddings."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from time import monotonic

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

from action_clustering.clustering import cluster, score_sample
from action_clustering.config import fresh_dir, write_json
from action_clustering.data import read_index
from action_clustering.progress import duration, timed
from action_clustering.visual_aux import evaluate_motion, file_hash


def compare(source, output, k_values, threads=4):
    source, output = source.resolve(), output.resolve()
    if threads < 1 or not k_values or len(set(k_values)) != len(k_values) or min(k_values) < 2:
        raise ValueError('Positive threads and distinct cluster counts >= 2 are required')
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    started = monotonic()
    meta = json.loads((source / 'embeddings/metadata.json').read_text())
    if meta.get('method') != 'frozen_motion_plus_frozen_resnet18_train_only_pca_late_fusion':
        raise ValueError('Source must be a completed visual_aux run')
    if file_hash(source / 'windows/index.jsonl') != meta['index_sha256']:
        raise ValueError('Source window index hash mismatch')
    motion_source = Path(meta['motion_source'])
    if file_hash(motion_source / 'embeddings/z.npy') != meta['motion_embedding_sha256']:
        raise ValueError('Motion baseline embeddings have changed')
    if file_hash(motion_source / 'windows/index.jsonl') != meta['index_sha256']:
        raise ValueError('Motion and visual runs have different window indices')
    rows = read_index(source / 'windows')
    if [r['window_id'] for r in rows] != list(range(len(rows))):
        raise ValueError('Window IDs must match embedding row order')
    z = np.load(source / 'embeddings/z.npy')
    motion = np.load(motion_source / 'embeddings/z.npy')
    if z.shape != (len(rows), meta['motion_dim'] + meta['visual_dim']) or motion.shape != (len(rows), meta['motion_dim']):
        raise ValueError('Embedding dimensions do not match metadata')
    for values in (z, motion):
        if not np.isfinite(values).all() or not np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-5):
            raise ValueError('Invalid unit embeddings')
    np.testing.assert_allclose(z[:, :meta['motion_dim']],
                               np.sqrt(1 - meta['visual_distance_weight']) * motion, atol=1e-6)
    train, val, motion_score = evaluate_motion(source, rows)
    if len(val) < 3 or max(k_values) >= len(train):
        raise ValueError('Insufficient training or validation windows for requested cluster counts')
    train_episodes = {rows[i]['episode_id'] for i in train}
    if train_episodes & {rows[i]['episode_id'] for i in val}:
        raise ValueError('Training and validation episodes overlap')
    cfg = json.loads((source / 'training/config.json').read_text())
    cfg['output_dir'] = str(output)
    cfg['training']['threads'] = threads
    cfg['clustering'].update(k_values=k_values, hdbscan=False, silhouette_sample_size=2000)
    seed = cfg['seed']
    fresh_dir(output)
    for name in ('windows', 'training', 'embeddings'):
        (output / name).symlink_to((source / name).resolve(), target_is_directory=True)
    write_json(output / 'comparison_config.json', {
        'source_run': str(source), 'motion_source': str(motion_source), 'k_values': k_values,
        'visual_distance_weight': meta['visual_distance_weight'], 'fused_dimensions': z.shape[1],
        'seed': seed, 'n_init': 10, 'stability_seeds': [0, 1, 2], 'threads': threads,
        'index_sha256': meta['index_sha256'], 'embedding_sha256': file_hash(source / 'embeddings/z.npy'),
        'fit_split': 'train', 'evaluation_split': 'validation', 'silhouette_sample_size': 2000,
    })
    print(f"输出目录: {output}\n复用 {len(rows)} 个 {z.shape[1]} 维向量，视觉距离系数 "
          f"{meta['visual_distance_weight']:.0%}，K={k_values}。\n"
          '只重新聚类及评估；不重复图片编码、不重训动作网络。', flush=True)
    summary = cluster(cfg)
    baseline_folder = fresh_dir(output / 'motion_baselines')
    results = []
    with threadpool_limits(limits=threads):
        for k in k_values:
            with timed(f'K={k}：纯动作对照、三次初始化稳定性及动作一致性'):
                visual_labels = np.load(output / f'clusters/kmeans_k{k}/labels.npy')
                baseline = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(motion[train])
                baseline_labels = baseline.predict(motion)
                np.save(baseline_folder / f'labels_k{k}.npy', baseline_labels)
                np.save(baseline_folder / f'centers_k{k}.npy', baseline.cluster_centers_)
                record = {'k': k, 'validation_ari_between_modes': float(adjusted_rand_score(
                    baseline_labels[val], visual_labels[val]))}
                for mode, values, labels in [('motion_only', motion, baseline_labels), ('visual_aux', z, visual_labels)]:
                    stability = []
                    for alternate_seed in (0, 1, 2):
                        alternate = KMeans(n_clusters=k, n_init=10, random_state=alternate_seed)
                        predicted = alternate.fit(values[train]).predict(values[val])
                        stability.append(float(adjusted_rand_score(labels[val], predicted)))
                    counts = np.bincount(labels, minlength=k)
                    record[mode] = {
                        'validation_silhouette': score_sample(values, labels, val, seed, 2000),
                        'validation_motion_space_silhouette': score_sample(motion, labels, val, seed, 2000),
                        'kmeans_seed_ari': stability, 'kmeans_seed_ari_mean': float(np.mean(stability)),
                        'motion_feature_r2': motion_score(labels, k), 'occupied_clusters': int(np.count_nonzero(counts)),
                        'cluster_size_min': int(counts.min()), 'cluster_size_max': int(counts.max()),
                        'largest_cluster_fraction': float(counts.max() / len(rows)),
                    }
                results.append(record)
                visual_metrics = {**summary[f'kmeans_k{k}'], **record['visual_aux'],
                                  'validation_ari_to_same_k_motion_baseline': record['validation_ari_between_modes']}
                summary[f'kmeans_k{k}'] = visual_metrics
                write_json(output / f'clusters/kmeans_k{k}/metrics.json', visual_metrics)
                write_json(output / 'clusters/comparison.json', summary)
                write_json(output / 'k_comparison.json', results)
                print(f"K={k}: 视觉轮廓系数={record['visual_aux']['validation_silhouette']:.4f}, "
                      f"稳定性 ARI={record['visual_aux']['kmeans_seed_ari_mean']:.4f}, "
                      f"最小/最大簇={record['visual_aux']['cluster_size_min']}/{record['visual_aux']['cluster_size_max']}", flush=True)
    lines = ['# 固定视觉表示下的簇数比较', '',
             f"{len(rows)} 个窗口，{len(train)} 个训练窗口、{len(val)} 个验证窗口；"
             f"动作 {meta['motion_dim']} 维 + 视觉 {meta['visual_dim']} 维，视觉平方距离系数 {meta['visual_distance_weight']:.0%}。", '',
             '## 视觉辅助结果', '',
             '| K | 融合空间轮廓系数 | K-means 稳定性 ARI | 位置 R² | 姿态 R² | 夹爪 R² | 夹爪变化 R² | 最小/最大簇 |',
             '|---|---|---|---|---|---|---|---|']
    for r in results:
        v = r['visual_aux']; f = v['motion_feature_r2']
        lines.append(f"| {r['k']} | {v['validation_silhouette']:.4f} | {v['kmeans_seed_ari_mean']:.4f} | "
                     f"{f['position']:.4f} | {f['rotation']:.4f} | {f['gripper']:.4f} | {f['gripper_delta']:.4f} | "
                     f"{v['cluster_size_min']}/{v['cluster_size_max']} |")
    lines += ['', '## 同 K 的纯动作对照', '',
              '下表轮廓系数统一在原动作空间计算，避免把不同空间的分数直接当作语义提升。', '',
              '| K | 纯动作划分的动作轮廓系数 | 视觉辅助划分的动作轮廓系数 | 纯动作稳定性 ARI | 视觉辅助稳定性 ARI | 两种划分 ARI |',
              '|---|---|---|---|---|---|']
    for r in results:
        b, v = r['motion_only'], r['visual_aux']
        lines.append(f"| {r['k']} | {b['validation_motion_space_silhouette']:.4f} | "
                     f"{v['validation_motion_space_silhouette']:.4f} | {b['kmeans_seed_ari_mean']:.4f} | "
                     f"{v['kmeans_seed_ari_mean']:.4f} | {r['validation_ari_between_modes']:.4f} |")
    elapsed = monotonic() - started
    lines += ['', '所有 K 使用相同向量、episode 划分及验证抽样。K-means 仅在训练集拟合，n_init=10。',
              '稳定性为固定编码器后，种子 0/1/2 相对主种子划分的验证 ARI 均值，不是准确率。',
              '动作 R² 使用训练簇均值重建验证窗口的标准化动作特征，是运动一致性代理。簇数增加通常有利于细化重建，不能据此单独认定更优。',
              '这些结果不提供物体交互语义真值或下游成功率；需结合代表片段和随机验证片段判断是否过粗或过细。',
              f'共享 windows/training/embeddings 来自 {source}，须保留其源目录。',
              f'总耗时 {duration(elapsed)}。']
    (output / 'report.md').write_text('\n'.join(lines) + '\n')
    write_json(output / 'timing.json', {'total_seconds': elapsed})
    print('\n' + '\n'.join(lines[4:10 + len(results)]), flush=True)
    print(f"完成：{output / 'report.md'} | 耗时 {duration(elapsed)}", flush=True)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='复用图片辅助向量，比较不同 K 及相同 K 的纯动作基线')
    parser.add_argument('--source', type=Path, default=BASE / 'runs/lerobot_ee_visual_aux')
    parser.add_argument('--output', type=Path, help='必须是新目录；默认自动创建带时间戳的目录')
    parser.add_argument('--k', type=int, nargs='+', default=[16, 32, 48])
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    destination = args.output or BASE / 'runs' / ('visual_k_comparison_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    compare(args.source, destination, args.k, args.threads)
