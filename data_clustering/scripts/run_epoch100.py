"""Controlled 100-epoch experiment using the existing windows and visual codes."""

import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

import numpy as np

from action_clustering.config import fresh_dir, write_json
from action_clustering.data import read_index
from action_clustering.learning import embed, fit, metadata
from action_clustering.visual_aux import file_hash, fuse
from compare_visual_k import compare


def read(path):
    return json.loads(path.read_text())


def switching(rows, labels, stride):
    episodes = defaultdict(list)
    for i, row in enumerate(rows):
        if row['split'] == 'validation':
            episodes[row['episode_id']].append(i)
    changed = total = 0
    for ids in episodes.values():
        ids.sort(key=lambda i: rows[i]['start'])
        for a, b in zip(ids, ids[1:]):
            if rows[b]['start'] - rows[a]['start'] == stride:
                total += 1
                changed += int(labels[a] != labels[b])
    return {'changed': changed, 'pairs': total, 'rate': changed / total if total else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', help='Validate inputs without training or creating a run')
    args = parser.parse_args()
    visual_source = BASE / 'runs/lerobot_ee_visual_aux'
    baseline = BASE / 'runs/visual_k_comparison_20261104_232645_990948'
    visual_meta = read(visual_source / 'embeddings/metadata.json')
    source = Path(visual_meta['motion_source'])
    cfg = read(source / 'training/config.json')
    cfg['output_dir'] = str(source)
    windows, _ = metadata(cfg)
    rows = read_index(windows)
    index_hash = file_hash(windows / 'index.jsonl')
    assert index_hash == visual_meta['index_sha256'] == file_hash(baseline / 'windows/index.jsonl')
    assert file_hash(source / 'embeddings/z.npy') == visual_meta['motion_embedding_sha256']
    old_config = read(baseline / 'comparison_config.json')
    assert old_config['seed'] == cfg['seed'] and old_config['k_values'] == [16, 32, 48]
    assert old_config['embedding_sha256'] == file_hash(visual_source / 'embeddings/z.npy')
    visual = np.load(visual_source / 'embeddings/visual.npy')
    old_motion = np.load(source / 'embeddings/z.npy')
    np.testing.assert_allclose(fuse(old_motion, visual, visual_meta['visual_distance_weight']),
                               np.load(visual_source / 'embeddings/z.npy'), atol=1e-6)
    old_results = read(baseline / 'k_comparison.json')
    old_history = read(source / 'training/history.json')
    assert len(old_history) == 30 and cfg['training']['latent_dim'] == 64
    assert visual.shape == (len(rows), 8) and visual_meta['visual_distance_weight'] == 0.05
    # Import plotting dependencies before committing to a long training run.
    import matplotlib.pyplot  # noqa: F401
    print(f'检查通过：{len(rows)} 个窗口；相同轨迹划分、64+8 维、视觉权重 5%、CPU。', flush=True)
    print('从相同 seed 重新训练 100 轮；选择验证损失最低的 checkpoint；复用已保存的视觉向量。', flush=True)
    if args.check_only:
        return

    root = fresh_dir(BASE / 'runs' / ('epoch100_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
    motion = fresh_dir(root / 'motion')
    (motion / 'windows').symlink_to(windows.resolve(), target_is_directory=True)
    cfg['output_dir'] = str(motion)
    cfg['training']['epochs'] = 100
    write_json(root / 'motion_config.json', cfg)
    print(f'实验目录：{root}\n[1/4] 训练 100 轮（每批显示进度、耗时及 ETA）', flush=True)
    history = fit(cfg)
    print('[2/4] 编码并检查归一化统计量', flush=True)
    embed(cfg)
    with np.load(source / 'training/normalization.npz') as old, np.load(motion / 'training/normalization.npz') as new:
        for key in old.files:
            np.testing.assert_array_equal(old[key], new[key])
    selected = read(motion / 'embeddings/metadata.json')['checkpoint_epoch']

    print('[3/4] 与原有 8 维视觉向量融合', flush=True)
    fused_run = fresh_dir(root / 'visual')
    (fused_run / 'windows').symlink_to(windows.resolve(), target_is_directory=True)
    (fused_run / 'training').symlink_to(motion / 'training', target_is_directory=True)
    embeddings = fresh_dir(fused_run / 'embeddings')
    np.save(embeddings / 'z.npy', fuse(np.load(motion / 'embeddings/z.npy'), visual, 0.05))
    for name in ('visual.npy', 'visual_pca.npz', 'image_provenance.jsonl'):
        shutil.copy2(visual_source / 'embeddings' / name, embeddings / name)
    visual_meta.update(motion_source=str(motion),
                       motion_embedding_sha256=file_hash(motion / 'embeddings/z.npy'),
                       reused_visual_source=str(visual_source), checkpoint_epoch=selected)
    write_json(embeddings / 'metadata.json', visual_meta)
    print('[4/4] K=16/32/48 聚类、稳定性评估及 30/100 轮对比', flush=True)
    results = compare(fused_run, root / 'k_comparison', [16, 32, 48], cfg['training']['threads'])
    lines = ['# 30 轮与 100 轮训练预算对比', '',
             f'100 轮训练完成；按验证损失选中的模型为第 {selected} 轮。',
             '除训练轮数外保持动作训练参数不变，复用相同窗口和视觉编码；归一化统计量已验证完全一致。', '',
             '| 训练预算 | 最佳轮次 | 验证动作 MSE | 验证状态 MSE |',
             '|---|---|---|---|']
    for budget, hist in [(30, old_history), (100, history)]:
        best = min(hist, key=lambda x: x['validation_loss'])
        lines.append(f"| {budget} | {best['epoch']} | {best['validation_action_loss']:.6f} | {best['validation_state_loss']:.6f} |")
    lines += ['', '| 训练预算 | K | 轮廓系数 ↑ | 稳定性 ARI ↑ | 位置 R² ↑ | 姿态 R² ↑ | 夹爪 R² ↑ | 夹爪变化 R² ↑ | 相邻窗口切换率 |',
              '|---|---|---|---|---|---|---|---|---|']
    temporal = {}
    for record_old, record_new in zip(old_results, results, strict=True):
        assert record_old['k'] == record_new['k']
        for budget, record, folder in [(30, record_old, baseline), (100, record_new, root / 'k_comparison')]:
            k = record['k']
            labels = np.load(folder / f'clusters/kmeans_k{k}/labels.npy')
            switch = switching(rows, labels, cfg['window']['stride'])
            temporal[f'epochs{budget}_k{k}'] = switch
            metrics = record['visual_aux']
            r2 = metrics['motion_feature_r2']
            lines.append(f"| {budget} | {k} | {metrics['validation_silhouette']:.4f} | "
                         f"{metrics['kmeans_seed_ari_mean']:.4f} | {r2['position']:.4f} | "
                         f"{r2['rotation']:.4f} | {r2['gripper']:.4f} | {r2['gripper_delta']:.4f} | {switch['rate']:.2%} |")
    lines += ['', 'ARI 是固定编码器后 K-means 初始化稳定性，不是准确率。',
              '不同训练预算会改变编码空间，轮廓系数需结合共同原始动作特征的 R²、稳定性和片段检查解释。',
              '切换率不是错误率；更低也可能来自簇坍缩。重建损失降低不保证技能语义或时间连续性改善。',
              '本实验使用同一个随机种子，验证集用于选模；不等于独立测试集结果或多次训练的统计结论。',
              f'窗口通过符号链接复用 {windows.resolve()}，请保留该目录。']
    write_json(root / 'temporal_comparison.json', temporal)
    (root / 'comparison_30_vs_100.md').write_text('\n'.join(lines) + '\n')
    print('\n' + '\n'.join(lines), flush=True)
    print(f'\n最终对比报告：{root / "comparison_30_vs_100.md"}', flush=True)


if __name__ == '__main__':
    main()
