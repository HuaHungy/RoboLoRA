"""MLP coefficient sweep, visual-distance controls and gripper-event review."""

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
import torch

from action_clustering.clustering import score_sample
from action_clustering.config import fresh_dir, write_json
from action_clustering.data import read_index
from action_clustering.gripper_review import collect_events, select_events, event_scores, export_review
from action_clustering.mlp_temporal import (chains_and_pairs, make_steps, fit_gate, temporal_metrics,
                                           diagnostics, motion_r2, sample_pairs, distance_contribution,
                                           calibrate_alpha, guarded_weights, train_mlp)
from action_clustering.progress import duration, timed
from action_clustering.visual_aux import file_hash, fuse


def read(path):
    return json.loads(path.read_text())


def evaluate(out, name, motion, visual, alpha, calibration, rows, action, pairs, chains, scores,
             val_distance_pairs, events, args):
    fresh_dir(out)
    z = fuse(motion, visual, alpha)
    np.save(out / 'z.npy', z)
    train = np.asarray([i for i, r in enumerate(rows) if r['split'] == 'train'])
    val = np.asarray([i for i, r in enumerate(rows) if r['split'] == 'validation'])
    val_mask = np.zeros(len(rows), dtype=bool); val_mask[val] = True
    contribution = distance_contribution(motion, visual, val_distance_pairs, alpha)
    spread = diagnostics(motion, train)
    write_json(out / 'visual_distance.json', {'validation': contribution, 'calibration': calibration})
    records, labels_by_name = [], {}
    for k in args.k:
        with timed(f'{name} K={k} 聚类和初始化稳定性'):
            fits = [KMeans(n_clusters=k, n_init=10, random_state=s).fit(z[train]) for s in (42, 0, 1, 2)]
            labels = fits[0].predict(z)
            folder = fresh_dir(out / f'k{k}')
            np.save(folder / 'labels.npy', labels); np.save(folder / 'centers.npy', fits[0].cluster_centers_)
            counts = np.bincount(labels, minlength=k)
            record = {'variant': name, 'k': k, 'fit_split': 'train', 'evaluation_split': 'validation',
                      'validation_silhouette': score_sample(z, labels, val, 42, 2000),
                      'stability_ari': float(np.mean([adjusted_rand_score(labels[val], km.predict(z[val])) for km in fits[1:]])),
                      'r2': motion_r2(action, labels, train, val, k),
                      'temporal': temporal_metrics(labels, pairs, scores, val_mask, chains),
                      'visual_distance': contribution, 'motion_spread': spread,
                      'occupied_clusters': int(np.count_nonzero(counts)), 'cluster_min': int(counts.min()),
                      'largest_cluster_fraction': float(counts.max() / len(rows)),
                      'gripper_events': event_scores(events, rows, labels)}
            write_json(folder / 'metrics.json', record)
            with (folder / 'assignments.jsonl').open('w') as stream:
                for row, label in zip(rows, labels):
                    stream.write(json.dumps({**row, 'cluster_id': int(label)}, ensure_ascii=False) + '\n')
            representatives = {}
            rng = np.random.default_rng(42)
            for c in range(k):
                candidates = train[labels[train] == c]
                distance = np.square(z[candidates] - fits[0].cluster_centers_[c]).sum(-1)
                picked = candidates[np.argsort(distance)[:2]]
                validation = val[labels[val] == c]
                sampled = rng.choice(validation, min(2, len(validation)), replace=False)
                representatives[str(c)] = {'representative_train': [rows[i] for i in picked],
                                           'random_validation': [rows[i] for i in sampled]}
            write_json(folder / 'representatives.json', representatives)
            records.append(record); labels_by_name[f'{name}_k{k}'] = labels
            print(f"{name} K={k}: switch={record['temporal']['all_switch_rate']}, "
                  f"gripper_delta_R2={record['r2']['gripper_delta']}, visual distance fraction="
                  f"{contribution['visual_mean_pair_fraction']}, ARI={record['stability_ari']:.4f}", flush=True)
    return records, labels_by_name


def report(root, records, smoke):
    fmt = lambda x: 'N/A' if x is None else f'{x:.4f}'
    review_only = read(root / 'experiment.json').get('review_only', False)
    title_note = '（仅功能验证）' if smoke else ('（仅历史片段复核，未训练新模型）' if review_only else '')
    lines = ['# MLP 时间一致性探索' + title_note, '',
             '固定数据划分、原动作/状态标准化、模型结构、每轮采样顺序与视觉向量。默认只改变一致性系数。',
             'fixed5：融合系数5%；calibrated5：只用训练窗口对校准平均平方距离贡献为5%；motion_only：不融合视觉。',
             '三个视觉模式复用同一个已训练动作模型，不额外训练。启用guard的组是单独的夹爪事件保护消融。', '',
             '| 方案 | K | 总切换率 | 平稳代理区切换率 | 位置R² | 姿态R² | 夹爪R² | 夹爪变化R² | ARI | 视觉融合系数 | 验证视觉贡献：逐对均值 | 视觉贡献：均值之比 | 占用簇 |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in records:
        values = [r['variant'], str(r['k']), fmt(r['temporal']['all_switch_rate']), fmt(r['temporal']['quiet_switch_rate']),
                  *[fmt(r['r2'][key]) for key in ('position', 'rotation', 'gripper', 'gripper_delta')], fmt(r['stability_ari']),
                  fmt(r['visual_distance']['alpha']), fmt(r['visual_distance']['visual_mean_pair_fraction']),
                  fmt(r['visual_distance']['visual_ratio_of_mean_distances']), str(r['occupied_clusters'])]
        lines.append('| ' + ' | '.join(values) + ' |')
    lines += ['', '## 选中模型', '', '| 模型 | 最佳轮次 | 验证动作MSE | 验证状态MSE | 动作平均坐标标准差 | 有效秩 |', '|---|---|---|---|---|---|']
    for path in sorted(root.glob('models/*/embeddings/metadata.json')):
        meta = read(path); training = path.parent.parent / 'training'
        best = read(training / 'history.json')[meta['checkpoint_epoch'] - 1]
        lines.append(f"| {path.parent.parent.name} | {meta['checkpoint_epoch']} | {best['validation_action_loss']:.6f} | "
                     f"{best['validation_state_loss']:.6f} | {meta['motion_spread']['mean_feature_std']:.4f} | {meta['motion_spread']['effective_rank']:.3f} |")
    event_path = root / 'event_summary.json'
    if event_path.exists():
        lines += ['', '## 夹爪转换筛查（不是误合并率）', '',
                  '| 方案 | 可用事件数 | 事件中心两侧窗口同簇比例 |', '|---|---|---|']
        for name, metrics in read(event_path).items():
            all_events = metrics.get('all')
            if all_events:
                lines.append(f"| {name} | {all_events['eligible_events']} | {all_events['same_cluster_fraction']:.4f} |")
            else:
                lines.append(f'| {name} | 0 | N/A |')
    lines += ['', '比例列均为0~1。校准目标是训练集“平均视觉距离/平均总距离”，不是逐对比值的均值；验证集不参与校准。',
              '动作距离收缩可能提高固定视觉系数下的实际视觉影响。需同时看两个距离比例、动作有效秩、簇占用及夹爪变化R²。',
              '平稳/事件代理非技能真值；跨夹爪事件仍为同簇也不等于误合并。事件中心两侧窗口可能都包含完整转换。',
              'gripper_review/index.html 提供实际图像与标签对照；manual_review.csv 留给人工记录物体交互和短行为是否保留。',
              '历史平滑结果只读入已有标签，不再生成新平滑结果；历史组只作参考，系数效果优先比较本次匹配的模型。',
              '模型按验证重建损失选取best.pt，同时保留last.pt。不同编码空间轮廓系数不可单独排名；单seed无统计显著性结论。']
    (root / 'report.md').write_text('\n'.join(lines) + '\n'); write_json(root / 'summary.json', records)
    if not records:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # One page per K keeps the sweep readable.
    for k in sorted({r['k'] for r in records}):
        subset = [r for r in records if r['k'] == k]
        names = [r['variant'].replace('__', '\n') for r in subset]
        fig, axes = plt.subplots(2, 2, figsize=(max(11, len(subset) * .8), 9))
        values = [[r['temporal']['all_switch_rate'] for r in subset], [r['r2']['gripper_delta'] for r in subset],
                  [r['r2']['position'] for r in subset], [r['visual_distance']['visual_mean_pair_fraction'] for r in subset]]
        for ax, title, series in zip(axes.flat, ['Adjacent switches', 'Gripper change R2', 'Position R2', 'Visual distance fraction (mean of ratios)'], values):
            ax.bar(np.arange(len(series)), [np.nan if x is None else x for x in series])
            ax.set_title(title); ax.set_xticks(range(len(names)), names, rotation=50, ha='right')
        fig.suptitle(('SMOKE TEST ONLY | ' if smoke else '') + f'MLP coefficient sweep K={k}')
        fig.tight_layout(); fig.savefig(root / f'metrics_k{k}.png', dpi=140); plt.close(fig)


def reference_labels(reference, rows_all, selection, requested_k):
    exp = read(reference / 'experiment.json')
    if exp.get('smoke_test') or read(reference / 'timing.json').get('status') != 'complete':
        raise ValueError('Reference must be a complete full-data historical experiment')
    old_rows = read_index(reference)
    keys = ('window_id', 'episode_id', 'start', 'end', 'split')
    if len(old_rows) != len(rows_all) or any(any(a[key] != b[key] for key in keys) for a, b in zip(old_rows, rows_all)):
        raise ValueError('Historical reference window mapping differs')
    labels, provenance = {}, {}
    for source, suffix, title in [('baseline30', 'smooth0', 'historical_raw30'),
                                   ('baseline30', 'smooth2', 'historical_smoothed30'),
                                   ('mlp_temporal', 'smooth0', 'historical_lambda0_1')]:
        for k in requested_k:
            path = reference / 'evaluation' / source / f'k{k}_{suffix}' / 'labels.npy'
            values = np.load(path)
            if len(values) != len(rows_all):
                raise ValueError('Historical label count mismatch')
            name = f'{title}_k{k}'; labels[name] = values[selection]
            provenance[name] = {'path': str(path.resolve()), 'sha256': file_hash(path), 'k': k,
                                'read_only': True, 'historical_parameters': exp}
    return labels, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coefficients', nargs='+', type=float, default=[0., .03, .1])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--k', nargs='+', type=int, default=[32, 48])
    parser.add_argument('--visual-modes', nargs='+', choices=['fixed5', 'calibrated5', 'motion_only'],
                        default=['fixed5', 'calibrated5', 'motion_only'])
    parser.add_argument('--with-gripper-guard', action='store_true', help='Add an independent lambda=.03 gripper-guard ablation')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--events-per-group', type=int, default=8)
    parser.add_argument('--skip-images', action='store_true', help='Explicitly skip GIFs; still export numerical review')
    parser.add_argument('--review-only', action='store_true', help='Review historical labels; no new training or clustering')
    parser.add_argument('--reference-run', type=Path, default=BASE / 'runs/temporal_20261109_222250_840101')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--smoke-test', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if min(args.epochs, args.threads, args.events_per_group) < 1 or min(args.k) < 2:
        parser.error('Positive epochs/threads/events and K>=2 required')
    if any(not np.isfinite(x) or x < 0 for x in args.coefficients) or len(set(args.coefficients)) != len(args.coefficients):
        parser.error('Distinct finite nonnegative coefficients required')
    if len(set(args.k)) != len(args.k) or len(set(args.visual_modes)) != len(args.visual_modes):
        parser.error('Duplicate K or visual mode')
    if len({f'{x:g}' for x in args.coefficients}) != len(args.coefficients):
        parser.error('Coefficients are too close to distinguish in output names')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        parser.error('CUDA unavailable; use --device cpu')
    source = BASE / 'runs/lerobot_ee_visual_aux'
    meta = read(source / 'embeddings/metadata.json'); motion_source = Path(meta['motion_source'])
    windows = (source / 'windows').resolve(); rows_all = read_index(windows)
    index_hash = file_hash(windows / 'index.jsonl')
    if index_hash != meta['index_sha256'] or file_hash(motion_source / 'embeddings/z.npy') != meta['motion_embedding_sha256']:
        raise ValueError('Source provenance mismatch')
    cfg = read(motion_source / 'training/config.json')
    if cfg['window'] != {'length': 50, 'stride': 25} or cfg['training']['latent_dim'] != 64:
        raise ValueError('Expected the reviewed 50/25, latent64 baseline')
    chains_and_pairs(rows_all, 25)
    if [r['window_id'] for r in rows_all] != list(range(len(rows_all))):
        raise ValueError('Window order mismatch')
    for cache in {r['episode_cache'] for r in rows_all}:
        if not (windows / cache).is_file():
            raise FileNotFoundError(windows / cache)
    with np.load(motion_source / 'training/normalization.npz') as f:
        stats = {key: f[key] for key in f.files if key.startswith(('action_', 'state_'))}
    visual_all = np.load(source / 'embeddings/visual.npy')
    np.testing.assert_allclose(fuse(np.load(motion_source / 'embeddings/z.npy'), visual_all, .05),
                               np.load(source / 'embeddings/z.npy'), atol=1e-6)
    import matplotlib.pyplot
    import pyarrow.parquet
    from PIL import Image
    selection = np.arange(len(rows_all))
    reference_k = args.k
    if args.smoke_test:
        chosen = set()
        for split in ('train', 'validation'):
            episodes = sorted({r['episode_id'] for r in rows_all if r['split'] == split})
            chosen.update(episodes[i] for i in np.linspace(0, len(episodes)-1, min(8, len(episodes)), dtype=int))
        selection = np.asarray([i for i,r in enumerate(rows_all) if r['episode_id'] in chosen])
        args.epochs, args.k, args.events_per_group = 2, [3], 1
        reference_k = [32]
    historical, history_provenance = reference_labels(args.reference_run.resolve(), rows_all, selection, reference_k)
    print(f'预检查通过：原始 {len(rows_all)} 窗口，历史标签仅用于对照；只训练 MLP。', flush=True)
    if args.check_only:
        return
    rows = [rows_all[i] for i in selection]
    train = np.asarray([i for i,r in enumerate(rows) if r['split']=='train'])
    val = np.asarray([i for i,r in enumerate(rows) if r['split']=='validation'])
    if max(args.k) >= len(train):
        raise ValueError('Not enough training windows for K')
    prefix = 'mlp_temporal_' + ('smoke_' if args.smoke_test else '')
    root = fresh_dir((args.output or BASE/'runs'/(prefix+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve())
    started = monotonic()
    (root/'source_windows').symlink_to(windows, target_is_directory=True)
    (root/'index.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))
    np.save(root/'window_ids.npy',selection)
    write_json(root/'experiment.json', {**vars(args), 'output':str(root), 'reference_run':str(args.reference_run.resolve()),
                                       'index_sha256':index_hash, 'visual_sha256':file_hash(source/'embeddings/visual.npy'),
                                       'review_seed':20261110, 'clustering_seed':42, 'distance_sample_seed':42})
    write_json(root/'historical_provenance.json',history_provenance)
    events=collect_events(windows,rows); selected=select_events(events,args.events_per_group)
    write_json(root/'all_validation_events.json',events); write_json(root/'review_selection_before_training.json',selected)
    print(f'输出：{root}\n提前固定 {len(selected)} 个随机验证事件；候选共 {len(events)} 个。',flush=True)
    label_sets=dict(historical); records=[]
    if not args.review_only:
        cfg['seed']=args.seed;cfg['training'].update(epochs=args.epochs,threads=args.threads,device=args.device)
        action=np.asarray(np.load(windows/'action.npy',mmap_mode='r')[selection])
        state=np.asarray(np.load(windows/'state.npy',mmap_mode='r')[selection])
        steps=make_steps(windows,rows,root/'local_steps_for_boundary.npy')
        chains,pairs=chains_and_pairs(rows,25)
        train_mask=np.zeros(len(rows),dtype=bool);train_mask[train]=True
        scores,weights,gate=fit_gate(steps,pairs,train_mask,25)
        write_json(root/'boundary_gate.json',gate);np.savez(root/'boundary_pairs.npz',pairs=pairs,score=scores,weight=weights)
        train_pairs,val_pairs=sample_pairs(train),sample_pairs(val)
        np.savez(root/'distance_pairs.npz',train=train_pairs,validation=val_pairs)
        normalized=(action-stats['action_mean'])/stats['action_std'];visual=visual_all[selection]
        models=fresh_dir(root/'models');evaluations=fresh_dir(root/'evaluation')
        experiments=[(f'lambda{x:g}'.replace('.','_'),x,False) for x in args.coefficients]
        if args.with_gripper_guard:
            experiments.append(('lambda0_03_guard',.03,True))
        for n,(name,coefficient,guard) in enumerate(experiments,1):
            out=fresh_dir(models/name)
            active_weights,blocked=guarded_weights(action,pairs,weights) if guard else (weights,np.zeros(len(pairs),dtype=bool))
            write_json(out/'gate_activity.json',{'guard':guard,'blocked_train_pairs':int(blocked[train_mask[pairs[:,0]]].sum()),
                                                'train_pair_count':int(train_mask[pairs[:,0]].sum()),
                                                'mean_train_weight':float(active_weights[train_mask[pairs[:,0]]].mean())})
            print(f'模型 {n}/{len(experiments)}：{name}，{args.epochs} 轮。ETA 为当前模型。',flush=True)
            with timed(name):
                motion=train_mlp(out,coefficient,cfg,rows,{'action':action,'state':state},stats,pairs,active_weights,args.device,guard)
            for mode in args.visual_modes:
                alpha,calibration=(.05,{'fit_split':None,'mode':'fixed coefficient'}) if mode=='fixed5' else (0.,{'fit_split':None,'mode':'motion only'})
                if mode=='calibrated5':
                    alpha,calibration=calibrate_alpha(motion,visual,train_pairs)
                variant=name+'__'+mode
                with threadpool_limits(limits=args.threads):
                    new,labels=evaluate(evaluations/variant,variant,motion,visual,alpha,calibration,rows,normalized,pairs,chains,scores,val_pairs,events,args)
                records.extend(new);label_sets.update(labels);report(root,records,args.smoke_test)
    event_summary={name:event_scores(events,rows,labels) for name,labels in label_sets.items()}
    write_json(root/'event_summary.json',event_summary)
    review_status=export_review(root/'gripper_review',windows,rows,selected,label_sets,args.skip_images,args.smoke_test)
    report(root,records,args.smoke_test)
    write_json(root/'timing.json',{'status':'complete','total_seconds':monotonic()-started,'review':review_status})
    print(f'完成：{root}/report.md\n夹爪复核：{root}/gripper_review/index.html\n图像状态：{review_status}\n耗时 {duration(monotonic()-started)}',flush=True)


if __name__=='__main__':
    main()
