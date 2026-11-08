"""Frozen image features with an explicit, bounded weight in clustering distance.

This is late fusion: the reviewed motion checkpoint is reused unchanged. No
language, task labels, validation-fitted PCA, or random visual features are used.
"""

import hashlib
from io import BytesIO
import json
from pathlib import Path
from time import monotonic

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

from .config import fresh_dir, write_json
from .data import read_index
from .progress import Progress, duration, timed


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def unit_rows(values):
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def visual_codes(first, last, train_ids, dimensions=8, context_weight=0.25, seed=42):
    """Separate context/change PCA, fitted exclusively on training windows."""
    if dimensions < 2 or dimensions % 2:
        raise ValueError('visual_dim must be an even integer >= 2')
    if not 0 < context_weight < 1:
        raise ValueError('context_weight must be strictly between 0 and 1')
    first, last = unit_rows(first), unit_rows(last)
    sources = {'context': (first + last) / 2, 'change': last - first}
    codes, parameters, explained = [], {}, {}
    for name, values in sources.items():
        if min(len(train_ids), values.shape[1]) <= dimensions // 2:
            raise ValueError('Too few training windows/features for visual PCA')
        pca = PCA(n_components=dimensions // 2, svd_solver='randomized', random_state=seed)
        pca.fit(values[train_ids])
        if not np.isfinite(pca.explained_variance_ratio_).all() or pca.explained_variance_.sum() < 1e-12:
            raise ValueError(f'Degenerate {name} image features; cannot create informative visual codes')
        codes.append(unit_rows(pca.transform(values)))
        parameters[name + '_mean'] = pca.mean_
        parameters[name + '_components'] = pca.components_
        parameters[name + '_explained_variance_ratio'] = pca.explained_variance_ratio_
        explained[name] = float(pca.explained_variance_ratio_.sum())
    visual = unit_rows(np.concatenate([np.sqrt(context_weight) * codes[0],
                                      np.sqrt(1 - context_weight) * codes[1]], axis=1))
    if not np.isfinite(visual).all() or not np.allclose(np.linalg.norm(visual, axis=1), 1, atol=1e-5):
        raise ValueError('Degenerate visual code for one or more windows')
    return visual, parameters, explained


def fuse(motion, visual, weight):
    """Squared distance = (1-weight)*motion_distance² + weight*visual_distance²."""
    if not 0 <= weight < 1:
        raise ValueError('visual distance weight must be in [0,1)')
    for name, values in [('motion', motion), ('visual', visual)]:
        if values.ndim != 2 or not np.isfinite(values).all() or not np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-5):
            raise ValueError(f'{name} features must be finite unit vectors')
    if len(motion) != len(visual):
        raise ValueError('Motion/visual window counts differ')
    if weight == 0:
        return np.asarray(motion, dtype=np.float32).copy()
    return np.concatenate([np.sqrt(1 - weight) * motion, np.sqrt(weight) * visual], axis=1).astype(np.float32)


def episode_frames(rows):
    """Use real Parquet observations at start and end-1, including final windows.

    The reconstructed terminal numeric state has no corresponding Parquet image.
    Never substitute an action target or invent an image for that missing frame.
    """
    frames = sorted({int(row[key]) - (key == 'end') for row in rows for key in ('start', 'end')})
    if any(row['end'] <= row['start'] for row in rows) or min(frames) < 0:
        raise ValueError('Invalid window interval')
    return np.asarray(frames, dtype=np.int64)


def extract_frames(rows, cfg, cache):
    import pyarrow.parquet as pq
    import torch
    import torchvision
    from PIL import Image
    from torchvision import transforms

    torch.set_num_threads(cfg['threads'])
    weights = Path(cfg['encoder_weights'])
    if not weights.is_file():
        raise FileNotFoundError(f'Local pretrained ResNet18 weights required: {weights}; no automatic download')
    device = torch.device(cfg['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable; select device=cpu')
    spec = {'encoder': 'torchvision.resnet18', 'pretraining': 'ImageNet-1K',
            'weights_sha256': file_hash(weights), 'weights_path': str(weights),
            'pooling': 'global_average_pool_before_fc', 'feature_dim': 512,
            'camera': cfg['camera'], 'frame_policy': 'start_and_end_minus_one',
            'preprocessing': 'PIL RGB; resize full image to 224x224 bilinear; ImageNet mean/std; no crop',
            'torch': torch.__version__, 'torchvision': torchvision.__version__,
            'device': str(device)}
    cache.mkdir(parents=True, exist_ok=True)
    manifest = cache / 'encoder.json'
    if manifest.exists() and json.loads(manifest.read_text()) != spec:
        raise ValueError('Visual cache encoder differs; use a new feature_cache directory')
    write_json(manifest, spec)
    model = torchvision.models.resnet18(weights=None)
    model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
    model.fc = torch.nn.Identity()
    model.eval().requires_grad_(False).to(device)
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    groups = {}
    for row in rows:
        groups.setdefault(row['source_path'], []).append(row)
    total = sum(len(episode_frames(group)) for group in groups.values())
    progress = Progress('图片编码（含可复用缓存）', total, '张')
    first = np.empty((len(rows), 512), dtype=np.float32)
    last = np.empty_like(first)
    done, reused = 0, 0
    with torch.inference_mode():
        for source, group in groups.items():
            path = Path(source)
            stat = path.stat()
            frames = episode_frames(group)
            identity = {'source_path': str(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
            destination = cache / (hashlib.sha256(str(path).encode()).hexdigest() + '.npz')
            if destination.exists():
                with np.load(destination, allow_pickle=False) as saved:
                    if json.loads(str(saved['source_identity'])) != identity or not np.array_equal(saved['frames'], frames):
                        raise ValueError(f'Stale visual cache: {destination}; use a new cache directory')
                    features = saved['features'].copy()
                reused += len(frames)
            else:
                parquet = pq.ParquetFile(path)
                if frames[-1] >= parquet.metadata.num_rows:
                    raise ValueError(f'Window image index out of range: {source}')
                column = parquet.read(columns=[cfg['camera']])[cfg['camera']]
                encoded = []
                for offset in range(0, len(frames), cfg['batch_size']):
                    tensors = []
                    for frame in frames[offset:offset + cfg['batch_size']]:
                        item = column[int(frame)].as_py()
                        if not item or not item.get('bytes'):
                            raise ValueError(f'{source} frame {frame}: embedded image bytes required')
                        with Image.open(BytesIO(item['bytes'])) as image:
                            tensors.append(transform(image.convert('RGB')))
                    encoded.append(model(torch.stack(tensors).to(device)).cpu().numpy())
                features = np.concatenate(encoded).astype(np.float32)
                temporary = destination.with_suffix('.partial')
                with temporary.open('wb') as stream:
                    np.savez_compressed(stream, frames=frames, features=features,
                                        source_identity=json.dumps(identity, sort_keys=True))
                temporary.replace(destination)
            if features.shape != (len(frames), 512) or not np.isfinite(features).all():
                raise ValueError(f'Invalid visual cache: {destination}')
            lookup = {int(frame): i for i, frame in enumerate(frames)}
            for row in group:
                first[row['window_id']] = features[lookup[row['start']]]
                last[row['window_id']] = features[lookup[row['end'] - 1]]
            done += len(frames)
            progress.update(done)
    return first, last, {**spec, 'unique_encoded_frames': total, 'reused_frames': reused,
                         'episode_count': len(groups), 'cache': str(cache)}


def evaluate_motion(source, rows):
    train = np.asarray([i for i, row in enumerate(rows) if row['split'] == 'train'])
    val = np.asarray([i for i, row in enumerate(rows) if row['split'] == 'validation'])
    action = np.load(source / 'windows/action.npy')
    with np.load(source / 'training/normalization.npz') as stats:
        normalized = (action - stats['action_mean']) / stats['action_std']
    groups = {'position': [0, 1, 2, 6, 7, 8], 'rotation': [3, 4, 5, 9, 10, 11],
              'gripper': [12, 13], 'gripper_delta': [14, 15]}
    features = {key: normalized[:, :, columns].reshape(len(rows), -1) for key, columns in groups.items()}

    def score(labels, k):
        result = {}
        for key, values in features.items():
            centers = np.stack([values[train[labels[train] == c]].mean(axis=0) for c in range(k)])
            baseline = np.square(values[val] - values[train].mean(axis=0)).sum()
            result[key] = float(1 - np.square(values[val] - centers[labels[val]]).sum() / baseline)
        return result

    return train, val, score


def run(config_path):
    from .clustering import cluster, score_sample

    config_path = Path(config_path).resolve()
    cfg = json.loads(config_path.read_text())
    for key in ('source_run', 'output_dir', 'feature_cache', 'encoder_weights'):
        cfg[key] = str((config_path.parent / cfg[key]).resolve())
    source, output = Path(cfg['source_run']), Path(cfg['output_dir'])
    if output.exists():
        raise FileExistsError(f'Result directory already exists: {output}; use a new output_dir')
    if not 0 < cfg['visual_distance_weight'] <= 0.1:
        raise ValueError('Auxiliary visual_distance_weight must be >0 and <=0.1')
    if any(not 0 <= w <= 0.1 for w in cfg['comparison_weights']):
        raise ValueError('Auxiliary comparison weights must be in [0,0.1]')
    if min(cfg['batch_size'], cfg['threads']) < 1:
        raise ValueError('batch_size and threads must be positive')
    started = monotonic()
    rows = read_index(source / 'windows')
    if [row['window_id'] for row in rows] != list(range(len(rows))):
        raise ValueError('Window IDs must match embedding row order')
    meta = json.loads((source / 'windows/metadata.json').read_text())
    index_hash = file_hash(source / 'windows/index.jsonl')
    motion_meta = json.loads((source / 'embeddings/metadata.json').read_text())
    if index_hash != meta['index_sha256'] or index_hash != motion_meta['index_sha256']:
        raise ValueError('Motion embeddings/window index provenance mismatch')
    motion = np.load(source / 'embeddings/z.npy')
    if motion.shape != (len(rows), 64) or meta['action_dim'] != 16:
        raise ValueError('This auxiliary experiment requires the reviewed 64D dual-arm motion baseline')
    print(f"基线: {source}\n输出: {output}\n{len(rows)} 个窗口；冻结 ResNet18 + 8 维视觉辅助；"
          f"视觉距离系数 {cfg['visual_distance_weight']:.1%}；不重新训练动作网络。", flush=True)
    extraction_started = monotonic()
    first, last, encoder = extract_frames(rows, cfg, Path(cfg['feature_cache']))
    extraction_seconds = monotonic() - extraction_started
    train, val, motion_score = evaluate_motion(source, rows)
    with threadpool_limits(limits=cfg['threads']):
        with timed('训练集视觉 PCA 和低权重融合'):
            visual, parameters, explained = visual_codes(first, last, train, cfg['visual_dim'],
                                                         cfg['context_weight'], cfg['seed'])
    fused = fuse(motion, visual, cfg['visual_distance_weight'])
    fresh_dir(output)
    # Share read-only inputs/checkpoint; no duplication of the original image dataset.
    (output / 'windows').symlink_to((source / 'windows').resolve(), target_is_directory=True)
    (output / 'training').symlink_to((source / 'training').resolve(), target_is_directory=True)
    (output / 'embeddings').mkdir()
    np.save(output / 'embeddings/z.npy', fused)
    np.save(output / 'embeddings/visual.npy', visual)
    np.savez(output / 'embeddings/visual_pca.npz', **parameters)
    write_json(output / 'visual_config.json', cfg)
    write_json(output / 'embeddings/metadata.json', {
        'index_sha256': index_hash, 'window_count': len(rows), 'latent_dim': fused.shape[1],
        'motion_dim': motion.shape[1], 'visual_dim': visual.shape[1],
        'method': 'frozen_motion_plus_frozen_resnet18_train_only_pca_late_fusion',
        'motion_source': str(source), 'motion_embedding_sha256': file_hash(source / 'embeddings/z.npy'),
        'encoder': encoder, 'pca_fit_split': 'train', 'pca_explained_variance_ratio_sum': explained,
        'visual_distance_weight': cfg['visual_distance_weight'], 'visual_context_weight': cfg['context_weight'],
        'distance_formula': '(1-alpha)*squared_motion_distance + alpha*squared_visual_distance',
        'images': 'actual Parquet observations at start and end-1; not the terminal numeric state',
        'language_used': False, 'object_state_used': False,
    })
    with (output / 'embeddings/image_provenance.jsonl').open('w') as stream:
        for row in rows:
            stream.write(json.dumps({'window_id': row['window_id'], 'source_path': row['source_path'],
                                     'camera': cfg['camera'], 'frames': [row['start'], row['end'] - 1]}) + '\n')
    base_config = json.loads((source / 'training/config.json').read_text())
    base_config['output_dir'] = str(output)
    base_config['clustering'].update(k_values=[cfg['k']], hdbscan=False)
    base_config['training']['threads'] = cfg['threads']
    with timed('导出视觉辅助聚类'):
        cluster(base_config)
    results = []
    comparisons = fresh_dir(output / 'comparison')
    baseline = np.load(source / f"clusters/kmeans_k{cfg['k']}/labels.npy")
    rng = np.random.default_rng(cfg['seed'])
    pairs = rng.choice(val, (20000, 2), replace=True)
    dm = np.square(motion[pairs[:, 0]] - motion[pairs[:, 1]]).sum(axis=1)
    dv = np.square(visual[pairs[:, 0]] - visual[pairs[:, 1]]).sum(axis=1)
    with threadpool_limits(limits=cfg['threads']):
        for weight in sorted(set([*cfg['comparison_weights'], cfg['visual_distance_weight']])):
            with timed(f'对照评估：视觉距离系数 {weight:.0%}'):
                z = fuse(motion, visual, weight)
                if weight == cfg['visual_distance_weight']:
                    labels = np.load(output / f"clusters/kmeans_k{cfg['k']}/labels.npy")
                elif weight == 0:
                    labels = baseline
                else:
                    km = KMeans(n_clusters=cfg['k'], n_init=10, random_state=cfg['seed']).fit(z[train])
                    labels = km.predict(z)
                stability = []
                for seed in (0, 1, 2):
                    other = KMeans(n_clusters=cfg['k'], n_init=10, random_state=seed).fit(z[train]).predict(z[val])
                    stability.append(float(adjusted_rand_score(labels[val], other)))
                sizes = np.bincount(labels, minlength=cfg['k'])
                distance = (1 - weight) * dm + weight * dv
                nonzero = distance > 1e-12
                result = {'visual_distance_weight': weight, 'dimensions': z.shape[1],
                          'validation_silhouette': score_sample(z, labels, val, cfg['seed'], 2000),
                          'validation_motion_space_silhouette': score_sample(motion, labels, val, cfg['seed'], 2000),
                          'validation_ari_to_motion_baseline': float(adjusted_rand_score(baseline[val], labels[val])),
                          'kmeans_seed_ari': stability, 'kmeans_seed_ari_mean': float(np.mean(stability)),
                          'motion_feature_r2': motion_score(labels, cfg['k']),
                          'cluster_size_min': int(sizes.min()), 'cluster_size_max': int(sizes.max()),
                          'mean_pairwise_visual_distance_fraction': float(np.mean(weight * dv[nonzero] / distance[nonzero]))}
                results.append(result)
                np.save(comparisons / f'labels_visual_{weight:.3f}.npy', labels)
                write_json(comparisons / 'metrics.json', results)
                print(f"视觉 {weight:.0%}: silhouette={result['validation_silhouette']:.3f}, "
                      f"初始化稳定性 ARI={result['kmeans_seed_ari_mean']:.3f}, "
                      f"与纯动作结果 ARI={result['validation_ari_to_motion_baseline']:.3f}", flush=True)
    timings = {'image_extraction_seconds': extraction_seconds, 'total_seconds': monotonic() - started}
    write_json(output / 'timing_visual.json', timings)
    summary_lines = [
        '# 低权重视觉辅助聚类', '',
        f"采用 {cfg['visual_distance_weight']:.0%} 视觉距离系数，64 维动作 + {cfg['visual_dim']} 维视觉。", '',
        'ResNet18 冻结；主相机窗口首帧和末帧（end-1）；训练集拟合 PCA；不使用语言或任务标签。',
        '视觉包含 4 维画面上下文和 4 维画面变化，内部距离系数分别为 25% 和 75%。',
        '图像变化也可能来自机械臂或遮挡，不能等同于物体交互效果。', '',
        '| 视觉距离系数 | 验证轮廓系数 | 动作空间轮廓系数 | K-means 稳定性 ARI | 与纯动作 ARI |',
        '|---|---|---|---|---|',
    ]
    for r in results:
        summary_lines.append(f"| {r['visual_distance_weight']:.0%} | {r['validation_silhouette']:.4f} | "
                             f"{r['validation_motion_space_silhouette']:.4f} | {r['kmeans_seed_ari_mean']:.4f} | "
                             f"{r['validation_ari_to_motion_baseline']:.4f} |")
    summary_lines.extend(['', '距离系数是显式的几何权重，不是语义贡献百分比或准确率。',
                          '不同特征空间的轮廓系数不能单独用来判定语义质量提高；仍需检查代表及随机验证片段。',
                          f'windows/ 和 training/ 链接到 {source}，需保留原目录。',
                          'training/ 是原有动作编码器；新增视觉步骤由 visual_config.json 和 embeddings/ 中的 PCA 参数记录。',
                          f"总耗时：{duration(timings['total_seconds'])}；图片编码：{duration(extraction_seconds)}。"])
    (output / 'report.md').write_text('\n'.join(summary_lines) + '\n')
    print(f"完成：{output} | 总耗时 {duration(timings['total_seconds'])}", flush=True)
    return results
