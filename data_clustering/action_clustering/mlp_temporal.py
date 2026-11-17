"""MLP temporal consistency and train-only distance calibration utilities."""

from collections import defaultdict

import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch import nn
from torch.nn import functional as F

from .config import fresh_dir, write_json
from .learning import seed_all
from .model import ActionRepresentation
from .progress import Progress

def episode_steps(episode):
    """Each row describes t -> t+1 in the EE body frame at t, not at window start."""
    pose, grip = episode['ee_pose'], episode['gripper']
    parts = []
    for arm in range(2):
        rotations = Rotation.from_quat(pose[:, arm, 3:])
        translation = rotations[:-1].inv().apply(np.diff(pose[:, arm, :3], axis=0))
        angular = (rotations[:-1].inv() * rotations[1:]).as_rotvec()
        parts.append(np.concatenate([translation, angular], axis=1))
    return np.concatenate([*parts, grip[:-1], np.diff(grip, axis=0)], axis=1).astype(np.float32)


def chains_and_pairs(rows, stride):
    """Do not bridge episodes, split changes, missing windows, or reordered rows."""
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row['episode_id']].append(i)
    chains, pairs = [], []
    for ids in groups.values():
        if len({rows[i]['split'] for i in ids}) != 1:
            raise ValueError('One episode appears in multiple splits')
        ids.sort(key=lambda i: rows[i]['start'])
        chain = []
        for i in ids:
            if chain and rows[i]['start'] - rows[chain[-1]]['start'] != stride:
                chains.append(np.asarray(chain)); chain = []
            if chain:
                pairs.append((chain[-1], i))
            chain.append(i)
        if chain:
            chains.append(np.asarray(chain))
    return chains, np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def make_steps(windows, rows, destination):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row['episode_cache']].append(i)
    length = rows[0]['end'] - rows[0]['start']
    array = np.lib.format.open_memmap(destination, mode='w+', dtype='float32', shape=(len(rows), length, 16))
    progress = Progress('构造共享时间坐标的逐步运动特征', len(groups), '条轨迹')
    for count, (cache, ids) in enumerate(groups.items(), 1):
        with np.load(windows / cache) as episode:
            values = episode_steps(episode)
        for i in ids:
            array[i] = values[rows[i]['start']:rows[i]['end']]
        progress.update(count)
    array.flush()
    if not np.isfinite(array).all():
        raise ValueError('Non-finite local motion features')
    return array


def fit_gate(steps, pairs, train_mask, stride):
    """Heuristic event evidence between window centers; thresholds are train-only.

    These are motion-change proxies, not annotated skill boundaries.
    """
    length = steps.shape[1]
    if not len(pairs) or stride > length // 2:
        raise ValueError('Pairs and at least 50% window overlap are required')
    # Center of left window to center of right window, with one preceding sample.
    segment = np.asarray(steps[pairs[:, 0], length // 2 - 1:length // 2 + stride])
    position = segment[:, :, [0, 1, 2, 6, 7, 8]].reshape(len(pairs), -1, 2, 3)
    rotation = segment[:, :, [3, 4, 5, 9, 10, 11]].reshape(len(pairs), -1, 2, 3)
    evidence = np.stack([
        np.abs(segment[:, 1:, 14:16]).max(axis=(1, 2)),
        np.linalg.norm(np.diff(position, axis=1), axis=-1).max(axis=(1, 2)),
        np.linalg.norm(np.diff(rotation, axis=1), axis=-1).max(axis=(1, 2)),
    ], axis=1)
    train_pairs = train_mask[pairs[:, 0]]
    if not train_pairs.any():
        raise ValueError('No training pairs')
    scales = np.maximum(np.quantile(evidence[train_pairs], 0.90, axis=0), 1e-6)
    score = np.max(evidence / scales, axis=1)
    weights = np.exp(-3 * np.minimum(score, 20) ** 2).astype(np.float32)
    return score, weights, {'features': ['gripper_step', 'translation_increment_change', 'rotation_increment_change'],
                            'train_q90': scales.tolist(), 'weight': 'exp(-3 * score^2)',
                            'quiet_threshold': 0.5, 'event_threshold': 1.0,
                            'meaning': 'heuristic motion event proxy, NOT ground-truth skill boundaries'}


def temporal_metrics(labels, pairs, score, val_mask, chains):
    valid = val_mask[pairs[:, 0]]
    changed = labels[pairs[:, 0]] != labels[pairs[:, 1]]
    result = {}
    for name, mask in [('all', valid), ('quiet', valid & (score <= .5)), ('event', valid & (score >= 1))]:
        result[name + '_pairs'] = int(mask.sum())
        result[name + '_switch_rate'] = float(changed[mask].mean()) if mask.any() else None
    durations = []
    for ids in chains:
        if not val_mask[ids[0]]:
            continue
        cuts = np.r_[0, np.flatnonzero(np.diff(labels[ids])) + 1, len(ids)]
        durations.extend(np.diff(cuts).tolist())
    result['median_run_windows'] = float(np.median(durations)) if durations else None
    result['single_window_run_fraction'] = float(np.mean(np.asarray(durations) == 1)) if durations else None
    return result


def motion_r2(action, labels, train, val, k):
    counts = np.bincount(labels[train], minlength=k)
    missing = np.flatnonzero(counts == 0)
    groups = {'position': [0, 1, 2, 6, 7, 8], 'rotation': [3, 4, 5, 9, 10, 11],
              'gripper': [12, 13], 'gripper_delta': [14, 15]}
    result = {}
    for name, columns in groups.items():
        # No validation-fitted centers and no fabricated score for unseen train clusters.
        if np.isin(labels[val], missing).any():
            result[name] = None
            continue
        values = action[:, :, columns].reshape(len(labels), -1)
        centers = np.zeros((k, values.shape[1]), dtype=np.float32)
        for c in np.flatnonzero(counts):
            centers[c] = values[train[labels[train] == c]].mean(0)
        denominator = np.square(values[val] - values[train].mean(0)).sum()
        result[name] = float(1 - np.square(values[val] - centers[labels[val]]).sum() / denominator) if denominator > 0 else None
    return result


def diagnostics(z, train):
    singular = np.linalg.svd(z[train] - z[train].mean(0), compute_uv=False)
    energy = singular ** 2
    if energy.sum() == 0:
        return {'mean_feature_std': 0.0, 'effective_rank': 0.0}
    p = energy / energy.sum()
    return {'mean_feature_std': float(z[train].std(0).mean()),
            'effective_rank': float(np.exp(-(p * np.log(np.maximum(p, 1e-30))).sum()))}


def sample_pairs(ids, count=10000, seed=42):
    if len(ids) < 2:
        raise ValueError('At least two windows are needed for distance diagnostics')
    rng = np.random.default_rng(seed)
    a, b = rng.choice(ids, count), rng.choice(ids, count)
    return np.stack([a[a != b], b[a != b]], axis=1)


def distance_contribution(motion, visual, pairs, alpha):
    m = np.square(motion[pairs[:, 0]] - motion[pairs[:, 1]]).sum(-1)
    v = np.square(visual[pairs[:, 0]] - visual[pairs[:, 1]]).sum(-1)
    numerator, total = alpha * v, (1 - alpha) * m + alpha * v
    valid = total > 1e-12
    return {'pair_count': len(pairs), 'nonzero_distance_pairs': int(valid.sum()), 'alpha': float(alpha),
            'raw_motion_distance2_mean': float(m.mean()), 'raw_visual_distance2_mean': float(v.mean()),
            'visual_mean_pair_fraction': float((numerator[valid] / total[valid]).mean()) if valid.any() else None,
            'visual_ratio_of_mean_distances': float(numerator.mean() / total.mean()) if total.mean() > 1e-12 else None}


def calibrate_alpha(motion, visual, train_pairs, target=.05):
    """Fit the ratio of mean squared distances on TRAIN pairs, never validation."""
    if not 0 < target < 1:
        raise ValueError('Target contribution must be in (0,1)')
    measurements = distance_contribution(motion, visual, train_pairs, .05)
    m, v = measurements['raw_motion_distance2_mean'], measurements['raw_visual_distance2_mean']
    if min(m, v) <= 1e-12:
        raise ValueError('Degenerate representations cannot be distance-calibrated')
    unconstrained = target * m / ((1 - target) * v + target * m)
    alpha = min(unconstrained, .1)
    return alpha, {'fit_split': 'train', 'target_ratio_of_mean_distances': target,
                   'unconstrained_alpha': unconstrained, 'alpha_cap': .1, 'cap_active': unconstrained > .1,
                   'achieved_train': distance_contribution(motion, visual, train_pairs, alpha)}


def guarded_weights(action, pairs, weights, threshold=.2):
    """Optional ablation: do not pull whole windows together across gripper activity."""
    activity = np.abs(action[:, :, 14:16]).sum(axis=1).max(axis=1)
    mask = (activity[pairs[:, 0]] >= threshold) | (activity[pairs[:, 1]] >= threshold)
    return np.where(mask, 0, weights).astype(np.float32), mask


def train_mlp(out, coefficient, cfg, rows, arrays, stats, pairs, weights, device, guard=False):
    seed_all(cfg['seed'], cfg['training']['threads'])
    tr = cfg['training']
    model = ActionRepresentation(cfg['window']['length'], 16, 14, 0, tr['hidden_dim'], 64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tr['lr'])
    ids = {s: np.asarray([i for i, row in enumerate(rows) if row['split'] == s]) for s in ('train', 'validation')}
    neighbor = np.arange(len(rows)); edge = np.zeros(len(rows), dtype=np.float32)
    neighbor[pairs[:, 0]] = pairs[:, 1]; edge[pairs[:, 0]] = weights
    loaders = {s: torch.utils.data.DataLoader(torch.as_tensor(values), batch_size=tr['batch_size'], shuffle=s == 'train',
                                             generator=torch.Generator().manual_seed(cfg['seed'])) for s, values in ids.items()}
    training = fresh_dir(out / 'training')
    np.savez(training / 'normalization.npz', **stats)
    write_json(training / 'config.json', {**cfg, 'output_dir': str(out)})
    parameters = {'model': 'MLP', 'temporal_weight': coefficient, 'gripper_guard': guard,
                  'gripper_guard_threshold': .2 if guard else None, 'seed': cfg['seed'],
                  'selection': 'minimum validation action+state reconstruction MSE',
                  'epochs': tr['epochs'], 'hidden': tr['hidden_dim'], 'latent': 64,
                  'optimizer': 'AdamW weight_decay=0.01, gradient_clip=1.0', 'device': device}
    write_json(training / 'experiment.json', parameters)

    def forward(selection):
        inputs = [torch.as_tensor((arrays[key][selection] - stats[key + '_mean']) / stats[key + '_std'],
                                  dtype=torch.float32, device=device) for key in ('action', 'state')]
        return model(*inputs, torch.empty(0, device=device)), inputs

    progress = Progress(out.name, tr['epochs'] * sum(len(x) for x in loaders.values()), '批次')
    best, history, done = float('inf'), [], 0
    for epoch in range(1, tr['epochs'] + 1):
        record = {'epoch': epoch}
        for split, loader in loaders.items():
            model.train(split == 'train')
            sums = np.zeros(4)
            for anchors in loader:
                anchors = anchors.numpy()
                with torch.set_grad_enabled(split == 'train'):
                    (z, pred_action, pred_state), inputs = forward(anchors)
                    action_loss, state_loss = F.mse_loss(pred_action, inputs[0]), F.mse_loss(pred_state, inputs[1])
                    temporal = torch.zeros((), device=device)
                    if coefficient:
                        other = forward(neighbor[anchors])[0][0]
                        temporal = (torch.as_tensor(edge[anchors], device=device) *
                                    (1 - (z * other).sum(-1)).clamp_min(0)).mean()
                    loss = action_loss + state_loss + coefficient * temporal
                    if not torch.isfinite(loss):
                        raise ValueError('Non-finite loss')
                    if split == 'train':
                        optimizer.zero_grad(set_to_none=True); loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 1.)
                        optimizer.step()
                sums += np.asarray([x.item() for x in (loss, action_loss, state_loss, temporal)]) * len(anchors)
                done += 1; progress.update(done)
            for key, value in zip(('loss', 'action_loss', 'state_loss', 'temporal_loss'), sums / len(ids[split])):
                record[split + '_' + key] = float(value)
        reconstruction = record['validation_action_loss'] + record['validation_state_loss']
        checkpoint = {'model': model.state_dict(), 'epoch': epoch, 'parameters': parameters,
                      'validation_reconstruction': reconstruction}
        if reconstruction < best:
            best = reconstruction; torch.save(checkpoint, training / 'best.pt')
        history.append(record); write_json(training / 'history.json', history)
        print(f"{out.name} epoch {epoch}/{tr['epochs']}: val reconstruction={reconstruction:.6f}, "
              f"temporal={record['validation_temporal_loss']:.6f}", flush=True)
    torch.save(checkpoint, training / 'last.pt')
    selected = torch.load(training / 'best.pt', map_location=device, weights_only=True)
    model.load_state_dict(selected['model']); model.eval()
    embeddings = fresh_dir(out / 'embeddings')
    z_all = np.empty((len(rows), 64), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(rows), tr['batch_size']):
            selection = np.arange(start, min(start + tr['batch_size'], len(rows)))
            z_all[selection] = forward(selection)[0][0].cpu().numpy()
    np.save(embeddings / 'motion.npy', z_all)
    write_json(embeddings / 'metadata.json', {'checkpoint_epoch': selected['epoch'], 'dimensions': 64,
                                             'parameters': parameters, 'motion_spread': diagnostics(z_all, ids['train'])})
    return z_all
