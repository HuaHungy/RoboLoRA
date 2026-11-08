"""Bounded validation sweep, retaining checkpoints and never touching the source run.

Fixed episode split, seed, optimizer, window schema and normalization across trials.
Metrics are diagnostics, not skill ground truth. Run from data_clustering/.
"""

import argparse
import json
from pathlib import Path
import sys
from time import monotonic

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from threadpoolctl import threadpool_limits
from torch.utils.data import DataLoader

from action_clustering.config import write_json
from action_clustering.learning import WindowDataset, seed_all
from action_clustering.model import ActionRepresentation


GROUPS = {'position': [0, 1, 2, 6, 7, 8], 'rotation': [3, 4, 5, 9, 10, 11],
          'gripper': [12, 13], 'gripper_delta': [14, 15]}


def run(source, out, trial_name=None, epochs=None, training_seed=42, k_values=(16, 32, 48)):
    out.mkdir(parents=True, exist_ok=False)
    started = monotonic()
    rows = [json.loads(s) for s in (source / 'windows/index.jsonl').read_text().splitlines()]
    metadata = json.loads((source / 'windows/metadata.json').read_text())
    cfg = json.loads((source / 'training/config.json').read_text())
    ids = {s: np.array([i for i, r in enumerate(rows) if r['split'] == s]) for s in ['train', 'validation']}
    with np.load(source / 'training/normalization.npz') as f: stats = dict(f)
    action = np.load(source / 'windows/action.npy')
    normalized = (action - stats['action_mean']) / stats['action_std']
    original_grip_patterns = np.where(action[:, -1, 12:14] + action[:, -1, 14:16] - action[:, 0, 12:14] > .2, 1,
                                     np.where(action[:, -1, 12:14] + action[:, -1, 14:16] - action[:, 0, 12:14] < -.2, -1, 0))
    grip_classes = (original_grip_patterns[:, 0] + 1) * 3 + original_grip_patterns[:, 1] + 1
    features = {g: normalized[:, :, columns].reshape(len(rows), -1) for g, columns in GROUPS.items()}
    baselines = {g: float(np.square(v[ids['validation']] - v[ids['train']].mean(axis=0)).sum()) for g, v in features.items()}
    # Adjacent windows overlap: report switching, but do not optimize it in isolation.
    adjacent = np.array([(i - 1, i) for i in ids['validation'] if i > 0 and rows[i]['episode_id'] == rows[i - 1]['episode_id']])
    silhouette_ids = np.random.default_rng(42).choice(ids['validation'], min(2000, len(ids['validation'])), replace=False)
    settings = [
        {'name': 'baseline128', 'latent': 128, 'state_weight': 1., 'epochs': [3, 10, 20, 30, 50, 60]},
        {'name': 'latent64', 'latent': 64, 'state_weight': 1., 'epochs': [10, 20, 30, 50]},
        {'name': 'state025', 'latent': 128, 'state_weight': .25, 'epochs': [10, 20, 30, 50]},
    ]
    if trial_name:
        settings = [s for s in settings if s['name'] == trial_name]
    if epochs:
        settings = [{**s, 'epochs': epochs} for s in settings]
    results = []
    for trial in settings:
        folder = out / trial['name']; folder.mkdir()
        seed_all(training_seed, 4)
        args = {'length': 50, 'action_dim': 16, 'state_dim': 14, 'visual_dim': 0,
                'hidden_dim': 256, 'latent_dim': trial['latent'], 'visual_weight': 0.}
        model = ActionRepresentation(**args)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        loaders = {s: DataLoader(WindowDataset(source / 'windows', ix, stats), batch_size=256, shuffle=s == 'train') for s, ix in ids.items()}
        all_loader = DataLoader(WindowDataset(source / 'windows', np.arange(len(rows)), stats), batch_size=256, shuffle=False)
        history = []
        # DataLoader iteration consumes torch RNG even for nonshuffled evaluation.
        # Restore RNG around extra checkpoint diagnostics to preserve training order.
        for epoch in range(1, max(trial['epochs']) + 1):
            losses = {}
            for split, loader in loaders.items():
                training = split == 'train'; model.train(training); sums = np.zeros(2)
                for a, s, v in loader:
                    with torch.set_grad_enabled(training):
                        _, pa, ps = model(a, s, v)
                        la = (pa - a).square().mean(); ls = (ps - s).square().mean()
                        loss = la + trial['state_weight'] * ls
                        if not torch.isfinite(loss): raise ValueError('Non-finite training loss')
                        if training:
                            optimizer.zero_grad(set_to_none=True); loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                            optimizer.step()
                    sums += [la.item() * len(a), ls.item() * len(a)]
                losses[split] = (sums / len(loader.dataset)).tolist()
            history.append({'epoch': epoch, **losses})
            if epoch % 10 == 0 or epoch in trial['epochs']:
                print(f"{trial['name']} epoch {epoch}: validation action/state={losses['validation']}", flush=True)
            if epoch not in trial['epochs']: continue
            state = torch.get_rng_state()
            destination = folder / f'epoch_{epoch:03d}'; destination.mkdir()
            torch.save({'model': model.state_dict(), 'model_args': args, 'epoch': epoch,
                        'validation_loss': losses['validation'][0] + trial['state_weight'] * losses['validation'][1],
                        'index_sha256': metadata['index_sha256']}, destination / 'checkpoint.pt')
            model.eval()
            with torch.inference_mode(): z = np.concatenate([model.encode(*batch).numpy() for batch in all_loader])
            torch.set_rng_state(state)
            assert np.isfinite(z).all() and np.allclose(np.linalg.norm(z, axis=1), 1, atol=1e-5)
            np.save(destination / 'z.npy', z)
            with threadpool_limits(limits=4):
                for k in k_values:
                    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(z[ids['train']])
                    labels = km.predict(z)
                    np.save(destination / f'labels_k{k}.npy', labels)
                    np.save(destination / f'centers_k{k}.npy', km.cluster_centers_)
                    record = {'trial': trial['name'], 'training_seed': training_seed, 'epoch': epoch, 'latent_dim': trial['latent'],
                              'state_loss_weight': trial['state_weight'], 'k': k,
                              'validation_action_mse': losses['validation'][0], 'validation_state_mse': losses['validation'][1],
                              'validation_silhouette': float(silhouette_score(z[silhouette_ids], labels[silhouette_ids])),
                              'cluster_size_min': int(np.bincount(labels, minlength=k).min()),
                              'cluster_size_max': int(np.bincount(labels, minlength=k).max()), 'feature_r2': {},
                              'adjacent_label_switch_rate': float(np.mean(labels[adjacent[:, 0]] != labels[adjacent[:, 1]])),
                              'checkpoint': str(destination / 'checkpoint.pt')}
                    for g, values in features.items():
                        centers = np.stack([values[ids['train'][labels[ids['train']] == c]].mean(axis=0) for c in range(k)])
                        sse = np.square(values[ids['validation']] - centers[labels[ids['validation']]]).sum()
                        record['feature_r2'][g] = float(1 - sse / baselines[g])
                    correct = 0
                    for c in range(k):
                        train_c = ids['train'][labels[ids['train']] == c]
                        majority = np.bincount(grip_classes[train_c], minlength=9).argmax()
                        val_c = ids['validation'][labels[ids['validation']] == c]
                        correct += np.sum(grip_classes[val_c] == majority)
                    record['gripper_event_majority_accuracy'] = float(correct / len(ids['validation']))
                    if k == 32:
                        alternate = KMeans(n_clusters=k, n_init=10, random_state=0).fit(z[ids['train']]).predict(z[ids['validation']])
                        record['seed0_validation_ari'] = float(adjusted_rand_score(labels[ids['validation']], alternate))
                    results.append(record)
                    write_json(out / 'results.json', {'source': str(source.resolve()), 'trials': settings,
                                                     'selection_split': 'validation; no independent test set', 'results': results,
                                                     'elapsed_seconds': monotonic() - started})
                    print(f"  K={k}: silhouette={record['validation_silhouette']:.3f}, feature R2={record['feature_r2']}", flush=True)
            write_json(out / 'results.json', {'source': str(source.resolve()), 'trials': settings,
                                            'selection_split': 'validation; no independent test set', 'results': results,
                                            'elapsed_seconds': monotonic() - started})
            write_json(folder / 'history.json', history)
        write_json(folder / 'history.json', history)
    print(f'Completed {len(results)} candidates in {monotonic() - started:.1f}s -> {out}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path('runs/lerobot_ee_all'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trial', choices=['baseline128', 'latent64', 'state025'])
    parser.add_argument('--epochs', type=int, nargs='+')
    parser.add_argument('--training-seed', type=int, default=42)
    parser.add_argument('--k', type=int, nargs='+', default=[16, 32, 48])
    args = parser.parse_args()
    run(args.source, args.output, args.trial, args.epochs, args.training_seed, args.k)
