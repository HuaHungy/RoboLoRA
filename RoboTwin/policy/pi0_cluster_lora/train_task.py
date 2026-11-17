"""Train one cluster LoRA expert with start-only window samples."""

import argparse
import json
import logging
from pathlib import Path

from common import CLUSTERS, PI0, build_config, latest_checkpoint, load_module, read_settings, repo_id, setup, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cluster', type=int, choices=CLUSTERS, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    settings = read_settings(args.run_dir / 'run_config.json')
    setup(settings)
    from cluster_data import install_cluster_dataset
    from pi0_ee_five.train_compat import install_train_state_compat
    install_cluster_dataset('train')
    install_train_state_compat()
    import jax
    if not any(device.platform == 'gpu' for device in jax.devices()):
        raise RuntimeError('GPU is unavailable in JAX; refusing to start π0 cluster training on CPU')
    logging.basicConfig(level=logging.INFO)
    cfg = build_config(args.cluster, settings, args.run_dir, args.resume)
    norm_path = cfg.assets_dirs / repo_id(args.cluster) / 'norm_stats.json'
    if not norm_path.is_file():
        raise FileNotFoundError(f'Missing cluster normalization: {norm_path}; run prepare first')
    print(f'Cluster={args.cluster}, samples=start-only, batch={cfg.batch_size}, steps={cfg.num_train_steps}, '
          f'lr={settings["peak_lr"]}, horizon=50, trainable=LoRA-only, EMA disabled', flush=True)
    trainer = load_module('pi0_cluster_upstream_train', PI0 / 'scripts/train.py')
    from pi0_ee_five.train_loop import main as train
    train(trainer, cfg)
    checkpoint = latest_checkpoint(args.run_dir, args.cluster, cfg.num_train_steps)
    write_json(args.run_dir / 'status' / f'cluster_{args.cluster:02d}_trained.json',
               {'cluster_id': args.cluster, 'checkpoint': str(checkpoint), 'steps': cfg.num_train_steps,
                'repo_id': repo_id(args.cluster)})


if __name__ == '__main__':
    main()
