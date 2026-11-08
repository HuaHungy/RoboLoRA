"""Run upstream pi0 training with a local EE configuration object."""

import argparse
import json
import logging
from pathlib import Path

from common import JOINT, PI0, TASKS, build_config, latest_checkpoint, load_module, repo_id, setup, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=(*TASKS, JOINT), required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    settings = json.loads((args.run_dir / 'run_config.json').read_text())
    if (args.task == JOINT) != (settings.get('training_mode') == 'joint'):
        raise ValueError('Requested model does not match the saved joint/single-task training mode')
    setup(settings)
    from joint_data import install_joint_dataset
    from train_compat import install_train_state_compat
    install_joint_dataset()
    install_train_state_compat()
    import jax
    if not any(device.platform == 'gpu' for device in jax.devices()):
        raise RuntimeError('GPU is unavailable in JAX; refusing to start full pi0 training on CPU')
    logging.basicConfig(level=logging.INFO)
    cfg = build_config(args.task, settings, args.run_dir, args.resume)
    norm_path = cfg.assets_dirs / repo_id(args.task) / 'norm_stats.json'
    if not norm_path.is_file():
        raise FileNotFoundError(f'Missing EE normalization: {norm_path}; run prepare first')
    print(f'Task={args.task}, batch_size={cfg.batch_size}, steps={cfg.num_train_steps}, '
          f'EE horizon={cfg.model.action_horizon}, vision frozen, EMA disabled', flush=True)
    trainer = load_module('pi0_ee_upstream_train', PI0 / 'scripts/train.py')
    from train_loop import main as train
    train(trainer, cfg)
    checkpoint = latest_checkpoint(args.run_dir, args.task, cfg.num_train_steps)
    write_json(args.run_dir / 'status' / f'{args.task}_trained.json',
               {'task': args.task, 'checkpoint': str(checkpoint), 'steps': cfg.num_train_steps})


if __name__ == '__main__':
    main()
