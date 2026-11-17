"""Prepare and sequentially train seven independent cluster LoRA experts."""

import argparse
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

from common import (CLUSTERS, HERE, PI0, ROBOTWIN, latest_checkpoint, parse_cluster_steps,
                    read_settings, setup, write_json)


def invoke(script, arguments, log_dir, *, cpu=False):
    env = os.environ.copy()
    env['JAX_PLATFORMS'] = 'cpu' if cpu else 'cuda'
    env['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    env['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.85'
    env.setdefault('XLA_FLAGS', '--xla_gpu_autotune_level=0')
    env.setdefault('OMP_NUM_THREADS', '4')
    env['PYTHONPATH'] = os.pathsep.join([str(HERE), str(HERE.parent), str(PI0 / 'src'),
                                        env.get('PYTHONPATH', '')])
    torch_lib = PI0 / '.venv/lib/python3.11/site-packages/torch/lib'
    env['LD_LIBRARY_PATH'] = str(torch_lib) + os.pathsep + env.get('LD_LIBRARY_PATH', '')
    log_dir.mkdir(parents=True, exist_ok=True)
    cluster = arguments[arguments.index('--cluster') + 1] if '--cluster' in arguments else 'all'
    log = log_dir / f"{Path(script).stem}_{cluster}_{datetime.now():%Y%m%d_%H%M%S_%f}.log"
    command = [sys.executable, '-u', str(HERE / script), *map(str, arguments)]
    print(f"\nCommand: {' '.join(command)}\nLog: {log}", flush=True)
    with log.open('w') as stream:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   env=env, cwd=ROBOTWIN, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end='', flush=True); stream.write(line); stream.flush()
            code = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
            raise
    if code:
        raise SystemExit(f'{script} failed with exit code {code}; inspect {log}')


def main():
    parser = argparse.ArgumentParser(description='Seven independent π0 LoRA experts from clustered EE windows')
    parser.add_argument('stage', choices=['prepare', 'check', 'train'])
    parser.add_argument('--run-name', default='cluster7_lora_v1')
    parser.add_argument('--config', type=Path, default=HERE / 'config.json')
    parser.add_argument('--clusters', nargs='+', type=int, choices=CLUSTERS, default=list(CLUSTERS))
    parser.add_argument('--gpu', default=os.environ.get('CUDA_VISIBLE_DEVICES', '0'))
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--epochs', type=float,
                        help='Set every cluster step count from its train windows and the effective batch size')
    parser.add_argument('--train-steps', type=int, help='Override steps for every selected cluster')
    parser.add_argument('--cluster-step', action='append', help='Per-cluster override, repeat CLUSTER=STEPS')
    parser.add_argument('--save-interval', type=int)
    parser.add_argument('--keep-period', type=int, help='Retain checkpoints at this period; default retains latest only')
    parser.add_argument('--fsdp-devices', type=int)
    parser.add_argument('--num-workers', type=int)
    parser.add_argument('--peak-lr', type=float)
    parser.add_argument('--warmup-steps', type=int)
    parser.add_argument('--dataset-root', type=Path)
    parser.add_argument('--pretrained-params', type=Path)
    parser.add_argument('--assets-root', type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.run_name):
        parser.error('run-name must be a simple name, not a path')
    if not re.fullmatch(r'\d+(,\d+)*', args.gpu) or len(set(args.gpu.split(','))) != len(args.gpu.split(',')):
        parser.error('Select distinct GPU indices, for example --gpu 0,1,2,3')
    if len(set(args.clusters)) != len(args.clusters):
        parser.error('clusters must be unique')
    settings = read_settings(args.config)
    for key, value in [('batch_size', args.batch_size), ('save_interval', args.save_interval),
                       ('fsdp_devices', args.fsdp_devices), ('num_workers', args.num_workers)]:
        if value is not None:
            if value <= 0 and key != 'num_workers':
                parser.error(f'{key} must be positive')
            if key == 'num_workers' and value < 0:
                parser.error('num_workers must be nonnegative')
            settings[key] = value
    if args.keep_period is not None:
        if args.keep_period <= 0:
            parser.error('keep-period must be positive')
        settings['keep_period'] = args.keep_period
    if args.peak_lr is not None:
        if args.peak_lr <= 0:
            parser.error('peak-lr must be positive')
        settings['peak_lr'] = args.peak_lr
    if args.warmup_steps is not None:
        if args.warmup_steps < 0:
            parser.error('warmup-steps must be nonnegative')
        settings['warmup_steps'] = args.warmup_steps
    for key, value in [('dataset_root', args.dataset_root), ('pretrained_params', args.pretrained_params),
                       ('assets_root', args.assets_root)]:
        if value is not None:
            settings[key] = str(value.expanduser().resolve())
    if args.epochs is not None:
        if args.epochs <= 0 or args.train_steps is not None or args.cluster_step:
            parser.error('--epochs must be positive and cannot be combined with --train-steps/--cluster-step')
        for cluster in CLUSTERS:
            manifest_path = Path(settings['dataset_root']) / f'cluster_{cluster:02d}_pi0_ee/meta/cluster_export.json'
            try:
                train_windows = int(json.loads(manifest_path.read_text())['train_windows'])
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                parser.error(f'Cannot determine train windows for cluster {cluster}: {manifest_path}: {exc}')
            settings['cluster_train_steps'][str(cluster)] = math.ceil(
                args.epochs * train_windows / settings['batch_size'])
        settings['target_epochs'] = args.epochs
    if args.train_steps is not None:
        if args.train_steps <= 0:
            parser.error('train-steps must be positive')
        for cluster in args.clusters:
            settings['cluster_train_steps'][str(cluster)] = args.train_steps
    for cluster, steps in parse_cluster_steps(args.cluster_step, parser).items():
        settings['cluster_train_steps'][str(cluster)] = steps
    gpu_count = len(args.gpu.split(','))
    if args.stage == 'train' and (settings['batch_size'] % gpu_count
                                  or gpu_count % settings.get('fsdp_devices', 1)):
        parser.error('Global batch_size must be divisible by GPU count; fsdp-devices must divide GPU count')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    setup(settings)
    if args.stage == 'check':
        invoke('gpu_check.py', [], HERE / 'logs')
        return
    if args.stage == 'prepare':
        snapshot = HERE / 'logs' / f'prepare_settings_{datetime.now():%Y%m%d_%H%M%S_%f}.json'
        write_json(snapshot, settings)
        invoke('prepare.py', ['--settings', snapshot, '--clusters', *args.clusters], HERE / 'logs', cpu=True)
        return
    if not (Path(settings['pretrained_params']) / '_METADATA').is_file():
        parser.error(f'Missing π0 base checkpoint: {settings["pretrained_params"]}')
    print(f"Cluster LoRA training: clusters={args.clusters}, global batch={settings['batch_size']}, "
          f"GPUs={args.gpu}, batch/GPU={settings['batch_size']//gpu_count}, lr={settings['peak_lr']}", flush=True)
    run_dir = HERE / 'runs' / args.run_name
    config_path = run_dir / 'run_config.json'
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = (run_dir / '.run.lock').open('a')
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error('This run-name is already active')
    if config_path.exists():
        previous = read_settings(config_path)
        if previous != settings:
            parser.error('Run configuration differs; use a new run-name or exactly matching arguments')
        if not args.resume:
            parser.error('Run already exists; pass --resume or choose a new run-name')
    else:
        write_json(config_path, settings)
    log_dir = run_dir / 'logs'
    invoke('prepare.py', ['--settings', config_path, '--clusters', *args.clusters], log_dir, cpu=True)
    for cluster in args.clusters:
        marker = run_dir / 'status' / f'cluster_{cluster:02d}_trained.json'
        steps = settings['cluster_train_steps'][str(cluster)]
        if marker.is_file() and args.resume:
            latest_checkpoint(run_dir, cluster, steps)
            print(f'Cluster {cluster:02d}: verified completed checkpoint; skipping', flush=True)
            continue
        command = ['--cluster', cluster, '--run-dir', run_dir]
        if args.resume:
            command.append('--resume')
        invoke('train_task.py', command, log_dir)
    completed = {}
    for cluster in CLUSTERS:
        marker = run_dir / 'status' / f'cluster_{cluster:02d}_trained.json'
        if marker.is_file():
            checkpoint = latest_checkpoint(run_dir, cluster, settings['cluster_train_steps'][str(cluster)])
            completed[str(cluster)] = str(checkpoint)
    write_json(run_dir / 'training_summary.json', {
        'clusters_requested_this_invocation': args.clusters,
        'completed_checkpoints': completed,
        'all_configured_clusters_complete': set(map(int, completed)) == set(CLUSTERS),
        'settings': settings})
    print(f'Completed cluster LoRA training: {run_dir}', flush=True)
    lock.close()


if __name__ == '__main__':
    main()
