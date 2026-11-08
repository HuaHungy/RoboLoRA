"""Train one mixed-task model, then evaluate that checkpoint on each task."""

import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from common import HERE, JOINT, PI0, ROBOTWIN, TASKS, latest_checkpoint, model_key, read_settings, setup, write_json


def invoke(script, arguments, log_dir, *, cpu=False, evaluation=False):
    env = os.environ.copy()
    if evaluation:
        env['CUDA_VISIBLE_DEVICES'] = env.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0]
    env['JAX_PLATFORMS'] = 'cpu' if cpu else 'cuda'
    env['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    env['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.4' if evaluation else '0.85'
    env.setdefault('ROBOTWIN_CUROBO_USE_CUDA_GRAPH', '0')
    # Match the existing RoboTwin workaround for this driver/toolchain.
    env.setdefault('XLA_FLAGS', '--xla_gpu_autotune_level=0')
    env.setdefault('OMP_NUM_THREADS', '4')
    env['PYTHONPATH'] = os.pathsep.join([str(HERE.parent), str(PI0 / 'src'), env.get('PYTHONPATH', '')])
    torch_lib = PI0 / '.venv/lib/python3.11/site-packages/torch/lib'
    env['LD_LIBRARY_PATH'] = str(torch_lib) + os.pathsep + env.get('LD_LIBRARY_PATH', '')
    log_dir.mkdir(parents=True, exist_ok=True)
    task = arguments[arguments.index('--task') + 1] if '--task' in arguments else 'all'
    log = log_dir / f"{Path(script).stem}_{task}_{datetime.now():%Y%m%d_%H%M%S_%f}.log"
    command = [sys.executable, '-u', str(HERE / script), *map(str, arguments)]
    print(f"\nCommand: {' '.join(command)}\nLog: {log}", flush=True)
    with log.open('w') as stream:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   env=env, cwd=ROBOTWIN, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                stream.write(line); stream.flush()
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
    parser = argparse.ArgumentParser(description='五任务 π0 末端 LoRA 微调与原生 RoboTwin 评测；不修改原有文件')
    parser.add_argument('stage', choices=['prepare', 'check', 'train', 'eval', 'all'])
    parser.add_argument('--run-name', default='ee5_joint_v1')
    parser.add_argument('--config', type=Path, default=HERE / 'config.json')
    parser.add_argument('--tasks', nargs='+', choices=TASKS, default=list(TASKS), help='Evaluation tasks; joint training always uses all five')
    parser.add_argument('--gpu', default=os.environ.get('CUDA_VISIBLE_DEVICES', '0'))
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--train-steps', type=int)
    parser.add_argument('--save-interval', type=int)
    parser.add_argument('--keep-period', type=int)
    parser.add_argument('--fsdp-devices', type=int)
    parser.add_argument('--dataset-root', type=Path)
    parser.add_argument('--pretrained-params', type=Path)
    parser.add_argument('--eval-episodes', type=int)
    parser.add_argument('--eval-seed', type=int)
    parser.add_argument('--checkpoint-step', type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--no-video', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.run_name):
        parser.error('run-name must be a simple name, not a path')
    if not re.fullmatch(r'\d+(,\d+)*', args.gpu) or len(set(args.gpu.split(','))) != len(args.gpu.split(',')):
        parser.error('Select distinct GPU indices, e.g. --gpu 0 or --gpu 0,1,2,3')
    if len(set(args.tasks)) != len(args.tasks):
        parser.error('Each task may only be listed once')
    if (args.eval_episodes is not None and args.eval_episodes < 1) or (args.eval_seed is not None and args.eval_seed < 0):
        parser.error('eval-episodes must be positive; eval-seed must be nonnegative')
    if args.stage in ('train', 'all', 'prepare') and tuple(args.tasks) != TASKS:
        parser.error('Joint training/preparation requires all five tasks; --tasks is for the eval stage only')
    settings = read_settings(args.config)
    for key, value in [('batch_size', args.batch_size), ('num_train_steps', args.train_steps),
                       ('save_interval', args.save_interval), ('keep_period', args.keep_period),
                       ('fsdp_devices', args.fsdp_devices)]:
        if value is not None:
            if value <= 0:
                parser.error(f'{key} must be positive')
            settings[key] = value
    for key, value in [('dataset_root', args.dataset_root), ('pretrained_params', args.pretrained_params)]:
        if value is not None:
            settings[key] = str(value.expanduser().resolve())
    gpu_count = len(args.gpu.split(','))
    if args.stage in ('train', 'all'):
        if settings['batch_size'] % gpu_count or gpu_count % settings.get('fsdp_devices', 1):
            parser.error('Global batch_size must be divisible by GPU count; fsdp-devices must divide GPU count')
        print(f"Joint training: global batch={settings['batch_size']}, GPUs={args.gpu}, "
              f"batch/GPU={settings['batch_size'] // gpu_count}, FSDP={settings.get('fsdp_devices', 1)}", flush=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    setup(settings)
    if args.stage == 'prepare':
        # Snapshot resolved CLI paths too, so preparation uses exactly these inputs.
        snapshot = HERE / 'logs' / f'prepare_settings_{datetime.now():%Y%m%d_%H%M%S_%f}.json'
        write_json(snapshot, settings)
        invoke('prepare.py', ['--settings', str(snapshot)], HERE / 'logs', cpu=True)
        return
    invoke('gpu_check.py', ['--evaluation'] if args.stage in ('eval', 'all') else [], HERE / 'logs')
    if args.stage == 'check':
        return
    run_dir = HERE / 'runs' / args.run_name
    config_path = run_dir / 'run_config.json'
    if args.stage == 'eval' and not config_path.is_file():
        parser.error(f'No saved training configuration: {config_path}')
    run_dir.mkdir(parents=True, exist_ok=True)
    run_lock = (run_dir / '.run.lock').open('a')
    try:
        fcntl.flock(run_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error('This run-name is already active in another process')
    if args.stage == 'eval':
        if not config_path.is_file():
            parser.error(f'No saved training configuration: {config_path}')
        settings = json.loads(config_path.read_text())
    elif config_path.exists():
        previous = json.loads(config_path.read_text())
        if previous != settings:
            parser.error('Run configuration differs from existing run; use a new run-name or matching arguments')
        if not args.resume:
            parser.error('Run already exists; pass --resume to continue, or choose a new run-name')
    else:
        write_json(config_path, settings)
    log_dir = run_dir / 'logs'
    if args.stage in ('train', 'all'):
        # Idempotent CPU validation and stats, never silently changes existing normalization.
        invoke('prepare.py', ['--settings', str(config_path)], log_dir, cpu=True)
        marker = run_dir / 'status' / f'{JOINT}_trained.json'
        if marker.is_file() and args.resume:
            latest_checkpoint(run_dir, JOINT, settings['num_train_steps'])
            print('Joint training checkpoint verified; skipping training', flush=True)
        else:
            command = ['--task', JOINT, '--run-dir', str(run_dir)]
            if args.resume:
                command.append('--resume')
            invoke('train_task.py', command, log_dir)
    for task in args.tasks:
        if args.stage in ('eval', 'all'):
            count = settings['eval_episodes'] if args.eval_episodes is None else args.eval_episodes
            seed = settings['eval_seed'] if args.eval_seed is None else args.eval_seed
            if count < 1 or seed < 0:
                parser.error('eval-episodes must be positive; eval-seed must be nonnegative')
            checkpoint = latest_checkpoint(run_dir, model_key(settings, task), args.checkpoint_step)
            result_path = run_dir / 'evaluations' / task / f'step_{checkpoint.name}_seed_{seed}_n{count}' / 'result.json'
            if result_path.is_file() and args.resume:
                print(f'{task}: completed evaluation found; skipping', flush=True)
            else:
                command = ['--task', task, '--run-dir', str(run_dir), '--episodes', str(count), '--seed', str(seed),
                           '--checkpoint-step', checkpoint.name]
                if args.no_video or not settings['eval_video']:
                    command.append('--no-video')
                if args.resume:
                    command.append('--restart-incomplete')
                invoke('evaluate.py', command, log_dir, evaluation=True)
            reports = [json.loads(p.read_text()) for p in (run_dir / 'evaluations').glob('*/step_*/result.json')]
            write_json(run_dir / 'evaluation_summary.json', reports)
    print(f'Completed stage {args.stage}; outputs: {run_dir}', flush=True)
    run_lock.close()


if __name__ == '__main__':
    main()
