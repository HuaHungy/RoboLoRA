"""Use RoboTwin's native expert-filtered evaluator with the new EE policy."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import random
import sys
from time import monotonic

from common import HERE, ROBOTWIN, TASKS, latest_checkpoint, load_module, model_key, repo_id, setup, write_json


class SeedSearchLimit(BaseException):
    """Stop the native retry loop even though it catches ordinary exceptions."""


def run(task, run_dir, episodes, seed, checkpoint_step=None, video=True, restart_incomplete=False):
    settings = json.loads((run_dir / 'run_config.json').read_text())
    setup(settings)
    checkpoint = latest_checkpoint(run_dir, model_key(settings, task), checkpoint_step)
    destination = run_dir / 'evaluations' / task / f'step_{checkpoint.name}_seed_{seed}_n{episodes}'
    if destination.exists():
        if restart_incomplete and not (destination / 'result.json').exists():
            archived = destination.with_name(destination.name + datetime.now().strftime('_interrupted_%Y%m%d_%H%M%S_%f'))
            destination.rename(archived)
            print(f'Preserved interrupted evaluation at {archived}; restarting this evaluation', flush=True)
        else:
            raise FileExistsError(f'Evaluation output exists: {destination}; use a different evaluation seed or episode count')
    destination.mkdir(parents=True)
    os.chdir(ROBOTWIN)
    for path in (ROBOTWIN, ROBOTWIN / 'policy', ROBOTWIN / 'script', ROBOTWIN / 'description/utils'):
        sys.path.insert(0, str(path))
    import numpy as np
    import yaml
    import jax
    if not any(d.platform == 'gpu' for d in jax.devices()):
        raise RuntimeError('JAX GPU is required for pi0 evaluation')
    native = load_module('pi0_ee_native_evaluator', ROBOTWIN / 'script/eval_policy.py')
    from pi0_ee_five.deploy_policy import get_model

    args = yaml.safe_load((ROBOTWIN / 'task_config' / f"{settings['task_config']}.yml").read_text())
    args.update(task_name=task, task_config=settings['task_config'], policy_name='pi0_ee_five',
                ckpt_setting=run_dir.name, render_freq=0, eval_mode=True, eval_video_log=video,
                collect_data=False)
    args['data_type'].update(rgb=True, endpose=True)
    args['camera'].update(collect_head_camera=True, collect_wrist_camera=True)
    embodiment = args['embodiment']
    if embodiment != ['aloha-agilex']:
        raise ValueError('The prepared 16D datasets require the demo_aloha dual-arm embodiment')
    mapping = yaml.safe_load((ROBOTWIN / 'task_config/_embodiment_config.yml').read_text())
    robot_file = mapping[embodiment[0]]['file_path']
    args.update(left_robot_file=robot_file, right_robot_file=robot_file, dual_arm_embodied=True,
                left_embodiment_config=native.get_embodiment_config(robot_file),
                right_embodiment_config=native.get_embodiment_config(robot_file))
    camera = native.get_camera_config(args['camera']['head_camera_type'])
    args.update(head_camera_h=camera['h'], head_camera_w=camera['w'])
    if video:
        args['eval_video_save_dir'] = destination
    np.random.seed(seed); random.seed(seed)
    env = native.class_decorator(task)
    model = get_model({'task_name': task, 'run_dir': str(run_dir), 'checkpoint': str(checkpoint)})
    records, attempted_seeds = [], set()
    active = {'seed': None, 'instruction': None}
    original_setup, original_instruction, original_close = env.setup_demo, env.set_instruction, env.close_env

    def tracked_setup(*positional, **kwargs):
        active['seed'] = int(kwargs['seed'])
        attempted_seeds.add(active['seed'])
        if len(attempted_seeds) > max(100, episodes * 20):
            raise SeedSearchLimit('Too many expert-rejected seeds; inspect simulator/planner errors before retrying')
        random.seed(active['seed'])
        return original_setup(*positional, **kwargs)

    def tracked_instruction(*positional, **kwargs):
        active['instruction'] = kwargs.get('instruction', positional[0] if positional else None)
        return original_instruction(*positional, **kwargs)

    def tracked_close(*positional, **kwargs):
        if active['instruction'] is not None:
            records.append({'seed': active['seed'], 'instruction': active['instruction'],
                            'success': bool(env.eval_success), 'actions': int(env.take_action_cnt)})
            write_json(destination / 'episodes.json', records)
            active['instruction'] = None
        return original_close(*positional, **kwargs)

    env.setup_demo, env.set_instruction, env.close_env = tracked_setup, tracked_instruction, tracked_close
    started = monotonic()
    metadata = {'task': task, 'checkpoint': str(checkpoint), 'requested_episodes': episodes, 'seed': seed,
                'normalization_asset': repo_id(model_key(settings, task)),
                'instruction_type': settings['instruction_type'], 'execute_steps': settings['execute_steps'],
                'action_horizon': settings['action_horizon'], 'control': 'absolute world EE, xyz+wxyz+gripper, 16D',
                'protocol': 'native RoboTwin eval_policy, expert-solvable test seeds starting at 100000*(1+seed)'}
    write_json(destination / 'evaluation_config.json', metadata)
    try:
        next_seed, success = native.eval_policy(task, env, args, model, 100000 * (1 + seed),
                                                test_num=episodes, video_size=f"{camera['w']}x{camera['h']}" if video else None,
                                                instruction_type=settings['instruction_type'])
        if len(records) != episodes or sum(r['success'] for r in records) != success:
            raise RuntimeError('Episode audit does not match native success count')
        result = {**metadata, 'status': 'complete', 'successes': int(success), 'episodes': episodes,
                  'success_rate': float(success / episodes), 'attempted_seeds': len(attempted_seeds),
                  'next_seed': int(next_seed), 'elapsed_seconds': monotonic() - started}
        write_json(destination / 'result.json', result)
        print(f"{task}: {success}/{episodes} = {success / episodes:.1%}; results -> {destination}", flush=True)
    except BaseException as error:
        write_json(destination / 'failure.json', {**metadata, 'error': repr(error),
                                                  'completed_episodes': len(records), 'elapsed_seconds': monotonic() - started})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=TASKS, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--checkpoint-step', type=int)
    parser.add_argument('--no-video', action='store_true')
    parser.add_argument('--restart-incomplete', action='store_true')
    options = parser.parse_args()
    if options.episodes < 1 or options.seed < 0:
        parser.error('episodes must be positive and seed nonnegative')
    run(options.task, options.run_dir.resolve(), options.episodes, options.seed, options.checkpoint_step,
        not options.no_video, options.restart_incomplete)
