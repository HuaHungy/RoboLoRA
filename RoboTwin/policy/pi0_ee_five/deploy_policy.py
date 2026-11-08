"""16D world-frame EE policy; native RoboTwin end-effector control only."""

from pathlib import Path
import json
import numpy as np


def encode_observation(observation, previous_state=None):
    endpose = observation['endpose']
    state = np.concatenate([
        np.asarray(endpose['left_endpose']).reshape(7), np.asarray(endpose['left_gripper']).reshape(1),
        np.asarray(endpose['right_endpose']).reshape(7), np.asarray(endpose['right_gripper']).reshape(1),
    ]).astype(np.float32)
    if not np.isfinite(state).all() or np.any(state[[7, 15]] < -1e-5) or np.any(state[[7, 15]] > 1 + 1e-5):
        raise ValueError('Invalid measured EE state')
    state[[7, 15]] = np.clip(state[[7, 15]], 0, 1)
    for offset in (3, 11):
        quaternion = state[offset:offset + 4]
        norm = np.linalg.norm(quaternion)
        if norm < 1e-8:
            raise ValueError('Measured EE quaternion has zero length')
        quaternion /= norm
        flip = (quaternion[np.argmax(np.abs(quaternion))] < 0 if previous_state is None
                else np.dot(previous_state[offset:offset + 4], quaternion) < 0)
        if flip:
            quaternion *= -1
    images = {}
    for key, camera in [('cam_high', 'head_camera'), ('cam_left_wrist', 'left_camera'), ('cam_right_wrist', 'right_camera')]:
        rgb = np.asarray(observation['observation'][camera]['rgb'])
        if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
            raise ValueError(f'{camera} must contain HWC uint8 RGB')
        images[key] = np.moveaxis(rgb, -1, 0)
    return {'state': state, 'images': images}


class EEPolicy:
    def __init__(self, settings, task, run_dir, checkpoint):
        from .common import build_config, model_key, repo_id
        from openpi.policies.policy_config import create_trained_policy
        key = model_key(settings, task)
        cfg = build_config(key, settings, run_dir)
        self.policy = create_trained_policy(cfg, checkpoint, robotwin_repo_id=repo_id(key),
                                           sample_kwargs={'num_steps': settings['diffusion_steps']})
        self.execute_steps = settings['execute_steps']
        self.previous_state = None


def get_model(usr_args):
    settings = json.loads((Path(usr_args['run_dir']) / 'run_config.json').read_text())
    return EEPolicy(settings, usr_args['task_name'], usr_args['run_dir'], usr_args['checkpoint'])


def eval(TASK_ENV, model, observation):
    inputs = encode_observation(observation, model.previous_state)
    model.previous_state = inputs['state'].copy()
    inputs['prompt'] = TASK_ENV.get_instruction()
    actions = np.asarray(model.policy.infer(inputs)['actions'])
    if actions.ndim != 2 or actions.shape[1] != 16 or not np.isfinite(actions).all():
        raise ValueError('Expected finite [horizon,16] absolute EE actions')
    for action in actions[:model.execute_steps]:
        if TASK_ENV.eval_success or TASK_ENV.take_action_cnt >= TASK_ENV.step_lim:
            break
        TASK_ENV.take_action(action, action_type='ee')
        # Refresh native video observations and preserve quaternion sign continuity
        # even between the observations used for policy inference.
        current = TASK_ENV.get_obs()
        model.previous_state = encode_observation(current, model.previous_state)['state']


def reset_model(model):
    model.previous_state = None
