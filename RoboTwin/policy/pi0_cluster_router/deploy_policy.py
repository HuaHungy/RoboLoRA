"""Deploy the routed seven-expert model with RoboTwin's native EE control."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np

from pi0_ee_five.deploy_policy import encode_observation


class RoutedEEPolicy:
    def __init__(self, settings, run_dir, checkpoint, execute_steps=10, diffusion_steps=10):
        from .common import JOINT_REPO, build_config
        from openpi.policies.policy_config import create_trained_policy

        cfg = build_config(settings, run_dir)
        self.policy = create_trained_policy(
            cfg,
            checkpoint,
            robotwin_repo_id=JOINT_REPO,
            sample_kwargs={"num_steps": diffusion_steps},
        )
        self.execute_steps = execute_steps
        self.previous_state = None
        self.inference_calls = 0
        self.total_inference_seconds = 0.0
        self.total_action_seconds = 0.0


def get_model(usr_args):
    run_dir = Path(usr_args["run_dir"])
    settings = json.loads((run_dir / "run_config.json").read_text())
    return RoutedEEPolicy(
        settings,
        run_dir,
        usr_args["checkpoint"],
        execute_steps=int(usr_args.get("execute_steps", 10)),
        diffusion_steps=int(usr_args.get("diffusion_steps", 10)),
    )


def eval(TASK_ENV, model, observation):
    inputs = encode_observation(observation, model.previous_state)
    model.previous_state = inputs["state"].copy()
    inputs["prompt"] = TASK_ENV.get_instruction()
    inference_started = perf_counter()
    actions = np.asarray(model.policy.infer(inputs)["actions"])
    inference_seconds = perf_counter() - inference_started
    if actions.ndim != 2 or actions.shape[1] != 16 or not np.isfinite(actions).all():
        raise ValueError("Expected finite [horizon,16] absolute EE actions")
    action_started = perf_counter()
    executed = 0
    for action in actions[:model.execute_steps]:
        if TASK_ENV.eval_success or TASK_ENV.take_action_cnt >= TASK_ENV.step_lim:
            break
        TASK_ENV.take_action(action, action_type="ee")
        executed += 1
        # Video recording needs a fresh rendered frame after every action.
        # Without video, the native outer loop obtains a new full observation
        # at the next chunk boundary, so rendering three cameras here is pure
        # duplicate work.
        if getattr(TASK_ENV, "eval_video_path", None) is not None:
            current = TASK_ENV.get_obs()
            model.previous_state = encode_observation(current, model.previous_state)["state"]
    action_seconds = perf_counter() - action_started
    model.inference_calls += 1
    model.total_inference_seconds += inference_seconds
    model.total_action_seconds += action_seconds
    print(
        f"\n[router timing] chunk={model.inference_calls} actions={executed} "
        f"inference={inference_seconds:.2f}s environment={action_seconds:.2f}s "
        f"cumulative_inference={model.total_inference_seconds:.1f}s "
        f"cumulative_environment={model.total_action_seconds:.1f}s",
        flush=True,
    )


def reset_model(model):
    model.previous_state = None
