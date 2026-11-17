"""Evaluate one routed checkpoint with RoboTwin's expert-filtered protocol."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import random
import sys
from time import monotonic

from common import CLUSTERS, HERE, JOINT_REPO, ROBOTWIN, TASKS, latest_checkpoint, load_module, setup, write_json


class SeedSearchLimit(BaseException):
    """Stop the native retry loop, which catches ordinary exceptions."""


def run(task, run_dir, episodes, seed, checkpoint_step=None, video=True,
        restart_incomplete=False, execute_steps=10, diffusion_steps=10):
    settings = json.loads((run_dir / "run_config.json").read_text())
    setup(settings)
    checkpoint = latest_checkpoint(run_dir, checkpoint_step)
    destination = run_dir / "evaluations" / task / f"step_{checkpoint.name}_seed_{seed}_n{episodes}"
    if destination.exists():
        if restart_incomplete and not (destination / "result.json").is_file():
            archived = destination.with_name(
                destination.name + datetime.now().strftime("_interrupted_%Y%m%d_%H%M%S_%f"))
            destination.rename(archived)
            print(f"Preserved interrupted evaluation at {archived}; restarting", flush=True)
        else:
            raise FileExistsError(
                f"Evaluation output exists: {destination}; change seed/count or pass --restart-incomplete")
    destination.mkdir(parents=True)

    os.chdir(ROBOTWIN)
    for path in (ROBOTWIN, ROBOTWIN / "policy", ROBOTWIN / "script", ROBOTWIN / "description/utils"):
        sys.path.insert(0, str(path))
    import jax
    import numpy as np
    import yaml

    if not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("JAX GPU is required for pi0 evaluation")
    native = load_module("pi0_cluster_router_native_evaluator", ROBOTWIN / "script/eval_policy.py")
    from pi0_cluster_router.deploy_policy import get_model

    task_config = "demo_aloha"
    instruction_type = "unseen"
    args = yaml.safe_load((ROBOTWIN / "task_config" / f"{task_config}.yml").read_text())
    args.update(
        task_name=task,
        task_config=task_config,
        policy_name="pi0_cluster_router",
        ckpt_setting=run_dir.name,
        render_freq=0,
        eval_mode=True,
        eval_video_log=video,
        collect_data=False,
    )
    args["data_type"].update(rgb=True, endpose=True)
    args["camera"].update(collect_head_camera=True, collect_wrist_camera=True)
    embodiment = args["embodiment"]
    if embodiment != ["aloha-agilex"]:
        raise ValueError("The 16D datasets require the demo_aloha dual-arm embodiment")
    mapping = yaml.safe_load((ROBOTWIN / "task_config/_embodiment_config.yml").read_text())
    robot_file = mapping[embodiment[0]]["file_path"]
    args.update(
        left_robot_file=robot_file,
        right_robot_file=robot_file,
        dual_arm_embodied=True,
        left_embodiment_config=native.get_embodiment_config(robot_file),
        right_embodiment_config=native.get_embodiment_config(robot_file),
    )
    camera = native.get_camera_config(args["camera"]["head_camera_type"])
    args.update(head_camera_h=camera["h"], head_camera_w=camera["w"])
    if video:
        args["eval_video_save_dir"] = destination

    np.random.seed(seed)
    random.seed(seed)
    env = native.class_decorator(task)
    # The policy observes only at action-chunk boundaries. During a no-video
    # evaluation, intermediate renderer refreshes do not affect physics,
    # planning, observations, or success checks and are needlessly expensive.
    env.fast_eval_no_render = not video
    model = get_model({
        "task_name": task,
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "execute_steps": execute_steps,
        "diffusion_steps": diffusion_steps,
    })
    records, attempted_seeds = [], set()
    active = {"seed": None, "instruction": None}
    original_setup = env.setup_demo
    original_instruction = env.set_instruction
    original_close = env.close_env

    def tracked_setup(*positional, **kwargs):
        active["seed"] = int(kwargs["seed"])
        attempted_seeds.add(active["seed"])
        if len(attempted_seeds) > max(100, episodes * 20):
            raise SeedSearchLimit("Too many expert-rejected seeds; inspect simulator/planner errors")
        random.seed(active["seed"])
        return original_setup(*positional, **kwargs)

    def tracked_instruction(*positional, **kwargs):
        active["instruction"] = kwargs.get("instruction", positional[0] if positional else None)
        return original_instruction(*positional, **kwargs)

    def tracked_close(*positional, **kwargs):
        if active["instruction"] is not None:
            records.append({
                "seed": active["seed"],
                "instruction": active["instruction"],
                "success": bool(env.eval_success),
                "actions": int(env.take_action_cnt),
            })
            write_json(destination / "episodes.json", records)
            active["instruction"] = None
        return original_close(*positional, **kwargs)

    env.setup_demo = tracked_setup
    env.set_instruction = tracked_instruction
    env.close_env = tracked_close
    started = monotonic()
    metadata = {
        "task": task,
        "checkpoint": str(checkpoint),
        "requested_episodes": episodes,
        "seed": seed,
        "normalization_asset": JOINT_REPO,
        "instruction_type": instruction_type,
        "execute_steps": execute_steps,
        "diffusion_steps": diffusion_steps,
        "action_horizon": settings["action_horizon"],
        "control": "absolute world EE, xyz+wxyz+gripper, 16D",
        "clusters": list(CLUSTERS),
        "router_top_k": settings["router_top_k"],
        "fast_no_video_render": not video,
        "protocol": "native RoboTwin eval_policy, expert-solvable test seeds starting at 100000*(1+seed)",
    }
    write_json(destination / "evaluation_config.json", metadata)
    try:
        next_seed, success = native.eval_policy(
            task,
            env,
            args,
            model,
            100000 * (1 + seed),
            test_num=episodes,
            video_size=f"{camera['w']}x{camera['h']}" if video else None,
            instruction_type=instruction_type,
        )
        if len(records) != episodes or sum(record["success"] for record in records) != success:
            raise RuntimeError("Episode audit does not match native success count")
        result = {
            **metadata,
            "status": "complete",
            "successes": int(success),
            "episodes": episodes,
            "success_rate": float(success / episodes),
            "attempted_seeds": len(attempted_seeds),
            "next_seed": int(next_seed),
            "elapsed_seconds": monotonic() - started,
        }
        write_json(destination / "result.json", result)
        print(f"{task}: {success}/{episodes} = {success / episodes:.1%}; results -> {destination}", flush=True)
    except BaseException as error:
        write_json(destination / "failure.json", {
            **metadata,
            "error": repr(error),
            "completed_episodes": len(records),
            "elapsed_seconds": monotonic() - started,
        })
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-step", type=int)
    parser.add_argument("--execute-steps", type=int, default=10)
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--restart-incomplete", action="store_true")
    options = parser.parse_args()
    if min(options.episodes, options.execute_steps, options.diffusion_steps) < 1 or options.seed < 0:
        parser.error("episodes/steps must be positive and seed nonnegative")
    run(
        options.task,
        options.run_dir.resolve(),
        options.episodes,
        options.seed,
        options.checkpoint_step,
        not options.no_video,
        options.restart_incomplete,
        options.execute_steps,
        options.diffusion_steps,
    )
