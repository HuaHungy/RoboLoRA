"""Prepare, train, and evaluate the seven-expert Top-3 LoRA router."""

import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from common import HERE, PI0, ROBOTWIN, TASKS, latest_checkpoint, read_settings, setup, write_json


def invoke(script, arguments, log_dir, *, cpu=False, evaluation=False):
    env = os.environ.copy()
    if evaluation:
        env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    env["JAX_PLATFORMS"] = "cpu" if cpu else "cuda"
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.50" if evaluation else "0.90"
    env.setdefault("ROBOTWIN_CUROBO_USE_CUDA_GRAPH", "0")
    env.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")
    env.setdefault("OMP_NUM_THREADS", "4")
    env["PYTHONPATH"] = os.pathsep.join([
        str(HERE), str(HERE.parent), str(PI0 / "src"), env.get("PYTHONPATH", "")])
    torch_lib = PI0 / ".venv/lib/python3.11/site-packages/torch/lib"
    env["LD_LIBRARY_PATH"] = str(torch_lib) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / f"{Path(script).stem}_{datetime.now():%Y%m%d_%H%M%S_%f}.log"
    command = [sys.executable, "-u", str(HERE / script), *map(str, arguments)]
    print(f"Command: {' '.join(command)}\nLog: {log}", flush=True)
    with log.open("w") as stream:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, env=env, cwd=ROBOTWIN, bufsize=1)
        for line in process.stdout:
            print(line, end="", flush=True)
            stream.write(line)
            stream.flush()
        code = process.wait()
    if code:
        raise SystemExit(f"{script} failed with exit code {code}; inspect {log}")


def main():
    parser = argparse.ArgumentParser(description="Top-3 MLP router over seven cluster LoRAs")
    parser.add_argument("stage", choices=("prepare", "train", "eval"))
    parser.add_argument("--run-name", default="cluster7_top3_router_v1")
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--train-steps", type=int)
    parser.add_argument("--save-interval", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--peak-lr", type=float)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--checkpoint-step", type=int)
    parser.add_argument("--execute-steps", type=int, default=10)
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reuse-bank", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_name):
        parser.error("run-name must be a simple name")
    if not re.fullmatch(r"\d+(,\d+)*", args.gpu):
        parser.error("gpu must look like 0 or 0,1")
    if len(set(args.tasks)) != len(args.tasks):
        parser.error("Each evaluation task may only be listed once")
    if min(args.eval_episodes, args.execute_steps, args.diffusion_steps) < 1 or args.eval_seed < 0:
        parser.error("evaluation counts/steps must be positive and eval-seed nonnegative")
    run_dir = HERE / "runs" / args.run_name
    config_path = run_dir / "run_config.json"
    if args.stage == "eval":
        if not config_path.is_file():
            parser.error(f"No saved training configuration: {config_path}")
        settings = read_settings(config_path)
    else:
        settings = read_settings(args.config)
    for key, value in (("batch_size", args.batch_size),
                       ("num_train_steps", args.train_steps),
                       ("save_interval", args.save_interval),
                       ("num_workers", args.num_workers),
                       ("peak_lr", args.peak_lr)):
        if value is not None:
            if value < 0 or (key != "num_workers" and value == 0):
                parser.error(f"invalid {key}")
            settings[key] = value
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    setup(settings)
    if args.stage == "prepare":
        snapshot = HERE / "logs" / f"prepare_settings_{datetime.now():%Y%m%d_%H%M%S_%f}.json"
        write_json(snapshot, settings)
        arguments = ["--settings", snapshot]
        if args.reuse_bank:
            arguments.append("--reuse-bank")
        invoke("prepare.py", arguments, HERE / "logs", cpu=True)
        return

    if args.stage == "eval":
        checkpoint = latest_checkpoint(run_dir, args.checkpoint_step)
        lock = (run_dir / ".run.lock").open("a")
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("this run-name is already active")
        for task in args.tasks:
            result_path = (run_dir / "evaluations" / task /
                           f"step_{checkpoint.name}_seed_{args.eval_seed}_n{args.eval_episodes}" / "result.json")
            if result_path.is_file() and args.resume:
                print(f"{task}: completed evaluation found; skipping", flush=True)
                continue
            restart_incomplete = result_path.parent.exists() and not result_path.is_file()
            command = [
                "--task", task,
                "--run-dir", run_dir,
                "--episodes", args.eval_episodes,
                "--seed", args.eval_seed,
                "--checkpoint-step", checkpoint.name,
                "--execute-steps", args.execute_steps,
                "--diffusion-steps", args.diffusion_steps,
            ]
            if args.no_video:
                command.append("--no-video")
            if args.resume or restart_incomplete:
                command.append("--restart-incomplete")
            invoke("evaluate.py", command, run_dir / "logs", evaluation=True)
        reports = [
            json.loads(path.read_text())
            for path in sorted((run_dir / "evaluations").glob("*/step_*/result.json"))
        ]
        matching = [
            report for report in reports
            if Path(report["checkpoint"]).name == checkpoint.name
            and report["seed"] == args.eval_seed
            and report["episodes"] == args.eval_episodes
        ]
        total_successes = sum(report["successes"] for report in matching)
        total_episodes = sum(report["episodes"] for report in matching)
        summary = {
            "checkpoint": str(checkpoint),
            "seed": args.eval_seed,
            "episodes_per_task": args.eval_episodes,
            "tasks": matching,
            "macro_success_rate": (sum(report["success_rate"] for report in matching) / len(matching)
                                   if matching else None),
            "overall_success_rate": total_successes / total_episodes if total_episodes else None,
        }
        write_json(run_dir / "evaluation_summary.json", summary)
        print(f"Completed evaluation; summary: {run_dir / 'evaluation_summary.json'}", flush=True)
        return

    gpu_count = len(args.gpu.split(","))
    if settings["batch_size"] % gpu_count:
        parser.error("global batch size must be divisible by visible GPU count")
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = (run_dir / ".run.lock").open("a")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error("this run-name is already active")
    if config_path.exists():
        if read_settings(config_path) != settings:
            parser.error("existing run has a different configuration")
        if not args.resume:
            parser.error("run exists; pass --resume or choose a new name")
    else:
        write_json(config_path, settings)
    prepare_args = ["--settings", config_path]
    if args.reuse_bank or Path(settings["bank_path"]).is_file():
        prepare_args.append("--reuse-bank")
    invoke("prepare.py", prepare_args, run_dir / "logs", cpu=True)
    if args.resume and (run_dir / "status/trained.json").is_file():
        print(f"Already complete: {latest_checkpoint(run_dir, settings['num_train_steps'])}")
        return
    train_args = ["--run-dir", run_dir]
    if args.resume:
        train_args.append("--resume")
    invoke("train_task.py", train_args, run_dir / "logs")
    print(f"Completed routed composition fine-tuning: {latest_checkpoint(run_dir)}", flush=True)


if __name__ == "__main__":
    main()
