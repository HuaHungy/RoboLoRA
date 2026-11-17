#!/usr/bin/env python3
"""Register the matching action-expert config, then run RoboTwin evaluation."""

import os
import runpy
import sys

from train_action_expert_only import register_config

register_config()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROBOTWIN_ROOT = os.path.join(ROOT, "..", "RoboTwin")
# RoboTwin's evaluation entry point resolves ``envs``, ``policy`` and its
# YAML paths relative to the repository root.  ``runpy`` also does not add
# the executed file's directory to sys.path, so expose ``script/`` for local
# imports such as ``test_render``.
os.chdir(ROBOTWIN_ROOT)
sys.path.insert(0, os.path.join(ROBOTWIN_ROOT, "script"))
sys.argv = [
    os.path.join(ROBOTWIN_ROOT, "script", "eval_policy.py"),
    "--config", "policy/pi0/deploy_policy.yml",
    "--overrides",
    "--task_name", sys.argv[1],
    "--task_config", sys.argv[2],
    "--train_config_name", sys.argv[3],
    "--model_name", sys.argv[4],
    "--ckpt_setting", os.environ.get("ACTION_EXPERT_CKPT_SETTING", sys.argv[4]),
    "--checkpoint_id", os.environ.get("ACTION_EXPERT_CHECKPOINT_ID", "3000"),
    "--seed", sys.argv[5],
    "--policy_name", "pi0",
]
runpy.run_path(sys.argv[0], run_name="__main__")
