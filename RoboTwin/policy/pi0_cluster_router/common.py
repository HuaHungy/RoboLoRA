"""Configuration for Top-3 routed cluster-LoRA composition fine-tuning."""

import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PI0 = HERE.parent / "pi0"
ROBOTWIN = HERE.parents[1]
TASKS = ("adjust_bottle", "click_alarmclock", "place_can_basket",
         "place_container_plate", "rotate_qrcode")
CLUSTERS = (8, 24, 3, 6, 9, 31, 23)
JOINT_REPO = "robotwin_five_pi0_ee"
CONFIG_NAME = "cluster7_top3_router_pi0_ee"


def read_settings(path=None):
    path = Path(path or HERE / "config.json").resolve()
    settings = json.loads(path.read_text())
    for key in ("dataset_root", "pretrained_params", "cluster_training_summary",
                "bank_path", "bank_metadata", "source_joint_assets", "assets_root"):
        settings[key] = str((path.parent / Path(settings[key]).expanduser()).resolve())
    if tuple(settings["tasks"]) != TASKS or tuple(settings["clusters"]) != CLUSTERS:
        raise ValueError("Task or cluster order changed; it is part of the router-bank contract")
    for key in ("batch_size", "num_train_steps", "save_interval", "action_horizon",
                "router_top_k", "router_hidden_dim", "fsdp_devices"):
        if not isinstance(settings[key], int) or settings[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if settings["router_top_k"] > len(CLUSTERS):
        raise ValueError("router_top_k cannot exceed the seven-expert bank")
    if settings["action_horizon"] != 50:
        raise ValueError("The five-task EE datasets use 50-step action chunks")
    if settings["num_workers"] < 0 or settings["warmup_steps"] < 0:
        raise ValueError("num_workers and warmup_steps must be nonnegative")
    return settings


def setup(settings):
    sys.path[:0] = [str(PI0 / "src"), str(HERE.parent), str(HERE)]
    os.environ["HF_LEROBOT_HOME"] = settings["dataset_root"]
    os.environ.setdefault("HF_DATASETS_CACHE", str(HERE / ".cache/hf_datasets"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def build_config(settings, run_dir, resume=False):
    setup(settings)
    import flax.nnx as nnx
    from openpi.models.pi0 import Pi0Config
    from openpi.shared.nnx_utils import PathRegex
    from openpi.training.config import DataConfig, LeRobotRoboTwinEEDataConfig, TrainConfig
    from openpi.training.optimizer import CosineDecaySchedule
    from weight_loader import RoutedLoRABankWeightLoader

    model = Pi0Config(
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
        action_horizon=settings["action_horizon"],
        lora_expert_bank_size=len(CLUSTERS),
        lora_router_top_k=settings["router_top_k"],
        lora_router_hidden_dim=settings["router_hidden_dim"],
        lora_router_temperature=settings["router_temperature"],
        lora_router_balance_loss=settings["router_balance_loss"],
        lora_residual_scale=settings["residual_scale"],
    )

    class RouterTrainConfig(TrainConfig):
        @property
        def trainable_filter(self):
            # Train the MLP router, residual A/B and eta.  Historical bank,
            # backbone and control projections remain fixed.
            return nnx.All(
                nnx.Param,
                nnx.Any(
                    PathRegex(".*lora_router.*"),
                    nnx.All(
                        PathRegex(".*lora.*"),
                        nnx.Not(PathRegex(".*lora_bank.*")),
                    ),
                ),
            )

    freeze = nnx.Any(
        model.get_freeze_filter(),
        PathRegex(".*PaliGemma/img/.*"),
        PathRegex(".*lora_bank.*"),
    )
    steps = settings["num_train_steps"]
    return RouterTrainConfig(
        name=CONFIG_NAME,
        exp_name="model",
        project_name="robotwin_pi0_cluster_router",
        model=model,
        data=LeRobotRoboTwinEEDataConfig(
            repo_id=JOINT_REPO, base_config=DataConfig(local_files_only=True)),
        freeze_filter=freeze,
        ema_decay=None,
        batch_size=settings["batch_size"],
        num_workers=settings["num_workers"],
        num_train_steps=steps,
        weight_loader=RoutedLoRABankWeightLoader(
            settings["pretrained_params"], settings["bank_path"]),
        lr_schedule=CosineDecaySchedule(
            warmup_steps=min(settings["warmup_steps"], steps - 1),
            peak_lr=settings["peak_lr"], decay_steps=steps),
        assets_base_dir=settings["assets_root"],
        checkpoint_base_dir=str(Path(run_dir) / "checkpoints"),
        save_interval=settings["save_interval"],
        keep_period=settings.get("keep_period"),
        log_interval=50,
        seed=settings["seed"],
        fsdp_devices=settings["fsdp_devices"],
        wandb_enabled=False,
        overwrite=False,
        resume=resume,
        policy_metadata={
            "action_representation": "absolute_world_ee_xyz_wxyz_gripper",
            "training_tasks": list(TASKS),
            "cluster_order": list(CLUSTERS),
            "router_groups": 10,
            "router_top_k": settings["router_top_k"],
            "history_branch": "frozen_exact_weighted_delta",
            "residual_branch": "trainable_A_normal_B_zero",
        },
    )


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def latest_checkpoint(run_dir, step=None):
    root = Path(run_dir) / "checkpoints" / CONFIG_NAME / "model"
    candidates = [root / str(step)] if step is not None else sorted(
        (path for path in root.glob("*") if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name), reverse=True)
    for path in candidates:
        if ((path / "_CHECKPOINT_METADATA").is_file()
                and (path / "params").is_dir()
                and (path / "train_state").is_dir()
                and (path / "assets" / JOINT_REPO / "norm_stats.json").is_file()):
            return path
    raise FileNotFoundError(f"No committed routed checkpoint under {root}")
