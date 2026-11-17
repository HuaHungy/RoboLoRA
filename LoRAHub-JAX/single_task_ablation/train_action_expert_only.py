#!/usr/bin/env python3
"""Register and launch a native Pi0 action-expert-only training config.

This keeps the OpenPI checkout untouched.  The stock ``place_empty_cup``
config trains LoRA leaves; this experiment instead trains the full action
expert while freezing the PaliGemma language/vision branch.
"""

import dataclasses
import os
import runpy
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PI0_ROOT = os.path.join(ROOT, "..", "RoboTwin", "policy", "pi0")
sys.path.insert(0, os.path.join(PI0_ROOT, "src"))

from flax import nnx
from openpi.models import pi0
from openpi.shared import nnx_utils
from openpi.training import config as training_config
from openpi.training import weight_loaders


DEFAULT_CONFIG_NAME = "place_empty_cup_pi0_action_expert_only_10ep"


def register_config() -> None:
    config_name = os.environ.get("ACTION_EXPERT_CONFIG_NAME", DEFAULT_CONFIG_NAME)
    reference_name = os.environ.get(
        "ACTION_EXPERT_REFERENCE_CONFIG", "place_empty_cup_pi0_lora_10ep"
    )
    reference = training_config.get_config(reference_name)
    repo_id = os.environ.get("ACTION_EXPERT_REPO_ID", "place_empty_cup_10")
    asset_id = os.environ.get(
        "ACTION_EXPERT_ASSET_ID", "place_empty_cup_pi0_10"
    )
    assets_dir = os.environ.get(
        "ACTION_EXPERT_ASSETS_DIR",
        os.path.join(
            ROOT, "outputs", "place_empty_cup_10_v3_pi0_seq",
            "checkpoint", "10000", "assets",
        ),
    )
    local_data = dataclasses.replace(
        reference.data,
        repo_id=repo_id,
        assets=training_config.AssetsConfig(
            assets_dir=assets_dir,
            asset_id=asset_id,
        ),
    )

    use_lorahub_init = os.environ.get("ACTION_EXPERT_LORAHUB_MODEL", "0") == "1"
    init_params = os.environ.get(
        "ACTION_EXPERT_INIT_PARAMS",
        os.path.join(ROOT, "openpi_base", "pi0_base", "params"),
    )

    # LoRAHub's processed checkpoint contains PaliGemma LoRA leaves, so the
    # model graph must retain the PaliGemma LoRA variant for that arm.  The
    # native arm uses the ordinary Pi0 graph.  Both arms train only the same
    # action-expert branch after initialization.
    model = pi0.Pi0Config(
        paligemma_variant="gemma_2b_lora" if use_lorahub_init else "gemma_2b",
        action_expert_variant="gemma_300m",
    )

    # Strict action-expert-only training.  Train the ``*_1`` action-expert
    # transformer plus Pi0's action/state/time projections; freeze the vision
    # encoder, main language model, embeddings, and all LoRA leaves.
    action_expert_trainable = nnx.Any(
        nnx_utils.PathRegex(r".*llm.*_1.*"),
        nnx_utils.PathRegex(
            r".*(action_in_proj|action_out_proj|action_time_mlp_in|"
            r"action_time_mlp_out|state_proj).*"
        ),
    )
    freeze_non_action_expert = nnx.Not(action_expert_trainable)
    config = dataclasses.replace(
        reference,
        name=config_name,
        model=model,
        data=local_data,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            init_params
        ),
        freeze_filter=freeze_non_action_expert,
        checkpoint_base_dir=os.path.join(PI0_ROOT, "checkpoints"),
        num_train_steps=int(os.environ.get("ACTION_EXPERT_STEPS", "3000")),
        batch_size=int(os.environ.get("ACTION_EXPERT_BATCH_SIZE", "8")),
        # LeRobot 0.1 + datasets 5.x returns Column objects in worker
        # processes.  The compatibility patch is installed in the launcher
        # process, so keep workers off by default for reliable training.
        num_workers=int(os.environ.get("ACTION_EXPERT_NUM_WORKERS", "0")),
        save_interval=int(os.environ.get("ACTION_EXPERT_SAVE_INTERVAL", "1000")),
        # Orbax uses max_to_keep=1 in this OpenPI checkout.  Setting
        # keep_period=save_interval is what actually preserves the requested
        # 1000/2000/3000 checkpoints in one run.
        keep_period=int(os.environ.get("ACTION_EXPERT_KEEP_PERIOD", "5000")),
        # OpenPI's EMA duplicates the full parameter tree, including frozen
        # Pi0 weights.  Disable it for this memory-constrained ablation.
        ema_decay=None,
        overwrite=os.environ.get("ACTION_EXPERT_OVERWRITE", "1") == "1",
        resume=os.environ.get("ACTION_EXPERT_RESUME", "0") == "1",
        wandb_enabled=False,
    )
    training_config._CONFIGS_DICT[config_name] = config


def patch_lerobot_column_stack() -> None:
    """Compatibility for datasets 5.x Column objects with old LeRobot 0.1."""
    import torch

    if getattr(torch, "_lorahub_column_stack_patched", False):
        return
    original_stack = torch.stack

    def compatible_stack(tensors, *args, **kwargs):
        if type(tensors).__name__ == "Column":
            tensors = tuple(tensors)
        return original_stack(tensors, *args, **kwargs)

    torch.stack = compatible_stack
    torch._lorahub_column_stack_patched = True


def patch_openpi_optax_forward_ref() -> None:
    """Resolve Optax's ArrayTree forward reference used by OpenPI TrainState.

    The pinned OpenPI source annotates ``optax.OptState`` with a forward
    reference named ``ArrayTree``.  The installed Optax exposes that alias in
    the annotation but not in ``openpi.training.utils``, which makes the
    runtime beartype check fail during TrainState initialization.
    """
    import openpi.training.utils as training_utils

    if not hasattr(training_utils, "ArrayTree"):
        # ``typing.Any`` cannot be used by beartype in isinstance checks.
        # ``object`` is intentionally broad but runtime-checkable; the actual
        # JAX/Optax pytree structure is still preserved by TrainState.
        training_utils.ArrayTree = object


if __name__ == "__main__":
    patch_lerobot_column_stack()
    patch_openpi_optax_forward_ref()
    register_config()
    # Execute the unmodified OpenPI training entry point after registration.
    sys.argv = [os.path.join(PI0_ROOT, "scripts", "train.py"), *sys.argv[1:]]
    runpy.run_path(sys.argv[0], run_name="__main__")
