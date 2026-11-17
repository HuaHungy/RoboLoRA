"""Frozen LoRAHub-Pi0 feature backbone for direct action decoders."""

from __future__ import annotations

import pathlib

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from PIL import Image

from script.common.data_loader import load_norm_stats


def _resize_with_pad(image: np.ndarray, height: int = 224, width: int = 224):
    image = np.asarray(image)
    if image.shape[:2] == (height, width):
        return image
    h, w = image.shape[:2]
    ratio = max(w / width, h / height)
    resized_h, resized_w = int(h / ratio), int(w / ratio)
    resized = np.asarray(Image.fromarray(image.astype(np.uint8)).resize(
        (resized_w, resized_h), Image.BILINEAR))
    output = np.zeros((height, width, image.shape[2]), np.uint8)
    top, left = (height - resized_h) // 2, (width - resized_w) // 2
    output[top:top + resized_h, left:left + resized_w] = resized
    return output


def prepare_observation_batch(inputs: list[dict], norm_stats: dict, tokenizer):
    """Create the OpenPI observation consumed only by the frozen prefix encoder."""
    from openpi.models.model import Observation

    image_batches = {name: [] for name in (
        "base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")}
    source_names = {
        "base_0_rgb": "cam_high",
        "left_wrist_0_rgb": "cam_left_wrist",
        "right_wrist_0_rgb": "cam_right_wrist",
    }
    states, tokens, token_masks = [], [], []
    state_stats = norm_stats["state"]
    state_size = len(state_stats["mean"])
    for item in inputs:
        for destination, source in source_names.items():
            image = np.asarray(item["images"][source])
            if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
                image = np.transpose(image, (1, 2, 0))
            if image.dtype != np.uint8:
                image = image.astype(np.float32)
                if float(np.nanmax(image)) <= 2.0 and float(np.nanmin(image)) >= -2.0:
                    image = np.clip((image + 1.0) * 127.5, 0, 255)
                image = np.clip(image, 0, 255).astype(np.uint8)
            image_batches[destination].append(
                _resize_with_pad(image).astype(np.float32) / 127.5 - 1.0)

        state = np.asarray(item["state"], np.float32).reshape(-1)
        padded = np.zeros(state_size, np.float32)
        padded[:min(state.size, state_size)] = state[:state_size]
        if not bool(item.get("state_is_normalized", False)):
            padded = (
                padded - np.asarray(state_stats["mean"], np.float32)
            ) / (np.asarray(state_stats["std"], np.float32) + 1e-6)
        states.append(padded)
        token, token_mask = tokenizer.tokenize(item["instruction"])
        tokens.append(np.asarray(token, np.int32))
        token_masks.append(np.asarray(token_mask, bool))

    valid = jnp.ones((len(inputs),), bool)
    return Observation(
        images={key: jnp.asarray(np.stack(value)) for key, value in image_batches.items()},
        image_masks={key: valid for key in image_batches},
        state=jnp.asarray(np.stack(states), jnp.float32),
        tokenized_prompt=jnp.asarray(np.stack(tokens), jnp.int32),
        tokenized_prompt_mask=jnp.asarray(np.stack(token_masks), bool),
    )


def load_frozen_backbone(checkpoint_dir: str | pathlib.Path, train_config_name: str):
    from openpi.models import model as model_lib
    from openpi.training import config as config_lib

    checkpoint_dir = pathlib.Path(checkpoint_dir).expanduser().resolve()
    params_dir = checkpoint_dir / "params"
    if not params_dir.is_dir():
        raise FileNotFoundError(f"LoRAHub checkpoint params not found: {params_dir}")
    train_config = config_lib.get_config(train_config_name)
    params = model_lib.restore_params(params_dir, dtype=jnp.bfloat16)
    model = train_config.model.load(params, remove_extra_params=False)
    graphdef, state = nnx.split(model)
    return graphdef, state, model


def make_prefix_feature_fn(graphdef):
    """Return final LoRAHub prefix hidden tokens; never run Pi0 action suffix."""
    from openpi.models import model as model_lib
    from openpi.models.pi0 import make_attn_mask

    @jax.jit
    def extract(state, observation):
        model = nnx.merge(graphdef, state)
        observation = model_lib.preprocess_observation(None, observation, train=False)
        prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
        attention_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_hidden, _), _ = model.PaliGemma.llm(
            [prefix_tokens, None], mask=attention_mask, positions=positions)
        return (
            jax.lax.stop_gradient(prefix_hidden.astype(jnp.float32)),
            jax.lax.stop_gradient(prefix_mask),
            jax.lax.stop_gradient(observation.state.astype(jnp.float32)),
        )

    return extract


def normalize_actions(outputs, norm_stats: dict, action_dim: int = 14) -> np.ndarray:
    actions = np.stack([np.asarray(x, np.float32)[:, :action_dim] for x in outputs])
    mean = np.asarray(norm_stats["actions"]["mean"], np.float32)[:action_dim]
    std = np.asarray(norm_stats["actions"]["std"], np.float32)[:action_dim]
    return (actions - mean[None, None, :]) / (std[None, None, :] + 1e-6)


def unnormalize_actions(actions, norm_stats: dict, action_dim: int = 14):
    mean = jnp.asarray(norm_stats["actions"]["mean"], jnp.float32)[:action_dim]
    std = jnp.asarray(norm_stats["actions"]["std"], jnp.float32)[:action_dim]
    return actions * (std[None, None, :] + 1e-6) + mean[None, None, :]


__all__ = [
    "load_frozen_backbone", "load_norm_stats", "make_prefix_feature_fn",
    "normalize_actions", "prepare_observation_batch", "unnormalize_actions",
]
