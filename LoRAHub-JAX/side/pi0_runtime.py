"""Frozen Pi0 loading and feature extraction for Side-Tuning.

This module deliberately keeps the backbone out of the optimizer.  It loads a
RoboTwin/OpenPI checkpoint, runs the frozen flow-matching forward pass, and
returns detached prefix K/V summaries plus action-expert hidden states.
"""

from __future__ import annotations

import pathlib
from functools import partial

import einops
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from PIL import Image

from script.common.data_loader import load_norm_stats
from side.kv_side_network import KVSideConfig, KVSideQFormerConfig


def _resize_with_pad(image: np.ndarray, height: int = 224, width: int = 224) -> np.ndarray:
    image = np.asarray(image)
    if image.shape[:2] == (height, width):
        return image
    h, w = image.shape[:2]
    ratio = max(w / width, h / height)
    resized_h = int(h / ratio)
    resized_w = int(w / ratio)
    resized = np.asarray(Image.fromarray(image.astype(np.uint8)).resize(
        (resized_w, resized_h), Image.BILINEAR))
    out = np.zeros((height, width, image.shape[2]), dtype=np.uint8)
    pad_h, rem_h = divmod(height - resized_h, 2)
    pad_w, rem_w = divmod(width - resized_w, 2)
    del rem_h, rem_w
    out[pad_h:pad_h + resized_h, pad_w:pad_w + resized_w] = resized
    return out


def load_frozen_pi0(checkpoint_dir: str | pathlib.Path, train_config_name: str):
    """Load a composed Pi0 checkpoint and split it for dynamic JIT execution."""
    from openpi.models import model as model_lib
    from openpi.training import config as config_lib

    checkpoint_dir = pathlib.Path(checkpoint_dir).expanduser().resolve()
    params_dir = checkpoint_dir / "params"
    if not params_dir.is_dir():
        raise FileNotFoundError(f"Pi0 params directory not found: {params_dir}")

    train_config = config_lib.get_config(train_config_name)
    params = model_lib.restore_params(params_dir, dtype=jnp.bfloat16)
    model = train_config.model.load(params, remove_extra_params=False)
    graphdef, state = nnx.split(model)
    return graphdef, state, model


def prepare_observation_batch(inputs: list[dict], norm_stats: dict, tokenizer):
    """Convert LoRAHub frame samples into an OpenPI ``Observation`` batch."""
    from openpi.models.model import Observation

    high, left, right, states, tokens, masks = [], [], [], [], [], []
    for item in inputs:
        images = item["images"]
        image_values = []
        for name in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
            image = np.asarray(images[name])
            if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
                image = np.transpose(image, (1, 2, 0))
            if image.dtype != np.uint8:
                image = image.astype(np.float32)
                if float(np.nanmax(image)) <= 2.0 and float(np.nanmin(image)) >= -2.0:
                    image = np.clip((image + 1.0) * 127.5, 0, 255).astype(np.uint8)
                else:
                    image = np.clip(image, 0, 255).astype(np.uint8)
            image_values.append(_resize_with_pad(image))

        high.append(image_values[0].astype(np.float32) / 127.5 - 1.0)
        left.append(image_values[1].astype(np.float32) / 127.5 - 1.0)
        right.append(image_values[2].astype(np.float32) / 127.5 - 1.0)

        state = np.asarray(item["state"], dtype=np.float32).reshape(-1)
        if state.shape[0] < len(norm_stats["state"]["mean"]):
            padded = np.zeros(len(norm_stats["state"]["mean"]), dtype=np.float32)
            padded[:state.shape[0]] = state
            state = padded
        else:
            state = state[:len(norm_stats["state"]["mean"])]
        if not bool(item.get("state_is_normalized", False)):
            stats = norm_stats["state"]
            state = (state - np.asarray(stats["mean"], np.float32)) / (
                np.asarray(stats["std"], np.float32) + 1e-6)
        states.append(state)

        token, mask = tokenizer.tokenize(item["instruction"])
        tokens.append(np.asarray(token, dtype=np.int32))
        masks.append(np.asarray(mask, dtype=bool))

    batch = len(inputs)
    valid = np.ones(batch, dtype=bool)
    return Observation(
        images={
            "base_0_rgb": jnp.asarray(np.stack(high)),
            "left_wrist_0_rgb": jnp.asarray(np.stack(left)),
            "right_wrist_0_rgb": jnp.asarray(np.stack(right)),
        },
        image_masks={
            "base_0_rgb": jnp.asarray(valid),
            "left_wrist_0_rgb": jnp.asarray(valid),
            "right_wrist_0_rgb": jnp.asarray(valid),
        },
        state=jnp.asarray(np.stack(states), dtype=jnp.float32),
        tokenized_prompt=jnp.asarray(np.stack(tokens), dtype=jnp.int32),
        tokenized_prompt_mask=jnp.asarray(np.stack(masks), dtype=bool),
    )


def build_normalized_action_batch(outputs: list[np.ndarray], norm_stats: dict,
                                  action_dim: int = 14, model_action_dim: int = 32):
    """Pad 14D delta-action targets to Pi0's 32D normalized action space."""
    raw = np.zeros((len(outputs), outputs[0].shape[0], model_action_dim), np.float32)
    for index, value in enumerate(outputs):
        value = np.asarray(value, dtype=np.float32)
        raw[index, :, :action_dim] = value[:, :action_dim]
    stats = norm_stats["actions"]
    mean = np.asarray(stats["mean"], dtype=np.float32)[:model_action_dim]
    std = np.asarray(stats["std"], dtype=np.float32)[:model_action_dim]
    return (raw - mean[None, None, :]) / (std[None, None, :] + 1e-6)


def make_flow_feature_fn(graphdef, *, layer_indices=(5, 11, 17),
                         kv_mode: str = "pooled"):
    """Build a JIT function returning frozen flow outputs and side features.

    The returned function accepts ``(state, observation, actions, noise, time)``
    where actions/noise are normalized 32D action chunks.  It returns:

      ``v_base``: normalized Pi0 flow velocity, ``[B, 50, 32]``;
      ``kv_context``: pooled prefix K/V, ``[B, 512]`` for Pi0;
      ``action_hidden``: action-expert hidden, ``[B, 50, 1024]``;
      ``u_target``: flow-matching target, ``[B, 50, 32]``.
    """
    from openpi.models import model as model_lib
    from openpi.models.pi0 import make_attn_mask

    selected_layers = tuple(int(x) for x in layer_indices)
    if kv_mode not in {"pooled", "tokens"}:
        raise ValueError(f"unsupported kv_mode: {kv_mode}")

    @jax.jit
    def flow_features(state, observation, actions, noise, time):
        model = nnx.merge(graphdef, state)
        observation = model_lib.preprocess_observation(None, observation, train=False)
        time = jnp.asarray(time, dtype=jnp.float32).reshape(-1)
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1.0 - time_expanded) * actions
        u_target = noise - actions

        prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask = model.embed_suffix(
            observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), kv_cache = model.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
        )
        del prefix_out
        v_base = model.action_out_proj(suffix_out[:, -model.action_horizon:])

        k_cache, v_cache = kv_cache
        prefix_len = prefix_tokens.shape[1]
        selected = jnp.asarray(selected_layers, dtype=jnp.int32)
        k_prefix = k_cache[selected, :, :prefix_len, :, :]
        v_prefix = v_cache[selected, :, :prefix_len, :, :]
        if kv_mode == "tokens":
            # Preserve prefix-token information for the Q-Former while
            # averaging only selected layers and attention heads.
            k_tokens = jnp.mean(k_prefix.astype(jnp.float32), axis=(0, 3))
            v_tokens = jnp.mean(v_prefix.astype(jnp.float32), axis=(0, 3))
            kv_context = jnp.concatenate([k_tokens, v_tokens], axis=-1)
        else:
            k_pool = jnp.mean(k_prefix.astype(jnp.float32), axis=(0, 2, 3))
            v_pool = jnp.mean(v_prefix.astype(jnp.float32), axis=(0, 2, 3))
            kv_context = jnp.concatenate([k_pool, v_pool], axis=-1)
        action_hidden = suffix_out[:, -model.action_horizon:].astype(jnp.float32)

        values = (v_base, kv_context, action_hidden, u_target)
        return tuple(jax.lax.stop_gradient(x.astype(jnp.float32)) for x in values)

    return flow_features


def make_side_sampler_fn(graphdef, side_model, *, layer_indices=(5, 11, 17),
                         kv_mode: str = "pooled"):
    """Build a JIT sampler that injects the Side residual at every ODE step."""
    from openpi.models import model as model_lib
    from openpi.models.pi0 import make_attn_mask

    selected_layers = tuple(int(x) for x in layer_indices)
    if kv_mode not in {"pooled", "tokens"}:
        raise ValueError(f"unsupported kv_mode: {kv_mode}")

    @partial(jax.jit, static_argnames=("num_steps",))
    def sample(state, side_params, rng, observation, *, num_steps, residual_scale):
        model = nnx.merge(graphdef, state)
        observation = model_lib.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        noise = jax.random.normal(
            rng, (batch_size, model.action_horizon, model.action_dim), dtype=jnp.float32)

        prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = model.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=positions,
        )
        k_cache, v_cache = kv_cache
        prefix_len = prefix_tokens.shape[1]
        selected = jnp.asarray(selected_layers, dtype=jnp.int32)
        k_prefix = k_cache[selected, :, :prefix_len, :, :]
        v_prefix = v_cache[selected, :, :prefix_len, :, :]
        if kv_mode == "tokens":
            context = jnp.concatenate([
                jnp.mean(k_prefix.astype(jnp.float32), axis=(0, 3)),
                jnp.mean(v_prefix.astype(jnp.float32), axis=(0, 3)),
            ], axis=-1)
        else:
            context = jnp.concatenate([
                jnp.mean(k_prefix.astype(jnp.float32), axis=(0, 2, 3)),
                jnp.mean(v_prefix.astype(jnp.float32), axis=(0, 2, 3)),
            ], axis=-1)

        def step(carry):
            x_t, time = carry
            time_batch = jnp.broadcast_to(time, (batch_size,))
            suffix_tokens, suffix_mask, suffix_ar_mask = model.embed_suffix(
                observation, x_t, time_batch)
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_mask_for_suffix = einops.repeat(
                prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate(
                [prefix_mask_for_suffix, suffix_attn_mask], axis=-1)
            suffix_positions = (
                jnp.sum(prefix_mask, axis=-1)[:, None]
                + jnp.cumsum(suffix_mask, axis=-1) - 1
            )
            (prefix_out, suffix_out), _ = model.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=suffix_positions,
                kv_cache=kv_cache,
            )
            del prefix_out
            action_hidden = suffix_out[:, -model.action_horizon:]
            v_base = model.action_out_proj(action_hidden)
            residual = side_model.apply(
                {"params": side_params}, context, action_hidden, time_batch)
            residual = jnp.pad(
                residual, ((0, 0), (0, 0), (0, model.action_dim - residual.shape[-1])))
            velocity = v_base.astype(jnp.float32) + residual_scale * residual
            return x_t + dt * velocity, time + dt

        # A static fori_loop avoids a data-dependent loop condition and keeps
        # the compiled sampler shape-stable for deployment.
        def body(_, carry):
            return step(carry)

        x_0, _ = jax.lax.fori_loop(0, num_steps, body, (noise, jnp.asarray(1.0)))
        return x_0

    return sample


def side_shape_config(model, *, output_dim: int = 14) -> KVSideConfig:
    """Infer Side input dimensions from a loaded Pi0 model."""
    gemma = model.PaliGemma.llm
    del gemma
    # Pi0's action expert width is the final hidden dimension.  This is stable
    # for the current pi0 config but inferred from the model projection shape.
    action_hidden_dim = int(model.action_out_proj.in_features)
    # One KV head with head_dim 256 for the current Pi0 family.
    context_dim = 2 * 256
    return KVSideConfig(
        context_dim=context_dim,
        action_hidden_dim=action_hidden_dim,
        output_dim=output_dim,
    )


def qformer_shape_config(model, *, output_dim: int = 14,
                         num_queries: int = 4,
                         qformer_dim: int = 128,
                         qformer_heads: int = 4,
                         qformer_layers: int = 2,
                         qformer_ff_dim: int = 256) -> KVSideQFormerConfig:
    """Infer the small token-level Q-Former Side dimensions from Pi0."""
    base = side_shape_config(model, output_dim=output_dim)
    return KVSideQFormerConfig(
        context_dim=base.context_dim,
        action_hidden_dim=base.action_hidden_dim,
        output_dim=base.output_dim,
        kv_token_dim=base.context_dim,
        num_queries=int(num_queries),
        qformer_dim=int(qformer_dim),
        qformer_heads=int(qformer_heads),
        qformer_layers=int(qformer_layers),
        qformer_ff_dim=int(qformer_ff_dim),
    )
