"""Flax action decoders conditioned on frozen LoRAHub prefix features.

These modules are policy heads, not Side-Tuning residuals.  They directly
produce a normalized 50x14 action chunk and never consume Pi0's action expert
hidden state or flow velocity.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
from flax import linen as nn


@dataclasses.dataclass(frozen=True)
class DecoderConfig:
    action_dim: int = 14
    action_horizon: int = 50
    state_dim: int = 32
    model_dim: int = 256
    heads: int = 4
    layers: int = 4
    ff_dim: int = 1024
    separate_query_pos: bool = False
    predict_is_pad: bool = False
    use_cvae: bool = False
    latent_dim: int = 32
    cvae_encoder_layers: int = 2


def sinusoidal_embedding(value: jax.Array, dim: int) -> jax.Array:
    if dim % 2:
        raise ValueError(f"embedding dim must be even, got {dim}")
    value = jnp.asarray(value, jnp.float32).reshape(-1, 1)
    frequency = jnp.exp(jnp.linspace(jnp.log(1.0), jnp.log(10_000.0), dim // 2))
    angle = value / frequency[None, :]
    return jnp.concatenate([jnp.sin(angle), jnp.cos(angle)], axis=-1)


def _attention(q, k, v, *, dim: int, heads: int, name: str, key_mask=None):
    if dim % heads:
        raise ValueError(f"model_dim={dim} must be divisible by heads={heads}")
    batch, q_len = q.shape[:2]
    k_len = k.shape[1]
    head_dim = dim // heads
    q = nn.Dense(dim, name=f"{name}_q")(q).reshape(batch, q_len, heads, head_dim)
    k = nn.Dense(dim, name=f"{name}_k")(k).reshape(batch, k_len, heads, head_dim)
    v = nn.Dense(dim, name=f"{name}_v")(v).reshape(batch, k_len, heads, head_dim)
    logits = jnp.einsum("bqhd,bkhd->bhqk", q, k) * (head_dim ** -0.5)
    if key_mask is not None:
        mask = jnp.asarray(key_mask, bool)[:, None, None, :]
        logits = jnp.where(mask, logits, jnp.finfo(logits.dtype).min)
    weights = jax.nn.softmax(logits, axis=-1)
    output = jnp.einsum("bhqk,bkhd->bqhd", weights, v)
    return nn.Dense(dim, name=f"{name}_out")(
        output.reshape(batch, q_len, dim))


def _condition_memory(memory, memory_mask, state, cfg: DecoderConfig):
    memory = nn.Dense(cfg.model_dim, name="memory_projection")(
        jnp.asarray(memory, jnp.float32))
    state_token = nn.Dense(cfg.model_dim, name="state_projection")(
        jnp.asarray(state, jnp.float32)[..., :cfg.state_dim])[:, None, :]
    memory = jnp.concatenate([state_token, memory], axis=1)
    state_mask = jnp.ones((memory.shape[0], 1), dtype=bool)
    memory_mask = jnp.concatenate([state_mask, jnp.asarray(memory_mask, bool)], axis=1)
    return memory, memory_mask


def _decoder_stack(query, memory, memory_mask, cfg: DecoderConfig, query_pos=None):
    """DETR/ACT-style self-attention, cross-attention and FFN stack."""
    for layer in range(cfg.layers):
        prefix = f"decoder_{layer}"
        norm = nn.LayerNorm(name=f"{prefix}_self_norm")(query)
        query_with_pos = norm if query_pos is None else norm + query_pos
        query = query + _attention(
            query_with_pos, query_with_pos, norm,
            dim=cfg.model_dim, heads=cfg.heads,
            name=f"{prefix}_self")
        q_norm = nn.LayerNorm(name=f"{prefix}_cross_q_norm")(query)
        cross_query = q_norm if query_pos is None else q_norm + query_pos
        m_norm = nn.LayerNorm(name=f"{prefix}_cross_m_norm")(memory)
        query = query + _attention(
            cross_query, m_norm, m_norm, dim=cfg.model_dim, heads=cfg.heads,
            name=f"{prefix}_cross", key_mask=memory_mask)
        ff = nn.LayerNorm(name=f"{prefix}_ff_norm")(query)
        ff = nn.Dense(cfg.ff_dim, name=f"{prefix}_ff_in")(ff)
        ff = nn.gelu(ff)
        ff = nn.Dense(cfg.model_dim, name=f"{prefix}_ff_out")(ff)
        query = query + ff
    return nn.LayerNorm(name="decoder_output_norm")(query)


def _cvae_encoder(target_actions, state, action_valid_mask, cls_embedding,
                  cfg: DecoderConfig):
    """Encode the demonstrated chunk into an ACT-style Gaussian latent."""
    batch = target_actions.shape[0]
    cls = jnp.broadcast_to(
        cls_embedding[None, :, :], (batch, 1, cfg.model_dim))
    state_token = nn.Dense(cfg.model_dim, name="cvae_state_projection")(
        jnp.asarray(state, jnp.float32)[..., :cfg.state_dim])[:, None, :]
    action_tokens = nn.Dense(cfg.model_dim, name="cvae_action_projection")(
        jnp.asarray(target_actions, jnp.float32))
    tokens = jnp.concatenate([cls, state_token, action_tokens], axis=1)
    position = sinusoidal_embedding(
        jnp.arange(tokens.shape[1], dtype=jnp.float32), cfg.model_dim)
    tokens = tokens + position[None, :, :]

    if action_valid_mask is None:
        action_valid_mask = jnp.ones(target_actions.shape[:2], bool)
    prefix_mask = jnp.ones((batch, 2), bool)
    token_mask = jnp.concatenate(
        [prefix_mask, jnp.asarray(action_valid_mask, bool)], axis=1)
    for layer in range(cfg.cvae_encoder_layers):
        prefix = f"cvae_encoder_{layer}"
        norm = nn.LayerNorm(name=f"{prefix}_attn_norm")(tokens)
        tokens = tokens + _attention(
            norm, norm, norm, dim=cfg.model_dim, heads=cfg.heads,
            name=f"{prefix}_attn", key_mask=token_mask)
        ff = nn.LayerNorm(name=f"{prefix}_ff_norm")(tokens)
        ff = nn.Dense(cfg.ff_dim, name=f"{prefix}_ff_in")(ff)
        ff = nn.gelu(ff)
        ff = nn.Dense(cfg.model_dim, name=f"{prefix}_ff_out")(ff)
        tokens = tokens + ff
    cls_output = nn.LayerNorm(name="cvae_output_norm")(tokens[:, 0])
    statistics = nn.Dense(2 * cfg.latent_dim, name="cvae_statistics")(cls_output)
    return jnp.split(statistics, 2, axis=-1)


class ACTDecoder(nn.Module):
    """Direct deterministic ACT head with one learned query per action step."""

    config: DecoderConfig = dataclasses.field(default_factory=DecoderConfig)

    @nn.compact
    def __call__(self, memory, memory_mask, state, target_actions=None,
                 action_valid_mask=None, latent_noise=None):
        cfg = self.config
        memory, memory_mask = _condition_memory(memory, memory_mask, state, cfg)
        batch = memory.shape[0]

        mu = jnp.zeros((batch, cfg.latent_dim), jnp.float32)
        logvar = jnp.zeros_like(mu)
        if cfg.use_cvae:
            if target_actions is not None:
                cls_embedding = self.param(
                    "cvae_cls", nn.initializers.normal(stddev=0.02),
                    (1, cfg.model_dim))
                mu, logvar = _cvae_encoder(
                    target_actions, state, action_valid_mask, cls_embedding, cfg)
                if latent_noise is None:
                    latent_noise = jnp.zeros_like(mu)
                latent = mu + jnp.exp(0.5 * logvar) * latent_noise
            else:
                # Match native ACT inference: use the centre of the N(0, I) prior.
                latent = jnp.zeros_like(mu)
            latent_token = nn.Dense(
                cfg.model_dim, name="cvae_latent_projection")(latent)[:, None, :]
            memory = jnp.concatenate([latent_token, memory], axis=1)
            memory_mask = jnp.concatenate(
                [jnp.ones((batch, 1), bool), memory_mask], axis=1)

        if cfg.separate_query_pos:
            learned_query_pos = self.param(
                "action_query_pos", nn.initializers.normal(stddev=0.02),
                (cfg.action_horizon, cfg.model_dim))
            query_pos = jnp.broadcast_to(
                learned_query_pos[None, :, :],
                (batch, cfg.action_horizon, cfg.model_dim))
            query = jnp.zeros_like(query_pos)
        else:
            learned_queries = self.param(
                "action_queries", nn.initializers.normal(stddev=0.02),
                (cfg.action_horizon, cfg.model_dim))
            query = jnp.broadcast_to(
                learned_queries[None, :, :],
                (batch, cfg.action_horizon, cfg.model_dim))
            query_pos = None

        query = _decoder_stack(
            query, memory, memory_mask, cfg, query_pos=query_pos)
        actions = nn.Dense(cfg.action_dim, name="action_output")(query)
        if cfg.predict_is_pad:
            is_pad_logits = nn.Dense(1, name="is_pad_output")(query)[..., 0]
        else:
            is_pad_logits = jnp.zeros(actions.shape[:2], jnp.float32)
        return {
            "actions": actions,
            "is_pad_logits": is_pad_logits,
            "mu": mu,
            "logvar": logvar,
        }


class DiffusionDecoder(nn.Module):
    """Transformer diffusion-policy head that directly predicts action noise."""

    config: DecoderConfig = dataclasses.field(default_factory=DecoderConfig)

    @nn.compact
    def __call__(self, memory, memory_mask, state, noisy_action, timestep):
        cfg = self.config
        memory, memory_mask = _condition_memory(memory, memory_mask, state, cfg)
        noisy_action = jnp.asarray(noisy_action, jnp.float32)
        query = nn.Dense(cfg.model_dim, name="noisy_action_projection")(noisy_action)
        position = self.param(
            "action_position", nn.initializers.normal(stddev=0.02),
            (cfg.action_horizon, cfg.model_dim))
        time = sinusoidal_embedding(timestep, cfg.model_dim)
        time = nn.Dense(cfg.model_dim, name="time_projection_1")(time)
        time = nn.silu(time)
        time = nn.Dense(cfg.model_dim, name="time_projection_2")(time)
        query = query + position[None, :, :] + time[:, None, :]
        query = _decoder_stack(query, memory, memory_mask, cfg)
        return nn.Dense(cfg.action_dim, name="noise_output")(query)


def parameter_count(params) -> int:
    return int(sum(int(x.size) for x in jax.tree_util.tree_leaves(params)))
