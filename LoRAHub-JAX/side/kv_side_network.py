"""A small KV-conditioned residual network for Pi0 flow matching.

The Pi0 backbone is frozen.  This module consumes detached prefix K/V
summaries and detached action-expert hidden states, then predicts a residual
flow velocity for the first ``output_dim`` action dimensions.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
from flax import linen as nn


@dataclasses.dataclass(frozen=True)
class KVSideConfig:
    """Shape/configuration for the deliberately small side network."""

    context_dim: int = 512
    action_hidden_dim: int = 1024
    action_projection_dim: int = 64
    time_dim: int = 32
    hidden_dim: int = 128
    output_dim: int = 14
    residual_scale: float = 0.1


def sinusoidal_time_embedding(time: jax.Array, dim: int) -> jax.Array:
    """Return a deterministic sin/cos embedding for scalar flow time."""
    if dim % 2:
        raise ValueError(f"time embedding dimension must be even, got {dim}")
    time = jnp.asarray(time, dtype=jnp.float32).reshape(-1, 1)
    frequencies = jnp.exp(
        jnp.linspace(jnp.log(1.0), jnp.log(1000.0), dim // 2))
    angles = time * frequencies[None, :]
    return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)


class KVSideResidual(nn.Module):
    """Tiny residual head attached to a frozen Pi0 flow field.

    Inputs:
      ``kv_context``: ``[B, context_dim]`` pooled frozen prefix K/V.
      ``action_hidden``: ``[B, action_horizon, action_hidden_dim]`` frozen
        action-expert hidden states for the current noisy action.
      ``time``: ``[B]`` flow-matching time.

    Output:
      ``[B, action_horizon, output_dim]`` normalized flow-velocity residual.

    The final layer is zero initialized, so the composed Pi0 behavior is
    unchanged at the beginning of side training.
    """

    config: KVSideConfig = dataclasses.field(default_factory=KVSideConfig)

    @nn.compact
    def __call__(self, kv_context, action_hidden, time):
        cfg = self.config
        kv_context = jnp.asarray(kv_context, dtype=jnp.float32)
        action_hidden = jnp.asarray(action_hidden, dtype=jnp.float32)
        time = jnp.asarray(time, dtype=jnp.float32).reshape(-1)

        if kv_context.ndim != 2 or kv_context.shape[-1] != cfg.context_dim:
            raise ValueError(
                f"kv_context must be [B,{cfg.context_dim}], got {kv_context.shape}")
        if action_hidden.ndim != 3 or action_hidden.shape[-1] != cfg.action_hidden_dim:
            raise ValueError(
                "action_hidden must be [B,H,"
                f"{cfg.action_hidden_dim}], got {action_hidden.shape}")

        action = nn.Dense(
            cfg.action_projection_dim,
            name="action_projection",
            kernel_init=nn.initializers.lecun_normal(),
        )(action_hidden)
        action = jax.nn.gelu(action)

        t = sinusoidal_time_embedding(time, cfg.time_dim)
        t = jnp.broadcast_to(t[:, None, :],
                             (action_hidden.shape[0], action_hidden.shape[1], cfg.time_dim))
        context = jnp.broadcast_to(
            kv_context[:, None, :],
            (action_hidden.shape[0], action_hidden.shape[1], cfg.context_dim),
        )

        x = jnp.concatenate([context, action, t], axis=-1)
        x = nn.Dense(cfg.hidden_dim, name="hidden")(x)
        x = jax.nn.gelu(x)
        residual = nn.Dense(
            cfg.output_dim,
            name="output",
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.zeros_init(),
        )(x)
        return residual


@dataclasses.dataclass(frozen=True)
class KVSideQFormerConfig(KVSideConfig):
    """Configuration for the token-level KV Q-Former Side adapter.

    The frozen Pi0 supplies ``[B, prefix_tokens, kv_token_dim]`` K/V tokens.
    A few learned queries compress them before the same small action residual
    head used by the pooling baseline.
    """

    kv_token_dim: int = 512
    num_queries: int = 4
    qformer_dim: int = 128
    qformer_heads: int = 4
    qformer_layers: int = 2
    qformer_ff_dim: int = 256


def _attention(q, k, v, *, dim: int, heads: int, name: str):
    """Small explicit multi-head attention helper for stable Linen shapes."""
    if dim % heads:
        raise ValueError(f"qformer_dim={dim} must be divisible by heads={heads}")
    head_dim = dim // heads
    batch, q_len = q.shape[:2]
    k_len = k.shape[1]

    q = nn.Dense(dim, name=f"{name}_q")(q)
    k = nn.Dense(dim, name=f"{name}_k")(k)
    v = nn.Dense(dim, name=f"{name}_v")(v)
    q = q.reshape(batch, q_len, heads, head_dim)
    k = k.reshape(batch, k_len, heads, head_dim)
    v = v.reshape(batch, k_len, heads, head_dim)

    logits = jnp.einsum("bqhd,bkhd->bhqk", q, k)
    logits = logits * (head_dim ** -0.5)
    weights = jax.nn.softmax(logits, axis=-1)
    output = jnp.einsum("bhqk,bkhd->bqhd", weights, v)
    output = output.reshape(batch, q_len, dim)
    return nn.Dense(dim, name=f"{name}_out")(output)


class KVSideQFormerResidual(nn.Module):
    """A lightweight Q-Former-conditioned version of ``KVSideResidual``.

    Only the frozen Pi0 K/V pooling operation changes: token-level K/V is
    compressed by learned queries, while the action/time residual head remains
    deliberately small and outputs the same 14D residual.
    """

    config: KVSideQFormerConfig = dataclasses.field(
        default_factory=KVSideQFormerConfig)

    @nn.compact
    def __call__(self, kv_tokens, action_hidden, time):
        cfg = self.config
        kv_tokens = jnp.asarray(kv_tokens, dtype=jnp.float32)
        action_hidden = jnp.asarray(action_hidden, dtype=jnp.float32)
        time = jnp.asarray(time, dtype=jnp.float32).reshape(-1)

        if kv_tokens.ndim != 3 or kv_tokens.shape[-1] != cfg.kv_token_dim:
            raise ValueError(
                f"kv_tokens must be [B,T,{cfg.kv_token_dim}], got {kv_tokens.shape}")
        if action_hidden.ndim != 3 or action_hidden.shape[-1] != cfg.action_hidden_dim:
            raise ValueError(
                "action_hidden must be [B,H,"
                f"{cfg.action_hidden_dim}], got {action_hidden.shape}")
        if cfg.num_queries <= 0 or cfg.qformer_layers <= 0:
            raise ValueError("num_queries and qformer_layers must be positive")

        batch = kv_tokens.shape[0]
        query_tokens = self.param(
            "query_tokens",
            nn.initializers.normal(stddev=0.02),
            (cfg.num_queries, cfg.qformer_dim),
        )
        x = jnp.broadcast_to(
            query_tokens[None, :, :],
            (batch, cfg.num_queries, cfg.qformer_dim),
        )

        for layer in range(cfg.qformer_layers):
            prefix = f"qformer_{layer}"
            x_norm = nn.LayerNorm(name=f"{prefix}_self_norm")(x)
            x = x + _attention(
                x_norm, x_norm, x_norm,
                dim=cfg.qformer_dim,
                heads=cfg.qformer_heads,
                name=f"{prefix}_self_attn",
            )

            query_norm = nn.LayerNorm(name=f"{prefix}_cross_norm")(x)
            kv_norm = nn.LayerNorm(name=f"{prefix}_kv_norm")(kv_tokens)
            x = x + _attention(
                query_norm, kv_norm, kv_norm,
                dim=cfg.qformer_dim,
                heads=cfg.qformer_heads,
                name=f"{prefix}_cross_attn",
            )

            ff_norm = nn.LayerNorm(name=f"{prefix}_ff_norm")(x)
            ff = nn.Dense(cfg.qformer_ff_dim, name=f"{prefix}_ff_in")(ff_norm)
            ff = jax.nn.gelu(ff)
            ff = nn.Dense(cfg.qformer_dim, name=f"{prefix}_ff_out")(ff)
            x = x + ff

        # Keep one context vector per query.  With the production setting
        # num_queries == action_horizon == 50, query t is aligned with action
        # chunk t instead of being flattened into one global context.
        context = nn.Dense(
            cfg.context_dim,
            name="qformer_context_projection",
        )(x)
        context = jax.nn.gelu(context)

        if context.shape[1] == action_hidden.shape[1]:
            # Temporal Q-Former path: [B, 50, context_dim].
            pass
        else:
            # Keep small unit-test/legacy configurations functional.  The
            # current training configuration uses 50 queries and takes the
            # temporal branch above.
            context = jnp.mean(context, axis=1)

        action = nn.Dense(
            cfg.action_projection_dim,
            name="action_projection",
            kernel_init=nn.initializers.lecun_normal(),
        )(action_hidden)
        action = jax.nn.gelu(action)

        t = sinusoidal_time_embedding(time, cfg.time_dim)
        t = jnp.broadcast_to(
            t[:, None, :],
            (action_hidden.shape[0], action_hidden.shape[1], cfg.time_dim),
        )
        if context.ndim == 2:
            context = jnp.broadcast_to(
                context[:, None, :],
                (action_hidden.shape[0], action_hidden.shape[1], cfg.context_dim),
            )
        x = jnp.concatenate([context, action, t], axis=-1)
        x = nn.Dense(cfg.hidden_dim, name="hidden")(x)
        x = jax.nn.gelu(x)
        return nn.Dense(
            cfg.output_dim,
            name="output",
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.zeros_init(),
        )(x)


def parameter_count(params) -> int:
    """Count scalar parameters in a Linen parameter tree."""
    return int(sum(int(x.size) for x in jax.tree_util.tree_leaves(params)))
