"""Inference helper for a trained KV side residual."""

from __future__ import annotations

import json
import pathlib
import pickle
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from script.common.data_loader import load_norm_stats
from side.kv_side_network import KVSideQFormerResidual, KVSideResidual
from side.pi0_runtime import (
    load_frozen_pi0,
    make_side_sampler_fn,
    prepare_observation_batch,
    qformer_shape_config,
    side_shape_config,
)


class KVSidePi0:
    """Frozen merged Pi0 plus a trained KV residual side network.

    ``predict`` expects the same HWC camera/state/instruction dictionaries used
    by ``prepare_observation_batch`` and returns unnormalized 14D delta actions.
    """

    def __init__(self, config_path: str | pathlib.Path):
        config_path = pathlib.Path(config_path).expanduser().resolve()
        with config_path.open(encoding="utf-8") as handle:
            self.config = json.load(handle)
        # Configs may live in the standard config/ directory or in a nested
        # experiment suite.  Resolve the LoRAHub root by its package layout.
        root = next(
            (parent for parent in config_path.parents
             if (parent / "side").is_dir() and (parent / "vla_lorahub").is_dir()),
            None,
        )
        if root is None:
            raise FileNotFoundError(
                f"无法从 side config 定位 LoRAHub-JAX 根目录: {config_path}")
        model_cfg = self.config["model"]
        data_cfg = self.config["data"]
        self.norm_stats = load_norm_stats(str(root / data_cfg["norm_stats_path"]))
        self.graphdef, self.state, self.model = load_frozen_pi0(
            root / model_cfg["merged_checkpoint"],
            model_cfg["train_config_name"],
        )
        self.tokenizer = __import__(
            "openpi.models.tokenizer", fromlist=["PaligemmaTokenizer"]
        ).PaligemmaTokenizer(max_len=self.model.max_token_len)
        architecture = str(model_cfg.get("side_architecture", "pool")).lower()
        residual_scale = float(model_cfg.get("residual_scale", 0.1))
        if architecture in {"qformer", "q-former"}:
            side_cfg = qformer_shape_config(
                self.model,
                output_dim=int(model_cfg.get("action_dim", 14)),
                num_queries=int(model_cfg.get("qformer_num_queries", 4)),
                qformer_dim=int(model_cfg.get("qformer_dim", 128)),
                qformer_heads=int(model_cfg.get("qformer_heads", 4)),
                qformer_layers=int(model_cfg.get("qformer_layers", 2)),
                qformer_ff_dim=int(model_cfg.get("qformer_ff_dim", 256)),
            )
            self.side_model = KVSideQFormerResidual(config=side_cfg)
            kv_mode = "tokens"
        elif architecture in {"pool", "pooled", "mean_pool"}:
            side_cfg = side_shape_config(
                self.model, output_dim=int(model_cfg.get("action_dim", 14)))
            self.side_model = KVSideResidual(config=side_cfg)
            kv_mode = "pooled"
        else:
            raise ValueError(
                f"unsupported model.side_architecture={architecture!r}; "
                "expected 'pool' or 'qformer'"
            )
        side_cfg = dataclasses.replace(side_cfg, residual_scale=residual_scale)
        if architecture in {"qformer", "q-former"}:
            dummy_context = jnp.zeros(
                (1, 1, side_cfg.kv_token_dim), jnp.float32)
        else:
            dummy_context = jnp.zeros((1, side_cfg.context_dim), jnp.float32)
        dummy_hidden = jnp.zeros((1, self.model.action_horizon,
                                  side_cfg.action_hidden_dim), jnp.float32)
        dummy_time = jnp.zeros((1,), jnp.float32)
        template = self.side_model.init(
            jax.random.key(0), dummy_context, dummy_hidden, dummy_time)["params"]
        side_ckpt = root / self.config["output_dir"] / "checkpoints" / "latest.pkl"
        with side_ckpt.open("rb") as handle:
            payload = pickle.load(handle)
        self.side_params = serialization.from_state_dict(template, payload["params"])
        self.sample_fn = make_side_sampler_fn(
            self.graphdef, self.side_model,
            layer_indices=tuple(model_cfg.get("kv_layer_indices", [5, 11, 17])),
            kv_mode=kv_mode,
        )
        self.residual_scale = residual_scale

    def predict(self, inputs: list[dict], *, diffusion_steps: int = 10,
                seed: int = 0) -> list[np.ndarray]:
        observation = prepare_observation_batch(
            inputs, self.norm_stats, self.tokenizer)
        normalized = self.sample_fn(
            self.state,
            self.side_params,
            jax.random.key(seed),
            observation,
            num_steps=int(diffusion_steps),
            residual_scale=self.residual_scale,
        )
        stats = self.norm_stats["actions"]
        mean = jnp.asarray(stats["mean"], dtype=jnp.float32)
        std = jnp.asarray(stats["std"], dtype=jnp.float32)
        actions = normalized * (std[None, None, :] + 1e-6) + mean[None, None, :]
        actions = np.asarray(actions[..., :14], dtype=np.float32)
        return [actions[i] for i in range(actions.shape[0])]
