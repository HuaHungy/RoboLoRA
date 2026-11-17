"""Deployment policy for direct ACT/DP heads on a frozen LoRAHub backbone."""

from __future__ import annotations

import json
import pathlib
import pickle

import jax
import jax.numpy as jnp
import numpy as np

from action_decoders.backbone import (
    load_frozen_backbone,
    make_prefix_feature_fn,
    prepare_observation_batch,
    unnormalize_actions,
)
from action_decoders.diffusion import cosine_schedule
from action_decoders.models import ACTDecoder, DecoderConfig, DiffusionDecoder
from script.common.data_loader import load_norm_stats


def _decoder_config(cfg: dict) -> DecoderConfig:
    model = cfg["decoder"]
    return DecoderConfig(
        action_dim=int(model.get("action_dim", 14)),
        action_horizon=int(model.get("action_horizon", 50)),
        state_dim=int(model.get("state_dim", 32)),
        model_dim=int(model.get("model_dim", 256)),
        heads=int(model.get("heads", 4)),
        layers=int(model.get("layers", 4)),
        ff_dim=int(model.get("ff_dim", 1024)),
        separate_query_pos=bool(model.get("separate_query_pos", False)),
        predict_is_pad=bool(model.get("predict_is_pad", False)),
        use_cvae=bool(model.get("use_cvae", False)),
        latent_dim=int(model.get("latent_dim", 32)),
        cvae_encoder_layers=int(model.get("cvae_encoder_layers", 2)),
    )


class LoRAHubActionDecoderPolicy:
    """LoRAHub observation encoder followed by one direct action decoder."""

    def __init__(self, config_path: str | pathlib.Path):
        config_path = pathlib.Path(config_path).expanduser().resolve()
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        root = next(
            (parent for parent in config_path.parents
             if (parent / "vla_lorahub").is_dir() and (parent / "action_decoders").is_dir()),
            None)
        if root is None:
            raise FileNotFoundError(f"cannot locate LoRAHub-JAX root from {config_path}")
        self.root = root
        backbone_cfg = self.config["backbone"]
        decoder_cfg = _decoder_config(self.config)
        self.decoder_cfg = decoder_cfg
        self.decoder_type = str(self.config["decoder"]["type"]).lower()
        self.norm_stats = load_norm_stats(str(root / self.config["data"]["norm_stats_path"]))
        self.graphdef, self.frozen_state, self.backbone = load_frozen_backbone(
            root / backbone_cfg["merged_checkpoint"], backbone_cfg["train_config_name"])
        self.tokenizer = __import__(
            "openpi.models.tokenizer", fromlist=["PaligemmaTokenizer"]
        ).PaligemmaTokenizer(max_len=self.backbone.max_token_len)
        self.feature_fn = make_prefix_feature_fn(self.graphdef)

        if self.decoder_type == "act":
            self.decoder = ACTDecoder(config=decoder_cfg)
        elif self.decoder_type == "dp":
            self.decoder = DiffusionDecoder(config=decoder_cfg)
        else:
            raise ValueError(f"unsupported decoder.type={self.decoder_type!r}")

        checkpoint_name = str(self.config.get("deployment", {}).get("checkpoint", "best.pkl"))
        checkpoint = root / self.config["output_dir"] / "checkpoints" / checkpoint_name
        if not checkpoint.is_file():
            raise FileNotFoundError(f"decoder checkpoint not found: {checkpoint}")
        with checkpoint.open("rb") as handle:
            payload = pickle.load(handle)
        state_dict = payload.get("ema_params", payload["params"])
        self.params = jax.tree_util.tree_map(jnp.asarray, state_dict)
        self._counter = 0
        deployment_cfg = self.config.get("deployment", {})
        self.temporal_aggregation = bool(
            deployment_cfg.get("temporal_aggregation", False))
        self.temporal_agg_decay = float(
            deployment_cfg.get("temporal_agg_decay", 0.01))
        self.temporal_agg_recent_weighted = bool(
            deployment_cfg.get("temporal_agg_recent_weighted", True))
        self._aggregation_step = 0
        self._action_history: list[tuple[int, np.ndarray]] = []

        if self.decoder_type == "act":
            self._act_fn = jax.jit(lambda params, memory, mask, state: self.decoder.apply(
                {"params": params}, memory, mask, state)["actions"])
        else:
            diffusion_cfg = self.config.get("diffusion", {})
            self.train_timesteps = int(diffusion_cfg.get("train_timesteps", 100))
            self.inference_steps = int(diffusion_cfg.get("inference_steps", 10))
            self.sample_clip = float(diffusion_cfg.get("sample_clip", 5.0))
            self.alpha_bars = jnp.asarray(cosine_schedule(self.train_timesteps)[2])
            self._dp_fn = self._build_dp_sampler()

    @property
    def action_steps(self) -> int:
        steps = int(self.config.get("deployment", {}).get("action_steps", 5))
        if self.temporal_aggregation and steps != 1:
            raise ValueError("temporal_aggregation requires deployment.action_steps=1")
        return steps

    def reset(self):
        self._counter = 0
        self._aggregation_step = 0
        self._action_history.clear()

    def aggregate_absolute_actions(self, action_chunk: np.ndarray) -> np.ndarray:
        """Fuse overlapping absolute-action chunks at the current control step."""
        chunk = np.asarray(action_chunk, np.float32)
        if chunk.ndim != 2 or chunk.shape[-1] != self.decoder_cfg.action_dim:
            raise ValueError(f"invalid action chunk shape: {chunk.shape}")
        if not self.temporal_aggregation:
            return chunk

        current = self._aggregation_step
        self._action_history.append((current, chunk.copy()))
        candidates, ages = [], []
        for start, values in self._action_history:
            offset = current - start
            if 0 <= offset < len(values):
                candidates.append(values[offset])
                ages.append(offset)
        if not candidates:
            raise RuntimeError("temporal aggregation has no action candidate")
        ages = np.asarray(ages, np.float32)
        if self.temporal_agg_recent_weighted:
            weights = np.exp(-self.temporal_agg_decay * ages)
        else:
            weights = np.exp(-self.temporal_agg_decay * (ages.max() - ages))
        weights /= np.maximum(weights.sum(), 1e-12)
        action = np.sum(np.stack(candidates) * weights[:, None], axis=0)

        self._aggregation_step += 1
        next_step = self._aggregation_step
        self._action_history = [
            (start, values) for start, values in self._action_history
            if start + len(values) > next_step
        ]
        return action[None, :].astype(np.float32)

    def _build_dp_sampler(self):
        decoder = self.decoder
        cfg = self.decoder_cfg
        alpha_bars = self.alpha_bars
        inference_steps = self.inference_steps
        sample_clip = self.sample_clip
        timesteps = jnp.asarray(
            np.rint(np.linspace(self.train_timesteps - 1, 0, inference_steps)).astype(np.int32))

        @jax.jit
        def sample(params, memory, memory_mask, state, key):
            action = jax.random.normal(
                key, (memory.shape[0], cfg.action_horizon, cfg.action_dim), jnp.float32)

            def body(index, current):
                timestep_scalar = timesteps[index]
                timestep = jnp.full((current.shape[0],), timestep_scalar, jnp.int32)
                predicted_noise = decoder.apply(
                    {"params": params}, memory, memory_mask, state, current, timestep)
                alpha = alpha_bars[timestep_scalar]
                clean = (current - jnp.sqrt(1.0 - alpha) * predicted_noise) / jnp.sqrt(alpha)
                clean = jnp.clip(clean, -sample_clip, sample_clip)
                has_next = index + 1 < inference_steps
                next_index = jnp.minimum(index + 1, inference_steps - 1)
                next_alpha = alpha_bars[timesteps[next_index]]
                next_sample = (
                    jnp.sqrt(next_alpha) * clean
                    + jnp.sqrt(1.0 - next_alpha) * predicted_noise)
                return jnp.where(has_next, next_sample, clean)

            return jax.lax.fori_loop(0, inference_steps, body, action)

        return sample

    def predict(self, inputs: list[dict], *, seed: int = 0) -> list[np.ndarray]:
        observation = prepare_observation_batch(inputs, self.norm_stats, self.tokenizer)
        memory, memory_mask, state = self.feature_fn(self.frozen_state, observation)
        if self.decoder_type == "act":
            normalized = self._act_fn(self.params, memory, memory_mask, state)
        else:
            normalized = self._dp_fn(
                self.params, memory, memory_mask, state,
                jax.random.key(int(seed) + self._counter))
            self._counter += 1
        actions = unnormalize_actions(
            normalized, self.norm_stats, self.decoder_cfg.action_dim)
        actions = np.asarray(actions, np.float32)
        return [actions[index] for index in range(actions.shape[0])]
