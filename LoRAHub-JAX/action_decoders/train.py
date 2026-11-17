#!/usr/bin/env python
"""Train a direct ACT or diffusion action head on frozen LoRAHub features."""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import pickle
import sys
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
from tqdm import tqdm

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from action_decoders.backbone import (
    load_frozen_backbone,
    make_prefix_feature_fn,
    normalize_actions,
    prepare_observation_batch,
)
from action_decoders.diffusion import cosine_schedule
from action_decoders.models import ACTDecoder, DecoderConfig, DiffusionDecoder, parameter_count
from script.common.data_loader import Pi0FrameDataset, load_norm_stats


def _resolve(value: str) -> pathlib.Path:
    path = pathlib.Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def _atomic_pickle(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            pickle.dump(jax.device_get(payload), handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _episode_indices(dataset: Pi0FrameDataset, selected) -> np.ndarray:
    selected = set(int(x) for x in selected)
    values = []
    start = 0
    for episode, length in enumerate(dataset.episode_lengths):
        if episode in selected:
            values.extend(range(start, start + length))
        start += length
    return np.asarray(values, np.int64)


def _padded_batch(values: np.ndarray, start: int, size: int):
    batch = values[start:start + size]
    mask = np.ones(len(batch), np.float32)
    if len(batch) < size:
        if not len(batch):
            raise ValueError("empty batch")
        batch = np.pad(batch, (0, size - len(batch)), mode="edge")
        mask = np.pad(mask, (0, size - len(mask)), constant_values=0.0)
    return batch, mask


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


def _action_objective(prediction, target, sample_mask, action_valid_mask,
                      action_dim, grip_weight, smooth_weight, jerk_weight,
                      reconstruction_loss="weighted_mse", huber_delta=1.0):
    weights = jnp.ones((action_dim,), jnp.float32)
    gripper_mask = jnp.zeros((action_dim,), bool)
    for index in (6, 13):
        if index < action_dim:
            weights = weights.at[index].set(grip_weight)
            gripper_mask = gripper_mask.at[index].set(True)
    mask = sample_mask[:, None, None] * action_valid_mask[:, :, None]
    error = prediction - target
    if reconstruction_loss == "weighted_mse":
        point_loss = jnp.square(error)
    elif reconstruction_loss == "joint_huber_gripper_l1":
        absolute = jnp.abs(error)
        joint_huber = jnp.where(
            absolute <= huber_delta,
            0.5 * jnp.square(error),
            huber_delta * (absolute - 0.5 * huber_delta))
        point_loss = jnp.where(gripper_mask[None, None, :], absolute, joint_huber)
    else:
        raise ValueError(f"unknown reconstruction_loss={reconstruction_loss!r}")
    denom = jnp.maximum(jnp.sum(mask) * jnp.sum(weights), 1.0)
    reconstruction = jnp.sum(point_loss * weights * mask) / denom

    velocity_error = jnp.diff(prediction, axis=1) - jnp.diff(target, axis=1)
    velocity_mask = mask[:, 1:] * mask[:, :-1]
    smooth_denom = jnp.maximum(jnp.sum(velocity_mask) * action_dim, 1.0)
    smooth = jnp.sum(jnp.square(velocity_error) * velocity_mask) / smooth_denom

    acceleration_error = jnp.diff(prediction, n=2, axis=1) - jnp.diff(target, n=2, axis=1)
    acceleration_mask = mask[:, 2:] * mask[:, 1:-1] * mask[:, :-2]
    jerk_denom = jnp.maximum(jnp.sum(acceleration_mask) * action_dim, 1.0)
    jerk = jnp.sum(jnp.square(acceleration_error) * acceleration_mask) / jerk_denom
    total = reconstruction + smooth_weight * smooth + jerk_weight * jerk
    return total, reconstruction, smooth, jerk


def _make_act_steps(model, optimizer, decoder_cfg, training_cfg):
    grip_weight = float(training_cfg.get("gripper_weight", 2.0))
    smooth_weight = float(training_cfg.get("smooth_weight", 0.05))
    jerk_weight = float(training_cfg.get("jerk_weight", 0.01))
    ema_decay = float(training_cfg.get("ema_decay", 0.999))
    use_action_valid_mask = bool(training_cfg.get("use_action_valid_mask", False))
    reconstruction_loss = str(training_cfg.get("reconstruction_loss", "weighted_mse"))
    huber_delta = float(training_cfg.get("huber_delta", 1.0))
    pad_loss_weight = float(training_cfg.get("pad_loss_weight", 0.1))
    kl_weight = float(training_cfg.get("kl_weight", 0.01))

    @jax.jit
    def train_step(params, ema_params, opt_state, memory, memory_mask, state,
                   target, sample_mask, action_valid_mask, key, kl_factor):
        def loss_fn(current):
            if decoder_cfg.use_cvae:
                latent_noise = jax.random.normal(
                    key, (target.shape[0], decoder_cfg.latent_dim), jnp.float32)
                output = model.apply(
                    {"params": current}, memory, memory_mask, state,
                    target, action_valid_mask, latent_noise)
            else:
                output = model.apply(
                    {"params": current}, memory, memory_mask, state)
            prediction = output["actions"]
            objective_mask = (action_valid_mask if use_action_valid_mask else
                              jnp.ones_like(action_valid_mask))
            total, reconstruction, smooth, jerk = _action_objective(
                prediction, target, sample_mask, objective_mask,
                decoder_cfg.action_dim, grip_weight, smooth_weight, jerk_weight,
                reconstruction_loss, huber_delta)
            pad_bce = jnp.asarray(0.0)
            if decoder_cfg.predict_is_pad:
                pad_target = 1.0 - action_valid_mask
                pad_values = optax.sigmoid_binary_cross_entropy(
                    output["is_pad_logits"], pad_target)
                pad_denom = jnp.maximum(
                    jnp.sum(sample_mask) * pad_values.shape[1], 1.0)
                pad_bce = jnp.sum(pad_values * sample_mask[:, None]) / pad_denom
                total = total + pad_loss_weight * pad_bce
            kl = jnp.asarray(0.0)
            if decoder_cfg.use_cvae:
                mu, logvar = output["mu"], output["logvar"]
                kl = -0.5 * jnp.mean(jnp.sum(
                    1.0 + logvar - jnp.square(mu) - jnp.exp(logvar), axis=-1))
                total = total + kl_weight * kl_factor * kl
            return total, (reconstruction, smooth, jerk, pad_bce, kl)

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        ema_params = optax.incremental_update(params, ema_params, 1.0 - ema_decay)
        return params, ema_params, opt_state, loss, metrics

    @jax.jit
    def eval_step(params, memory, memory_mask, state, target, sample_mask,
                  action_valid_mask):
        if decoder_cfg.use_cvae:
            output = model.apply(
                {"params": params}, memory, memory_mask, state,
                target, action_valid_mask,
                jnp.zeros((target.shape[0], decoder_cfg.latent_dim), jnp.float32))
        else:
            output = model.apply({"params": params}, memory, memory_mask, state)
        prediction = output["actions"]
        objective_mask = (action_valid_mask if use_action_valid_mask else
                          jnp.ones_like(action_valid_mask))
        total, reconstruction, smooth, jerk = _action_objective(
            prediction, target, sample_mask, objective_mask,
            decoder_cfg.action_dim, grip_weight, smooth_weight, jerk_weight,
            reconstruction_loss, huber_delta)
        pad_bce = jnp.asarray(0.0)
        if decoder_cfg.predict_is_pad:
            pad_target = 1.0 - action_valid_mask
            pad_values = optax.sigmoid_binary_cross_entropy(
                output["is_pad_logits"], pad_target)
            pad_denom = jnp.maximum(
                jnp.sum(sample_mask) * pad_values.shape[1], 1.0)
            pad_bce = jnp.sum(pad_values * sample_mask[:, None]) / pad_denom
            total = total + pad_loss_weight * pad_bce
        kl = jnp.asarray(0.0)
        if decoder_cfg.use_cvae:
            mu, logvar = output["mu"], output["logvar"]
            kl = -0.5 * jnp.mean(jnp.sum(
                1.0 + logvar - jnp.square(mu) - jnp.exp(logvar), axis=-1))
            total = total + kl_weight * kl
        return total, (reconstruction, smooth, jerk, pad_bce, kl)

    return train_step, eval_step


def _make_dp_steps(model, optimizer, decoder_cfg, training_cfg, alpha_bars):
    ema_decay = float(training_cfg.get("ema_decay", 0.999))
    grip_weight = float(training_cfg.get("gripper_weight", 1.0))
    action_weights = jnp.ones((decoder_cfg.action_dim,), jnp.float32)
    for index in (6, 13):
        if index < decoder_cfg.action_dim:
            action_weights = action_weights.at[index].set(grip_weight)
    use_action_valid_mask = bool(training_cfg.get("use_action_valid_mask", False))

    @jax.jit
    def train_step(params, ema_params, opt_state, memory, memory_mask, state,
                   target, sample_mask, action_valid_mask, key):
        noise_key, time_key = jax.random.split(key)
        noise = jax.random.normal(noise_key, target.shape, jnp.float32)
        timestep = jax.random.randint(
            time_key, (target.shape[0],), 0, alpha_bars.shape[0])
        alpha = alpha_bars[timestep][:, None, None]
        noisy = jnp.sqrt(alpha) * target + jnp.sqrt(1.0 - alpha) * noise

        def loss_fn(current):
            prediction = model.apply(
                {"params": current}, memory, memory_mask, state, noisy, timestep)
            step_mask = (action_valid_mask if use_action_valid_mask else
                         jnp.ones_like(action_valid_mask))
            mask = sample_mask[:, None, None] * step_mask[:, :, None]
            denom = jnp.maximum(
                jnp.sum(mask) * jnp.sum(action_weights), 1.0)
            loss = jnp.sum(jnp.square(prediction - noise) * action_weights * mask) / denom
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        ema_params = optax.incremental_update(params, ema_params, 1.0 - ema_decay)
        zero = jnp.asarray(0.0)
        return params, ema_params, opt_state, loss, (loss, zero, zero, zero, zero)

    @jax.jit
    def eval_step(params, memory, memory_mask, state, target, sample_mask,
                  action_valid_mask, key):
        noise_key, time_key = jax.random.split(key)
        noise = jax.random.normal(noise_key, target.shape, jnp.float32)
        timestep = jax.random.randint(
            time_key, (target.shape[0],), 0, alpha_bars.shape[0])
        alpha = alpha_bars[timestep][:, None, None]
        noisy = jnp.sqrt(alpha) * target + jnp.sqrt(1.0 - alpha) * noise
        prediction = model.apply(
            {"params": params}, memory, memory_mask, state, noisy, timestep)
        step_mask = (action_valid_mask if use_action_valid_mask else
                     jnp.ones_like(action_valid_mask))
        mask = sample_mask[:, None, None] * step_mask[:, :, None]
        denom = jnp.maximum(
            jnp.sum(mask) * jnp.sum(action_weights), 1.0)
        loss = jnp.sum(jnp.square(prediction - noise) * action_weights * mask) / denom
        zero = jnp.asarray(0.0)
        return loss, (loss, zero, zero, zero, zero)

    return train_step, eval_step


def _plot_epochs(path: pathlib.Path, output_dir: pathlib.Path) -> None:
    if not path.is_file():
        return
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    epochs = [int(row["epoch"]) for row in rows]
    train = [float(row["train_loss"]) for row in rows]
    validation = [float(row["val_loss"]) for row in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train, marker="o", markersize=3, label="train")
    ax.plot(epochs, validation, marker="o", markersize=3, label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Direct action decoder loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "logs" / "loss_curve.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke-test", action="store_true",
                        help="load the real backbone and run exactly one update without writing output")
    args = parser.parse_args()
    config_path = pathlib.Path(args.config).expanduser().resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    backbone_cfg = cfg["backbone"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    decoder_cfg = _decoder_config(cfg)
    decoder_type = str(cfg["decoder"]["type"]).lower()
    if decoder_type not in {"act", "dp"}:
        raise ValueError("decoder.type must be 'act' or 'dp'")

    norm_path = _resolve(data_cfg["norm_stats_path"])
    norm_stats = load_norm_stats(str(norm_path))
    dataset = Pi0FrameDataset(
        str(_resolve(data_cfg["data_dir"])), str(norm_path),
        state_dim=decoder_cfg.state_dim,
        action_dim=decoder_cfg.action_dim,
        action_horizon=decoder_cfg.action_horizon,
        episode_cache_size=int(data_cfg.get("episode_cache_size", 10)))
    validation_episodes = list(map(int, data_cfg.get("validation_episodes", [8, 9])))
    all_episodes = set(range(len(dataset.episode_paths)))
    if not validation_episodes or not set(validation_episodes) < all_episodes:
        raise ValueError("validation_episodes must be a non-empty proper subset of episodes")
    train_episodes = sorted(all_episodes - set(validation_episodes))
    train_indices = _episode_indices(dataset, train_episodes)
    val_indices = _episode_indices(dataset, validation_episodes)

    batch_size = int(train_cfg.get("batch_size", 2))
    epochs = int(train_cfg.get("epochs", 30))
    seed = int(train_cfg.get("seed", 42))
    print(f"[Decoder] type={decoder_type} train_episodes={train_episodes} "
          f"val_episodes={validation_episodes} frames={len(train_indices)}/{len(val_indices)} "
          f"batch={batch_size} epochs={epochs}")

    graphdef, frozen_state, frozen_model = load_frozen_backbone(
        _resolve(backbone_cfg["merged_checkpoint"]),
        backbone_cfg["train_config_name"])
    tokenizer = __import__(
        "openpi.models.tokenizer", fromlist=["PaligemmaTokenizer"]
    ).PaligemmaTokenizer(max_len=frozen_model.max_token_len)
    feature_fn = make_prefix_feature_fn(graphdef)

    initial_indices, initial_mask = _padded_batch(train_indices, 0, batch_size)
    initial_inputs, initial_outputs = dataset.get_batch(initial_indices)
    initial_observation = prepare_observation_batch(initial_inputs, norm_stats, tokenizer)
    memory, memory_mask, state = feature_fn(frozen_state, initial_observation)
    del initial_outputs, initial_mask

    if decoder_type == "act":
        decoder = ACTDecoder(config=decoder_cfg)
        if decoder_cfg.use_cvae:
            params = decoder.init(
                jax.random.key(seed), memory, memory_mask, state,
                jnp.zeros((batch_size, decoder_cfg.action_horizon,
                           decoder_cfg.action_dim), jnp.float32),
                jnp.ones((batch_size, decoder_cfg.action_horizon), jnp.float32),
                jnp.zeros((batch_size, decoder_cfg.latent_dim), jnp.float32))["params"]
        else:
            params = decoder.init(
                jax.random.key(seed), memory, memory_mask, state)["params"]
    else:
        decoder = DiffusionDecoder(config=decoder_cfg)
        params = decoder.init(
            jax.random.key(seed), memory, memory_mask, state,
            jnp.zeros((batch_size, decoder_cfg.action_horizon, decoder_cfg.action_dim), jnp.float32),
            jnp.zeros((batch_size,), jnp.int32))["params"]
    print(f"[Decoder] prefix_memory={tuple(memory.shape)} params={parameter_count(params):,}")

    optimizer = optax.chain(
        optax.clip_by_global_norm(float(train_cfg.get("gradient_clip", 1.0))),
        optax.adamw(
            float(train_cfg.get("learning_rate", 1e-4)),
            weight_decay=float(train_cfg.get("weight_decay", 1e-5))))
    opt_state = optimizer.init(params)
    ema_params = params
    alpha_bars = jnp.asarray(cosine_schedule(
        int(cfg.get("diffusion", {}).get("train_timesteps", 100)))[2])
    if decoder_type == "act":
        train_step, eval_step = _make_act_steps(decoder, optimizer, decoder_cfg, train_cfg)
    else:
        train_step, eval_step = _make_dp_steps(
            decoder, optimizer, decoder_cfg, train_cfg, alpha_bars)

    def load_batch(indices, sample_mask):
        inputs, outputs, action_valid_mask = dataset.get_batch(
            indices, return_action_mask=True)
        observation = prepare_observation_batch(inputs, norm_stats, tokenizer)
        features = feature_fn(frozen_state, observation)
        target = jnp.asarray(normalize_actions(outputs, norm_stats, decoder_cfg.action_dim))
        return (*features, target, jnp.asarray(sample_mask),
                jnp.asarray(action_valid_mask))

    if args.smoke_test:
        memory, memory_mask, state, target, sample_mask, action_valid_mask = load_batch(
            initial_indices, np.ones(batch_size, np.float32))
        if decoder_type == "act":
            result = train_step(
                params, ema_params, opt_state, memory, memory_mask, state,
                target, sample_mask, action_valid_mask,
                jax.random.key(seed + 1), jnp.asarray(1.0))
        else:
            result = train_step(
                params, ema_params, opt_state, memory, memory_mask, state,
                target, sample_mask, action_valid_mask, jax.random.key(seed + 1))
        loss = float(result[3])
        if not np.isfinite(loss):
            raise FloatingPointError(f"non-finite smoke loss: {loss}")
        print(f"[OK] real-checkpoint smoke update: type={decoder_type} loss={loss:.6f}")
        return

    output_dir = _resolve(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    start_epoch = 0
    best_validation = float("inf")
    latest = output_dir / "checkpoints" / "latest.pkl"
    if bool(train_cfg.get("resume", True)) and latest.is_file():
        with latest.open("rb") as handle:
            payload = pickle.load(handle)
        params = serialization.from_state_dict(params, payload["params"])
        ema_params = serialization.from_state_dict(ema_params, payload["ema_params"])
        opt_state = serialization.from_state_dict(opt_state, payload["opt_state"])
        start_epoch = int(payload["epoch"])
        best_validation = float(payload.get("best_validation", best_validation))
        print(f"[Decoder] resumed epoch={start_epoch} best_val={best_validation:.6f}")

    epoch_log = output_dir / "epoch_history.csv"
    mode = "a" if start_epoch and epoch_log.is_file() else "w"
    with epoch_log.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if mode == "w":
            writer.writerow([
                "epoch", "train_loss", "val_loss", "reconstruction",
                "smooth", "jerk", "pad_bce", "kl"])
        for epoch in range(start_epoch, epochs):
            order = np.random.default_rng(seed + epoch).permutation(train_indices)
            train_sums = np.zeros(6, np.float64)
            train_count = 0
            batches = (len(order) + batch_size - 1) // batch_size
            kl_warmup_epochs = max(1, int(train_cfg.get("kl_warmup_epochs", 1)))
            kl_factor = min(1.0, float(epoch + 1) / kl_warmup_epochs)
            with tqdm(total=batches, desc=f"{decoder_type.upper()} epoch {epoch + 1}/{epochs}",
                      unit="batch", dynamic_ncols=True) as progress:
                for batch_index, start in enumerate(range(0, len(order), batch_size)):
                    indices, sample_mask = _padded_batch(order, start, batch_size)
                    (memory, memory_mask, state, target, sample_mask,
                     action_valid_mask) = load_batch(indices, sample_mask)
                    if decoder_type == "act":
                        key = jax.random.fold_in(
                            jax.random.key(seed), epoch * batches + batch_index)
                        result = train_step(
                            params, ema_params, opt_state, memory, memory_mask,
                            state, target, sample_mask, action_valid_mask,
                            key, jnp.asarray(kl_factor, jnp.float32))
                    else:
                        key = jax.random.fold_in(
                            jax.random.key(seed), epoch * batches + batch_index)
                        result = train_step(
                            params, ema_params, opt_state, memory, memory_mask,
                            state, target, sample_mask, action_valid_mask, key)
                    params, ema_params, opt_state, loss, metrics = result
                    values = np.asarray([loss, *metrics], np.float64)
                    if not np.all(np.isfinite(values)):
                        raise FloatingPointError(
                            f"non-finite loss epoch={epoch + 1} batch={batch_index}: {values}")
                    train_sums += values
                    train_count += 1
                    progress.update(1)
                    progress.set_postfix(loss=f"{values[0]:.4f}")

            validation_sum = 0.0
            validation_count = 0
            for batch_index, start in enumerate(range(0, len(val_indices), batch_size)):
                indices, sample_mask = _padded_batch(val_indices, start, batch_size)
                (memory, memory_mask, state, target, sample_mask,
                 action_valid_mask) = load_batch(indices, sample_mask)
                if decoder_type == "act":
                    validation, _ = eval_step(
                        ema_params, memory, memory_mask, state, target,
                        sample_mask, action_valid_mask)
                else:
                    key = jax.random.fold_in(jax.random.key(seed + 100_000), batch_index)
                    validation, _ = eval_step(
                        ema_params, memory, memory_mask, state, target,
                        sample_mask, action_valid_mask, key)
                validation_sum += float(validation)
                validation_count += 1

            averaged = train_sums / max(train_count, 1)
            val_loss = validation_sum / max(validation_count, 1)
            writer.writerow([
                epoch + 1, averaged[0], val_loss, averaged[1], averaged[2],
                averaged[3], averaged[4], averaged[5]])
            handle.flush()
            print(f"[Decoder epoch {epoch + 1}/{epochs}] train={averaged[0]:.6f} "
                  f"val={val_loss:.6f}", flush=True)

            payload = {
                "format_version": 1,
                "decoder_type": decoder_type,
                "epoch": epoch + 1,
                "params": serialization.to_state_dict(params),
                "ema_params": serialization.to_state_dict(ema_params),
                "opt_state": serialization.to_state_dict(opt_state),
                "best_validation": min(best_validation, val_loss),
                "config": cfg,
            }
            _atomic_pickle(output_dir / "checkpoints" / "latest.pkl", payload)
            interval = int(train_cfg.get("checkpoint_every_epochs", 5))
            if interval > 0 and (epoch + 1) % interval == 0:
                _atomic_pickle(
                    output_dir / "checkpoints" / f"epoch_{epoch + 1:03d}.pkl", payload)
            if val_loss < best_validation:
                best_validation = val_loss
                payload["best_validation"] = best_validation
                _atomic_pickle(output_dir / "checkpoints" / "best.pkl", payload)
            _plot_epochs(epoch_log, output_dir)

    print(f"[Decoder] complete: {output_dir / 'checkpoints' / 'best.pkl'}")


if __name__ == "__main__":
    main()
