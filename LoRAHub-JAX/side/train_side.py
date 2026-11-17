#!/usr/bin/env python
"""Train a tiny KV-conditioned residual on a frozen composed Pi0.

Example:
    python side/train_side.py --config config/place_empty_cup_10_v3_pi0_side.json

The data iterator is deterministic and sequential: one epoch visits every
frame of every episode exactly once.  The Pi0 backbone is never optimized.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import pathlib
import pickle
import sys
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import optax
from flax import serialization
from tqdm import tqdm

from script.common import data_loader
from side.kv_side_network import (
    KVSideQFormerConfig,
    KVSideQFormerResidual,
    KVSideResidual,
    parameter_count,
)
from side.pi0_runtime import (
    build_normalized_action_batch,
    load_frozen_pi0,
    make_flow_feature_fn,
    prepare_observation_batch,
    qformer_shape_config,
    side_shape_config,
)


def _resolve(root: pathlib.Path, value: str) -> pathlib.Path:
    path = pathlib.Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _atomic_pickle(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            pickle.dump(jax.device_get(payload), handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _load_checkpoint(path: pathlib.Path, params_template, opt_template):
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if payload.get("format_version") != 1:
        raise ValueError(f"unsupported side checkpoint format: {path}")
    params = serialization.from_state_dict(params_template, payload["params"])
    opt_state = serialization.from_state_dict(opt_template, payload["opt_state"])
    return (
        params,
        opt_state,
        int(payload["step"]),
        int(payload.get("next_epoch", payload.get("epoch", 0))),
        int(payload.get("next_batch", 0)),
    )


def _save_checkpoint(
    out_dir, step, next_epoch, next_batch, params, opt_state, cfg, *, numbered=True
):
    ckpt_dir = out_dir / "checkpoints"
    payload = {
        "format_version": 1,
        "step": int(step),
        "next_epoch": int(next_epoch),
        "next_batch": int(next_batch),
        "params": serialization.to_state_dict(params),
        "opt_state": serialization.to_state_dict(opt_state),
        "config": cfg,
    }
    if numbered:
        _atomic_pickle(ckpt_dir / f"step_{step:08d}.pkl", payload)
    _atomic_pickle(ckpt_dir / "latest.pkl", payload)


def _find_latest(out_dir: pathlib.Path) -> pathlib.Path | None:
    """Return the highest valid checkpoint, not just the latest pointer.

    A full disk can occur after the numbered checkpoint is written but before
    ``latest.pkl`` is atomically replaced.  In that case the numbered file is
    still a valid and newer resume point.
    """
    ckpt_dir = out_dir / "checkpoints"
    candidates = []
    latest = ckpt_dir / "latest.pkl"
    if latest.is_file():
        candidates.append(latest)
    candidates.extend(sorted(ckpt_dir.glob("step_*.pkl"), reverse=True))

    valid = []
    for path in candidates:
        try:
            with path.open("rb") as handle:
                payload = pickle.load(handle)
            step = int(payload["step"])
            if payload.get("format_version") != 1:
                continue
            valid.append((step, path))
        except (OSError, KeyError, TypeError, ValueError, EOFError, pickle.PickleError):
            continue
    return max(valid, key=lambda item: item[0])[1] if valid else None


def _plot_loss_curve(log_path: pathlib.Path, out_dir: pathlib.Path) -> None:
    """Plot the CSV losses without making plotting a training dependency."""
    rows = []
    if not log_path.is_file():
        return
    with log_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                rows.append({
                    "step": int(row["step"]),
                    "loss": float(row["loss"]),
                    "flow_loss": float(row["flow_loss"]),
                    "residual_loss": float(row["residual_loss"]),
                })
            except (KeyError, TypeError, ValueError):
                continue
    if not rows:
        return

    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [row["step"] for row in rows]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, [row["loss"] for row in rows], label="total loss", linewidth=1.2)
    ax.plot(steps, [row["flow_loss"] for row in rows], label="flow loss", linewidth=1.0)
    ax.plot(steps, [row["residual_loss"] for row in rows], label="residual penalty", linewidth=1.0)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("loss")
    ax.set_title("KV Side Residual Training Loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(logs_dir / "loss_curve.png", dpi=140)
    plt.close(fig)


def _batch_indices(start: int, total: int, batch_size: int):
    end = min(start + batch_size, total)
    indices = np.arange(start, end, dtype=np.int64)
    mask = np.ones(len(indices), dtype=np.float32)
    if len(indices) < batch_size:
        if not len(indices):
            raise ValueError("empty batch")
        indices = np.pad(indices, (0, batch_size - len(indices)), mode="edge")
        mask = np.pad(mask, (0, batch_size - len(mask)), constant_values=0.0)
    return indices, mask


def _make_train_step(side_model, optimizer, residual_scale, action_dim):
    @jax.jit
    def train_step(params, opt_state, v_base, kv_context, action_hidden,
                   time, u_target, sample_mask):
        def loss_fn(current_params):
            residual = side_model.apply(
                {"params": current_params}, kv_context, action_hidden, time)
            residual_full = jnp.pad(
                residual, ((0, 0), (0, 0), (0, u_target.shape[-1] - action_dim)))
            prediction = v_base + jnp.asarray(residual_scale, jnp.float32) * residual_full
            error = prediction[..., :action_dim] - u_target[..., :action_dim]
            mask = sample_mask[:, None, None]
            denom = jnp.maximum(jnp.sum(mask) * error.shape[1] * error.shape[2], 1.0)
            flow_loss = jnp.sum(jnp.square(error) * mask) / denom
            residual_loss = jnp.sum(jnp.square(residual) * mask) / denom
            return flow_loss + 1e-4 * residual_loss, (flow_loss, residual_loss)

        (loss, (flow_loss, residual_loss)), grads = jax.value_and_grad(
            loss_fn, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, flow_loss, residual_loss

    return train_step


def _make_eval_step(side_model, residual_scale, action_dim):
    @jax.jit
    def eval_step(params, v_base, kv_context, action_hidden, time,
                  u_target, sample_mask):
        residual = side_model.apply(
            {"params": params}, kv_context, action_hidden, time)
        residual_full = jnp.pad(
            residual, ((0, 0), (0, 0), (0, u_target.shape[-1] - action_dim)))
        prediction = v_base + jnp.asarray(residual_scale, jnp.float32) * residual_full
        error = prediction[..., :action_dim] - u_target[..., :action_dim]
        mask = sample_mask[:, None, None]
        denom = jnp.maximum(jnp.sum(mask) * error.shape[1] * error.shape[2], 1.0)
        return jnp.sum(jnp.square(error) * mask) / denom

    return eval_step


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument(
        "--init-only", action="store_true",
        help="加载数据和冻结 Pi0、初始化 residual 参数后退出，不训练",
    )
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as handle:
        cfg = json.load(handle)

    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    out_dir = _resolve(ROOT, cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(cfg, handle, indent=2, ensure_ascii=False)

    seed = int(train_cfg.get("seed", 42))
    batch_size = int(train_cfg.get("batch_size", 32))
    epochs = int(train_cfg.get("epochs", 20))
    action_dim = int(model_cfg.get("action_dim", 14))
    model_action_dim = int(model_cfg.get("model_action_dim", 32))
    if batch_size <= 0 or epochs <= 0:
        raise ValueError("batch_size and epochs must be positive")

    norm_stats_path = _resolve(ROOT, data_cfg["norm_stats_path"])
    norm_stats = data_loader.load_norm_stats(str(norm_stats_path))
    dataset = data_loader.Pi0FrameDataset(
        str(_resolve(ROOT, data_cfg["data_dir"])),
        str(norm_stats_path),
        action_dim=action_dim,
        action_horizon=int(model_cfg.get("action_horizon", 50)),
        episode_cache_size=int(data_cfg.get("episode_cache_size", 10)),
    )
    if len(dataset) == 0:
        raise ValueError("side training dataset is empty")
    batches_per_epoch = (len(dataset) + batch_size - 1) // batch_size
    total_steps = epochs * batches_per_epoch

    print(f"[Side] dataset frames={len(dataset)} episodes={len(dataset.episode_paths)} "
          f"batch={batch_size} epochs={epochs} batches/epoch={batches_per_epoch}")
    print("[Side] order=sequential, every frame is visited once per epoch")

    graphdef, frozen_state, frozen_model = load_frozen_pi0(
        _resolve(ROOT, model_cfg["merged_checkpoint"]),
        model_cfg["train_config_name"],
    )
    layer_indices = tuple(int(x) for x in model_cfg.get("kv_layer_indices", [5, 11, 17]))
    architecture = str(model_cfg.get("side_architecture", "pool")).lower()
    if architecture in {"qformer", "q-former"}:
        side_cfg = qformer_shape_config(
            frozen_model,
            output_dim=action_dim,
            num_queries=int(model_cfg.get("qformer_num_queries", 4)),
            qformer_dim=int(model_cfg.get("qformer_dim", 128)),
            qformer_heads=int(model_cfg.get("qformer_heads", 4)),
            qformer_layers=int(model_cfg.get("qformer_layers", 2)),
            qformer_ff_dim=int(model_cfg.get("qformer_ff_dim", 256)),
        )
        side_model = KVSideQFormerResidual(config=side_cfg)
        kv_mode = "tokens"
    elif architecture in {"pool", "pooled", "mean_pool"}:
        side_cfg = side_shape_config(frozen_model, output_dim=action_dim)
        side_model = KVSideResidual(config=side_cfg)
        kv_mode = "pooled"
    else:
        raise ValueError(
            f"unsupported model.side_architecture={architecture!r}; "
            "expected 'pool' or 'qformer'"
        )
    feature_fn = make_flow_feature_fn(
        graphdef, layer_indices=layer_indices, kv_mode=kv_mode)
    residual_scale = float(model_cfg.get("residual_scale", 0.1))
    side_cfg = dataclasses.replace(side_cfg, residual_scale=residual_scale)
    tokenizer_cls = __import__(
        "openpi.models.tokenizer", fromlist=["PaligemmaTokenizer"]
    ).PaligemmaTokenizer
    tokenizer = tokenizer_cls(max_len=frozen_model.max_token_len)
    if architecture in {"qformer", "q-former"}:
        dummy_context = jnp.zeros(
            (batch_size, 1, side_cfg.kv_token_dim), jnp.float32)
    else:
        dummy_context = jnp.zeros((batch_size, side_cfg.context_dim), jnp.float32)
    dummy_hidden = jnp.zeros((batch_size, frozen_model.action_horizon,
                              side_cfg.action_hidden_dim), jnp.float32)
    dummy_time = jnp.zeros((batch_size,), jnp.float32)
    params = side_model.init(
        jax.random.key(seed), dummy_context, dummy_hidden, dummy_time)["params"]
    print(f"[Side] architecture={architecture} kv_mode={kv_mode}")
    count = parameter_count(params)
    print(f"[Side] config={side_cfg} parameters={count}")

    if args.init_only:
        print("[Side] init-only 检查通过；未启动优化器训练。")
        return

    learning_rate = float(train_cfg.get("learning_rate", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 1e-5))
    optimizer = optax.adamw(learning_rate, weight_decay=weight_decay)
    opt_state = optimizer.init(params)
    train_step = _make_train_step(
        side_model, optimizer, residual_scale, action_dim)
    eval_step = _make_eval_step(
        side_model, residual_scale, action_dim)

    start_step = 0
    start_epoch = 0
    start_batch = 0
    if bool(train_cfg.get("resume", True)):
        latest = _find_latest(out_dir)
        if latest is not None:
            params, opt_state, start_step, start_epoch, start_batch = _load_checkpoint(
                latest, params, opt_state)
            print(f"[Side] resumed from {latest} step={start_step} "
                  f"epoch={start_epoch} batch={start_batch}")

    log_path = out_dir / "training.csv"
    # A fresh run gets a clean curve; only a real checkpoint resume appends.
    log_mode = "a" if start_step > 0 else "w"
    with log_path.open(log_mode, newline="", encoding="utf-8") as log_handle:
        writer = csv.writer(log_handle)
        if start_step == 0:
            writer.writerow(["step", "epoch", "batch", "loss", "flow_loss", "residual_loss"])

        global_step = start_step
        interval = int(train_cfg.get("checkpoint_interval", 100))
        with tqdm(
            total=total_steps,
            initial=min(start_step, total_steps),
            desc="Side 网络训练",
            unit="step",
            dynamic_ncols=True,
        ) as progress:
            for epoch in range(start_epoch, epochs):
                for batch_index in range(batches_per_epoch):
                    if global_step >= total_steps:
                        break
                    if epoch == start_epoch and batch_index < start_batch:
                        continue
                    frame_start = batch_index * batch_size
                    indices, sample_mask = _batch_indices(frame_start, len(dataset), batch_size)
                    inputs, outputs = dataset.get_batch(indices)
                    observation = prepare_observation_batch(inputs, norm_stats, tokenizer)
                    actions = build_normalized_action_batch(
                        outputs, norm_stats, action_dim=action_dim,
                        model_action_dim=model_action_dim)
                    key = jax.random.fold_in(jax.random.key(seed), global_step)
                    noise_key, time_key = jax.random.split(key)
                    noise = jax.random.normal(noise_key, actions.shape, dtype=jnp.float32)
                    time = jax.random.uniform(
                        time_key, (batch_size,), minval=0.001, maxval=0.999,
                    )
                    features = feature_fn(
                        frozen_state, observation, jnp.asarray(actions), noise, time)
                    v_base, context, hidden, target = features
                    step_result = train_step(
                        params, opt_state, v_base, context, hidden, time, target,
                        jnp.asarray(sample_mask))
                    params, opt_state, loss, flow_loss, residual_loss = step_result
                    global_step += 1
                    values = [float(x) for x in (loss, flow_loss, residual_loss)]
                    writer.writerow([global_step, epoch + 1, batch_index, *values])
                    log_handle.flush()
                    progress.update(1)
                    progress.set_postfix(
                        epoch=f"{epoch + 1}/{epochs}",
                        loss=f"{values[0]:.4f}",
                    )

                    if global_step == 1 or global_step % 10 == 0:
                        print(f"[Side step {global_step}/{total_steps}] "
                              f"loss={values[0]:.6f} flow={values[1]:.6f} "
                              f"residual={values[2]:.6f}", flush=True)

                    if interval > 0 and global_step % interval == 0:
                        next_epoch = epoch
                        next_batch = batch_index + 1
                        if next_batch >= batches_per_epoch:
                            next_epoch += 1
                            next_batch = 0
                        _save_checkpoint(
                            out_dir, global_step, next_epoch, next_batch,
                            params, opt_state, cfg)
                        _plot_loss_curve(log_path, out_dir)

                if global_step > start_step:
                    eval_loss = 0.0
                    count = 0
                    for batch_index in range(batches_per_epoch):
                        frame_start = batch_index * batch_size
                        indices, sample_mask = _batch_indices(frame_start, len(dataset), batch_size)
                        inputs, outputs = dataset.get_batch(indices)
                        observation = prepare_observation_batch(inputs, norm_stats, tokenizer)
                        actions = build_normalized_action_batch(
                            outputs, norm_stats, action_dim=action_dim,
                            model_action_dim=model_action_dim)
                        key = jax.random.fold_in(jax.random.key(seed + 1), epoch * batches_per_epoch + batch_index)
                        noise_key, time_key = jax.random.split(key)
                        noise = jax.random.normal(noise_key, actions.shape, dtype=jnp.float32)
                        time = jax.random.uniform(time_key, (batch_size,), minval=0.001, maxval=0.999)
                        features = feature_fn(
                            frozen_state, observation, jnp.asarray(actions), noise, time)
                        v_base, context, hidden, target = features
                        eval_loss += float(eval_step(
                            params, v_base, context, hidden, time, target,
                            jnp.asarray(sample_mask)))
                        count += 1
                    print(f"[Side epoch {epoch + 1}/{epochs}] "
                          f"eval_flow={eval_loss / max(count, 1):.6f}")
                    _save_checkpoint(
                        out_dir, global_step, epoch + 1, 0,
                        params, opt_state, cfg, numbered=False)
                    _plot_loss_curve(log_path, out_dir)

        _plot_loss_curve(log_path, out_dir)

    print(f"[Side] complete: {out_dir / 'checkpoints' / 'latest.pkl'}")
if __name__ == "__main__":
    main()
