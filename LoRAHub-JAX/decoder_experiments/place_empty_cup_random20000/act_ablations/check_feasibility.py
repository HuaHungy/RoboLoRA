#!/usr/bin/env python3
"""Validate the ACT ablation matrix without loading the full Pi0 checkpoint."""

from __future__ import annotations

import json
import pathlib
import py_compile
import shutil
import sys

import jax
import jax.numpy as jnp
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EXPECTED = {
    "E0_baseline", "E1_query_pos", "E2_action_mask", "E3_is_pad_head",
    "E4_temporal_agg", "E5_cvae", "E6_huber_gripper_l1",
}


def resolve(value: str) -> pathlib.Path:
    path = pathlib.Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def model_config(payload):
    from action_decoders.models import DecoderConfig
    cfg = payload["decoder"]
    return DecoderConfig(
        action_dim=int(cfg.get("action_dim", 14)),
        action_horizon=int(cfg.get("action_horizon", 50)),
        state_dim=int(cfg.get("state_dim", 32)),
        model_dim=int(cfg.get("model_dim", 256)),
        heads=int(cfg.get("heads", 4)),
        layers=int(cfg.get("layers", 4)),
        ff_dim=int(cfg.get("ff_dim", 1024)),
        separate_query_pos=bool(cfg.get("separate_query_pos", False)),
        predict_is_pad=bool(cfg.get("predict_is_pad", False)),
        use_cvae=bool(cfg.get("use_cvae", False)),
        latent_dim=int(cfg.get("latent_dim", 32)),
        cvae_encoder_layers=int(cfg.get("cvae_encoder_layers", 2)),
    )


def main() -> int:
    sys.path.insert(0, str(ROOT))
    from action_decoders.models import ACTDecoder, parameter_count
    from script.common.data_loader import Pi0FrameDataset

    configs = sorted((HERE / "configs").glob("*.json"))
    if {path.stem for path in configs} != EXPECTED:
        raise ValueError(f"配置集合错误: {[path.name for path in configs]}")

    required = [
        ROOT / "action_decoders/models.py",
        ROOT / "action_decoders/train.py",
        ROOT / "action_decoders/predict.py",
        ROOT / "script/common/data_loader.py",
        ROOT.parent / "RoboTwin/policy/pi0/pi_model.py",
        pathlib.Path(__file__),
    ]
    for path in required:
        py_compile.compile(str(path), doraise=True)

    baseline = json.loads((HERE / "configs/E0_baseline.json").read_text())
    immutable = ("batch_size", "epochs", "learning_rate", "weight_decay")
    for path in configs:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["experiment_id"] != path.stem:
            raise ValueError(f"{path}: experiment_id与文件名不一致")
        if payload["decoder"]["type"] != "act":
            raise ValueError(f"{path}: only ACT is allowed")
        for key in immutable:
            if payload["training"][key] != baseline["training"][key]:
                raise ValueError(f"{path}: 非单变量训练配置 {key}")
        for dependency in (
            resolve(payload["backbone"]["merged_checkpoint"]) / "params",
            resolve(payload["data"]["norm_stats_path"]),
            resolve(payload["data"]["data_dir"]) / "meta/episodes.jsonl",
        ):
            if not dependency.exists():
                raise FileNotFoundError(f"{path}: missing {dependency}")

        cfg = model_config(payload)
        model = ACTDecoder(config=cfg)
        memory = jnp.zeros((2, 9, 128), jnp.float32)
        memory_mask = jnp.ones((2, 9), bool)
        state = jnp.zeros((2, 32), jnp.float32)
        target = jnp.zeros((2, 50, 14), jnp.float32)
        valid = jnp.ones((2, 50), jnp.float32)
        latent_noise = jnp.zeros((2, cfg.latent_dim), jnp.float32)
        if cfg.use_cvae:
            variables = model.init(
                jax.random.key(0), memory, memory_mask, state,
                target, valid, latent_noise)
            output = model.apply(
                variables, memory, memory_mask, state, target, valid, latent_noise)
        else:
            variables = model.init(jax.random.key(0), memory, memory_mask, state)
            output = model.apply(variables, memory, memory_mask, state)
        if output["actions"].shape != (2, 50, 14):
            raise ValueError(f"{path}: bad action shape {output['actions'].shape}")
        if output["is_pad_logits"].shape != (2, 50):
            raise ValueError(f"{path}: bad pad shape")
        print(f"[OK] {path.stem}: params={parameter_count(variables['params']):,}")

    data_cfg = baseline["data"]
    decoder_cfg = baseline["decoder"]
    dataset = Pi0FrameDataset(
        str(resolve(data_cfg["data_dir"])),
        str(resolve(data_cfg["norm_stats_path"])),
        state_dim=decoder_cfg["state_dim"], action_dim=decoder_cfg["action_dim"],
        action_horizon=decoder_cfg["action_horizon"], episode_cache_size=1)
    last_index = dataset.episode_lengths[0] - 1
    _, _, masks = dataset.get_batch([0, last_index], return_action_mask=True)
    if not np.all(masks[0]) or masks[1].sum() != 1:
        raise ValueError(f"action mask语义错误: sums={masks.sum(axis=1)}")
    if dataset.episode_lengths != [168, 172, 177, 168, 172, 171, 183, 169, 182, 165]:
        raise ValueError(f"unexpected dataset lengths: {dataset.episode_lengths}")

    free_gib = shutil.disk_usage(ROOT).free / 2**30
    if free_gib < 15:
        raise OSError(f"剩余磁盘空间仅 {free_gib:.1f} GiB，需要至少15 GiB")
    print(f"[OK] 7个实验配置、模型分支、逐步mask、依赖和磁盘通过（free={free_gib:.1f} GiB）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
