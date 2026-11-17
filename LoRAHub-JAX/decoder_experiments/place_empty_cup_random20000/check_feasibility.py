#!/usr/bin/env python3
"""CPU-only schema, path and decoder-shape checks for the direct suite."""

from __future__ import annotations

import json
import pathlib
import py_compile
import shutil
import sys

import jax
import jax.numpy as jnp

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def resolve(value: str) -> pathlib.Path:
    path = pathlib.Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    required = [
        ROOT / "action_decoders/models.py",
        ROOT / "action_decoders/backbone.py",
        ROOT / "action_decoders/train.py",
        ROOT / "action_decoders/predict.py",
        ROOT.parent / "RoboTwin/policy/pi0/eval.sh",
    ]
    configs = sorted((HERE / "configs").glob("*.json"))
    if {path.stem for path in configs} != {"ACT_direct", "DP_direct"}:
        raise ValueError(f"配置不完整: {[path.name for path in configs]}")
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    for path in configs:
        cfg = json.loads(path.read_text(encoding="utf-8"))
        if cfg["decoder"]["type"] not in {"act", "dp"}:
            raise ValueError(f"{path}: invalid decoder type")
        for dependency in (
            resolve(cfg["backbone"]["merged_checkpoint"]) / "params",
            resolve(cfg["data"]["norm_stats_path"]),
            resolve(cfg["data"]["data_dir"]) / "meta/episodes.jsonl",
        ):
            if not dependency.exists():
                raise FileNotFoundError(f"{path}: missing {dependency}")
    free_gib = shutil.disk_usage(ROOT).free / 2**30
    if free_gib < 5:
        raise OSError(f"剩余磁盘空间仅 {free_gib:.1f} GiB，需要至少 5 GiB")

    for path in required[:4] + [pathlib.Path(__file__)]:
        py_compile.compile(str(path), doraise=True)
    sys.path.insert(0, str(ROOT))
    from action_decoders.models import ACTDecoder, DecoderConfig, DiffusionDecoder, parameter_count
    cfg = DecoderConfig(model_dim=64, heads=4, layers=2, ff_dim=128)
    memory = jnp.zeros((2, 9, 128), jnp.float32)
    mask = jnp.ones((2, 9), bool)
    state = jnp.zeros((2, 32), jnp.float32)
    act = ACTDecoder(config=cfg)
    act_params = act.init(jax.random.key(0), memory, mask, state)["params"]
    act_output = act.apply({"params": act_params}, memory, mask, state)["actions"]
    dp = DiffusionDecoder(config=cfg)
    noisy = jnp.zeros((2, 50, 14), jnp.float32)
    timestep = jnp.zeros((2,), jnp.int32)
    dp_params = dp.init(jax.random.key(1), memory, mask, state, noisy, timestep)["params"]
    dp_output = dp.apply({"params": dp_params}, memory, mask, state, noisy, timestep)
    assert act_output.shape == dp_output.shape == (2, 50, 14)
    print(f"[OK] ACT shape={act_output.shape} params={parameter_count(act_params):,}")
    print(f"[OK] DP  shape={dp_output.shape} params={parameter_count(dp_params):,}")
    print(f"[OK] 配置、依赖、编译、模型形状、磁盘空间通过（free={free_gib:.1f} GiB）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
