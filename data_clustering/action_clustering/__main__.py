"""Run from data_clustering/: python -m action_clustering --help."""

import argparse
from pathlib import Path
from time import monotonic

import numpy as np

from .config import fresh_dir, load_config, write_json
from .progress import duration, timed


def demo(destination):
    """Unlabelled synthetic episodes for pipeline checks, not skill evaluation."""
    root = fresh_dir(destination).resolve()
    data = root / "data"
    data.mkdir()
    rng = np.random.default_rng(7)
    for i in range(12):
        count = 180
        time = np.arange(count + 1, dtype=np.float32) / 25
        velocity = rng.normal(0, 0.006, (count, 12)).astype(np.float32)
        velocity += 0.02 * np.sin(time[:-1, None] * rng.uniform(0.5, 2, (1, 12)) + i)
        arms = np.concatenate([np.zeros((1, 12), dtype=np.float32), np.cumsum(velocity, axis=0)])
        grip = (0.5 + 0.4 * np.sin(time[:, None] + np.array([[i, i + 1]]))).astype(np.float32)
        qpos = np.concatenate([arms[:, :6], grip[:, :1], arms[:, 6:], grip[:, 1:]], axis=1)
        pose = np.zeros((count + 1, 2, 7), dtype=np.float32)
        pose[:, :, 6] = 1
        pose[:, 0, :3], pose[:, 1, :3] = arms[:, :3], arms[:, 6:9]
        np.savez_compressed(data / f"episode_{i:03d}.npz", actions=qpos[1:], qpos=qpos,
                            gripper=grip, ee_pose=pose)
    cfg = {"output_dir": "run", "datasets": [{"id": "synthetic", "root": "data", "glob": "*.npz",
           "format": "npz", "robot_id": "synthetic_dual_arm", "fps": 25,
           "action_semantics": "absolute_joint_target", "gripper_indices": [6, 13]}],
           "training": {"epochs": 3, "batch_size": 32, "hidden_dim": 64, "latent_dim": 128},
           "clustering": {"k_values": [16, 32, 48], "hdbscan": True, "min_cluster_size": 5}}
    write_json(root / "config.json", cfg)
    return root / "config.json"


def main():
    parser = argparse.ArgumentParser(description="机器人动作窗口自监督表示与隐式聚类（第一部分）")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "train", "embed", "cluster", "run"):
        stage = sub.add_parser(name)
        stage.add_argument("--config", required=True, type=Path)
    example = sub.add_parser("demo", help="生成合成数据并运行 CPU 冒烟验证")
    example.add_argument("--output", type=Path, required=True, help="必须是尚不存在的新目录")
    args = parser.parse_args()
    if args.command == "demo":
        config_path = demo(args.output)
        cfg = load_config(config_path)
        command = "run"
    else:
        cfg = load_config(args.config)
        command = args.command
    from .data import prepare
    from .learning import fit, embed
    from .clustering import cluster

    stages = {"prepare": prepare, "train": fit, "embed": embed, "cluster": cluster}
    selected = list(stages) if command == "run" else [command]
    labels = {"prepare": "准备数据", "train": "训练表示网络", "embed": "提取表示", "cluster": "聚类及导出"}
    started = monotonic()
    timings = {}
    print(f"输出目录: {cfg['output_dir']}\n设备: {cfg['training']['device']} | "
          f"线程: {cfg['training']['threads']} | 训练轮数: {cfg['training']['epochs']}\n"
          "时间估计按本次实测速度动态更新；各阶段 ETA 不包含后续阶段，机器负载和 K-means 收敛速度会影响估计。", flush=True)
    for number, stage in enumerate(selected, 1):
        stage_started = monotonic()
        with timed(f"阶段 {number}/{len(selected)}：{labels[stage]}"):
            stages[stage](cfg)
        timings[stage] = monotonic() - stage_started
        print(f"[流程] 累计耗时 {duration(monotonic() - started)}", flush=True)
    root = Path(cfg["output_dir"])
    write_json(root / f"timing_{command}.json", {"stage_seconds": timings, "total_seconds": monotonic() - started})
    files = [p for p in root.rglob("*") if p.is_file()]
    size = sum(p.stat().st_size for p in files)
    allocated = sum(p.stat().st_blocks * 512 for p in files)
    print(f"[完成] 总耗时 {duration(monotonic() - started)} | 产物文件大小 {size / 2**20:.1f} MiB | "
          f"磁盘占用 {allocated / 2**20:.1f} MiB | {root}", flush=True)


if __name__ == "__main__":
    main()
