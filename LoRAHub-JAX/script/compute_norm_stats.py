#!/usr/bin/env python
"""按 openpi 方式从任务数据重算归一化统计并保存到 outputs/<task>/norm_stats.json。

用法:
    python script/compute_norm_stats.py --config config/xxx.json [--force]

说明:
    1. 数据来源: 配置文件 data_dir 的全部帧；actions 先按 OpenPI 转 delta chunk
    2. 计算方式: 与 openpi RunningStats 一致 (mean, ddof=0 的 std, q01/q99)
    3. 输出: outputs/<task_name>/norm_stats.json, 并自动更新配置中的 norm_stats_path
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from common import data_loader


def main():
    ap = argparse.ArgumentParser(description="重算任务数据的归一化统计 (openpi 方式)")
    ap.add_argument("--config", "-c", required=True,
                    help="任务配置文件 (config/xxx.json)")
    ap.add_argument("--force", action="store_true",
                    help="即使 norm_stats.json 已存在也强制重算")
    args = ap.parse_args()

    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(ROOT, args.config)
    with open(cfg_path) as f:
        cfg = json.load(f)
    task = cfg["task_name"]
    data_dir = cfg["data_dir"] if os.path.isabs(cfg["data_dir"]) else os.path.join(ROOT, cfg["data_dir"])

    out_path = os.path.join(ROOT, "outputs", task, "norm_stats.json")
    if os.path.exists(out_path) and not args.force:
        print(f"norm_stats 已存在: {out_path} (加 --force 强制重算)")
        return

    print(f"任务: {task}")
    print(f"数据: {data_dir}")
    reference = cfg.get("reference_norm_stats_path")
    if reference and not os.path.isabs(reference):
        reference = os.path.abspath(os.path.join(ROOT, reference))
    if cfg.get("use_reference_norm_stats", False):
        if not reference or not os.path.exists(reference):
            raise FileNotFoundError(
                "use_reference_norm_stats=true，但参考统计不存在: "
                f"{reference}")
        print(f"使用官方 checkpoint 统计: {reference}")
        stats = data_loader.load_norm_stats(reference)
    else:
        action_horizon = cfg.get("algorithm", {}).get("action_horizon", 50)
        stats = data_loader.compute_norm_stats(
            data_dir, action_horizon=action_horizon)
    data_loader.save_norm_stats(out_path, stats)

    # 同步配置中的 norm_stats_path (相对项目根)
    rel = os.path.relpath(out_path, ROOT)
    if cfg.get("norm_stats_path") != rel:
        cfg["norm_stats_path"] = rel
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"  已更新 {cfg_path}: norm_stats_path → {rel}")


if __name__ == "__main__":
    main()
