#!/usr/bin/env python
"""LoRAHub 统一优化入口: 读任务配置 -> 加载数据/LoRA -> 黑盒搜索 -> 输出权重。

用法:
    python script/run.py --config config/place_empty_cup_10.json
或通过 .sh 入口:
    bash run_lorahub.sh place_empty_cup_10

输出: outputs/<task_name>/lorahub_weights.json
"""

import os
import sys
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

# 将 RoboTwin 的 policy/pi0/src 加入 Python path (Pi0 模型代码)
_ROBOTWIN_SRC = os.path.join(ROOT, "..", "RoboTwin", "policy", "pi0", "src")
if os.path.isdir(_ROBOTWIN_SRC):
    sys.path.insert(0, _ROBOTWIN_SRC)

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

from vla_lorahub import algorithm as algo
from vla_lorahub.constant import BASE_CHECKPOINT_DIR, get_all_modules
from script.common import data_loader
from script.common.pi0_forward import pi0_forward, get_pi0_runtime_stats


def load_config(path: str) -> dict:
    """加载任务配置, 相对路径基于项目根目录 (ROOT) 解析。"""
    with open(path) as f:
        cfg = json.load(f)
    for key in ("data_dir", "norm_stats_path"):
        if cfg.get(key) and not os.path.isabs(cfg[key]):
            cfg[key] = os.path.join(ROOT, cfg[key])
    return cfg


def get_reference_norm_stats_path(cfg: dict) -> str | None:
    """返回 compose.sync_dir 对应的目标 checkpoint norm_stats 路径。"""
    explicit = cfg.get("reference_norm_stats_path")
    if explicit:
        return explicit if os.path.isabs(explicit) else os.path.abspath(
            os.path.join(ROOT, explicit))
    compose = cfg.get("compose", {})
    sync_dir = compose.get("sync_dir")
    asset_id = compose.get("asset_id")
    if not sync_dir or not asset_id:
        return None
    if not os.path.isabs(sync_dir):
        sync_dir = os.path.abspath(os.path.join(ROOT, sync_dir))
    return os.path.join(sync_dir, "10000", "assets", asset_id, "norm_stats.json")


def select_lora_modules(cfg: dict) -> list:
    """选择参与组合的 LoRA 模块。

    优先使用 config.lora_modules 显式指定; 为空则自动发现 lora_modules/
    并排除目标任务 (target_task) 自己的 LoRA。
    """
    target = cfg["target_task"]
    explicit = cfg.get("lora_modules") or []
    if explicit:
        return [os.path.join(ROOT, m) if not os.path.isabs(m) else m for m in explicit]
    all_mods = get_all_modules()
    selected = [m for m in all_mods if target not in m]
    excluded = [m for m in all_mods if target in m]
    print(f"  排除模块: {[os.path.basename(x) for x in excluded]}")
    print(f"  使用模块: {[os.path.basename(x) for x in selected]}")
    return selected


def resolve_epochs(label: str, epoch_value, steps_per_epoch):
    """把 epoch 数换算成候选评估步数。

    1 epoch 的长度 = 帧池批数 x 每批复用候选数, 只在顺序采样模式下有定义。
    """
    if steps_per_epoch is None:
        raise ValueError(
            f"{label} 使用 epoch 单位时要求 pi0_random + sequential 采样模式")
    return int(round(float(epoch_value) * steps_per_epoch))


def resolve_interval(label: str, step_value, epoch_value, steps_per_epoch):
    """间隔二选一: epoch 字段优先于候选数字段, 两者不一致时告警。"""
    if epoch_value is None:
        return int(step_value or 0)
    resolved = resolve_epochs(label, epoch_value, steps_per_epoch)
    if step_value is not None and int(step_value) != resolved:
        print(f"  [WARN] {label}: 同时设置 {step_value} 候选与 {epoch_value} "
              f"epoch; epochs 优先 -> {resolved}")
    return resolved


def plot_loss_curves(version: str, out_dir: str, history: list) -> None:
    """绘制优化过程 loss/reg/total 曲线, 与日志一同保存到 outputs/<task>/logs/."""
    if not history:
        print("  无优化历史, 跳过曲线绘制")
        return
    import csv
    import matplotlib
    matplotlib.use("Agg")  # 无头环境, 不弹窗
    import matplotlib.pyplot as plt

    steps = [h["step"] for h in history]
    loss = [h["loss"] for h in history]
    reg = [h["reg"] for h in history]
    total = [h["total"] for h in history]

    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Persist numeric history first so a server-side font problem cannot discard
    # the useful optimization record after a long run.
    csv_path = os.path.join(log_dir, "loss_history.csv")
    with open(csv_path, "w", newline="") as f:
        # SCORE_HISTORY can gain diagnostic fields as optimizers evolve (for
        # example ``kind`` for search/validation records).  Keep the CSV
        # export forward-compatible instead of failing after a long search.
        preferred = [
            "step", "kind", "loss", "joint_loss", "gripper_loss",
            "batch_generation", "batch_samples", "reg", "total",
        ]
        present = {key for row in history for key in row}
        fieldnames = [key for key in preferred if key in present]
        fieldnames.extend(
            key for key in sorted(present) if key not in fieldnames
        )
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

    png_path = os.path.join(log_dir, "loss_curve.png")
    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(steps, loss, label="loss", linewidth=1)
        ax.plot(steps, total, label="total (loss+reg)", linewidth=1, alpha=0.7)
        if any(reg):
            ax.plot(steps, reg, label="reg", linewidth=1, alpha=0.4)
        ax.set_xlabel("step")
        ax.set_ylabel("objective loss")
        ax.set_title(f"LoRAHub [{version}] optimization curve ({len(steps)} steps)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(png_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  曲线: {png_path}")
    except Exception as exc:
        print(f"  [WARN] 曲线绘制失败，CSV 和权重不受影响: {exc}")
    print(f"  明细: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="LoRAHub 黑盒优化入口")
    parser.add_argument("--config", "-c", required=True, help="任务配置文件 (config/xxx.json)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    task = cfg["task_name"]
    out_dir = os.path.join(ROOT, "outputs", task)
    os.makedirs(out_dir, exist_ok=True)

    a_cfg = cfg.get("algorithm", {})
    version = a_cfg.get("version", "v3")           # v1 / v2 / v2.5 / v3
    budget_steps = a_cfg.get("max_steps")          # 候选评估步数 (显式)
    budget_epochs = a_cfg.get("epochs")            # 数据集 epoch 数 (自动换算)
    seed = a_cfg.get("seed", 42)
    weight_bounds = tuple(a_cfg.get("weight_bounds", [1e-6, 1.5]))
    frames_per_ep = a_cfg.get("frames_per_ep", 1)       # 每条 episode 采样起始帧数
    keyframe_strategy = a_cfg.get("keyframe_strategy", "motion_stratified")
    sampling_mode = a_cfg.get("sampling_mode", "keyframes")
    batch_sampling_order = a_cfg.get("batch_sampling_order", "random")
    action_horizon = a_cfg.get("action_horizon", 50)    # 与 Pi0 action_horizon 对齐
    max_batch = a_cfg.get("max_batch", 2)               # 50 步动作窗口下控制显存峰值
    softmax_temperature = a_cfg.get("softmax_temperature", 1.0)
    reg_weight = a_cfg.get("reg_weight", 0.0)           # 稀疏正则强度 (0=关闭)
    regularization_type = a_cfg.get("regularization_type", "uniform_kl")
    logit_bound = a_cfg.get("logit_bound")
    cma_scale = a_cfg.get("cma_scale", 1.0)
    cma_popsize = a_cfg.get("cma_popsize")
    reuse_candidates = a_cfg.get("batch_reuse_candidates", cma_popsize or 1)
    composition_mode = a_cfg.get("composition_mode", "factor")
    v2_mode = a_cfg.get("v2_mode", "full")
    loss_cfg = a_cfg.get("loss", {})
    diffusion_steps = cfg.get("diffusion_steps", 10)  # 搜索阶段扩散步数 (可调 4 提速)
    search_diffusion_seeds = [
        int(x) for x in a_cfg.get("search_diffusion_seeds", [42])]
    validation_diffusion_steps = int(
        a_cfg.get("validation_diffusion_steps", 10))
    validation_diffusion_seeds = [
        int(x) for x in a_cfg.get(
            "validation_diffusion_seeds", [42, 43, 44, 45])]
    # 间隔: 候选数单位 (xxx_interval / xxx_interval_candidates) 与 epoch 单位
    # (xxx_interval_epochs / xxx_epochs) 二选一; 两者都写且不一致时 epochs 优先。
    intervals = {
        "validation": {
            "steps": a_cfg.get("validation_interval_candidates"),
            "epochs": a_cfg.get("validation_interval_epochs")},
        "weight_update": {
            "steps": a_cfg.get("weight_update_interval"),
            "epochs": a_cfg.get("weight_update_epochs")},
        "checkpoint": {
            "steps": a_cfg.get("checkpoint_interval"),
            "epochs": a_cfg.get("checkpoint_epochs")},
    }
    resume = bool(a_cfg.get("resume", False))
    if not search_diffusion_seeds or not validation_diffusion_seeds:
        raise ValueError("search/validation_diffusion_seeds 不得为空")
    if budget_epochs is not None:
        budget_text = f"{budget_epochs} epoch"
    elif budget_steps is not None:
        budget_text = f"{budget_steps} steps"
    else:
        raise ValueError("algorithm 必须设置 max_steps 或 epochs 之一")

    print("=" * 60)
    print(f"VLA LoRAHub [{version}] 任务={task} 优化开始")
    if sampling_mode == "pi0_random":
        order_text = ("顺序 epoch" if batch_sampling_order == "sequential"
                      else "随机")
        sample_text = (
            f"PI0 全帧 {order_text} mini-batch="
            f"{a_cfg.get('candidate_batch_size', max_batch)}"
            f"x{a_cfg.get('batches_per_evaluation', 1)}")
    else:
        sample_text = (
            f"{frames_per_ep} 帧/episode, keyframes={keyframe_strategy}")
    print(f"  数据模式: {sampling_mode} | 样本: {sample_text} × "
          f"GT {action_horizon} 步轨迹 | max_batch={max_batch} | "
          f"softmax τ={softmax_temperature} | "
          f"reg={regularization_type}:{reg_weight} | "
          f"compose={composition_mode} | budget={budget_text}")
    print(f"  diffusion: search={diffusion_steps} steps×"
          f"{search_diffusion_seeds} | validation={validation_diffusion_steps} "
          f"steps×{validation_diffusion_seeds} | resume={resume}")
    print("  间隔配置: " + " | ".join(
        f"{name}={item['epochs']} epoch" if item["epochs"] is not None
        else f"{name}={int(item['steps'] or 0)} 候选"
        for name, item in intervals.items()))
    print("=" * 60)

    print("\n[1] 准备归一化统计 + 加载 few-shot 数据...")
    norm_stats_path = cfg["norm_stats_path"]
    reference_norm_stats_path = get_reference_norm_stats_path(cfg)
    use_reference_norm_stats = bool(cfg.get("use_reference_norm_stats", False))
    if not os.path.exists(norm_stats_path):
        print(f"  norm_stats 不存在: {norm_stats_path}")
        norm_stats_path = os.path.join(out_dir, "norm_stats.json")
        if use_reference_norm_stats:
            if not reference_norm_stats_path or not os.path.exists(
                    reference_norm_stats_path):
                raise FileNotFoundError(
                    "use_reference_norm_stats=true，但官方 checkpoint 的 "
                    f"norm_stats 不存在: {reference_norm_stats_path}")
            print("  复制官方 checkpoint 的 delta-action norm_stats...")
            data_loader.save_norm_stats(
                norm_stats_path,
                data_loader.load_norm_stats(reference_norm_stats_path),
            )
        else:
            print("  按 openpi DeltaActions→Normalize 顺序从数据重算并保存...")
            recomputed = data_loader.compute_norm_stats(
                cfg["data_dir"], action_horizon=action_horizon)
            data_loader.save_norm_stats(norm_stats_path, recomputed)
    if reference_norm_stats_path and os.path.exists(reference_norm_stats_path):
        data_loader.assert_norm_stats_match(norm_stats_path, reference_norm_stats_path)
    else:
        print("  [WARN] 未找到目标 checkpoint norm_stats，跳过一致性校验: "
              f"{reference_norm_stats_path}")
    batch_sampler = None
    if sampling_mode == "pi0_random":
        frame_dataset = data_loader.Pi0FrameDataset(
            cfg["data_dir"], norm_stats_path,
            action_horizon=action_horizon,
            episode_cache_size=a_cfg.get("episode_cache_size", 10),
        )
        candidate_batch_size = a_cfg.get("candidate_batch_size", max_batch)
        batches_per_evaluation = a_cfg.get("batches_per_evaluation", 1)
        frame_pool = None
        if (batch_sampling_order == "sequential"
                and a_cfg.get("frame_pool_enabled", False)):
            frame_pool = frame_dataset.build_frame_pool(
                gripper_delta_threshold=a_cfg.get(
                    "gripper_delta_threshold", 0.005),
                static_stride=a_cfg.get("static_stride", 2),
            )
        batch_sampler = data_loader.Pi0RandomBatchSampler(
            frame_dataset,
            batch_size=candidate_batch_size,
            batches_per_evaluation=batches_per_evaluation,
            reuse_candidates=reuse_candidates,
            final_evaluation_batches=a_cfg.get("final_evaluation_batches", 2),
            seed=seed,
            order=batch_sampling_order,
            frame_pool=frame_pool,
        )
        few_inputs, few_outputs, _ = batch_sampler.get_batch(0)
        objective_outputs = frame_dataset.all_delta_outputs()
        sample_manifest = frame_dataset.manifest()
        if frame_pool is not None:
            total = len(frame_dataset)
            kept = len(frame_pool)
            print(f"  Pi0 gripper 自适应帧池: {kept}/{total} "
                  f"(节省 {100.0 * (1 - kept / total):.1f}%; "
                  f"阈值={a_cfg.get('gripper_delta_threshold', 0.005)}, "
                  f"静止隔帧={a_cfg.get('static_stride', 2)})")
        if batch_sampling_order == "sequential":
            pool_text = (f"帧池 {batch_sampler.pool_size}/{len(frame_dataset)}"
                         if frame_pool is not None
                         else f"全帧池 {len(frame_dataset)}")
            candidates_per_epoch = (
                batch_sampler.batches_per_epoch * reuse_candidates)
            print(f"  Pi0 {pool_text} 条；顺序遍历，每候选 "
                  f"{candidate_batch_size}x{batches_per_evaluation} 条连续帧，"
                  f"每 {reuse_candidates} 个候选前进一批，"
                  f"1 epoch={batch_sampler.batches_per_epoch} 批"
                  f"({candidates_per_epoch} 候选)")
        else:
            print(f"  Pi0 全帧池 {len(frame_dataset)} 条；每候选随机 "
                  f"{candidate_batch_size}x{batches_per_evaluation} 条，"
                  f"每 {reuse_candidates} 个候选刷新")
    elif sampling_mode == "keyframes":
        few_inputs, few_outputs = data_loader.load_few_shot_data(
            cfg["data_dir"], norm_stats_path,
            frames_per_ep=frames_per_ep, action_horizon=action_horizon,
            keyframe_strategy=keyframe_strategy)
        objective_outputs = few_outputs
        sample_manifest = [
            {
                "episode_index": item["episode_index"],
                "frame_index": item["frame_index"],
                "instruction": item["instruction"],
            }
            for item in few_inputs
        ]
        print(f"  共 {len(few_inputs)} 条固定样本")
    else:
        raise ValueError(f"未知 sampling_mode: {sampling_mode}")
    with open(os.path.join(out_dir, "sample_manifest.json"), "w") as f:
        json.dump(sample_manifest, f, indent=2, ensure_ascii=False)

    # ---- epoch 单位换算: 1 epoch = batches_per_epoch 批 x reuse_candidates 候选 ----
    if batch_sampler is not None and batch_sampling_order == "sequential":
        steps_per_epoch = batch_sampler.batches_per_epoch * reuse_candidates
    else:
        steps_per_epoch = None

    if budget_epochs is None:
        max_steps = int(budget_steps)
    else:
        max_steps = resolve_epochs("epochs", budget_epochs, steps_per_epoch)
        if budget_steps is not None and int(budget_steps) != max_steps:
            print(f"  [WARN] 同时设置了 max_steps={budget_steps} 与 "
                  f"epochs={budget_epochs}; epochs 优先 -> max_steps={max_steps}")
    validation_interval_candidates = resolve_interval(
        "validation_interval_epochs", intervals["validation"]["steps"],
        intervals["validation"]["epochs"], steps_per_epoch)
    weight_update_interval = resolve_interval(
        "weight_update_epochs", intervals["weight_update"]["steps"],
        intervals["weight_update"]["epochs"], steps_per_epoch)
    checkpoint_interval = resolve_interval(
        "checkpoint_epochs", intervals["checkpoint"]["steps"],
        intervals["checkpoint"]["epochs"], steps_per_epoch)
    if budget_epochs is not None:
        print(f"  训练预算换算: {budget_epochs} epoch x {steps_per_epoch} "
              f"候选/epoch = max_steps={max_steps}")
        print(f"  间隔换算(候选数): validation={validation_interval_candidates} | "
              f"weight_update={weight_update_interval} | "
              f"checkpoint={checkpoint_interval}")

    delta_stats = data_loader.compute_delta_loss_stats(
        objective_outputs, std_floor=loss_cfg.get("std_floor", 1e-3))
    delta_stats_path = os.path.join(out_dir, "delta_loss_stats.json")
    data_loader.save_delta_loss_stats(delta_stats_path, delta_stats)

    print("\n[2] 选择 LoRA 模块...")
    modules = select_lora_modules(cfg)
    print(f"  最终 {len(modules)} 个模块参与组合")

    norm_stats = data_loader.load_norm_stats(norm_stats_path)

    print(f"\n[3] LoRAHub {version} 黑盒学习 (max_steps={max_steps})...")
    learners = {
        "v1": algo.vla_lorahub_learning,
        "v2": algo.vla_lorahub_learning_v2,
        "v2.5": algo.vla_lorahub_learning_v25,
        "v3": algo.vla_lorahub_learning_v3,
    }
    learn = learners[version]
    from functools import partial
    if loss_cfg.get("type", "delta_grouped_huber") == "delta_grouped_huber":
        base_loss = partial(
            algo.delta_grouped_vla_loss,
            delta_stats=delta_stats,
            joint_weight=loss_cfg.get("joint_weight", 0.8),
            gripper_weight=loss_cfg.get("gripper_weight", 0.2),
            huber_delta=loss_cfg.get("huber_delta", 1.0),
        )
    elif loss_cfg.get("type") == "mse":
        base_loss = algo.default_vla_loss
    else:
        raise ValueError(f"未知 loss.type: {loss_cfg.get('type')}")

    def make_seeded_loss(num_steps, diffusion_seeds):
        """Average objective over common diffusion noise seeds."""
        def seeded_loss(params, inputs, outputs, _unused_forward=None):
            values, components = [], []
            for diffusion_seed in diffusion_seeds:
                def seeded_forward(p, x, *, _seed=diffusion_seed):
                    return pi0_forward(
                        p, x, cfg["train_config_name"], norm_stats,
                        num_steps, max_batch, diffusion_seed=_seed)

                algo.LAST_LOSS_COMPONENTS.clear()
                values.append(float(base_loss(
                    params, inputs, outputs, seeded_forward)))
                if algo.LAST_LOSS_COMPONENTS:
                    components.append(dict(algo.LAST_LOSS_COMPONENTS))
            if components:
                keys = set.intersection(*(set(item) for item in components))
                algo.LAST_LOSS_COMPONENTS.clear()
                algo.LAST_LOSS_COMPONENTS.update({
                    key: float(np.mean([item[key] for item in components]))
                    for key in keys
                })
            return float(np.mean(values))
        return seeded_loss

    get_loss = make_seeded_loss(diffusion_steps, search_diffusion_seeds)
    final_get_loss = make_seeded_loss(
        validation_diffusion_steps, validation_diffusion_seeds)

    if regularization_type == "uniform_kl":
        get_regular = partial(algo.uniform_kl_regularization, lam=reg_weight)
    elif regularization_type == "sparse_entropy":
        get_regular = partial(algo.default_regularization, lam=reg_weight)
    else:
        raise ValueError(f"未知 regularization_type: {regularization_type}")
    learn_kwargs = dict(
        lora_module_paths=modules,
        example_inputs=few_inputs,
        example_outputs=few_outputs,
        base_checkpoint_dir=BASE_CHECKPOINT_DIR,
        max_inference_step=max_steps,
        forward_fn=None,
        get_loss=get_loss,
        get_regular=get_regular,
        seed=seed,
        weight_bounds=weight_bounds,
        softmax_temperature=softmax_temperature,
    )
    if version in ("v1", "v2", "v2.5", "v3"):
        learn_kwargs.update(
            logit_bound=logit_bound,
            cma_scale=cma_scale,
            cma_popsize=cma_popsize,
        )
    if version in ("v1", "v2", "v2.5", "v3"):
        learn_kwargs["composition_mode"] = composition_mode
    if version == "v2":
        learn_kwargs["v2_mode"] = v2_mode
    if version in ("v2", "v2.5", "v3"):
        learn_kwargs["batch_sampler"] = batch_sampler
        learn_kwargs["final_get_loss"] = final_get_loss
        learn_kwargs["validation_interval_candidates"] = (
            validation_interval_candidates)
    if version == "v3":
        learn_kwargs.update(
            progress_dir=out_dir,
            weight_update_interval=weight_update_interval,
            checkpoint_interval=checkpoint_interval,
            resume=resume,
        )
    optimal_weights, _, _ = learn(**learn_kwargs)

    runtime_stats = get_pi0_runtime_stats()
    if runtime_stats["partial_updates"] or runtime_stats["full_updates"]:
        update_mib = runtime_stats["updated_bytes"] / 2**20
        print("\n[4] Pi0 参数更新统计...")
        print(f"  initial_device={runtime_stats['initial_device_bytes'] / 2**30:.2f} GiB "
              f"({runtime_stats['initial_device_seconds']:.2f}s)")
        print(f"  partial/full updates={runtime_stats['partial_updates']}/"
              f"{runtime_stats['full_updates']} | leaves={runtime_stats['updated_leaves']} | "
              f"transfer={update_mib:.1f} MiB | "
              f"update_time={runtime_stats['update_seconds']:.2f}s")
        print(f"  non_lora_identity_checks="
              f"{runtime_stats['non_lora_identity_checks']}")

    print("\n[5] 导出权重...")
    w = np.asarray(optimal_weights)
    mod_names = [os.path.basename(m) for m in modules]
    if version == "v1":
        weight_info = {
            "version": "v1",
            "module_names": mod_names,
            "optimal_weights": w.tolist(),
            "composition_mode": composition_mode,
        }
    elif version in ("v2", "v2.5"):
        weight_info = {
            "version": ("v2_5_per_group" if version == "v2.5"
                        else "v2_per_group"),
            "module_names": mod_names,
            "group_names": [g["name"] for g in algo.LAYER_GROUPS_V2],
            "optimal_weights": w.tolist(),
            "composition_mode": composition_mode,
            "v2_mode": v2_mode,
            "sampling_mode": sampling_mode,
        }
    else:  # v3
        weight_info = {
            "version": "v3_per_layer",
            "module_names": mod_names,
            "layer_names": [g["name"] for g in algo.LAYER_GROUPS_V3],
            "optimal_weights": w.tolist(),
            "composition_mode": composition_mode,
            "sampling_mode": sampling_mode,
            "batch_sampling_order": batch_sampling_order,
        }
    weight_info["objective"] = {
        "max_steps": int(max_steps),
        "search_diffusion_steps": int(diffusion_steps),
        "search_diffusion_seeds": search_diffusion_seeds,
        "validation_diffusion_steps": validation_diffusion_steps,
        "validation_diffusion_seeds": validation_diffusion_seeds,
        "validation_interval_candidates": validation_interval_candidates,
        "norm_stats_path": os.path.relpath(norm_stats_path, ROOT),
        "reference_norm_stats_path": reference_norm_stats_path,
    }
    if budget_epochs is not None:
        weight_info["objective"]["epochs"] = budget_epochs
        weight_info["objective"]["steps_per_epoch"] = int(steps_per_epoch)
    if batch_sampler is not None and batch_sampler.frame_pool is not None:
        weight_info["objective"]["frame_pool"] = {
            "enabled": True,
            "gripper_delta_threshold": a_cfg.get(
                "gripper_delta_threshold", 0.005),
            "static_stride": a_cfg.get("static_stride", 2),
            "pool_size": int(batch_sampler.pool_size),
            "dataset_size": len(frame_dataset),
        }

    out_path = os.path.join(out_dir, "lorahub_weights.json")
    with open(out_path, "w") as f:
        json.dump(weight_info, f, indent=2)
    print(f"  已保存到 {out_path}")

    print("\n[6] 绘制优化曲线...")
    plot_loss_curves(version, out_dir, algo.SCORE_HISTORY[version])

    print(f"\n{'=' * 60}")
    print(f"[{version}] 优化完成! 成果: outputs/{task}/lorahub_weights.json")
    print(f"下一步导出评测 checkpoint: bash compose_checkpoint.sh {task}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
