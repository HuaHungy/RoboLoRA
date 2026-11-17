#!/usr/bin/env python
"""LoRAHub 统一组合导出: 基于 Pi0 基座模型直接生成评测 checkpoint。

用法:
    python script/compose.py --config config/place_empty_cup_10.json
或通过 .sh 入口:
    bash compose_checkpoint.sh place_empty_cup_10

原理:
    非精确模式把组合后的 LoRA 叶子注入基座；delta_concat 则先精确计算
    Σ w_i A_iB_i，将其合并到基座权重，并注入原生 rank 的零 LoRA 叶子。
    两者都输出 RoboTwin/Pi0 可直接加载的标准模型树。

支持权重版本: v1 (1D), v2_per_group (4xN), v3_per_layer (10xN)。

输出: outputs/<task_name>/checkpoint/10000/{params,assets}
      (可选) 同步到 config.compose.sync_dir 供 RoboTwin 评测直接加载。
"""

import os
import sys
import json
import shutil
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import numpy as np

from vla_lorahub.algorithm import (
    _get_layer_id_for_key, LAYER_GROUPS_V3,
    _get_group_id_for_key, LAYER_GROUPS_V2,
    compose_lora_weights, compose_lora_weights_per_group,
    compose_lora_weights_per_layer,
)
from vla_lorahub.checkpoint_utils import load_lora_module
from vla_lorahub.constant import BASE_CHECKPOINT_DIR

LORA_DIR = os.path.join(ROOT, "lora_modules")


def load_config(path: str) -> dict:
    """加载任务配置, 相对路径基于项目根目录 (ROOT) 解析。"""
    with open(path) as f:
        cfg = json.load(f)
    v = cfg.get("compose", {}).get("sync_dir")
    if v and not os.path.isabs(v):
        cfg.setdefault("compose", {})["sync_dir"] = os.path.abspath(os.path.join(ROOT, v))
    return cfg


def load_lora_cache(names: list) -> tuple:
    cache, mods = {}, []
    for n in names:
        name = n.replace(".npz", "")
        path = os.path.join(LORA_DIR, n if n.endswith(".npz") else n + ".npz")
        lp, _ = load_lora_module(path)
        cache[name] = lp
        mods.append(name)
        print(f"  加载 {name}: {len(lp)} params")
    return cache, mods


def norm_key(k: str) -> str:
    """去掉 "params/" 前缀与 "value" 包裹, 得到统一 flat key。"""
    k = k[7:] if k.startswith("params/") else k
    parts = k.split("/")
    if parts[-1] == "value":
        parts = parts[:-1]
    return "/".join(parts)


def compose_weights(cache: dict, W: np.ndarray, mods: list, version: str,
                    composition_mode: str = "factor") -> dict:
    """按版本加权组合 LoRA: v1 -> 1 组, v2 -> 4 组, v3 -> 10 层。"""
    if version == "v1":
        return compose_lora_weights(
            cache, np.asarray(W, dtype=np.float32).reshape(-1), mods,
            composition_mode=composition_mode)
    if version in ("v2_per_group", "v2_5_per_group"):
        return compose_lora_weights_per_group(
            cache, np.asarray(W, dtype=np.float32), mods,
            composition_mode=composition_mode)
    if version == "v3_per_layer":
        return compose_lora_weights_per_layer(
            cache, np.asarray(W, dtype=np.float32), mods,
            composition_mode=composition_mode)
    else:
        raise ValueError(f"未知权重版本: {version}")


def inject_lora_into_base(tree: dict, composed_flat: dict, verbose: bool = False) -> dict:
    """将组合后的 LoRA 叶子注入基座树 (基座只有 w, 无 lora 分支)。

    composed_flat 的 key 形如 "PaliGemma/llm/layers/attn/q_einsum/lora_a",
    其父路径在基座树中必然存在 (w 所在模块)。直接在对应 dict 节点插入叶子,
    与 RoboTwin 模型定义 (lora.py) 的参数名一一对应:
      Einsum      -> <module>/lora_a, <module>/lora_b
      FeedForward -> <module>/gating_einsum_lora_a|b, <module>/linear_lora_a|b
    """
    n = 0
    root = tree["params"] if isinstance(tree, dict) and "params" in tree else tree
    for key, val in composed_flat.items():
        parts = key.split("/")
        node = root
        ok = True
        for p in parts[:-1]:  # 父路径 (w 所在模块)
            if not isinstance(node, dict) or p not in node:
                ok = False
                break
            node = node[p]
        if not ok:
            print(f"  跳过 (基座无此模块): {key}")
            continue
        if not isinstance(node, dict):
            print(f"  跳过 (父节点非 dict): {key}")
            continue
        node[parts[-1]] = val
        n += 1
        if verbose:
            print(f"    注入: {key} {np.shape(val)}")
    print(f"  共注入 {n} 个 lora 叶子 (组合后)")
    return tree


def merge_exact_delta_into_base(tree: dict, exact_flat: dict,
                                zero_lora_flat: dict,
                                verbose: bool = False) -> dict:
    """Merge exact concatenated LoRA deltas into base weights.

    ``delta_concat`` is exact but has N-times rank.  Deployment keeps the
    official Pi0 rank and graph by folding ``A_cat @ B_cat`` into each base
    weight and injecting zero-valued native-rank LoRA leaves.  This removes all
    cross-adapter terms without requiring a custom RoboTwin model definition.
    """
    root = tree["params"] if isinstance(tree, dict) and "params" in tree else tree
    normalized = {norm_key(key): value for key, value in exact_flat.items()}
    merged_count = 0
    for a_key, lora_a in normalized.items():
        if "lora_a" not in a_key:
            continue
        b_key = a_key.replace("lora_a", "lora_b")
        if b_key not in normalized:
            raise KeyError(f"精确组合缺少配对参数: {b_key}")
        parts = a_key.split("/")
        leaf = parts[-1]
        if leaf == "lora_a":
            base_leaf = "w"
        elif leaf.endswith("_lora_a"):
            base_leaf = leaf[:-len("_lora_a")]
        else:
            raise ValueError(f"无法推导 LoRA 对应基座叶子: {a_key}")
        node = root
        for part in parts[:-1]:
            node = node[part]
        if base_leaf not in node:
            raise KeyError(f"基座缺少待合并参数: {'/'.join(parts[:-1])}/{base_leaf}")

        base = np.asarray(node[base_leaf])
        lora_b = normalized[b_key]
        # Work one transformer layer at a time.  This avoids materializing a
        # second full MLP delta tensor (several GiB) during checkpoint export.
        merged = np.array(base, copy=True)
        if base.ndim >= 3 and lora_a.shape[0] == base.shape[0]:
            for layer in range(base.shape[0]):
                delta = np.matmul(lora_a[layer], lora_b[layer])
                merged[layer] = (base[layer].astype(np.float32) + delta).astype(
                    base.dtype)
        else:
            delta = np.matmul(lora_a, lora_b)
            merged = (base.astype(np.float32) + delta).astype(base.dtype)
        node[base_leaf] = merged
        merged_count += 1
        if verbose:
            print(f"    合并 delta: {a_key} -> {base_leaf} {base.shape}")

    print(f"  共精确合并 {merged_count} 个 LoRA delta 到基座权重")
    return inject_lora_into_base(tree, zero_lora_flat, verbose=verbose)


def build_assets(out_assets: str, task: str, comp: dict) -> str:
    """从任务 norm_stats 生成 assets/<asset_id>/norm_stats.json (基座无现成 assets)。"""
    asset_id = comp.get("asset_id")
    assert asset_id, "缺少 config.compose.asset_id (数据资产目录名, 如 place_empty_cup_pi0)"
    norm_stats_src = os.path.join(ROOT, "outputs", task, "norm_stats.json")
    assert os.path.exists(norm_stats_src), \
        f"norm_stats 不存在: {norm_stats_src} (先运行 script/compute_norm_stats.py)"
    dst_dir = os.path.join(out_assets, asset_id)
    os.makedirs(dst_dir, exist_ok=True)
    shutil.copy(norm_stats_src, os.path.join(dst_dir, "norm_stats.json"))
    print(f"  assets -> {dst_dir}/norm_stats.json")
    return dst_dir


def main():
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    parser = argparse.ArgumentParser(description="LoRAHub 组合导出入口 (基座注入模式)")
    parser.add_argument("--config", "-c", required=True, help="任务配置文件 (config/xxx.json)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    task = cfg["task_name"]
    comp = cfg.get("compose", {})

    with open(os.path.join(ROOT, "outputs", task, "lorahub_weights.json")) as f:
        info = json.load(f)
    version = info["version"]
    composition_mode = info.get("composition_mode", "factor")
    names = info["module_names"]
    W = np.array(info["optimal_weights"], dtype=np.float32)
    print(f"权重版本: {version}, shape: {W.shape}, compose={composition_mode}")

    base_params = os.path.join(BASE_CHECKPOINT_DIR, "params")
    assert os.path.isdir(base_params), f"基座模型不存在: {base_params}"

    print("\n[1/4] 加载并组合 LoRA...")
    cache, mods = load_lora_cache(names)
    composed = compose_weights(cache, W, mods, version, composition_mode)
    composed_flat = {norm_key(k): v for k, v in composed.items()}
    print(f"  组合后 key 数: {len(composed_flat)}")

    print(f"[2/4] 加载基座模型...\n  {base_params}")
    import orbax.checkpoint as ocp
    from etils import epath
    ckptr = ocp.PyTreeCheckpointer()
    tree = ckptr.restore(epath.Path(base_params))

    print("[3/4] 注入组合 LoRA...")
    if composition_mode in ("delta_concat", "action_delta_concat"):
        first = cache[mods[0]]
        zero_lora_flat = {
            norm_key(key): np.zeros_like(value)
            for key, value in first.items()
            if "lora_a" in key or "lora_b" in key
        }
        new_tree = merge_exact_delta_into_base(
            tree, composed_flat, zero_lora_flat, verbose=True)
    else:
        new_tree = inject_lora_into_base(tree, composed_flat, verbose=True)

    print("[4/4] 保存...")
    out = os.path.join(ROOT, "outputs", task, "checkpoint", "10000")
    if os.path.exists(out):
        shutil.rmtree(out)
    pd_ = os.path.join(out, "params")
    ckptr.save(epath.Path(pd_), new_tree, force=True)
    print(f"  已保存到 {pd_}/")

    dst_assets = os.path.join(out, "assets")
    build_assets(dst_assets, task, comp)

    sync_dir = comp.get("sync_dir")
    if sync_dir:
        rw = os.path.join(sync_dir, "10000")
        if os.path.exists(rw):
            shutil.rmtree(rw)
        shutil.copytree(pd_, os.path.join(rw, "params"))
        shutil.copytree(dst_assets, os.path.join(rw, "assets"))
        print(f"  同步到 {rw}")

    # A separate marker lets orchestration scripts distinguish a complete
    # export from a stale checkpoint left by an earlier run or a killed export.
    marker = os.path.join(out, ".compose_complete")
    with open(marker, "w") as f:
        f.write("ok\n")
    print(f"  完成标记: {marker}")

    print(f"\n完成! 评测 checkpoint 位于 outputs/{task}/checkpoint/10000/")


if __name__ == "__main__":
    main()
