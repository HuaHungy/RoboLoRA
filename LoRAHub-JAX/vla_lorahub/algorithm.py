"""
VLA LoRAHub 核心算法
====================

复刻 lorahub/algorithm.py 的架构，适配 JAX/Orbax checkpoint 和 VLA (Vision-Language-Action) 场景。

核心差异 vs 原始 LoRAHub:
  1. 基座模型: Pi0 (PaliGemma + Action Expert) 替代 Flan-T5
  2. Checkpoint 格式: Orbax OCDBT 替代 HuggingFace safetensors
  3. 损失函数: 动作预测 MSE 替代文本生成 Cross-Entropy
  4. LoRA 组合: 同样使用加权线性求和 (Σ w_i * LoRA_i)

黑盒优化流程 (与原始 LoRAHub 一致):
  1. 加载基座模型参数 + N 个 LoRA 模块的 lora_a/lora_b
  2. 为每个 LoRA 分配正权重 w_i > 0
  3. Nevergrad NGOpt 优化器搜索最优权重
  4. 每步: 组合 LoRA → 前向推理 → 计算动作 MSE loss
  5. 无反向传播! 纯黑盒搜索
"""

import os
import json
import copy
import glob
import pickle
import tempfile
import warnings
from typing import List, Dict, Tuple, Optional, Callable, Union
from functools import partial

import numpy as np
import nevergrad as ng
from tqdm import tqdm

# 手动 ask/tell 循环时 nevergrad 注入机制会逐代输出无害警告, 屏蔽之
warnings.filterwarnings("ignore", message=".*orphanated injected solution.*")

from .checkpoint_utils import (
    load_lora_module,
    load_ocdbt_params,
    extract_lora_weights,
    extract_base_weights,
    is_lora_param,
)


# ============================================================
# LoRA 组合逻辑
# ============================================================

# Pi0 LoRA 参数配对关系 (从 checkpoint _METADATA 确认):
#   每个投影层有 lora_a (降维矩阵) 和 lora_b (升维矩阵)
#   前向传播时: output = base(x) + (x @ lora_a) @ lora_b
#
#   提取 .npz 中的 key 示例:
#     "params/PaliGemma/llm/layers/attn/q_einsum/lora_a/value"
#     "params/PaliGemma/llm/layers/attn/q_einsum/lora_b/value"
#     "params/PaliGemma/llm/layers/attn/q_einsum/w/value"  (基座, 不在 .npz 中)
#
#   配对规则: key 中将 "lora_a" 替换为 "lora_b" 即得到配对参数

def _get_lora_pair_key(lora_key: str) -> str:
    """获取配对的 lora_a ↔ lora_b key

    处理两种命名格式:
      attn 层: .../lora_a/value  ↔ .../lora_b/value
      mlp 层:  ..._lora_a/value  ↔ ..._lora_b/value
    """
    if "lora_a" in lora_key:
        return lora_key.replace("lora_a", "lora_b")
    elif "lora_b" in lora_key:
        return lora_key.replace("lora_b", "lora_a")
    return lora_key


def _get_lora_base_path(lora_key: str) -> str:
    """从 lora_a 或 lora_b 的 key 提取模块基础路径"""
    # 去掉 /lora_a/value 或 /lora_b/value 或 _lora_a/value 等后缀
    import re
    return re.sub(r'[/_]?lora_[ab]/value$', '', lora_key)


def _get_base_w_key(lora_a_key: str, base_params: dict = None) -> str:
    """从 lora_a 的 key 推导基座 w 的 key。

    Pi0 中不同层有不同命名, 且新旧 checkpoint 格式不同:
      attn 层: .../q_einsum/w/value 或 .../q_einsum/w
      mlp 层:  .../gating_einsum/value 或 .../gating_einsum

    同时处理 LoRA .npz 有 "params/" 前缀而基座没有的情况。
    """
    base_path = _get_lora_base_path(lora_a_key)

    # 候选后缀 (从具体到抽象)
    suffixes = ["/w/value", "/w", "/value", ""]
    candidates = [base_path + s for s in suffixes]

    # 如果 base_path 有 "params/" 前缀, 也尝试去掉前缀的版本
    if base_path.startswith("params/"):
        base_path_no_prefix = base_path[len("params/"):]
        candidates += [base_path_no_prefix + s for s in suffixes]

    if base_params is not None:
        for c in candidates:
            if c in base_params:
                return c

    return candidates[0]  # 默认返回最具体的


def compose_lora_weights(
    lora_cache: Dict[str, Dict[str, np.ndarray]],
    weights: np.ndarray,
    lora_module_list: List[str],
    composition_mode: str = "factor",
) -> Dict[str, np.ndarray]:
    """
    在激活空间对 LoRA 进行加权线性组合。

    ``factor`` reproduces the historical LoRAHub factor blend and may contain
    cross-adapter A_i @ B_j terms.  ``delta_concat`` concatenates rank blocks so
    that A_cat @ B_cat is exactly Σ w_i (A_i @ B_i) for positive weights.
    """
    if len(lora_module_list) == 0:
        return {}

    weights = np.asarray(weights, dtype=np.float32)
    return _compose_lora_generic(
        lora_cache, lora_module_list,
        weight_for_key=lambda _key, index: float(weights[index]),
        exact_for_key=lambda _key: composition_mode == "delta_concat",
        sketch_for_key=lambda _key: composition_mode == "delta_sketch",
        composition_mode=composition_mode,
    )


def _normalize_lora_key(key: str) -> str:
    return key[7:] if key.startswith("params/") else key


def _get_lora_value(params: dict, normalized_key: str) -> np.ndarray:
    value = params.get(normalized_key)
    if value is None:
        value = params.get("params/" + normalized_key)
    if value is None:
        raise KeyError(f"LoRA 模块缺少参数: {normalized_key}")
    return np.asarray(value, dtype=np.float32)


def _compose_lora_generic(lora_cache, lora_module_list, weight_for_key,
                          exact_for_key, sketch_for_key,
                          composition_mode: str, include_key=None):
    valid_modes = (
        "factor", "delta_concat", "action_delta_concat",
        "delta_sketch", "action_delta_sketch",
    )
    if composition_mode not in valid_modes:
        raise ValueError(f"未知 LoRA composition_mode: {composition_mode}")

    first = lora_cache[lora_module_list[0]]
    keys = [_normalize_lora_key(k) for k in first.keys()]
    if include_key is not None:
        keys = [key for key in keys if include_key(key)]
    final = {}
    handled = set()
    for key in keys:
        if key in handled:
            continue
        pair_key = _get_lora_pair_key(key)
        use_exact = ("lora_a" in key and pair_key in keys and exact_for_key(key))
        if use_exact:
            a_blocks, b_blocks = [], []
            for index, module_name in enumerate(lora_module_list):
                weight = max(0.0, float(weight_for_key(key, index)))
                scale = np.float32(np.sqrt(weight))
                params = lora_cache[module_name]
                a_blocks.append(scale * _get_lora_value(params, key))
                b_blocks.append(scale * _get_lora_value(params, pair_key))
            # A[..., in, rank] and B[..., rank, out].  Block concatenation
            # removes all cross-module products while preserving exact deltas.
            final[key] = np.concatenate(a_blocks, axis=-1)
            final[pair_key] = np.concatenate(b_blocks, axis=-2)
            handled.update((key, pair_key))
            continue

        accumulated = None
        for index, module_name in enumerate(lora_module_list):
            weight = float(weight_for_key(key, index))
            if sketch_for_key(key) and ("lora_a" in key or "lora_b" in key):
                # A single fixed-sign rank sketch is shape-compatible with the
                # trained Pi0 LoRA.  Its diagonal terms are w_i A_iB_i, but
                # cross terms remain; this is an explicitly approximate mode.
                sign = 1.0 if index % 2 == 0 else -1.0
                coefficient = sign * np.sqrt(max(0.0, weight))
            else:
                coefficient = weight
            weighted = coefficient * _get_lora_value(lora_cache[module_name], key)
            accumulated = weighted if accumulated is None else accumulated + weighted
        final[key] = accumulated
        handled.add(key)
    return final


def build_composed_model_params(
    base_params: Dict[str, np.ndarray],
    composed_lora: Dict[str, np.ndarray],
    update_lora_keys: Optional[set[str]] = None,
) -> Dict[str, np.ndarray]:
    """
    构建推理用的完整参数字典: base_params + composed_lora 键值。

    不进行 weight 级别的合并! 原因:
      1. LoRA key (如 .../lora_a/value) 和基座 key (如 .../w/value) 是不同参数
      2. 真正的合并发生在 Pi0 推理时的 activation space
         output = base(x) + (x @ composed_lora_a) @ composed_lora_b

    此函数将两者放入同一个字典，标记哪些是 LoRA、哪些是基座，
    以便 forward 函数区分处理。

    Args:
        base_params: 基座参数字典
        composed_lora: 组合后的 LoRA 参数字典

    Returns:
        {**base_params, **composed_lora, "__is_lora_keys__": set()}
    """
    # Base arrays are immutable during search.  A deep copy duplicated roughly
    # 12GB on every candidate; a shallow dictionary is sufficient because only
    # LoRA keys and the marker below are added/replaced.
    merged = dict(base_params)

    # 直接加入 composed_lora 的 key (不会与 base 冲突，因为 key 名不同)
    for key, val in composed_lora.items():
        merged[key] = val

    # 存储全部 LoRA key，供 forward 函数从基座参数中分离 LoRA。
    merged["__lora_keys__"] = set(composed_lora.keys())

    # 首次前向始终加载全部 LoRA。模型运行时建立后，forward 只更新这里
    # 标记的叶子。V2 action_focused 可据此跳过固定不变的主干 LoRA。
    if update_lora_keys is None:
        update_lora_keys = set(composed_lora.keys())
    else:
        update_lora_keys = set(update_lora_keys)
        unknown = update_lora_keys - set(composed_lora.keys())
        if unknown:
            raise ValueError(f"update_lora_keys 包含未知 LoRA key: {sorted(unknown)}")
    merged["__lora_update_keys__"] = update_lora_keys

    return merged


def get_lora_pair_shapes(composed_lora: Dict[str, np.ndarray],
                         base_params: Dict[str, np.ndarray] = None) -> Dict[str, dict]:
    """
    返回每个 LoRA 模块的配对信息，用于在 Pi0 forward 中正确应用。

    对于每个 lora_a key，找到其配对的 lora_b:
      delta_weight = lora_b[l,h] @ lora_a[l,h].T   (假设 A 是 [d_in, r], B 是 [r, d_out])

    实际验证 (从 checkpoint 数据):
      q_einsum: lora_a=(18,8,2048,16)  lora_b=(18,8,16,256)  w=(18,8,2048,256)
      → lora_a 最后两维 @ lora_b 最后两维:
        (2048, 16) @ (16, 256) = (2048, 256) ✓ 匹配 w 的最后两维
      → 所以 delta = lora_a @ lora_b (保留 batch dims 18,8)

    Returns:
        {module_base_key: {"lora_a_key": ..., "lora_b_key": ..., "base_w_key": ...}}
        例如: {".../q_einsum": {"lora_a": "...lora_a/value", "lora_b": "...lora_b/value", "w": "...w/value"}}
    """
    pairs = {}
    lora_a_keys = [k for k in composed_lora if "lora_a" in k]

    for a_key in lora_a_keys:
        b_key = _get_lora_pair_key(a_key)
        base_path = _get_lora_base_path(a_key)
        w_key = _get_base_w_key(a_key, base_params)

        if b_key in composed_lora:
            pairs[base_path] = {
                "lora_a_key": a_key,
                "lora_b_key": b_key,
                "w_key": w_key,
                "lora_a_shape": composed_lora[a_key].shape,
                "lora_b_shape": composed_lora[b_key].shape,
            }
        else:
            print(f"[WARN] 找不到配对 lora_b: {a_key} → {b_key}")

    return pairs


def compute_merged_weight(
    base_w: np.ndarray,
    lora_a: np.ndarray,
    lora_b: np.ndarray,
) -> np.ndarray:
    """
    计算 weight-space 合并: merged_w = w + lora_a @ lora_b

    用于在将 LoRA 合并到基座权重后导出为普通 checkpoint 的场景。
    注意: 合并后不能再分离 LoRA，适合推理部署。

    验证 (q_einsum 为例):
      w:      (18, 8, 2048, 256)
      lora_a: (18, 8, 2048, 16)
      lora_b: (18, 8, 16, 256)
      lora_a @ lora_b: (18, 8, 2048, 256)  ← 最后两维 matmul
    """
    # 对最后两维做 matmul，保留前面的 batch 维度
    delta = np.matmul(lora_a, lora_b)  # 自动对最后两维做矩阵乘法
    return base_w + delta


def merge_all_lora_to_base(
    base_params: Dict[str, np.ndarray],
    composed_lora: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    将所有 LoRA 合并到基座权重中 (用于最终模型导出)。

    遍历所有 lora_a → 找到配对的 lora_b 和基座 w
    → 计算 merged_w = w + lora_a @ lora_b
    → 替换基座中的 w

    最终返回的字典只包含基座参数 (含合并后的 w)，不包含 lora_a/b。
    """
    merged = copy.deepcopy(base_params)
    pairs = get_lora_pair_shapes(composed_lora, base_params)

    for base_path, info in pairs.items():
        lora_a = composed_lora[info["lora_a_key"]]
        lora_b = composed_lora[info["lora_b_key"]]
        w_key = info["w_key"]

        if w_key in merged:
            base_w = merged[w_key]
            merged_w = compute_merged_weight(base_w, lora_a, lora_b)
            merged[w_key] = merged_w
        else:
            print(f"[WARN] 找不到基座参数 '{w_key}'，跳过合并 {base_path}")

    return merged


# ============================================================
# VLA 前向推理 (模拟 — 实际使用时替换为真实的 Pi0 推理)
# ============================================================

def default_vla_forward(
    params: Dict[str, np.ndarray],
    example_inputs: List[Dict],
) -> List[np.ndarray]:
    """
    默认 VLA 前向推理函数。

    这是占位实现 — 在实际使用中需要替换为真正的 Pi0 模型推理。
    当前返回随机动作，用于演示框架流程。

    Args:
        params: 模型参数字典 (base + LoRA 合并后)
        example_inputs: 输入列表，每个元素是 {"image": ..., "state": ..., "instruction": ...}

    Returns:
        预测的动作列表 [np.ndarray shape (action_dim,), ...]
    """
    # 实际实现需要:
    # 1. 将 params 加载到 Pi0 模型中
    # 2. 对每个输入执行前向推理
    # 3. 返回预测的动作

    # 占位: 返回与 norm_stats 一致的 32 维动作
    action_dim = 32
    return [np.zeros(action_dim, dtype=np.float32) for _ in example_inputs]


# ============================================================
# 损失函数
# ============================================================

def default_vla_loss(
    composed_params: Dict[str, np.ndarray],
    example_inputs: List[Dict],
    example_outputs: List[np.ndarray],
    forward_fn: Callable = None,
) -> float:
    """
    默认 VLA 损失函数: 动作预测的 MSE。

    Args:
        composed_params: 组合后的模型参数
        example_inputs: few-shot 输入
        example_outputs: few-shot 输出 (ground-truth actions)
        forward_fn: 前向推理函数

    Returns:
        平均 MSE loss
    """
    if forward_fn is None:
        forward_fn = default_vla_forward

    predictions = forward_fn(composed_params, example_inputs)

    total_loss = 0.0
    for pred, gt in zip(predictions, example_outputs):
        pred = np.asarray(pred, dtype=np.float32)
        gt = np.asarray(gt, dtype=np.float32)
        # 支持轨迹序列 (horizon, dim) 与单帧 (dim,) 两种格式
        if pred.ndim == 1:
            pred = pred.reshape(1, -1)
        if gt.ndim == 1:
            gt = gt.reshape(1, -1)
        h = min(pred.shape[0], gt.shape[0])
        mse = np.mean((pred[:h] - gt[:h]) ** 2)
        total_loss += float(mse)

    return total_loss / len(example_outputs)


LAST_LOSS_COMPONENTS = {}


def delta_grouped_vla_loss(
    composed_params: Dict[str, np.ndarray],
    example_inputs: List[Dict],
    example_outputs: List[np.ndarray],
    forward_fn: Callable = None,
    delta_stats: dict = None,
    joint_weight: float = 0.8,
    gripper_weight: float = 0.2,
    huber_delta: float = 1.0,
) -> float:
    """Scale-balanced Huber objective in unnormalized Pi0 delta space.

    Checkpoint normalization remains untouched.  ``delta_stats`` is computed
    only from the selected GT windows and is used to balance objective units.
    """
    if forward_fn is None:
        forward_fn = default_vla_forward
    if delta_stats is None:
        raise ValueError("delta_grouped_vla_loss 需要 delta_stats")
    if joint_weight < 0 or gripper_weight < 0 or joint_weight + gripper_weight <= 0:
        raise ValueError("joint/gripper loss 权重必须非负且总和 > 0")

    predictions = forward_fn(composed_params, example_inputs)
    scale = np.asarray(delta_stats["scale"], dtype=np.float32)
    active = np.asarray(delta_stats.get("active", np.ones_like(scale)), dtype=bool)
    action_dim = len(scale)
    joint_mask = np.ones(action_dim, dtype=bool)
    if action_dim >= 14:
        joint_mask[[6, 13]] = False
    grip_mask = ~joint_mask
    joint_mask &= active
    grip_mask &= active

    def huber(values):
        absolute = np.abs(values)
        delta = float(huber_delta)
        return np.where(absolute <= delta, 0.5 * values ** 2,
                        delta * (absolute - 0.5 * delta))

    joint_losses, grip_losses = [], []
    for pred, gt in zip(predictions, example_outputs):
        pred = np.atleast_2d(np.asarray(pred, dtype=np.float32))
        gt = np.atleast_2d(np.asarray(gt, dtype=np.float32))
        horizon = min(pred.shape[0], gt.shape[0])
        dim = min(pred.shape[1], gt.shape[1], action_dim)
        residual = (pred[:horizon, :dim] - gt[:horizon, :dim]) / scale[:dim]
        jmask = joint_mask[:dim]
        gmask = grip_mask[:dim]
        if np.any(jmask):
            joint_losses.append(float(np.mean(huber(residual[:, jmask]))))
        if np.any(gmask):
            grip_losses.append(float(np.mean(huber(residual[:, gmask]))))

    joint_loss = float(np.mean(joint_losses)) if joint_losses else 0.0
    grip_loss = float(np.mean(grip_losses)) if grip_losses else 0.0
    weight_sum = float(joint_weight + gripper_weight)
    total = (float(joint_weight) * joint_loss +
             float(gripper_weight) * grip_loss) / weight_sum
    LAST_LOSS_COMPONENTS.clear()
    LAST_LOSS_COMPONENTS.update({
        "joint_loss": joint_loss,
        "gripper_loss": grip_loss,
    })
    return total


def default_l1_regularization(weights: np.ndarray) -> float:
    """
    L1 正则化: 0.05 * Σ|w_i| / N
    鼓励权重稀疏，避免过拟合。
    """
    return 0.05 * float(np.sum(np.abs(weights))) / len(weights)


def softmax_normalize(raw: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """按行做数值稳定的 Softmax 归一化 (替代 simplex 投影)。
    - 输入任意实数 logits, 输出 (0,1) 且每行和为 1。
    - 无边界裁剪, 保持梯度平滑, 对 CMA-ES 的协方差自适应更友好。
    - temperature < 1 使分布更尖锐, > 1 更均匀。
    """
    z = np.asarray(raw, dtype=np.float32) / max(float(temperature), 1e-6)
    z = z - np.max(z, axis=-1, keepdims=True)  # 数值稳定
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


def logits_bounds_from_weight_bounds(weight_bounds) -> Tuple[float, float]:
    """将原权重界变换为以 0 为中心的 logits 搜索界。

    Softmax 方案下 nevergrad 搜索的是 logits (任意实数), 且 init=0
    (均匀权重) 必须落在界内。原权重界 [low, high] (如 [1e-6, 1.5])
    不含 0, 直接使用会导致 ng.p.Array 初始化报错。
    """
    low, high = weight_bounds
    bound = max(abs(low), abs(high)) * 4.0
    return -bound, bound


def default_regularization(weights: np.ndarray, lam: float = 0.0) -> float:
    """Softmax 归一化后的稀疏正则 (替代对归一化权重恒为常数的 L1)。

    softmax 权重有界 (0,1) 且行和为 1, 此时 mean|w| 是常数, L1 对优化无效。
    默认 lam=0 不加正则; 设 lam>0 时惩罚行熵, 鼓励每层聚焦少数模块 (稀疏组合)。
    """
    w = np.asarray(weights, dtype=np.float32)
    if w.ndim == 1:
        w = w.reshape(1, -1)
    ent = -np.sum(w * np.log(w + 1e-12), axis=-1)  # 行熵 (0=集中在1个模块)
    return float(lam * np.mean(ent) / w.shape[1])


def uniform_kl_regularization(weights: np.ndarray, lam: float = 0.0) -> float:
    """Small anti-collapse penalty KL(weights || uniform)."""
    w = np.asarray(weights, dtype=np.float32)
    if w.ndim == 1:
        w = w.reshape(1, -1)
    n = w.shape[1]
    kl = np.sum(w * np.log(np.maximum(w * n, 1e-12)), axis=-1)
    return float(lam * np.mean(kl))


# ============================================================
# 统一评分函数 (供 Nevergrad 优化)
# ============================================================

# 每步优化历史 (V1/V2/V3 各一份, 由 run.py 读取并绘制 loss 曲线)
# 元素: {"step": int, "loss": float, "reg": float, "total": float}
SCORE_HISTORY = {"v1": [], "v2": [], "v2.5": [], "v3": []}


def _atomic_json_dump(obj, path: str) -> None:
    """Write JSON atomically so a power loss cannot leave a half-written file."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise


def _atomic_pickle_dump(obj, path: str) -> None:
    """Write a binary optimization checkpoint atomically."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise

def get_score(
    weights: np.ndarray,
    lora_cache: Dict[str, Dict[str, np.ndarray]],
    lora_module_list: List[str],
    base_params: Dict[str, np.ndarray],
    example_inputs: List[Dict],
    example_outputs: List[np.ndarray],
    get_loss: Callable,
    get_regular: Callable,
    forward_fn: Callable = None,
    composition_mode: str = "factor",
) -> float:
    """
    评分函数: 组合 LoRA → 构建推理参数 → 前向推理 → 计算 loss + 正则化。

    组合方式 (与原始 LoRAHub 一致):
      composed_lora_a = Σ w_i * lora_a_i
      composed_lora_b = Σ w_i * lora_b_i

    forward 函数接收 base_params + composed_lora，在 Pi0 推理时:
      output = base(x) + (x @ composed_lora_a) @ composed_lora_b

    这是 Nevergrad 优化的目标函数。
    """
    # 1. 在激活空间组合 LoRA (与原始 LoRAHub 的 PEFT forward 方式一致)
    composed_lora = compose_lora_weights(
        lora_cache, weights, lora_module_list, composition_mode=composition_mode)

    # 2. 构建推理参数 (base + composed_lora，不做 weight 级合并)
    model_params = build_composed_model_params(base_params, composed_lora)

    # 3. 计算 loss
    LAST_LOSS_COMPONENTS.clear()
    loss = get_loss(model_params, example_inputs, example_outputs, forward_fn)

    # 4. 加正则化
    regular = get_regular(weights)
    total = loss + regular

    # 5. 可视化输出
    get_score._step = getattr(get_score, '_step', 0) + 1
    w_str = "  ".join(f"{w:+.3f}" for w in weights)
    components = dict(LAST_LOSS_COMPONENTS)
    component_text = (f"  joint={components['joint_loss']:.6f}"
                      f"  grip={components['gripper_loss']:.6f}"
                      if components else "")
    print(f"  [Step {get_score._step:3d}] loss={loss:.6f}{component_text}  "
          f"reg={regular:.6f}  total={total:.6f}  |  weights: [{w_str}]", flush=True)
    entry = {"step": get_score._step, "loss": loss,
             "reg": regular, "total": total}
    entry.update(components)
    SCORE_HISTORY["v1"].append(entry)

    return total


# ============================================================
# LoRAHub 学习 (黑盒优化)
# ============================================================

def load_lora_modules(
    lora_module_paths: List[str],
) -> Tuple[Dict[str, Dict[str, np.ndarray]], List[str]]:
    """
    加载所有 LoRA 模块到缓存。

    Args:
        lora_module_paths: .npz 文件路径列表

    Returns:
        (lora_cache, module_names)
        lora_cache: {module_name: {param_key: np.ndarray}}
    """
    cache = {}
    module_names = []

    for path in lora_module_paths:
        name = os.path.splitext(os.path.basename(path))[0]
        lora_params, metadata = load_lora_module(path)
        cache[name] = lora_params
        module_names.append(name)

        print(f"[INFO] 加载 LoRA 模块 '{name}': {len(lora_params)} 个参数, "
              f"任务: {metadata.get('task_name', 'unknown')}")

    # 验证所有模块的 key 一致
    if len(cache) > 1:
        first_keys = set(list(cache.values())[0].keys())
        for name, params in cache.items():
            cur_keys = set(params.keys())
            if cur_keys != first_keys:
                print(f"[WARN] 模块 '{name}' 的 LoRA key 不一致!")
                print(f"  只有第一个: {first_keys - cur_keys}")
                print(f"  只有当前: {cur_keys - first_keys}")

    return cache, module_names


def load_base_params_from_checkpoint(checkpoint_dir: str) -> Dict[str, np.ndarray]:
    """
    从 checkpoint 加载基座参数 (非 LoRA 部分)。

    Args:
        checkpoint_dir: 例如 pi0_weights/5000/ (或基座模型的 checkpoint 目录)

    Returns:
        基座参数字典
    """
    params_dir = os.path.join(checkpoint_dir, "params")
    all_params = load_ocdbt_params(params_dir)
    base_params = extract_base_weights(all_params)
    print(f"[INFO] 加载基座参数: {len(base_params)} 个 (非 LoRA)")
    return base_params


def vla_lorahub_learning(
    lora_module_paths: List[str],
    example_inputs: List[Dict],
    example_outputs: List[np.ndarray],
    base_checkpoint_dir: str,
    max_inference_step: int = 40,
    get_loss: Callable = default_vla_loss,
    get_regular: Callable = default_regularization,
    forward_fn: Callable = None,
    seed: int = 42,
    weight_bounds: Tuple[float, float] = (1e-6, 1.5),
    softmax_temperature: float = 1.0,
    logit_bound: float = None,
    cma_scale: float = 1.0,
    cma_popsize: int = None,
    composition_mode: str = "factor",
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]]]:
    """
    VLA LoRAHub 黑盒学习主函数。

    完全复刻 lorahub/algorithm.py 的 lorahub_learning() 流程。

    Args:
        lora_module_paths: LoRA .npz 文件路径列表
        example_inputs: few-shot 输入样本
        example_outputs: few-shot 输出 (ground-truth actions)
        base_checkpoint_dir: 基座模型 checkpoint 目录
        max_inference_step: Nevergrad 优化步数 (默认 40)
        get_loss: 损失函数
        get_regular: 正则化函数
        forward_fn: 前向推理函数 (需要实现真实的 Pi0 推理)
        seed: 随机种子
        weight_bounds: 权重范围 (low, high)

    Returns:
        (optimal_weights, merged_params, lora_cache)
        - optimal_weights: 最优权重向量
        - merged_params: 合并后的完整参数
        - lora_cache: LoRA 模块缓存
    """
    # 设置随机种子
    np.random.seed(seed)

    n_loras = len(lora_module_paths)
    if n_loras == 0:
        raise ValueError("至少需要 1 个 LoRA 模块")

    print(f"\n{'='*60}")
    print(f"VLA LoRAHub 学习开始")
    print(f"  LoRA 模块数: {n_loras}")
    print(f"  Few-shot 样本数: {len(example_inputs)}")
    print(f"  优化步数: {max_inference_step}")
    shown_bound = (abs(float(logit_bound)) if logit_bound is not None
                   else abs(logits_bounds_from_weight_bounds(weight_bounds)[1]))
    print(f"  logits 范围: [-{shown_bound}, {shown_bound}], "
          f"CMA scale={cma_scale}, popsize={cma_popsize}")
    print(f"{'='*60}\n")

    # 1. 加载基座参数
    print("[Step 1] 加载基座参数...")
    base_params = load_base_params_from_checkpoint(base_checkpoint_dir)

    # 2. 加载所有 LoRA 模块
    print("[Step 2] 加载 LoRA 模块...")
    lora_cache, module_names = load_lora_modules(lora_module_paths)

    # 3. 构建评分函数
    score_fn = partial(
        get_score,
        lora_cache=lora_cache,
        lora_module_list=module_names,
        base_params=base_params,
        example_inputs=example_inputs,
        example_outputs=example_outputs,
        get_loss=get_loss,
        get_regular=get_regular,
        forward_fn=forward_fn,
        composition_mode=composition_mode,
    )

    # 4. 设置 Nevergrad 优化器 (Softmax 映射到 simplex, 搜索空间为 logits)
    if logit_bound is None:
        low, high = logits_bounds_from_weight_bounds(weight_bounds)
    else:
        low, high = -abs(float(logit_bound)), abs(float(logit_bound))
    # 初始 logits 全 0 -> Softmax 后为均匀权重 (每个模块等权)
    init_weights = np.zeros(n_loras)

    instrum = ng.p.Array(
        init=list(init_weights),
        upper=[high] * n_loras,
        lower=[low] * n_loras,
    )
    instrum.random_state.seed(seed)
    cma_factory = ng.optimizers.ParametrizedCMA(
        scale=float(cma_scale), popsize=cma_popsize)
    optimizer = cma_factory(
        parametrization=instrum,
        budget=max_inference_step,
        num_workers=1,  # Pi0/JAX 前向显存重，顺序评估更稳定
    )

    # 约束包装: 每次评估前用 Softmax 映射到 simplex (w_i ∈ (0,1), Σw=1)
    def constrained_score(raw_weights):
        # nevergrad ask() 返回 Parameter, 需取 .value (numpy) 再做数值运算
        normalized = softmax_normalize(raw_weights.value, softmax_temperature)
        return score_fn(normalized)

    # 5. 执行黑盒优化 (ask/tell 循环, tqdm 显示进度与预计时间)
    print("[Step 3] 开始黑盒优化 (Nevergrad CMA-ES, 约束: w_i>0 且 Σw_i=1)...")
    print(f"  初始权重 (Softmax 均匀): {[f'{w:.3f}' for w in softmax_normalize(init_weights, softmax_temperature)]}")
    SCORE_HISTORY["v1"].clear()
    get_score._step = 0
    best_raw, best_score = None, float("inf")
    for _ in tqdm(range(max_inference_step), desc="CMA-ES 优化", unit="step"):
        x = optimizer.ask()
        y = constrained_score(x)
        optimizer.tell(x, y)
        if y < best_score:
            best_score = float(y)
            best_raw = np.asarray(x.value, dtype=np.float32).copy()
    recommendation = optimizer.provide_recommendation()

    optimal_weights = softmax_normalize(recommendation.value, softmax_temperature)
    # CMA 不自动填充 .loss (V2/V3 已验证), 用最优权重重算最终 loss
    final_score = score_fn(optimal_weights)
    if best_raw is not None and best_score < final_score:
        optimal_weights = softmax_normalize(best_raw, softmax_temperature)
        final_score = best_score
        print("\n[结果] CMA 均值推荐不如历史最优，采用历史最优候选")
    print(f"[结果] 最优权重: {optimal_weights}")
    print(f"[结果] 最优 loss: {final_score:.6f}")

    # 6. 用最优权重组合最终 LoRA
    composed_lora = compose_lora_weights(
        lora_cache, optimal_weights, module_names, composition_mode=composition_mode)

    # 构建推理参数 (保留 lora_a/lora_b 独立，不合并到 w)
    model_params = build_composed_model_params(base_params, composed_lora)

    # 7. 打印每个模块的权重
    print(f"\n[最终权重分配]")
    for name, w in zip(module_names, optimal_weights):
        bar = "█" * int(w * 10) if w > 0.01 else "-"
        print(f"  {w:.4f} {bar} {name}")

    return optimal_weights, model_params, lora_cache


# ============================================================
# LoRAHub 推理
# ============================================================

def vla_lorahub_inference(
    merged_params: Dict[str, np.ndarray],
    example_inputs: List[Dict],
    example_outputs: Optional[List[np.ndarray]] = None,
    forward_fn: Callable = None,
) -> Tuple[List[np.ndarray], Optional[float]]:
    """
    使用优化后的组合模型进行推理。

    Args:
        merged_params: 合并后的模型参数
        example_inputs: 推理输入
        example_outputs: ground truth (可选，用于计算精度)
        forward_fn: 前向推理函数

    Returns:
        (predictions, mse_loss or None)
    """
    if forward_fn is None:
        forward_fn = default_vla_forward

    predictions = forward_fn(merged_params, example_inputs)

    mse = None
    if example_outputs is not None:
        total = 0.0
        for pred, gt in zip(predictions, example_outputs):
            total += float(np.mean((np.asarray(pred) - np.asarray(gt)) ** 2))
        mse = total / len(example_outputs)
        print(f"[推理] 平均 MSE: {mse:.6f}")

    return predictions, mse


# ============================================================
# 保存/加载优化后的模型
# ============================================================

def save_composed_model(
    merged_params: Dict[str, np.ndarray],
    optimal_weights: np.ndarray,
    module_names: List[str],
    save_dir: str,
):
    """
    保存组合后的模型参数。

    Args:
        merged_params: 完整的合并参数
        optimal_weights: 最优权重
        module_names: LoRA 模块名
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)

    # 保存合并后的参数
    save_path = os.path.join(save_dir, "composed_model.npz")
    np.savez_compressed(save_path, **merged_params)
    print(f"[OK] 合并模型已保存到 {save_path}")

    # 保存权重信息
    weight_info = {
        "module_names": module_names,
        "optimal_weights": optimal_weights.tolist(),
    }
    weight_path = os.path.join(save_dir, "lorahub_weights.json")
    with open(weight_path, "w") as f:
        json.dump(weight_info, f, indent=2)
    print(f"[OK] 权重信息已保存到 {weight_path}")

# ============================================================
# V2: Per-Group 64D 优化 (追加, 不覆盖原版)
# ============================================================

# 4 组层划分
LAYER_GROUPS_V2 = [
    {   # 0: Attention (主 LLM)
        "name": "attn_main",
        "description": "q_einsum, kv_einsum, attn_vec_einsum (主 LLM)",
    },
    {   # 1: Attention (Action Expert)
        "name": "attn_expert",
        "description": "q_einsum_1, kv_einsum_1, attn_vec_einsum_1 (Action Expert)",
    },
    {   # 2: MLP (主 LLM)
        "name": "mlp_main",
        "description": "mlp/gating_einsum, mlp/linear (主 LLM)",
    },
    {   # 3: MLP (Action Expert)
        "name": "mlp_expert",
        "description": "mlp_1/gating_einsum, mlp_1/linear (Action Expert)",
    },
]

def _get_group_id_for_key(lora_key: str) -> int:
    """根据 lora key 返回所属 group id (0-3)。"""
    key = lora_key
    if key.startswith("params/"):
        key = key[7:]
    if key.endswith("/value"):
        key = key[:-6]
    if "mlp_1" in key:
        return 3  # MLP Expert
    if "mlp" in key:
        return 2  # MLP Main
    # q_einsum_1, kv_einsum_1, attn_vec_einsum_1
    if "_1/" in key or "_1/" in key.replace("_lora_", "/"):
        return 1  # Attn Expert
    return 0  # Attn Main


def compose_lora_weights_per_group(
    lora_cache,
    weights,
    lora_module_list,
    composition_mode="factor",
    include_group_ids=None,
):
    """
    Per-group LoRA 加权组合。weights shape (4, N_modules).

    每个 lora key 根据其所属层分组, 使用该组的权重:
      composed_lora[key] = Σ w[group_id, i] * lora_i[key]
    """
    import numpy as np
    if len(lora_module_list) == 0:
        return {}

    assert weights.ndim == 2 and weights.shape[0] == 4, \
        f"weights shape must be (4, N), got {weights.shape}"

    def exact_for_key(key):
        if composition_mode == "delta_concat":
            return True
        if composition_mode == "action_delta_concat":
            return _get_group_id_for_key(key) in (1, 3)
        return False

    def sketch_for_key(key):
        if composition_mode == "delta_sketch":
            return True
        if composition_mode == "action_delta_sketch":
            return _get_group_id_for_key(key) in (1, 3)
        return False

    included = None if include_group_ids is None else set(include_group_ids)
    return _compose_lora_generic(
        lora_cache, lora_module_list,
        weight_for_key=lambda key, index: float(
            weights[_get_group_id_for_key(key), index]),
        exact_for_key=exact_for_key,
        sketch_for_key=sketch_for_key,
        composition_mode=composition_mode,
        include_key=(None if included is None else
                     lambda key: _get_group_id_for_key(key) in included),
    )


def get_score_v2(
    weights_flat,
    lora_cache,
    lora_module_list,
    base_params,
    example_inputs,
    example_outputs,
    get_loss,
    get_regular,
    forward_fn=None,
    n_modules=None,
    composition_mode="factor",
    update_group_ids=None,
    fixed_composed_lora=None,
    history_version="v2",
    batch_info=None,
):
    """
    V2 评分函数: weights_flat shape (4 * N_modules,) → 分组组合 → loss + reg。
    """
    import numpy as np
    if n_modules is None:
        n_modules = len(lora_module_list)
    weights = np.asarray(weights_flat, dtype=np.float32).reshape(4, n_modules)
    dynamic_group_ids = None if update_group_ids is None else set(update_group_ids)
    composed_lora = compose_lora_weights_per_group(
        lora_cache, weights, lora_module_list,
        composition_mode=composition_mode,
        include_group_ids=dynamic_group_ids)
    if fixed_composed_lora:
        composed_lora = {**fixed_composed_lora, **composed_lora}
    update_lora_keys = None
    if dynamic_group_ids is not None:
        update_lora_keys = {
            key for key in composed_lora
            if _get_group_id_for_key(key) in dynamic_group_ids
        }
    model_params = build_composed_model_params(
        base_params, composed_lora, update_lora_keys=update_lora_keys)
    LAST_LOSS_COMPONENTS.clear()
    loss = get_loss(model_params, example_inputs, example_outputs, forward_fn)
    regular = get_regular(weights)
    total = loss + regular

    get_score_v2._step = getattr(get_score_v2, '_step', 0) + 1
    step = get_score_v2._step
    components = dict(LAST_LOSS_COMPONENTS)
    component_text = (f"  joint={components['joint_loss']:.6f}"
                      f"  grip={components['gripper_loss']:.6f}"
                      if components else "")
    print(f"  [Step {step:3d}] loss={loss:.6f}{component_text}  "
          f"reg={regular:.6f}  total={total:.6f}", flush=True)
    for gid, ginfo in enumerate(LAYER_GROUPS_V2):
        w_str = "  ".join(f"{w:+.3f}" for w in weights[gid])
        print(f"         [{ginfo['name']:15s}] [{w_str}]")
    entry = {"step": step, "loss": loss, "reg": regular, "total": total}
    if batch_info:
        entry["batch_generation"] = batch_info.get("generation")
        entry["batch_samples"] = batch_info.get("num_samples")
    entry.update(components)
    SCORE_HISTORY[history_version].append(entry)
    return total


def get_v2_search_group_ids(v2_mode: str) -> list[int]:
    """Return searched V2 groups; V2.5 adds main attention for 45D/15 LoRAs."""
    if v2_mode == "full":
        return list(range(len(LAYER_GROUPS_V2)))
    if v2_mode == "action_focused":
        return [1, 3]
    if v2_mode == "vision_action_focused":
        # Main attention mixes visual/language prefix tokens; both Action Expert
        # groups retain direct control over trajectory generation.  Main MLP is
        # fixed to preserve a stable semantic backbone.
        return [0, 1, 3]
    raise ValueError(f"未知 v2_mode: {v2_mode}")


def vla_lorahub_learning_v2(
    lora_module_paths,
    example_inputs,
    example_outputs,
    base_checkpoint_dir,
    max_inference_step=300,
    get_loss=None,
    get_regular=None,
    forward_fn=None,
    seed=42,
    weight_bounds=(1e-6, 1.5),
    softmax_temperature: float = 1.0,
    logit_bound: float = None,
    cma_scale: float = 1.0,
    cma_popsize: int = None,
    composition_mode: str = "factor",
    v2_mode: str = "full",
    batch_sampler=None,
    history_version: str = "v2",
    display_version: str = "V2",
    final_get_loss=None,
    final_forward_fn=None,
    validation_interval_candidates: int = 0,
):
    """
    V2 黑盒学习。``action_focused`` 固定主干两组为均匀权重，只搜索
    Action Expert attention/MLP 两组，将 4N 维降为 2N 维。

    Returns:
        (optimal_weights_4xN, model_params, lora_cache)
    """
    import numpy as np
    from functools import partial
    import nevergrad as ng

    if get_loss is None:
        get_loss = default_vla_loss
    if get_regular is None:
        get_regular = default_regularization

    np.random.seed(seed)
    n_loras = len(lora_module_paths)
    if n_loras == 0:
        raise ValueError("至少需要 1 个 LoRA 模块")

    n_groups = len(LAYER_GROUPS_V2)
    search_group_ids = get_v2_search_group_ids(v2_mode)
    fixed_group_ids = set(range(n_groups)) - set(search_group_ids)
    n_search_groups = len(search_group_ids)
    total_dim = n_search_groups * n_loras

    print(f"\n{'='*60}")
    print(f"VLA LoRAHub {display_version} ({v2_mode}) 学习开始")
    print(f"  LoRA 模块数: {n_loras}")
    print(f"  层组数: {n_groups}")
    for g in LAYER_GROUPS_V2:
        print(f"    [{g['name']}] {g['description']}")
    print(f"  搜索组: {[LAYER_GROUPS_V2[i]['name'] for i in search_group_ids]}")
    print(f"  搜索维度: {n_search_groups}x{n_loras} = {total_dim}D")
    if batch_sampler is None:
        print(f"  固定评估样本数: {len(example_inputs)}")
    else:
        order_text = ("顺序 epoch" if getattr(batch_sampler, "order", "random") == "sequential"
                      else "随机")
        print(f"  Pi0 全帧池: {len(batch_sampler.dataset)} 帧 [{order_text}] | "
              f"每候选 {batch_sampler.batch_size}x"
              f"{batch_sampler.batches_per_evaluation} 样本 | "
              f"每 {getattr(batch_sampler, 'reuse_candidates', '?')} 候选一批 | "
              f"{getattr(batch_sampler, 'batches_per_epoch', '?')} 批/epoch")
    print(f"  优化步数: {max_inference_step}")
    shown_bound = (abs(float(logit_bound)) if logit_bound is not None
                   else abs(logits_bounds_from_weight_bounds(weight_bounds)[1]))
    print(f"  logits 范围: [-{shown_bound}, {shown_bound}], "
          f"CMA scale={cma_scale}, popsize={cma_popsize}")
    print(f"{'='*60}\n")

    print("[Step 1] 加载基座参数...")
    base_params = load_base_params_from_checkpoint(base_checkpoint_dir)

    print("[Step 2] 加载 LoRA 模块...")
    lora_cache, module_names = load_lora_modules(lora_module_paths)

    fixed_composed_lora = None
    if fixed_group_ids:
        uniform_weights = np.full(
            (n_groups, n_loras), 1.0 / n_loras, dtype=np.float32)
        fixed_composed_lora = compose_lora_weights_per_group(
            lora_cache, uniform_weights, module_names,
            composition_mode=composition_mode,
            include_group_ids=fixed_group_ids)
        fixed_mib = sum(value.nbytes for value in fixed_composed_lora.values()) / 2**20
        fixed_names = [LAYER_GROUPS_V2[i]["name"] for i in sorted(fixed_group_ids)]
        print(f"[INFO] 已缓存固定组 {fixed_names}: "
              f"{len(fixed_composed_lora)} 个参数, "
              f"{fixed_mib:.1f} MiB")

    score_base = partial(
        get_score_v2,
        lora_cache=lora_cache,
        lora_module_list=module_names,
        base_params=base_params,
        get_loss=get_loss,
        get_regular=get_regular,
        forward_fn=forward_fn,
        n_modules=n_loras,
        composition_mode=composition_mode,
        update_group_ids=(search_group_ids if fixed_group_ids else None),
        fixed_composed_lora=fixed_composed_lora,
        history_version=history_version,
    )
    final_score_base = partial(
        get_score_v2,
        lora_cache=lora_cache,
        lora_module_list=module_names,
        base_params=base_params,
        get_loss=(final_get_loss or get_loss),
        get_regular=get_regular,
        forward_fn=(final_forward_fn or forward_fn),
        n_modules=n_loras,
        composition_mode=composition_mode,
        update_group_ids=(search_group_ids if fixed_group_ids else None),
        fixed_composed_lora=fixed_composed_lora,
        history_version=history_version,
    )

    def score_fn(weights, evaluation_index=0, *, final=False,
                 final_objective=False):
        if batch_sampler is None:
            batch_inputs, batch_outputs, batch_info = (
                example_inputs, example_outputs, None)
        elif final:
            batch_inputs, batch_outputs, batch_info = (
                batch_sampler.get_final_batch())
        else:
            batch_inputs, batch_outputs, batch_info = (
                batch_sampler.get_batch(evaluation_index))
        scorer = final_score_base if final_objective else score_base
        return scorer(
            weights,
            example_inputs=batch_inputs,
            example_outputs=batch_outputs,
            batch_info=batch_info,
        )

    if logit_bound is None:
        low, high = logits_bounds_from_weight_bounds(weight_bounds)
    else:
        low, high = -abs(float(logit_bound)), abs(float(logit_bound))
    # 初始 logits 全 0 -> Softmax 后为均匀权重 (每个模块等权)
    init_flat = [0.0] * total_dim

    instrum = ng.p.Array(
        init=init_flat,
        upper=[high] * total_dim,
        lower=[low] * total_dim,
    )
    instrum.random_state.seed(seed)
    cma_factory = ng.optimizers.ParametrizedCMA(
        scale=float(cma_scale), popsize=cma_popsize)
    optimizer = cma_factory(
        parametrization=instrum,
        budget=max_inference_step,
        num_workers=1,
    )

    def expand_groups(search_weights):
        full = np.full((n_groups, n_loras), 1.0 / n_loras, dtype=np.float32)
        for row, group_id in enumerate(search_group_ids):
            full[group_id] = search_weights[row]
        return full

    def constrained_score(raw_weights, evaluation_index):
        raw = np.asarray(raw_weights.value, dtype=np.float32).reshape(
            n_search_groups, n_loras)
        searched = softmax_normalize(raw, softmax_temperature)
        return score_fn(
            expand_groups(searched).ravel(), evaluation_index=evaluation_index)

    print(f"[Step 3] 开始黑盒优化 (CMA-ES, {n_search_groups}-group simplex)...")
    print(f"  搜索组初始权重: {[f'{1.0 / n_loras:.3f}'] * n_loras}")
    if fixed_group_ids:
        print("  固定均匀组: " + ", ".join(
            LAYER_GROUPS_V2[i]["name"] for i in sorted(fixed_group_ids)))
    SCORE_HISTORY[history_version].clear()
    get_score_v2._step = 0
    best_raw, best_score = None, float("inf")
    validation_candidates = []
    validation_interval_candidates = max(
        0, int(validation_interval_candidates or 0))
    for evaluation_index in tqdm(
            range(max_inference_step), desc="CMA-ES 优化", unit="step"):
        x = optimizer.ask()
        y = constrained_score(x, evaluation_index)
        optimizer.tell(x, y)
        if y < best_score:
            best_score = float(y)
            best_raw = np.asarray(x.value, dtype=np.float32).copy()
        if (validation_interval_candidates and
                (evaluation_index + 1) % validation_interval_candidates == 0):
            snapshot = optimizer.provide_recommendation()
            raw_snapshot = np.asarray(snapshot.value, dtype=np.float32).copy()
            snapshot_groups = softmax_normalize(
                raw_snapshot.reshape(n_search_groups, n_loras),
                softmax_temperature)
            snapshot_score = score_fn(
                expand_groups(snapshot_groups).ravel(), evaluation_index,
                final=batch_sampler is not None, final_objective=True)
            validation_candidates.append((raw_snapshot, float(snapshot_score)))
            print(f"\n  [10-step validation @ {evaluation_index + 1}] "
                  f"loss={snapshot_score:.6f}")
    recommendation = optimizer.provide_recommendation()
    validation_candidates.append((
        np.asarray(recommendation.value, dtype=np.float32).copy(), None))
    if best_raw is not None:
        validation_candidates.append((best_raw, None))
    deduplicated = {}
    for raw_snapshot, known_score in validation_candidates:
        key = raw_snapshot.tobytes()
        previous = deduplicated.get(key)
        if previous is None or (previous[1] is None and known_score is not None):
            deduplicated[key] = (raw_snapshot, known_score)
    validation_candidates = list(deduplicated.values())

    # Search remains cheap (e.g. 2 diffusion steps), while checkpoint selection
    # uses a deployment-like objective on one common fixed validation batch.
    opt_flat, final_score = None, float("inf")
    for snapshot_index, (raw_snapshot, known_score) in enumerate(
            validation_candidates):
        candidate_groups = softmax_normalize(
            raw_snapshot.reshape(n_search_groups, n_loras),
            softmax_temperature)
        candidate_weights = expand_groups(candidate_groups)
        candidate_score = known_score
        if candidate_score is None:
            candidate_score = score_fn(
                candidate_weights.ravel(), max_inference_step + snapshot_index,
                final=batch_sampler is not None, final_objective=True)
        if candidate_score < final_score:
            final_score = float(candidate_score)
            opt_flat = candidate_weights
    assert opt_flat is not None
    # Scores from changing random mini-batches are not comparable across CMA
    # generations.  Only fixed-dataset V2 may safely select a historical best.
    if (batch_sampler is None and final_get_loss is None and
            validation_interval_candidates == 0 and best_raw is not None and
            best_score < final_score):
        best_groups = softmax_normalize(
            best_raw.reshape(n_search_groups, n_loras), softmax_temperature)
        opt_flat = expand_groups(best_groups)
        final_score = best_score
        print("\n[结果] CMA 均值推荐不如历史最优，采用历史最优候选")
    print(f"\n[结果] 最优 loss: {final_score:.6f}")

    composed_lora = compose_lora_weights_per_group(
        lora_cache, opt_flat, module_names, composition_mode=composition_mode)
    model_params = build_composed_model_params(base_params, composed_lora)

    print(f"\n[最终权重分配]")
    for gid, ginfo in enumerate(LAYER_GROUPS_V2):
        print(f"  [{ginfo['name']}] {ginfo['description']}")
        for i, name in enumerate(module_names):
            w = opt_flat[gid, i]
            bar = "\u2588" * int(w * 10) if w > 0.01 else "-"
            print(f"    {w:.4f} {bar} {name}")

    return opt_flat, model_params, lora_cache


def vla_lorahub_learning_v25(*args, **kwargs):
    """V2.5: 45D for 15 modules (main attention + both Action Expert groups)."""
    kwargs["v2_mode"] = "vision_action_focused"
    kwargs["history_version"] = "v2.5"
    kwargs["display_version"] = "V2.5"
    return vla_lorahub_learning_v2(*args, **kwargs)

# ============================================================
# V3: Per-Layer 优化 (10 层独立权重，搜索维度 = 10 × LoRA 数量)
# ============================================================

LAYER_GROUPS_V3 = [
    {"name": "attn/q_einsum",     "key": "q_einsum",     "exclude": "_1"},
    {"name": "attn/q_einsum_1",   "key": "q_einsum_1"},
    {"name": "attn/kv_einsum",    "key": "kv_einsum",    "exclude": "_1"},
    {"name": "attn/kv_einsum_1",  "key": "kv_einsum_1"},
    {"name": "attn/attn_vec",     "key": "attn_vec_einsum", "exclude": "_1"},
    {"name": "attn/attn_vec_1",   "key": "attn_vec_einsum_1"},
    {"name": "mlp/gating",        "key": "gating_einsum", "exclude": "_1"},
    {"name": "mlp/linear",        "key": "linear",       "exclude": "_1"},
    {"name": "mlp_1/gating",      "key": "gating_einsum", "require": "mlp_1"},
    {"name": "mlp_1/linear",      "key": "linear",       "require": "mlp_1"},
]

def _get_layer_id_for_key(lora_key: str) -> int:
    """根据 lora key 返回 layer id (0-9). 1:1 映射到 10 个 lora 层."""
    key = lora_key
    if key.startswith("params/"):
        key = key[7:]
    if key.endswith("/value"):
        key = key[:-6]

    for lid, info in enumerate(LAYER_GROUPS_V3):
        if info["key"] not in key:
            continue
        if "exclude" in info and info["exclude"] in key:
            continue
        if "require" in info and info["require"] not in key:
            continue
        return lid
    return 0  # fallback


def compose_lora_weights_per_layer(
        lora_cache, weights, lora_module_list, composition_mode="factor"):
    """Per-layer LoRA 加权组合。weights shape (10, N_modules)."""
    import numpy as np
    if len(lora_module_list) == 0:
        return {}
    assert weights.ndim == 2 and weights.shape[0] == 10

    action_layer_ids = {1, 3, 5, 8, 9}

    def exact_for_key(key):
        if composition_mode == "delta_concat":
            return True
        if composition_mode == "action_delta_concat":
            return _get_layer_id_for_key(key) in action_layer_ids
        return False

    def sketch_for_key(key):
        if composition_mode == "delta_sketch":
            return True
        if composition_mode == "action_delta_sketch":
            return _get_layer_id_for_key(key) in action_layer_ids
        return False

    return _compose_lora_generic(
        lora_cache, lora_module_list,
        weight_for_key=lambda key, index: float(
            weights[_get_layer_id_for_key(key), index]),
        exact_for_key=exact_for_key,
        sketch_for_key=sketch_for_key,
        composition_mode=composition_mode,
    )


def get_score_v3(weights_flat, lora_cache, lora_module_list, base_params,
                 example_inputs, example_outputs, get_loss, get_regular,
                 forward_fn=None, n_modules=None, batch_info=None,
                 composition_mode="factor", history_kind="search"):
    """V3 评分函数: weights_flat (10 * N,) → per-layer 组合 → loss."""
    import numpy as np
    if n_modules is None:
        n_modules = len(lora_module_list)
    weights = np.asarray(weights_flat, dtype=np.float32).reshape(10, n_modules)
    composed = compose_lora_weights_per_layer(
        lora_cache, weights, lora_module_list, composition_mode)
    model_params = build_composed_model_params(base_params, composed)
    LAST_LOSS_COMPONENTS.clear()
    loss = get_loss(model_params, example_inputs, example_outputs, forward_fn)
    regular = get_regular(weights)
    total = loss + regular
    get_score_v3._step = getattr(get_score_v3, '_step', 0) + 1
    s = get_score_v3._step
    components = dict(LAST_LOSS_COMPONENTS)
    component_text = (f"  joint={components['joint_loss']:.6f}"
                      f"  grip={components['gripper_loss']:.6f}"
                      if components else "")
    print(f"  [Step {s:3d}] loss={loss:.6f}{component_text}  "
          f"reg={regular:.6f}  total={total:.6f}")
    for lid, info in enumerate(LAYER_GROUPS_V3):
        print(f"         [{info['name']:16s}] " + "  ".join(f"{w:+.3f}" for w in weights[lid][:4])
              + " ... " + "  ".join(f"{w:+.3f}" for w in weights[lid][-4:]))
    entry = {"step": s, "loss": loss, "reg": regular, "total": total,
             "kind": history_kind}
    if batch_info:
        entry["batch_generation"] = batch_info.get("generation")
        entry["batch_samples"] = batch_info.get("num_samples")
    entry.update(components)
    SCORE_HISTORY["v3"].append(entry)
    return total


def vla_lorahub_learning_v3(
    lora_module_paths, example_inputs, example_outputs, base_checkpoint_dir,
    max_inference_step=300, get_loss=None, get_regular=None,
    forward_fn=None, seed=42, weight_bounds=(1e-6, 1.5),
    softmax_temperature: float = 1.0,
    logit_bound: float = None,
    cma_scale: float = 1.0,
    cma_popsize: int = None,
    batch_sampler=None,
    composition_mode: str = "factor",
    final_get_loss=None,
    final_forward_fn=None,
    validation_interval_candidates: int = 0,
    progress_dir: str = None,
    weight_update_interval: int = 0,
    checkpoint_interval: int = 0,
    resume: bool = False,
):
    """V3: 10 层 × N_modules 搜索, CMA-ES, 每层独立 Softmax。"""
    import numpy as np
    from functools import partial
    import nevergrad as ng

    if get_loss is None:
        get_loss = default_vla_loss
    if get_regular is None:
        get_regular = default_regularization

    np.random.seed(seed)
    n_loras = len(lora_module_paths)
    if n_loras == 0:
        raise ValueError("至少需要 1 个 LoRA 模块")
    n_groups = len(LAYER_GROUPS_V3)
    total_dim = n_groups * n_loras

    print(f"\n{'='*60}")
    print(f"VLA LoRAHub V3 (Per-Layer {total_dim}D) 学习开始")
    print(f"  LoRA 模块数: {n_loras}")
    print(f"  层数: {n_groups}")
    for info in LAYER_GROUPS_V3:
        print(f"    [{info['name']}]")
    print(f"  搜索维度: {n_groups}x{n_loras} = {total_dim}D")
    if batch_sampler is None:
        print(f"  固定评估样本数: {len(example_inputs)}")
    else:
        order_text = ("顺序 epoch" if getattr(batch_sampler, "order", "random") == "sequential"
                      else "随机")
        print(f"  Pi0 全帧池: {len(batch_sampler.dataset)} 帧 [{order_text}] | "
              f"每候选 {batch_sampler.batch_size}x"
              f"{batch_sampler.batches_per_evaluation} 样本 | "
              f"每 {getattr(batch_sampler, 'reuse_candidates', '?')} 候选一批 | "
              f"{getattr(batch_sampler, 'batches_per_epoch', '?')} 批/epoch")
    print(f"  优化步数: {max_inference_step}")
    shown_bound = (abs(float(logit_bound)) if logit_bound is not None
                   else abs(logits_bounds_from_weight_bounds(weight_bounds)[1]))
    print(f"  logits 范围: [-{shown_bound}, {shown_bound}], "
          f"CMA scale={cma_scale}, popsize={cma_popsize}")
    print(f"{'='*60}\n")

    print("[Step 1] 加载基座参数...")
    base_params = load_base_params_from_checkpoint(base_checkpoint_dir)

    print("[Step 2] 加载 LoRA 模块...")
    lora_cache, module_names = load_lora_modules(lora_module_paths)

    score_base = partial(
        get_score_v3,
        lora_cache=lora_cache,
        lora_module_list=module_names,
        base_params=base_params,
        get_loss=get_loss,
        get_regular=get_regular,
        forward_fn=forward_fn,
        n_modules=n_loras,
        composition_mode=composition_mode,
    )
    final_score_base = partial(
        get_score_v3,
        lora_cache=lora_cache,
        lora_module_list=module_names,
        base_params=base_params,
        get_loss=(final_get_loss or get_loss),
        get_regular=get_regular,
        forward_fn=(final_forward_fn or forward_fn),
        n_modules=n_loras,
        composition_mode=composition_mode,
    )

    def score_fn(weights, evaluation_index=0, *, final=False,
                 final_objective=False):
        if batch_sampler is None:
            batch_inputs, batch_outputs, batch_info = (
                example_inputs, example_outputs, None)
        elif final:
            batch_inputs, batch_outputs, batch_info = (
                batch_sampler.get_final_batch())
        else:
            batch_inputs, batch_outputs, batch_info = (
                batch_sampler.get_batch(evaluation_index))
        scorer = final_score_base if final_objective else score_base
        return scorer(
            weights,
            example_inputs=batch_inputs,
            example_outputs=batch_outputs,
            batch_info=batch_info,
            history_kind=("validation" if final_objective else "search"),
        )

    if logit_bound is None:
        low, high = logits_bounds_from_weight_bounds(weight_bounds)
    else:
        low, high = -abs(float(logit_bound)), abs(float(logit_bound))
    # 初始 logits 全 0 -> Softmax 后为均匀权重 (每个模块等权)
    init_flat = [0.0] * total_dim

    progress_dir = os.path.abspath(progress_dir) if progress_dir else None
    weight_update_interval = max(0, int(weight_update_interval or 0))
    checkpoint_interval = max(0, int(checkpoint_interval or 0))
    checkpoint_dir = (
        os.path.join(progress_dir, "optimization_checkpoints")
        if progress_dir else None
    )

    def make_optimizer():
        instrum = ng.p.Array(
            init=init_flat, upper=[high] * total_dim, lower=[low] * total_dim)
        instrum.random_state.seed(seed)
        cma_factory = ng.optimizers.ParametrizedCMA(
            scale=float(cma_scale), popsize=cma_popsize)
        return cma_factory(
            parametrization=instrum,
            budget=max_inference_step,
            num_workers=1,
        )

    def checkpoint_candidates():
        if not checkpoint_dir:
            return []
        latest = os.path.join(checkpoint_dir, "latest.pkl")
        numbered = sorted(
            glob.glob(os.path.join(checkpoint_dir, "step_*.pkl")),
            reverse=True,
        )
        return [latest] + [path for path in numbered if path != latest]

    def load_checkpoint():
        for path in checkpoint_candidates():
            if not os.path.exists(path):
                continue
            try:
                with open(path, "rb") as f:
                    state = pickle.load(f)
            except Exception as exc:
                print(f"  [WARN] checkpoint 损坏，跳过 {path}: {exc}")
                continue
            expected = {
                "algorithm": "v3",
                "module_names": module_names,
                "n_groups": n_groups,
                "n_modules": n_loras,
                "seed": seed,
                "max_inference_step": max_inference_step,
            }
            mismatch = [
                key for key, value in expected.items()
                if state.get(key) != value
            ]
            if mismatch:
                print(f"  [WARN] checkpoint 配置不匹配，跳过 {path}: {mismatch}")
                continue
            completed_steps = int(state.get("completed_steps", -1))
            if not 0 <= completed_steps <= max_inference_step:
                print(f"  [WARN] checkpoint 步数无效，跳过 {path}: {completed_steps}")
                continue
            print(f"  [恢复] 从 {path} 继续，已完成 {completed_steps}/{max_inference_step} 步")
            return state
        return None

    state = load_checkpoint() if resume and checkpoint_dir else None
    if state is None:
        optimizer = make_optimizer()
        start_step = 0
        best_raw, best_score = None, float("inf")
        validation_candidates = []
        SCORE_HISTORY["v3"].clear()
        get_score_v3._step = 0
    else:
        optimizer = state["optimizer"]
        start_step = int(state["completed_steps"])
        best_raw = state.get("best_raw")
        if best_raw is not None:
            best_raw = np.asarray(best_raw, dtype=np.float32)
        best_score = float(state.get("best_score", float("inf")))
        validation_candidates = state.get("validation_candidates", [])
        SCORE_HISTORY["v3"] = list(state.get("score_history", []))
        get_score_v3._step = int(state.get("score_step", start_step))
        if state.get("numpy_random_state") is not None:
            np.random.set_state(state["numpy_random_state"])

    def constrained_score(raw_weights, evaluation_index):
        raw = np.asarray(raw_weights.value, dtype=np.float32).reshape(n_groups, n_loras)
        normalized = softmax_normalize(raw, softmax_temperature)
        return score_fn(normalized.ravel(), evaluation_index=evaluation_index)

    print("[Step 3] 开始黑盒优化 (CMA-ES, 10-layer simplex)...")
    validation_interval_candidates = max(
        0, int(validation_interval_candidates or 0))

    def progress_weight_info(step, weights):
        info = {
            "version": "v3_per_layer",
            "status": "in_progress",
            "step": int(step),
            "max_steps": int(max_inference_step),
            "module_names": module_names,
            "layer_names": [g["name"] for g in LAYER_GROUPS_V3],
            "optimal_weights": np.asarray(weights).tolist(),
            "composition_mode": composition_mode,
            "sampling_mode": "pi0_random" if batch_sampler is not None
            else "keyframes",
            "weights_are_provisional": True,
            "best_observed_score": (None if not np.isfinite(best_score)
                                     else float(best_score)),
        }
        if best_raw is not None:
            info["best_observed_weights"] = softmax_normalize(
                best_raw.reshape(n_groups, n_loras), softmax_temperature
            ).tolist()
        return info

    def write_progress(step, *, force=False):
        if not progress_dir or not weight_update_interval:
            return
        if not force and step % weight_update_interval != 0:
            return
        recommendation = optimizer.provide_recommendation()
        weights = softmax_normalize(
            np.asarray(recommendation.value, dtype=np.float32).reshape(
                n_groups, n_loras),
            softmax_temperature,
        )
        _atomic_json_dump(
            progress_weight_info(step, weights),
            os.path.join(progress_dir, "lorahub_weights.json"),
        )
        print(f"  [快照] 权重 JSON 已更新: step={step}", flush=True)

    def write_checkpoint(step):
        if not checkpoint_dir or not checkpoint_interval:
            return
        if step % checkpoint_interval != 0:
            return
        checkpoint_path = os.path.join(
            checkpoint_dir, f"step_{step:09d}.pkl")
        state_to_save = {
            "format_version": 1,
            "algorithm": "v3",
            "completed_steps": int(step),
            "max_inference_step": int(max_inference_step),
            "seed": int(seed),
            "module_names": list(module_names),
            "n_groups": int(n_groups),
            "n_modules": int(n_loras),
            "softmax_temperature": float(softmax_temperature),
            "composition_mode": composition_mode,
            "optimizer": optimizer,
            "best_raw": best_raw,
            "best_score": float(best_score),
            "validation_candidates": validation_candidates,
            "score_history": list(SCORE_HISTORY["v3"]),
            "score_step": int(get_score_v3._step),
            "numpy_random_state": np.random.get_state(),
        }
        _atomic_pickle_dump(state_to_save, checkpoint_path)
        _atomic_pickle_dump(state_to_save, os.path.join(checkpoint_dir, "latest.pkl"))
        print(f"  [checkpoint] 已保存: {checkpoint_path}", flush=True)

    for evaluation_index in tqdm(
            range(start_step, max_inference_step), desc="CMA-ES 优化", unit="step",
            initial=start_step, total=max_inference_step):
        x = optimizer.ask()
        y = constrained_score(x, evaluation_index)
        optimizer.tell(x, y)
        if y < best_score:
            best_score = float(y)
            best_raw = np.asarray(x.value, dtype=np.float32).copy()
        if (validation_interval_candidates and
                (evaluation_index + 1) % validation_interval_candidates == 0):
            snapshot = optimizer.provide_recommendation()
            raw_snapshot = np.asarray(snapshot.value, dtype=np.float32).copy()
            snapshot_weights = softmax_normalize(
                raw_snapshot.reshape(n_groups, n_loras),
                softmax_temperature)
            snapshot_score = score_fn(
                snapshot_weights.ravel(), evaluation_index,
                final=batch_sampler is not None, final_objective=True)
            validation_candidates.append((raw_snapshot, float(snapshot_score)))
            print(f"\n  [10-step validation @ {evaluation_index + 1}] "
                  f"loss={snapshot_score:.6f}")
        completed_step = evaluation_index + 1
        write_progress(completed_step)
        write_checkpoint(completed_step)
    recommendation = optimizer.provide_recommendation()
    validation_candidates.append((
        np.asarray(recommendation.value, dtype=np.float32).copy(), None))
    if best_raw is not None:
        validation_candidates.append((best_raw, None))
    deduplicated = {}
    for raw_snapshot, known_score in validation_candidates:
        key = raw_snapshot.tobytes()
        previous = deduplicated.get(key)
        if previous is None or (previous[1] is None and known_score is not None):
            deduplicated[key] = (raw_snapshot, known_score)
    validation_candidates = list(deduplicated.values())

    opt_flat, final_score = None, float("inf")
    for snapshot_index, (raw_snapshot, known_score) in enumerate(
            validation_candidates):
        candidate_weights = softmax_normalize(
            raw_snapshot.reshape(n_groups, n_loras), softmax_temperature)
        candidate_score = known_score
        if candidate_score is None:
            candidate_score = score_fn(
                candidate_weights.ravel(), max_inference_step + snapshot_index,
                final=batch_sampler is not None, final_objective=True)
        if candidate_score < final_score:
            final_score = float(candidate_score)
            opt_flat = candidate_weights
    assert opt_flat is not None
    # 随机 mini-batch 下不同 generation 的 loss 不可直接比较；固定数据模式
    # 才能安全地用历史最佳候选替换 CMA recommendation。
    if (batch_sampler is None and final_get_loss is None and
            validation_interval_candidates == 0 and best_raw is not None and
            best_score < final_score):
        opt_flat = softmax_normalize(
            best_raw.reshape(n_groups, n_loras), softmax_temperature)
        final_score = best_score
        print("\n[结果] CMA 均值推荐不如历史最优，采用历史最优候选")
    print(f"\n[结果] 最优 loss: {final_score:.6f}")

    composed_lora = compose_lora_weights_per_layer(
        lora_cache, opt_flat, module_names, composition_mode)
    model_params = build_composed_model_params(base_params, composed_lora)

    print(f"\n[最终权重分配]")
    for lid, info in enumerate(LAYER_GROUPS_V3):
        print(f"  [{info['name']:16s}] " + "  ".join(
            f"{w:.3f}" for w in opt_flat[lid]))

    return opt_flat, model_params, lora_cache
