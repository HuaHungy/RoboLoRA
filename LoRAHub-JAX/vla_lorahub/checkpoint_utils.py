"""
Orbax Checkpoint 工具：从 Pi0 的 JAX/Orbax checkpoint 中提取 LoRA 权重。

Pi0 = PaliGemma (ViT + LLM) + Action Expert

确认的 LoRA 参数 (step 5000 checkpoint, 共 20 个):
  attn/q_einsum/lora_a, lora_b        (Q 投影, 主 LLM)
  attn/q_einsum_1/lora_a, lora_b      (Q 投影, 辅助模块)
  attn/kv_einsum/lora_a, lora_b       (KV 投影)
  attn/kv_einsum_1/lora_a, lora_b
  attn/attn_vec_einsum/lora_a, lora_b (注意力输出)
  attn/attn_vec_einsum_1/lora_a, lora_b
  mlp/gating_einsum_lora_a, lora_b    (MLP 门控)
  mlp/linear_lora_a, lora_b           (MLP 线性)
  mlp_1/gating_einsum_lora_a, lora_b
  mlp_1/linear_lora_a, lora_b
"""

import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional


# ---- 核心: 使用 orbax 官方 API 加载 checkpoint ----

def _flatten_pytree(tree, prefix: str = "") -> Dict[str, np.ndarray]:
    """
    将 JAX PyTree 扁平化为 {path: np.ndarray}
    JAX Array 转换为 numpy (CPU) 以便后续操作。
    """
    result = {}
    if isinstance(tree, dict):
        for k, v in tree.items():
            new_prefix = f"{prefix}/{k}" if prefix else str(k)
            result.update(_flatten_pytree(v, new_prefix))
    elif isinstance(tree, (list, tuple)):
        for i, v in enumerate(tree):
            new_prefix = f"{prefix}/{i}" if prefix else str(i)
            result.update(_flatten_pytree(v, new_prefix))
    elif isinstance(tree, np.ndarray):
        result[prefix] = tree
    elif hasattr(tree, 'shape'):
        # jax.Array — 转为 numpy
        result[prefix] = np.array(tree)
    elif hasattr(tree, 'item'):
        result[prefix] = np.array(tree)
    return result


def load_ocdbt_params(params_dir: str) -> Dict[str, np.ndarray]:
    """
    从 Orbax/OCDBT 格式的 params 目录加载所有参数。
    """
    import orbax.checkpoint as ocp
    from etils import epath
    import jax

    path = epath.Path(os.path.abspath(params_dir))
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint 路径不存在: {path}")

    mesh = jax.sharding.Mesh(jax.devices(), ("x",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    restore_args = ocp.ArrayRestoreArgs(sharding=sharding, restore_type=np.ndarray)

    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(path)
        # metadata 在 orbax 0.11 是 dict, 在 0.12+ 可能是 StepMetadata
        if isinstance(metadata, dict):
            item = {"params": metadata["params"]}
        else:
            # StepMetadata: 使用 item_metadata 或直接 restore
            item = getattr(metadata, "item_metadata", None)
            if item is None:
                # 无法获取 item, 回退到直接 restore
                item = {"params": metadata["params"]}

        restored = ckptr.restore(
            path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree.map(lambda _: restore_args, item),
            ),
        )
        if isinstance(restored, dict) and "params" in restored:
            restored = restored["params"]
    return _flatten_pytree(restored)


def load_checkpoint_params(checkpoint_dir: str) -> Dict[str, np.ndarray]:
    """
    一站式: 从 checkpoint 目录加载所有参数。

    Args:
        checkpoint_dir: pi0_weights/5000/ 这样的目录

    Returns:
        {param_path: np.ndarray} 包含全部参数 (基座 + LoRA)
    """
    params_dir = os.path.join(checkpoint_dir, "params")
    return load_ocdbt_params(params_dir)


# ---- LoRA 相关常量 ----
# 从 _METADATA 中确认的 LoRA 参数路径模式
LORA_KEY_PATTERNS = [
    "lora_a",
    "lora_b",
    "gating_einsum_lora_a",
    "gating_einsum_lora_b",
    "linear_lora_a",
    "linear_lora_b",
]

NON_LORA_KEY_PATTERNS = [
    "lora_a",
    "lora_b",
]


def is_lora_param(key: str) -> bool:
    """判断参数名是否为 LoRA 参数"""
    key_lower = key.lower()
    return any(pat in key_lower for pat in LORA_KEY_PATTERNS) and "lora" in key_lower


def extract_lora_weights(params: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    从完整参数中提取仅 LoRA 相关的权重。

    Args:
        params: load_ocdbt_params() 的返回值

    Returns:
        {key: array} 仅包含 lora_a 和 lora_b 的参数
    """
    lora_params = {}
    for key, value in params.items():
        if is_lora_param(key):
            lora_params[key] = value
    return lora_params


def extract_base_weights(params: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """提取非 LoRA 的基座参数（w, bias, scale 等）"""
    base_params = {}
    for key, value in params.items():
        if not is_lora_param(key):
            base_params[key] = value
    return base_params


def save_lora_module(lora_params: Dict[str, np.ndarray],
                     save_path: str,
                     metadata: Optional[dict] = None):
    """保存 LoRA 模块到 .npz 文件"""
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

    save_dict = {}
    for key, arr in lora_params.items():
        save_dict[key] = arr

    if metadata:
        save_dict["__metadata__"] = np.array([json.dumps(metadata)], dtype=object)

    np.savez(save_path, **save_dict)
    print(f"[OK] 保存 {len(lora_params)} 个 LoRA 参数到 {save_path}")


def load_lora_module(load_path: str) -> Tuple[Dict[str, np.ndarray], dict]:
    """加载 LoRA 模块"""
    data = np.load(load_path, allow_pickle=True)
    lora_params = {}
    metadata = {}
    for key in data.files:
        if key == "__metadata__":
            metadata = json.loads(str(data[key][0]))
        else:
            lora_params[key] = data[key]
    return lora_params, metadata


def list_checkpoint_lora(checkpoint_dir: str) -> List[str]:
    """列出 checkpoint 中所有 LoRA 参数的路径"""
    params_dir = os.path.join(checkpoint_dir, "params")
    if not os.path.exists(params_dir):
        raise FileNotFoundError(f"找不到 params 目录: {params_dir}")

    params = load_ocdbt_params(params_dir)
    lora_keys = [k for k in params.keys() if is_lora_param(k)]
    return lora_keys


# ---- 便捷函数 ----
def extract_lora_from_checkpoint(checkpoint_dir: str,
                                  output_path: str,
                                  task_name: str = "unknown"):
    """
    一站式：从 checkpoint 提取 LoRA 权重并保存。

    Args:
        checkpoint_dir: pi0_weights/5000/ 这样的目录
        output_path: 输出的 .npz 文件路径
        task_name: 任务名称，记录到 metadata 中
    """
    params_dir = os.path.join(checkpoint_dir, "params")
    print(f"[INFO] 从 {params_dir} 加载参数...")

    params = load_ocdbt_params(params_dir)
    print(f"[INFO] 共加载 {len(params)} 个参数")

    lora_params = extract_lora_weights(params)
    print(f"[INFO] 其中 {len(lora_params)} 个是 LoRA 参数")

    metadata = {
        "task_name": task_name,
        "checkpoint_dir": checkpoint_dir,
        "num_lora_params": len(lora_params),
        "lora_keys": list(lora_params.keys()),
    }

    save_lora_module(lora_params, output_path, metadata)
    return lora_params


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python checkpoint_utils.py <checkpoint_dir> [output_path]")
        print("示例: python checkpoint_utils.py pi0_weights/5000 lora_modules/grab_roller.npz")
        sys.exit(1)

    ckpt_dir = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "extracted_lora.npz"

    extract_lora_from_checkpoint(ckpt_dir, out_path, task_name=os.path.basename(ckpt_dir))
