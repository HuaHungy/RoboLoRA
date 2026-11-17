"""
VLA LoRAHub - 模块路径注册表

列出所有可用的 LoRA 模块 (.npz 文件)。
当你在不同任务上训练了多个 LoRA 后，将它们的路径添加到这里。

模块命名规范:
  {base_model}/{task_name}/{robot_embodiment}

示例:
  pi0/grab_roller/realman.npz
  pi0/pick_place/realman.npz
  pi0/pour_water/realman.npz
"""

import os

# --- 基础路径 ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LORA_MODULES_DIR = os.path.join(BASE_DIR, "lora_modules")

# --- 已注册的 LoRA 模块 ---
# 格式: 每个元素是 .npz 文件路径
# 从你的现有 checkpoint 中提取后，会自动填充这里

LORA_MODULE_PATHS = [
    # 示例: os.path.join(LORA_MODULES_DIR, "task_grab_roller.npz"),
    # 示例: os.path.join(LORA_MODULES_DIR, "task_pick_place.npz"),
    # 示例: os.path.join(LORA_MODULES_DIR, "task_pour_water.npz"),
]

# --- 动态发现 ---
def discover_lora_modules(modules_dir: str = None) -> list:
    """自动发现 lora_modules/ 目录下的所有 .npz 文件"""
    if modules_dir is None:
        modules_dir = LORA_MODULES_DIR

    if not os.path.isdir(modules_dir):
        return []

    modules = []
    for f in sorted(os.listdir(modules_dir)):
        if f.endswith(".npz"):
            modules.append(os.path.join(modules_dir, f))
    return modules


def get_all_modules() -> list:
    """获取所有可用模块 (已注册 + 自动发现)"""
    discovered = discover_lora_modules()
    all_modules = list(LORA_MODULE_PATHS)
    for d in discovered:
        if d not in all_modules:
            all_modules.append(d)
    return all_modules


# --- Pi0 基座模型 checkpoint ---
# 纯净的 Pi0 基座模型 (未微调), 所有 LoRA 模块在此基座上独立训练
# 格式: 40 个非 LoRA 参数的纯基座 (无 train_state/assets)
BASE_CHECKPOINT_DIR = os.path.join(BASE_DIR, "openpi_base", "pi0_base")
