# LoRAHub-JAX 服务器环境安装与训练

本文档用于把当前 `RoboLoRAHub` 工作区迁移到 Linux NVIDIA 服务器，并创建一套可复现的 Conda 环境。依赖版本来自当前本地已运行的 Pi0 环境；这里只安装 LoRAHub 黑盒组合所需运行时，不安装 RoboTwin 仿真、ROS、Sapien、MuJoCo、Open3D 或完整 LeRobot 训练栈。

## 1. 服务器要求

- Linux x86_64，NVIDIA GPU。
- NVIDIA 驱动支持 CUDA 12；JAX 的 CUDA 运行库由 pip 安装，一般不要求单独安装系统 CUDA Toolkit 或 `nvcc`。
- Python 3.11（本地验证版本为 3.11.14）。
- 显存：24 GB 可运行基础配置；大显存配置建议 40 GB 及以上。
- 内存：至少 32 GB，建议 64 GB。
- 磁盘：项目训练必需文件约 26 GB，建议预留至少 40 GB，若保留多个导出 checkpoint 建议 80 GB 以上。
- 建议工具：Git、rsync、tmux、Miniconda/Anaconda。

先在服务器确认驱动和磁盘：

```bash
nvidia-smi
df -h
```

## 2. 必须保留的目录结构

`LoRAHub-JAX/script/run.py` 会从相邻的 RoboTwin 目录加载 OpenPI 源码，因此服务器上应保持下面的相对位置：

```text
RoboLoRAHub/
├── LoRAHub-JAX/
│   ├── config/
│   ├── data/place_empty_cup_10/
│   ├── lora_modules/
│   └── openpi_base/pi0_base/params/
└── RoboTwin/
    └── policy/pi0/
        ├── src/openpi/
        └── packages/openpi-client/
```

以下内容缺一不可：

- `LoRAHub-JAX/data/place_empty_cup_10`：10 条 episode 数据。
- `LoRAHub-JAX/lora_modules/*.npz`：当前 15 个候选 LoRA，约 3.2 GB。
- `LoRAHub-JAX/openpi_base/pi0_base/params`：Pi0 基座 Orbax checkpoint，约 22 GB。
- `RoboTwin/policy/pi0/src/openpi` 和 `packages/openpi-client`：Pi0 模型与输入处理源码。

重要：本机的 `LoRAHub-JAX/openpi_base` 是指向 `/home/huahungy/.cache/openpi/openpi-assets/checkpoints` 的绝对符号链接。仅执行普通 Git push/clone 不会上传其目标文件；复制时必须跟随符号链接，或在服务器上重新下载基座。

从本机复制完整工作区时可使用：

```bash
rsync -a --info=progress2 \
  --exclude 'LoRAHub-JAX/outputs/*/checkpoint' \
  --exclude 'LoRAHub-JAX/openpi_base' \
  /home/huahungy/RoboLoRAHub/ user@server:/path/to/RoboLoRAHub/

rsync -aL --info=progress2 \
  /home/huahungy/RoboLoRAHub/LoRAHub-JAX/openpi_base/pi0_base/ \
  user@server:/path/to/RoboLoRAHub/LoRAHub-JAX/openpi_base/pi0_base/
```

第一条命令明确排除了本机的绝对符号链接，第二条命令会在服务器创建真实的 `openpi_base/pi0_base` 目录并复制模型内容。

## 3. 创建 Conda 环境

在服务器进入仓库根目录：

```bash
cd /path/to/RoboLoRAHub
conda create -n lorahub-jax python=3.11 pip=24.2 -y
conda activate lorahub-jax

python -m pip install --upgrade pip==24.2 setuptools wheel
cd LoRAHub-JAX
python -m pip install -r requirements-server.txt
```

`requirements-server.txt` 中包含 `openpi-client` 的本地 editable 安装，因此安装命令必须在 `LoRAHub-JAX` 目录执行，并保持上一节的目录结构。

这里不需要安装 `RoboTwin/policy/pi0` 的完整 `pyproject.toml` 依赖。完整安装会额外拉入仿真、数据训练和可视化组件，显著增加环境体积与冲突概率；LoRAHub 启动脚本会直接加载相邻目录中的 OpenPI 源码。

## 4. 准备模型与 tokenizer 缓存

基座可以直接放在项目内：

```text
LoRAHub-JAX/openpi_base/pi0_base/params/
```

Pi0 第一次处理 prompt 时还会下载：

```text
gs://big_vision/paligemma_tokenizer.model
```

默认缓存目录为 `~/.cache/openpi`。服务器有高速数据盘时建议设置：

```bash
mkdir -p /data/$USER/openpi-cache
export OPENPI_DATA_HOME=/data/$USER/openpi-cache
```

如果服务器不能访问 Google Cloud Storage，可从本机复制已有文件：

```bash
rsync -a --info=progress2 \
  /home/huahungy/.cache/openpi/big_vision/paligemma_tokenizer.model \
  user@server:/data/user/openpi-cache/big_vision/paligemma_tokenizer.model
```

把示例中的 `/data/user` 改为服务器实际目录，并保证启动训练时 `OPENPI_DATA_HOME` 指向同一个 `openpi-cache`。

## 5. 安装后验证

### 5.1 核对文件

```bash
cd /path/to/RoboLoRAHub/LoRAHub-JAX
test -d openpi_base/pi0_base/params
test -d data/place_empty_cup_10
test "$(find lora_modules -maxdepth 1 -name '*.npz' | wc -l)" -ge 15
du -sh openpi_base/pi0_base lora_modules data/place_empty_cup_10
```

### 5.2 核对 Python、JAX 与 GPU

```bash
python - <<'PY'
import sys
import jax
import flax
import nevergrad
import orbax.checkpoint
import pandas
import pyarrow
import sentencepiece
import transformers

print("Python:", sys.version)
print("JAX:", jax.__version__)
print("Flax:", flax.__version__)
print("devices:", jax.devices())
assert any(device.platform == "gpu" for device in jax.devices()), "JAX 未识别到 GPU"
print("环境检查通过")
PY
```

如果这里仅显示 `CpuDevice`，先检查 `nvidia-smi`、NVIDIA 驱动和当前 shell 是否确实激活了 `lorahub-jax`，不要直接开始正式训练。

### 5.3 核对 OpenPI 导入和任务配置

```bash
PYTHONPATH="../RoboTwin/policy/pi0/src${PYTHONPATH:+:$PYTHONPATH}" python - <<'PY'
from openpi.training import config

cfg = config.get_config("place_empty_cup_pi0_lora_10ep")
print("model:", type(cfg.model).__name__)
print("action horizon:", cfg.model.action_horizon)
print("OpenPI 配置检查通过")
PY
```

## 6. 运行 smoke 验证

启动脚本会优先使用显式的 `PYTHON`，其次使用项目原有 Pi0 venv，最后使用当前 Conda 环境中的 `python`。服务器建议显式指定当前解释器：

```bash
cd /path/to/RoboLoRAHub/LoRAHub-JAX
conda activate lorahub-jax

export PYTHON="$(command -v python)"
export OPENPI_DATA_HOME=/data/$USER/openpi-cache
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export PI0_LORA_UPDATE_MODE=partial
export PI0_CACHE_CLEAR_INTERVAL=4

bash run_lorahub.sh config/place_empty_cup_10_4090_v2_smoke.json
```

首次运行会加载约 22 GB 基座、读取 LoRA、编译 Pi0 前向并可能下载 tokenizer，因此第一步明显慢于后续候选是正常现象。smoke 成功的最低标准：

- `jax.devices()` 为 GPU；
- 10 条 episode 各选择 4 个关键帧，共 40 个样本；
- 模型和 tokenizer 初始化完成；
- 至少产生一次有限的 loss，没有 OOM、SIGKILL 或 traceback；
- 输出目录生成日志、`sample_manifest.json` 和 loss 记录。

训练时可在另一个终端观察资源：

```bash
watch -n 2 nvidia-smi
tail -f outputs/place_empty_cup_10_4090_v2_smoke/logs/run_*.log
```

## 7. 启动正式训练

大显存服务器使用 800 次评估配置：

```bash
cd /path/to/RoboLoRAHub/LoRAHub-JAX
conda activate lorahub-jax
export PYTHON="$(command -v python)"
export OPENPI_DATA_HOME=/data/$USER/openpi-cache
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export PI0_LORA_UPDATE_MODE=partial
export PI0_CACHE_CLEAR_INTERVAL=4

bash run_lorahub.sh config/place_empty_cup_10_fast_large_vram.json
```

该配置保持当前方案：10 条数据、每条 4 个 `motion_stratified` 关键帧、50-step action horizon、`diffusion_steps=2`、V2 `action_focused` 搜索、`action_delta_sketch` 组合、CMA population 10、总预算 800。默认 `max_batch=8`；如果服务器仍然 OOM，先改为 4，而不是减少数据或 action horizon。

推荐在 tmux 中运行，避免 SSH 断开终止任务：

```bash
tmux new -s lorahub-jax
# 在 tmux 内执行上面的正式训练命令
# Ctrl-b d 退出会话；重新连接：tmux attach -t lorahub-jax
```

当前搜索过程不提供候选级断点续训。重新执行同一配置会以相同 seed 重新搜索，并覆盖同一任务目录中的最终权重与曲线；需要保留旧结果时，先复制或重命名对应 `outputs/<task_name>`。

## 8. 导出 checkpoint

正式训练成功并生成 `outputs/place_empty_cup_10_fast_large_vram/lorahub_weights.json` 后运行：

```bash
bash compose_checkpoint.sh config/place_empty_cup_10_fast_large_vram.json
```

导出结果位于：

```text
outputs/place_empty_cup_10_fast_large_vram/checkpoint/10000/
├── params/
└── assets/place_empty_cup_pi0/norm_stats.json
```

最后检查：

```bash
test -d outputs/place_empty_cup_10_fast_large_vram/checkpoint/10000/params
test -f outputs/place_empty_cup_10_fast_large_vram/checkpoint/10000/assets/place_empty_cup_pi0/norm_stats.json
du -sh outputs/place_empty_cup_10_fast_large_vram/checkpoint/10000
```

## 9. 常见问题

### JAX 没有识别 GPU

确认安装的是 `jax[cuda12]==0.5.0`，且 `nvidia-smi` 正常。不要设置 `JAX_PLATFORMS=cpu`；该变量只在 `compose.py` 导出阶段由程序设置。

### 找不到 `openpi` 或 `openpi_client`

确认仓库目录结构未改变，并在 `LoRAHub-JAX` 目录重新执行：

```bash
python -m pip install -e ../RoboTwin/policy/pi0/packages/openpi-client
```

`run.py` 会自动把 `../RoboTwin/policy/pi0/src` 加入模块搜索路径。

### 找不到基座 checkpoint

确认下面目录真实存在，不是指向本机旧路径的失效符号链接：

```bash
readlink -f openpi_base
test -f openpi_base/pi0_base/params/_METADATA
```

### tokenizer 下载失败

设置 `OPENPI_DATA_HOME`，并把 `paligemma_tokenizer.model` 预先复制到：

```text
$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model
```

### 显存不足或显存持续增长

- 保持 `PI0_LORA_UPDATE_MODE=partial`；
- 保持 `PI0_CACHE_CLEAR_INTERVAL=4`；
- 将配置中的 `algorithm.max_batch` 从 8 降到 4 或 2；
- 保持 `XLA_PYTHON_CLIENT_PREALLOCATE=false`；
- 用 `nvidia-smi` 排除其他进程占用。

不要首先修改关键帧数、action horizon、delta loss 或 V2 action-focused 搜索空间，这些属于当前训练目标的一部分。
