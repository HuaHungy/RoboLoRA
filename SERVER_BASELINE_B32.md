# 服务器运行 Pi0 Baseline100（全局 Batch Size 32）

本文只针对当前准备跑通的标准 Pi0 baseline：五个任务各取 20 条轨迹，合并为一个 100 条轨迹的数据集；先训练一个联合模型，再用同一个 5,000 步 checkpoint 逐个评测五个任务。

任务如下：

- `adjust_bottle`
- `click_alarmclock`
- `place_can_basket`
- `place_container_plate`
- `rotate_qrcode`

固定实验参数：

| 参数 | 值 |
|---|---:|
| 训练数据 | 100 条轨迹，五任务各 20 条 |
| 总帧数 | 15,789 |
| 动作表示 | 双臂末端世界坐标，`xyz + wxyz + gripper`，原始 16 维、模型补齐到 32 维 |
| 全局 batch size | 32 |
| 优化步数 | 5,000 |
| 学习率 | `1e-5` |
| warmup | 500 步 |
| action horizon | 50 |
| checkpoint 间隔 | 1,000 步；当前配置最终保留最新的完整 checkpoint |
| 评测 | 五任务各 20 次，seed 0，保存视频 |
| 训练内容 | 标准 Pi0 LoRA；不使用簇专家、MLP Router 或残差 LoRA |

## 1. 推荐的服务器条件

训练和评测建议使用：

- Linux x86-64。
- Python 3.11。
- NVIDIA 驱动支持 CUDA 12。
- 推荐 4 张 GPU：全局 batch 32 时，每张卡处理 batch 8；也可以使用 2、8 张卡，但全局 batch 32 必须能被 GPU 数量整除。
- 多卡训练由单个 JAX 进程负责，不使用 `torchrun`。
- 建议至少 64GB 系统内存。
- 建议为代码、Python 环境、数据、基础权重、仿真资源、checkpoint 和视频预留至少 80GB 可用空间。

显存能否容纳 batch 32 与 GPU 型号、卡数和 FSDP 分片数有关。正式训练前必须执行本文的 `prepare`、`check` 和两步 smoke test。

仅训练需要 NVIDIA 驱动和 JAX CUDA 运行环境。完整的“训练并评测”还需要：

- CUDA Toolkit 12.x 和 `nvcc`，用于构建 cuRobo。
- C/C++ 编译工具、`ninja`。
- NVIDIA Vulkan/EGL 无头渲染环境。
- 系统命令 `ffmpeg`，用于保存 MP4 视频。
- 能够安装锁定的 Python 依赖并克隆指定版本 cuRobo；无外网服务器需要提前准备内部镜像或离线包。

基础检查：

```bash
nvidia-smi
nvcc --version
ffmpeg -version
python3 --version
```

## 2. 服务器目录结构

仓库可以放在任意位置。下文以 `/workspace/RoboLoRA` 为例：

```text
/workspace/RoboLoRA/
├── SERVER_BASELINE_B32.md
├── data_clustering/
│   └── data/merged/
│       └── robotwin_five_pi0_ee_subset20_seed42/
│           ├── data/chunk-000/
│           │   ├── episode_000000.parquet
│           │   ├── ...
│           │   └── episode_000099.parquet
│           └── meta/
│               ├── info.json
│               ├── episodes.jsonl
│               ├── episodes_stats.jsonl
│               ├── tasks.jsonl
│               ├── merge_manifest.json
│               ├── source_episodes.jsonl
│               └── robotwin_ee_conversion.json
├── transfer/openpi/
│   ├── big_vision/paligemma_tokenizer.model
│   └── openpi-assets/checkpoints/pi0_base/params/
│       ├── _CHECKPOINT_METADATA
│       ├── _METADATA
│       ├── _sharding
│       ├── d/
│       ├── manifest.ocdbt
│       └── ocdbt.process_0/
└── RoboTwin/
    ├── assets/                         # 仿真评测必需
    └── policy/
        ├── pi0/.venv/                 # 在服务器重新创建
        └── pi0_ee_five/
            ├── baseline100_config.json
            ├── run_baseline100_b32.sh
            ├── assets/                # 服务器自动生成归一化统计
            └── runs/                  # 服务器自动生成训练与评测产物
```

## 3. 必须单独搬运的内容

Git 仓库不包含数据、基础权重、归一化文件、checkpoint、仿真资源或评测结果。需要搬运：

| 内容 | 本机来源 | 服务器目标 | 大小 |
|---|---|---|---:|
| Baseline100 合并数据集 | `/home/huahungy/RoboLoRAHub/data_clustering/data/merged/robotwin_five_pi0_ee_subset20_seed42/` | `data_clustering/data/merged/robotwin_five_pi0_ee_subset20_seed42/` | 约 1.6GB |
| Pi0 基础参数 | `/home/huahungy/.cache/openpi/openpi-assets/checkpoints/pi0_base/params/` | `transfer/openpi/openpi-assets/checkpoints/pi0_base/params/` | 约 12GB |
| PaliGemma tokenizer | `/home/huahungy/.cache/openpi/big_vision/paligemma_tokenizer.model` | `transfer/openpi/big_vision/paligemma_tokenizer.model` | 约 4.1MB |
| RoboTwin 仿真资源 | `/home/huahungy/RoboLoRAHub/RoboTwin/assets/` | `RoboTwin/assets/` | 约 16GB |

不需要搬运以下内容：

- 本机 `.venv`、JAX 编译缓存或 Hugging Face Arrow 缓存。
- 本机 baseline 的归一化文件；服务器会使用这100条轨迹重新计算。
- 本机 batch-4 checkpoint、日志和评测视频。
- 七簇 LoRA、Router checkpoint 或聚类数据。
- 原始 HDF5 数据。图像已经作为压缩字节保存在 Parquet 的三个相机列中。

在本机执行以下搬运示例；先修改服务器地址和目标目录：

```bash
SERVER=user@server
TARGET=/workspace/RoboLoRA

rsync -a --info=progress2 \
  /home/huahungy/RoboLoRAHub/data_clustering/data/merged/robotwin_five_pi0_ee_subset20_seed42/ \
  "$SERVER:$TARGET/data_clustering/data/merged/robotwin_five_pi0_ee_subset20_seed42/"

rsync -a --info=progress2 \
  /home/huahungy/.cache/openpi/openpi-assets/checkpoints/pi0_base/params/ \
  "$SERVER:$TARGET/transfer/openpi/openpi-assets/checkpoints/pi0_base/params/"

rsync -a --info=progress2 \
  /home/huahungy/.cache/openpi/big_vision/paligemma_tokenizer.model \
  "$SERVER:$TARGET/transfer/openpi/big_vision/"

rsync -a --info=progress2 \
  /home/huahungy/RoboLoRAHub/RoboTwin/assets/ \
  "$SERVER:$TARGET/RoboTwin/assets/"
```

## 4. 克隆代码与安装环境

服务器执行：

```bash
git clone https://github.com/HuaHungy/RoboLoRA.git /workspace/RoboLoRA
cd /workspace/RoboLoRA
git checkout main
```

如果仓库已经存在：

```bash
cd /workspace/RoboLoRA
git pull origin main
```

创建目标目录：

```bash
mkdir -p \
  data_clustering/data/merged/robotwin_five_pi0_ee_subset20_seed42 \
  transfer/openpi/big_vision \
  transfer/openpi/openpi-assets/checkpoints/pi0_base/params
```

不要从本机复制 `.venv`。在服务器根据锁文件重新创建环境。因为一键脚本会直接进入评测，所以这里安装包含 cuRobo 的评测环境：

```bash
cd /workspace/RoboLoRA/RoboTwin/policy/pi0
python3 -m pip install --user uv
export PATH="$HOME/.local/bin:$PATH"
bash setup_server_ee.sh eval
```

该命令会执行锁定依赖安装，并构建指定 revision 的 cuRobo。只安装训练依赖可使用 `bash setup_server_ee.sh train`，但随后运行完整的一键训练评测脚本会在开始时检查评测依赖并报错，因此本实验应直接准备 `eval` 环境。

## 5. 数据和权重完整性检查

回到仓库根目录并设置 tokenizer 缓存位置：

```bash
cd /workspace/RoboLoRA
export OPENPI_DATA_HOME="$PWD/transfer/openpi"
```

确认数据集恰好有100个 Parquet：

```bash
find data_clustering/data/merged/robotwin_five_pi0_ee_subset20_seed42/data \
  -name 'episode_*.parquet' | wc -l
```

预期输出：

```text
100
```

检查元数据：

```bash
RoboTwin/policy/pi0/.venv/bin/python - <<'PY'
import json
from pathlib import Path

root = Path('data_clustering/data/merged/robotwin_five_pi0_ee_subset20_seed42')
info = json.loads((root / 'meta/info.json').read_text())
print({key: info[key] for key in ('total_episodes', 'total_frames', 'total_tasks')})
PY
```

预期结果是100条轨迹、15,789帧、100条语言任务描述：

```text
{'total_episodes': 100, 'total_frames': 15789, 'total_tasks': 100}
```

检查基础权重和 tokenizer：

```bash
test -f transfer/openpi/big_vision/paligemma_tokenizer.model
test -f transfer/openpi/openpi-assets/checkpoints/pi0_base/params/_CHECKPOINT_METADATA
test -f transfer/openpi/openpi-assets/checkpoints/pi0_base/params/manifest.ocdbt
test -d transfer/openpi/openpi-assets/checkpoints/pi0_base/params/d
test -d transfer/openpi/openpi-assets/checkpoints/pi0_base/params/ocdbt.process_0
echo 'data assets present'
```

## 6. 正式训练前检查

以下命令均在仓库根目录执行：

```bash
cd /workspace/RoboLoRA
export OPENPI_DATA_HOME="$PWD/transfer/openpi"
PI0_PARAMS="$PWD/transfer/openpi/openpi-assets/checkpoints/pi0_base/params"
```

先检查并计算100条数据对应的共享32维归一化统计：

```bash
bash RoboTwin/policy/pi0_ee_five/run.sh prepare \
  --config RoboTwin/policy/pi0_ee_five/baseline100_config.json \
  --batch-size 32 \
  --fsdp-devices 4 \
  --pretrained-params "$PI0_PARAMS"
```

归一化结果会生成到：

```text
RoboTwin/policy/pi0_ee_five/assets/
└── joint_pi0_ee_five_lora/
    └── robotwin_five_pi0_ee_subset20_seed42/
        ├── norm_stats.json
        └── provenance.json
```

然后检查4张GPU是否同时被JAX和PyTorch识别，以及训练数据依赖是否齐全：

```bash
bash RoboTwin/policy/pi0_ee_five/run.sh check \
  --config RoboTwin/policy/pi0_ee_five/baseline100_config.json \
  --gpu 0,1,2,3 \
  --batch-size 32 \
  --fsdp-devices 4 \
  --pretrained-params "$PI0_PARAMS"
```

单独检查仿真评测依赖：

```bash
RoboTwin/policy/pi0/.venv/bin/python -c \
  'import sapien, mplib, curobo; print("evaluation imports PASS")'
command -v nvcc
command -v ffmpeg
```

最后用独立目录进行两步训练 smoke test，避免污染正式实验：

```bash
bash RoboTwin/policy/pi0_ee_five/run.sh train \
  --config RoboTwin/policy/pi0_ee_five/baseline100_config.json \
  --run-name pi0_baseline100_b32_smoke \
  --gpu 0,1,2,3 \
  --batch-size 32 \
  --fsdp-devices 4 \
  --train-steps 2 \
  --save-interval 1 \
  --pretrained-params "$PI0_PARAMS"
```

smoke test 能完成两个优化步骤并保存 checkpoint 后，再启动正式实验。

## 7. 启动完整训练和评测

四卡命令：

```bash
cd /workspace/RoboLoRA

nohup env \
  OPENPI_DATA_HOME="$PWD/transfer/openpi" \
  GPU_IDS=0,1,2,3 \
  FSDP_DEVICES=4 \
  bash RoboTwin/policy/pi0_ee_five/run_baseline100_b32.sh \
  --pretrained-params "$PWD/transfer/openpi/openpi-assets/checkpoints/pi0_base/params" \
  > pi0_baseline100_b32_s5000.log 2>&1 &

echo $!
```

这里的 batch 32 是全局 batch。GPU 与每卡样本数示例：

| GPU | `GPU_IDS` | `FSDP_DEVICES` | 每卡 batch |
|---:|---|---:|---:|
| 2 | `0,1` | `2` | 16 |
| 4 | `0,1,2,3` | `4` | 8 |
| 8 | `0,1,2,3,4,5,6,7` | `8` | 4 |

`FSDP_DEVICES` 必须整除所选GPU数量。服务器调度系统已经屏蔽设备时，`GPU_IDS` 使用容器内可见的编号。不要在脚本外再套 `torchrun`。

查看主日志：

```bash
tail -f pi0_baseline100_b32_s5000.log
```

查看GPU：

```bash
watch -n 2 nvidia-smi
```

查看最新内部日志：

```bash
tail -f "$(find RoboTwin/policy/pi0_ee_five/runs/pi0_baseline100_b32_s5000/logs \
  -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)"
```

按 `Ctrl+C` 只会退出 `tail` 或 `watch`，不会停止 `nohup` 后台进程。

## 8. 续训

如果训练中断，保持相同数据、基础权重、batch、FSDP设置和运行目录，增加 `--resume`：

```bash
cd /workspace/RoboLoRA

nohup env \
  OPENPI_DATA_HOME="$PWD/transfer/openpi" \
  GPU_IDS=0,1,2,3 \
  FSDP_DEVICES=4 \
  bash RoboTwin/policy/pi0_ee_five/run_baseline100_b32.sh \
  --pretrained-params "$PWD/transfer/openpi/openpi-assets/checkpoints/pi0_base/params" \
  --resume \
  >> pi0_baseline100_b32_s5000.log 2>&1 &
```

不要用同一个 `RUN_NAME` 改 batch size、训练步数、FSDP大小或数据位置，否则运行配置检查会拒绝启动。确需改变实验配置时使用新的名称：

```bash
RUN_NAME=pi0_baseline100_b32_s5000_v2 GPU_IDS=0,1,2,3 FSDP_DEVICES=4 \
  bash RoboTwin/policy/pi0_ee_five/run_baseline100_b32.sh \
  --pretrained-params "$PWD/transfer/openpi/openpi-assets/checkpoints/pi0_base/params"
```

## 9. 输出位置

正式实验目录：

```text
RoboTwin/policy/pi0_ee_five/runs/pi0_baseline100_b32_s5000/
├── run_config.json
├── logs/
├── status/
├── checkpoints/
│   └── joint_pi0_ee_five_lora/model/5000/
│       ├── params/
│       ├── train_state/
│       └── assets/robotwin_five_pi0_ee_subset20_seed42/norm_stats.json
├── evaluations/
│   ├── adjust_bottle/step_5000_seed_0_n20/
│   ├── click_alarmclock/step_5000_seed_0_n20/
│   ├── place_can_basket/step_5000_seed_0_n20/
│   ├── place_container_plate/step_5000_seed_0_n20/
│   └── rotate_qrcode/step_5000_seed_0_n20/
└── evaluation_summary.json
```

每个评测目录包含 `result.json` 和保存的视频。五个任务按顺序评测，评测阶段只使用 `GPU_IDS` 中的第一张卡。服务器没有历史 Router 运行目录时，只会跳过最终 baseline-vs-router 汇总，不影响训练、逐任务评测或视频保存。

## 10. 常见问题

### `FileNotFoundError: Merged baseline dataset metadata is missing`

数据目录层级不正确。确认 `info.json` 位于：

```text
data_clustering/data/merged/robotwin_five_pi0_ee_subset20_seed42/meta/info.json
```

### tokenizer 尝试联网或找不到文件

确认启动命令设置了：

```bash
export OPENPI_DATA_HOME=/workspace/RoboLoRA/transfer/openpi
```

并存在：

```text
transfer/openpi/big_vision/paligemma_tokenizer.model
```

### JAX 只看到一张卡

检查调度器的 `CUDA_VISIBLE_DEVICES`、容器GPU挂载和 `GPU_IDS`。脚本输出中应列出全部目标 JAX GPU。

### CUDA OOM

优先确认4张卡都被使用，并将 `FSDP_DEVICES` 设为4。如果仍然不足，可以增加GPU数量并保持全局 batch 32。降低 batch 后就不再是本次 batch-32 对比实验，应使用新的运行名称记录。

### 训练结束后评测报 `curobo`、`nvcc`、Vulkan/EGL 或 `ffmpeg` 错误

说明训练依赖已经可用，但评测环境不完整。重新执行：

```bash
cd /workspace/RoboLoRA/RoboTwin/policy/pi0
bash setup_server_ee.sh eval
```

并检查服务器的 NVIDIA 无头渲染与 Vulkan 配置。保留训练目录后，可以只运行 `eval` 阶段，无需重新训练。

## 11. 启动前清单

- [ ] GitHub `main` 已更新到最新提交。
- [ ] Python 3.11 和 `uv` 可用。
- [ ] `setup_server_ee.sh eval` 成功完成。
- [ ] JAX 和 PyTorch 都能识别目标GPU。
- [ ] 100个 Parquet 文件完整，`info.json` 显示100条、15,789帧。
- [ ] Pi0 `params/` 目录完整，约12GB。
- [ ] PaliGemma tokenizer 存在，并设置 `OPENPI_DATA_HOME`。
- [ ] RoboTwin `assets/` 已完整搬运，约16GB。
- [ ] `prepare` 成功生成100条数据对应的共享归一化。
- [ ] 两步 batch-32 smoke test 成功。
- [ ] 至少预留约80GB可用空间。
- [ ] 使用 `nohup` 或服务器作业调度器启动正式实验。
