# 服务器：五任务联合 π0 末端训练

本入口使用 RoboTwin 的 π0/JAX LoRA：五个任务共同训练一个模型，同一 checkpoint 分别评测五个任务。代码仓库：<https://github.com/HuaHungy/RoboLoRA>。

## 1. 目录结构

仓库可以克隆到任意位置，不需要沿用本机的 `/home/huahungy`。以下以 `/workspace/RoboLoRA` 为例：

```text
RoboLoRA/
├── data_clustering/
│   ├── action_clustering/ configs/ scripts/ tests/ docs/
│   └── data/lerobot/                         # 手工搬运，Git 不上传
│       ├── adjust_bottle_pi0_ee/
│       ├── click_alarmclock_pi0_ee/
│       ├── place_can_basket_pi0_ee/
│       ├── place_container_plate_pi0_ee/
│       └── rotate_qrcode_pi0_ee/
├── transfer/openpi/                          # 手工搬运，Git 不上传
│   ├── big_vision/paligemma_tokenizer.model
│   └── openpi-assets/checkpoints/pi0_base/params/
├── RoboTwin/
│   ├── assets/                              # 仅仿真评测需要，手工搬运
│   ├── task_config/                         # 已上传，含 demo_aloha.yml
│   ├── envs/ script/ description/           # 原生任务与评测代码
│   └── policy/
│       ├── pi0/
│       │   ├── SERVER_TRAINING.md            # 本文
│       │   ├── server_train_ee.sh            # 服务器统一启动入口
│       │   ├── setup_server_ee.sh            # 依赖安装入口
│       │   ├── pyproject.toml / uv.lock
│       │   └── .venv/                       # 在服务器重建，Git 不上传
│       └── pi0_ee_five/
│           ├── config.server.json           # batch=32 / steps=20000 / save=5000
│           ├── pipeline.py / train_loop.py / joint_data.py / ...
│           ├── assets/                      # 自动生成共享归一化统计
│           └── runs/ee5_joint_b32_s20000/    # 训练后自动生成
└── README.md
```

`data_clustering` 和 `RoboTwin` 在同一仓库中作为普通目录保存，不是需要另外初始化的 Git 子模块。保留原项目许可证。原始数据、训练产物、缓存、第三方编译环境不放入 Git。

## 2. 需要单独搬运的内容

| 内容 | 本机来源 | 服务器仓库内目标 | 必需性 / 约大小 |
|---|---|---|---|
| 五个 LeRobot 数据集 | `/home/huahungy/RoboLoRAHub/data_clustering/data/lerobot/<task>_pi0_ee/` | `data_clustering/data/lerobot/<task>_pi0_ee/` | 训练必需，合计约 4.1 GB |
| π0 基座参数 | `/home/huahungy/.cache/openpi/openpi-assets/checkpoints/pi0_base/params/` | `transfer/openpi/openpi-assets/checkpoints/pi0_base/params/` | 训练必需，约 12 GiB |
| PaliGemma tokenizer | `/home/huahungy/.cache/openpi/big_vision/paligemma_tokenizer.model` | `transfer/openpi/big_vision/paligemma_tokenizer.model` | 训练、推理必需，约 4.1 MB |
| RoboTwin 仿真资源 | `/home/huahungy/RoboLoRAHub/RoboTwin/assets/` | `RoboTwin/assets/` | 仅评测必需，约 16 GB；包含 objects、embodiments、background_texture |
| 历史联合 checkpoint / 运行配置 | `RoboTwin/policy/pi0_ee_five/runs/<run>/` | 同一相对位置 | 仅继续已有实验需要；从基座训练不需要 |
| 全量聚类数据、结果和视觉权重 | `data_clustering/data/`、`runs/`、`reports/`；`~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth` | 对应目录，视觉权重可放服务器相应 Torch 缓存 | 仅重跑旧聚类实验需要，本次五任务训练不需要 |

每个 LeRobot 数据集必须完整复制 `data/` 和 `meta/`，包括 `meta/info.json`、`episodes.jsonl`、`tasks.jsonl`、`robotwin_ee_conversion.json` 等已有文件。图像 PNG 字节已经嵌在 Parquet 中，不需要重新提取图片，不需要搬原始 HDF5，也不需要先做聚类。

基座 `params/` 必须完整复制，包含隐藏/下划线元数据、`d/`、`ocdbt.process_0/`、`manifest.ocdbt` 等；不要只复制单个文件。

**不要直接搬 `.venv`**：它包含本机绝对路径和编译库，应在服务器重新安装。无需搬 HF Arrow 缓存、JAX 编译缓存、旧日志或旧归一化统计。启动会重新检查数据并生成统一统计；训练时图片/语言仍保留。

当前单个 batch=1 验证 checkpoint 约 5.3 GiB；服务器最终保留四个完整 checkpoint，建议训练目录至少预留约 80 GB 空闲空间（数据、环境、缓存、临时保存另有开销）；评测视频额外占用空间。

## 3. 克隆与搬运示例

服务器：

```bash
git clone https://github.com/HuaHungy/RoboLoRA.git /workspace/RoboLoRA
cd /workspace/RoboLoRA
mkdir -p data_clustering/data/lerobot \
  transfer/openpi/big_vision \
  transfer/openpi/openpi-assets/checkpoints/pi0_base/params
```

以下命令在**本机**执行，先将 `SERVER` 和 `TARGET` 替换为你的服务器 SSH 地址和已克隆目录；非标准 SSH 端口可给 rsync 加 `-e 'ssh -p 端口'`：

```bash
SERVER=user@server
TARGET=/workspace/RoboLoRA
for task in adjust_bottle click_alarmclock place_can_basket place_container_plate rotate_qrcode; do
  rsync -a --info=progress2 \
    "/home/huahungy/RoboLoRAHub/data_clustering/data/lerobot/${task}_pi0_ee" \
    "$SERVER:$TARGET/data_clustering/data/lerobot/"
done
rsync -a --info=progress2 \
  /home/huahungy/.cache/openpi/openpi-assets/checkpoints/pi0_base/params/ \
  "$SERVER:$TARGET/transfer/openpi/openpi-assets/checkpoints/pi0_base/params/"
rsync -a --info=progress2 \
  /home/huahungy/.cache/openpi/big_vision/paligemma_tokenizer.model \
  "$SERVER:$TARGET/transfer/openpi/big_vision/"
# 仅需要在服务器进行仿真评测时：
rsync -a --info=progress2 /home/huahungy/RoboLoRAHub/RoboTwin/assets/ \
  "$SERVER:$TARGET/RoboTwin/assets/"
```

也可以将 `data_clustering/data/lerobot` 或基座参数目录软链接到共享存储；或者启动时用 `--dataset-root /shared/lerobot --pretrained-params /shared/pi0_base/params` 指定位置。tokenizer 缓存由 `OPENPI_DATA_HOME` 指定，其下仍须有 `big_vision/paligemma_tokenizer.model`。恢复训练时保持路径和训练参数与首次运行一致。

## 4. 重建环境

在服务器上执行（Linux、NVIDIA GPU、Python 3.11；需要访问依赖镜像和 GitHub）：

```bash
cd /workspace/RoboLoRA/RoboTwin/policy/pi0
python3 -m pip install uv
bash setup_server_ee.sh train
```

安装沿用仓库 `uv.lock`，包含 JAX 0.5.0 CUDA 12、Flax 0.10.2、固定版本 LeRobot 和现有数据加载兼容层。脚本不会下载训练数据/基座。只运行训练不要求安装 cuRobo 或搬运仿真资源。不要用 `torchrun` 启动：这是单进程、多 GPU 的 JAX 训练。

如需评测，服务器还需可用的 NVIDIA Vulkan/EGL 渲染环境、CUDA Toolkit 12.x（含 `nvcc`）、C++ 编译工具和系统 `ffmpeg`。然后：

```bash
bash setup_server_ee.sh eval
```

评测安装额外构建本机使用的 cuRobo revision `cca894de9ec74e77a0a4071319c81958991a9108`，不会上传已有机器的 CUDA 二进制。无外网时，应提前通过组织内部镜像/离线依赖包准备这些依赖；Git 仓库本身不包含 Python 包或 CUDA 运行环境。

## 5. 多卡训练：主要入口

四卡例子，**显卡选择只需改 `--gpu` 后面的列表**：

```bash
cd /workspace/RoboLoRA/RoboTwin/policy/pi0
bash server_train_ee.sh train --gpu 0,1,2,3 --fsdp-devices 4
```

两张指定卡：

```bash
bash server_train_ee.sh train --gpu 2,3 --fsdp-devices 2
```

默认参数已经写入 `config.server.json`，不需要 IDE：

- 五任务共同训练一个模型，共 250 条轨迹、39,505 帧。
- **全局 batch_size=32，num_train_steps=20000**；四卡每卡 8，双卡每卡 16。
- `--fsdp-devices` 是模型参数分片组大小，必须整除所选卡数；设 1 则不分片。不同配置的显存需求应以实际服务器验证为准，本机只验证过单卡 batch=1。
- 全局 batch 必须整除所选 GPU 数量。这里支持**单台服务器内多卡**，不包含跨节点启动。
- 总计 20,000 次优化更新，约对应 16.2 次混合数据集遍历，不是每任务 20,000 步。
- 保持学习率 peak=2.5e-5、warmup=500，不自动按 batch 放大学习率。
- **在完成 5,000 / 10,000 / 15,000 / 20,000 次更新时保存，并保留这四份 checkpoint**。已修正上游循环的 checkpoint 标签与真实更新次数相差 1 的问题。
- 训练前自动计算共享 state/action z-score（动作统计含 50 步窗口尾部重复权重），推理读取相同 checkpoint 内统计。

如果调度器已经设置 `CUDA_VISIBLE_DEVICES`，不传 `--gpu` 会继承它；显式传入则覆盖。因此容器或调度平台内的设备编号以该运行环境可见的编号为准。

先验证搬运的数据、归一化和 batch=32 的 CPU 加载：

```bash
bash server_train_ee.sh prepare
bash server_train_ee.sh check --gpu 0,1,2,3
```

可以用独立目录验证服务器 GPU 是否能承载配置（不会评测）：

```bash
bash server_train_ee.sh train --gpu 0,1,2,3 --fsdp-devices 4 \
  --run-name ee5_server_smoke --train-steps 2
```

这次短验证和正式训练目录不同。短验证不代表收敛；如果显存不足，可先增加卡数/分片，或用命令行 `--batch-size` 调整实际全局 batch，不能把梯度累积误认为当前已有功能。

SSH 断开后继续运行：

```bash
nohup bash server_train_ee.sh train --gpu 0,1,2,3 --fsdp-devices 4 \
  > ee5_server_console.log 2>&1 &
tail -f ee5_server_console.log
```

## 6. 恢复、评测与输出

断点恢复必须保留整个 run 目录，并使用同样的 GPU/训练参数：

```bash
bash server_train_ee.sh train --gpu 0,1,2,3 --fsdp-devices 4 --resume
```

如尚未写出第一个 checkpoint，则从基座重新开始；已保存后恢复优化器及模型状态。学习器沿用原加载器的随机打乱机制，数据采样器位置没有随 checkpoint 保存，所以恢复不保证逐样本完全复现未中断训练。

训练完后逐任务评测，默认加载 20,000 步的同一个联合模型：

```bash
bash server_train_ee.sh eval --gpu 0
# 选择一个保留下来的 checkpoint：
bash server_train_ee.sh eval --gpu 0 --checkpoint-step 10000
```

若服务器已具备完整仿真环境，也可以一次执行训练与评测：

```bash
bash server_train_ee.sh all --gpu 0,1,2,3 --fsdp-devices 4
```

`all` 的训练使用全部所选卡；后续评测仅使用列表第一张卡，五任务顺序执行，各 100 个专家可解场景。可用 `--eval-episodes 10`、`--no-video` 改变评测预算，或用 `eval --tasks adjust_bottle` 只测一个任务。

```text
RoboTwin/policy/pi0_ee_five/runs/ee5_joint_b32_s20000/
├── run_config.json
├── logs/
├── checkpoints/joint_pi0_ee_five_lora/model/
│   ├── 5000/
│   ├── 10000/
│   ├── 15000/
│   └── 20000/
│       ├── params/
│       ├── train_state/
│       └── assets/robotwin_five_pi0_ee/norm_stats.json
├── status/joint_trained.json
├── evaluations/<task>/step_<N>_seed_0_n100/
└── evaluation_summary.json
```

每份 checkpoint 是可恢复的完整模型与优化器，不只是 LoRA adapter。`result.json` 含各任务成功率，汇总在 `evaluation_summary.json`。这些输出和归一化缓存都被 Git 忽略，之后备份时单独搬运。
