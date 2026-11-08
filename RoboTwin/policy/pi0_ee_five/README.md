# 五任务联合 π0 末端 LoRA 微调与评测

服务器多卡入口见 [SERVER_TRAINING.md](../pi0/SERVER_TRAINING.md)：预设全局 batch=32、20,000 步，每 5,000 步保存并保留，支持 `--gpu 0,1,2,3 --fsdp-devices 4`。本文件下方仍说明本地默认配置。

本目录扩展现有 π0 训练与 RoboTwin 评测功能，不修改原 `policy/pi0/`、仿真环境或源 LeRobot 数据。现在的默认流程是 **五个任务混合训练一个模型，再用同一个 checkpoint 逐任务评测**。

训练任务：`adjust_bottle`、`click_alarmclock`、`place_can_basket`、`place_container_plate`、`rotate_qrcode`。使用全部 250 条示范、39,505 帧。每个数据集先用自己的 task_index 查出语言指令，再合并；50 步动作窗口始终限制在原 episode 内，尾部重复最后动作。无需复制图片或合并 Parquet。

## 启动

```bash
cd /home/huahungy/RoboLoRAHub/RoboTwin/policy/pi0_ee_five
bash run.sh all --run-name ee5_joint_v1 --gpu 0 --batch-size 1 --train-steps 100000
```

自动顺序：GPU 检查 → 全量数值校验、共享归一化、真实混合数据加载检查 → 一次联合训练 → 五任务分别评测 100 个场景。默认保存评测视频；加 `--no-video` 可关闭。

中断后使用同样参数恢复：

```bash
bash run.sh all --run-name ee5_joint_v1 --gpu 0 --batch-size 1 --train-steps 100000 --resume
```

不要复用原先分别训练的 `ee5_v1` 目录。新旧配置不兼容时脚本会拒绝混用。旧单任务实验保留不变，`eval` 仍按其保存配置加载。

也可分开执行：

```bash
bash run.sh prepare
bash run.sh train --run-name ee5_joint_v1 --gpu 0 --batch-size 1 --train-steps 100000
bash run.sh eval --run-name ee5_joint_v1 --gpu 0
# 只评测某个任务，仍使用联合 checkpoint：
bash run.sh eval --run-name ee5_joint_v1 --gpu 0 --tasks adjust_bottle --eval-episodes 10
```

`--tasks` 仅用于选择评测任务；联合训练始终使用五个任务。相同评测目录已存在时需 `--resume`，或换 seed/episode 数。恢复会跳过完整训练和完整评测；未完成的评测先归档，再重新运行该任务。

## 训练与归一化

| 项目 | 设置 |
|---|---|
| 输入 | 三路 RGB、原任务语言、16 维绝对末端状态补零到 32 维 |
| 输出 | 50 步 × 32 维；有效动作是前 16 维 |
| 16 维顺序 | 左 xyz + wxyz 四元数 + 夹爪，右 xyz + wxyz 四元数 + 夹爪 |
| 采样 | 五任务全部帧拼接后随机打乱，按帧均匀采样；不是按任务等概率 |
| 归一化 | 五任务共享的 state/action 逐维 z-score，使用总体均值与标准差 |
| 动作统计 | 严格计入每一帧对应的 50 步窗口及 episode 尾部重复权重 |
| 图像 | π0 图像变换、224×224、像素映射 [-1,1]；不混入末端统计 |
| 语言 | π0 tokenizer；不做数值 z-score |
| 微调 | PaliGemma 与 action expert LoRA、控制投射层；冻结 LLM 基座和视觉骨干 |
| batch / EMA | batch=1，无梯度累积；EMA 关闭 |
| 默认预算 | 整个混合数据集共 100,000 optimizer steps，约 2.53 次全量遍历 |
| 学习率 | cosine，peak 2.5e-5，warmup 500 steps |
| checkpoint | 每 2,000 步及最终步保存，原 Orbax 策略仅保留最近一个 |
| 评测 | 同一个 checkpoint，逐任务 100 个原生专家可解场景，unseen 指令 |

100,000 步保留原先五任务各 20,000 步的总更新预算，不是 100,000 轮，也不保证收敛。batch=1 下，20,000 步约为混合数据集的 0.51 轮。任务样本比例由帧数决定：7,044 / 4,250 / 12,536 / 7,927 / 7,748。训练效果应以各任务仿真成功率判断。

归一化在加载时应用，不改写原始数据。计算共享统计时同时包含任务内及任务间方差，不直接平均各任务的标准差。保存源数据指纹，重跑准备阶段会校验一致性。checkpoint 内携带同一份共享统计；五个任务推理都读取它，预测动作反归一化后再单位化四元数、裁剪夹爪并调用 `take_action(..., action_type='ee')`。使用 get_left/right_ee_pose 的世界坐标参考点，不添加 TCP 偏移，不使用关节 DeltaActions。

## 输出与检查

```text
assets/joint_pi0_ee_five_lora/robotwin_five_pi0_ee/
  norm_stats.json
  provenance.json
reports/joint_preparation.json
runs/ee5_joint_v1/
  run_config.json
  logs/
  checkpoints/joint_pi0_ee_five_lora/model/<step>/
    params/ train_state/ assets/robotwin_five_pi0_ee/norm_stats.json
  status/joint_trained.json
  evaluations/<task>/step_<N>_seed_<S>_n100/
    evaluation_config.json episodes.json result.json episode*.mp4
  evaluation_summary.json
```

只保存一个联合模型，但 checkpoint 包含完整参数和优化器，不只是 LoRA 小文件。图片缓存和视频另占空间。旧单任务统计文件保留，仅用于计算/核验共享统计，不用于联合模型的推理。

`train_compat.py` 仅在新启动进程内修复当前 Optax/beartype 的 `ArrayTree` 前向类型引用错误，保留 TrainState 的字段、优化器、梯度和 checkpoint 布局。未修改原依赖文件。

```bash
JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 \
  ../pi0/.venv/bin/python -m unittest discover -s tests -v
```

检查覆盖混合调度、共享统计、语言映射、窗口末尾、模型/统计复用、末端控制和原训练器优化器初始化。CPU 检查不代表完整 GPU 训练或仿真已通过；有限步 GPU 检查也不代表收敛和成功率已验证。
