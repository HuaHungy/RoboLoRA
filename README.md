# RoboLoRA

机器人动作表示与聚类、RoboTwin π0 末端控制，以及可组合 LoRA/Router 实验代码。

- `data_clustering/`：动作窗口构造、表示学习、视觉辅助、时间一致性与 K=32 聚类代码。
- `RoboTwin/`：RoboTwin、π0 末端数据训练、七簇 LoRA、MLP Router、标准 Pi0 baseline 与评测代码。
- `LoRAHub-JAX/`：历史 LoRA 提取、组合、低秩模块与消融实验代码。
- [服务器训练与数据搬运文档](RoboTwin/policy/pi0/SERVER_TRAINING.md)：基础权重、数据集、环境以及多卡训练说明。
- [五任务 100 条 baseline 文档](RoboTwin/policy/pi0_ee_five/BASELINE100.md)：归一化、训练、视频评测和对比说明。

服务器以全局 batch size 32 运行五任务 100 条标准 Pi0 baseline：

```bash
cd /path/to/RoboLoRA
GPU_IDS=0,1,2,3 FSDP_DEVICES=4 \
  bash RoboTwin/policy/pi0_ee_five/run_baseline100_b32.sh
```

默认训练 5,000 步，每 1,000 步保存一次，五个任务各评测 20 次并保存视频。训练目录为
`RoboTwin/policy/pi0_ee_five/runs/pi0_baseline100_b32_s5000`。`GPU_IDS` 控制使用的显卡，
`FSDP_DEVICES` 控制参数分片组大小；全局 batch 32 必须能被显卡数量整除。

数据集、归一化资产、基础模型、LoRA 权重、checkpoint、缓存、日志、评测结果和视频均未上传，
需要按照各目录文档在服务器上单独放置。

RoboTwin 来源：<https://github.com/RoboTwin-Platform/RoboTwin>。各组件许可证保留在对应目录。
