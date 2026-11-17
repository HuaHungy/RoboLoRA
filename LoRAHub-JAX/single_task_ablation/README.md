# 单任务 LoRAHub 对照实验

目标任务固定为 `place_empty_cup`，使用同一个 Pi0 base、同一批 10 条训练
episode 和同一套 RoboTwin evaluation。LoRAHub 采用顺序 epoch 读取：每个
候选按固定 frame pool 顺序遍历，候选之间共享相同 batch，避免随机 batch
造成黑盒目标噪声。比较两条路径：

| 组别 | 可训练/可搜索内容 | 基座 |
|---|---|---|
| Native | full action expert；冻结 PaliGemma | 官方 `pi0_base` |
| LoRAHub | 先黑盒搜索并合并 15 个其他任务 LoRA，再训练同样的 full action expert；冻结 PaliGemma | LoRAHub 处理后的 Pi0 base |

Native baseline 不是现有的 `place_empty_cup_pi0_lora_10ep`：后者只训练 LoRA
叶子。`train_action_expert_only.py` 在运行时注册一个临时 OpenPI config，冻结
`llm`（PaliGemma）并训练 `llm_1`（action expert）以及 action head 的投影层，
不修改相邻 RoboTwin checkout。

先做配置和数据检查：

```bash
cd /home/huahungy/RoboLoRAHub/LoRAHub-JAX
single_task_ablation/run_single_task.sh native
```

再运行 LoRAHub 搜索、导出、LoRAHub 初始化后的 action-expert 训练和两组评测：

```bash
single_task_ablation/run_single_task.sh all
```

建议先分别运行 `native` 和 `lorahub`，确认 checkpoint 成功后再运行
`eval-native`、`eval-lorahub`。正式结果以 RoboTwin success rate 为主；同时
记录同一 seed、同一 GPU、同一 evaluation episode 数，并保存两个视频目录。

判定 LoRAHub 是否有效：相对于 Native，LoRAHub 初始化后再训练 action expert
在未见 evaluation episode 上 success rate 更高或在误差范围内相当，同时动作
loss/轨迹抖动不明显恶化。两组必须使用相同训练步数、batch size、学习率、seed
和评测 seed。

后续多任务版本只需把 `data_dir`/LeRobot repo 换成 5 个任务各 10 条数据，
并为 native 与 LoRAHub 保持相同的总 episode 数、训练步数和评测协议；这次不
把多任务结果混入单任务结论。
