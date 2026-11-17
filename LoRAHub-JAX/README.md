# VLA LoRAHub (JAX)

基于 Pi0 (PaliGemma + Action Expert) 的 VLA 黑盒 LoRA 组合框架。

**论文:** [LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition](https://arxiv.org/abs/2307.13269)

## 核心思想

1. **阶段 A (独立训练):** 在多个机器人操作任务上用 LoRA 模式分别微调同一个 Pi0 基座，得到 N 个 LoRA 模块 (`lora_modules/*.npz`)
2. **阶段 B (黑盒组合):** 对新任务，仅需 few-shot 样本，通过 Nevergrad CMA-ES 搜索最优组合权重 `{w_1, ..., w_N}`
3. **组合公式:** `composed_lora = Σ w_i * LoRA_i`，搜索阶段在激活空间组合，导出阶段基于 Pi0 基座模型直接注入组合 LoRA，得到可评测 checkpoint

## 目录结构

```
LoRAHub-JAX/
├── config/                      # 任务配置 (数据集来源、LoRA 条目、算法参数)
│   └── place_empty_cup_10.json  #   示例: place_empty_cup 10 条数据
├── script/                      # 统一入口与公共模块
│   ├── run.py                   #   黑盒优化入口 (读 config, V1/V2/V2.5/V3 通用)
│   ├── compose.py               #   组合导出入口 (权重 -> RoboTwin checkpoint)
│   ├── compute_norm_stats.py    #   归一化统计重算入口 (openpi 方式, 数据驱动)
│   ├── extract_lora.py          #   LoRA 提取 CLI (checkpoint -> .npz)
│   └── common/                  #   公共模块
│       ├── data_loader.py       #     few-shot 数据加载 + norm_stats 计算/保存
│       └── pi0_forward.py       #     Pi0 前向推理 (模型缓存, diffusion 步数可调)
├── outputs/                     # 输出产物 (按任务名组织)
│   └── <task_name>/
│       ├── lorahub_weights.json #   优化出的组合权重
│       ├── norm_stats.json      #   按任务数据重算的归一化统计
│       ├── delta_loss_stats.json#   仅供黑盒目标使用的 delta 尺度
│       ├── sample_manifest.json #   episode/frame 关键帧清单
│       ├── checkpoint/          #   组合导出的评测 checkpoint
│       ├── logs/                #   运行日志 + loss 曲线 + 逐步明细
│       └── artifacts/           #   历史/附带产物
├── vla_lorahub/                 # 核心库
│   ├── algorithm.py             #   LoRA 组合 + Nevergrad 黑盒优化 (V1/V2/V3)
│   ├── checkpoint_utils.py      #   Orbax/OCDBT checkpoint 读写 & LoRA 提取
│   └── constant.py              #   基座路径 & LoRA 模块自动发现
├── lora_modules/                # 已提取的 LoRA 模块 (.npz)
├── data/                        # few-shot 评测数据 (RoboTwin episode 格式)
├── run_lorahub.sh               # 优化入口脚本
├── compose_checkpoint.sh        # 组合导出脚本
└── TECHNICAL.md                 # 技术原理文档 (checkpoint 结构 / 组合数学)
```

## 使用流程

### 1. 准备 LoRA 模块 (一次性)

```bash
# 从训练好的 checkpoint 提取 LoRA 到 lora_modules/
python script/extract_lora.py extract -c <checkpoint_dir>/ -o lora_modules/<task>.npz -t <task>
python script/extract_lora.py info          # 查看已提取的模块
```

### 2. 重算归一化统计（从任务数据，openpi 方式）

```bash
python script/compute_norm_stats.py -c config/<task_name>.json [--force]
```

- 通用任务按 OpenPI 的 `AlohaInputs → DeltaActions → Normalize` 顺序，从全部帧和 future action chunks 计算 `state`/delta `actions` 的统计量
- 对齐已有官方 checkpoint 时设置 `use_reference_norm_stats: true` 和 `reference_norm_stats_path`，直接复制官方资产并做严格一致性校验
- 保存到 `outputs/<task_name>/norm_stats.json`，并自动更新配置的 `norm_stats_path`
- 若跳过此步，`run.py` 在 `norm_stats_path` 缺失时也会自动重算

### 3. 编写任务配置 `config/<task_name>.json`

```json
{
  "task_name": "place_empty_cup_10",
  "target_task": "place_empty_cup",
  "train_config_name": "place_empty_cup_pi0_lora",
  "data_dir": "data/place_empty_cup_10",
  "lora_modules": [],
  "norm_stats_path": "outputs/place_empty_cup_10/norm_stats.json",
  "diffusion_steps": 10,
  "algorithm": {
    "version": "v3",
    "max_steps": 100,
    "seed": 42,
    "weight_bounds": [1e-6, 1.5]
  },
  "compose": {
    "asset_id": "place_empty_cup_pi0",
    "sync_dir": "../RoboTwin/policy/pi0/checkpoints/place_empty_cup_pi0_lora_10ep/place_empty_cup_pi0_lora_10ep"
  }
}
```

| 字段 | 说明 |
|------|------|
| `task_name` | 任务名，输出目录 `outputs/<task_name>/` 以此命名 |
| `target_task` | 目标任务名，自动发现 LoRA 模块时排除它自身的 LoRA |
| `train_config_name` | RoboTwin 训练配置名 (模型初始化用) |
| `data_dir` | few-shot 数据集目录 (相对项目根目录) |
| `lora_modules` | 显式指定参与组合的 LoRA 条目；留空 `[]` 则自动发现并排除目标任务 |
| `norm_stats_path` | state/delta-action 归一化统计；必须与最终 checkpoint 的资产相同 |
| `use_reference_norm_stats` | 为 `true` 时使用 `reference_norm_stats_path` 的官方统计，避免数据版本差异 |
| `diffusion_steps` | 搜索阶段 diffusion 采样步数（对齐配置为 2） |
| `algorithm.version` | `v1` 全局 / `v2` 按组 / `v2.5` 视觉-动作 45D / `v3` 按层 |
| `algorithm.max_steps` | 黑盒优化预算；V3 维度为 `10×N`（15 个 LoRA 时为 150D） |
| `algorithm.sampling_mode` | `keyframes` 固定关键帧，或 `pi0_random` 全帧随机 mini-batch |
| `algorithm.candidate_batch_size` | `pi0_random` 下每个随机 batch 的帧数 |
| `algorithm.batches_per_evaluation` | 每个 CMA 候选评分使用的 batch 数量 |
| `algorithm.batch_reuse_candidates` | 同一随机 batch 复用的候选数；应等于 `cma_popsize` |
| `algorithm.composition_mode` | `delta_concat` 为无交叉项的精确 delta 组合；`delta_sketch` 为省显存近似 |
| `algorithm.validation_diffusion_steps` | 周期/最终精排的 diffusion 步数（推荐 10） |
| `algorithm.validation_diffusion_seeds` | 精排固定噪声 seeds（推荐 2–4 个） |
| `algorithm.validation_interval_candidates` | 每隔多少候选用部署步数验证一次 CMA recommendation |
| `algorithm.weight_update_interval` | V3 每隔多少步原子更新一次 `lorahub_weights.json` |
| `algorithm.checkpoint_interval` | V3 每隔多少步保存一次可恢复的 CMA-ES checkpoint |
| `algorithm.resume` | V3 启动时是否自动从 `optimization_checkpoints/latest.pkl` 恢复 |
| `compose.asset_id` | 数据资产目录名，导出的 `assets/<asset_id>/norm_stats.json`（来自本任务 norm_stats） |
| `compose.sync_dir` | 可选；导出后同步到的 RoboTwin checkpoint 根目录（其下写 `10000/{params,assets}`） |

所有相对路径均基于项目根目录 `LoRAHub-JAX/` 解析。

RTX 4090 推荐从 `config/place_empty_cup_10_local_min.json` 开始。该配置使用
10 episodes × 4 个分阶段动作关键帧、`max_batch=4`、V2
`action_focused`（只搜索 Action Expert attention/MLP，共 `2N` 维）、
分组 delta Huber loss，以及保持原始 LoRA rank 的
`action_delta_sketch` 组合。`delta_loss_stats` 只平衡搜索目标，不替换
checkpoint 的 `norm_stats`。

V2.5 推荐配置为 `config/place_empty_cup_10_v25_pi0_batch.json`：数据候选池
包含每条 episode 的全部帧，尾部动作按 LeRobot 规则重复 episode 最后一帧；
每个 CMA 世代共享一个随机 mini-batch，下一世代重新采样。图像 resize、state
归一化、动作反归一化和模型采样均复用固定 shape 的 JAX-JIT executable。
该配置用 `delta_concat` 精确计算 `Σw_iA_iB_i`；搜索保持 2-step，每 500 个
候选及最终选择使用 10-step×4 fixed seeds。精确组合会把搜索态 LoRA rank
扩大 N 倍，适合大显存服务器；导出时 delta 会合并进基座并恢复原生 rank 的
零 LoRA 叶子，因此 RoboTwin 评测图保持官方兼容。

### 4. 运行黑盒优化

```bash
bash run_lorahub.sh place_empty_cup_10        # 按任务名
bash run_lorahub.sh config/place_empty_cup_10.json   # 或直接指定配置
PYTHON=/path/to/python bash run_lorahub.sh place_empty_cup_10   # 指定解释器
```

日志自动写入 `outputs/place_empty_cup_10/logs/run_<时间戳>.log`，权重输出到 `outputs/place_empty_cup_10/lorahub_weights.json`。
V3 可设置周期性权重快照和断点恢复；checkpoint 保存在
`outputs/<task_name>/optimization_checkpoints/step_XXXXXXXXX.pkl`，并维护
`latest.pkl`。权重 JSON 在优化完成前标记为 `status: in_progress`，仅供中间查看。
优化过程带 tqdm 实时进度条与预计剩余时间；完成后自动在 `outputs/<task_name>/logs/` 生成 `loss_curve.png`（loss/reg/total 曲线）与 `loss_history.csv`（逐步明细）。

### 5. 导出评测 checkpoint

```bash
bash compose_checkpoint.sh place_empty_cup_10
```

**基座注入模式**：加载 Pi0 基座模型 `openpi_base/pi0_base/params`，按
`lorahub_weights.json` 记录的 `composition_mode` 组合各 LoRA 模块，把组合后的
参数写入基座树，输出到
`outputs/<task_name>/checkpoint/10000/{params,assets}`，可选同步到
`compose.sync_dir` 供 RoboTwin 直接评测。
`delta_concat` 会将精确 delta 合并到基座权重并注入原生 rank 的零 LoRA；其他
模式直接注入组合 LoRA。资产 `norm_stats.json` 与搜索时使用的统计完全相同。

## 优化方案对照

| 版本 | 分组方式 | 维度 | 权重格式 |
|------|---------|------|---------|
| v1 | 全局单一权重 | N | `{version, module_names, optimal_weights[N]}` |
| v2 full | 4 组 (attn/mlp × 主/辅) | 4×N | `{version, group_names, module_names, optimal_weights[4][N]}` |
| v2 action_focused | 主干固定、只搜索 Action Expert 两组 | 2×N 搜索，4×N 导出 | 同上 |
| v2.5 vision_action_focused | 搜索主干 attention + 两个 Action Expert 组，固定主干 MLP | 3×N 搜索，4×N 导出；15 个模块时为 45D | 同上 |
| v3 | 10 层逐层 | 10×N；15 个模块时为 150D | `{version, layer_names, module_names, optimal_weights[10][N]}` |

## 环境要求

- RTX 4090 (24GB) 已验证 40 条样本、50-step horizon、`max_batch=4`
- 显存受限时建议 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.75`
- 依赖: `jax`, `flax`, `orbax-checkpoint`, `nevergrad`, `numpy` (使用 RoboTwin 的 pi0 venv 即可)

详细技术原理见 [TECHNICAL.md](TECHNICAL.md)。
