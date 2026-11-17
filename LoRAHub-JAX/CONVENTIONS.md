# 项目组织规范（Conventions）

> 本文档基于 2026-08-19 的项目重构确立，**所有后续改动必须遵守**。
> 违反本规范 = 需要重构整改的改动，请勿引入。

## 1. 核心原则

本项目采用 **配置驱动 + 统一入口** 的组织形式：

1. **每个任务一份配置**（`config/<task_name>.json`），一切可变参数（数据集来源、LoRA 条目、算法参数）只能放在配置里，**禁止硬编码在脚本中**。
2. **所有可执行入口集中在 `script/`**，功能上不允许出现两个干同样事情的脚本。
3. **所有输出产物按任务名归入 `outputs/<task_name>/`**，根目录不残留任何产物、日志、中间文件。
4. **公共逻辑必须进 `script/common/`**，禁止复制粘贴式复用（历史教训：V1/V2/V3 曾各有一份几乎相同的 `run_lorahub_vX.py` / `compose_from_template_vX.py`）。

## 2. 目录结构（权威）

```
LoRAHub-JAX/
├── config/                        # 任务配置 (唯一参数入口)
│   └── <task_name>.json
├── script/                        # 统一入口 + 公共模块
│   ├── run.py                     # 黑盒优化入口 (V1/V2/V3 通过 config 分发)
│   ├── compose.py                 # 组合导出入口 (权重 -> checkpoint)
│   ├── compute_norm_stats.py      # 归一化统计重算入口 (openpi 方式)
│   ├── extract_lora.py            # LoRA 提取 CLI
│   └── common/                    # 公共模块 (只放可复用逻辑)
│       ├── data_loader.py         #   few-shot 数据加载 + norm_stats 计算/保存
│       └── pi0_forward.py
├── outputs/                       # 全部输出产物 (按任务名组织)
│   └── <task_name>/
│       ├── lorahub_weights.json   # 优化权重 (run.py 产出)
│       ├── norm_stats.json        # 按数据重算的归一化统计 (compute_norm_stats.py 产出)
│       ├── checkpoint/            # 评测 checkpoint (compose.py 产出)
│       ├── logs/                  # 日志 + loss 曲线 (由 .sh / run.py 写入)
│       └── artifacts/             # 历史/附带产物
├── vla_lorahub/                   # 核心库
│   ├── algorithm.py               # 算法 (含 V1/V2/V3 三种方案)
│   ├── checkpoint_utils.py
│   └── constant.py                # 基座路径 / LoRA 模块发现
├── lora_modules/                  # 提取的 LoRA 模块 (.npz) — 只读资源
├── data/                          # few-shot 数据 — 只读资源
├── run_lorahub.sh                 # 优化入口
├── compose_checkpoint.sh          # 组合导出入口
└── README.md / TECHNICAL.md / CONVENTIONS.md
```

## 3. 铁律（违反即整改）

### 3.1 根目录纪律
- **禁止**在根目录新增脚本（`.py`）、日志（`.log`）、产物目录。
- **禁止**出现 `run_lorahub_vX.py`、`compose_from_template_vX.py` 这类按版本复制的脚本；算法版本演进只允许改 `vla_lorahub/algorithm.py`，并在 `script/run.py` 的分发表中登记。
- **禁止**把 `.bak`、`__pycache__`、`.deps` 等杂物留在项目内。

### 3.2 配置纪律
- 新增/修改任务参数**只能**改 `config/<task_name>.json`，不改 `script/` 代码。
- 配置文件内相对路径基于项目根目录 `LoRAHub-JAX/` 解析（`run.py` / `compose.py` 的 `load_config` 已实现）。
- `lora_modules` 留空 `[]` = 自动发现并排除 `target_task`；显式填写 = 精确控制条目。

### 3.3 输出纪律
- 优化权重 → `outputs/<task_name>/lorahub_weights.json`
- 归一化统计 → `outputs/<task_name>/norm_stats.json`（**禁止**复用其他任务/数据集现成的 norm_stats）
- 评测 checkpoint → `outputs/<task_name>/checkpoint/10000/{params,assets}`（**基座注入模式**：基座 `openpi_base/pi0_base/params` + 注入组合 LoRA 叶子，禁止用训练产物当模板复制）
- 日志 → `outputs/<task_name>/logs/`（由 `.sh` 自动写，手动跑 `python script/run.py` 时也要重定向到这里）
- loss 曲线与明细 → `outputs/<task_name>/logs/loss_curve.png` + `loss_history.csv`（run.py 自动产出）
- 历史产物（.npz 等）→ `outputs/<task_name>/artifacts/`

### 3.4 组合导出纪律
- 评测 checkpoint = **基座模型 + 注入组合 LoRA**，由 `script/compose.py` 统一完成；配置文件里只写 `compose.asset_id`（资产目录名）与 `compose.sync_dir`（RoboTwin 同步目标），**禁止**出现 `template_params` / `template_assets` 这类模板字段。
- `assets/<asset_id>/norm_stats.json` 必须来自本任务数据（`compute_norm_stats.py` 产出），禁止拷贝其他任务的资产。

### 3.5 代码纪律
- 公共数据加载、Pi0 推理只存在于 `script/common/`，其他模块引用它，不得复制。
- 模型加载/前向缓存逻辑（graphdef、base 浅拷贝）属于 `pi0_forward.py`，不要在其他文件重复实现。
- 优化历史（每步 loss/reg/total）由 `vla_lorahub/algorithm.py` 的 `SCORE_HISTORY` 统一收集，`script/run.py` 只负责读取与绘图；禁止在 run.py 里自行再算一遍。
- norm_stats 计算逻辑（mean / ddof=0 std / q01/q99 + 32 维补齐）只在 `script/common/data_loader.py` 的 `compute_norm_stats` 实现。
- 硬编码路径仅允许出现在 `vla_lorahub/constant.py`；业务脚本一律走配置。

## 4. 新增任务的标准流程

1. **准备数据**：few-shot 数据放入 `data/<task_name>/`（episode parquet 格式，见 `data_loader.py` 约定）。
2. **提取 LoRA**（如需要）：
   ```bash
   python script/extract_lora.py extract -c <ckpt_dir>/ -o lora_modules/<task>.npz -t <task>
   ```
3. **重算归一化统计**（从本任务数据，openpi 方式，自动更新配置）：
   ```bash
   python script/compute_norm_stats.py -c config/<task_name>.json [--force]
   ```
   或直接运行优化——`run.py` 在 `norm_stats_path` 缺失时也会自动重算并保存。
4. **写配置**：`cp config/place_empty_cup_10.json config/<task_name>.json`，修改：
   - `task_name`（决定输出目录名）
   - `data_dir`（数据来源）
   - `norm_stats_path`（建议指向 `outputs/<task_name>/norm_stats.json`）
   - `lora_modules`（使用的 LoRA 条目，或留空）
   - `algorithm`（version / max_steps / seed / diffusion_steps）
   - `compose`（asset_id 资产目录名 + sync_dir 同步目标，见下）
5. **运行**：
   ```bash
   bash run_lorahub.sh <task_name>          # 优化 -> outputs/<task>/lorahub_weights.json
   bash compose_checkpoint.sh <task_name>   # 导出 -> outputs/<task>/checkpoint/
   ```
6. 运行日志与 loss 曲线自动落入 `outputs/<task_name>/logs/`，无需手动管理。

## 5. 路径与命名约定

| 项 | 约定 |
|----|------|
| 任务配置 | `config/<task_name>.json`，`task_name` 用 snake_case |
| 输出目录 | `outputs/<task_name>/`，与 `task_name` 一一对应 |
| LoRA 模块 | `lora_modules/<task>_pi0_lora.npz` |
| 权重文件 | 固定名 `lorahub_weights.json`（version 字段区分 v1/v2/v3） |
| checkpoint 根 | `outputs/<task>/checkpoint/10000/`（step=10000 固定） |
| 日志文件 | `run_<YYYYmmdd_HHMMSS>.log` / `compose_<YYYYmmdd_HHMMSS>.log` |
| 版本标识 | 算法版本只体现在权重文件的 `version` 字段，不体现在文件名 |

## 6. 修改边界

**可自由修改**：
- `config/*.json` — 任务级参数
- `script/common/` — 公共工具（新增工具函数时注意被引用处）
- `vla_lorahub/algorithm.py` — 算法实现（新增 V4 方案：加新函数 + 在 `run.py` 的 `learners` 分发表登记 + 在 `compose.py` 的 `compose_weights` 登记 layer 映射）

**需谨慎修改**：
- `script/run.py` / `script/compose.py` 的输出路径与权重格式（下游 RoboTwin 评测依赖 `outputs/<task>/checkpoint/10000/` 结构）
- `vla_lorahub/checkpoint_utils.py` — Orbax 读写与 LoRA 提取格式（.npz 键名被 compose.py / run.py 依赖）

**禁止修改**：
- 根目录布局与入口脚本命名（`run_lorahub.sh`、`compose_checkpoint.sh`）
- `lora_modules/`、`data/`、`openpi_base/` 等符号链接目标（只读资源，沙箱外管理）

## 7. 环境与运行约束（历史教训）

- 4090 24GB：`XLA_FLAGS="--xla_gpu_autotune_level=0"`、`XLA_PYTHON_CLIENT_MEM_FRACTION=0.75`（`script/run.py` 已默认，勿移除）。
- 搜索阶段 `diffusion_steps` 可调 4 提速，且不影响权重排序趋势；评测导出时恢复 10。
- Nevergrad 用 `num_workers=1` 串行评估；搜索中避免 `config.model.load()` 重复 JIT（用 nnx 原地更新，见 `pi0_forward.py` 缓存逻辑）。
- 权重归一化禁止裸 `sum(raw_weights)`（趋零时会爆炸），用 Softmax/Tanh。

## 8. 提交前自查清单

- [ ] 根目录无新增 `.py` / `.log` / 产物目录
- [ ] 参数没有硬编码进脚本（都在 config 里）
- [ ] 产物与日志都落在 `outputs/<task_name>/`
- [ ] 没有复制粘贴公共代码
- [ ] `script/run.py`、`script/compose.py` 均可 `py_compile` 通过
- [ ] README 的目录结构章节与实际一致（如有新增模块需同步更新）
