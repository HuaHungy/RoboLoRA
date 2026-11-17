# VLA LoRAHub — 技术文档

> 从 NLP LoRAHub 到 VLA (Vision-Language-Action) LoRAHub 的完整技术记录

---

## 目录

1. [原始 LoRAHub 代码架构分析](#1-原始-lorahub-代码架构分析)
2. [Pi0 Checkpoint 结构分析](#2-pi0-checkpoint-结构分析)
3. [VLA LoRAHub 框架设计](#3-vla-lorahub-框架设计)
4. [LoRA 提取：原理与无损性证明](#4-lora-提取原理与无损性证明)
5. [LoRA 重组：激活空间组合 vs Weight 空间合并](#5-lora-重组激活空间组合-vs-weight-空间合并)
6. [LoRA 配对验证结果](#6-lora-配对验证结果)
7. [项目文件结构](#7-项目文件结构)
8. [操作流程](#8-操作流程)

---

## 1. 原始 LoRAHub 代码架构分析

### 1.1 论文核心思想

**论文:** [LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition](https://arxiv.org/abs/2307.13269) (COLM 2024)

LoRAHub 提出了一种动态组合多个预训练 LoRA 模块的方法，通过**黑盒优化（无梯度、无反向传播）**学习一组权重系数，使组合后的模型能够适应新任务。

### 1.2 两阶段流程

```
┌─────────────────────────────────────────────────────┐
│  阶段一：独立 LoRA 训练（有梯度微调）                   │
│  基座模型: google/flan-t5-large (Seq2Seq LM)          │
│  每个 LoRA: PEFT LoraConfig(r=16, alpha=32)           │
│  196 个模块在不同 NLP 任务/模板上训练                   │
│  → 上传到 HuggingFace Hub                              │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│  阶段二：LoRAHub 黑盒组合学习（无梯度优化）              │
│  1. 从 HuggingFace Hub 加载候选 LoRA 模块的 state_dict  │
│  2. 给每个模块分配权重 w_i ∈ [-1.5, 1.5]               │
│  3. Nevergrad NGOpt 黑盒优化 40 步                     │
│     - 每步: 组合 LoRA state_dict → 加载到 PEFT model   │
│            → 前向推理 → 计算 CE loss                   │
│     - 无需反向传播！                                     │
│  4. 返回最优权重 + merge_and_unload() 后的模型          │
└─────────────────────────────────────────────────────┘
```

### 1.3 关键代码实现 (algorithm.py)

**LoRA 组合公式（在 PEFT state_dict 空间）:**

```python
# 在每个 LoRA 参数的 key 上做加权线性求和
final_state_dict[key] = Σ w_i * lora_state_dict_i[key]

# 注意: 组合发生在 state_dict (参数) 空间
# PEFT forward 时: output = base(x) + lora_B(lora_A(x))
#   其中 lora_A = Σ w_i * lora_A_i, lora_B = Σ w_i * lora_B_i
# 因此实际 forward 为:
#   output = base(x) + (Σ w_j * B_j) @ ((Σ w_i * A_i) @ x)
#          = base(x) + Σ_i Σ_j w_i w_j * B_j @ A_i @ x
# 交叉项 w_i w_j (i ≠ j) 是 LoRAHub 设计的核心特征
```

**评分函数 (get_score):**

```python
def get_score(weights, model, cache, example_dataset, ...):
    # 1. 用权重组合 LoRA state_dict
    final_state_dict = Σ w_i * lora_i
    set_peft_model_state_dict(model, final_state_dict)

    # 2. 无梯度前向推理，计算 loss
    with torch.no_grad():
        outputs = model(**batch)
    loss = outputs.loss / len(dataset)

    # 3. L1 正则化: 0.05 * Σ|w_i| / N
    return loss + get_regular(weights)
```

**优化器设置:**

```python
instrum = ng.p.Array(
    init=[0] * number_of_loras,
    upper=[1.5] * number_of_loras,
    lower=[-1.5] * number_of_loras,
)
optimizer = ng.optimizers.NGOpt(parametrization=instrum, budget=max_inference_step)
```

### 1.4 与原始 LoRAHub 的维度对比

| 维度 | 原始 LoRAHub | VLA LoRAHub (本项目) |
|------|-------------|---------------------|
| 基座模型 | Flan-T5 (PyTorch) | Pi0 (JAX/Orbax) |
| LoRA 格式 | HuggingFace PEFT safetensors | NumPy .npz |
| 优化目标 | Cross-entropy (文本生成) | 标准化 delta 分组 Huber (动作预测) |
| 任务 | NLP (QA, reasoning) | 机器人操作 (VLA) |
| 输入 | 文本 prompt | 图像 + 状态 + 指令 |
| Checkpoint | HuggingFace Hub | 本地 Orbax OCDBT |
| 优化器 | Nevergrad NGOpt | Nevergrad NGOpt |
| 组合方式 | state_dict 空间加权求和 | **激活空间加权求和**（等效） |

---

## 2. Pi0 Checkpoint 结构分析

### 2.1 目录结构

```
pi0_weights/5000/              ← 训练 step=5000 的快照
├── _CHECKPOINT_METADATA       ← 顶层索引 (JSON)
├── params/                    ← 模型权重 (70 个参数)
│   ├── _METADATA              ← 参数路径 → shape/dtype 映射表
│   ├── _sharding              ← JAX 分片策略 (base64)
│   ├── array_metadatas/       ← 参数的可选元数据
│   ├── manifest.ocdbt         ← OCDBT 顶层 manifest
│   ├── d/                     ← 扁平 OCDBT 数据 (兼容)
│   └── ocdbt.process_0/       ← 多进程 OCDBT
│       ├── manifest.ocdbt     ← [(idx, file_name), ...]
│       └── d/                 ← msgpack 序列化的 jax.Array
│           ├── 09796e78...    ← 第 0 个参数
│           └── ...            ← 共 70 个文件
├── train_state/               ← 训练状态
│   ├── step                   ← 当前步数 (标量)
│   ├── opt_state/1/0/mu       ← Adam μ (一阶动量)
│   ├── opt_state/1/0/nu       ← Adam ν (二阶动量)
│   └── opt_state/1/0/count    ← 优化器步数
└── assets/                    ← 辅助资源
    └── realman/v3/norm_stats.json  ← state/action 归一化统计
```

### 2.2 参数统计

| 类别 | 数量 | 参数量 | 存储 (float32) |
|------|------|--------|----------------|
| **总参数** | 70 | 32.88 亿 | ~12.25 GB |
| **基座参数** | 50 | 32.38 亿 | ~12.06 GB |
| **LoRA 参数** | 20 | 4998 万 | ~190 MB |
| **LoRA 占比** | — | 1.52% | — |

### 2.3 LoRA 参数详细列表 (20 个)

| # | 路径 (相对 PaliGemma/llm/layers/) | Shape | 角色 | rank |
|---|-----------------------------------|-------|------|------|
| 1 | `attn/q_einsum/lora_a/value` | (18, 8, 2048, 16) | Q 投影 A (主 LLM) | r=16 |
| 2 | `attn/q_einsum/lora_b/value` | (18, 8, 16, 256) | Q 投影 B (主 LLM) | r=16 |
| 3 | `attn/q_einsum_1/lora_a/value` | (18, 8, 1024, 32) | Q 投影 A (辅助模块) | r=32 |
| 4 | `attn/q_einsum_1/lora_b/value` | (18, 8, 32, 256) | Q 投影 B (辅助模块) | r=32 |
| 5 | `attn/kv_einsum/lora_a/value` | (18, 2, 1, 2048, 16) | KV 投影 A | r=16 |
| 6 | `attn/kv_einsum/lora_b/value` | (18, 2, 1, 16, 256) | KV 投影 B | r=16 |
| 7 | `attn/kv_einsum_1/lora_a/value` | (18, 2, 1, 1024, 32) | KV 投影 A (辅助) | r=32 |
| 8 | `attn/kv_einsum_1/lora_b/value` | (18, 2, 1, 32, 256) | KV 投影 B (辅助) | r=32 |
| 9 | `attn/attn_vec_einsum/lora_a/value` | (18, 8, 256, 16) | 注意力输出 A | r=16 |
| 10 | `attn/attn_vec_einsum/lora_b/value` | (18, 8, 16, 2048) | 注意力输出 B | r=16 |
| 11 | `attn/attn_vec_einsum_1/lora_a/value` | (18, 8, 256, 32) | 注意力输出 A (辅助) | r=32 |
| 12 | `attn/attn_vec_einsum_1/lora_b/value` | (18, 8, 32, 1024) | 注意力输出 B (辅助) | r=32 |
| 13 | `mlp/gating_einsum_lora_a/value` | (18, 2, 2048, 16) | MLP 门控 A | r=16 |
| 14 | `mlp/gating_einsum_lora_b/value` | (18, 2, 16, 16384) | MLP 门控 B | r=16 |
| 15 | `mlp/linear_lora_a/value` | (18, 16384, 16) | MLP 线性 A | r=16 |
| 16 | `mlp/linear_lora_b/value` | (18, 16, 2048) | MLP 线性 B | r=16 |
| 17 | `mlp_1/gating_einsum_lora_a/value` | (18, 2, 1024, 32) | MLP 门控 A (辅助) | r=32 |
| 18 | `mlp_1/gating_einsum_lora_b/value` | (18, 2, 32, 4096) | MLP 门控 B (辅助) | r=32 |
| 19 | `mlp_1/linear_lora_a/value` | (18, 4096, 32) | MLP 线性 A (辅助) | r=32 |
| 20 | `mlp_1/linear_lora_b/value` | (18, 32, 1024) | MLP 线性 B (辅助) | r=32 |

- `18` = 18 层 transformer layers
- `8` = 8 attention heads (主 LLM 用 2048 维)
- `2` = 2 组 KV (GQA)
- 主 LLM 用 rank=16, 辅助模块用 rank=32

### 2.4 LoRA 参数实际数值特征 (以 q_einsum 为例)

```
lora_a: shape=(18, 8, 2048, 16)
  均值: 0.000003  标准差: 0.0102  范围: [-0.053, +0.051]

lora_b: shape=(18, 8, 16, 256)
  均值: 0.000026  标准差: 0.0102  范围: [-0.047, +0.051]

w (base): shape=(18, 8, 2048, 256)
  [基座权重，不在 .npz 中保存]
```

符合 LoRA 典型行为: 通过低秩适配学习**任务相关的微小扰动**。

---

## 3. VLA LoRAHub 框架设计

### 3.1 设计原则

完全复刻原始 LoRAHub 的黑盒优化架构，适配以下差异:

1. **存储格式**: PyTorch safetensors → NumPy .npz
2. **加载方式**: PEFT → Orbax PyTreeCheckpointHandler
3. **损失函数**: Cross-Entropy (文本) → MSE (动作预测)
4. **模型架构**: Flan-T5 Seq2Seq → Pi0 (PaliGemma + Action Expert)

### 3.2 两阶段流程 (VLA 版)

```
阶段 A: 独立 LoRA 训练
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
基座模型: Pi0 (PaliGemma ViT + LLM + Action Expert)
在每个任务上 LoRA 微调:
  任务 A (grab_roller) → pi0_weights/grab_roller/29999/
  任务 B (pick_place)  → pi0_weights/pick_place/29999/
  任务 C (pour_water)  → pi0_weights/pour_water/29999/
  ...

阶段 B: LoRAHub 黑盒组合
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                                       ┌──────────────────┐
  ┌──────────────────┐  加权系数 w_i    │                  │
  │ LoRA 模块 A      │ ──────────────→  │  组合后的        │
  ├──────────────────┤                 │  LoRA            │
  │ LoRA 模块 B      │ ──────────────→  │  composed_a =    │
  ├──────────────────┤                 │  Σ w_i·A_i       │
  │ ...              │                 │  composed_b =    │
  ├──────────────────┤                 │  Σ w_i·B_i       │
  │ LoRA 模块 N      │ ──────────────→  │                  │
  └──────────────────┘                 └────────┬─────────┘
                                               │
                                      ┌────────▼─────────┐
                                      │  Pi0 推理         │
                                      │  output =         │
                                      │  base(x) +        │
                                      │  (x@lora_a)@lora_b│
                                      └────────┬─────────┘
                                               │
                                      ┌────────▼─────────┐
                                      │  MSE Loss         │ ← Nevergrad
                                      └──────────────────┘   黑盒优化
```

### 3.3 Nevergrad 黑盒优化配置

```python
# 权重空间: 每个 LoRA 模块一个连续权重
instrum = ng.p.Array(
    init=[0.0] * N,           # 初始化全 0
    upper=[1.5] * N,          # 上界
    lower=[-1.5] * N,         # 下界
)
optimizer = ng.optimizers.NGOpt(
    parametrization=instrum,
    budget=40                 # 迭代步数
)

# 目标函数: loss = MSE(actions, ground_truth) + 0.05 * Σ|w_i|/N
```

---

## 4. LoRA 提取：原理与无损性证明

### 4.1 提取流程

```
Orbax OCDBT 二进制 (params/ocdbt.process_0/d/*)
    ↓
orbax.checkpoint.PyTreeCheckpointHandler().restore()
    ↓  ← 官方 API，精确反序列化
jax.Array (嵌套 PyTree)
    ↓
_flatten_pytree() → {path: jax.Array}
    ↓  ← np.array() 转换，float32 逐 bit 保留
{path: np.ndarray}  (70 个参数)
    ↓
extract_lora_weights() → 筛选 key 包含 "lora" 的参数
    ↓
{path: np.ndarray}  (20 个 LoRA 参数)
    ↓
np.savez_compressed() → .npz 文件
```

### 4.2 无损性证明

1. **数值精度**: Orbax 存储的 jax.Array (float32) → numpy array (float32)，逐 bit 相同，无精度损失
2. **信息完整性**: LoRA 训练中基座权重 w 是冻结的，只有 lora_a 和 lora_b 被更新。这 20 个参数完整描述了该任务的全部可训练知识
3. **可恢复性**:
   - 基座权重: 从基座 checkpoint (`pi0_weights/5000/`) 加载(50 个非 LoRA 参数)
   - LoRA 增量: 从 .npz 加载 20 个 lora_a/lora_b
   - 组合后模型 = 基座模型 + LoRA delta，与原始 checkpoint 数学等价

---

## 5. LoRA 重组：激活空间组合 vs Weight 空间合并

### 5.1 两种重组方式

LoRAHub 支持两种方式将 LoRA 应用到基座模型:

#### 方式 A: 传统 factor 组合（兼容模式）
```python
# 保持 lora_a/lora_b 独立
composed_lora_a = Σ w_i * lora_a_i
composed_lora_b = Σ w_i * lora_b_i

# Pi0 forward 时:
# output = base(x) + (x @ composed_lora_a) @ composed_lora_b
#        = base(x) + Σ_i Σ_j w_i·w_j · (x @ A_i) @ B_j
```
**特点:**
- 保留低秩结构 (rank 不变)
- 交叉项 w_i·w_j (i≠j) 捕获模块间交互
- 与原始 LoRAHub PEFT forward 完全等价
- 适合优化阶段 (每次调整 w 时无需重新计算合并)

#### 方式 B: rank-compatible delta sketch（4090 近似模式）

对需要重点搜索的 Action Expert 组使用固定 Rademacher 符号：

```python
A' = Σ s_i·sqrt(w_i)·A_i
B' = Σ s_i·sqrt(w_i)·B_i       # s_i ∈ {-1, +1}
```

其对角项严格为 `Σ w_i A_iB_i`；交叉项具有平衡符号并在随机符号意义下
抵消。单个固定 sketch 并不等价于精确 delta，只应在显存不足时使用。

服务器对齐配置使用 `delta_concat`：把各源 LoRA 的 rank block 拼接，严格得到
`A_cat B_cat = Σw_iA_iB_i`。搜索态 rank 扩大 N 倍；导出时将精确 delta 合并到
基座权重，并注入原生 rank 的零 LoRA 叶子，所以最终 checkpoint 仍可由官方
RoboTwin/Pi0 图加载。

#### 方式 C: Weight 空间合并
```python
# 对每个模块计算 delta 并合并到基座 w
delta = lora_a @ lora_b   # 最后两维 matmul
merged_w = w + delta

# 合并后模型为标准 checkpoint (无 LoRA 包装)
# 适合推理部署
```

### 5.2 Pi0 中 LoRA 的数学形式

对于 `q_einsum`:
```
w:      shape (18, 8, 2048, 256)    ← 基座 Q 投影权重
lora_a: shape (18, 8, 2048, 16)     ← 降维矩阵 (d_in=2048 → r=16)
lora_b: shape (18, 8, 16, 256)      ← 升维矩阵 (r=16 → d_out=256)

delta = lora_a @ lora_b   ← np.matmul 自动对最后两维
      = (18, 8, 2048, 16) @ (18, 8, 16, 256)
      = (18, 8, 2048, 256)  ✓ 匹配 w 的 shape

merged_w = w + delta      ← 直接相加
```

### 5.3 关键 Bug 修正

**错误做法 (已修正):**
```python
# 错误: 试图按 key 匹配合并 (lora key ≠ base key)
merged[key] = base_val + lora_val  # key 不匹配，无法合并
```

**正确做法:**
```python
# 正确: 保持 lora_a/lora_b 独立，forward 时处理
model_params = base_params.copy()
for key, val in composed_lora.items():
    model_params[key] = val    # lora key 与 base key 不冲突
model_params["__lora_keys__"] = set(composed_lora.keys())
```

### 5.4 关键实现函数

| 函数 | 作用 |
|------|------|
| `compose_lora_weights()` | `final_lora[key] = Σ w_i * lora_i[key]` |
| `build_composed_model_params()` | 合并 base + composed_lora 字典 (不合并 weight) |
| `get_lora_pair_shapes()` | 自动检测 lora_a ↔ lora_b 配对 |
| `compute_merged_weight()` | 计算 `merged_w = w + lora_a @ lora_b` |
| `merge_all_lora_to_base()` | 遍历所有配对，导出合并后的纯基座参数 |

---

## 6. LoRA 配对验证结果

全部 10 对 LoRA 的 lora_a @ lora_b 的 delta shape 均与基座 w 匹配:

| 模块 | lora_a shape | lora_b shape | delta → w | rank |
|------|-------------|-------------|-----------|------|
| `attn/q_einsum` | (18,8,2048,16) | (18,8,16,256) | (18,8,2048,256) ✓ | r=16 |
| `attn/q_einsum_1` | (18,8,1024,32) | (18,8,32,256) | (18,8,1024,256) ✓ | r=32 |
| `attn/kv_einsum` | (18,2,1,2048,16) | (18,2,1,16,256) | (18,2,1,2048,256) ✓ | r=16 |
| `attn/kv_einsum_1` | (18,2,1,1024,32) | (18,2,1,32,256) | (18,2,1,1024,256) ✓ | r=32 |
| `attn/attn_vec_einsum` | (18,8,256,16) | (18,8,16,2048) | (18,8,256,2048) ✓ | r=16 |
| `attn/attn_vec_einsum_1` | (18,8,256,32) | (18,8,32,1024) | (18,8,256,1024) ✓ | r=32 |
| `mlp/gating_einsum` | (18,2,2048,16) | (18,2,16,16384) | (18,2,2048,16384) ✓ | r=16 |
| `mlp/linear` | (18,16384,16) | (18,16,2048) | (18,16384,2048) ✓ | r=16 |
| `mlp_1/gating_einsum` | (18,2,1024,32) | (18,2,32,4096) | (18,2,1024,4096) ✓ | r=32 |
| `mlp_1/linear` | (18,4096,32) | (18,32,1024) | (18,4096,1024) ✓ | r=32 |

**结论**: 所有 delta 的 shape 与基座 w 完全匹配，weight 级合并方案正确。

---

## 7. 项目文件结构

```
LoRAHub-JAX/
├── config/                      # 任务配置 (数据集来源、LoRA 条目、算法参数)
│   └── place_empty_cup_10.json  #   示例配置
├── script/                      # 统一入口与公共模块
│   ├── run.py                   #   黑盒优化入口 (读 config, V1/V2/V3 通用)
│   ├── compose.py               #   组合导出入口 (权重 -> RoboTwin checkpoint)
│   ├── extract_lora.py          #   LoRA 提取 CLI (checkpoint -> .npz)
│   └── common/
│       ├── data_loader.py       #     few-shot 数据加载 + 归一化
│       └── pi0_forward.py       #     Pi0 前向推理 (模型缓存, diffusion 步数可调)
├── outputs/                     # 输出产物 (按任务名组织)
│   └── <task_name>/
│       ├── lorahub_weights.json #   组合权重
│       ├── checkpoint/          #   导出的评测 checkpoint
│       ├── logs/                #   运行日志
│       └── artifacts/           #   历史/附带产物
├── vla_lorahub/                 # 核心库
│   ├── __init__.py              # 包入口
│   ├── algorithm.py             # LoRA组合 + Nevergrad黑盒优化 + 推理 (V1/V2/V3)
│   ├── checkpoint_utils.py      # Orbax checkpoint 读写 & LoRA 提取
│   └── constant.py              # 模块路径注册表 & 自动发现
├── lora_modules/                # 已提取的 LoRA 模块 (.npz)
├── data/                        # few-shot 评测数据 (RoboTwin episode 格式)
├── run_lorahub.sh               # 优化入口脚本
├── compose_checkpoint.sh        # 组合导出脚本
├── README.md                    # 用户操作文档
└── TECHNICAL.md                 # 本文档
```

### 核心依赖

| 包 | 用途 |
|----|------|
| `orbax-checkpoint` | 读取 JAX/OCDBT checkpoint |
| `nevergrad` | 黑盒进化优化 (NGOpt/CMA-ES) |
| `numpy` | 数组存储与计算 |
| `jax` / `flax` | (间接依赖) 数组类型支持 |

---

## 8. 操作流程

### 8.1 提取 LoRA 模块

```bash
# 单个提取
python script/extract_lora.py extract \
  -c <checkpoint_dir>/ \
  -o lora_modules/task_grab_roller_29999.npz \
  -t grab_roller

# 查看已提取的模块
python script/extract_lora.py info
```

### 8.2 编写任务配置

编辑 `config/<task_name>.json`，可配置: 数据集来源 (`data_dir`)、参与组合的 LoRA 条目 (`lora_modules`，留空自动发现)、优化方案 (`algorithm.version`: v1/v2/v3)、优化预算 (`max_steps`)、diffusion 采样步数 (`diffusion_steps`) 等。字段说明见 README。

### 8.3 运行 LoRAHub 黑盒学习

```bash
bash run_lorahub.sh <task_name>
```

程序会:
1. 按配置加载 few-shot 数据与 LoRA 模块 (自动排除目标任务自身 LoRA)
2. 加载基座参数 + 所有 LoRA 模块
3. Nevergrad 黑盒优化 (预算 = max_steps)
4. 输出最优权重 `lorahub_weights.json` 到 `outputs/<task_name>/`
5. 日志写入 `outputs/<task_name>/logs/`

### 8.4 组合导出评测 checkpoint

```bash
bash compose_checkpoint.sh <task_name>
```

从 `lorahub_weights.json` 组合 LoRA 并重建为完整 Orbax checkpoint (输出到 `outputs/<task_name>/checkpoint/10000/`)，可选同步到 RoboTwin 训练配置目录供直接评测。
