# 机器人动作数据聚类：第一部分

在旧轨迹上构造动作窗口，学习无人工技能标签的潜在表示，再聚类形成候选行为簇。此目录是独立实验模块，不修改 `LoRAHub-JAX/`、`RoboTwin/` 或原始数据。

**已转换 LeRobot 末端数据的正式入口：**[检查结果、处理维度、全量聚类命令与产物说明](docs/lerobot_ee_clustering.md)。已完成全量数据核对及去重，50 个任务的 18,325 个窗口通过 3 epoch 全流程验证；运行 `bash run_lerobot_ee.sh` 可启动正式 30 epoch 实验。

**2026-11-04 参数复核：**完成两套 K=32 结果各 128 个片段的图像抽查、36 组参数比较及 4 次额外网络种子复核。当前较均衡的候选是 **64 维、30 轮、K=32**，详见[参数与局限报告](reports/parameter_review/report.md)、[原结果图库](reports/k32_visual_review/index.html)和[候选图库](reports/k32_latent64_review/index.html)。已导出的候选结果在 `runs/lerobot_ee_reviewed64/`；重跑使用 `bash run_lerobot_ee.sh configs/lerobot_ee_reviewed64.json`，写入新目录 `runs/lerobot_ee_reviewed64_rerun/`。尚未通过独立任务或下游控制成功率验证，不能将候选簇直接视为技能真值。

**低权重图片辅助版本：**`bash run_visual_aux.sh` 从 Parquet 内嵌 PNG 提取主相机首尾帧，用冻结的预训练 ResNet18 和仅训练集拟合的 PCA 得到 8 维视觉表示，追加到现有 64 维动作表示。默认平方距离系数为动作 95%、视觉 5%，并比较 0/2/5/10% 视觉权重。详见[编码、权重、运行及产物说明](docs/visual_aux_clustering.md)。使用独立结果目录和可复用图片特征缓存，原纯动作结果保持可对照。

设计依据是本次提供的第一部分文字。飞书链接未能读取，消息中的 `[图片]` 也没有实际图像内容，因此这里的公式、数据约定与工程边界见 [设计说明](docs/design.md)，不声称逐字复现原文或 OPAL / AtomSkill。

## 已实现

- HDF5 / NPZ / 当前转换器生成的 LeRobot EE Parquet 轨迹读取；默认 **50 时间步窗口、25 步滑动间隔**，窗口不跨 episode。
- `joint` 同构机器人关节动态表示，及 `ee` 末端相对位姿表示。跨机器人禁止直接拼接原始关节角。
- 动作时序 MLP、状态变化 MLP、可选缓存视觉特征分支；两层投射 MLP、可配置潜在维度（已比较 64 / 128 维）、L2 归一化。
- 自监督动作动态重建与状态变化重建/预测损失；按 episode 划分验证集，仅用训练集统计归一化参数。
- K-means 候选 `K=16,32,48`；可选 HDBSCAN 噪声簇 `-1`。
- 保存模型、归一化参数、潜在向量、簇分配、中心、代表窗口、来源索引、指标及 PCA 图。

**当前是可运行的研究基线。** π0 视觉编码器在线提取、对象跟踪、不同采样率重采样、自动时间边界发现、专家 LoRA 训练和下游成功率评测尚未实现。视觉缓存关闭时运行双分支消融，不用随机向量代替视觉特征。

本次已通过 6 项测试，并用工作区 10 条真实轨迹跑通 3 epoch CPU 冒烟验证；结果与限制见 [验证记录](docs/verification.md)。

## 目录

```text
data_clustering/
├── action_clustering/
│   ├── __main__.py        # CLI 与合成数据演示
│   ├── config.py          # 配置校验、相对路径解析
│   ├── data.py            # 适配器、相对动作表示、窗口缓存和轨迹划分
│   ├── model.py           # 编码器、投射器、重建头
│   ├── learning.py        # 训练、验证、潜在向量导出
│   └── clustering.py      # 聚类、代表窗口与诊断图
├── configs/
│   ├── aloha_hdf5.json    # 工作区现有 place_empty_cup 数据
│   └── canonical_ee.json  # 统一末端 NPZ 数据接口模板
├── docs/design.md
├── tests/test_pipeline.py
└── requirements.txt
```

## 快速运行

```bash
cd /home/huahungy/RoboLoRAHub/data_clustering
# 在你选择的 Python 环境中安装依赖（现有环境可直接运行）
python -m pip install -r requirements.txt

# 合成数据端到端示例：12 条轨迹，3 个 epoch，CPU
python -m action_clustering demo --output runs/demo

# 有针对性的测试，不需要下载模型或访问网络
MPLCONFIGDIR=/tmp/robot_action_clustering_mpl python -m unittest discover -s tests -v
```

`demo` 创建的新目录包含合成轨迹、配置和完整输出。它验证流程能运行，不代表真实技能发现效果。所有阶段拒绝覆盖已有阶段目录；重复实验请使用新的 `output_dir`。中途失败的目录也不会被静默覆盖。当前没有断点续训功能。

## 使用工作区的真实 HDF5 数据

```bash
python -m action_clustering run --config configs/aloha_hdf5.json
```

此配置指向已存在的 `RoboTwin/policy/pi0/training_data/place_empty_cup-demo_aloha-10`。**配置中的 `fps=25` 是示例假设，未从这些 HDF5 元数据中确认；正式实验前必须核实并修改。** 同一组数据内部改变该值只影响时间元数据，程序不会根据 fps 自动重采样。50 步是一个动作块的长度，不是 50 个动作块。

检查到的 HDF5 结构为：

```text
action                              [T, 14]
observations/qpos                   [T, 14]
observations/images/cam_high         [T] JPEG bytes
observations/images/cam_left_wrist   [T] JPEG bytes
observations/images/cam_right_wrist  [T] JPEG bytes
```

`gripper_indices=[6,13]`，其余为左右臂关节。项目里的 `process_data.py` 将下一帧关节状态写作当前动作目标，因此配置显式声明 `absolute_joint_target`。这一动作语义不能直接用于扭矩、速度或其他机器人数据。

现有 HDF5 的状态与动作同为 T 帧，缺少最后一个动作后的真实观测。因此实际只使用前 T−1 个动作；50 步窗口需要 51 个真实状态。不会把目标动作冒充为测量到的末端状态。短于窗口的轨迹记录到 `metadata.json` 并跳过，未满整窗的尾部丢弃。

可以分阶段运行，后续阶段读取前一阶段产物：

```bash
python -m action_clustering prepare --config configs/aloha_hdf5.json
python -m action_clustering train   --config configs/aloha_hdf5.json
python -m action_clustering embed   --config configs/aloha_hdf5.json
python -m action_clustering cluster --config configs/aloha_hdf5.json
```

配置路径相对于当前工作目录；配置内部的 `root`、`visual_cache`、`output_dir` **相对于该配置文件所在目录**。设备默认 `cpu`，有兼容的 PyTorch GPU 环境时可以改成 `cuda:0`。

## 通用末端数据接口

`configs/canonical_ee.json` 是待填入实际数据的模板。每个 NPZ 对应一条 episode，必须不含 pickle/object 数组：

| 字段 | 形状 | 含义 |
|---|---|---|
| `actions` | `[T, A_raw]` | 原机器人动作，必须保留；不同 robot_id 可不同维度 |
| `gripper` | `[T+1, arms]` | 实测夹爪状态，事先统一开合方向与尺度 |
| `ee_pose` | `[T+1, arms, 7]` | `xyz + qx,qy,qz,qw`，米、单位四元数、统一臂角色顺序 |
| `qpos` | `[T+1, Q]`，可选 | 原始关节状态；joint 模式必须提供 |
| `object_features` | `[T+1, O]`，可选 | 预先提取且语义对齐的物体相对位置/目标距离等 |

`ee` 模式对每只手臂使用窗口起始末端坐标系：相对平移与旋转向量；不直接相减四元数。不同机器人必须先对齐单位、采样率、臂/对象槽位与夹爪语义。单臂与双臂尚不能自动混合，需要另做槽位与 mask 适配。没有 FK/末端数据时程序报错，不会从关节维度猜测末端位姿。

## π0 视觉特征接口

先由外部特征提取流程按帧生成 `[T, D_v]` 的 `.npy` 缓存。每个 dataset 增加：

```json
"visual_cache": "../data/pi0_visual_features"
```

并设置：

```json
"representation": {"mode": "joint", "visual_weight": 0.1}
```

缓存路径与原文件相对路径对应，例如 `episode_0/episode_0.hdf5` 对应 `episode_0/episode_0.npy`。NPZ 同样替换后缀为 `.npy`。仅读取窗口起始帧特征，多相机池化/拼接应在外部保持统一。开启后缺少缓存将直接报错。

请为缓存另存模型 checkpoint、视觉层选择、池化方式、相机顺序和预处理版本。编码器应固定，以便比较不同 K 的效果。此基线没有调用 π0 的视觉编码器，也没有额外下载模型。低视觉权重与 dropout 是抑制视觉主导的启发式，不能保证背景不影响簇；需要做有/无视觉消融。

## 输出与解释

```text
<output_dir>/
├── windows/
│   ├── index.jsonl          # 每个窗口的 source/episode/start/end/robot/split/instruction
│   ├── metadata.json       # 形状、参数快照、跳过轨迹、索引校验
│   ├── action.npy          # 规范化之前的动态特征；mmap 文件
│   ├── state.npy           # 状态变化目标
│   ├── visual.npy          # 仅启用视觉时存在
│   └── episodes/*.npz      # 原始动作及起止状态的数值数据缓存
├── training/
│   ├── best.pt             # 验证损失最优权重
│   ├── normalization.npz   # 只在训练窗口上计算
│   ├── config.json
│   └── history.json
├── embeddings/
│   ├── z.npy               # 与 index.jsonl 同序，[N,128 或 256]，单位范数
│   └── metadata.json
└── clusters/
    ├── comparison.json
    ├── kmeans_k16/          # k32、k48 同结构，样本不足则记录 skipped
    │   ├── labels.npy
    │   ├── assignments.jsonl
    │   ├── centers.npy
    │   ├── cluster_means.json
    │   ├── representatives.json
    │   ├── metrics.json
    │   └── latent_pca.png
    └── hdbscan/             # 仅训练集分配，额外保存 window_ids/probabilities
```

所有窗口动作范围为 `[start,end)`，状态端点为 `state[start]`、`state[end]`。`episode_cache` 相对于 `windows/`；读取缓存中的 `actions[start:end]` 可取得原动作，图像通过 `source_path + image_key + start` 定位，避免重复存储大规模图片。NPZ 图像索引需要由上游保留，此版本的 NPZ 接口只承载数值。

K-means 在训练窗口拟合，预测所有窗口；HDBSCAN 使用 sklearn 实现，没有验证集 `predict`，因此只导出训练窗口，`labels.npy` 必须结合 `window_ids.npy` 使用。`-1` 仅表示 HDBSCAN 噪声；没有把未预测的验证窗口误记为噪声。

代表窗口按到簇训练样本均值的距离选取，输出含来源、指令与时间范围，供事后回放/语义命名。原任务指令仅保留为元数据，不作为类别监督。PCA 只是投影诊断，聚类实际在完整潜在空间计算。

`comparison.json` 提供簇大小、噪声比例、验证 silhouette 和验证中心距离等。**这些无监督指标不能替代下游控制成功率，也不能自动确认最佳 K。** 本模块不自动给簇命名为“抓取”“放置”等人工动作类别。

## 数据规模与实验边界

窗口数值使用 mmap，准备阶段按轨迹两遍处理，不一次加载全部图片。训练按 batch 读取；但元数据索引仍驻内存，标准 K-means/HDBSCAN 也不是分布式或无限规模算法。百万级窗口需要进一步实现分片索引、MiniBatchKMeans、抽样密度聚类与增量特征提取。

至少需要两条足够长的 episode。多任务实验建议每任务提供多条独立轨迹，避免把同一条轨迹的不同转换副本作为不同数据源。当前随机 episode 留出用于流程验证；跨任务/跨物体/跨场景泛化需另加分组留出实验。
