# 七个相关簇的数据提取

选择簇 **8、24、3、6、9、31、23**（0 起始编号），来自 `mlp_temporal_20261109_230731_563725/evaluation/lambda0_03__calibrated5/k32`。不重新训练或聚类。默认检索原始全库 50 个任务；100 条五任务子集用于选择相关簇，不限制专家数据来源。

## 提取命令

```bash
cd /home/huahungy/RoboLoRAHub
RoboTwin/policy/pi0/.venv/bin/python -u \
  data_clustering/scripts/extract_cluster_lerobot.py \
  --scope global --clusters 8 24 3 6 9 31 23
```

仅查看规模和预计空间，加 `--dry-run`。无需 GPU。默认七簇共 5,082 个窗口，254,100 个存储帧，预计 Parquet 约 26 GiB；实际大小可能有偏差，建议预留 35 GiB。每簇每完成 50 个窗口输出进度、耗时和预计剩余时间，初期估计会波动。

需要断开 SSH 后继续：

```bash
cd /home/huahungy/RoboLoRAHub
mkdir -p data_clustering/logs
nohup RoboTwin/policy/pi0/.venv/bin/python -u \
  data_clustering/scripts/extract_cluster_lerobot.py \
  --scope global --clusters 8 24 3 6 9 31 23 \
  > data_clustering/logs/extract_seven_clusters.log 2>&1 &
tail -f data_clustering/logs/extract_seven_clusters.log
```

原源数据不会更改。每个簇先写入独立 `.partial` 目录，完成后重命名。重跑会验证并跳过配置匹配的已完成簇，自动重建本脚本产生、请求参数一致的未完成簇；不支持同时启动两个相同输出目录的提取进程。

可用 `--scope five-full` 限制到五任务全量 250 条轨迹，或 `--scope subset20` 限制到已抽取的 100 条；两者自动使用不同输出目录。不要将不同 scope 的簇数据当成同一训练实验。

## 输出结构

```text
data_clustering/data/cluster_exports/lambda003_calibrated5_k32_global/
├── cluster_08_pi0_ee/
├── cluster_24_pi0_ee/
├── cluster_03_pi0_ee/
├── cluster_06_pi0_ee/
├── cluster_09_pi0_ee/
├── cluster_31_pi0_ee/
├── cluster_23_pi0_ee/
└── export_summary.json
```

每个目录是独立的 LeRobot v2.1 数据集：

- `data/chunk-*/episode_*.parquet`：一窗口一片段，每片段 50 帧。三路相机 RGB 字节、语言指令、16 维双臂末端状态/动作保留；动作仍是下一时刻记录的末端状态。最后一个动作保留真实记录目标，不改成零或停留动作。
- `meta/info.json`、`episodes.jsonl`、`tasks.jsonl`、`episodes_stats.jsonl`：重新编号，按片段重算统计；训练片段在前、验证片段在后。
- `meta/source_windows.jsonl`：簇、原窗口、原轨迹、起止帧、原划分、源与目标文件 SHA256。
- `meta/sample_starts.jsonl`：后续训练每个片段唯一合法的起点索引。
- `meta/train_window_stats.json`：仅训练片段的起始状态和全部 50 步动作的原始 16 维统计；验证片段不参与。
- `meta/cluster_export.json`：配置、完成标志、规模和训练采样要求。

窗口长度 50、步长 25，可能重叠，也可能含动作转换。标签属于整个窗口，**不代表窗口内每帧都具有纯净的原子动作语义**；不同簇也可能含同一原轨迹的重叠帧。原轨迹的 train/validation 划分在所有簇中一致，不重新随机划分片段。

## 后续 LoRA 训练必须配套的采样

本次准备数据与采样接口，**没有启动或改写现有五任务 LoRA 训练**。

不能直接沿用逐帧采样：50 帧片段从中间采样 50 步动作会出现大量重复尾帧。使用新增 `action_clustering.cluster_window_dataset.load_cluster_windows(root, split)`，每个窗口只从起点取一次 50 步完整动作块：

```python
from action_clustering.cluster_window_dataset import load_cluster_windows

train_dataset = load_cluster_windows('/path/to/cluster_08_pi0_ee', split='train')
sample = train_dataset[0]  # images + state[16] + action[50,16] + task
```

Python 路径需包含 `data_clustering` 和 `RoboTwin/policy/pi0/src`。该读取器仅在自己的视图内对单列查询做投影，避免读取动作/时间戳时反复解码无关图像；完整观测仍解码三路图像。

后续 π0 训练应在此数据集上应用 EE 重打包、语言 tokenization 和模型维度填充；将训练片段的统计转换成对应模型的 `norm_stats.json`。`train_window_stats.json` 是原始 16 维统计，不是可直接替换旧五任务 checkpoint 的归一化文件；也不能使用包含验证片段的全数据聚合统计来拟合训练归一化。

## 验证

仅做小规模测试（独立 `_smoke` 目录，每簇 train/validation 各一个窗口）：

```bash
RoboTwin/policy/pi0/.venv/bin/python -u data_clustering/scripts/extract_cluster_lerobot.py --smoke-test
RoboTwin/policy/pi0/.venv/bin/python -u data_clustering/scripts/validate_cluster_exports.py
```

全量导出后可选执行完整源数据对比与抽样加载验证（会额外读写较大的临时 HF Arrow 缓存，需要时间与临时空间）：

```bash
RoboTwin/policy/pi0/.venv/bin/python -u data_clustering/scripts/validate_cluster_exports.py \
  --root data_clustering/data/cluster_exports/lambda003_calibrated5_k32_global
```
