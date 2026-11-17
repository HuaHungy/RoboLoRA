# 五任务物理合并、20 条子集与相关簇选择

本功能新增独立 LeRobot v2.1 数据集，保留原任务目录及现有 π0 联合训练入口。原入口是在运行时用 `ConcatDataset` 合并五个源数据集；本功能将数据真正写成一个可搬运、可由 LeRobot 读取的目录。

任务按固定顺序：`adjust_bottle`、`click_alarmclock`、`place_can_basket`、`place_container_plate`、`rotate_qrcode`。

## 运行

在 RoboLoRAHub 根目录运行，使用已有 π0 环境（需要 numpy、pyarrow）：

```bash
RoboTwin/policy/pi0/.venv/bin/python -u \
  data_clustering/scripts/prepare_five_task_subset.py

# 可选：用实际 π0 LeRobot 加载器检查全部轨迹首尾、50 步动作块和聚合统计
RoboTwin/policy/pi0/.venv/bin/python -u \
  data_clustering/scripts/validate_five_task_datasets.py
```

默认使用 `runs/mlp_temporal_20261109_230731_563725/evaluation/lambda0_03__calibrated5/k32`。如果搬运到服务器，需要一起搬运该 run 的 `index.jsonl` 和该分支的 `z.npy`、`visual_distance.json`、K32 的标签、中心及 assignments。脚本支持 `--run`、`--source-root`、`--output-root`、`--report`。聚类索引中的源路径来自旧机器，因此本入口的完整来源核验应在原工作区执行；**生成的合并数据集可以单独搬运并加载，不依赖原始 HDF5 或旧绝对路径**。

每任务从原 50 条轨迹中用 NumPy `default_rng(42)` 顺序抽 20 条（不放回），按原轨迹编号排序写入；抽样不参考簇标签。原有 train/validation 标签只用于报告，不用于本次抽样。新增数据集 `splits.train` 表示所有可用演示，不意味着新建了独立验证集。

## 目录

```text
data_clustering/
├── data/lerobot/<task>_pi0_ee/          # 原五个数据集，保留
├── data/merged/
│   ├── robotwin_five_pi0_ee_full/       # 250 条，39,505 帧
│   └── robotwin_five_pi0_ee_subset20_seed42/ # 100 条，15,789 帧
│       ├── data/chunk-000/episode_*.parquet
│       └── meta/
│           ├── info.json / tasks.jsonl / episodes.jsonl
│           ├── episodes_stats.jsonl
│           ├── robotwin_ee_conversion.json
│           ├── source_episodes.jsonl   # 原轨迹→新轨迹映射及文件 SHA256
│           └── merge_manifest.json
└── reports/five_task_subset20_lambda003_calibrated5_k32/
    ├── analysis.md / summary.json / cluster_ranking.csv
    ├── dataset_summary.json / loader_validation.json
    ├── subset_windows.jsonl
    ├── subset_top5_windows.jsonl
    ├── global_top5_windows.jsonl
    └── subset_representatives.json
```

图像字节仍嵌在 Parquet 中，三路相机、原指令及双臂 16 维末端 state/action 原样保留。轨迹索引、全局帧索引、语言指令索引重新编号，对应统计同步更新。LeRobot `total_tasks` 统计的是独立指令字符串数量，不是 RoboTwin 的五个任务数量。

完整数据写入 `.partial` 目录后再重命名；已有完成数据集只有抽样配置一致且源/目标 Parquet SHA256 全部一致才复用，拒绝覆盖不兼容数据。若执行中断，保留 `.partial` 便于检查，确认后可自行清理再运行。

## 相关性定义

不重新拟合编码器或 K-means，而是通过原任务、原轨迹、窗口起止帧匹配原有标签。主评分为五个任务各自簇窗口占比的平均值，保持五任务等权。保存原始窗口覆盖率、相对全库的富集倍数、全量五任务和剩余轨迹的排名，并按任务重采样整条轨迹 1000 次检查前五排名敏感性。

默认子集前五是 **8、24、3、6、9**（从 0 编号）；第六名 31 与第五名很接近。该结果是指定子集上的检索排序，并非五种已确认的语义技能，也不是下游策略成功率。详细数值见生成的 `analysis.md`。

`global_top5_windows.jsonl` 仅是原全库候选窗口清单，不会再复制这些簇的数据。它包含五任务之外的原始任务。`subset_top5_windows.jsonl` 只包含 100 条子集中的命中窗口。完整子集仍保留所有动作，不因未入选五簇而删除轨迹内容。

新数据集可用当前 `openpi.training.lerobot_compat.LeRobotDataset(repo_id, root=...)` 读取。现有 `pi0_ee_five/run.sh` 仍固定读取原五个目录，尚未改为训练这个小子集；不能仅把 `--dataset-root` 指向 `data/merged` 就期望旧入口自动切换。后续切换训练数据需要选择对应 repo_id，并重新计算该数据集的共享归一化统计。

验证脚本使用 π0 的 LeRobot 读取、语言解析和动作块边界逻辑；默认仅在验证进程给单列读取增加列投影，避免检查时间戳和 action 列时解码无关图像。每条轨迹首尾的完整观测仍走原图像解码。此加速不修改生产加载器；加 `--original-column-reader` 可使用原单列读取路径，但其初始化会反复解码整库图像，耗时明显更长。
