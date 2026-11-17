# MLP 时间一致性：系数、视觉距离与夹爪事件复核

旧的 TCN、重叠对齐训练、动态规划标签平滑及其实验入口已删除。历史 `runs/temporal_...`、报告、模型与标签保留。历史平滑标签仅作为只读复核对象，后续不再运行平滑算法。

## 推荐命令

```bash
cd /home/huahungy/RoboLoRAHub/data_clustering
bash run_mlp_temporal.sh --check-only
bash run_mlp_temporal.sh --epochs 100 --coefficients 0 0.03 0.1 --k 32 48
```

默认只训练 **3个MLP，每个100轮**。原结构、动作/状态标准化、训练验证episode划分、batch=256、lr=0.001、seed=42和每轮样本顺序保持一致。系数为0是匹配对照；0.03与0.1用于检验更弱约束是否保留运动一致性并恢复夹爪变化信息。按验证动作+状态重建MSE最低选best.pt，同时保存last.pt与完整历史。

每个动作模型都评估以下三个视觉设置，**不额外训练网络或重编码图片**：

1. `fixed5`：原有公式 `sqrt(.95)*motion + sqrt(.05)*visual`（实际是向量拼接）。
2. `calibrated5`：用固定随机抽取的训练窗口对校准视觉系数，使“平均视觉平方距离/平均总平方距离”为5%。公式 `alpha=target*M/((1-target)*V+target*M)`，M/V是未加权的训练动作/视觉平方距离均值。alpha上限0.1；记录是否触及上限及实际达到的比例。验证集不参与校准。
3. `motion_only`：只用64维动作编码，排除视觉贡献变化造成的混淆。

每组保存**验证集逐对视觉比例均值**与**平均距离之比**，两者不是同一指标；与历史“视觉平均贡献19%”比较时使用前者。校准模式的5%是训练集均值之比目标，不保证验证集逐对均值也恰为5%。

如希望增加夹爪保护方向：

```bash
bash run_mlp_temporal.sh --epochs 100 --coefficients 0 0.03 0.1 --k 32 48 --with-gripper-guard
```

增加第4个模型 `lambda0_03_guard`：如果相邻窗口任一窗口中任一夹爪的累计绝对变化达到0.2，就关闭这对窗口的全局时间拉近约束。这是防止缓慢夹爪开合被逐步阈值漏掉的单独消融，不改变重建目标，也不自动定义“抓取/释放”语义。记录关闭了多少训练对；不能仅追求切换率下降。

## 夹爪事件复核

启动训练前，扫描验证轨迹的真实夹爪曲线。将同方向、间隔不超过2步的有效增量（单步绝对值≥0.005）合成事件，净开合变化≥0.2才进入候选。左右臂×开/闭分4层，每层随机抽8个来自不同轨迹的事件（不足则按实际数量），固定review seed=20261110。采样不使用任何聚类标签，所有模型使用完全相同样本。

训练结束后导出：头部相机GIF、事件前/中/后真实图像、双臂夹爪曲线、各方案窗口标签、原始窗口来源及人工复核表。图片从LeRobot Parquet内嵌字节读取；不会为只存在数值状态的轨迹终点虚构图像。图像缺失会记录错误和完整数量，不伪造成功；`--skip-images`明确只做数值复核。

对照同时包含原实验中的未平滑30轮基线、平滑强度2的基线、MLP系数0.1结果。默认历史参考：
`runs/temporal_20261109_222250_840101`，可用 `--reference-run /path/to/run` 指定具备相同窗口映射的完整历史实验。

仅导出当前历史方案的随机片段，不训练新模型：

```bash
bash run_mlp_temporal.sh --review-only --k 32 48
```

“事件中心两侧窗口仍同簇”只用于筛查，不等于技能被合并：50步窗口可能都覆盖同一开合事件。另提供完全位于事件前后的窗口标签，但其间可能相隔较远甚至存在其他事件。应观察GIF确认短行为和物体交互，填写 `manual_review.csv`。没有人工填写时，语义复核状态保持 `pending_human_inspection`。

## 输出与运行选项

```text
runs/mlp_temporal_<时间戳>/
  experiment.json / report.md / summary.json / metrics_k*.png
  models/lambda0、lambda0_03、lambda0_1/  # checkpoint、标准化、历史、64维向量
  evaluation/<模型>__<视觉设置>/k32、k48/
    labels.npy / centers.npy / assignments.jsonl / representatives.json / metrics.json
  evaluation/<模型>__<视觉设置>/visual_distance.json
  distance_pairs.npz                     # 固定训练/验证距离抽样对
  boundary_pairs.npz / boundary_gate.json
  review_selection_before_training.json  # 提前固定的随机事件
  all_validation_events.json / event_summary.json
  historical_provenance.json             # 旧标签路径和哈希
  gripper_review/
    index.html / assets/*.gif、*.jpg、*_gripper.png
    selected_events.json / manual_review.csv / status.json
```

每次新建目录，保留历史结果，不自动续训。CPU默认4线程，与原有环境一致；进度ETA只涵盖当前模型。环境可用 `CLUSTER_PYTHON=/path/to/python` 替换；可选 `--device cuda`，不自动安装软件。

- `--k 16 32 48`：增加K16对照，不增加模型训练次数。
- `--visual-modes fixed5`：只跑原视觉融合方式。
- `--seed 7`：做另一组匹配训练seed；历史结果仍明确标记为历史参考。
- `--events-per-group 4`：每层最多4个随机事件，缩小图像产物。
- `--smoke-test --with-gripper-guard`：16条完整轨迹、每模型2轮、K3，仅功能验证，图像标记SMOKE。

SSH后台运行：

```bash
cd /home/huahungy/RoboLoRAHub/data_clustering
mkdir -p logs
nohup bash run_mlp_temporal.sh --epochs 100 --coefficients 0 0.03 0.1 --k 32 48 > logs/mlp_launcher_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
```

依赖原窗口及episode缓存、冻结视觉向量、历史标签；生成GIF还需要索引所指向的LeRobot Parquet图片。已准备好检查入口，但预检查不解码全量图像；最终图像成功/缺失数量见 `gripper_review/status.json`。
