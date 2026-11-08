# 本次验证记录

## 环境

- Python 3.11，`/home/huahungy/anaconda3/bin/python`
- PyTorch 2.10.0+cpu，NumPy 2.4.4，scikit-learn 1.7.2
- CPU 执行，未下载模型、未启动 GPU 训练、未修改原始数据。

## 自动化测试

以下命令通过，6 项测试全部成功：

```bash
cd /home/huahungy/RoboLoRAHub/data_clustering
XDG_CACHE_HOME=/tmp/robot_action_clustering_cache \
MPLCONFIGDIR=/tmp/robot_action_clustering_mpl \
python -m unittest discover -s tests -v
```

覆盖末端表示对全局刚体变换与四元数符号变化的不变性、HDF5 真实末帧边界、禁止混合不同机器人关节、拒绝不同采样率、按轨迹划分且标准化不使用验证数据、潜在向量单位范数、K-means/HDBSCAN 输出对应关系、末端加视觉缓存分支的完整训练与编码。

## 现有真实数据冒烟运行

配置：`../runs/aloha_smoke_config.json`；产物：`../runs/aloha_smoke/`。这是 `configs/aloha_hdf5.json` 的 3 epoch CPU 副本，未改写正式配置。

| 项目 | 结果 |
|---|---|
| 原始数据 | place_empty_cup-demo_aloha-10，10 条轨迹 |
| 窗口 | H=50，stride=25，共 53 个 |
| 训练/验证 | 42 / 11 个窗口，episode 不交叉 |
| 表示 | joint，无视觉，128 维单位范数 |
| 训练损失 | 2.00242 → 1.95182 |
| 验证损失 | 2.30637 → 2.25973 |
| K=16 / 32 | 正常拟合、分配与导出 |
| K=48 | 自动跳过：只有 42 个训练窗口 |
| HDBSCAN | min_cluster_size=20，训练窗口全部为噪声，正常保留 -1 |
| PCA 图 | 已生成并目视检查 |

验证 silhouette 因分配退化/验证样本不足而记录 `null`，不是 0，也不是缺失结果被伪造为良好指标。HDBSCAN 全噪声说明这组小样本在当前设置下没有密度簇，不能解释为发现了技能。

此运行仅确认代码与真实 HDF5 适配器可用：数据只有一个任务、训练仅 3 epoch、fps=25 仍是未核实假设，没有评估下游控制。不能据此选择最佳 K 或声称跨任务技能发现成功。
