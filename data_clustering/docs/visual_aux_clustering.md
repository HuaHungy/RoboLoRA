# 低权重图片辅助聚类

这一版本在已经复核的 **64 维动作表示**后追加 **8 维视觉表示**，形成 72 维聚类输入。动作网络沿用原来训练 30 轮的 checkpoint，不重新训练。视觉编码器使用本地 ImageNet 预训练 ResNet18，冻结全部参数。不使用语言指令、任务名称、人工技能标签或物体位姿。

## 图片在哪里

当前 LeRobot Parquet 包含三个图像列：

- `observation.images.cam_high`
- `observation.images.cam_left_wrist`
- `observation.images.cam_right_wrist`

每列是 `struct<bytes: binary, path: string>`。当前转换器把 RGB 图片编码成 PNG，字节实际存放在 `bytes` 字段里；`path` 是名称元数据，不需要另找这个名称对应的图片文件。

本次先使用主相机 `cam_high`。程序按轨迹读取该列，只解码每个窗口需要的首尾画面，避免重复导出图片。窗口动作范围是 `[start,end)`，因此图片使用 `start` 和 `end-1`。数字状态可以恢复到 `end`，但 Parquet 最后一个动作之后的终止图像不存在；程序不捏造或复制终止图像。50 步动作窗口的首尾图像间隔是 49 个记录帧。

## 编码与权重

1. PIL 按 RGB 解码 PNG，整幅画面双线性缩放到 224×224，保留完整视野但改变宽高比；不做中心裁剪。使用 ImageNet 均值、标准差归一化。
2. 冻结 ResNet18，取分类头之前的全局平均池化向量，每张图 512 维，并做 L2 归一化，得到 `h_start` 和 `h_last`。
3. 构造上下文 `(h_start+h_last)/2` 与变化 `h_last-h_start`。分别只在训练窗口拟合 PCA，各压缩到 4 维，分别归一化。
4. 将两部分按 `sqrt(0.25)` 和 `sqrt(0.75)` 拼接并归一化，得到 8 维视觉向量 `v`。变化分支获得较大内部系数，但图像变化也可能由机械臂运动或遮挡造成，并不等同于物体效果。
5. 与原来的单位范数 64 维动作向量 `m` 拼接：

```text
z = concat(sqrt(1-alpha) * m, sqrt(alpha) * v)
alpha = 0.05
```

两点的平方欧氏距离严格满足：

```text
distance(z_i,z_j)^2
  = 0.95 * distance(m_i,m_j)^2
  + 0.05 * distance(v_i,v_j)^2
```

8 维视觉不会再经过可训练的融合 MLP，因而不会被后续网络重新放大。**5% 是平方距离的系数，不是每对样本视觉贡献恰好 5%，也不是只有 5% 的样本会改变标签。** 两个分支的实际距离分布会影响实际贡献；对照报告同时记录随机验证样本对的平均视觉距离贡献比例。

这一模式与旧 `representation.visual_weight` 的训练前融合接口不同。后者是特征进入可训练投射 MLP 前的缩放，不能直接保证最终聚类距离中的视觉比例。本次使用独立入口实现可明确控制的距离融合，不修改旧实验。

## 运行

```bash
bash /home/huahungy/RoboLoRAHub/data_clustering/run_visual_aux.sh
```

本次全量结果已完成，见 [实测解释](../runs/lerobot_ee_visual_aux/analysis.md) 和 [权重对照](../runs/lerobot_ee_visual_aux/report.md)。需要复现时使用新输出目录的配置，复用已提取的图片缓存：

```bash
bash /home/huahungy/RoboLoRAHub/data_clustering/run_visual_aux.sh /home/huahungy/RoboLoRAHub/data_clustering/configs/lerobot_ee_visual_aux_rerun.json
```

配置：`configs/lerobot_ee_visual_aux.json`。路径相对于配置文件；默认使用当前可用的 CPU PyTorch 环境。额外依赖 `torchvision` 已在该环境中验证。预训练权重使用已有的 `/home/huahungy/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth`，不会自动下载，也不会在缺少权重时退化成随机网络。

脚本显示图片编码的已完成数、百分比、耗时和 ETA，并记录到 `logs/visual_aux_*.log`。按 episode 原子写入图片特征缓存，图片提取中断后可用同一命令复用已完成缓存。若最终结果目录已经存在，程序拒绝覆盖；重新实验须在配置中更换 `output_dir`，可继续复用同一个 `feature_cache`。缓存校验模型 SHA256、预处理、相机、源文件路径/大小/修改时间与帧索引，不兼容时需指定新的缓存目录。

## 产物与对照

```text
data/visual_cache/resnet18_head_start_last_v1/
  encoder.json                 # 模型哈希、预处理、版本
  <episode-hash>.npz           # 所需帧的 512 维编码，不复制 PNG

runs/lerobot_ee_visual_aux/
  visual_config.json           # 本次图片提取与融合配置
  windows/                    # 指向已有动作窗口的链接
  training/                   # 指向已有动作网络和归一化参数的链接
  embeddings/
    z.npy                     # [18325,72]，实际用于聚类
    visual.npy                # [18325,8]
    visual_pca.npz            # 训练集拟合的 PCA 均值和投射矩阵
    metadata.json             # 编码器、融合公式、基线和索引溯源
    image_provenance.jsonl    # 每个窗口的 Parquet、相机和帧索引
  clusters/kmeans_k32/        # 标签、中心、代表片段、指标和 PCA 图
  comparison/
    metrics.json              # 0%、2%、5%、10% 视觉系数的对照
    labels_visual_*.npy
  report.md
  timing_visual.json
```

PCA 与 K-means 均只在原有训练集拟合，验证集保持不变。默认正式输出采用 5%，其余权重用于对照，不按验证轮廓系数自动宣称语义最优。报告包含融合空间及原动作空间的轮廓系数、K-means 三次不同初始化的 ARI、与纯动作分组的 ARI、动作特征的簇均值重建 R²。

`training/` 仍是原动作网络，不能单靠它重建 72 维融合表示；需同时保留视觉编码器、PCA 参数和融合配置。`windows/`、`training/` 是链接，必须保留其指向的原目录。通用 `train/embed` 命令不用于重建这个视觉辅助实验，重跑应使用上面的独立入口。

这些指标没有人工技能真值，不能作为语义准确率。加入视觉后，还需要检查“运动相似但交互不同”的代表与随机验证片段，确认是否改善了实际分组。主相机全局特征也不能保证识别遮挡的小物体；此版本不包含对象跟踪或视觉语言模型。

## 比较 K=16、32、48

直接复用已完成的 72 维融合向量，固定 5% 视觉距离系数，分别运行三种 K；无需重训或重新读取图片：

```bash
bash /home/huahungy/RoboLoRAHub/data_clustering/run_visual_k_comparison.sh --k 16 32 48
```

若只运行一种 K，可传入 `--k 16`、`--k 32` 或 `--k 48`。默认每次新建 `runs/visual_k_comparison_<时间戳>/`，打印真实输出位置并记录到 `logs/visual_k_comparison_*.log`；也可用 `--output` 指定尚不存在的目录。

每种 K 还会拟合相同 K 的纯动作对照，计算三次额外 K-means 初始化的稳定性、验证轮廓系数、位置/姿态/夹爪特征的簇均值重建 R²、簇大小。主报告为 `report.md`，完整数值为 `k_comparison.json`；各视觉辅助结果在 `clusters/kmeans_k16/`、`kmeans_k32/`、`kmeans_k48/`，包含标签、中心、代表窗口来源和 PCA 图。原始向量及训练数据以链接共享，须保留源目录。
