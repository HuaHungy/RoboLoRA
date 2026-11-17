# 七簇 π0 LoRA 训练适配

本目录独立于 `pi0_ee_five`。它分别训练簇 `8、24、3、6、9、31、23` 的七个 LoRA，不修改五任务联合模型、原始 LeRobot 数据或上游配置注册表。

## 已解决的数据接口问题

- 每个导出的50帧片段只从第0帧产生一个样本：三路起始图像、16维起始末端状态、原语言指令和完整的50步16维末端动作。
- 训练加载器仅打开 `meta/sample_starts.jsonl` 中的训练 episode；验证片段不进入训练。
- 所有动作目标均为真实记录值，不使用片段末端重复填充。
- 每簇归一化只使用训练窗口：状态统计每窗口起点一次，动作统计每窗口50步；16维末端量按模型输入规则零填充到32维。
- 每簇独立 repo id、归一化资产、checkpoint 和完成标记。
- 为与 `LoRAHub-JAX` 的提取和组合前提严格一致，仅20个路径含 `lora` 的参数叶子参与优化；视觉、LLM基座、动作专家基座以及 state/action 控制投影全部冻结。

## 默认训练参数

相比五任务联合配置（batch 32、20,000步、peak LR `2.5e-5`），小规模动作簇默认使用：

| 簇 | 训练窗口 | batch | 更新步数 | 等效遍历 |
|---:|---:|---:|---:|---:|
| 8 | 736 | 16 | 2,300 | 50.0 |
| 24 | 366 | 16 | 1,200 | 52.5 |
| 3 | 1,019 | 16 | 3,200 | 50.2 |
| 6 | 403 | 16 | 1,300 | 51.6 |
| 9 | 409 | 16 | 1,300 | 50.9 |
| 31 | 353 | 16 | 1,100 | 49.9 |
| 23 | 795 | 16 | 2,500 | 50.3 |

共同参数：peak LR `1e-5`、warmup 100、action horizon 50、每500步保存、默认只保留最新 checkpoint、仅训练LoRA、EMA关闭。步数按 batch 16 下约50次遍历设置，是第一轮对照参数，不代表已验证最优收敛点。覆盖 batch size 时必须同步按 `steps = ceil(epochs × train_windows / batch_size)` 调整步数。

参数集中在 `config.json`。命令行可以使用 `--batch-size`、`--peak-lr`、`--warmup-steps`、`--save-interval`、`--num-workers`、`--fsdp-devices`。改变 batch 时推荐传入 `--epochs E`，入口会按照每簇训练窗口数计算 `ceil(E × windows / batch)`；它不能与步数覆盖同时使用。`--train-steps N` 覆盖本次所选全部簇，重复传入 `--cluster-step 8=1500` 可单独覆盖。

本地24GB RTX 4090已分别用 `batch=2、4、8` 完成真实的一步前向、反向和checkpoint保存，推荐使用已验证的最大值 `batch=8`；默认 `batch=16` 面向服务器，未在本机验证。建议先以20轮作为七簇基线，入口会自动换算为总计10,205个更新步：

```bash
bash RoboTwin/policy/pi0_cluster_lora/run.sh train \
  --gpu 0 --batch-size 8 --epochs 20 --fsdp-devices 1 \
  --run-name cluster7_lora_local_e20 \
  --pretrained-params /home/huahungy/.cache/openpi/openpi-assets/checkpoints/pi0_base/params
```

## 数据和模型文件

服务器仓库中需要：

```text
data_clustering/data/cluster_exports/lambda003_calibrated5_k32_global/
├── cluster_08_pi0_ee/ ... cluster_23_pi0_ee/
└── export_summary.json

transfer/openpi/
├── big_vision/paligemma_tokenizer.model
└── openpi-assets/checkpoints/pi0_base/params/
```

每个簇必须完整搬运 `data/` 和 `meta/`。七簇数据约25 GiB。π0基座约12 GiB；七个最终 checkpoint 按此前实测约需37 GiB，加上HF Arrow缓存、临时checkpoint和环境，建议至少预留100 GiB。

## 准备与检查

在仓库根目录执行：

```bash
cd /workspace/RoboLoRA
bash RoboTwin/policy/pi0/server_train_clusters.sh prepare
bash RoboTwin/policy/pi0/server_train_clusters.sh check --gpu 0,1,2,3
```

`prepare` 使用CPU，逐簇生成32维归一化资产并检查真实加载器、图像、语言和batch。输出位于：

```text
RoboTwin/policy/pi0_cluster_lora/assets/cluster_XX_pi0_ee_lora/
└── cluster_XX_pi0_ee/
    ├── norm_stats.json
    └── provenance.json
```

如果 tokenizer/基座放在共享位置：

```bash
export OPENPI_DATA_HOME=/shared/openpi
bash RoboTwin/policy/pi0/server_train_clusters.sh prepare \
  --pretrained-params /shared/pi0_base/params
```

## 训练

四张指定GPU依次训练七个专家，每个专家内部使用全部四卡：

```bash
cd /workspace/RoboLoRA
bash RoboTwin/policy/pi0/server_train_clusters.sh train \
  --gpu 0,1,2,3 --fsdp-devices 4 \
  --run-name cluster7_lora_v1
```

训练单个或部分簇：

```bash
bash RoboTwin/policy/pi0/server_train_clusters.sh train \
  --clusters 8 24 --gpu 0,1,2,3 --fsdp-devices 4 \
  --run-name cluster7_lora_v1
```

同一个 run 可在后续训练其余簇，但必须带 `--resume`，且训练配置和首次完全一致：

```bash
bash RoboTwin/policy/pi0/server_train_clusters.sh train \
  --clusters 3 6 9 31 23 --gpu 0,1,2,3 --fsdp-devices 4 \
  --run-name cluster7_lora_v1 --resume
```

后台运行：

```bash
nohup bash RoboTwin/policy/pi0/server_train_clusters.sh train \
  --gpu 0,1,2,3 --fsdp-devices 4 --run-name cluster7_lora_v1 \
  > cluster7_train_console.log 2>&1 &
tail -f cluster7_train_console.log
```

checkpoint结构：

```text
RoboTwin/policy/pi0_cluster_lora/runs/cluster7_lora_v1/
├── run_config.json
├── logs/
├── checkpoints/
│   ├── cluster_08_pi0_ee_lora/model/<step>/
│   └── ...
├── status/cluster_08_trained.json
└── training_summary.json
```

恢复时保留整个 run 目录并重复相同参数。默认 `keep_period=null`，中间保存用于故障恢复，但在后续保存后只保留最新checkpoint；每个簇的最终非整500步checkpoint也会保存。

## 当前范围

本适配完成数据准备和七个独立 LoRA 的训练管线。簇专家如何路由、组合，以及在五个完整任务中的仿真评测属于下一阶段；当前入口不会把七个 LoRA 合并成一个模型，也不会自动用簇编号替换语言指令。
