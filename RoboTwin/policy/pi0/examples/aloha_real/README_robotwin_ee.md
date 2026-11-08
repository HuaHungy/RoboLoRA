# RoboTwin 原生末端轨迹 → π0 LeRobot 数据

转换入口现默认读取 **RoboTwin 原生 HDF5**，直接导出末端表示的 LeRobot 数据。无需先运行 `scripts/process_data.py` 生成关节中间文件。

## 数据定义

`observation.state` 和 `action` 都是 16 维 float32：

```text
[left_x, left_y, left_z, left_qw, left_qx, left_qy, left_qz, left_gripper,
 right_x, right_y, right_z, right_qw, right_qx, right_qy, right_qz, right_gripper]
```

- 保留原生世界坐标位置（米）和 **wxyz** 四元数；四元数做单位化及 episode 内符号连续处理。
- 参考点采用原生 `get_left/right_ee_pose()`，不转换为 `get_left/right_tcp_pose()`。这两个参考点在当前实现中沿局部轴相差 0.12 米。
- 夹爪取原生归一化值，0 关闭、1 打开。
- 对原始 N 帧，生成 N−1 条记录：`state[t]=EE[t]`、`action[t]=EE[t+1]`、`image[t]=RGB[t]`。
- 动作标签是示范轨迹的**下一采样点绝对末端目标**，不是低层电机指令，也不是末端增量。最后一个真实状态用于末条记录的动作目标，不复制伪造终点。
- 50 步动作块由 π0 加载器读取；episode 尾部由 LeRobot 按其约定重复末动作并标记 padding，不跨越 episode。
- 原关节字段不写入训练 state/action。原生 HDF5 保持完整，可通过转换清单追溯。

## 一步转换

使用项目已有 `.venv`，沿用该项目固定的 LeRobot API，避免安装另一套 LeRobot 覆盖环境。

```bash
cd /home/huahungy/RoboLoRAHub/RoboTwin/policy/pi0

# 将新数据保存到工作区；训练时也要保留这个环境变量。
export HF_LEROBOT_HOME=/home/huahungy/RoboLoRAHub/data_clustering/data/lerobot

# 从原生采集目录读取前 50 条轨迹，直接生成 LeRobot 数据。
bash process_data_pi0.sh place_empty_cup demo_aloha 50 place_empty_cup_pi0_ee
```

输入目录为 `RoboTwin/data/place_empty_cup/demo_aloha/`，输出目录为 `$HF_LEROBOT_HOME/place_empty_cup_pi0_ee/`。省略第四个参数时，repo_id 默认 `<task_name>_pi0_ee`。`PYTHON=/path/to/python` 可选择其他具备相同依赖的解释器。

也可以指定原始目录：

```bash
bash generate.sh ../../data/place_empty_cup/demo_aloha place_empty_cup_pi0_ee

# 独立入口，可选择轨迹、保存模式和输出位置。
.venv/bin/python examples/aloha_real/convert_robotwin_ee_to_lerobot.py \
  --raw-dir ../../data/place_empty_cup/demo_aloha \
  --repo-id place_empty_cup_pi0_ee \
  --max-episodes 50 \
  --mode image
```

上述命令是同一转换的替代入口，不要对同一输出目录重复执行。脚本拒绝覆盖已有目录，也不自动删除失败转换留下的目录；重跑请选择新 repo_id/root。`--root` 指的是完整数据集目录，而不是缓存根目录。π0 训练按 `$HF_LEROBOT_HOME/<repo_id>` 找数据，两者必须对应。

默认按文件名中的数字自然排序。`--episodes 0 2 5` 指自然排序后的文件位置，`--max-episodes N` 取所选列表的前 N 条。每条 episode 从 `instructions/episodeN.json` 的 `seen` 列表选取一个固定种子控制的指令；缺少指令时可显式提供 `--task "..."`。脚本默认仅本地保存。

输入必须包含：

```text
endpose/{left,right}_endpose        [N,7]
endpose/{left,right}_gripper        [N] 或 [N,1]
observation/head_camera/rgb         [N,...]
observation/left_camera/rgb         [N,...]
observation/right_camera/rgb        [N,...]
```

图像映射为 `cam_high / cam_left_wrist / cam_right_wrist`，保存实际图像尺寸，在训练变换中调整到 224×224。原生 JPEG 经 RoboTwin 的 cv2 编解码后已恢复源 RGB 数组，不再额外交换 RGB/BGR。PNG 图像存储模式已完成回读验证；`--mode video` 使用 LeRobot 的视频编码路径，需要其 ffmpeg/解码依赖。

## 时间戳约定

默认 `fps=250/15≈16.6667`，对应当前常规保存间隔；可通过 `--fps` 指定其他采集设置的名义帧率。旧原生 HDF5 不含逐帧仿真时间，动作段起止还可能额外保存，因此**该值用于 LeRobot 的均匀帧索引时间，不代表恢复了精确物理时间，也没有执行重采样**。

转换清单明确保存该约定。这样可以按顺序训练 π0 的下一采样点动作块；如果实验依赖固定物理时长或速度估计，应先给采集器补充真实时间戳，再做时间重采样。

## π0 训练配置

新增 `place_empty_cup_pi0_ee_lora`，绑定 `place_empty_cup_pi0_ee` 数据集，使用 `LeRobotRoboTwinEEDataConfig`：

- 16 维输入/目标补零到 π0 的 32 维模型接口，动作块长度为 50。
- 不使用旧 Aloha 的关节符号翻转、夹爪角度换算或 `DeltaActions(6,-1,6,-1)`。
- 输出保留完整 16 维，归一化预测四元数并限制夹爪到 `[0,1]`；非法零四元数报错。
- 使用当前末端数据重新计算归一化统计，不能复用关节数据的统计文件或已训练关节专家。

转换后运行：

```bash
# 同一终端保持 HF_LEROBOT_HOME 指向上面的目录。
.venv/bin/python scripts/compute_norm_stats.py \
  --config-name place_empty_cup_pi0_ee_lora --num-workers 0

.venv/bin/python scripts/train.py place_empty_cup_pi0_ee_lora --exp-name ee_v1
```

归一化文件保存到 `assets/place_empty_cup_pi0_ee_lora/place_empty_cup_pi0_ee/norm_stats.json`。其他任务需新增/复制 `LeRobotRoboTwinEEDataConfig` 配置并修改 repo_id。当前配置沿用 π0 基座的 LoRA 训练设置；本次仅验证数据管道，没有进行完整模型训练或仿真成功率评测。

原来的 `deploy_policy.py` 仍按关节观测和关节控制工作。训练好 EE 模型后，部署端需要构造相同 16 维末端状态、使用新配置加载模型并调用 `take_action(..., action_type="ee")`；当前旧部署入口不能直接用于 EE 模型。

## 兼容性与旧入口

- `process_data_pi0.sh` 现在一步生成 EE LeRobot 数据，不再仅写 `processed_data/` 中间文件。
- `generate.sh` 及 `convert_aloha_data_to_lerobot_robotwin.py` 默认 `--action-representation ee`。
- 如需旧的关节转换流程，仍可直接运行 `scripts/process_data.py`，再给旧 LeRobot 入口指定 `--action-representation joint`。关节路径的旧动作/帧率约定未用于新的 EE 路径。
- 现有环境的 `datasets 5.0.1` 返回 `Column`，旧 LeRobot 在 `torch.stack` 时不兼容。新增 `openpi.training.lerobot_compat.LeRobotDataset` 适配列访问；π0 加载器已使用它，无全局 monkey patch、无环境降级。
- 修复 `compute_norm_stats.py` 的原有统计错误：现在遍历有限数据集一次、使用整个 batch、保留最后不足一批的记录；`max_frames` 选择可复现的不重复子集。

## 输出与验证

LeRobot 保存 `meta/info.json`、任务/episode 元数据以及 `data/chunk-000/*.parquet`；默认 image 模式的图像嵌入 parquet。额外的 `meta/robotwin_ee_conversion.json` 记录：源文件、帧数、所选指令、16 维字段名、动作时间对齐、坐标/四元数约定及完成状态。

回归测试：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
HF_DATASETS_CACHE=/tmp/robotwin_ee_hf_datasets \
HF_HUB_OFFLINE=1 JAX_PLATFORMS=cpu \
.venv/bin/python -m pytest scripts/robotwin_ee_test.py -q
```

本次 5 项测试通过，覆盖四元数连续性、图像通道、动作错位检查、跨 episode 边界 padding、拒绝关节文件/不同步相机、拒绝覆盖、16 维绝对目标不被关节变换，以及全 batch/尾部统计。

真实数据验证使用 `place_empty_cup` 的 episode0：178 个源帧 → 177 条记录。已通过实际训练加载器的归一化、分词、图像缩放及 JAX batch 构造：状态 `[2,32]`，动作 `[2,50,32]`，三相机各 `[2,224,224,3]`。

验证数据和报告位于：

```text
/home/huahungy/RoboLoRAHub/data_clustering/data/lerobot/place_empty_cup_pi0_ee_smoke/
└── meta/pi0_validation.json
```

这只是 1 条轨迹的管道验证产物；完整 50 条数据需要执行上面的正式转换命令。
