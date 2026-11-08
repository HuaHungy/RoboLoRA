# RoboLoRA

机器人动作表示与聚类，以及 RoboTwin 五任务联合 π0 末端 LoRA 训练/评测。

- `data_clustering/`：动作窗口、表示学习、视觉辅助与聚类代码。
- `RoboTwin/`：RoboTwin 任务、π0 代码和本地扩展。
- **[服务器训练与数据搬运文档](RoboTwin/policy/pi0/SERVER_TRAINING.md)**：数据/权重位置、环境安装、多卡选择、训练和评测命令。

完成文档中的环境安装及数据搬运后：

```bash
cd RoboTwin/policy/pi0
bash server_train_ee.sh train --gpu 0,1,2,3 --fsdp-devices 4
```

默认全局 batch=32，总计 20,000 步；保留 5,000 / 10,000 / 15,000 / 20,000 四个 checkpoint。
五个任务共同训练一个模型。显卡编号通过 `--gpu` 指定，`--fsdp-devices` 指定参数分片组大小。
只训练不需要仿真资源；评测另见文档。

数据集、模型权重、虚拟环境、缓存、历史训练产物及含凭据的本地登录脚本未上传。
本快照不包含 `LoRAHub-JAX/`，因为该目录不属于本次发布范围。

RoboTwin 来源：<https://github.com/RoboTwin-Platform/RoboTwin>，本地基准提交 `c3ddfa8b97d5519efa828b075999bd0006778e5e`，包含工作区现有修改与新增扩展。各组件许可证保留在对应目录（包括 `RoboTwin/LICENSE`、`RoboTwin/policy/pi0/LICENSE`）。
