# 仿真数据闭环总说明

本项目把七个仿真任务的数据闭环分成三个稳定阶段，并为每个阶段提供一个统一命令入口：

```text
任务控制器
   │
   ▼
统一采集 → 共享 Raw HDF5
   │
   ▼
格式 adapter → EgoSteer / π0.5 / EgoTouch 数据集
   │
   ▼
训练 checkpoint + deployment manifest
   │
   ▼
评估 dispatcher → 任务 runner → 结果、指标与复核视频
```

## 三个主文档

| 阶段 | 主文档 | 统一命令 |
|---|---|---|
| 数据采集 | [数据采集](data_collection.md) | `pixi run collect -- ...` |
| 数据转换 | [数据转换](data_conversion.md) | `pixi run convert-dataset -- ...` |
| 模型推理评估 | [模型推理与评估](model_evaluation.md) | `pixi run evaluate-policy -- ...` |

后续日常操作优先查看这三份文档，不再使用固定批次、个人绝对路径或某个 checkpoint 专属的历史命令。

## 核心合同

- 新采集固定保存 `head + left_wrist + right_wrist` 三路 320×240、30 Hz RGB，不维护 head-only、双相机、三相机三套采集 profile。
- 七个任务共用 Raw schema 和调度接口，但动作、随机化、任务专属状态及成功条件仍由各任务实现。
- 转换层可以自由指定输入路径、输出路径、目标格式、任务和相机；action/state 的含义不能凭目录名推测，必须存在对应语义 adapter。
- 推理评估由 deployment manifest 冻结模型输入合同。评估参数不能把旧 checkpoint 静默扩展为三相机模型。
- 新任务接入时扩展 adapter/runner 注册表，不新建另一套采集、转换或评估总流程。

## 当前覆盖范围

| 任务 | 统一采集 | EgoSteer 转换 | π0.5 转换 | EgoTouch 转换 | EgoSteer/π0.5 评估 |
|---|---:|---:|---:|---:|---:|
| PickPlace | 是 | head | head | 待接入 | 是 |
| Card (`poker-draw`) | 是 | 可选相机 | head + right_wrist | 可选相机 | 是 |
| USB (`usb-insert`) | 是 | 可选相机 | head + right_wrist | 可选相机 | 是 |
| Bulb (`bulb-screw`) | 是 | 可选相机 | 可选相机 | 待接入 | 是 |
| RAM (`install-ram`) | 是 | 可选相机 | 可选相机 | 待接入 | 是 |
| Vase (`vase-wipe`) | 是 | 待接入 | 待接入 | 待接入 | 是 |
| Whiteboard (`whiteboard-wipe`) | 是 | 可选相机 | 可选相机 | 可选相机 | 是 |

“可选相机”表示从三路 Raw 中选择包含 `head` 的子集。表中的“待接入”会明确报错，不会输出语义不完整的数据或调用错误 runner。

## 任务与数据质量文档

- Bulb：[灯泡旋拧](bulb_screw.md)
- RAM：[安装任务](install_ram.md)、[尺寸与物理模型](install_ram_dimensions.md)、[原始力诊断](install_ram_force_diagnosis.md)
- Vase：[花瓶内壁擦拭](vase_wipe.md)
- Whiteboard：[斜置白板擦拭](whiteboard_wipe.md)
- USB：[数据 Cleaning](usb_cleaning.md)、[接触力学检查](usb_contact_mechanics.md)、[接触阻力与控制](usb_contact_resistance.md)、[初态随机化](usb_randomization.md)、[接触前控制噪声](usb_precontact_noise.md)
- 数据复核：[原始数据可视化](data_visualization.md)、[模型评测视频](policy_evaluation_video.md)
- 场景外观：[White / Silver Lab 外壳与兼容性验证](appearance.md)
- 抓取动作：[USB 与内存先到位、再夹紧](grasp_timing.md)
- 机器人初始姿势：[共享双臂对称收臂配置](shared_home.md)
- PickPlace、Card 和共享工作台的任务细节见项目根目录 [README](../README.md)。

## 文档维护规则

- 三个阶段的当前命令只维护在对应主文档中；脚本目录的 README 仅提供跳转。
- 示例使用占位路径，不固化个人目录、批次日期、端口或 checkpoint 编号。
- 有时间戳的实验记录放在 `artifacts/` 或 Git 历史；`docs/` 只保留当前仍有效的接口与工程合同。
- 修改相机、Raw schema、模型输入、action 定义或成功指标时，同时更新本页、对应阶段主文档和任务文档。
