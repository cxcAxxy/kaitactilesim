# 文档导航

本仓库按“采集 Raw → 转换训练数据 → 部署模型 → 闭环评估”组织文档。
日常运行从[统一命令速查](workflows/pipeline_quickstart.md)开始；任务和模型的实时支持范围
以 `pixi run convert-dataset -- --list-support`、`pixi run evaluate-policy -- --list-support`
的输出为准，不在多份文档里维护重复矩阵。

## 目录结构

```text
docs/
├── README.md                # 本页：唯一文档导航
├── workflows/               # 采集、转换、回放、评估的当前命令与合同
├── tasks/                   # 任务实现、随机化、物理模型及专项诊断
└── architecture/            # 跨任务共用的机器人姿势、抓取时序和外观
```

## 工作流

| 需求 | 文档 |
|---|---|
| 一套命令跑通流程 | [采集 → 转换 → 评估速查](workflows/pipeline_quickstart.md) |
| 采集 Raw 与续采 | [统一数据采集](workflows/data_collection.md) |
| 转换为 π0.5、EgoSteer、EgoTouch 或 LeRobot v3 | [统一数据转换](workflows/data_conversion.md) |
| LeRobot v3 字段和时钟细节 | [LeRobot v3 合同](workflows/lerobot_v3.md) |
| 离线回放 Raw | [原始数据可视化](workflows/data_visualization.md) |
| 运行模型评估 | [统一模型评估](workflows/model_evaluation.md) |
| 理解评估视频、触觉热图及轨迹图 | [模型评测视频](workflows/policy_evaluation_video.md) |

## 任务

| 任务 | 文档 |
|---|---|
| PickPlace | [采集示例](tasks/pick_place_example.md)、[LingBot 专项评估](tasks/lingbot_pickplace_evaluation.md) |
| Poker / Card | [示例与导出](tasks/poker_example.md) |
| USB insert | [执行](tasks/usb_insert_execution.md)、[初态随机化](tasks/usb_randomization.md)、[接触前噪声](tasks/usb_precontact_noise.md)、[接触阻力](tasks/usb_contact_resistance.md)、[力学检查](tasks/usb_contact_mechanics.md)、[Raw Cleaning](tasks/usb_cleaning.md) |
| Bulb screw | [旋拧任务](tasks/bulb_screw.md) |
| Install RAM | [安装任务](tasks/install_ram.md)、[尺寸与物理模型](tasks/install_ram_dimensions.md)、[原始力诊断](tasks/install_ram_force_diagnosis.md) |
| Vase wipe | [花瓶擦拭](tasks/vase_wipe.md) |
| Whiteboard wipe | [白板擦拭](tasks/whiteboard_wipe.md) |
| Sponge grasp | [海绵抓取与入盘](tasks/sponge_grasp.md) |

跨任务设计见[共享双臂初始姿势](architecture/shared_home.md)、
[抓取时序](architecture/grasp_timing.md)和[场景外观](architecture/appearance.md)。

## 维护原则

- 统一流水线的当前命令只在 `workflows/` 维护；任务页可以保留任务专属的预览、诊断命令与物理合同。
- 统一采集支持八个任务；Sponge 现支持 π0.5 和 LeRobot v3 转换，但尚无统一闭环评估 runner。
- 训练数据格式不等于模型评估支持；评估必须使用匹配的 checkpoint、deployment manifest 和 runner。
- 历史实验结果保留在原任务文档、`artifacts/` 或 Git 历史，不用旧批次命令替代当前入口。
