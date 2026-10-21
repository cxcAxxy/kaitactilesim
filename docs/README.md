# 文档导航与维护分工

先按[环境说明](workflows/environment.md)核对本仓源码，再用[流程速查](workflows/pipeline_quickstart.md)选择操作顺序。
每类信息由下面指定的一份文档负责；其他页面通过链接引用，避免复制规则后分别修改。

## 文档职责

| 文档 | 负责内容 | 其他内容去哪里查 |
|---|---|---|
| [环境](workflows/environment.md) | 安装、解释器与源码来源、模型环境选择、渲染配置 | 实际工作流见各操作指南 |
| [流程速查](workflows/pipeline_quickstart.md) | 采集到评估的最短命令顺序 | 详细参数、格式、成功条件不在速查页重复定义 |
| [数据采集](workflows/data_collection.md) | Raw 任务调度、尝试/成功数、录制、校验发布、续采 | 初态与专家动作见任务指南 |
| [数据转换](workflows/data_conversion.md) | adapter、CLI、相机选择、输出发布与恢复 | 字段、时钟、动作监督见格式合同 |
| [数据格式合同](reference/dataset_formats.md) | π0.5 / EgoSteer / EgoTouch / LeRobot v3 / FTP-1 的数据语义与来源边界 | 模型 checkpoint 和在线输入见模型专项 |
| [Raw 可视化](workflows/data_visualization.md) | 项目内专家示例的目录/来源校验、已有 HDF5 的离线回放与质量复核 | 模型 rollout 视频见评测视频 |
| [模型部署](workflows/model_deployment.md) | manifest、prepare / verify / serve / run 及进程生命周期 | 模型专有配置见模型专项，评估指标见模型评估 |
| [模型专项](workflows/model_specific_evaluation.md) | π0.5、EgoTouch、CPM、FTP-1、LingBot 的 checkpoint、输入/动作合同与额外参数 | 共用生命周期和 trial 参数不在本页重复维护 |
| [模型评估](workflows/model_evaluation.md) | 闭环参数、保护/停止、结果字段、有效性和统计分母 | 原生任务成功见成功判据，阶段进度见阶段判定 |
| [评测视频](workflows/policy_evaluation_video.md) | 模型视频布局、触觉显示、录像与轨迹产物、播放时钟 | 成功率与停止条件引用模型评估 |
| [阶段判定](workflows/staged_evaluation.md) | staged 支持范围、有序里程碑、数值/持续时间、累计与缺测 | 原生成功和专家完整流程另见成功判据 |
| [统一观察台](workflows/unified_evaluation.md) | 已有评估的离线发现、导入、来源核验和站点查看 | 新 rollout 见模型评估，发布历史见档案 |
| [成功判据](tasks/success_criteria.md) | 十三个 task 的原生成功与专家验收差异、对应源码 | 任务动作、几何和诊断见任务指南 |
| [共享场景](architecture/shared_scene.md) | 共用机器人姿势、抓取时序边界、外观资产与共享配置影响 | USB / RAM 等专用力学见任务指南 |
| [历史记录](archive/README.md) | 具体日期的版本、数据、训练、模型批次及发布验收 | 新运行使用当前工作流和全新输出目录 |

## 任务指南

任务按物理对象归并为八类，task ID 与数据身份仍独立。任务指南维护初态、专家动作、物理模型、
任务专用运行/诊断命令和数据记录；成功条件以[成功判据](tasks/success_criteria.md)为准。

| 指南 | 包含的 task ID | 主要内容 |
|---|---|---|
| [抓放](tasks/pick_place_example.md) | `pick-place` | 抓取、稳定放置、示例生成 |
| [摸牌](tasks/poker_example.md) | `poker-draw` | 版本关系、拖牌/夹牌/看牌、派生展示导出 |
| [USB](tasks/usb_tasks.md) | `usb-insert`、`usb-unplug` | 插拔流程、随机化、接触前噪声、阻力、力学诊断和 Cleaning |
| [灯泡](tasks/bulb_tasks.md) | `bulb-screw`、`bulb-screw-inplace`、`bulb-unscrew` | 三种独立初态、旋入/原位旋入/旋出、螺纹与数据身份 |
| [RAM](tasks/install_ram.md) | `install-ram` | 安装流程、尺寸/几何、产物模式、逐物理步力诊断 |
| [花瓶](tasks/vase_wipe.md) | `vase-wipe` | 柔体海绵、承力擦拭、视觉复查和污渍模型 |
| [白板](tasks/whiteboard_wipe.md) | `whiteboard-wipe`、`whiteboard-wipe-horizontal`、`whiteboard-wipe-horizontal-line` | 三类板面/轨迹、接触去污与离线触觉诊断 |
| [海绵](tasks/sponge_grasp.md) | `sponge-grasp` | 柔性海绵拿取与入盘 |

## 目录与脚本入口

```text
docs/
├── README.md        # 本页：总导航、文档职责和维护规则
├── workflows/       # 操作指南：如何运行
├── tasks/           # 按任务族组织的物理/专家指南与成功判据
├── reference/       # 格式合同：保存和训练的数据是什么意思
├── architecture/    # 跨任务共享设计
└── archive/         # 具体日期的历史事实与复现记录
```

根 [README](../README.md)只提供工程概览与常用入口；
[scripts/README.md](../scripts/README.md)说明稳定、兼容、历史脚本的等级。
各脚本目录的 README 只保留入口、查询方式和权威指南链接。

## 维护规则

- 一处定义，多处引用。更新数值成功条件先改成功判据；更新阶段阈值先改阶段判定；更新模型输入/执行例外先改模型专项。
- 操作指南维护可复用命令，任务指南维护任务专用物理和诊断，具体数据路径、checkpoint 批次与验收计数放入档案。
- 采集、转换、部署和评估能力分别以各自 `--list-support` 为准；接口注册、匹配权重就绪和实际评估完成分别记录。
- Raw 中缺失的相机或时钟不补造；原生结果、专家验收、staged 进度、trial 有效性和正式指标资格分别解释。
- 合并或移动页面时同时更新所有本仓引用和章节锚点，不保留只转发的空壳页。
- 历史路径、来源 SHA、规则版本和统计身份保留原值；不能通过文档整理改判旧结果。
