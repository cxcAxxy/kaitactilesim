# KaiHand Tactile Simulation

KaiHand 双臂触觉操作仿真、Raw 数据采集、模型数据转换与闭环评估代码库。
共享 MuJoCo 机器人、相机、触觉和 Raw HDF5 合同；各任务保留自己的动作控制器、
随机化、任务状态和成功条件。

## 从这里开始

| 需求 | 入口 |
|---|---|
| 第一次运行完整流程 | [采集 → 转换 → 评估命令速查](docs/workflows/pipeline_quickstart.md) |
| 采集 Raw 与断点续采 | [统一数据采集](docs/workflows/data_collection.md) |
| 转换为 π0.5、EgoSteer、EgoTouch 或 LeRobot v3 | [统一数据转换](docs/workflows/data_conversion.md) |
| 回放与检查 Raw | [原始数据可视化](docs/workflows/data_visualization.md) |
| 部署模型并评估 | [统一模型评估](docs/workflows/model_evaluation.md) |
| 查任务、物理模型和专项诊断 | [文档导航](docs/README.md) |

在仓库根目录安装环境并检查场景：

```bash
pixi install --locked
pixi run test-scenes
pixi run collect -- --task all --check --render-backend hardware
```

没有 NVIDIA EGL 时显式改用 `--render-backend software`；硬件模式不会
静默回退。所有正式数据与评估应写入新的输出目录，不覆盖既有批次。

## 数据闭环与覆盖范围

```text
任务控制器 → 统一采集 → 共享 Raw HDF5
                         ├→ 离线回放与质量检查
                         └→ 语义 adapter → 模型数据集 / LeRobot v3
                                            → checkpoint + deployment manifest
                                            → 统一评估 dispatcher → 指标与复核视频
```

统一采集与 LeRobot v3 转换覆盖八个任务：

| CLI 任务名 | 任务 |
|---|---|
| `pick-place` | 抓取圆柱并稳定放入目标盒 |
| `poker-draw` | 将扑克牌移到桌沿、夹起并转向机器人 |
| `usb-insert` | 抓取、对准、插入 USB-A 插头并释放 |
| `bulb-screw` | 抓取灯泡并旋入灯座 |
| `install-ram` | 对准定位键并安装内存条 |
| `vase-wipe` | 用柔性海绵擦除花瓶内壁污渍 |
| `whiteboard-wipe` | 拿起板擦、擦净斜置白板、放回并释放 |
| `sponge-grasp` | 抓取柔性海绵并放入盘子 |

新采集默认保存 `head + left_wrist + right_wrist` 三路 RGB，以及状态、控制、
触觉、时间戳和任务诊断。训练时从 Raw 选相机子集；评估必须遵守
deployment manifest 冻结的模型输入合同。LeRobot v3 是数据格式，
不能作为 `--model-family`；Sponge 目前支持采集、π0.5 与 LeRobot v3
转换，尚无统一闭环评估 runner。具体任务 × 格式/模型组合以代码注册表为准：

```bash
pixi run convert-dataset -- --list-support
pixi run evaluate-policy -- --list-support
```

## 代码目录

```text
src/kaihand_tactile_env/
├── shared/             # 机器人、相机、触觉、Raw 与评估公共组件
├── pipeline/           # 转换注册表和格式合同
└── tasks/              # 八个任务的场景、控制与成功判定
scripts/
├── collect/            # 统一采集入口
├── convert/            # 统一转换入口
├── evaluate/           # 任务 × 模型评估入口
└── workcell/           # 专项 runner、adapter、回放与诊断
docs/
├── workflows/          # 当前命令和数据合同
├── tasks/              # 任务说明和专项诊断
└── architecture/       # 跨任务共享设计
```

历史 Raw 按采集时的真实相机、时间和动作合同使用；新代码不会补造旧批次缺失的
视角，也不会自动把不同动作版本的数据混成同一训练版本。
