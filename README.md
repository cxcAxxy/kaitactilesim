# KaiHand Tactile Simulation

## 统一评估整合版（2026-10-15）

本目录是独立整合工程：保留 `kaitactilesim` 的完整仿真/任务判定/评估执行，
接入 `kaitactilesim_laest` 的离线分析与本地观察台。两个原工程、原数据和 8876 H50 站点不修改。
快照来源与文件 SHA 见 `integration_manifest.json`。

- 在线入口：`scripts/evaluate/evaluate.py`；离线导入入口：`scripts/analysis/evaluate_unified.py`。
- 共享阶段规则：`src/kaihand_tactile_env/shared/evaluation_rules.py`，由监控、记录和规则导出共用。
- 新版导入：只读已经完成的 H50 staged 记录；不启动推理、不重跑物理、不按当前规则重判。
- 原任务判定、归档阶段判定、数据有效性分层呈现；不同归档源码身份不混算。
- [使用方法、限制与回退](docs/workflows/unified_evaluation.md)。

```bash
cd /cpfs_infra/user/chenxianchi/code/kaitactilesim_unified
pixi run --as-is unified-evaluation -- rules
pixi run --as-is unified-evaluation -- discover --source-root /cpfs_infra/user/chenxianchi/kaitactilesim_evaluations
pixi run --as-is serve-unified
```

当前命令均从本整合工程根目录运行；历史 checkpoint、数据和归档的身份保持原值。

KaiHand 双臂触觉操作仿真、Raw 数据采集、模型数据转换与闭环评估代码库。
共享 MuJoCo 机器人、相机、触觉和 Raw HDF5 合同；各任务保留自己的动作控制器、
随机化、任务状态和成功条件。

## 从这里开始

| 需求 | 入口 |
|---|---|
| 第一次运行完整流程 | [采集 → 转换 → 部署命令速查](docs/workflows/pipeline_quickstart.md) |
| 采集 Raw 与断点续采 | [统一数据采集](docs/workflows/data_collection.md) |
| 转换为 π0.5、EgoSteer、EgoTouch 或 LeRobot v3 | [统一数据转换](docs/workflows/data_conversion.md) |
| 回放与检查 Raw | [原始数据可视化](docs/workflows/data_visualization.md) |
| 准备、核验、启动模型并接入仿真 | [统一模型部署](docs/workflows/model_deployment.md) |
| 已有模型服务的专项评估 | [统一模型评估](docs/workflows/model_evaluation.md) |
| 各任务具体成功条件与专家验收区别 | [13 个任务成功条件](docs/tasks/success_criteria.md) |
| 阶段判定、数值门槛与累计规则 | [阶段评估规则](docs/workflows/staged_evaluation.md) |
| 导入已有 H50 记录和查看本地网页 | [统一评估与观察台](docs/workflows/unified_evaluation.md) |
| 查任务、物理模型和专项诊断 | [文档导航](docs/README.md) |

在仓库根目录安装环境并检查场景。以下使用本节点的 Pixi 完整路径；
NVIDIA EGL 还需要显式指定 vendor 配置：

```bash
cd /cpfs_infra/user/chenxianchi/code/kaitactilesim_unified
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi install --locked
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run test-scenes
__EGL_VENDOR_LIBRARY_FILENAMES=/cpfs_infra/user/chenxianchi/.config/kaitactilesim/nvidia_egl_vendor.json \
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run collect -- --task all --check --render-backend hardware
```

没有 NVIDIA EGL 时去掉 vendor 环境变量前缀，显式改用 `--render-backend software`；硬件模式不会
静默回退。所有正式数据与评估应写入新的输出目录，不覆盖既有批次。

本节点的独立 `.pixi/envs/default` 已离线安装供 CPU 物理验证，暂缺 `pyarrow`。
使用已安装环境查询和预览时可用 `pixi run --as-is ...`，或直接调用
`.pixi/envs/default/bin/python -B`，不依赖已退役旧工程的解释器。
完整采集/转换/部署前应按所选后端补齐依赖；`--as-is` 不会补装缺包。

## 数据闭环与覆盖范围

```text
任务控制器 → 统一采集 → 共享 Raw HDF5
                         ├→ 离线回放与质量检查
                         └→ 语义 adapter → 模型数据集 / LeRobot v3
                                            → 外部模型训练 → checkpoint
                                            → 统一部署 prepare/verify/serve/run
                                            → 统一评估 → 指标与复核视频
```

统一采集与 LeRobot v3 转换覆盖十三个独立任务（原任务全部保留）：

| CLI 任务名 | 任务 |
|---|---|
| `pick-place` | 抓取圆柱并稳定放入目标盒 |
| `poker-draw` | 将扑克牌移到桌沿、夹起并转向机器人 |
| `usb-insert` | 抓取、对准、插入 USB-A 插头并释放 |
| `bulb-screw` | 抓取灯泡并旋入灯座 |
| `bulb-screw-inplace` | 灯泡已在灯座入口入牙，原位抓稳、旋入、拧紧并释放；[任务说明](docs/tasks/bulb_screw_inplace.md) |
| `install-ram` | 对准定位键并安装内存条 |
| `vase-wipe` | 用柔性海绵擦除花瓶内壁污渍 |
| `whiteboard-wipe` | 拿起板擦、擦净斜置白板、放回并释放 |
| `sponge-grasp` | 抓取柔性海绵并放入盘子 |
| `usb-unplug` | 抓住已插入的 USB，完全拔出后手持、无桌面接触连续保持 2 秒；旧版放回流程归档保留 |
| `whiteboard-wipe-horizontal` | 拿起板擦、擦净水平朝上的白板、放回并释放 |
| `whiteboard-wipe-horizontal-line` | 拿起板擦、只擦水平板上原 Z 顶部横线、放回并回位 |
| `bulb-unscrew` | 抓住已旋入的灯泡，逆时针取出、放稳普通桌面并释放、回位；旧返回卡座版不可混用 |

新采集默认保存 `head + left_wrist + right_wrist` 三路 RGB，以及状态、控制、
触觉、时间戳和任务诊断。训练时从 Raw 选相机子集；评估必须遵守
deployment manifest 冻结的模型输入合同。LeRobot v3 是数据格式，
不能作为 `--model-family`；Sponge 目前支持采集、π0.5 与 LeRobot v3
转换，以及 π0.5 统一闭环评估。具体任务 × 格式/模型组合以代码注册表为准：

```bash
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is collect -- --list-support
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is convert-dataset -- --list-support
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is deploy-policy -- --list-support
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is evaluate-policy -- --list-support
```

转换与部署可以共用命名运行环境：`--runtime-config FILE --runtime NAME`。
示例见 [runtimes.example.toml](configs/runtimes.example.toml)；sidecar 和 local-worker
模型使用所选 Python/CUDA 环境，仿真仍在 Pixi 环境运行。注册表标为 `embedded`
的模型在 rollout runner 内执行，具体边界以部署能力查询为准。

## 代码目录

```text
src/kaihand_tactile_env/
├── shared/             # 机器人、相机、触觉、Raw 与评估公共组件
├── pipeline/           # 任务、采集/转换/部署注册表与运行环境
└── tasks/              # 十三个独立任务的场景、控制与成功判定
scripts/
├── collect/            # 统一采集入口
├── convert/            # 统一转换入口
├── deploy/             # prepare/verify/serve/run 与进程管理
├── evaluate/           # 任务 × 模型评估入口
└── workcell/           # 专项 runner、adapter、回放与诊断
docs/
├── workflows/          # 当前命令和数据合同
├── tasks/              # 任务说明和专项诊断
└── architecture/       # 跨任务共享设计
```

历史 Raw 按采集时的真实相机、时间和动作合同使用；新代码不会补造旧批次缺失的
视角，也不会自动把不同动作版本的数据混成同一训练版本。

脚本的正式、兼容和历史入口约定见 [scripts/README.md](scripts/README.md)。
默认 `pytest` 只收集 `tests/`，实验复现包保留在 `runs/`。
