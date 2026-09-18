# KaiHand Tactile Simulation

这是 KaiHand 双臂触觉操作仿真、数据采集、数据转换和模型闭环评估的统一代码库。
环境基于 MuJoCo，复用同一套天机 M6 双臂、左右 KaiHand、共享相机、触觉和 Raw HDF5
合同，同时保留每个任务原有的动作控制器、随机化和成功判定。

当前合并版包含七个彼此隔离的任务：

- PickPlace：抓取圆柱并放入目标盒；
- Card：将扑克牌滑至桌沿、夹起并转向机器人；
- USB：抓取 USB-A 插头、对准并插入插座；
- Bulb：抓取灯泡、对准灯座并旋入；
- RAM：抓取内存条、按定位键对准并插入卡槽；
- Vase：用柔性海绵擦除花瓶内壁污渍；
- Whiteboard：拿起板擦、擦净斜置白板并放回释放。

## 当前版本边界

本目录是在服务器统一工作流版本上合入 GitHub 最新 `main` 后形成的独立版本：

- 旧目录 `sim_code_merged_20260917` 保持不变，历史采集和转换任务不受影响；
- 当前开发分支为 `integration/github-main-20260918`；
- GitHub 新增的 Whiteboard 任务及 USB 动作更新已合入；
- Whiteboard 已接入统一采集、转换、回放和闭环评估；
- 旧任务动作与成功判定不因统一框架而被改写。

日常操作从[文档总入口](docs/README.md)开始。三个主流程只维护一套公共命令：

| 阶段 | 统一入口 | 主文档 |
|---|---|---|
| 数据采集 | `pixi run collect -- ...` | [统一数据采集](docs/data_collection.md) |
| 数据转换 | `pixi run convert-dataset -- ...` | [统一数据转换](docs/data_conversion.md) |
| 模型推理评估 | `pixi run evaluate-policy -- ...` | [统一模型推理与评估](docs/model_evaluation.md) |
| Raw 轨迹回放 | `pixi run replay -- ...` | [原始数据可视化](docs/data_visualization.md) |

## 七个任务

| CLI 任务名 | 操作与成功条件摘要 | 查看/执行入口 |
|---|---|---|
| `pick-place` | 圆柱完成抓取、搬运、稳定放置 | `pixi run view-pick-place` |
| `poker-draw` | 牌到达桌沿、完成夹持抬起并保持 | `pixi run view-poker-draw` |
| `usb-insert` | 插头完成抓取、对准、插入和释放验收 | `pixi run view-usb-insert` / `pixi run run-usb-insert` |
| `bulb-screw` | 灯泡完成抓取、对准、旋入、就位和释放 | `pixi run view-bulb-screw` / `pixi run run-bulb-screw` |
| `install-ram` | RAM 对准定位键、插入卡槽、就位和释放 | `pixi run view-install-ram` / `pixi run run-install-ram` |
| `vase-wipe` | 海绵在真实承力和摩擦滑动下达到清洁阈值 | `pixi run view-vase-wipe` / `pixi run run-vase-wipe` |
| `whiteboard-wipe` | 板擦被拿起、笔迹清零、放回桌面并稳定释放 | `pixi run view-whiteboard-wipe` / `pixi run run-whiteboard-wipe` |

表中只是摘要。最终成功与失败以任务实现写入 HDF5/sidecar 的 outcome 为准，统一调度器
不会用进程返回码替代物理成功判定。

## 数据闭环

```text
任务控制器与成功判定
        │
        ▼
统一采集 ──► 共享 Raw HDF5 + outcome + 最小验证 sidecar
        │
        ├──► 离线回放：相机 + 触觉热力图 + 十指力曲线
        │
        ▼
语义 adapter ──► EgoSteer / π0.5 / EgoTouch 数据集
        │
        ▼
checkpoint + deployment manifest
        │
        ▼
统一评估 dispatcher ──► 任务 runner ──► 指标 + 可视化视频
```

共享的是接口、数据合同和调度方式；任务动作、任务专属状态、随机化和成功判定仍位于
`src/kaihand_tactile_env/tasks/<task>/` 中。

## 安装与开始前检查

在本仓库根目录执行：

```bash
pixi install --locked

# 轻量回归
pixi run test-scenes

# H20 / NVIDIA EGL：编译所选场景并检查三路相机
pixi run collect -- \
  --task all --check --render-backend hardware
```

没有硬件 EGL 时，可以显式使用软件渲染：

```bash
pixi run collect -- \
  --task all --check --render-backend software
```

硬件模式不会静默回退到 Mesa。GPU 主要负责三路 RGB 渲染，MuJoCo 物理、控制器和
HDF5 写入仍主要消耗 CPU、内存和存储带宽。

## 统一 Raw 合同

所有通过统一入口新采集的 Raw 默认保存：

- `head + left_wrist + right_wrist` 三路 320×240、30 Hz RGB；
- 相机内参、逐帧动态外参、图像时间戳和对应状态索引；
- 机器人关节状态、实际控制量、动作阶段和 taskspace；
- 物体位姿、速度、任务随机化和模型/源码指纹；
- 求解器触觉或任务声明的触觉源、接触事件及时间戳；
- 任务专属状态，例如擦拭进度、插入力、旋入量或笔迹状态；
- HDF5、任务结果和带 SHA-256/结构验证结果的最小 sidecar。

新采集不维护 head-only、双相机和三相机三套 profile。采集阶段保存完整三相机，模型
需要哪些视角由转换参数和 deployment manifest 决定。

正式批采只保存训练所需数据，不在采集进程中生成 MP4、逐帧 PNG、PDF、CSV 或 HTML。
需要检查轨迹时，后续从 Raw 离线恢复可视化。

## 数据采集

先用一个 episode 做 smoke test：

```bash
pixi run collect -- \
  --task whiteboard-wipe \
  --episodes 1 \
  --workers 1 \
  --seed 20260918 \
  --output-dir /path/to/raw_smoke \
  --render-backend hardware
```

正式收满 200 条成功数据：

```bash
pixi run collect -- \
  --task bulb-screw \
  --target-successes 200 \
  --max-attempts 400 \
  --workers 4 \
  --seed 20260918 \
  --staging-root /path/to/fast_staging \
  --output-dir /path/to/raw_batch \
  --render-backend hardware
```

默认推荐 4 workers。同一输出目录中断后使用完全相同的任务、seed、源码和采集合同续采：

```bash
pixi run collect -- \
  --task bulb-screw \
  --target-successes 200 \
  --max-attempts 400 \
  --workers 4 \
  --seed 20260918 \
  --staging-root /path/to/fast_staging \
  --output-dir /path/to/raw_batch \
  --render-backend hardware \
  --resume
```

数据先写入 staging，成功且校验通过后才复制并原子发布到目标目录。详细的目录结构、失败
重试、锁和恢复规则见[统一数据采集](docs/data_collection.md)。

## Raw 轨迹回放

七个任务均可从一条已完成的 Raw HDF5 离线生成所选相机、双手触觉热力图和十指触觉值
时间曲线，不重新执行任务，也不修改源文件：

```bash
pixi run replay -- /path/to/episode.h5 \
  --viewer none \
  --review-dir /path/to/episode_review \
  --cameras head left_wrist right_wrist \
  --fps 10 --width 1920 --height 1080
```

输出包含 `review.mp4`、首末帧、逐帧数值 `frames.jsonl` 和 `replay.json`。只有源 HDF5
真实保存的相机才能被选择；回放不会补造历史数据中缺失的视角。

## 数据转换

统一转换入口不依赖固定任务目录。调用者显式指定任意 Raw 输入路径、输出路径、模型格式、
任务和相机；注册表负责选择经过审核的语义 adapter。

先查看当前代码实际支持的组合：

```bash
pixi run convert-dataset -- --list-support
```

只做合同和命令检查，不写输出：

```bash
pixi run convert-dataset -- \
  --input-dir /path/to/raw \
  --output-dir /path/to/planned_output \
  --format egosteer \
  --task auto \
  --cameras head left_wrist right_wrist \
  --expected-episodes 200 \
  --dry-run
```

正式转换时去掉 `--dry-run`，并按需要设置 `--workers`、`--staging-root` 和模型专属参数。
转换器直接读取 HDF5，优先复用采集 sidecar 的摘要，避免无意义重复校验；支持能力和恢复
边界见[统一数据转换](docs/data_conversion.md)。

### 当前转换覆盖

“任意 head 子集”表示 `head`、`head + left_wrist`、`head + right_wrist` 或三相机，前提是
源 Raw 中真实存在对应视角。

| 任务 | EgoSteer | π0.5 | EgoTouch |
|---|---|---|---|
| PickPlace | `head` | `head` | 待接入 |
| Card | 任意 head 子集 | `head + right_wrist` | 任意 head 子集 |
| USB | 任意 head 子集 | `head + right_wrist` | 任意 head 子集 |
| Bulb | 任意 head 子集 | 任意 head 子集 | 待接入 |
| RAM | 任意 head 子集 | 任意 head 子集 | 待接入 |
| Vase | 待接入 | 待接入 | 待接入 |
| Whiteboard | 任意 head 子集 | 任意 head 子集 | 任意 head 子集 |

未注册的组合会在写输出前明确失败。公共路径层不会仅凭 HDF5 结构猜测一个新任务的 action
语义。

## 模型推理与评估

deployment manifest 冻结 checkpoint、模型族和相机输入合同。统一入口可以配置评估次数、
录像次数、每次推理后实际执行步数、最大仿真时长和任务 runner 参数：

```bash
pixi run evaluate-policy -- \
  --task install-ram \
  --model-family pi05 \
  --deployment-manifest /path/to/deployment.json \
  --output-dir /path/to/evaluation \
  --cameras head left_wrist right_wrist \
  --num-trials 20 \
  --video-count 5 \
  --execute-steps 5 \
  --max-sim-seconds 60 \
  -- --server ws://127.0.0.1:18783
```

- `--num-trials` 默认 20；
- `--video-count` 默认 `min(3, num_trials)`，可设为 0；
- `--execute-steps` 控制每个预测 action chunk 实际执行多少步；
- `--cameras` 只校验 manifest，不会把旧 checkpoint 临时扩成三相机模型；
- 评估视频可包含模型实际输入视角、双手触觉热力图、十指三轴力曲线、任务指标和结果。

### 当前评估覆盖

| 任务 | EgoSteer | π0.5 | EgoTouch |
|---|---:|---:|---:|
| PickPlace | 已接入 | 已接入 | 待接入 |
| Card | 已接入 | 已接入 | 待接入 |
| USB | 已接入 | 已接入 | 待接入 |
| Bulb | 已接入 | 已接入 | 已接入（head-only） |
| RAM | 已接入 | 已接入 | 已接入（head-only） |
| Vase | 已接入 | 已接入 | 已接入（head-only） |
| Whiteboard | 已接入 | 已接入 | 已接入（head-only） |

“已接入”表示闭环 observation/action、任务状态、安全限制、成功判定、批量统计和视频链路
已经存在；实际运行仍需提供该任务训练得到且合同一致的 checkpoint、manifest 和模型服务。

## 历史数据兼容性

合并不会修改或迁移已有 HDF5。现有 0914/0917 数据应按采集时的真实合同使用：

| 历史任务 | 已保存相机 | 兼容结论 |
|---|---|---|
| PickPlace | `head` | Raw、回放和 head-only 转换可用；不能补造腕部视角 |
| Card | `head + right_wrist` | Raw、回放和历史双相机转换可用；不能选择 `left_wrist` |
| Bulb / RAM | 三相机 | Raw、回放及已注册的 EgoSteer/π0.5 转换可用 |
| Vase | 三相机 | Raw、验证和回放可用；模型数据语义 adapter 仍待实现 |
| USB | `head + right_wrist` | 文件结构仍可读；新版动作分布有变化，必须与新 USB 批次分版管理 |

USB 新版调整了高位悬停、对准和接近动作。旧 USB 数据不是损坏，而是不应与新版数据静默
混成同一数据版本。相同原则也适用于未来任何修改了任务动作、随机化、物理参数或成功判定
的数据。

## 相机与模型输入原则

共享训练相机的规范顺序是：

```text
head, left_wrist, right_wrist
```

- 新 Raw 固定采三相机；
- 转换阶段从源数据真实存在的相机中选子集；
- 训练产物记录最终相机合同；
- 推理时由 deployment manifest 复现相同合同；
- head-only、双相机和三相机 checkpoint 是不同模型合同，不能在评估命令中互相替代。

## 代码结构

```text
src/kaihand_tactile_env/
├── shared/                  # 机器人、相机、触觉、Raw、回放和策略公共组件
├── pipeline/                # 数据转换注册表与合同
└── tasks/
    ├── pick_place/
    ├── poker_draw/
    ├── usb_insert/
    ├── bulb_screw/
    ├── install_ram/
    ├── vase_wipe/
    └── whiteboard_wipe/

scripts/
├── collect/                 # 七任务统一采集调度
├── convert/                 # Raw → 模型数据集统一入口
├── evaluate/                # 任务 × 模型评估 dispatcher
└── workcell/                # 任务 runner、转换 adapter、回放和诊断脚本

docs/                        # 当前接口与任务文档
tests/                       # 场景、数据合同、调度、转换和评估测试
artifacts/                   # 带日期的实验与验收记录
datasets/                    # 仓库内的小型示例，不是正式 NAS 数据根目录
```

## 文档导航

- 总说明：[仿真数据闭环](docs/README.md)
- 采集：[统一数据采集](docs/data_collection.md)
- 转换：[统一数据转换](docs/data_conversion.md)
- 推理评估：[统一模型推理与评估](docs/model_evaluation.md)
- Raw 可视化：[原始数据可视化](docs/data_visualization.md)
- 评估视频：[模型评测视频](docs/policy_evaluation_video.md)
- Bulb：[灯泡旋拧](docs/bulb_screw.md)
- RAM：[安装任务](docs/install_ram.md)、[尺寸与物理模型](docs/install_ram_dimensions.md)、[原始力诊断](docs/install_ram_force_diagnosis.md)
- Vase：[花瓶内壁擦拭](docs/vase_wipe.md)
- Whiteboard：[斜置白板擦拭](docs/whiteboard_wipe.md)
- USB：[动作更新](docs/usb_insert_execution.md)、[数据 Cleaning](docs/usb_cleaning.md)、[接触力学](docs/usb_contact_mechanics.md)、[随机化](docs/usb_randomization.md)

带日期的单次实验、特定 checkpoint、固定端口和历史批次命令不再放在根 README；这类记录
保留在 `artifacts/`、Git 历史或对应任务文档中。
