# 统一数据采集

统一入口是 `scripts/collect/collect.py`，Pixi 快捷命令为：

```bash
pixi run collect -- --help
```

入口负责七任务调度、独立进程、seed 派生、输出编号、状态记录、中断处理和断点续采。它调用各任务现有控制器，不使用训练模型，也不改变任务动作。

## 采集合同

所有新 Raw 默认固定采集：

- `head`
- `left_wrist`
- `right_wrist`

三路均为 320×240 RGB、30 Hz。这里没有相机 profile；模型训练使用哪些视角由后续转换决定。

共享 Raw 主要包含：

- 机器人关节状态、实际控制量与阶段；
- 物体位姿和速度；
- 求解器触觉、接触事件及时间戳；
- RGB、相机内参、逐帧动态外参和对应状态索引；
- 模型指纹、随机参数、任务 outcome 和采集合同；
- 任务专属数据组，例如 `vase_wipe/` 中的软体节点、污渍进度和摩擦功。

物理、状态、触觉和相机频率以每条 HDF5 的属性为准；不同任务可以有不同的高频诊断流。

## 开始前检查

安装锁定环境后，先检查七个场景和三路相机：

```bash
pixi install --locked

# NVIDIA/硬件 EGL
pixi run collect -- \
  --task all --check --render-backend hardware

# 没有硬件 EGL 时显式使用 Mesa 软件渲染
pixi run collect -- \
  --task all --check --render-backend software
```

硬件模式不会静默回退到软件 renderer。GPU 主要负责渲染，MuJoCo 物理仍主要使用 CPU。

## 采一条与批量采集

先对目标任务采一条 smoke 数据：

```bash
pixi run collect -- \
  --task usb-insert --episodes 1 --seed 2026 \
  --output-dir /path/to/raw_smoke \
  --render-backend hardware
```

七个任务各采十次尝试：

```bash
pixi run collect -- \
  --task all --episodes 10 --seed 2026 \
  --output-dir /path/to/raw_batch \
  --render-backend hardware
```

也可以一次选择多个任务：

```bash
pixi run collect -- \
  --task poker-draw usb-insert --episodes 20 --seed 2026 \
  --output-dir /path/to/card_usb_raw \
  --render-backend hardware
```

`--episodes` 是每个任务的尝试数，不保证成功条数。先使用 `--dry-run` 查看实际子命令且不写数据：

```bash
pixi run collect -- \
  --task all --episodes 10 \
  --output-dir /path/to/raw_batch --dry-run
```

需要收满指定数量的成功轨迹时，使用成功数模式：

```bash
pixi run collect -- \
  --task bulb-screw \
  --target-successes 200 \
  --max-attempts 400 \
  --workers 4 \
  --seed 20260917 \
  --output-dir /path/to/bulb_raw_0917_200 \
  --render-backend hardware
```

`--target-successes` 与 `--episodes` 互斥。失败和超时会保留，但不计入目标；达到 200 条
成功轨迹后立即停止。`--max-attempts` 是每个任务的安全上限，避免任务持续失败时无限运行。
如果 400 个索引仍未收满，可以用相同命令加 `--resume` 并把上限提高到 600；目标数、
seed、任务、渲染后端和源码必须保持不变。

`--workers 4` 表示同时运行最多 4 个独立 episode 子进程，也是 H20 上当前的
稳定默认值。成功数模式会把在途任务数限制为“目标成功数减去当前成功数”，
所以不会因并发超过目标。不同任务应分别启动；不要再同时启动多个 4-worker 采集命令。

## 七个任务

| CLI 任务名 | 控制器与随机性 | 额外输出 |
|---|---|---|
| `pick-place` | 原 production 抓放与原初态随机参数 | Raw HDF5、episode sidecar |
| `poker-draw` | `middle-force-precontact-v1` 与任务完成验收 | 右手求解器触觉、episode sidecar |
| `usb-insert` | fast 动作、物体位姿和接触前噪声 | summary、result sidecar |
| `bulb-screw` | five-finger/fast，灯泡 X/Y 小范围随机化 | HDF5、结果、最小验证 sidecar |
| `install-ram` | RAM 与支架共同 X/Y 小范围随机化 | HDF5、结果、最小验证 sidecar |
| `vase-wipe` | 污渍布局随机化，动作与清洁逻辑保持原实现 | HDF5、结果、最小验证 sidecar |
| `whiteboard-wipe` | 保留 GitHub 新版拿起板擦、擦净笔迹、放回并释放的原动作 | HDF5、结果、笔迹随机化与最小验证 sidecar |

统一的是 Raw、三相机和调度；动作与验收不被统一入口重写。

Bulb、RAM、Vase、Whiteboard 的正式批采自动走 raw-only 入口。单条 `data/` 不生成 MP4、逐帧 PNG、
PDF、CSV、HTML 或验收视频；可视化全部留给离线轨迹回放。批次级 `status.json`、
`collector.log`、`process.json` 不属于样本模态，用于失败定位与断点续采。

## 输出与断点续采

统一入口在任务外层保存批次状态：

```text
/path/to/raw_batch/
├── collection.json
├── preflight.log
├── summary.json
└── usb-insert/000000/attempt_001/
    ├── status.json
    ├── process.json
    ├── collector.log
    └── data/
```

续采必须使用相同输出目录和根 seed：

```bash
pixi run collect -- \
  --task usb-insert --episodes 100 --seed 2026 \
  --output-dir /path/to/raw_batch \
  --render-backend hardware --resume
```

- 已完成的成功、失败和超时尝试都会跳过。
- 中断或崩溃的条目在新的 `attempt_002/` 重试，旧目录保留。
- 成功产物会重新核对哈希；缺失或损坏时拒绝续采。
- 可以提高 `--episodes` 扩展同一批次，但不能更换 seed、任务合同或源码版本后继续混写。
- 成功数模式可在续采时提高 `--max-attempts`，但不能更换 `--target-successes`。
- 同一输出目录有互斥锁，不能由两个采集器同时使用。

## 新任务接入

新增任务时需要：

1. 使用共享机器人、相机命名和 `EpisodeRecorder` Raw schema；
2. 在 `collect.py` 注册任务命令、seed 和 outcome 检查；
3. 保存 `metadata_json.scene`、随机参数、任务结果和三路相机；
4. 保留任务控制器、专属状态及成功判定，不把动作逻辑写进统一调度器；
5. 增加命令构造、Raw 验证、中断和续采测试。

采集完成不等于数据可训练。正式发布前仍需执行 Raw 验证、任务成功筛选及对应数据质量检查。
