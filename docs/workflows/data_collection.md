# 统一数据采集

本页维护 Raw 采集配置、调度、校验发布与续采。任务控制器和物理验收由任务指南负责；
模型数据字段与动作监督见[数据格式合同](../reference/dataset_formats.md)，
已录制 Raw 的回放见[离线可视化](data_visualization.md)。

统一入口是 `scripts/collect/collect.py`。从仓库根目录使用本节点的 Pixi 完整路径：

```bash
cd /cpfs_infra/user/chenxianchi/code/kaitactilesim
PYTHONPATH=src /cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is collect -- --help
PYTHONPATH=src /cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is collect -- --list-support
PYTHONPATH=src /cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is collect -- --describe usb-insert
```

入口负责十三个任务的调度、独立进程、seed 派生、输出编号、状态记录、中断处理和断点续采。它调用各任务控制器，不使用训练模型，也不改变任务动作。

环境安装、已安装环境的限制和源码导入检查统一见[环境与工作目录](environment.md)。
示例使用 `--as-is` 复用已安装环境，并用 `PYTHONPATH=src` 指向当前仓库源码。
实际验收条件见[成功判据总表](../tasks/success_criteria.md)；
可选的阶段诊断见[分阶段评估](staged_evaluation.md)，不能把阶段通过当作采集成功。

`--profile production-v1` 为默认任务配置，`legacy-compatible` 是同一配置的兼容别名。
`--describe TASK` 展示实际 recorder、随机化、相机、验收规则和合同来源。
新 `collection.json` 保存解析后的任务配置和首条具体命令，每次尝试的真实命令
仍保存在 `status.json`。`--dry-run` 保留每任务一行 JSON 的输出形式，并增加
`kaihand_collection_plan_v1` 版本标识。

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

新录制的共享 HDF5 增加 `runtime_contract_json`：它在 recorder 初始化时从实际
编译后的模型读取 timestep、integrator、solver、NoSlip、impratio 和显式接触对，
并保存触觉 provider 的 source、单位、kernel、阈值和 debounce（适用时）。
任务运行时对模型的修改会进入该快照；`depth×1e4` 的 probe 聚合值仍保留自身单位，
不会改标为 N。历史 HDF5 缺少此属性时，保留“未记录”的事实。

## 开始前检查

先按[环境与工作目录](environment.md)准备环境，再检查所有场景和三路相机。硬件渲染需要
`__EGL_VENDOR_LIBRARY_FILENAMES` 指向 NVIDIA EGL vendor JSON；
`--render-backend hardware` 不会自动设置这个路径。以下硬件命令均显式包含该环境变量：

```bash
cd /cpfs_infra/user/chenxianchi/code/kaitactilesim
# 安装与源码导入检查见 environment.md。

# NVIDIA/硬件 EGL
__EGL_VENDOR_LIBRARY_FILENAMES=/cpfs_infra/user/chenxianchi/.config/kaitactilesim/nvidia_egl_vendor.json \
PYTHONPATH=src /cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is collect -- \
  --task all --check --render-backend hardware

# 没有硬件 EGL 时显式使用 Mesa 软件渲染
PYTHONPATH=src /cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is collect -- \
  --task all --check --render-backend software
```

硬件模式不会静默回退到软件 renderer。GPU 主要负责渲染，MuJoCo 物理仍主要使用 CPU。

## 采一条与批量采集

先对目标任务采一条 smoke 数据：

```bash
__EGL_VENDOR_LIBRARY_FILENAMES=/cpfs_infra/user/chenxianchi/.config/kaitactilesim/nvidia_egl_vendor.json \
PYTHONPATH=src /cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is collect -- \
  --task usb-insert --episodes 1 --seed 2026 \
  --output-dir /path/to/raw_smoke \
  --render-backend hardware
```

全部任务各采十次尝试：

```bash
__EGL_VENDOR_LIBRARY_FILENAMES=/cpfs_infra/user/chenxianchi/.config/kaitactilesim/nvidia_egl_vendor.json \
PYTHONPATH=src /cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is collect -- \
  --task all --episodes 10 --seed 2026 \
  --output-dir /path/to/raw_batch \
  --render-backend hardware
```

也可以一次选择多个任务：

```bash
__EGL_VENDOR_LIBRARY_FILENAMES=/cpfs_infra/user/chenxianchi/.config/kaitactilesim/nvidia_egl_vendor.json \
PYTHONPATH=src /cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is collect -- \
  --task poker-draw usb-insert --episodes 20 --seed 2026 \
  --output-dir /path/to/card_usb_raw \
  --render-backend hardware
```

`--episodes` 是每个任务的尝试数，不保证成功条数。先使用 `--dry-run` 查看实际子命令且不写数据：

```bash
PYTHONPATH=src /cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is collect -- \
  --task all --episodes 10 \
  --output-dir /path/to/raw_batch --dry-run
```

需要收满指定数量的成功轨迹时，使用成功数模式：

```bash
__EGL_VENDOR_LIBRARY_FILENAMES=/cpfs_infra/user/chenxianchi/.config/kaitactilesim/nvidia_egl_vendor.json \
PYTHONPATH=src /cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is collect -- \
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

海绵抓取放盘子的 8-worker 命令示例（不同 seed 对海绵初始位置进行每轴 ±1 mm 的可复现随机化；盘子与动作控制不变）：

```bash
__EGL_VENDOR_LIBRARY_FILENAMES=/cpfs_infra/user/chenxianchi/.config/kaitactilesim/nvidia_egl_vendor.json \
PYTHONPATH=src /cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is collect -- \
  --task sponge-grasp --target-successes 200 --max-attempts 1000 \
  --workers 8 --seed 20260924 \
  --output-dir /path/to/sponge_grasp_raw_200 \
  --render-backend hardware
```

随机化样本仍须连续保持五指接触、满足原峰值力上限并稳定落盘；失败尝试不计入成功数。
因软体求解器的瞬时接触点会随初态切换，带 seed 的样本在悬空运动阶段使用
法向/切向 0.8/0.8 N 的逐步跳变上限；海绵落盘后的释放阶段分别为 1.0/1.1 N。
未随机化的固定样本仍使用原有的 0.6 N 全程法向、0.6 N 运动中切向和
0.8 N 释放切向上限。

先用 `--task sponge-grasp --check --render-backend hardware` 检查渲染；没有可用 EGL
设备时改用 `--render-backend software`。请按实际内存和 CPU 容量决定是否保持 8 workers。
上述环境变量会被预检和每个采集子进程继承，使 EGL 使用 NVIDIA 驱动；
预检应报告实际 NVIDIA renderer 和 `software: false`。软件模式不需要该 NVIDIA 配置前缀。

`--workers 4` 表示同时运行最多 4 个独立 episode 子进程，也是当前接口默认值。
实际并发应按节点 CPU、内存与渲染容量调整。成功数模式会把在途任务数限制为“目标成功数减去当前成功数”，
所以不会因并发超过目标。不同任务应分别启动；不要再同时启动多个 4-worker 采集命令。

## 十三个采集任务

| CLI 任务名 | 控制器与随机性 | 额外输出 |
|---|---|---|
| `pick-place` | 原 production 抓放与原初态随机参数 | Raw HDF5、episode sidecar |
| `poker-draw` | `middle-force-precontact-v1` 与任务完成验收 | 右手求解器触觉、episode sidecar |
| `usb-insert` | fast 动作、物体位姿和接触前噪声 | summary、result sidecar |
| `bulb-screw` | five-finger/fast，灯泡 X/Y 小范围随机化 | HDF5、结果、最小验证 sidecar |
| `install-ram` | RAM 与支架共同 X/Y 小范围随机化 | HDF5、结果、最小验证 sidecar |
| `vase-wipe` | 污渍布局随机化，动作与清洁逻辑保持原实现 | HDF5、结果、最小验证 sidecar |
| `whiteboard-wipe` | 保留 GitHub 新版拿起板擦、擦净笔迹、放回并释放的原动作 | HDF5、结果、笔迹随机化与最小验证 sidecar |
| `sponge-grasp` | 软海绵从桌面抓起并放入盘子；初始 XY 每轴 ±1 mm、由 seed 可复现 | HDF5、结果、严格任务审计与验证 sidecar |
| `usb-unplug` | 已插入初态，双指抓取、完全拔出后手持且不触桌 ≥2 s；seed 不给已插入插头自由 XY 扰动；旧放回版不混用 | HDF5、signed withdrawal load、连续手持计时与协议标识、物理步审计 |
| `whiteboard-wipe-horizontal` | 固定水平朝上白板，随机墨迹中心；`--fixed-stains` 固定笔迹 | HDF5、实际接触清洁进度、结果和验证 sidecar |
| `whiteboard-wipe-horizontal-line` | 固定水平板，仅保留原 Z 顶部横线；随机墨迹中心，`--fixed-stains` 固定笔迹 | HDF5、8 片墨迹实际清洁进度、拿取/释放/回位与物理步审计 |
| `bulb-unscrew` | 已安装灯泡初态，五指逆拧、真实脱扣、普通桌面放稳；seed 只随机化桌面返回目标 XY ±2 mm | HDF5、反向转数/脱扣/桌面支持、历史验收、物理步审计 |
| `bulb-screw-inplace` | 灯座入口已啮合但未到座；原位抓稳、旋入、承力停转、释放验稳；当前初态固定，seed 不改变入口位姿 | HDF5、初始化证据、旋入/拧紧/释放标志、独立 home 收尾及物理步审计 |

统一的是 Raw、三相机和调度；动作与验收不被统一入口重写。

五个变体均使用独立 task 身份。新 `bulb-screw-inplace` 不包含拿取/搬运；物理成功与
完整专家流程正常结束分开记录，回 home 不作为物理旋入成功条件。多个 seed 仅是固定初态
重复性检查，不能统计成初态泛化。运行与单条三相机采集示例见
[灯泡原位旋入命令](../tasks/bulb_tasks.md#bulb-screw-inplace)。

Bulb、RAM、Vase、Whiteboard、Sponge 的正式批采自动走 raw-only 入口。单条 `data/` 不生成 MP4、逐帧 PNG、
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
__EGL_VENDOR_LIBRARY_FILENAMES=/cpfs_infra/user/chenxianchi/.config/kaitactilesim/nvidia_egl_vendor.json \
PYTHONPATH=src /cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is collect -- \
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
- 旧结构的 `collection.json` 只有在完整旧合同（含 `source_sha256`）仍匹配时才可续采；
  新增或修改源码后应使用新采集目录。

## 发布前数据质量检查

首次使用一套新的采集设置时，先采至少 5 条完整 episode，再决定是否批量采集：

1. 检查相机、触觉、状态和动作时间戳及索引；图像所对应的触觉不得来自未来。
2. 检查 `state_t → action_t → state_{t+1}` 是否一致，并核对动作确实产生合理状态变化。
3. 检查缺帧、异常重复帧、非有限值和状态轨迹突变；静止场景的相同像素不自动算异常。
4. 抽查首次接触和最后接触附近的触觉与 RGB，记录人工复核结论。

后续批次仍逐条运行自动校验；若修改相机、时间同步、触觉、控制或导出流程，
重新执行 5 条小批量人工复核。USB 的具体命令、阈值和触觉起止对照见
[USB Raw Cleaning](../tasks/usb_tasks.md#usb-cleaning)。

## 新任务接入

新增任务时需要：

1. 使用共享机器人、相机命名和 `EpisodeRecorder` Raw schema；
2. 在 `src/kaihand_tactile_env/pipeline/catalog.py` 注册任务名，并在
   `pipeline/collection.py` 的 `COLLECTION_SPECS` 注册命令构造、seed、配置和 outcome 检查；
   `scripts/collect/collect.py` 负责共享调度，不直接维护任务专属命令；
3. 保存 `metadata_json.scene`、随机参数、任务结果和三路相机；
4. 保留任务控制器、专属状态及成功判定，不把动作逻辑写进统一调度器；
5. 增加命令构造、Raw 验证、中断和续采测试。

采集完成不等于数据可训练。正式发布前仍需执行 Raw 验证、任务成功筛选及对应数据质量检查。
