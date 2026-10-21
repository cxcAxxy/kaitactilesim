# 原始数据可视化

本页负责读取已完成的 Raw HDF5，离线回放保存的 RGB、状态和触觉并导出复核产物。
它读取专家或采集器录制的源数据；[模型评估视频](policy_evaluation_video.md)则读取模型闭环
rollout 中的实际观测与任务诊断。两者的时间线与产物来源分别保留，不能将专家回放当作模型成绩。
新录制操作见[数据采集](data_collection.md)及各任务指南。

## 项目内专家示例

专家示例位于本项目根目录的 `task_demo/`，不是 `runs/` 下的临时实验，也不是 Python 包资源。
白板诊断与统一观察台默认使用此目录，仍可用各自的 `--demo-root`、
`--contact-demo-root` 或 `--task-demo-root` 显式选择其他示例包。默认值根据代码位置确定，
不依赖终端当前目录，也不回退 NAS 来掩盖本地文件缺失。

```text
task_demo/
├── *.mp4                    # 任务示例视频；版本名和历史别名原样保留
├── _metadata/               # 原 replay/publication、逐帧时钟、首末帧与部分 Raw
├── _raw_sources/<SHA256>/   # 从原目录外补齐的、由 replay 明确绑定的 Raw
└── migration_manifest.json # 原绝对路径 → 本地相对路径、文件 SHA/大小、回放索引
```

2026-10-21 迁入时保留了原目录的 109 个文件及其内部硬链接，补齐 10 个外置 Raw，
共 14 个不同 Raw、17 份 replay（包含旧版本与别名）。每个载荷都已完整 SHA-256 校验；
旧审计 JSON 没有替换绝对路径或重写字节，而是通过单独的迁移清单解析本地文件。
原始路径表示来源身份，不再意味着运行时必须读取该 NAS 路径。训练集、转换数据集与
publication 中的历史 dataset_root 不随此包搬入，也不承诺重建整个训练批次。

这些是记录的脚本专家示例，不是真机人工示教，也不是专家总体表现的统计样本。
现有示例覆盖十一种当前 task；尚缺 `bulb-screw-inplace`、`vase-wipe`，
且旧版本成功记录不等于按当前规则重新验收。目录内的历史变体不能混作同一条轨迹。

在项目根目录执行以下命令可重新校验全部本地载荷与 replay 绑定，不访问 NAS、不运行仿真：

```bash
python3 -B tools/migrate_task_demo.py verify
```

需要取得某份原始 replay 对应的本地 HDF5 时，使用同一解析器，避免手改原 JSON：

```bash
PYTHONPATH=src .pixi/envs/default/bin/python -B - <<'PY'
import json
from kaihand_tactile_env.shared.task_demo import DEFAULT_DEMO_ROOT, resolve_demo_path
replay_path = DEFAULT_DEMO_ROOT / "_metadata/whiteboard-wipe/replay.json"
replay = json.loads(replay_path.read_text())
print(resolve_demo_path(replay["source_hdf5"], demo_root=DEFAULT_DEMO_ROOT,
                        expected_sha256=replay["source_sha256"]))
PY
```

`tools/migrate_task_demo.py copy --source /path/to/archive --output /new/path/task_demo`
只向尚不存在的目录复制并校验，保留原数据、不做删除。完整数据目录被 `.gitignore` 排除，
迁移工具和使用说明随源码维护。已冻结源码的 deployment 应重新 prepare；旧运行和已发布
报告不改写、不续接新路径身份不同的 batch。NAS 原始目录目前保留作回退副本。

## 十三任务统一轨迹回放

`replay` 是 PickPlace、Card、USB、Bulb、RAM、Vase、Whiteboard、Sponge 与五个独立变体
共十三任务的回放入口；变体包含 `usb-unplug`、`bulb-unscrew`、
`whiteboard-wipe-horizontal`、`whiteboard-wipe-horizontal-line`、`bulb-screw-inplace`。
下面的命令直接读取一条已完成的 Raw HDF5，生成相机画面、双手触觉热力图和十指触觉值时间曲线，
不重新运行仿真，也不修改源文件。命令从当前 `kaitactilesim` 根目录执行，
使用 `--as-is` 复用已安装环境，并用 `PYTHONPATH=src` 指向当前源码；
环境准备和导入检查见[环境与工作目录](environment.md)。
这里“不重新运行”指不执行策略、专家控制或物理步；入口仍会加载对应场景并恢复记录的状态以验证回放。

```bash
cd /cpfs_infra/user/chenxianchi/code/kaitactilesim
PYTHONPATH=src pixi run --as-is replay -- /绝对路径/episode.h5 \
  --viewer none \
  --review-dir /绝对路径/episode_replay
```

默认显示 HDF5 中存在的 `head + left_wrist + right_wrist`。也可以按顺序选择一至四路已保存相机，例如：

```bash
PYTHONPATH=src pixi run --as-is replay -- /绝对路径/episode.h5 \
  --viewer none \
  --review-dir /绝对路径/episode_replay_head_right \
  --cameras head right_wrist \
  --fps 10 --width 1920 --height 1080
```

输出目录必须不存在，完成后包含：

- `review.mp4`：所选 RGB、双手热力图和十指三轴曲线；
- `first_frame.png`、`last_frame.png`：快速检查布局和末帧；
- `frames.jsonl`：逐帧相机、状态、触觉索引、时间戳和十指曲线数值；
- `replay.json`：任务、源文件哈希、相机、触觉语义、固定量程和最大时间偏差。

Card、USB、Whiteboard 使用十指 solver taxel 的 `Fn`、`|Ft|` 和每指合力曲线；Bulb、RAM 只有右手五指空间力时，左手面板明确补零，不伪造测量；Vase 使用柔性接触力图。PickPlace 的原始触觉是 Genesis probe，因此热力图显示 probe depth（mm）和接触布尔值，曲线标成 Genesis local proxy，**不会标成牛顿力**。旧数据若只有每指聚合力，则热力图明确使用均匀铺到 7×5 的 fallback，曲线仍取保存的聚合值。

需要同时观察 MuJoCo 三维运动时使用：

```bash
PYTHONPATH=src pixi run --as-is replay -- /绝对路径/episode.h5 \
  --viewer native --speed 1.0
```

native viewer 按保存的 `qpos/qvel` 和状态时间戳恢复运动；多模态 MP4 使用保存的 RGB/触觉时钟。二者可以在一次命令中同时请求，但服务器批处理通常使用 `--viewer none`。

### 灯泡原位旋入样例

专家样例 Raw 保留在 `kaitactilesim_unified/runs/`，本仓不再重复保存。
下面读取该归档副本，写入一个新的复核目录；不是重跑模型评估，也不会覆盖原有 H50 报告：

```bash
PYTHONPATH=src /cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is replay -- \
  /cpfs_infra/user/chenxianchi/code/kaitactilesim_unified/runs/bulb_screw_inplace_example_20261015/raw/episode.h5 \
  --viewer none --review-dir /new/path/bulb_screw_inplace_review \
  --cameras head right_wrist --fps 10 --width 1920 --height 1080
```

原始 Raw 保留三路相机，上例仅选择头部与右腕显示。专家按自己的真实采样时间线播放，
不与模型强行逐时刻对齐；回放中亮灯、回位或“四段旋拧”不是成功证据。判定应核对
[成功判据总表](../tasks/success_criteria.md)、[原位旋入物理门槛](../tasks/bulb_tasks.md#判定与记录)
和真实记录标志；[分阶段诊断](staged_evaluation.md)与原生成功不同，缺失历史信号不补成零。

## 任务专属复核产物

PickPlace、Card、USB 的旧导出脚本仍可读取其历史固定布局。统一批采只保存 Raw，
下面的复核产物从已有 Raw 离线生成。

在仓库根目录执行，并将输入文件和输出目录替换为实际路径。输出目录必须尚不存在。

| 数据 | 命令入口 | 画面 |
| --- | --- | --- |
| PickPlace | `scripts/workcell/export_pickplace_tactile_review.py` | 相机、Genesis probe 图和曲线 |
| Card | `scripts/workcell/export_card_tactile_review.py` | `head`、`right_wrist`、十指力图和曲线；源 episode 须成功 |
| Card 四类示例 | `PYTHONPATH=src pixi run --as-is export-poker-example -- ...` | `head`＋右腕＋右手触觉视频、独立全局回放、五指原始力曲线；见[说明](../tasks/poker_example.md) |
| USB | `scripts/workcell/export_usb_review.py` | `head`、`right_wrist`、十指力图和曲线 |

例如，复核一条 USB 原始数据：

```bash
PYTHONPATH=src .pixi/envs/default/bin/python -B scripts/workcell/export_usb_review.py \
  /绝对路径/usb_000000.h5 \
  --output-dir /绝对路径/usb_000000_review
```

PickPlace 或 Card 换成表中的脚本即可。三个入口都接受 `--fps 5` 或 `--fps 10`，默认 10 fps、1920×1080。输出目录包含 `review.mp4`、`review.json`、`frames.jsonl`、`first_frame.png` 和 `last_frame.png`；先查看首末帧，再观看视频，并用 JSON/JSONL 核对来源、时间和逐帧数值。

Bulb、RAM、Vase、Whiteboard 的任务专属示例采集器会重新执行物理任务，其使用方式分别见
[灯泡](../tasks/bulb_tasks.md)、[内存](../tasks/install_ram.md)、[花瓶](../tasks/vase_wipe.md)
和[白板](../tasks/whiteboard_wipe.md)指南；它们不属于本页的离线导出命令。
已有 Raw 优先使用上面的统一 `replay`。模型闭环视频的布局与输出单独见
[模型评估视频](policy_evaluation_video.md)，逐帧数值保存在对应的 `frames.jsonl`。

USB 的**接触开始与结束静态对照图**另用 [Cleaning 文档中的 `review-usb-tactile`](../tasks/usb_tasks.md#触觉起止与视觉对应)。模型闭环评估的视频参数和输出见[模型评估视频](policy_evaluation_video.md)，运行入口见[统一模型推理与评估](model_evaluation.md)。
