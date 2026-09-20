# 原始数据可视化

## 七任务统一轨迹回放

`replay` 是 PickPlace、Card、USB、Bulb、RAM、Vase、Whiteboard 共用的回放入口。下面的命令直接读取一条已完成的 Raw HDF5，生成相机画面、双手触觉热力图和十指触觉值时间曲线，不重新运行仿真，也不修改源文件：

```bash
pixi run replay -- /绝对路径/episode.h5 \
  --viewer none \
  --review-dir /绝对路径/episode_replay
```

默认显示 HDF5 中存在的 `head + left_wrist + right_wrist`。也可以按顺序选择一至四路已保存相机，例如：

```bash
pixi run replay -- /绝对路径/episode.h5 \
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
pixi run replay -- /绝对路径/episode.h5 \
  --viewer native --speed 1.0
```

native viewer 按保存的 `qpos/qvel` 和状态时间戳恢复运动；多模态 MP4 使用保存的 RGB/触觉时钟。二者可以在一次命令中同时请求，但服务器批处理通常使用 `--viewer none`。

## 任务专属复核产物

PickPlace、Card、USB 的旧导出脚本仍可用于其历史固定布局。Bulb、RAM、Vase、Whiteboard 的专项示例入口也能在一次物理 rollout 后生成任务专属视频和曲线。统一批采仍只保存 Raw，复核产物从 Raw 离线生成。

在仓库根目录执行，并将输入文件和输出目录替换为实际路径。输出目录必须尚不存在。

| 数据 | 命令入口 | 画面 |
| --- | --- | --- |
| PickPlace | `scripts/workcell/export_pickplace_tactile_review.py` | 相机、Genesis probe 图和曲线 |
| Card | `scripts/workcell/export_card_tactile_review.py` | `head`、`right_wrist`、十指力图和曲线；源 episode 须成功 |
| Card 四类示例 | `pixi run export-poker-example -- ...` | `head`＋右腕＋右手触觉视频、独立全局回放、五指原始力曲线；见[说明](poker_example.md) |
| USB | `scripts/workcell/export_usb_review.py` | `head`、`right_wrist`、十指力图和曲线 |
| Bulb | `scripts/workcell/refresh_light_bulb_example.py` | 三路机器人相机、灯座近景、十指力图和五指力/拧紧曲线 |
| RAM | `scripts/workcell/record_install_ram_example.py` | 三路机器人相机、RAM 近景、十指力图和分阶段插入力曲线 |
| Vase | `scripts/workcell/view_vase_wipe.py --run-task --output-dir ...` | 三路机器人相机、花瓶内外视角、柔性接触力图和清洁/力曲线 |
| Whiteboard | `scripts/workcell/record_whiteboard_example.py --output-dir ...` | 机器人相机、板面载荷、十指力图和清洁/力曲线 |

例如，复核一条 USB 原始数据：

```bash
.venv/bin/python scripts/workcell/export_usb_review.py \
  /绝对路径/usb_000000.h5 \
  --output-dir /绝对路径/usb_000000_review
```

PickPlace 或 Card 换成表中的脚本即可。三个入口都接受 `--fps 5` 或 `--fps 10`，默认 10 fps、1920×1080。输出目录包含 `review.mp4`、`review.json`、`frames.jsonl`、`first_frame.png` 和 `last_frame.png`；先查看首末帧，再观看视频，并用 JSON/JSONL 核对来源、时间和逐帧数值。

后四类任务的完整示例采集与导出：

```bash
python scripts/workcell/refresh_light_bulb_example.py \
  --output /new/path/light_bulb_example

python scripts/workcell/record_install_ram_example.py \
  --output-dir /new/path/install_ram_example

python scripts/workcell/view_vase_wipe.py --headless --run-task \
  --output-dir /new/path/vase_wipe_example

python scripts/workcell/record_whiteboard_example.py \
  --output-dir /new/path/whiteboard_wipe_example
```

这些入口生成的 review 尺寸和附加产物由任务文档定义；Raw 中的相机与触觉仍遵守共享采集合同。若需要与模型评估完全一致的 1920×1080 十指热力图和三轴时间曲线，使用[模型评估视频](policy_evaluation_video.md)中的 `--evaluation-review-dir`。

USB 的**接触开始与结束静态对照图**另用 [Cleaning 文档中的 `review-usb-tactile`](usb_cleaning.md#触觉起止与视觉对应)。模型闭环评估的视频参数和输出见[模型评估视频](policy_evaluation_video.md)，运行入口见[统一模型推理与评估](model_evaluation.md)。
