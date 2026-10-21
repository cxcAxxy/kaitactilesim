# 模型评测视频

本页维护画面布局、录制参数、导出产物与可视化解释；
trial、保护、停止和统计见[模型评估](model_evaluation.md)，原始 Raw 回放见[数据可视化](data_visualization.md)。

命令和导入路径按[运行环境](environment.md)设置；模型特有录像/数据例外见[模型专项](model_specific_evaluation.md)。
网页查看、离线导入和站点核验见[统一观察台](unified_evaluation.md)。

## 画面内容与单位

公共 `EvaluationVideo` 生成默认 1920×1080、10 FPS 的只读评测视频，统一入口已注册的
13 个任务均接入它：`pick-place`、`poker-draw`、`usb-insert`、`bulb-screw`、
`install-ram`、`vase-wipe`、`whiteboard-wipe`、`sponge-grasp`、`usb-unplug`、
`whiteboard-wipe-horizontal`、`bulb-unscrew`、`whiteboard-wipe-horizontal-line`、`bulb-screw-inplace`。
各任务支持的模型族并不相同，以
`evaluate-policy -- --list-support` 为准；部分任务执行入口也复用同一组件。
下面描述公共视频组件，不改变不同模型的输入合同。视频包含：

- `head`、右腕和可配置的第三人称画面（第二画面设为右腕时避免重复）；
- 左右手十指放大的 `Fn`、`|Ft|` 7×5 空间分布；
- `frames.jsonl` 中每帧的原始有符号 taxel 力值、三轴 taxel 均值和
  `review.json` 元数据；
- 卡牌任务的当前/峰值机器人方向移动距离和 50 mm 阈值。

灯泡三类任务的旋拧读数以角度（°）显示：旋入 `clockwise_angle_deg`、
旋出 `angle_removed_deg`，原位旋入另记录相对实际初态的 `rotation_from_initial_deg`。
这些是原净圈数乘以 360 的单位换算，反转仍会抵消；不表示手指动作次数或累计转动总角度。
名义机械到座角为 100°，旋出名义脱扣角为 131.5°；达到参考角度还需满足相应接触、
拧紧或脱扣、释放等成功条件。底层圈数和原生判据保留，JSON 同时保存角度别名。
本显示变更用于新录制的视频；已编码的历史 MP4 保留原样。

`review.mp4` 不再绘制触觉随时间变化的曲线，腾出的画面用于放大 RGB 和
触觉热力图。逐 taxel 的 `Fn`、有符号 `Ft_col/Ft_row` 以及兼容旧分析脚本的
每指三轴均值仍完整写入 `frames.jsonl`，视频布局变化不会删减这些数据。

`Ft_col/Ft_row/Fn` 是触觉网格坐标，不是世界坐标。需要使用 XYZ 名称时，其别名是
`Fx(sensor_chart)/Fy(sensor_chart)/Fz(sensor_chart)`。

## 录像参数

`--video-count` 选择按 seed 顺序录制的 trial，不是只保存成功视频；次数和默认值统一见
[评估次数与录像次数](model_evaluation.md#评估次数与录像次数)。
`--record-fps 5|10` 是统一入口参数，写在 `--` 前。其它常见 runner 视频选项如下，
各 batch 的透传能力和任务执行器的参数名以自身 `--help` 为准：

```text
--record / --no-record
--record-fps 5|10
--review-width 1920
--review-height 1080
--review-render-width 640
--review-render-height 480
--review-second-camera global|right_wrist|overhead
```

模型输入图像的 `--width/--height` 与 review 渲染分辨率相互独立；修改 review 参数不会改变模型输入。

模型输入相机以 deployment manifest 的 `observation_contract.cameras` 为准，评估视频的第二画面可以独立选择为 `global` 或腕部相机。合同含 `right_wrist` 时会自动增加右腕模型输入画面；若第二画面已经是 `right_wrist`，则不重复。当前公共布局不保证单独展示左腕画面，但 runner 仍会按照合同向模型传入该视角。运行方式见[统一模型推理与评估](model_evaluation.md)。

对于只使用头部相机的模型，正式评估也可增加右腕画面供回看，画面标为
`RIGHT WRIST / REVIEW ONLY`，`review.json.model_input_cameras_displayed`
仍只列真实模型输入。右腕画面的展示不改变模型输入或评估协议。

## 录像收尾与播放时长

仿真为何停止、guard 何时恢复或终止，统一见[停止原因与批次中断](model_evaluation.md#停止原因与批次中断)。
正常收尾只封装已有有效帧，不冻结末帧补足预算；失败 trial 也可保留截至退出的录像。
数值失效后不读取或渲染坏态终帧，尝试封装此前有效帧；首帧前失败、编码故障或强制终止可能没有完整 MP4。

### 仿真时间、墙钟与播放时长

`--max-sim-seconds` 限制 rollout 的仿真时间预算，不是运行进程的真实耗时；等待
模型推理、渲染和编码会增加 `wall_seconds`，等待时仿真时间不前进。
`sim_seconds` 通常是终止时的仿真时钟值；部分任务有初始化/预滚动，应结合该任务
记录的控制 epoch/初始时间计算实际 rollout 时长，不要把它直接当作 MP4 时长。

公共录像按 30 Hz 控制 tick 降采样到 5/10 FPS，不按墙钟等待补帧。
`review.json.playback_duration_s = frame_count / fps`，而真实取样跨度为
`last_pose_time_s - first_pose_time_s`。初帧和有效的非采样网格终帧会使两者略有差异；
终帧与上一帧同时间戳时不会重复录制。`frames.jsonl.simulation_time_s`、
`camera_pose_time_s` 和 `tactile_time_s` 用于定位帧对应的仿真状态；录像组件本身
不推进或 `forward` 物理。

## 视频输出

批次文件与 seed 目录先查[结果字段与统计口径](model_evaluation.md#结果字段与统计口径)，
本节列单条选中录像的公共产物。触觉后端及 LingBot 的例外见[模型专项](model_specific_evaluation.md)。

新评估不再创建 `review/` 子目录，视频、帧日志、轨迹和参考图都直接放在单条
`seed_<seed>/` 下，与 `summary.json` 同级；`review.mp4` 和 `review.json` 的文件名
不变。历史输出保持原来的 `seed_<seed>/review/` 布局，不自动迁移或删除；批次读取
和续跑校验同时支持新旧布局，源码/协议一致性检查仍然适用。

正常完成录制的 trial 在对应的 `seed_<seed>/` 下生成：

```text
review.mp4
review.json
frames.jsonl
first_frame.png
last_frame.png
```

注册模型的新评估在每个控制步记录真实物理状态与右手五指的 7×5
触觉网格，而视频仍按 `--record-fps` 降采样。如果提供且可以读取与 checkpoint
对应的数据集，该 seed 目录还会生成三张参考轨迹/闭环真实轨迹对比图（默认 episode
`00`，可用 `--reference-episode-index` 改变）。
与视频一起输出的通用文件为：

```text
evaluation_rollout_trace.npz
evaluation_comparison.json
evaluation_right_wrist_state.png
evaluation_right_hand_actuated_dof.png
evaluation_right_fingertip_tactile.png
evaluation_evaluation_plots.json
```

三张 PNG 与绘图 metadata 需要参考字段可用且绘图成功；上述重复的
`evaluation_evaluation_plots.json` 文件名来自公共绘图组件，不表示额外一级目录。

参考图的绝对时间轴只适合查看各自的变化，不构成“同一时刻应做同一动作”的评判。
专家与模型速度、持续时间、接触先后可能不同；任务版本、初态、触觉 provider 和单位也必须
一致后才可比较。解释成功/失败应回到真实物理事件、任务效果和相应阶段，而非要求触觉曲线
形状一致；时间曲线本身不能证明触觉变化导致了某次动作。阶段诊断的当前范围见
[分阶段评估](staged_evaluation.md)。

未提供参考数据，或参考数据没有所需字段时，正常录制仍保留控制步采样的
`evaluation_rollout_trace.npz`；原因写在 `evaluation_comparison.json` 和
`review.json.comparison_plots`，不把其他版本的数据冒充参考。若批次已声明
`--reference-dataset`，对应 runner 还会校验参考对比产物；缺失/不可用可能被认定为
产物合同失败，并不是“有 MP4 就有效”。数值失效或录制故障时，也不保证完整轨迹。

### 录像元数据与 Raw

`task_success`、`stop_reason`、`physics_valid`、`record_error` 和正式计分资格由 trial summary 与 batch 共同解释，
定义统一见[评估结果](model_evaluation.md#结果字段与统计口径)，不能只看 MP4 是否存在。

`review.json` 的公共字段为 `completed`、`task_status`、`task_error`、`evaluation`、
`frame_count`、`fps` 和上述视频时间字段；runner 还会在其顶层或 `evaluation` 中
补入结果/保护信息。各 runner 补充字段位置不完全相同，完整结束原因仍以 trial
`summary.json` 为准。`completed=true` 只说明存在帧且视频封装流程完成，不代表
任务成功、参考图正确或 trial 有效。`frames.jsonl` 保存逐视频帧触觉和任务指标，
不是完整的保护事件日志。

卡牌 π0.5/TREX 链路在选中录像的 trial 还记录 `raw/episode.h5`；正常完成 Raw
收尾时生成 `raw/episode.result.json` 和 `curves/right_hand_force_curves.png`。
物理失效后会关闭已记录的 Raw，不追加坏态终样本，并以
`raw.validated=false`、`raw.terminal_capture_skipped=physics_invalid` 标明不完整；
它不能当作通过验证的训练 episode。

旧 OpenWAM USB 使用独立 runner/绘图入口，不在当前统一入口注册表内，也不应套用
统一保护协议；其对比图仍在同一目录生成以下同口径文件：

```text
openwam_right_wrist_state.png
openwam_right_hand_actuated_dof.png
openwam_right_fingertip_tactile.png
openwam_evaluation_plots.json
```

实线为训练数据集输出 episode `00`，虚线为本次 rollout 完成物理步之后读取的
真实状态，不使用模型 action 代替观测。右腕图包含 native
`hand_r_base_link_site` 的 `xyz + rot6d` 九维状态；右手关节图把 20 个独立驱动
自由度放在一张 5×4 大图中。触觉图按拇指到小指排列，`Fn` 是 7×5 法向 taxel
力之和，`|Ft|` 先分别对有符号 `Ft_col/Ft_row` 求和，再计算
`sqrt(sum(Ft_col)^2 + sum(Ft_row)^2)`。

单次录像可传 `--reference-dataset`；OpenWAM 批量入口默认从 checkpoint 的
`dataloader.dataset_dir` 读取。episode `00` 优先通过
`meta/kaihand_source_episodes.jsonl` 映射到原始 HDF5；fast pi0.5 v2.1 数据集则从
`meta/kaihand_fast_pi05_conversion.json` 取得 output/source episode 映射，再以 Raw
collection `summary.json` 中唯一的成功记录定位 HDF5，不依赖文件名猜测。

卡牌峰值位移由 runner 在每个物理步更新，视频帧只读取缓存值，因此不会遗漏帧间峰值，也不会推进或 `forward` 仿真。

## 新任务接入检查

Bulb、RAM、Vase 的模型闭环 runner 和原任务执行器都可生成公共可视化；前者用于
模型评估（是否正式取决于保护模式和有效性），后者只用于检查任务与传感器。
这些命令运行的是已知状态控制器，不计入模型成功率：

```bash
cd /cpfs_infra/user/chenxianchi/code/kaitactilesim
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHONPATH=src .pixi/envs/default/bin/python -B scripts/workcell/view_bulb_screw.py --headless --run-task \
  --evaluation-review-dir /new/path/bulb_review

PYTHONPATH=src .pixi/envs/default/bin/python -B scripts/workcell/view_install_ram.py --headless --run-task \
  --evaluation-review-dir /new/path/ram_review

PYTHONPATH=src .pixi/envs/default/bin/python -B scripts/workcell/view_vase_wipe.py --headless --run-task \
  --evaluation-review-dir /new/path/vase_review
```

三个入口都接受 `--evaluation-video-fps 5|10`。Vase 自动使用能识别柔性海绵
接触的触觉 provider；其他任务按自身配置选择刚体/柔体 provider，不要由视频任务名
推断触觉来源，应查看 `review.json.tactile_source`。正常完成录制时三者均生成
`review.mp4`、`review.json`、`frames.jsonl`、`first_frame.png` 和
`last_frame.png`。

不要把上面三个旧任务执行入口的 `--evaluation-review-dir` 套用到新
`view_bulb_screw_inplace.py`；新入口仅提供场景与专家流程，视频使用
`record_task_variant.py --task bulb-screw-inplace` 保存 Raw 后经 `replay` 导出。
新任务的物理成功不由亮灯、回 home 或拧动次数推断；具体门槛见[专项判定](../tasks/bulb_tasks.md#判定与记录)。


白板额外接触强度、五指空间分布和整体触觉变化的解释及离线命令统一见[白板任务](../tasks/whiteboard_wipe.md)。
