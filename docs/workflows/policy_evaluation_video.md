# 模型评测视频

本页说明模型闭环评估视频。若要查看采集得到的原始 HDF5，请用[原始数据可视化](data_visualization.md)中的离线导出命令。

公共 `EvaluationVideo` 可供七类任务使用，生成 1920×1080 的只读评测视频。当前已注册的 PickPlace、Card、USB、Bulb、RAM、Vase、Whiteboard 模型 runner 会直接调用它，任务执行入口也复用同一组件。视频包含：

- `head`、右腕和可配置的第三人称画面（第二画面设为右腕时避免重复）；
- 左右手十指放大的 `Fn`、`|Ft|` 7×5 空间分布；
- `frames.jsonl` 中每帧的原始有符号 taxel 力值、三轴 taxel 均值和
  `review.json` 元数据；
- 卡牌任务的当前/峰值机器人方向移动距离和 50 mm 阈值。

`review.mp4` 不再绘制触觉随时间变化的曲线，腾出的画面用于放大 RGB 和
触觉热力图。逐 taxel 的 `Fn`、有符号 `Ft_col/Ft_row` 以及兼容旧分析脚本的
每指三轴均值仍完整写入 `frames.jsonl`，视频布局变化不会删减这些数据。

`Ft_col/Ft_row/Fn` 是触觉网格坐标，不是世界坐标。需要使用 XYZ 名称时，其别名是
`Fx(sensor_chart)/Fy(sensor_chart)/Fz(sensor_chart)`。

## 通用参数

统一批量入口的次数参数：

```text
--num-trials N       # 默认 20
--seed-start S       # 默认 0
--video-count V      # 默认 N，即默认全部录像；允许 0..N
```

`--num-trials` 决定成功率统计的总 trial 数；`--video-count` 决定前多少个
trial 生成 review 视频及其逐控制步轨迹、对比图，不改变模型推理、任务执行或指标统计。

单个 runner/任务执行入口的视频参数：

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

预测步数由模型通过 `action_horizon` 声明，评测 runner 不再限定为固定 32 步。
`--execute-steps` 控制每次实际执行多少步，必须满足
`1 <= execute_steps <= action_horizon`。统一正式评估默认只运行完整
`action_horizon`；需要比较两组时可显式传 `--execute-steps 16 horizon`
（前提是模型 horizon 大于 16）。评测元数据会同时记录
`prediction_horizon`、`execute_steps` 和 `replan_period_s`。

## 视频输出

所有任务的模型服务参数通过统一评估入口后的 `--` 传给具体 runner。每次评估使用与 checkpoint 配套的 deployment manifest 和新输出目录，不复用其他模型的相机合同或结果目录。

每个被录像的 trial 在 `review/` 下生成：

```text
review.mp4
review.json
frames.jsonl
first_frame.png
last_frame.png
```

注册模型的新版正式评估在每个控制步记录真实物理状态与右手五指的 7×5
触觉网格，而视频仍按 `--record-fps` 降采样。如果提供且可以读取与 checkpoint
对应的数据集，`review/` 还会生成三张 episode `00` 参考轨迹/闭环真实轨迹对比图。
与视频一起输出的通用文件为：

```text
evaluation_rollout_trace.npz
evaluation_comparison.json
evaluation_right_wrist_state.png
evaluation_right_hand_actuated_dof.png
evaluation_right_fingertip_tactile.png
```

缺少参考数据或参考数据没有所需字段时，仍保留 30 Hz 的
`evaluation_rollout_trace.npz`；原因写在 `evaluation_comparison.json` 和
`review.json.comparison_plots`，不把其他版本的数据冒充参考。

OpenWAM USB 当前使用独立绘图入口，仍在同一目录生成以下同口径文件：

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

Bulb、RAM、Vase 的模型闭环 runner 和原任务执行器都可生成公共可视化；前者用于正式模型评估，后者只用于检查任务与传感器。
这些命令运行的是已知状态控制器，不计入模型成功率：

```bash
python scripts/workcell/view_bulb_screw.py --headless --run-task \
  --evaluation-review-dir /new/path/bulb_review

python scripts/workcell/view_install_ram.py --headless --run-task \
  --evaluation-review-dir /new/path/ram_review

python scripts/workcell/view_vase_wipe.py --headless --run-task \
  --evaluation-review-dir /new/path/vase_review
```

三个入口都接受 `--evaluation-video-fps 5|10`。Vase 自动使用能识别柔性海绵
接触的触觉 provider；另外五类任务使用共享刚体接触 provider。三者均生成
`review.mp4`、`review.json`、`frames.jsonl`、`first_frame.png` 和
`last_frame.png`。
