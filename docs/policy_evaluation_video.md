# 模型评测视频

本页说明模型闭环评估视频。若要查看采集得到的原始 HDF5，请用[原始数据可视化](data_visualization.md)中的离线导出命令。

公共 `EvaluationVideo` 可供七类任务使用，生成 1920×1080 的只读评测视频。当前已注册的 PickPlace、Card、USB、Bulb、RAM、Vase、Whiteboard 模型 runner 会直接调用它，任务执行入口也复用同一组件。视频包含：

- `head` 第一人称与可配置的第三人称画面；
- 左右手十指的 `Fn`、`|Ft|` 7×5 空间分布；
- 每根手指的 `Ft_col`、`Ft_row`、`Fn` taxel 均值曲线；
- `frames.jsonl` 原始有符号力值和 `review.json` 元数据；
- 卡牌任务的当前/峰值机器人方向移动距离和 50 mm 阈值。

`Ft_col/Ft_row/Fn` 是触觉网格坐标，不是世界坐标。需要使用 XYZ 名称时，其别名是
`Fx(sensor_chart)/Fy(sensor_chart)/Fz(sensor_chart)`。

## 通用参数

统一批量入口的次数参数：

```text
--num-trials N       # 默认 20
--seed-start S       # 默认 0
--video-count V      # 默认 min(3, N)，允许 0..N
```

`--num-trials` 决定成功率统计的总 trial 数；`--video-count` 只决定前多少个
trial 生成视频，不改变模型推理、任务执行或指标统计。

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

模型输入相机以 deployment manifest 的 `observation_contract.cameras` 为准，评估视频的第二画面可以独立选择为 `global` 或腕部相机。视频未显示某个模型输入画面，不表示 runner 没有向模型传入该视角；运行方式见[统一模型推理与评估](model_evaluation.md)。

预测步数由模型通过 `action_horizon` 声明，评测 runner 不再限定为固定 32 步。
`--execute-steps` 控制每次实际执行多少步，必须满足
`1 <= execute_steps <= action_horizon`。不同模型 runner 的默认值可能不同，正式对比时应显式指定同一值。评测元数据会同时记录 `prediction_horizon`、`execute_steps` 和
`replan_period_s`。

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
