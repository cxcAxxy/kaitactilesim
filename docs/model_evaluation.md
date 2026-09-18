# 统一模型推理与评估

统一入口是 `scripts/evaluate/evaluate.py`，Pixi 快捷命令为：

```bash
pixi run evaluate-policy -- --help
```

入口根据 `(task, model_family)` 选择经过任务适配的 runner，读取 deployment manifest，并把 `--` 后的参数原样传给具体 runner。

## 基本命令

```bash
pixi run evaluate-policy -- \
  --task usb-insert \
  --model-family pi05 \
  --deployment-manifest /path/to/deployment.json \
  --output-dir /path/to/evaluation \
  --cameras head right_wrist \
  --num-trials 20 \
  --video-count 5 \
  -- --server ws://127.0.0.1:18783
```

## 评估次数与录像次数

这两个参数相互独立，并由统一入口负责，不再到各任务 runner 中分别设置：

- `--num-trials N`：运行 N 次评估，默认 `20`；
- `--seed-start S`：使用连续 seed `S ... S+N-1`，默认从 `0` 开始；
- `--video-count V`：前 V 次评估生成公共 review 视频，默认
  `min(3, N)`；`0` 表示不录像，最大不能超过 N。

例如评估 20 次但只保存前 6 次视频：

```bash
pixi run evaluate-policy -- \
  --task usb-insert --model-family pi05 \
  --deployment-manifest /path/to/deployment.json \
  --output-dir /path/to/evaluation \
  --num-trials 20 --seed-start 0 --video-count 6 \
  -- --server ws://127.0.0.1:18783
```

需要每次都有视频时设置 `--video-count` 与 `--num-trials` 相同。具体
runner 参数仍放在 `--` 后，但 `--seeds` 和 `--video-count` 已归统一入口管理。

`--dry-run` 只验证 manifest、相机合同和 runner 分发，并打印最终子命令：

```bash
pixi run evaluate-policy -- \
  --task poker-draw --model-family egosteer \
  --deployment-manifest /path/to/deployment.json \
  --output-dir /path/to/evaluation \
  --dry-run
```

## Deployment manifest

统一入口至少检查以下字段；具体 runner 可以要求更多模型服务参数：

```json
{
  "deployment_id": "usb_pi05_example",
  "task": "usb-insert",
  "model_family": "pi0.5",
  "observation_contract": {
    "cameras": ["head", "right_wrist"]
  }
}
```

- manifest 中声明的任务必须与 `--task` 一致。
- `model_family` 必须与 `--model-family` 一致；`pi0.5` 和 `pi05` 作为同一族处理。
- `--cameras` 是对冻结合同的断言，不会修改模型输入或 checkpoint。
- 旧模型继续使用其训练时的 head-only/双相机合同；需要三相机时必须重新转换数据、训练并发布新的 manifest。

相机合同统一使用规范顺序，当前网络 runner 支持以下四种组合：

- `head`
- `head left_wrist`
- `head right_wrist`
- `head left_wrist right_wrist`

其中 EgoSteer 和 π0.5 会按照 manifest 同步采集并发送所声明的全部视角。
π0.5 的 head-only 请求沿用 `image`；多相机请求默认使用
`head_image`、`left_wrist_image`、`right_wrist_image`，也可以由 manifest 的
`observation_contract.request_image_keys` 冻结服务端实际字段名。当前 EgoTouch
worker 对应既有单路 RGB 网络，只接受 `cameras: ["head"]`；给它声明腕部相机会在
启动模型前明确报错。这里的“可选”表示 runner 能按 checkpoint 合同选择，并不表示
head-only checkpoint 能在评估时临时增加腕部输入。

## 当前 runner 覆盖

| 任务 | 公共可视化 | EgoSteer | π0.5 | EgoTouch |
|---|---:|---:|---:|---:|
| PickPlace | 已接入 | 已接入 | 已接入 | 待接入 |
| Card | 已接入 | 已接入 | 已接入 | 待接入 |
| USB | 已接入 | 已接入 | 已接入 | 待接入 |
| Bulb | 已接入 | 已接入 | 已接入 | 已接入（head-only） |
| RAM | 已接入 | 已接入 | 已接入 | 已接入（head-only） |
| Vase | 已接入 | 已接入 | 已接入 | 已接入（head-only） |
| Whiteboard | 已接入 | 已接入 | 已接入 | 已接入（head-only） |

不支持的组合会在启动仿真前报错，不会退回其他任务 runner。
“已接入”表示闭环 observation/action、任务判据、批量统计和视频链路已经实现；实际
运行仍必须提供针对该任务训练且合同匹配的 checkpoint、deployment manifest，以及
EgoSteer/π0.5 模型服务或 EgoTouch 本地模型环境。

七类任务的仿真对象均可接入公共 `EvaluationVideo`。Bulb、RAM、Vase、Whiteboard 的三个
模型 runner 共用任务适配层：模型只接收 manifest 声明的 observation，输出动作直接
闭环执行；适配层只负责每个物理步更新任务状态、安全限制与成功判据，不调用已知状态
动作控制器或专家轨迹。

新任务使用同一个统一命令。例如 RAM 的三相机 π0.5 模型：

```bash
pixi run evaluate-policy -- \
  --task install-ram --model-family pi05 \
  --deployment-manifest /path/to/ram_pi05/deployment.json \
  --output-dir /path/to/ram_pi05_eval \
  --cameras head left_wrist right_wrist \
  --num-trials 20 --video-count 5 \
  -- --server ws://127.0.0.1:18783
```

EgoTouch 使用本地私有模型环境，manifest 仍负责冻结 checkpoint，运行参数放在
统一入口的 `--` 后：

```bash
pixi run evaluate-policy -- \
  --task vase-wipe --model-family egotouch \
  --deployment-manifest /path/to/vase_egotouch/deployment.json \
  --output-dir /path/to/vase_egotouch_eval \
  --cameras head --num-trials 20 --video-count 5 \
  -- --model-python /path/to/egotouch/python \
     --model-project /path/to/egotouch
```

## 评估职责边界

统一 dispatcher 负责：

- manifest、任务、模型族和相机合同检查；
- runner 查找与参数转发；
- 防止覆盖已有评估目录；
- 输出可审计的实际运行命令。

任务 runner 负责：

- 模型 observation 编码和 action 解码；
- action horizon、重规划频率和控制执行；
- 任务 reset、seed、超时和安全限制；
- 任务成功指标、失败原因、逐 trial 结果和聚合统计；
- 可选评估视频与触觉复核产物。

评估视频内容和参数见[模型评测视频](policy_evaluation_video.md)。

## 输出与运行原则

- 每次评估使用新的 `--output-dir`，统一入口拒绝覆盖已有目录。
- 正式批量前先执行 `--dry-run`，再运行单 seed smoke test。
- 模型服务端口、checkpoint、推理配置和 runner 参数放在 manifest 或 `--` 后，不写进通用文档。
- 比较模型时保持任务版本、随机 seed、最大时长、action horizon、execute steps、相机合同和成功指标一致。
- 评估视频是复核产物，不代替结构化成功指标和失败原因。

## 新 runner 接入

新增模型或任务时：

1. 冻结 observation/action 和相机合同；
2. 实现任务专属 runner 与成功指标；
3. 在 `RUNNERS` 注册 `(task, model_family)`；
4. 生成与 checkpoint 配套的 deployment manifest；
5. 增加 manifest 不匹配、相机不匹配、dry-run 分发和单 seed smoke 测试。

接入 runner 不应修改任务原有动作控制器；模型闭环行为属于评估适配层。
