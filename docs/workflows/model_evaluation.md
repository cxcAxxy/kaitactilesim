# 统一模型推理与评估

统一入口是 `scripts/evaluate/evaluate.py`，Pixi 快捷命令为：

```bash
pixi run evaluate-policy -- --help
pixi run evaluate-policy -- --list-support
```

若当前 shell 报 `pixi: 未找到命令`，但仓库已有 `.pixi/envs/default/` 环境，
可从仓库根目录直接用
`.pixi/envs/default/bin/python scripts/evaluate/evaluate.py` 替换
`pixi run evaluate-policy --`，后面的参数保持不变。本机也可直接调用
`/cpfs_infra/user/chenxianchi/tools/pixi/bin/pixi`；不要改用当前激活的
`openwam_alpha` Python 运行评估。

三阶段命令模板见[统一命令速查](pipeline_quickstart.md)。`--list-support` 直接从
runner 注册表列出任务、模型和底层脚本，无需先准备 deployment manifest。

入口根据 `(task, model_family)` 选择经过任务适配的 runner，读取 deployment manifest。
`--` 后可传任务 runner 的专属参数，但不能覆盖统一入口已校验的 manifest、输出目录、
任务、模型族、seed 和执行长度。

## 基本命令

```bash
pixi run evaluate-policy -- \
  --task usb-insert \
  --model-family pi05 \
  --deployment-manifest /path/to/deployment.json \
  --output-dir /path/to/evaluation \
  --cameras head right_wrist \
  --num-trials 20 \
  --video-count 20 \
  --execute-steps horizon \
  --reference-dataset /path/to/matching/training_dataset \
  -- --server ws://127.0.0.1:18783
```

## 评估次数与录像次数

这两个参数相互独立，并由统一入口负责，不再到各任务 runner 中分别设置：

- `--num-trials N`：运行 N 次评估，默认 `20`；
- `--seed-start S`：使用连续 seed `S ... S+N-1`，默认从 `0` 开始；
- `--video-count V`：前 V 次评估生成公共 review 视频及逐步轨迹/对比图，默认
  与 `N` 相同；默认 20 次全部录像，`0` 表示不录像，最大不能超过 N。

例如评估 20 次但只保存前 6 次视频：

```bash
pixi run evaluate-policy -- \
  --task usb-insert --model-family pi05 \
  --deployment-manifest /path/to/deployment.json \
  --output-dir /path/to/evaluation \
  --num-trials 20 --seed-start 0 --video-count 6 \
  -- --server ws://127.0.0.1:18783
```

具体 runner 参数仍放在 `--` 后，但 `--seeds` 和 `--video-count` 已归统一入口管理。

## Action chunk 执行长度

默认执行 deployment manifest 声明的完整 `prediction_horizon`：

```text
--execute-steps horizon
```

这里控制的是每次模型预测后实际执行的 action 数量，不是改变模型自身的预测 horizon。
执行长度不能超过 checkpoint 的 horizon。若特意比较不同重规划间隔，仍可显式传
`--execute-steps 16 horizon`；此时两组使用相同 seed，输出到不同子目录，根目录写
`evaluation_matrix.json`。不同执行长度属于不同评估协议，结果不能混为一组。

`--reference-dataset` 可明确指定与 checkpoint 对应的 LeRobot 数据集，默认对比
第 00 条。若未指定，统一入口仅在 checkpoint 路径包含明确的同任务、同版本
`sim/<task>/<model>/<version>/checkpoints` 结构时推导参考路径：已存在的旧 sibling
`lerobot_v3/<version>` 布局保持优先；若旧布局不存在，则选择同版本目录中带
`meta/info.json` 的训练数据集。
参考字段或版本缺失时，新录像仍保存逐控制步物理轨迹及不可用原因，不会用别的数据集
补画参考实线。

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

### 共享任务 π0.5 deployment

Bulb、RAM、Vase 和 Whiteboard 使用同一套严格 deployment 工具。先在 OpenPI
环境中冻结 task、prompt、normalizer、checkpoint 内容哈希和模型合同：

```bash
/path/to/openpi/.venv-pi05/bin/python \
  scripts/workcell/prepare_shared_task_pi05_deployment.py \
  --task bulb-screw \
  --checkpoint /path/to/checkpoint/25000 \
  --openpi-root /path/to/openpi \
  --output-dir /path/to/bulb_step25000_deployment
```

可先增加 `--dry-run`，只检查 checkpoint、OpenPI 配置、normalizer 和相机合同，
不计算完整哈希且不写目录。生成 manifest 后启动与其绑定的服务。
OpenPI 的 Python 环境未必安装本仓库的 `kaihand_tactile_env`，所以服务命令需把
本仓库 `src/` 加入 `PYTHONPATH`；这不修改 checkpoint 或 manifest：

```bash
PYTHONPATH=/path/to/kaitactilesim/src \
  CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /path/to/openpi/.venv-pi05/bin/python \
  scripts/workcell/serve_shared_task_pi05_policy.py \
  --deployment-manifest /path/to/bulb_step25000_deployment/deployment_manifest.json \
  --port 18783
```

服务会拒绝任务 prompt、checkpoint、normalizer、相机、action 或 OpenPI 配置不一致的
manifest，不能把 USB deployment 改路径后用于共享任务。

### Bulb 0923：20000 step 完整命令

以下命令均在本仓库根目录执行，选择的是这一个**已提交的 checkpoint 目录**：

```text
/nas/chenxianchi/datasets/sim/bulb-screw/pi05/0923_200/checkpoints/pi05_kaihand/bulb-screw_bs128_25k/20000
```

`20000` 是训练 step，不是评估次数；要评估其他 step，就把下面生成命令中的
`--checkpoint` 改为对应的已提交目录，并为新 manifest、评估结果选择不同的
输出目录。训练进程可能清理或轮换 checkpoint；使用前需确认目录中有
`_CHECKPOINT_METADATA`、`params/` 和 `assets/normalizer/norm_stats.json`。
不要选 `*.orbax-checkpoint-tmp-*` 临时目录。

先生成 deployment manifest（可先在末尾加 `--dry-run` 只检查合同；正式执行时去掉）：

```bash
cd /cpfs_infra/user/chenxianchi/code/kaitactilesim
/cpfs_infra/user/chenxianchi/code/openpi/.venv-pi05/bin/python \
  scripts/workcell/prepare_shared_task_pi05_deployment.py \
  --task bulb-screw \
  --checkpoint /nas/chenxianchi/datasets/sim/bulb-screw/pi05/0923_200/checkpoints/pi05_kaihand/bulb-screw_bs128_25k/20000 \
  --openpi-root /cpfs_infra/user/chenxianchi/code/openpi \
  --output-dir /cpfs_infra/user/chenxianchi/evaluations/bulb/deploy_0923_step20000
```

生成文件是
`/cpfs_infra/user/chenxianchi/evaluations/bulb/deploy_0923_step20000/deployment_manifest.json`。
它冻结 checkpoint 内容哈希、任务、prompt、normalizer、相机与动作合同；生成目录必须
事先不存在。该 checkpoint 的只读预检报告 `head + right_wrist`、预测 horizon 30。

在**另一个终端**启动与此 manifest 绑定的模型服务，保持进程运行：

```bash
cd /cpfs_infra/user/chenxianchi/code/kaitactilesim
PYTHONPATH=/cpfs_infra/user/chenxianchi/code/kaitactilesim/src \
  CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /cpfs_infra/user/chenxianchi/code/openpi/.venv-pi05/bin/python \
  scripts/workcell/serve_shared_task_pi05_policy.py \
  --deployment-manifest /cpfs_infra/user/chenxianchi/evaluations/bulb/deploy_0923_step20000/deployment_manifest.json \
  --port 18783
```

服务就绪后，在评估终端运行 20 个 seed、20 个复核视频。
这里直接使用仓库的 Pixi Python，不依赖 `pixi` 命令是否在 `PATH` 中：

```bash
cd /cpfs_infra/user/chenxianchi/code/kaitactilesim
.pixi/envs/default/bin/python scripts/evaluate/evaluate.py \
  --task bulb-screw --model-family pi05 \
  --deployment-manifest /cpfs_infra/user/chenxianchi/evaluations/bulb/deploy_0923_step20000/deployment_manifest.json \
  --output-dir /cpfs_infra/user/chenxianchi/evaluations/bulb/pi05_0923_step20000_n20 \
  --cameras head right_wrist \
  --num-trials 20 --seed-start 0 --video-count 20 \
  --execute-steps horizon \
  --reference-dataset /nas/chenxianchi/datasets/sim/bulb-screw/pi05/0923_200 \
  -- --server ws://127.0.0.1:18783
```

正式运行前可先给评估命令加 `--dry-run` 检查分发，再把 `--num-trials` 和
`--video-count` 都改为 `1`、改用**另一个新输出目录**做单 seed smoke。
`--dry-run` 必须放在分隔符 `-- --server` 之前；它不连接模型服务，也不验证输出目录尚未存在。上例的 `horizon` 为
30 个控制步；manifest 的建议重规划间隔为 8 步，若改成 `--execute-steps 8`
即属于另一套评估协议，结果须分目录比较。参考数据集只是绘图参考，
不是模型权重；这里显式选与 checkpoint 同版本的 π0.5 训练数据集。
服务和评估在不同机器时，`127.0.0.1` 必须换为服务所在机器的可达地址。

Bulb 0920 的 20k/25k 正式评估可由独立子脚本完成。它会逐 checkpoint 执行 deployment、
单 seed 录像 smoke、模型服务和按模型 horizon 的正式评估（20 个 trial、20 个视频）：

```bash
cd /cpfs_infra/user/chenxianchi/code/kaitactilesim
bash scripts/workcell/run_bulb_pi05_0920_evaluation.sh 20000 25000
```

脚本会从 checkpoint root 的 `/checkpoints/` 前缀推导参考数据集，也可用
`BULB_PI05_REFERENCE_DATASET` 显式冻结。fast pi0.5 v2.1 数据集会通过
`meta/kaihand_fast_pi05_conversion.json` 和 Raw collection `summary.json` 回溯原始 HDF5；
smoke 必须成功生成三张参考对比图后才进入正式 20 次评估。

脚本只清理自己启动的模型服务，并通过服务日志判断 WebSocket 是否就绪，不会用裸 TCP
连接制造握手错误。即使脚本失败，退出的也只是 `bash` 子进程，不会关闭当前终端。

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

当相机合同含 `right_wrist` 时，公共 review 会自动增加右腕模型输入画面；若第二画面
本身已经选择 `right_wrist`，则不重复显示。该规则同时适用于 EgoSteer 和 π0.5。

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

Card 另有 `pi05+trex` 专项 runner，依赖本地 OpenPI、T-Rex expert 与触觉权重；
它与上述常规模型服务的部署流程不同，使用前须检查专项脚本中的路径及 checkpoint。
PickPlace 的 `lingbot-vla2` 也已接入统一分发，但仍使用自己的冻结 deployment
manifest、模型服务和[专项合同](../tasks/lingbot_pickplace_evaluation.md)。USB OpenWAM 的
batch 参数与 deployment 合同不同，目前保留专项命令，不在本表中冒充通用 runner。
Sponge 目前没有统一评估 runner。

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
  --num-trials 20 --video-count 20 --execute-steps horizon \
  -- --server ws://127.0.0.1:18783
```

EgoTouch 使用本地私有模型环境，manifest 仍负责冻结 checkpoint，运行参数放在
统一入口的 `--` 后：

```bash
pixi run evaluate-policy -- \
  --task vase-wipe --model-family egotouch \
  --deployment-manifest /path/to/vase_egotouch/deployment.json \
  --output-dir /path/to/vase_egotouch_eval \
  --cameras head --num-trials 20 --video-count 20 \
  --execute-steps horizon \
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

## 串行与并行

当前正式 batch 是串行评估，固定 `workers=1`。protocol 中的 `workers` 字段是运行记录，
不是仅修改数值就会生效的并行开关。要并行需要 batch 调度器同时管理多个独立 trial
子进程、输出目录、失败回收和聚合写入；还必须确认模型服务的并发语义。当前 EgoSteer
和 π0.5 WebSocket 服务在事件循环中同步调用单个模型的 `infer`，多个 rollout 即使并发
连接，GPU 推理仍会串行，并可能改变随机采样请求顺序。因此正式可比结果仍使用串行模式。

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
