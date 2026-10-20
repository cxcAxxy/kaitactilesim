# 统一模型部署

公共入口为 `scripts/deploy/deploy.py`，Pixi 命令为 `deploy-policy`。
它调用现有模型后端，覆盖 checkpoint 准备、冻结合同核验、模型启动与仿真评估。
完整数据流程见[统一命令速查](pipeline_quickstart.md)。

以下命令均从 `/cpfs_infra/user/chenxianchi/code/kaitactilesim_unified` 执行。
本节点使用 `pixi run --as-is` 复用已安装环境，避免因缺失 pyarrow 自动重装；
这不替代完整依赖检查，模型服务仍必须使用对应的完整模型 Python。
环境状态见[环境与工作目录](pipeline_quickstart.md#环境与工作目录)，
各任务成功定义见[成功判据总表](../tasks/success_criteria.md)，诊断阶段见[分阶段评估](staged_evaluation.md)。

USB 插入的 bound checkpoint 支持 `pi05_kaihand`/H30 与 `pi05_kaihand_h50`/H50。
准备与服务从 checkpoint 的训练记录读取 `normalizer_asset_id`，不从数据目录借用旧统计；
H50 手指尾部修复使用独立资产 `normalizer_h50_usb_insert_hand_tail_v1`。
这类 bound checkpoint 生成 `usb_pi05_deployment_v2`，统一部署、批量评估与单次 runner
均验证其身份。旧无训练 provenance 的 H30 `normalizer`/v1 部署保持原行为，不能改名
伪装成 H50。新旧统计与 checkpoint 不覆盖、不混用；具体训练流程见
[USB H50 归一化](../../../openpi/examples/kaihand/README.md#usb-插入-h50手指尾部安全归一化)。

```bash
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is deploy-policy -- --help
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is deploy-policy -- --list-support
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is deploy-policy -- prepare --help
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is deploy-policy -- run --help
```

能力查询分别列出 prepare、verify、serve、run/evaluate 的支持情况。
某个模型有评估 runner，并不意味着本仓库能够为它生成新的 deployment。
π0.5 的十三个任务有完整准备与服务后端；其他模型按查询结果使用。
新增 `usb-unplug`、`whiteboard-wipe-horizontal`、`bulb-unscrew`、
`whiteboard-wipe-horizontal-line`、`bulb-screw-inplace` 五个变体必须使用新任务
训练来源绑定的 checkpoint；冻结合同会核对 task、mechanics version 和场景指纹，
旧任务 checkpoint 不能仅通过改任务名变成新任务模型。
未实现的阶段明确报错，已有模型和训练 provenance 的校验仍由专项后端执行。

`bulb-screw-inplace` 从灯座入口开始，不是原 `bulb-screw` 的 checkpoint 别名。
它只接受 `shared_task_pi05_deployment_v2`：训练来源必须绑定该任务的新数据、
`recorded-controller-target-v1`、一致的 instruction/mechanics/场景指纹。
OpenPI 的独立 H50 配置为 `pi05_kaihand_bulb_screw_inplace_controller_target_h50_v1`；
旧拿取旋入的 `pi05_kaihand_bulb_controller_target[_h50]_v1` 只用于原 `bulb-screw`，不得互换。
旧 v1 manifest 或旧抓取旋入 checkpoint 不能重标为原位旋入。
当前已通过的是新任务专家与接口验证，尚无本次新增任务的模型训练或闭环成绩；
准备前应先完成独立采集、转换和训练，见[专项命令](pipeline_quickstart.md#6-灯泡原位旋入任务)。

### USB 拔出：旧策略按新规则的显式诊断

旧 `1010_200` USB 拔出模型绑定“拔出、放回、释放”的训练提示词，不能默认为当前
`usb-unplug-held-clear-2s-v2` 的匹配模型。默认 prepare 仍拒绝这种协议不一致。
只有显式传给 shared prepare 后端 `--allow-usb-unplug-protocol-transfer-diagnostic` 时，
才允许已绑定、源哈希重算、mechanics/场景指纹完全相同的旧策略按新两秒保持规则评分。

此入口不改 checkpoint、提示词、动作语义、normalizer 或训练来源；模型仍收到原训练指令。
迁移证据记录在被 deployment ID 认证的
`training_provenance.completion_protocol_transfer` 中，包含训练/评估协议、两套场景身份和明确原因。
prepare、verify、server metadata 与 runner 必须一致，且始终
`diagnostic_only=true`、`formal_metrics_valid=false`，不能用作新任务正式成功率。
`--allow-legacy-unbound-diagnostic` 不能代替这个入口；未绑定来源、未知协议、物理场景变化仍拒绝。

九任务 10000/15000、每组五条的完整新批次命令见
[阶段诊断批次](pipeline_quickstart.md#7-九任务-1000015000-checkpoints-的五次阶段诊断)。

## 命名运行环境

转换与部署共享 TOML/JSON 配置。可将
[runtimes.example.toml](../../configs/runtimes.example.toml) 复制为
`configs/runtimes.local.toml`，修改实际 Python 和项目路径：

```toml
[runtimes.pi05]
python = "../../openpi/.venv-pi05/bin/python"
project_root = "../../openpi"

[runtimes.pi05.env]
CUDA_VISIBLE_DEVICES = "0"
OPENBLAS_NUM_THREADS = "1"
OMP_NUM_THREADS = "1"
```

路径相对配置文件所在目录解析，环境变量值是字面字符串。
通过 `--runtime-config configs/runtimes.local.toml --runtime pi05` 选择。
配置中也可设置顶层 `default_runtime = "pi05"`。prepare/run 计划及其 pipeline
记录保存所选 profile 和配置 SHA-256；TOKEN、PASSWORD 等敏感环境值会被脱敏。

WebSocket sidecar 与 local worker 使用对应模型 Python，仿真评估器继续使用 Pixi
仿真环境。`prepare/serve` 使用模型 runtime；EgoTouch worker 继承所选 runtime
环境。注册表标为 `embedded` 的 Poker π0.5+T-Rex 是例外，它在 rollout runner
内加载模型，不提供独立 `serve`。

## 准备与核验

以 Whiteboard π0.5 为例，checkpoint 必须带与该任务匹配的训练来源和转换快照：

```bash
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is deploy-policy -- prepare \
  --task whiteboard-wipe --model-family pi05 \
  --checkpoint /path/to/checkpoint \
  --output-dir /path/to/deployment \
  --runtime-config configs/runtimes.local.toml --runtime pi05 \
  --dry-run
```

核对实际命令后移除 `--dry-run`。输出目录必须是新目录。
`deployment_manifest.json` 保留原模型的冻结合同；统一层额外保存
`pipeline_deployment.json`，记录入口、运行环境和原始 manifest 的身份。
normalizer、binding、预训练视觉模型等专属资产通过 `--` 后的后端参数传入。
例如 EgoSteer 仍需要 `--model-config`、`--normalizer`、`--pretrained-vlm`；
PickPlace EgoTouch 仍需要 `--binding`。

```bash
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is deploy-policy -- verify \
  --deployment-manifest /path/to/deployment/deployment_manifest.json \
  --runtime-config configs/runtimes.local.toml --runtime pi05
```

核验使用现有 checkpoint、normalizer、模型源码与评估源码校验器。
注册表列出的旧 manifest schema（例如 shared π0.5 v1）可以作为输入，但其冻结的
源码与当前 checkout 不匹配时会拒绝执行；应使用对应源码版本，或按正式流程准备
新部署。未注册的 schema 不会被当作兼容格式猜测。历史诊断选项继续由原后端显式处理。

本次保护策略调整修改了评估源码：旧冻结 deployment 如报源码身份或 SHA-256
不匹配，需要从原 checkpoint 在**新的输出目录**重新 `prepare`，再 `verify` 和
`run`。不要手改旧 manifest 的 hash、覆盖原目录或绕过核验；历史部署与评估保留原样。
这与修改本次运行的时限、保护模式或 action 执行步数不同，后者本身不要求重新准备。

## 自动接入仿真

```bash
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is deploy-policy -- run \
  --deployment-manifest /path/to/deployment/deployment_manifest.json \
  --runtime-config configs/runtimes.local.toml --runtime pi05 \
  --output-dir /path/to/evaluation_run \
  --num-trials 20 --seed-start 0 --video-count 20 \
  --execute-steps horizon \
  --max-sim-seconds 90 --guard-mode observe \
  --dry-run
```

确认计划后移除 `--dry-run`。默认管理模式会启动模型服务、等待 WebSocket
metadata、核对部署身份与输入输出合同，再运行统一评估器，最后回收自己启动的进程。
`--num-trials 1 --video-count 1` 可先做单次完整 smoke。
默认执行步数沿用评估器：PickPlace EgoTouch 为 5，其余使用冻结的预测 horizon；
可用 `--execute-steps` 显式选择后端允许的值。上例为 π0.5，显式执行完整 horizon。
若要每次只执行 8 步，把 `--execute-steps horizon` 改为 `--execute-steps 8`。
π0.5 的合法范围为 `1 ≤ K ≤ prediction_horizon`；例如 H50/K8，模型仍预测
50 步，执行前 8 步后重新观测和推理，剩余 42 步丢弃。此参数不改变 checkpoint
或冻结 manifest，也不需要重新 prepare。
`deploy-policy run` 一次只接受一个执行长度，参数放在 `--` 分隔符之前。
需要同 seed 对比多个执行长度时，使用独立 `evaluate-policy`，见
[Action chunk 执行长度](model_evaluation.md#action-chunk-执行长度)。

### 评估时限、保护与指标

`run` 与独立 `evaluate-policy` 默认都是 `--max-sim-seconds 90 --guard-mode observe`。
时限是每次 trial 的**仿真时间**，不是墙钟运行时间，必须是有限正数；例如改成
`--max-sim-seconds 120` 即允许最多 120 秒仿真。这些统一参数放在 `--` 分隔符之前，
不能作为后端参数传入；改变它们不会修改 checkpoint 或冻结 manifest。

`observe` 用于诊断：力过大、碰撞、掉落等任务保护记录后继续仿真与录像；已识别的
无效/不可达动作保持上一条完整有效控制指令，丢弃当前 action chunk 并重新推理。
达到时限结束本次 rollout；任务成功、数值故障或基础设施故障仍可提前结束。
录像只保存已有有效帧，不冻结补足 90 秒；数值故障后不再渲染坏状态。
后端的推理请求上限等 watchdog 也保留，详见[评估停止原因](model_evaluation.md)。

需要原来的任务保护终止规则时，显式加 `--guard-mode strict`。`observe` 的结果标为
`diagnostic_only=true`、`formal_metrics_valid=false`，即使任务完成，也不能混入历史
strict 成功率；正式比较应使用 `strict`，并核对每次结果的有效性、任务版本和运行配置。
strict 只是请求正式指标，不保证存在数值或基础设施故障的 trial 仍有效。

首次可用 `--num-trials 1 --video-count 1` 检查完整闭环；若只想做短时服务/录像检查，
再加 `--smoke-seconds 10`。该选项仅属于 `deploy-policy run`，会覆盖
`--max-sim-seconds` 的实际时限；即使同时选择 `strict`，这次运行仍标为诊断，
不作为正式任务指标。独立评估没有 `--smoke-seconds`，直接设置 `--max-sim-seconds`。

输出根目录保存 `run_manifest.json`，评估结果在 `evaluation/`。运行记录保存有效
`max_sim_seconds`、`guard_mode`、`diagnostic_only` 与 `formal_metrics_requested`；
trial 的停止原因和保护事件在评估结果中检查。仅 managed
sidecar 生成根目录下的 `server.log`；local worker 的模型日志由对应 evaluator
写入评估目录。错误和子进程退出码保留在运行记录中。
运行时 endpoint 与设备选择属于本次运行，不修改冻结的模型 manifest。

接入已经启动的模型服务：

```bash
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is deploy-policy -- run \
  --deployment-manifest /path/to/deployment/deployment_manifest.json \
  --service-mode external --endpoint ws://127.0.0.1:18783 \
  --output-dir /path/to/external_evaluation \
  --num-trials 20 --video-count 20
```

外部模式同样检查服务 metadata，退出时不停止外部服务。远程地址或 `wss://`
TLS 服务必须使用 external；此模式不要求提供 `--runtime-config/--runtime`，
仿真 evaluator 不会继承模型 profile 的 CUDA/PYTHONPATH 覆盖。
冻结部署的本地资产与源码核验仍会执行；部分后端仍需要 manifest 声明的
模型项目和校验环境，不代表可以只凭远程 endpoint 跳过部署核验。
EgoTouch 使用本地 stdin/stdout worker，T-Rex 在 rollout 进程内加载模型；
这两类后端没有单独的 `serve` 阶段。T-Rex 需要显式
`--model-family pi05+trex` 选择其 π0.5 base manifest，并传入触觉 expert 等专项资产；
当前尚未提供组合模型的统一 prepare。

## 单独启动服务与独立评估

```bash
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is deploy-policy -- serve \
  --deployment-manifest /path/to/deployment/deployment_manifest.json \
  --runtime-config configs/runtimes.local.toml --runtime pi05 \
  --port 18783
```

`serve` 前台运行，便于由作业系统管理。实际绑定地址以 `--list-support` 的
`bind_host` 为准：部分既有 π0.5 后端固定绑定 `0.0.0.0`，PickPlace 等后端固定
绑定 loopback；不能实现的 `--host` 会直接拒绝。`run --service-mode managed`
使用的连接 endpoint 则只能是本机明文 `ws://`。仅需要评估已经管理好的服务时，
继续使用 [`evaluate-policy`](model_evaluation.md)。

没有 Pixi 命令但已有项目环境时，可以在仓库根目录使用
`.pixi/envs/default/bin/python scripts/deploy/deploy.py`，后接相同参数。
