# 统一模型部署

本页维护 prepare / verify / serve / run 生命周期、manifest 和进程管理；
运行参数与统计见[模型评估](model_evaluation.md)，模型特有资产和合同见[模型专项](model_specific_evaluation.md)。

公共入口为 `scripts/deploy/deploy.py`，Pixi 命令为 `deploy-policy`。它是无模型依赖的协调层，
调用 `scripts/workcell` 专项后端，不导入模型包，也不替换原任务/模型校验器。
命令从当前仓库执行；仿真/模型 Python 和依赖要求见[运行环境](environment.md)。

```bash
cd /cpfs_infra/user/chenxianchi/code/kaitactilesim
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
pixi run --as-is deploy-policy -- --list-support
pixi run --as-is deploy-policy -- prepare --help
pixi run --as-is deploy-policy -- verify --help
pixi run --as-is deploy-policy -- serve --help
pixi run --as-is deploy-policy -- run --help
```

能力查询分别列出 prepare、verify、serve、run/evaluate。评估 runner 已注册不代表支持所有生命周期阶段，
未实现的阶段会明确拒绝。三种服务形态如下：

| 形态 | 生命周期 |
|---|---|
| `websocket-sidecar` | managed run 启动并回收自己的模型服务；external 只接入调用者管理的服务 |
| `local-worker` | evaluator 用声明的模型 Python 创建私有 worker，不提供独立 serve |
| `embedded` | 模型在 rollout runner 内加载，不提供独立 serve；组合模型的 prepare 能力须查注册表 |

具体模型属于哪种形态以支持查询为准；运行时环境和专项资产见[模型专项](model_specific_evaluation.md)。

## Deployment manifest

专项 prepare 冻结 checkpoint、normalizer、输入/动作、模型与评估源码身份。
统一评估入口至少读取下列字段，专项后端还会要求自身完整 schema，不能把此最小示例作为可运行 deployment：

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

manifest 的 task/model_family 必须与评估参数一致；`pi0.5` 与 `pi05` 归为同族。
`observation_contract.cameras` 是冻结训练合同，支持规范组合 `head`、`head left_wrist`、
`head right_wrist`、`head left_wrist right_wrist`。实际模型支持的组合与请求键见[模型专项](model_specific_evaluation.md)。
新增训练输入视角须重新转换、训练并 prepare，不能手改旧 manifest。

prepare 保留专项 `deployment_manifest.json`，另保存 `pipeline_deployment.json`，记录准备命令、
runtime config 身份、原 manifest SHA/deployment ID 和 verify 结果。
既有 π0.5 哈希与训练 provenance 仍由 `verify_pi05_deployment` 等专项校验器检查，不降级为泛化检查。

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
normalizer、binding、预训练视觉模型等专属资产通过 `--` 后的后端参数传入。
专项资产和绑定参数见[模型专项](model_specific_evaluation.md)。

```bash
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is deploy-policy -- verify \
  --deployment-manifest /path/to/deployment/deployment_manifest.json \
  --runtime-config configs/runtimes.local.toml --runtime pi05
```

核验使用现有 checkpoint、normalizer、模型源码与评估源码校验器。
注册表列出的旧 manifest schema（例如 shared π0.5 v1）可以作为输入，但其冻结的
源码与当前 checkout 不匹配时会拒绝执行；应使用对应源码版本，或按正式流程准备
新部署。未注册的 schema 不会被当作兼容格式猜测。历史诊断选项继续由原后端显式处理。

源码或模型资产变化导致旧 deployment 身份不符时，从原 checkpoint 在新目录重新 prepare/verify；
不要手改旧 hash、覆盖部署或绕过校验。运行参数和冻结字段的区别见[模型评估](model_evaluation.md#action-chunk-执行长度)，
后端特有冻结项见[模型专项](model_specific_evaluation.md)。

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

`run` 默认只执行一条且不录像（`--num-trials 1 --video-count 0`）。此例显式覆盖次数，
完整 trial、时限、保护和执行长度参数统一见[模型评估](model_evaluation.md)。
`run --execute-steps` 一次只接受一个值；多值矩阵使用独立 `evaluate-policy`。
额外 evaluator 参数放在 `--` 后，不能覆盖统一入口已拥有的参数。

managed run 验证部署、启动模型服务、等待 metadata 后才执行评估，结束时回收自己启动的进程。
就绪检查通过 WebSocket 握手核对 deployment ID、checkpoint SHA、observation/action 合同及 horizon/dimension；
服务声明 task/model_family 时也一并核对。只监听 TCP 端口不算就绪，`--server-timeout` 控制服务等待上限。
模型请求绕过 HTTP 代理，但保留模型下载所需的代理环境，不全局清空代理设置。

owned prepare、verify、server、evaluation 子进程使用独立进程组，SIGINT/SIGTERM 时统一回收。
sidecar 与私有 stdin/stdout local worker 使用模型环境；仿真 evaluator 保持仿真环境，local worker 继承所选 profile。
只有注册为 embedded 的后端在 rollout 内加载模型。

### 短时服务与录像检查

`--num-trials 1 --video-count 1 --smoke-seconds 10` 用于检查短时闭环。
`--smoke-seconds` 仅属于 deploy run，会覆盖 `--max-sim-seconds`，即使选择 strict 也标为诊断；
独立评估使用 `--max-sim-seconds`，完整停止和有效性解释见[模型评估](model_evaluation.md)。

### 运行记录

每次使用不存在的新根目录：

```text
run_root/
├── run_manifest.json          # deployment ID/路径、runtime、参数、错误与子进程退出码
├── server.log                # 仅 managed WebSocket sidecar
└── evaluation/               # 对应 evaluator 的输出
```

运行记录保存实际 `max_sim_seconds`、`guard_mode`、`diagnostic_only` 和 `formal_metrics_requested`。
local worker 日志由 evaluator 写入评估目录；endpoint 和设备是本次运行选择，不改冻结模型 manifest。
trial 结果与后端目录差异见[评估结果](model_evaluation.md#结果字段与统计口径)。

## 接入外部模型服务

```bash
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is deploy-policy -- run \
  --deployment-manifest /path/to/deployment/deployment_manifest.json \
  --service-mode external --endpoint ws://127.0.0.1:18783 \
  --output-dir /path/to/external_evaluation \
  --num-trials 20 --video-count 20
```

外部模式核对同样的 metadata，退出时从不停止或发送信号给外部服务。
远程地址和 `wss://` TLS 必须使用 external；managed 连接 endpoint 仅允许 loopback 明文 `ws://`。
external 不要求 runtime config，仿真 evaluator 不继承模型 profile 的 CUDA/PYTHONPATH 覆盖。
本地模型资产、源码和专项校验环境仍按 manifest 核验，远程 endpoint 不绕过部署完整性。

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
