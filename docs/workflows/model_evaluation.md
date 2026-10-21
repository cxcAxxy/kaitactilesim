# 统一模型推理与评估

本页维护模型评估的运行参数、停止原因、结果统计与 runner 接入；
部署与 manifest 生命周期见[模型部署](model_deployment.md)，模型特有合同与例外见[模型专项](model_specific_evaluation.md)。

独立 `evaluate-policy` 使用已有 deployment；调用者管理网络服务，本地 worker 由对应 runner 启动。
环境与当前工作目录见[运行环境](environment.md)，原生成功判据见[任务成功条件](../tasks/success_criteria.md)，
有序阶段的阈值、连续时间和缺测处理只在[阶段判定](staged_evaluation.md)维护。

统一入口为 `scripts/evaluate/evaluate.py`：

```bash
cd /cpfs_infra/user/chenxianchi/code/kaitactilesim
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is evaluate-policy -- --help
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is evaluate-policy -- --list-support
```

入口按 `(task, model_family)` 选择 runner，核对 manifest 后分发。`--` 后可传 runner 专属参数，
不能覆盖统一入口拥有的 manifest、输出、任务、模型、seed、执行长度、时限和保护模式。

## 基本命令

```bash
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is evaluate-policy -- \
  --task usb-insert \
  --model-family pi05 \
  --deployment-manifest /path/to/deployment.json \
  --output-dir /path/to/evaluation \
  --cameras head right_wrist \
  --num-trials 20 \
  --video-count 20 \
  --execute-steps 5 \
  --max-sim-seconds 90 \
  --guard-mode observe \
  --reference-dataset /path/to/matching/training_dataset \
  -- --server ws://127.0.0.1:18783
```

以上是观察/诊断评估。需要原保护规则下的正式比较时，改为
`--guard-mode strict`，并检查部署、任务版本和最终 trial 有效性。
`strict` 不会把历史不匹配的 checkpoint 或失败的基础设施自动变成正式结果。
只改变时限、模式或执行长度，不需要重新训练或重新 prepare；只有冻结的源码、
模型资产或合同发生变化时，才需要生成新 deployment。
模型特有的冻结 K 和强制诊断限制见[模型专项](model_specific_evaluation.md)，不能把通用运行时选择套到这些后端。

## 评估次数与录像次数

这两个参数相互独立，并由统一入口负责，不再到各任务 runner 中分别设置：

- `--num-trials N`：运行 N 次评估，默认 `20`；
- `--seed-start S`：使用连续 seed `S ... S+N-1`，默认从 `0` 开始；
- `--video-count V`：前 V 次评估生成公共 review 视频及逐步轨迹/对比图，默认
  与 `N` 相同；默认 20 次全部录像，`0` 表示不录像，最大不能超过 N。

例如评估 20 次但只保存前 6 次视频：

```bash
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is evaluate-policy -- \
  --task usb-insert --model-family pi05 \
  --deployment-manifest /path/to/deployment.json \
  --output-dir /path/to/evaluation \
  --num-trials 20 --seed-start 0 --video-count 6 \
  -- --server ws://127.0.0.1:18783
```

`--max-sim-seconds`、`--guard-mode`、`--execute-steps`、`--record-fps` 等统一参数都
放在 `--` 前；具体 runner 参数放在后面。不能在 `--` 后再次覆盖统一参数，
也不能用底层 `--seeds` 替代统一入口的 `--num-trials/--seed-start`。

## 仿真时限与保护事件

统一模型评估默认 `--max-sim-seconds 90 --guard-mode observe`。时限以**仿真时间**
计量，不包括模型推理等待的真实时间；可以在运行时改成任意有限正数，例如
`--max-sim-seconds 120`。`evaluate-policy`、`deploy-policy run` 和注册的任务 batch/runner
使用相同默认值。`deploy-policy run --smoke-seconds 10` 可覆盖为 10 秒短时诊断；
独立 `evaluate-policy` 没有该参数，短时检查直接设置 `--max-sim-seconds`。

默认 `observe` 模式保留原任务阈值，但力过大、碰撞、掉落、有限状态下的接触穿透、
提前脱牙等只记录事件，不提前停止仿真或视频。不可执行的策略动作（腕跳变、IK
不可达、非有限动作）会被拒绝，保持上一完整有效控制指令并推进仿真，丢弃剩余
chunk 后重新推理。不会执行半提交的“新手臂目标 + 旧手指目标”。

默认模式与原严格模式的区别：

| 情况 | `observe`（默认） | `strict` |
|---|---|---|
| 力、碰撞、掉落、有限接触穿透等任务保护 | 记录事件，继续执行与录像 | 提前停止 |
| 已识别的不可达/非有限模型动作 | 拒绝动作，保持上一指令推进一个控制 tick 后重规划 | 不使用 observe 恢复路径；按具体 runner 终止规则处理 |
| 原任务成功判据满足（未启用 staged） | 提前停止 | 提前停止 |
| 时限到达 | 停止 | 停止 |
| 数值故障、不可恢复控制器或基础设施故障 | 提前停止 | 提前停止 |

正常收尾只封装已有帧，**不冻结补帧**。数值故障后不再读取/渲染坏物理态作为
终帧；已有正常帧仍尝试保存。编码器故障、强制杀进程或尚未录到首帧时，不能保证
生成完整 MP4；是否创建视频还由 `--video-count` 控制，详见[视频说明](policy_evaluation_video.md)。

真实时间 watchdog 仍保留：等待模型时仿真时间不前进，仿真时限不能替代推理和
子进程 timeout。默认不设置额外请求上限时，未发生提前结束条件才会持续到仿真时限。

`observe` 是新的观察/诊断协议，标记 `diagnostic_only=true`、
`formal_metrics_valid=false`，不能与历史严格模式成功率混合。需要原来的任务保护
提前终止行为时，显式使用 `--guard-mode strict`。该选项写在统一入口的 `--` 分隔符
之前；专家采集和未配置观察模式的仿真保持原来的严格保护。
支持的 π0.5 runner 启用 `--staged-evaluation-version staged_v1` 后，原生成功不提前停止，
但故障、strict guard 和显式请求/墙钟上限仍可中断。详见[阶段评估](staged_evaluation.md)。

修改代码后，冻结了 evaluation source 哈希的旧 deployment 会正常拒绝源码漂移。
应重新 prepare 对应 deployment；不要修改旧 manifest 的哈希或绕过完整性校验。

## 停止原因与批次中断

`guard_events` 表示过程中触发过什么；`stop_reason` 表示这条最终为什么退出。
例如白板曾超过 8 N，随后完成擦拭，仍可以是 `task_success=true`、
`stop_reason=success`，同时保留 force guard 事件。不能用“有保护记录”直接判定失败。

| `stop_reason` | 含义与检查方法 |
|---|---|
| `success` | 原任务成功判据满足，提前结束；不表示没有触发过保护 |
| `time_limit` | 尚未完成任务，通常到达仿真时限；结合 `sim_seconds` 和运行参数确认 |
| `guard` | 严格模式的任务/动作保护终止；看事件和 `error`/`policy_failure` |
| `request_limit` | 部分 runner 显式设置 `--max-requests` 后达到推理请求上限 |
| `fatal_error` | 数值、不可恢复控制器、服务、渲染/编码器等错误；查看 `physics_valid`、`error`、`record_error` |
| 缺失或 `null` | 老版本、启动前失败、子进程被强制终止等；不能推断为成功或正常到时限 |

`--max-requests` 默认 `0`（不限）。仅在底层支持此选项时作为连接诊断使用。
USB insert 两个 runner 当前把请求上限结束也映射成 `time_limit`；因此若手工设置了
请求上限，必须核对 `stats.requests`/batch `requests` 和 `sim_seconds`，不能仅凭
该字符串认为已经跑满 90 秒。仿真步按 timestep 对齐，非整步时限可能有一个物理步的误差。

MuJoCo 数值检查包括时间和 `qpos/qvel/qacc/ctrl` 的有限性/数值范围、warning 计数、
以及每个物理步恰好推进一个 timestep。自动 reset 后数组恢复有限值，也不会把轨迹
重新判为有效。柔体网格翻转同样是数值故障，不能按普通接触穿透继续执行。
非有限**模型动作**可以被拒绝并恢复，不等于非有限**物理状态**可以继续。
动作 shape/协议不匹配、服务失联，以及未被恢复逻辑明确识别的异常仍会终止。

推理 response timeout、batch `--trial-wall-limit`、手工关闭 viewer 等也会提前退出。
这些是服务/进程的真实时间限制，不受 90 秒仿真时限替代。强制 timeout 可能来不及
写 trial summary 或完成视频封装，应查看 `seed_NNN.log` 与批量 `termination`/退出码。

普通成功和到时限未完成通常会继续下一个 seed；数值/基础设施错误、录像或产物校验
失败会使 trial 无效，并可能暂停批次。不同 backend 的严格保护退出码/统计处理并不
完全相同，应以实际 batch 行为准。`--num-trials 20` 是计划次数，不保证最终完成 20 条。

## 各任务保护原因与触发条件

下表是当前注册评估链路的默认保护条件，不是任务成功阈值。在 `observe` 下，除
明确注明“数值致命”的项目外，任务保护只记录并继续；`strict` 保留提前终止行为。
阈值及传感器坐标含义以运行代码、参数和事件里的 `threshold/message` 为准。

| 任务 | 原因/默认触发条件 |
|---|---|
| `usb-insert` | socket 总法向负载 > 4 N；轴向阻力 > 2 N；侧壁负载 > 1.5 N；socket 穿透 > **0.3 mm**；plug 参考 pose 的 z < 0.5 m |
| `usb-unplug` | 拔出阻力 > 1.5 N；向内阻力 > 2 N；总法向负载 > 4 N；侧壁负载 > 1.5 N；穿透 > **0.1 mm**；plug pose 的 z < 0.5 m |
| `install-ram` | 轴向阻力 > 5 N；socket 穿透 > 0.2 mm；RAM pose 的 z < 0.5 m |
| `bulb-screw` / `bulb-unscrew` / `bulb-screw-inplace` | 任意指尖法向力 > 35 N；bulb pose 的 z < 0.5 m |
| `bulb-unscrew` 额外 | 螺纹提前从 engaged 变为 disengaged：breakaway 尚未验证，或有效移除圈数不足；记录 `task_failure`，该历史故障也会阻止最终成功判据通过 |
| `whiteboard-wipe` / `whiteboard-wipe-horizontal` / `whiteboard-wipe-horizontal-line` | 板擦-板法向力 > 8 N；手直接撞板负载 > 0.5 N；eraser pose 的 z < 0.5 m |
| `vase-wipe` | sponge-花瓶壁法向力 > 5 N；手-花瓶/桌面合计负载 > 2 N；手-环境或 sponge 外部接触距离 < 0；sponge **参考 pose** 的 z < 0.4 m |
| `sponge-grasp` | 任意指尖法向力 > 5 N 或切向力 > 3 N；手-桌或 sponge 外部接触距离 < 0；sponge **bottom z** < 0.4 m |
| Vase / Sponge 数值致命 | 柔体 tetrahedron 当前有符号体积/rest volume 的最小比值 ≤ 0，记录 `sponge_integrity_guard`，**两种模式都立即停止** |
| `poker-draw` | card 有桌面支撑时穿透 > 0.6 mm；card pose 的 z < 0.5 m |
| Poker EgoSteer 额外 | press 超过 12 s 仍未建立四指各 ≥ 0.125 N、连续 80 ms 的负载；slide 阶段少于 3 指达到 0.125 N 持续 ≥ 0.30 s；已识别的力预算伺服保护 |
| `pick-place` | 当前没有单独的 force/drop/penetration 任务 guard；仍检查动作有效性、IK 与公共数值完整性 |

USB insert 这里的 **0.3 mm** 是评估 runner 阈值，不能用任务配置里的 0.1 mm 代替。
Vase/Sponge 的负接触距离保护与柔体体积翻转不是同一种故障。
主要实现见[共享任务适配层](../../src/kaihand_tactile_env/shared/policy_tasks.py)、
[USB insert runner](../../scripts/workcell/run_usb_egosteer_policy.py)和
[Poker 控制器](../../src/kaihand_tactile_env/tasks/poker_draw/policy_control.py)。

任务空间动作还可能记录腕目标跳变、arm/hand IK 拒绝。EgoSteer/共享任务/USB/Poker
通常使用腕跳变 > 0.15 m、失败 arm IK 位置误差 > 0.03 m 或方向误差 > 0.30 rad、
hand 最大位置误差 > 0.02 m；PickPlace 的默认腕跳变/arm 方向阈值为 0.35 m/0.35 rad。
这些不是所有模型族统一的硬编码合同，应看具体 runner 参数。`--strict-ik` 与 bounded
best-effort 的判断也不同，不能把 `stats.hand_ik_failures` 直接当成保护次数。

Poker 的 4 N 是水平伺服 wrench `Fx` 预算，不是瞬态接触冲量上限。已识别的关节/力矩
越界和 `Fx` 超预算可回退到关节位置伺服；并非所有控制器异常都可恢复，例如当前
雅可比 condition > 1e6 抛出的异常仍会终止。详细物理合同见任务页，不能仅凭本表
把无法继续推进的故障强行归为“只记录”。

## 结果字段与统计口径

普通 batch 先看 `summary.json`，再打开其中对应 seed 的 trial `summary.json`。
共享任务的 batch 行在 `results`，PickPlace/Poker/USB 的行通常在 `trials`。
`deploy-policy run` 则把这套目录放在输出根目录的 `evaluation/` 下。

```text
评估输出目录/
├── protocol.json              # seeds、时限、模式、checkpoint/代码指纹等
├── summary.json               # 已执行批次的结果与汇总；可能尚未完成全部 seed
├── seed_000.log               # 子进程日志
└── seed_000/
    ├── summary.json           # 单条任务状态、终止原因与保护事件
    ├── review.mp4             # 以下复核产物仅在选中录像且成功导出时存在
    ├── review.json
    ├── frames.jsonl
    ├── first_frame.png
    ├── last_frame.png
    ├── evaluation_rollout_trace.npz
    └── evaluation_comparison.json
```

新评估把视频、帧日志、轨迹与参考对比产物直接放在对应的 `seed_<seed>/` 中，
与单条 `summary.json` 同级；不再创建 `review/` 子目录，文件名 `review.mp4` 和
`review.json` 保持不变。可用参考数据对应的三张 `evaluation_*.png` 和
`evaluation_evaluation_plots.json` 也放在该层。
历史结果仍可保留 `seed_<seed>/review/` 布局，批次产物读取与续跑校验兼容旧目录，
不会自动搬动或删除旧评估；这项路径兼容不绕过 deployment/续跑的源码哈希检查。

以上是普通后端目录；触觉后端的批次文件名、seed 布局和有效性字段例外见
[模型专项](model_specific_evaluation.md#tactile-policy)。

| 字段 | 如何理解 |
|---|---|
| `status` / `error` | runner 最终状态及异常；基础设施原因不一定进入 `guard_events` |
| `task_success` | 原任务判据是否完成，不被观察到的保护覆盖；完成后录像故障也可能仍为 true |
| `stop_reason` | 最终退出原因，区别于过程中记录的全部原因 |
| `guard_mode` / `guard_events` | 本次模式，以及聚合的保护记录 |
| `physics_valid` | 物理轨迹/时钟完整性，不等同于产物或批次是否有效 |
| `valid_trial` | batch 行是否通过本 backend 的结果/基础设施校验；observe 的正常 trial 也可为 true |
| `diagnostic_only` / `formal_metrics_valid` | 是否仅作诊断、是否满足正式指标条件；Poker 等部分 batch 行用 `formal_result_valid` 转存后一个字段 |
| `record_error` / `raw_record_error` | 视频或 Raw 导出问题；看单条 summary，不能用已存在 MP4 反推这些校验成功 |
| `sim_seconds` / `wall_seconds` | 仿真时间与实际耗时；数值时间 NaN/Inf 时新版 runner 写 `sim_seconds=null` |

示意：触发过力保护、后来仍完成任务的观察评估可以保存如下字段（不是实际批次数据）：

```json
{
  "status": "success",
  "task_success": true,
  "stop_reason": "success",
  "guard_mode": "observe",
  "physics_valid": true,
  "formal_metrics_valid": false,
  "guard_events": [
    {"kind": "force_guard", "count": 3, "threshold": 8.0,
     "measured_max": 9.5, "recovery": "continue"}
  ]
}
```

单条事件按 `(kind, message, threshold)` 聚合，实际还包含
`first_sim_time_s/last_sim_time_s`、`measured_min/measured_max`，可选
`last_measured/last_details/recovery`。`count` 是检查发现违规的次数，**不是独立
碰撞次数或失败 trial 数**；极值只统计违规样本，`last_details` 只保留最近一次详情。
多个条件可以同一步记录。想数“发生过该保护的 trial 数”，应按 seed 去重，而不是加 count。
`recovery` 是调用点记录的处理提示；严格模式可能保留同样的提示文本后抛出保护异常，
最终是否停止仍以 `stop_reason/status` 为准。

batch 的 `planned` 是计划条数，`attempted` 是已尝试条数，`valid_trials` 是有效
条数；`task_successes` 是原任务成功计数，可能包括随后产物失败的无效条目。
`successes/success_rate` 使用 backend 的可统计成功判定，`success_rate` 的分母是
`valid_trials` 而非无条件 `planned`。observe 的非空成功率仍只是诊断结果。
USB/共享 batch 另有 `formal_success_rate`，不满足正式条件时为 `null`；不是所有
backend 都提供此字段。比较结果前还要检查是否完成所有 seed、有无无效 trial、
源文件一致性及各条的正式有效性，不能只取一个成功率数字。

两个现有跨 backend 差异尤其不能忽略：

- USB 的顶级 `guard_event_counts` 累加每条事件的 `count`；Poker/PickPlace/LingBot
  按聚合事件记录数计数；共享 batch 没有这个顶级字段。跨任务比较应自己按统一
  口径读取各条 `guard_events`，不要直接横向比较顶级 counts。
- USB/共享 batch 的顶级 `formal_metrics_valid` 会结合各条结果；
  Poker/PickPlace/LingBot 的顶级值主要是模式/协议声明，**不保证全批有效**。
  应检查各条 `valid_trial`、`formal_metrics_valid` 或 `formal_result_valid`，以及
  `complete`（存在时）和 `valid_trials == planned`。`strict` 只是必要条件，不是充分条件。

历史 summary 可能没有新字段。不要把缺失当成 false/零，也不要直接与新协议混合。

## Action chunk 执行长度

记 deployment manifest 冻结的预测长度为 H。π0.5 runner 默认令执行 chunk
K=H；可省略参数，或显式写：

```text
--execute-steps horizon
```

也可以固定 K，且必须满足 `1 <= K <= H`。例如 H50K8 表示模型仍预测 50 个
action，但 rollout 只执行前 8 步，随后重新观测、重新推理，上一轮剩余 42 步直接
弃用。这只改变闭环重规划间隔，不修改模型权重、预测 horizon 或冻结 manifest。
其它模型的固定 K、动作起始帧和可修改范围以[模型专项](model_specific_evaluation.md)为准。

`--execute-steps` 属于统一评估入口参数，必须写在 runner 分隔符及服务参数之前，
例如 `--execute-steps 8 -- --server ws://127.0.0.1:18783`。若特意比较不同重规划
间隔，`evaluate-policy` 可一次传多个值，例如 `--execute-steps 8 horizon`；各组
使用相同 seed，输出到不同子目录，根目录写 `evaluation_matrix.json`。不同执行长度
属于不同评估协议，结果不能混为一组。`deploy-policy run --execute-steps` 每次只接受
一个整数或 `horizon`，多值矩阵比较应直接使用 `evaluate-policy`。

`--reference-dataset` 可明确指定与 checkpoint 对应的 LeRobot 数据集，默认对比
第 00 条。若未指定，统一入口仅在 checkpoint 路径包含明确的同任务、同版本
`sim/<task>/<model>/<version>/checkpoints` 结构时推导参考路径：已存在的旧 sibling
`lerobot_v3/<version>` 布局保持优先；若旧布局不存在，则选择同版本目录中带
`meta/info.json` 的训练数据集。
参考字段或版本缺失时，新录像仍保存逐控制步物理轨迹及不可用原因，不会用别的数据集
补画参考实线。

## dry-run 与 smoke

`--dry-run` 只验证 manifest、相机合同和 runner 分发，并打印最终子命令：

```bash
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is evaluate-policy -- \
  --task poker-draw --model-family egosteer \
  --deployment-manifest /path/to/deployment.json \
  --output-dir /path/to/evaluation \
  --dry-run
```

它不连接模型服务、不启动仿真、不测试渲染，也不执行底层 runner 的全部资产/源码
完整性校验；输出目录存在性检查也在 dry-run 返回之后。因此 dry-run 成功不能代替
`deploy-policy verify` 或一条实际 smoke。smoke 使用另一个新输出目录，例如把
`--num-trials/--video-count` 都设为 `1`，并设置 `--max-sim-seconds 10`。


## 冻结部署与相机断言

`--deployment-manifest` 选择已冻结模型，`--cameras` 仅断言合同，不修改 checkpoint。
manifest schema、准备与完整性核验见[模型部署](model_deployment.md#deployment-manifest)；
各模型视角、图像请求键和动作编码见[模型专项](model_specific_evaluation.md)。

三相机 RAM π0.5 的统一调用示例：

```bash
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is evaluate-policy -- \
  --task install-ram --model-family pi05 \
  --deployment-manifest /path/to/ram_pi05/deployment.json \
  --output-dir /path/to/ram_pi05_eval \
  --cameras head left_wrist right_wrist \
  --num-trials 20 --video-count 20 --execute-steps horizon \
  -- --server ws://127.0.0.1:18783
```

## 当前 runner 覆盖

支持组合直接查询注册表：

```bash
.pixi/envs/default/bin/python -B scripts/evaluate/evaluate.py --list-support
```

注册只表示 observation/action、任务判据、批量结果和视频的适配链路存在，不等于已有匹配权重或真实验收。
不支持的组合在仿真启动前报错，不回退到其它任务。普通 runner 覆盖也不等于 staged 覆盖，后者见[阶段判定](staged_evaluation.md#当前支持范围)。
共享任务模型只接收 manifest 声明的 observation，模型输出闭环执行；适配层更新物理任务状态、保护和成功判据，
不调用专家动作控制器或轨迹。专属/历史后端的限制见[模型专项](model_specific_evaluation.md)。

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

当前统一 batch 是串行评估，固定 `workers=1`。protocol 中的 `workers` 字段是运行记录，
不是仅修改数值就会生效的并行开关。要并行需要 batch 调度器同时管理多个独立 trial
子进程、输出目录、失败回收和聚合写入；还必须确认模型服务的并发语义。当前 EgoSteer
和 π0.5 WebSocket 服务在事件循环中同步调用单个模型的 `infer`，多个 rollout 即使并发
连接，GPU 推理仍会串行，并可能改变随机采样请求顺序。因此正式可比结果仍使用串行模式。

评估视频内容和参数见[模型评测视频](policy_evaluation_video.md)。

## 运行检查与常见问题

- 每次评估使用新的 `--output-dir`，统一入口拒绝覆盖已有目录。
- 完整批量前先执行 `--dry-run`，再在独立目录运行单 seed smoke test。
- 模型服务端口、checkpoint、推理配置和 runner 参数放在 manifest 或 `--` 后，不写进通用文档。
- 比较模型时保持任务版本、随机 seed、保护模式、最大时长、action horizon、execute steps、相机合同和成功指标一致。
- 评估视频是复核产物，不代替结构化成功指标和失败原因。

| 现象 | 先检查什么 / 如何处理 |
|---|---|
| 有 force/drop/collision 记录，但视频继续 | 默认 observe 的预期行为；看最终 `stop_reason`，不要把事件直接当终止原因 |
| 视频不足 90 秒 | 查成功、数值/基础设施错误、手工关闭或请求/墙钟上限；MP4 播放时长也不等于严格仿真时长 |
| 没有视频或 MP4 不完整 | 查 `video_count`、是否已有首帧、`record_error`、seed 日志和进程退出码；强制终止不保证产物完整 |
| 批次没有跑完全部 seed | 看 `planned/attempted/valid_trials` 和最后一条日志；不要把未运行 seed 直接当成任务失败 |
| 源码/manifest 哈希漂移 | 核实代码与资产后在新目录重新 prepare；不要手改哈希、删除旧证据或绕过完整性校验 |
| 输出目录已存在 | 为新的评估选择新目录；统一入口没有通用续跑协议，不能直接覆盖原结果 |
| `formal_metrics_valid=false` 但任务成功 | observe/其他诊断协议的正常情况，或正式条件未通过；分别看任务判据、物理和产物有效性 |
| 三张参考图不可用 | 查参考版本、字段和 `evaluation_comparison.json`；参考只是绘图数据，不会代替模型或任务状态 |

## 新 runner 接入

新增模型或任务时：

1. 冻结 observation/action 和相机合同；
2. 实现任务专属 runner 与成功指标；
3. 在 `RUNNERS` 注册 `(task, model_family)`；
4. 生成与 checkpoint 配套的 deployment manifest；
5. 增加 manifest 不匹配、相机不匹配、dry-run 分发和单 seed smoke 测试。

接入 runner 不应修改任务原有动作控制器；模型闭环行为属于评估适配层。


## 分阶段诊断评估 `staged_v1`

已接入阶段观测的 runner 在统一 `--` 后接收 `--staged-evaluation-version staged_v1`。
启用后原生成功不提前结束；故障、strict guard 或显式上限仍可中断。
任务/模型覆盖、逐任务判据、产物和历史兼容性统一见[阶段判定](staged_evaluation.md)。
具体批次从[档案索引](../archive/README.md)查阅。
