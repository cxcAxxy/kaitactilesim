# 统一模型推理与评估

从 checkpoint 准备部署、自动启动模型服务并接入仿真，使用
[统一部署命令](model_deployment.md)。本文的 `evaluate-policy` 适用于已有 deployment、
由调用者管理网络模型服务的独立评估；EgoTouch、CPM 和 FTP-1 的本地 worker 由 runner 启动。
历史模型的严格合同和诊断例子仍保留，并不代表它们能用于当前正式指标。

本文的默认时限和保护规则适用于 `--list-support` 注册的任务/模型组合。先看
[基本命令](#基本命令)，再按需要查[停止原因](#停止原因与批次中断)、
[任务保护条件](#各任务保护原因与触发条件)和[结果解读](#结果字段与统计口径)。
13 个任务的原生成功、专家流程与数值门槛见[各任务成功判定](../tasks/success_criteria.md)；
10 个已注册任务的有序阶段、全部 24 个共享阈值和历史归档边界见[阶段判定专页](staged_evaluation.md)。

统一入口是 `scripts/evaluate/evaluate.py`，Pixi 快捷命令为：

```bash
cd /cpfs_infra/user/chenxianchi/code/kaitactilesim_unified_v2
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is evaluate-policy -- --help
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is evaluate-policy -- --list-support
```

本文统一使用已验证的
`/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi`。若仓库已有
`.pixi/envs/default/` 环境，也可从仓库根目录直接用
`.pixi/envs/default/bin/python scripts/evaluate/evaluate.py` 替换
`/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is evaluate-policy --`，后面的参数
保持不变。本文所有相对命令都从上述 unified 根目录执行，使用 unified 自己的环境，不复用原两份工程的环境。
本机缓存恢复环境暂缺 pyarrow；`--as-is` 不触发补装，物理/规则检查可运行，数据转换或依赖 pyarrow 的
参考数据读图前须先补齐依赖。不要改用当前激活的 `openwam_alpha` Python 运行评估。

三阶段命令模板见[统一命令速查](pipeline_quickstart.md)。`--list-support` 直接从
runner 注册表列出任务、模型和底层脚本，无需先准备 deployment manifest。

`poker-draw` 新增 `pi05+cpm`（H30）和 `ftp1`（H32，从 a1 执行）。两者首版为独立
原生 horizon 的诊断评估，不混入历史 H50 汇总；模型环境、prepare、视频与报告命令见
[CPM / FTP-1 触觉闭环](policy_evaluation_video.md#cpm--ftp-1-触觉闭环)。FTP-1 另为三种灯泡、
RAM、拔 USB、三种擦白板、擦花瓶和海绵抓取登记了诊断闭环接口；接口登记不代表已有
匹配权重或真实模型成绩。非扑克 CPM、抓放/插 USB 的 FTP-1 尚未接通，完整状态见
[跨任务就绪度](policy_evaluation_video.md#跨任务触觉模型就绪度只读预检)。

入口根据 `(task, model_family)` 选择经过任务适配的 runner，读取 deployment manifest。
`--` 后可传任务 runner 的专属参数，但不能覆盖统一入口已校验的 manifest、输出目录、
任务、模型族、seed、执行长度、时限和保护模式。

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
例外：新 CPM / FTP-1 deployment 将执行长度 K 也冻结；改变 K 需重新 prepare，
不需要重新训练。它们的首版结果即使使用 strict，也仍保持 diagnostic 标记。

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

先看 batch `summary.json`，再打开其中对应 seed 的 trial `summary.json`。
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
PickPlace EgoTouch 是单独的已审计合同：H50 固定 K=5，不沿用 π0.5 的 K=H 默认值。
其他模型族应以对应 runner 的限制为准。

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

### PickPlace EgoTouch：选择 checkpoint、生成部署文件、评估

这里的部署文件是 `deployment_manifest.json`，用于绑定 checkpoint、归一化统计、
模型代码和 Python 环境；不是生成新的训练权重。统一评估入口通过
`--deployment-manifest` 选择 checkpoint，不直接接收 `--checkpoint`。

当前这条链路支持 PickPlace **no_tactile、EMA、H50、108 维 wrist/fingertip pose9**
的 `external_sim_candidates/epoch_NNNN.pt`。支持不同 epoch（例如 0025、0050、0075），
不再限定 epoch 50。不要用普通训练恢复文件、其他任务或 tactile checkpoint 替代。
`egotouch_binding.json` 必须来自同一个训练 run，脚本会校验其与 checkpoint 的绑定。

以下三段在同一个 Bash 终端依次运行。更换 epoch 时修改 `EGOTOUCH_EPOCH`；更换训练
run 时同时修改 `EGOTOUCH_RUN` 和 `EGOTOUCH_RUN_LABEL`。参考数据也必须对应训练版本。

**1. 选择权重与输出位置。**

```bash
cd /cpfs_infra/user/chenxianchi/code/kaitactilesim_unified

EGOTOUCH_RUN=/cpfs_infra/user/chenxianchi/code/egotouch/runs/pickplace_0920_200_tict_pair/no_tactile_bs128
EGOTOUCH_RUN_LABEL=no_tactile_bs128
EGOTOUCH_EPOCH=0075
EGOTOUCH_TAG="${EGOTOUCH_RUN_LABEL}_epoch${EGOTOUCH_EPOCH}"
EGOTOUCH_CHECKPOINT="${EGOTOUCH_RUN}/external_sim_candidates/epoch_${EGOTOUCH_EPOCH}.pt"
EGOTOUCH_DEPLOY="/cpfs_infra/user/chenxianchi/code/kaitactilesim_unified/runs/pickplace_egotouch_${EGOTOUCH_TAG}_deployment"
EGOTOUCH_OUTPUT="/cpfs_infra/user/chenxianchi/evaluations/pick-place/egotouch/${EGOTOUCH_TAG}_n20_h50"
EGOTOUCH_REFERENCE=/nas/chenxianchi/datasets/sim/pick-place/lerobot_v3/0920_200
```

例如评估第 50 epoch，将 `EGOTOUCH_EPOCH=0075` 改为 `EGOTOUCH_EPOCH=0050`，再从
这段开始依次执行。相同 checkpoint 重跑时为 `EGOTOUCH_OUTPUT` 增加新后缀（如
`_v2`），不要覆盖已有目录；若源码未变且校验通过，可复用对应的 deployment。

**2. 生成部署文件。** 不需要提前启动模型服务。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
.pixi/envs/default/bin/python scripts/workcell/prepare_pickplace_egotouch_deployment.py \
  --checkpoint "$EGOTOUCH_CHECKPOINT" \
  --binding "${EGOTOUCH_RUN}/egotouch_binding.json" \
  --model-project /cpfs_infra/user/chenxianchi/code/egotouch \
  --model-python /cpfs_infra/user/chenxianchi/miniconda3/envs/egosteer_touch/bin/python \
  --output-dir "$EGOTOUCH_DEPLOY"
```

成功后生成 `${EGOTOUCH_DEPLOY}/deployment_manifest.json`。生成器读取权重内部的 epoch，
校验文件名、模型配置及 binding，并记录 SHA-256。不要手改 manifest 的 checkpoint 路径
或哈希。目标部署目录必须尚不存在；若已经为同一权重生成部署文件，可以跳过这一步复用。

**3. 运行 20 条评估，全部录像。**

```bash
OPENBLAS_NUM_THREADS=1 \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
LP_NUM_THREADS=4 \
MPLCONFIGDIR=/tmp/egotouch-mpl-cache \
.pixi/envs/default/bin/python scripts/evaluate/evaluate.py \
  --task pick-place \
  --model-family egotouch \
  --deployment-manifest "${EGOTOUCH_DEPLOY}/deployment_manifest.json" \
  --output-dir "$EGOTOUCH_OUTPUT" \
  --num-trials 20 \
  --seed-start 0 \
  --video-count 20 \
  --execute-steps 5 \
  --max-sim-seconds 50 \
  --guard-mode observe \
  --record-fps 10 \
  --reference-dataset "$EGOTOUCH_REFERENCE" \
  --reference-episode-index 0 \
  -- \
  --model-device cuda:0 \
  --render-backend hardware-egl
```

先看一条时，改为 `--num-trials 1 --video-count 1`，并使用独立的输出目录，例如
`--output-dir "${EGOTOUCH_OUTPUT}_preview"`。完整 20 条仍使用上面的输出目录。
命令使用 GPU 推理和 NVIDIA EGL 渲染；模型进程由评估脚本自动启动。

评估协议：seed 0–19，30 Hz。模型每次预测 H50，但 runner 只执行前 5 步
（约 0.167 秒）便重新观测和规划；本例显式选择每条最多 50 秒**仿真时间**，
不是当前统一默认 90 秒。需要默认协议时将其改为 90，比较不同设置必须分目录记录。
手部 IK 保留完整 pose9 目标，但默认只给拇指施加 `0.02 m/rad` 的软姿态约束，
其余四指保持位置 IK。这个设置用于消除拇指 XYZ 逆解的冗余错误分支，同时避免把
四指较噪的完整 SO(3) 预测变成抓取时的过强约束。批量消融可通过
`--hand-ik-orientation-weights TH IN MI RI PI` 显式覆盖。
observe 下动作可达性限制或正确 shape 的非有限预测只记录并 hold/replan，
不会立即结束该条；strict 下明确的 `policy_action_rejected` 计作失败并继续下一条。
任务成功仍提前结束；数值、模型加载、渲染或产物校验异常会使 trial 无效并可能中止
批次。20 条是计划次数，不保证完成全部 20 条或每条跑满 50 秒。
本例为观察评估，不产生原严格协议的正式成功率；正式比较显式选择 `strict` 并核对有效性。
默认录像为 1920×1080、10 FPS，包含头部、右腕、全局画面及双手触觉热力图；
no_tactile 表示触觉不送入模型，仍会采集并可视化仿真触觉。
EgoTouch 模型 RGB 和共用录像沿用原有默认可见组 0/1/2，不显示红色碰撞几何。
这项接入不修改其他模型的专属渲染设置。

结果位置（相对于 `$EGOTOUCH_OUTPUT`）：

- `summary.json`：批次汇总；`protocol.json`：运行配置和代码版本指纹。
- `seed_000/summary.json`：单条成功/失败状态、执行步数及终止原因。
- `seed_000/review.mp4`：复核视频，其他 seed 位于对应目录。
- 同一 `seed_000/` 下的 `evaluation_right_wrist_state.png`、
  `evaluation_right_hand_actuated_dof.png`、`evaluation_right_fingertip_tactile.png`：
  与指定参考 episode 的腕部、关节及触觉曲线对比。

常见错误：目录已存在时使用新标签；binding 不匹配时选回同一训练 run 的 binding；
模型源码哈希变化时核实改动并重新生成部署文件，不手工跳过哈希校验。
评估执行期间不要改动相关评估代码或模型源码，否则批次的版本一致性检查会中止运行。

### 共享任务 π0.5 deployment

Bulb、RAM、Vase、Whiteboard、Sponge 及其已注册变体使用共享严格 deployment 工具。先在 OpenPI
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
PYTHONPATH=/cpfs_infra/user/chenxianchi/code/kaitactilesim_unified/src \
  CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /path/to/openpi/.venv-pi05/bin/python \
  scripts/workcell/serve_shared_task_pi05_policy.py \
  --deployment-manifest /path/to/bulb_step25000_deployment/deployment_manifest.json \
  --port 18783
```

服务会拒绝任务 prompt、checkpoint、normalizer、相机、action 或 OpenPI 配置不一致的
manifest，不能把 USB deployment 改路径后用于共享任务。模型加载后会先用冻结的输入
合同完成一次 JAX warmup，日志出现 `WARMUP_OK` 后才开放 WebSocket 端口；因此首次编译
不占用 smoke 的仿真时间，也不会触发 WebSocket keepalive 超时。

π0.5 输出的是 30 Hz 绝对关节目标端点。runner 默认从仿真器当前 controller target
开始，在每个 2 ms 物理步线性插值到新端点，并跨 action chunk 和重新推理保留上一个
端点；action horizon 从 checkpoint provenance 读取，新训练默认 H50，历史 checkpoint
仍可保留 H30；`--action-interpolation zoh` 只用于消融。插值不能修复训练标签语义错误：Bulb
checkpoint 必须来自 `recorded-controller-target-v1` 数据，历史
`legacy-measured-arm-hybrid-v1` checkpoint 应重新转换数据并训练，不能仅靠切换 runner
参数继续作为正式结果。

`bulb-screw-inplace` 使用独立 task / prompt / 环境身份，当前只注册 π0.5。
它要求新任务匹配的 `shared_task_pi05_deployment_v2`、记录的 controller-target 动作和来源合同；
不能把旧 `bulb-screw` checkpoint 或 manifest 改名后用于新任务。
专家流程已验证不等于已有训练模型；新任务的采集与命令见[任务页](../tasks/bulb_screw_inplace.md)。

### Bulb 0923：20000 step 历史诊断命令

本节保留历史 checkpoint / 数据版本标签，但命令入口统一指向新工程。
历史输出路径只是当时命名示例，可能已存在；复跑必须选新的 deployment 和 evaluation 目录，
不能覆盖旧 H50/H30 归档。下述 H30 示例不纳入当前 H50 汇总或 staged 观察台。

0923/0926 checkpoint 早于当前 action/source provenance 与 v15 mechanics，下面流程只能用于复现历史诊断，不能产生正式成功率。生成 deployment 时必须显式加入
`--allow-legacy-unbound-diagnostic`；正式评估应改用同一 v15 环境重新采集、按
`recorded-controller-target-v1` 转换并训练得到的绑定 checkpoint。

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
cd /cpfs_infra/user/chenxianchi/code/kaitactilesim_unified
/cpfs_infra/user/chenxianchi/code/openpi/.venv-pi05/bin/python \
  scripts/workcell/prepare_shared_task_pi05_deployment.py \
  --task bulb-screw \
  --checkpoint /nas/chenxianchi/datasets/sim/bulb-screw/pi05/0923_200/checkpoints/pi05_kaihand/bulb-screw_bs128_25k/20000 \
  --openpi-root /cpfs_infra/user/chenxianchi/code/openpi \
  --output-dir /cpfs_infra/user/chenxianchi/evaluations/bulb/deploy_0923_step20000 \
  --allow-legacy-unbound-diagnostic
```

生成文件是
`/cpfs_infra/user/chenxianchi/evaluations/bulb/deploy_0923_step20000/deployment_manifest.json`。
它冻结 checkpoint 内容哈希、任务、prompt、normalizer、相机与动作合同；生成目录必须
事先不存在。该 checkpoint 的只读预检报告 `head + right_wrist`、预测 horizon 30。

在**另一个终端**启动与此 manifest 绑定的模型服务，保持进程运行：

```bash
cd /cpfs_infra/user/chenxianchi/code/kaitactilesim_unified
PYTHONPATH=/cpfs_infra/user/chenxianchi/code/kaitactilesim_unified/src \
  CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /cpfs_infra/user/chenxianchi/code/openpi/.venv-pi05/bin/python \
  scripts/workcell/serve_shared_task_pi05_policy.py \
  --deployment-manifest /cpfs_infra/user/chenxianchi/evaluations/bulb/deploy_0923_step20000/deployment_manifest.json \
  --port 18783
```

服务就绪后，在评估终端运行 20 个 seed、20 个复核视频。
这里直接使用仓库的 Pixi Python，不依赖 `PATH` 中的 Pixi 可执行文件：

```bash
cd /cpfs_infra/user/chenxianchi/code/kaitactilesim_unified
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

完整批量运行前可先给评估命令加 `--dry-run` 检查分发，再把 `--num-trials` 和
`--video-count` 都改为 `1`、改用**另一个新输出目录**做单 seed smoke。
`--dry-run` 必须放在分隔符 `-- --server` 之前；它不连接模型服务，也不验证输出目录尚未存在。上例的 `horizon` 为
30 个控制步；manifest 的建议重规划间隔为 8 步，若改成 `--execute-steps 8`
即属于另一套评估协议，结果须分目录比较。参考数据集只是绘图参考，
不是模型权重；这里显式选与 checkpoint 同版本的 π0.5 训练数据集。
服务和评估在不同机器时，`127.0.0.1` 必须换为服务所在机器的可达地址。

Bulb 0920 的 20k/25k 历史诊断可由独立子脚本完成。它会逐 checkpoint 执行 deployment、
单 seed 录像 smoke、模型服务和按模型 horizon 的历史诊断批次（20 个 trial、20 个视频）：

```bash
cd /cpfs_infra/user/chenxianchi/code/kaitactilesim_unified
SHARED_PI05_ALLOW_LEGACY_UNBOUND_DIAGNOSTIC=1 \
  bash scripts/workcell/run_bulb_pi05_0920_evaluation.sh 20000 25000
```

脚本会从 checkpoint root 的 `/checkpoints/` 前缀推导参考数据集，也可用
`BULB_PI05_REFERENCE_DATASET` 显式冻结。fast pi0.5 v2.1 数据集会通过
`meta/kaihand_fast_pi05_conversion.json` 和 Raw collection `summary.json` 回溯原始 HDF5；
smoke 必须成功生成三张参考对比图后才进入完整 20 次历史诊断评估。
历史 shell 封装脚本的显式参数以脚本为准，不等同于统一入口的全部默认值；
需要明确控制当前时限/保护模式时，使用上面的统一命令。

脚本只清理自己启动的模型服务，并通过服务日志判断 WebSocket 是否就绪，不会用裸 TCP
连接制造握手错误。即使脚本失败，退出的也只是 `bash` 子进程，不会关闭当前终端。

### Sponge 0924：20000/24999 step 一键评估

Sponge 的专用入口会依次为两个 checkpoint 生成冻结 deployment manifest、启动模型服务、
执行单条录像 smoke，再分别运行 20 条评估并全部录像：

```bash
cd /cpfs_infra/user/chenxianchi/code/kaitactilesim_unified
bash scripts/workcell/run_sponge_pi05_0924_evaluation.sh 20000 24999
```

脚本默认使用 0924 的 π0.5 数据集与 checkpoint，设置 H20 所需的 NVIDIA EGL vendor
文件，并在启动前运行硬件渲染预检；预检必须报告 `software: false`。可用
`SPONGE_PI05_OUTPUT_DIR`、`SPONGE_PI05_GPU` 和 `SPONGE_PI05_PORT` 修改输出根目录、
GPU 与端口。每个 seed 目录中的 `review.mp4` 是复核视频。

相机合同统一使用规范顺序，当前网络 runner 支持以下四种组合：

- `head`
- `head left_wrist`
- `head right_wrist`
- `head left_wrist right_wrist`

其中 EgoSteer 和 π0.5 会按照 manifest 同步采集并发送所声明的全部视角。
π0.5 的 head-only 请求沿用 `image`；多相机请求默认使用
`head_image`、`left_wrist_image`、`right_wrist_image`，也可以由 manifest 的
`observation_contract.request_image_keys` 冻结服务端实际字段名。既有共享任务 EgoTouch
worker 对应单路 RGB 网络，只接受 `cameras: ["head"]`；PickPlace 使用专用多视角
worker，固定槽位为 `head/left_wrist/right_wrist`，实际输入 `head/right_wrist`，并以
`view_mask=[true,false,true]` 保留但屏蔽左腕位置。相机合同必须和 checkpoint 一致，
不能在评估时给 head-only checkpoint 临时增加腕部输入。

当相机合同含 `right_wrist` 时，公共 review 会自动增加右腕模型输入画面；若第二画面
本身已经选择 `right_wrist`，则不重复显示。该规则适用于 EgoSteer、π0.5 和
PickPlace EgoTouch。

## 当前 runner 覆盖

支持组合以注册表的实时输出为准，不维护会漂移的另一份支持矩阵：

```bash
.pixi/envs/default/bin/python scripts/evaluate/evaluate.py --list-support
```

统一任务名包括 PickPlace、Card、USB insert、Bulb screw、RAM、Vase、Whiteboard、
Sponge，以及 `usb-unplug`、`whiteboard-wipe-horizontal`、`bulb-unscrew`、
`whiteboard-wipe-horizontal-line`、`bulb-screw-inplace`，共 **13 个独立任务**；任务存在不代表每个模型族都已接入。
新增五个变体任务和 Sponge 当前只注册
π0.5，其他组合须以查询结果为准。采集/转换支持也不等于评估支持。

Card 另有 `pi05+trex` 专项 runner，依赖本地 OpenPI、T-Rex expert 与触觉权重；
它与上述常规模型服务的部署流程不同，使用前须检查专项脚本中的路径及 checkpoint。
PickPlace 的 `lingbot-vla2` 也已接入统一分发，但仍使用自己的冻结 deployment
manifest、模型服务和[专项合同](../tasks/lingbot_pickplace_evaluation.md)。USB OpenWAM 的
batch 参数与 deployment 合同不同，目前保留专项命令；**未接入本文的 observe 规则**，
其旧 runner/batch 仍默认 50 秒、力/穿透/掉落保护立即终止。
仓库中的历史 Poker EgoTouch 脚本也不等于统一入口已支持该组合。
Sponge 的 π0.5 闭环 runner 已接入；EgoSteer 和 EgoTouch 尚未接入。

不支持的组合会在启动仿真前报错，不会退回其他任务 runner。
“已接入”表示闭环 observation/action、任务判据、批量统计和视频链路已经实现；实际
运行仍必须提供针对该任务训练且合同匹配的 checkpoint、deployment manifest，以及
EgoSteer/π0.5 模型服务或 EgoTouch 本地模型环境。

注册任务的仿真对象可接入公共 `EvaluationVideo`。已注册的共享任务
模型 runner 共用任务适配层：模型只接收 manifest 声明的 observation，输出动作直接
闭环执行；适配层只负责每个物理步更新任务状态、安全限制与成功判据，不调用已知状态
动作控制器或专家轨迹。

新任务使用同一个统一命令。例如 RAM 的三相机 π0.5 模型：

```bash
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is evaluate-policy -- \
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
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is evaluate-policy -- \
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

## 白板接触强度与专家对比

以下是白板专项诊断及其离线预览工具，不是通用的专家动作相似度评分。
某些预览把两条轨迹按各自的绝对仿真时间展示在一张图中；同一秒不代表同一动作阶段，
不能据此计算轨迹优劣或推断触觉变化的因果影响。当前观察台保留专家独立播放时间线；
阶段进展按[实际物理判据](staged_evaluation.md)解释，不把专家同秒画面当作模型应有动作。

`whiteboard-wipe` 和 `whiteboard-wipe-horizontal` 的 shared π0.5、EgoSteer、EgoTouch
runner 默认记录接触强度，每个 trial 都记录，包括 `--video-count 0` / `--no-record`。
该指标仅用于诊断，不改变成功判据、保护模式或结束条件。新评估直接在 `seed_<seed>/`
输出 `contact_intensity.json`、`contact_intensity.npz`、`contact_intensity.png`，
`summary.json.contact_intensity` 保存诊断状态和产物信息，batch 的逐 seed 结果也保留该字段。
有足够手指数据时另输出 `contact_intensity_fingers.png`；双方有共同接触状态时另输出
`contact_intensity_phases.png`。图中文字使用中文，绘图时自动选择可用的中文字体
（例如 WenQuanYi）；JSON 字段名和产物文件名保留英文，以兼容程序读取。总览和阶段图
标注双方各自的实际采样频率。

总览将接触力与真实清洁进度放在同一时间轴上，同时展示任务结果、停止原因及停止时刻。
这能区分“已经碰到白板”和“接触产生了擦除进度”。直接手碰白板的时间曲线缺失时，图中
明确展示日志中已有的碰撞峰值和保护阈值，并标注曲线缺失，不能把空图理解为没有碰撞。
五指图展示各手指受力变化；阶段图上两行并排比较五指合计、板擦的接触期均值与 P95，
下方展示各接触状态的持续时长，以及板擦接触白板期间五指各自的接触时间占比。
状态时长保留只有一方观测到的阶段，便于看到模型早停或尚未进入的阶段。五指占比以
该指有效已知样本覆盖的板擦接触时长为分母，按各指 `> 0.05 N` 的实际覆盖时长计算；
缺失覆盖比例另列，图中提示覆盖不足，不能把缺失样本当作零接触。只有接触时的均值
相似，不能说明整个阶段接触同样稳定。

它分别统计右手五指指尖及其合计抓握载荷、板擦与白板的法向/切向接触、手与白板的
直接碰撞。抓握载荷只计指尖触觉垫与板擦的接触，不等于整手所有接触的合力；各接触点
法向力幅值分别累加，切向力也先取幅值再累加，避免方向相反的接触相互抵消。
PNG 总览与五指图展示法向载荷，切向统计及轨迹保存在 JSON/NPZ。
阶段按接触事件推断为 `pre_contact`、`grasp_transport`、`board_contact`、`post_board`，
不代表已经验证抓取或擦拭成功。`board_contact` 只表示板擦与白板存在大于 `0.05 N`
的接触，静止压着白板也可能进入这一状态。仿真的实际擦除还取决于局部滑动、切向作用
与墨迹覆盖，需结合真实清洁进度判断。清洁进度始终为零的 trial 必须保留并显示，它是
接触没有产生任务效果的诊断证据。对比按这些阶段分别统计力分布、峰值及接触时长；专家
和模型的执行速度可以不同，不能直接把同一绝对秒数当作相同动作阶段。
无接触时的接触期均值/P95 是 `null`；专家值为零时不计算倍数。
接触的统计门限为 `> 0.05 N`。均值/P95 按样本覆盖时长加权；另外保存接触时长/占比、
峰值及采样时间、力时间积分（N·s），以及模型相对专家的差值/倍数。
全阶段均值包含零接触样本，接触期均值只统计超过接触门限的样本，两者分别展示。
白板法向力 `8 N`、手直接碰白板 `0.5 N` 是当前任务保护阈值；每指没有人为追加安全阈值。
专家接触期 P95 仅作为相对参考水平，另统计模型高于该水平的时长/占比，不能据此判定失败。
新模型轨迹按物理步采样，专家 Raw 通常以较低频率采样；峰值比较必须同时查看采样口径，
不能把专家采样峰值当作专家真实物理峰值上界。

这些统计按用途分开阅读，不合成单个接触得分：

| 用途 | 优先查看 | 解释限制 |
|---|---|---|
| 任务效果 | 清洁进度曲线、最终清洁比例、任务结果 | 有接触或力接近专家，不保证已经擦除墨迹 |
| 接触强度 | 法向/切向载荷的接触期均值、P95、峰值及峰值时刻 | 均值和 P95 会忽略短促冲击，需同时查看峰值和曲线 |
| 安全 | 直接碰撞峰值、任务保护阈值、超阈时长/占比 | 超过专家 P95 只代表高于参考水平，不等于触发保护 |
| 接触持续性 | 接触时长/占比、断触情况、五指各自接触占比 | 不同手指分配可能有效，不能仅因与专家不同便判为失败 |

JSON 保留原有力统计，同时以 `quality.effect`、`quality.safety`、`quality.continuity` 和
`quality.phase_variability` 分别保存效果、安全、持续性与阶段内波动信息，每项区分
`rollout` 和 `reference`。其中“接触但未观察到清洁进展”的时长只根据连续真实进度样本
推断；已擦干净后继续接触也可能没有新增进展，所以该数值本身不是有效擦拭判据。

新评估采集实际清洁进度，随力轨迹一起保存；进度字段缺失时应显示缺失，不得以最终清洁
比例插值或复制成全程曲线。数值故障后的导出只使用此前通过检查的缓存，不追加故障态。

默认从 `/nas/chenxianchi/datasets/sim/task_demo` 的专家示例元数据定位 Raw HDF5。
例如 `<demo-root>/_metadata/whiteboard-wipe/replay.json`；参考必须对应同一任务且记录成功。
可指定接触指标的专家文件；这与现有触觉/关节对比图的 `--reference-dataset` 是独立参数：

```bash
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is evaluate-policy -- \
  --task whiteboard-wipe --model-family pi05 \
  --deployment-manifest /path/to/whiteboard_pi05/deployment.json \
  --output-dir /new/path/whiteboard_contact_eval \
  --cameras head right_wrist --num-trials 1 --video-count 1 \
  --execute-steps horizon \
  -- --server ws://127.0.0.1:18784 \
     --contact-reference-hdf5 /path/to/expert/episode.h5
```

`--contact-demo-root /path/to/task_demo` 可替换默认示例目录。两项设置写入白板 batch 的
`protocol.json` 与 trial 身份检查；显式 `--contact-reference-hdf5` 只接受白板任务。
参数解析和 `--dry-run` 不读取专家文件，实际对比在 trial 收尾时完成。参考缺失或绘图失败
记录在 `contact_intensity` 自身的状态/错误中，不写成 `record_error`，也不改变任务结果。
已冻结旧评估源码的 deployment 需要按新代码重新 prepare。

单条专家示例是参考基准，不是专家分布或安全边界。超出专家参考水平与超出任务保护阈值
分别统计；比较模型时应保持参考文件、任务配置与随机初态一致，并分开查看成功和失败 trial。
诊断图不会生成跨任务综合分数，也不以“力越小”直接判定表现更好。
早停 trial 的累计接触时间、力时间积分和清洁进度覆盖范围可能更短，不能据此判断更高效；
阶段内的均值/P95 也应结合阶段时长阅读。若模型与专家墨迹位置或初始姿态不同，比较只用于
解释该条轨迹，不能直接当作同条件性能排名。

已有评估可生成独立预览，兼容历史 `review/` 布局；输出目录使用新目录以保留原始证据：

```bash
PYTHONPATH=src .pixi/envs/default/bin/python -B scripts/workcell/compare_whiteboard_contact_intensity.py \
  --trial-dir /path/to/existing/seed_001 \
  --output-dir /new/path/seed_001_contact_preview \
  --demo-root /nas/chenxianchi/datasets/sim/task_demo --preview
```

该离线命令可加 `--reference-hdf5 /path/to/expert/episode.h5` 覆盖示例参考。历史日志缺少
瞬时板擦接触力或抓握对象分类时，只展示有证据的字段；累计任务峰值单列，不能当成逐步轨迹。
清洁进度只从历史逐步日志中的真实记录恢复；如果只有终态比例，保留终态读数并标注进度
曲线缺失。历史预览不会改写旧 `summary.json`、原视频或已有评估结果。

### 按专家时间线的指尖空间触觉预览

除总接触力外，已有白板评估可另生成中文“指尖触觉接触对比”。参考固定来自
`task_demo` 对应任务的成功专家原始 HDF5，保留专家物理时间线；模型使用自己实际
经过的仿真时间，早停之后留空，不补零、不拉伸。此命令仅做离线诊断，不修改评估
结果或运行仿真，也不默认加入新评估的 `summary.json`。

```bash
PYTHONPATH=src .pixi/envs/default/bin/python -B scripts/workcell/compare_whiteboard_tactile.py \
  --trial-dir /path/to/existing/seed_001 \
  --demo-root /nas/chenxianchi/datasets/sim/task_demo \
  --snapshot-times-s 20.0 28.3 34.0
```

默认在该 trial 根目录另存五个文件，已有同名文件时拒绝覆盖，可用 `--output-dir`
指定新的目录。`--snapshot-times-s` 是三个专家仿真时间，可省略，由专家接触阶段
自动选取；`--contact-threshold-n` 默认 `0.05 N`，不是新增的安全阈值。

| 文件 | 内容 |
|---|---|
| `tactile_contact_timeline.png` | 五指法向力与净切向力、双方有接触时的局部中心偏差和归一化空间分布差异；标注漏接触、额外接触、数据缺失和早停 |
| `tactile_contact_centers.png` | 指尖固定局部坐标中的受力中心轨迹与分布 |
| `tactile_contact_heatmaps.png` | 三个快照的专家／模型／差值 7×5 法向力网格；共用实际力值色标，白圈标注法向力加权的行列网格中心 |
| `tactile_contact_comparison.json` | 时间、数据来源、坐标核验、有效覆盖和逐指统计，说明使用限制 |
| `tactile_contact_comparison.npz` | 原生网格及派生数据，可独立复核 |

受力中心按法向力加权，以指尖局部毫米坐标表示，避免整只手运动污染指标。
归一化网格先除以每指总法向力，再比较空间分布；因此总力变大与接触形状变化分开
阅读。无接触时中心未定义，不放到网格原点；中心／形状偏差仅在双方有接触时计算，
并同时报告接触状态不一致，不能通过忽略漏接触得到好看的平均偏差。

专家和模型原生采样频率分别标注。时间配对只使用最新且未过期的非未来模型样本，
模型早停后和采样缺口处不生成配对结果；同一秒的差异还包含动作进度不同，不能全部
解释为接触质量差。切向网格保留有符号分量，合力幅值不等于不抵消的切向载荷。

历史 NPZ 没保存局部坐标时，只在冻结部署的网格与坐标生成代码哈希核验通过，且与
专家坐标匹配后才恢复毫米坐标；核验不通过则拒绝生成，不能静默使用当前布局。
当前网格是 solver 接触力经归一化高斯核分配的空间估计，单位为 `N/taxel`，不是
柔性皮肤压力真值。指尖图用于解释抓持，清洁进度与板擦—白板接触仍单独评价。

### 五指整体触觉变化概览

已有评估还可生成“圆＋箭头”的整体变化预览，与逐格阵列图配套阅读：圆的位置为
法向力加权受力中心，圆面积表示当前每指总法向力，圆颜色表示固定时间窗口内的
法向力增减，箭头方向和长度表示切向力变化向量 `ΔFt`。箭头不是当前受力方向，也
不是受力中心的移动方向；它的长度单位为 N，不是 N/s。圈是力值示意，不是真实
接触面积，不能把它当作柔性皮肤压力图。

```bash
PYTHONPATH=src .pixi/envs/default/bin/python -B scripts/workcell/preview_whiteboard_tactile_glyphs.py \
  --trial-dir /path/to/existing/seed_001 \
  --snapshot-times-s 20.0 28.5 34.0 \
  --window-s 0.1
```

另存 `tactile_overall_change.png`、`tactile_overall_change.mp4`、
`tactile_overall_change.json` 和 `tactile_overall_change.npz`，已有同名文件拒绝覆盖。
可用 `--output-dir` 另存；`--no-video` 只生成静态图及数据。视频由已记录触觉数据绘制，
不是再次运行模型或重新录制仿真相机，也不修改原 `summary.json`、视频及阵列报告。

专家在上排、模型在下排，五指固定列，固定使用专家物理时间线。默认变化窗口为
`0.1 s`，两边都按 `t` 和 `t−0.1 s` 配对，而不是各自减上一帧。默认每 `0.1` 仿真秒
显示一帧，播放按相同时间比例进行；跨越缺失数据或大采样缺口的变化不计算，模型
结束后留空。无接触、变化未知和未观测分别标注，未知中心不放到指尖原点。

所有指尖、两条轨迹和全部时刻使用相同圆面积、箭头长度和颜色尺度。若箭头为避免
遮挡而设置固定显示上限，超过上限时必须注明截断并显示实际变化量；真实数据在
JSON/NPZ 中保留，不随图形截断。净切向力可能抵消，因此短箭头不证明各个触觉
单元都没有切向变化，仍需结合阵列图和真实清洁进度判断。

## 分阶段诊断评估 `staged_v1`

完整的逐任务条件、24 个共享阈值、接触角色、连续保持时间与归档兼容性，统一维护在
[阶段判定专页](staged_evaluation.md)，不以本节摘要代替实际判据。

当前 13 个任务中 10 个注册 staged；`pick-place`、`vase-wipe`、`sponge-grasp` 暂不支持。
支持 staged 的 π0.5 runner 通过统一入口分隔符后的
`-- --server ws://127.0.0.1:18783 --staged-evaluation-version staged_v1` 启用。
阶段观测只读，不修改物理、模型输入、原生成功或保护阈值；原生成功不再提前结束，
但仍可能被故障、strict guard 或显式运行上限中断。

| 任务 | Stage1 | Stage2 | Stage3 | Stage4 |
|---|---|---|---|---|
| `bulb-screw` | 抓持拿离支撑 | 持有并靠近灯座 | 螺纹啮合、机械就位稳定 0.20 s | — |
| `bulb-screw-inplace` | 原位抓稳 | 有效旋入、机械就位稳定 0.20 s | 原生承力拧紧证据 | 松手后稳定就位 |
| `bulb-unscrew` | 完整取出稳定 0.10 s | 抓持运输 ≥ 2 cm、受控触桌 ≥ 0.05 s、无机器人支撑稳定松手 ≥ 0.20 s | — | — |
| `install-ram` | 拿离支架后持有靠近槽口 | 原生机械就位稳定 0.20 s | — | — |
| `usb-insert` | 抓持拿离支撑 | 持有并靠近插口 | 原生机械就位稳定 0.20 s | — |
| `usb-unplug` | 完整取出稳定 0.10 s | 完全清离插座、右手仍接触且不触桌共同连续 ≥2 s，不需放回或松手 | — | — |
| `poker-draw` | 桌面有效拖动 ≥ 2 cm、持续 ≥ 0.05 s | 对向抓持 ≥ 0.15 s | 满足位置、卡面朝向、离桌和低速条件，连续持牌 **> 3 s** | — |
| 三种白板擦拭 | 抓持板擦拿离桌面 | 未定义；清洁/贴合/覆盖独立报告 | — | — |

阶段有序累计“曾达到”，不是最后一帧状态；未达成的首次时间为 `null`，缺测不会变成零或成功。
拿起通用门槛为相对初态 ≥ 2 cm、脱离桌面/支架、有对向抓持连续 ≥ 0.15 s；
靠近是工作端到入口 ≤ 5 cm。三种白板的 `complete_success` 始终 `null`，不是 0 或完整擦净。
当前控制端点日志约 30 Hz，但 observer 按物理子步更新；间断会重置未完成连续计时。

已有观察台的 9 任务、360 条历史 H50 与新 runtime 10 任务目录不是同一份覆盖范围。
新任务尚无模型评估轨迹，不能将其专家视频记入模型成功率；
历史 H30 权重可以保留用于复现，但不得混入当前 H50 汇总。
同为 `staged_v1` 也必须核对归档定义、阈值与 observer/recording/native source 身份，
不能用当前规则重写旧记录。新 trial 使用独立输出目录，既有 H50 报告不修改。
