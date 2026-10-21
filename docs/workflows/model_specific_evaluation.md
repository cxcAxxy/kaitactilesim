# 模型特有评估合同与参数

本页维护各模型特有的运行环境、checkpoint、输入/动作合同和额外参数；
通用部署生命周期见[模型部署](model_deployment.md)，trial、保护、K 和统计解释见[模型评估](model_evaluation.md)。

| 模型 | 专项入口 | 模型环境与资产 | 输入 / 动作及主要例外 |
|---|---|---|---|
| EgoTouch | [PickPlace 与共享任务](#egotouch-pickplace) | 独立模型 Python、同 run binding、EMA candidate | PickPlace no_tactile、108D pose9、H50K5；共享 worker 仅 head |
| CPM + π0.5 | [触觉策略](#tactile-policy) | CPM runtime、匹配 H30 base、guidance 权重与 tokenizer | 同帧 RGB/状态/taxel；H30、K 冻结，保留动作局部梯度 |
| FTP-1 | [触觉策略与就绪度](#tactile-policy) | FTP-1 runtime、domain normalization、HPT、conversion 绑定 | H32×120，从 a1 执行，K≤31 且冻结；仅右臂/手映射 |
| LingBot-VLA-2.0 | [PickPlace](#lingbot-pickplace) | CUDA 服务、hf_ckpt、训练 YAML、机器人映射 | head/right_wrist RGB、27D 状态与绝对关节目标；tokenizer 模板恢复 |
| π0.5 / EgoSteer | [基础与任务变体合同](#pi05-contract) | OpenPI / EgoSteer 资产、normalizer 与训练来源 | π0.5 controller-target、相机请求键与 USB/inplace/迁移约束 |

所有命令从 `/cpfs_infra/user/chenxianchi/code/kaitactilesim` 执行；依赖、导入路径和仿真/模型 Python
分工见[运行环境](environment.md)。模型是否注册以 `evaluate-policy -- --list-support` 为准，注册不证明已有匹配权重。
具体运行和发布证据统一从[档案索引](../archive/README.md)查阅。

<a id="egotouch-pickplace"></a>

## EgoTouch：PickPlace 与共享任务

当前这条链路支持 PickPlace **no_tactile、EMA、H50、108 维 wrist/fingertip pose9**
的 `external_sim_candidates/epoch_NNNN.pt`。支持不同 epoch（例如 0025、0050、0075），
不再限定 epoch 50。不要用普通训练恢复文件、其他任务或 tactile checkpoint 替代。
`egotouch_binding.json` 必须来自同一个训练 run，脚本会校验其与 checkpoint 的绑定。

以下三段在同一个 Bash 终端依次运行。更换 epoch 时修改 `EGOTOUCH_EPOCH`；更换训练
run 时同时修改 `EGOTOUCH_RUN` 和 `EGOTOUCH_RUN_LABEL`。参考数据也必须对应训练版本。

**1. 选择权重与输出位置。**

```bash
cd /cpfs_infra/user/chenxianchi/code/kaitactilesim
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

EGOTOUCH_RUN=/cpfs_infra/user/chenxianchi/code/egotouch/runs/pickplace_0920_200_tict_pair/no_tactile_bs128
EGOTOUCH_RUN_LABEL=no_tactile_bs128
EGOTOUCH_EPOCH=0075
EGOTOUCH_TAG="${EGOTOUCH_RUN_LABEL}_epoch${EGOTOUCH_EPOCH}"
EGOTOUCH_CHECKPOINT="${EGOTOUCH_RUN}/external_sim_candidates/epoch_${EGOTOUCH_EPOCH}.pt"
EGOTOUCH_DEPLOY="/cpfs_infra/user/chenxianchi/code/kaitactilesim/runs/pickplace_egotouch_${EGOTOUCH_TAG}_deployment"
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

strict 下明确的 `policy_action_rejected` 计作失败并继续下一条；其它停止与结果有效性统一见
[停止原因](model_evaluation.md#停止原因与批次中断)及[统计口径](model_evaluation.md#结果字段与统计口径)。
no_tactile 表示触觉不送入模型，仿真仍可采集和可视化触觉。RGB 与共用录像沿用可见组 0/1/2，
不显示红色碰撞几何；视频布局和产物见[模型评测视频](policy_evaluation_video.md)。
运行中修改评估或模型源码会触发批次版本一致性检查。

### 共享任务 EgoTouch 与相机差异

既有共享任务 worker 为单路 RGB，仅接受 `cameras: ["head"]`。PickPlace 使用专用多视角 worker，
固定槽位为 `head/left_wrist/right_wrist`，实际输入 `head/right_wrist`，通过
`view_mask=[true,false,true]` 保留并屏蔽左腕。不能临时给 head-only 权重增加腕部输入。
共享任务本地环境参数放在统一入口 `--` 后，例如：

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

<a id="tactile-policy"></a>

## CPM / FTP-1：触觉策略

### 模型合同与部署

模型族 `pi05+cpm` 与 `ftp1`。CPM 只注册 `poker-draw`；FTP-1 除扑克外还注册了
下文列出的共享任务诊断接口，但不表示这些任务已有训练数据或权重。仿真沿用各任务的
原生场景、动作适配、成功判据、保护及已支持的记录和视频链路；模型各自在独立 Python worker 中运行，
无须手动 `serve`，也不共用两个仓库的 `openpi`。

| 模型 | 原生预测 | 实际执行 | 解码 |
| --- | --- | --- | --- |
| CPM + π0.5 | H30 | `a[0:K]`，K≤30 | 原训练归一化逆变换，臂增量恢复为绝对关节目标 |
| FTP-1 | H32 | `a[1:K+1]`，K≤31 | 120D 中仅解码右臂 7 + FAAS 映射右手 20；手臂 delta 相对当前实测关节恢复；扑克 `mix` 的手也是 delta，新任务依 checkpoint 使用 `mix` 手 delta 或 `arm_relative_hand_absolute` 手绝对值 |

FTP-1 的第 0 帧是当前时刻一致性目标，不占未来控制周期；其余手臂、头部和 padding
维度不执行。腕位姿解码记录一致性诊断，首版不改用 IK 控制。
每次重规划从同一仿真状态读取双 RGB、27 维关节、位姿和任务合同指定目标几何的原始
三通道 taxel 力（扑克为 `card_core_geom`）；模型推理期间仿真时间暂停。触觉不是视频
热力图，也不是旧 TReX F6。

本机环境见 `configs/runtimes.tactile.toml`。CPM 必须使用与该权重配套的 20k base，
不能替换为当前 H50 π0.5。示例（输出目录必须不存在）：

```bash
cd /cpfs_infra/user/chenxianchi/code/kaitactilesim
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
.pixi/envs/default/bin/python scripts/deploy/deploy.py prepare \
  --task poker-draw --model-family pi05+cpm \
  --checkpoint /nas/chenxianchi/datasets/touchguidence/recovery_state_cpm_ratio_1_1_b1024_g8_20261013_165142/best.pt \
  --output-dir deployments/poker_cpm_h30_k5 \
  --runtime-config configs/runtimes.tactile.toml --runtime cpm -- \
  --base-checkpoint /nas/chenxianchi/datasets/sim/poker-draw/pi05_trex/pi05_pytorch_20k \
  --data-root /nas/chenxianchi/datasets/touchguidence/augmentation/poker_fingertip_recovery_rules_v1_random10/ratio_1_1 \
  --tokenizer-model /root/.cache/openpi/big_vision/paligemma_tokenizer.model \
  --execute-steps 5 --guidance-scale 1 --guidance-steps 3

.pixi/envs/default/bin/python scripts/deploy/deploy.py prepare \
  --task poker-draw --model-family ftp1 \
  --checkpoint /nas/chenxianchi/datasets/sim/poker-draw/ftp1_checkpoints/ftp1/poker_draw_0920_200_h32_bs128_25k_ddp8_20261012_114006/24999 \
  --output-dir deployments/poker_ftp1_h32_k1 \
  --runtime-config configs/runtimes.tactile.toml --runtime ftp1 -- --execute-steps 1

.pixi/envs/default/bin/python scripts/evaluate/evaluate.py \
  --task poker-draw --model-family ftp1 \
  --deployment-manifest deployments/poker_ftp1_h32_k1/deployment_manifest.json \
  --output-dir reports/poker_ftp1_smoke --num-trials 1 --video-count 1 \
  --max-sim-seconds 2 --guard-mode observe -- --save-raw
```

K、权重/配置/归一化/tokenizer、两端源码及运行环境路径均绑定在 deployment 中；修改
后应重新 prepare，禁止修改旧 manifest 的字段绕过验证。CPM scale=0 对照需另外
prepare；相同 episode seed / request index 使用相同初始噪声，不在采样外包
`torch.inference_mode()`，以保留 CPM 的局部动作梯度。

这两类 runner 保持 `diagnostic_only=true`、`formal_metrics_valid=false`；原生任务结果仍单列。
通用有效性和分母解释见[结果统计](model_evaluation.md#结果字段与统计口径)。
产物含每条 `summary.json`、`requests.jsonl`、`model_worker.log`、`first_request.npz`，
可选 `review.mp4` 和 Raw HDF5。请求日志保留实际触觉数值及同帧时钟。

原生 H30/H32 汇总独立导出，不修改历史 H50 页面与 MP4：

```bash
.pixi/envs/default/bin/python scripts/analysis/tactile_evaluation_summary.py \
  reports/poker_ftp1_smoke \
  --output-json reports/poker_tactile_summary.json \
  --output-html reports/poker_tactile_summary.html
```

汇总按模型、manifest、H、动作起始帧、K 和 guidance 分组，分别列出 valid / invalid /
missing、diagnostic / formal；不能把 H30/H32 填充或标记为 H50。


### 执行长度、阶段与结果目录

CPM 的 K 不超过 30；FTP-1 从预测 a1 开始，K 不超过 31。两者运行时默认读取
manifest 冻结的 `execute_steps`，不默认执行完整 H。更换 K 应在新目录重新 prepare，
不能通过 `evaluate-policy --execute-steps` 覆盖旧 deployment。其它评估协议参数见
[Action chunk 执行长度](model_evaluation.md#action-chunk-执行长度)。

扑克 CPM/FTP-1 runner 可透传 `--staged-evaluation-version staged_v1`；已注册阶段任务的
共享 FTP-1 runner 也接入同一阶段观测。阶段覆盖与连续保持条件见[阶段判定](staged_evaluation.md)。
接口接入不表示所有任务都已有匹配权重或真实模型验收；`vase-wipe`、`sponge-grasp` 未注册 staged，
不能给它们传此开关。触觉结果保持 diagnostic 标记。

触觉 batch 使用以下目录；它与普通 batch 的 `summary.json` / `protocol.json` 布局不同：

```text
评估目录/
├── batch_summary.json          # trials、valid_trials、invalid_trials 与诊断成功率
├── seed_0000.log
└── seed_0000/
    ├── summary.json            # 单条原生结果、物理/协议有效性与保护事件
    ├── requests.jsonl
    ├── model_worker.log
    ├── first_request.npz
    └── review.mp4              # 选择录像并成功导出时生成；另有 review.json 等
```

`batch_summary.json.trials[].valid` 是此后端的批次有效性字段；不要只找其它后端的
`valid_trial`。manifest 和 worker 握手保存冻结模型身份；每条 summary 是单条结果依据。
扑克可选 Raw HDF5，共享 FTP-1 接口当前拒绝 `--save-raw`。公共视频与轨迹文件见
[视频输出](policy_evaluation_video.md#视频输出)。

### 跨任务触觉模型就绪度（只读预检）

各任务是否已有合同匹配权重，应以以下只读脚本的现场结果为准；
接口注册、checkpoint 合同匹配和真实闭环验收分别记录。
扑克 CPM 固定绑定扑克 H30 base、任务 instruction、训练数据和归一化，
不能改 manifest 中的 task 名后用于其他任务。

共享任务层现已为以下 11 个任务登记 FTP-1 的任务 instruction、domain 和触觉目标物体：
`bulb-screw`、`bulb-screw-inplace`、`bulb-unscrew`、`install-ram`、`usb-unplug`、
`whiteboard-wipe`、`whiteboard-wipe-horizontal`、`whiteboard-wipe-horizontal-line`、
`vase-wipe`、`sponge-grasp`、`usb-insert`。这是供后续 checkpoint 接入使用的诊断接口，不表示已有
训练 corpus、checkpoint 或真实闭环成绩；尤其新增的 vase/sponge domain 名只是合同约定。
`pick-place` 的 FTP-1 触觉闭环 adapter 仍为 `adapter_pending`。

`usb-insert` adapter 使用原生 `usb-insert` 场景、`initialize_for_insertion` 的 10 mm XY /
5° yaw 有界初始化，以及 `UsbInsertionMonitor` 的机械就座判据；触觉、RGB、27 维实测状态
共享同一仿真时刻，动作仍经通用 FTP-1 H32/D120 codec 解码为 27 维绝对关节目标。训练
数据预检继续要求 USB conversion manifest 的 500 Hz 原生采集、30 Hz/H32 导出、
`KaiHandSimUsbInsert` domain 和 future measured-state action supervision 全部匹配。
instruction 含“release and withdraw”，而原生 success 是对准近底、当前 backstop 支撑或
保留的实测 bottom-out 证据、静止并连续就座，并不要求 release/home；两者必须在报告中
分别保留。未找到合同匹配的 USB 专用 checkpoint 时，readiness 显示
`adapter_ready_awaiting_checkpoint`；找到匹配项仍需独立部署与真实闭环验收。
不能用扑克 checkpoint、扑克 normalization 或仅有 normalization 的目录伪装 USB 模型，
也不能把接口 CPU/native smoke 当作真实模型成绩。

用轻量只读清单检查 13 个任务：

```bash
.pixi/envs/default/bin/python scripts/analysis/tactile_task_readiness.py

# 只看一个任务
.pixi/envs/default/bin/python scripts/analysis/tactile_task_readiness.py \
  --task bulb-screw

# 显式检查一个待接入 FTP-1 checkpoint，不加载或散列权重
.pixi/envs/default/bin/python scripts/analysis/tactile_task_readiness.py \
  --task bulb-screw --checkpoint /absolute/path/to/checkpoint
```

显式预检返回 `checkpoint_contract_matched` 后，仍须用同一 task、checkpoint 和 FTP-1
运行环境重新 prepare，再从新 manifest 启动诊断闭环；以下仅示范已匹配的
`bulb-screw` 权重路径，不能把扑克权重或 normalization 路径代入：

```bash
.pixi/envs/default/bin/python scripts/deploy/deploy.py prepare \
  --task bulb-screw --model-family ftp1 \
  --checkpoint /absolute/path/to/matched/bulb-screw/checkpoint \
  --output-dir deployments/bulb_screw_ftp1_h32_k5 \
  --runtime-config configs/runtimes.tactile.toml --runtime ftp1 -- \
  --execute-steps 5

.pixi/envs/default/bin/python scripts/deploy/deploy.py run \
  --deployment-manifest deployments/bulb_screw_ftp1_h32_k5/deployment_manifest.json \
  --output-dir reports/bulb_screw_ftp1_smoke \
  --runtime-config configs/runtimes.tactile.toml --runtime ftp1 \
  --num-trials 1 --video-count 1 --smoke-seconds 12
```

该脚本只在每个任务的 `ftp1_checkpoints` 浅层目录中找候选，并读取小型 JSON 元数据；
不导入模型运行时、不加载 GPU、不启动仿真，也不散列大型权重。FTP-1 只有在 domain、
H32×120 模型合同、训练动作表示、task dataset/conversion manifest 绑定、domain
normalization、HPT tokenizer 和非空 `model.safetensors` 同时存在时才标为合同匹配。
单独生成或复制 normalization 绝不算“已训练”。合同匹配也只是 preflight，仍需独立的
真实权重加载、同帧触觉检查和任务原生闭环验收，不能据此报告成功率。
共享任务入口保存请求、控制步、首帧缓存、原生指标和可选视频；目前不支持 `--save-raw`，
会明确拒绝该参数，不会生成不完整的 Raw 记录。

共享新接口和 readiness 都不会训练模型、不会修改正在使用的 normalization，也不会
原地升级旧 deployment。任务合同或源码变化后必须重新运行 `deploy.py prepare` 生成新
manifest；旧 manifest 保持原样，不能手改字段绕过任务、checkpoint 或数据绑定。

`bulb-screw-inplace` 的 π0.5 与 FTP-1 使用各自独立训练合同，π0.5 的模型记录不能证明 FTP-1 已有匹配权重。
CPM 仍未注册该任务；专家 Raw 预览与模型结果分别解释，预览命令见[原始数据回放](data_visualization.md#灯泡原位旋入样例)。

<a id="lingbot-pickplace"></a>

## LingBot-VLA-2.0：PickPlace

服务端使用 LingBot CUDA 环境，仿真端使用本仓库 Pixi 环境。模型族为 `lingbot-vla2`，
成功判据沿用[PickPlace 原生稳态规则](../tasks/success_criteria.md#pick-place)。
以下占位路径需替换为实际模型环境、deployment 和新输出目录。

在有 GPU 的节点启动模型服务：

```bash
cd /cpfs_infra/user/chenxianchi/code/kaitactilesim
/path/to/lingbot_vla2/bin/python \
  scripts/workcell/serve_pickplace_lingbot_vla2_policy.py \
  --deployment-manifest /path/to/deployment_manifest.json \
  --host 127.0.0.1 --port 8006
```

另一个已按[环境说明](environment.md)初始化的终端运行一条评估；多条评估使用统一次数参数：

```bash
pixi run --as-is evaluate-policy -- \
  --task pick-place --model-family lingbot-vla2 \
  --deployment-manifest /path/to/deployment_manifest.json \
  --output-dir /path/to/lingbot_eval \
  --num-trials 1 --video-count 1 --execute-steps horizon \
  -- --server 127.0.0.1:8006
```

以下底层命令保留用于直接调试 runner：

```bash
cd /cpfs_infra/user/chenxianchi/code/kaitactilesim
.pixi/envs/default/bin/python \
  scripts/workcell/evaluate_pickplace_lingbot_vla2_batch.py \
  --server 127.0.0.1:8006 \
  --deployment-manifest /path/to/deployment_manifest.json \
  --output-dir /path/to/lingbot_eval_new \
  --seeds 0 --max-sim-seconds 90
```


模型 H50、控制频率 30 Hz；输入仅包含 head/right_wrist RGB 和右侧 27D 关节状态，
服务返回反归一化的右臂 7＋右手 20 维绝对关节目标。此后端 `--video-count 0` 只关闭 MP4，
不关闭图表；其它录像布局、路径及参考图解释见[视频说明](policy_evaluation_video.md)。

这份训练集 Raw HDF5 没有直接保存 7×5 牛顿力网格，参考曲线从原始 MuJoCo 接触求解器的
逐帧 wrench 还原 Fn 和切向合力。网格分配保守地保持相应总量，与网格求和口径一致；
不能用深度传感器 proxy 冒充真实力。

无需 GPU 可先给服务端加 `--validate-only`，只核验 checkpoint、训练 YAML、机器人映射、归一化统计及数据集。批量评估加 `--dry-run` 可核对协议和命令而不启动仿真。如果仿真节点没有 EGL GPU，批量评估命令前设置 `MUJOCO_GL=osmesa` 即可使用软件渲染；这不替代模型服务端所需的 CUDA GPU。

模型服务从已哈希的 `hf_ckpt/chat_template.jinja` 恢复 tokenizer 的提示格式：训练基础模型目录的 tokenizer 本身未设置模板。修改服务脚本后必须重启服务，旧进程不会自动加载新代码。失败的评估目录会保留供排查，重试时请使用新的 `--output-dir`。

需要重建清单时，用 `scripts/workcell/prepare_pickplace_lingbot_vla2_deployment.py --checkpoint <hf_ckpt> --output-dir <新的空目录>`；脚本会哈希约 24 GB 文件，且拒绝覆盖已有目录。

<a id="pi05-contract"></a>

## π0.5 / EgoSteer：基础与任务变体合同

### 共享任务 π0.5

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
PYTHONPATH=/cpfs_infra/user/chenxianchi/code/kaitactilesim/src \
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

### USB π0.5 normalizer 与版本

USB 插入的训练数据名为 `usb_insert`，部署 task 为 `usb-insert`。bound checkpoint 支持
`pi05_kaihand`/H30 与 `pi05_kaihand_h50`/H50；prepare/serve 从 checkpoint 训练 provenance
读取实际 config、horizon 和 `normalizer_asset_id`，训练与服务使用同份 checkpoint 内统计。
USB H50 手指尾部修复采用独立 `normalizer_h50_usb_insert_hand_tail_v1`，生成 `usb_pi05_deployment_v2`；
统一部署、batch 和单次 runner 均核对其身份，不从数据目录借用旧统计。
不覆盖旧 `normalizer_h50`，也不能用新统计 resume 旧 run。旧 H30 `normalizer` / v1 deployment
保留兼容，不得手改旧 manifest 的 horizon 或资产名重标为 H50。
训练修复和命令见[OpenPI USB H50 说明](../../../openpi/examples/kaihand/README.md#usb-插入-h50手指尾部安全归一化)。

### 原位灯泡 π0.5

π0.5 的十三个任务有完整准备与服务后端；其他模型按查询结果使用。
新增 `usb-unplug`、`whiteboard-wipe-horizontal`、`bulb-unscrew`、
`whiteboard-wipe-horizontal-line`、`bulb-screw-inplace` 五个变体必须使用新任务
训练来源绑定的 checkpoint；冻结合同会核对 task、mechanics version 和场景指纹，
旧任务 checkpoint 不能仅通过改任务名变成新任务模型。
未实现的阶段明确报错，已有模型和训练 provenance 的校验仍由专项后端执行。

`bulb-screw-inplace` 从灯座入口开始，不是原 `bulb-screw` 的 checkpoint 别名。
该任务的 π0.5 部署只接受 `shared_task_pi05_deployment_v2`：训练来源必须绑定该任务的新数据、
`recorded-controller-target-v1`、一致的 instruction/mechanics/场景指纹。
OpenPI 的独立 H50 配置为 `pi05_kaihand_bulb_screw_inplace_controller_target_h50_v1`；
旧拿取旋入的 `pi05_kaihand_bulb_controller_target[_h50]_v1` 只用于原 `bulb-screw`，不得互换。
旧 v1 manifest 或旧抓取旋入 checkpoint 不得重标为原位任务。

### USB 拔出协议迁移

旧 `1010_200` USB 拔出模型绑定“拔出、放回、释放”的训练提示词，不能默认为当前
`usb-unplug-held-clear-2s-v2` 的匹配模型。默认 prepare 仍拒绝这种协议不一致。
只有显式传给 shared prepare 后端 `--allow-usb-unplug-protocol-transfer-diagnostic` 时，
才允许已绑定、源哈希重算、mechanics/场景指纹完全相同的旧策略按新两秒保持规则评分。

此入口不改 checkpoint、提示词、动作语义、normalizer 或训练来源；模型仍收到原训练指令。
迁移证据记录在被 deployment ID 认证的
`training_provenance.completion_protocol_transfer` 中，包含训练/评估协议、两套场景身份和明确原因。
prepare、verify、server metadata 与 runner 必须一致，且始终
`diagnostic_only=true`、`formal_metrics_valid=false`，不能用作新任务正式成功率。

`--allow-legacy-unbound-diagnostic` 不能代替该显式迁移入口；未绑定来源、未知协议或场景变化仍拒绝。

### 图像请求键与其它专属后端

EgoSteer 和 π0.5 按 manifest 采集全部声明视角。π0.5 head-only 请求沿用 `image`；
多相机默认键为 `head_image`、`left_wrist_image`、`right_wrist_image`，也可由 manifest 的
`observation_contract.request_image_keys` 冻结实际字段名。
EgoSteer prepare 的专属资产为 `--model-config`、`--normalizer`、`--pretrained-vlm`。

Poker `pi05+trex` 在 rollout 内加载 OpenPI base 与 T-Rex expert，不提供独立 serve，
统一层尚无组合模型 prepare。使用时显式传 `--model-family pi05+trex`，选择其 π0.5 base manifest，
并按专项脚本提供触觉 expert 等资产；不能把普通 π0.5 deployment 当作完整组合合同。

USB OpenWAM 保留独立 runner/batch，不在统一注册表内；旧默认 50 秒且保护立即终止，
不能套用统一 observe 协议。历史 Poker EgoTouch 脚本也不等于已注册对应组合。
