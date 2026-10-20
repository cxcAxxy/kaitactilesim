# 模型评测视频

本页说明模型闭环评估视频。若要查看采集得到的原始 HDF5，请用[原始数据可视化](data_visualization.md)中的离线导出命令。

命令均在 `/cpfs_infra/user/chenxianchi/code/kaitactilesim_unified_v2` 执行，直接脚本使用
本项目 `.pixi/envs/default/bin/python`，Pixi 使用 `run --as-is` 复用已安装环境。
本节点 pyarrow 缺包及转换前依赖要求见[环境说明](pipeline_quickstart.md#环境与工作目录)。
看视频不能替代[各任务成功判据](../tasks/success_criteria.md)；
阶段图的覆盖范围和缺失值解释见[分阶段评估](staged_evaluation.md)。

公共 `EvaluationVideo` 生成默认 1920×1080、10 FPS 的只读评测视频，统一入口已注册的
13 个任务均接入它：`pick-place`、`poker-draw`、`usb-insert`、`bulb-screw`、
`install-ram`、`vase-wipe`、`whiteboard-wipe`、`sponge-grasp`、`usb-unplug`、
`whiteboard-wipe-horizontal`、`bulb-unscrew`、`whiteboard-wipe-horizontal-line`、`bulb-screw-inplace`。
各任务支持的模型族并不相同，以
`evaluate-policy -- --list-support` 为准；部分任务执行入口也复用同一组件。
本文的默认保护和收尾规则指统一注册的模型评估链路，不代表旧 OpenWAM 专用脚本、
专家采集或任务执行器也改为相同协议。视频包含：

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

## CPM / FTP-1 触觉闭环

新增模型族 `pi05+cpm` 与 `ftp1`。CPM 只注册 `poker-draw`；FTP-1 除扑克外还注册了
下文列出的共享任务诊断接口，但不表示这些任务已有训练数据或权重。仿真沿用各任务的
原生控制器、成功判据、保护、Raw 记录和视频；模型各自在独立 Python worker 中运行，
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

初版结果明确标为 `diagnostic_only=true`、`formal_metrics_valid=false`；原生任务成功
仍单独记录，接口/基础设施失败不计入成功率分母。短时 smoke 不是 90 秒正式任务成绩。
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

### 统一工作台：跨任务、模型与 checkpoint 对比

平台首页使用原 KaiTactileSim 工作台的任务标签、模型/checkpoint 筛选及卡片/紧凑列表，
在同一页展示 π0.5、EgoTouch、CPM 和 FTP-1，不再以三个独立实验入口作为首页。
成功率总表同时展示原生成功、完整阶段、各阶段到达率及已归档的白板清洁度/贴合面积指标；
任务/模型/checkpoint/协议/批次筛选联动总表，结果/阶段/搜索只筛轨迹列表，不改变成功率分母。
页面只保留一张“595 条成功率与阶段汇总”，默认按“任务 × 模型 × checkpoint”分别展示
47 个原协议分组。模型与 checkpoint 独立成列，每行的原生成功率、完整阶段成功率、各阶段
到达率和量化指标只使用本行分组的有效样本，不混合其他模型或 checkpoint；协议或批次不同
也保留分行。顶部仅展示归档/有效/隔离样本覆盖量，不把所有策略的成功率混成一个总分。
同表仍可显式切换“按任务合计（跨模型，仅描述性）”查看 10 个任务合计。模型选项全局常驻 CPM、FTP-1、
π0.5、EgoTouch，不因当前任务暂无结果而隐藏；空组合明确显示暂无评估记录，不写成 0%。

统一汇总只读取已发布快照，不重新扫描任务目录或重跑模型：H50 450 条（430 计分、20 隔离）
与 EgoTouch 120 条、本轮触觉 25 条，共 595 条成员、575 条有效。原 360 条和旧 550 条状态表
已包含在这些记录中，不能再次加入。原 360 条汇总并入同一张表，不再另列历史统计表；
原始汇总仍留在归档链接及 catalog 中供审计。可选的跨模型任务合计沿用各记录原判据，
仅为描述性计数，不是同协议公平策略排名，也不是任务等权平均。USB 拔出两套不同完成定义分别
显示，白板均值按已知有效 episode 数量加权；缺失与未验证阶段不补零。
当前 CPM 的 5 条和 FTP-1 的 20 条均只覆盖扑克任务，其他任务暂无这两类已发布结果。

```bash
.pixi/envs/default/bin/python -B scripts/analysis/export_unified_evaluation_workbench.py \
  --h50-snapshot reports/h50_checkpoint_experiment_20261019 \
  --tactile-snapshot reports/poker_tactile_benchmark_20261019_n5 \
  --output reports/kaitactilesim_unified_workbench_20261020_r3 --publish-index

.pixi/envs/default/bin/python -B scripts/analysis/export_unified_evaluation_workbench.py \
  --verify reports/kaitactilesim_unified_workbench_20261020_r3
```

后续先导出各模型的不可变结果快照，再用上述入口生成新的统一工作台版本。不要再使用旧触觉
实验导出器的 `--publish-index` 将首页换回实验选择页。所有旧页面、明细和视频保持原样。

#### 内网直接访问

公司或实验室用户可通过服务器的内网地址访问平台，无需各自设置 IDE 端口转发。
当前节点的直接入口为 `http://192.168.0.54:8877/`，首页指向已发布的 r5。
客户端网络需能路由到该内网地址；若办公网与云 VPC 不连通，需要由网络侧接通 VPN、路由
或配置内网网关。仅启动监听不会自动建立这些网络连接。

```bash
.pixi/envs/default/bin/python -B scripts/analysis/serve_evaluation.py \
  --directory reports --host 192.168.0.54 --port 8877
```

`--host` 默认仍为 `127.0.0.1`，可保留原转发入口；显式指定一个 loopback 或 RFC1918 IPv4
时只监听该地址，不开放其他网卡。服务只提供 GET/HEAD，支持视频 Range 分段播放。
根目录应使用整个 `reports`，以便详情页中的归档链接可访问，不应只托管 r5 子目录。
原始评估目录及 checkpoints 无需对外提供。启动前检查该地址/端口是否已有服务。

该入口依赖当前实例运行，实例停止或重启后需重新启动服务；若内网 IP 改变，也需更新
`--host` 和共享网址。用于长期固定访问时，可在内网常驻服务器配置固定地址或内网域名。
深层的“原 H50 站点（8876）”属于旧历史服务，仍保留本机地址；当前统一页面、指标、
视频和归档详情均可通过上述内网入口查看。

#### 完整轨迹详情与交付验收

统一首页的成员应全部进入完整分析页，不再混用简版原生状态页和触觉实验页。
`export_rich_evaluation_workbench.py` 在新快照中复用历史 rich 数据，并仅对发布 catalog 明确绑定的
新增轨迹读取原生记录；不会扫描其他实验、读取模型权重、启动仿真或重判成功。
每页显示模型、checkpoint、seed、真实 H/K，可返回统一总览或在同任务全部模型之间切换。
有记录时展示视频、任务曲线、五指 Fn/|Ft| 与 taxel 空间分布，以及手指关节和腕部诊断图。
灯泡旋拧量使用角度（°）；原生时钟未核验的视频保持独立播放，不伪造曲线同步。

缺少已归档参考图但具有明确 state29、Fn/Ft 及原生时钟合同的轨迹，可离线生成三类
**仅模型实际记录**诊断图，不补造专家、动作目标或物体状态。EgoTouch 缺少任务阶段记录时
保留未知；CPM 来源校验失效时显示“已记录，但来源校验未通过／暂不计分”，不能把原始阶段
达成回填到成绩。隔离轨迹仍可查看诊断，但不进入任何成绩分母，审计页同样提示当前隔离状态。

```bash
# 先建立新的不可变详情快照；完成浏览器验收后才切换入口。
.pixi/envs/default/bin/python -B scripts/analysis/export_rich_evaluation_workbench.py \
  --baseline reports/kaitactilesim_unified_workbench_20261020_r3 \
  --evaluation-root /cpfs_infra/user/chenxianchi/kaitactilesim_evaluations \
  --output reports/kaitactilesim_unified_workbench_20261020_r4 --workers 4

node scripts/analysis/check_rich_evaluation_workbench_browser.cjs \
  http://127.0.0.1:8877/kaitactilesim_unified_workbench_20261020_r4/index.html \
  reports/qa_kaitactilesim_unified_workbench_20261020_r4

.pixi/envs/default/bin/python -B scripts/analysis/export_rich_evaluation_workbench.py \
  --output reports/kaitactilesim_unified_workbench_20261020_r4 --publish-only
```

验收同时检查全部 595 条的详情/视频/数据身份、原 47 个分组的成员与分母完全不变，以及每个
分组和所有隔离/CPM 例外页面的实际浏览器显示；`--all-dom` 可额外逐条渲染所有详情。
r4 详情整理本身没有启动 `bulb-screw-inplace` 新评估；后续独立批次见下文。

#### 任务目标参考与灯泡角度（r5）

`export_reference_evaluation_workbench.py` 从上一版已发布快照生成新的参考值扩展版。
原 360 条的目标参考原样保留；新增 π0.5 90 条、CPM 5 条及 FTP-1 20 条，按各自归档
summary、manifest 和已审阅源码的哈希绑定恢复目标值。页面同时展示目标虚线、适用的容差带
及参考来源；EgoTouch 120 条未记录物体任务进度，明确标为不适用，不补造曲线或目标。
参考值不是专家轨迹，也不替代完整成功判据；不同协议的阈值不互相覆盖。

100 条灯泡轨迹统一展示“当前角度 / 终态角度 / 目标参考角度”。当前值跟随原生曲线游标，
终态值仅从归档终态读取，不用曲线末采样点替代；本次补齐旧 80 条的终态角度。
角度是净旋转量（净圈数 × 360），反转会抵消，不是累计转动路程。原位拧入的相对初始角度
不能直接套用固定 100° 目标；该任务不包含在这版 595 条基线中。

```bash
# 只做离线参考标注，不读取模型权重、不运行仿真、不重判成绩。
.pixi/envs/default/bin/python -B scripts/analysis/export_reference_evaluation_workbench.py \
  --baseline reports/kaitactilesim_unified_workbench_20261020_r4 \
  --evaluation-root /cpfs_infra/user/chenxianchi/kaitactilesim_evaluations \
  --output reports/kaitactilesim_unified_workbench_20261020_r5 --workers 4

node scripts/analysis/check_reference_evaluation_workbench_browser.cjs \
  http://127.0.0.1:8877/kaitactilesim_unified_workbench_20261020_r5/index.html \
  reports/qa_kaitactilesim_unified_workbench_20261020_r5/references

.pixi/envs/default/bin/python -B scripts/analysis/export_reference_evaluation_workbench.py \
  --output reports/kaitactilesim_unified_workbench_20261020_r5 --publish-only
```

发布前还需运行上一节完整详情浏览器验收。参考扩展验收检查全部 595 条原始采样、视频与
诊断文件未改变，475 条轨迹的目标参考和 100 条灯泡终态角度均有来源；按任务 × 模型 ×
checkpoint 的 47 个分组成绩与分母完全不变。CPM 失效阶段证据仍不计分，20 条隔离记录仍
不进入分母；目标参考可用不代表阶段证据重新获得计分资格。

#### 原位灯泡 π0.5：五个 checkpoint、各五次

专用执行器 `scripts/evaluate/evaluate_bulb_inplace_pi05_batch.py` 固定使用
`bulb-screw-inplace/pi05/1018_200/checkpoints/`
`pi05_kaihand_bulb_screw_inplace_controller_target_h50_v1/`
`bulb-screw-inplace_bs128_25k` 下的 `15000 / 18000 / 21000 / 24000 / 24999`，
每组 seeds `0..4`，共 25 条。使用独立模型环境 `code/openpi/.venv-pi05`，
不能替换为 CPM 的 `yidu/openpi-main`。协议为 H50/K50、30 Hz、物理子步线性插值、
90 秒完整仿真、`observe`、`staged_v1`，五条全部录像；即使阶段成功也完成该段记录。

这是固定物理入牙初态的重复试验，seed 不代表随机物体初态，不能据此声称初态泛化。
四个独立阶段为原位稳定握持 → 旋入到座 → 紧固确认 → 脱手稳定。
终态净旋入角度取归档 `rotation_from_initial_turns × 360`；相对角度的机械参考使用
经来源校验的绝对到座目标减去本条实际初始角度，不直接画固定 100° 相对目标。
角度到线不替代握持、实际到座、紧固及释放条件。

```bash
# 只读查看本轮进度；没有结束的组不作为零成功率填入平台。
less /cpfs_infra/user/chenxianchi/kaitactilesim_evaluations/_jobs/pi05_bulb_screw_inplace_h50_k50_n5_20261020/status.json

# 新批次输出必须不存在；默认 --plan 只做 CPU 合同预检。
.pixi/envs/default/bin/python -B scripts/evaluate/evaluate_bulb_inplace_pi05_batch.py \
  --plan --output-dir /absolute/path/to/fresh_bulb_inplace_job
```

完整批次通过原生与阶段校验后，`export_additive_evaluation_workbench.py` 可在新目录中
增量生成 620 条／52 个分组的工作台：保留 r5 原有 595 条全部分组、计分资格和详情字节，
只新增原位任务的五组。导出不自动发布；独立原始记录、旧文件 SHA 和浏览器验收通过后，
才使用 `--publish-only` 切换首页。任何失败保留旧入口与全部 attempt，不按成功率挑选重跑。

本机接入验收（2026-10-19）已完成：两种模型均通过统一 `deploy run` 的 12 秒 K5
真实权重闭环，各 72 次重规划 / 360 个控制步，部署和物理检查有效，均进入卡牌接触阶段；
12 秒内均未完成原生任务，不能将此结果解释为正式成功率。两种模型的 2 秒检查还验证了
Raw HDF5 与视频收尾。CPM 同噪声 scale=0 对照、FTP-1 真实训练 Zarr codec parity
以及相关测试结果见[验收记录](../../reports/tactile_integration_20261019/acceptance.json)和
[原生 H30/H32 报告](../../reports/tactile_integration_20261019/accepted_summary.html)。

记录中的 `*_ready/deployment_manifest.json` 只保留为当时真实权重诊断链路的证据。
任务/触觉运行源码已经更新，旧 manifest 的源码 inventory 会 fail closed；复跑前必须用
当前源码重新 prepare，模型资产或 K 变化时也同样如此。不要把上述历史 manifest 直接
交给 `deploy run`；本节末尾给出先 prepare、再从新 manifest 启动的命令。

### 五个 checkpoint 的固定小样本评估

本轮按每个 checkpoint **5 次**固定为 seeds `0..4`：CPM `best.pt`（H30）和
FTP-1 `10000`、`15000`、`20000`、`24999`（H32），共 25 条，不包含 FTP-1 5000。
每条最多 90 秒仿真时间，30 Hz 控制、K5、ZOH、`guard_mode=observe`，保存 10 FPS
评估视频、请求日志和 `staged_v1` 状态证据；本轮不保存 Raw HDF5。
模型推理期间仿真暂停，实际耗时不等于 90 秒。

```bash
.pixi/envs/default/bin/python -B scripts/evaluate/evaluate_poker_tactile_experiment.py \
  --job-root /absolute/path/to/fresh/poker_tactile_job \
  --num-trials 5 --max-sim-seconds 90 --concurrency 3 --gpu 0

# 只在 supervisor 已结束后生成新的不可变快照；输出目录必须不存在。
.pixi/envs/default/bin/python -B scripts/analysis/export_tactile_evaluation_experiment.py \
  --job-root /absolute/path/to/finished/poker_tactile_job \
  --h50-snapshot reports/h50_checkpoint_experiment_20261019 \
  --output reports/poker_tactile_benchmark_snapshot --require-complete

.pixi/envs/default/bin/python -B scripts/analysis/export_tactile_evaluation_experiment.py \
  --verify reports/poker_tactile_benchmark_snapshot
```

单模型族结果快照生成后，使用上节统一工作台导出器发布首页，不另建互斥的模型入口。

每个组固定接受 `attempt_001`，不按成功率挑选或删除失败轨迹。无效/缺失条目单列，
不能当作 0% 或进入有效分母。原生任务成功与三个阶段（拖牌→稳定夹持→展示持牌超过
3 秒）分开统计，且仍明确标为 diagnostic，而不是 formal 任务验收。
H30/H32 结果独立于旧 H50 的 450 条实验；旧页面、视频和隔离条目不原地修改。
触觉 v1 的阶段导入显式使用 `tactile_manifest_inventory_v1`：核对 manifest 的完整
文件清单、实际文件 SHA、worker ready 握手、轨迹身份和状态日志，保存录制时的规则
指纹；不重跑当前规则，也不补写旧 H50 格式的 `deployment_integrity`。默认 H50
导入仍保留原来的严格校验，不因兼容 H30/H32 而放宽。

本轮已发布 [25 条 Poker 触觉诊断结果](../../reports/poker_tactile_benchmark_20261019_n5/index.html)
和[验收及限制说明](../../reports/qa_poker_tactile_benchmark_20261019_n5/acceptance.json)。五组原生
记录各 5 条有效，均未成功。FTP-1 的 20 条阶段证据通过严格校验；CPM `best.pt` 在评估结束后
发生 SHA 变化，因此其 5 条阶段成绩不计分，原生记录、原权重身份和视频仍保留。不要把当前
同名 `best.pt` 当作本轮评估的冻结版本，也不要因此放宽源码/权重校验。

### 跨任务触觉模型就绪度（只读预检）

当前只有 `poker-draw` 具有匹配且做过真实权重诊断闭环链路测试的 CPM 和 FTP-1
checkpoint；该 12 秒 smoke 不是正式任务评估或成功率验收。
其余 12 个任务没有非扑克 CPM 的真实模型验收；扑克 CPM 固定绑定扑克 H30 base、任务
instruction、训练数据和归一化，不能改 manifest 中的 task 名后用于其他任务。

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
分别保留。当前没有已登记并验证的 USB 专用 checkpoint，因此 readiness 只应显示
`adapter_ready_awaiting_checkpoint`，不能用扑克 checkpoint、扑克 normalization 或仅有
normalization 的目录伪装 USB 模型，也不能把接口 CPU/native smoke 当作真实模型成绩。

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

`bulb-screw-inplace` 的 π0.5 runner 与 FTP-1 共享诊断接口已经注册，但不代表已有该任务的
训练数据、训练权重或真实模型评估；CPM 未注册到该任务。
已生成的新任务专家预览来自保存的三相机 Raw 离线导出，采用专家自己的播放时间线，
不是模型成绩；操作命令见[原始数据回放](data_visualization.md#灯泡原位旋入样例)。

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

`--num-trials` 决定计划运行的 trial 数；`--video-count` 决定按 seed 顺序前多少个
trial 生成 review 视频及其逐控制步轨迹、对比图，不是“前 V 条成功视频”。录像本身
不改变物理控制或模型输入，但渲染、编码及产物校验失败会使 trial 无效，并可能中止
批次，因此不能保证统计结果完全不受录像故障影响。`success_rate` 的分母是有效
trial 数 `valid_trials`，不能只用计划次数或视频数量推算。

统一入口还接受以下参数，均放在 `--` 分隔符之前：

```text
--max-sim-seconds 90       # 每条的仿真时间预算，有限正数；默认 90 秒
--guard-mode observe      # 默认记录可恢复保护并继续；strict 恢复保护终止
--record-fps 5|10         # 默认 10 FPS
```

单个模型 runner 的常见视频参数如下；任务执行入口使用自己的参数名，见文末示例。
并非所有 batch 都透传下面的分辨率/相机参数，使用前应查看对应入口的 `--help`：

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

预测步数由 deployment 的 `prediction_horizon` 和模型服务的 `action_horizon` 合同
共同约束，评测 runner 不再限定为固定 32 步。
`--execute-steps` 控制每次实际执行多少步，必须满足
`1 <= execute_steps <= action_horizon`。统一入口默认执行完整 horizon；
**PickPlace EgoTouch 除外，它固定为 H50K5**。需要比较两组时可显式传
`--execute-steps 16 horizon`（前提是模型 horizon 大于 16，且该模型支持此设置）。
多值比较分组输出，不合并成功率。评测元数据会同时记录
`prediction_horizon`、`execute_steps` 和 `replan_period_s`。

## 何时结束录像

默认 `--guard-mode observe` 保留任务的原阈值，但将可恢复保护从“终止原因”改为
“观察事件”。完整保护与结果说明见[统一模型推理与评估](model_evaluation.md#仿真时限与保护事件)。

| 触发情况 | 默认 observe 的仿真与录像行为 |
| --- | --- |
| 力/载荷过大、碰撞、掉落、有限状态下的穿透、提前脱牙等任务保护 | 写入 `guard_events`，继续仿真和录像；事件本身不表示此次已结束 |
| 腕目标跳变、IK 不可达、非有限策略动作 | 拒绝本条动作，保持上一完整有效控制指令推进一个控制 tick，丢弃剩余 chunk 后重新推理；继续录像 |
| 达到仿真时间预算，或任务成功 | 结束 rollout，尝试捕获有效终态并封装已有帧；成功不再等到时间预算结束 |
| 数值失效、求解器报警、仿真时钟异常、柔体网格翻转，或服务/合同/渲染/编码等基础设施故障 | 提前结束；数值坏态不再读取新观测或渲染终帧，只尝试封装此前已录制的有效帧 |

`strict` 会在任务保护触发时提前结束。可显式配置的请求次数上限（`max_requests`）
也可能提前停止；部分 runner 记录 `stop_reason=request_limit`，USB insert 当前仍
映射成 `time_limit`，需结合请求数与仿真时间检查，详见
[停止原因与批次中断](model_evaluation.md#停止原因与批次中断)。墙钟 watchdog、
用户关闭 viewer 或中断进程也不是仿真时限，不能保证 90 秒录像一定生成。

收尾不会将最后一帧冻结到 90 秒，也不会删除正常完成的失败录像。只要已选择该 trial
录像、录制初始化成功且编码器正常，失败 trial 也会保存截至终止时的录像。若在首帧
之前失败、被 watchdog 强制终止、磁盘写入失败或编码器异常，视频可能缺失、不完整
或不可播放；不能保证每种故障都有完整 MP4。`--video-count 0` 则不创建公共 review
视频/轨迹/对比图，但仍运行模型和记录 trial 结果。

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

已支持的模型服务/worker 参数通过统一评估入口后的 `--` 传给对应 batch/runner。
每次评估使用与 checkpoint 配套的 deployment manifest 和新输出目录，不复用其他
模型的相机合同或结果目录。统一入口可用 `--dry-run` 检查最终子命令。

单组评估的相对目录结构为（按该任务支持情况还会有其他产物）：

```text
summary.json                   # 批次汇总和各 trial 行
protocol.json                  # 批次协议、身份与源码指纹
seed_000.log                   # 子进程运行日志
seed_000/
  summary.json                 # 单条的结果、保护事件和录制错误
  review.mp4                   # 以下复核产物仅前 video-count 条录制成功时生成
  review.json
  frames.jsonl
  first_frame.png
  last_frame.png
  evaluation_rollout_trace.npz
  evaluation_comparison.json
```

新评估不再创建 `review/` 子目录，视频、帧日志、轨迹和参考图都直接放在单条
`seed_<seed>/` 下，与 `summary.json` 同级；`review.mp4` 和 `review.json` 的文件名
不变。历史输出保持原来的 `seed_<seed>/review/` 布局，不自动迁移或删除；批次读取
和续跑校验同时支持新旧布局，源码/协议一致性检查仍然适用。

一次比较多个 `execute_steps` 时，上述结构分别位于
`execute_steps_<K>/` 或 `execute_steps_horizon_<H>/`，根目录另有
`evaluation_matrix.json`。

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

### 如何判断结束原因与录像是否有效

以 `seed_<seed>/summary.json` 和批次汇总行判断结果，不能只看 MP4 是否存在：

- `task_success`：原任务成功判据是否达到，与录像编码是否成功分开记录。
- `guard_mode` / `guard_events`：观察模式以及触发过的保护。同一原因/阈值聚合
  首次和末次仿真时间、`count`、实测极值及恢复方式；`count` 是超限检查次数，
  不是独立碰撞或独立失败次数。
- `stop_reason`：`success`、`time_limit`、`request_limit`、`guard` 或 `fatal_error`；
  `guard` 是 strict 下的提前终止，不是 observe 中每个事件的终止原因。
- `physics_valid`：物理轨迹/时钟完整性是否仍有效；为 `false` 时不可将视频最后一帧误认为
  故障状态的最终渲染，最后一帧来自此前有效状态。
- `record_error`（卡牌还有 `raw_record_error`）：录制/导出错误。任务即使已成功，
  有此错误或产物校验未通过，也不能当作有效评估结果。
- `valid_trial`、`diagnostic_only`、`formal_metrics_valid`：批次有效性和协议口径。
  默认 observe 标记 `diagnostic_only=true`、`formal_metrics_valid=false`，观察到的
  完成数 `task_successes` 不等于可报告的正式成功率，不能与历史 strict 成功率混合。
  部分 batch 行用 `formal_result_valid` 转存单条正式有效性；不要仅凭 batch 顶级
  `formal_metrics_valid=true` 判断所有 trial 有效，详见[统计口径](model_evaluation.md#结果字段与统计口径)。

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
本文默认 observe 保护协议；其对比图仍在同一目录生成以下同口径文件：

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
新任务的物理成功不由亮灯、回 home 或拧动次数推断；具体门槛见[专项判定](../tasks/bulb_screw_inplace.md#判定与记录)。

## 白板接触强度产物

白板模型评估另外在 trial 根目录生成 `contact_intensity.json`、`contact_intensity.npz`
和 `contact_intensity.png`，与 `review.mp4` 同级。它们独立于视频录制开关：没有录像的
白板 trial 也会采集接触数据。参考参数、统计口径和运行示例见
[白板接触强度与专家对比](model_evaluation.md#白板接触强度与专家对比)。
多指数据可用时另生成 `contact_intensity_fingers.png`；双方有共同接触状态时另生成
`contact_intensity_phases.png`，上两行比较五指合计、板擦的接触期均值与 P95，下方展示
各接触状态持续时长及板擦接触期间五指接触占比。状态时长保留只有一方观测到的阶段；
五指占比以该指有效已知样本覆盖的板擦接触时长为分母，缺失覆盖比例另列并提示覆盖
不足，不能将缺失样本当作零接触。
图中标题、图例和提示使用中文；自动检测可用中文字体（例如 WenQuanYi），JSON 键与文件名
继续使用英文。

总览在接触力下方展示真实清洁进度，并标注任务结果、停止原因和停止时刻。直接碰撞曲线
缺失但存在累计碰撞峰值时，图中醒目标明峰值、保护阈值和曲线缺失。清洁进度为零的失败
trial 仍参与展示，便于发现“接触到了但没有擦掉”的情况。

新轨迹从已完成且通过数值有效性检查的物理步采样，故障收尾只导出此前缓存，
不追加故障态样本。图中专家和模型按接触事件推断的阶段分别展示；这些阶段不是任务成功
标签，目的是避免把两者不同的执行速度解释成接触异常。新接触指标中的切向强度是
接触点切向力幅值之和；现有
`evaluation_right_fingertip_tactile.png` 中的 `|Ft|` 仍是向量相加后的模长，两个口径不可混用。
`board_contact` 只表示大于 `0.05 N` 的板擦接触状态，不是有效擦拭标签；有效擦除还取决于
局部滑动、切向作用及墨迹覆盖。解读时分别查看效果、接触强度、安全和持续性，不合成总分。

旧 trial 的视频帧日志只有低频采样及部分累计任务峰值，无法还原完整的物理步接触过程。
历史预览会明确标记采样频率、缺失通道和可比较范围；缺失值不补成零，也不据累计峰值
推算接触期均值、P95 或力时间积分。历史清洁进度只从真实逐步记录恢复，不能把最终比例
造为全程进度曲线。单条专家不是合理范围；不同墨迹/初态、不同采样频率以及模型早停都会
限制比较口径，尤其不能把更短的累计用力解释成更高效。单独生成的历史预览不会改写旧
trial 的结果、`summary.json` 或视频。
