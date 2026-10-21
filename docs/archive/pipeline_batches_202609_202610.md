# 数据转换训练与模型评估批次记录

本页只保存绑定具体数据、权重、seeds 和输出目录的批次命令与验收记录。
通用采集和转换操作见工作流，字段与动作语义见[数据格式合同](../reference/dataset_formats.md)，
模型专属部署要求见[模型专项评估](../workflows/model_specific_evaluation.md)。
工作台页面的导入、发布及回退历史单独保存在[发布记录](evaluation_workbench_release_history_20261015_20261021.md)。

日期、协议、场景指纹、来源路径与成绩保持记录时的身份。复跑前先按
[运行环境](../workflows/environment.md)核对源码，选择新输出目录；旧命令不表示授权覆盖已有批次。

| 批次范围 | 内容 |
|---|---|
| [9 月 checkpoint](#historical-checkpoints) | Bulb / Sponge 历史模型诊断 |
| [10 月任务流水线示例](#task-pipeline-examples) | 固定任务和数据目录的采集、转换示例 |
| [10 月 18 日 π0.5](#pi05-stage-batches) | 原位灯泡训练配置及九任务阶段评估 |
| [10 月 19 日 FTP-1](#ftp1-nine-task) | 九任务转换与训练资产 |
| [10 月 19–21 日诊断](#tactile-evaluation-runs) | CPM、FTP-1 与原位灯泡模型运行 |

<a id="historical-checkpoints"></a>

## 2026 年 9 月 checkpoint 诊断

本节从模型评估主文档归档 0920/0923/0924 专项示例，保留 checkpoint、数据版本与历史输出身份。
这些是历史诊断，不是推荐的新训练/正式评估协议。复跑前检查资产是否仍可访问，
使用新 deployment 和 evaluation 目录，并按[当前环境说明](../workflows/environment.md)从当前仓库执行。
命令中的当前执行目录已调整为 `kaitactilesim`；历史数据、来源 SHA 和原有评估不改写。
通用入口见[模型部署](../workflows/model_deployment.md)和[模型评估](../workflows/model_evaluation.md)。

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
cd /cpfs_infra/user/chenxianchi/code/kaitactilesim
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
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
cd /cpfs_infra/user/chenxianchi/code/kaitactilesim
PYTHONPATH=/cpfs_infra/user/chenxianchi/code/kaitactilesim/src \
  CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /cpfs_infra/user/chenxianchi/code/openpi/.venv-pi05/bin/python \
  scripts/workcell/serve_shared_task_pi05_policy.py \
  --deployment-manifest /cpfs_infra/user/chenxianchi/evaluations/bulb/deploy_0923_step20000/deployment_manifest.json \
  --port 18783
```

服务就绪后，在评估终端运行 20 个 seed、20 个复核视频。
这里直接使用仓库的 Pixi Python，不依赖 `PATH` 中的 Pixi 可执行文件：

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
cd /cpfs_infra/user/chenxianchi/code/kaitactilesim
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
cd /cpfs_infra/user/chenxianchi/code/kaitactilesim
bash scripts/workcell/run_sponge_pi05_0924_evaluation.sh 20000 24999
```

脚本默认使用 0924 的 π0.5 数据集与 checkpoint，设置 H20 所需的 NVIDIA EGL vendor
文件，并在启动前运行硬件渲染预检；预检必须报告 `software: false`。可用
`SPONGE_PI05_OUTPUT_DIR`、`SPONGE_PI05_GPU` 和 `SPONGE_PI05_PORT` 修改输出根目录、
GPU 与端口。每个 seed 目录中的 `review.mp4` 是复核视频。

<a id="pi05-stage-batches"></a>

## 2026-10-18 原位灯泡训练与九任务阶段评估

记录日期范围为 2026-10-12 至 2026-10-18。下列命令绑定具体数据、训练配置与输出目录，
保留用于理解和复现当时的工作；已有输出不可当作新批次覆盖执行。
当前通用入口见[统一命令速查](../workflows/pipeline_quickstart.md)，专项采集见
[任务流水线示例](#task-pipeline-examples)，运行前核对[环境与源码来源](../workflows/environment.md)。

### 原位旋入的 H50 八卡训练（2026-10-18 接入）

本任务使用独立配置
`pi05_kaihand_bulb_screw_inplace_controller_target_h50_v1`，不能复用原拿取旋入的配置或统计。
它绑定 `recorded-controller-target-v1`、新任务 instruction、mechanics version 和场景指纹；
转换默认值及训练检查器已识别新任务。`1018_200` 是本次实际转换完成的批次。

先做无 GPU、无输出的合同元数据预检（不代表完成全部训练就绪检查）：

```bash
cd /cpfs_infra/user/chenxianchi/code/openpi
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' \
  bash scripts/train_bulb_screw_inplace_pi05_1018_h50_8gpu.sh --check-only
```

在八张 GPU 均可用时启动全新训练。此命令会实际计算并核验本任务的
`normalizer_h50`，完整检查数据，然后从本地 `pi05_base` 初始化训练；不沿用旧任务的统计。
`WANDB_MODE=offline` 避免登录/网络依赖，日志仍保存在本批次的 `.train_runtime`。

```bash
cd /cpfs_infra/user/chenxianchi/code/openpi
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 WANDB_MODE=offline \
  bash scripts/train_bulb_screw_inplace_pi05_1018_h50_8gpu.sh
```

固定为 H50、全局 batch 128、8 卡 JAX FSDP（每卡 16）、25,000 steps，
保存 `5000/10000/15000/20000/24999`。默认输出为：

```text
/nas/chenxianchi/datasets/sim/bulb-screw-inplace/pi05/1018_200/checkpoints/pi05_kaihand_bulb_screw_inplace_controller_target_h50_v1/bulb_screw_inplace_1018_h50_bs128_25k_fsdp8_v1/
```

独立启动器拒绝已有实验目录，不接受 `--overwrite` 或 `--resume`，用原子锁阻止同名并发；
如需重复实验，只能在脚本后传一个全新实验名。训练期间不要再在相同八张卡上启动评估。
以上记录 2026-10-18 引入启动器时的命令与配置，不据此推断任意后续训练作业状态。

### 九任务 10000/15000 checkpoints 的五次阶段诊断

新入口 `scripts/evaluate/evaluate_pi05_stage_batch.py` 固定选择已核验的 9 个 H50 任务与数据批次，
不使用历史专用的 `run_pi05_staged_groups.py`。每任务 `10000/15000` 各运行 seed `0..4`，
共 18 组、90 条；K50、90 秒仿真、`observe`、`staged_v1`，各组串行且保存 5 份头部/右腕回放。

旧 USB 拔出模型来自“放回释放”训练，本次评分采用“拔出并手持清离 2 秒”。
必须显式开启以下跨协议诊断选项；保留原模型、提示词和训练来源，归档训练/评估两套协议，
永远不能将此结果声明为正式有效。默认无该选项仍拒绝不匹配的旧 USB 来源。

```bash
cd /cpfs_infra/user/chenxianchi/code/kaitactilesim
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
PI05_STAGE_BATCH=/cpfs_infra/user/chenxianchi/kaitactilesim_evaluations/_jobs/pi05_h50_k50_steps10000_15000_n5_20261018

# 只读 CPU 合同检查，不加载模型、不创建输出、不占 GPU。
.pixi/envs/default/bin/python -B scripts/evaluate/evaluate_pi05_stage_batch.py \
  --plan --output-dir "$PI05_STAGE_BATCH" \
  --gpu 0 --port 18980 --steps 10000 15000 --num-trials 5 \
  --allow-usb-unplug-protocol-transfer-diagnostic

# 实际评估：将 0 改为一张可用的 GPU；整个批次仅串行使用这一张卡。
.pixi/envs/default/bin/python -B scripts/evaluate/evaluate_pi05_stage_batch.py \
  --run --output-dir "$PI05_STAGE_BATCH" \
  --gpu 0 --port 18980 --steps 10000 15000 --num-trials 5 \
  --allow-usb-unplug-protocol-transfer-diagnostic

# 只读输出每任务、每 checkpoint 的阶段计数和比例。
.pixi/envs/default/bin/python -B scripts/evaluate/evaluate_pi05_stage_batch.py \
  --summarize --output-dir "$PI05_STAGE_BATCH" --format csv
```

`--run` 拒绝覆盖已有目录；中断后先检查 `status.json`，若确认恢复同一批次，
在原 `--run` 命令中增加 `--resume`。失败 attempt 保留，新建下一 attempt；不会覆盖旧轨迹。
同一批次续跑必须保持原输出目录、GPU、端口及诊断选项，不要改端口来绕过检查。
例如上述批次的完整续跑命令为：

```bash
cd /cpfs_infra/user/chenxianchi/code/kaitactilesim
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
PI05_STAGE_BATCH=/cpfs_infra/user/chenxianchi/kaitactilesim_evaluations/_jobs/pi05_h50_k50_steps10000_15000_n5_20261018

.pixi/envs/default/bin/python -B scripts/evaluate/evaluate_pi05_stage_batch.py \
  --run --resume --output-dir "$PI05_STAGE_BATCH" \
  --gpu 0 --port 18980 --steps 10000 15000 --num-trials 5 \
  --allow-usb-unplug-protocol-transfer-diagnostic
```

批次与部署层的端口预检均使用与 Unix 模型服务一致的 `SO_REUSEADDR`，
允许上一组连接关闭后的 `TIME_WAIT`，但仍拒绝存活的监听服务；不会使用 `SO_REUSEPORT`、
探测连接或终止占用者。权限/地址等其他 socket 错误会保留真实原因，不再一律称为端口占用。
各组的部署和评估日志位于 `groups/<task>/step<step>/attempt_NNN/`，
全批通过验收后自动写 `stage_summary.json`。`--summarize` 仅向 stdout 输出，不改既有结果。

阶段率的分母是有效 staged 样本；必须每组 5/5 有效、全程与录像完整，才能宣布 90 条全部完成。
缺失/无效不会当作失败零值；原生任务成功与阶段累计到达分别统计。
三类白板只有“拿起板擦”阶段，staged 完整成功率为 `null`，不能解释为擦净成功或零成功率。
这套命令只生成新的诊断批次，不修改历史 H50 结果，也不会自动替换网站。

<a id="ftp1-nine-task"></a>

## 2026-10-19 九任务 FTP-1 数据与资产

本节保存原统一转换页中的固定九任务补转配方。任务范围、来源批次与输出版本仅属于
这次补转；通用字段和训练语义见[FTP-1 数据合同](../reference/dataset_formats.md#ftp1)。
原批次文档的工作目录为 `/cpfs_infra/user/chenxianchi/code/kaitactilesim_unified`；
下面的复现命令已明确使用当前 `kaitactilesim` 仓库。执行前先检查
[环境与源码来源](../workflows/environment.md)，并核对目标数据与发布合同。

FTP-1 使用独立入口 `scripts/convert/convert_kaihand_ftp1_batch.py`，不是给统一入口
`convert.py` 传 `--format ftp1`。本次只补下面九个任务，不处理已完成的 `poker-draw`，
也不转换 `sponge-grasp`、`pick-place`、`vase-wipe`。所有路径相对于
`/nas/chenxianchi/datasets/sim/`：

| 任务 | 已发布 RAW 来源 | FTP-1 输出 |
|---|---|---|
| `bulb-screw` | `bulb-screw/raw/1012_v15_200_r2` | `bulb-screw/ftp1/1012_200` |
| `bulb-screw-inplace` | `bulb-screw-inplace/shared_raw/1018_200` | `bulb-screw-inplace/ftp1/1018_200` |
| `bulb-unscrew` | `bulb-unscrew/raw/1010_200` | `bulb-unscrew/ftp1/1010_200` |
| `install-ram` | `install-ram/raw/0920_200` | `install-ram/ftp1/0920_200` |
| `usb-insert` | `usb_insert/raw/0920_200` | `usb_insert/ftp1/0920_200` |
| `usb-unplug` | `usb-unplug/raw/1010_200` | `usb-unplug/ftp1/1010_200` |
| `whiteboard-wipe` | `whiteboard-wipe/raw/0929_200` | `whiteboard-wipe/ftp1/0929_200` |
| `whiteboard-wipe-horizontal` | `whiteboard-wipe-horizontal/raw/1010_200` | `whiteboard-wipe-horizontal/ftp1/1010_200` |
| `whiteboard-wipe-horizontal-line` | `whiteboard-wipe-horizontal-line/raw/1012_200` | `whiteboard-wipe-horizontal-line/ftp1/1012_200` |

USB 插入采用原始批次 `0920_200`，不借用 π0.5 后处理版本 `0921_200` 的标签。
USB 拔出的 `1010_200` 是旧“拔出、放回、释放”示范，转换后仍保留该语义；
不能仅修改提示词就宣称变成新版“拔出并手持清离两秒”的训练数据。

### 批次命令

先做全部九任务、每任务 200 条成功示范的只读元数据检查。检查相机时钟、状态索引、
触觉形状/单位、场景身份及已有输出的恢复合同；不读取 RGB 像素或完整触觉载荷，
不重新计算源文件 SHA，不创建转换结果：

```bash
cd /cpfs_infra/user/chenxianchi/code/kaitactilesim

.pixi/envs/default/bin/python -B scripts/convert/convert_kaihand_ftp1_batch.py \
  --plan \
  --resume
```

确认全部通过后，显式开始转换。下面仅使用 CPU，依次处理九个任务；每任务最多两个
shard worker，每 20 条示范写一个 Zarr shard。源 HDF5 摘要会重新计算，增加一遍源数据读取：

```bash
.pixi/envs/default/bin/python -B scripts/convert/convert_kaihand_ftp1_batch.py \
  --run \
  --resume \
  --workers 2 \
  --episodes-per-shard 20 \
  --verify-source-hash
```

`--resume` 同时适用于初次运行与中断恢复：只复用来源、任务、指令、环境、分片及合同
相符的已发布结果，不覆盖不匹配的 Zarr、数据配置或训练资产。全部九组的预检均通过后才开始第一个转换。
默认运行不会转换数据，必须显式传 `--run`。

全部转换完成后，另行准备每任务的归一化和触觉输入配置。这不是训练，不加载完整模型：

```bash
.pixi/envs/default/bin/python -B scripts/convert/convert_kaihand_ftp1_batch.py \
  --prepare-training-assets \
  --norm-workers 4
```

资产写入 `<task>/ftp1_assets/ftp1/<repo_id>/`。底层 FTP-1 归一化脚本会删除同名输出，
因此这里强制在新临时目录计算，校验后原子发布；不会把该脚本直接指向已有资产。
已有资产只有在发布标记、数据清单摘要、配置及全部资产摘要一致时才能复用，否则拒绝。
资产准备前要求全部九组输出已经完整发布，残缺 Zarr 不会被当成完整训练集。
可用 `--tasks bulb-screw-inplace` 等参数限制为九任务集合中的子集。

### 格式依据与语义边界

这些命令依据本机 FTP-1 的实际 reader 与训练参数，不是照搬 π0.5 命令：

- [FTP-1 数据准备说明](../../../ftp1-policy/README_zh.md#data-preparation-zh)：要求兼容 key 的 Zarr。
- [FTP-1 数据加载器](../../../ftp1-policy/src/openpi/dataset_zarr.py)：读取腕部位姿、
  手臂/手部关节、FAAS 映射和触觉阵列，生成统一状态与动作槽。
- [触觉编码器](../../../ftp1-policy/src/openpi/models_pytorch/ftp1_blocks.py)：`matrix`
  路径接收数值阵列，而非伪彩色触觉图片。
- [归一化示例](../../../ftp1-policy/scripts_exp_zarr/univtac/compute_norm_stats_univtac_example.sh)
  与 [KaiHand 训练入口](../../../ftp1-policy/scripts/train_kaihand_ftp1.sh)：归一化和训练必须匹配。
- 本工程的 `scripts/workcell/convert_kaihand_raw_to_ftp1_zarr.py` 负责 RAW 到上述合同的适配，
  锁定每任务的来源、相机、关节顺序、传感器方向、物理版本和成功记录。

转换保留右手五指 `[T,5,7,5,3]` 阵列，通道为法向载荷和沿阵列列/行方向的有符号切向分量，
单位 `N/taxel`。它们是 MuJoCo 求解器接触力的空间分配估计，不是 Genesis 深度代理，
也不是独立柔性皮肤传感器的实测真值。头部/右腕图像使用相同名义 30 Hz 时钟，
状态和触觉按已记录的最近非未来 `state_index` 选取。RAW 源频率不强制混同：
USB 插入状态/触觉为 500 Hz，另外八任务为 100 Hz；三类白板物理为 1000 Hz，其余为 500 Hz。

训练约定沿用现有 FTP-1 配方：H32、动作步长 1、关闭历史、相对 pose、绝对 proprioception joints、
`action_joint_rep=mix`、z-score，并开启触觉。H32 表示当前帧加 31 个后续 30 Hz 轨迹状态，
不是 32 Hz，也不是 32 帧触觉历史。`mix` 动作监督来自实测轨迹，不等于 π0.5 新灯泡任务的
`recorded-controller-target-v1`；转换不会将前者冒充后者。数据转换与归一化完成也不表示
已经接通 FTP-1 在线仿真部署、验证真实推理频率或完成模型训练。

<a id="tactile-evaluation-runs"></a>

## 2026-10-19 至 10-21 模型诊断批次

本节保留具体运行、checkpoint 身份、验收限制及发布回执。记录中的“本轮”对应各节标注日期，
不是自动更新的作业状态。595/620 等数量是当时基线或发布计划，当前入口以
[统一评估与观察台](../workflows/unified_evaluation.md)及实际 `reports/index.html` 为准。
通用触觉部署与就绪度见[CPM / FTP-1 评估](../workflows/model_specific_evaluation.md#tactile-policy)。

### 2026-10-19 接入验收

本机接入验收（2026-10-19）已完成：两种模型均通过统一 `deploy run` 的 12 秒 K5
真实权重闭环，各 72 次重规划 / 360 个控制步，部署和物理检查有效，均进入卡牌接触阶段；
12 秒内均未完成原生任务，不能将此结果解释为正式成功率。两种模型的 2 秒检查还验证了
Raw HDF5 与视频收尾。CPM 同噪声 scale=0 对照、FTP-1 真实训练 Zarr codec parity
以及相关测试结果见[验收记录](../../reports/tactile_integration_20261019/acceptance.json)和
[原生 H30/H32 报告](../../reports/tactile_integration_20261019/accepted_summary.html)。

记录中的 `*_ready/deployment_manifest.json` 只保留为当时真实权重诊断链路的证据。
任务/触觉运行源码已经更新，旧 manifest 的源码 inventory 会 fail closed；复跑前必须用
当前源码重新 prepare，模型资产或 K 变化时也同样如此。不要把上述历史 manifest 直接
交给 `deploy run`；重新 prepare 与启动命令见[触觉策略评估](../workflows/model_specific_evaluation.md#tactile-policy)。

### 五个 checkpoint 的固定小样本评估

本轮按每个 checkpoint **5 次**固定为 seeds `0..4`：CPM `best.pt`（H30）和
FTP-1 `10000`、`15000`、`20000`、`24999`（H32），共 25 条，不包含 FTP-1 5000。
每条最多 90 秒仿真时间，30 Hz 控制、K5、ZOH、`guard_mode=observe`，保存 10 FPS
评估视频、请求日志和 `staged_v1` 状态证据；本轮不保存 Raw HDF5。
模型推理期间仿真暂停，实际耗时不等于 90 秒。

```bash
cd /cpfs_infra/user/chenxianchi/code/kaitactilesim
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
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

单模型族结果快照生成后，使用[统一工作台导出流程](evaluation_workbench_release_history_20261015_20261021.md#统一工作台跨任务模型与-checkpoint-对比)
发布首页，不另建互斥的模型入口。

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

### 灯泡原位旋入：五个 checkpoint、各五次

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

该独立作业现已完成：`status.json.state=complete`；`stage_summary.json` 记录五组、25 条，
各组均有五条有效 staged 轨迹（2026-10-21 只读核对）。这说明已有模型诊断记录，
不自动表示已通过平台增量发布或正式任务验收。上段 620 条计划以当时 595 条基线计算，
不得直接用于后续已增加 CPM 10 条的新版工作台。

### 当前 CPM 权重的 10 条独立复评

2026-10-21 的复评只运行扑克 CPM，不重跑 FTP-1。固定使用当前 `best.pt` 的
SHA-256 `e5f859a5755254e55e5fec660e8d11e8b5f27b7400cc4243ad48a133951186f3`，
先复制到新 job 的 `assets/cpm/best.pt` 并核对复制前后的原件和副本，再准备新的部署清单。
不能复用旧 5 条的部署，也不能把不同 SHA 的两个 `best` 合并为同一个 checkpoint 成绩。

```bash
# 输出目录必须不存在。此入口只接受上述已核对版本，不自动重试或按成绩挑选 attempt。
.pixi/envs/default/bin/python -B scripts/evaluate/evaluate_cpm_poker_isolated_checkpoint.py \
  --job-root /absolute/path/to/fresh/cpm_poker_job --start
```

实际 job 为
`/cpfs_infra/user/chenxianchi/kaitactilesim_evaluations/_jobs/cpm_poker_best_n10_20261021_r2`；
`plan.json` 固定 10 条 seeds `0..9`，`status.json` 和 `logs/controller.log` 可查看进度。
首次无后缀 job 因运行后训练仓库的两个源码文件被外部修改而中断，全部产物保留，
不计入本次 10 条。`_r2` 将模拟器及 OpenPI 源码复制到独立只读目录，复制前后核对
文件集合、原件和副本 SHA；`runtime_snapshot.json` 与部署清单绑定实际执行的副本，
不复用或修改原 job 清单，不按成功率挑选重跑。
每条 90 秒、H30/K5、30 Hz、ZOH、`observe`、`staged_v1`，全部保存 10 FPS 视频、
同帧触觉请求和 Raw HDF5；属于诊断评估，不改变原生成功判据。
初始卡牌按默认 ±4 mm XY / ±0.5° yaw 做 reset-only 随机化，不是固定初态重复。
共用 Raw 记录器的历史 `model_family` 标签可能仍写作 `pi0.5`，模型身份以对应
deployment SHA、checkpoint SHA 和 worker 握手中的 `pi05+cpm` 为准，不能只看该标签。
新批次完成后须通过权重、原生/阶段、原始记录和浏览器检查，才增量发布到统一工作台；
旧 595 条记录和旧 CPM 的失效阶段资格均保留。

本次已配置 `finalize_cpm_poker_evaluation.py` 的只监控/验收发布流程，不额外启动或
重试评估。当前发布控制记录位于 `_jobs/cpm_poker_best_n10_publication_20261021_r3`；
可用下面命令查看阶段，任一验收失败均保留旧入口和全部日志，不自动重新发布：

```bash
.pixi/envs/default/bin/python -B scripts/analysis/finalize_cpm_poker_evaluation.py --status
```

本批 10 条已完成并发布到 `reports/kaitactilesim_unified_workbench_20261021_cpm_n10_r2`：
原生有效 10/10、成功 0/10；`staged_v1` 三阶段依次为 9/10、0/10、0/10，
独立原始数据、平台快照及全部 10 页浏览器验收通过。工作台为 605 条／48 组，
旧 595 条的计分资格、指标与详情保留，新权重显示为 `best@e5f859a57552`。
数据与平台回执分别为 job 内 `qa/raw_evidence_r2.json`、`qa/platform_evidence_r2.json`；
浏览器回执在快照的 `browser_qa/cpm_additive_browser_verification.json`。
浏览器曾因长临时路径无法创建 Unix socket，后使用独立短目录续验；失败日志、
此前状态及人工续验计划均保存在发布控制目录，未因此改判、重跑模型或覆盖旧快照。

<a id="task-pipeline-examples"></a>

## 2026-10-12 至 10-18 任务流水线示例

本节保留水平单横线和灯泡原位旋入的任务专属检查；通用采集、转换、部署与评估见
[统一命令速查](../workflows/pipeline_quickstart.md)。先按[运行环境](../workflows/environment.md)核对本仓源码与依赖。
带日期的路径是原示例批次名；实际执行必须更换为全新输出目录。

### 水平单横线任务

`whiteboard-wipe-horizontal-line` 只保留原 Z 字顶部的 16 cm 横线；板仍水平朝上。
原 `whiteboard-wipe`（斜板 Z 字）和 `whiteboard-wipe-horizontal`（水平 Z 字）保持不变。
流程和物理成功门槛见[任务说明](../tasks/whiteboard_wipe.md#水平单横线whiteboard-wipe-horizontal-line)。

先采一条并转换，使用 NAS 的 `sim/<task>/{raw,pi05}/<批次>` 目录规范。
以下是待执行的示例命令，不代表已经产生该数据集；输出路径须不存在：

```bash
cd /cpfs_infra/user/chenxianchi/code/kaitactilesim
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

__EGL_VENDOR_LIBRARY_FILENAMES=/cpfs_infra/user/chenxianchi/.config/kaitactilesim/nvidia_egl_vendor.json \
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is collect -- \
  --task whiteboard-wipe-horizontal-line \
  --target-successes 1 --max-attempts 5 --workers 1 --seed 20261012 \
  --output-dir /nas/chenxianchi/datasets/sim/whiteboard-wipe-horizontal-line/raw/1012_1 \
  --staging-root /nas/chenxianchi/datasets/sim/_staging/whiteboard-wipe-horizontal-line/collect/1012_1 \
  --render-backend hardware

HF_HOME=/nas/chenxianchi/datasets/sim/_cache/huggingface \
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is convert-dataset -- \
  --task whiteboard-wipe-horizontal-line --format pi05 \
  --input-dir /nas/chenxianchi/datasets/sim/whiteboard-wipe-horizontal-line/raw/1012_1 \
  --output-dir /nas/chenxianchi/datasets/sim/whiteboard-wipe-horizontal-line/pi05/1012_1 \
  --dataset-name whiteboard_wipe_horizontal_line_1012_1 \
  --cameras head right_wrist --workers 1 --expected-episodes 1 \
  --staging-root /nas/chenxianchi/datasets/sim/_staging/whiteboard-wipe-horizontal-line/pi05/1012_1 \
  --openpi-root /cpfs_infra/user/chenxianchi/code/openpi \
  --pi05-python /cpfs_infra/user/chenxianchi/code/openpi/.venv-pi05/bin/python \
  --verify-source-hash
```

批量采集时使用全新批次目录，例如 `1012_200`，将采集参数改为
`--target-successes 200 --max-attempts 1000 --workers 8`，转换参数改为
`--workers 8 --expected-episodes 200`，并同步修改路径和 dataset name。
训练启动器现在默认选择 `pi05_kaihand_h50`；归一化统计与数据绑定，不能借用 Z 字数据集。
训练后，使用[统一速查](../workflows/pipeline_quickstart.md)中的 `deploy-policy prepare --task whiteboard-wipe-horizontal-line --model-family pi05`
及相应的 `verify/run`、`evaluate-policy`。π0.5 示例要求匹配的训练来源；其他模型接口与权重就绪度分别查询，见[触觉模型评估](../workflows/model_specific_evaluation.md#tactile-policy)。

### 灯泡原位旋入任务

`bulb-screw-inplace` 从灯座入口已啮合、未到座、未拧紧且灯灭的状态开始；
原 `bulb-screw` 仍包含桌面拿取、搬运和入牙，不能用旧轨迹改名代替。
先查询合同、检查初态，再运行无渲染专家；以下都是待执行示例：

```bash
cd /cpfs_infra/user/chenxianchi/code/kaitactilesim
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is collect -- --describe bulb-screw-inplace
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is check-bulb-screw-inplace
.pixi/envs/default/bin/python -B scripts/workcell/view_bulb_screw_inplace.py \
  --headless --run-task --seed 0
```

需要一条可供离线回放的三相机 Raw 时，通过统一采集入口写入新的批次目录：

```bash
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is collect -- \
  --task bulb-screw-inplace --target-successes 1 --max-attempts 1 \
  --workers 1 --seed 20261015 --render-backend software \
  --output-dir /path/to/bulb_screw_inplace_raw_new \
  --staging-root /path/to/bulb_screw_inplace_staging_new
```

软件渲染较慢；使用 GPU 前按[环境说明](../workflows/environment.md#nvidia-egl-与软件渲染)做硬件预检。当前初态固定，seed 只标识可复现运行，
多 seed 不代表不同初态泛化。物理成功要求抓稳、真实旋入到座、承力停转和释放后稳定，
不看灯亮或固定拧动次数；回 home 单独作为收尾记录，完整专家流程的结果另行保留。

采集后可将[统一速查第 2 节](../workflows/pipeline_quickstart.md#2-转换为模型数据或-lerobot-v3)的 `TASK` 换为 `bulb-screw-inplace`，独立转换为 π0.5 或 LeRobot v3。
只有使用匹配新任务数据训练的 checkpoint，才可进入[部署与评估](../workflows/pipeline_quickstart.md#3-准备模型并自动接入仿真)；该任务的 π0.5 部署只接受
`shared_task_pi05_deployment_v2`，并要求新的 recorded-controller-target 来源合同、
任务 instruction、mechanics version 与场景指纹一致。不得重标旧 `bulb-screw` checkpoint
或手改旧 manifest。2026-10-20 的五 checkpoint 模型诊断批次已完成，固定初态与结果边界见
[评估批次记录](#tactile-evaluation-runs)。
