# 采集 → 转换 → 部署：统一命令速查

日常流程使用 `collect`、`convert-dataset`、`deploy-policy` 三个入口。
改任务时替换 `TASK`，改训练数据格式时替换 `FORMAT`，改模型时替换 `MODEL`。
`evaluate-policy` 保留为已有部署的独立评估入口。

各任务的实际成功门槛见[成功判据总表](../tasks/success_criteria.md)，阶段诊断的口径与支持范围见
[分阶段评估](staged_evaluation.md)。原生任务成功、阶段进度和 trial 有效性不能互相替代。

## 环境与工作目录

下面使用本节点已核对的 Pixi 绝对路径，不依赖 shell 的 `PATH`。
先进入仓库根目录，再查当前代码支持什么组合：

```bash
cd /cpfs_infra/user/chenxianchi/code/kaitactilesim_unified

/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is collect -- --list-support
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is convert-dataset -- --list-support
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is deploy-policy -- --list-support
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is evaluate-policy -- --list-support
```

若其他节点没有上述 Pixi，但已安装仓库环境，也可使用
`/cpfs_infra/user/chenxianchi/code/kaitactilesim_unified/.pixi/envs/default/bin/python`
直接运行对应的 `scripts/collect/collect.py`、`scripts/convert/convert.py`、
`scripts/deploy/deploy.py` 或 `scripts/evaluate/evaluate.py`，后续参数相同。

本节点的 unified 独立环境是从缓存恢复的：安装时使用了
`pixi install --locked --offline --skip-with-deps pyarrow`，因此目前缺少 pyarrow。
本页统一使用 `pixi run --as-is` 复用已安装环境，不触发自动安装；也可用上面的项目 Python。
这不是完整生产转换环境：正式转换前需补齐对应后端的全部依赖，并核对实际执行转换的
OpenPI/LeRobot Python。`--as-is` 不会安装缺失包，也不会绕过任务或部署合同校验。
新节点应按锁文件完整安装，不要把跳过 pyarrow 当作通用安装要求。

## 1. 采集共享 Raw

`TASK` 可选 `pick-place`、`poker-draw`、`usb-insert`、`bulb-screw`、
`install-ram`、`vase-wipe`、`whiteboard-wipe`、`sponge-grasp`、`usb-unplug`、
`whiteboard-wipe-horizontal`、`bulb-unscrew`、`whiteboard-wipe-horizontal-line`、
`bulb-screw-inplace`，共十三个独立任务；采集时不指定模型。
原 `usb-insert`、`bulb-screw`、斜板 `whiteboard-wipe` 不删除、不改名。
五个变体的初态、流程与验收见[任务说明](../tasks/task_variants.md)；
灯泡原位旋入的五步流程另见[专项说明](../tasks/bulb_screw_inplace.md)。

当前 `usb-unplug` 已改为完全拔出后手持且不触桌连续保持 2 秒，协议
`usb-unplug-held-clear-2s-v2`；不再要求放回、松手或退手。新采集使用独立输出目录，
不能 `--resume` 到旧放回版批次。当前转换入口要求新版协议和连续保持证据，
不把旧放回版示教换成新的短任务指令；旧数据仍按旧代码/归档合同复现。

本节点的 NVIDIA EGL 需要显式指定 vendor 配置。
`--render-backend hardware` 只选择并检查硬件模式，不会自动设置这个配置路径；
下面通过环境变量 `__EGL_VENDOR_LIBRARY_FILENAMES` 传入，它不是采集 CLI 参数，
也不是后面转换/部署使用的 `--runtime-config`。先对目标任务做渲染预检：

```bash
__EGL_VENDOR_LIBRARY_FILENAMES=/cpfs_infra/user/chenxianchi/.config/kaitactilesim/nvidia_egl_vendor.json \
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is collect -- \
  --task TASK --check --render-backend hardware
```

预检应报告实际 NVIDIA renderer 和 `software: false`，通过后再采集。
软件渲染改为 `--render-backend software`，并去掉 NVIDIA vendor 环境变量前缀。

```bash
__EGL_VENDOR_LIBRARY_FILENAMES=/cpfs_infra/user/chenxianchi/.config/kaitactilesim/nvidia_egl_vendor.json \
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is collect -- \
  --task TASK --target-successes 200 --max-attempts 400 \
  --workers 4 --seed 20260926 \
  --output-dir /path/to/TASK_raw \
  --render-backend hardware
```

先把 `--target-successes 200 --max-attempts 400` 换成 `--episodes 1` 做单条 smoke。
`--episodes` 统计尝试数，`--target-successes` 统计成功数，两者互斥。预览子命令
加 `--dry-run`，中断续采加 `--resume`；后者必须保持原任务、seed 和输出目录。

## 2. 转换为模型数据或 LeRobot v3

同一份 Raw 可以分别转换到不同的全新输出目录，不修改原始 HDF5：

```bash
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is convert-dataset -- \
  --input-dir /path/to/TASK_raw \
  --output-dir /path/to/TASK_FORMAT \
  --task TASK --format FORMAT \
  --workers 4 --expected-episodes 200 \
  --dry-run
```

确认计划后移除 `--dry-run`。`FORMAT` 为 `pi05`、`lerobot-v3`、`egosteer`
或 `egotouch`；并非每个任务都支持每种格式，以 `--list-support` 为准。
`--cameras` 可省略：固定相机合同的 adapter 会自动选择正确视角，其余默认
`head`。需要更多训练视角时显式传 `--cameras head left_wrist right_wrist`，但不能
临时改变固定的模型合同或要求 Raw 中不存在的相机。

`--format pi05` 是 π0.5 训练适配器：任务语义动作和模型输入按对应 adapter
冻结，当前高速后端输出 LeRobot v2.1。`--format lerobot-v3` 是另一条模型中立
的 LeRobot v3 路径，固定使用 `head + right_wrist`，目前支持全部十三个采集任务。
Sponge 使用 100 Hz 控制步上的 30 Hz 相机 deadline，转换器会按实际控制步验证
其时间戳。它不是 π0.5 数据的版本升级，也不能直接当作 π0.5 训练集。

LeRobot v3 中断后，使用相同参数和路径加 `--resume`；也可指定
`--work-dir /path/to/checkpoints`，把断点放在持久磁盘上。π0.5 的高速 adapter
使用自动断点工作目录；USB π0.5 等未标注可续跑的 adapter 不接受统一
`--resume`。正式输出目录应是新路径，不覆盖已发布数据。

LeRobot v3 需要安装 LeRobot 0.4.x 的 Python 环境；通过 Pixi 统一命令调用时增加
`--lerobot-v3-python /path/to/lerobot-env/bin/python`。若默认 HOME 不可写，
命令前设置 `HF_HOME=/path/to/writable/huggingface_cache`。该格式目前只接受达到成功数
目标的统一采集批次或其支持的合并批次；多任务采集根目录要显式指定 `--task`，
转换器只取该任务的成功轨迹。

## 3. 准备模型并自动接入仿真

模型训练在对应外部项目中完成。转换和部署共用命名运行环境配置，示例见
[runtimes.example.toml](../../configs/runtimes.example.toml)。复制为本地配置并修改环境路径；
配置里的路径相对配置文件解析。π0.5 转换可在上一步增加
`--runtime-config configs/runtimes.local.toml --runtime pi05`。
新增五个变体任务的 π0.5 转换同样固定使用 `head + right_wrist`，并绑定自己的
task、mechanics version 和场景指纹。需要用对应的新数据训练 checkpoint；
不能把插 USB、旋入灯泡或斜板的旧 manifest 改名当作新任务部署。

```bash
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is deploy-policy -- prepare \
  --task TASK --model-family pi05 \
  --checkpoint /path/to/checkpoint \
  --output-dir /path/to/deployment \
  --runtime-config configs/runtimes.local.toml --runtime pi05 \
  --dry-run

/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is deploy-policy -- verify \
  --deployment-manifest /path/to/deployment/deployment_manifest.json \
  --runtime-config configs/runtimes.local.toml --runtime pi05

/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is deploy-policy -- run \
  --deployment-manifest /path/to/deployment/deployment_manifest.json \
  --runtime-config configs/runtimes.local.toml --runtime pi05 \
  --output-dir /path/to/TASK_pi05_run \
  --num-trials 20 --video-count 20 \
  --execute-steps horizon \
  --max-sim-seconds 90 --guard-mode observe \
  --dry-run
```

分别核对 prepare/run 计划后移除 `--dry-run`。上例的 π0.5 使用默认 managed
模式：`run` 根据冻结 manifest 识别任务与模型，管理本机服务启动、身份和就绪检查、
评估与退出清理。managed 如显式传 `--endpoint`，只接受 loopback `ws://`；远程或
TLS 服务使用 `--service-mode external --endpoint ...`，统一入口只连接且不会停止它。
EgoTouch local-worker 和 T-Rex embedded 不接受 external。T-Rex 复用 π0.5 base
manifest，必须显式加 `--model-family pi05+trex`，不能只从 manifest 自动推断。
首次可以用 `--num-trials 1 --video-count 1` 检查完整闭环；短时检查再加
`--smoke-seconds 10`，它会覆盖本次 `--max-sim-seconds`，且无论保护模式如何都不作为
正式任务指标。旧冻结部署如因评估源码变化核验失败，应在新目录重新 `prepare`，
不能手改 hash 或绕过检查；详见[准备与核验](model_deployment.md#准备与核验)。
根目录保存运行记录，结果写入 `evaluation/`。其他模型按 `--list-support` 中的逐阶段
能力使用；模型专属准备参数放在 `--` 后。详细用法见[统一模型部署](model_deployment.md)。

USB 插入的训练数据名是 `usb_insert`，此处部署任务名仍为 `usb-insert`。
新的 USB H50 checkpoint 使用独立的 `normalizer_h50_usb_insert_hand_tail_v1`，
`prepare` 从 checkpoint 的训练 provenance 读取实际 config、horizon 和归一化资产，
生成 `usb_pi05_deployment_v2`；训练与服务使用同一份 checkpoint 内统计。
不覆盖旧 `normalizer_h50`，也不能用新统计 resume 旧 run。旧 H30/v1 部署保留兼容，
不要手改旧 manifest 的 horizon 或资产名；训练修复与命令见
[OpenPI USB H50 说明](../../../openpi/examples/kaihand/README.md#usb-插入-h50手指尾部安全归一化)。

### 修改 π0.5 每次推理后的执行步数

`--execute-steps` 控制每次预测后实际执行多少个 action，不修改模型训练规定的
预测 horizon。π0.5 不指定时默认执行完整 horizon；上例显式写成
`--execute-steps horizon`。若想每次只执行 8 步，将这一行改为：

```text
--execute-steps 8
```

例如训练 horizon 为 50，模型仍每次预测 50 步，但只执行前 8 步，随后重新观测和
推理，旧预测的剩余 42 步丢弃。执行步数必须满足 `1 ≤ K ≤ horizon`。
这是本次评估的运行参数，不需要重新训练、重新 prepare 或修改 deployment manifest；
使用不同输出目录比较不同执行步数的结果。

## 4. 按任务和模型独立评估

评估需要对应模型事先准备好的 deployment manifest；网络模型还需要已启动的
模型服务。这些身份信息不能由 Raw 或任务名自动推断。

```bash
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is evaluate-policy -- \
  --task TASK --model-family MODEL \
  --deployment-manifest /path/to/deployment_manifest.json \
  --output-dir /path/to/TASK_MODEL_eval \
  --num-trials 20 --video-count 20 \
  --execute-steps horizon \
  --max-sim-seconds 90 --guard-mode observe --dry-run \
  -- --server ws://127.0.0.1:18783
```

独立评估同样可以把 `--execute-steps horizon` 改为 `--execute-steps 8`，
该参数必须放在 `-- --server` 之前。`evaluate-policy` 还支持一次比较多个值，
如 `--execute-steps 5 8 16 horizon`，各组使用相同 seeds 并分目录保存；
`deploy-policy run` 当前一次只接受一个值。非 π0.5 模型还需遵守后端的专项限制，
例如 PickPlace EgoTouch 固定执行 5 步。详细说明见[Action chunk 执行长度](model_evaluation.md#action-chunk-执行长度)。

部署 `run` 和独立评估均默认每次最多 **90 秒仿真时间**，不等于墙钟运行时间。
可将 `--max-sim-seconds 90` 改为任意有限正值，例如 `120`；该参数和 `--guard-mode`
都放在 `-- --server` 之前。默认 `observe` 会记录力过大、碰撞、掉落等任务保护，
继续仿真与录像；已识别的无效动作保持上一条完整有效指令并重新推理。任务成功、
时限到达、数值或基础设施故障仍结束本次 rollout，视频不会冻结补帧。
observe 结果仅用于诊断（`formal_metrics_valid=false`），不能与旧 strict 成功率混用。
需要任务保护触发即停止、用于正式比较时，改成 `--guard-mode strict` 并检查 trial
有效性。运行参数本身不改变冻结模型，不要求重新 prepare；独立评估不接受
`--smoke-seconds`。事件字段与其他后端停止条件见[评估文档](model_evaluation.md)。

确认计划后移除 `--dry-run`。`MODEL` 以 `--list-support` 为准，常见为
`pi05`、`egosteer`、`egotouch`，Card 另有 `pi05+trex`、PickPlace 另有
`lingbot-vla2` 专项 runner。
`--` 后只放该 runner 的专属参数；EgoTouch 本地模型等没有 WebSocket 的
runner 不使用示例中的 `--server`。统一入口会校验 manifest 的任务、模型族
和相机合同，并拒绝不支持的组合。目前 `sponge-grasp` 支持采集、π0.5 和
LeRobot v3 转换，以及 π0.5 统一闭环评估。

Bulb 0923 的 `20000` step checkpoint 的部署、服务与 20 次评估命令，见
[Bulb 历史诊断实例](model_evaluation.md)。该历史 checkpoint 不用于当前正式指标；
正式评估需使用与当前任务版本及数据 provenance 绑定的 checkpoint。

详细参数和异常处理分别见[采集](data_collection.md)、[转换](data_conversion.md)、
[LeRobot v3](lerobot_v3.md)、[部署](model_deployment.md)和[评估](model_evaluation.md)。

## 5. 水平单横线任务

`whiteboard-wipe-horizontal-line` 只保留原 Z 字顶部的 16 cm 横线；板仍水平朝上。
原 `whiteboard-wipe`（斜板 Z 字）和 `whiteboard-wipe-horizontal`（水平 Z 字）保持不变。
流程和物理成功门槛见[任务说明](../tasks/task_variants.md#水平单横线whiteboard-wipe-horizontal-line)。

先采一条并转换，使用 NAS 的 `sim/<task>/{raw,pi05}/<批次>` 目录规范。
以下是待执行的示例命令，不代表已经产生该数据集；输出路径须不存在：

```bash
cd /cpfs_infra/user/chenxianchi/code/kaitactilesim_unified

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
训练后，使用上文 `deploy-policy prepare --task whiteboard-wipe-horizontal-line --model-family pi05`
及相应的 `verify/run`、`evaluate-policy`。目前新任务只注册 π0.5 模型评估。

## 6. 灯泡原位旋入任务

`bulb-screw-inplace` 从灯座入口已啮合、未到座、未拧紧且灯灭的状态开始；
原 `bulb-screw` 仍包含桌面拿取、搬运和入牙，不能用旧轨迹改名代替。
先查询合同、检查初态，再运行无渲染专家；以下都是待执行示例：

```bash
cd /cpfs_infra/user/chenxianchi/code/kaitactilesim_unified
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

软件渲染较慢；使用 GPU 前按第 1 节做硬件预检。当前初态固定，seed 只标识可复现运行，
多 seed 不代表不同初态泛化。物理成功要求抓稳、真实旋入到座、承力停转和释放后稳定，
不看灯亮或固定拧动次数；回 home 单独作为收尾记录，完整专家流程的结果另行保留。

采集后可将第 2 节的 `TASK` 换为 `bulb-screw-inplace`，独立转换为 π0.5 或 LeRobot v3。
只有使用匹配新任务数据训练的 checkpoint，才可进入第 3–4 节；该任务只接受
`shared_task_pi05_deployment_v2`，并要求新的 recorded-controller-target 来源合同、
任务 instruction、mechanics version 与场景指纹一致。不得重标旧 `bulb-screw` checkpoint
或手改旧 manifest。目前已验证的是专家和接口，不代表已经完成新任务模型评估。

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
以上为可执行命令，不表示已经启动或完成了这次新任务训练。

## 7. 九任务 10000/15000 checkpoints 的五次阶段诊断

新入口 `scripts/evaluate/evaluate_pi05_stage_batch.py` 固定选择已核验的 9 个 H50 任务与数据批次，
不使用历史专用的 `run_pi05_staged_groups.py`。每任务 `10000/15000` 各运行 seed `0..4`，
共 18 组、90 条；K50、90 秒仿真、`observe`、`staged_v1`，各组串行且保存 5 份头部/右腕回放。

旧 USB 拔出模型来自“放回释放”训练，本次评分采用“拔出并手持清离 2 秒”。
必须显式开启以下跨协议诊断选项；保留原模型、提示词和训练来源，归档训练/评估两套协议，
永远不能将此结果声明为正式有效。默认无该选项仍拒绝不匹配的旧 USB 来源。

```bash
cd /cpfs_infra/user/chenxianchi/code/kaitactilesim_unified
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
cd /cpfs_infra/user/chenxianchi/code/kaitactilesim_unified
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
