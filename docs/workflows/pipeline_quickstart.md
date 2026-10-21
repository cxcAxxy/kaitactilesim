# 采集转换部署与评估速查

本页只提供完整流程的最短命令顺序。参数含义、恢复方式、数据合同和模型专属要求，
分别由[采集](data_collection.md)、[转换](data_conversion.md)、[部署](model_deployment.md)、
[模型评估](model_evaluation.md)和[模型专项配置](model_specific_evaluation.md)维护。
任务初态与专家动作见[任务指南](../README.md#任务指南)，具体旧批次见[历史记录](../archive/README.md)。

## 环境与工作目录

先按[运行环境](environment.md)安装依赖、核对导入来源并检查渲染。
每个新终端从本仓库执行：

```bash
cd /cpfs_infra/user/chenxianchi/code/kaitactilesim
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is collect -- --list-support
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is convert-dataset -- --list-support
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is deploy-policy -- --list-support
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is evaluate-policy -- --list-support
```

下面把 `TASK` 换成实际任务名，把 `FORMAT` 换成该任务支持的格式。
输出目录使用全新路径；`--as-is` 只复用已安装环境，不补装依赖。

## 1. 采集共享 Raw

先对目标任务做硬件渲染检查：

```bash
__EGL_VENDOR_LIBRARY_FILENAMES=/cpfs_infra/user/chenxianchi/.config/kaitactilesim/nvidia_egl_vendor.json \
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is collect -- \
  --task TASK --check --render-backend hardware
```

检查通过后采集：

```bash
__EGL_VENDOR_LIBRARY_FILENAMES=/cpfs_infra/user/chenxianchi/.config/kaitactilesim/nvidia_egl_vendor.json \
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is collect -- \
  --task TASK --target-successes 200 --max-attempts 400 \
  --workers 4 --seed 20261021 \
  --output-dir /path/to/TASK_raw --render-backend hardware
```

首次可以把 `--target-successes 200 --max-attempts 400` 换为 `--episodes 1`。
没有 NVIDIA EGL 时去掉 vendor 前缀，显式使用 `--render-backend software`。
尝试数、成功数、续采和发布校验见[统一采集](data_collection.md)；先用 `--dry-run` 可预览 recorder 计划。

## 2. 转换为模型数据或 LeRobot v3

```bash
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is convert-dataset -- \
  --input-dir /path/to/TASK_raw --output-dir /path/to/TASK_FORMAT \
  --task TASK --format FORMAT --workers 4 --expected-episodes 200 \
  --dry-run
```

核对计划后移除 `--dry-run`。统一格式为 `pi05`、`lerobot-v3`、`egosteer`、`egotouch`，
按 `--list-support` 选择组合与相机；FTP-1 使用独立转换入口。
恢复能力及外部 Python 参数见[统一转换](data_conversion.md)，字段、时钟与动作语义见
[数据格式合同](../reference/dataset_formats.md)。历史数据不能通过改任务名或补造相机变成新合同。

## 3. 准备模型并自动接入仿真

模型在外部项目完成训练。复制并修改 [runtimes.example.toml](../../configs/runtimes.example.toml)
为本地 `configs/runtimes.local.toml`，设置实际模型 Python 和项目路径。
下面以任务匹配的 π0.5 checkpoint 为例，先预览 prepare：

```bash
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is deploy-policy -- prepare \
  --task TASK --model-family pi05 --checkpoint /path/to/checkpoint \
  --output-dir /path/to/deployment \
  --runtime-config configs/runtimes.local.toml --runtime pi05 --dry-run
```

核对计划后移除 `--dry-run` 生成 deployment，再执行核验和运行计划：

```bash
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is deploy-policy -- verify \
  --deployment-manifest /path/to/deployment/deployment_manifest.json \
  --runtime-config configs/runtimes.local.toml --runtime pi05

/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is deploy-policy -- run \
  --deployment-manifest /path/to/deployment/deployment_manifest.json \
  --runtime-config configs/runtimes.local.toml --runtime pi05 \
  --output-dir /path/to/TASK_pi05_run \
  --num-trials 20 --video-count 20 --execute-steps horizon \
  --max-sim-seconds 90 --guard-mode observe --dry-run
```

移除 run 的 `--dry-run` 才启动完整闭环。首次可用 `--num-trials 1 --video-count 1`。
服务管理、核验、退出清理和默认值只在[模型部署](model_deployment.md)维护；
任务专属 checkpoint、normalizer 与动作绑定见[模型专项配置](model_specific_evaluation.md)。

## 4. 按任务和模型独立评估

已有 deployment 且 π0.5 服务已经启动时，可单独运行评估：

```bash
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is evaluate-policy -- \
  --task TASK --model-family pi05 \
  --deployment-manifest /path/to/deployment/deployment_manifest.json \
  --output-dir /path/to/TASK_pi05_eval \
  --num-trials 20 --video-count 20 --execute-steps horizon \
  --max-sim-seconds 90 --guard-mode observe --dry-run \
  -- --server ws://127.0.0.1:18783
```

确认计划后移除 `--dry-run`。`observe` 是诊断模式，结果不能直接作为正式指标；
`strict` 也仍须通过单条有效性检查。`--max-sim-seconds` 是仿真时间，`--execute-steps`
是每次预测后的执行长度，两者的范围与例外见[模型评估](model_evaluation.md)。

上例仅适用于 π0.5 网络服务。EgoTouch、CPM、FTP-1、LingBot 等的 worker、输入合同、
冻结 K 与专属参数见[模型专项配置](model_specific_evaluation.md)，不要原样套用 `--server`。

结果字段和统计分母见[模型评估](model_evaluation.md)，视频画面与文件见[评测视频](policy_evaluation_video.md)，
原生成功见[成功判据](../tasks/success_criteria.md)，阶段里程碑见[阶段判定](staged_evaluation.md)。
