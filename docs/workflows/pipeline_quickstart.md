# 采集 → 转换 → 评估：统一命令速查

现有任务采集器、模型转换器和评估 runner 保持原样；三个统一入口只负责选择、
校验和分发。改任务时替换 `TASK`，改训练数据格式时替换 `FORMAT`，改评估模型时
替换 `MODEL`，不需要改脚本路径或复制一套控制逻辑。

先查当前代码支持什么组合：

```bash
pixi run collect -- --help
pixi run convert-dataset -- --list-support
pixi run evaluate-policy -- --list-support
```

若当前 shell 没有 `pixi` 命令而仓库已安装 `.pixi/envs/default/`，
统一评估可在仓库根目录改用
`.pixi/envs/default/bin/python scripts/evaluate/evaluate.py`；参数与
`pixi run evaluate-policy --` 完全相同。

## 1. 采集共享 Raw

`TASK` 可选 `pick-place`、`poker-draw`、`usb-insert`、`bulb-screw`、
`install-ram`、`vase-wipe`、`whiteboard-wipe`、`sponge-grasp`；采集时不指定模型。

```bash
pixi run collect -- \
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
pixi run convert-dataset -- \
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
的 LeRobot v3 路径，固定使用 `head + right_wrist`，目前支持全部八个采集任务。
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

## 3. 按任务和模型评估

评估需要对应模型事先准备好的 deployment manifest；网络模型还需要已启动的
模型服务。这些身份信息不能由 Raw 或任务名自动推断。

```bash
pixi run evaluate-policy -- \
  --task TASK --model-family MODEL \
  --deployment-manifest /path/to/deployment_manifest.json \
  --output-dir /path/to/TASK_MODEL_eval \
  --num-trials 20 --video-count 20 \
  --execute-steps horizon --dry-run \
  -- --server ws://127.0.0.1:18783
```

确认计划后移除 `--dry-run`。`MODEL` 以 `--list-support` 为准，常见为
`pi05`、`egosteer`、`egotouch`，Card 另有 `pi05+trex`、PickPlace 另有
`lingbot-vla2` 专项 runner。
`--` 后只放该 runner 的专属参数；EgoTouch 本地模型等没有 WebSocket 的
runner 不使用示例中的 `--server`。统一入口会校验 manifest 的任务、模型族
和相机合同，并拒绝不支持的组合。目前 `sponge-grasp` 支持采集、π0.5 和
LeRobot v3 转换，尚无统一评估 runner。

Bulb 0923 的 `20000` step checkpoint 如何生成 deployment manifest、
启动服务和运行完整 20 次评估，见[Bulb 完整评估实例](model_evaluation.md)。

详细参数和异常处理分别见[采集](data_collection.md)、[转换](data_conversion.md)、
[LeRobot v3](lerobot_v3.md)和[评估](model_evaluation.md)。
