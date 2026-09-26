# 统一数据转换

统一入口是 `scripts/convert/convert.py`，Pixi 快捷命令为：

```bash
pixi run convert-dataset -- --help
pixi run convert-dataset -- --list-support
```

只想复用一套采集、转换、评估命令时，先看[统一命令速查](pipeline_quickstart.md)。

调用者显式指定 Raw 输入、输出目录、模型数据格式、任务和相机。统一入口负责发现、合同检查和 adapter 分发；它不会根据文件名猜测任务 action 的含义。

## 基本参数

```text
--input-dir PATH
--output-dir PATH
--format egosteer|pi05|egotouch|lerobot-v3
--task auto|pick-place|poker-draw|usb-insert|bulb-screw|vase-wipe|install-ram|whiteboard-wipe|sponge-grasp
--cameras head [left_wrist] [right_wrist]
--workers N
--expected-episodes N
--staging-root PATH
--work-dir PATH                 # 仅 LeRobot v3
--repo-id ID                   # 仅 LeRobot v3
--lerobot-v3-python PATH        # 当前 Python 未安装 LeRobot 时必填
--verify-source-hash
--resume
--dry-run
```

固定相机合同的 adapter 会自动选择其相机集合，其余默认只有 `head`；
显式选择时必须包含 `head`，顺序会规范为
`head, left_wrist, right_wrist` 的共享顺序。Raw 可以采三相机，但转换时只读取
`--cameras` 指定的子集；未选相机不会进入模型数据集。

`--format lerobot-v3` 接入模型中立的右臂/右手 LeRobot v3 转换器，支持全部
八个采集任务，固定使用 `head` 与 `right_wrist`。它和 `--format pi05` 的任务专属
训练数据不是同一种合同。Sponge 额外核对已通过的采集审计，并按 100 Hz 控制步
上的 30 Hz 相机 deadline 验证源时间戳；其柔体顶点和盘子支撑力保留为任务诊断。
该后端接受统一采集批次和其支持的历史/合并批次，
续跑时传 `--resume`，可用 `--work-dir` 指定持久断点位置。

`--expected-episodes 0` 接受实际发现数量；正式批次建议显式给出预期数量，避免把
路径写错后仍转换一个不完整批次。

## 任务识别

- `--task auto` 从 HDF5 的 `metadata_json.scene` 读取任务，要求输入根目录下只有一种任务。
- 显式 `--task` 会从混合 Raw 根目录中只选择该任务。
- 没有任务元数据的分析 HDF5 会被忽略。
- adapter 不支持某个 `(task, format)` 时明确失败，不产生猜测结果。
- `--list-support` 从代码里的 adapter 注册表直接输出当前支持能力，文档表格不是
  调度依据。

统一采集批次以 `summary.json` 中已成功、已发布的 episode 为准；转换入口和后端
使用相同的选择范围。PickPlace、Card、USB 的 EgoSteer 与 π0.5 适配器同时接受历史
平铺批次和 `task_collection_v2` 嵌套批次，保存外层采集编号。Bulb、RAM、Vase、
Whiteboard 和 Sponge 的共享后端也接受嵌套 Raw。

## 使用示例

Card 转 EgoSteer，使用三相机：

```bash
pixi run convert-dataset -- \
  --input-dir /path/to/card_raw \
  --output-dir /path/to/card_egosteer \
  --format egosteer --task poker-draw \
  --cameras head left_wrist right_wrist \
  --workers 4
```

USB 转现有双相机 π0.5 合同：

```bash
pixi run convert-dataset -- \
  --input-dir /path/to/usb_raw \
  --output-dir /path/to/usb_pi05 \
  --format pi05 --task usb-insert \
  --cameras head right_wrist \
  --workers 4 --staging-root /path/to/fast_staging
```

USB π0.5 适配器同时接受历史扁平 Raw 批次和统一采集器生成的
`task_collection_v2` 嵌套目录，包含 `attempts`、`target-successes` 和多任务采集根目录。
统一批次只选择 USB 的成功 episode；外层采集编号会映射为 LeRobot 来源编号，
原始 HDF5 不会改名或改写。

Sponge Raw 转 π0.5（模型使用头部和右腕相机）：

```bash
pixi run convert-dataset -- \
  --input-dir /path/to/sponge_raw \
  --output-dir /path/to/sponge_pi05 \
  --format pi05 --task sponge-grasp \
  --cameras head right_wrist \
  --workers 4 --expected-episodes 200 \
  --staging-root /path/to/staging \
  --dataset-name sponge_grasp_dataset \
  --openpi-root /path/to/openpi \
  --pi05-python /path/to/openpi/.venv-pi05/bin/python
```

Sponge 的 30 Hz 相机帧落在 100 Hz 控制步上；专属校验会核对每帧的精确控制步
位置，保留 Raw 时间戳与来源编号，不修改其他任务的相机时间规则。正式输出目录须不存在。

同一批 Sponge Raw 转模型中立 LeRobot v3：

```bash
HF_HOME=/path/to/writable/huggingface_cache pixi run convert-dataset -- \
  --input-dir /path/to/sponge_raw_collection \
  --output-dir /path/to/sponge_lerobot_v3 \
  --format lerobot-v3 --task sponge-grasp \
  --expected-episodes 200 --workers 3 \
  --lerobot-v3-python /path/to/lerobot-env/bin/python \
  --dry-run
```

确认计划后移除 `--dry-run`。该格式要求已达成功数目标的统一采集批次；
训练集的视频以名义 30 fps 编码，来源帧的真实 30/40 ms 控制步间隔保存在 provenance 中。
官方 LeRobot reader 会使用 Hugging Face 缓存；在默认 HOME 不可写的节点上必须把
`HF_HOME` 指向可写目录。

USB 等八任务转换为模型中立 LeRobot v3（现有 Pixi 环境未安装 LeRobot，故显式选择
安装了 LeRobot 0.4.x 的 Python）：

```bash
pixi run convert-dataset -- \
  --input-dir /path/to/raw_collection \
  --output-dir /path/to/usb_lerobot_v3 \
  --format lerobot-v3 --task usb-insert \
  --expected-episodes 200 --workers 3 \
  --lerobot-v3-python /path/to/lerobot-env/bin/python \
  --work-dir /path/to/usb_v3_checkpoints --dry-run
```

确认后移除 `--dry-run`；中断时沿用相同命令加 `--resume`。多任务成功数批次中，
`--task` 只选择该任务的已发布成功轨迹；该后端暂不接受 attempts-only 批次。

PickPlace 和 Card 的 π0.5 wrapper 不改名、不复制也不改写 Raw。统一采集批次中每个
隔离 recorder 可以都生成内部编号 0；wrapper 会使用 `summary.json` 的外层编号作为
LeRobot 来源编号，避免 200 条样本发生编号冲突。

除 USB 外的 π0.5 本地 wrapper 使用统一高速发布后端：每个 worker 独立转换一条
episode，HDF5 RGB 以批次直接送入 FFmpeg，输出 LeRobot v2.1 的 video feature，
不会生成逐帧临时 PNG。状态、动作、时间戳和来源编号仍由 OpenPI 锁定的任务转换器
计算，因此加速不会改变任务语义。

Card/USB 转 EgoTouch：

```bash
pixi run convert-dataset -- \
  --input-dir /path/to/raw \
  --output-dir /path/to/egotouch \
  --format egotouch --task auto \
  --cameras head left_wrist right_wrist \
  --workers 4 --resume
```

`--resume` 不会仅凭目录存在就跳过：它会检查发布审计、session、来源路径/大小、
采集 sidecar 摘要、相机名、selector 和触觉 NPZ。残缺或串用的目录会报错并保留
现场，不会混进完成 manifest。

先只检查任务、相机和 adapter，不写输出：

```bash
pixi run convert-dataset -- \
  --input-dir /path/to/raw \
  --output-dir /path/to/planned_output \
  --format egosteer --task auto \
  --cameras head --dry-run
```

## 当前支持矩阵

| 任务 | EgoSteer | π0.5 | LeRobot v3 | EgoTouch |
|---|---|---|---|---|
| PickPlace | head | head | head + right_wrist | 待接入 |
| Card | 包含 head 的任意 Raw 相机子集 | head + right_wrist | head + right_wrist | 包含 head 的任意子集 |
| USB | 包含 head 的任意 Raw 相机子集 | head + right_wrist | head + right_wrist | 包含 head 的任意子集 |
| Bulb | 包含 head 的任意 Raw 相机子集 | 包含 head 的任意 Raw 相机子集 | head + right_wrist | 待接入 |
| RAM | 包含 head 的任意 Raw 相机子集 | 包含 head 的任意 Raw 相机子集 | head + right_wrist | 待接入 |
| Vase | 待接入 | 包含 head 的任意 Raw 相机子集 | head + right_wrist | 待接入 |
| Whiteboard | 包含 head 的任意 Raw 相机子集 | 包含 head 的任意 Raw 相机子集 | head + right_wrist | 包含 head 的任意子集 |
| Sponge grasp | 待接入 | head + right_wrist | head + right_wrist | 待接入 |

Bulb/RAM 的 EgoSteer 与 π0.5 adapter 已冻结并注册：EgoSteer 使用归档的双手腕/
十指 taskspace 形成 48 维下一帧动作；π0.5 使用右臂 7 关节与右手 20 关节形成
27 维绝对关节状态/动作。两者只接受成功 sidecar，并检查 30 Hz 相机同步。
Vase 的 π0.5 adapter 使用相同的 27 维右臂+右手合同，保留真实 500 Hz 源状态时钟，
在 30 Hz 相机时刻插值状态并使用下一相机帧作为动作，不就地降采样或改写 Raw。
Bulb、RAM、Vase 的 EgoTouch 以及 Vase 的 EgoSteer 仍须完成各自的模型数据合同，
不能仅凭 Raw 结构宣称可训练。Whiteboard 使用共享右臂+右手动作合同，三种
输出格式均已在注册表中显式接入。Sponge 使用相同的右臂 7 关节＋右手 20 关节
合同，但只注册 π0.5 的头部＋右腕双相机转换。

## 历史 Raw 数据兼容性

新合并不会重写历史 HDF5。已抽查现有 0914/0917 批次，结论如下：

| 历史任务 | 实际相机合同 | 在新代码中的用法 |
|---|---|---|
| PickPlace | `head` | 可回放、验证，并使用 head-only adapter；不能伪造缺失的腕部视角 |
| Card | `head + right_wrist` | 可按历史双相机合同回放和转换；不能选择缺失的 `left_wrist` |
| Bulb / RAM | 三相机 | Raw 、回放及已注册的 EgoSteer/π0.5 转换继续可用 |
| Vase | 三相机 | Raw、回放及 π0.5 共享右臂+右手 adapter 可用；EgoSteer/EgoTouch 待实现 |
| USB | `head + right_wrist` | 文件结构仍可读；新版改了高位悬停/对准/接近运动分布，须与新 USB 批次分版管理，不要静默混合 |

因此“可用”分为两层：Raw 可读/可回放不等于已有对应模型的语义转换 adapter；
转换时必须严格选择该批数据真实存在的相机。

## Bulb/RAM 完整转换

对每个任务分别选择输出格式并先执行 `--dry-run`。例如 RAM 的 EgoSteer 转换：

```bash
pixi run convert-dataset -- \
  --input-dir /path/to/ram_raw \
  --output-dir /path/to/ram_egosteer \
  --format egosteer --task install-ram \
  --cameras head left_wrist right_wrist \
  --workers 4 --expected-episodes 200 \
  --staging-root /path/to/staging
```

正式输出目录必须不存在。需要独立复核源 HDF5 摘要时增加 `--verify-source-hash`；
采集续跑和高速 π0.5 转换续跑会重新核对已发布产物的内容哈希。

## 性能、校验与恢复

- PickPlace、Poker、Bulb、RAM、Vase、Whiteboard 和 Sponge 的 π0.5 `--workers N`
  表示同时转换 N 条 episode；每个 worker 使用一个 FFmpeg 编码线程。
- 这些高速 π0.5 adapter 直接把 HDF5 RGB 管道送入 H.264 MP4，不生成逐帧临时
  PNG；输出仍是 OpenPI 锁定环境可读取的标准 LeRobot v2.1 数据集。
- `--staging-root` 存放 episode checkpoint 和组装目录。每条 episode 的 Parquet、
  MP4、统计量和 `done.json` 原子提交，全部完成并通过官方 LeRobot reader 抽查后才
  发布正式输出。
- 中断后用相同参数加 `--resume` 重跑；已提交且身份、大小与转换计划一致的 episode
  不会重新编码。计划不一致或已提交 checkpoint 损坏时明确失败，不静默混用。
- 默认信任采集 sidecar 的 HDF5 SHA-256，避免为校验再次完整读取大文件；只有指定
  `--verify-source-hash` 才并行重算源摘要。结构、时钟、动作对齐、FFprobe 视频合同、
  metadata 总数和官方 reader 解码检查仍然保留。
- EgoSteer 后端同样直接从 HDF5 编码，不需要临时 PNG。EgoTouch 支持
  episode/camera 级 `--resume`；续跑要求已有转换 manifest 中的源根目录、episode
  集合与当前输入完全一致，并重新核对已有源 HDF5 的 SHA-256。旧版缺少这些字段的
  EgoTouch 输出应使用新的输出目录重新转换。
- USB 的独立 π0.5 adapter 仍沿用既有 image-backed v2.1 合同；另有模型中立的
  LeRobot v3 高速转换器，二者不要混为同一种训练格式。
- 正式输出目录必须不存在，避免覆盖已经发布的数据集。

## 新 adapter 接入

新增任务或模型格式时：

1. 在 `pipeline/conversion.py` 注册 `(task, format)` 和允许的相机集合；
2. 实现任务状态到模型 observation/action 的语义映射；
3. 验证帧时钟、动作对齐、相机顺序、训练/验证划分和来源清单；
4. 通过 staging 写入、完整校验和原子发布，禁止直接覆盖正式目录；
5. 增加 `--dry-run`、缺失相机、不支持组合和实际小样本转换测试。

共享 CLI 只负责稳定接口，不能代替任务语义审核。
