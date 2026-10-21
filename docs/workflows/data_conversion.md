# 统一数据转换

本页维护转换 CLI、Raw 发现、adapter 分发、相机选择、发布与恢复。
模型字段、相机时钟、动作监督及来源合同统一在[数据格式合同](../reference/dataset_formats.md)维护；
这里的支持矩阵描述命令可分发的组合，不代表已有匹配的训练权重。

统一入口是 `scripts/convert/convert.py`，Pixi 快捷命令为：

```bash
cd /cpfs_infra/user/chenxianchi/code/kaitactilesim
PYTHONPATH=src /cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is convert-dataset -- --help
PYTHONPATH=src /cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is convert-dataset -- --list-support
```

只想复用一套采集、转换、评估命令时，先看[统一命令速查](pipeline_quickstart.md)。

环境安装、依赖限制和源码导入检查统一见[环境与工作目录](environment.md)。
示例用 `--as-is` 复用已安装环境，并用 `PYTHONPATH=src` 指向当前仓库源码；
实际转换前仍须补齐所选后端依赖，或配置完整的 OpenPI/LeRobot Python。
任务数据的成功筛选口径见
[成功判据](../tasks/success_criteria.md)，阶段诊断与原生成功的区别见[分阶段评估](staged_evaluation.md)。

调用者显式指定 Raw 输入、输出目录、模型数据格式、任务和相机。统一入口负责发现、合同检查和 adapter 分发；它不会根据文件名猜测任务 action 的含义。

## 基本参数

```text
--input-dir PATH
--output-dir PATH
--format egosteer|pi05|egotouch|lerobot-v3
--task auto|pick-place|poker-draw|usb-insert|bulb-screw|vase-wipe|install-ram|whiteboard-wipe|sponge-grasp
# 也支持 usb-unplug、bulb-unscrew、whiteboard-wipe-horizontal、whiteboard-wipe-horizontal-line、bulb-screw-inplace
--cameras head [left_wrist] [right_wrist]
--workers N
--expected-episodes N
--staging-root PATH
--work-dir PATH                 # 仅 LeRobot v3
--repo-id ID                   # 仅 LeRobot v3
--lerobot-v3-python PATH        # 当前 Python 未安装 LeRobot 时必填
--runtime-config FILE          # TOML/JSON 命名运行环境
--runtime NAME                 # 选择配置内的环境
--verify-source-hash
--resume
--dry-run
```

命名运行环境可以集中保存 Python、模型项目根目录和环境变量，示例见
[runtimes.example.toml](../../configs/runtimes.example.toml)。路径相对配置文件所在目录
解析，环境变量值按字面使用。转换与部署共用同一配置格式；原有
`--openpi-root`、`--pi05-python`、`--lerobot-v3-python` 参数继续可用。
先用 `--dry-run` 查看解析后的实际命令，参数冲突会明确报错。

TOML 与 JSON 均可使用；`default_runtime` 允许省略 `--runtime`。配置中的 `python`
与 `project_root` 相对配置文件解析，不随调用目录变化。新旧参数指定冲突路径时会报错。
计划阶段可解析规划机器上尚不存在的路径，执行时会验证实际路径；疑似密钥的环境值在
输出与保存的元数据中脱敏。

外部 Python 子进程的 `PYTHONPATH` 会优先加入本仓库 `src`；π0.5 还加入所选
OpenPI 的 `src` 与 `packages/openpi-client/src`。dry-run 的
`runtime.pythonpath_prepend` 显示这些实际来源，不输出其余进程环境。

dry-run 计划使用 `schema=kaihand_conversion_plan_v1`、`schema_version=1`、
`stage=convert`，记录 runtime、workers 语义及从 HDF5 读取的动作与相机合同；
历史 Raw 未声明 `runtime_contract_json` 时记录 `unknown_not_declared`，不推测补齐。
成功转换后原子写入 `pipeline_conversion.json`，保存统一计划、runtime 配置指纹、
选中的 Raw 身份及已识别后端 manifest 的 SHA-256 引用。失败时不发布这份清单；
任务和模型专属的原始 manifest 仍是转换语义的依据。
`--list-support` 还会说明各 adapter 的运行环境和 workers 含义；workers 在
episode 转换、指纹计算和 image writer 中可能承担不同工作。

固定相机合同的 adapter 会自动选择其相机集合，其余默认只有 `head`；
显式选择时必须包含 `head`，顺序会规范为
`head, left_wrist, right_wrist` 的共享顺序。Raw 可以采三相机，但转换时只读取
`--cameras` 指定的子集；未选相机不会进入模型数据集。

`--format lerobot-v3` 支持十三个注册任务，固定选择 `head + right_wrist`。
它接受已达成功数目标的统一采集批次及后端支持的历史/合并批次，暂不接受 attempts-only
统一批次。字段、时钟与专属参数示例见[LeRobot v3 合同](../reference/dataset_formats.md#lerobot-v3)。

LeRobot v3 需要安装 LeRobot 0.4.x 的 Python，可用 `--lerobot-v3-python` 或命名环境选择。
官方 reader 使用 Hugging Face 缓存；默认目录不可写时，在命令前设置
`HF_HOME=/path/to/writable/huggingface_cache`。

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
PYTHONPATH=src /cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is convert-dataset -- \
  --input-dir /path/to/card_raw \
  --output-dir /path/to/card_egosteer \
  --format egosteer --task poker-draw \
  --cameras head left_wrist right_wrist \
  --workers 4
```

USB 转现有双相机 π0.5 合同：

```bash
PYTHONPATH=src /cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is convert-dataset -- \
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
PYTHONPATH=src /cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is convert-dataset -- \
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

Sponge π0.5 的源时钟约束见[格式合同](../reference/dataset_formats.md#pi05)。
同一批 Raw 可在独立输出目录转换为 LeRobot v3；双相机、外部 Python 与 `--repo-id`
的专属示例见[LeRobot v3](../reference/dataset_formats.md#lerobot-v3)。

Card/USB 首次转 EgoTouch，输出目录须不存在：

```bash
PYTHONPATH=src /cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is convert-dataset -- \
  --input-dir /path/to/raw \
  --output-dir /path/to/egotouch \
  --format egotouch --task auto \
  --cameras head left_wrist right_wrist \
  --workers 4
```

EgoTouch 中断后沿用上述参数加 `--resume`；该参数要求输出目录已经存在，
不能用于首次转换。它不会仅凭目录存在就跳过，而会检查发布审计、session、来源路径/大小、
采集 sidecar 摘要、相机名、selector 和触觉 NPZ。残缺或串用的目录会报错并保留
现场，不会混进完成 manifest。

先只检查任务、相机和 adapter，不写输出：

```bash
PYTHONPATH=src /cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is convert-dataset -- \
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
| USB unplug | 待接入 | head + right_wrist | head + right_wrist | 待接入 |
| Bulb unscrew | 待接入 | head + right_wrist | head + right_wrist | 待接入 |
| Horizontal whiteboard Z | 待接入 | head + right_wrist | head + right_wrist | 待接入 |
| Horizontal whiteboard top line | 待接入 | head + right_wrist | head + right_wrist | 待接入 |
| Bulb screw inplace | 待接入 | head + right_wrist | head + right_wrist | 待接入 |

上表是当前注册能力的说明，实际分发仍以 `--list-support` 为准。
五个变体以各自 task 身份选择 adapter；动作语义与来源协议在
[数据格式合同](../reference/dataset_formats.md)中分别定义，不能通过改任务名转换成另一种合同。

## 历史 Raw 数据兼容性

新合并不会重写历史 HDF5。已抽查现有 0914/0917 批次，结论如下：

| 历史任务 | 实际相机合同 | 在新代码中的用法 |
|---|---|---|
| PickPlace | `head` | 可回放、验证，并使用 head-only adapter；不能伪造缺失的腕部视角 |
| Card | `head + right_wrist` | 可按历史双相机合同回放和转换；不能选择缺失的 `left_wrist` |
| Bulb / RAM | 三相机 | Raw、回放及已注册的 EgoSteer/π0.5 转换继续可用；旧 Bulb mechanics 仅作同版本/诊断用途，不能冒充当前 v15 正式训练数据 |
| Vase | 三相机 | Raw、回放及 π0.5 共享右臂+右手 adapter 可用；EgoSteer/EgoTouch 待实现 |
| USB | `head + right_wrist` | 文件结构仍可读；新版改了高位悬停/对准/接近运动分布，须与新 USB 批次分版管理，不要静默混合 |

因此“可用”分为两层：Raw 可读/可回放不等于已有对应模型的语义转换 adapter；
转换时必须严格选择该批数据真实存在的相机。

## Bulb/RAM 完整转换

对每个任务分别选择输出格式并先执行 `--dry-run`。例如 RAM 的 EgoSteer 转换：

```bash
PYTHONPATH=src /cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is convert-dataset -- \
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

| Adapter | `--workers N` 的实际含义 |
|---|---|
| PickPlace EgoSteer | 后端无 worker 参数，不使用该值 |
| 其他 EgoSteer | 同时转换的 episode 进程上限 |
| EgoTouch | 同时导出的 episode × camera 进程上限 |
| LeRobot v3 | 源预检与 episode 转换进程数 |
| 高速 π0.5（USB 除外） | 源指纹/哈希线程与 episode 转换进程数；每个编码进程使用一个 FFmpeg 线程 |
| USB π0.5 | 源指纹/哈希线程与 image-writer 线程数；不启用 image-writer 进程 |

各 adapter 的实际支持仍以 `--list-support` 为准。

- 高速 π0.5 adapter（USB 除外）直接把 HDF5 RGB 管道送入 H.264 MP4，不生成逐帧临时
  PNG；字段与动作计算仍使用对应任务的冻结转换器，见[π0.5 合同](../reference/dataset_formats.md#pi05)。
- `--staging-root` 存放 episode checkpoint 和组装目录。每条 episode 的 Parquet、
  MP4、统计量和 `done.json` 原子提交，全部完成并通过官方 LeRobot reader 抽查后才
  发布正式输出。
- LeRobot v3 的工作目录在每条 episode 完成后原子提交；续跑用相同参数加 `--resume`，
  可用 `--work-dir` 指定持久断点目录，最好与最终输出在同一文件系统。
  续跑核对已提交 episode 的大小与哈希；损坏或身份不一致时使用新工作目录。
- 注册为 `resumable: true` 的高速 π0.5 adapter 中断后可沿用相同参数加 `--resume`；
  已提交且身份、大小与转换计划一致的 episode 不会重新编码。计划不一致或已提交
  checkpoint 损坏时明确失败。当前 Sponge 与 USB 的 π0.5 adapter 未开放统一
  `--resume` 参数，不应照搬该续跑命令；先查询 `--list-support`。
- 默认信任采集 sidecar 的 HDF5 SHA-256，避免为校验再次完整读取大文件；只有指定
  `--verify-source-hash` 才并行重算源摘要。结构、时钟、动作对齐、FFprobe 视频合同、
  metadata 总数和官方 reader 解码检查仍然保留。
- EgoSteer 后端同样直接从 HDF5 编码，不需要临时 PNG。EgoTouch 支持
  episode/camera 级 `--resume`；续跑要求已有转换 manifest 中的源根目录、episode
  集合与当前输入完全一致，并重新核对已有源 HDF5 的 SHA-256。旧版缺少这些字段的
  EgoTouch 输出应使用新的输出目录重新转换。
- USB π0.5 使用独立发布后端；模型中立 LeRobot v3 使用另一份转换计划和工作目录，
  不混用两者的断点。
- 正式输出目录必须不存在，避免覆盖已经发布的数据集。

## 新 adapter 接入

新增任务或模型格式时：

1. 在 `pipeline/conversion.py` 注册 `(task, format)` 和允许的相机集合；
2. 实现任务状态到模型 observation/action 的语义映射；
3. 验证帧时钟、动作对齐、相机顺序、训练/验证划分和来源清单；
4. 通过 staging 写入、完整校验和原子发布，禁止直接覆盖正式目录；
5. 增加 `--dry-run`、缺失相机、不支持组合和实际小样本转换测试。

共享 CLI 只负责稳定接口，不能代替任务语义审核。

<a id="ftp1-conversion"></a>
## FTP-1 独立转换入口

FTP-1 使用 `scripts/workcell/convert_kaihand_raw_to_ftp1_zarr.py`，不属于
`convert-dataset --format` 的四种格式。输入与字段合同见[FTP-1](../reference/dataset_formats.md#ftp1)。
先使用已配置好转换依赖的 Python 做元数据预检：

```bash
cd /cpfs_infra/user/chenxianchi/code/kaitactilesim
PYTHONPATH=src /path/to/ftp1-env/bin/python -B scripts/workcell/convert_kaihand_raw_to_ftp1_zarr.py \
  --input-dir /path/to/published_raw \
  --output-dir /path/to/new_ftp1_dataset \
  --task usb-insert --expected-episodes 200 \
  --dry-run --plan-json
```

预检不读取 RGB 像素或完整触觉载荷，不重新计算源摘要，也不写转换结果。
确认计划后移除 `--dry-run --plan-json` 才执行转换。`--workers` 是并行 Zarr shard 数，
`--episodes-per-shard` 控制每个 shard 的 episode 数。`--resume` 只复用来源及合同匹配
的已发布分片；`--verify-source-hash` 会完整读取源 HDF5 并核对 sidecar 摘要。
新批次显式选择任务与来源，不套用固定批次脚本的默认路径。

固定九任务来源、归一化资产和批次命令保存在
[历史训练与转换批次](../archive/pipeline_batches_202609_202610.md)。
