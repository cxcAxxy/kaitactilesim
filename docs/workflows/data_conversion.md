# 统一数据转换

统一入口是 `scripts/convert/convert.py`，Pixi 快捷命令为：

```bash
cd /cpfs_infra/user/chenxianchi/code/kaitactilesim_unified
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is convert-dataset -- --help
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is convert-dataset -- --list-support
```

只想复用一套采集、转换、评估命令时，先看[统一命令速查](pipeline_quickstart.md)。

示例用 `pixi run --as-is` 复用已安装环境，不自动补依赖。本节点 unified 环境曾以
`--offline --skip-with-deps pyarrow` 恢复，当前缺 pyarrow；真正转换必须先补齐所选后端的
全部依赖或配置完整的 OpenPI/LeRobot Python，不能因 `--help` 可运行就认定转换环境完整。
详见[环境与工作目录](pipeline_quickstart.md#环境与工作目录)。任务数据的成功筛选口径见
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

成功转换额外生成 `pipeline_conversion.json`，记录统一计划、runtime 配置指纹、
选中的 Raw 与后端产物身份。任务和模型专属的原始 manifest 保留为转换语义的依据。
`--list-support` 还会说明各 adapter 的运行环境和 workers 含义；workers 在
episode 转换、指纹计算和 image writer 中可能承担不同工作。

固定相机合同的 adapter 会自动选择其相机集合，其余默认只有 `head`；
显式选择时必须包含 `head`，顺序会规范为
`head, left_wrist, right_wrist` 的共享顺序。Raw 可以采三相机，但转换时只读取
`--cameras` 指定的子集；未选相机不会进入模型数据集。

`--format lerobot-v3` 接入模型中立的右臂/右手 LeRobot v3 转换器，支持全部
十三个采集任务，固定使用 `head` 与 `right_wrist`。它和 `--format pi05` 的任务专属
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
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is convert-dataset -- \
  --input-dir /path/to/card_raw \
  --output-dir /path/to/card_egosteer \
  --format egosteer --task poker-draw \
  --cameras head left_wrist right_wrist \
  --workers 4
```

USB 转现有双相机 π0.5 合同：

```bash
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is convert-dataset -- \
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
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is convert-dataset -- \
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
HF_HOME=/path/to/writable/huggingface_cache \
  /cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is convert-dataset -- \
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

USB 等十三个任务转换为模型中立 LeRobot v3（现有 Pixi 环境未安装 LeRobot，故显式选择
安装了 LeRobot 0.4.x 的 Python）：

```bash
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is convert-dataset -- \
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
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is convert-dataset -- \
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
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is convert-dataset -- \
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

上表是当前注册能力的说明，实际分发仍以 `--list-support` 为准。五个变体保留独立
task、mechanics version、场景指纹和 instruction，不能用旧任务数据重标替代。
新 `bulb-screw-inplace` 的 π0.5 `task-default` 必须选择
`recorded-controller-target-v1`，并在独立新目录转换。当前入口位姿不随 seed 随机化，
转换不会把固定初态重复数据变成泛化数据；其物理完成与 home 收尾也不能合并成同一个标签。

Bulb/RAM 的 EgoSteer 与 π0.5 adapter 已冻结并注册：EgoSteer 使用归档的双手腕/
十指 taskspace 形成 48 维下一帧动作；π0.5 使用右臂 7 关节与右手 20 关节形成
27 维绝对关节状态/动作。观测是实测关节位置；新转换的 `bulb-screw` 与 `bulb-screw-inplace` 动作必须使用
`recorded-controller-target-v1`，即在同一个 non-future 源状态索引读取 7 维手臂和
20 维手部控制器目标。`--action-contract task-default` 对这两个任务选择该合同，对其他
共享任务暂时保留 `legacy-measured-arm-hybrid-v1`。旧 Bulb 输出中的 hybrid 合同把
手臂实测 qpos 当成动作、手部却使用控制目标，只可作为显式兼容模式，不能用于新的
Bulb 训练。转换 manifest 会记录完整合同，恢复和合并时拒绝混用。两类 adapter 均只
接受成功 sidecar，并检查 30 Hz 相机同步。
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
| Bulb / RAM | 三相机 | Raw、回放及已注册的 EgoSteer/π0.5 转换继续可用；旧 Bulb mechanics 仅作同版本/诊断用途，不能冒充当前 v15 正式训练数据 |
| Vase | 三相机 | Raw、回放及 π0.5 共享右臂+右手 adapter 可用；EgoSteer/EgoTouch 待实现 |
| USB | `head + right_wrist` | 文件结构仍可读；新版改了高位悬停/对准/接近运动分布，须与新 USB 批次分版管理，不要静默混合 |

因此“可用”分为两层：Raw 可读/可回放不等于已有对应模型的语义转换 adapter；
转换时必须严格选择该批数据真实存在的相机。

## Bulb/RAM 完整转换

对每个任务分别选择输出格式并先执行 `--dry-run`。例如 RAM 的 EgoSteer 转换：

```bash
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is convert-dataset -- \
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

## FTP-1：九任务三轴触觉训练数据

FTP-1 使用独立入口 `scripts/convert/convert_kaihand_ftp1_batch.py`，不是给上面的
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

### 完整命令

先做全部九任务、每任务 200 条成功示范的只读元数据检查。检查相机时钟、状态索引、
触觉形状/单位、场景身份及已有输出的恢复合同；不读取 RGB 像素或完整触觉载荷，
不重新计算源文件 SHA，不创建转换结果：

```bash
cd /cpfs_infra/user/chenxianchi/code/kaitactilesim_unified

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
