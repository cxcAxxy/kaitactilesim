# 统一数据转换

统一入口是 `scripts/convert/convert.py`，Pixi 快捷命令为：

```bash
pixi run convert-dataset -- --help
pixi run convert-dataset -- --list-support
```

调用者显式指定 Raw 输入、输出目录、模型数据格式、任务和相机。统一入口负责发现、合同检查和 adapter 分发；它不会根据文件名猜测任务 action 的含义。

## 基本参数

```text
--input-dir PATH
--output-dir PATH
--format egosteer|pi05|egotouch
--task auto|pick-place|poker-draw|usb-insert|bulb-screw|vase-wipe|install-ram|whiteboard-wipe
--cameras head [left_wrist] [right_wrist]
--workers N
--expected-episodes N
--staging-root PATH
--verify-source-hash
--resume
--dry-run
```

相机默认只有 `head`；显式选择时必须包含 `head`，顺序会规范为
`head, left_wrist, right_wrist` 的共享顺序。Raw 可以采三相机，但转换时只读取
`--cameras` 指定的子集；未选相机不会进入模型数据集。

`--expected-episodes 0` 接受实际发现数量；正式批次建议显式给出预期数量，避免把
路径写错后仍转换一个不完整批次。

## 任务识别

- `--task auto` 从 HDF5 的 `metadata_json.scene` 读取任务，要求输入根目录下只有一种任务。
- 显式 `--task` 会从混合 Raw 根目录中只选择该任务。
- 没有任务元数据的分析 HDF5 会被忽略。
- adapter 不支持某个 `(task, format)` 时明确失败，不产生猜测结果。
- `--list-support` 从代码里的 adapter 注册表直接输出当前支持能力，文档表格不是
  调度依据。

目录发现是递归的；但现有 EgoSteer/π0.5 后端仍保留各自原始批次的内部布局检查。也就是说，输入和输出根路径可以自由指定，但历史后端要求根目录内的 episode/sidecar/summary 结构符合对应格式。

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
`task_collection_v2` 嵌套目录。统一批次只选择 `summary.json` 中已成功、已发布的
episode；外层采集编号会映射为 LeRobot 来源编号，原始 HDF5 不会改名或改写。

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

| 任务 | EgoSteer | π0.5 | EgoTouch |
|---|---|---|---|
| PickPlace | head | head | 待接入 |
| Card | 包含 head 的任意 Raw 相机子集 | head + right_wrist | 包含 head 的任意子集 |
| USB | 包含 head 的任意 Raw 相机子集 | head + right_wrist | 包含 head 的任意子集 |
| Bulb | 包含 head 的任意 Raw 相机子集 | 包含 head 的任意 Raw 相机子集 | 待接入 |
| RAM | 包含 head 的任意 Raw 相机子集 | 包含 head 的任意 Raw 相机子集 | 待接入 |
| Vase | 待接入 | 待接入 | 待接入 |
| Whiteboard | 包含 head 的任意 Raw 相机子集 | 包含 head 的任意 Raw 相机子集 | 包含 head 的任意子集 |

Bulb/RAM 的 EgoSteer 与 π0.5 adapter 已冻结并注册：EgoSteer 使用归档的双手腕/
十指 taskspace 形成 48 维下一帧动作；π0.5 使用右臂 7 关节与右手 20 关节形成
27 维绝对关节状态/动作。两者只接受成功 sidecar，并检查 30 Hz 相机同步。
Bulb、RAM、Vase 的 EgoTouch 以及 Vase 的 EgoSteer/π0.5 仍须完成各自的模型数据合同，
不能仅凭 Raw 结构宣称可训练。Whiteboard 使用共享右臂+右手动作合同，三种
输出格式均已在注册表中显式接入。

## 历史 Raw 数据兼容性

新合并不会重写历史 HDF5。已抽查现有 0914/0917 批次，结论如下：

| 历史任务 | 实际相机合同 | 在新代码中的用法 |
|---|---|---|
| PickPlace | `head` | 可回放、验证，并使用 head-only adapter；不能伪造缺失的腕部视角 |
| Card | `head + right_wrist` | 可按历史双相机合同回放和转换；不能选择缺失的 `left_wrist` |
| Bulb / RAM | 三相机 | Raw 、回放及已注册的 EgoSteer/π0.5 转换继续可用 |
| Vase | 三相机 | Raw 、验证和回放可用；三种训练格式的语义 adapter 仍待实现 |
| USB | `head + right_wrist` | 文件结构仍可读；新版改了高位悬停/对准/接近运动分布，须与新 USB 批次分版管理，不要静默混合 |

因此“可用”分为两层：Raw 可读/可回放不等于已有对应模型的语义转换 adapter；
转换时必须严格选择该批数据真实存在的相机。

## Bulb/RAM 0917_200 完整转换

默认转换 `head + left_wrist + right_wrist`，200 条源数据先在 CPFS staging 中完成，
验证后原子发布到各任务的 `egosteer/0917_200` 与 `pi05/0917_200`。四项任务默认
串行运行，避免同时读取约四份三相机数据而拖慢 NAS：

```bash
cd /cpfs_infra/user/chenxianchi/code/sim_code_merged_20260917
bash scripts/convert/convert_bulb_ram_0917_200.sh
```

后台运行并保存总日志：

```bash
cd /cpfs_infra/user/chenxianchi/code/sim_code_merged_20260917
mkdir -p /cpfs_infra/user/chenxianchi/conversion_logs/0917_200
nohup bash scripts/convert/convert_bulb_ram_0917_200.sh \
  > /cpfs_infra/user/chenxianchi/conversion_logs/0917_200/all.log 2>&1 &
echo $!
```

查看进度：

```bash
tail -f /cpfs_infra/user/chenxianchi/conversion_logs/0917_200/all.log
```

正式输出目录必须不存在。转换器默认信任采集时已写入 sidecar 的 SHA-256，避免重复
读取全部 HDF5；若需要独立复核源摘要，再在单项命令中增加 `--verify-source-hash`。

## 性能、校验与恢复

- `--workers` 控制可并行的 episode/编码工作数。
- `--staging-root` 可把临时工作放到较快的本地或 CPFS 目录，再原子发布到目标路径。
- 默认信任采集 sidecar 的源摘要，只有指定 `--verify-source-hash` 才重新读取全部 HDF5 计算哈希。
- EgoSteer 后端直接从 HDF5 编码，不需要临时 PNG。
- 现有 π0.5/LeRobot 后端仍使用 image-writer 临时 PNG；统一入口没有伪装成已消除这一开销。
- EgoTouch 支持 episode/camera 级 `--resume`；只有通过现有产物身份与完整性检查的
  目录才会跳过，并持续原子更新 batch manifest。
- EgoSteer/π0.5 当前使用完整 staging 后原子发布，不支持中途 episode 级续转；对它们使用 `--resume` 会明确报错。
- 正式输出目录必须不存在，避免覆盖已经发布的数据集。

## 新 adapter 接入

新增任务或模型格式时：

1. 在 `pipeline/conversion.py` 注册 `(task, format)` 和允许的相机集合；
2. 实现任务状态到模型 observation/action 的语义映射；
3. 验证帧时钟、动作对齐、相机顺序、训练/验证划分和来源清单；
4. 通过 staging 写入、完整校验和原子发布，禁止直接覆盖正式目录；
5. 增加 `--dry-run`、缺失相机、不支持组合和实际小样本转换测试。

共享 CLI 只负责稳定接口，不能代替任务语义审核。
