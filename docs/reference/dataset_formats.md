# 模型数据格式与来源合同

本页维护模型数据的字段、相机、时钟、动作监督和来源语义。转换参数、adapter 分发、
相机选择、发布和恢复统一见[数据转换工作流](../workflows/data_conversion.md)；
Raw 采集合同见[数据采集](../workflows/data_collection.md)。格式支持、模型接口与可用权重分别核对。

| 格式 | 合同入口 |
|---|---|
| π0.5 训练数据 | [LeRobot v2.1 与任务动作合同](#pi05) |
| EgoSteer | [任务空间合同](#egosteer) |
| EgoTouch | [Raw 模态转换](#egotouch) |
| 模型中立 LeRobot v3 | [字段、时钟与动作](#lerobot-v3) |
| FTP-1 数值触觉 Zarr | [字段、H32 与实测轨迹监督](#ftp1) |

## 共同的来源要求

每个任务及变体保留自己的 task、instruction、mechanics version、场景指纹和完成协议。
不能改名、改提示词或重标成功标签，把旧任务的数据变成新任务训练集。
`bulb-screw-inplace` 的入口初态当前固定；不同 seed 是重复性身份，不表示初态泛化。
物理旋入成功和回 home 收尾也不能合并成同一个标签。

统一采集输入只选择已发布成功 episode，来源编号使用外层采集身份。
例如 PickPlace、Card 的隔离 recorder 可重复生成内部编号 0，π0.5 wrapper 使用
`summary.json` 的外层编号避免冲突，不改名、复制或改写源 HDF5。
历史批次只有实际存在的视角，不能补造缺失相机；不同物理、轨迹或动作版本分版管理。

任务成功标签按[成功判据](../tasks/success_criteria.md)保留，
[阶段诊断](../workflows/staged_evaluation.md)不能替代原生成功。
实际模型动作与 observation 必须按所选格式核对，不能因同为 27 维就互换语义。
不同触觉 provider 的单位和标定也必须分别核对，不能将 probe 深度代理当作求解器牛顿力。

<a id="pi05"></a>
## π0.5：LeRobot v2.1 与任务动作

`--format pi05` 生成任务专属训练数据；它与下面的模型中立 LeRobot v3 分别维护。
除 USB 独立 image-backed 后端外，本地高速 wrapper 输出 LeRobot v2.1 video feature。
状态、动作、时间戳与来源仍由对应的 OpenPI 冻结任务转换器计算，加速不改变动作语义。

Bulb、RAM、Vase、Whiteboard、Sponge 的共享关节表示为右臂 7 关节加右手 20 关节，
共 27 维绝对关节状态/动作；观测来自实测关节位置。PickPlace、Card、USB 各自的
冻结合同不得直接套用这一共享表示，具体相机与 adapter 以
[转换能力表](../workflows/data_conversion.md#当前支持矩阵)为准。

新转换的 `bulb-screw` 与 `bulb-screw-inplace` 必须使用
`recorded-controller-target-v1`：在同一个 non-future 源状态索引读取右臂 7 维和
右手 20 维控制器目标。下游 OpenPI 转换器的 `--action-contract task-default`
为这两项选择该合同，其他共享任务暂时保留 `legacy-measured-arm-hybrid-v1`。
统一转换入口沿用后端默认值，没有单独的 `--action-contract` 参数。

旧 Bulb hybrid 合同把手臂实测 qpos 当动作、手部却使用控制目标，只可显式兼容，
不能用于新的 Bulb 训练。转换 manifest 保存合同，恢复或合并时拒绝混用。
新原位旋入数据须绑定独立任务与场景身份，不得用旧拿取旋入数据替代。

Vase 保留真实 500 Hz 源状态时钟，在 30 Hz 相机时刻插值状态，并以下一相机帧形成动作；
不在源 Raw 中就地降采样。Sponge 的 30 Hz 帧落在 100 Hz 控制步上，转换器按实际
控制步检查每帧时间，保留源时间戳与编号，不把其他任务的时钟规则套到它上面。

<a id="egosteer"></a>
## EgoSteer：任务空间合同

Bulb/RAM 的 EgoSteer adapter 使用归档的双手腕与十指 taskspace，形成 48 维下一帧动作；
其 π0.5 版本则采用上面的 27 维右臂/右手关节表示，二者不能互换。
adapter 只接受成功 sidecar，并检查 30 Hz 相机同步。
其他任务的 observation/action 映射仍由对应的任务 adapter 定义。

<a id="egotouch"></a>
## EgoTouch：Raw 模态转换

EgoTouch 按选定相机导出 episode，保存其触觉 NPZ、session、selector 与来源身份。
任务映射由 `shared.tict_export` 定义，不从文件名猜动作语义。
支持的任务与相机集合见[转换能力表](../workflows/data_conversion.md#当前支持矩阵)；
Raw 可读不代表已经存在对应任务的训练转换合同。

<a id="lerobot-v3"></a>
## 模型中立 LeRobot v3

该格式覆盖十三个注册任务，固定读取 `head + right_wrist`，不是 π0.5 训练集的版本升级。
它要求安装 LeRobot 0.4.x 的 Python；环境准备见[运行环境](../workflows/environment.md)。
统一采集输入须使用 `target-successes` 且已达目标，暂不接受 attempts-only 批次；
受支持的历史/合并输入按其来源合同识别，混合根目录须显式选择 `--task`。

下面保留 `--repo-id` 与外部 Python 的格式专属示例；通用执行和续跑规则见
[转换与恢复](../workflows/data_conversion.md#性能校验与恢复)。

```bash
cd /cpfs_infra/user/chenxianchi/code/kaitactilesim
PYTHONPATH=src pixi run --as-is convert-dataset -- \
  --input-dir /path/to/usb_raw_collection \
  --output-dir /path/to/usb_lerobot_v3 \
  --task usb-insert --format lerobot-v3 \
  --expected-episodes 200 --workers 4 \
  --repo-id kaihand/usb_insert_example \
  --lerobot-v3-python /path/to/lerobot-env/bin/python \
  --dry-run
```

输出包含 head/right-wrist MP4 与 Parquet；RGB 从 HDF5 直接送入 FFmpeg，
不生成逐帧临时 PNG。首次处理大批次前先以少量数据实转，再用官方 LeRobot reader 抽查。
reader 缓存和外部 Python 的运行参数见[转换参数](../workflows/data_conversion.md#基本参数)。

<a id="lerobot-v3-clock-actions"></a>
### 时钟与动作

- 数据集 FPS 为 30，按 Raw 中真实的 head/right-wrist 相机时刻取样；两路必须同步。
- 状态、腕部力和触觉从记录的源状态时钟取不晚于相机帧的样本。USB 等任务有
  500 Hz 源状态；Sponge 为 100 Hz 控制步与 30 Hz 相机 deadline 的组合，不能
  假设所有任务都是 500 Hz。
- `action` 是下一保留相机时刻的右腕世界位姿
  `[x,y,z,qx,qy,qz,qw]` 加右手 20 个驱动关节目标，共 27 维；最后一帧无下一
  waypoint，因此不作为训练样本。
- `auxiliary.action.right_joint_target` 另保存同一动作时刻的右臂 7 关节目标
  加右手 20 关节目标。两种动作语义不能互换。

右腕姿态四元数是绝对 XYZW。需要相对末端动作时必须计算 SE(3) 差，
不能逐分量做 `action - observation.state`。

Sponge 视频按名义 30 fps 编码，真实的 30/40 ms 控制步间隔保存在 provenance 中；
柔体顶点与盘子支撑力保留为任务诊断，不作为另一种固定频率的测量。

### 主要字段与来源

- `observation.images.head`、`observation.images.right_wrist`
- `observation.state`：当前实际右腕位姿与右手 20 关节
- `observation.state.right_joint_position`：当前实际右臂 7 关节与右手 20 关节
- 拆分的右臂/右手位置、速度和力；实际腕/指尖位姿；腕部局部与世界系 6D wrench
- 右手五指 `(5,7,5,3)` taxel 力、接触与聚合力
- `action` 与 `auxiliary.action.right_joint_target`；物体/阶段等任务诊断字段
- 原始 episode、帧、状态索引和时间戳来源

`meta/kaihand_schema.json` 是坐标系、四元数、动作、触觉和阶段语义的机器可读
权威；`meta/kaihand_source_episodes.jsonl` 可追溯到不可变 Raw HDF5 及其 SHA-256。
源数据中的左侧机器人、全频物理状态、稀疏接触事件和控制诊断不会伪装成
30 Hz 训练字段。物体状态、任务阶段与成功标签属于仿真特权信息；部署时不可得
的信号不应直接作为模型输入。

Vase 的受支持合并批次可由 `merge_manifest.json` 识别，不要求伪造普通
`collection.json`。参考样本的相机、触觉与力标定若来自不同 provider，
比较力的大小前须先核对两者标定。

<a id="ftp1"></a>
## FTP-1：数值触觉 Zarr

FTP-1 使用独立转换器
[convert_kaihand_raw_to_ftp1_zarr.py](../../scripts/workcell/convert_kaihand_raw_to_ftp1_zarr.py)，
不传 `convert-dataset --format ftp1`。支持任务以该转换器的 `TASK_CONTRACTS` 与
`--help` 为准；专属预检、分片和恢复命令见
[独立转换入口](../workflows/data_conversion.md#ftp1-conversion)。

输入是已发布 Raw 批次，任务成功、相机、关节顺序、触觉方向和物理身份必须匹配。
旧 USB 拔出训练源采用“放回并释放”协议，不能仅修改 instruction 就变成
“拔出并手持清离两秒”。固定九任务来源与归一化资产配方见
[历史流水线批次](../archive/pipeline_batches_202609_202610.md)。

输出包含 Zarr shards、`conversion_manifest.json` 与 `dataset_config.json`；
manifest 记录源 episode、摘要策略、时钟、触觉与动作监督合同。

### FTP-1 字段与时钟

| Zarr 字段 | 来源或含义 |
|---|---|
| `camera_ego_rgb`、`right_wrist_camera_rgb` | Raw 的 head、right_wrist RGB |
| `camera_ego_pose`、`right_wrist_pose` | 记录的相机与实际右腕位姿 |
| `right_arm_joints`、`right_hand_joints` | 实测右臂 7 关节、右手 20 驱动关节 |
| `right_hand_joints_idx` | KaiHand 到 FTP-1 FAAS 的关节映射 |
| `right_tactile_data_fingers` | 右手五指 `[T,5,7,5,3]` 数值触觉阵列 |

触觉三通道为法向载荷、沿阵列列方向和行方向的有符号切向分量，单位 `N/taxel`。
它们是 MuJoCo 求解器接触力的空间分配估计，不是 Genesis 深度代理，也不是独立
柔性皮肤传感器的实测真值。`matrix` 触觉编码器接收这些数值阵列，不接收伪彩色图片。

head 与 right_wrist 必须具有相同的名义 30 Hz 时钟和来源索引。
状态与触觉按记录的最近非未来 `state_index` 选取，触觉可比积分后的状态时间早一个
物理步。Raw 的状态与物理频率以源文件及任务合同为准，不统一改为 500 Hz；
九任务补转中各来源的实际频率保存在对应[批次记录](../archive/pipeline_batches_202609_202610.md)。

### FTP-1 动作与训练配置

Zarr 不预存动作 chunk；原生 FTP-1 loader 从实测腕位姿和关节轨迹构造监督。
非 Card 任务的 manifest 明确使用
`ftp1-native-future-measured-state-trajectory-v1`，并标记
`controller_targets_used=false`。Raw 即使保存控制器目标，也不将其用于这份导出的动作监督。

现有训练配方为 H32、动作步长 1、关闭历史、相对 pose、绝对 proprioception joints、
`action_joint_rep=mix`、z-score，并开启触觉。H32 表示当前帧加 31 个后续 30 Hz 状态，
不是 32 Hz，也不是 32 帧触觉历史。`mix` 的实测轨迹监督不等于 π0.5 灯泡任务的
`recorded-controller-target-v1`，两种数据合同不可互换。

归一化、触觉配置与训练必须匹配同一份数据配置及转换清单。转换和归一化完成不代表
已经完成训练、在线部署或闭环验收；是否有模型入口与权重，应按部署能力表和实际训练资产核对。

### FTP-1 格式依据

- [FTP-1 数据准备说明](../../../ftp1-policy/README_zh.md#data-preparation-zh)
- [数据加载器](../../../ftp1-policy/src/openpi/dataset_zarr.py)
- [触觉编码器](../../../ftp1-policy/src/openpi/models_pytorch/ftp1_blocks.py)
- [归一化示例](../../../ftp1-policy/scripts_exp_zarr/univtac/compute_norm_stats_univtac_example.sh)
- [KaiHand 训练入口](../../../ftp1-policy/scripts/train_kaihand_ftp1.sh)

以上 FTP-1 链接指向相邻的 `ftp1-policy` 项目；缺少该项目时先配置实际模型环境。
