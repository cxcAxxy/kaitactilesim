# LeRobot v3 转换合同

这是模型中立的 LeRobot v3 数据集，不是 `--format pi05` 生成的 π0.5 训练集。
统一入口支持 PickPlace、Card、USB、Bulb、RAM、Vase、Whiteboard 和 Sponge；
加上 `usb-unplug`、`bulb-unscrew`、`whiteboard-wipe-horizontal`、
`whiteboard-wipe-horizontal-line`、`bulb-screw-inplace` 五个变体，共十三个独立任务。
可用组合以 `pixi run --as-is convert-dataset -- --list-support` 为准。
一般参数先看[统一数据转换](data_conversion.md)，三阶段命令见[速查](pipeline_quickstart.md)。

## 运行

从 `/cpfs_infra/user/chenxianchi/code/kaitactilesim_unified` 执行。示例使用 `pixi run --as-is`
避免触发本节点环境自动安装；本节点 offline 安装跳过了 pyarrow，不能据此直接做正式转换，
需要补全实际后端依赖或配置完整 LeRobot 环境。详见[环境与工作目录](pipeline_quickstart.md#环境与工作目录)。

使用安装 LeRobot 0.4.x 的 Python 环境。输入可以是统一采集批次或转换器支持的历史、
合并批次；混合任务根目录必须显式传 `--task`，只转换该任务已发布的成功 episode。
USB 示例，RAM 或 Sponge 只需修改任务名及路径：

```bash
cd /cpfs_infra/user/chenxianchi/code/kaitactilesim_unified
pixi run --as-is convert-dataset -- \
  --input-dir /path/to/usb_raw \
  --output-dir /path/to/usb_lerobot_v3 \
  --task usb-insert --format lerobot-v3 \
  --expected-episodes 200 --workers 4 \
  --repo-id kaihand/usb_insert_example \
  --lerobot-v3-python /path/to/lerobot-env/bin/python \
  --dry-run
```

确认计划后移除 `--dry-run`。输出目录应为新路径；中断后使用相同参数加
`--resume`。可用 `--work-dir /path/to/checkpoints` 把断点放在持久磁盘上，
最好与最终输出处于同一文件系统。需要重新核对采集 sidecar 中的 HDF5 SHA-256
时加 `--verify-source-hash`。首次处理大批次前可先用少量数据实转并用
LeRobot reader 抽查。

十三个任务的物理成功筛选见[成功判据总表](../tasks/success_criteria.md)，
[分阶段诊断](staged_evaluation.md)不替代原生成功标签。`bulb-screw-inplace` 的初态固定，
不同 seed 只是重复性身份；不可把旧 `bulb-screw` Raw 改任务名后混入新任务。
模型中立 LeRobot v3 与 π0.5 的 recorded-controller-target 训练合同也不能互相冒充。

后端逐 episode 编码 head/right-wrist MP4 与 Parquet，完成后原子提交断点；
RGB 从 HDF5 直接送入 FFmpeg，不生成逐帧临时 PNG。续跑会核对已提交 episode
的大小和哈希，正常完成后再发布整个数据集。工作目录损坏或身份不一致时应
使用新的工作目录，不混用旧断点。

## 时钟与动作

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

## 主要字段与来源

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
