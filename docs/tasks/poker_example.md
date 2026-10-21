# 摸牌：共享配置与示例导出

本页负责摸牌场景、专家动作版本与成功 Raw 的派生展示。原生成功数值由[成功判定](success_criteria.md#poker-draw)维护，
模型专项配置见[模型专项评估](../workflows/model_specific_evaluation.md)。

`poker-draw` 从共享收臂初态连续接近牌，受力拖到桌沿、对向夹起，再抬臂转向看牌。
模型原生成功、可调终态 hold、专家压力准入及历史版本判据见
[摸牌成功判定](success_criteria.md#poker-draw)。运行前确认[环境与工作目录](../workflows/environment.md)。

采集使用[统一采集入口](../workflows/data_collection.md)；本页说明当前动作配置和成功 Raw 的离线展示导出。
2026-09-20 的动作改造过程与单 seed 验收保留在[任务历史档案](../archive/task_scene_history.md#poker-2026-09-20)。

摸牌的 `tasks/poker_draw/scene.xml` 直接包含 `shared/mjcf/robot.xml`，
机器人、双手外观、主桌、背景和相机沿用 shared；任务 config 直接导入
`shared/posture.py::ARM_HOME`。加宽矮桌、扑克牌及任务接触参数由摸牌任务管理。
矮桌使用浅灰石纹，牌背使用红色/暖金菱形格纹理；视觉资产的改变需要与采集时版本一起保留。

## 连续动作版本

当前实现同时记录多个版本，它们描述不同维度：

| 字段 | 当前值 | 描述 |
| --- | --- | --- |
| `draw_posture_version` | `tip-pad-v7` | 拖牌时弯曲手指根部、以指腹前端接触牌的姿态；拖牌角度上限按该版本验收 |
| `motion_profile_version` | `right-side-low-chatter-v4` | 中档专家的右侧接近、短腕夹牌、抬牌与看牌轨迹 |
| `contact_model_version` | `poker-compliant-contact-v3` | 中档预设的桌牌、牌与指腹柔顺接触配置 |
| `acceptance_policy` | 专家 API 默认 `strict-force-v1`；统一采集 `production-v1` 显式使用 `task-completion-v1` | 专家最终结果是否同时强制整套拖牌压力质量 |

`tip-pad-v7` 与 `right-side-low-chatter-v4` 可以同时出现在一条当前记录中。
旧 `legacy-flat-v1` 记录按旧姿态合同解释；对应角度门槛和压力准入以
[成功判定](success_criteria.md#poker-draw)及记录元数据为准。
姿态配置在 `tasks/poker_draw/config.py`，中档预设、版本字段和交接控制在 `mid_full.py`。

中档轨迹从右侧笛卡尔路径接近：最多向外避让 55 mm、抬高 35 mm，到牌上方后下降；
初始 `clear_card` 用 3 s 平滑到净空路点，再进入拖牌工作阶段。
拖牌名义速度为 12 mm/s。拇指从露出桌沿的牌正面托起，与四指对夹；手臂和手指目标逐物理步推进。
接触短暂减弱时暂停路径，保留伺服预载。双面支撑建立后稳定 0.45 s，再分别用 0.65 s、1.25 s
完成 4 mm 转移和后续 21 mm 抬升。看牌目标由实际夹持后的牌/手相对位姿计算，
使牌的 88 mm 长轴竖直、63 mm 窄边朝下，牌正面朝机器人，并选择较低的肘部逆解分支。

基础场景与中档采集的控制目标不同：

| 配置 | 拖牌每指目标 | 夹牌四指各自目标 | 拇指夹牌目标 |
| --- | ---: | ---: | ---: |
| 基础场景 `view -- --task poker-draw` | 0.35 N | 0.75 N | 3 N |
| 中档柔顺采集预设 | 0.50 N | 0.25 N | 1 N |

这些是控制目标，实际瞬态载荷由接触求解器产生。反馈中的低通只影响控制，保存的触觉不滤波。
轨迹、外观和接触姿态变化会影响数据分布；历史模型需独立闭环评估。
公开 `maximum_palm_normal_z` 包括共享初态与 `clear_card` 的转掌过程；专家成功使用的工作阶段姿态值，
以及它的适用 phase，见[成功判定](success_criteria.md#poker-draw)。

## 重新生成一条独立示例

先按[统一采集](../workflows/data_collection.md)取得并验证成功 Raw，使用尚未存在的新批次目录。
下面的导出命令读取原 Raw，在新目录生成展示视频和力曲线；**它不复制 Raw HDF5 或同名 JSON**：

```bash
pixi run --as-is export-poker-example -- \
  /path/to/successful/episode_000000_card_right.h5 \
  --output-dir /new/path/poker_review
```

导出器使用单线程软件渲染，只接受校验通过且 outcome 成功的 `poker-draw` Raw，不执行新物理任务。
输入 Raw 保留在原路径；输出目录可新建，也可使用已准备好 `raw/` 的示例目录，
但已有 `review/` 或 `curves/` 时拒绝覆盖。要交付完整独立示例，需一并保留原始 HDF5、sidecar 和采集时资产；
仅导出的展示目录不能替代 Raw 数据备份。

## 四类产物

完整示例通常按 `raw/`、`review/`、`curves/` 组织；第一项来自采集，其余三项由上述离线命令生成：

1. `raw/*.h5` 与同名 JSON：实际保存的相机 RGB、状态、动作、触觉、时钟、相机标定及 outcome。
   统一新采集默认保存 `head`、`left_wrist`、`right_wrist`；历史紧凑示例可能只有 `head`＋`right_wrist`。
2. `review/review.mp4`：头部＋右腕原始图像、右手五指法向/切向热图；展示为 10 fps，Raw 相机仍为 30 Hz。
3. `review/robot_global_short_path.mp4`：保存 qpos 的全局运动学回放，使用最邻近保存状态，
   不插值、不执行物理步进；该视角仅供展示。
4. `curves/right_hand_force_curves.png`：右手五指原始 Fn 与 |Ft|。
   Fn 为每指 taxel 法向之和，Ft 为局部有符号切向 X/Y 分量分别求和后的向量模，曲线不额外滤波。

导出器检查当前 MJCF 指纹与 Raw 一致；该指纹不涵盖独立网格/纹理内容，跨版本回放需保留采集时的运行资产。
