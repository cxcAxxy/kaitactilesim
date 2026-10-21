# 分阶段评估：当前判据、时间与归档边界

本页唯一维护 `staged_v1` 的阶段名称、阈值、连续计时、缺测和归档可比较性；
原生成功见[任务判据](../tasks/success_criteria.md)，运行参数见[模型评估](model_evaluation.md)，离线流程见[统一观察台](unified_evaluation.md)。

## 先区分四种结果

| 层次 | 读取位置 | 能说明什么 |
|---|---|---|
| 原生物理任务结果 | `summary.json.task_success`、任务 monitor/state | 满足该任务自己的成功规则；不因 staged 观测而被替换 |
| 专家完整流程 | 专家 `task_result.success`、阶段/清理结果 | 专家控制器是否完整执行；可能另要求退手、回位等收尾，不等同于所有模型的原生成功 |
| 有序阶段结果 | `staged_evaluation.stages`、`first_times` | 该轮按当时规则**曾经**依序达到哪些阶段；不是当前姿态或最终安全性证明 |
| 数据与协议有效性 | `physics_valid`、`valid_trial`、`diagnostic_only`、`formal_metrics_valid`、来源 SHA | 结果是否具备对应证据、是否可用于所声明的统计；有阶段进展不自动变成正式有效 trial |

另有执行器 `phase` 或适配层 `stage()` 返回的 `approach`、`grasp`、`turn`、`return` 等文本，
它们是过程提示/状态摘要，不是这里的 Stage1…Stage4 布尔里程碑，也不是独立的成功证据。
控制器进入 `turn` 不证明灯泡已经转动，进入 `return` 也不证明物体已经合法取出；
应核对同轮原生物理测量、累计 `stages` 和最后一条 `measurements`。

例如旧 `bulb-screw` 的 staged 最后一阶段只要求机械就位稳定，不要求它原生判据中的承力拧紧和松手；
所以 staged 完整成功不等于原生成功。反过来，模型也可能满足原生终态，但没有经过 staged 要求的前置动作。
新增 `bulb-screw-inplace` 则明确将抓稳、有效旋入、拧紧、释放的原生证据接入独立四阶段。
触觉曲线、专家视频或控制器 `phase` 不能代替这些物理证据。

## 当前支持范围

当前统一流水线有 **13 个任务**；其中 **10 个注册了 staged 观测**：

| 任务 ID | 观测对象 | 有序阶段 | staged 完整成功 |
|---|---|---|---|
| `bulb-screw` | `bulb` | 拿起 → 持有靠近灯座 → 旋入并机械就位 | Stage3 曾达成 |
| `bulb-screw-inplace` | `bulb` | 原位抓稳 → 有效旋入并就位 → 承力拧紧 → 松手后稳定 | Stage4 曾达成 |
| `bulb-unscrew` | `bulb` | 完全脱离灯座 → 搬运、受控放桌并稳定松手 | Stage2 曾达成 |
| `install-ram` | `ram` | 拿起后持有靠近槽口 → 稳定机械插入 | Stage2 曾达成 |
| `usb-insert` | `usb_plug` | 拿起 → 持有靠近插口 → 稳定机械插入 | Stage3 曾达成 |
| `usb-unplug` | `usb_plug` | 完全脱离插口 → 手持且不触桌连续保持 2 秒 | Stage2 曾达成 |
| `poker-draw` | `card` | 桌面有效拖动 → 稳定对向抓持 → 终态持牌超过 3 秒 | Stage3 曾达成 |
| `whiteboard-wipe` | `eraser` | 拿起板擦 | 未定义，`null` |
| `whiteboard-wipe-horizontal` | `eraser` | 拿起板擦 | 未定义，`null` |
| `whiteboard-wipe-horizontal-line` | `eraser` | 拿起板擦 | 未定义，`null` |

`pick-place`、`vase-wipe`、`sponge-grasp` **未注册 staged**。它们仍有自己的原生成功判据和可视化，
但不能强行套用本表或传入 staged 开关；`StagedEvaluationMonitor` 对未注册任务直接拒绝。

该开关已接入 **π0.5** 的 shared / USB insert / Poker runner，也接入复用这些执行器的
Poker CPM/FTP-1 和相应的共享 FTP-1 runner；模型特有环境与 frozen K 见[模型专项](model_specific_evaluation.md#tactile-policy)。
任务注册 staged 不表示所有模型族的 runner 都支持它；共享 FTP-1 仍仅能对本表注册的任务启用阶段观测。
EgoSteer、EgoTouch、T-Rex、LingBot 等是否支持普通评估，另查 `evaluate.py --list-support`；
该查询列普通 runner 注册，不是 staged 能力查询。接口存在也不代表已有匹配权重或真实模型验收。

## 共同测量定义

以下是阶段条件的组成量，不是另加的安全保护阈值。

- 指尖载荷：只累加当前任务对象与右手 `tactile_pad_col` 碰撞面的法向力幅值，按五指分类。
  接触点方向相反也不抵消该载荷；其单位是 N，不是压力或世界坐标净合力。
- 对向抓持 `opposed_grasp_now`：拇指载荷 ≥ 0.025 N，其余四指**合计** ≥ 0.025 N，
  且拇指法向力向量与其它四指合向量的余弦 ≤ 0.50。向量长度不足时余弦未知，不能获得对向抓持。
  普通 staged 对向抓持不要求五指各自接触；新原位旋入的五指抓稳由其原生 monitor 更严格验证。
- 稳定抓持：上述对向抓持连续 ≥ 0.15 秒。
- 拿起 `picked_up`：当前对向抓持、对象参考位置比 monitor 初始化时升高 ≥ 0.02 m、
  对象与桌面和初始支架均无接触，三者共同连续 ≥ 0.15 秒。
  “无支撑”这里仅排除桌面和初始支架，不额外排除 socket 或全部环境接触。
- 靠近入口：当前对向抓持，且对象**工作端 site** 到入口 site 的三维距离 ≤ 0.05 m，连续 ≥ 0.15 秒；
  同时还要满足稳定抓持。不是用对象中心或图像距离替代。
- 机械就位 `seated_now`：读取新鲜原生任务状态的 `seated`，不是根据控制阶段、最终灯光或深度单变量猜测。
  staged 在此基础上另要求连续 ≥ 0.20 秒，不能与原生 monitor 自己的成功 dwell 混为一个参数。

接触角色按 MuJoCo 编译场景中的实际几何分类：

| 角色 | 当前分类 | 用途/限制 |
|---|---|---|
| `hand` | 右手 `hand_r_` 的对象接触 | Poker 桌面拖动的手接触；不是整个机器人的接触 |
| `robot` | 左右机械臂、右手安装结构及 `hand_r_` / `hand_l_` 身体的对象接触 | 释放要求排除整个机器人支撑，不只清空五指 |
| `table` | `tabletop`、`poker_table_top` | 拿起/放桌/扑克支撑；不是灯座、支架或任意环境平面 |
| `initial_support` | `bulb_pickup_cradle*`、`ram_presentation_stand*` 身体 | 拿起必须离开取料支架 |
| `socket` | `usb_socket_*`、`ram_socket_*`、`bulb_socket_*` | 插口/灯座接触数、载荷及完整取出检查 |

`*_normal_load_n` 是对应接触点法向力**幅值之和**；
同一接触可能同时属于 hand 和 robot，不能把角色载荷再次相加当成总力。
`support_free_now` 检查接触数为 0，并非“载荷小于某阈值”；很小但仍存在的接触也会阻止拿起条件。

## 逐任务的实际阶段判据

### Bulb screw：抓取并旋入

| 阶段键 / 定义 | 当前条件 |
|---|---|
| Stage1 `picked_up` | 共同“拿起”条件连续 ≥ 0.15 s |
| Stage2 `held_near_socket` | Stage1 已达成；当前稳定对向抓持，且工作端到灯座入口 ≤ 5 cm 的抓持靠近持续 ≥ 0.15 s |
| Stage3 `threaded_and_seated` | Stage2 已达成；当前螺纹仍啮合，且原生 `seated=true` 连续 ≥ 0.20 s |

工作端/入口为 `bulb_tip` / `bulb_socket_mouth`。
原生 `seated` 包含真实插深、目标圈数、螺距误差、侧偏、倾角、速度、灯座及肩部承载与肩部间隙；
具体机械数值见[成功判据总表](../tasks/success_criteria.md)。
Stage3 **不检查**原生 `tightening_verified`、松手或回 home，不可改名为“完整专家流程成功”。

### Bulb screw inplace：灯泡原位旋入

| 阶段键 / 定义 | 当前条件 |
|---|---|
| Stage1 `stable_inplace_grasp` | 新鲜原生 `initially_threaded_verified=true` 且 `grasp_verified=true` |
| Stage2 `threaded_and_seated` | Stage1 已达成；原生 `effective_rotation_verified=true`；当前螺纹啮合；原生 `seated=true` 连续 ≥ 0.20 s |
| Stage3 `tightening_verified` | Stage2 已达成；新鲜原生 `tightening_verified=true` |
| Stage4 `released_stably` | Stage3 已达成；新鲜原生 `released_stably=true`，当前 `seated=true`，当前 `robot_contact_count=0` |

这里不套用共享“拿起”或“工作端靠近入口”的前置，因为灯泡已经在入口啮合。
原生抓稳要求五指各 ≥ 0.025 N 连续 0.10 s；有效推进需要抓稳后实际旋转至少 30°及对应轴向推进，
并扣除初态漂移影响。承力停转要求连续 0.30 s 内旋转 ≤ 0.30°、顺时针接触扭矩 ≥ 0.10 N·m、
五指各 ≥ 4 N，且实际到座、无非指腹/其它机器人支撑；释放由原生 monitor 验证无机器人接触下到座稳定 ≥ 0.20 s。
这些是原生独立证据，不是把通用 `grasp_hold_s=0.15` 改成了 0.10。
灯亮、固定拧动次数、手已经回原位均不能替代证据。`home_return_verified` 单列，不属于 staged 第五阶段。
详见[独立任务说明](../tasks/bulb_tasks.md#bulb-screw-inplace)。

### RAM install：拿取并安装内存条

| 阶段键 / 定义 | 当前条件 |
|---|---|
| Stage1 `picked_up_and_held_near_slot` | 曾满足共同拿起条件；随后当前稳定对向抓持，工作端到槽口 ≤ 5 cm 的抓持靠近持续 ≥ 0.15 s |
| Stage2 `inserted_and_seated` | Stage1 已达成；原生 `seated=true` 连续 ≥ 0.20 s |

工作端/入口为 `ram_bottom` / `ram_socket_mouth`；曾拿起的事实内部保留，不要求靠近时还高于初态 2 cm。
原生 `seated` 要求目标深度 6 mm ± 0.2 mm、键形几何能进入、方向误差 ≤ 1°、穿透 ≤ 0.2 mm、
触底负载 > 0.01 N 或有效保留的真实触底证书、线速度 < 0.001 m/s、角速度 < 0.05 rad/s。
原生成功还要求自己的 0.30 s 稳定时间；staged 的 0.20 s 不能代替它，也不要求松手、回位。

### USB insert：拿取并插入

| 阶段键 / 定义 | 当前条件 |
|---|---|
| Stage1 `picked_up` | 共同拿起条件连续 ≥ 0.15 s |
| Stage2 `held_near_socket` | Stage1 已达成；当前稳定对向抓持，工作端到插口 ≤ 5 cm 的抓持靠近持续 ≥ 0.15 s |
| Stage3 `inserted_and_seated` | Stage2 已达成；原生 `seated=true` 连续 ≥ 0.20 s |

工作端/入口为 `usb_plug_tip` / `usb_socket_mouth`。
原生 seating 要求插深落在 `[11.9, 12.1] mm`、壳体适配、方向误差 ≤ 3°、穿透 ≤ 0.1 mm、
线速度 ≤ 0.0005 m/s、角速度 ≤ 0.05 rad/s，并有当前有效真实止挡接触或仍在有效近底几何范围的触底证书。
真实止挡接触需同时有止挡接触点、止挡法向载荷及轴向阻力各 ≥ 0.0001 N。
这是任务 seating 内部条件；不是模型 runner 额外 0.3 mm 穿透保护阈值。
原生成功 dwell 为 0.10 s，staged Stage3 的 dwell 仍为 0.20 s。
原生触底证书的 0.60 N / 0.15 s 是保留历史支撑的另一条路径，不是当前止挡接触路径的强制前置；
不能把它写成每次 staged 就位都额外需要的条件。

### USB unplug：完全拔出并手持保持 2 秒

当前协议为 `usb-unplug-held-clear-2s-v2`，与旧的放回、释放规则分开记录。

| 阶段键 / 定义 | 当前条件 |
|---|---|
| Stage1 `fully_extracted` | 工作端在插口局部轴向深度 ≤ −2 mm 且无插座接触，连续 ≥0.10 s |
| Stage2 `extracted_and_held_2s` | Stage1 曾达成；当前完全清离、`hand_contact_count > 0`、`table_contact_count == 0` 共同连续 ≥2.0 s |

两秒从完整条件首次同时成立时开始，不在 Stage1 后额外再加 0.1 秒。
不要求搬运距离、对向双指压力、固定姿态或低速，不要求放回、松手、退让或回 home。
缺少清离、手接触或桌面接触证据时不获得成功；回插、断手接触、触桌或采样断档会重置该连续窗口。
阶段仍是累计曾达成，不随播放游标变化，也不改写原生轨迹标签。
旧 `released_stably_on_table` 徽标和成绩继续按旧归档解释，不直接换名成两秒保持。

### Bulb unscrew：完整取出，再受控放桌

灯泡旋出流程不随 USB 简化而改变。当前 `fully_clear_now` 要求螺纹已脱离、
插入深度 ≤−3 mm、灯泡与灯座接触数为 0 且法向载荷 ≤0.01 N。

优先读新鲜原生状态的 `fully_clear_now`；缺少该字段时可用上述当前真实几何/接触量计算，
绝不从“灯灭”“控制阶段已 return”推断。

| 阶段键 / 定义 | 当前条件 |
|---|---|
| Stage1 `fully_extracted` | 当前完整取出共同连续 ≥ 0.10 s |
| Stage2 `released_stably_on_table` | Stage1 已达成，并依次获得下述搬运、受控触桌、松手稳定证据 |

Stage2 的证据链是：

1. 对向抓持、当前完整取出、无桌面/初始支架接触共同连续 ≥ 0.15 s 后，记录一次持有位置原点。
   维持这些条件时，从该原点的**三维直线位移**达到 ≥ 0.02 m 才记 `transport_verified=true`。
   不是路径长度累加，也不是从拔出前初态直接计算；计时这 0.15 s 之前的位移不自动补计。
2. 在已验证搬运后，完整取出、对向抓持、桌面接触数 ≥ 1、桌面法向支撑 ≥ 0.01 N、
   线速度 ≤ 0.02 m/s、角速度 ≤ 0.20 rad/s 共同持续 ≥ 0.05 s，记
   `controlled_table_placement_verified=true`。直接摔落后停稳不能补出受控持有触桌证据。
3. 保留完整取出、搬运和受控触桌的事实后，当前桌面接触数 ≥ 1、支撑 ≥ 0.01 N、
   整个机器人接触数为 0、线速度 ≤ 0.02 m/s、角速度 ≤ 0.20 rad/s，共同持续 ≥ 0.20 s。

staged **不要求**回原落点、直立、朝向恢复或机械臂回 home；这些可能是原生/专家流程额外要求。
它也不会把原生“提前脱扣”等历史失败自动清除：几何完整取出阶段与原生合法旋出成功是两种证据，必须并列看。

### Poker draw：桌面拖动、抓持与终态保持

| 阶段键 / 定义 | 当前条件 |
|---|---|
| Stage1 `dragged_on_table` | 当前右手与卡接触，手-卡载荷 ≥ 0.025 N；桌面接触数 ≥ 1；相对初态 XY 位移 ≥ 0.02 m；卡底距牌桌面 ≤ 0.005 m；共同连续 ≥ 0.05 s |
| Stage2 `stable_grasp` | Stage1 已达成；当前对向抓持连续 ≥ 0.15 s |
| Stage3 `terminal_hold_over_3s` | Stage2 已达成，且以下终态条件共同连续**超过** 3.0 s（不是 ≥ 3.0 s） |

终态条件：当前对向抓持；卡底高于牌桌 ≥ 0.02 m；无桌面接触；
对象参考位置到 `(0.55, −0.10, 1.05) m` 的三维误差 ≤ 0.03 m；
卡面朝头部相机的余弦 ≥ 0.80，且朝机器人方向（世界 −X）的余弦 ≥ 0.80；
线速度 ≤ 0.02 m/s、角速度 ≤ 0.20 rad/s。
卡面法向为卡身体旋转矩阵负 Z 轴；head 朝向以物体到真实 head 相机位置的方向计算。

卡底间隙用 `card_core_geom` 的实际姿态半尺寸投影与 `poker_table_top` 顶面计算，
不是对象中心高度，也不是下面那张普通工作台。缺任一几何或相机方向证据时不能凑齐成功。
Stage1 的间隙上限没有额外新增穿透下限；穿透保护是独立 guard，不要据此把阶段达成视为安全认证。

### 三种白板擦拭：仅定义拿起阶段

三个独立任务均只有 Stage1 `picked_up`，采用共同拿起条件（2 cm、对向抓持、脱离桌面/支架、0.15 s）。
`success` / `complete_success` **始终是 `null`**，`whiteboard_complete_success_defined=false`；
在线 `staged_recording` 的 batch `complete_success_count` / `complete_success_rate` 也为 `null`。
离线网站的 `stage_groups()` 则记录 `complete_success_count=0`、`complete_success_defined=false`：
该 0 仅是“没有 true 值”的程序计数，不表示已定义完整成功率或白板全部失败。
拿起不代表擦净；不能造出 Stage2、将 `null` 当作 0，或因为唯一阶段通过就显示完整成功。

清洁与接触几何另保存在 `continuous_metrics`：

- 原墨迹块均值清洁度，以及初始墨迹并集的面积加权清洁度；
- 有效贴合面积 / 完整板擦底面积；
- 去重累计触及的初始墨迹并集覆盖率；
- 接触时间、几何采样和未知/分辨率信息。

默认约 30 Hz 采样几何，板擦底面网格约 1 mm、墨迹网格约 0.5 mm；
使用编译 mesh 底面和有限白板，法向夹角 ≤ 20°、间隙及穿透各 ≤ 2 mm、
实际参与该物理步积分的板擦-板法向载荷达到几何门控，才计有效贴合。
这是刚体几何代理，不是柔性材料真实接触面积；不插值补扫两次采样间的轨迹。
静止接触可以增加几何覆盖，却不自动增加真实去污效果。

白板抓持接触单独读取既有积分步的 solver 接触缓存（后续 `mj_forward` 之前），
不拿后刷新瞬时力冒充整步证据。若时间已经推进但缺该步缓存，`opposed_grasp_now=null`，
记 `integrated_whiteboard_grasp_contacts` 缺测，当前拿起连续计时不能继续。

## 全部 24 个共享阈值

下表是当前 `StageThresholds()` 默认值；每条轨迹自己的 `thresholds` 才是归档权威。
这些阈值不是所有任务原生判定或安全阈值，也不是触觉接触图的 0.05 N 诊断阈值。

| 字段 | 默认值 | 单位 | 用途 |
|---|---:|---|---|
| `lift_m` | 0.02 | m | 相对初态拿起高度；扑克终态离桌间隙 |
| `near_mouth_m` | 0.05 | m | 工作端到入口的最大三维距离 |
| `grasp_hold_s` | 0.15 | s | 稳定抓持、拿起；取出后持有建立搬运原点 |
| `finger_load_n` | 0.025 | N | 拇指与其它四指合计的最小载荷；扑克手接触最小载荷 |
| `maximum_opposition_cosine` | 0.50 | 1 | 对向抓持力方向余弦上限 |
| `near_hold_s` | 0.15 | s | 抓持靠近入口连续时长 |
| `seating_hold_s` | 0.20 | s | 当前原生就位连续时长 |
| `clear_hold_s` | 0.10 | s | 完整取出连续时长 |
| `usb_extracted_hold_s` | 2.0 | s | 新版 USB 完整清离并由手持、无桌面接触的共同连续时长 |
| `table_placement_hold_s` | 0.05 | s | 仍持有时受控触桌时长 |
| `release_hold_s` | 0.20 | s | 松手放桌稳定时长 |
| `minimum_table_support_n` | 0.01 | N | 放桌/释放所需桌面法向支撑 |
| `maximum_release_linear_speed_m_s` | 0.02 | m/s | 放桌/释放线速度上限 |
| `maximum_release_angular_speed_rad_s` | 0.20 | rad/s | 放桌/释放角速度上限 |
| `drag_displacement_m` | 0.02 | m | 扑克相对初态平面位移下限 |
| `drag_hold_s` | 0.05 | s | 扑克有效桌面拖动时长 |
| `maximum_drag_clearance_m` | 0.005 | m | 拖动时卡底间隙上限 |
| `minimum_transport_m` | 0.02 | m | 取出后受持有的三维位移下限 |
| `poker_terminal_hold_s` | 3.0 | s | 扑克终态保持严格超过此时长 |
| `poker_terminal_position_tolerance_m` | 0.03 | m | 扑克检查位置误差上限 |
| `poker_terminal_face_cosine` | 0.80 | 1 | 同时朝 head 与机器人方向的余弦下限 |
| `poker_terminal_linear_speed_m_s` | 0.02 | m/s | 扑克终态线速度上限 |
| `poker_terminal_angular_speed_rad_s` | 0.20 | rad/s | 扑克终态角速度上限 |
| `observation_gap_steps` | 1.5 | 1（物理步倍数） | 相邻观测超过该倍 timestep，重置未完成连续窗口 |

门槛比较使用代码中的微小浮点容差（一般 `1e-10 s`）；不要将图上四舍五入到 3.00 秒视为 Poker 已通过。

## 顺序、累计、缺测与时钟

- `stage_achievement_policy=ordered_cumulative_ever_reached`：只在所有前置阶段已达成时锁存下一阶段；
  已锁存的阶段不会因为后续掉落、失去接触或离开终态而退回 false。
  `highest_stage` 是有序达成数，`failure_stage` 是下一未达成的定义，不是致命失败原因。
- 连续条件分别从首次 true 的**观测时间**开始计时；该首样本持续时间为 0。
  false 或未知立即清空相应连续窗口。不能用控制帧数、视频帧数或已运行总秒数代替。
  各条件并行积累计时，未要求每个下一阶段在上一阶段锁存后再重新计满；同一物理观测可能依序锁存多个阶段。
- 取出后搬运中断会清空当前搬运位置原点；已验证的搬运/受控触桌事实仍保留。
  观测断档同样清空所有未完成连续窗口和当前搬运原点，但不撤销已累计阶段或已完成物理证据。
- 缺失值及非有限数值落盘为 `null`，不变成 0 或成功。需要的原生 seating 缺失会列
  `native_seating_predicate`，不能套用另一个任务的几何规则补判。
- 原生状态带 timestamp 时，必须与当前物理时间相差不超过 `1.5 × timestep`；否则不接纳，记录
  `native_task_state_timestamp_mismatch`。无 timestamp 的合法当前状态（例如白板适配状态）随本次物理观测读取。
- 时基是 `mujoco_data_time_seconds`，不是 wall time、专家时间或 MP4 播放位置。
  `first_times` 是首次锁存时的实际 `data.time`（秒），未达成为 `null`；不要假定所有任务的计时起点都严格为 0。
- 相邻观测间隔 > `1.5 × timestep`（加数值容差）计入 `observation_gap_count` 并重置连续窗口；
  时间倒退直接报错，必须为新 episode 新建 monitor。
  同一 timestamp 的刷新不会增加连续时长；归档状态日志另要求时间严格递增。
- monitor 在**每个物理子步、原生状态更新后**更新，通常 500 Hz；
  `staged_states.jsonl` 只保存已完成的约 30 Hz 控制端点。不能事后拿这份稀疏日志重跑 observer
  并声称重新验证过 500 Hz 连续保持；视频/触觉也不能按行号对齐。

## 启用命令与记录产物

以下在当前 `kaitactilesim` 根目录执行，使用本仓库 Pixi Python；
环境安装、`--as-is` 与缺包要求见[环境与工作目录](environment.md)。

只读检查不连接模型、不开仿真：

```bash
cd /cpfs_infra/user/chenxianchi/code/kaitactilesim
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
.pixi/envs/default/bin/python -B scripts/analysis/evaluate_unified.py rules
.pixi/envs/default/bin/python -B scripts/evaluate/evaluate.py --list-support
PYTHONPATH=src .pixi/envs/default/bin/python -B scripts/workcell/run_shared_task_pi05_policy.py --help
```

下面是使用匹配 `bulb-screw-inplace` checkpoint/deployment 的单 seed staged 评估模板。
模板不自动选择已有权重；应换入当前源码 prepare/verify 通过的 manifest 和新输出目录。
新任务只能使用自己的 `shared_task_pi05_deployment_v2` manifest，不允许修改旧 `bulb-screw` manifest 的 task 冒充：

```bash
cd /cpfs_infra/user/chenxianchi/code/kaitactilesim
.pixi/envs/default/bin/python -B scripts/evaluate/evaluate.py \
  --task bulb-screw-inplace --model-family pi05 \
  --deployment-manifest /path/to/new_inplace/deployment_manifest.json \
  --output-dir /new/path/bulb_screw_inplace_h50_staged \
  --num-trials 1 --seed-start 0 --video-count 1 \
  --execute-steps horizon --max-sim-seconds 90 --guard-mode observe \
  -- --server ws://127.0.0.1:18783 --staged-evaluation-version staged_v1
```

`--staged-evaluation-version` 是 runner 参数，写在统一分隔符 `--` 后。
其它已注册任务使用其对应的匹配 manifest；USB/Poker π0.5 batch 会转发同名参数。
H50 来自真实 checkpoint/manifest 的 `prediction_horizon=50`，不是因为目录写了 `h50`；
`--execute-steps horizon` 才表示 K=H。开 staged 不会把 H30 权重变成 H50，也不会给模型增加历史或触觉输入。

staged 下原生成功不提前结束，目标是跑满 `--max-sim-seconds`（默认 90 **仿真**秒）；
数值/基础设施故障、strict guard、显式请求上限/墙钟上限等仍可能提前退出。
`full_duration_evaluation=true` 是协议声明，还必须核对实际末时刻和完成状态，不能只凭这个布尔值认为跑满。
`observe` 保留 `diagnostic_only=true` / `formal_metrics_valid=false`；阶段统计不升级为正式成功率。

每个 trial 新增：

```text
seed_NNN/
├── summary.json                 # 原生结果不覆盖；另嵌入 staged_evaluation
├── staged_evaluation.json       # 终态副本、累计阶段、首次时间、阈值、测量、identity
└── staged_states.jsonl          # 控制端点快照；observer 本身按物理步更新
```

`stage_definitions`、`stages` / `stage_flags`、`named_stages`、`first_times`、
`instantaneous_conditions`、`continuous_durations_s`、`measurements` 应并列解释：
累计为 true 不表示最后一条瞬时条件仍为 true。`legacy_task_success` 保留原生结果。
当前 batch 汇总只计 `valid_trial=true` 且有同任务/版本 sidecar 的条目，要求阶段定义和阈值一致；
分母是 `valid_staged_trials`，不是无条件的 `planned`。跨批次还必须额外检查来源身份和整轮完成条件。

## 历史归档、H50 筛选与可比较性

固定 checkpoint 作业、plan/run/resume 与逐组验收见[π0.5 阶段批次档案](../archive/pipeline_batches_202609_202610.md#pi05-stage-batches)。
旧 USB 策略按新协议诊断的训练/评分绑定见[模型专项](model_specific_evaluation.md#usb-拔出协议迁移)。

### 历史归档读取

历史观察台目前展示 **9 个任务、360 条 H50 记录**，不含后来新增的 `bulb-screw-inplace` 模型轨迹。
当前 runtime 有 10 个 staged 任务，并不意味着现有历史站自动多出第 10 个任务。
离线 reader 依据记录自身的合同读取，网站整条导入/触觉/时钟/任务字段链路受实际来源审核约束；
新任务的 registry 和专家 MP4 不能证明已完成新模型归档的导入验收。

`evaluation_unified/staged.py` 读取归档，不执行当前物理规则：

- 显式 `prediction_horizon=50` 才纳入；未知 horizon、H30 均排除，不根据目录名猜。
- summary、sidecar 必须逐字段一致，task/seed/checkpoint SHA/deployment ID/horizon 明确匹配。
- 只接纳正常终态、无执行/录像错误、显式 full-duration 且末时刻达到声明上限的记录；
  录像字段存在时要求完成。原生成功为 false 并不排除一条完整有效的失败案例。
- 校验绑定 manifest 的 SHA 与冻结 evaluation source inventory；保存 observer、recorder、完整源码 inventory
  及所录阶段定义/阈值的身份。它没有重新执行历史源码，也不替当前源码作兼容背书。
- 日志逐行流式校验：身份与合同不变、时钟一致且严格递增、control tick 递增、阶段不可回退、样本数及最后快照匹配；
  不接纳截断/并发改变的输入。重复别名只计一次，同身份内容冲突全部排除，不挑成功的那份。
- 保留 state/contact/tactile/video 的独立时钟，不插值、补零、归一化时间轴或偷偷把专家同秒配到模型同动作。

同叫 `staged_v1` **不代表规则实现相同**。当前纯规则导出的 `catalog_sha256` 只描述目录/阈值，
不包含 observer、原生任务或录制代码；`assert_rules_compatible()` 还要求调用者提供已验证的实现身份，
但它自己不会替调用者读文件验证 SHA。
跨批次比较至少分开 task、model、checkpoint 内容、cohort、保护/执行协议、归档规则与来源身份；
旧九任务记录沿用自己的 stage_definitions/thresholds/observer SHA，绝不套用今日新任务列表重写。

离线 export/verify 还要求受保护原工程 source 可访问且核验通过；当前原 A/B 旧路径已退役时不能绕过保护重导出。
已有站点和原有报告保持不动，具体可用命令及受阻前提见[统一评估与观察台](unified_evaluation.md)。

## 判据来源

- [当前规则目录、24 个共享阈值及兼容合同](../../src/kaihand_tactile_env/shared/evaluation_rules.py)
- [测量角色、连续计时与各任务阶段逻辑](../../src/kaihand_tactile_env/shared/staged_evaluation.py)
- [原生状态更新后接入 staged 的任务适配层](../../src/kaihand_tactile_env/shared/policy_tasks.py)
- [控制端点记录与 batch 阶段汇总](../../src/kaihand_tactile_env/shared/staged_recording.py)
- [历史 H50 staged 归档 reader](../../scripts/analysis/evaluation_unified/staged.py)
