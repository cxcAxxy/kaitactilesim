# 十三个任务的成功判定

本页说明 `kaitactilesim` 源码中模型闭环评估实际使用的条件（判据初次核对日期为 2026-10-15），
不是建议中的更严格规则，也不用于重算、改写历史 H50 标签。
命令见[流水线速查](../workflows/pipeline_quickstart.md)；
阶段诊断另见 [staged_v1](../workflows/staged_evaluation.md)。

## 先区分三种结果

- **原生任务成功**：runner 中的 `task_success`、monitor 的 `state.success` 或专用 outcome monitor。
  不同任务的条件不同；没有统一的“回 home 即成功”规则。
- **专家完整流程成功**：executor 的 `result.success`。通常还有拿取、真实松手、退让、轨迹与触觉质量等额外检查，
  不应把专家验收条件直接当成模型 runner 已实施的条件。
- **分阶段诊断**：独立 `staged_evaluation`；记录曾经按序到达的阶段，不覆盖原生结果。
  三种白板的 staged 完整成功仍为 `null`，不是擦净或未擦净的替代判据。

`task_success=true` 也不保证正式有效：物理/模型/产物完整性、`guard_mode`、
`diagnostic_only` 和 `formal_metrics_valid` 另行检查。在 `observe` 下，记录过力/碰撞保护
事件的轨迹可能继续并取得任务成功；这不把它变成严格协议下的正式成功。
本页载荷与材料门槛均是当前仿真参数，不是硬件安全额定值或真实材料标定。

连续时间由真实物理步累积。USB、RAM 和灯泡 monitor 不能用稀疏快照或重复读取同一时刻
代替连续证据；观测回退、断档和 reset 的具体处理见对应函数。

## pick-place

取放浅灰圆柱到白盒。模型 runner 的 `StablePlacementDetector` 要求同时满足：

- `cylinder_is_in_box=true`：圆柱中心 XY 位于盒内包络，包络由墙内缘减圆柱半径和 **2 mm** 余量得到；
  中心高度与“盒底上表面 + 圆柱半高”的差 **<5 cm**。
- 环境的抓持稳定器 `grasp_active=false`。
- 物体线速度 **<0.02 m/s**、角速度 **<0.2 rad/s**，连续 **0.1 s**。

成功一旦达到就锁存。这里的释放条件是“抓持稳定器未启用”，**没有额外检查全部机器人接触为零**；
几何函数也没有圆柱倾角门槛。不得把它描述成已经证明无任何手部接触、直立就位或回 home。

专家 `PickPlaceExecutor.execute()` 自身返回值只以最终 `cylinder_is_in_box` 为准；
执行器走过松手、退让、回臂路径，但这些路径不是额外的终态布尔门槛。
正式 Raw 采集还有独立末态稳定检查，不能只看 executor 的一个 `success`。

源码：[几何与专家](../../src/kaihand_tactile_env/tasks/pick_place/task.py)
`cylinder_is_in_box` / `PickPlaceExecutor.execute`；
[模型终态](../../scripts/workcell/run_egosteer_policy.py) `StablePlacementDetector.update`；
[默认阈值与 Raw 验收](../../src/kaihand_tactile_env/shared/recording.py)。

## poker-draw

模型使用 `PokerOutcomeMonitor`，**不是**专家的整套压力质量验收：

1. 曾有指腹接触牌，至少一指法向力 **>0.01 N**。
2. 此后牌仍受桌面支撑、牌法向世界 Z 分量 **≥0.94**，悬出桌沿比例 **≥40%**，
   且朝机器人方向的位移 **≥5 cm**。支撑由实际牌—桌接触法向力 **>0.00001 N** 识别。
3. 此后出现对向夹持并离桌 **≥2 cm**。对向夹持要求牌上侧至少一根非拇指接触、
   拇指在下侧，拇指载荷 **≥0.05 N**、另外四指载荷之和 **≥0.05 N**；并非要求四指各自都接触。
4. 最后对向夹持、牌正面朝 head 及机器人方向余弦均 **≥0.8**，
   距检查位置 `(0.55,-0.10,1.05) m` **≤3 cm**，线速度 **<0.02 m/s**、角速度 **<0.2 rad/s**，
   连续达到 runner 的 `--success-hold-seconds`，默认 **0.10 s**。

当前 π0.5 / π0.5+TReX runner 在该参数 **>0.10 s** 时，额外要求整个终态窗口仍离桌 ≥2 cm；
默认短窗口只要求此前已满足抬牌阶段。阶段和最终成功一旦达到即锁存。
这与 staged 的“持牌超过 3 s”不是同一个指标。

专家还检查历史动作与夹持质量：悬出比例 **≥49%**、四指都曾参与且曾同时接触、拇指与四指形成持续夹持；
抬牌对向接触比例 **≥95%**、最长缺口 **≤0.03 s**，初始持牌对向比例 **≥95%**；
看牌前倾角 **<35°**，看牌转角和腕内转角各 **>80°**；
看牌运动期间对向比例 **≥90%**、四指比例 **≥55%**、最长对向缺口 **≤0.03 s**，
看牌保持期间两种比例均 **≥95%**。看牌期间最低牌高需高于桌沿时刻牌高 **≥5 mm**，
终态至少高出 **8 cm**；朝向余弦各 **≥0.8**、检查位置误差 **<3 cm**。
末态仍需拇指在下、四指在上，拇指和对侧总法向载荷各 **≥0.05 N**，
四指 pad 对齐余弦均 **≥0.94**、拇指 **≥0.84**、夹牌指尖平面角 **≤21°**。
初始 `clear_card` 转掌后的拖牌工作阶段，最大掌面法向 Z 须 **<-0.95**；拖牌朝机器人位移 **>8 cm**、
横向漂移 **<15 mm**，受桌支撑时最低离面 **≥-0.6 mm**，牌背最低间隙 **>0**。
该姿态检查使用 `ready_card`、`hover_card`、`precontact_card`、`four_finger_press`、`slide_card`、`edge_hold` 的内部工作阶段最大值；
公开导出的 `maximum_palm_normal_z` 还包含初态与 `clear_card`，不能直接代替此检查。

专家最终 `success = task_completed AND (strict_force_quality OR acceptance_policy == task-completion-v1)`。
专家 API 默认的 `strict-force-v1` 还要求滑动有样本、每指及四指同时接触比例 **100%**、
每指目标带内比例 **≥95%**、四指同时带内比例 **≥90%**、每指均值误差 **≤0.035 N**、
最大接触缺口 **0 s**；力带容差默认 **±0.075 N**。
统一采集的 `production-v1` 显式采用 `task-completion-v1`，实际准入策略应读取采集元数据。
基础示例目标每指 **0.35 N**，中档示例 **0.50 N**，属于控制/专家质量参数，**不是模型原生成功条件**。
当前 `tip-pad-v7` 的拖牌角上限为 **50°**，夹牌仍 **21°**；历史 `legacy-flat-v1` 拖牌上限 **21°**，
不能用新版本改判旧数据。中档专家另有真实桌沿停稳与交接验收。

源码：[模型 outcome](../../src/kaihand_tactile_env/tasks/poker_draw/rollout_eval.py)
`PokerOutcomeMonitor.observe` / `observe_simulation`；
[对向接触](../../src/kaihand_tactile_env/tasks/poker_draw/policy_control.py) `terminal_metrics`；
[专家](../../src/kaihand_tactile_env/tasks/poker_draw/task.py) `PokerDrawExecutor.execute` / `_slide_press_control_qualified`；
[准入版本](../../src/kaihand_tactile_env/tasks/poker_draw/acceptance.py) `accept_task`；
[中档交接](../../src/kaihand_tactile_env/tasks/poker_draw/mid_full.py) `_prepare_and_slide`。

## usb-insert

原生 `UsbInsertionMonitor.update()` 的成功是 **`seated` 连续 ≥0.1 s**。其中：

- 插头尖端插深在 **11.9–12.1 mm**（底挡深度 12 mm，容差 0.1 mm），完整壳体包络通过孔口；
  姿态误差 **≤3°**、最大接触穿透 **≤0.1 mm**。
- 线速度 **≤0.0005 m/s**、角速度 **≤0.05 rad/s**。
- 当前有真实底挡接触，**或者**保留了此前的 `bottom_out_confirmed`。
  当前底挡接触要求接触数 >0、底挡法向及轴向反力均 **≥0.0001 N**。
- `bottom_out_confirmed` 的取得条件更强：上述几何与低速成立、底挡轴向反力 **≥0.6 N** 连续 **0.15 s**；
  留在近底区域时可保留，离开该区域或观测断档时清除。

**实现边界：原生 `success` 不直接强制 `bottom_out_confirmed=true`，也不强制松手或回 home。**
持续的当前底挡接触也可满足原生成功，不能把它写成必经 0.6 N / 0.15 s 的完整主动压底。

专家完整结果额外要求无流程失败、原生成功、已经释放且 `active_bottom_out_confirmed=true`；
有抓稳/抬升、对准、主动压底、松手和退让过程。当前动作压底门槛为 0.6 N / 0.15 s，
不是旧 V2 的 0.8 N / 0.10 s；Raw Cleaning 按记录版本验收，不能混用。

源码：[原生](../../src/kaihand_tactile_env/tasks/usb_insert/task.py)
`_near_bottom` / `_stationary` / `_with_seating` / `UsbInsertionMonitor.update`；
[参数](../../src/kaihand_tactile_env/tasks/usb_insert/config.py)；
[专家](../../src/kaihand_tactile_env/tasks/usb_insert/execution.py) `UsbInsertionExecutor.execute`。

## usb-unplug

2026-10-15 按新要求简化为 **完全拔出后仍由右手持有，连续保持 ≥2.0 s**，
协议标识 `usb-unplug-held-clear-2s-v2`。以下条件须在同一连续时间窗口共同成立：

1. 尖端相对插座入口的有符号轴向位置 **≤−2 mm**。
2. 插头与插座的接触数 **为零**。
3. 右手与插头仍有实际接触（`right_hand_contact_count > 0`）。
4. 插头与桌面的接触数 **为零**，不能把放桌或掉落后静止计为手持。

不再要求放回标定位置、松手、退手或回 home；不另设保持位置、姿态和速度误差，
也不将初态确认、双指压力阈值或搬运距离作为新版成功的前置门槛。
场景仍从已插入的 reset 初态开始，原有物理安全保护与数据有效性检查不取消。
右手接触只是仿真中的手持观测代理，不等于证明稳定双指夹持或硬件防滑安全。

`extraction_hold_now` 表示当前满足上述条件，`extraction_hold_duration_s` 为本次连续时长，
`extraction_hold_verified` 表示当前连续窗口已满 2 秒（中断后清零）。中断手持、回插、触桌、观测断档或时间回退
会重置未完成连续窗口；重复读取同一时刻不会增加时间。原生当前成功还要求当前仍满足手持清离条件，
模型 adapter 的轨迹级成功按既有方式锁存。初始就位、抓稳、完全拔出以及旧放回/释放/退手字段继续作为诊断保留。
专家现在在实际拔出后手持满 2 秒结束，不再执行旧的放回收尾。

**历史规则单独保留：**既有 H50 归档使用“初始就位 → 双指抓稳 → 完全拔出 → 标定桌面放回 →
释放 → 退手”的严格原生流程，staged 也要求受控搬运放桌和松手。历史网页和原始成功标签不改写，
不能用新版规则名称解释旧徽标。按新规则复评必须另存结果并核对连续证据，不能只凭末帧或原来的拔出布尔值改成成功。

源码：[原生](../../src/kaihand_tactile_env/tasks/usb_unplug/task.py)
`UsbUnplugMonitor.measure` / `update`；
[参数](../../src/kaihand_tactile_env/tasks/usb_unplug/config.py)；
[专家](../../src/kaihand_tactile_env/tasks/usb_unplug/execution.py) `UsbUnplugExecutor.run`。

## bulb-screw

这是“拿起并旋入”，不是原位任务。模型 adapter 使用 `auto_confirm_tightening=True`。
原生成功要求 **机械到座 + 已确认拧紧 + 全机器人脱手**，连续 **≥0.2 s**。

机械到座同时要求：

- 螺纹仍啮合；目标插深 **7.111… mm**、误差 **≤0.3 mm**，目标转数 **100/360 圈**、误差 **≤0.08 圈**；
  螺距误差 **≤0.3 mm**、横偏 **≤0.4 mm**、倾角 **≤2°**。
- 线速度 **≤0.001 m/s**、角速度 **≤0.05 rad/s**；灯座接触载荷 **≥0.01 N**。
- 外螺纹露出高度 **≤0.1 mm**，肩部间隙绝对值 **≤0.1 mm**、肩部接触载荷 **≥0.01 N**。

拧紧确认：到座状态下，至少连续 **0.3 s** 的角度极差 **≤0.3°**，
顺时针指腹扭矩始终 **≥0.1 N·m**，每指法向力始终 **≥4 N**，
没有右手非指腹接触、其它机器人部位隐藏支撑。随后点亮反馈灯。
全机器人脱手要求每指载荷 **≤0.01 N**、右手及整个机器人接触总载荷各 **≤0.01 N**，
并且整个机器人与灯泡接触数为零。

**原生条件没有单独要求此前拿起 6 cm 或回 home。** 专家 `_motion()` 则从自由桌面灯泡开始，
真实抬起 **≥6 cm**、安全搬运、对正入牙、旋拧与加载停转，再松手、退让、执行回位并验收。
正常完成流程才返回 `result.success=true`；计数“四下/五下”、仅到座或手回位均不能替代此流程。

旧 monitor 会把 `bulb_lit` 作为已有拧紧确认的锁存标志。在合法评估路径中灯由物理确认点亮，
但这不是一个与视觉灯完全独立、可抵御直接改灯状态的证书；新原位任务已独立实现该证书，旧实现没有追溯修改。
另注意专家 monitor 默认 `auto_confirm_tightening=False` 时，底层 `state.success` 使用连续到座时间；
专家 executor 自己另外实施严格加载确认与脱手检查，不能只抽取这个中间 `state.success` 作为完整专家结果。

源码：[原生](../../src/kaihand_tactile_env/tasks/bulb_screw/task.py)
`BulbScrewMonitor.measure` / `_update_tightening` / `update`；
[参数](../../src/kaihand_tactile_env/tasks/bulb_screw/config.py)；
[专家](../../src/kaihand_tactile_env/tasks/bulb_screw/execution.py) `_motion` / `_check_tightening_stall` / `execute`。

## bulb-screw-inplace

初态已在灯座入口刚啮合，未到座、未亮灯；没有 pickup 阶段。
原生 `state.success` / `physical_success` 要求依次取得：

1. reset 真实啮合、静置并验证过的 `initially_threaded_verified`。
2. 五指各 **≥0.025 N** 连续 **≥0.10 s**，无非指腹或其它机器人支撑，取得 `grasp_verified`。
3. 相对抓持起点 **以及** episode 初态，实际顺时针推进各 **≥30°**、实际轴向推进各 **≥0.2167 mm**
   （4 mm 螺距 × 30/360 × 0.65），且满足当前有效五指抓持与啮合，取得 `effective_rotation_verified`。
4. 取得与上一节相同数值门槛的独立加载停转证书；不读取灯亮来伪造证书。
5. 当前机械到座（几何/速度/接触门槛同上一节）并全机器人脱手，连续 **≥0.2 s**。

观测断档清除未完成的抓持、释放、回位与停转窗口；真正脱扣或插深回退到
目标以下 **>0.6 mm** 会使旧停转证书失效。已取得的抓稳/有效推进是历史里程碑，
不意味着换指时必须五指始终全部加载。

物理成功**不要求回 home**。`home_return_verified` 另外要求物理成功且关节回位条件持续 **≥0.10 s**：
右臂最大误差 **≤1°**、右手最大误差 **≤4°**、相应关节最大速度 **≤2°/s**、机器人与灯泡接触数为零。
专家完整 `task_result.success` 还要求正常执行完回位收尾；物理成功后若取消收尾，
保留 `physical_success=true`，但流程结果为失败并标记 `cleanup_failed`。
名义有效旋入约 100°，不按行程次数判成功；seed 当前不随机入口位姿。

源码：[独立 monitor](../../src/kaihand_tactile_env/tasks/bulb_screw_inplace/task.py)
`BulbScrewInplaceMonitor.update` / `_update_tightening` / `_extra`；
[参数](../../src/kaihand_tactile_env/tasks/bulb_screw_inplace/config.py)；
[专家](../../src/kaihand_tactile_env/tasks/bulb_screw_inplace/execution.py) `_motion` / `execute`。
初态、五步流程、命令与实测产物见[任务页](bulb_tasks.md#bulb-screw-inplace)。

## bulb-unscrew

当前为无返回支架的普通桌面版，不是旧返回卡座版。原生 monitor 按序要求：

1. 初态有有效初始化记录、仍啮合且到座、灯亮、无机器人接触，
   线速度 **≤0.001 m/s**、角速度 **≤0.05 rad/s**，连续 **≥0.05 s**。
2. 五指各 **≥0.025 N**、没有右手非指腹接触，连续 **≥0.10 s**。
3. 已脱离到座且灯灭，实测旋出 **≥30°**，曾取得逆时针扭矩 **≥0.002 N·m**，确认 breakaway。
4. 真实 engaged→disengaged 时必须已有 breakaway，且旋出量 **≥121.5°**：
   名义自动脱扣量 **131.5°** 减去 **10°** 容差。无效提前脱扣会保留失败原因并阻止最终成功。
5. 不再啮合、插深 **≤-3 mm**、灯座总载荷 **≤0.01 N** 且灯座接触数为零，连续 **≥0.08 s**。
6. 真实放到本条桌面目标：XY 误差 **≤1.5 mm**、Z 误差 **≤4 mm**、倾角 **≤8°**、
   线速度 **≤0.004 m/s**、角速度 **≤0.20 rad/s**，桌面接触载荷 **0.01–12 N** 且有接触、
   无灯座接触、未啮合，连续 **≥0.20 s**。
7. 上述桌面条件仍成立，每指载荷 **≤0.01 N**、整个机器人接触载荷 **≤0.01 N** 且接触数零，连续 **≥0.10 s**。
8. 继续桌面稳定、自由释放，并真实回位：右臂最大误差 **≤1°**、右手 **≤4°**、
   最大关节速度 **≤2°/s**、腕—灯泡距离 **≥10 cm**，连续 **≥0.10 s**。

最终还要求没有历史失败原因、当前未啮合、灯灭、桌面目标/释放/回位条件仍为真。
灯灭只是中间现象，既不证明脱扣，也不证明完成取出。
专家完整结果要求上述原生成功且无流程失败；不会按拧动次数或视觉相似程度直接结束。
桌面目标的 seed 仅对名义 XY 做 ±2 mm 抖动，不随机化固定灯座里的已安装灯泡。

源码：[原生](../../src/kaihand_tactile_env/tasks/bulb_unscrew/task.py)
`BulbUnscrewMonitor.measure` / `update`；
[参数](../../src/kaihand_tactile_env/tasks/bulb_unscrew/config.py)；
[专家](../../src/kaihand_tactile_env/tasks/bulb_unscrew/execution.py) `_motion` / `execute`。

## install-ram

原生 `RamInstallationMonitor.update()` 要求已确认真实到底，并且就位连续 **≥0.3 s**：

- 近底几何：插深距 **6 mm ≤0.2 mm**，完整 DIMM 插入边适合开口，
  朝向误差 **≤1°**，最大穿透 **≤0.2 mm**。
- 低速：线速度 **<0.001 m/s**、角速度 **<0.05 rad/s**。
- 在上述条件下底挡载荷 **≥1.2 N** 连续 **≥0.4 s**，取得 `bottom_out_confirmed`。
  离开近底几何区域、跳步或 reset 清除证据；仍在近底区域时，瞬时卸载/速度波动不会单独清除已取得的证书。
- 当前就位还需要低速和近底，以及当前底挡载荷 **>0.01 N** 或已有到底证书。

原生成功没有单独要求先抬起、释放或回 home。
专家额外要求真实抬起 **≥2 cm**，槽外先对正并低速稳定 **≥0.2 s**；
插入/压底时只推进腕部 Z，保持 XY、方向与手指目标；
主动压底控制目标 **1.7 N**，实际底挡载荷 **≥1.5 N** 等加载条件至少保持 **0.5 s**，
并且已取得原生 1.2 N / 0.4 s 到底证书；不能把较低的原生门槛当作整个专家压底验收。
松手时两指最大载荷 **≤0.03 N**，退让后不得存在 **>0.01 N** 的机器人手部支持，
最终原生成功仍为真，才正常返回专家成功。不模拟卡扣自动闭锁或电气连通。

源码：[原生](../../src/kaihand_tactile_env/tasks/install_ram/task.py)
`RamInstallationMonitor.update`；
[参数](../../src/kaihand_tactile_env/tasks/install_ram/config.py)；
[专家](../../src/kaihand_tactile_env/tasks/install_ram/execution.py) `_motion` / `_press_bottom` / `run`。

## vase-wipe

模型 adapter 的原生 `task_success` **仅等于 `simulation.cleaning.success`**：
全部污渍单元平均残留 **≤5%**，并且每一块残留 **≤10%**。
没有额外的终态 hold，没有要求模型完成专家的视觉复查、离壁退出或回 home。

残留不是按动作阶段/时间直接消失。每块清洁进度取累计摩擦功、滑程、加载时间三者归一化的最小值。
有效累计要求全局内壁法向力 **0.20–5.0 N**、全局切向力 **≥0.30 N**，
该块局部切向载荷 **≥0.025 N**、真实滑速 **≥0.005 m/s**、耗散功率 **>0**。
接触汇总器还会先剔除实际滑速 **>0.15 m/s** 的单个接触，再分配局部载荷/功率；
不能仅从 `CleaningProgress.update` 的局部门槛推断高速冲击会产生清洁。
基准每块要求 **0.045 mJ** 摩擦功、**1.6 mm** 有载行程、**0.14 s** 有效加载；
面积缩放时功门槛乘 `area_scale`，行程门槛乘其平方根。

专家完整结果还要求 `pickup_report.success`、物理清洁成功和最终 `inspection.visually_clean` 同时为真，
正常完成离壁/退出且没有抛出掉落、穿入、翻转或数值异常。
视觉复查使用初始污渍区域：浅黑色残留像素为零，且 **≥98%** 目标像素深度变化 **<6 mm**；
遮挡时先抬高再查。专家的视觉与流程验收**不能反推模型原生成功也检查了这些项**。

源码：[adapter](../../src/kaihand_tactile_env/shared/policy_tasks.py) `TaskPolicyAdapter.step`；
[物理清洁](../../src/kaihand_tactile_env/tasks/vase_wipe/cleaning.py) `CleaningProgress.update` / `success`；
[参数](../../src/kaihand_tactile_env/tasks/vase_wipe/config.py)；
[专家](../../src/kaihand_tactile_env/tasks/vase_wipe/task.py) `VaseWipeExecutor.execute`；
[视觉](../../src/kaihand_tactile_env/tasks/vase_wipe/vision.py)。

## whiteboard-wipe

模型原生成功同时要求：

- 本 episode 最大真实抬升高度 **>12 cm**。
- 所有 **25** 块 Z 字墨迹残留 **≤1e-9**。
- 当前释放条件连续 **≥0.10 s**：桌面支持力 **>0.2 N**、板擦线速度 **<0.01 m/s**、
  右手五指法向载荷之和 **<0.05 N**。

原斜板任务不强制回 home；释放代理只检查上述五指合载，**不是整个机器人接触数为零的证明**。
也没有独立抓持历史布尔门槛；不要把水平变体的额外抓持门槛写进原任务。

清洁只累计真实带载滑动：板擦—板面总法向力 **≥1.5 N**，该块局部切向载荷 **≥0.02 N**、
滑速 **0.005–0.15 m/s**、耗散功率 **>0**。每块累计摩擦功 **≥1 mJ**、
有效行程 **≥3 cm**、有效加载 **≥0.25 s** 才完全清零。
**8 N 是 guard，不是清洁有效性上限**；observe 超限后仍可累计物理清洁，但不是严格协议正式结果。

网页另显示平均清洁进度的 100% 理想参考与最大单元墨迹余量 `≤1e-9` 的擦净条件，
不能用平均进度足够高替代逐块擦净。接触时刻/接触期间平均贴合面积比属于已有归档的
刚体几何诊断，不是真实形变接触面积，也没有通用成功百分比；未接触时均值不适用。
展示公式、来源和导出命令见[目标参考与白板贴合诊断](../archive/evaluation_workbench_release_history_20261015_20261021.md#白板接触期间贴合面积比)。

专家完整结果还要求正确的张手/上方/垂直下降/闭合/抬升顺序；张手误差 **<0.06 rad**、
上方腕位置误差 **<8 mm**、下降横偏和深度误差各 **<1 cm**、闭手前板擦位移 **<2 mm**；
闭手前指尖峰值 **<0.05 N**、对向抓握为拇指 **≥0.05 N** 且其余四指和 **≥0.15 N**，
抓取/擦拭时指尖峰值各 **<6 N**、擦拭阶段板面峰值 **<3 N**、直接手碰板峰值 **<1e-9 N**，
松手末态五指和 **<0.05 N**；全部擦净、真实自由放桌并正常结束。
这些专家质量门槛与模型的 8 N 安全 guard 不是同一条规则。

源码：[模型](../../src/kaihand_tactile_env/shared/policy_tasks.py)
`TaskPolicyAdapter.step` / `_whiteboard_state` / `_whiteboard_release_condition`；
[清洁](../../src/kaihand_tactile_env/tasks/whiteboard_wipe/cleaning.py) `CleaningProgress.update`；
[专家](../../src/kaihand_tactile_env/tasks/whiteboard_wipe/execution.py) `WhiteboardWipeExecutor.run`。

## whiteboard-wipe-horizontal

清洁物理门槛与上一节完全相同，仍为 **25** 块 Z 字。原生成功保留最大抬升 **>12 cm**、全部擦净和
上述桌面自由释放持续 **≥0.10 s**，再增加：

- 先出现对向抓握（拇指 **≥0.05 N**、其它四指和 **≥0.15 N**）；
  在其后物理步中，桌面支持 **≤0.01 N** 且抬升 **≥12 cm**，取得 `horizontal_pickup_verified`。
- 释放计时只能在已经取得 pickup 且已擦净之后开始，不能将初态放在桌面算成完成后的释放。
- 当前右臂相对 ARM_HOME 最大关节误差 **<0.01 rad**，右手相对 spread_grip 最大误差 **<0.06 rad**。

专家还继承上一节的拾取顺序、触觉与清洁质量检查；回位两段实际关节速度峰值 **<70°/s**、
加速度峰值 **<100°/s²**，最终关节误差满足相同门槛、板擦仍自由受桌面支撑，且没有直接手碰板。
专家额外运动质量并未加入模型原生成功布尔式。

源码：[模型](../../src/kaihand_tactile_env/shared/policy_tasks.py)
`_update_horizontal_pickup` / `_whiteboard_state` / `_whiteboard_release_condition`；
[专家](../../src/kaihand_tactile_env/tasks/whiteboard_wipe_horizontal/execution.py)
`HorizontalWhiteboardWipeExecutor.run`。

## whiteboard-wipe-horizontal-line

原生与专家成功门槛继承上一节，但只对独立 scene 中的 **8** 块顶部横线验收。
没有隐藏的另外 17 块 Z 字，也不能用清掉 Z 字的平均值代替 8 块全部清零。
pattern 为 `z_top_bar_v1`；行程是局部 Y 正到负的一条横线。
拿取、对向抓握历史、自由放桌/0.10 s、臂手回位及专家运动质量条件均不删除。

源码：[任务配置](../../src/kaihand_tactile_env/tasks/whiteboard_wipe_horizontal_line/config.py)；
[独立路径、继承验收](../../src/kaihand_tactile_env/tasks/whiteboard_wipe_horizontal_line/execution.py)
`HorizontalLineWhiteboardWipeExecutor.run`；原生判据仍在 shared `policy_tasks.py`。

## sponge-grasp

模型 adapter 以 **100 Hz 审计采样**建立下列过程与终态证据；不应把柔体每个物理子步均接触当作已检查事实。

1. 五指各法向力 **≥0.05 N** 连续 **≥0.5 s**，建立抓持。
2. 在允许释放之前，各审计样本五指都须保持 ≥0.05 N；失去一指会使本轮搬运接触有效性失效。
   真实海绵底部离桌 **≥12 cm**、桌面支持 **≤0.01 N** 才确认抬起。
3. 抬起后、开始放盘前，最低离桌高度 **≥8 cm**、最大桌面支持 **<0.05 N**。
   质心到盘中心 **≤5 cm** 或盘底支持力 **>0.02 N** 时认为开始放盘。
4. 完成上述历史门槛，且最近约 **0.25 s** 的盘底平均支持力 **≥0.08 N**，才允许释放。
   所有指尖最大载荷 **<0.10 N** 连续 **≥0.10 s** 后确认释放，并关闭抓持 patch。
5. 当前质心距盘中心 **≤3.5 cm**，所有柔体顶点 XY 半径 **≤8.2 cm**，
   海绵最低点在盘内底表面 **-5 至 +15 mm**（盘面 Z=0.688 m），
   桌面支持 **<0.05 N**、最大指尖载荷 **<0.10 N**、抓持 patch 未启用。
   在此状态持续 **≥1 s**，该窗口盘底平均支持力 **≥0.20 N**，
   材料 pin、质心和最低点三个高度各自极差均 **≤3 mm**。

最终原生成功是抓稳、抬起、接触有效、允许释放及上述稳定窗口的合取；没有回 home 要求。
抓持 patch 是本任务已有的有限接触斑力矩近似，不是新增物体 weld；原始接触力和物理证据仍需独立记录。

专家也检查五指真实加载、离桌/盘支持及释放，但还要求接近时不提前碰海绵（峰值 **≤0.05 N**）、
闭合前位移 **≤1 cm**；持物相对手部漂移 **≤4.5 cm**，搬运下滑 **≤5 mm**、
搬运桌面支持 **≤0.01 N**；接近腕 Y **≤-0.08 m** 且双手间距 **≥23 cm**。
专家按既定日程执行松手并检查时长、末态盘中心偏移、盘底平均支持和底部高度；
其 `result.success=true` 不等于另一次模型已通过该 adapter 的所有审计窗口。

源码：[模型](../../src/kaihand_tactile_env/shared/policy_tasks.py)
`_sponge_sample` / `_update_sponge` / `_sponge_settle_statistics`；
[参数](../../src/kaihand_tactile_env/tasks/sponge_grasp/config.py)；
[专家](../../src/kaihand_tactile_env/tasks/sponge_grasp/execution.py) `SpongeGraspExecutor.run` / `_record`。

## 维护与复核

新增或修改任务时，必须一起核对原生 monitor、模型 adapter、专家 executor、记录器和 staged observer；
“都叫 success”不保证同义。先保留各自原字段，再解释差异，不擅自把较强的专家门槛追溯套用到旧模型归档。
本页是说明文档，不是机器可执行规则；发生冲突以对应版本的源代码、运行参数和归档源码指纹为准。
