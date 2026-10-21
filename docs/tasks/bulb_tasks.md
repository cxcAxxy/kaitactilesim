# 灯泡任务指南

本页负责三种灯泡任务的场景、被动螺纹模型、专家动作、初始化和 Raw 诊断。
原生成功数值只在[任务成功判定](success_criteria.md#bulb-screw)维护；本页的几何尺寸、控制目标和工程检查另行标明。
通用运行先看[环境说明](../workflows/environment.md)与[流水线速查](../workflows/pipeline_quickstart.md)，
模型专属配置和部署资产规则见[π0.5 合同](../workflows/model_specific_evaluation.md#pi05-contract)。

| task | 初态 | 专家流程 | 取料/返回支架 |
| --- | --- | --- | --- |
| `bulb-screw` | 桌面自由灯泡 | 拿取、搬运、入牙、旋入、拧紧、释放和回位 | 保留取料卡座 |
| `bulb-screw-inplace` | 灯座入口刚啮合，灯灭 | 原位抓稳、旋入、拧紧、释放和回位 | 移除无用取料卡座 |
| `bulb-unscrew` | 已安装且灯亮 | 反向旋出、脱扣、清离、普通桌面放稳释放和回位 | 返回位置无卡座 |

章节入口：[拿取并旋入](#bulb-screw)、[原位旋入](#bulb-screw-inplace)、[旋出放桌](#bulb-unscrew)。

三项独立采集、转换和训练。旧示教倒放、改 task 或 deployment 名字不能生成新任务；
支持模型以统一入口 `--list-support` 为准，已注册接口不表示已有匹配权重。

<a id="bulb-screw"></a>
## 拿取并旋入：bulb-screw

`bulb-screw` 复用 shared 机器人、双手、350 个触觉 probe、相机和大桌面。head、left_wrist 和 right_wrist 与 USB 使用同一套共享标定，另有任务近景 bulb_closeup。

### 运行与当前验收

底层专家和样例刷新入口分别为 `scripts/workcell/view_bulb_screw.py`、`scripts/workcell/refresh_light_bulb_example.py`。

```bash
pixi run --as-is run-bulb-screw
pixi run --as-is run-bulb-screw --headless
# 复现到新目录，目录必须事先不存在
pixi run --as-is refresh-light-bulb-example --output datasets/light_bulb_example_new
```

默认快速五指策略从桌面抓取灯泡，先抬高并平移到灯座中心上方，确认灯头距离原螺纹口沿 5 cm（高于漏斗上口 2.8 cm）且水平对准后才竖直下降入牙；随后固定手腕目标、用五指分 4 段旋入约 100°，施力确认拧紧，再松手退回 home。目标是完整任务不超过 60 秒仿真时间；录制和离线导出需要更长的墙钟时间。

默认验收目录是 `datasets/light_bulb_example/`，按 `install_ram_example` 的精简结构仅保留 `raw/` 下的 HDF5、清单 JSON、结果 JSON，`review/review.mp4` 和 `curves/right_hand_force_curves.png`。详细力矩、螺纹及亮灯诊断保留在 HDF5 和结果 JSON 中。该目录属于生成数据，不随源码仓库提交。

日常复现使用上述新输出目录。`pixi run --as-is refresh-light-bulb-example` 不带 `--output`
是**显式替换维护中开发样例**的操作：验证通过后替换 `datasets/light_bulb_example`，
失败自动清理临时目录。只有明确要更新该样例时才执行，不用于覆盖既有 H50 报告或历史数据。
该开发样例采用精简布局；有效回归测试保留在 tests。

`pixi run --as-is view-bulb-screw` 查看初始场景，`--threaded` 从入牙位置调试。命令行及 Python executor 默认均为快速五指手指驱动。显式 `--grasp pinch --speed normal` 保留旧双指腕部驱动作为兼容对照，结果标记 `legacy_pinch_wrist`；它不用于当前验收样例。
旧双指模式先在收臂位置转腕，再进入原标定解支；五指模式沿原抓取路径接近，两种模式分别保存自己的轨迹身份。

### 连续旋入与末端阻力

| 参数 | 配置 |
| --- | --- |
| 桌面高度 | 0.680 m |
| 原螺纹口沿 | (0.620, -0.180, 0.736) m |
| 底座 / 漏斗 | 圆柱外径 88 mm；上口内径 60 mm（z = 0.758 m），下端内径 33.6 mm（z = 0.734 m）；斜面相对水平面约 61° |
| 入牙零位 | 接触头底端 z = 0.730 m |
| 螺距 / 行程 | 每圈 4 mm / 入牙后下降约 1.11 mm，共 100° |
| 最终插入深度 | 约 7.11 mm |
| 金属灯头有效长度 | 约 5.11 mm，完整旋入后肩部贴住口沿 |
| 灯泡质量 / 球壳横向直径 | 60 g / 58 mm（高度及球心不变） |
| 物理时钟 | 500 Hz |

初始取料位增加四爪被动卡座：四根圆角竖直卡爪围住半径 16 mm 的颈部，内半径 16.9 mm、总高 28 mm，使用低摩擦接触。卡座顶部完全开放，灯泡仍为 freejoint，可直接向上取出；没有 weld、joint 或隐藏执行器。位置随机化时，卡座与灯泡应用同一组 ±2 mm XY 偏移，保持相对间隙不变。球壳仅将 XY 半轴从 30 mm 缩为 29 mm，Z 半轴、球心以及 60 g 基准质量/惯量保持不变。

内螺纹有 48 段、外螺纹有 32 段胶囊牙顶碰撞体。实际进入约 6 mm、牙顶接触载荷超过 0.02 N，且满足对准、倾角和速度门限后，才允许啮合。灯泡保留 freejoint，通过接触触发的原生 weld 连接被动导向体；没有手—灯泡 weld、灯泡执行器或机器人轨迹中的灯泡外加力矩。

捕获时还要求灯泡线速度不超过 0.025 m/s、角速度不超过 0.5 rad/s。满足门限后，先根据实测插入深度初始化被动 slide/hinge 的螺距相位，再计算导向体到自由灯泡的完整相对位姿；启用 weld 的当步不改写灯泡 freejoint 位姿或速度。门限内最多 0.4 mm 的横向残差和 2° 的倾角残差在 0.2 s 内连续收敛，轴向与偏航相位保持螺距一致，避免高速或一次性吸附产生速度尖峰。反向旋出超过捕获带后脱扣，只有灯头重新退出入口后才重新布防，避免边界处反复接合。

底座上方的漏斗由 24 段凸扇区围成空心斜面，斜面参与真实碰撞，可把偏心灯泡导向原螺纹入口。下端留出灯泡颈部空间，居中入牙时不会先碰漏斗；原螺纹、口沿缓冲垫和底部挡块仍承担最终旋入与压紧。

被动 slide/hinge 用 equality 按 `θ = (2π / 0.004) × d` 耦合，每个 2 ms 物理步连续求解旋转和下降。动作段末没有单独下移指令，牙面接触前不会启用悬托约束。

末端口沿有 16 个柔顺接触区，在最后约 0.4 mm 行程受压，随后肩部—硬口沿和底部挡块共同限制进给。接触参数全程固定。握力目标根据实测口沿载荷从每指约 2.2 N 增至约 6 N，通过执行器闭环施力；实际力有动态波动，目标不是力值上限。

搬运先达到高于原螺纹口沿至少 5 cm 的实际灯头高度，再沿水平路段移到灯座中心；执行器目标到位不直接视为灯泡到位。实际水平偏差小于 0.5 mm、在原螺纹口沿上方 50 ± 1 mm 且姿态对准后，才开始向入牙位下降。测试禁止搬运和高空下降阶段出现灯泡—灯座接触，避免斜向扫到漏斗边缘。

样例刷新还会检查原始灯泡位姿：灯头下降到漏斗上口高度及以下时，横向偏差不得超过 0.4 mm、倾角不得超过 0.01 rad。未经过这段下降或未对准的采集不会替换当前样例。

松手并离开灯座后，右手先沿右侧移动到 home 上方，再垂直回到 home 位姿，最后收敛到精确关节 home；直接从灯座旁按关节角度插值回 home 会扫到左手。完整动作测试和样例刷新均拒绝左右手臂之间的接触。

旋拧开始后固定右臂七关节执行器目标；指腹通过手指关节雅可比沿切向运动，并跟随螺距连续下降。每段目标最多 30°，缩短后的 100° 有效螺纹行程约分 4 段完成；换指时中间三指与拇指/小指交替松开和复位，其余手指保持灯泡。整个过程不发送手腕旋转或回腕指令。实际手腕仍有执行器柔顺带来的小幅偏移，记录并单独验收。实际段数和完整耗时记录在当前样例中，并检查耗时低于 60 秒。

手指控制器每 20 ms 计算运动和握力修正，在接下来的每个 2 ms 物理步线性推进关节目标，避免整段指令一次跳变。交替松开、复位和重新接触时，支撑组继续根据实测法向力调整手指目标，不再锁死旧角度而积累挤压力。入牙后的重新预加载也使用同样的连续指令。这里改变的是实际执行器运动，导出的力数据不做平滑。

旋拧前的抬升、搬运、对准及入牙补偿采用单独的稳力控制：手臂 IK 仍以 50 Hz 计算目标，各关节目标在每个 2 ms 物理步连续推进；手指每个物理步重新读取实际接触力，以较小的径向指腹位移修正握力，并锁住手指展收关节，减少纠偏附带的侧向拖动。位移增量和握力目标变化率按物理步长缩放。搬运中拇指保持对抗，四根长指的目标法向力约为 3 N；入牙后拇指也参与握力调节。这些控制只用于灯泡任务的默认五指策略。

到名义圈数后进入 `tighten`：交替换指恢复行程，再给出有限顺时针指腹目标。
手指在 1.2 s 内累积有限切向目标并保持加载 0.4 s，逐步核验实际角度、接触扭矩、五指载荷和隐藏支撑。
承力停转与机械到座的数值条件见[成功判定](success_criteria.md#bulb-screw)。
结果记录 `tightening_verified`、`tightening_peak_torque_nm` 和 `tightening_stall_duration_s`。

默认五指策略在 tighten 中逐个物理步检查承力停转，避免漏掉 20 ms 控制周期之间的短暂卸载；首次满足检查的当步，立即切换为暖色发光材质并打开随灯泡移动的局部光源，不等阶段结束或松手。完成原有拧紧检查后进入 hold_tight，继续保持加载目标 0.4 秒，确保视频中先在施力状态下亮灯，再松手；松手后保持亮灯。重置时熄灭，重新旋出超过就位深度容差或脱扣时也会熄灭。旋拧力均值使用 tighten 阶段，亮灯后固定目标的 hold_tight 单独标记并检查承力。亮灯是自动控制器确认拧紧后的视觉反馈，仅修改材质和照明；`BulbScrewState.bulb_lit` 及 HDF5 的 `bulb_screw/bulb_lit` 记录亮灭状态。

松手后检查全机器人脱手和独立机械稳定；完整条件见[成功判定](success_criteria.md#bulb-screw)。
持续失去参与指接触、单指载荷超过 35 N、IK 不可达或脱扣均会使专家流程失败。
`BulbScrewMonitor` 默认核验完整物理结果；专家 executor 自己实施 tighten/release 检查，并显式使用 seated-only 内部信号。

这是“真实接触力 + 理想螺旋导向 + 简化末端顺应”模型。柔顺性用软接触近似，不是可视网格变形。尺寸、材料、拧紧力矩未经实物标定；尚不覆盖错牙、磨损、玻璃破坏或电连接。

### 原始数据

样例复用 shared EpisodeRecorder、SolverDistributedTactileProvider 和视频面板。状态与触觉 100 Hz，三路训练相机 Raw RGB（head、left_wrist、right_wrist）各 30 Hz、320×240；复核视频为 10 Hz。场景内 `bulb_closeup` 相机仍可用于交互观察，但不额外复制到精简样例。Fn 为每指 35 单元之和，|Ft| 为局部两个有符号分量分别求和后的向量模。原始值及曲线不滤波、不裁剪、不插值；固定色标热图可能饱和。换手和松手应卸载，力不是全程单调上升。

HDF5 保存共享 probe、实际接触、Fn/有符号 Ft、机器人指令/状态、物体位姿、螺纹进度、口沿/底部载荷和实际手力矩，以及手指关节位置、手腕实测位姿、固定的右臂目标。任务监测器每个积分步后刷新 mj_forward，因此力、图像和状态在同一时刻；视频匹配同源且非未来的触觉样本。

结果 JSON 的 `example_validation.turn_force_variation` 记录旋入进度 15%–80% 内、各旋拧段去掉开头 0.2 秒和末尾 0.1 秒后的原始 Fn/Ft 标准差及相邻采样跳变 RMS。该统计不包含主动换指卸载和末端阻力上升；500 Hz 原始接触力的抖动及换指期间支撑力另由完整动作回归测试检查。100 Hz 导出本身不用于证明更高频率接触动力学已经收敛。

`example_validation.transport_force_variation` 分别统计 lift、transfer 和 align_thread 阶段入牙前的原始力波动，各阶段去掉开头 0.2 秒和末尾 0.3 秒。完整动作回归检查搬运和高空下降时不与灯座接触，同时检查 500 Hz 接触力和 100 Hz 采样下的跳变，避免只验证旋拧段而遗漏前半程。

额外保存 eq_active 和捕获后的 eq_data 供诊断。通用自由物体回放器尚不支持完整恢复螺纹捕获逻辑；视频直接使用本次仿真原始 RGB。

### 数据来源与动作合同

当前 `bulb-screw` mechanics 为 `phase_aware_strict_completion_v15`。采集、转换和模型部署必须绑定相同 task、
mechanics、scene 指纹和 instruction；旧数据仅重标动作不能消除场景差异。
`0926_200` 的 v12/v15 完整指纹、动作标签回放和当时 0/20 的解释保留在
[历史档案](../archive/task_scene_history.md#bulb-action-contract)。

模型输入、`recorded-controller-target-v1`、H30/H50 配置、normalizer/provenance、线性执行插值及诊断恢复阶段，
统一见[π0.5 数据与部署合同](../workflows/model_specific_evaluation.md#pi05-contract)。
物理任务本身的连续执行控制与 Raw 诊断保持本页的定义。

### Python 接口

```python
from kaihand_tactile_env.tasks.bulb_screw.task import BulbScrewSimulation
from kaihand_tactile_env.tasks.bulb_screw.execution import BulbScrewExecutor
sim = BulbScrewSimulation()
result = BulbScrewExecutor(sim, grasp_mode="five-finger", speed="fast").execute()
print(result.success, result.tightening_verified, result.state.clockwise_turns)
```

observer(sim, phase) 在每个物理步调用，should_stop() 每步检查。每条任务创建新的 executor；重置用 sim.reset()。自动捕获/释放需使用 BulbScrewSimulation 子类。

### test.pdf 第 11 节参考核查


- [FurnitureBench 官方仓库](https://github.com/clvrai/furniture-bench)：提供 FurnitureSim（基于 Isaac Gym），作为装配任务组织的参考。本实现未拷贝其灯具网格或代码，避免引入另一套机器人和仿真依赖。
- [MuJoCo 官方关节等式说明](https://github.com/google-deepmind/mujoco/blob/main/doc/computation/index.rst)：明确说明多项式关节耦合可用于 helical joint；这里采用线性旋转—轴向耦合。
- [MuJoCo 官方 weld 文档](https://mujoco.readthedocs.io/en/stable/XMLreference.html#equality-weld)：用于门限内捕获后的相对位姿约束。
- [MuJoCo 官方 SDF bolt 实现](https://github.com/google-deepmind/mujoco/blob/main/plugin/sdf/bolt.cc)：可作为未来牙面接触模型的参考。本版未启用 SDF，也不声称具有真实螺纹接触精度。

第 11 节其余条目未作为本版实现依赖；文档表格中关于外部项目的资产和螺纹精度描述未一概视为已核实。

<a id="bulb-screw-inplace"></a>
## 原位旋入：bulb-screw-inplace

`bulb-screw-inplace` 使用独立 task、mechanics 和 scene 身份，从灯座入口开始，保留原拿取旋入任务的独立身份。

### 初态与五步流程

初态不算专家操作：灯泡已对正灯座并在入口刚啮合，未到座、未拧紧、灯灭，机器人双手在 home。
沿用原 v15 被动螺旋约束和接触模型，没有物体执行器、手—灯泡固定或新增外部旋转力。
场景仅去掉不再使用的桌面取料支架，灯座和灯泡碰撞参数不变。

1. 张手接近灯座上的灯泡。
2. 原位五指抓稳，确认实际接触稳定，不执行抬起、搬运或重新入牙。
3. 通过手指旋拧、卸载换指、重新抓稳循环推进，按实测灯泡角度到座。
4. 到座后继续施力，确认承力停转，再点亮反馈灯。
5. 松手、短距离撤离并验证无机器人支撑下稳定就位，随后回 home 收尾。

入口到终点的名义旋入量沿用约 100°，每段最多 30°，一般有四个有效旋入行程。
换指和末端 tighten 不算额外有效旋入次数；不以固定“四下”或“五下”作为成功条件。

### 初始化与随机性

reset 先通过真实齿接触门限啮合，再真实静置 0.3 秒，验证未到座、未亮灯、无机器人接触、
低速且剩余旋入行程充足，之后才归零计时和速度。记录静置前后的真实角度、深度和速度，
不能通过归零掩盖漂移。初态通常约 0.144°、6.0019 mm 插深；实际值以每条 initialization 为准。

原被动模型存在轻微自然漂移，不把物体冻结在坐标上：90 秒无手空跑约停在 10.24°，
仍未到座、未亮、无成功。只有抓稳之后实际推进才可获得有效旋入确认。
当前 `--seed` 保留运行身份，入口位姿**不随机化**；多个 seed 不能当成不同初态的泛化试验。

### 判定与记录

原生 monitor 按“有效初始化 → 原位抓稳 → 实际推进 → 承力停转 → 到座脱手稳定”建立证据；
抓持、有效旋入、到座、释放及历史证书失效的全部数值条件只在
[成功判定](success_criteria.md#bulb-screw-inplace)维护。

完整阶段解释见 [staged_v1](../workflows/staged_evaluation.md)；
观察台按真实导入的 task 身份显示轨迹，不把旧任务模型轨迹改名。

`state.success` / `physical_success` 是物理任务结果，`home_return_verified` 单独记录收尾。
专家 `task_result.success` 表示完整执行流程正常结束；若物理完成后收尾被取消，
结果保留 `physical_success=true`，但完整流程失败并给出 `cleanup_failed` 原因。
灯光是反馈，不是拧紧证据：直接打开视觉灯、数够次数或回到 home 都不能伪造成功。

staged_v1 使用独立 task 身份和四个有序阶段：
`stable_inplace_grasp → threaded_and_seated → tightening_verified → released_stably`。
该协议没有 pickup 前置；其它任务继续使用各自阶段定义。
离线观察台注册了本任务的角度、深度、扭矩和原生标志，实际轨迹由对应评估导入。

### 本地运行

在 `kaitactilesim` 根目录执行，无渲染检查或跑完整专家。
依赖、渲染后端与本机 runtime 配置见[环境与工作目录](../workflows/environment.md)。

```bash
.pixi/envs/default/bin/python -B scripts/workcell/view_bulb_screw_inplace.py --headless --duration 2
.pixi/envs/default/bin/python -B scripts/workcell/view_bulb_screw_inplace.py --headless --run-task --seed 0
```

带显示环境时去掉 `--headless`，默认用灯泡近景；也可以使用
`pixi run --as-is view-bulb-screw-inplace`、`pixi run --as-is run-bulb-screw-inplace` 和 `pixi run --as-is check-bulb-screw-inplace`。

通用采集入口已接通 `--task bulb-screw-inplace`，三路相机、控制目标、状态、触觉和初态证据均独立保存。
新采集默认 Raw 相机 30 Hz、状态/触觉 100 Hz，执行器与判定仍在 500 Hz 物理步更新。
生产采集不允许把旧灯泡轨迹仅改名字充当此任务数据。
π0.5 / LeRobot v3 转换和 π0.5 部署评估均使用独立任务与环境身份；正式部署要求
新的 recorded-controller-target 数据合同，不允许重标旧任务 checkpoint 的 manifest。

任务独立的训练配置、动作来源、归一化与 `shared_task_pi05_deployment_v2` 合同见
[π0.5 数据与部署合同](../workflows/model_specific_evaluation.md#pi05-contract)。

### 已完成的本地验证（2026-10-15）

当时的 CPU 专家、回归计数、Raw 来源、localhost 预览和依赖安装情况，以及 10-18 接入快照，
见[历史验收档案](../archive/task_scene_history.md#bulb-inplace-2026-10-15)。这些是带日期的专家/接口记录，
模型闭环成绩应从匹配该任务的实际评估产物查询。
新采集使用[统一采集入口](../workflows/data_collection.md)，选择新输出目录。

Raw 诊断组为 `bulb_screw_inplace/`，同时保留 reset 初始化证据。新模型评估必须真实导入该 task，
不能通过改名复用原 `bulb-screw` 轨迹。原位旋入的场景、monitor 与专家实现在
`src/kaihand_tactile_env/tasks/bulb_screw_inplace/{config.py,task.py,execution.py}`。

<a id="bulb-unscrew"></a>
## 旋出并放桌：bulb-unscrew

流程：验证已安装状态 → 接近并五指抓稳 → 逆时针打破座底压紧
→ 五指逆向 gait/regrasp，沿螺距上升 → 被动模型真实触发脱扣 → 抬起完全越过漏斗
→ 转移到普通桌面位置 → 桌面有载支撑、放稳、卸载、松手 → 退让并返回共享初始姿势。

继承当前旋入任务 v15 的短螺纹、被动螺旋导向和自动脱扣机制；不是校准后的真实
多圈 E27 模型。reset 同步灯泡、carriage slide/hinge 和 weld 相对姿态，先物理静置
并检查残余速度、载荷和 thread error，再建立 episode 零时刻。运行时不能直接关闭
weld、修改灯泡 qpos/qvel 或施加外力来制造脱扣。

灯灭仅是中间状态，不代表成功。完整成功要求已安装 → 抓稳 → 有效反向旋出并脱扣
→ 真实清离 socket/funnel → 普通桌面有载支持和稳定 → 释放 → 手退让和回位。
各连续时长、真实脱扣所需旋出量、自由放桌与回位容差见
[灯泡旋出成功判定](success_criteria.md#bulb-unscrew)。
新取出任务已删除桌面返回位置的 pickup cradle（四根固定架支柱），安装端的灯座、
漏斗和被动螺纹保留；原 `bulb-screw` 场景及其拿取架不改。
普通返回位置沿用原桌面放置点附近 `(0.500, -0.180)` m；seed 仅随机化该桌面
目标的 XY（±2 mm），不是对已经约束在固定灯座里的灯泡做自由平移。
放置必须先取得真实桌面支撑并稳定，再松手；记录松手时的落下高度和支持
峰值，不能先放掉灯泡再补写“已放稳”。home 数值条件见[成功判定](success_criteria.md#bulb-unscrew)；
手部容差考虑速度执行器/静摩擦的静止残差（曾测得约 `3.20°`）。

桌面新版以 `placed_on_table` 为成功里程碑，并升级 mechanics version、场景指纹和
训练 instruction；旧的返回卡座 Raw/checkpoint/deployment 不能重命名为新版。
重新采集、转换并训练后，才可用于新版模型闭环评估。

桌面版示教与转换预检的来源路径、验证范围见[历史快照](../archive/task_scene_history.md#task-variants-initial-validation)。
数据能否用于当前训练，以实际 Raw、mechanics、场景指纹和转换清单核对为准。

Raw 诊断组为 `bulb_unscrew/`，状态/控制默认 100 Hz、相机 30 Hz，任务 monitor 在每物理步更新。
场景、参数、初始化与专家执行见 `src/kaihand_tactile_env/tasks/bulb_unscrew/{config.py,task.py,execution.py}`。
转换和部署须保持任务 instruction、mechanics 和场景指纹一致，不能将返回卡座版 manifest 重标为桌面版。
