# 拧灯泡环境

第四个独立任务 `bulb-screw` 复用 shared 机器人、双手、350 个触觉 probe、相机和大桌面。head、left_wrist 和 right_wrist 与 USB 使用同一套共享标定，另有任务近景 bulb_closeup。

## 运行与当前验收

```bash
pixi run run-bulb-screw
pixi run run-bulb-screw --headless
pixi run refresh-light-bulb-example
```

默认快速五指策略从桌面抓取灯泡，先抬高并平移到灯座中心上方，确认灯头距离原螺纹口沿 5 cm（高于漏斗上口 2.8 cm）且水平对准后才竖直下降入牙；随后固定手腕目标、用五指分 4 段旋入约 100°，施力确认拧紧，再松手退回 home。目标是完整任务不超过 60 秒仿真时间；录制和离线导出需要更长的墙钟时间。

默认验收目录是 `datasets/light_bulb_example/`，按 `install_ram_example` 的精简结构仅保留 `raw/` 下的 HDF5、清单 JSON、结果 JSON，`review/review.mp4` 和 `curves/right_hand_force_curves.png`。详细力矩、螺纹及亮灯诊断保留在 HDF5 和结果 JSON 中。该目录属于生成数据，不随源码仓库提交。

每次修改后运行 refresh 命令，以一次成功采集更新 HDF5、视频和曲线。验证通过后替换旧样例；失败自动清理临时目录。不再保留日期目录、历版验收报告、重复逐帧图像或中途诊断脚本。有效回归测试保留在 tests。

`pixi run view-bulb-screw` 查看初始场景，`--threaded` 从入牙位置调试。命令行及 Python executor 默认均为快速五指手指驱动。显式 `--grasp pinch --speed normal` 保留旧双指腕部驱动作为兼容对照，结果标记 `legacy_pinch_wrist`；它不用于当前验收样例。

## 连续旋入与末端阻力

| 参数 | 配置 |
| --- | --- |
| 桌面高度 | 0.680 m |
| 原螺纹口沿 | (0.620, -0.180, 0.736) m |
| 底座 / 漏斗 | 圆柱外径 88 mm；上口内径 60 mm（z = 0.758 m），下端内径 33.6 mm（z = 0.734 m）；斜面相对水平面约 61° |
| 入牙零位 | 接触头底端 z = 0.730 m |
| 螺距 / 行程 | 每圈 4 mm / 入牙后下降约 1.11 mm，共 100° |
| 最终插入深度 | 约 7.11 mm |
| 金属灯头有效长度 | 约 5.11 mm，完整旋入后肩部贴住口沿 |
| 灯泡质量 / 球壳直径 | 60 g / 60 mm |
| 物理时钟 | 500 Hz |

内螺纹有 48 段、外螺纹有 32 段胶囊牙顶碰撞体。实际进入约 6 mm、牙顶接触载荷超过 0.02 N，且满足对准、倾角和速度门限后，才允许啮合。灯泡保留 freejoint，通过接触触发的原生 weld 连接被动导向体；没有手—灯泡 weld、灯泡执行器或机器人轨迹中的灯泡外加力矩。

底座上方的漏斗由 24 段凸扇区围成空心斜面，斜面参与真实碰撞，可把偏心灯泡导向原螺纹入口。下端留出灯泡颈部空间，居中入牙时不会先碰漏斗；原螺纹、口沿缓冲垫和底部挡块仍承担最终旋入与压紧。

被动 slide/hinge 用 equality 按 `θ = (2π / 0.004) × d` 耦合，每个 2 ms 物理步连续求解旋转和下降。动作段末没有单独下移指令，牙面接触前不会启用悬托约束。

末端口沿有 16 个柔顺接触区，在最后约 0.4 mm 行程受压，随后肩部—硬口沿和底部挡块共同限制进给。接触参数全程固定。握力目标根据实测口沿载荷从每指约 2.2 N 增至约 6 N，通过执行器闭环施力；实际力有动态波动，目标不是力值上限。

搬运先达到高于原螺纹口沿至少 5 cm 的实际灯头高度，再沿水平路段移到灯座中心；执行器目标到位不直接视为灯泡到位。实际水平偏差小于 0.5 mm、在原螺纹口沿上方 50 ± 1 mm 且姿态对准后，才开始向入牙位下降。测试禁止搬运和高空下降阶段出现灯泡—灯座接触，避免斜向扫到漏斗边缘。

样例刷新还会检查原始灯泡位姿：灯头下降到漏斗上口高度及以下时，横向偏差不得超过 0.4 mm、倾角不得超过 0.01 rad。未经过这段下降或未对准的采集不会替换当前样例。

松手并离开灯座后，右手先沿右侧移动到 home 上方，再垂直回到 home 位姿，最后收敛到精确关节 home；直接从灯座旁按关节角度插值回 home 会扫到左手。完整动作测试和样例刷新均拒绝左右手臂之间的接触。

旋拧开始后固定右臂七关节执行器目标；指腹通过手指关节雅可比沿切向运动，并跟随螺距连续下降。每段目标最多 30°，缩短后的 100° 有效螺纹行程约分 4 段完成；换指时中间三指与拇指/小指交替松开和复位，其余手指保持灯泡。整个过程不发送手腕旋转或回腕指令。实际手腕仍有执行器柔顺带来的小幅偏移，记录并单独验收。实际段数和完整耗时记录在当前样例中，并检查耗时低于 60 秒。

手指控制器每 20 ms 计算运动和握力修正，在接下来的每个 2 ms 物理步线性推进关节目标，避免整段指令一次跳变。交替松开、复位和重新接触时，支撑组继续根据实测法向力调整手指目标，不再锁死旧角度而积累挤压力。入牙后的重新预加载也使用同样的连续指令。这里改变的是实际执行器运动，导出的力数据不做平滑。

旋拧前的抬升、搬运、对准及入牙补偿采用单独的稳力控制：手臂 IK 仍以 50 Hz 计算目标，各关节目标在每个 2 ms 物理步连续推进；手指每个物理步重新读取实际接触力，以较小的径向指腹位移修正握力，并锁住手指展收关节，减少纠偏附带的侧向拖动。位移增量和握力目标变化率按物理步长缩放。搬运中拇指保持对抗，四根长指的目标法向力约为 3 N；入牙后拇指也参与握力调节。这些控制只用于灯泡任务的默认五指策略。

到名义圈数后进入 tighten：先交替换指恢复关节行程，再继续给出有限的顺时针指腹运动目标，检测实际接触力矩和灯泡转角。手指在 1.2 秒内累积有限的切向目标，然后保持加载目标 0.4 秒；检查最后至少 0.3 秒的实际灯泡转角变化 ≤ 0.3°、顺时针力矩 ≥ 0.1 N·m、每指法向力 ≥ 4 N 且实际就位，才允许松手。结果包含 tightening_verified、tightening_peak_torque_nm 和 tightening_stall_duration_s；只碰到底座不能代表机器人任务完成。

默认五指策略在 tighten 中逐个物理步检查承力停转，避免漏掉 20 ms 控制周期之间的短暂卸载；首次满足检查的当步，立即切换为暖色发光材质并打开随灯泡移动的局部光源，不等阶段结束或松手。完成原有拧紧检查后进入 hold_tight，继续保持加载目标 0.4 秒，确保视频中先在施力状态下亮灯，再松手；松手后保持亮灯。重置时熄灭，重新旋出超过就位深度容差或脱扣时也会熄灭。旋拧力均值使用 tighten 阶段，亮灯后固定目标的 hold_tight 单独标记并检查承力。亮灯是自动控制器确认拧紧后的视觉反馈，仅修改材质和照明；`BulbScrewState.bulb_lit` 及 HDF5 的 `bulb_screw/bulb_lit` 记录亮灭状态。

松手后要求灯泡独立稳定：约 100°、7.11 ± 0.3 mm 插入深度、外露螺纹 ≤ 0.1 mm、肩部间隙绝对值 ≤ 0.1 mm、底部与肩部均有接触载荷，并在速度门限内连续稳定 0.2 s。持续失去任一参与指的接触、单指载荷超过 35 N、IK 不可达或脱扣均失败。

这是“真实接触力 + 理想螺旋导向 + 简化末端顺应”模型。柔顺性用软接触近似，不是可视网格变形。尺寸、材料、拧紧力矩未经实物标定；尚不覆盖错牙、磨损、玻璃破坏或电连接。

## 原始数据

样例复用 shared EpisodeRecorder、SolverDistributedTactileProvider 和视频面板。状态与触觉 100 Hz，三路训练相机 Raw RGB（head、left_wrist、right_wrist）各 30 Hz、320×240；复核视频为 10 Hz。场景内 `bulb_closeup` 相机仍可用于交互观察，但不额外复制到精简样例。Fn 为每指 35 单元之和，|Ft| 为局部两个有符号分量分别求和后的向量模。原始值及曲线不滤波、不裁剪、不插值；固定色标热图可能饱和。换手和松手应卸载，力不是全程单调上升。

HDF5 保存共享 probe、实际接触、Fn/有符号 Ft、机器人指令/状态、物体位姿、螺纹进度、口沿/底部载荷和实际手力矩，以及手指关节位置、手腕实测位姿、固定的右臂目标。任务监测器每个积分步后刷新 mj_forward，因此力、图像和状态在同一时刻；视频匹配同源且非未来的触觉样本。

结果 JSON 的 `example_validation.turn_force_variation` 记录旋入进度 15%–80% 内、各旋拧段去掉开头 0.2 秒和末尾 0.1 秒后的原始 Fn/Ft 标准差及相邻采样跳变 RMS。该统计不包含主动换指卸载和末端阻力上升；500 Hz 原始接触力的抖动及换指期间支撑力另由完整动作回归测试检查。100 Hz 导出本身不用于证明更高频率接触动力学已经收敛。

`example_validation.transport_force_variation` 分别统计 lift、transfer 和 align_thread 阶段入牙前的原始力波动，各阶段去掉开头 0.2 秒和末尾 0.3 秒。完整动作回归检查搬运和高空下降时不与灯座接触，同时检查 500 Hz 接触力和 100 Hz 采样下的跳变，避免只验证旋拧段而遗漏前半程。

额外保存 eq_active 和捕获后的 eq_data 供诊断。通用自由物体回放器尚不支持完整恢复螺纹捕获逻辑；视频直接使用本次仿真原始 RGB。

## Python 接口

```python
from kaihand_tactile_env.tasks.bulb_screw.task import BulbScrewSimulation
from kaihand_tactile_env.tasks.bulb_screw.execution import BulbScrewExecutor
sim = BulbScrewSimulation()
result = BulbScrewExecutor(sim, grasp_mode="five-finger", speed="fast").execute()
print(result.success, result.tightening_verified, result.state.clockwise_turns)
```

observer(sim, phase) 在每个物理步调用，should_stop() 每步检查。每条任务创建新的 executor；重置用 sim.reset()。自动捕获/释放需使用 BulbScrewSimulation 子类。

## test.pdf 第 11 节参考核查


- [FurnitureBench 官方仓库](https://github.com/clvrai/furniture-bench)：提供 FurnitureSim（基于 Isaac Gym），作为装配任务组织的参考。本实现未拷贝其灯具网格或代码，避免引入另一套机器人和仿真依赖。
- [MuJoCo 官方关节等式说明](https://github.com/google-deepmind/mujoco/blob/main/doc/computation/index.rst)：明确说明多项式关节耦合可用于 helical joint；这里采用线性旋转—轴向耦合。
- [MuJoCo 官方 weld 文档](https://mujoco.readthedocs.io/en/stable/XMLreference.html#equality-weld)：用于门限内捕获后的相对位姿约束。
- [MuJoCo 官方 SDF bolt 实现](https://github.com/google-deepmind/mujoco/blob/main/plugin/sdf/bolt.cc)：可作为未来牙面接触模型的参考。本版未启用 SDF，也不声称具有真实螺纹接触精度。

第 11 节其余条目未作为本版实现依赖；文档表格中关于外部项目的资产和螺纹精度描述未一概视为已核实。
