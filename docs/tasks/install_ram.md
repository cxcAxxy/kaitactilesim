# 内存条安装任务

本页负责 `install-ram` 的场景、尺寸、专家抓取/插入、录制产物及原始力诊断。
原生成功数值由[成功判定](success_criteria.md#install-ram)维护；通用命令见[流水线速查](../workflows/pipeline_quickstart.md)，
模型专项设置见[模型专项评估](../workflows/model_specific_evaluation.md)。

原尺寸说明和力诊断已合入本页。旧录制器及生成结果中出现的
`docs/tasks/install_ram_dimensions.md` 指向下文[尺寸与物理模型](#ram-dimensions)，
原力诊断页对应[原始切向力诊断](#ram-force-diagnosis)；源码与既有结果的来源身份保留。

常用章节：[录制模式与产物](#ram-recording)、[尺寸和物理模型](#ram-dimensions)、[原始力诊断](#ram-force-diagnosis)。
原生模型成功本身不强制松手或回 home，专家完整流程另检查收尾。

`install-ram` 是独立的 DDR4 台式机 DIMM 安装场景。模块长约 133.35 mm、高 31.25 mm，PCB 厚 1.40 mm，采用黑色 PCB 和散热马甲外观，包含两面芯片、金手指和偏心定位缺口。卡槽具有真正开放的插入口、对应定位键、两侧壁与底部止挡；主板安装在共享桌面上。尺寸来源及模型近似见[尺寸说明](#ram-dimensions)。

以下命令在 `kaitactilesim` 根目录执行，环境准备见[环境与工作目录](../workflows/environment.md)。

场景通过 MJCF include 直接引用 `shared/mjcf/robot.xml` 和 `shared/mjcf/control.xml`。机械臂、左右手、相机安装参数和触觉垫沿用 shared；任务只新增内存条、卡槽、主板、上料支架和检查用近景相机 `ram_closeup`。采集使用共享的 `head`、`left_wrist` 和 `right_wrist`，近景相机不替代任何共享相机。

```bash
# 查看场景
pixi run --as-is view-install-ram

# 执行夹持、提升、移到卡槽、插入、释放和终态检查
pixi run --as-is run-install-ram
pixi run --as-is run-install-ram -- --headless

# 只检查场景启动
pixi run --as-is check-install-ram

# 日常复现到新目录，不覆盖旧样例
pixi run --as-is record-install-ram-example -- --compact --output-dir datasets/install_ram_example_new

# 用户明确允许覆盖时：验证成功后替换旧例，不保留历史数据
pixi run --as-is record-install-ram-example -- --compact --replace-existing

# 仅检查场景和完整专家，不写录制样例：使用上述 check/run 命令
```

若环境已安装，也可以直接使用 `.pixi/envs/default/bin/python scripts/workcell/view_install_ram.py --headless --run-task`。无窗口录制需要设置 `MUJOCO_GL=egl`；默认 Pixi 录制命令已设置该变量、单线程物理计算和四线程软件渲染参数。

模块沿世界 X 轴伸展、Y 轴为厚度方向、Z 轴向上，插入沿 −Z。初始状态使用竖直上料支架，双手从 shared home 的统一屈肘姿势开始。`prepare()` 只计算抓取目标，不改机器人或内存条的 qpos；右手先用 4 秒移到支架上方并预成形，再用 3 秒下降、静置 0.4 秒后闭指。抓点位于内存条长边中心附近，以减少重力引起的滚转；运输和插入时保持手背朝上、掌面朝下的正手姿势。控制器使用仿真真值位姿完成已知初态的演示，动作通过机器人执行器和原生接触传递到自由运动的内存条。

卡槽内有两组被动弹片等效机构，滑动关节、弹簧和阻尼产生侧向预紧，实际库仑接触摩擦抵抗向下插入。在 3.5–5.5 mm 深度范围可观察滑动阻力；约 6 mm 到底后，控制器进入独立的 `bottom_press` 阶段，根据实际底挡力缓慢加压并保持。两阶段的力均来自接触求解，不按阶段直接写入物体外力或触觉读数。力的绝对值是仿真参数，不是真实连接器插入力标定。

先对准，再下插：机器人在槽口上方约 0.5 mm 处建立约 4 N 的夹持力，固定手指目标后完成最后对准，连续至少 0.2 秒满足位置、方向和低速度条件才开始插入。此后 `insert` 与 `bottom_press` 全程保持腕部 XY、腕部旋转和手指关节目标不变，仅改变腕部 Z。过程中失去开口对准就停止，不通过扭动腕部补救。到底目标载荷在 0.8 秒内平滑建立，采用限速的小增益反馈，没有普通移动分支的参数跳变。存档包含这些目标值，自动核验它们在插入和承压期间确实恒定。

原生 monitor 结合开口包络、方向、近底几何、低速和连续真实底挡加载建立到底证据；数值以[成功判定](success_criteria.md#install-ram)为准。
卸载后的弹片静摩擦可能分担重量，历史证据的保留/清除由 monitor 决定。专家另外执行松手和退离检查；一个位置快照不足以证明受力到底。

抓取前先张开拇指和食指，在腕位置误差 ≤3 mm、姿态误差 ≤0.03 rad 后保持腕部目标不动，
用 0.6 s 平滑收拢再夹紧；未到位则失败。两指尖 site 中心间距约 52 mm，不能作为指腹净空解释。
该过程沿用 `grasp` 阶段名，其余三指保持避障弯曲。时序变化应按真实 phase 和时间戳对齐。
底层录制入口为 `scripts/workcell/record_install_ram_example.py`，场景执行入口为 `scripts/workcell/view_install_ram.py`。
`minimum_palm_down_cosine` 的全流程统计可能包含初始收臂转掌，应结合 phase 阅读；
持物搬运、对准、下插和承压期间的姿态检查与初始收臂使用不同口径。

<a id="ram-recording"></a>
## 录制模式与产物

录制入口提供三种产物模式，物理步频均为 500 Hz，主 HDF5 的状态/触觉为 100 Hz、三路 Raw RGB 默认各 30 Hz：

| 模式 | 命令选项 | 最终保留内容 |
| --- | --- | --- |
| 精简示例 | `--compact` | `raw/` 中 HDF5、校验 JSON、结果 JSON；`review/` 中合成与全局两个 MP4；`curves/` 中五指力 PNG，共六文件 |
| 完整诊断 | 不加 `--compact` 或 `--raw-only` | 上述数据及同次运行的 `raw/force_trace_500hz.npz`、逐帧复核、CSV、`processing/`、交付清单和来源信息 |
| 仅 Raw | `--raw-only` | HDF5、任务结果与最小校验摘要；不生成展示视频、曲线或独立 500 Hz 力档案 |

精简模式先在工作目录完成完整采集、500 Hz 力诊断、Raw 和视频验证，再仅发布六文件并清理工作目录。
因此其结果 JSON 保留验证结论，**不保留逐物理步 NPZ**；需要重新计算高频力统计时应使用完整诊断模式：

```bash
pixi run --as-is record-install-ram-example -- --output-dir datasets/install_ram_diagnostics_new
pixi run --as-is record-install-ram-example -- --raw-only --output-dir datasets/install_ram_raw_new
```

`--raw-only` 不能与 `--compact`、`--replace-existing`、`--diagnostics-dir` 同用。
完整诊断中的 NPZ 保存五指原始法向力、有符号双轴切向力及卡槽载荷，录制器核对它与主 HDF5 在相同时刻的力值。
复核视频为 10 Hz；100 Hz 主存档和 10 Hz 视频均不能代替 500 Hz 原始力档案进行高频振荡分析。
五指触觉来自接触求解器，允许未接触手指为零；热图色标饱和不改变底层原始值。

指尖与内存条采用带滚动／扭转阻力的有限接触面近似。它只定义在本任务的接触对中，不改变 shared 触觉垫几何或其他任务。接触对使用 `solimp="0.99 0.999 0.00055"` 和 `solreffriction="0.004 1"`，抑制固定夹持时软接触产生的缓慢滑移；摩擦系数、共享垫几何和卡槽参数不变。手臂和手指目标在每个 2 ms 物理步连续插值，握力反馈只在槽外建立夹持和之后松手时使用。原始存档和曲线没有时间滤波；诊断依据及参数对照见 [力振荡排查](#ram-force-diagnosis)。

这是刚性模块与简化卡槽的机械插入任务。定位键和止挡参加碰撞，端部卡扣采用固定打开状态；本版不模拟卡扣自动闭锁、电气连通、逐根金属触点弹性或 PCB 弯曲。间隙与接触参数服务于稳定仿真，不能用于连接器制造或插入力标定。

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .pixi/envs/default/bin/python -m pytest -q tests/test_install_ram.py
```

腕部画面中的移动黑色斑点经同状态开/关阴影对照确认来自投射阴影渲染。任务在加载共享模型后将 `light_castshadow` 关闭，对录制、辅助近景和交互查看一致生效；保留共享相机位姿、内参、光照和真实物体遮挡，不修改碰撞或触觉。五指力图按实际阶段时间用橙色标出下插、粉色标出到底承压。

## Shared 配置继承

维护中精简示例使用 `datasets/install_ram_example`，新运行使用独立输出目录。场景直接包含 shared MJCF，双臂从 `shared/posture.py` 的统一屈肘姿势开始，通过执行器接近和抓取，不在 reset/prepare 中重设关节。共享 home 的掌面方向可与抓取方向不同，因此录制验收仅在持物搬运、对准、下插和到底承压阶段要求掌面朝下，与 500 Hz 物理检查一致。

各模式的文件保留范围见上方三模式表。全局相机仅用于展示，不加入训练观测，不改变 Raw schema。
新姿势和外观改变轨迹及图像分布；接口兼容不代表历史模型性能不变。

## 黑色散热马甲与短转接近

`grasp.py` 使用另一侧的正手夹持标定，接近由位置与 SO(3) 短旋转插值驱动，避免从 shared home 插值到另一 IK 分支时的长转。中指、无名指和小指折叠避让长 PCB，拇指/食指到位后夹紧。共享初始姿势不变；新轨迹和图像分布与历史数据不同，不保证旧模型的成功率不变。

黑色 PCB 和阳极氧化散热马甲带倒角、顶边棱线、金属刻线及文字；主板增加 ATX 插座、供电器件、芯片引脚、贴片元件和安装螺钉。目标插槽的槽口四周增加银色镶边，便于在黑色主板上辨认插入口，镶边不侵入插入通道。新增外观几何不参与接触、质量为零，原有单槽、金手指碰撞形状、物体质量和插入阻力参数保留。装饰壳体是现有薄板/芯片包络的视觉近似，不额外模拟散热马甲的弹性。

按 USB 示例的简洁布局生成。下面是**明确允许替换维护中开发样例**时才执行的命令，
不是日常评估命令，不触及既有 H50 报告；新实验应使用上方的新输出目录：

```bash
pixi run --as-is record-install-ram-example -- --compact --replace-existing --output-dir datasets/install_ram_example
```

新一轮完整验证后才覆盖目标，交付仅含 `raw`（HDF5、校验和结果 JSON）、`review`（相机与触觉合成、机器人全局两个 MP4）、`curves`（五指法向/切向力 PNG）。不保存旧示例或版本子目录；验证细节嵌入新的结果 JSON。

<a id="ram-dimensions"></a>
## 尺寸依据与物理模型

尺寸统一以米为单位，模块采用 DDR4 UDIMM 大致真实外形，没有整体放大。机器人、双手和传感器校准沿用 shared。

### 尺寸依据

- [Kingston KVR32N22S8/8 官方数据表，第 2 页](https://www.kingston.com/datasheets/KVR32N22S8_8.pdf)
  标出 PCB 长 133.35 mm、高 31.25 mm，防呆缺口中心距左端 72.25 mm，
  缺口宽 1.50 mm、高 3.80 mm。数据表说明该类 DIMM 为 288 针。
- [TE Connectivity Memory Sockets 官方选型表](https://www.te.com/content/dam/te-com/documents/consumer-devices/global/memory-sockets-final-flyer.pdf)
  给出 DDR4 DIMM 的板厚 1.40 ± 0.10 mm、288 针、0.85 mm 间距。
- [TE Connectivity DDR4 DIMM 官方产品页](https://www.te.com/en/product-1-2308107-1.html)
  描述竖直插入、一个偏心定位键和两端退出卡扣。
- [TE DDR4 DIMM Sockets 官方介绍](https://www.te.com/content/dam/te-com/documents/consumer-devices/global/ddr4-flyer-en.pdf)
  给出 288 个触点、0.85 mm 间距，并说明接口对应 JEDEC MO-309 模块。

以上链接于 2026-09-16 核对；场景是参数化的装配仿真，不声称满足连接器的生产公差。

| 项目 | 场景尺寸 | 依据或处理 |
| --- | ---: | --- |
| PCB 长 × 高 × 厚 | 133.35 × 31.25 × 1.40 mm | 制造商标称外形 |
| 含两面芯片的总厚度 | 3.40 mm | 每面 1.00 mm 高的简化封装 |
| 金手指 | 两面各 144 个 | 0.85 mm 节距的视觉条带 |
| 偏心防呆缺口 | 1.50 mm 宽、3.80 mm 高 | 距左端 72.25 mm |
| 卡槽内部长度 | 133.95 mm | 每端留 0.30 mm 仿真间隙 |
| 卡槽内部宽度 | 2.00 mm | PCB 每面留 0.30 mm 仿真间隙 |
| 卡槽插入深度 | 6.00 mm | 仿真选择的机械止挡位置 |
| 槽内定位键 | 1.00 mm 宽、3.50 mm 高 | 保留 0.25 mm 横向和 0.30 mm 顶部间隙 |
| 主板 | 200 × 140 × 1.60 mm | 任务夹具，非某一主板产品的复刻 |
| 内存质量 | 20 g | 含芯片模块的近似质量 |

### 几何与坐标

内存 `ram` 和卡槽 `ram_socket` 使用相同的局部方向：X 沿长边，Y 穿过
PCB 厚度，Z 向上，插入方向为 −Z。内存刚体原点位于 PCB 外形中心，底边
中心 site `ram_bottom` 位于 `(0, 0, -0.015625)`。抓取参考 site
`ram_grasp` 位于 `(0, 0, 0.006)`，抓点沿长边居中以减小重力滚转矩。

卡槽口 site `ram_socket_mouth` 位于世界坐标 `(0.500, -0.150, 0.704)`。
机械止挡 `ram_socket_backstop` 顶面为世界 Z = 0.698 m，因此模块底边下降
6 mm 后到底。偏心键位于模块/卡槽局部 X = 0.005575 m，反向模块会被实体键
阻挡。

槽内空腔由两个长壁、端部导向、定位键和底部止挡组成，没有跨过空腔的实体
凸包。PCB 下边缘分成两段，缺口是真实碰撞空间。初始内存由两个固定支承
竖直托住，两面可以接近；模块具有自由关节，没有附着到手上的焊接约束。

### 物理近似和范围

- PCB 和芯片为刚体盒体；省略 PCB 弯曲、焊点、细小倒角、真实金手指下边缘
  的微小弧度以及两端锁扣凹槽。金手指与板上电阻/走线主要用于视觉。
- 卡槽两端卡扣以打开状态固定表示。任务评估对齐、插入及机械到底，不包含
  卡扣自动闭合、拔出保持力、电气导通或主板通电检测。
- 288 根弹性触点由两组被动侧向弹簧等效，每组有两个圆滑接触面，刚度
  3000 N/m、阻尼 5 N·s/m、等效质量 5 g。标称侧向让位约 0.3 mm，接触
  摩擦系数 0.35。在慢速居中插入的机械测试中，实际轴向滑动阻力约 0.61 N；
  机器人实际轨迹的测量以示例原始记录为准。到底阻力由独立实体止挡产生。
  这些是降阶的仿真参数，没有根据厂家规定标定插入力。0.30 mm 单侧间隙
  是用于毫米尺度刚体装配的数值余量，不是现实插座的制造间隙。
- 原始 `scene.xml` 初始模块中心为 `(0.470, -0.290, 0.755635)`，待抓支架顶面
  Z = 0.740 m。自动演示如使用不同的抓取起始标定，需在对应轨迹元数据记录。

可执行尺寸常量和 `DIMENSIONS_M` 位于
`src/kaihand_tactile_env/tasks/install_ram/config.py`；完整场景位于同目录
`scene.xml`。

<a id="ram-force-diagnosis"></a>
## 原始切向力诊断

本节规定力信号口径和工程诊断，成功判据见[RAM 成功判定](success_criteria.md#install-ram)。

本页保留当前 `install-ram` 实现的诊断结论和验收合同。早期逐次 A/B 数字和临时候选结果未随本页交付；
复核历史数据时需使用其原始归档、模型指纹和对应源码。

### 问题结论

早期实现的切向力振荡并不是单一的采样或绘图问题，主要来自偏心夹持、有限接触斑被简化为点接触，以及槽内持续进行腕部姿态和握力修正。仅提高求解器迭代数、减小物理步长或对腕部目标插值，不能稳定消除该问题。

当前实现采用以下任务局部修正：

- 抓点移到 RAM 长边中心附近，减少运输滚转。
- 指尖—RAM 使用具有滚动／扭转阻力的有限接触面近似。
- 手臂和手指目标在每个 2 ms 物理步连续插值。
- 在槽外建立最终握力并完成对准；进入 `insert` 后固定腕部 XY、腕部方向和手指关节目标，只改变腕部 Z。
- 仅对本任务的指尖—RAM 接触对使用 `solimp="0.99 0.999 0.00055"` 和 `solreffriction="0.004 1"`，不修改 shared 指垫或其他任务。
- 到底载荷在 0.8 秒内平滑建立；失去开口对准时停止，不在槽内扭动补救。

这些变化修改的是仿真接触和执行器目标，不对保存后的力信号做滤波、裁剪或重采样。

### 数据口径

完整诊断模式（`record-install-ram-example` 不加 `--compact` 或 `--raw-only`）同时保存：

- 共享 Raw HDF5 中的 100 Hz 状态、命令和求解器触觉；
- `force_trace_500hz.npz` 中同一次 rollout 的逐物理步五指 Fn、有符号双轴 Ft 和卡槽载荷；
- 任务阶段、插入深度、弹片阻力、底挡载荷以及腕部／手指目标。

`--compact` 在发布前执行这些核验，但只保留六文件示例和嵌入结果 JSON 的验证结论，清理包含 NPZ 的工作目录。
`--raw-only` 也不生成独立 500 Hz 力档案。需要复算以下高频统计时，必须保留完整诊断模式的 NPZ；
三种模式的命令与目录区别见[录制模式](#ram-recording)。

Ft 使用共享触觉垫的局部切向基，先合计每指有符号两轴分量，再计算向量模或相邻向量差。统计不得跨动作阶段拼接，也不能用 100 Hz 数据推断 100 Hz 以上频谱。

纯统计入口为：

```text
src/kaihand_tactile_env/tasks/install_ram/diagnostics.py::summarize_force_trace
```

### 当前验收

插入和承压阶段必须同时满足：

- 腕部 XY、腕部旋转和手指关节目标保持恒定；
- 3.5–5.5 mm 插入深度窗口内，实际承载的拇指和食指 Ft 标准差不超过 0.015 N；
- 同一窗口的 2–10 Hz Ft 频带 RMS 不超过 0.005 N；
- 到底、开口包络、方向、速度和连续承载条件通过；
- 松手后五指载荷归零，RAM 保持终态就位。

力曲线、CSV、HDF5 和 NPZ 始终保留原始值。频谱统计可以使用去均值和 Hann 窗，但不能把处理后的信号写回 Raw 或当作传感器输出。

### 复核原则

- 修改接触参数、抓点或插入控制后，必须重新生成一条完整记录，不能沿用旧版本的统计结论。
- 平滑的力曲线不能代替任务成功；几何对准、真实底挡载荷和卸载后稳定性必须独立通过。
- 仿真阈值用于当前模型的数值质量控制，不代表真实 DIMM、卡槽或机械手的硬件额定值。
