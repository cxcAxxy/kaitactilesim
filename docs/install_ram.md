# 内存条安装任务

`install-ram` 是独立的 DDR4 台式机 DIMM 安装场景。模块长约 133.35 mm、高 31.25 mm，PCB 厚 1.40 mm，包含两面芯片、金手指和偏心定位缺口。卡槽具有真正开放的插入口、对应定位键、两侧壁与底部止挡；主板安装在共享桌面上。尺寸来源及模型近似见 [尺寸说明](install_ram_dimensions.md)。

场景通过 MJCF include 直接引用 `shared/mjcf/robot.xml` 和 `shared/mjcf/control.xml`。机械臂、左右手、相机安装参数和触觉垫沿用 shared；任务只新增内存条、卡槽、主板、上料支架和检查用近景相机 `ram_closeup`。采集使用共享的 `head`、`left_wrist` 和 `right_wrist`，近景相机不替代任何共享相机。

```bash
# 查看场景
pixi run view-install-ram

# 执行夹持、提升、移到卡槽、插入、释放和终态检查
pixi run run-install-ram
pixi run run-install-ram -- --headless

# 只检查场景启动
pixi run check-install-ram

# 生成一条示例，默认保存到 datasets/install_ram_example
pixi run record-install-ram-example

# 成功后更新已有RAM示例；旧例原始记录和曲线保存在新例provenance内
pixi run record-install-ram-example -- --replace-existing

# 在新目录复现，已有示例不会被覆盖
pixi run record-install-ram-example -- --output-dir datasets/install_ram_example_new
```

若环境已安装，也可以直接使用 `.pixi/envs/default/bin/python scripts/workcell/view_install_ram.py --headless --run-task`。无窗口录制需要设置 `MUJOCO_GL=egl`；默认 Pixi 录制命令已设置该变量、单线程物理计算和四线程软件渲染参数。

模块沿世界 X 轴伸展、Y 轴为厚度方向、Z 轴向上，插入沿 −Z。初始状态使用竖直上料支架，双手从 home 位置张开平放开始。`prepare()` 只计算抓取目标，不改机器人或内存条的 qpos；右手先用 4 秒移到支架上方并预成形，再用 3 秒下降、静置 0.4 秒后闭指。抓点位于内存条长边中心附近，以减少重力引起的滚转；运输和插入时保持手背朝上、掌面朝下的正手姿势。控制器使用仿真真值位姿完成已知初态的演示，动作通过机器人执行器和原生接触传递到自由运动的内存条。

卡槽内有两组被动弹片等效机构，滑动关节、弹簧和阻尼产生侧向预紧，实际库仑接触摩擦抵抗向下插入。在 3.5–5.5 mm 深度范围可观察滑动阻力；约 6 mm 到底后，控制器进入独立的 `bottom_press` 阶段，根据实际底挡力缓慢加压并保持。两阶段的力均来自接触求解，不按阶段直接写入物体外力或触觉读数。力的绝对值是仿真参数，不是真实连接器插入力标定。

先对准，再下插：机器人在槽口上方约 0.5 mm 处建立约 4 N 的夹持力，固定手指目标后完成最后对准，连续至少 0.2 秒满足位置、方向和低速度条件才开始插入。此后 `insert` 与 `bottom_press` 全程保持腕部 XY、腕部旋转和手指关节目标不变，仅改变腕部 Z。过程中失去开口对准就停止，不通过扭动腕部补救。到底目标载荷在 0.8 秒内平滑建立，采用限速的小增益反馈，没有普通移动分支的参数跳变。存档包含这些目标值，自动核验它们在插入和承压期间确实恒定。

安装判据同时检查整条插入边的开口包络、方向、约 6 mm 的插入深度、低速度和真实到底记录：底挡承载至少 1.2 N、连续 0.4 秒，才取得到底确认。机器人控制器继续保持足够承载后松手退离；最终对齐和低速度还须连续满足 0.3 秒。卸载后弹片静摩擦可能分担模块重量，因此允许在已验证到底、且几何仍在到底范围时保留确认。离开到底范围、跳步观测或重置都会清除该证据。重力静置或一个位置快照不能证明受力到底。

`datasets/install_ram_example` 的交付结构参考 USB 与抽牌示例：

- `raw/`：本次动作的 HDF5、JSON、任务结果；包含状态、命令、模块位姿、相机标定、共享触觉及实际接触力。`force_trace_500hz.npz` 保存同次运行每个物理步的五指原始法向力、有符号双轴切向力及卡槽载荷，JSON 保存统计与来源。
- `review/review.mp4`：头部／右腕图像与五指法向／切向触觉的组合回放。
- `review/frames/`、`review/raw/`、`review/frames.csv`：图像、热图、原始有符号切向力 NPZ 和时刻对应关系。
- `curves/`：未平滑的五指法向与切向力曲线及 CSV，以及插入深度／弹片阻力／到底承载的阶段图和旧新原始力对比。
- `processing/`、`delivery_manifest.json`：验证报告、来源信息和文件校验。

物理步频为共享的 500 Hz，主 HDF5 保存状态与触觉 100 Hz、三路 Raw RGB 各 30 Hz，另存同次运行的 500 Hz 原始力；复核视频为 10 Hz。录制器逐值核对两份力数据在相同时刻的一致性。触觉曲线来自接触求解器，五指均保留，未接触的手指允许为零；热图的色标饱和不改变底层原始力。10 Hz 视频和 100 Hz 主存档不能展示每一个 500 Hz 物理瞬态，振荡诊断使用完整物理步记录。

指尖与内存条采用带滚动／扭转阻力的有限接触面近似。它只定义在本任务的接触对中，不改变 shared 触觉垫几何或其他任务。接触对使用 `solimp="0.99 0.999 0.00055"` 和 `solreffriction="0.004 1"`，抑制固定夹持时软接触产生的缓慢滑移；摩擦系数、共享垫几何和卡槽参数不变。手臂和手指目标在每个 2 ms 物理步连续插值，握力反馈只在槽外建立夹持和之后松手时使用。原始存档和曲线没有时间滤波；诊断依据及参数对照见 [力振荡排查](install_ram_force_diagnosis.md)。

这是刚性模块与简化卡槽的机械插入任务。定位键和止挡参加碰撞，端部卡扣采用固定打开状态；本版不模拟卡扣自动闭锁、电气连通、逐根金属触点弹性或 PCB 弯曲。间隙与接触参数服务于稳定仿真，不能用于连接器制造或插入力标定。

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .pixi/envs/default/bin/python -m pytest -q tests/test_install_ram.py
```

腕部画面中的移动黑色斑点经同状态开/关阴影对照确认来自投射阴影渲染。任务在加载共享模型后将 `light_castshadow` 关闭，对录制、辅助近景和交互查看一致生效；保留共享相机位姿、内参、光照和真实物体遮挡，不修改碰撞或触觉。五指力图按实际阶段时间用橙色标出下插、粉色标出到底承压。
