# 五个独立变体任务

当前五个变体为 `usb-unplug`、`whiteboard-wipe-horizontal`、`bulb-unscrew`、
`whiteboard-wipe-horizontal-line`、`bulb-screw-inplace`，均与原任务分开采集、转换、训练和评估。
原生成功的所有数值/时序条件、专家额外条件见[十三任务成功判定](success_criteria.md)；
阶段诊断另见 [staged_v1](../workflows/staged_evaluation.md)，不以阶段结果替换原生标签。

落地及验收顺序为 `usb-unplug` → `whiteboard-wipe-horizontal` → `bulb-unscrew`。
这三项是独立场景和独立 task identity，原 `usb-insert`、`whiteboard-wipe`（45°）、
`bulb-screw` 均保留。禁止把旧示教倒放、旧数据重标 task 或旧 deployment 改名冒充新任务。
初版闭环、Raw 和转换验收原报告位于 `runs/task_variants_validation_20261010/report.md`；
该历史产物未随当前工程保留，路径仅供追溯。
其中取出灯泡的初版验收对应返回卡座场景；当前已更新为下文的无返回架桌面版，
历史报告和数据保持原样，不能作为桌面新版的数据使用。
另新增 `whiteboard-wipe-horizontal-line`，仅擦除水平白板上原 Z 字的顶部横线。
它与完整 Z 字的 `whiteboard-wipe-horizontal`、原斜板 `whiteboard-wipe` 独立并存。
2026-10-15 再新增 `bulb-screw-inplace`，初态已在灯座入口，只执行旋入，不替换原 `bulb-screw`。

## 拔 USB：usb-unplug

当前流程（`usb-unplug-held-clear-2s-v2`）：验证已插入的静态初态 → 接近外露塑料柄
→ 双指对向抓稳 → 沿 socket 退出轴低速拔出 → 完全离开插座，右手持有且不触桌连续保持 2 秒。
不再要求转移到原桌面位置、放稳松手或退让回位。

初态只在 reset 时设置插头自由关节和两个被动弹片的弹性位移，再检查静态稳定、
接触和穿透。不添加锁扣、物体 weld、吸附或拔出助力；采集开始后只改变机器人控制目标。
seed 保持可复现，但不能对已经插入的插头沿用桌面自由 XY/yaw 抖动。

原插入任务的轴向阻力符号不能用于退出：新 monitor 保留 socket 局部坐标下的
signed force，分别计算向内与向外阻力。新版物理成功只要求当前真实完全清离且手持、
无桌面接触共同连续 ≥2 秒；双指抓稳是专家动作，不是新增的模型压力门槛。
初始场景仍由 reset 安装，不能将任意自由插头初态冒充该任务。

原生具体阈值见 [USB 拔出成功判定](success_criteria.md#usb-unplug)。
旧版曾要求放回标定桌面位置、释放和退手；旧示教、deployment、H50 报告保持原样，
必须按各自归档协议解释，不与新版手持 2 秒结果直接混算。

物理范围沿用原任务的简化 target socket / 被动弹片模型：排插外壳、相邻插孔、
开关和电缆是无碰撞的视觉件。验收只证明该模型下的名义拔出流程成立，不能外推
为真实 USB 插头的机械合规性，也不能作为整台排插的通用碰撞安全结论。

## 水平擦白板：whiteboard-wipe-horizontal

流程：拿起擦子 → 抬升 → 移到白板上方，毛毡面朝下 → 建立实际法向预载
→ 按板坐标系的 Z 路径带载擦除 → 卸载抬起 → 放回原桌面位置并释放 → 返回共享初始姿势。

白板是固定、水平朝上的桌面板，不是可移动白板；中心 `(0.52, -0.12, 0.692)` m，
板法向 `+Z`，表面高 `0.700` m。旧斜支架只从新场景删除，原斜板 scene 不变。
板、墨迹、擦子姿态与控力全部通过 instance config 指定；切换场景不修改模块全局常量。

保留原 25 处 Z 形墨迹和真实摩擦清洁门槛：实际接触、法向载荷、切向载荷、
滑动速度、累计功、行程和带载时间。轨迹经过墨迹但没有满足门槛，不会自动清除。
“从上到下”在此指板的局部坐标，不指水平板上不存在的世界 Z 高度下降。
成功仍要求实际拿取/抬升、墨迹全部清零、桌面支持下自由释放，而非只清零计数。
新水平任务还要求右臂真实返回初始姿势；初态擦子在桌面上不算一次完成后的释放。
原生与专家的抓持历史、抬升/释放和回位门槛见
[水平 Z 字成功判定](success_criteria.md#whiteboard-wipe-horizontal)。
当前 expert 目标法向载荷为 `1.85 N`，建立预载门槛为 `1.60 N`；清洁物理门槛
沿用原任务。回位先充分张手，再恢复已验证的 elevated approach IK 分支，最后
返回 shared home；不直接跨擦板末端 IK 分支折回。两段回位均检查实际关节速度
`<70°/s`、加速度 `<100°/s²`，末态臂误差 `<0.01 rad`、手误差 `<0.06 rad`。

## 水平单横线：whiteboard-wipe-horizontal-line

流程：共享初始姿势 → 拿起板擦并真实抬升 → 移到顶部横线起点上方
→ 毛毡面朝下、建立法向预载 → 仅沿这条横线带载擦净 → 抬离白板
→ 放回原桌面位置、由桌面承重后释放 → 展手并返回共享初始姿势。

沿用水平板的固定位置、相机和接触清洁模型，保留原 Z 字的 `ink_0` 至 `ink_7`：
板局部 `x = 0.07 m`，`y = +0.08 → -0.08 m`，名义长度 16 cm、8 个墨迹片段。
不把横线重新移到板中心，也不保留隐藏的斜线或下横线。seed 仍只平移整条墨迹；
`--fixed-stains` 使用原顶部横线的固定位置。控制轨迹只包含预加载和顶部横线，
在两端保留带载擦拭余量，并按墨迹平移裁剪至板框允许范围。

成功要求真实抓取与抬升、8 片墨迹全部清零、板擦在桌面支持下自由释放及手臂回位。
法向载荷、切向载荷、摩擦功、滑动距离和带载时间门槛全部沿用原物理模型，
不能通过走完路径、静态压住或直接清零墨迹判成功。
完整继承关系见[水平单横线成功判定](success_criteria.md#whiteboard-wipe-horizontal-line)。

任务、pattern `z_top_bar_v1`、mechanics version
`horizontal_fixed_board_top_line_contact_v1` 和场景指纹均独立。
统一采集、π0.5 / LeRobot v3 转换、π0.5 部署评估均使用完整新 task 名；
Raw 诊断组为 `whiteboard_wipe_horizontal_line`，清洁状态为 8 维。
旧 Z 字 Raw、checkpoint 或 deployment 不能重命名为此任务；新任务需单独采集、
转换和训练。完整采集与转换命令见[统一命令速查](../workflows/pipeline_quickstart.md#5-水平单横线任务)。

## 取出灯泡：bulb-unscrew

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
峰值，不能先放掉灯泡再补写“已放稳”。末态 home 门槛为臂 `1°`、手 `4°`、
关节速度 `2°/s`；手的容差来自速度执行器/静摩擦的实际静止残差约 `3.20°`。

桌面新版以 `placed_on_table` 为成功里程碑，并升级 mechanics version、场景指纹和
训练 instruction；旧的返回卡座 Raw/checkpoint/deployment 不能重命名为新版。
重新采集、转换并训练后，才可用于新版模型闭环评估。

桌面新版成功采集视频为
`/nas/chenxianchi/datasets/sim/task_demo/bulb-unscrew-tabletop.mp4`（38.3 秒，1080p/30 fps）；
对应 Raw 位于
`/nas/chenxianchi/datasets/sim/task_demo/_metadata/bulb-unscrew-tabletop/collection`。
目录名用于区别旧录像，实际 task identity 仍为 `bulb-unscrew`。
该 Raw 的完整成功门槛、每物理步审计、三路相机首尾一致性及源 SHA 均通过；
实际 π0.5 环境的只读转换预检通过，规划 1148 个训练帧，尚未写出训练数据集。
旧 `bulb-unscrew.mp4` 和其卡座版 Raw 保留，但不可混入桌面新版数据。

## 灯泡原位旋入：bulb-screw-inplace

初态已对正灯座、入口刚啮合，未到座、未拧紧、灯灭，双手在 home。
流程为接近 → 原位五指抓稳 → 实际旋入/换指 → 到座承力停转确认 → 松手验稳并回位。
不执行桌面拿取、搬运、重新入牙；也不将亮灯、四段行程或回位当作独立成功证明。

它与 `bulb-screw` 及 `bulb-unscrew` 有独立 task / mechanics / scene 身份。
详细初态、五步流程、CPU 验证和本地视频见[原位旋入任务页](bulb_screw_inplace.md)，
数值判据见[原位旋入成功判定](success_criteria.md#bulb-screw-inplace)。
物理成功不强制回 home；专家完整流程另外检查收尾，不能混用两种 `success`。
当前 seed 只保留运行身份，不随机灯座入口位姿。

## 统一命令

先进入 `kaitactilesim_unified` 根目录。以下 `TASK` 可取上述五个独立变体名字；
新原位任务也是 `--task bulb-screw-inplace`，建议逐个单条验收后再批采。
本机环境与缺失依赖说明见[环境与工作目录](../workflows/pipeline_quickstart.md#环境与工作目录)。
这里使用 `pixi run --as-is` 运行已有环境，避免隐式安装；不代表缺少 pyarrow 时也能执行正式训练数据转换。
硬件渲染必须显式传本节点的 EGL vendor 配置：

```bash
__EGL_VENDOR_LIBRARY_FILENAMES=/cpfs_infra/user/chenxianchi/.config/kaitactilesim/nvidia_egl_vendor.json \
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is collect -- \
  --task TASK --check --render-backend hardware

__EGL_VENDOR_LIBRARY_FILENAMES=/cpfs_infra/user/chenxianchi/.config/kaitactilesim/nvidia_egl_vendor.json \
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is collect -- \
  --task TASK --episodes 1 --workers 1 --seed 20261010 \
  --output-dir /path/to/TASK_raw --staging-root /path/to/TASK_staging \
  --render-backend hardware

/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is convert-dataset -- \
  --task TASK --input-dir /path/to/TASK_raw --format pi05 \
  --output-dir /path/to/TASK_pi05 --expected-episodes 1 \
  --openpi-root /cpfs_infra/user/chenxianchi/code/openpi \
  --pi05-python /cpfs_infra/user/chenxianchi/code/openpi/.venv-pi05/bin/python \
  --verify-source-hash --dry-run
```

确认转换计划正确后，去掉 `--dry-run` 执行实际转换。
上例显式使用当前机器的 OpenPI 环境；也可以先配置 `configs/runtimes.local.toml`，
再用 `--runtime-config configs/runtimes.local.toml --runtime pi05` 代替这两个路径参数。
本地 runtime 配置文件不是仓库自带的必备文件，不要直接引用尚未创建的示例路径。

拔 USB 和水平白板已接入统一采集、π0.5 转换入口，并用成功采集的三路相机 Raw
通过了实际 π0.5 Python 的 `--validate-only --verify-source-hash` 预检：分别规划
710 和 2507 个训练帧。示范 Raw 位于
`/nas/chenxianchi/datasets/sim/task_demo/_metadata/TASK/collection`；这两条数据的
代码接入和预检通过不代表已自动执行完整转换或训练。

无 GPU 时用 `--render-backend software` 并去掉 vendor 前缀；正式批采需确认输出盘
和 staging 盘都有足够空间，不建议降低磁盘安全阈值。五个变体均输出共享 Raw，
state/control 100 Hz、camera 30 Hz，以每物理步 observer 保证真实时钟。
额外诊断保存在 `usb_unplug/`、`whiteboard_wipe_horizontal/`、`bulb_unscrew/`。
水平单横线使用独立的 `whiteboard_wipe_horizontal_line/` 诊断组。
原位旋入使用独立的 `bulb_screw_inplace/` 诊断组，并保留 reset 初始化证据。

π0.5 转换固定 `head + right_wrist`，沿用 shared-task adapter 的动作表示，
保留 task、mechanics version、模型指纹及实际参数来源。也支持独立 `lerobot-v3` 转换。
目前只注册 π0.5 模型部署评估，不据此宣称 EgoSteer/EgoTouch 新任务已适配。
新任务部署还严格比对训练转换快照中的 instruction 与当前任务 instruction；
如自定义训练 prompt 或后续改变任务指令，不能直接用默认部署入口替换训练时指令。

使用新数据训练匹配任务的 checkpoint 后，仍走
`deploy-policy prepare --task TASK --model-family pi05` → `deploy-policy run`，
执行步数仍用 `--execute-steps K`（`1 <= K <= horizon`）。完整部署命令见
[统一命令速查](../workflows/pipeline_quickstart.md)。训练和 checkpoint 模型闭环评估
不属于新增 expert 的物理 smoke 验证，不能用 expert 成功率冒充模型成功率。
