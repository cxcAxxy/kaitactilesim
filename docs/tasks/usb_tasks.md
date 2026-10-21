# USB 任务指南

本页负责 `usb-insert` 与 `usb-unplug` 的场景、专家动作、随机化、接触模型及保存后诊断。
原生成功的数值只在[任务成功判定](success_criteria.md#usb-insert)维护；本页保留控制目标和工程诊断阈值，不能替代成功规则。
通用采集、转换、部署与评估命令见[流水线速查](../workflows/pipeline_quickstart.md)，环境见[环境说明](../workflows/environment.md)；
模型专项配置见[模型专项评估](../workflows/model_specific_evaluation.md)。

| task | 初态 | 专家流程与身份 |
| --- | --- | --- |
| `usb-insert` | 桌面自由插头 | 拿取、插入、主动触底、卸力和释放 |
| `usb-unplug` | 已插入且物理稳定 | 抓稳、退出、手持清离；独立协议 `usb-unplug-held-clear-2s-v2` |

常用章节：[插入](#usb-insert)、[拔出](#usb-unplug)、[初态随机化](#usb-randomization)、
[接触前噪声](#usb-precontact-noise)、[阻力模型](#usb-contact-resistance)、[力学检查](#usb-contact-mechanics)、[Cleaning](#usb-cleaning)。

<a id="usb-insert"></a>
## 插入：usb-insert

`usb-insert` 从桌面自由插头开始，经正手抓取、抬升、搬运、悬停对准、低速插入、主动触底、卸力、松手和撤离完成专家流程。
原生模型与专家完整结果的区别见[USB 插入成功判定](success_criteria.md#usb-insert)。

任务专用预览与专家执行命令：

```bash
pixi run --as-is run-usb-insert
# 无窗口执行并保存报告；输出文件必须尚不存在
pixi run --as-is run-usb-insert -- --headless --result-json artifacts/usb_run.json
```

自动入口显式使用 `--run-task --plug-for-insertion`。USB 初态为标记朝下、yaw 330°，连接端朝机器人前方偏右。
只在首个物理步前设置朝向；空闲查看场景和旧 `--plug-face-down` 预设保留自己的初态。
取物时小拇指在远端、拇指和食指在近端；插入时掌心朝下，拇指和食指在上、小拇指在下。
从共享收臂初态到接近前悬停路点的名义时间为 2.1 s；具体阶段以本次保存的时间戳和 phase 为准。

搬运后先把尖端移到孔口上方 100 mm。横向误差 <0.08 mm、姿态误差 <0.004 rad、
高度误差 <0.15 mm、线速度 <1 mm/s、角速度 <0.01 rad/s 连续保持 0.6 秒后，
才下降到孔口上方 2 mm 并进入低速插入。停顿时继续反馈修正，失准重计时、超时失败。
悬停和下降均记为 `align`，参数定义在任务的 `motion.py` 并写入结果 `motion_parameters`。

高位到孔口的名义下降时间为 3 秒；距孔口 30 mm 内启用插入用肘部外展修正。
高位利用冗余自由度调整关节姿态并保持腕部目标，关节余量至少 10°；近口和插入阶段恢复至少 15°。
机械臂目标逐物理步平滑推进，只通过执行器和接触移动自由插头。
主动压底、卸力与释放的连续控制见[接触阻力与控制](#控制和任务完成)，
接触质量检查见[接触力学工程检查](#usb-contact-mechanics)。

Python 入口在 reset 后、首个物理步前初始化插入初态：

```python
from kaihand_tactile_env.shared.simulation import ArmHandSimulation
from kaihand_tactile_env.tasks.usb_insert.setup import initialize_for_insertion
from kaihand_tactile_env.tasks.usb_insert.execution import UsbInsertionExecutor

sim = ArmHandSimulation(scene="usb-insert")
initialize_for_insertion(sim)
result = UsbInsertionExecutor(sim).execute()
print(result.success, result.failure_reason)
```

CLI 与 Python API 的接触前噪声默认值不同，见[接触前控制噪声](#usb-precontact-noise)；
初始位置与 yaw 的范围、独立 seed 见[初态随机化](#usb-randomization)。
批量数据使用[统一采集入口](../workflows/data_collection.md)，保存后做[USB Cleaning](#usb-cleaning)，
视频按[统一回放流程](../workflows/data_visualization.md)从对应 Raw 生成。

2026-09-09 的姿态/耗时实测和 09-17 的悬停回归计数移入[历史验收档案](../archive/task_scene_history.md#usb-2026-09)。
历史单次动作只说明当时模型和固定初态的检查结果；当前验收需绑定实际 Raw、模型指纹与控制源码。
控制器使用仿真真值反馈，尚不覆盖视觉定位误差、翻面重抓或电气连接性能。

源码入口：`scripts/workcell/view_usb_insert.py`；任务配置、初态与专家执行位于
`src/kaihand_tactile_env/tasks/usb_insert/{config.py,setup.py,motion.py,execution.py}`。
EgoSteer 转换可读取 `task_collection_v2` 嵌套批次；历史平铺输入仍要求 `summary.json`、`usb_*.h5` 和配套 JSON 同目录，
具体选择规则由[数据转换](../workflows/data_conversion.md#任务识别)维护，输出目录必须独立于输入。

有效目标是白色排插靠近机器人一端的第一个 USB-A 孔，site 为 `usb_socket_mouth`，世界位置 `[0.620,-0.180,0.720] m`。
排插外观件不参加碰撞；其它孔位没有对应任务或电气功能。

排插视觉资产与生成器分别为 `src/kaihand_tactile_env/tasks/usb_insert/power_strip_visual.xml`、
`src/kaihand_tactile_env/assets/workcell/meshes/usb_power_strip_visual.json` 和 `scripts/workcell/build_usb_power_strip.py`。
需要用当前外观重绘已保存轨迹时，使用独立预览目录：

```bash
PYTHONPATH=src pixi run --as-is python scripts/workcell/preview_usb_appearance.py \
  /path/to/usb_episode.h5 --output-dir /path/to/new_preview
```

可加 `--reference-model BEFORE.mjb` 对比原模型。重绘不重算触觉，也不产生新采集；旧 Raw RGB 与重绘图像需分别标识。

抓取前先张开拇指和食指，腕位置误差 ≤3 mm、姿态误差 ≤0.03 rad 后才允许原地收拢，否则报失败。
两指尖 site 中心间距约 61 mm，表示 site 间距而非指腹净空。`close` 阶段的静止收拢时间为
`0.6 × grasp_ramp_s`（fast 为 0.6 s，baseline 为 0.75 s）。

<a id="usb-randomization"></a>
## 初态随机化

初始位姿随机化只作用于 USB 场景 reset 后、首个物理步之前的自由物体状态。它在世界 X/Y 方向分别采样有界平移，并围绕世界 Z 轴采样 yaw；高度、正反面、插座、摩擦和物体尺寸不变。

```bash
# X/Y 各 ±10 mm，基准朝向附近 yaw ±5°
pixi run --as-is run-usb-insert -- \
  --xy-jitter-mm 10 --yaw-jitter-deg 5 --seed 42

# 同时固定接触前控制噪声
pixi run --as-is run-usb-insert -- --headless \
  --xy-jitter-mm 10 --yaw-jitter-deg 5 --seed 42 \
  --precontact-noise-mm 0.5 --noise-seed 1 \
  --result-json /path/to/usb_random_seed42.json
```

`--seed` 决定物体初态，`--noise-seed` 决定另一套接触前运动噪声。只固定物体 seed 不保证动作轨迹相同；需要无噪声对照时增加 `--precontact-noise-mm 0`。噪声合同见 [USB 接触前控制噪声](#usb-precontact-noise)。

### 几何与控制合同

- `--xy-jitter-mm 10` 表示 X、Y 各自位于 ±10 mm，不是总宽 10 mm。
- `--yaw-jitter-deg 5` 表示基准朝向 330° 附近 ±5°。
- `initial_pose_randomization` 保存 seed、采样范围、实际 `offset_xy_m`、`yaw_offset_rad` 和首个物理步前的 `initial_pose_wxyz`。
- 抓取目标围绕 USB 同时变换位置和方向，保持标定的物体—手指捏持关系。
- 物体只在首个物理步前设置一次；执行期间由机器人执行器和接触推动，不重设自由物体状态。
- 随机 yaw 会调整机械臂 IK 初值以保持肘部外展，但不会直接写入实际关节状态。

初始翻面或明显倾斜仍会被拒绝；当前功能不包含翻面重抓，也不模拟视觉定位误差。

### 批量检查

`check-usb-random` 使用无噪声 API，串行检查固定初态、均匀样本和 X/Y/yaw 边界组合：

```bash
nice -n 10 pixi run --as-is check-usb-random -- \
  --samples 8 --seed 20260909 \
  --output-dir /path/to/new_usb_random_check
```

输出目录必须不存在。报告分别统计任务成功和质量合格，质量项包括关节余量、桌面／支架／插座碰撞、正手姿态、肘部外展、动作时间和腕部额外往返角程。

当前第一轮采集合同允许 X/Y 各 ±10 mm、yaw ±5°，但这不是对连续范围的成功保证。正式采集仍须保留失败样本及 seed，并对每条成功数据执行 Raw 验证和 USB Cleaning；启用接触前噪声后应单独统计成功率，不能沿用无噪声检查结果。

<a id="usb-precontact-noise"></a>
## 接触前控制噪声

USB 自动任务在接触物体前，可向手腕目标的世界 X/Y 平移加入平滑噪声。CLI 默认标准差为 0.5 mm；`--precontact-noise-mm 0` 可关闭。空闲查看场景不施加噪声。

```bash
# 使用默认噪声；每次自动生成并记录 noise seed
pixi run --as-is run-usb-insert

# 固定物体初态和控制噪声，保存完整控制轨迹
pixi run --as-is run-usb-insert -- --headless \
  --xy-jitter-mm 10 --yaw-jitter-deg 5 --seed 42 \
  --precontact-noise-mm 0.5 --noise-seed 1 \
  --result-json /path/to/usb_seed42_noise1.json

# 无噪声基线
pixi run --as-is run-usb-insert -- --precontact-noise-mm 0
```

物体初态的 `--seed` 与控制噪声的 `--noise-seed` 相互独立。只固定前者能复现摆放，不能保证控制轨迹相同；在模型和控制器版本相同的前提下，同时固定两种 seed 及相关参数才能复现加噪动作。初态定义见 [USB 初始位姿随机化](#usb-randomization)。

### 噪声合同

- X/Y 使用独立高斯结点，每轴截断到 `±min(3σ, 2 mm)`；默认范围为 ±1.5 mm。
- 结点间隔 0.30 秒，通过五次平滑插值生成连续偏移，不是逐物理步白噪声。
- 接近抓取位置时，偏移倍率平滑缩小到 0.2。
- 任一指尖与 USB 的真实法向力大于 `1e-6 N` 时锁存首次接触并停止采样。
- 已施加偏移在 0.12 秒内连续归零；此后即使短暂失去接触，本条动作也不重新启动噪声。
- 噪声只修改手腕目标的世界 X/Y，不修改 Z、旋转、手指目标、物体状态或传感器读数。

### API 与记录

`UsbInsertionExecutor` 的 Python API 默认不加噪，需要显式传入米单位：

```python
result = UsbInsertionExecutor(
    sim,
    precontact_noise_std_m=0.0005,
    noise_seed=1,
).execute()
```

结果的 `precontact_noise` 保存实际 seed、标准差、限幅、首次接触时间／指垫／法向力、采样次数和最大偏移。完整 JSON 中的 `commands` 保存名义与实际手腕目标、随机偏移、恢复偏移和机械臂关节目标；终端只打印摘要。

### 验证要求

- 首次真实触觉后，采样次数不得继续增加，随机偏移保持为零，恢复偏移按时归零。
- 控制目标在阶段边界必须连续，不能把上一阶段偏移重复叠加。
- 固定 seed 的加噪动作仍须通过抓取、关节余量、碰撞、姿态、插入和松手终态检查。
- 有限 seed 通过只证明实现和记录一致，不代表随机范围成功率或训练收益。

<a id="usb-contact-resistance"></a>
## 接触阻力与连续控制

本节负责任务局部接触模型与专家控制；原生、专家及 Raw 准入条件统一见[成功判定](success_criteria.md#usb-insert)。

本页记录 USB 当前的任务专用接触模型、连续控制和完成条件。模型仍是近似接触模型，不能由同步或触底验收推导出真实软指腹力学。

本改动仅用于 `usb-insert`，共享机器人、机械臂、手、指垫触觉和摸牌场景均不修改。
当前物理版本为 `passive_spring_shoes_bottom_out_v3`。V3 的主动触底证据按[成功判定](success_criteria.md#usb-insert)核对；历史 V2 仍按原 0.8 N / 100 ms 审计。

### 接触力学复核

接触滑移、力/力矩平衡与小扰动的验收方法见 [接触力学检查](#usb-contact-mechanics)。USB 局部求解参数 `impratio=5` 抑制缓慢漂移；释放前两指法向力须低于 0.1 N 并稳定 100 ms，之后额外分离 2 mm。最新参数以本页及绑定的 MJCF/控制源码为准。

### 模型

- 插头外壳截面仍为 12×4.5 mm，孔口仍为 12.6×5.1 mm，底挡前面仍为 12 mm 深。
- 孔内两组接触件分别沿插座局部 ±Z 被动让位，每组包含两个横向分开的扁椭球接触面。每组刚度 1200 N/m、阻尼 3 N·s/m、有效质量 5 g、最大行程 1.5 mm。插入后约 0.3 mm 的压缩形成法向预压力。
- 接触件滑动摩擦系数 0.6，四个分布接触点共同产生抗转力矩。浅导入曲面让阻力随接触逐渐建立；孔口至接触件前可以有短距离自由行程。接触件仅与外壳上下表面碰撞，不与手或固定孔壁碰撞。
- 底挡使用独立接触参数（滑动摩擦系数 0.2、接触时间常数 12 ms），阻止继续插深。孔壁延伸到深度 13 mm，避免与底挡共面的末端棱边形成多余的停止面。
- 宽握柄前端后移 1 mm，并用窄颈连接外壳。到 12 mm 底挡时肩部与孔口约有 1 mm 余量，避免握柄先顶住口沿。

接触力来自 MuJoCo 求解器，弹簧力来自被动关节。不向 USB 施加外部帮助力，
不移动或焊接物体，不向触觉添加理想阻力曲线。所有弹簧、摩擦和接触参数是
可解释的仿真近似，尚未以实物连接器的力—位移曲线标定。

### 控制和任务完成

抓取、转移和初态随机化沿用原流程。插入阶段保持姿态、位置反馈，受实际
轴向力和侧壁载荷保护。检测到真实底挡接触后进入 `bottom_out`：保持插头的横向与姿态反馈，通过轴向手腕位置的小幅调整建立底挡预压力，目标 1.0 N；达到确认条件后即开始卸力，不继续追求更大载荷。
控制器使用最近 20 ms 的力均值反馈和小死区；原始 500 Hz 触觉、接触力以及
成功判定都保留逐步未平滑的数值。

插入与触底力反馈采用 1 µm 的 IK 位置收敛精度，避免小幅轴向调整被数值容差
吞掉。对准后沿七自由度手臂的冗余方向保持肘部略向外展开，仍保持原来
的手腕目标；只改变关节控制目标，不直接设置物体或机器人实际状态。

先按[成功判定](success_criteria.md#usb-insert)取得真实主动触底证据。之后进入 `unload`，保持腕部，通过指尖 IK 沿接触外法线以 1 mm/s 逐渐松开夹持。底挡力不大于 0.4 N、两指法向力均不大于 0.1 N、物体速度小于 0.5 mm/s 连续 100 ms 后，才进入 `release` 继续分离两指，随后撤手。卸力期间可能已失去部分指腹接触，阶段名称并不等同于接触开关。超时失败，不能用固定等待代替稳定判定。主动触底历史
与当前底挡载荷分开记录：卸载后弹片静摩擦可能承担插头重量，底挡可以暂时
没有载荷；**专家完整执行与相应 Raw 准入**要求已经确认触底且仍在近底区域保持稳定，
才允许最终成功。原生 monitor 还存在“当前真实底挡接触”的就位路径，不能混为同一个布尔条件。
离开近底位置、失配或重置会清除触底历史，不能用沿程摩擦冒充触底。

保护阈值：插座总法向载荷 4 N、轴向阻力 2 N、四侧壁法向载荷 1.5 N。
插入反馈在轴向阻力达到 1.5 N 时停止增加前向偏置。主动触底和保持失败时
仍返回失败，不能只因几何插入深度合适而提前松手。

### 数据和验收

`pixi run --as-is run-usb-insert` 使用新模型和流程。任务专用 500 Hz USB 诊断采集命令（与通用 `collect` 的保存模式不同），串行采一条带全局相机的数据：

```bash
pixi run --as-is record-usb --output-dir datasets/usb_resistance_new_run --episodes 1 --seed 20260909 --include-overhead
```

输出目录必须是新目录。采集不使用并行 worker，数值和渲染线程均受限。
`/usb_insertion` 逐步保存深度、轴向速度、弹片载荷、侧壁载荷、底挡载荷及
触底确认历史，与 state、视觉和原始触觉同一时钟。五指 Fn/Ft 的定义未改变：
Fn 主要反映夹紧力，判断插入阻力须对照插座轴向力和世界系指端传力。

新物理版本的 Cleaning/T-ICT 源审计要求这些流完整，并从逐步深度、速度、
姿态与底挡力独立检查主动触底和确认历史。旧版数据仍可做原管线一致性检查，
但不能因此标为通过新版物理验收。示例的独立阻力脚本还会从原始接触事件
重建分项世界系力，并与监测器逐帧比较。

### 验证范围

回归检查应覆盖完整无辅助物理动作、接触力学、初态边界和接触参数敏感性，并直接读取求解器力，约束插入／触底的相邻 2 ms 切向力跳变、卸力的法向力跳变和松手二次尖峰。有限仿真检查不是实物力学校准或随机成功率估计。

机械臂每 20 ms 的目标在 2 ms 物理步之间插值；达到关节余量边界时使用受限目标的实际 FK 更新反馈，避免继续积累不可达命令。fast 抓取倾角为 0.06 rad，baseline 保留 0.05 rad。撤手恢复插入前的腕部方向，避免保压形变继续带入撤手姿态。共享机械臂、手部模型和摸牌场景未修改。

<a id="usb-contact-mechanics"></a>
## 接触力学工程检查

本检查用于简化 USB 插入任务的内部一致性和有限范围稳定性，不要求与实物曲线逐项一致，不要求两指切向力相等或同时上升，也不要求夹持法向力恒定。

### 方法与预设限值

- **力和力矩平衡**：收集 USB 的全部接触，按接触双方转换受力方向，累加接触点力及其对 USB 质心的力臂力矩，同时保留扭转和滚转接触力矩。检验世界坐标中的 `ΣF + mg = ma_COM` 与 `Στ_COM = Iα + ω×Iω`。力残差限值 `1e-7 N`，力矩残差限值 `1e-9 N·m`。
- **时间语义**：使用原始 HDF5 同一行的接触力和 `physics/qacc`，即积分后 `mj_forward` 的同次求解状态。它不是上一步积分所使用的力，也不是用相邻速度差分构造的加速度。该检查验证内部受力一致性，不代替已有动作—状态检查。
- **相对滑移**：在真实接触位置用两刚体雅可比求相对材料点速度，去掉接触法向分量。几何接触位置的移动不直接当作滑移。`insert` 和 `bottom_out` 全段累计滑移路径每指不超过 **1 mm**，瞬时速度不超过 **10 mm/s**，100 ms 窗口平均速度不超过 **2 mm/s**。
- **相对位姿**：以 `insert` 起点为参考，USB 相对手掌平移不超过 **2 mm**，转动不超过 **3°**。它不是 USB 相对插座的对准误差；后者由原来的孔口、穿透和触底检查独立约束。
- **接触与载荷**：持握期间两指各有一个承载接触，每指法向力不低于 **0.4 N**。检查包含滑动和力矩维度的椭圆摩擦锥利用率，不允许超过 1（数值容差 `1e-5`）。
- **连续性**：持握时 2 ms 法向力变化不超过 **0.35 N**，切向力变化不超过 **0.2 N**；卸力/释放时法向力变化不超过 **0.35 N**，释放切向力峰值不超过 **0.15 N**。已有回归测试还对搬运等持握阶段执行更严格的 **0.15 N** 法向步变检查。

这些是检查前声明的工程限值，适用于当前尺寸和动作。1 mm 累计路径比约 14–21 mm 的指垫面内尺寸小得多，且另设位姿、持续速度和插座对准约束。限值不是 USB 标准，也不是来自实物传感器的标定结果。

### 原版问题与修正

早期接触检查发现软接触下的缓慢滑移与相对手掌转动，单条示例数值和七个初态对照见
[历史记录](../archive/task_scene_history.md#usb-2026-09)。

USB 场景现在把 `impratio` 从默认 1 调到 **5**，使切向约束相对法向更硬，降低软接触的缓慢漂移。没有提高滑动摩擦系数，也没有添加夹持力补偿或平滑触觉输出。这种调整的依据见 [MuJoCo 的慢滑移说明](https://mujoco.readthedocs.io/en/stable/modeling.html#preventing-slip)。

切向约束改变后，原先剩余 0.4 N 握力时就加速分离，会产生释放小峰值。因此控制器改为两指均低于 **0.1 N**、底挡力低于 0.4 N、低速连续 100 ms 后再完成分离。额外分离由 3 mm 改为 **2 mm**，为已有慢速卸力行程留出手指 IK 余量。触底确认仍按[成功判定](success_criteria.md#usb-insert)执行。机器人、相机、弹片、底挡、摩擦系数及初态随机范围保持原设置；修改只在 USB 任务内生效。

物理版本仍为 `passive_spring_shoes_bottom_out_v3`（相同弹片/底挡模型与触底判据）。不同求解参数及控制实现通过完整 MJCF 指纹与控制源码 SHA256 区分，不能仅凭版本字符串视为相同数据。

### 串行验证与交付

`scripts/workcell/check_usb_mechanics.py --suite --output 新目录` 默认逐条运行九个完整动作：原位、XY ±10 mm 四个角点及 ±5° 偏航、两个独立随机初态，以及抗扭转/滚转参数同时 ±20% 的两个敏感性案例。每条都含接触前 0.5 mm 高斯扰动，触觉出现后停止随机扰动。敏感性参数只改测试模型实例，不写入场景。

`--raw 原始H5 --output 新目录` 只做离线 FK、接触和刚体计算，不重新执行动作，不修改原始数据。输出 JSON、逐步 NPZ 和独立 PNG/PDF 图。负例测试覆盖遗漏接触力矩、力臂、陀螺项，以及缺帧、缺相位、失去接触、持续滑移和超限力跳变。

正式样例的结果在 `processing/mechanics/report.json`；九条测试结果在 `processing/mechanics/robustness.json`。报告绑定原始 SHA256、模型指纹和控制源码。有限案例通过后可开展该范围内的试采与训练；它不构成任意初态或大量采样的成功率保证，后续每条仍须执行 Cleaning 和接触质量检查。

<a id="usb-cleaning"></a>
## Raw Cleaning 与时钟复核

本节负责保存后的准入与物理证据复核；模型原生成功见[成功判定](success_criteria.md#usb-insert)。
当前 V3 与历史 V2 使用各自记录版本的门槛，不能交叉补写或重判历史标签。

首次验收使用 5 条完整 episode，按同一采集代码和设置生成，保留每次尝试的结果。所有采集、测试和离线检查依次执行；采集运行期间不另开仿真或测试进程。已经有同代码、同设置的完整样本时，可以通过 `--start-index` 在新目录补足，避免重复采集。

### 保存后的自动检查

`record-usb` 在保存时已检查原始格式和抓取、插入、松手后的稳定终态。新增 Cleaning 作为保存后、发布前的独立步骤，读取原始文件，不覆盖或改写原始记录：

以下命令在项目根目录执行，输出目录必须是尚不存在的新目录。

```bash
nice -n 10 pixi run --as-is check-usb-cleaning -- \
  --input datasets/usb_insert/raw_01/raw/usb_000000.h5 \
  --input datasets/usb_insert/raw_01/raw/usb_000001.h5 \
  --input datasets/usb_insert/raw_01/raw/usb_000002.h5 \
  --input datasets/usb_insert/raw_01/raw/usb_000003.h5 \
  --input datasets/usb_insert/raw_01/raw/usb_000004.h5 \
  --output-dir datasets/usb_insert/cleaning_01
```

检查包括：

1. 状态和物理触觉完整记录在 500 Hz；图像按采集参数记录，例如 10 Hz 或 30 Hz。验证每条流的时钟和源索引，图像所用触觉不来自未来。
2. 按真实施加的执行器控制建立状态转移；检查手部控制与前一状态、目标的关系，关节和 USB 平移是否与积分后的速度一致。另检查步后刷新的执行器力是否符合该步控制量与结果状态的模型方程。
3. 检查预期时间网格、流长度、接触事件索引、图像内容和重复 episode。静止时同样的像素不自动算丢帧；末尾为完整记录终态追加的短间隔帧也不是异常。
4. 检查非有限值、关节与 USB 平移/转动跳变；四元数取反不算姿态跳变。加速度尖峰单独提示结合接触阶段查看。

报告列出实际最大值、发生帧和阈值。默认突变筛查阈值是关节 30 rad/s、物体平移 2 m/s、物体转动 30 rad/s，加速度超过 500 rad/s² 提示复查。这些是明显数据损坏的筛查阈值，机械臂控制速度、关节余量和碰撞等还由动作检查及独立源审计核对。阈值不能视为真实硬件安全额定值。

当前 V3（`passive_spring_shoes_bottom_out_v3`）按[成功判定](success_criteria.md#usb-insert)中的主动触底证据核验，同时检查近底、低速、低穿透和姿态。清洗器按记录的版本选择门槛：历史 V2 不会按 V3 的较低力阈值重新判定通过。卸力后的稳定窗口和二次尖峰另见示例 `processing/control_quality.json`。

对于 `contact_model_version=passive_spring_shoes_bottom_out_v2`，报告另列 `pipeline_integrity` 和 `recorded_physics_gate`：前者检查逐步监测流的时间、索引、长度和有限值；后者核对插入深度、速度、底挡受力及成功标记。`bottom_out` 阶段须在松手前出现连续至少 100 ms、底挡轴向反力至少 0.8 N 的真实触底证据，同时保持近底、对准、低穿透和低速条件。50 个相邻的 500 Hz 有效受力样本覆盖 100 ms；重复读取同一物理时间不能增加确认时长。

**下面的 0.8 N / 100 ms 仅指历史 V2**：`bottom_out_confirmed` 保存已观测到的主动触底历史（bottom contact confirmation retained after unloading）。松手后，弹片静摩擦可能承担 USB 的重量，底挡反力可以降为零；只要此前的 0.8 N / 100 ms 确认成立，插深仍在 11.9–12.1 mm、穿透不超过 100 µm、线速度不超过 0.5 mm/s，且姿态、孔内截面和角速度满足条件，仍可保持 `seated`。V3 按当前版本判据执行，不能沿用 V2 数值。新启动的监测器不能仅凭这个位置或弹片摩擦宣布主动触底证书成立。离开近底区域、姿态失配、穿透越限、时间倒退、观测缺口或显式 reset 都清除这段历史；恢复到原位置也必须重新获得触底证据。短时运动即使未离开近底区域，也会使 `seated` 和稳定时长失效。

离线清洗与 T-ICT 源审计从每一步的实际时间、底挡力、几何条件独立重建这段历史，再比较 `bottom_out_confirmed`、`seated` 和连续稳定的 `success`，不直接信任布尔标签。`bottom_out_latch_reconstructed=true` 表示归档锁存序列与独立重建一致。旧源没有该接触模型版本时保留原格式检查，明确标记这项物理验收未执行；时序检查通过不能代替触底模型验收。

执行器核验加载 MJCF 参数、计算代数方程，不重新运行物理步或渲染。还会用图像所对应时刻的原始关节状态重新计算相机、双腕和十指运动学位姿，核对缓存位姿的真实时刻。它要求归档的代码和模型指纹与本地版本匹配，避免用新版本的控制方程验证旧数据。

### 时间与动作索引

原始记录第 `k≥1` 行的含义：

| 内容 | 所属时刻 |
| --- | --- |
| `state/qpos[k-1]`、`state/qvel[k-1]` | 本步开始前状态 |
| `commands/actuator_control[k]` | 本步实际使用的控制量，记录于步结束时 |
| `physics/actuator_force[k]`、触觉、物体缓存位姿 | 步后监测器刷新时刻，即 `state/timestamp[k]` |
| `state/qpos[k]`、`state/qvel[k]` | 积分后的结果，`state/timestamp[k]` |
| 相机 `pose_timestamp` | 与该图像对应的触觉、腕/指尖位姿同一观测时刻 |

因此 `state_t → action_t → state_{t+1}` 的原始行号应为 `k → control[k+1] → k+1`。动作的日志时间和结果状态同为最晚时间，实际应用发生在这两个状态之间。初始化第 0 行的控制量没有前驱转移，不能作为已执行动作。接触锁存可能在一次物理步结束后更新请求的关节目标，因此不能把 `arm_joint_target` 直接当成该步已施加的控制量。

USB 监测器在每个物理步后已有一次 `mj_forward`，所以观测时钟必须声明为 `post_step_forward_v1`，触觉、solver、位姿、对应状态时刻完全一致。`physics/actuator_force` 是保持该控制量、在结果状态重新评估的力；原来的积分前执行器力已被这次刷新覆盖，不能把记录值冒充积分时使用的力。重建这些步后观测不等于完整重放约束动力学。

早期 USB 采集器曾把这些观测统一标早 2 ms；问题和当时的修复验收状态见
[历史记录](../archive/task_scene_history.md#usb-2026-09)。图像与触觉相互一致不足以证明时钟正确，
需用独立关节状态/FK 检查；当前转换器和源审计拒绝旧时钟合同。初始化控制行不能用于计算机械臂速度，
检查从第一条实际命令与初始关节位置开始。

PDF 发布格式中的 H50 action 是后续 50 帧的腕/指尖目标位姿标签，属于从未来示范状态构造的学习目标；它不是原始执行器控制或力矩。两层需要分别验证，发布时仍执行 `validate-tict` 与 `audit-tict-source`。

### 触觉起止与视觉对应

对 5 条样本分别运行以下命令，修改输入编号和新输出目录：

```bash
nice -n 10 pixi run --as-is review-usb-tactile -- \
  --input datasets/usb_insert/raw_01/raw/usb_000000.h5 \
  --output-dir datasets/usb_insert/tactile_review_000000
```

查看生成的 `index.html` 和对照图，核对首次接触前后、峰值受力、最后接触与松手后画面。默认同时报告 `1e-6 N` 和 `0.01 N` 两种阈值，使用真实求解器的十指法向力/切向负载，保留有符号的两轴切向力。Genesis 几何探针的接触防抖标记不代替物理力信号。

图像时刻与触觉时刻匹配，不表示相机捕捉了每个 2 ms 接触事件：10 Hz 图像通常相隔 100 ms，30 Hz 图像通常相隔约 33 ms。报告保留事件前后图像的实际时间差，后帧只用于检查上下文，不进入当前观测。头部图像为 320×240 全景；裁剪只放大已有像素。可以检查抓取/松手阶段是否对应，不能凭这些图像分辨微米级间隙或触觉皮肤形变。

自动检查通过后仍需逐条查看对照图并记录判断。首次 5 条通过用于确认这版 pipeline 的基本一致性；后续批量数据继续做第 1–4 项，修改相机、时间同步、触觉、控制或导出流程后重新做小批量视觉验收。

<a id="usb-h50"></a>
## USB H50 训练来源

USB 插入的训练数据名为 `usb_insert`，部署任务名为 `usb-insert`。
H50 normalizer、旧 H30 兼容、checkpoint 内统计与 `usb_pi05_deployment_v2` 的完整绑定规则统一维护在
[π0.5 数据与部署合同](../workflows/model_specific_evaluation.md#pi05-contract)。
训练修复来源见[OpenPI USB H50 说明](../../../openpi/examples/kaihand/README.md#usb-插入-h50手指尾部安全归一化)。

<a id="usb-unplug"></a>
## 拔出：usb-unplug

当前流程（`usb-unplug-held-clear-2s-v2`）：验证已插入的静态初态 → 接近外露塑料柄
→ 双指对向抓稳 → 沿 socket 退出轴低速拔出 → 完全离开插座，右手持有且不触桌，保持至原生监测确认。
不再要求转移到原桌面位置、放稳松手或退让回位。

初态只在 reset 时设置插头自由关节和两个被动弹片的弹性位移，再检查静态稳定、
接触和穿透。不添加锁扣、物体 weld、吸附或拔出助力；采集开始后只改变机器人控制目标。
seed 保持可复现，但不能对已经插入的插头沿用桌面自由 XY/yaw 抖动。

原插入任务的轴向阻力符号不能用于退出：新 monitor 保留 socket 局部坐标下的
signed force，分别计算向内与向外阻力。原生手持清离窗口见[成功判定](success_criteria.md#usb-unplug)；双指抓稳属于专家动作。
初始场景仍由 reset 安装，不能将任意自由插头初态冒充该任务。

原生具体阈值见 [USB 拔出成功判定](success_criteria.md#usb-unplug)。
旧版曾要求放回标定桌面位置、释放和退手；旧示教、deployment、H50 报告保持原样，
必须按各自归档协议解释，不与新版手持 2 秒结果直接混算。

物理范围沿用原任务的简化 target socket / 被动弹片模型：排插外壳、相邻插孔、
开关和电缆是无碰撞的视觉件。验收只证明该模型下的名义拔出流程成立，不能外推
为真实 USB 插头的机械合规性，也不能作为整台排插的通用碰撞安全结论。

统一采集、转换和部署使用完整 task 名 `usb-unplug`；Raw 诊断组为 `usb_unplug/`，
状态/控制默认 100 Hz、相机 30 Hz，任务 monitor 在每物理步更新。
π0.5 转换采用 `head + right_wrist` 并保留 mechanics、场景指纹与 instruction；也支持独立 LeRobot v3。
模型与任务支持以统一入口 `--list-support` 为准，训练转换快照的 instruction 必须与 deployment 匹配。
