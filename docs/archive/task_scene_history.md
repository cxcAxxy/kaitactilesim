# 场景修改与任务专家验收记录

本页负责共享外观修改和任务专家的历史单次验收，保留版本、来源、指纹与验证范围。
当前共享配置见[共享场景](../architecture/shared_scene.md)，任务用法见[任务指南](../README.md#任务指南)。
固定数据、模型训练和模型评估批次见[流水线批次记录](pipeline_batches_202609_202610.md)，
站点发布见[工作台发布记录](evaluation_workbench_release_history_20261015_20261021.md)。
历史路径可能没有随当前仓库保留；单次专家验收不表示当前模型成功率或随机初态泛化能力。

<a id="appearance-history"></a>

## 共享场景外观修改

本页保留各轮修改时的验证结果、路径和限制；文中的“当前”“最新”和任务数量均属于
对应记录时点。当前配置与查看方式见[场景外观与资产](../architecture/shared_scene.md#appearance)。
原 `datasets/usb_insert_example/` 及其预览页面未随当前仓库保留，以下路径仅用于历史溯源。

当前版本按用户要求采用**白色手臂和机身、银色手、深灰色大理石桌面、白色周围环境**。

双臂初始姿势后续升级为 [v9 共享对称收臂配置](../architecture/shared_scene.md#shared-home)。本页 v1–v7 记录描述的是当时的外观改动。
下文的 v7 记录描述当时的 USB 外观改动；其中“仅 USB 左臂变化、右臂保留”不再代表当前初始姿势。

### v7：水平左前臂与参考图风格的白色排插

USB 左肩关节 1 从 100° 调为 112°，其余左臂关节不变；前臂上扬角从约 13.6° 降至约 2.0°，
左手下降约 6.8 cm。右臂和其他任务姿态保留。

原插座夹具改为白色，外围新增圆角白色排插外观：三个五孔插座、四个 USB-A 孔、开关和电源线。
靠近机器人一端的**第一个 USB-A 孔**是唯一有效目标，继续使用 `usb_socket_mouth`，
世界位置 `[0.620, -0.180, 0.720]` m 不变。原孔壁、弹簧、插入判定和触觉接触模型保留。
其他孔位没有插入任务和电气功能；新增外观不参与碰撞。

`scripts/workcell/build_usb_power_strip.py` 离线生成两份封闭 STL 和 `tasks/usb_insert/power_strip_visual.xml`，
SciPy 仅用于离线网格三角化，运行时直接加载资产。STL/包含文件哈希保存在
`assets/workcell/meshes/usb_power_strip_visual.json`。生成的可视模型在原任务体之后追加，
保留原 body/geom/joint/site/actuator/camera 索引；增加一个零质量固定体和 34 个可视 geom。
Python 包资源包含新的任务 XML 和网格。

最新预览为 `datasets/usb_insert_example/appearance_power_strip_v7/index.html`，通用入口 `appearance.html`
指向此页。新采集 `power_strip_v7_capture/` 成功完成插入和释放，23.418 s、11,710 个状态样本、
三路相机各 704 帧。Raw 校验与三相机 EgoTouch 转换 dry-run 通过。
134 项场景、采集、转换和推理接口测试通过；带/不带探针的编译模型各核对 181 个原有字段，
原物理模型、相机、控制维度和对象索引保留。可视外壳封闭性和有效孔口无遮挡测试通过。
未重新评测外部模型权重的闭环性能；新外观与历史 RGB 分布不同。

### v6：USB 左臂自然待机

仅将 USB 左臂初始角调整为 `[100, -75, -80, -120, 150, 0, 0]` 度。
大臂接近竖直下垂，肘部相对 v5 内收约 13 cm、降低约 7 cm；前臂保持约 67° 屈肘，
手掌朝内，腕部弯曲关节保持中立。右臂、手指控制、碰撞与触觉模型、相机安装和数据格式保留。
左腕相机仍随真实关节姿态更新世界位姿。深灰石纹桌面和约 8 cm 机身间距保留。

v6 历史预览 `datasets/usb_insert_example/appearance_natural_arm_v6/index.html` 包含正面、侧面新旧对照。`natural_arm_v6_capture/` 为独立的新采集目录。
93 项相关场景、采集、转换和推理接口测试通过；完整 USB 动作成功，Raw 校验与三相机 EgoTouch
转换 dry-run 通过。外部 checkpoint 的闭环性能未重新评测。

### v5：深灰石纹、桌沿退让与左臂屈肘

- 共享工作台使用 `assets/workcell/textures/dark_grey_marble_v1.png`，内置 image_gen 生成；同目录 JSON 保存完整提示词与纹理哈希，PNG 纳入 Python 包资源。
- 桌面近侧边界从 x = −0.92 m 收到 x = 0.15 m。在桌面高度处，机身网格前表面 x = 0.07 m，间距约 8 cm。远侧边界 x = 2.08 m、宽度 3 m、桌面高度 0.68 m、摩擦及接触参数保留；可视和碰撞边界同步修改。固定桌子的自动推导质量/惯性随几何变化，它没有自由关节。
- USB 左臂初始角改为 `[90, -45, -70, -120, -120, 0, 0]` 度；大小臂实际夹角约 67°，左手 x 从原始约 0.566 m 收至 0.353 m。其他任务姿态、右臂初态及控制器保留。
- 相机安装位姿/FOV/分辨率/频率、触觉、动作维度及 Raw 格式保留；左腕相机随左臂姿态运动，正常记录新外参。

v5 历史预览为 `datasets/usb_insert_example/appearance_marble_fold_v5/index.html`。`marble_fold_v5_capture/` 是新场景下实际重新采集的成功 USB 轨迹，
23.418 s、11,710 个状态样本、三路相机各 704 帧；Raw 校验与三相机 EgoTouch 转换 dry-run 通过。
119 项相关回归测试通过；七个模型的机器人动力学、任务几何和固定相机标定核对通过；
七条历史任务轨迹各抽查 100 个姿态，已有桌面接触未落入被裁掉的区域。
花瓶固定布局及 10 个随机种子的污渍/清洁视觉检查通过。
其他任务未完整重新执行，也未重新评测外部模型权重成功率；历史结果不与新数据混写。

### v4：USB 左臂外展待机姿态

USB 场景的左肩 `left_arm_joint2` 初始角由 −65° 改为 −45°，左手向外移动约 17 cm。
仅修改该任务的 `ARM_HOME["left"]`，右臂与其他任务的初始姿态保留。
这是实际关节姿态调整；左腕相机随手臂移动，安装标定不改，逐帧外参按实际姿态记录。
关节/动作映射、碰撞与触觉参数、数据格式、转换和评估入口均未修改。

v4 历史预览位于 `datasets/usb_insert_example/appearance_left_open_v4/index.html`。
实际重新采集保存在 `left_open_v4_capture/`：USB 成功完成，23.418 s、11,710 个状态样本，
head/left_wrist/right_wrist 各 704 帧。Raw 校验和三相机 EgoTouch 转换 dry-run 通过；
93 项 USB 场景、采集、转换及推理接口测试通过。未运行外部模型 checkpoint 的闭环评估。
历史轨迹仍保留原来的左臂姿态，没有修改旧 HDF5。

### v3：银灰色桌面

仅修改共享 `worktop_finish` 纹理的底色和微纹理颜色，保留 v2 的其余外观和腕部修正。
USB 编译模型 484 个数组对比仅 `tex_data` 不同。相机、触觉和物理配置未修改。
v3 历史预览位于 `datasets/usb_insert_example/appearance_silver_table_v3/index.html`。视频为原轨迹的新外观重绘，235 帧；
内参与原记录一致，外参最大误差为 0，源 HDF5 哈希未变。本次未重新采集或运行模型闭环。
v2 的黑色桌面采集与验证结果仍保留在原目录。

### v2：封闭手部外壳与腕部画面修正

新增 38 份仅用于显示的封闭 STL：双手手掌与近端指节 36 份、腕部护罩 2 份。
它们由原始 CAD 顶点构建封闭包络；腕部护罩在横截面方向收窄，为既有腕部相机光心留出空间。
替换现有 visual geom 引用，没有增加 body、geom 或关节。原始碰撞 STL、碰撞 capsule、
质量、惯量、控制和指尖触觉接触网格全部保留。

腕部穿模包含近裁剪切开外壳及装饰护罩包围相机两种原因。`visual/map znear` 从默认
0.01 改为 0.0005，对 USB 对应约 42.4 mm → 2.1 mm。此项是渲染裁剪修正，
相机位置、朝向、FOV、分辨率、采样频率和输出的像素内参均未修改。
原位相机仍会看到靠近镜头的外表面，未通过移动视角或后处理遮罩隐藏机器人。

MuJoCo 的 near 距离等于 `znear × extent`，见[官方定义](https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mjmodel.h)。
新增视觉网格会改变编译器的网格缓冲地址、包围体和派生 `dof_length` 尺度。
FOV 相机未设置物理 sensor size 时，内部 `cam_intrinsic` 占位值也随 znear 变化；
实际输出的像素内参及逐帧外参已在完整录制中逐值核对，不将这些内部字段宣称为全部不变。

v2 历史预览入口为 `datasets/usb_insert_example/appearance_white_silver_v2/index.html`。v2 采集在 `white_silver_v2_capture/`，v1 和原始记录保留。

v2 验证：118 项回归测试通过；新 USB 完整采集成功，11,710 个状态样本、控制量和
法向/切向触觉，以及两路相机内外参与原示例逐值一致；真实 Raw 校验和转换 dry-run 通过。
花瓶固定布局和 10 个随机种子的污渍/清洁图像判定通过。尚未重新评测外部模型权重的成功率。

资产生成脚本为 `scripts/workcell/build_hand_visual_shells.py`。SciPy 只用于离线生成，
本次使用临时独立工具环境，未改变 Pixi 依赖；运行仿真直接加载已生成的 STL。
`hand_visual_shells.json` 记录原网格、输出网格的哈希和仅视觉缩放。
测试检查原网格未改变、网格封闭、外壳不参与碰撞、指尖触觉网格不被替换。

### v1 历史记录

以下描述和数字属于第一版，保留用于对比；当前外观和 v2 结果以上文为准。

这次将共享天机 M6、左右 KaiHand 改为银白色系，并统一工作台、地面、背景与 USB 材质。
这是现有 MuJoCo 渲染器内的第一轮外观整理，不代表已经达到光线追踪或扫描资产的写实程度。

### 设计参考

- [RoboDojo 官方项目](https://github.com/RoboDojo-Benchmark/RoboDojo)：参考其机器人操作场景的完整资产呈现与 benchmark 定位；没有复制其资产或引擎实现。
- [CARLA 材质制作](https://carla.readthedocs.io/en/latest/tuto_content_authoring_vehicles_materials/)：参考按表面类型区分金属、涂层和橡胶，以及克制的微表面变化。CARLA 的 PBR/clear-coat 功能不等于当前 MuJoCo 原生渲染器的能力。
- [ManiSkill 外观与物理材质](https://maniskill.readthedocs.io/en/latest/user_guide/tutorials/domain_randomization.html#actor-link-physical-and-visual-randomizations)：参考视觉材质与物理接触属性分别配置的方式，本次不引入随机化。
- [MuJoCo 材质定义](https://mujoco.readthedocs.io/en/stable/XMLreference.html#asset-material)：实际使用原生颜色、specular、shininess、程序纹理及灯光。

### 修改范围

- `shared/mjcf/robot.xml`：银白机身、灰色细纹理工作台、低对比度地面、浅色背景；调整灯光与阴影质量。旧的 `arm_red/green/blue` 材质名保留，其内容改为不同亮度的银白材质。
- `assets/workcell/mjcf/hand_asset.xml` 与左右 `hand_*_body.xml`：手部外壳、关节及触觉垫**可视几何**使用银白材质。碰撞几何、触觉垫物理属性与 probe 定义不改。
- `tasks/usb_insert/scene.xml`：细化金属壳、深色橡胶、夹具和触点的视觉属性。

共享外观应用于七个任务。所有场景物体的位置、尺寸、质量、惯量、碰撞掩码、摩擦、求解参数、机器人控制及具名相机保持不变。没有增加或删除 geom/body，分割对象索引也保持不变。
额外的全景展示使用临时 free camera，不写入场景或采集配置。

### 查看效果

本次本地示例放在 `datasets/usb_insert_example/`，没有覆盖原始记录、旧发布包或其校验清单：

- `appearance_silver_v1/index.html`：新旧全景/相机对比、完整动作视频及验证报告。
- `silver_lab_capture/raw/usb_000000.h5`：新外观下重新物理执行并采集的成功轨迹，沿用该示例的 head/right_wrist 两路相机。
- `silver_lab_capture/review/review.mp4`：上述新采集 RGB 与触觉的离线回放。
- `appearance_silver_v1/review.mp4`：将旧轨迹的 qpos 用新外观重绘；触觉读取旧记录，不重新计算，不作为新的训练 RGB 来源。

正式统一采集入口的三相机默认配置未修改。新示例两路 RGB 仍为 320×240、30 Hz；状态与触觉为 500 Hz。展示视频为 10 fps。

重绘工具只读取源文件，拒绝覆盖已有输出：

```bash
pixi run python scripts/workcell/preview_usb_appearance.py \
  datasets/usb_insert_example/raw/usb_000000.h5 \
  --output-dir /path/to/new_preview
```

可选 `--reference-model /path/to/before.mjb` 生成相同姿态与视角的外观对照。

### 本次验证

- 五个 XML 去除视觉属性后，与更新前的结构及参数一致。
- 七个编译模型的非视觉物理数组与相机参数一致。
- USB 新旧完整采集的 11,710 个状态样本、控制量、法向和切向触觉逐值一致；两路相机的内参和 704 帧外参逐值一致。
- 新 USB 采集成功，Raw 校验及真实数据转换 dry-run 通过。
- 116 项已有测试通过，覆盖场景、USB、采集、转换/评估分发、触觉以及花瓶数据接口。
- 花瓶固定布局和 10 个随机种子的图像检测通过：有污渍时正确识别，隔离的清洁视觉 fixture 中剩余红色像素为零。这不是完整擦拭轨迹成功率测试。
- 未运行外部 checkpoint 的闭环性能评估。视觉分布变化可能影响旧模型表现，即使所有接口和物理参数保持一致。

原始图像不应与新外观重绘混用。模型指纹会随 XML 外观内容变化；这是正常的来源追踪，应使用新采集目录，不绕过旧批次的源码一致性检查。

<a id="task-history"></a>

## 任务专家与示例验收

归档日期：2026-10-21。内容从 `docs/tasks/` 的任务教程移入，保留原记录的版本、数值、来源路径与指纹。
这些记录说明当时的检查范围；历史路径、localhost 服务、依赖安装状态和“尚未训练”等表述均是当时快照。
当前命令从[流水线速查](../workflows/pipeline_quickstart.md)查询，当前判据见[任务成功判定](../tasks/success_criteria.md)。
下列命令只用于追溯旧运行，不应直接作为当前采集、训练或评估入口。

<a id="poker-2026-09-20"></a>
### Poker：2026-09-20 动作改造与单 seed 验收

来源：[摸牌示例页](../tasks/poker_example.md)整理前的“连续动作版本”。
`right-side-low-chatter-v4` 是轨迹版本；整理时它仍与当前 `tip-pad-v7` 接触姿态同时使用，二者不是替代关系。

当时的轨迹与接触改动：

- 起始手指和右臂目标按同一最小急动度曲线逐物理步更新，小指不再突然追逐整段弯曲目标。
- 起步从关节空间插值改为右侧笛卡尔轨迹，向右外侧最多避让 55 mm、同时抬高 35 mm；
  实测末端横向范围 `y=-0.228～-0.164 m`，回归同时检查双手间距。到牌上方后只下降。
- 中档拖牌速度由 6 改为 12 mm/s；建立压力时使用单独的有界增益/速度，之后恢复拖动反馈。
  每指 0.5 N 目标、桌牌摩擦和 4 N 水平限力不变。
- 夹牌改用同侧腕关节解，拇指从桌沿下托起，避免另一侧腕姿约 300° 绕行；牌仍为自由刚体。
- 拇指最后两关节接近时间由 0.20/0.24 s 改为 0.40/0.80 s；Cartesian IK 的 20 ms 分段目标连续插值到物理步。
- 短暂失去接触时保持原关节目标，避免用受力偏移后的实测关节值反复消除伺服预载。
- 桌牌及牌/指腹柔顺接触阻尼比由 1.0 提高到 1.5，时间常数从 10 ms 调为 6.67 ms，
  使等效刚度和静态压入量近似不变；整条 episode 固定参数，不修改摩擦或滤波保存的触觉。
- 双面支撑后稳定 0.45 s；4 mm 转移、后续 21 mm 抬升分别使用 0.65 s、1.25 s 最小急动度轨迹。
- 看牌由实际夹持后的牌/手相对位姿求目标，88 mm 长轴竖直、63 mm 窄边朝下；
  终态肘部不高于肩部 40 mm。
- 基础夹持积分增益为 0.035；中档从 0.10 降为 0.06，最大修正速度从 1.5 降为 1.0 deg/s，
  最大修正范围从 1.0 降为 0.75 deg。基础夹牌目标四指各 0.75 N、拇指 3 N；中档各 0.25 N、拇指 1 N。

2026-09-20 串行验收：233 项影响范围内回归通过；中档 seed 4（原有位置随机化与接触前微扰）采集成功，
Raw 校验和 Cleaning 前四项通过。两路 30 Hz RGB 各 909 帧，状态/触觉 3028 帧，仿真 30.264 s。

| 指标（仿真时间/实测关节） | 更新前 | 更新后 |
| --- | ---: | ---: |
| 全任务时长 | 29.794 s | 30.264 s |
| 拖牌四指接触比例 | 99.68% | 99.81% |
| 桌沿多指同步低载 | 3 个连续周期性记录帧 | 2 个分离的 2 ms 接触切换 |
| 抬牌双面支撑比例 | 100% | 100% |
| 抬牌四指同时接触比例 | 82.91% | 90.61% |
| 转向看牌拇指 Fn / Ft 峰值 | 1.08 / 0.71 N | 1.06 / 0.63 N |
| 静止看牌五指 Fn/Ft 标准差 | 均小于 0.002 N | 均约 0.001 N 或更低 |

该条记录的抬牌及抬臂/看牌过渡双面支撑均 100%；持牌、抬臂和看牌阶段四指接触均 100%。
桌沿保留两个真实单物理步接触流形切换，没有滤波、补帧或修改 Raw。
当时验证采集与相关转换/推理接口，未运行远端模型闭环，也未据此保证随机种子成功率。
当时声明未放宽成功门槛或改变触觉计算、Raw 时钟/schema、相机及关节映射；该声明只适用于此次动作改造。

<a id="usb-2026-09"></a>
### USB：2026-09-09 姿态与 2026-09-17 悬停验收

来源：[USB 执行页](../tasks/usb_tasks.md#usb-insert)。09-09 表格属于被动接触阻力改造之前，
其阻力和耗时不能替代 V3 的物理验收。V2 按 0.8 N / 100 ms、V3 按 0.6 N / 150 ms 分别解释。

2026-09-09：MuJoCo 3.12.0、500 Hz 物理步、50 Hz 末端反馈、350 个触觉 probes，完整动作约 21 秒。
先抬升约 5 cm，再搬运和对准，插入目标速度 3 mm/s；当时释放为 0.4 秒平滑卸力加 0.2 秒等待。
同一末端姿态下调整冗余机械臂解，肘部 Y 从约 −0.1349 m 到 −0.1530 m，外展约 18.1 mm；
撤离高度从 8 cm 改为 7 cm，用时仍 1 秒，保留至少 5 cm 实测净空验收。

| 指标 | 09-09 实测 |
| --- | ---: |
| 完整仿真时间 | 21.002 s |
| 最终插深 | 11.960 mm |
| 全程最大抬升 | 108.629 mm |
| 最终孔口局部 Y / Z 偏差 | −0.035 / −0.090 mm |
| 最终姿态误差 | 0.349° |
| 峰值插座法向载荷 / 轴向阻力 | 0.771 / 0.768 N |
| 最大插头—插座穿透 | 0.01019 mm；终态 0 |
| 松手后稳定驻留 | 2.124 s |
| 抬升阶段最小拇指 / 食指夹持力 | 2.088 / 2.183 N |
| 取物时小拇指 MCP 比食指向前 | 49.80 mm，方向余弦 0.729 |
| 插入掌面法向 Z 分量 | 约 −0.679 |
| 插入食指 MCP 高于小拇指 | 约 48.5 mm |
| 插入末端腕部 J6 / J7 | −30.87° / +61.48° |
| 全程实际关节最小限位余量 | 15.744° |
| 插入阶段可见模型最小桌面间隙 | 超过 30 mm（当时网格审计） |
| 最终手指—插头接触 | 0 |

搬运 J6 / J7 累计角程约 66.3° / 25.2°，抓稳到插完约 82.4° / 67.2°；肘部低于肩部，
放置阶段世界 Y 不超过 −0.1528 m。10,501 个物理步中，机器人与桌面、支架、插座，以及非夹持手指与 USB 的接触均为零。
当时 `pixi run --locked test-usb-insert` 60 项通过，包含旧场景隔离、完整动作、退出、正手姿态、
关节角程、Y≤−0.15 m、至少 15° 关节限位余量、22 秒内完成、真实脱手和禁止瞬移/额外外力检查。
记录对应固定初态、仿真真值反馈；当时 USB 实例使用 8 次摩擦后求解迭代。

2026-09-17：新增孔口上方 100 mm 悬停，连续 0.6 秒满足横偏 <0.08 mm、角差 <0.004 rad、
高度误差 <0.15 mm、线速 <1 mm/s、角速 <0.01 rad/s 后才下降至孔口上方 2 mm。
下降名义 3 秒，距孔口 30 mm 内启用插入肘部外展；高位关节余量至少 10°，近口/插入恢复至少 15°。
当时串行通过 6 个完整动作回归和 63 项相关单元/入口测试，未重新采样例。
覆盖固定、随机边界、两组噪声、边界加噪声与 baseline，逐帧检查下降至孔口上方 50 mm 前完成悬停，
以及限位、外展、禁止瞬移/外力、真实触底及松手。力跳变阈值保持当时原值。
旧动作录像和样例页未随当前源码交付；展示应从对应 Raw 按版本重新生成。

原 USB Cleaning 页另记录过一次未注明日期的时钟修复：旧采集器将步后观测统一标早 2 ms，
图像和触觉相互一致没有暴露它，独立关节状态/FK 检查才发现。原文称旧数据保留供诊断，
“修复后的完整采集另行验收”；后一句是当时的工作状态，不是当前尚未验收的结论。

接触力学页早期对照（原文未给日期）：完整力/力矩平衡与连续性通过，但两指累计滑移约 1.14/1.20 mm，
相对手掌最大转动约 3.09°，超过预声明限值；摩擦锥利用率最高约 0.315。
七个初态出现相近漂移，未表现为某个位置突发失稳。这是任务局部调整 `impratio` 与释放控制之前的诊断数据。

<a id="bulb-action-contract"></a>
### Bulb：0926_200 动作合同与域偏移诊断

来源：[旋入任务页](../tasks/bulb_tasks.md#bulb-screw)原“π0.5 动作与评估契约”；原页未提供这次消融的精确运行日期。
本节按批次名 `0926_200` 归档，保留原身份，不将归档日期当作实验日期。

该批 π0.5 只接收 head、right_wrist 和 27 维实测关节状态，部署清单记录 `tactile_sent_to_model=false`。
因此原文所述 0/20 不能直接归因于“模型没有理解触觉”；触觉增量输入需要另设实验。

旧 `0926_200` 发布数据的 7 维 arm action 是下一相机帧实测 qpos，而评估器将 27 维均当作控制器目标。
成功 episode 0 回放消融：旧 published label 的 ZOH/linear 均不能入牙；改用原始 controller target 后，
ZOH 只能入牙，controller target 加 500 Hz 物理步线性插值才实现就位、亮灯和成功。

源 Raw mechanics 为 `short_thread_finger_tightening_cradled_pickup_v12`，model fingerprint：
`8314902cd7250e41789f240f746adf067120e7e542c7b94b9837a5835e266e87`。
对比的 v15 环境为 `phase_aware_strict_completion_v15`，fingerprint：
`cac34f5fc061c4d854be34563a312e83b9352dd2629a4215908fcbf778eb1d46`。
这份旧 Raw 仅重标 action 后仍有 mechanics/domain shift，只适合动作合同回放和诊断；正式 v15 训练需要匹配场景的新示教。

原文给出的转换示意（历史路径与 200 条批量参数，仅供追溯）：

```bash
cd /cpfs_infra/user/chenxianchi/code/kaitactilesim_unified
/cpfs_infra/user/chenxianchi/code/openpi/.venv-pi05/bin/python \
  scripts/workcell/convert_shared_to_lerobot.py \
  --openpi-root /cpfs_infra/user/chenxianchi/code/openpi \
  --task bulb-screw \
  --input-dir /path/to/bulb_v15_raw_collection \
  --output-dir /path/to/bulb_v15_pi05_controller_target_v1 \
  --dataset-name bulb_v15_controller_target_v1 \
  --cameras head right_wrist \
  --action-contract recorded-controller-target-v1 \
  --expected-episodes 200 --verify-source-hash --fingerprint-workers 8 \
  --staging-root /cpfs_infra/user/chenxianchi/conversion_work/bulb_v15_controller_target_v1
```

<a id="bulb-inplace-2026-10-15"></a>
### Bulb inplace：2026-10-15 本机验证与 10-18 接入快照

来源：[原位旋入任务页](../tasks/bulb_tasks.md#bulb-screw-inplace)。以下均为当时快照，不描述当前机器或模型训练状态。

2026-10-15 初次接入说明：新任务未混入当时现有 360 条模型评估；两个原工程与已有 H50 报告未修改。
当时 `.pixi` 环境从缓存离线安装，执行过 `pixi install --locked --offline --skip-with-deps pyarrow`，
以 `pixi run --as-is ...` 做物理检查；转换前仍需完整依赖。这不是新的环境安装建议。

- 完整 CPU 专家用时 33.67 秒，4 个有效行程，相对初态推进约 100.65° / 1.082 mm，
  无抬起搬运；抓持、有效旋入、到座承力、无支撑释放及回 home 均通过。
- 新旧任务、staged、归档和报告接口选定回归 232 项与 44 个子测试通过；独立 executor 另有 20 项通过。
- 真实 policy adapter 接入 staged_v1 空跑 0.2 秒，四阶段均未通过，未误判成功。
- Raw 有 3,368 条状态，三路各 1,012 帧；格式检查通过。16,835 个物理步全审计，
  物体额外外力/广义力峰值 0，无终态事后写入。触觉来自 link-level solver contact 空间力阵列。
- seed 0、1 为固定初态重复性，不是两种初态或模型成功率。
- 当时 unified 内原 `bulb_screw` 七个文件与修改前 SHA256 一致，既有 staged H50 的 4,334 个归档文件大小/哈希一致。
  当时两个旧工程根目录不在原路径，无法重新核验其 856 项 source；未恢复或写入这两个路径。

当时 Raw 来源：`runs/bulb_screw_inplace_example_20261015/raw/episode.h5`。
2026-10-21 清理本仓重复产物后，Raw、sidecar、结果及 seed 0/1 smoke 记录保留在
`/cpfs_infra/user/chenxianchi/code/kaitactilesim_unified/runs/` 的对应同名目录。
Raw 完整 SHA-256 复核为 `53a189d89b018186a1b7a9de603cbe110913ff634dfa498a467056028db73da3`，
与移除前本仓副本一致；此处保留原来源身份，不将已有视频视为 Raw 替代品。
当时本机预览服务地址：`http://127.0.0.1:8877/bulb_screw_inplace_demo_20261015/review.mp4`；
服务存在与否不由文档保证。预览从保存帧导出，三路 Raw 全保留，没有重跑或修改物理轨迹。
当时未训练或运行该任务的模型策略；去掉搬运不会自动解决历史记忆或视觉捷径。

2026-10-18 接入快照：补齐独立 H50 配置 `pi05_kaihand_bulb_screw_inplace_controller_target_h50_v1`、
自动归一化与八卡入口，绑定独立环境身份。`1018_200` 做过 metadata-only 只读预检，未计算统计或启动模型。
这次接入本身没有提供模型训练或闭环成绩；真正训练仍需完整数据/统计校验。

<a id="task-variants-initial-validation"></a>
### 任务变体：初版验收与 NAS 预检快照

来源：[五个变体任务页](../README.md#任务指南)。初版报告按路径日期标识，NAS 预检原文未给精确执行日期。
以下“尚未转换”仅为归档前记录的现场状态。

初版落地顺序为 `usb-unplug` → `whiteboard-wipe-horizontal` → `bulb-unscrew`。
初版闭环、Raw、转换报告原路径 `runs/task_variants_validation_20261010/report.md`，历史产物未随源码交付。
其中 bulb-unscrew 使用返回卡座版，不能作为后来普通桌面版的数据。单横线随后独立加入，原位旋入于 2026-10-15 加入。

桌面版 bulb-unscrew 视频来源：`/nas/chenxianchi/datasets/sim/task_demo/bulb-unscrew-tabletop.mp4`，38.3 秒、1080p/30 fps。
对应 Raw 来源：`/nas/chenxianchi/datasets/sim/task_demo/_metadata/bulb-unscrew-tabletop/collection`。
目录名区分历史录像，task identity 仍为 `bulb-unscrew`。原记录称完整成功判据、逐步审计、三相机首尾一致性与源 SHA 均通过；
π0.5 只读转换预检规划 1148 帧，当时尚未写出训练集。旧 `bulb-unscrew.mp4` 与卡座 Raw 不能混入桌面版。

USB unplug 与水平白板的成功三相机 Raw 当时通过实际 π0.5 Python 的
`--validate-only --verify-source-hash`，分别规划 710、2507 帧，来源模板为
`/nas/chenxianchi/datasets/sim/task_demo/_metadata/TASK/collection`。
预检通过不表示当时已完整转换或训练，也不表示当前仍未完成。

2026-10-21 专家示例迁入本项目根目录 `task_demo/`，并补齐原 replay 绑定的外置 Raw。
上面的 NAS 路径及原 SHA 保留为历史来源身份；当前加载通过迁移清单定位本地文件，
不改写原 replay/publication。数据分工及本地校验见[项目内专家示例](../workflows/data_visualization.md#项目内专家示例)。

<a id="whiteboard-2026-09"></a>
### 白板：历史两横线示例与 2026-09-29 Z 字验收

来源：[白板任务页](../tasks/whiteboard_wipe.md)。两种布局必须按对应 scene/pattern 解释。

原页还记录过一次未单独标日期的短路径测量：5 秒初始接近中腕部第 5 关节净转 −12.95°，
替代从 −150° 到 +146° 的绕行；七关节五次轨迹最大速度 48.45°/s、最大加速度 67.17°/s²。
抓起后 5 秒展开到白板前的峰值分别为 61.49°/s、37.83°/s²。它们是该条轨迹的测量值。

历史两横线示例用时 57.10 s，100 Hz 状态 5712 条、每路 30 Hz 图像 1715 张，Raw 校验通过；
1000 Hz 板面峰值 3.193 N、100 Hz 稳定擦拭板面峰值 2.348 N。
来源路径曾为 `datasets/erase_whiteboard_example/raw/erase_whiteboard_000000.result.json` 与
`datasets/erase_whiteboard_example/curves/right_hand_force_curves.png`；历史产物未随源码交付。
当时 40±1 s 跨第二条擦拭与 `leave_board`，52±1 s 为 `release`；两个展示视频均 573 帧、57.3 s。
这些时间窗不能用于解释 Z 字流程。

2026-09-29 Z 字验收：白板相关 49 项测试通过，覆盖完整物理过程、正边界位置和自上而下轨迹。
默认、(+4 cm,+6 cm)、(−4 cm,−6 cm) 三次均擦净并放回松手，用时 66.69 / 66.735 / 67.17 s；
擦拭 1000 Hz 板面峰值 2.774 / 2.756 / 2.745 N，相隔 20 ms 最大变化 0.118 / 0.115 / 0.131 N，无直接手碰板。
当时未放宽力稳定阈值或修改摩擦/清洁门槛；结果来自已知状态专家，不是重新训练后的 PI05 成绩。

<a id="lingbot-step3905"></a>
### LingBot PickPlace：step3905 现场路径

来源：[LingBot 专项页](../workflows/model_specific_evaluation.md#lingbot-pickplace)整理前的部署清单和本机命令；原文未给验证日期。
当时静态校验的 manifest 来源为
`/cpfs_infra/user/chenxianchi/evaluations/pick-place/lingbot_vla2/step3905_deployment/deployment_manifest.json`。
服务 Python 为 `/cpfs_infra/user/chenxianchi/miniconda3/envs/lingbot_vla2/bin/python`，
旧工作目录为 `/cpfs_infra/user/chenxianchi/code/kaitactilesim_unified`，服务端口 8006。
单 seed 输出目录曾为 `/cpfs_infra/user/chenxianchi/evaluations/pick-place/lingbot_vla2/step3905_seed000`，
参数 `--seeds 0 --max-sim-seconds 90`。这些路径保留 checkpoint 和运行身份，不作为当前默认值。
