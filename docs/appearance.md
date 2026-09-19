# 场景外观与待机姿态：Power Strip / Level Forearm v7

当前版本按用户要求采用**白色手臂和机身、银色手、深灰色大理石桌面、白色周围环境**。

双臂初始姿势现已升级为 [v9 共享对称收臂配置](shared_home.md)，覆盖所有七个任务。
下文的 v7 记录描述当时的 USB 外观改动；其中“仅 USB 左臂变化、右臂保留”不再代表当前初始姿势。

## v7：水平左前臂与参考图风格的白色排插

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

## v6：USB 左臂自然待机

仅将 USB 左臂初始角调整为 `[100, -75, -80, -120, 150, 0, 0]` 度。
大臂接近竖直下垂，肘部相对 v5 内收约 13 cm、降低约 7 cm；前臂保持约 67° 屈肘，
手掌朝内，腕部弯曲关节保持中立。右臂、手指控制、碰撞与触觉模型、相机安装和数据格式保留。
左腕相机仍随真实关节姿态更新世界位姿。深灰石纹桌面和约 8 cm 机身间距保留。

v6 历史预览 `datasets/usb_insert_example/appearance_natural_arm_v6/index.html` 包含正面、侧面新旧对照。`natural_arm_v6_capture/` 为独立的新采集目录。
93 项相关场景、采集、转换和推理接口测试通过；完整 USB 动作成功，Raw 校验与三相机 EgoTouch
转换 dry-run 通过。外部 checkpoint 的闭环性能未重新评测。

## v5：深灰石纹、桌沿退让与左臂屈肘

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

## v4：USB 左臂外展待机姿态

USB 场景的左肩 `left_arm_joint2` 初始角由 −65° 改为 −45°，左手向外移动约 17 cm。
仅修改该任务的 `ARM_HOME["left"]`，右臂与其他任务的初始姿态保留。
这是实际关节姿态调整；左腕相机随手臂移动，安装标定不改，逐帧外参按实际姿态记录。
关节/动作映射、碰撞与触觉参数、数据格式、转换和评估入口均未修改。

v4 历史预览位于 `datasets/usb_insert_example/appearance_left_open_v4/index.html`。
实际重新采集保存在 `left_open_v4_capture/`：USB 成功完成，23.418 s、11,710 个状态样本，
head/left_wrist/right_wrist 各 704 帧。Raw 校验和三相机 EgoTouch 转换 dry-run 通过；
93 项 USB 场景、采集、转换及推理接口测试通过。未运行外部模型 checkpoint 的闭环评估。
历史轨迹仍保留原来的左臂姿态，没有修改旧 HDF5。

## v3：银灰色桌面

仅修改共享 `worktop_finish` 纹理的底色和微纹理颜色，保留 v2 的其余外观和腕部修正。
USB 编译模型 484 个数组对比仅 `tex_data` 不同。相机、触觉和物理配置未修改。
v3 历史预览位于 `datasets/usb_insert_example/appearance_silver_table_v3/index.html`。视频为原轨迹的新外观重绘，235 帧；
内参与原记录一致，外参最大误差为 0，源 HDF5 哈希未变。本次未重新采集或运行模型闭环。
v2 的黑色桌面采集与验证结果仍保留在原目录。

## v2：封闭手部外壳与腕部画面修正

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

## v1 历史记录

以下描述和数字属于第一版，保留用于对比；当前外观和 v2 结果以上文为准。

这次将共享天机 M6、左右 KaiHand 改为银白色系，并统一工作台、地面、背景与 USB 材质。
这是现有 MuJoCo 渲染器内的第一轮外观整理，不代表已经达到光线追踪或扫描资产的写实程度。

## 设计参考

- [RoboDojo 官方项目](https://github.com/RoboDojo-Benchmark/RoboDojo)：参考其机器人操作场景的完整资产呈现与 benchmark 定位；没有复制其资产或引擎实现。
- [CARLA 材质制作](https://carla.readthedocs.io/en/latest/tuto_content_authoring_vehicles_materials/)：参考按表面类型区分金属、涂层和橡胶，以及克制的微表面变化。CARLA 的 PBR/clear-coat 功能不等于当前 MuJoCo 原生渲染器的能力。
- [ManiSkill 外观与物理材质](https://maniskill.readthedocs.io/en/latest/user_guide/tutorials/domain_randomization.html#actor-link-physical-and-visual-randomizations)：参考视觉材质与物理接触属性分别配置的方式，本次不引入随机化。
- [MuJoCo 材质定义](https://mujoco.readthedocs.io/en/stable/XMLreference.html#asset-material)：实际使用原生颜色、specular、shininess、程序纹理及灯光。

## 修改范围

- `shared/mjcf/robot.xml`：银白机身、灰色细纹理工作台、低对比度地面、浅色背景；调整灯光与阴影质量。旧的 `arm_red/green/blue` 材质名保留，其内容改为不同亮度的银白材质。
- `assets/workcell/mjcf/hand_asset.xml` 与左右 `hand_*_body.xml`：手部外壳、关节及触觉垫**可视几何**使用银白材质。碰撞几何、触觉垫物理属性与 probe 定义不改。
- `tasks/usb_insert/scene.xml`：细化金属壳、深色橡胶、夹具和触点的视觉属性。

共享外观应用于七个任务。所有场景物体的位置、尺寸、质量、惯量、碰撞掩码、摩擦、求解参数、机器人控制及具名相机保持不变。没有增加或删除 geom/body，分割对象索引也保持不变。
额外的全景展示使用临时 free camera，不写入场景或采集配置。

## 查看效果

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

## 本次验证

- 五个 XML 去除视觉属性后，与更新前的结构及参数一致。
- 七个编译模型的非视觉物理数组与相机参数一致。
- USB 新旧完整采集的 11,710 个状态样本、控制量、法向和切向触觉逐值一致；两路相机的内参和 704 帧外参逐值一致。
- 新 USB 采集成功，Raw 校验及真实数据转换 dry-run 通过。
- 116 项已有测试通过，覆盖场景、USB、采集、转换/评估分发、触觉以及花瓶数据接口。
- 花瓶固定布局和 10 个随机种子的图像检测通过：有污渍时正确识别，隔离的清洁视觉 fixture 中剩余红色像素为零。这不是完整擦拭轨迹成功率测试。
- 未运行外部 checkpoint 的闭环性能评估。视觉分布变化可能影响旧模型表现，即使所有接口和物理参数保持一致。

原始图像不应与新外观重绘混用。模型指纹会随 XML 外观内容变化；这是正常的来源追踪，应使用新采集目录，不绕过旧批次的源码一致性检查。
