# 共享场景、初态与接近边界

本页维护跨任务共用的机器人姿势、场景资产与接近/抓取约束。
任务几何、接触阈值、抓取标定和成功判定由各任务指南维护；Raw 数据与模型数据语义分别见
[数据采集](../workflows/data_collection.md)和[数据格式合同](../reference/dataset_formats.md)。

<a id="shared-home"></a>
## 共享双臂初始姿势

十三个注册任务使用同一份 `shared/posture.py::ARM_HOME`。
左臂自然屈肘、小臂近乎水平，右臂按机器人关节轴约定镜像。

| 手臂 | Joint 1–7，单位：度 |
|---|---|
| 左臂 | `112, -75, -80, -120, 150, 0, 0` |
| 右臂 | `-112, -75, 80, -120, -150, 0, 0` |

两个上臂靠近身体，肘部夹角约 67°，小臂相对桌面约 2°。
配置只读，各任务的 `config.ARM_HOME` 与旧 `workcell` 导入作为别名。
`ArmHandSimulation` 的构造与重置均从它设置关节状态、目标和内部命令。
两手以共享的放松张手姿势为基础；海绵任务有左右拇指的无载稳定设置。

初态定义机器人从哪里开始，不重新定义抓取 IK 分支。
抓取方向、接近点与工具相对姿势属于任务标定，执行器在录像开始后连续移动到这些目标，
不能在 reset 中把手腕直接搬到任务工作点来替代真实接近过程。
白板和花瓶的 reset 也遵守这一边界。

`tests/test_shared_home.py` 按 `shared/config.py::SCENE_NAMES` 的十三个场景检查构造和重置、
左右肩肘腕镜像、锐角屈肘与近水平小臂。具体任务的执行验收仍由任务测试覆盖。

<a id="grasp-timing"></a>
## 接近与抓取边界

共享初态、任务接近与工作阶段分开维护。接近路点应通过执行器连续到达，
任务控制器在自己的阶段内核对手腕到位、夹紧和接触，再继续搬运、插入或擦拭。
等待时长、指尖间距与到位门槛由各任务标定，不设一个覆盖所有任务的统一数值。

USB 与 RAM 的“先到位、再夹紧”流程、两指间距和静止收拢时间分别见
[USB 指南](../tasks/usb_tasks.md#usb-insert)与[内存任务](../tasks/install_ram.md)；
卡片、灯泡、白板和柔体任务保持自己的接近分支，不能将其动作替换为统一抓取模板。

初态与接近轨迹变化会影响腕部相机的世界位姿、初始画面、时长与接触历史。
采集按实际姿态记录逐帧外参与时间戳，下游按保存的时间和 phase 对齐，
不按旧示例固定帧号判断接触。全程姿态统计可能包含初始收臂、转掌和接近，
工作阶段的掌心方向要求不能扩展为全程要求。

关节/执行器映射、观测与动作维度、相机安装及内参、触觉配置和 Raw 字段是分别维护的合同。
修改初态或时序后应核对这些边界，并重新验证真实录制；
接口回归与专家自动执行不能替代历史 checkpoint 的完整闭环评估。

<a id="appearance"></a>
## 外观与资产

共享场景使用白色手臂和机身、银色手、深灰石纹桌面与浅色背景。
以下配置与资产路径相对于 `src/kaihand_tactile_env/`，生成脚本相对于仓库根目录：

| 内容 | 配置或资产 | 生成与来源记录 |
|---|---|---|
| 机器人、桌面、灯光和背景 | `shared/mjcf/robot.xml` | 共享 MJCF |
| 双手显示外壳与腕部护罩 | `assets/workcell/mjcf/`、`assets/workcell/meshes/` | `assets/workcell/meshes/hand_visual_shells.json`；`scripts/workcell/build_hand_visual_shells.py` |
| 深灰石纹桌面 | `assets/workcell/textures/dark_grey_marble_v1.png` | 同目录 JSON 保存提示词与 SHA-256；[纹理说明](../../src/kaihand_tactile_env/assets/workcell/textures/README.md) |

SciPy 仅用于离线网格生成，运行时加载已有 STL；XML、网格、纹理与来源 JSON 纳入 Python 包资源。
视觉外壳、碰撞几何与触觉接触网格分开维护。任务专属视觉资产、有效孔位和装饰几何的
边界由各任务指南维护，例如[USB 场景](../tasks/usb_tasks.md#usb-insert)。

已有 Raw 的实际相机画面与三维运动使用[离线回放](../workflows/data_visualization.md)查看。
按当前外观重绘历史轨迹属于派生可视化，不是新的物理采集；原始 RGB 与重绘画面分别标识，
不把旧触觉与重绘 RGB 当作新采集训练数据。专属预览入口见各任务指南。

历史记录保留保存时的 qpos，不会被新 shared home 或新外观改写。
外观和接近分布变化可能影响旧模型；模型指纹变化后应使用新采集目录，
不绕过历史批次的源码一致性检查。v1–v7 外观修改与旧预览位置保存在
[任务与场景历史](../archive/task_scene_history.md)，未随仓库保留的预览不可作为当前入口。
资产来源与分发限制继续保留在
[第三方声明](../../src/kaihand_tactile_env/assets/workcell/THIRD_PARTY_NOTICES.md)。
