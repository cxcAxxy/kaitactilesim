# Sponge grasp：柔性海绵抓取与入盘

任务场景为 `sponge-grasp`。右手从桌面抓起柔性海绵，保持五指接触，搬运到右侧盘子，
松开并等待稳定。海绵由 MuJoCo flex 和真实接触力驱动，没有隐藏的物体附着或位姿控制。

任务配置、初始位置和接触参数在
[`tasks/sponge_grasp/config.py`](../../src/kaihand_tactile_env/tasks/sponge_grasp/config.py)；
仿真、随机化和任务指标在
[`tasks/sponge_grasp/task.py`](../../src/kaihand_tactile_env/tasks/sponge_grasp/task.py)。
正式采集的水平位置偏移每轴限制在 ±1 mm，并由 seed 复现。

采集使用[统一采集入口](../workflows/data_collection.md)，任务名为 `sponge-grasp`。Raw 使用共享
HDF5 与结果 sidecar；质量检查应同时确认任务成功、五指接触、海绵入盘和 Raw 校验。
回放使用[Raw 可视化入口](../workflows/data_visualization.md)。

目前注册了 π0.5 和模型中立 LeRobot v3 的 `head + right_wrist` 转换，
见[转换文档](../workflows/data_conversion.md)。LeRobot v3 会验证 100 Hz 控制步相机时间戳、
采集 sidecar 的任务审计，并保留柔体顶点和盘子支撑力等诊断。EgoSteer、EgoTouch
转换和统一闭环评估尚未注册，不能用其他任务的 action 合同替代。
