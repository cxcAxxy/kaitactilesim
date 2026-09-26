# Pick and place 当前示例

场景继承 shared 的机器人、双臂初姿、桌面和相机。任务物体为浅灰圆柱和白色盒子。
自动控制器从 shared home 连续运动，不在 reset 中预定位。

抓取前的抬高路点由 20 cm 降到 4 cm，搬运抬升由 30 cm 降到 18 cm，
释放后只抬手 9 cm 再收臂。手臂目标每个 2 ms 物理步更新；相邻五次曲线段保持
位置、速度和加速度连续。松手时保持手臂静止 0.18 秒，沿用原有张手力上限，
等手指充分打开后再撤离；松手前已解除原有搬运软约束。
初始和最终姿势仍为共享 ARM_HOME。

为减少 place 减速时的触觉粘滑，圆柱滑动摩擦系数由 5.0 调整为 1.5，搬运约束的
时间常数由 5 ms 放宽为 10 ms，并删除 place 阶段仅对食指施加的额外闭合量和限力。
五指使用抓取阶段确定的保持目标。下降前根据实测圆柱轴线做最小角度扶正，再用
1.10 秒完成垂直下降。默认示例中，释放前倾角约由 6.7° 降到 2°以内；place 的
总切向力峰值和最大相邻采样跳变也明显降低。曲线仍保存 100 Hz 原始求解器接触力，
没有用滤波掩盖波动。

示例目录 `datasets/pick_place_example` 按 `poker_draw_example` 的紧凑结构组织：

- `raw/episode_000000_cylinder_right.h5` 与同名 JSON：完整 Raw 和采集清单。
- `review/review.mp4`：采集的 head / right_wrist RGB 与右手五指 Fn / Ft 热图，10 fps。
- `review/robot_global_short_path.mp4`：同次采集 qpos 的全局运动学回放，10 fps。
- `curves/right_hand_force_curves.png`：原始右手五指法向力和合成切向力，不平滑、不裁剪。

Raw 保留三路共享相机 head、left_wrist、right_wrist，320×240、30 Hz；
物理 500 Hz，状态和触觉 100 Hz。求解器接触力与相机位姿时钟比积分后状态早 2 ms，
初始帧为 0。全局视频采用距离视频时刻最近的保存状态，仅执行运动学，不执行物理步进，
不替换训练图像。逐帧索引、校验和与检查结果放在 Raw 同名 JSON 的 `example` 字段中。

重新生成并替换当前示例：

```bash
MUJOCO_GL=egl .pixi/envs/default/bin/python scripts/workcell/record_pick_place_example.py --overwrite
```

先在临时目录完成采集和验证，成功后替换旧示例，失败保留原目录。
新的动作和外观会改变图像、轨迹和接触力分布，不能据格式兼容推断旧模型性能不变。
新数据转换应使用 `--instruction "Put the light grey cylinder into the white box."`；
历史红柱蓝盒数据与旧部署清单的指令保持原值。模型服务闭环需独立评估。
