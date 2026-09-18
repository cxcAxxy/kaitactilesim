# USB 初始位姿随机化

初始位姿随机化只作用于 USB 场景 reset 后、首个物理步之前的自由物体状态。它在世界 X/Y 方向分别采样有界平移，并围绕世界 Z 轴采样 yaw；高度、正反面、插座、摩擦和物体尺寸不变。

```bash
# X/Y 各 ±10 mm，基准朝向附近 yaw ±5°
pixi run run-usb-insert -- \
  --xy-jitter-mm 10 --yaw-jitter-deg 5 --seed 42

# 同时固定接触前控制噪声
pixi run run-usb-insert -- --headless \
  --xy-jitter-mm 10 --yaw-jitter-deg 5 --seed 42 \
  --precontact-noise-mm 0.5 --noise-seed 1 \
  --result-json /path/to/usb_random_seed42.json
```

`--seed` 决定物体初态，`--noise-seed` 决定另一套接触前运动噪声。只固定物体 seed 不保证动作轨迹相同；需要无噪声对照时增加 `--precontact-noise-mm 0`。噪声合同见 [USB 接触前控制噪声](usb_precontact_noise.md)。

## 几何与控制合同

- `--xy-jitter-mm 10` 表示 X、Y 各自位于 ±10 mm，不是总宽 10 mm。
- `--yaw-jitter-deg 5` 表示基准朝向 330° 附近 ±5°。
- `initial_pose_randomization` 保存 seed、采样范围、实际 `offset_xy_m`、`yaw_offset_rad` 和首个物理步前的 `initial_pose_wxyz`。
- 抓取目标围绕 USB 同时变换位置和方向，保持标定的物体—手指捏持关系。
- 物体只在首个物理步前设置一次；执行期间由机器人执行器和接触推动，不重设自由物体状态。
- 随机 yaw 会调整机械臂 IK 初值以保持肘部外展，但不会直接写入实际关节状态。

初始翻面或明显倾斜仍会被拒绝；当前功能不包含翻面重抓，也不模拟视觉定位误差。

## 批量检查

`check-usb-random` 使用无噪声 API，串行检查固定初态、均匀样本和 X/Y/yaw 边界组合：

```bash
nice -n 10 pixi run check-usb-random -- \
  --samples 8 --seed 20260909 \
  --output-dir /path/to/new_usb_random_check
```

输出目录必须不存在。报告分别统计任务成功和质量合格，质量项包括关节余量、桌面／支架／插座碰撞、正手姿态、肘部外展、动作时间和腕部额外往返角程。

当前第一轮采集合同允许 X/Y 各 ±10 mm、yaw ±5°，但这不是对连续范围的成功保证。正式采集仍须保留失败样本及 seed，并对每条成功数据执行 Raw 验证和 USB Cleaning；启用接触前噪声后应单独统计成功率，不能沿用无噪声检查结果。
