# USB 与内存：先到位，再夹紧

USB 插入和内存安装的自动任务控制器采用分阶段抓取：

1. 张开拇指与食指，移动到物体上方，再接近抓取位置。
2. 保持张开并等待手腕到位。位置误差必须不超过 3 mm，姿态误差不超过 0.03 rad；否则终止本次任务，报告抓取位置未到达。
3. 保持手腕目标不动，平滑收拢到原有精细抓取姿态，再夹紧并验证接触力。
4. 确认抓稳后，继续提起、转移、插入和释放。

USB 接近时两个指尖 site 的中心距离约 61 mm，内存约 52 mm。这不是指腹表面的净空。其余三个手指保持避障用的弯曲姿态。原有最终夹持标定和力控制继续使用。

USB 新增的静止收拢时间为 `0.6 * grasp_ramp_s`（fast 为 0.6 s，baseline 为 0.75 s），内存为 0.6 s。轨迹分布及总时长因此改变；下游应按保存的时间戳和 phase 对齐，不能依赖历史示例的固定帧号。

## 兼容性

本次修改只涉及两项任务的抓取姿态与控制时序。相机、触觉配置、碰撞模型、动作维度、关节顺序和 Raw HDF5 字段没有变动。现有 `close` / `grasp` 阶段包含新增的收拢过程，不增加公共 phase 名称。

回归测试覆盖 USB 随机初态、接触前控制噪声、两个任务的完整物理执行、采集、转换调度和 USB 推理接口。实际采集样本还需要经过 Raw 校验及目标格式转换；这些验证不能替代已有模型 checkpoint 的完整闭环评估，也不保证旧模型在新轨迹分布上的性能。

USB 的 EgoSteer 后端要求平铺批次：`summary.json`、`usb_*.h5` 和配套 JSON 位于同一输入目录。USB 采集器输出的 `raw/` 子目录需要先与批次 summary 整理到独立的转换输入目录；保留源文件内容、校验和及原始批次，不覆盖旧数据。RAM 后端可以直接读取采集根目录。转换输出必须与输入目录相互独立。

## 检查命令

```bash
pixi run python scripts/workcell/view_usb_insert.py --headless --run-task --plug-for-insertion --precontact-noise-mm 0
pixi run python scripts/workcell/view_install_ram.py --headless --run-task

pixi run python scripts/workcell/validate_episode.py /path/to/new_episode.h5
pixi run python scripts/workcell/replay_episode.py /path/to/new_episode.h5 \
  --viewer none --review-dir /path/to/new_review --fps 10
```

预览页分别位于 [USB](../datasets/usb_insert_example/approach_then_grasp_v8/index.html) 和 [内存](../datasets/install_ram_example/approach_then_grasp_v8/index.html)，包括抓取近景、指尖距离曲线和新采集的完整多模态回放。近景使用记录的关节状态重渲染，展示相机不参与采集。
