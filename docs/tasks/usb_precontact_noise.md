# USB 接触前控制噪声

USB 自动任务在接触物体前，可向手腕目标的世界 X/Y 平移加入平滑噪声。CLI 默认标准差为 0.5 mm；`--precontact-noise-mm 0` 可关闭。空闲查看场景不施加噪声。

```bash
# 使用默认噪声；每次自动生成并记录 noise seed
pixi run run-usb-insert

# 固定物体初态和控制噪声，保存完整控制轨迹
pixi run run-usb-insert -- --headless \
  --xy-jitter-mm 10 --yaw-jitter-deg 5 --seed 42 \
  --precontact-noise-mm 0.5 --noise-seed 1 \
  --result-json /path/to/usb_seed42_noise1.json

# 无噪声基线
pixi run run-usb-insert -- --precontact-noise-mm 0
```

物体初态的 `--seed` 与控制噪声的 `--noise-seed` 相互独立。只固定前者能复现摆放，不能保证控制轨迹相同；在模型和控制器版本相同的前提下，同时固定两种 seed 及相关参数才能复现加噪动作。初态定义见 [USB 初始位姿随机化](usb_randomization.md)。

## 噪声合同

- X/Y 使用独立高斯结点，每轴截断到 `±min(3σ, 2 mm)`；默认范围为 ±1.5 mm。
- 结点间隔 0.30 秒，通过五次平滑插值生成连续偏移，不是逐物理步白噪声。
- 接近抓取位置时，偏移倍率平滑缩小到 0.2。
- 任一指尖与 USB 的真实法向力大于 `1e-6 N` 时锁存首次接触并停止采样。
- 已施加偏移在 0.12 秒内连续归零；此后即使短暂失去接触，本条动作也不重新启动噪声。
- 噪声只修改手腕目标的世界 X/Y，不修改 Z、旋转、手指目标、物体状态或传感器读数。

## API 与记录

`UsbInsertionExecutor` 的 Python API 默认不加噪，需要显式传入米单位：

```python
result = UsbInsertionExecutor(
    sim,
    precontact_noise_std_m=0.0005,
    noise_seed=1,
).execute()
```

结果的 `precontact_noise` 保存实际 seed、标准差、限幅、首次接触时间／指垫／法向力、采样次数和最大偏移。完整 JSON 中的 `commands` 保存名义与实际手腕目标、随机偏移、恢复偏移和机械臂关节目标；终端只打印摘要。

## 验证要求

- 首次真实触觉后，采样次数不得继续增加，随机偏移保持为零，恢复偏移按时归零。
- 控制目标在阶段边界必须连续，不能把上一阶段偏移重复叠加。
- 固定 seed 的加噪动作仍须通过抓取、关节余量、碰撞、姿态、插入和松手终态检查。
- 有限 seed 通过只证明实现和记录一致，不代表随机范围成功率或训练收益。
