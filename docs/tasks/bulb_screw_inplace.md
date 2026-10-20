# 灯泡原位旋入：bulb-screw-inplace

这是独立的新任务，不是对 `bulb-screw` 的重命名或简化标签。
旧任务仍为“灯泡抓取并旋入”，保留桌面拾取、抬高、搬运、对正、入牙。
两份原工程及已有 H50 报告均不修改，新任务没有混入现有 360 条模型评估。

## 初态与五步流程

初态不算专家操作：灯泡已对正灯座并在入口刚啮合，未到座、未拧紧、灯灭，机器人双手在 home。
沿用原 v15 被动螺旋约束和接触模型，没有物体执行器、手—灯泡固定或新增外部旋转力。
场景仅去掉不再使用的桌面取料支架，灯座和灯泡碰撞参数不变。

1. 张手接近灯座上的灯泡。
2. 原位五指抓稳，确认实际接触稳定，不执行抬起、搬运或重新入牙。
3. 通过手指旋拧、卸载换指、重新抓稳循环推进，按实测灯泡角度到座。
4. 到座后继续施力，确认承力停转，再点亮反馈灯。
5. 松手、短距离撤离并验证无机器人支撑下稳定就位，随后回 home 收尾。

入口到终点的名义旋入量沿用约 100°，每段最多 30°，一般有四个有效旋入行程。
换指和末端 tighten 不算额外有效旋入次数；不以固定“四下”或“五下”作为成功条件。

## 初始化与随机性

reset 先通过真实齿接触门限啮合，再真实静置 0.3 秒，验证未到座、未亮灯、无机器人接触、
低速且剩余旋入行程充足，之后才归零计时和速度。记录静置前后的真实角度、深度和速度，
不能通过归零掩盖漂移。初态通常约 0.144°、6.0019 mm 插深；实际值以每条 initialization 为准。

原被动模型存在轻微自然漂移，不把物体冻结在坐标上：90 秒无手空跑约停在 10.24°，
仍未到座、未亮、无成功。只有抓稳之后实际推进才可获得有效旋入确认。
当前 `--seed` 保留运行身份，入口位姿**不随机化**；多个 seed 不能当成不同初态的泛化试验。

## 判定与记录

- 抓持：五指真实指腹法向载荷各至少 0.025 N、连续至少 0.10 秒，
  无掌部/指节或其它机器人部位作为隐藏支撑。
- 有效旋入：相对抓持起点和 episode 初态均至少实际旋转 30°、轴向推进至少 0.2167 mm，
  且仍满足有效抓持与啮合；不能由指令次数或空跑漂移获得。
- 机械到座：沿用旧任务的实际插深、角度、螺距误差、侧偏、倾角、肩部/止挡接触及速度检查。
- 拧紧：至少连续 0.3 秒，实际角变化不超过 0.3°，顺时针接触扭矩至少 0.1 N·m，五指各至少 4 N，
  并且物理到座、没有非指腹支撑。观测断档清除未完成窗口；真正脱扣或明显回退会使旧证书失效。
- 释放：整个机器人不再接触灯泡，并在实际到座状态稳定至少 0.2 秒。

全部机械到座几何/速度门槛、各历史锁存和回位容差，见
[原位旋入成功判定](success_criteria.md#bulb-screw-inplace)。
完整阶段解释见 [staged_v1](../workflows/staged_evaluation.md)；
新任务仅在未来真实导入对应评估时显示，不把旧站的模型轨迹改名。

`state.success` / `physical_success` 是物理任务结果，`home_return_verified` 单独记录收尾。
专家 `task_result.success` 表示完整执行流程正常结束；若物理完成后收尾被取消，
结果保留 `physical_success=true`，但完整流程失败并给出 `cleanup_failed` 原因。
灯光是反馈，不是拧紧证据：直接打开视觉灯、数够次数或回到 home 都不能伪造成功。

staged_v1 使用独立 task 身份和四个有序阶段：
`stable_inplace_grasp → threaded_and_seated → tightening_verified → released_stably`。
没有沿用旧任务的 pickup 前置，也没有更改旧九个任务的定义或阈值。
离线观察台已注册新任务的角度、深度、扭矩和原生标志；只有未来真实导入相应新评估才显示新轨迹。

## 本地运行

在 `kaitactilesim_unified` 根目录执行，无渲染检查或跑完整专家：

```bash
.pixi/envs/default/bin/python -B scripts/workcell/view_bulb_screw_inplace.py --headless --duration 2
.pixi/envs/default/bin/python -B scripts/workcell/view_bulb_screw_inplace.py --headless --run-task --seed 0
```

带显示环境时去掉 `--headless`，默认用灯泡近景；也可以使用
`pixi run --as-is view-bulb-screw-inplace`、`pixi run --as-is run-bulb-screw-inplace` 和 `pixi run --as-is check-bulb-screw-inplace`。
本机环境从缓存离线安装，暂排除了无缓存且物理测试不需要的 pyarrow：
`pixi install --locked --offline --skip-with-deps pyarrow`。当前可用 `pixi run --as-is ...`，
避免重新尝试安装缺失包；训练数据转换前需补齐完整依赖。

通用采集入口已接通 `--task bulb-screw-inplace`，三路相机、控制目标、状态、触觉和初态证据均独立保存。
新采集默认 Raw 相机 30 Hz、状态/触觉 100 Hz，执行器与判定仍在 500 Hz 物理步更新。
生产采集不允许把旧灯泡轨迹仅改名字充当此任务数据。
π0.5 / LeRobot v3 转换和 π0.5 部署评估均使用独立任务与环境身份；正式部署要求
新的 recorded-controller-target 数据合同，不允许重标旧任务 checkpoint 的 manifest。

2026-10-15 的本地验证仅覆盖专家与接口，没有训练或运行这个新任务的模型策略；
去掉拿取搬运也不会自动解决重复旋拧时的历史记忆或视觉捷径问题。

2026-10-18 已补齐独立 OpenPI H50 训练合同、自动归一化及安全八卡入口：
`pi05_kaihand_bulb_screw_inplace_controller_target_h50_v1`，绑定本任务自己的场景身份。
对 `1018_200` 的 metadata-only 只读预检不计算统计、不启动模型；真正训练前仍需完整数据/统计校验。
完整训练与评估命令见[流水线速查](../workflows/pipeline_quickstart.md#6-灯泡原位旋入任务)。
这次接入本身不代表已有模型训练或闭环成绩，旧拿取旋入配置及 checkpoints 保持原样。

## 已完成的本地验证（2026-10-15）

- 完整 CPU 专家：33.67 秒，4 个有效旋入行程，相对初态实际推进约 100.65° / 1.082 mm，
  无抬起搬运；原位抓持、有效旋入、到座承力、无机器人支撑释放及回 home 均通过。
- 新旧任务、staged 规则、归档导入及报告接口的选定回归：232 项测试和 44 个子测试通过；
  独立执行器另有 20 项测试通过（包括完整物理流程）。
- 真实 policy adapter 接入 staged_v1 空跑 0.2 秒：四阶段全部未通过，未误判成功。
- 实际 Raw 样例保存了 3,368 条状态、三路各 1,012 张相机帧；格式验证通过。
  16,835 个物理步全部进入记录审计，物体额外外力/广义力峰值为 0，无终态事后写入。
  触觉来自 link-level solver contact 的空间力阵列，并非独立球形探针或真机标定结果。
- `seed=0`、`seed=1` 为固定初态重复性检查，不算两种初态或模型成功率。
- unified 内原 `bulb_screw` 七个文件与修改前 SHA256 一致；既有 staged H50 报告的
  4,334 个归档文件大小与 SHA256 全量一致。两个旧工程根目录目前不在原路径，
  因此无法重新核验其 856 项原 source；本次没有恢复、创建或写入这两个旧路径。

专家 Raw：`runs/bulb_screw_inplace_example_20261015/raw/episode.h5`。
独立预览视频（头部 + 右腕视角及触觉显示）：
[本机 8877 播放](http://127.0.0.1:8877/bulb_screw_inplace_demo_20261015/review.mp4)。
三路原始相机仍全部保留；预览由已保存帧导出，没有重跑/修改物理轨迹，亦没有覆盖旧报告。

再次采集时使用新的输出目录：

```bash
PYTHONPATH=src .pixi/envs/default/bin/python -B scripts/workcell/record_task_variant.py \
  --task bulb-screw-inplace --seed 0 --output-dir runs/bulb_screw_inplace_next
```

本机没有显示设备时，可在命令前增加 `MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa`
及 `KAIHAND_RENDER_BACKEND=software LIBGL_ALWAYS_SOFTWARE=1`，使用 CPU 渲染。
当前示例是专家数据，不是已训练新任务模型的评估结果。
