# LingBot-VLA-2.0 KaiHand PickPlace 评估

这套底层服务和 batch runner 仅用于 LingBot-VLA-2.0 PickPlace，不改变 pi0.5 或
其他模型的推理与成功判定。统一评估器已注册 PickPlace 的 `lingbot-vla2`
模型族，负责通用参数与分发。服务端需要 LingBot 的 CUDA
环境，仿真端使用 kaitactilesim 的 `.pixi/envs/default` 环境。

已生成并静态校验的部署清单：

```text
/cpfs_infra/user/chenxianchi/evaluations/pick-place/lingbot_vla2/step3905_deployment/deployment_manifest.json
```

在有 GPU 的节点启动模型服务：

```bash
cd /cpfs_infra/user/chenxianchi/code/kaitactilesim
/cpfs_infra/user/chenxianchi/miniconda3/envs/lingbot_vla2/bin/python \
  scripts/workcell/serve_pickplace_lingbot_vla2_policy.py \
  --deployment-manifest /cpfs_infra/user/chenxianchi/evaluations/pick-place/lingbot_vla2/step3905_deployment/deployment_manifest.json \
  --host 127.0.0.1 --port 8006
```

另一个终端运行一条评估，或把 `--seeds` 扩展为多个 seed：

```bash
pixi run evaluate-policy -- \
  --task pick-place --model-family lingbot-vla2 \
  --deployment-manifest /path/to/deployment_manifest.json \
  --output-dir /path/to/lingbot_eval \
  --num-trials 1 --video-count 1 --execute-steps horizon \
  -- --server 127.0.0.1:8006
```

以下底层命令保留用于直接调试 runner：

```bash
cd /cpfs_infra/user/chenxianchi/code/kaitactilesim
.pixi/envs/default/bin/python \
  scripts/workcell/evaluate_pickplace_lingbot_vla2_batch.py \
  --server 127.0.0.1:8006 \
  --deployment-manifest /cpfs_infra/user/chenxianchi/evaluations/pick-place/lingbot_vla2/step3905_deployment/deployment_manifest.json \
  --output-dir /cpfs_infra/user/chenxianchi/evaluations/pick-place/lingbot_vla2/step3905_seed000 \
  --seeds 0 --max-sim-seconds 90
```

执行步数默认等于模型 horizon（50）；控制频率 30 Hz，视频 10 fps，最长仿真时间默认 90 秒。模型只接收头部、右腕 RGB 相机和右侧 27 维关节状态。服务端返回反归一化后的右臂 7＋右手 20 维绝对关节目标。视频保留头部、右腕、全局三个视角和触觉热力图，不嵌入时间曲线。每个 seed 的 `review/` 目录都有完整控制步轨迹，以及右腕位姿、右手 20 关节、五指 Fn 和 |Ft| 三张独立对比图；`--video-count 0` 只关闭 MP4，不关闭图表。数据集第 00 条是实线，本次仿真是虚线。

这份训练集的原始 HDF5 没有直接保存 7×5 牛顿力网格。参考曲线通过原始 MuJoCo 接触求解器的逐帧 wrench 还原：Fn 对法向力求和；|Ft| 对两个有符号切向分量分别求和，再计算二维合力大小。当前触觉提供器的网格分配保守地保持这两个总量，因此与“35 个网格求和”的定义一致；不会把深度传感器的 proxy 当作真实力。

无需 GPU 可先给服务端加 `--validate-only`，只核验 checkpoint、训练 YAML、机器人映射、归一化统计及数据集。批量评估加 `--dry-run` 可核对协议和命令而不启动仿真。如果仿真节点没有 EGL GPU，批量评估命令前设置 `MUJOCO_GL=osmesa` 即可使用软件渲染；这不替代模型服务端所需的 CUDA GPU。

模型服务从已哈希的 `hf_ckpt/chat_template.jinja` 恢复 tokenizer 的提示格式：训练基础模型目录的 tokenizer 本身未设置模板。修改服务脚本后必须重启服务，旧进程不会自动加载新代码。失败的评估目录会保留供排查，重试时请使用新的 `--output-dir`。

需要重建清单时，用 `scripts/workcell/prepare_pickplace_lingbot_vla2_deployment.py --checkpoint <hf_ckpt> --output-dir <新的空目录>`；脚本会哈希约 24 GB 文件，且拒绝覆盖已有目录。
