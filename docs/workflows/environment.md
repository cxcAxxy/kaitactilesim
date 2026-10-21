# 运行环境与源码来源

当前工程根目录为 `/cpfs_infra/user/chenxianchi/code/kaitactilesim`。采集、转换、部署、
评估和离线报告命令都从这里运行；外部模型训练使用对应项目自己的环境。
历史归档中的 source root、checkpoint 路径和 SHA 表示当时的来源身份，不能随着工作目录改名。

## 进入工程并核对源码

每个新终端先执行：

```bash
cd /cpfs_infra/user/chenxianchi/code/kaitactilesim
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
.pixi/envs/default/bin/python -B -c 'import sys, kaihand_tactile_env; print(sys.executable); print(kaihand_tactile_env.__file__)'
```

第二行输出应位于本仓库的 `src/kaihand_tactile_env/`。`PYTHONPATH` 前缀使本次终端及
其子进程优先加载这份源码；没有重写安装环境或历史部署。直接运行脚本时也可使用
`PYTHONPATH=src .pixi/envs/default/bin/python -B scripts/collect/collect.py --list-support`。

2026-10-21 核对时，本仓 `.pixi/envs/default` 的 editable 安装仍指向
`/cpfs_infra/user/chenxianchi/code/kaitactilesim_unified/src`。仅修改 `cd` 或选择本仓
Python 不足以保证导入本仓源码，尤其是采集、转换入口。使用上述前缀后再查支持范围。

## 安装与依赖检查

本节点 Pixi 的完整路径为 `/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi`。
新节点在工程根目录按锁文件完整安装：

```bash
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi install --locked
```

安装完成后仍应核对上面的源码位置。其他机器可以使用自己安装的 `pixi`，无需复制本节点的绝对路径。
已安装环境可通过 `pixi run --as-is ...` 或 `.pixi/envs/default/bin/python -B ...` 查询和运行；
`--as-is` 不会补装缺失依赖。

2026-10-21 本节点环境没有 `pyarrow` 和 `lerobot`。它曾通过缓存离线恢复并跳过 pyarrow，
该操作不是通用安装步骤。可用下面的只读命令查看所选 Python 的当前依赖状态：

```bash
.pixi/envs/default/bin/python -B -c 'import importlib.util; print({name: importlib.util.find_spec(name) is not None for name in ("mujoco", "pyarrow", "lerobot")})'
```

正式转换和参考数据读图前补齐对应后端的依赖。OpenPI、LeRobot、EgoSteer、EgoTouch、
CPM 和 FTP-1 等模型依赖使用自己的 Python/CUDA 环境，不能以仿真环境是否装有模型包判断就绪。
转换和部署的命名环境配置见 [runtimes.example.toml](../../configs/runtimes.example.toml)，
触觉模型环境见 [runtimes.tactile.toml](../../configs/runtimes.tactile.toml)。配置中的相对路径
以配置文件所在目录解析；复制配置后先修改为自己的实际环境。

## NVIDIA EGL 与软件渲染

本节点硬件渲染需要显式指定 NVIDIA EGL vendor 配置。先选择目标任务做检查：

```bash
__EGL_VENDOR_LIBRARY_FILENAMES=/cpfs_infra/user/chenxianchi/.config/kaitactilesim/nvidia_egl_vendor.json \
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is collect -- \
  --task poker-draw --check --render-backend hardware
```

检查应报告实际 NVIDIA renderer 与 `software: false`。`--render-backend hardware`
只选择并检查硬件模式，不会自动设置 vendor 路径，也不会静默回退。
没有 NVIDIA EGL 时去掉环境变量前缀，改为 `--render-backend software`；软件渲染通常较慢。

## 查询当前能力

在已进入工程并设置 `PYTHONPATH` 的终端中执行：

```bash
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is collect -- --list-support
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is convert-dataset -- --list-support
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is deploy-policy -- --list-support
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is evaluate-policy -- --list-support
```

任务注册、格式支持、部署阶段能力与已有匹配 checkpoint 分别核对。
具体操作从[统一命令速查](pipeline_quickstart.md)开始，旧批次和发布过程见[历史记录](../archive/README.md)。
