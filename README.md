# KaiHand Tactile Simulation

KaiHand 双臂触觉操作仿真、Raw 数据采集、模型数据转换、闭环评估与离线观察台。
工程根目录为 `/cpfs_infra/user/chenxianchi/code/kaitactilesim`。

## 从这里开始

| 需要做什么 | 入口 |
|---|---|
| 查文档分工、选择任务指南 | [文档导航](docs/README.md) |
| 安装环境、核对源码、配置渲染 | [运行环境](docs/workflows/environment.md) |
| 按顺序完成采集、转换、部署、评估 | [流程速查](docs/workflows/pipeline_quickstart.md) |
| 查原生成功与专家验收 | [成功判据](docs/tasks/success_criteria.md) |
| 查阶段里程碑 | [阶段判定](docs/workflows/staged_evaluation.md) |
| 查看已有评估网页与离线归档 | [统一观察台](docs/workflows/unified_evaluation.md) |
| 查看项目内专家示例、Raw 与来源校验 | [专家示例数据](docs/workflows/data_visualization.md#项目内专家示例) |
| 复查具体旧批次与发布记录 | [历史记录](docs/archive/README.md) |

## 运行与覆盖范围

先按环境页安装依赖，确保当前终端加载本仓源码：

```bash
cd /cpfs_infra/user/chenxianchi/code/kaitactilesim
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run --as-is collect -- --list-support
```

流水线包含十三个独立任务，按抓放、摸牌、USB、灯泡、RAM、花瓶、白板、海绵八类维护
[任务指南](docs/README.md#任务指南)。每个 task 仍保留独立初态、mechanics、场景与完成协议。
采集、转换、部署和模型评估分别查询各自的 `--list-support`；
任务存在、模型接口接入和已有匹配权重是不同状态。

新 Raw 默认记录三路 RGB、状态、动作、触觉、时间戳与任务诊断。训练观测与动作受
[格式合同](docs/reference/dataset_formats.md)和模型来源约束；原生任务成功、专家结果、
有序阶段进度和统计有效性分别保存，不能互相替代。

```text
任务专家 → Raw → 转换训练数据 → 外部模型训练
             └→ 离线回放          └→ 部署 → 闭环评估 → 结果与视频
                                                     └→ 离线观察台
```

## 工程目录

```text
src/kaihand_tactile_env/
├── shared/          # 机器人、相机、触觉、Raw 和评估公共组件
├── pipeline/        # 任务、采集、转换、部署注册表与运行环境
└── tasks/           # 十三个独立任务的场景、控制和原生判定
scripts/
├── collect/         # 统一 Raw 采集
├── convert/         # 数据转换
├── deploy/          # 部署准备、核验和服务生命周期
├── evaluate/        # 模型闭环评估
├── analysis/        # 离线导入、核验和工作台导出
└── workcell/        # 任务后端、预览和诊断
docs/                # 分工与全部入口见 docs/README.md
task_demo/           # 专家示例媒体、原始审计元数据及绑定 Raw；本地配套数据
```

脚本等级及兼容入口见 [scripts/README.md](scripts/README.md)。默认 `pytest` 只收集 `tests/`，
`runs/` 保存对应实验复现包。新输出使用新目录，历史 Raw、部署和归档保留其来源身份。
`task_demo/` 是工程根目录下的独立数据目录，不属于 Python 安装包，也不随普通 Git 提交。
当前收录十一种任务的历史示例，不代表全部十三任务的当前版本验收集。

## 整合来源

2026-10-15 的整合保留仿真与评估执行，并接入离线分析和本地观察台；
快照来源与文件 SHA 见 [integration_manifest.json](integration_manifest.json)。
在线入口为 `scripts/evaluate/evaluate.py`，离线入口为 `scripts/analysis/evaluate_unified.py`。
离线导入不启动推理、不重跑物理、不按当前规则重判历史成绩。
