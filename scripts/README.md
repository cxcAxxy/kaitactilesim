# 脚本入口与兼容约定

日常流程使用以下正式命令；任务、格式与模型能力由对应命令的
`--list-support` 查询。完整例子见
[统一命令速查](../docs/workflows/pipeline_quickstart.md)。

| 接口 | 用途 |
|---|---|
| `/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run collect -- ...` | 任务配置、Raw 采集、校验发布与续采 |
| `/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run convert-dataset -- ...` | Raw 发现、语义转换与转换记录 |
| `/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run deploy-policy -- prepare/verify/serve/run ...` | 模型准备、核验、启动与接入仿真 |
| `/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run evaluate-policy -- ...` | 已有 deployment 的专项评估 |
| `/cpfs_infra/user/chenxianchi/miniconda3/bin/pixi run replay -- ...` | Raw 离线回放与复核 |

`pixi.toml` 用 `[stable]` 标记上述入口。`record`、`convert-egosteer`、
`egosteer-rollout` 为 `[compatibility]`，参数和既有行为仍可使用。

`workcell/` 中的任务 recorder、转换器、prepare、server、runner 是统一接口调用的
后端；预览和诊断也是开发工具。日期命名的脚本保留为历史批次复现入口，
其中默认路径与参数只对应原实验。新批次优先使用正式入口与配置文件。

第一轮整理保留后端文件路径：现有测试、裸模块导入和冻结的部署源码指纹都可能
引用这些文件。后续归档前应逐项检查 registry、导入、Pixi、文档、测试与部署引用。
`runs/` 下的脚本、报告和数据继续作为对应实验的复现包保存。
