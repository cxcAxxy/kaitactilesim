# 脚本入口与兼容约定

日常流程从当前仓库根目录执行。环境安装、依赖限制和源码导入检查见
[环境与工作目录](../docs/workflows/environment.md)；完整例子见
[统一命令速查](../docs/workflows/pipeline_quickstart.md)。

```bash
cd /cpfs_infra/user/chenxianchi/code/kaitactilesim
PYTHONPATH=src pixi run --as-is collect -- --list-support
```

以下示例以 `PYTHONPATH=src pixi run --as-is` 为前缀。任务、格式与模型能力
由对应命令的 `--list-support` 查询。

| 命令 | 用途 | 说明 |
|---|---|---|
| `collect -- ...` | 任务配置、Raw 采集、校验发布与续采 | [采集](../docs/workflows/data_collection.md) |
| `convert-dataset -- ...` | Raw 发现、语义转换与转换记录 | [转换](../docs/workflows/data_conversion.md) |
| `deploy-policy -- prepare/verify/serve/run ...` | 模型准备、核验、启动与接入仿真 | [部署](../docs/workflows/model_deployment.md) |
| `evaluate-policy -- ...` | 已有 deployment 的专项评估 | [评估](../docs/workflows/model_evaluation.md) |
| `replay -- ...` | Raw 离线回放与复核 | [回放](../docs/workflows/data_visualization.md) |

`pixi.toml` 用 `[stable]` 标记上述入口。`record`、`convert-egosteer`、
`egosteer-rollout` 为 `[compatibility]`，参数和既有行为仍可使用。
FTP-1 使用独立转换入口，见[数据合同](../docs/reference/dataset_formats.md#ftp1)。

`workcell/` 中的 recorder、转换器、prepare、server、runner 是统一接口调用的后端；
预览和诊断是开发工具。日期命名的脚本用于对应历史批次复现，默认路径与参数只对应原实验。
新批次使用正式入口与配置文件。

后端文件路径属于兼容约定：测试、裸模块导入和冻结的部署源码指纹可能引用它们。
归档前须逐项检查 registry、导入、Pixi、文档、测试与部署引用。
`runs/` 下的脚本、报告和数据作为对应实验的复现包保存。
