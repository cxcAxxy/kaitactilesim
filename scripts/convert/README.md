# 数据转换入口

本目录提供转换 CLI；参数、adapter 分发、发布与恢复只在工作流说明维护。
字段、时钟、动作监督与来源语义统一见[数据格式合同](../../docs/reference/dataset_formats.md)。

正式说明见[统一数据转换](../../docs/workflows/data_conversion.md)，环境与导入来源检查见
[环境与工作目录](../../docs/workflows/environment.md)。

```bash
cd /cpfs_infra/user/chenxianchi/code/kaitactilesim
PYTHONPATH=src pixi run --as-is convert-dataset -- --help
PYTHONPATH=src pixi run --as-is convert-dataset -- --list-support
```

`--list-support` 是实际 adapter 能力表，包含任务、格式、相机、runtime、workers 含义、
恢复能力和后端特性。后端文件名不作为能力合同。

命名运行环境、计划与转换清单见[基本参数](../../docs/workflows/data_conversion.md#基本参数)，
各后端并发与恢复区别见[性能、校验与恢复](../../docs/workflows/data_conversion.md#性能校验与恢复)。
LeRobot v3 与 FTP-1 的差异见[数据格式合同](../../docs/reference/dataset_formats.md)。
FTP-1 使用独立入口，不传 `--format ftp1`。
