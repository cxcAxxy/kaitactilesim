# 统一采集入口

本目录提供 Raw 采集的命令入口；采集配置、调度、发布和续采只在工作流说明维护。
模型数据格式与离线回放分别由对应文档负责。

正式说明见[统一数据采集](../../docs/workflows/data_collection.md)，环境与导入来源检查见
[环境与工作目录](../../docs/workflows/environment.md)。

采集器注册了十三个任务。下面的查询不会启动 MuJoCo 或 recorder：

```bash
cd /cpfs_infra/user/chenxianchi/code/kaitactilesim
PYTHONPATH=src pixi run --as-is collect -- --help
PYTHONPATH=src pixi run --as-is collect -- --list-support
PYTHONPATH=src pixi run --as-is collect -- --describe vase-wipe
```

任务配置、三相机合同、硬件/软件预检、成功数采集、输出与续采规则集中在正式说明中维护。
默认配置为 `production-v1`；旧名称 `legacy-compatible` 是同一配置的兼容别名。
新增任务时修改 `src/kaihand_tactile_env/pipeline/catalog.py` 与
`pipeline/collection.py::COLLECTION_SPECS`，再补任务配置和对应测试。
