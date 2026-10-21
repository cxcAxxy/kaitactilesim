# 统一评估：离线导入、核验与观察台

本页维护已有归档的离线发现、导入、来源核验和站点查看/发布；
新模型 rollout 与结果统计见[模型评估](model_evaluation.md)，阶段判据见[阶段规则](staged_evaluation.md)。

命令从当前仓库执行，Python 与导入路径见[运行环境](environment.md)。

## 查看当前工作台

当前网页入口是 [`reports/index.html`](../../reports/index.html)，它指向已发布的不可变快照。
2026-10-21 只读复核时，入口为 `kaitactilesim_unified_workbench_20261021_cpm_n10_r2`，
605 条成员／48 个原协议分组；后续以实际入口和该快照 catalog 为准。
历史 360／450／595 条站点仍有独立链接，不能把这些重叠成员再次累加。
当前表按任务、模型、checkpoint、协议和批次分组；成员数不等于有效计分条数。

```bash
cd /cpfs_infra/user/chenxianchi/code/kaitactilesim
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
.pixi/envs/default/bin/python -B scripts/analysis/serve_evaluation.py --directory reports --port 8877
```

启动前检查现有服务；端口已由正确目录提供服务时无需重复启动。
默认监听 `127.0.0.1`，远程 IDE 转发 8877 后访问 `http://127.0.0.1:8877/`。
根目录使用整个 `reports`，以便详情页中的归档链接可访问。

### 内网直接访问

需要内网直连时，可显式绑定本节点可用的 loopback 或 RFC1918 IPv4，例如：

```bash
.pixi/envs/default/bin/python -B scripts/analysis/serve_evaluation.py \
  --directory reports --host 192.168.0.54 --port 8877
```

上例 IP 是原节点示例，执行前应核对实际网卡；客户端须能路由至该私网地址。
服务只监听指定地址，提供 GET/HEAD 与视频 Range 播放。实例停止或重启后须重新启动；
监听不会建立 VPN 或网络路由。长期访问可使用内网常驻服务与固定地址。

## 查询规则与发现已有归档

以下查询不启动推理或仿真：

```bash
.pixi/envs/default/bin/python -B scripts/analysis/evaluate_unified.py rules
.pixi/envs/default/bin/python -B scripts/analysis/evaluate_unified.py discover \
  --source-root /cpfs_infra/user/chenxianchi/kaitactilesim_evaluations
```

规则查询描述当前代码，发现归档不表示它们都满足导入、物理、来源或完整时长要求。
导出器应使用记录自身的 checkpoint、规则与来源身份，不以当前规则重判历史轨迹。

### USB 拔出规则变更与历史边界

当前源码的 `usb-unplug` 使用 `usb-unplug-held-clear-2s-v2`：完全清离插座后，
右手仍有实际接触、插头无桌面接触共同连续 ≥2 秒，不再要求放回、释放、退手。
原生与 staged 记录各自的协议和连续计时，具体见[成功判据](../tasks/success_criteria.md#usb-unplug)
及[阶段判定](staged_evaluation.md#usb-unplug完全拔出并手持保持-2-秒)。

现有 360 条 H50 网站仍展示当时归档的严格放回规则和原成绩，未重算或重命名其阶段。
新试验应使用新的输出目录及匹配当前源码、指令和完成协议的 deployment；
不要修改旧 manifest 或旧来源 SHA 绕过冻结检查。若需对旧轨迹按新规则复评，应另外导出
明确标记的复评结果，并核对完整连续物理证据；单个终态/旧布尔标志不足以证明手持两秒。

`scripts/deploy/run_pi05_staged_groups.py` 及其 handoff/retry 工具针对旧的只读冻结快照和
9 任务批次，仍按旧阶段定义验收，保留用于历史队列复现；它不是新版 USB 两秒保持的发布器。
新版试验使用 `scripts/evaluate/evaluate.py` 的通用入口及匹配当前协议的部署，不复用旧批次计划。

### 重新导出及完整核验的前提

`export` 和 `verify` 都会严格核验 `integration_manifest.json` 中两份历史原工程的 source。
该清单保存合并时的文件内容身份，不因当前仓库路径变化而更新；当前同名目录也不代表原 source tree 已恢复。
本节点尚不具备可核验的两份旧来源，完整 `verify-originals` / `verify` / `export` 会受到来源检查阻挡。
已有网页文件仍可单独核验。不要删除来源保护、重写 SHA，或把历史核验成功当作当前复核结果。
重新导出前，须恢复可验证的原来源，并提供仍可访问、具有原回退清单的旧观察台目录。

满足上述前提后，从本工程根目录执行以下模板，替换 `/path/to/...` 为真实路径：

```bash
.pixi/envs/default/bin/python -B scripts/analysis/evaluate_unified.py verify-originals
.pixi/envs/default/bin/python -B scripts/analysis/evaluate_unified.py export \
  --source-root /cpfs_infra/user/chenxianchi/kaitactilesim_evaluations \
  --previous-workbench /path/to/preserved_h50_workbench \
  --output reports/staged_h50_new --workers 4
.pixi/envs/default/bin/python -B scripts/analysis/evaluate_unified.py verify reports/staged_h50_new
```

输出必须是不存在的新目录；已有报告不会就地更新。小批检查可增加 `--trial-limit 9`，
优先选取每个任务各一条；`--no-model-plots` 可跳过缺图补绘。
额外的完整 staged 记录通过 `--extra-root /path/to/evaluation` 显式加入；每条仍须通过归档身份和完整性校验，不能仅凭目录名导入。
默认回退锚点指向旧工程 H50 多任务站点；旧目录退役时，不能省略上述来源前提。

### 只核验已有网页文件

若原来源已不可访问，仍可按原生成物清单逐文件核验已有报告。以下只读检查只证明
`artifact_manifest.json` 所列文件的内容，不等价于完整 `verify`、原源码保护或模型评估复现：

```bash
.pixi/envs/default/bin/python -B - <<'PY'
import hashlib
import json
from pathlib import Path

manifest = Path("reports/staged_h50_20261015/artifact_manifest.json")
rows = json.loads(manifest.read_text())["files"]
for row in rows:
    path = Path(row["path"])
    assert path.stat().st_size == row["size"], path
    with path.open("rb") as file:
        assert hashlib.file_digest(file, "sha256").hexdigest() == row["sha256"], path
print(f"Verified {len(rows)} archived report files")
PY
```


## 分层证据与统计边界

| 层 | 来源 | 含义 |
|---|---|---|
| 原任务结果 | 归档 summary 的 task_success/status | 当次执行记录的任务结果，保留原值 |
| 阶段结果 | staged_evaluation.json / staged_states.jsonl | 按当时规则累计曾到达的阶段，保留首次到达时间 |
| 数据审核 | 物理标志、来源 SHA、日志时钟、合同校验 | 是否具备所需证据；不覆盖上两层事实 |
| 动作与触觉观察 | 模型/专家视频、Fn/Ft、关节与腕部曲线 | 帮助观察行为与载荷关系，不构成因果证明 |

阶段统计仅在相同任务、模型、权重内容、批次和归档规则身份内计数。
同为 `staged_v1` 不代表源码相同：例如扑克修复批次单独保留其 observer SHA。
当前规则导出的目录 SHA 也不是历史规则实现的 SHA。导入验证归档 inventory/manifest 与记录合同，
不会声称已经按历史源码重新仿真或重算判定。离线读取依照实际归档合同及来源身份，
不会用当前在线规则替换归档定义。

当前代码共有 13 个任务，其中 10 个注册了 `staged_v1`；完整列表、数值门槛和累计方式见
[阶段评估规则](staged_evaluation.md)。原生任务成功及专家完整执行的条件见
[各任务成功条件](../tasks/success_criteria.md)。历史 360 条归档只包含当时的 9 个任务，
没有新任务 `bulb-screw-inplace`；新任务已能产生当前阶段观测，不等于已验证历史站导入链路。

上述历史 H50 站点导入的 staged 记录均为 diagnostic_only，不应标为正式模型成功率或跨任务综合排名。
白板三类任务仅定义拿起阶段，`complete_success=null`：未定义完整擦拭结果，不是 0，也不等于成功擦净。
几何覆盖与真实去污/接触面积亦不混同。失败、未知及证据缺失都保留。

## 时间与力的解释

- 专家视频使用自己的播放时间线，不与模型按秒对齐，不计算时间轴相似度。
- 只有明确共享原生时钟的 staged 记录才提供“跳到模型首次到达”；否则保留独立绝对时间和 unmapped 标志。
- 保留日志真实采样间隔和缺口，不将累计阶段标志解释为当前接触，不插值补造状态。
- 五指法向载荷与切向载荷是局部测点/手指载荷的标量汇总，不是统一世界坐标的净合力。
- 台面支撑、板擦—白板、插座等对象交互力来自各任务物理记录，与手指载荷分开标注。
- USB 插入缺少原三类 PNG 时，由模型原生 NPZ 补绘模型自身曲线，不伪造专家对照。


## 工作台快照与发布

完整历史 H50 重导出、规范归档状态快照、统一工作台展示扩展是不同入口。
后两类可以复用已发布快照及已绑定归档，不代表已恢复或重验退役原始 source tree，
也不取消前述 full-H50 `export` 的来源保护。

按对应导出器生成不存在的新目录，核对身份、分组分母、视频/数据文件和浏览器页面后，
才使用该导出器的显式发布步骤切换 `reports/index.html`。不要仅为更新展示重跑模型或覆盖旧快照。
历史站点迭代、目标参考与贴合面积说明、各模型固定批次和验收证据统一从[档案索引](../archive/README.md)查阅。
