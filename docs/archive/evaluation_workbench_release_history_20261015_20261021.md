# 评估工作台发布与来源档案：2026-10-15 至 2026-10-21

本页保留 H50 整合、状态快照、实验页及工作台展示迭代的操作和证据。各节的数量、
“当前/本批”与服务地址均属于该次发布时的历史状态，不能据此确定今天的首页或计分范围。
当前查看、导入与核验流程见[统一评估与观察台](../workflows/unified_evaluation.md)，
节点环境见[环境说明](../workflows/environment.md)。

原整合来源为 `kaitactilesim` / `kaitactilesim_laest` 当时的工作树；`integration_manifest.json`
保留其大小、SHA 与 Git HEAD。这些历史来源身份不随当前仓库重命名或文档整理改写。
本页复用模板的 `cd` 已指向当前 `kaitactilesim`，但旧 source tree 的独立验证要求仍然保留。

## 同一 H50 实验的四个 checkpoint（450 条统一入口）

2026-10-19 的平台根入口按实验组织 π0.5 H50 结果，而不是将历史 360 条与新增 90 条分成两个主入口。
四个 checkpoint 为 `10000`、`15000`、`20000`、`24999`，分别有 45、45、180、180 条成员记录，
覆盖同一组 9 个任务，共 450 条。每个任务的新旧 checkpoint 使用同一个训练目录；
成员身份仍绑定任务、H/K、checkpoint 与原始摘要 SHA，不靠 seed 或成绩匹配。

450 是实验成员数，不是 450 条都可计分：其中 430 条维持现有可统计身份，20 条旧水平线迹白板
step20000 记录保持 `quarantined_replay_only`，仅能经隔离说明查看原工作台回放，不进入原生或
阶段成功分母。本次没有修复、重判或解除其原有发布隔离。USB-out 新旧完成协议仍分组显示；
同一实验不意味着不同 checkpoint、规则、原生成功和 staged 成功可混成一个总成功率。
EgoTouch 的 120 条放在独立实验页面，不计入此 450 条。

```bash
cd /cpfs_infra/user/chenxianchi/code/kaitactilesim
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
.pixi/envs/default/bin/python -B scripts/analysis/export_evaluation_experiment.py \
  --status-snapshot reports/evaluation_status_20261019_r2 \
  --historical-report reports/kaitactilesim_official_h50_20261015_r2 \
  --output reports/h50_checkpoint_experiment_20261019 \
  --publish-index
.pixi/envs/default/bin/python -B scripts/analysis/export_evaluation_experiment.py \
  --verify reports/h50_checkpoint_experiment_20261019
```

输出仍必须为不存在的新目录。导出器核验父快照、原工作台的生成物清单和来源身份，复用已有详情、
视频与曲线，不重跑评估、不重复复制媒体、不改写旧站点。旧 source tree 的完整原始核验保护不变，
退役原始 source tree 不宣称本次已验证。根入口仅在快照验证通过后原子切换，并保存旧入口备份。

本机服务仍为 `http://127.0.0.1:8877/`（远程 IDE 转发 8877）。统一入口直达
`http://127.0.0.1:8877/h50_checkpoint_experiment_20261019/index.html`，独立 EgoTouch 实验位于
`http://127.0.0.1:8877/h50_checkpoint_experiment_20261019/egotouch.html`。

## 先前的各任务归档状态快照（只读归档发布）

`export_evaluation_status.py` 从当前规范归档生成一个静态状态快照，并保留现有 360 条 H50
工作台入口。它不是上文的旧 full-H50 `export`：不会重跑仿真、重判历史轨迹或改写来源目录，
也不会取消 `verify-originals` 对完整 H50 重导出的保护。此快照只核验当前归档元数据、复制的
生成物字节和历史站点的生成物清单；退役的原始 source tree 仍明确记录为**未在本次验证**，
不得据此宣称它们已重新核验。

从本工程根目录发布一个不存在的新目录：

```bash
cd /cpfs_infra/user/chenxianchi/code/kaitactilesim
.pixi/envs/default/bin/python -B scripts/analysis/export_evaluation_status.py \
  --source-root /cpfs_infra/user/chenxianchi/kaitactilesim_evaluations \
  --historical-report reports/kaitactilesim_official_h50_20261015_r2 \
  --output reports/evaluation_status_20261019_r2 \
  --publish-index
.pixi/envs/default/bin/python -B scripts/analysis/export_evaluation_status.py \
  --verify reports/evaluation_status_20261019_r2
```

输出目录必须是与历史站点同级的 fresh canonical directory；同名目录已存在时应换用新的时间戳或
版本号，不能覆盖。导出器先完成并核验快照，只有核验通过且根入口在导出期间未被其他进程修改时，
`--publish-index` 才原子替换 `reports/index.html`。替换前的根入口保存在新快照的
`previous_root_index.html`，入口切换另记在 `portal_publication.json`；若导出或核验失败，旧入口保持不变。
发布后也应执行上面的独立 `--verify`，并以其成功结果作为交付记录。

归档选择以 canonical index 为准。与历史 H50 `summary.json` SHA-256 完全匹配的轨迹直接链接已保留的
旧详情，不复制或修改旧 payload；规范归档中的 EgoTouch 与新评估组使用各自冻结的原生摘要、视频和
证据生成新详情。canonical index 明确隔离的旧未发布记录继续保持隔离，不导入、不补算，也不按当前
规则重新判定。实际任务、分组和轨迹数量应读取新快照的 `catalog.json` 与验证输出，不在流程文档中
写死候选扫描数量。

展示和统计始终分开保存原生成功、staged 阶段、`diagnostic`／`formal` 身份及完成协议；只在相同任务、
checkpoint、模式、协议和规则身份内统计，禁止把不同协议的结果汇成一个总成功率。归一化作业只能显示
为作业状态，不是模型评估轨迹，也不产生成功率。来源评估目录、历史 360 站点及其 payload 全程只读。

该页面是发布时刻的静态快照，不是自动监控面板；来源变化后须用新的输出目录重新导出、核验并切换入口。
本机仍复用只读静态服务：

```bash
.pixi/envs/default/bin/python -B scripts/analysis/serve_evaluation.py --directory reports --port 8877
```

服务默认监听 `127.0.0.1`；远程 IDE 转发 8877 后访问 `http://127.0.0.1:8877/`，也可直接访问
`http://127.0.0.1:8877/evaluation_status_20261019_r2/index.html`。启动前先检查端口，已有服务时无需重启。

## 各任务目标参考与白板贴合诊断版

保留的 360 条 H50 工作台入口为 `http://127.0.0.1:8877/kaitactilesim_official_h50_20261015_r2/index.html`。
它基于 `staged_h50_studio_20261015_r2`，在现有布局上统一 KaiTactileSim 官方品牌与栏目名称，
保留 USB/白板参考版全部目标线、白板贴合指标、视频和原始判定。
保留全部 360 条 H50 轨迹，9 种任务各 40 条，逐条增加来源核验的单项参考线。
三种白板的 120 条轨迹还补充已归档的几何贴合指标和最大单元墨迹余量；不重跑仿真。
原 `staged_h50_20261015` 稳定版及仅有灯泡参考线的 `staged_h50_goal_refs_20261015`
和 RAM/抽卡参考版 `staged_h50_goal_refs_ram_poker_20261015` 的网页、数据和生成物清单保持不变。

### KaiTactileSim 官方品牌展示版

`kaitactilesim_official_h50_20261015_r2` 使用主标题“KaiTactileSim 官方评估平台”，
英文品牌“KaiTactileSim Evaluation Platform”，副标题“机器人操作任务评估 · 动作回放 · 触觉分析”。
总览浏览器标题保留 H50 批次标识；详情页采用“任务名 · 轨迹分析”和“任务名 | KaiTactileSim”。
栏目统一为评估总览、轨迹分析、专家示范、阶段记录与评估规则，保留来源审计和诊断说明。

官方品牌表示平台归属，不表示该批记录已成为正式认证或新的排名；
`diagnostic_only`、`formal_metrics_valid`、原生结果、阶段记录及分组统计均保持原样。
“同规则、同权重的阶段统计”仍位于数量提示下方、轨迹列表上方。
这是本地展示改版，没有新增局域网监听、开放防火墙端口或改变访问权限。

```bash
.pixi/envs/default/bin/python -B scripts/analysis/evaluation_goal_reference_snapshot.py \
  --source-report reports/staged_h50_studio_20261015_r2 \
  --output-dir reports/kaitactilesim_official_h50_next \
  --presentation-only --detail-navigation --visual-refresh
.pixi/envs/default/bin/python -B scripts/analysis/evaluation_goal_reference_snapshot.py \
  --verify reports/kaitactilesim_official_h50_next
```

品牌信息独立保存在 `presentation_extension.branding`，不写入原生／阶段判定。
逐条 HTML/JSON、曲线和媒体保持归档字节不变；详情品牌由新的 UI 资产显示，旧站点不加载新资产。

### 科研评估工作台视觉版

`staged_h50_studio_20261015` 及其 `r2` 是独立的展示快照，不是重新评估：

- 详情页压缩页首与固定播放工具栏，首屏并排展示模型动作和任务关键曲线；原任务结果与累计阶段仍独立。
- 触觉区集中呈现五指法向/切向载荷及单指阵列/接触力图；页内分区导航可直达任务、触觉、专家和全程图。
- 专家参考默认收起，可单独展开播放，始终保留独立时间线，不与模型强行对齐。
- 总览用紧凑数量摘要代替大统计卡，保留筛选与分组统计，提供卡片/紧凑列表两种浏览方式。
- `r2` 将“同规则、同权重的阶段统计”放在“显示 X / 360 条轨迹”下方、轨迹列表上方，
  仍默认折叠，保留原始统计口径及任务/权重筛选；首版视觉站点原样保留。
- 保留已有任务目标线、白板贴合数据、腕部视角、三类诊断图及缺测说明；当前游标与归档终态值分别标注。

本批仍是 9 任务、360 条既有 H50；USB 继续显示该批当时的严格放回规则和原成绩，
不因当前源码增加两秒保持协议而改判。新原位旋入任务没有被补成模型评估轨迹。

从已核验快照另存视觉版（输出目录须不存在）：

```bash
.pixi/envs/default/bin/python -B scripts/analysis/evaluation_goal_reference_snapshot.py \
  --source-report reports/staged_h50_studio_20261015 \
  --output-dir reports/staged_h50_studio_next \
  --presentation-only --detail-navigation --visual-refresh
.pixi/envs/default/bin/python -B scripts/analysis/evaluation_goal_reference_snapshot.py \
  --verify reports/staged_h50_studio_next
```

`--visual-refresh` 需显式搭配前两个展示开关，记录 `evaluation-studio-v1` 展示版本。
只替换既有白名单中的总览/详情 UI 资产与展示元数据；逐条报告 HTML/JSON、原始曲线、
参考线、贴合数据和视频均逐文件核验为不变，不放宽 replay/data 资产保护。
所有旧网址保留；回退可直接打开 `staged_h50_detail_navigation_20261015` 或更早稳定版。

### 总览展示与内部审计分离

精简总览移除运行模式筛选、模式统计/卡片徽标，以及“数据覆盖与来源”整个展示区
（覆盖数量、快照时间、USB 图来源说明和清单/筛选/验证下载链接）。顶部仅保留任务数和轨迹数；
任务、模型、checkpoint、原记录结果、阶段及 seed 筛选仍可用，每条轨迹的媒体可用性提示仍保留。

`formal` 是 `formal_metrics_valid=true` 且 `diagnostic_only=false` 的来源协议标记，
不是任务成功或独立认证；本批 360 条均为 `diagnostic`。移出界面不改变这些内部标记，
`batch.json`、`selection.json`、`verification.json`、原生/阶段和来源证据仍保存。
这是展示精简，不是删除原始文件或提供访问控制；详情页原有的协议说明仍保留。

只精简既有快照的总览时使用以下命令，不重读原始仿真数据、不重算目标值或贴合指标：

```bash
.pixi/envs/default/bin/python -B scripts/analysis/evaluation_goal_reference_snapshot.py \
  --source-report reports/staged_h50_goal_refs_usb_whiteboard_20261015 \
  --output-dir reports/staged_h50_clean_view_next \
  --presentation-only
.pixi/envs/default/bin/python -B scripts/analysis/evaluation_goal_reference_snapshot.py \
  --verify reports/staged_h50_clean_view_next
```

该模式完整独立复制父快照，只允许更改总览 HTML、`suite.js/css`、新增展示元数据和验证清单；
逐条报告 HTML/JSON、参考线与贴合数据、回放资源均核对为字节不变。
所有旧网址保留，回退只需打开原地址。

### 详情页阶段概览与同任务切换

轨迹详情页头部直接显示归档的任务专属阶段徽标：`✓` 已达到、`—` 未达到、`?` 未记录。
这些是整条轨迹**累计曾达到**的结果，拖动模型时间游标不会改写它们；当前时刻的阶段观测仍在原详细面板中查看。
原任务结果严格读取该条归档的原生 `task_success`（真/假/未知），与 staged 阶段结果分开展示。
本批两条原生成功轨迹的 staged 完整阶段仍未通过；三类白板仅定义拿起阶段，绿色拿起徽标不等于擦净或完整成功。

模型时间条顶部增加“同任务切换”：

- 上一条／下一条，以及按模型与 checkpoint 分组的 seed 下拉选择；当前批次每个任务有 40 条。
- 按任务精确匹配，不将水平、倾斜和横线白板混为同一个任务；可在同任务不同权重之间切换。
- 导航随模型时间条固定，滚动查看图表时也可操作；切换完整加载目标页面，从该条自己的时间起点开始，
  不沿用上一条游标，也不对齐模型与专家时间线。
- 首尾方向禁用；索引缺失、路径或身份校验失败、`file://` 无法读取索引时明确提示不可用，
  当前回放和指标继续可用，并保留“返回任务列表”入口。

轻量导航索引 `trajectory_index.json` 只含 H50 轨迹身份、任务、模型、权重、seed、相对报告路径和原生成功标记，
不包含源文件系统路径或内部 provenance。它只是批次清单的展示投影，不重判任何阶段。

从已有快照生成独立详情导航版：

```bash
.pixi/envs/default/bin/python -B scripts/analysis/evaluation_goal_reference_snapshot.py \
  --source-report reports/staged_h50_clean_view_20261015 \
  --output-dir reports/staged_h50_detail_navigation_next \
  --presentation-only --detail-navigation
.pixi/envs/default/bin/python -B scripts/analysis/evaluation_goal_reference_snapshot.py \
  --verify reports/staged_h50_detail_navigation_next
```

`--detail-navigation` 必须与 `--presentation-only` 一起使用，才额外允许替换 `workbench.js/css` 并生成导航索引；
原 portal-only 模式的回放资产保护不放宽。逐条报告 HTML/JSON、阶段/目标/贴合数据和媒体仍核对为字节不变。
`staged_h50_clean_view_20261015` 及更早各版均原样保留。

### 单项参考线

- 紫色虚线表示机械位置目标或单侧阈值（`≤` / `≥`）；淡色带表示已核实的几何容差或允许范围。
- 灯泡旋拧曲线和读数统一显示角度（°），由原净圈数 ×360 得到，反向转动会抵消，
  不将角度归一化为整个任务完成率。新导出使用 `clockwise_angle_deg`、`angle_removed_deg`，
  原位任务另用 `rotation_from_initial_deg` 表示相对实际初态的净旋入角度。
  新版观察台也可在显示时换算旧 `turn` 曲线及其目标线，不改写归档样本。
- 旋入角度目标为 100°，轴向目标为 7.111 mm；
  几何容差分别为 ±28.8°（原 ±0.08 圈）、±0.3 mm。它们只是机械到座条件的一部分。
  原位任务的相对初态角度需扣除本条实际初始角，不直接套用固定的 100° 终点。
- 旋出角度参考为 131.5°的名义脱扣位置，**不画对称脱扣容差带**。
  121.5°只是合法脱扣事件的最低旋出量校验，不是名义终点。
  轴向 `≤−3 mm` 是完全清离的单侧几何条件，不是放回桌面后的目标。
- RAM 插入深度目标为 **6 mm**，原就位容差 **±0.2 mm**，显示 **5.8–6.2 mm** 容差带；
  该有符号几何量不是越大越好，必须结合完整插入边与孔口匹配观察。
  姿态误差理想值为 **0°**，原就位门槛为 **≤1° / 0.017453 rad**，显示 **0–1°** 允许区。
  两项满足仍不证明真实压底；原生成功另需压底证据及稳定保持，不证明电连接或卡扣闭锁。
- 抽卡 `card_displacement_now_mm`（当前位移）和 `card_displacement_peak_mm`（累计最大位移）
  均标 **≥50 mm** 的模型原生抽出边缘阶段位移参考。
  当前位移是相对本轨迹初态朝机器人方向的有符号位移；累计最大值只说明曾达到，卡片回退后仍会保持峰值。
  达到该位移还须同一物理时刻桌面支撑、平放和悬出比例满足，之后还要夹持、抬牌和持牌检查；
  不把位移越线标成完整成功，也不混用专家完整流程中的 **>80 mm** 条件。
  这也不是独立 staged Stage 1 的 **XY 位移 ≥20 mm + 桌面有效接触持续 0.05 s** 门槛。
- USB 插入：背挡目标 **12 mm**，机械近底范围 **11.9–12.1 mm**；机械就位姿态上限 **3°**，
  允许区 **0–3°**。图中的对准条件插深原来以孔口匹配和 **≤5°** 决定是否记非零，
  未对准记 0 是记录约定；不要把该 5°门槛与机械就位的 3°混同，也不改写原深度。
- USB 拔出：有符号轴向位置 **≤−2 mm** 是完全清离的几何条件之一，
  还需无插座接触并保持 **0.10 s**；不是放回桌面的最终位置。
- 白板：平均 `cleaned_fraction=1` / **100%** 只画“理想参考”，不是独立的成功阈值。
  从同轨迹已归档 `staged_states.jsonl:measurements.native_task_state.maximum_ink_remaining`
  补充“最大单元墨迹余量”曲线，标出真实原生擦净条件 **≤1e−9**。
  三种白板仍分别检查 25/25/8 块墨迹；还须满足拿取/抬升、释放以及水平变体回位等各自条件。
  科学计数法保留微小非零残留，不把显示四舍五入的 0% 当作已经擦净。
- 图的纵轴同时包含真实观测和参考值，即使模型没有推进也能看到离目标的距离。
  原始数值、采样时刻、缺口和游标不改变；扭矩与螺纹载荷不强设“完成终点”。

参考值只在该 trial 绑定的 summary、deployment manifest、source inventory 和已审阅
配置/测量源码 SHA 全部核实后生成；不是直接取当前默认值重判历史数据。
目标元数据保存在各 trial 的 `goal_references.json`；来源不足时状态为 `unavailable`，不补造参考线。
入带或到线不等于真实压底、拧紧、释放、真实脱扣、完成持牌检查或整个任务成功，仍须结合原生/阶段标志。

### 白板接触期间贴合面积比

旧网页缺少字段映射，并非本批没有记录。新版从 SHA 绑定的 staged 状态日志和最终汇总提取：

- `effective_pad_contact_area_ratio`：当前有板面承载接触时的有效贴合面积 / 完整毡底面积；
  非接触时留空，而有接触却不贴合的真实 0 保留。
- `contact_time_weighted_mean_effective_area_ratio`：截至当前的接触期间累计均值，
  公式为 `∫有效贴合面积 dt /（完整毡底面积 × 板面承载接触时长）`。
  直接采用原物理 dt 积分记录，不对低频展示点或视频帧重新取平均。
- 终态摘要另外显示整个 episode 的接触期均值和累计板面承载接触时长，不随当前回放游标变化。

该测量是**带载刚体几何贴合估计**，分母为编译后的完整毡底平面多边形，
当前批次按有限板内重叠、法向偏差 ≤20°、表面间隙/穿透 ±2 mm、实际承载 >1e−8 N 筛选；
它不是真实形变接触面积，不是墨迹累计覆盖率，也不等于有效擦除。
指标没有预设“达到某百分比就成功”的线。6 条零板面接触轨迹的均值为 `null / no_contact`，
显示“无接触，均值未定义”，不显示伪造的 0%；来源冲突则明确不可用，不回填终态值。
原生、staged 和完整任务成功三层结果保持原样，白板 staged 仍只定义拾取而没有完整清洁成功布尔式。

此入口只为**现有报告的独立副本**增加展示标注及已有记录的曲线投影，不启动推理或重算物理指标，
与[通用离线导入](../workflows/unified_evaluation.md#重新导出及完整核验的前提)中需要两份原工程来源的
`evaluate_unified.py export` 不是同一个操作。
新网页、媒体与图像都是独立文件副本，不依赖指向旧站的软/硬链接；
其来源复核不应被误读为已重新验证退役原工程或重跑物理。

从 unified 根目录执行；再次生成时选择尚不存在的新目录：

```bash
.pixi/envs/default/bin/python -B scripts/analysis/evaluation_goal_reference_snapshot.py \
  --source-report reports/staged_h50_goal_refs_ram_poker_20261015 \
  --output-dir reports/staged_h50_goal_refs_next \
  --verified-usb-source-copy /cpfs_infra/user/chenxianchi/code/kaitactilesim_unified
.pixi/envs/default/bin/python -B scripts/analysis/evaluation_goal_reference_snapshot.py \
  --verify reports/staged_h50_goal_refs_next
```

本批 USB 插入归档没有源码 root 字段，因此命令**显式**提供 USB 源码字节副本：每个必需配置、
监视器及记录模块文件仍须与该 trial 冻结 manifest 的 SHA/大小匹配；不是使用当前默认参数。
不提供该选项时缺源则不显示 USB 参考线。其它任务仍从各自 SHA 绑定的冻结源码 root 读取。
白板补充记录另保存在 `whiteboard_contact_area.json`，来源、时钟原点、分母和几何参数均可追溯；
新增序列带 `presentation_extension` 标记，完整保留已有曲线的数值、时刻和缺口。

旧参考版可以作为新导出的父报告；每条目标线仍重新核验来源，证据缺失时删除新副本中的旧线，
而不是把旧标注当作可信默认值。整个导出不会覆盖任何父报告。

`--verify` 检查新版文件清单/哈希及原站完整文件清单/哈希；原站仍须可访问。
回退时打开原网址即可，不需覆盖文件或撤销代码。

## 此次导入与回退

默认 canonical staged 批次有 340 条；显式额外导入水平线擦拭 step20000 的 20 条完整记录，合计 360 条。
额外记录所属运行组标为 failed，这一批处理警告保留，不抹去每条已完成日志，也不解释成每条任务失败。
历史扑克中断/重试目录不重复纳入。`selection.json` 记录选择、排除和来源，
`source_manifest.json` / `artifact_manifest.json` / `verification.json` 可复核导出证据。

原站文件及服务仍可访问时，继续访问原站 8876 即可回退；不需要 checkout 或覆盖目录。
导航锚点本身不复制旧网页或恢复已退役目录；若原路径已缺失，须从实际保存位置恢复访问。
新站 `_preserved_h50` 是原站的导航和校验锚点，不是原报告副本。
实际数据盘在本次整合前已清理部分旧输入：新锚点会严格核对仍可访问的历史来源，
对仅路径迁移且 SHA/大小一致的别名留痕重绑定；已缺失的历史输入明确记录为不可验证。
旧站自身的原 artifact manifest 及其中全部生成物另行逐文件校验保护，
不会把“旧网页保留完好”误写成“旧原始数据全部还在”。此模式使用 `_preserved_h50_site_bound`。
本工程包含完整仿真源码；整合站是已有记录的离线导入，不是新模型 rollout。未来启动新 rollout 时，
必须重新 prepare/verify 生成绑定新源码的部署 manifest，不能复用旧部署的源码 SHA。

## 2026-10-15 整合初版归档验收

- 完整输出：`reports/staged_h50_20261015`，本地入口 `http://127.0.0.1:8877/`。
- 9 个任务各 40 条、18 个规则/权重/批次分组各 20 条；共 360 条 H50/K50，没有 H30 或重试重复。
- 360 条逐条核对原 summary/sidecar：原生字段、完整阶段结果、首次时间和诊断标志一致；白板 120 条完整成功保留 null。
- Fn、Ft、专家回放和三类全程图均覆盖 360 条；USB 插入 40 条的三类图为模型自身日志补绘。
- 319 项 Python 测试和 117 项子测试通过；前端力学/游标辅助测试通过。
- 两组浏览器验收均通过 9 任务；新版另核对 18 分组、7 次实际阶段跳转、专家独立时钟、腕部切换与 390px 布局。
- 整合初版验收时，`verify` 全量核验通过，856 个原来源文件与合并清单一致；这是历史验收，不保证这些路径今天仍可访问。
- 整合初版只做离线导入，没有新建模型 rollout、训练或 GPU 仿真。

之后新增的[灯泡原位旋入](../tasks/bulb_tasks.md#bulb-screw-inplace)已完成 CPU 专家物理验证，
三相机 Raw 与独立预览位于新目录，未混入上述 360 条模型评估。其预览可直接打开
`http://127.0.0.1:8877/bulb_screw_inplace_demo_20261015/review.mp4`。
最近一次只读复核中，原整合站 4,334 个归档文件的大小和 SHA256 全部一致。

### 统一工作台：跨任务、模型与 checkpoint 对比

平台首页使用原 KaiTactileSim 工作台的任务标签、模型/checkpoint 筛选及卡片/紧凑列表，
在同一页展示 π0.5、EgoTouch、CPM 和 FTP-1，不再以三个独立实验入口作为首页。
成功率总表同时展示原生成功、完整阶段、各阶段到达率及已归档的白板清洁度/贴合面积指标；
任务/模型/checkpoint/协议/批次筛选联动总表，结果/阶段/搜索只筛轨迹列表，不改变成功率分母。
页面只保留一张“595 条成功率与阶段汇总”，默认按“任务 × 模型 × checkpoint”分别展示
47 个原协议分组。模型与 checkpoint 独立成列，每行的原生成功率、完整阶段成功率、各阶段
到达率和量化指标只使用本行分组的有效样本，不混合其他模型或 checkpoint；协议或批次不同
也保留分行。顶部仅展示归档/有效/隔离样本覆盖量，不把所有策略的成功率混成一个总分。
同表仍可显式切换“按任务合计（跨模型，仅描述性）”查看 10 个任务合计。模型选项全局常驻 CPM、FTP-1、
π0.5、EgoTouch，不因当前任务暂无结果而隐藏；空组合明确显示暂无评估记录，不写成 0%。

统一汇总只读取已发布快照，不重新扫描任务目录或重跑模型：H50 450 条（430 计分、20 隔离）
与 EgoTouch 120 条、本轮触觉 25 条，共 595 条成员、575 条有效。原 360 条和旧 550 条状态表
已包含在这些记录中，不能再次加入。原 360 条汇总并入同一张表，不再另列历史统计表；
原始汇总仍留在归档链接及 catalog 中供审计。可选的跨模型任务合计沿用各记录原判据，
仅为描述性计数，不是同协议公平策略排名，也不是任务等权平均。USB 拔出两套不同完成定义分别
显示，白板均值按已知有效 episode 数量加权；缺失与未验证阶段不补零。
当前 CPM 的 5 条和 FTP-1 的 20 条均只覆盖扑克任务，其他任务暂无这两类已发布结果。

```bash
.pixi/envs/default/bin/python -B scripts/analysis/export_unified_evaluation_workbench.py \
  --h50-snapshot reports/h50_checkpoint_experiment_20261019 \
  --tactile-snapshot reports/poker_tactile_benchmark_20261019_n5 \
  --output reports/kaitactilesim_unified_workbench_20261020_r3 --publish-index

.pixi/envs/default/bin/python -B scripts/analysis/export_unified_evaluation_workbench.py \
  --verify reports/kaitactilesim_unified_workbench_20261020_r3
```

后续先导出各模型的不可变结果快照，再用上述入口生成新的统一工作台版本。不要再使用旧触觉
实验导出器的 `--publish-index` 将首页换回实验选择页。所有旧页面、明细和视频保持原样。


#### 完整轨迹详情与交付验收

统一首页的成员应全部进入完整分析页，不再混用简版原生状态页和触觉实验页。
`export_rich_evaluation_workbench.py` 在新快照中复用历史 rich 数据，并仅对发布 catalog 明确绑定的
新增轨迹读取原生记录；不会扫描其他实验、读取模型权重、启动仿真或重判成功。
每页显示模型、checkpoint、seed、真实 H/K，可返回统一总览或在同任务全部模型之间切换。
有记录时展示视频、任务曲线、五指 Fn/|Ft| 与 taxel 空间分布，以及手指关节和腕部诊断图。
灯泡旋拧量使用角度（°）；原生时钟未核验的视频保持独立播放，不伪造曲线同步。

缺少已归档参考图但具有明确 state29、Fn/Ft 及原生时钟合同的轨迹，可离线生成三类
**仅模型实际记录**诊断图，不补造专家、动作目标或物体状态。EgoTouch 缺少任务阶段记录时
保留未知；CPM 来源校验失效时显示“已记录，但来源校验未通过／暂不计分”，不能把原始阶段
达成回填到成绩。隔离轨迹仍可查看诊断，但不进入任何成绩分母，审计页同样提示当前隔离状态。

```bash
# 先建立新的不可变详情快照；完成浏览器验收后才切换入口。
.pixi/envs/default/bin/python -B scripts/analysis/export_rich_evaluation_workbench.py \
  --baseline reports/kaitactilesim_unified_workbench_20261020_r3 \
  --evaluation-root /cpfs_infra/user/chenxianchi/kaitactilesim_evaluations \
  --output reports/kaitactilesim_unified_workbench_20261020_r4 --workers 4

node scripts/analysis/check_rich_evaluation_workbench_browser.cjs \
  http://127.0.0.1:8877/kaitactilesim_unified_workbench_20261020_r4/index.html \
  reports/qa_kaitactilesim_unified_workbench_20261020_r4

.pixi/envs/default/bin/python -B scripts/analysis/export_rich_evaluation_workbench.py \
  --output reports/kaitactilesim_unified_workbench_20261020_r4 --publish-only
```

验收同时检查全部 595 条的详情/视频/数据身份、原 47 个分组的成员与分母完全不变，以及每个
分组和所有隔离/CPM 例外页面的实际浏览器显示；`--all-dom` 可额外逐条渲染所有详情。
r4 详情整理本身没有启动 `bulb-screw-inplace` 新评估；后续独立批次见下文。

#### 任务目标参考与灯泡角度（r5）

`export_reference_evaluation_workbench.py` 从上一版已发布快照生成新的参考值扩展版。
原 360 条的目标参考原样保留；新增 π0.5 90 条、CPM 5 条及 FTP-1 20 条，按各自归档
summary、manifest 和已审阅源码的哈希绑定恢复目标值。页面同时展示目标虚线、适用的容差带
及参考来源；EgoTouch 120 条未记录物体任务进度，明确标为不适用，不补造曲线或目标。
参考值不是专家轨迹，也不替代完整成功判据；不同协议的阈值不互相覆盖。

100 条灯泡轨迹统一展示“当前角度 / 终态角度 / 目标参考角度”。当前值跟随原生曲线游标，
终态值仅从归档终态读取，不用曲线末采样点替代；本次补齐旧 80 条的终态角度。
角度是净旋转量（净圈数 × 360），反转会抵消，不是累计转动路程。原位拧入的相对初始角度
不能直接套用固定 100° 目标；该任务不包含在这版 595 条基线中。

```bash
# 只做离线参考标注，不读取模型权重、不运行仿真、不重判成绩。
.pixi/envs/default/bin/python -B scripts/analysis/export_reference_evaluation_workbench.py \
  --baseline reports/kaitactilesim_unified_workbench_20261020_r4 \
  --evaluation-root /cpfs_infra/user/chenxianchi/kaitactilesim_evaluations \
  --output reports/kaitactilesim_unified_workbench_20261020_r5 --workers 4

node scripts/analysis/check_reference_evaluation_workbench_browser.cjs \
  http://127.0.0.1:8877/kaitactilesim_unified_workbench_20261020_r5/index.html \
  reports/qa_kaitactilesim_unified_workbench_20261020_r5/references

.pixi/envs/default/bin/python -B scripts/analysis/export_reference_evaluation_workbench.py \
  --output reports/kaitactilesim_unified_workbench_20261020_r5 --publish-only
```

发布前还需运行上一节完整详情浏览器验收。参考扩展验收检查全部 595 条原始采样、视频与
诊断文件未改变，475 条轨迹的目标参考和 100 条灯泡终态角度均有来源；按任务 × 模型 ×
checkpoint 的 47 个分组成绩与分母完全不变。CPM 失效阶段证据仍不计分，20 条隔离记录仍
不进入分母；目标参考可用不代表阶段证据重新获得计分资格。


## 2026-10-21 CPM 独立权重增量发布

10 条新 CPM 记录已追加至 `reports/kaitactilesim_unified_workbench_20261021_cpm_n10_r2`；
该版为 605 条／48 组，旧 595 条身份和计分资格保留。权重 SHA、失败 attempt、独立 Raw 与
浏览器验收回执见[触觉运行档案](pipeline_batches_202609_202610.md#当前-cpm-权重的-10-条独立复评)。
