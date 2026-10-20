"""Local-only simulator evaluation hub, using existing portable report widgets."""
from html import escape
import json
from pathlib import Path

from evaluation_v1.dashboard import _embedded_json, _document
from evaluation_v1.render import render_report, render_batch


def _audit_sections(values):
    return ''.join('<section><h2>' + escape(title) + '</h2><pre>' +
                   escape(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)) +
                   '</pre></section>' for title, value in values)


def render_suite_report(report):
    extra = _audit_sections([(title, report.get(key)) for title, key in (
        ('V2 · 预测 Horizon 依据', 'prediction_horizon_evidence'),
        ('V2 · 证据完整性', 'evidence_v2'), ('V2 · 条件与任务规则复核', 'task_review'),
        ('V2 · 接触质量', 'contact_quality'), ('V2 · 专家条件分布', 'reference_comparison'),
        ('V2 · 参考分布统计', 'reference_distributions'))])
    return render_report(report).replace('</main>', extra + '</main>')


def render_suite_batch(batch):
    extra = _audit_sections([(title, batch.get(key)) for title, key in (
        ('V2 · Horizon 纳入与排除清单', 'horizon_selection'),
        ('V2 · 样本选择范围', 'input_selection'), ('V2 · 实际验证覆盖', 'validation'),
        ('V2 · 专家参照状态', 'reference_comparison'))])
    return render_batch(batch).replace('</main>', extra + '</main>')


def read_portal_assets():
    root = Path(__file__).parent / "web"
    return {name: (root / name).read_bytes() for name in ("hub.js", "hub.css")}


def render_portal(batch):
    # Embed only compact index metadata. Individual traces are fetched lazily.
    index = {key: batch[key] for key in ("suite_schema_version", "trial_count", "by_task", "trials", "validation")}
    index["reference_status"] = batch["reference_comparison"].get("status", "unknown")
    selection = batch.get("horizon_selection") or {}
    index["horizon_selection"] = {key: selection.get(key) for key in ("required_prediction_horizon", "input_count", "included_count")}
    index["horizon_selection"]["excluded_count"] = len(selection.get("excluded", []))
    index["task_catalog"] = [{key: entry.get(key) for key in ("task", "contract_id", "required_evidence")}
                             for entry in batch["task_catalog"]]
    return '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>KAI · 仿真评估工作台</title>
<link rel="stylesheet" href="assets/hub.css"><link rel="stylesheet" href="assets/dashboard.css">
<script src="assets/dashboard.js" defer></script><script src="assets/hub.js" defer></script></head>
<body><main id="evaluation-hub"><header class="hub-header"><div><div class="hub-eyebrow">KAI · SIMULATION EVALUATION</div>
<h1>仿真评估工作台</h1><p>任务结果、接触质量与证据，把每一次试验看清楚。</p></div>
<nav><a href="integration.html">使用与接入说明</a><a href="batch_audit.html">完整审计</a><a href="validation.json">覆盖清单 JSON</a></nav></header>
<div id="hub-content"><p>正在载入评估索引…</p></div>
<noscript><p>筛选与并排查看需要 JavaScript。<a href="batch_audit.html">查看静态报告列表</a></p></noscript></main>
<script type="application/json" id="hub-data">''' + _embedded_json(index) + '''</script></body></html>'''


def render_v2_integration():
    command = '''# 默认 H50-only；只读评估已完成轨迹，输出目录必须是新目录
.pixi/envs/default/bin/python -B scripts/analysis/evaluate_suite.py \\
  --batch-dir /path/to/completed/checkpoint \\
  --output-dir /path/to/new/local/site

# 可挂在已有评估启动命令成功退出之后；不会改写仿真脚本
# 或读取 status.json 中显式 complete 且 returncode=0 的组
.pixi/envs/default/bin/python -B scripts/analysis/evaluate_suite.py \\
  --status-json /path/to/status.json --output-dir /path/to/new/snapshot

.pixi/envs/default/bin/python -B scripts/analysis/serve_evaluation.py \\
  --directory /path/to/new/local/site --port 8877'''
    body = '<header><div><div class="eyebrow">LOCAL SIMULATION WORKSPACE · V2</div><h1>使用与接入说明</h1></div><a href="index.html">← 工作台</a></header>'
    body += '<section><h2>从评估到网站</h2><p>本地工作台读取离线产物，不启动模型、不占用仿真 GPU，也不会修改仿真源码、运行中的状态文件或原始轨迹。批次结束后调用导出入口，即可得到本地网站；不需要把生成的 HTML 手工改成网页。</p>'
    body += f'<pre><code>{escape(command)}</code></pre></section>'
    body += '<section><h2>当前统计口径：H50-only</h2><p>默认只计算归档元数据明确声明预测 horizon=50 的模型轨迹。H30、未知、冲突或非法声明不进入成功率、触觉汇总及任务覆盖分母；不是只在网页上隐藏。<code>execute_steps=50</code>、<code>control_hz=30</code> 或目录名均不能代替 prediction horizon 证据。任务、checkpoint 和控制配置仍分别分组。</p><p><a href="horizon_selection.json">查看完整纳入 / 排除依据</a>。历史混合版本仅保留为旧归档，不代表当前 H50 统计。脚本专家示范没有模型预测 horizon，不套用模型筛选，但仍需独立满足参考集兼容性检查。</p></section>'
    body += '<section><h2>先看证据，再看结论</h2><ul><li>“记录结果”来自原任务监视器；“规则复核”是对可用证据的重新计算，两者不相互覆盖。未知不是失败，也不是安全通过。</li><li>任务覆盖清单只统计实际提供的轨迹；支持 12 类契约不等于已经完成 12 类任务的闭环实验。</li><li>接触质量需要目标接触对、独立期望接触阶段及显式阈值等证据。旧档案缺少这些字段时，不自动推断。</li><li>专家比较按事件/阶段条件下的 episode 标量分布进行，不对齐专家与模型的时间轴。采样、单位、几何或参考成员证据不足时拒绝比较。</li><li>并排查看的时间游标彼此独立。统一量程只统一力的视觉映射，不证明不同任务、传感器或控制条件可以直接排名。</li></ul></section>'
    body += '<section><h2>新增证据与参考集</h2><p><code>evaluation_v2.evidence</code> 提供严格版本化的 metadata/JSONL 记录器；新记录使用独占创建，不覆盖旧档案。可通过独立采集程序使用该接口；本次不修改 <code>kaitactilesim</code> 源码，也不宣称已给现有运行进程插入采集钩子。</p><p>提供 <code>--reference-report</code>、<code>--reference-manifest</code> 和必要的 <code>--reference-compatibility</code> 才会启用专家基线。只挑选成功示例的参考集必须明确选择偏差；不能充当无偏专家成功率。</p></section>'
    reference_command = '''.pixi/envs/default/bin/python -B scripts/analysis/export_expert_reference.py \\
  --replay-json /archive/task_demo/_metadata/task/replay.json \\
  --output-dir /path/to/new/reference \\
  --reference-id curated-scripted-demo --declared-by curator \\
  --selection-rule 'Explicitly selected demonstrations; success-selection bias' \\
  --selection-evidence 'Describe the audited source and selection process' \\
  --declare-expert-reference

# 将导出的每个 report.json 作为 --reference-report（可重复），
# reference_manifest.json 作为 --reference-manifest 交给 evaluate_suite.py。
# 兼容性另以经过审计的 --reference-compatibility JSON 声明。
# 不应为了让图表出数而把未核实字段改成 verified。'''
    body += '<section><h2>直接导入归档专家 HDF</h2><p>已支持明确版本的 task_demo replay 元数据及其绑定的单 episode HDF5；会核验原文件 SHA，并保留原生采样、几何、缺失值和记录结果。没有触觉的任务不会补造曲线，也不会把同一 HDF 的重复导出当多条示范。</p><pre><code>' + escape(reference_command) + '</code></pre><p>导入成功不代表可比较：专家 100/500 Hz、模型约 30 Hz 的峰值和变化量不能直接同口径排名；单任务只有一条展示示范也不足以建立参考分布。本模块不会自动重采样、时间规整或补造阶段真值。</p></section>'
    example = '''# 外部 runner 的显式接入示意；不要复制到只读仿真源码
# 在 scripts/analysis 在 PYTHONPATH 的独立程序中运行。
from evaluation_v2.evidence import EvidenceRecorder, make_metadata

meta = make_metadata(task_name, clock_id="simulation_data_time",
                     source_kind="simulation_oracle",
                     nominal_interval_s=expected_interval_at_this_hook)
meta["task_fields"] = {
    key: {"kind": "simulation_oracle", "source": "sim.full_state() after adapter.step",
          "unit": "mixed generalized coordinates; model-dependent"}
    for key in ("qpos", "qvel")}
with EvidenceRecorder(meta, trial_dir=NEW_CAPTURE_DIRECTORY) as recorder:
    for _ in physics_ticks:  # 目标控制量仍由已有 runner 设置
        adapter.step()  # 现有无参物理步进；此示意不添加 mj_forward
        qpos, qvel = sim.full_state()
        recorder.record({"time_s": float(sim.data.time), "valid": True,
                         "task_values": {"qpos": qpos.tolist(), "qvel": qvel.tolist()}})
    # 仅在外部 runner 确认采集完整终止后：
    recorder.close(complete=True)
'''
    body += '<section><h2>采证接入的最小边界</h2><p>下面是调用示意，不是本次已经运行的新仿真实验。原始 qpos/qvel 不等于 12 类任务所需的语义字段，单靠它们不会得到独立成功结论。物体位姿缓存、积分接触力、接触点相对速度必须由任务适配器说明采样相位；不能把 before-forward 的力与 after-forward 的位姿伪称同一时刻，也不能覆盖原有 observer。</p><pre><code>' + escape(example) + '</code></pre><p>需要规则复核时，从 <code>get_contract(task)</code> 取得明确的 <code>contract_id</code> 与 <code>contract_sha256</code>，经确认后置于 metadata 的 <code>task_config</code>；记录所需的独立 primitive 字段与来源。声明不是外部认证，不能把原监视器的 success/phase 布尔值重新命名后当真值。</p></section>'
    body += '<section><h2>边界与后续集成</h2><p>这是本地基础版，不包含账号、云部署、任意文件上传或在线 GPU 作业调度。正式共享前需审核路径和媒体权限，并配置访问控制。</p><p><a href="component_integration.html">独立可视化组件接口</a> · <a href="task_catalog.json">任务契约</a> · <a href="reference_comparison.json">专家参照诊断</a> · <a href="validation.json">实际验证覆盖</a></p></section>'
    return _document("本地仿真评估工作台 · 使用说明", body)
