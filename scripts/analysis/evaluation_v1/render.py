"""Render evaluation dictionaries as self-contained, script-free Chinese HTML.

Only explicit ``pass`` / ``fail`` / ``unknown`` outcome statuses are interpreted.
All measurements are reported separately; this module does not score or infer
task quality. Paths and JSON are escaped text, never links or executable assets.
"""

from __future__ import annotations

from collections.abc import Mapping
from html import escape
import json
import math
from urllib.parse import unquote, urlsplit


_LABELS = {
    "status": "状态", "task": "任务", "trial_id": "试验编号",
    "source": "证据来源", "evidence": "判定证据", "evidence_sources": "证据来源",
    "reason": "判定原因", "reasons": "判定原因", "reason_codes": "原因代码",
    "terminal_state": "终态", "final_state": "终态", "success": "记录的成功标志",
    "reported_status": "原始状态", "reported_success": "原始成功标志",
    "duration_s": "时长（s）", "scope_duration_s": "区间时长（s）",
    "observed_duration_s": "有效观测时长（s）", "sample_count": "样本数",
    "coverage": "有效观测覆盖", "coverage_fraction": "有效观测覆盖比例（0–1）",
    "unknown_duration_s": "未知时长（s）", "mean_n": "均值（N）",
    "p50_n": "P50（N）", "p95_n": "P95（N）", "std_n": "标准差（N）",
    "sampled_peak_n": "原生采样峰值（N）", "sampled_peak_time_s": "采样峰值时刻（s）",
    "sampled_peak": "原生采样峰值", "analysis_bin_s": "分析时间箱（s）",
    "change_lag_s": "变化滞后窗口（s）", "complete_bin_count": "完整时间箱数",
    "pair_count": "有效配对数", "mean_absolute_change_n": "绝对变化均值（N）",
    "p95_absolute_change_n": "绝对变化 P95（N）", "rms_change_n": "变化 RMS（N）",
    "max_absolute_change_n": "绝对变化最大值（N）", "semantics": "指标定义",
    "observed_contact_duration_s": "观测接触时长（s）",
    "contact_episode_count": "接触段数", "candidate_episode_count": "候选接触段数",
    "short_contact_episode_count": "短接触段数",
    "observed_internal_break_count": "观测到的内部断触次数",
    "observed_internal_break_duration_s": "观测到的内部断触时长（s）",
    "longest_internal_break_s": "最长内部断触（s）", "episodes": "接触段明细",
    "start_s": "起点（s）", "end_s": "终点（s）",
    "left_censored": "左边界截断", "right_censored": "右边界截断",
    "distribution_change": "分布变化", "center_of_pressure": "法向力加权中心",
    "mean_total_variation": "总变差距离均值", "p95_total_variation": "总变差距离 P95",
    "mean_local_mm": "局部中心均值（mm）", "mean_displacement_mm": "位移均值（mm）",
    "p95_displacement_mm": "位移 P95（mm）", "net_magnitude": "净切向合力模长",
    "mean_fraction": "平均比例（0–1）", "p95_fraction": "比例 P95（0–1）",
    "channels": "通道", "overall": "完整轨迹", "protocol_sha256": "协议 SHA-256",
    "protocol": "评估协议", "provenance": "数据来源与版本", "warnings": "限制与提示",
    "trial_count": "全部试验数", "outcome_counts": "各判定数量",
    "confirmed_success_fraction": "确认成功数 / 全部试验数",
    "resolved_success_rate": "确认成功数 / 已明确判定试验数",
    "unknown_count": "无法判定数量", "metrics": "逐试验标量汇总",
    "macro_confirmed_success_fraction": "按任务等权的确认成功比例",
    "pass": "通过", "fail": "未通过", "unknown": "未知",
    "report_path": "报告路径", "error": "错误信息",
    "recorded_success": "任务监视器记录的成功标志", "termination_reason": "终止原因",
    "formal_metrics_valid": "原评估声明正式指标有效", "diagnostic_only": "仅诊断运行",
    "scope": "判定范围", "enforcement": "约束执行模式", "checks": "已记录检查",
    "name": "名称", "value": "观测值", "threshold": "阈值", "rule": "判定规则",
    "recorded_guard_events": "已记录保护事件", "limitations": "解释限制",
    "simulation_duration_s": "仿真经过时间（s）", "wall_duration_s": "实际耗时（s）",
    "recorded_time_budget_s": "记录的仿真时间预算（s）", "plugin": "任务适配器",
    "predicates": "任务条件", "partial_state": "记录的任务终态",
    "implementation": "评估实现版本", "files": "文件与摘要", "sha256": "SHA-256",
    "by_cohort": "相同配置的试验组", "configuration": "配置", "cohort": "试验配置",
    "possible_success_fraction": "将未知视为成功时的比例上界",
    "wilson95_interval": "Wilson 95% 区间", "interval_assumption": "区间成立假设",
    "interpretation": "解释", "aggregation_semantics": "汇总定义",
    "available_episode_count": "该指标可用试验数", "total_episode_count": "全部试验数",
    "median": "中位数", "p25": "P25", "p75": "P75", "min": "最小值", "max": "最大值",
}
_STATUS = {
    "pass": "通过（pass）", "fail": "未通过（fail）", "unknown": "未知／无法判定（unknown）",
    "available": "有有效观测", "ok": "有效", "partial": "部分可用",
    "not_reached": "未到达此阶段", "missing": "缺失／无观测",
    "observed": "已观察到此条件", "not_observed": "记录中未观察到此条件",
    "unavailable": "不可用／缺失", "no_valid_samples": "无有效样本",
    "insufficient_data": "数据不足", "insufficient_samples": "样本不足",
    "not_applicable": "不适用", "unsupported": "尚不支持",
}
_CSS = """
:root{font-family:system-ui,-apple-system,'Noto Sans CJK SC',sans-serif;color:#17212d;background:#f3f5f7}
*{box-sizing:border-box}body{margin:0}main{max-width:1180px;margin:auto;padding:28px 20px 64px}
h1{font-size:28px;margin:0 0 10px}h2{font-size:20px;margin:0 0 16px}h3{font-size:17px;margin:22px 0 12px}
h4{font-size:15px;margin:18px 0 10px}p{line-height:1.65;margin:8px 0}.muted{color:#586675;font-size:14px}
section,.card{background:white;border:1px solid #dae1e7;border-radius:10px;padding:20px;margin:18px 0}
.outcome{border-left:6px solid #657786}.outcome.pass{border-left-color:#14734c}.outcome.fail{border-left-color:#b93937}
.verdict{display:inline-block;font-size:23px;font-weight:700;margin:0 0 12px}.pass .verdict{color:#14734c}.fail .verdict{color:#a62c2a}
.columns{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:18px}.columns section{margin:0}
.table-wrap{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:14px;line-height:1.55}
th,td{text-align:left;vertical-align:top;padding:10px;border-bottom:1px solid #e3e8ed;overflow-wrap:anywhere}
thead th{background:#eef3f7}tbody th{font-weight:600;min-width:110px}td table{font-size:13px}td table th,td table td{padding:5px}
.empty{color:#697581}code,pre{font-family:ui-monospace,monospace}code{font-size:12px;color:#566472}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7f9;padding:16px;line-height:1.5;font-size:12px}
details{margin:12px 0}summary{cursor:pointer;font-weight:600;line-height:1.7}.values{margin:0;padding-left:20px}.values li{margin:5px 0}
.scope{border-top:2px solid #dce4eb;padding-top:8px;margin-top:25px}.notice{padding:12px;background:#fff7e9;border-left:3px solid #ba8129}
@media(max-width:760px){main{padding:18px 10px}.columns{grid-template-columns:1fr}section,.card{padding:14px}th,td{padding:7px}}
@media print{body{background:white}main{max-width:none;padding:0}section,.card{break-inside:avoid}details>summary{display:block}}
"""


def _mapping(value: object) -> Mapping:
    return value if isinstance(value, Mapping) else {}


def _label(key: object) -> str:
    text = str(key)
    translated = _LABELS.get(text)
    return escape(translated) if translated else escape(text)


def _value(value: object, *, key: str = "") -> str:
    """Render every supplied field, preserving null separately from zero/false."""
    if value is None:
        return '<span class="empty">缺失／未知（null）</span>'
    if isinstance(value, bool):
        return "是（true）" if value else "否（false）"
    if isinstance(value, float):
        if not math.isfinite(value):
            return '<span class="empty">缺失／非有限值</span>'
        return format(value, ".6g")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, Mapping):
        if not value:
            return '<span class="empty">未提供记录</span>'
        rows = "".join(
            f"<tr><th scope=\"row\">{_label(k)}</th><td>{_value(v, key=str(k))}</td></tr>"
            for k, v in value.items()
        )
        return f'<div class="table-wrap"><table><tbody>{rows}</tbody></table></div>'
    if isinstance(value, (list, tuple)):
        if not value:
            return '<span class="empty">无记录（空列表）</span>'
        return '<ul class="values">' + "".join(f"<li>{_value(v)}</li>" for v in value) + "</ul>"
    text = str(value)
    if key == "status":
        text = _STATUS.get(text, text)
    return escape(text) if text else '<span class="empty">未提供（空文本）</span>'


def _outcome_status(outcome: object) -> str:
    value = _mapping(outcome).get("status")
    return value if value in ("pass", "fail", "unknown") else "unknown"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f'<th scope="col">{escape(h)}</th>' for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _section(title: str, value: object, *, identity: str = "") -> str:
    attr = f' id="{escape(identity, quote=True)}"' if identity else ""
    return f"<section{attr}><h2>{escape(title)}</h2>{_value(value)}</section>"


def _json_details(report: dict) -> str:
    # JSON is inert text, not a script element; HTML escaping also prevents
    # closing pre/details tags in paths, messages or task-specific metadata.
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    return '<details><summary>完整详细 JSON（保留原始字段与精度）</summary><pre>' + escape(serialized) + '</pre></details>'


def _document(title: str, body: str) -> str:
    return ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>{escape(title)}</title><style>{_CSS}</style></head>'
            f'<body><main>{body}</main></body></html>')


def _report_link(path: object) -> str:
    """Allow only local relative HTML reports, never source-file navigation."""
    if not isinstance(path, str):
        return _value(path)
    decoded = unquote(path)
    try:
        parts = urlsplit(decoded)
    except ValueError:
        return _value(path)
    unsafe = (parts.scheme or parts.netloc or parts.query or parts.fragment
              or decoded.startswith("/") or "\\" in decoded
              or ".." in decoded.split("/") or "%" in decoded
              or any(ord(char) < 32 for char in decoded)
              or not decoded.lower().endswith(".html"))
    if unsafe:
        return _value(path)
    return f'<a href="{escape(path, quote=True)}">{escape(path)}</a>'


def _tactile(tactile: object) -> str:
    data = _mapping(tactile)
    scopes = _mapping(data.get("scopes"))
    intro = ('<section id="tactile"><h2>触觉诊断</h2>'
             '<p class="muted">力幅值与时序变化分开展示；标准差描述幅值分布，不能单独解释为动态稳定性。'
             '原生采样峰值仅指记录样本中的最大值，受采样频率限制。比例按 0–1 显示。'
             '不同阶段分别解释，不生成统一分数或专家逐秒误差。</p>')
    extras = {k: v for k, v in data.items() if k != "scopes"}
    if extras:
        intro += _value(extras)
    if not scopes:
        return intro + '<p class="empty">触觉区间数据缺失／尚未提供。</p></section>'
    for name, raw_scope in scopes.items():
        scope = _mapping(raw_scope)
        intro += f'<article class="scope"><h3>{_label(name)}</h3>'
        scope_meta = {k: v for k, v in scope.items() if k not in ("channels", "load_shares")}
        intro += _value(scope_meta)
        channels = _mapping(scope.get("channels"))
        if not channels:
            message = "未到达此阶段；不将缺失指标显示为零。" if scope.get("status") == "not_reached" else "无通道观测；指标缺失。"
            intro += f'<p class="empty">{message}</p></article>'
            continue
        intensity_rows, change_rows, other_rows = [], [], []
        for channel_name, raw_channel in channels.items():
            channel = _mapping(raw_channel)
            intensity = _mapping(channel.get("intensity"))
            coverage = {k: v for k, v in intensity.items() if k not in ("all_observed", "contact", "sampled_peak_n", "sampled_peak_time_s")}
            peak = {"sampled_peak_n": intensity.get("sampled_peak_n"), "sampled_peak_time_s": intensity.get("sampled_peak_time_s")}
            intensity_rows.append([
                escape(str(channel_name)), _value(channel.get("status"), key="status"),
                _value(intensity.get("all_observed")), _value(intensity.get("contact")),
                _value(peak), _value(coverage or None),
            ])
            change_rows.append([escape(str(channel_name)), _value(channel.get("change"))])
            other_rows.append([escape(str(channel_name)), _value(channel.get("continuity")),
                               _value(channel.get("spatial")), _value(channel.get("tangent"))])
            other = {k: v for k, v in channel.items() if k not in ("status", "intensity", "change", "continuity", "spatial", "tangent")}
            if other:
                other_rows[-1].append(_value(other))
        intro += '<h4>力幅值：全部有效观测与接触条件统计</h4>'
        intro += _table(["通道", "状态", "全部有效观测（包含零力）", "接触条件统计", "原生采样峰值", "覆盖与统计区间"], intensity_rows)
        intro += '<h4>时序变化：固定时间箱与滞后窗口</h4>'
        intro += _table(["通道", "变化指标及有效配对"], change_rows)
        intro += '<h4>接触连续性、空间变化与切向力</h4>'
        extra_column = any(len(row) > 4 for row in other_rows)
        if extra_column:
            other_rows = [row + [_value(None)] if len(row) == 4 else row for row in other_rows]
        intro += _table(["通道", "接触连续性", "空间诊断", "切向力"] + (["其他字段"] if extra_column else []), other_rows)
        intro += '<h4>通道载荷分配</h4>' + _value(scope.get("load_shares")) + '</article>'
    return intro + '</section>'


def render_report(report: dict) -> str:
    """Return an offline report; no files are read or written and no JS is used."""
    trial = _mapping(report.get("trial"))
    task = str(trial.get("task") or "未提供任务")
    trial_id = str(trial.get("trial_id") or "未提供编号")
    title = f"评估报告 · {task} · {trial_id}"
    outcome = report.get("outcome")
    status = _outcome_status(outcome)
    body = f'<h1>{escape(title)}</h1><p class="muted">离线评估记录 · 任务结果优先 · 各项指标独立报告</p>'
    body += f'<section id="outcome" class="outcome {status}"><h2>任务结果</h2><p class="verdict">{_STATUS[status]}</p>'
    body += _value(outcome)
    body += '<h3>记录的任务终态</h3>' + _value(_mapping(report.get("task_specific")).get("partial_state"))
    body += '<h3>本次证据来源</h3>' + _value(trial.get("source")) + '</section>'
    if report.get("warnings"):
        body += '<aside class="notice"><strong>解释限制</strong>' + _value(report["warnings"]) + '</aside>'
    body += '<div class="columns">' + _section("效率", report.get("efficiency"), identity="efficiency")
    body += _section("安全", report.get("safety"), identity="safety") + '</div>'
    body += _section("任务专项证据", report.get("task_specific"), identity="task-specific")
    body += _section("证据质量与不可判定项", report.get("evidence_quality"), identity="evidence-quality")
    body += _section("事件记录", report.get("events"), identity="events")
    body += _section("分析区间", report.get("scope_intervals"), identity="scope-intervals")
    body += _section("任务条件的观测覆盖", report.get("scope_availability"), identity="scope-availability")
    body += _tactile(report.get("tactile"))
    body += '<section><h2>可追溯信息</h2><details><summary>协议、来源版本与限制</summary>'
    body += _value({key: report.get(key) for key in ("schema_version", "trial", "protocol_sha256", "protocol", "implementation", "provenance", "warnings", "limitations")}) + '</details>'
    body += _json_details(report) + '</section>'
    return _document(title, body)


def render_batch(report: dict) -> str:
    """Render all batch rows, including failures and unresolved outcomes."""
    rows = report.get("trials")
    trials = rows if isinstance(rows, (list, tuple)) else []
    body = '<h1>批次评估报告</h1><section><h2>按任务汇总</h2>'
    body += ('<p class="muted">确认成功比例的分母包含全部试验；已明确判定成功率的分母仅含通过与未通过。'
             '未知结果不当作成功。各任务分别汇总；任务等权宏平均不等于合并所有试验的成功率。</p>')
    body += _value(report.get("by_task"))
    if "macro_confirmed_success_fraction" in report:
        body += _value({"macro_confirmed_success_fraction": report["macro_confirmed_success_fraction"]})
    body += '</section>'
    if "by_cohort" in report:
        body += '<section><h2>按相同评估配置分组</h2><p class="muted">检查点、控制频率、执行步数和保护模式等配置分开统计；不将不同配置的力分布混为一次对比。</p>'
        for name, group in _mapping(report["by_cohort"]).items():
            group = _mapping(group)
            body += f'<article class="scope"><h3>配置组 <code>{escape(str(name))}</code></h3>'
            body += _value({k: v for k, v in group.items() if k != "metrics"})
            if "metrics" in group:
                body += '<details><summary>逐试验标量分布（每条轨迹贡献一个标量）</summary>' + _value(group["metrics"]) + '</details>'
            body += '</article>'
        body += '</section>'
    body += f'<section><h2>全部试验（{len(trials)} 条）</h2>'
    rendered = []
    for raw_trial in trials:
        trial = _mapping(raw_trial)
        outcome = trial.get("outcome")
        # Batch rows may store either the explicit status or the outcome object.
        status = outcome if outcome in ("pass", "fail", "unknown") else _outcome_status(outcome)
        rendered.append([_value(trial.get("trial_id")), _value(trial.get("task")), escape(_STATUS[status]),
                         _value(outcome), _value(trial.get("source")), _report_link(trial.get("report_path"))])
    body += _table(["试验编号", "任务", "任务判定", "判定明细", "证据来源（文本）", "本地报告"], rendered) if trials else '<p class="empty">未提供试验记录。</p>'
    body += '</section><section><h2>批次可追溯信息</h2>'
    meta = {k: v for k, v in report.items() if k not in ("trials", "by_task", "by_cohort")}
    body += _value(meta) + _json_details(report) + '</section>'
    return _document("批次评估报告", body)
