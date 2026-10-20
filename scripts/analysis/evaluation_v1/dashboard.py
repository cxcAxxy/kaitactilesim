"""Portable visual reports; computation remains in the offline evaluator.

The exported dashboard is a buildless, dependency-free browser component.
It consumes report.json and visualization.json, not simulator internals. The
audit view remains available separately. Never publish archived provenance
without reviewing paths, permissions, and the intended audience first.
"""

from __future__ import annotations

from html import escape
import json
from pathlib import Path


ASSET_NAMES = ("dashboard.js", "dashboard.css", "bootstrap.js")


def read_assets() -> dict[str, bytes]:
    root = Path(__file__).resolve().parent / "web"
    return {name: (root / name).read_bytes() for name in ASSET_NAMES}


def _embedded_json(value: object) -> str:
    # Script data is inert JSON. Escape tag delimiters even inside strings.
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).replace(
        "&", "\\u0026").replace("<", "\\u003c").replace(
        ">", "\\u003e").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def render_dashboard(report: dict, visualization: dict) -> str:
    title = escape(f"任务与触觉评估 · {report['trial']['task']}")
    payload = _embedded_json({"report": report, "visualization": visualization})
    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light">
<title>{title}</title><link rel="stylesheet" href="../assets/dashboard.css">
<script src="../assets/dashboard.js" defer></script>
<script src="../assets/bootstrap.js" defer></script>
</head><body style="margin:0;background:#f3f6fa">
<div id="kai-evaluation"><p style="padding:2rem">正在载入评估图表…</p></div>
<noscript><p>交互图表需要 JavaScript。<a href="audit.html">查看完整静态审计报告</a></p></noscript>
<script type="application/json" id="kai-evaluation-data">{payload}</script>
</body></html>'''


_BATCH_CSS = '''
*{box-sizing:border-box}body{margin:0;background:#f3f6fa;color:#172c44;font:15px/1.6 system-ui,sans-serif}
main{max-width:1220px;margin:auto;padding:34px 24px 64px}a{color:#087d94;text-decoration:none}a:hover{text-decoration:underline}
header{display:flex;gap:24px;justify-content:space-between;align-items:center;margin-bottom:28px}
h1{font-size:30px;letter-spacing:-.7px;margin:4px 0}h2{font-size:20px;margin:0 0 14px}p{margin:6px 0}.eyebrow{font-size:12px;letter-spacing:2px;color:#537085}
.muted{color:#607486;font-size:13px}.links{display:flex;flex-wrap:wrap;gap:18px}.summary{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:24px 0}
.card,section{background:#fff;border:1px solid #dce5ed;border-radius:14px;padding:22px}.summary strong{font-size:30px;display:block;line-height:1.2}.summary span{font-size:13px;color:#607486}
section{margin:20px 0}.task-row{display:grid;grid-template-columns:170px 1fr 220px;gap:20px;align-items:center;margin:18px 0}.stack{display:flex;height:12px;border-radius:5px;overflow:hidden;background:#edf1f6}
.pass{background:#2b947c}.fail{background:#dc7869}.unknown{background:#94a3b8}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}
.run{display:flex;flex-direction:column;border:1px solid #dce5ed;background:white;border-radius:12px;padding:20px;min-width:0}.run:hover{border-color:#58aabd;text-decoration:none;box-shadow:0 5px 20px #12354b0a}
.run strong{font-size:16px;color:#172c44}.identity{font:12px/1.7 ui-monospace,monospace;color:#647b8d;overflow-wrap:anywhere;margin:8px 0 16px}.run-foot{margin-top:auto;display:flex;justify-content:space-between;gap:8px;color:#087d94;font-size:13px}
.badge{border-radius:99px;padding:3px 9px;font-size:11px;display:inline-block;color:#964939;background:#fff0ec}.badge-pass{color:#176d59;background:#e7f6ee}.badge-unknown{color:#63748a;background:#edf1f6}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#edf3f8;padding:20px;border-radius:10px;font-size:13px}code{font-family:ui-monospace,monospace}
@media(max-width:850px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}.task-row{grid-template-columns:1fr}.task-row .muted{margin-top:-10px}}
@media(max-width:560px){main{padding:22px 14px}header{display:block}.grid{grid-template-columns:1fr}.summary{gap:8px}.card{padding:16px}.summary strong{font-size:25px}h1{font-size:26px}.links{margin-top:16px}}
'''


def _document(title: str, body: str) -> str:
    return f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{escape(title)}</title><style>{_BATCH_CSS}</style></head><body><main>{body}</main></body></html>'


def render_visual_batch(batch: dict) -> str:
    tasks = batch.get("by_task", {})
    body = '<header><div><div class="eyebrow">KAI · EVALUATION WORKSPACE</div><h1>任务与触觉评估</h1><p class="muted">从任务表现到接触细节，逐条检查真实记录。</p></div><nav class="links"><a href="batch_audit.html">完整批次审计 ↗</a><a href="integration.html">网站接入说明 ↗</a></nav></header>'
    body += '<div class="summary">' + ''.join(f'<div class="card"><strong>{value}</strong><span>{label}</span></div>' for value, label in (
        (batch["trial_count"], "条已记录轨迹"), (len(tasks), "类任务"),
        (len(batch.get("by_cohort", {})), "组独立部署配置"))) + '</div>'
    body += '<section><h2>记录的任务结果</h2><p class="muted">全部轨迹均计入，失败与未知不剔除。不同配置的任务汇总仅为描述，不是模型排名。</p>'
    for task, group in tasks.items():
        count = group["trial_count"]
        counts = group["outcome_counts"]
        bars = ''.join(f'<span class="{key}" style="width:{100 * counts[key] / count if count else 0:.6f}%"></span>' for key in ("pass", "fail", "unknown"))
        description = f'通过 {counts["pass"]} / 未通过 {counts["fail"]} / 未知 {counts["unknown"]}'
        body += f'<div class="task-row"><strong>{escape(task)}</strong><div class="stack" role="img" aria-label="{description}">{bars}</div><span class="muted">{description}</span></div>'
    body += '</section><h2>选择一条轨迹，进入可视化</h2><div class="grid">'
    from .render import _report_link
    for trial in batch["trials"]:
        path = trial.get("report_path")
        # Reuse the audit renderer\'s conservative relative-link validator.
        validated = _report_link(path)
        if not validated.startswith('<a href="'):
            continue
        outcome = trial.get("outcome") or {}
        status = outcome.get("status", "unknown")
        status = status if status in ("pass", "fail", "unknown") else "unknown"
        label = {"pass": "记录为通过", "fail": "记录为未通过", "unknown": "无法判定"}[status]
        body += (f'<a class="run" href="{escape(path, quote=True)}"><strong>{escape(trial["task"])}</strong>'
                 f'<div class="identity">{escape(trial["trial_id"])}</div><div class="run-foot">'
                 f'<span class="badge badge-{status}">{label}</span><span>视频与触觉图表 →</span></div></a>')
    body += '</div><p class="muted" style="margin-top:24px">可视化不改变原评估结果，也不通过专家时间轴相似度评分。缺失数据、未验证时钟和触觉证据局限均单独标注。</p>'
    return _document("任务与触觉评估", body)


def render_integration() -> str:
    example = '''<link rel="stylesheet" href="/evaluation/assets/dashboard.css">
<div id="evaluation-panel"></div>
<script src="/evaluation/assets/dashboard.js"></script>
<script type="module">
  const base = "/evaluation/YOUR_TRIAL/";
  const [report, visualization] = await Promise.all([
    fetch(base + "report.json").then(r => r.json()),
    fetch(base + "visualization.json").then(r => r.json())
  ]);
  if (visualization.media?.url) {
    visualization.media.url = base + visualization.media.url;
  }
  const view = KaiEvaluationDashboard.mount(
    document.getElementById("evaluation-panel"),
    {report, visualization}
  );
  // Unmount: view.destroy();
</script>'''
    body = '<header><div><div class="eyebrow">COMPONENT CONTRACT · V1</div><h1>接入仿真网站</h1></div><a href="index.html">← 返回报告</a></header>'
    body += '<section><h2>计算与展示分离</h2><p>Python 离线评估输出 <code>report.json</code> 和 <code>visualization.json</code>；浏览器组件仅展示，不重新计算成功率或触觉评分。组件无框架、无 CDN 依赖，可挂载多个实例，CSS 限定在组件内。</p><p>评估 JSON 保存完整统计；可视化 JSON 保存原生力曲线、有限数量的原生热图快照、任务曲线及经核实的视频映射。减少热图快照不会改变任何评估指标。</p></section>'
    body += '<section><h2>触觉视图切换</h2><p>触觉阵列支持“阵列热图”和“接触力圆图”。热图保留逐单元法向力；新增的 <code>contact_glyph</code> 数据保存全部原生时刻的法向力、力加权中心和过去 0.1 秒的力变化，不从稀疏热图快照计算变化量。两个视图都遵循同一轨迹的时间游标。</p><ul><li>圆面积与法向力 Fn 成正比，所有通道和时刻使用固定比例；圆不是实际接触面积。</li><li>仅在记录了有效的非退化局部 XY 几何时使用毫米坐标；否则显示列/行索引，不能推断毫米间距或物理接触位置。</li><li>ΔFn = Fn(t) − Fn(t−0.1s)。ΔFt 先对有符号切向分量求和，再对两个时刻作向量差；|ΔFt| 不是当前 |Ft|，也不是各单元切向力模长之和。</li><li>只有原生记录中存在两个匹配端点且窗口内无对应通道缺测或时钟缺口时才显示变化；无足够历史时显示不可用，不补零、不插值。</li><li>切向箭头使用单独的记录分量坐标，不声称与阵列行列方向对应，也不代表运动方向或滑移真值。</li><li>旧版数据没有 <code>contact_glyph</code> 时，保留热图，圆图明确提示未导出。</li></ul></section>'
    body += f'<section><h2>最小挂载示例</h2><pre><code>{escape(example)}</code></pre><p>也可以直接通过 iframe 嵌入每条轨迹的 <code>report.html</code>。拆分嵌入时，请由宿主设置审计链接与 JSON 下载入口；<code>audit.html</code> 等默认链接相对于宿主页面。</p></section>'
    body += '<section><h2>数据和部署边界</h2><ul><li>保留所有通道和缺失值；空值不是零力，间隔缺失不会连成连续曲线。</li><li>热图坐标是阵列索引；没有几何校准时不是毫米，也不是皮肤压力 Pa。</li><li>视频仅在帧号、录制帧率和仿真时间证据一致时联动；否则独立播放。</li><li>原生时间轴只解释该条轨迹，不是模型与专家的逐秒评分基准。</li><li>当前产物仅在本地预览。公开部署前审核原始路径、模型信息、媒体授权和访问控制；本组件不提供账号或后端上传服务。</li><li>评估协议和实现哈希保留在审计报告中；展示组件升级不会默默替换旧版报告。</li></ul></section>'
    return _document("网站接入说明", body)
