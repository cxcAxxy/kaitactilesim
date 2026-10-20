"""Small self-contained local page, native HTML controls and canvas evidence plots."""
import json


def render(data):
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    return '''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>动作证据复核 · H50 实验版</title><link rel="stylesheet" href="assets/review.css"></head>
<body><header class="topbar"><div><span class="eyebrow">KAI / ACTION EVIDENCE</span><h1>看动作，再看接触与任务响应</h1></div>
<a id="baseline-link">返回 H50 稳定版 ↗</a></header>
<main><div class="episode-bar"><label for="episode-select">H50 · 白板试点</label><select id="episode-select"></select><span id="outcome"></span></div>
<p class="scope">实验性复核工具，不新增评分。时间轴只联读同一条轨迹；不把模型与专家按秒对齐，也不将相关性当作原因。</p>
<section class="review-layout" aria-label="动作与记录证据">
<div class="video-panel"><div class="panel-heading"><h2>01 / 看这一段动作</h2><label>画面 <select id="view-select"><option value="full">完整拼图</option></select></label></div>
<div id="video-viewport" class="video-viewport"><video id="main-video" preload="metadata" playsinline muted></video><div id="video-missing" hidden>没有可验证的视频映射</div></div>
<div class="transport"><button id="play" type="button">播放</button><button id="replay" type="button">重播选段</button><label><input id="loop" type="checkbox" checked> 循环选段</label><label><input id="context" type="checkbox" checked> 前后各 2 s</label><label>倍速 <select id="speed"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select></label><output id="clock"></output></div>
<p id="position-note" class="fine"></p>
<label class="scrub-label" for="scrubber">全程游标（拖动可回看任意记录）</label><input id="scrubber" class="scrubber" type="range" step="1" aria-label="全程记录游标">
<div id="instant" class="instant" aria-live="off"></div><p id="frame-note" class="fine"></p></div>
<aside class="fact-panel"><span class="eyebrow">02 / 这段记录能说明什么</span><h2 id="event-title"></h2><p id="event-range" class="muted"></p><div id="event-facts"></div>
<div class="evidence-boundary"><strong>还不能回答</strong><p>板擦具体擦到哪里、是否相对手掌滑动，以及这些读数变化的原因。模型记录缺少逐时板擦位姿和逐块污迹，不能自动归因。</p></div>
<button id="open-contrast" type="button">找一段本轨迹的进度增长片段对照 ↓</button></aside></section>
<section class="timeline-panel"><div class="panel-heading"><h2>03 / 放回全程看</h2><button id="whole" type="button">取消选段 · 看全程</button></div>
<div class="legend"><span class="plateau">清洁读数不变 ≥ 2 s</span><span class="growth">读数连续增长</span><span class="other">其余记录 ≠ 正常</span><span class="gap">缺口 / 进度缺测</span></div>
<canvas id="coverage-chart" class="coverage-chart" aria-label="全程候选片段与记录覆盖"></canvas><p id="coverage-text" class="fine"></p>
<div class="chart-heading"><span>沿同一条记录的原生时钟联读，点击曲线定位视频</span><label><input type="checkbox" id="hand-toggle" checked> 显示手指触觉</label></div>
<canvas id="signal-chart" class="signal-chart" aria-label="清洁进度、板面力和手指触觉的同轨迹时间线"></canvas>
<p class="fine">板面力：板擦—白板接触对的记录载荷。手指力：五指法向载荷之和，接触对象未逐点解析；两者不能互相替代。零读数不是几何分离证明。缺测断线，不补零。</p></section>
<section class="candidates-panel"><div class="panel-heading"><h2>04 / 全部候选片段</h2><label>筛选 <select id="event-filter"><option value="all">全部</option><option value="plateau">清洁读数不变</option><option value="growth">进度增长</option></select></label></div>
<p id="candidate-count" class="muted"></p><div id="event-list" class="event-list"></div></section>
<section id="contrast-panel" class="contrast-panel" hidden><div class="panel-heading"><h2>同一轨迹，换一段动作看</h2><button id="close-contrast" type="button">收起</button></div>
<p>两段独立播放，不强行同步。不代表相同位置、姿态或污迹条件；这是人工对照，不是“正确动作”标签。默认先选按时间排序首个 ≥ 0.3 s 的增长窗口，便于约 10 fps 视频观察；不足一帧的窗口仍全部可选。</p>
<label for="contrast-select">右侧：选择连续增长片段</label><select id="contrast-select"></select>
<div class="contrast-grid"><div><h3 id="selection-title"></h3><div id="selection-viewport" class="video-viewport"><video id="selection-video" preload="metadata" playsinline muted></video></div><button id="selection-replay" type="button">播放主选段（含上下文）</button><output id="selection-clock"></output><div id="selection-facts"></div></div>
<div><h3 id="contrast-title"></h3><div id="contrast-viewport" class="video-viewport"><video id="contrast-video" preload="metadata" playsinline muted controls></video></div><button id="contrast-replay" type="button">播放对照段（含上下文）</button><output id="contrast-clock"></output><div id="contrast-facts"></div></div></div></section>
<details class="method"><summary>筛选规则、证据边界与来源</summary><p>不变片段：连续样本的清洁比例总范围 ≤ 1e−12，持续至少 2 秒；缺口和未知值立即断开。增长片段：相邻比例差 &gt; 1e−12 的连续区间，全部保留，不设最短时长。0% 与 100% 的不变也保留，不当作失败。</p>
<p>默认推荐：每个板面力样本已知且为正的最长不变片段；没有则最长不变片段，再没有则增长片段。固定探索规则，不是经过验证的失效检测器。全候选和全程均可检查。</p>
<p>时间线表示记录时钟关联；视频按已存帧映射定位，不能保证传感器在物理上瞬时同相位。稀疏视频不能排除瞬时接触或滑移。手腕采样路径不是板擦轨迹。</p>
<p>专家示范不再占主视图；回到稳定版可查看既有指标。此试点仅含稳定快照内两条白板模型轨迹，不代表全任务或总体成功率，不读取正在写入的评估。</p>
<a id="source-link">查看这条轨迹的稳定报告 ↗</a> · <a href="review.json">下载证据数据</a> · <a href="rollback.json">稳定版保护清单</a> · <a href="verification.json">导出核验</a><pre id="provenance"></pre></details>
</main><footer>原 H50 稳定版保持不变 · 新入口仅用于观察、提问与复核</footer>
<script type="application/json" id="review-data">''' + payload + '''</script><script src="assets/review.js"></script></body></html>'''
