"""Buildless local experiment page. JSON is data, never injected as markup."""
import json


def render(data):
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    payload = payload.replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    return '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>任务进度参照 · 白板实验版</title>
<link rel="stylesheet" href="assets/progress.css"><script defer src="assets/progress.js"></script>
</head><body><main id="progress-app">
<header><div><p class="eyebrow">KAI · PROGRESS LENS / EXPERIMENT 01</p>
<h1>接触，与任务进展一起看。</h1><p class="subtitle">按真实清洁进度查看载荷与停留，不把专家和模型拉到同一条时间轴。</p></div>
<nav><a id="baseline-link">返回 H50 稳定版 ↗</a><a href="rollback.json">回退校验清单</a><a href="comparison.json">完整数据</a></nav></header>
<section class="notice"><strong>实验功能 · 案例参照，不是优劣评分</strong>
<p>专家与模型的采样率、物理采样相位及场景兼容性尚未核实。一条成功示范不能代表专家总体分布；本页不画“合格带”、不作安全或因果判断。</p></section>
<section id="overview" class="overview" aria-label="样本范围"></section>
<section class="controls panel"><div><label for="policy-select">模型轨迹 · H50</label><select id="policy-select"></select></div>
<div><label for="expert-select">专家参考 · 独立时钟</label><select id="expert-select"></select></div>
<div><label for="signal-select">观察哪一个接触？</label><select id="signal-select"><option value="board_normal">板擦—白板法向力</option><option value="hand_normal_sum">手指法向力标量总和</option></select></div></section>
<section class="panel progress-selector"><div class="section-heading"><div><h2>选一个真实进度区间</h2>
<p>分箱依据记录的清洁比例，不是“用了多少时间”。未观测区间保留空白。</p></div><span id="bin-caption" class="chip"></span></div>
<div id="bin-buttons" class="bin-buttons" role="group" aria-label="清洁进度区间"></div>
<p class="fine">注意：进度区间不等于动作阶段。0% 可能包含接近或尚未开始；100% 可能包含收尾。未变化不等于卡住，达到 100% 清洁也不等于完成放回、释放等全部任务。</p></section>
<section id="comparison-panes" class="comparison-panes" aria-label="模型和专家独立查看"></section>
<section class="panel"><h2>读图边界</h2><div class="boundaries">
<p><strong>力和效果各保留一个量。</strong> 力说明记录载荷；清洁比例说明记录效果。两者一起变化不证明前者导致后者。</p>
<p><strong>失败和缺失不消失。</strong> 失败轨迹不会被拉伸到 100%；跨采样缺口不连线。先看实际覆盖，再看区间内数据。</p>
<p><strong>不强行同步播放。</strong> 点击曲线或“跳至区间”只控制这一条轨迹。力图共用显示量程，不代表传感器、采样带宽或物理条件等价。</p>
<p><strong>试点不是完整动作解释。</strong> 本页尚未加入独立验证的姿态、运动速度、动作指令与阶段匹配；不能据此认证抓稳、滑脱、卡住或触觉带来的收益。</p>
</div></section><footer>独立实验目录 · 原评估入口和报告未覆盖 · 不修改仿真源码或运行中的任务</footer>
</main><script type="application/json" id="progress-data">''' + payload + '</script></body></html>'
