"""Original KaiTactileSim workbench shell for all published policy results."""
from __future__ import annotations

from html import escape

from .export import inert


def render_page(catalog):
    """Render a single workbench, not a selector for separate experiments."""
    members = catalog.get("counts", {}).get("members")
    if type(members) is not int:
        members = sum(len(group.get("trials", [])) for group in catalog.get("groups", []))
    links = []
    for row in catalog.get("links", []):
        url = str(row.get("url", ""))
        # Publisher validates exact report targets. Never emit a foreign URL
        # from a catalog as a navigation link, even in standalone previews.
        if (not url or url.startswith(("/", "\\")) or ":" in url
                or "\\" in url or any(c in url for c in "\r\n\0")):
            continue
        links.append(f'<a href="{escape(url, quote=True)}">'
                     f'{escape(str(row.get("label", "归档报告")))}</a>')
    return ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>KaiTactileSim 官方评估平台</title>'
            '<link rel="stylesheet" href="assets/suite.css">'
            '<link rel="stylesheet" href="assets/unified_workbench.css">'
            '<script defer src="assets/unified_workbench.js"></script>'
            '</head><body><main id="kai-suite">'
            '<header><p class="ks-eyebrow">KaiTactileSim Evaluation Platform</p>'
            '<h1>KaiTactileSim 官方评估平台</h1>'
            '<p>机器人操作任务评估 · 动作回放 · 触觉分析</p>'
            '<div id="ks-counts" class="ks-counts"></div></header>'
            '<section class="ks-controls"><div id="ks-task-tabs" class="ks-task-tabs" '
            'role="group" aria-label="选择任务"></div><div class="ks-filter-row">'
            '<label>模型<select id="ks-model"><option value="">全部模型</option></select></label>'
            '<label>Checkpoint<select id="ks-checkpoint"><option value="">全部 checkpoint</option></select></label>'
            '<label>评估协议<select id="ks-protocol"><option value="">全部协议（分行对照）</option></select></label>'
            '<label>批次<select id="ks-cohort"><option value="">全部批次</option></select></label>'
            '<label>原生结果<select id="ks-outcome"><option value="">全部</option>'
            '<option value="pass">成功</option><option value="fail">未完成</option>'
            '<option value="unknown">未知 / 不计分</option></select></label>'
            '<label>阶段记录<select id="ks-stage"><option value="">全部</option>'
            '<option value="complete">完整阶段达成</option><option value="partial">已达部分阶段</option>'
            '<option value="none">未达首阶段</option><option value="unknown">未知 / 不计分</option></select></label>'
            '<label class="ks-search">查找 seed / 轨迹<input id="ks-search" type="search" '
            'placeholder="例如 seed_005"></label></div></section>'
            '<aside class="ks-import-note">原 360 条已包含在全部归档中，成功率与阶段统一在下表展示，不重复计数。'
            '默认按任务 × 模型 × checkpoint 分行，每行只统计该组自己的成功率与阶段成功率，不混合其他模型或 checkpoint。'
            '隔离及未验证阶段不补零、不计分。</aside>'
            '<section class="ks-comparison-panel" aria-labelledby="ks-comparison-title">'
            '<div class="ks-summary-heading">'
            f'<h2 id="ks-comparison-title">{members} 条成功率与阶段汇总</h2>'
            '<label class="ks-summary-mode">汇总方式<select id="ks-summary-mode">'
            '<option value="policy" selected>按任务 × 模型 × checkpoint（分别统计）</option>'
            '<option value="task">按任务合计（跨模型，仅描述性）</option></select></label></div>'
            '<div id="ks-summary-total" class="ks-summary-total" role="status"></div>'
            '<p id="ks-summary-note"></p><div class="ks-group-scroll"><table><thead><tr>'
            '<th>任务</th><th>模型</th><th>Checkpoint</th><th>评估协议与口径</th><th>有效样本</th>'
            '<th>原生成功率</th><th>完整阶段成功率</th><th>阶段到达 / 量化指标</th>'
            '</tr></thead><tbody id="ks-comparison"></tbody></table></div></section>'
            '<section id="ks-task-note" class="ks-task-note"></section>'
            '<p id="ks-visible" class="ks-visible" role="status"></p>'
            '<div id="ks-trials" class="ks-trials"></div>'
            '<details class="ks-stage-groups ks-archive-links"><summary>原始归档（只读审计）</summary>'
            '<nav aria-label="原始归档">' + ' · '.join(links) + '</nav></details>'
            '<p class="ks-footnote">结果与搜索筛选只影响轨迹列表；成功率以选定任务、模型、checkpoint、协议和批次的完整样本计算，'
            '避免只看成功或失败轨迹造成选择偏差。所有历史结果与视频保留原样。</p>'
            '<p class="ks-brand-footer">KaiTactileSim 官方评估平台 · 评估规则与结果以各批次归档为准</p>'
            '</main><script type="application/json" id="kai-suite-data">' + inert(catalog)
            + '</script><noscript>请启用 JavaScript，或查看 <a href="catalog.json">完整汇总清单</a>。</noscript>'
            '</body></html>')
