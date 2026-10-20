from html.parser import HTMLParser

from evaluation_status.unified_workbench_page import render_page


class Tags(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get("id"):
            self.ids.append(attrs["id"])


def test_original_workbench_shell_and_unified_comparison():
    raw = render_page({"schema": "kaihand-unified-evaluation-workbench-v1", "groups": [],
                       "counts": {"members": 595}})
    tags = Tags()
    tags.feed(raw)
    assert len(tags.ids) == len(set(tags.ids))
    assert {"kai-suite", "kai-suite-data", "ks-task-tabs", "ks-model", "ks-checkpoint",
            "ks-outcome", "ks-stage", "ks-search", "ks-comparison", "ks-trials",
            "ks-protocol", "ks-cohort", "ks-summary-note", "ks-summary-mode",
            "ks-summary-total"} <= set(tags.ids)
    assert "KaiTactileSim 官方评估平台" in raw
    assert "assets/suite.css" in raw
    assert "原生成功率" in raw and "完整阶段成功率" in raw
    assert 'class="card"' not in raw
    assert "595 条成功率与阶段汇总" in raw
    assert '<option value="policy" selected>' in raw
    assert '<option value="task" selected>' not in raw
    assert '<th>模型</th><th>Checkpoint</th>' in raw
    assert '每行只统计该组自己的成功率与阶段成功率' in raw


def test_historical_records_remain_in_catalog_but_not_a_second_summary():
    raw = render_page({"groups": [], "historical_summary": {
        "rows": [{"task": "card", "title": "旧任务", "model": "pi05", "checkpoint": "20000",
                  "count": 20, "native_success_count": 0, "complete_success_defined": False,
                  "stages": [{"label": "拿起", "reached": 14, "unknown": 0}]}],
    }})
    assert 'id="ks-historical-summary"' not in raw
    assert '"historical_summary"' in raw and '"reached":14' in raw
    assert "原 360 条已包含" in raw and "不重复计数" in raw
    assert raw.count('id="ks-comparison"') == 1


def test_catalog_is_inert_and_untrusted_links_are_not_emitted():
    raw = render_page({"groups": [], "title": "</script><script>bad()</script>", "links": [
        {"label": "<unsafe>", "url": "javascript:alert(1)"},
        {"label": "safe", "url": "../original/index.html"},
    ]})
    assert "<script>bad()" not in raw
    assert "javascript:alert" in raw  # only inert JSON, never an href
    assert 'href="javascript:' not in raw
    assert 'href="../original/index.html"' in raw
