"""Visual report packaging is inert, portable, and independent of scoring."""

from copy import deepcopy
from html.parser import HTMLParser
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/analysis"))
from evaluation_v1.dashboard import render_dashboard, render_visual_batch, render_integration


class Document(HTMLParser):
    def __init__(self):
        super().__init__()
        self.scripts, self.current, self.data, self.tags = [], False, [], []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.tags.append(tag)
        if tag == "script":
            self.scripts.append(attrs)
            self.current = attrs.get("id") == "kai-evaluation-data"

    def handle_endtag(self, tag):
        if tag == "script":
            self.current = False

    def handle_data(self, data):
        if self.current:
            self.data.append(data)


class DashboardPackagingTests(unittest.TestCase):
    def test_embedded_json_cannot_escape_into_executable_markup(self):
        payload = '</script><img src=x onerror="alert(1)"><script>alert(1)</script>&\u2028'
        report = {"trial": {"task": payload}, "outcome": {"status": "fail"}, "warnings": [payload]}
        visual = {"schema_version": "evaluation-visualization-v1", "warnings": [payload]}
        page = render_dashboard(report, visual)
        parser = Document()
        parser.feed(page)
        self.assertEqual(len(parser.scripts), 3)
        self.assertNotIn("img", parser.tags)
        self.assertEqual(json.loads("".join(parser.data)), {"report": report, "visualization": visual})
        self.assertEqual([s.get("src") for s in parser.scripts[:2]],
                         ["../assets/dashboard.js", "../assets/bootstrap.js"])

    def test_rendering_does_not_mutate_metrics_and_uses_only_local_assets(self):
        report = {"trial": {"task": "custom"}, "outcome": {"status": "unknown"}}
        before = deepcopy(report)
        page = render_dashboard(report, {"tactile_series": {"time_s": []}})
        self.assertEqual(report, before)
        self.assertIn('href="audit.html"', page)
        self.assertNotIn("https://", page)
        self.assertIn('type="application/json"', page)

    def test_nonfinite_json_never_leaks_to_browser(self):
        with self.assertRaises(ValueError):
            render_dashboard({"trial": {"task": "custom"}}, {"values": [float("nan")]})

    def test_batch_visual_counts_preserve_failures_unknown_and_escape_labels(self):
        batch = {"trial_count": 3, "by_cohort": {"one": {}}, "by_task": {
            "<task>": {"trial_count": 3, "outcome_counts": {"pass": 1, "fail": 1, "unknown": 1}}},
            "trials": [{"task": "<task>", "trial_id": status, "outcome": {"status": status},
                        "report_path": f"{status}/report.html"} for status in ("pass", "fail", "unknown")]}
        page = render_visual_batch(batch)
        self.assertIn("通过 1 / 未通过 1 / 未知 1", page)
        for status in ("pass", "fail", "unknown"):
            self.assertIn(f'href="{status}/report.html"', page)
        self.assertIn("&lt;task&gt;", page)
        self.assertNotIn("<task>", page)

    def test_invalid_report_urls_are_not_executable(self):
        batch = {"trial_count": 1, "by_cohort": {}, "by_task": {}, "trials": [{
            "task": "custom", "trial_id": "one", "outcome": {"status": "pass"},
            "report_path": "javascript:alert(1).html"}]}
        self.assertNotIn("javascript:", render_visual_batch(batch))

    def test_integration_contract_documents_cleanup_and_publication_boundary(self):
        page = render_integration()
        self.assertIn("KaiEvaluationDashboard.mount", page)
        self.assertIn("view.destroy()", page)
        self.assertIn("访问控制", page)
        self.assertIn("原生时间轴", page)
        self.assertIn("contact_glyph", page)
        self.assertIn("圆不是实际接触面积", page)
        self.assertIn("不补零、不插值", page)


if __name__ == "__main__":
    unittest.main()
