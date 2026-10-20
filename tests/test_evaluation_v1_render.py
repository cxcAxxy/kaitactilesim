"""Behavioral checks for safe, complete, task-agnostic offline reports."""

from copy import deepcopy
from html.parser import HTMLParser
import importlib.util
from pathlib import Path
import unittest


_PATH = Path(__file__).resolve().parents[1] / "scripts/analysis/evaluation_v1/render.py"
_SPEC = importlib.util.spec_from_file_location("evaluation_v1_render", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class _Document(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tags = []
        self.text = []
        self.attributes = []

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        self.attributes.extend(attrs)

    def handle_data(self, text):
        self.text.append(text)


def _report():
    return {
        "schema_version": "evaluation-v1", "trial": {"task": "insert-custom-part", "trial_id": "trial-7", "source": "/data/trial-7"},
        "protocol": {"normal_threshold_n": .05}, "protocol_sha256": "a" * 64,
        "provenance": {"source_files": ["trace.npz"]}, "warnings": [],
        "outcome": {"status": "unknown", "reason": "终态证据缺失", "terminal_state": None},
        "safety": {"status": "unknown", "collision_duration_s": None},
        "efficiency": {"elapsed_s": 0, "success_time_s": None},
        "task_specific": {"distance_m": 0}, "events": [], "scope_intervals": {},
        "tactile": {"scopes": {}},
    }


class EvaluationRenderTests(unittest.TestCase):
    def test_outcome_first_and_zero_distinct_from_null(self):
        result = _MODULE.render_report(_report())
        self.assertIn('class="outcome unknown"', result)
        self.assertIn("未知／无法判定（unknown）", result)
        self.assertIn("缺失／未知（null）", result)
        self.assertIn('<td>0</td>', result)
        self.assertLess(result.index('id="outcome"'), result.index('id="efficiency"'))
        self.assertLess(result.index('id="safety"'), result.index('id="tactile"'))

    def test_explicit_status_only_no_bool_or_truthy_string_guessing(self):
        for status in (None, True, "success", "false", "available"):
            report = _report()
            report["outcome"] = {"status": status, "success": True}
            self.assertIn('class="outcome unknown"', _MODULE.render_report(report))
        for status in ("pass", "fail"):
            report = _report()
            report["outcome"]["status"] = status
            self.assertIn(f'class="outcome {status}"', _MODULE.render_report(report))

    def test_malicious_text_and_json_cannot_create_elements(self):
        payload = '</pre></details><script>alert("x")</script><img src=x onerror=alert(1)>&'
        report = _report()
        report["trial"].update(task=payload, source='javascript:alert("x")')
        report["outcome"]["reason"] = payload
        report["warnings"] = [payload]
        report["tactile"]["scopes"] = {payload: {"status": "available", "channels": {payload: {"status": "unknown"}}}}
        html = _MODULE.render_report(report)
        parsed = _Document()
        parsed.feed(html)
        self.assertNotIn("script", parsed.tags)
        self.assertNotIn("img", parsed.tags)
        self.assertNotIn("link", parsed.tags)
        self.assertFalse(any(name in ("href", "src", "onerror") for name, _ in parsed.attributes))
        self.assertIn(payload, "".join(parsed.text))
        self.assertIn("&lt;script&gt;", html)

    def test_arbitrary_channel_count_and_unknown_scope(self):
        report = _report()
        channels = {}
        for index in range(8):
            channels[f"pad-{index}"] = {
                "status": "available", "intensity": {"all_observed": {"mean_n": index}, "contact": None, "sampled_peak_n": index + .5, "coverage_fraction": 1},
                "change": {"status": "insufficient_samples", "p95_absolute_change_n": None},
                "continuity": {"contact_episode_count": 0}, "spatial": None, "tangent": None,
            }
        report["tactile"]["scopes"] = {
            "overall": {"status": "available", "duration_s": 10, "channels": channels, "load_shares": None},
            "custom-success-window": {"status": "not_reached", "duration_s": None, "channels": None, "load_shares": None},
        }
        result = _MODULE.render_report(report)
        for name in channels:
            self.assertIn(name, result)
        self.assertIn("未到达此阶段；不将缺失指标显示为零", result)
        self.assertIn("原生采样峰值", result)
        self.assertIn("力幅值", result)
        self.assertIn("时序变化", result)
        self.assertNotIn("白板", result)
        self.assertNotIn("小指", result)

    def test_preserves_input_and_full_precision_json(self):
        report = _report()
        report["task_specific"]["distance_m"] = .123456789123
        original = deepcopy(report)
        result = _MODULE.render_report(report)
        self.assertEqual(report, original)
        self.assertIn("0.123456789123", result)
        self.assertIn("a" * 64, result)

    def test_batch_keeps_failed_unknown_and_meta(self):
        report = {
            "schema_version": "batch-v1", "protocol_sha256": "hash-77",
            "trials": [{"trial_id": f"id-{status}", "task": "task-x", "outcome": {"status": status}, "source": f"/{status}", "report_path": f"{status}.html"} for status in ("pass", "fail", "unknown")],
            "by_task": {"task-x": {"trial_count": 3, "outcome_counts": {"pass": 1, "fail": 1, "unknown": 1}, "confirmed_success_fraction": 1/3, "resolved_success_rate": .5, "unknown_count": 1, "metrics": {}}},
            "macro_confirmed_success_fraction": 1/3, "extra_metadata": "keep-me",
        }
        result = _MODULE.render_batch(report)
        for status in ("pass", "fail", "unknown"):
            self.assertIn(f"id-{status}", result)
            self.assertIn(f"{status}.html", result)
        self.assertIn("全部试验（3 条）", result)
        self.assertIn("keep-me", result)
        self.assertIn("hash-77", result)

    def test_all_unknown_batch_does_not_invent_resolved_rate(self):
        report = {"trials": [{"trial_id": "unresolved", "outcome": "unknown"}], "by_task": {"task-x": {"resolved_success_rate": None, "confirmed_success_fraction": 0}}}
        result = _MODULE.render_batch(report)
        self.assertIn("未知／无法判定（unknown）", result)
        self.assertIn("缺失／未知（null）", result)
        self.assertIn('<td>0</td>', result)

    def test_batch_relative_links_only_and_distinct_cohorts(self):
        paths = ["000_custom_seed_0/report.html", "../report.html", "https://example.com/x.html",
                 "javascript:alert(1)", "//example.com/x.html", "%2e%2e/report.html",
                 "000/%252e%252e/x.html", "000/\\evil.html", "//[broken/report.html"]
        report = {"trials": [{"trial_id": str(i), "outcome": "fail", "report_path": path} for i, path in enumerate(paths)],
                  "by_cohort": {"checkpoint-A": {"configuration": {"checkpoint": "A"}, "trial_count": 1},
                                "checkpoint-B": {"configuration": {"checkpoint": "B"}, "trial_count": 1}}}
        parsed = _Document()
        parsed.feed(_MODULE.render_batch(report))
        self.assertEqual([value for key, value in parsed.attributes if key == "href"], [paths[0]])
        self.assertIn("checkpoint-A", "".join(parsed.text))
        self.assertIn("checkpoint-B", "".join(parsed.text))

    def test_task_terminal_evidence_precedes_efficiency(self):
        report = _report()
        report["task_specific"]["partial_state"] = {"insertion_depth": 0}
        result = _MODULE.render_report(report)
        self.assertLess(result.index("insertion_depth"), result.index('id="efficiency"'))


if __name__ == "__main__":
    unittest.main()
