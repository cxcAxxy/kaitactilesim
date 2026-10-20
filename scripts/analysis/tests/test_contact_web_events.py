"""Events describe the current recording without carrying seed001 interpretations."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np

SPEC = importlib.util.spec_from_file_location("web_events", Path(__file__).resolve().parents[1] / "export_contact_web_data.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ContactWebEventTests(unittest.TestCase):
    def test_long_model_keeps_both_ends_and_neutral_snapshots(self):
        report = {"native_finger_peaks": {"rollout_0": {"time_s": 72.034, "force_n": 8.1}},
                  "reference_end_s": 66.705, "rollout_end_s": 90.0,
                  "final_model_pinky_no_contact": None}
        display = np.unique(np.r_[np.arange(0, 90.01, .1), 66.705])
        events = MODULE.events_payload(report, display)
        by_id = {item["id"]: item for item in events}
        self.assertEqual(by_id["end"]["displayTime"], 90.0)
        self.assertEqual(by_id["reference-end"]["displayTime"], 66.705)
        self.assertEqual(by_id["peak"]["time"], 72.034)
        self.assertNotIn("contact-loss", by_id)
        self.assertEqual(by_id["snapshot"]["label"], "34 秒快照")
        self.assertNotIn("搬运", by_id["baseline"]["label"])
        self.assertNotIn("短峰", by_id["peak"]["label"])
        self.assertEqual([e["time"] for e in events], sorted(e["time"] for e in events))

    def test_actual_low_contact_run_is_reported_without_causal_claim(self):
        report = {"native_finger_peaks": {}, "reference_end_s": 2.0, "rollout_end_s": 1.15,
                  "final_model_pinky_no_contact": {"start_s": .87, "duration_s": .28}}
        events = MODULE.events_payload(report, np.arange(0, 2.01, .1))
        by_id = {e["id"]: e for e in events}
        self.assertNotIn("baseline", by_id)
        self.assertNotIn("snapshot", by_id)
        self.assertEqual(by_id["contact-loss"]["label"], "小指低于门限")
        self.assertAlmostEqual(by_id["contact-loss"]["displayTime"], .9)


if __name__ == "__main__":
    unittest.main()
