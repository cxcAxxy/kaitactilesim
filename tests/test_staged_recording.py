import json
import pytest

from kaihand_tactile_env.shared.staged_recording import StagedMetricView, StagedStateLog, add_staged_batch_results


class Observer:
  time = 0.0

  def snapshot(self):
    return {"version": "staged_v1", "task": "bulb-screw", "stages": {"stage1": True, "stage2": False, "stage3": False},
            "highest_stage": 1, "complete_success": False, "measurements": {"time_s": self.time}}

  report = snapshot


def test_independent_log_preserves_native_status_and_validity(tmp_path):
  observer = Observer()
  log = StagedStateLog(tmp_path, observer)
  log.record(0)
  log.record(0)
  observer.time = 0.034
  log.record(1)
  report = {"task_success": False, "status": "task_not_completed", "diagnostic_only": True,
            "formal_metrics_valid": False, "physics_valid": True}
  result = log.finish(report)
  assert report["task_success"] is False
  assert report["status"] == "task_not_completed"
  assert result["state_log_samples"] == 2
  assert result["formal_metrics_valid"] is False
  assert result["legacy_task_success"] is False
  assert json.loads((tmp_path / "staged_evaluation.json").read_text()) == result
  assert len((tmp_path / "staged_states.jsonl").read_text().splitlines()) == 2


def test_stage_view_keeps_legacy_outcome(tmp_path):
  class Legacy:
    def metadata(self):
      return {"diagnostic_only": True}

    def snapshot(self, simulation=None):
      return {"success": False, "stage": "pick_bulb"}

  view = StagedMetricView(Legacy(), Observer())
  assert view.snapshot()["stage1"] is True
  assert view.snapshot()["success"] is False
  assert view.metadata()["staged_evaluation_version"] == "staged_v1"


def test_batch_rates_ignore_invalid_trials_and_keep_original_success(tmp_path):
  for seed, stage in ((0, True), (1, False)):
    directory = tmp_path / f"seed_{seed:03d}"
    directory.mkdir()
    (directory / "staged_evaluation.json").write_text(json.dumps({
      "version": "staged_v1", "task": "usb-insert", "stage_definitions": {"stage1": "pickup"},
      "thresholds": {"lift_m": 0.02}, "stages": {"stage1": stage}, "complete_success": stage,
      "continuous_metrics": {},
    }))
  native = {"task": "usb-insert", "planned": 20, "success_rate": 0.0, "successes": 0}
  add_staged_batch_results(native, tmp_path, [{"seed": 0, "valid_trial": True}, {"seed": 1, "valid_trial": False}])
  assert native["success_rate"] == 0.0 and native["successes"] == 0
  assert native["staged_evaluation"]["stages"]["stage1"]["count"] == 1
  assert native["staged_evaluation"]["stages"]["stage1"]["rate"] == 1.0
  assert native["staged_evaluation"]["valid_staged_trials"] == 1
  assert "completion_protocol" not in native["staged_evaluation"]


def _protocol_batch_report(directory, seed, protocol_fields):
  trial = directory / f"seed_{seed:03d}"
  trial.mkdir()
  (trial / "staged_evaluation.json").write_text(json.dumps({
    "version": "staged_v1", "task": "usb-unplug",
    "stage_definitions": {"stage1": "fully_extracted", "stage2": "extracted_and_held_2s"},
    "thresholds": {"usb_extracted_hold_s": 2.0},
    "stages": {"stage1": True, "stage2": True}, "complete_success": True,
    "continuous_metrics": {}, **protocol_fields,
  }))


@pytest.mark.parametrize("protocol", ["usb-unplug-held-clear-2s-v2", "unknown-future-protocol", None])
def test_staged_batch_preserves_only_recorded_protocol(tmp_path, protocol):
  for seed in (0, 1):
    _protocol_batch_report(tmp_path, seed, {"completion_protocol": protocol})
  native = {"task": "usb-unplug", "planned": 2, "successes": 0}
  add_staged_batch_results(native, tmp_path, [{"seed": seed, "valid_trial": True} for seed in (0, 1)])
  assert native["staged_evaluation"]["completion_protocol"] == protocol
  assert native["staged_evaluation"]["valid_staged_trials"] == 2
  assert native["successes"] == 0


@pytest.mark.parametrize("first,second", [
  ({}, {"completion_protocol": "usb-unplug-held-clear-2s-v2"}),
  ({"completion_protocol": "usb-unplug-held-clear-2s-v2"}, {}),
  ({"completion_protocol": "legacy-protocol"}, {"completion_protocol": "usb-unplug-held-clear-2s-v2"}),
  ({}, {"completion_protocol": None}),
  ({"completion_protocol": None}, {}),
  ({"completion_protocol": None}, {"completion_protocol": "usb-unplug-held-clear-2s-v2"}),
])
def test_staged_batch_rejects_protocol_changes_even_with_same_stages_and_thresholds(tmp_path, first, second):
  _protocol_batch_report(tmp_path, 0, first)
  _protocol_batch_report(tmp_path, 1, second)
  native = {"task": "usb-unplug", "planned": 2, "successes": 0}
  with pytest.raises(ValueError, match="completion_protocol changed"):
    add_staged_batch_results(native, tmp_path, [{"seed": seed, "valid_trial": True} for seed in (0, 1)])
  assert "staged_evaluation" not in native


@pytest.mark.parametrize("protocol", [False, 2, "", []])
def test_staged_batch_rejects_malformed_protocol(tmp_path, protocol):
  _protocol_batch_report(tmp_path, 0, {"completion_protocol": protocol})
  with pytest.raises(ValueError, match="completion_protocol"):
    add_staged_batch_results({"task": "usb-unplug", "planned": 1}, tmp_path,
                             [{"seed": 0, "valid_trial": True}])
