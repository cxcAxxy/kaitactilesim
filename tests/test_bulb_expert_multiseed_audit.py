from __future__ import annotations

import importlib.util
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from kaihand_tactile_env.tasks.bulb_screw import config


def _module():
  path = (
    Path(__file__).resolve().parents[1]
    / "scripts/workcell/audit_bulb_expert_multiseed.py"
  )
  spec = importlib.util.spec_from_file_location("bulb_expert_multiseed_audit", path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


@dataclass
class _State:
  bulb_lit: bool = True
  seated: bool = True
  engaged: bool = True
  released: bool = True
  release_stable_duration_s: float = 0.25
  right_hand_contact_count: int = 0
  right_hand_nonpad_contact_count: int = 0
  right_hand_contact_load_n: float = 0.0
  robot_contact_count: int = 0
  robot_contact_load_n: float = 0.0


@dataclass
class _Result:
  success: bool = True
  reason: str = "completed"
  elapsed_s: float = 42.0
  tightening_verified: bool = True
  tightening_stall_duration_s: float = 0.302
  tightening_peak_torque_nm: float = 0.6
  final_fingertip_load_n: tuple[float, ...] = (0.0,) * 5
  state: _State | None = None
  grasp_mode: str = "five-finger"
  rotation_driver: str = config.ROTATION_DRIVER

  def __post_init__(self):
    if self.state is None:
      self.state = _State()


class _Simulation:
  def __init__(self, *, position_seed):
    self.position_seed = position_seed
    self.initial_position_randomization = {
      "version": "pickup_xy_v1",
      "seed": position_seed,
    }


def _history(*, clean=True):
  return deque(
    (
      time_s,
      angle,
      torque,
      load,
      True,
      clean,
    )
    for time_s, angle, torque, load in (
      (10.000, 0.0000, 0.15, 4.8),
      (10.150, 0.0010, 0.13, 4.5),
      (10.302, 0.0015, 0.14, 4.7),
    )
  )


def _executor(result=None, *, history=None):
  result = result or _Result()
  history = history or _history()

  class Executor:
    def __init__(self, simulation, observer, *, grasp_mode, speed):
      assert grasp_mode == "five-finger" and speed == "fast"
      self.simulation = simulation
      self.observer = observer
      self._tightening_verified = False
      self._tightening_history = deque()
      self._state = _State(released=False, release_stable_duration_s=0.0)

    def execute(self):
      self._tightening_history = history
      self._tightening_verified = True
      self.observer(self.simulation, "tighten")
      self._state = result.state
      self.observer(self.simulation, "verify_seated")
      return result

  return Executor


def _environment():
  return {
    "scene": config.SCENE_NAME,
    "mechanics_version": config.MECHANICS_VERSION,
    "model_fingerprint": "a" * 64,
  }


def test_json_bytes_normalizes_numpy_scalars_and_arrays():
  module = _module()

  encoded = module._json_bytes(
    {
      "flag": np.bool_(True),
      "count": np.int64(3),
      "values": np.asarray([1.25, 2.5], dtype=np.float32),
    }
  )

  assert json.loads(encoded) == {
    "count": 3,
    "flag": True,
    "values": [1.25, 2.5],
  }


def test_run_seed_freezes_first_exact_strict_window(monkeypatch):
  module = _module()
  monkeypatch.setattr(module, "BulbScrewSimulation", _Simulation)
  monkeypatch.setattr(module, "BulbScrewExecutor", _executor())

  row = module._run_seed(7, _environment(), 10.0)

  assert row["status"] == "completed"
  assert row["qualified_success"]
  assert row["qualification_failures"] == []
  assert row["initial_position_randomization"]["seed"] == 7
  window = row["strict_tightening"]["certified_window"]
  assert window["duration_s"] == pytest.approx(0.302)
  assert window["minimum_all_finger_load_n"] == pytest.approx(4.5)
  assert window["minimum_clockwise_torque_nm"] == pytest.approx(0.13)
  assert window["rotation_span_deg"] == pytest.approx(0.085943669, rel=1e-6)
  assert window["all_samples_clean_contact"]
  assert window["contract_satisfied"]
  assert window["margins"]["all_finger_load_n"] == pytest.approx(0.5)
  assert row["qualification"]["release_stable_duration_s"] == pytest.approx(0.25)


def test_qualified_success_rejects_light_release_dwell_and_robot_contact(
  monkeypatch,
):
  module = _module()
  failed = _Result(
    state=_State(
      bulb_lit=False,
      released=False,
      release_stable_duration_s=0.1,
      right_hand_contact_count=1,
      right_hand_nonpad_contact_count=1,
      right_hand_contact_load_n=0.02,
      robot_contact_count=1,
      robot_contact_load_n=0.02,
    )
  )
  monkeypatch.setattr(module, "BulbScrewSimulation", _Simulation)
  monkeypatch.setattr(module, "BulbScrewExecutor", _executor(failed))

  row = module._run_seed(3, _environment(), 10.0)

  assert row["status"] == "completed"
  assert not row["qualified_success"]
  assert {
    "bulb_lit",
    "released",
    "release_dwell",
    "final_right_hand_contact_count_zero",
    "final_right_hand_contact_load",
    "final_robot_contact_count_zero",
    "final_robot_contact_load",
  } <= set(row["qualification_failures"])


def test_release_qualification_uses_terminal_dwell_not_historical_maximum():
  module = _module()
  observer = module.StrictTighteningObserver(10.0)
  observer.certified_window = observer._freeze_history(_history())
  observer.maximum_release_stable_duration_s = 1.0
  result = _Result(
    state=_State(released=True, release_stable_duration_s=0.05),
  )

  qualification, failures = module._qualification(result, observer)

  assert qualification["release_stable_duration_s"] == pytest.approx(0.05)
  assert qualification["maximum_observed_release_stable_duration_s"] == pytest.approx(
    1.0
  )
  assert "release_dwell" in failures


def test_fresh_qualification_rejects_malformed_release_loads():
  module = _module()
  observer = module.StrictTighteningObserver(10.0)
  observer.certified_window = observer._freeze_history(_history())
  result = _Result(
    final_fingertip_load_n=(),
    state=_State(right_hand_contact_load_n=-1.0, robot_contact_load_n=-1.0),
  )

  _qualification, failures = module._qualification(result, observer)

  assert {
    "final_fingertip_load",
    "final_right_hand_contact_load",
    "final_robot_contact_load",
  } <= set(failures)


def _completed_row(module, seed, environment):
  return {
    "schema": module.SEED_SCHEMA,
    "seed": seed,
    "environment": environment,
    "status": "completed",
    "qualified_success": True,
    "qualification_failures": [],
    "initial_position_randomization": {"seed": seed},
    "strict_tightening": {
      "certified_window": {
        "found": True,
        "sample_rate_hz": 500,
        "samples": 152,
        "start_s": 31.642,
        "end_s": 31.944,
        "duration_s": 0.302,
        "minimum_all_finger_load_n": 4.5,
        "minimum_clockwise_torque_nm": 0.13,
        "rotation_span_deg": 0.1,
        "all_samples_seated": True,
        "all_samples_clean_contact": True,
        "contract_checks": {
          "duration": True,
          "rotation": True,
          "torque": True,
          "all_five_loaded": True,
          "seated": True,
          "clean_contact": True,
        },
        "contract_satisfied": True,
        "margins": {
          "duration_s": 0.002,
          "all_finger_load_n": 0.5,
          "clockwise_torque_nm": 0.03,
          "rotation_deg": 0.2,
        },
      }
    },
    "qualification": {
      "checks": {
        "task_result_success": True,
        "bulb_lit": True,
        "tightening_verified": True,
        "reported_tightening_stall_duration": True,
        "reported_tightening_peak_torque": True,
        "strict_window_captured": True,
        "strict_window_contract_satisfied": True,
        "seated": True,
        "thread_engaged": True,
        "released": True,
        "release_dwell": True,
        "final_fingertip_load": True,
        "final_right_hand_contact_count_zero": True,
        "final_right_hand_contact_load": True,
        "final_robot_contact_count_zero": True,
        "final_robot_contact_load": True,
        "five_finger_mode": True,
        "finger_rotation_driver": True,
      },
      "release_stable_duration_s": 0.25,
      "maximum_observed_release_stable_duration_s": 0.25,
      "release_dwell_margin_s": 0.05,
    },
    "task_result": {
      "success": True,
      "reason": "completed",
      "elapsed_s": 42.0,
      "tightening_verified": True,
      "tightening_stall_duration_s": 0.302,
      "tightening_peak_torque_nm": 0.6,
      "final_fingertip_load_n": [0.0] * 5,
      "grasp_mode": "five-finger",
      "rotation_driver": config.ROTATION_DRIVER,
      "state": {
        "bulb_lit": True,
        "seated": True,
        "engaged": True,
        "released": True,
        "release_stable_duration_s": 0.25,
        "right_hand_contact_count": 0,
        "right_hand_contact_load_n": 0.0,
        "robot_contact_count": 0,
        "robot_contact_load_n": 0.0,
      },
    },
    "error": None,
    "wall_duration_s": 50.0,
  }


def test_main_writes_new_seed_files_and_resumes_without_rerunning(
  tmp_path, monkeypatch
):
  module = _module()
  environment = _environment()
  calls = []

  def run_seed(seed, actual_environment, wall_limit_s):
    calls.append((seed, wall_limit_s))
    assert actual_environment == environment
    return _completed_row(module, seed, environment)

  monkeypatch.setattr(module, "_environment_identity", lambda: environment)
  monkeypatch.setattr(module, "_run_seed", run_seed)
  output = tmp_path / "audit"

  assert module.main(["--output-dir", str(output), "--seeds", "2", "5"]) == 0
  assert calls == [(2, 180.0), (5, 180.0)]
  first = json.loads((output / "seed_000002.json").read_text())
  aggregate = json.loads((output / "aggregate.json").read_text())
  assert first["seed"] == 2
  assert aggregate["requested_seeds"] == [2, 5]
  assert aggregate["mechanics_version"] == config.MECHANICS_VERSION
  assert aggregate["model_fingerprint"] == "a" * 64
  assert aggregate["qualified_success_rate"] == 1.0
  assert aggregate["margin_metrics"]["all_finger_load_n"]["minimum"] == 0.5
  assert not tuple(output.glob(".*.tmp"))

  monkeypatch.setattr(
    module,
    "_run_seed",
    lambda *args, **kwargs: pytest.fail("resume reran a completed seed"),
  )
  assert (
    module.main(["--output-dir", str(output), "--seeds", "2", "5", "--resume"]) == 0
  )
  with pytest.raises(FileExistsError, match="--resume"):
    module.main(["--output-dir", str(output), "--seeds", "2", "5"])

  seed_path = output / "seed_000002.json"
  tampered = json.loads(seed_path.read_text())
  tampered["wall_duration_s"] = 51.0
  seed_path.write_text(json.dumps(tampered))
  with pytest.raises(ValueError, match="hash does not match aggregate"):
    module.main(["--output-dir", str(output), "--seeds", "2", "5", "--resume"])

  seed_path.unlink()
  with pytest.raises(ValueError, match="listed by aggregate is missing"):
    module.main(["--output-dir", str(output), "--seeds", "2", "5", "--resume"])


def test_resume_rejects_stale_environment(tmp_path, monkeypatch):
  module = _module()
  old_environment = _environment()
  output = tmp_path / "audit"
  output.mkdir()
  module._write_json_atomic(
    output / "seed_000001.json",
    _completed_row(module, 1, old_environment),
    overwrite=False,
  )
  current = {**old_environment, "model_fingerprint": "b" * 64}
  monkeypatch.setattr(module, "_environment_identity", lambda: current)

  with pytest.raises(ValueError, match="stale environment provenance"):
    module.main(["--output-dir", str(output), "--seeds", "1", "--resume"])


def test_main_rejects_environment_drift_before_publishing(tmp_path, monkeypatch):
  module = _module()
  environment = _environment()
  current = environment.copy()
  monkeypatch.setattr(module, "_environment_identity", lambda: current.copy())

  def run_seed(seed, actual_environment, wall_limit_s):
    del wall_limit_s
    row = _completed_row(module, seed, actual_environment)
    current["model_fingerprint"] = "b" * 64
    return row

  monkeypatch.setattr(module, "_run_seed", run_seed)

  with pytest.raises(RuntimeError, match="environment changed"):
    module.main(["--output-dir", str(tmp_path / "audit"), "--seeds", "1"])


def test_resume_rejects_parseable_but_inconsistent_qualification(tmp_path, monkeypatch):
  module = _module()
  environment = _environment()
  output = tmp_path / "audit"
  output.mkdir()
  row = _completed_row(module, 1, environment)
  row["task_result"]["success"] = False
  module._write_json_atomic(
    output / "seed_000001.json",
    row,
    overwrite=False,
  )
  monkeypatch.setattr(module, "_environment_identity", lambda: environment)

  with pytest.raises(ValueError, match="stored qualification checks"):
    module.main(["--output-dir", str(output), "--seeds", "1", "--resume"])


def test_resume_rejects_non_boolean_success_flag(tmp_path, monkeypatch):
  module = _module()
  environment = _environment()
  output = tmp_path / "audit"
  output.mkdir()
  row = _completed_row(module, 1, environment)
  row["task_result"]["success"] = "false"
  module._write_json_atomic(
    output / "seed_000001.json",
    row,
    overwrite=False,
  )
  monkeypatch.setattr(module, "_environment_identity", lambda: environment)

  with pytest.raises(ValueError, match="must be a JSON boolean"):
    module.main(["--output-dir", str(output), "--seeds", "1", "--resume"])


def test_resume_rejects_non_numeric_infinite_string(tmp_path, monkeypatch):
  module = _module()
  environment = _environment()
  output = tmp_path / "audit"
  output.mkdir()
  row = _completed_row(module, 1, environment)
  row["strict_tightening"]["certified_window"]["duration_s"] = "inf"
  module._write_json_atomic(
    output / "seed_000001.json",
    row,
    overwrite=False,
  )
  monkeypatch.setattr(module, "_environment_identity", lambda: environment)

  with pytest.raises(ValueError, match="must be a finite JSON number"):
    module.main(["--output-dir", str(output), "--seeds", "1", "--resume"])


@pytest.mark.parametrize(
  ("mutation", "message"),
  (
    ("negative_rotation", "timing/load domain"),
    ("empty_fingertip_loads", "five nonnegative values"),
    ("negative_contact_load", "contact loads and peak torque"),
  ),
)
def test_resume_rejects_nonphysical_load_domains(
  tmp_path, monkeypatch, mutation, message
):
  module = _module()
  environment = _environment()
  output = tmp_path / "audit"
  output.mkdir()
  row = _completed_row(module, 1, environment)
  if mutation == "negative_rotation":
    window = row["strict_tightening"]["certified_window"]
    window["rotation_span_deg"] = -100.0
    window["margins"]["rotation_deg"] = 100.3
  elif mutation == "empty_fingertip_loads":
    row["task_result"]["final_fingertip_load_n"] = []
  else:
    row["task_result"]["state"]["right_hand_contact_load_n"] = -1.0
  module._write_json_atomic(
    output / "seed_000001.json",
    row,
    overwrite=False,
  )
  monkeypatch.setattr(module, "_environment_identity", lambda: environment)

  with pytest.raises(ValueError, match=message):
    module.main(["--output-dir", str(output), "--seeds", "1", "--resume"])


def test_resume_rejects_randomization_seed_mismatch(tmp_path, monkeypatch):
  module = _module()
  environment = _environment()
  output = tmp_path / "audit"
  output.mkdir()
  row = _completed_row(module, 1, environment)
  row["initial_position_randomization"]["seed"] = 2
  module._write_json_atomic(
    output / "seed_000001.json",
    row,
    overwrite=False,
  )
  monkeypatch.setattr(module, "_environment_identity", lambda: environment)

  with pytest.raises(ValueError, match="randomization seed is inconsistent"):
    module.main(["--output-dir", str(output), "--seeds", "1", "--resume"])


def test_resume_rejects_exception_with_task_result(tmp_path, monkeypatch):
  module = _module()
  environment = _environment()
  output = tmp_path / "audit"
  output.mkdir()
  row = _completed_row(module, 1, environment)
  row.update(
    status="exception",
    qualified_success=False,
    qualification_failures=["trial_exception"],
    qualification=None,
    error={"type": "RuntimeError", "message": "failed", "traceback": "trace"},
  )
  module._write_json_atomic(
    output / "seed_000001.json",
    row,
    overwrite=False,
  )
  monkeypatch.setattr(module, "_environment_identity", lambda: environment)

  with pytest.raises(ValueError, match="task_result and qualification must be null"):
    module.main(["--output-dir", str(output), "--seeds", "1", "--resume"])


def test_exception_certified_window_is_excluded_from_aggregate(tmp_path):
  module = _module()
  environment = _environment()
  row = _completed_row(module, 1, environment)
  row.update(
    status="exception",
    qualified_success=False,
    qualification_failures=["trial_exception"],
    qualification=None,
    task_result=None,
    error={"type": "RuntimeError", "message": "failed", "traceback": "trace"},
  )
  path = tmp_path / "seed_000001.json"
  module._write_json_atomic(path, row, overwrite=False)

  aggregate = module._aggregate([row], [path], [1], environment)

  assert aggregate["exception_count"] == 1
  assert aggregate["certified_window_count"] == 0
  assert aggregate["certified_clean_window_count"] == 0


def test_resume_rejects_boolean_seed_identity(tmp_path, monkeypatch):
  module = _module()
  environment = _environment()
  output = tmp_path / "audit"
  output.mkdir()
  row = _completed_row(module, 1, environment)
  row["seed"] = True
  module._write_json_atomic(
    output / "seed_000001.json",
    row,
    overwrite=False,
  )
  monkeypatch.setattr(module, "_environment_identity", lambda: environment)

  with pytest.raises(ValueError, match="wrong seed"):
    module.main(["--output-dir", str(output), "--seeds", "1", "--resume"])


def test_atomic_new_json_never_overwrites(tmp_path):
  module = _module()
  path = tmp_path / "result.json"
  module._write_json_atomic(path, {"value": 1}, overwrite=False)
  with pytest.raises(FileExistsError):
    module._write_json_atomic(path, {"value": 2}, overwrite=False)
  assert json.loads(path.read_text()) == {"value": 1}


def test_unclean_certified_history_cannot_be_qualified(monkeypatch):
  module = _module()
  monkeypatch.setattr(module, "BulbScrewSimulation", _Simulation)
  monkeypatch.setattr(
    module,
    "BulbScrewExecutor",
    _executor(history=_history(clean=False)),
  )

  row = module._run_seed(4, _environment(), 10.0)

  window = row["strict_tightening"]["certified_window"]
  assert not window["all_samples_clean_contact"]
  assert not window["contract_satisfied"]
  assert not row["qualified_success"]
  assert "strict_window_contract_satisfied" in row["qualification_failures"]


def test_seed_exception_is_returned_as_resumable_failure(tmp_path, monkeypatch):
  module = _module()

  class BrokenSimulation:
    def __init__(self, *, position_seed):
      raise RuntimeError(f"seed {position_seed} failed")

  monkeypatch.setattr(module, "BulbScrewSimulation", BrokenSimulation)
  row = module._run_seed(9, _environment(), 10.0)

  assert row["status"] == "exception"
  assert not row["qualified_success"]
  assert row["qualification_failures"] == ["trial_exception"]
  assert row["error"]["type"] == "RuntimeError"
  assert row["error"]["message"] == "seed 9 failed"

  output = tmp_path / "audit"
  module._write_json_atomic(
    output / "seed_000009.json",
    row,
    overwrite=False,
  )
  monkeypatch.setattr(module, "_environment_identity", _environment)
  assert module.main(["--output-dir", str(output), "--seeds", "9", "--resume"]) == 1


def test_cli_defaults_to_evaluation_seeds_zero_through_nineteen(tmp_path):
  module = _module()
  args = module._parse_args(["--output-dir", str(tmp_path / "audit")])
  assert args.seeds == list(range(20))

  with pytest.raises(SystemExit):
    module._parse_args(["--output-dir", str(tmp_path / "audit"), "--seeds", "1", "1"])
