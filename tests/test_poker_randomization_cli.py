"""Argument/success/observer tests without loading a robot or renderer."""

import importlib.util
import json
import signal
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_PATH = (
  Path(__file__).resolve().parents[1]
  / "scripts/workcell/validate_poker_randomization.py"
)
_SPEC = importlib.util.spec_from_file_location("randomization_cli_tests", _PATH)
cli = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cli)


def test_default_screen_is_small_and_position_only():
  args = cli._parse_args(["--output-dir", "unused"])
  assert args.xy_mm == 2
  assert args.yaw_deg == 0
  assert args.seeds == [0, 1, 2]
  assert args.fixed_offset_mm_deg is None
  assert args.precontact_noise is False


def test_precontact_defaults_expand_only_the_opt_in_screen():
  args = cli._parse_args(["--output-dir", "unused", "--precontact-noise"])
  assert args.xy_mm == 4
  assert args.yaw_deg == 0.5
  assert np.isclose(args.precontact_noise_std_deg, 0.03)
  assert args.seeds == [0, 1, 2]


def test_precontact_can_keep_card_fixed_and_disable_noise_for_paired_control():
  args = cli._parse_args(
    [
      "--output-dir",
      "unused",
      "--precontact-noise",
      "--xy-mm",
      "0",
      "--yaw-deg",
      "0",
      "--precontact-noise-std-deg",
      "0",
      "--seeds",
      "3",
    ]
  )
  assert args.xy_mm == args.yaw_deg == args.precontact_noise_std_deg == 0
  assert args.seeds == [3]


@pytest.mark.parametrize(
  "extra",
  [
    ["--precontact-noise-std-deg", ".03"],
    ["--precontact-noise", "--precontact-noise-std-deg", "nan"],
    ["--precontact-noise", "--precontact-noise-std-deg", "inf"],
    ["--precontact-noise", "--precontact-noise-std-deg", "-.01"],
    ["--precontact-noise", "--precontact-noise-std-deg", ".051"],
  ],
)
def test_noise_options_fail_before_model_creation(extra):
  with pytest.raises(SystemExit):
    cli._parse_args(["--output-dir", "unused", *extra])


@pytest.mark.parametrize(
  "extra",
  [
    ["--xy-mm", "5.1"],
    ["--xy-mm", "nan"],
    ["--yaw-deg", "1.1"],
    ["--yaw-deg", "-1"],
    ["--seeds", "0", "0"],
    ["--seeds", "-1"],
    ["--seeds", "0", "1", "2", "3", "4", "5", "6"],
    ["--trial-wall-limit", "301"],
    ["--trial-wall-limit", "nan"],
    ["--fixed-offset-mm-deg", "3", "0", "0"],
    ["--fixed-offset-mm-deg", "0", "0", ".1"],
  ],
)
def test_rejects_unbounded_or_ambiguous_screen(extra):
  with pytest.raises(SystemExit):
    cli._parse_args(["--output-dir", "unused", *extra])


def test_fixed_validation_offsets_are_explicit():
  args = cli._parse_args(
    [
      "--output-dir",
      "unused",
      "--xy-mm",
      "2",
      "--yaw-deg",
      ".5",
      "--fixed-offset-mm-deg",
      "-2",
      "2",
      "-.5",
    ]
  )
  assert args.fixed_offset_mm_deg == [[-2, 2, -0.5]]


@pytest.mark.parametrize(
  "gate",
  [
    "success",
    "slide_press_control_qualified",
    "target_reached",
    "held_at_edge",
    "full_slide_qualified",
    "completed",
  ],
)
def test_every_original_success_gate_is_mandatory(gate):
  result = SimpleNamespace(success=True, slide_press_control_qualified=True)
  executor = SimpleNamespace(
    edge_outcome={
      "target_reached": True,
      "held_at_edge": True,
      "full_slide_qualified": True,
    },
    handoff_outcome={"completed": True},
  )
  assert cli._accepted(result, executor)
  if gate in ("success", "slide_press_control_qualified"):
    setattr(result, gate, False)
  elif gate == "completed":
    executor.handoff_outcome[gate] = False
  else:
    executor.edge_outcome[gate] = False
  assert not cli._accepted(result, executor)


def test_json_does_not_replace_prior_results(tmp_path):
  path = tmp_path / "trial.json"
  cli._json(path, {"values": np.array([1.0, 2.0]), "success": False})
  assert json.loads(path.read_text())["values"] == [1.0, 2.0]
  with pytest.raises(FileExistsError):
    cli._json(path, {"success": True})
  assert json.loads(path.read_text())["success"] is False


def test_precontact_screen_selects_isolated_factory_without_robot(
  tmp_path, monkeypatch
):
  invoked = []

  @contextmanager
  def precontact_factory(path):
    invoked.append(path)
    yield object(), {"fixed_physics": True}

  def forbidden(*args, **kwargs):
    raise AssertionError("the original factory must not create a robot")

  def run_case(sim, args, index, seed, fixed):
    assert args.precontact_noise
    assert fixed is None
    return {"trial_index": index, "seed": seed, "success": True}

  monkeypatch.setattr(cli, "middle_force_simulation", forbidden)
  monkeypatch.setattr(cli, "precontact_force_simulation", precontact_factory)
  monkeypatch.setattr(cli, "model_fingerprint", lambda path: "fake-model")
  monkeypatch.setattr(cli, "_run_case", run_case)
  directory = tmp_path / "screen"
  assert (
    cli.main(
      [
        "--output-dir",
        str(directory),
        "--precontact-noise",
        "--seeds",
        "7",
      ]
    )
    == 0
  )
  assert len(invoked) == 1
  result = json.loads((directory / "summary.json").read_text())
  assert result["preset"] == cli.PRECONTACT_PRESET
  assert result["action_noise"] == cli.precontact_noise_settings(
    cli.DEFAULT_PRECONTACT_STD_RAD
  )
  assert result["observation_noise"] is None
  assert result["passed_count"] == result["trial_count"] == 1
  assert result["rendering"] is False
  with pytest.raises(FileExistsError):
    cli.main(["--output-dir", str(directory), "--precontact-noise"])
  assert len(invoked) == 1


@pytest.mark.parametrize("failed", [False, True, "interrupt"])
def test_precontact_case_wires_seed_and_saves_executed_trace_without_robot(
  tmp_path, monkeypatch, failed
):
  @dataclass
  class Result:
    success: bool = True
    slide_press_control_qualified: bool = True

  @dataclass
  class Stability:
    stable_seconds: float = 0.1

  events = []
  sim = SimpleNamespace(
    data=SimpleNamespace(time=0.0),
    observation_time=0.0,
    drive_limit_n=None,
    current_pose_matrix=lambda side: (np.zeros(3), np.eye(3)),
    object_pose=lambda name: np.array([0.58, -0.16, 0.84175, 1.0, 0.0, 0.0, 0.0]),
    drive_state=lambda: {},
  )
  sim.configure_precontact_noise = lambda **kwargs: events.append(("configure", kwargs))
  sim.precontact_noise_metadata = lambda: {"configured": True, "seed": 7}
  sim.precontact_noise_trace = lambda: {"time_s": np.array([0.0, 0.002])}

  def reset(*args, **kwargs):
    events.append(("reset", None))
    return {"sampled_offset_xy_m": [0.0, 0.0], "sampled_yaw_offset_rad": 0.0}

  def plan():
    events.append(("plan", None))
    return object()

  class Executor:
    def __init__(self, simulation, observer, acceptance_policy="strict-force-v1"):
      self.acceptance_policy = acceptance_policy
      assert simulation is sim
      self.edge_outcome = {
        "target_reached": True,
        "held_at_edge": True,
        "full_slide_qualified": True,
      }
      self.handoff_outcome = {"completed": True}

    def execute(self, plan):
      events.append(("execute", None))
      sim.data.time = 0.004
      if failed == "interrupt":
        raise KeyboardInterrupt("requested diagnostic stop")
      if failed:
        raise RuntimeError("injected failure, never resample")
      return Result()

    def refresh_terminal_result(self, result):
      return result

    def control_metadata(self):
      return {"physics_unchanged": True}

  monkeypatch.setattr(cli, "reset_randomized_card", reset)
  monkeypatch.setattr(
    cli, "PokerDrawPlanner", lambda simulation: SimpleNamespace(plan=plan)
  )
  monkeypatch.setattr(cli, "MidForcePokerExecutor", Executor)
  monkeypatch.setattr(cli, "wait_until_object_stable", lambda *a, **kw: Stability())
  args = cli._parse_args(["--output-dir", str(tmp_path), "--precontact-noise"])
  if failed == "interrupt":
    with pytest.raises(KeyboardInterrupt, match="requested diagnostic stop"):
      cli._run_case(sim, args, 1, 7, None)
    row = json.loads((tmp_path / "trial_001.json").read_text())
    assert row["interrupted"] is True
  else:
    row = cli._run_case(sim, args, 1, 7, None)
  assert [name for name, _ in events] == ["reset", "plan", "configure", "execute"]
  assert events[2][1] == {"seed": 7, "std_rad": cli.DEFAULT_PRECONTACT_STD_RAD}
  assert row["success"] is (not failed)
  assert row["precontact_noise"]["seed"] == 7
  with np.load(row["precontact_trace_npz"], allow_pickle=False) as trace:
    np.testing.assert_array_equal(trace["time_s"], [0.0, 0.002])
  assert json.loads((tmp_path / "trial_001.json").read_text())["success"] is (
    not failed
  )


def test_diagnostic_sigterm_guard_restores_prior_handler():
  previous = signal.getsignal(signal.SIGTERM)
  with pytest.raises(KeyboardInterrupt, match="terminated before completion"):
    with cli._termination_guard():
      signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
  assert signal.getsignal(signal.SIGTERM) is previous
