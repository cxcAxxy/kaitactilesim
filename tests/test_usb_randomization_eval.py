"""Case generation and reporting only; these tests never load a robot model."""

from __future__ import annotations

import importlib.util
import math
from copy import deepcopy
from pathlib import Path

import pytest

SCRIPT = (
  Path(__file__).resolve().parents[1] / "scripts/workcell/evaluate_usb_randomization.py"
)
SPEC = importlib.util.spec_from_file_location("usb_randomization_eval", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


def test_cases_have_reproducible_independent_uniform_seeds_and_all_corners():
  cases = evaluation.generate_cases(list(evaluation.LEVELS), samples=8, seed=42)
  assert cases == evaluation.generate_cases(list(evaluation.LEVELS), samples=8, seed=42)
  assert len(cases) == 51
  assert len({c["seed"] for c in cases}) == len(cases)
  assert len({c["index"] for c in cases}) == len(cases)
  assert len({c["id"] for c in cases}) == len(cases)
  for level, (xy, yaw) in evaluation.LEVELS.items():
    subset = [case for case in cases if case["level"] == level]
    assert subset == evaluation.generate_cases([level], samples=8, seed=42)
    assert [c["kind"] for c in subset].count("uniform") == 8
    corners = [c["initialization_kwargs"] for c in subset if c["kind"] == "corner"]
    assert len(corners) == 8
    assert {(tuple(c["offset_xy_m"]), c["yaw_offset_rad"]) for c in corners} == {
      ((x, y), angle) for x in (-xy, xy) for y in (-xy, xy) for angle in (-yaw, yaw)
    }
    for case in subset:
      kwargs = case["initialization_kwargs"]
      assert kwargs["seed"] == case["seed"]
      if case["kind"] == "uniform":
        assert kwargs["xy_jitter_m"] == xy
        assert kwargs["yaw_jitter_rad"] == yaw
        assert "offset_xy_m" not in kwargs and "yaw_offset_rad" not in kwargs
      else:
        assert kwargs["xy_jitter_m"] == kwargs["yaw_jitter_rad"] == 0.0
      if case["kind"] == "nominal":
        assert kwargs["offset_xy_m"] == [0.0, 0.0]
        assert kwargs["yaw_offset_rad"] == 0.0
  changed = evaluation.generate_cases(list(evaluation.LEVELS), samples=8, seed=43)
  assert [c["seed"] for c in changed] != [c["seed"] for c in cases]


def test_uniform_seeds_are_stable_when_requesting_more_samples():
  small = evaluation.generate_cases(["medium"], samples=2, seed=3)
  larger = evaluation.generate_cases(["medium"], samples=4, seed=3)
  for first, second in zip(small, larger, strict=False):
    assert first["id"] == second["id"]
    assert first["initialization_kwargs"] == second["initialization_kwargs"]


@pytest.mark.parametrize(
  "levels,samples,seed",
  [
    ([], 1, 0),
    (["bad"], 1, 0),
    (["small", "small"], 1, 0),
    (["small"], -1, 0),
    (["small"], 1, -1),
  ],
)
def test_invalid_case_requests_are_rejected(levels, samples, seed):
  with pytest.raises(ValueError):
    evaluation.generate_cases(levels, samples, seed)


def _report(case, *, success=True, quality=True, status="completed"):
  return {
    "case": case,
    "status": status,
    "task_success": success,
    "quality_pass": quality,
    "failure_stage": None if success else "insert",
    "failure_reason": None if success else "insertion failed",
    "quality": {"violations": [] if quality else ["robot_obstacle_contact"]},
    "wall_duration_s": 25.0,
    "result": {
      "insertion": {"insertion_depth_m": 0.0119 if success else 0.004},
      "elapsed_simulation_s": 21.0,
      "peak_socket_normal_load_n": 0.75,
      "peak_axial_resistance_n": 0.72,
    },
  }


def test_uniform_rates_do_not_include_nominal_corners_or_unstarted_cases():
  cases = evaluation.generate_cases(["small"], samples=3, seed=0)
  rows = [_report(case) for case in cases[:9]]
  rows += [_report(cases[9], quality=False), _report(cases[10], success=False)]
  summary = evaluation.summarize_cases(rows, cases)
  groups = summary["by_level"]["small"]
  assert groups["nominal"]["task_success_rate"] == 1.0
  assert groups["corner"]["task_success_rate"] == 1.0
  assert groups["uniform"]["task_success_rate"] == 0.5
  assert groups["uniform"]["quality_pass_rate"] == 0.5
  assert groups["uniform"]["qualified_success_rate"] == 0.0
  assert groups["uniform"]["unstarted_count"] == 1
  assert groups["uniform"]["failure_stages"] == {"insert": 1}
  assert groups["uniform"]["quality_failures"] == {"robot_obstacle_contact": 1}
  assert groups["uniform"]["insertion_depth_m"]["mean"] == pytest.approx(
    (0.0119 + 0.004) / 2
  )


def test_interruption_is_visible_but_exception_counts_as_failed_evaluation():
  cases = evaluation.generate_cases(["large"], samples=2, seed=0)[-2:]
  rows = [
    _report(cases[0], success=False, quality=False, status="interrupted"),
    _report(cases[1], success=False, quality=False, status="exception"),
  ]
  rows[1]["result"] = None
  original = deepcopy(rows)
  uniform = evaluation.summarize_cases(rows, cases)["by_level"]["large"]["uniform"]
  assert rows == original
  assert uniform["reported_count"] == 2
  assert (
    uniform["evaluated_count"]
    == uniform["exception_count"]
    == uniform["interrupted_count"]
    == 1
  )
  assert uniform["task_success_rate"] == 0.0
  assert uniform["insertion_depth_m"]["count"] == 0
  assert uniform["insertion_depth_m"]["mean"] is None


def test_empty_summary_reports_no_rate_and_excludes_nonfinite_metrics():
  cases = evaluation.generate_cases(["small"], samples=0, seed=0)
  uniform = evaluation.summarize_cases([], cases)["by_level"]["small"]["uniform"]
  assert uniform["task_success_rate"] is None
  assert uniform["planned_count"] == 0
  assert evaluation._numeric_summary([None, math.inf, math.nan, 1.0]) == {
    "count": 1,
    "minimum": 1.0,
    "mean": 1.0,
    "maximum": 1.0,
  }
