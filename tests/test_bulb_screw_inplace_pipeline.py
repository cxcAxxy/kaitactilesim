"""Model-free contracts for the independently named, initially threaded task."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

from kaihand_tactile_env.pipeline.catalog import EVALUATION_RUNNERS, TASKS
from kaihand_tactile_env.pipeline.collection import COLLECTION_SPECS
from kaihand_tactile_env.pipeline.conversion import ADAPTERS as CONVERSION_ADAPTERS
from kaihand_tactile_env.pipeline.deployment import adapter_for
from kaihand_tactile_env.shared.policy_tasks import TASK_INSTRUCTIONS, TaskPolicyAdapter
from kaihand_tactile_env.shared.task_variants import (
  VARIANT_SUCCESS_FIELDS,
  VARIANT_TASKS,
  create_variant_expert,
  validate_variant_outcome,
)

TASK = "bulb-screw-inplace"
ROOT = Path(__file__).resolve().parents[1]


def _script(filename):
  spec = importlib.util.spec_from_file_location(
    "inplace_pipeline_" + filename.replace(".", "_"),
    ROOT / "scripts/workcell" / filename,
  )
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def test_new_task_is_additional_and_old_pickup_task_remains_independent(tmp_path):
  assert TASK in TASKS and TASK in VARIANT_TASKS
  assert TASKS.index(TASK) > TASKS.index("bulb-screw")
  original = COLLECTION_SPECS["bulb-screw"]
  assert original.backend == "refresh_light_bulb_example.py"
  assert "Pick up the light bulb" in TASK_INSTRUCTIONS["bulb-screw"]
  assert "already aligned and initially threaded" in TASK_INSTRUCTIONS[TASK]
  assert "Pick up" not in TASK_INSTRUCTIONS[TASK]
  spec = COLLECTION_SPECS[TASK]
  command = spec.build_command(ROOT, sys.executable, tmp_path / "fresh", 13)
  assert command[1].endswith("record_task_variant.py")
  assert command[command.index("--task") + 1] == TASK
  assert command[command.index("--seed") + 1] == "13"
  assert spec.settings["initial_state"] == "initially-threaded-unseated-unlit"
  assert "home_return_verified" not in VARIANT_SUCCESS_FIELDS[TASK]
  for output_format in ("pi05", "lerobot-v3"):
    assert (TASK, output_format) in CONVERSION_ADAPTERS
  assert EVALUATION_RUNNERS[(TASK, "pi05")] == "evaluate_shared_task_policy_batch.py"
  assert adapter_for(TASK, "pi05").manifest_schemas == (
    "shared_task_pi05_deployment_v2",
  )


def test_success_is_physical_not_light_state_stroke_count_or_home_only():
  outcome = {
    "task": TASK, "success": True,
    **dict.fromkeys(VARIANT_SUCCESS_FIELDS[TASK], True),
    "home_return_verified": False,
  }
  validate_variant_outcome(TASK, outcome)
  for gate in VARIANT_SUCCESS_FIELDS[TASK]:
    with pytest.raises(ValueError, match=gate):
      validate_variant_outcome(TASK, {**outcome, gate: False, "bulb_lit": True,
                                      "strokes": 5, "home_return_verified": True})
  with pytest.raises(ValueError, match="mismatched task outcome"):
    validate_variant_outcome(TASK, {**outcome, "task": "bulb-screw"})
  unbound = dict(outcome)
  unbound.pop("task")
  with pytest.raises(ValueError, match="explicit independent task identity"):
    validate_variant_outcome(TASK, unbound)


def test_variant_factory_lazily_uses_only_new_simulation_and_executor(monkeypatch):
  task_module = types.ModuleType("kaihand_tactile_env.tasks.bulb_screw_inplace.task")
  execution_module = types.ModuleType(
    "kaihand_tactile_env.tasks.bulb_screw_inplace.execution"
  )

  class Simulation:
    def __init__(self, *, position_seed):
      self.seed = position_seed

  class Executor:
    def __init__(self, simulation):
      self.sim = simulation

  task_module.BulbScrewInplaceSimulation = Simulation
  execution_module.BulbScrewInplaceExecutor = Executor
  monkeypatch.setitem(sys.modules, task_module.__name__, task_module)
  monkeypatch.setitem(sys.modules, execution_module.__name__, execution_module)
  sim, executor = create_variant_expert(TASK, 23)
  assert isinstance(sim, Simulation) and sim.seed == 23
  assert isinstance(executor, Executor) and executor.sim is sim


@pytest.mark.parametrize("flags,expected", (
  ({}, "verify_initially_threaded"),
  ({"initially_threaded_verified": True}, "approach_and_grasp_inplace"),
  ({"grasp_verified": True}, "screw_clockwise_and_regrasp"),
  ({"seated": True}, "verify_loaded_tightening"),
  ({"tightening_verified": True}, "release_and_verify_stable"),
  ({"released_stably": True}, "cleanup_return_home"),
))
def test_policy_stage_has_no_tabletop_pickup_or_transport_phase(flags, expected):
  adapter = object.__new__(TaskPolicyAdapter)
  adapter.task = TASK
  adapter._success = False
  adapter.state = types.SimpleNamespace(**{
    "initially_threaded_verified": False, "grasp_verified": False,
    "seated": False, "tightening_verified": False, "released_stably": False,
    **flags,
  })
  assert adapter.stage() == expected


def test_prepare_serve_batch_bind_the_distinct_task_prompt_and_source_package():
  prepare = _script("prepare_shared_task_pi05_deployment.py")
  serve = _script("serve_shared_task_pi05_policy.py")
  batch = _script("evaluate_shared_task_policy_batch.py")
  assert prepare.TASK_INSTRUCTIONS[TASK] == TASK_INSTRUCTIONS[TASK]
  assert serve.TASK_INSTRUCTIONS[TASK] == TASK_INSTRUCTIONS[TASK]
  assert TASK in batch.TASKS
  assert prepare.TASK_PACKAGE_DIRS[TASK] == ("bulb_screw_inplace", "bulb_screw")
  sources = prepare.evaluation_source_paths(TASK)
  assert ROOT / "src/kaihand_tactile_env/shared/task_variants.py" in sources


def test_new_task_rejects_historical_manifest_without_own_training_binding():
  from kaihand_tactile_env.shared.pi05_deployment_integrity import (
    validate_shared_deployment_semantics,
  )

  with pytest.raises(RuntimeError, match="requires its own v2 deployment provenance"):
    validate_shared_deployment_semantics({
      "schema": "shared_task_pi05_deployment_v1", "task": TASK,
    })

