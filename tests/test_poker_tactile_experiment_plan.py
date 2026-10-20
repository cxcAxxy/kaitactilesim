from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("poker_experiment", ROOT / "scripts/evaluate/evaluate_poker_tactile_experiment.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_requested_five_models_use_five_fixed_seeds(tmp_path):
    plan = module.make_plan(tmp_path / "fresh")
    assert plan["seeds"] == [0, 1, 2, 3, 4]
    assert plan["planned_trials"] == 25
    assert [row["checkpoint_step"] for row in plan["groups"]] == ["best", "10000", "15000", "20000", "24999"]
    assert [row["prediction_horizon"] for row in plan["groups"]] == [30, 32, 32, 32, 32]
    assert all(row["execute_steps"] == 5 for row in plan["groups"])
    assert len({row["evaluation_path"] for row in plan["groups"]}) == 5


def test_commands_preserve_native_horizons_and_stage_protocol(tmp_path):
    plan = module.make_plan(tmp_path / "fresh")
    for group in plan["groups"]:
        command = module.evaluation_command(group, plan, 3600)
        assert command[command.index("--seeds") + 1:command.index("--video-count")] == ["0", "1", "2", "3", "4"]
        assert command[command.index("--max-sim-seconds") + 1] == "90"
        assert command[command.index("--staged-evaluation-version") + 1] == "staged_v1"
        assert command[command.index("--action-interpolation") + 1] == "zoh"
        assert "--save-raw" not in command
    prepare = module.prepare_command(plan["groups"][0])
    assert str(module.BASE) in prepare and str(module.CPM) in prepare
    assert "--guidance-scale" in prepare
