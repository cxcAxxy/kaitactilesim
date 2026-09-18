from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = (
  Path(__file__).resolve().parents[1]
  / "scripts/workcell/convert_card_0914_200.py"
)
SPEC = importlib.util.spec_from_file_location("card_0914_200_launcher", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
launcher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = launcher
SPEC.loader.exec_module(launcher)


def _plan(*, total=4, pi05=2, egosteer=2):
  return launcher.ResourcePlan(
    affinity_cpus=tuple(range(12)),
    load_1m=1.0,
    active_captures=(),
    total_workers=total,
    pi05_workers=pi05,
    egosteer_workers=egosteer,
  )


def _argument(command, name):
  index = command.index(name)
  return command[index + 1]


def test_layout_accepts_only_distinct_format_directories(tmp_path):
  raw = tmp_path / "card/raw/0914_200"
  pi05 = tmp_path / "card/pi05/0914_200"
  egosteer = tmp_path / "card/egosteer/0914_200"
  launcher.validate_layout(raw, pi05, egosteer)

  with pytest.raises(ValueError, match="Pi0.5 output"):
    launcher.validate_layout(raw, egosteer, pi05)
  with pytest.raises(ValueError, match="EgoSteer output"):
    launcher.validate_layout(raw, pi05, pi05)


def test_worker_budget_shrinks_while_capture_is_active():
  idle = launcher.choose_total_workers(
    None,
    selected_conversions=2,
    affinity_cpus=12,
    load_1m=0.5,
    active_captures=0,
  )
  busy = launcher.choose_total_workers(
    None,
    selected_conversions=2,
    affinity_cpus=12,
    load_1m=8.9,
    active_captures=4,
  )
  assert idle == 8
  assert busy == 2
  assert launcher.split_workers(busy, "both") == (1, 1)
  assert launcher.choose_total_workers(
    6,
    selected_conversions=2,
    affinity_cpus=12,
    load_1m=11.0,
    active_captures=4,
  ) == 6
  with pytest.raises(ValueError, match="exceed CPU affinity"):
    launcher.choose_total_workers(
      13,
      selected_conversions=2,
      affinity_cpus=12,
      load_1m=0.0,
      active_captures=0,
    )


def test_native_math_libraries_are_single_threaded():
  assert launcher.THREAD_LIMITS == {
    "BLIS_NUM_THREADS": "1",
    "LP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
  }


def test_active_capture_detection_uses_live_script_arguments(tmp_path):
  live = tmp_path / "101"
  live.mkdir()
  (live / "stat").write_text("101 (python) R 1 2 3")
  (live / "cmdline").write_bytes(
    b"/env/bin/python\0scripts/workcell/record_dataset.py\0--episodes\029\0"
  )
  zombie = tmp_path / "102"
  zombie.mkdir()
  (zombie / "stat").write_text("102 (python) Z 1 2 3")
  (zombie / "cmdline").write_bytes(
    b"python\0scripts/workcell/record_usb_dataset.py\0"
  )
  unrelated = tmp_path / "103"
  unrelated.mkdir()
  (unrelated / "stat").write_text("103 (python) S 1 2 3")
  (unrelated / "cmdline").write_bytes(b"python\0train.py\0")

  assert launcher.active_capture_processes(tmp_path) == (
    (101, "record_dataset.py"),
  )


def test_commands_keep_formats_separate_and_apply_limits():
  conversions = launcher.build_conversions(
    _plan(),
    validate_only=True,
    verify_source_hash=True,
    limit=1,
  )
  assert [item.name for item in conversions] == ["pi05", "egosteer"]
  pi05, egosteer = conversions
  assert _argument(pi05.command, "--output-dir") == str(launcher.PI05_OUTPUT)
  assert _argument(egosteer.command, "--output-dir") == str(
    launcher.EGOSTEER_OUTPUT
  )
  assert _argument(pi05.command, "--fingerprint-workers") == "2"
  assert _argument(pi05.command, "--staging-root") == str(
    launcher.DEFAULT_SCRATCH_ROOT
  )
  assert _argument(pi05.command, "--image-writer-processes") == "0"
  assert _argument(pi05.command, "--image-writer-threads") == "2"
  assert _argument(egosteer.command, "--workers") == "2"
  for conversion in conversions:
    assert not conversion.command[:8] == launcher._priority_prefix(True)
    assert "--validate-only" in conversion.command
    assert "--verify-source-hash" in conversion.command
    assert _argument(conversion.command, "--limit") == "1"


def test_one_pi05_worker_avoids_a_process_pool():
  conversion = launcher.build_conversions(
    _plan(total=1, pi05=1, egosteer=0),
    validate_only=False,
    verify_source_hash=False,
    limit=None,
  )[0]
  assert _argument(conversion.command, "--image-writer-processes") == "0"
  assert _argument(conversion.command, "--image-writer-threads") == "1"


def test_preflight_rejects_existing_selected_output(monkeypatch, tmp_path):
  raw = tmp_path / "card/raw/0914_200"
  raw.mkdir(parents=True)
  monkeypatch.setattr(launcher, "RAW_DIR", raw)
  cwd = tmp_path / "project"
  cwd.mkdir()
  python = cwd / "python"
  converter = cwd / "converter.py"
  python.touch()
  converter.touch()
  output = tmp_path / "card/pi05/0914_200"
  output.mkdir(parents=True)
  command = (*launcher._priority_prefix(), str(python), str(converter))
  selected = launcher.Conversion("pi05", cwd, output, command)

  with pytest.raises(FileExistsError, match="pi05"):
    launcher.preflight((selected,), no_write=True)


def test_dry_run_prints_commands_without_launching(monkeypatch, tmp_path, capsys):
  root = tmp_path / "card"
  raw = root / "raw/0914_200"
  raw.mkdir(parents=True)
  pi05 = root / "pi05/0914_200"
  egosteer = root / "egosteer/0914_200"
  (root / "pi05").mkdir()
  (root / "egosteer").mkdir()
  openpi = tmp_path / "openpi"
  sim = tmp_path / "sim_code"
  openpi.mkdir()
  sim.mkdir()
  openpi_python = openpi / "python"
  sim_python = sim / "python"
  pi_converter = openpi / "convert.py"
  ego_converter = sim / "convert.py"
  for path in (openpi_python, sim_python, pi_converter, ego_converter):
    path.touch()
  monkeypatch.setattr(launcher, "RAW_DIR", raw)
  monkeypatch.setattr(launcher, "PI05_OUTPUT", pi05)
  monkeypatch.setattr(launcher, "EGOSTEER_OUTPUT", egosteer)
  monkeypatch.setattr(launcher, "OPENPI_ROOT", openpi)
  monkeypatch.setattr(launcher, "SIM_ROOT", sim)
  monkeypatch.setattr(launcher, "OPENPI_PYTHON", openpi_python)
  monkeypatch.setattr(launcher, "SIM_PYTHON", sim_python)
  monkeypatch.setattr(launcher, "PI05_CONVERTER", pi_converter)
  monkeypatch.setattr(launcher, "EGOSTEER_CONVERTER", ego_converter)
  monkeypatch.setattr(launcher, "_affinity", lambda: tuple(range(12)))
  monkeypatch.setattr(launcher, "active_capture_processes", lambda: ())
  monkeypatch.setattr(launcher.os, "getloadavg", lambda: (1.0, 1.0, 1.0))

  def unexpected(*_args, **_kwargs):
    pytest.fail("dry-run started a subprocess")

  monkeypatch.setattr(launcher.subprocess, "Popen", unexpected)
  assert launcher.main(
    [
      "--dry-run",
      "--validate-only",
      "--total-workers",
      "4",
      "--limit",
      "1",
    ]
  ) == 0
  output = capsys.readouterr().out
  assert "priority=normal total_workers=4 pi05=2 egosteer=2" in output
  assert str(pi05) in output
  assert str(egosteer) in output
  assert "/usr/bin/ionice" not in output


def test_low_priority_is_explicit_and_keeps_both_converters_bounded():
  conversions = launcher.build_conversions(
    _plan(),
    validate_only=False,
    verify_source_hash=False,
    limit=None,
    low_priority=True,
  )
  for conversion in conversions:
    assert conversion.command[:8] == launcher._priority_prefix(True)


def test_launch_starts_both_before_polling(monkeypatch, tmp_path):
  conversions = (
    launcher.Conversion("pi05", tmp_path, tmp_path / "pi", ("pi",)),
    launcher.Conversion("egosteer", tmp_path, tmp_path / "ego", ("ego",)),
  )
  created = []

  class Process:
    def __init__(self, command, **kwargs):
      self.command = command
      self.kwargs = kwargs
      self.pid = 1000 + len(created)
      self.returncode = None
      created.append(self)

    def poll(self):
      assert len(created) == 2
      self.returncode = 0
      return self.returncode

  monkeypatch.setattr(launcher.subprocess, "Popen", Process)
  monkeypatch.setattr(launcher.time, "sleep", lambda _seconds: None)
  assert launcher.launch(conversions) == 0
  assert [item.command for item in created] == [("pi",), ("ego",)]
  for item in created:
    assert item.kwargs["start_new_session"] is True
    for key, value in launcher.THREAD_LIMITS.items():
      assert item.kwargs["env"][key] == value
    assert "LEROBOT_HOME" not in item.kwargs["env"]


@pytest.mark.parametrize(
  "arguments",
  (
    ("--total-workers", "1"),
    ("--limit", "0", "--validate-only"),
    ("--limit", "1"),
  ),
)
def test_invalid_cli_limits_fail(arguments):
  with pytest.raises(SystemExit):
    launcher.parse_args(arguments)
