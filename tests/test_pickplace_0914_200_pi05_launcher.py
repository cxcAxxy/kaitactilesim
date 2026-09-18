from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT = (
  Path(__file__).resolve().parents[1]
  / "scripts/workcell/convert_pickplace_0914_200_to_pi05.py"
)
SPEC = importlib.util.spec_from_file_location("pickplace_0914_200_pi05_launcher", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
launcher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = launcher
SPEC.loader.exec_module(launcher)


def _argument(command, name):
  return command[command.index(name) + 1]


def test_layout_is_fixed_to_pi05_sibling(tmp_path):
  raw = tmp_path / "pickplace/raw/0914_200"
  output = tmp_path / "pickplace/pi05/0914_200"
  launcher.validate_layout(raw, output)
  with pytest.raises(ValueError, match="Pi0.5 output"):
    launcher.validate_layout(raw, tmp_path / "pickplace/egotouch/0914_200")


def test_auto_workers_preserve_active_training():
  assert launcher.choose_workers(
    None, affinity_cpus=16, load_1m=1.0, active_sensitive=0
  ) == 8
  assert launcher.choose_workers(
    None, affinity_cpus=16, load_1m=1.0, active_sensitive=1
  ) == 1
  assert launcher.choose_workers(
    6, affinity_cpus=16, load_1m=15.0, active_sensitive=2
  ) == 6


def test_active_process_detection_includes_training(tmp_path):
  live = tmp_path / "101"
  live.mkdir()
  (live / "stat").write_text("101 (python) D 1 2 3")
  (live / "cmdline").write_bytes(b"python\0tools/train_usb_tict_ddp.py\0")
  zombie = tmp_path / "102"
  zombie.mkdir()
  (zombie / "stat").write_text("102 (python) Z 1 2 3")
  (zombie / "cmdline").write_bytes(b"python\0scripts/train.py\0")
  assert launcher.active_sensitive_processes(tmp_path) == (
    (101, "train_usb_tict_ddp.py"),
  )


def test_command_uses_fixed_paths_and_fast_staging(tmp_path):
  command = launcher.build_command(
    workers=7,
    scratch_root=tmp_path,
    validate_only=True,
    verify_source_hash=False,
    limit=1,
    low_priority=False,
  )
  assert _argument(command, "--input-dir") == str(launcher.RAW_DIR)
  assert _argument(command, "--output-dir") == str(launcher.PI05_OUTPUT)
  assert _argument(command, "--staging-root") == str(tmp_path)
  assert _argument(command, "--fingerprint-workers") == "7"
  assert _argument(command, "--image-writer-processes") == "0"
  assert _argument(command, "--image-writer-threads") == "7"
  assert "--validate-only" in command
  assert "--verify-source-hash" not in command
  assert _argument(command, "--limit") == "1"


def test_low_priority_wraps_converter():
  command = launcher.build_command(
    workers=1,
    scratch_root=launcher.DEFAULT_SCRATCH_ROOT,
    validate_only=False,
    verify_source_hash=True,
    limit=None,
    low_priority=True,
  )
  assert command[:8] == launcher._priority_prefix(True)
  assert "--verify-source-hash" in command


@pytest.mark.parametrize(
  "arguments",
  (
    ("--workers", "0"),
    ("--limit", "0", "--validate-only"),
    ("--limit", "1"),
  ),
)
def test_invalid_cli_limits_fail(arguments):
  with pytest.raises(SystemExit):
    launcher.parse_args(arguments)
