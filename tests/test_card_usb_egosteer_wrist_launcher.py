from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

SCRIPT = (
  Path(__file__).resolve().parents[1]
  / "scripts/workcell/convert_card_usb_0914_200_to_egosteer_wrist.py"
)
SPEC = importlib.util.spec_from_file_location("card_usb_wrist_launcher", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
launcher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = launcher
SPEC.loader.exec_module(launcher)


def _argument(command, name):
  index = command.index(name)
  return command[index + 1]


def _plan(*, active=(), total=8, card=4, usb=4):
  return launcher.ResourcePlan(
    affinity_cpus=tuple(range(16)),
    load_1m=1.0,
    active_workloads=active,
    total_workers=total,
    card_workers=card,
    usb_workers=usb,
  )


def test_adaptive_budget_is_fast_when_idle_and_conservative_when_busy():
  assert launcher.choose_total_workers(
    None,
    selected=2,
    affinity_cpus=16,
    load_1m=1.0,
    active_workloads=0,
  ) == 8
  assert launcher.choose_total_workers(
    None,
    selected=2,
    affinity_cpus=16,
    load_1m=1.0,
    active_workloads=1,
  ) == 2
  assert launcher.choose_total_workers(
    6,
    selected=2,
    affinity_cpus=16,
    load_1m=15.0,
    active_workloads=1,
  ) == 6


def test_commands_use_distinct_outputs_and_head_right_wrist(tmp_path):
  conversions = launcher.build_conversions(
    _plan(),
    validate_only=True,
    limit=1,
    verify_source_hash=False,
    scratch_root=tmp_path / "scratch",
    low_priority=False,
  )
  assert [item.name for item in conversions] == ["card", "usb"]
  card, usb = conversions
  assert card.output == launcher.CARD_OUTPUT
  assert usb.output == launcher.USB_OUTPUT
  assert card.output != usb.output
  assert _argument(card.command, "--output-dir") == str(launcher.CARD_OUTPUT)
  assert _argument(usb.command, "--output-dir") == str(launcher.USB_OUTPUT)
  assert _argument(card.command, "--workers") == "4"
  assert _argument(usb.command, "--workers") == "4"
  for conversion in conversions:
    camera_index = conversion.command.index("--cameras")
    assert conversion.command[camera_index + 1 : camera_index + 3] == (
      "head",
      "right_wrist",
    )
    assert "--validate-only" in conversion.command
    assert _argument(conversion.command, "--limit") == "1"
    assert "pi05" not in conversion.output.parts


def test_busy_plan_applies_low_io_and_cpu_priority(tmp_path):
  conversions = launcher.build_conversions(
    _plan(active=((123, "train.py"),), total=2, card=1, usb=1),
    validate_only=False,
    limit=None,
    verify_source_hash=False,
    scratch_root=tmp_path,
    low_priority=True,
  )
  for conversion in conversions:
    assert conversion.command[:8] == launcher._priority_prefix(True)


def test_sensitive_process_detection_ignores_zombies_and_self(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
  live = tmp_path / "101"
  live.mkdir()
  (live / "stat").write_text("101 (python) R 1 2 3")
  (live / "cmdline").write_bytes(b"python\0train.py\0")
  zombie = tmp_path / "102"
  zombie.mkdir()
  (zombie / "stat").write_text("102 (python) Z 1 2 3")
  (zombie / "cmdline").write_bytes(b"python\0record_dataset.py\0")
  self_dir = tmp_path / "103"
  self_dir.mkdir()
  (self_dir / "stat").write_text("103 (python) R 1 2 3")
  (self_dir / "cmdline").write_bytes(b"python\0train.py\0")
  monkeypatch.setattr(launcher.os, "getpid", lambda: 103)
  assert launcher.active_sensitive_processes(tmp_path) == ((101, "train.py"),)


def test_dry_run_does_not_create_logs_or_start_processes(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
):
  card_raw = tmp_path / "card/raw/0914_200"
  usb_raw = tmp_path / "usb_insert/raw/0914_200"
  card_raw.mkdir(parents=True)
  usb_raw.mkdir(parents=True)
  python = tmp_path / "python"
  card_converter = tmp_path / "card.py"
  usb_converter = tmp_path / "usb.py"
  for path in (python, card_converter, usb_converter):
    path.touch()
  monkeypatch.setattr(launcher, "CARD_RAW", card_raw)
  monkeypatch.setattr(launcher, "USB_RAW", usb_raw)
  monkeypatch.setattr(launcher, "CARD_OUTPUT", tmp_path / "card/egosteer/out")
  monkeypatch.setattr(launcher, "USB_OUTPUT", tmp_path / "usb_insert/egosteer/out")
  monkeypatch.setattr(launcher, "SIM_PYTHON", python)
  monkeypatch.setattr(launcher, "CARD_CONVERTER", card_converter)
  monkeypatch.setattr(launcher, "USB_CONVERTER", usb_converter)
  monkeypatch.setattr(launcher, "_affinity", lambda: tuple(range(16)))
  monkeypatch.setattr(launcher, "active_sensitive_processes", lambda: ())
  monkeypatch.setattr(launcher.os, "getloadavg", lambda: (1.0, 1.0, 1.0))

  def unexpected(*_args, **_kwargs):
    pytest.fail("dry-run started a subprocess")

  monkeypatch.setattr(launcher.subprocess, "Popen", unexpected)
  log_dir = tmp_path / "logs"
  assert launcher.main(
    [
      "--dry-run",
      "--validate-only",
      "--limit",
      "1",
      "--total-workers",
      "4",
      "--log-dir",
      str(log_dir),
    ]
  ) == 0
  assert not log_dir.exists()
  output = capsys.readouterr().out
  assert "total_workers=4 card=2 usb=2" in output
  assert "right_wrist" in output


@pytest.mark.parametrize(
  "arguments",
  (("--total-workers", "1"), ("--limit", "0", "--validate-only"), ("--limit", "1")),
)
def test_invalid_cli_limits_fail(arguments):
  with pytest.raises(SystemExit):
    launcher.parse_args(arguments)
