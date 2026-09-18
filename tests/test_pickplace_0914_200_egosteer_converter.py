from __future__ import annotations

import sys
from pathlib import Path

import pytest


WORKCELL = Path(__file__).parents[1] / "scripts" / "workcell"
sys.path.insert(0, str(WORKCELL))

import convert_pickplace_0914_200_to_egosteer as converter  # noqa: E402


def test_layout_is_fixed_to_egosteer_sibling(tmp_path: Path) -> None:
  raw = tmp_path / "pickplace/raw/0914_200"
  output = tmp_path / "pickplace/egosteer/0914_200"
  converter.validate_layout(raw, output)
  with pytest.raises(ValueError, match="EgoSteer output"):
    converter.validate_layout(raw, tmp_path / "pickplace/pi05/0914_200")


def test_default_uses_all_episodes_for_training() -> None:
  args = converter.parse_args(())

  assert args.val_fraction == 0.0
  assert args.dataset_name == "pickplace_0914_200"
  assert args.verify_source_hash is False


def test_auto_workers_preserve_active_training() -> None:
  assert converter.choose_workers(
    None, affinity_cpus=16, load_1m=1.0, active_sensitive=0
  ) == 8
  assert converter.choose_workers(
    None, affinity_cpus=16, load_1m=1.0, active_sensitive=1
  ) == 1
  assert converter.choose_workers(
    6, affinity_cpus=16, load_1m=15.0, active_sensitive=2
  ) == 6


def test_active_process_detection_ignores_zombies(tmp_path: Path) -> None:
  live = tmp_path / "101"
  live.mkdir()
  (live / "stat").write_text("101 (python) D 1 2 3")
  (live / "cmdline").write_bytes(b"python\0tools/train_usb_tict_ddp.py\0")
  zombie = tmp_path / "102"
  zombie.mkdir()
  (zombie / "stat").write_text("102 (python) Z 1 2 3")
  (zombie / "cmdline").write_bytes(b"python\0scripts/train.py\0")

  assert converter.active_sensitive_processes(tmp_path) == (
    (101, "train_usb_tict_ddp.py"),
  )


def test_tree_signature_detects_publish_size_difference(tmp_path: Path) -> None:
  source = tmp_path / "source"
  destination = tmp_path / "destination"
  source.mkdir()
  destination.mkdir()
  (source / "train").mkdir()
  (destination / "train").mkdir()
  (source / "train/shard.tar").write_bytes(b"abc")
  (destination / "train/shard.tar").write_bytes(b"abc")
  assert converter._tree_signature(source) == converter._tree_signature(destination)

  (destination / "train/shard.tar").write_bytes(b"abcd")
  assert converter._tree_signature(source) != converter._tree_signature(destination)


@pytest.mark.parametrize(
  "arguments",
  (
    ("--workers", "0"),
    ("--jpeg-quality", "101"),
    ("--val-fraction", "1"),
    ("--limit", "1"),
  ),
)
def test_invalid_cli_values_fail(arguments: tuple[str, ...]) -> None:
  with pytest.raises(SystemExit):
    converter.parse_args(arguments)
