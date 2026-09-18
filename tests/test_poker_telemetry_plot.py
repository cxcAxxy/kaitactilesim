from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from kaihand_tactile_env.tasks.poker_draw.telemetry_plot import (
  _REQUIRED_COLUMNS,
  _read_trace,
  plot_friction_trace,
)
from PIL import Image

_FINGERS = ("index", "middle", "ring", "pinky")


def _write_trace(path: Path, *, rows: int = 5) -> None:
  phases = ("approach", "slide_card", "slide_card", "edge_hold", "edge_hold")
  with path.open("w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=_REQUIRED_COLUMNS)
    writer.writeheader()
    for index in range(rows):
      row: dict[str, object] = {
        "time_s": index * 0.01,
        "phase": phases[index],
        "card_x_m": 0.58 - index * 0.0005,
        "card_y_m": -0.16,
        "card_z_m": 0.842,
        "card_yaw_rad": index * 0.002,
        "table_normal_force_n": 0.04 + index * 0.001,
        "table_force_on_card_x_n": -0.01 * index,
        "table_force_on_card_y_n": 0.0,
        "table_force_on_card_z_n": 0.04,
      }
      for finger_index, finger in enumerate(_FINGERS):
        row.update(
          {
            f"{finger}_fn_n": 0.3 + 0.01 * finger_index,
            f"{finger}_ft_x_n": 0.01 * index,
            f"{finger}_ft_y_n": 0.002 * finger_index,
            f"{finger}_ft_n": np.hypot(0.01 * index, 0.002 * finger_index),
            f"{finger}_slip_speed_m_s": (
              "nan" if finger == "middle" and index == 2 else index * 0.0001
            ),
            f"{finger}_cumulative_slip_m": index * 0.00002,
            f"{finger}_contact": index != 0,
          }
        )
      writer.writerow(row)


def test_plot_friction_trace_writes_headless_png_from_standard_csv(
  tmp_path: Path,
) -> None:
  source = tmp_path / "trace.csv"
  output = tmp_path / "plots" / "trace.png"
  _write_trace(source)

  telemetry, phases = _read_trace(source)
  assert np.isnan(telemetry["middle_slip_speed_m_s"][2])
  assert phases[1:4] == ("slide_card", "slide_card", "edge_hold")

  result = plot_friction_trace(source, output, target_force_n=0.35)

  assert result == output
  assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
  with Image.open(output) as image:
    assert image.format == "PNG"
    assert image.width > image.height > 500


def test_plot_friction_trace_never_overwrites_existing_output(tmp_path: Path) -> None:
  source = tmp_path / "trace.csv"
  output = tmp_path / "trace.png"
  _write_trace(source)
  output.write_bytes(b"keep me")

  with pytest.raises(FileExistsError, match="refusing to overwrite"):
    plot_friction_trace(source, output, target_force_n=0.35)

  assert output.read_bytes() == b"keep me"


def test_plot_friction_trace_rejects_empty_or_incomplete_csv(tmp_path: Path) -> None:
  empty = tmp_path / "empty.csv"
  with empty.open("w", encoding="utf-8", newline="") as stream:
    csv.DictWriter(stream, fieldnames=_REQUIRED_COLUMNS).writeheader()
  with pytest.raises(ValueError, match="no data rows"):
    plot_friction_trace(empty, tmp_path / "empty.png", target_force_n=0.35)

  incomplete = tmp_path / "incomplete.csv"
  incomplete.write_text("time_s,phase\n0.0,slide_card\n", encoding="utf-8")
  with pytest.raises(ValueError, match="missing columns"):
    plot_friction_trace(incomplete, tmp_path / "incomplete.png", target_force_n=0.35)


@pytest.mark.parametrize("target", (float("nan"), float("inf"), 0.0, -0.1))
def test_plot_friction_trace_rejects_invalid_target(
  tmp_path: Path,
  target: float,
) -> None:
  source = tmp_path / "trace.csv"
  _write_trace(source)

  with pytest.raises(ValueError, match="target_force_n"):
    plot_friction_trace(source, tmp_path / "trace.png", target_force_n=target)
