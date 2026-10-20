#!/usr/bin/env python3
"""Run CPM/FTP-1 through the native poker closed-loop evaluator."""
# ruff: noqa: E402
from __future__ import annotations

import json
import os
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("KAIHAND_RENDER_BACKEND", "hardware")
vendor = ROOT / "scripts/collect/nvidia_egl_vendor.json"
if vendor.is_file():
  os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", str(vendor))

from run_poker_pi05_policy import parse_args, run
from tactile_policy_binding import PokerTactileBinding

if __name__ == "__main__":
  def interrupted(signum, frame):
    raise KeyboardInterrupt(f"received signal {signum}")
  signal.signal(signal.SIGTERM, interrupted)
  args = parse_args()
  try:
    run(args, policy_binding=PokerTactileBinding())
  except Exception:
    # Native guards are evaluated failed trials, not subprocess/IO failures.
    summary = args.output_dir / "summary.json"
    if summary.is_file():
      report = json.loads(summary.read_text())
      if report.get("valid_trial") is True and report.get("stop_reason") == "guard":
        sys.exit(0)
    raise
