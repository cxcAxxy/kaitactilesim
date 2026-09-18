#!/usr/bin/env python3
"""Read-only raw audit or explicitly serial headless contact robustness checks."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def plot_mechanics(output, arrays):
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  import numpy as np

  t, phases = arrays["time"], arrays["phase"]
  hold = np.isin(phases, ("insert", "bottom_out"))
  start = np.flatnonzero(hold)[0]
  angle = np.rad2deg(
    np.arccos(
      np.clip(
        (
          np.einsum(
            "nij,ij->n", arrays["relative_rotation"], arrays["relative_rotation"][start]
          )
          - 1
        )
        / 2,
        -1,
        1,
      )
    )
  )
  shift = (
    np.linalg.norm(
      arrays["relative_position"] - arrays["relative_position"][start], axis=1
    )
    * 1000
  )
  # Restrict plotted data as well as x limits: autoscaling on the entire
  # episode would hide sub-mm slip under the initial approach/retreat range.
  arrays = {key: value[hold] for key, value in arrays.items()}
  t, phases, angle, shift = t[hold], phases[hold], angle[hold], shift[hold]
  hold = np.ones(len(t), dtype=bool)
  fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
  for i, name in enumerate(("Thumb", "Index")):
    axes[0, 0].plot(t, arrays["normal"][:, i], label=name, lw=1)
    axes[0, 1].plot(t, arrays["tangent"][:, i], label=name, lw=1)
    axes[1, 0].plot(t, arrays["slip"][:, i], label=name, lw=1)
  axes[1, 1].plot(t, shift, label="Translation (mm)", lw=1)
  axes[1, 1].plot(t, angle, label="Rotation (deg)", lw=1)
  axes[2, 0].semilogy(t, np.maximum(arrays["force_error"], 1e-20), lw=0.7)
  axes[2, 1].semilogy(t, np.maximum(arrays["torque_error"], 1e-20), lw=0.7)
  labels = [
    "Pad contact normal force (N)",
    "Pad contact tangent magnitude (N)",
    "Relative tangential contact speed (mm/s)",
    "Plug motion relative to palm, from insertion start",
    "Newton force residual (N)",
    "Euler COM torque residual (N m)",
  ]
  for ax, label in zip(axes.flat, labels, strict=True):
    ax.set_title(label, fontsize=10)
    ax.grid(alpha=0.2)
    ax.set_xlim(t[hold][0] - 0.1, t[hold][-1] + 0.1)
    for phase, color in (("insert", "#aad8aa"), ("bottom_out", "#ffc27a")):
      mask = phases == phase
      ax.axvspan(t[mask][0], t[mask][-1], color=color, alpha=0.25)
    if ax.get_legend_handles_labels()[0]:
      ax.legend(fontsize=8)
  for ax in axes[-1]:
    ax.set_xlabel("Time (s)")
  fig.suptitle("USB contact mechanics | original 500 Hz samples; no force smoothing")
  fig.tight_layout()
  fig.savefig(output / "mechanics.png", dpi=170)
  fig.savefig(output / "mechanics.pdf")
  plt.close(fig)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  mode = parser.add_mutually_exclusive_group(required=True)
  mode.add_argument("--raw", type=Path)
  mode.add_argument("--suite", action="store_true")
  parser.add_argument("--case", action="append", help="Select named suite cases")
  parser.add_argument(
    "--impratio", type=float, help="Explicit diagnostic model override"
  )
  parser.add_argument("--output", required=True, type=Path)
  args = parser.parse_args()
  for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
  import numpy as np
  from kaihand_tactile_env.tasks.usb_insert.mechanics_audit import (
    ContactAudit,
    audit_raw,
    digest,
    write_report,
  )

  args.output.mkdir(parents=True, exist_ok=False)
  if args.raw:
    report, arrays = audit_raw(args.raw)
    np.savez_compressed(args.output / "trace.npz", **arrays)
    write_report(args.output / "report.json", report)
    plot_mechanics(args.output, arrays)
    print(report, flush=True)
    return 0 if report["passed"] else 1

  from kaihand_tactile_env.shared.config import default_model_path, model_fingerprint
  from kaihand_tactile_env.shared.simulation import ArmHandSimulation
  from kaihand_tactile_env.tasks.usb_insert.execution import UsbInsertionExecutor
  from kaihand_tactile_env.tasks.usb_insert.recording import controller_source_hashes
  from kaihand_tactile_env.tasks.usb_insert.setup import initialize_for_insertion

  # Nominal; four XY corners and both yaw extrema; two independent random
  # placements. Sensitivity uses exactly the same placement/noise as nominal.
  cases = [
    ("nominal", [0.0, 0.0], 0.0, 11, 1.0),
    ("corner_pp", [0.01, 0.01], -5.0, 12, 1.0),
    ("corner_pm", [0.01, -0.01], 5.0, 13, 1.0),
    ("corner_mp", [-0.01, 0.01], 5.0, 14, 1.0),
    ("corner_mm", [-0.01, -0.01], -5.0, 15, 1.0),
  ]
  for seed in (41, 42):
    rng = np.random.default_rng(seed)
    cases.append(
      (
        f"random_{seed}",
        rng.uniform(-0.01, 0.01, 2).tolist(),
        float(rng.uniform(-5.0, 5.0)),
        seed,
        1.0,
      )
    )
  cases.extend(
    [
      ("moment_minus20", [0.0, 0.0], 0.0, 11, 0.8),
      ("moment_plus20", [0.0, 0.0], 0.0, 11, 1.2),
    ]
  )
  results = []
  if args.case:
    if set(args.case) - {c[0] for c in cases}:
      parser.error("unknown suite case")
    cases = [c for c in cases if c[0] in args.case]
  for name, xy, yaw, seed, scale in cases:
    print(f"serial case {len(results) + 1}/{len(cases)}: {name}", flush=True)
    sim = ArmHandSimulation(scene="usb-insert", add_genesis_probes=True)
    if args.impratio is not None:
      if not np.isfinite(args.impratio) or args.impratio <= 0:
        parser.error("impratio must be finite and positive")
      sim.model.opt.impratio = args.impratio
    initial = initialize_for_insertion(
      sim, offset_xy_m=xy, yaw_offset_rad=np.deg2rad(yaw)
    )
    for pair in ("usb_thumb_grip", "usb_index_grip"):
      sim.model.pair(pair).friction[2:] *= scale
    audit = ContactAudit(sim.model)
    result = UsbInsertionExecutor(
      sim, audit.live_observer, precontact_noise_std_m=0.0005, noise_seed=seed
    ).execute()
    report, arrays = audit.summarize()
    report.update(
      name=name,
      initial=initial,
      noise_seed=seed,
      moment_scale=scale,
      impratio=float(sim.model.opt.impratio),
      task_success=result.success,
      failure_reason=result.failure_reason,
      elapsed_simulation_s=result.elapsed_simulation_s,
    )
    report["passed"] = bool(report["passed"] and result.success)
    np.savez_compressed(args.output / f"{name}.npz", **arrays)
    write_report(args.output / f"{name}.json", report)
    results.append(report)
    print(
      f"  success={result.success}, mechanics={report['passed']}, "
      f"failed gates={[k for k, v in report.get('gates', {}).items() if not v]}",
      flush=True,
    )
    del sim, audit, arrays
  write_report(
    args.output / "report.json",
    {
      "passed": all(r["passed"] for r in results),
      "execution": "serial, no rendering",
      "cases": results,
      "case_count": len(results),
      "scope": "Only the explicitly listed cases were run; not a success-rate estimate.",
      "model_fingerprint": model_fingerprint(default_model_path("usb-insert")),
      "controller_source_sha256": controller_source_hashes(),
      "runner_source_sha256": digest(__file__),
      "audit_source_sha256": digest(
        Path(__file__).parents[2]
        / "src/kaihand_tactile_env/tasks/usb_insert/mechanics_audit.py"
      ),
    },
  )
  return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
  raise SystemExit(main())
