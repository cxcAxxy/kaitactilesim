#!/usr/bin/env python3
"""Cross-check USB split, full observation timelines and tactile between formats."""

import argparse
import json
from pathlib import Path

import numpy as np
from usb_delivery_common import save, sha256


def verify(touch, steer):
  touch, steer = Path(touch).resolve(), Path(steer).resolve()
  selection = json.loads((touch / "SOURCE_SELECTION.json").read_text())
  if selection != json.loads((steer / "SOURCE_SELECTION.json").read_text()):
    raise ValueError("format source selections differ")
  partitions = json.loads((touch / "split_manifest.json").read_text())["splits"]
  manifest = json.loads((steer / "dataset_manifest.json").read_text())
  touch_audit = json.loads((touch / "dataset_audit.json").read_text())
  steer_audit = json.loads((steer / "source_audit.json").read_text())
  for root, filename in (
    (touch, "package_validation.json"),
    (steer, "validation.json"),
  ):
    if json.loads((root / filename).read_text())["valid"] is not True:
      raise ValueError(f"failed local validation in {root}")
  if not touch_audit["all_source_audits_verified"] or not steer_audit["valid"]:
    raise ValueError("source audits must pass in both formats")
  if manifest["cameras"] != ["head"] or manifest["lowdim_dim"] != 116:
    raise ValueError("EgoSteer must be standard 116D with head-only training images")
  if manifest["tactile_included"] or not manifest["tactile_sidecars_included"]:
    raise ValueError("EgoSteer tactile must remain a separate sidecar")
  if "USB" not in manifest["instruction"]:
    raise ValueError("incorrect task instruction")
  for split in ("train", "val"):
    rows = [row for row in selection["episodes"] if row["split"] == split]
    if set(partitions["validation" if split == "val" else split]) != {
      row["session_id"] for row in rows
    }:
      raise ValueError("EgoTouch episode split mismatch")
    if set(manifest["splits"][split]["episode_indices"]) != {
      row["episode_index"] for row in rows
    }:
      raise ValueError("EgoSteer episode split mismatch")
  rows = []
  for row in selection["episodes"]:
    index, session = row["episode_index"], row["session_id"]
    descriptor = next(
      item for item in manifest["tactile_sidecars"] if item["episode_index"] == index
    )
    source = next(
      item for item in manifest["sources"] if item["episode_index"] == index
    )
    audit = next(
      item for item in touch_audit["sessions"] if item["session_id"] == session
    )
    steer_source_audit = next(
      item for item in steer_audit["episodes"] if item["episode_index"] == index
    )
    touch_source_audit = json.loads((touch / audit["source_audit_path"]).read_text())
    if (
      not touch_source_audit["valid"]
      or not steer_source_audit["valid"]
      or touch_source_audit["source"]["sha256"] != row["sha256"]
      or steer_source_audit["source_sha256"] != row["sha256"]
      or source["hdf5_sha256"] != row["sha256"]
    ):
      raise ValueError("source hash or audit mismatch")
    touch_path, steer_path = touch / audit["sidecar_path"], steer / descriptor["path"]
    if (
      sha256(touch_path) != audit["sidecar_sha256"]
      or sha256(steer_path) != descriptor["sha256"]
    ):
      raise ValueError("sidecar checksum mismatch")
    with (
      np.load(touch_path, allow_pickle=False) as a,
      np.load(steer_path, allow_pickle=False) as b,
    ):
      np.testing.assert_array_equal(
        a["timestamps_ns"], np.rint(b["pose_time_s"] * 1e9).astype(np.int64)
      )
      for name in (
        "tactile_source_index",
        "tactile_channel_mask",
        "tactile_frame_valid",
        "side_names",
        "finger_names",
      ):
        np.testing.assert_array_equal(a[name], b[name])
      difference = float(
        np.max(np.abs(a["tactile_mean"].astype(np.float64) - b["tactile_mean_n"]))
      )
      np.testing.assert_allclose(
        a["tactile_mean"], b["tactile_mean_n"], atol=1e-7, rtol=1e-6
      )
      frames = len(a["frame_names"])
      if frames != audit["frame_count"] or frames != descriptor["source_frames"]:
        raise ValueError("frame counts differ between formats")
      rows.append(
        {
          "episode_index": index,
          "split": row["split"],
          "source_sha256": row["sha256"],
          "observation_frames": frames,
          "egotouch_h50_windows": audit["window_count"],
          "egosteer_samples": source["exported_samples"],
          "maximum_tactile_mean_difference_n": difference,
          "valid": True,
        }
      )
  return {
    "schema_version": "usb-dual-format-cross-validation-v1",
    "valid": True,
    "episodes": rows,
    "episode_count": len(rows),
    "train_episodes": len(partitions["train"]),
    "validation_episodes": len(partitions["validation"]),
    "camera": "head",
    "frames": sum(row["observation_frames"] for row in rows),
    "egotouch_h50_windows": sum(row["egotouch_h50_windows"] for row in rows),
    "egosteer_samples": sum(row["egosteer_samples"] for row in rows),
    "upstream_training_executed": False,
  }


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--egotouch", type=Path, default=Path("datasets/usb_inset_0910_egotouch")
  )
  parser.add_argument(
    "--egosteer", type=Path, default=Path("datasets/usb_inset_0910_egosteer")
  )
  parser.add_argument(
    "--output",
    type=Path,
    default=Path(
      "datasets/usb_insert_0910/processing/dual_format_0910/dual_format_validation.json"
    ),
  )
  args = parser.parse_args()
  report = verify(args.egotouch, args.egosteer)
  save(args.output, report)
  print(
    json.dumps(
      {key: value for key, value in report.items() if key != "episodes"}, indent=2
    )
  )


if __name__ == "__main__":
  main()
