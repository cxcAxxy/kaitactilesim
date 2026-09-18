#!/usr/bin/env python3
"""Export the frozen successful poker selection from archived render-time poses.

Serial, read-only HDF5 conversion. No MuJoCo model loading or simulation.
The upstream 116D WebDataset remains unchanged; tactile is a separate NPZ.
"""

import argparse
import io
import json
import os
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
from convert_to_egosteer import (
  SITE_FROM_EGOSTEER_WRIST,
  EpisodeConversion,
  ShardSummary,
  SourceEpisode,
  _add_tar_member,
  _camera_from_world_cv,
  _dataset_manifest,
  _intrinsic_vector,
  _jpeg_bytes,
  _json_bytes,
  _npy_bytes,
  _source_manifest,
  _strict_30hz_prefix,
)
from kaihand_tactile_env.shared.egosteer_archive import (
  archived_motion,
  observation_archive,
  text,
  validate_observation_archive,
  validate_poker_outcome,
)
from poker_delivery_common import attach_quality, save, selection, sha256
from validate_egosteer_dataset import DatasetValidator

INSTRUCTION = (
  "Slide the face-down card toward the table edge, pinch it between the fingers "
  "and thumb, lift it, and turn its face toward the robot to look at it."
)


def describe_source(row):
  path = Path(row["source"])
  if sha256(path) != row["sha256"]:
    raise ValueError(f"frozen source changed: {path}")
  stat = path.stat()
  with h5py.File(path, "r") as file:
    validate_poker_outcome(json.loads(file.attrs["metadata_json"]),
                           json.loads(file.attrs["outcome_json"]))
    return SourceEpisode(path, path.with_suffix(".json"), row["episode_index"],
                         row["sha256"], sha256(path.with_suffix(".json")),
                         stat.st_size, stat.st_mtime_ns,
                         Path(text(file.attrs["model_path"])),
                         text(file.attrs["model_sha256"]))


def rgb_frames(dataset, count, block_size=16):
  """Bounded sequential reads: at most 16 RGB frames buffered, never an episode."""
  if not 0 <= count <= len(dataset) or block_size < 1:
    raise ValueError("invalid RGB read bounds")
  for start in range(0, count, block_size):
    block = dataset[start:min(count, start + block_size)]
    for offset, rgb in enumerate(block):
      yield start + offset, rgb


def convert(source, split, archive, root, dataset_name):
  with h5py.File(source.path, "r") as file:
    camera = file["cameras/head"]
    acquisition = np.asarray(camera["timestamp"])
    prefix, error = _strict_30hz_prefix(acquisition, int(file.attrs["physics_hz"]))
    if prefix < len(acquisition) - 1:
      raise ValueError("internal off-grid camera frame; refusing silent truncation")
    pose_times, wrists, hands = archived_motion(file, SITE_FROM_EGOSTEER_WRIST)
    intrinsic = _intrinsic_vector(np.asarray(camera["intrinsic"]))
    height, width = camera["rgb"].shape[1:3]
    samples = prefix - 1
    camera_poses = camera["world_from_camera"][:]
    for index, rgb in rgb_frames(camera["rgb"], samples):
      w2c = _camera_from_world_cv(camera_poses[index])
      lowdim = np.concatenate((wrists[index], hands[index], wrists[index + 1],
                               hands[index + 1], w2c.reshape(-1), intrinsic)).astype(np.float32)
      if lowdim.shape != (116,) or not np.isfinite(lowdim).all():
        raise ValueError("invalid unified lowdim")
      key = f"episode_{source.episode_index:06d}_frame_{index:06d}"
      meta = {"cameras": ["head"], "dataset_name": dataset_name,
              "episode_index": source.episode_index, "instruction": INSTRUCTION,
              "instruction_num": 1, "source_control": "scripted_simulation",
              "source_frame_index": index, "pose_time_s": float(pose_times[index]),
              "next_pose_time_s": float(pose_times[index + 1]),
              "acquisition_time_s": float(acquisition[index])}
      _add_tar_member(archive, key + ".image.jpg", _jpeg_bytes(rgb, 95))
      _add_tar_member(archive, key + ".lowdim.npy", _npy_bytes(lowdim))
      _add_tar_member(archive, key + ".meta.json", _json_bytes(meta))
    values = observation_archive(file, episode_index=source.episode_index, prefix=prefix,
                                 pose_times=pose_times, source_sha256=source.hdf5_sha256,
                                 include_tactile=True)
    validate_observation_archive(values, episode_index=source.episode_index,
                                 samples=samples, source_frames=len(acquisition), prefix=prefix,
                                 source_sha256=source.hdf5_sha256, include_tactile=True)
    if not values["tactile_frame_valid"].all():
      raise ValueError("successful poker release requires valid tactile on every camera frame")
    relative = f"tactile_sidecars/episode_{source.episode_index:06d}.npz"
    path = root / relative
    with path.open("xb") as stream:
      np.savez_compressed(stream, **values)
    conversion = EpisodeConversion(source.episode_index, split, len(acquisition), prefix,
                                    samples, len(acquisition) - prefix, error, width, height)
    sidecar = {"episode_index": source.episode_index, "path": relative,
               "sha256": sha256(path), "size_bytes": path.stat().st_size,
               "source_frames": len(acquisition), "training_samples": samples}
  stat = source.path.stat()
  if (stat.st_size, stat.st_mtime_ns) != (source.size_bytes, source.mtime_ns):
    raise ValueError("raw source changed during export")
  return conversion, sidecar


def reference_lowdim(camera, index):
  """Independent direct SE3 indexing, not the converter's vectorized assembly."""
  result = np.zeros(116, dtype=np.float32)
  for frame, offset in ((index, 0), (index + 1, 48)):
    wrists = camera["world_from_wrist"][frame]
    tips = camera["world_from_fingertip"][frame]
    for hand, side in enumerate(("left", "right")):
      result[offset + hand * 3:offset + hand * 3 + 3] = wrists[hand, :3, 3]
      rotation = wrists[hand, :3, :3] @ SITE_FROM_EGOSTEER_WRIST[side]
      start = offset + 6 + hand * 6
      result[start:start + 3] = rotation[:, 0]
      result[start + 3:start + 6] = rotation[:, 1]
      result[offset + 18 + hand * 15:offset + 33 + hand * 15] = tips[hand, :, :3, 3].reshape(-1)
  c2w = camera["world_from_camera"][index] @ np.diag([1., -1., -1., 1.])
  result[96:112] = np.linalg.inv(c2w).reshape(-1)
  intrinsic = camera["intrinsic"][:]
  result[112:] = [intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]]
  return result


def audit_source(root, sources, conversions, shards, sidecars):
  """Check every emitted image, pose, next-state target and tactile taxel."""
  reports = []
  for source in sources:
    index = source.episode_index
    conversion = next(item for item in conversions if item.episode_index == index)
    shard = next(item for items in shards.values() for item in items if index in item.episode_indices)
    descriptor = next(item for item in sidecars if item["episode_index"] == index)
    if sha256(root / descriptor["path"]) != descriptor["sha256"]:
      raise ValueError("tactile sidecar digest mismatch")
    with h5py.File(source.path, "r") as file, np.load(root / descriptor["path"], allow_pickle=False) as sidecar:
      validate_observation_archive(sidecar, episode_index=index,
                                    samples=conversion.exported_samples,
                                    source_frames=conversion.source_frames,
                                    prefix=conversion.strict_prefix_frames,
                                    source_sha256=source.hdf5_sha256, include_tactile=True)
      camera = file["cameras/head"]
      for target, raw in (("acquisition_time_s", "timestamp"), ("pose_time_s", "pose_timestamp"),
                           ("state_index", "state_index")):
        if not np.array_equal(sidecar[target], camera[raw][:]):
          raise ValueError(f"sidecar source timestamp/index mismatch: {target}")
      force = file["tactile_contact_force"]
      if not np.array_equal(sidecar["tactile_source_timestamps_s"], force["timestamp"][:]):
        raise ValueError("sidecar force clock mismatch")
      names = [text(value) for value in force["link_names"][:]]
      expected = [f"hand_{hand}_{finger}_{'link6' if finger == 'thumb' else 'link4'}"
                  for hand in ("l", "r") for finger in ("thumb", "index", "middle", "ring", "pinky")]
      order = [names.index(name) for name in expected]
      normal = force["normal_taxel_force_n"][:]
      tangent = force["tangent_taxel_force_n"][:]
      taxels = sidecar["tactile_taxel_force_n"]
      for frame, raw_index in enumerate(sidecar["tactile_source_index"]):
        if not sidecar["tactile_frame_valid"][frame]:
          raise ValueError("successful poker release requires valid tactile on every camera frame")
        if (not np.array_equal(taxels[frame, ..., 0].reshape(10, 7, 5), normal[raw_index, order])
            or not np.array_equal(taxels[frame, ..., 1:].reshape(10, 7, 5, 2), tangent[raw_index, order])):
          raise ValueError(f"tactile source mismatch episode={index} frame={frame}")
      with tarfile.open(root / shard.path, "r:") as archive:
        for frame in range(conversion.exported_samples):
          key = f"episode_{index:06d}_frame_{frame:06d}"
          lowdim = np.load(io.BytesIO(archive.extractfile(key + ".lowdim.npy").read()), allow_pickle=False)
          if not np.allclose(lowdim, reference_lowdim(camera, frame), atol=1e-7, rtol=0):
            raise ValueError(f"raw pose/camera/action mismatch {key}")
          image = archive.extractfile(key + ".image.jpg").read()
          if image != _jpeg_bytes(camera["rgb"][frame], 95):
            raise ValueError(f"raw image encoding mismatch {key}")
      if sha256(source.path) != source.hdf5_sha256:
        raise ValueError("raw digest changed")
      reports.append({"episode_index": index, "valid": True,
                      "source_sha256": source.hdf5_sha256,
                      "samples_checked": conversion.exported_samples,
                      "tactile_frames_checked": conversion.source_frames})
    print(f"EgoSteer source audit {index}: passed", flush=True)
  return {"valid": True, "episodes": reports, "scope": "all WDS samples and NPZ tactile taxels against raw HDF5; no external training executed"}


def verify_release(root, sources, conversions, shards, sidecars, verification):
  if verification not in ("schema", "source"):
    raise ValueError("unknown verification mode")
  print("EgoSteer: one standard schema validation", flush=True)
  validation = DatasetValidator(root).validate()
  save(root / "validation.json", validation)
  if not validation["valid"]:
    raise ValueError(f"validation failed: {root / 'validation.json'}")
  if verification == "source":
    report = audit_source(root, sources, conversions, shards, sidecars)
    save(root / "source_audit.json", report)
  else:
    print("EgoSteer: reusing existing Cleaning reports; no second raw/image/tactile audit", flush=True)
  save(root / "verification_scope.json", {
    "mode": verification, "schema_valid": True,
    "cleaning": "reuse frozen existing reports; not rerun",
    "source_sha256_checked_before_conversion": True,
    "tactile_arrays_validated_during_conversion": True,
    "full_source_audit_executed": verification == "source",
    "scope": "one output schema pass; in-memory pose/clock/tactile checks during export; source byte comparison only in source mode",
  })
  return validation


README = """# Poker draw — EgoSteer

50 successful episodes; train=43, val=7. Same source selection/split as EgoTouch.
`train/*.tar` and `val/*.tar` are standard head-only 30fps WebDataset shards.
Each sample contains `image.jpg`, `lowdim.npy` (float32,116), `meta.json`.
Use upstream `motion_type=fingertips`; action dimension stays 48. The instruction
is the poker task, not pick-and-place. Payloads are not pre-normalized.

lowdim: wrist state18, fingertip state30, next wrist18, next fingertips30,
OpenCV world-to-camera16, [fx,fy,cx,cy]4. World positions are metres.
Wrist rot6D is column0 followed by column1, with the documented canonical axes.
Pose values come directly from archived render-time SE3, not qpos interpolation.
Actual pose/acquisition times are in metadata and NPZ. 30fps is quantized by
the 500Hz physics clock. The last strict-grid frame provides a next-state target;
an extra off-grid terminal frame, if present, is excluded from WDS training.
Both are retained in the NPZ observation timeline, identified by masks.

## Additional tactile (ignored by unmodified EgoSteer)

`tactile_sidecars/episode_NNNNNN.npz` maps `sample_keys` to each WDS sample.
Load with `numpy.load(path, allow_pickle=False)`.
`tactile_taxel_force_n`: [N,2,5,7,5,3], units N, no clipping or smoothing.
Sides left/right; fingers thumb/index/middle/ring/little; channels normal,
signed tangent_x, signed tangent_y in the recorded fingertip tangent frame.
`tactile_force_n` sums taxels; `tactile_mean_n` averages them (EgoTouch feature).
Tangential magnitude is sqrt(Fx**2+Fy**2), not a fourth channel.
Source timestamps, nearest-nonfuture indices, validity masks, force metadata,
episode outcome and randomized initialization metadata are preserved.
Manifest `tactile_included=false` means the STANDARD 116D WDS contains no tactile;
`tactile_sidecars_included=true` explicitly describes this separate extension.

## Checks and use

`validation.json`: local full schema, JPEG, numeric, SE3, frame/next-state checks.
`source_audit.json`: every image, pose and tactile taxel compared to raw HDF5.
`QUALITY_NOTES.md` and `quality/`: original Cleaning scope and review warnings.
No model weights, normalizer or training success is claimed. Point the upstream
dataset configuration at train/val shards and compute normalization statistics
from TRAIN ONLY using that project's normalizer pipeline before training.
No old absolute source/model path is needed to read these self-contained shards.
"""


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--source-root", type=Path, default=Path("datasets/poker_draw_0909"))
  parser.add_argument("--processing", type=Path, default=Path("datasets/poker_draw_0909/processing/dual_format_20260909"))
  parser.add_argument("--output", type=Path, default=Path("datasets/poker_draw_0909_egosteer"))
  parser.add_argument("--expected-episodes", type=int, default=50)
  parser.add_argument("--verification", choices=("schema", "source"), default="source",
                      help="schema: reuse Cleaning and perform one output format check; source: additionally re-audit every image/pose/tactile against raw HDF5")
  args = parser.parse_args()
  output = args.output.resolve()
  if output.exists():
    raise FileExistsError(output)
  snapshot = selection(args.source_root, args.processing, args.expected_episodes)
  count = snapshot["episode_count"]
  train_count = sum(row["split"] == "train" for row in snapshot["episodes"])
  val_count = count - train_count
  started = time.monotonic()
  root = Path(tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=output.parent))
  print(f"EgoSteer staging: {root}; failed work is preserved", flush=True)
  sources = tuple(describe_source(row) for row in snapshot["episodes"])
  splits = {row["episode_index"]: row["split"] for row in snapshot["episodes"]}
  created = datetime.now(timezone.utc).isoformat()
  save(root / "source_snapshot_manifest.json", _source_manifest(args.source_root.resolve() / "raw", sources, created))
  (root / "tactile_sidecars").mkdir()
  shards, conversions, sidecars = {"train": [], "val": []}, [], []
  for split in ("train", "val"):
    (root / split).mkdir()
    selected = [source for source in sources if splits[source.episode_index] == split]
    for start in range(0, len(selected), 5):
      chunk = selected[start:start + 5]
      path = root / split / f"shard-{start // 5:06d}.tar"
      records = []
      with tarfile.open(path, "x", format=tarfile.USTAR_FORMAT) as archive:
        for source in chunk:
          episode_started = time.monotonic()
          record, descriptor = convert(source, split, archive, root, output.name)
          records.append(record)
          conversions.append(record)
          sidecars.append(descriptor)
          print(f"EgoSteer {source.episode_index}: {record.exported_samples} samples ({len(conversions)}/{count}), {time.monotonic()-episode_started:.1f}s", flush=True)
      shards[split].append(ShardSummary(path.relative_to(root).as_posix(), sha256(path),
                                       path.stat().st_size, len(chunk),
                                       tuple(source.episode_index for source in chunk),
                                       sum(record.exported_samples for record in records)))
  manifest = _dataset_manifest(created_utc=created, dataset_name=output.name, instruction=INSTRUCTION,
                               split_seed=0, val_fraction=val_count / count, jpeg_quality=95,
                               source_manifest_sha256=sha256(root / "source_snapshot_manifest.json"),
                               sources=sources, shards=shards, conversions=tuple(conversions))
  manifest["split_config"] = {"method": snapshot["split_rule"], "train_episodes": train_count, "val_episodes": val_count}
  manifest["time_alignment"] = {"pose_source": "archived native site SE3 at cameras/head/pose_timestamp",
                                "qpos_interpolation": False, "simulation_replay": False,
                                "terminal_action": "last strict-grid frame is action-only; N-1 samples",
                                "terminal_observation": "all source frames retained in NPZ; off-grid terminal not used as 30fps action",
                                "tactile": "latest-not-future at pose_timestamp, maximum age 20ms"}
  manifest["tactile_sidecars_included"] = True
  manifest["tactile_sidecars"] = sidecars
  save(root / "dataset_manifest.json", manifest)
  verify_release(root, sources, conversions, shards, sidecars, args.verification)
  attach_quality(root, snapshot)
  readme = README.replace("50 successful episodes; train=43, val=7.",
                         f"{count} successful episodes; train={train_count}, val={val_count}.")
  if args.verification == "schema":
    readme = readme.replace("`source_audit.json`: every image, pose and tactile taxel compared to raw HDF5.",
        "`verification_scope.json`: existing Cleaning reused; one output schema check.\n"
        "Pose/clock/tactile checks run during conversion; no second exhaustive raw comparison was executed.")
  (root / "README.md").write_text(readme, encoding="utf-8")
  if output.exists():
    raise FileExistsError(output)
  os.rename(root, output)
  save(args.processing / "egosteer_package_result.json",
       {"valid": True, "episodes": count, "elapsed_seconds": time.monotonic() - started,
        "output": str(output), "samples": sum(item.exported_samples for item in conversions),
        "verification": args.verification, "full_source_audit_executed": args.verification == "source"})
  print(f"EgoSteer complete: {output}", flush=True)


if __name__ == "__main__":
  main()
