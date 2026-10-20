#!/usr/bin/env python3
"""Plan or explicitly convert the nine remaining force-array KaiHand datasets.

The default is metadata-only preflight. --run converts Zarr; the separate
--prepare-training-assets mode computes normalization in fresh staging directories.
No mode trains a model or modifies an existing published dataset/asset directory.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).absolute().parents[2]
DATA_ROOT = Path("/nas/chenxianchi/datasets/sim")
FTP1_ROOT = ROOT.parent / "ftp1-policy"
CONVERTER = ROOT / "scripts/workcell/convert_kaihand_raw_to_ftp1_zarr.py"
TASKS = (
  ("bulb-screw", "bulb-screw", "raw/1012_v15_200_r2", "1012_200"),
  ("bulb-screw-inplace", "bulb-screw-inplace", "shared_raw/1018_200", "1018_200"),
  ("bulb-unscrew", "bulb-unscrew", "raw/1010_200", "1010_200"),
  ("install-ram", "install-ram", "raw/0920_200", "0920_200"),
  ("usb-insert", "usb_insert", "raw/0920_200", "0920_200"),
  ("usb-unplug", "usb-unplug", "raw/1010_200", "1010_200"),
  ("whiteboard-wipe", "whiteboard-wipe", "raw/0929_200", "0929_200"),
  (
    "whiteboard-wipe-horizontal",
    "whiteboard-wipe-horizontal",
    "raw/1010_200",
    "1010_200",
  ),
  (
    "whiteboard-wipe-horizontal-line",
    "whiteboard-wipe-horizontal-line",
    "raw/1012_200",
    "1012_200",
  ),
)
MARKER = "kaihand_ftp1_assets_binding.json"


def cpu_environment() -> dict[str, str]:
  return dict(
    os.environ,
    JAX_PLATFORMS="cpu",
    JAX_PLATFORM_NAME="cpu",
    CUDA_VISIBLE_DEVICES="",
    PYTHONDONTWRITEBYTECODE="1",
    OPENBLAS_NUM_THREADS="1",
    OMP_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
  )


def build_plan(args: argparse.Namespace) -> dict:
  if args.workers <= 0 or args.episodes_per_shard <= 0 or args.norm_workers < 0:
    raise ValueError(
      "workers/episodes-per-shard must be positive; norm-workers must be nonnegative"
    )
  selected = set(args.tasks or [row[0] for row in TASKS])
  unknown = selected.difference(row[0] for row in TASKS)
  if unknown:
    raise ValueError(f"Unsupported tasks: {sorted(unknown)}")
  dataset_root = args.dataset_root.absolute()
  groups = []
  for task, directory, raw_suffix, version in TASKS:
    if task not in selected:
      continue
    task_root = dataset_root / directory
    groups.append(
      dict(
        task=task,
        input_dir=str(task_root / raw_suffix),
        output_dir=str(task_root / "ftp1" / version),
        version=version,
        repo_id=f"{directory.replace('-', '_')}_{version}",
        assets_base_dir=str(task_root / "ftp1_assets"),
        checkpoint_base_dir=str(task_root / "ftp1_checkpoints"),
      )
    )
  return dict(
    schema="kaihand-nine-task-ftp1-conversion-v1",
    groups=groups,
    python_bin=str(args.python_bin.absolute()),
    converter=str(CONVERTER),
    expected_episodes=200,
    workers=args.workers,
    episodes_per_shard=args.episodes_per_shard,
    resume=args.resume,
    verify_source_hash=args.verify_source_hash,
    norm_workers=args.norm_workers,
    trajectory_fps=30,
    action_horizon=32,
    action_joint_rep="mix",
    tactile_type="matrix",
    tactile_shape=[5, 7, 5, 3],
  )


def conversion_command(plan: dict, group: dict, *, dry_run: bool) -> list[str]:
  command = [
    plan["python_bin"],
    "-B",
    plan["converter"],
    "--task",
    group["task"],
    "--input-dir",
    group["input_dir"],
    "--output-dir",
    group["output_dir"],
    "--expected-episodes",
    str(plan["expected_episodes"]),
    "--workers",
    str(plan["workers"]),
    "--episodes-per-shard",
    str(plan["episodes_per_shard"]),
  ]
  if plan["resume"]:
    command.append("--resume")
  if plan["verify_source_hash"]:
    command.append("--verify-source-hash")
  if dry_run:
    command += ["--dry-run", "--plan-json"]
  return command


def preflight(plan: dict, *, require_published: bool = False) -> list[dict]:
  """Check every source/output contract before allowing the first write."""
  if not os.access(plan["python_bin"], os.X_OK):
    raise ValueError(f"FTP-1 Python missing: {plan['python_bin']}")
  if not Path(plan["converter"]).is_file():
    raise ValueError(f"Converter missing: {plan['converter']}")
  reports = []
  for index, group in enumerate(plan["groups"], start=1):
    print(
      f"Metadata preflight {index}/{len(plan['groups'])}: {group['task']}",
      file=sys.stderr,
      flush=True,
    )
    command_plan = dict(plan, resume=True) if require_published else plan
    command = conversion_command(command_plan, group, dry_run=True)
    result = subprocess.run(
      command, env=cpu_environment(), cwd=ROOT, text=True, capture_output=True
    )
    row = dict(task=group["task"], valid=result.returncode == 0)
    if result.returncode:
      row["error"] = (result.stderr or result.stdout)[-12000:]
    elif require_published:
      try:
        converter_plan = json.loads(result.stdout)
        if converter_plan.get("published_complete") is not True:
          raise ValueError(
            "Converted dataset is not fully published; run conversion first"
          )
      except (ValueError, AttributeError) as error:
        row.update(valid=False, error=str(error))
    reports.append(row)
    print(
      f"Metadata preflight {group['task']}: {'passed' if row['valid'] else 'FAILED'}",
      file=sys.stderr,
      flush=True,
    )
  return reports


def execute_conversion(plan: dict, reports: list[dict]) -> None:
  if len(reports) != len(plan["groups"]) or not all(row["valid"] for row in reports):
    raise ValueError(
      "All selected dataset metadata preflights must pass before conversion; nothing was converted"
    )
  for group in plan["groups"]:
    print(f"Converting {group['task']} -> {group['output_dir']}", flush=True)
    subprocess.run(
      conversion_command(plan, group, dry_run=False),
      env=cpu_environment(),
      cwd=ROOT,
      check=True,
    )


def snapshot_for(group: dict) -> dict:
  return dict(
    repo_id=group["repo_id"],
    dataset_config_path=str(Path(group["output_dir"]) / "dataset_config.json"),
    action_down_sample_steps=1,
    state_down_sample_steps=[],
    image_down_sample_steps=[],
    tactile_down_sample_steps=[],
    disable_history=True,
    proprioception_pose_rep="relative",
    action_pose_rep="relative",
    proprioception_joint_rep="abs",
    action_joint_rep="mix",
    norm_type="zscore",
    independent_norm_mode="all",
    norm_time_dim=False,
    norm_image_tactile_mode="channel_wise",
    norm_image_channel_pool_max_size=2000000,
    norm_image_channel_batch_samples=0,
    contact_detection_threshold_k=3.0,
  )


def norm_command(plan: dict, group: dict, staging_base: Path) -> list[str]:
  return [
    plan["python_bin"],
    "-B",
    str(FTP1_ROOT / "scripts/zarr_compute_norm_stats.py"),
    "ftp1",
    "--repo-id",
    group["repo_id"],
    "--data.repo-id",
    group["repo_id"],
    "--exp-name",
    f"{group['repo_id']}_norm",
    "--dataset-config-path",
    str(Path(group["output_dir"]) / "dataset_config.json"),
    "--assets-base-dir",
    str(staging_base),
    "--checkpoint-base-dir",
    group["checkpoint_base_dir"],
    "--model.action-horizon",
    "32",
    "--action-down-sample-steps",
    "1",
    "--disable-history",
    "--proprioception-pose-rep",
    "relative",
    "--action-pose-rep",
    "relative",
    "--proprioception-joint-rep",
    "abs",
    "--action-joint-rep",
    "mix",
    "--norm-type",
    "zscore",
    "--independent-norm-mode",
    "all",
    "--no-norm-time-dim",
    "--norm-image-tactile-mode",
    "channel_wise",
    "--norm-image-channel-pool-max-size",
    "2000000",
    "--norm-image-channel-batch-samples",
    "0",
    "--contact-detection-threshold-k",
    "3.0",
    "--use-val-dataset",
    "--create-train-val-split",
    "--val-ratio",
    "0.01",
    "--seed",
    "42",
    "--norm-sample-ratio",
    "1.0",
    "--norm-batch-size",
    "64",
    "--norm-num-workers",
    str(plan["norm_workers"]),
    "--model.state-input-mode",
    "adarms",
    "--model.use-tactile-input",
    "--no-wandb-enabled",
  ]


def read_json(path: Path):
  with path.open(encoding="utf-8") as stream:
    return json.load(stream)


def file_sha(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def reject_symlinks(path: Path) -> None:
  for candidate in (path.absolute(), *path.absolute().parents):
    if candidate.is_symlink():
      raise ValueError(f"Symlinked asset/output paths are not allowed: {candidate}")
    if candidate.exists() and not candidate.is_dir():
      raise ValueError(f"Asset/output path component is not a directory: {candidate}")


def dataset_domain(group: dict) -> str:
  output = Path(group["output_dir"])
  config = read_json(output / "dataset_config.json")
  enabled = [row for row in config["datasets"] if row.get("enabled", True)]
  if len(enabled) != 1 or Path(enabled[0]["path"]).absolute() != output.absolute():
    raise ValueError("Exactly one task-bound dataset domain is required")
  name = enabled[0]["name"]
  if not isinstance(name, str) or not re.fullmatch(r"KaiHandSim[A-Za-z0-9]+", name):
    raise ValueError("Unsafe or unexpected dataset domain name")
  if enabled[0].get("use_trajectory_ratio", 1.0) != 1.0:
    raise ValueError("Full trajectory ratio is required")
  return name


def dataset_binding(group: dict) -> dict:
  output = Path(group["output_dir"])
  reject_symlinks(output)
  dataset_domain(group)
  manifest = output / "conversion_manifest.json"
  config = output / "dataset_config.json"
  record = read_json(manifest)
  if record.get("task") != group["task"]:
    raise ValueError(f"Converted task mismatch: {manifest}")
  expected = dict(
    schema="kaihand_raw_to_ftp1_zarr_v2",
    episodes=200,
    fps=30,
    ftp1_action_horizon=32,
    ftp1_action_down_sample_steps=1,
    output_root=str(output),
  )
  if any(record.get(key) != value for key, value in expected.items()):
    raise ValueError(
      f"Converted dataset is not the complete 200-episode H32/30Hz contract: {manifest}"
    )
  shards = record.get("shards", [])
  if not shards or sum(int(row.get("episodes", 0)) for row in shards) != 200:
    raise ValueError(f"Published shard episode count mismatch: {manifest}")
  if sum(int(row.get("frames", 0)) for row in shards) != record.get("frames"):
    raise ValueError(f"Published shard frame count mismatch: {manifest}")
  for row in shards:
    shard = Path(row["path"])
    if (
      shard.parent != output
      or shard.suffix != ".zarr"
      or shard.is_symlink()
      or not shard.is_dir()
    ):
      raise ValueError(f"Missing or out-of-domain Zarr shard: {shard}")
  if len(record.get("source_episodes", [])) != 200:
    raise ValueError(f"Published source episode provenance missing: {manifest}")
  return dict(
    schema="kaihand-ftp1-assets-recipe-v1",
    task=group["task"],
    norm_recipe=dict(sample_ratio=1.0, val_ratio=0.01, seed=42, expected_episodes=200),
    dataset_config_sha256=file_sha(config),
    conversion_manifest_sha256=file_sha(manifest),
    snapshot=snapshot_for(group),
  )


def finite_json(value, *, positive: bool = False) -> None:
  if isinstance(value, dict):
    for key, item in value.items():
      finite_json(item, positive=key == "std")
  elif isinstance(value, list):
    for item in value:
      finite_json(item, positive=positive)
  elif isinstance(value, (int, float)) and not isinstance(value, bool):
    if not math.isfinite(value) or (positive and value <= 0):
      raise ValueError(
        "Normalization contains nonfinite values or nonpositive standard deviations"
      )


def nested_shape(value) -> tuple:
  if not isinstance(value, list):
    return ()
  shapes = [nested_shape(item) for item in value]
  if shapes and any(shape != shapes[0] for shape in shapes):
    raise ValueError("Ragged normalization matrix")
  return (len(value), *(shapes[0] if shapes else ()))


def validate_assets(path: Path, group: dict) -> None:
  snapshot = read_json(path / "norm_params_snapshot.json")
  if snapshot != snapshot_for(group):
    raise ValueError(f"Normalization snapshot mismatch: {path}")
  name = dataset_domain(group)
  tactile_config = read_json(path / "tactile_input_config.json")
  tactile = tactile_config.get(name, {}).get("right_tactile_fingers", {})
  if tactile.get("type") != "matrix" or tactile.get("shape") != [1, 5, 7, 5, 3]:
    raise ValueError(f"Expected current-frame five-finger matrix tactile: {path}")
  if (
    tactile.get("function_areas") != [0, 1, 2, 3, 4]
    or tactile.get("sensor") != "KaiHandSimTaxelV1"
  ):
    raise ValueError(f"Unexpected tactile sensor/area identity: {path}")
  required = [
    path / "share_norm_stats_all_t0_zscore.json",
    path / name / "independent_norm_stats_all_t0_zscore.json",
    path / name / "train_val_split.json",
    path / "dataset_stats.json",
  ]
  for file in required:
    finite_json(read_json(file))
  independent = read_json(required[1])
  for key in ("actions", "state"):
    stats = independent.get("params", {}).get(key, {})
    if len(stats.get("mean", [])) != 120 or len(stats.get("std", [])) != 120:
      raise ValueError(f"Expected 120-dimensional {key} normalization")
  tactile_stats = independent.get("params", {}).get("tactile", {})
  key = "KaiHandSimTaxelV1_matrix_7_5_3"
  if set(tactile_stats) != {key}:
    raise ValueError("Tactile normalization sensor/type identity mismatch")
  for field in ("mean", "std"):
    if nested_shape(tactile_stats[key].get(field)) != (7, 5, 3):
      raise ValueError("Expected [7,5,3] shared-area tactile normalization")
  split = read_json(path / name / "train_val_split.json")
  train, val = split.get("train_episode_idx", []), split.get("val_episode_idx", [])
  if (
    len(train) != 198
    or len(val) != 2
    or len(set(train + val)) != 200
    or set(train + val) != set(range(200))
  ):
    raise ValueError("Expected disjoint 198/2 episode train/validation split")


def artifact_hashes(path: Path) -> dict[str, str]:
  return {
    str(file.relative_to(path)): file_sha(file)
    for file in sorted(path.rglob("*.json"))
    if file.name != MARKER
  }


def assets_status(group: dict) -> tuple[dict, Path, bool]:
  destination = Path(group["assets_base_dir"]) / "ftp1" / group["repo_id"]
  reject_symlinks(destination)
  binding = dataset_binding(group)
  if not destination.exists():
    return binding, destination, False
  if (
    destination.is_symlink()
    or not destination.is_dir()
    or not (destination / MARKER).is_file()
  ):
    raise ValueError(
      f"Existing unbound assets will not be overwritten: {destination}; choose a separate assets directory"
    )
  marker = read_json(destination / MARKER)
  if marker.get("binding") != binding or marker.get("artifacts") != artifact_hashes(
    destination
  ):
    raise ValueError(
      f"Existing assets identity/hash mismatch; preserving: {destination}"
    )
  validate_assets(destination, group)
  return binding, destination, True


def prepare_assets(plan: dict) -> None:
  """Precheck all outputs, then stage/publish each new task's own assets."""
  if not os.access(plan["python_bin"], os.X_OK):
    raise ValueError(f"FTP-1 Python missing: {plan['python_bin']}")
  script = FTP1_ROOT / "scripts/zarr_compute_norm_stats.py"
  if not script.is_file():
    raise ValueError(f"Normalization script missing: {script}")
  checked = [(group, *assets_status(group)) for group in plan["groups"]]
  reports = preflight(plan, require_published=True)
  if len(reports) != len(plan["groups"]) or not all(row["valid"] for row in reports):
    errors = "; ".join(
      f"{row['task']}: {row.get('error', 'invalid')}"
      for row in reports
      if not row["valid"]
    )
    raise ValueError(
      f"Every converted dataset must be completely published before normalization: {errors}"
    )
  for group, binding, destination, exists in checked:
    if exists:
      print(f"Verified existing assets: {destination}", flush=True)
      continue
    base = Path(group["assets_base_dir"])
    base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
      prefix=".kaihand-ftp1-norm-", dir=base
    ) as temporary:
      staging = Path(temporary)
      generated = staging / "ftp1" / group["repo_id"]
      print(f"Preparing fresh normalization: {group['task']}", flush=True)
      subprocess.run(
        norm_command(plan, group, staging),
        env=cpu_environment(),
        cwd=FTP1_ROOT,
        check=True,
      )
      validate_assets(generated, group)
      if dataset_binding(group) != binding:
        raise ValueError(
          "Converted dataset identity changed during normalization; publication refused"
        )
      with (generated / MARKER).open("x", encoding="utf-8") as stream:
        json.dump(
          dict(binding=binding, artifacts=artifact_hashes(generated)), stream, indent=2
        )
      destination.parent.mkdir(parents=True, exist_ok=True)
      with (destination.parent / f".{group['repo_id']}.publish.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.exists():
          raise FileExistsError(
            f"Assets appeared during generation; not replacing: {destination}"
          )
        generated.rename(destination)
      print(f"Published normalization: {destination}", flush=True)


def parse_args(argv=None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  mode = parser.add_mutually_exclusive_group()
  mode.add_argument(
    "--plan",
    action="store_true",
    help="Metadata-only preflight (default); no output writes",
  )
  mode.add_argument(
    "--run",
    action="store_true",
    help="Actually convert all selected RAW datasets to Zarr",
  )
  mode.add_argument(
    "--prepare-training-assets",
    action="store_true",
    help="After conversion, generate fresh per-task normalization assets",
  )
  parser.add_argument(
    "--resume",
    action="store_true",
    help="Validate and reuse identical published conversion shards",
  )
  parser.add_argument("--tasks", nargs="+", choices=[row[0] for row in TASKS])
  parser.add_argument("--dataset-root", type=Path, default=DATA_ROOT)
  parser.add_argument("--python-bin", type=Path, default=FTP1_ROOT / ".venv/bin/python")
  parser.add_argument("--workers", type=int, default=2)
  parser.add_argument("--episodes-per-shard", type=int, default=20)
  parser.add_argument(
    "--verify-source-hash",
    action="store_true",
    help="During conversion read/hash every original HDF5 (not in plan mode)",
  )
  parser.add_argument("--norm-workers", type=int, default=4)
  return parser.parse_args(argv)


def main(argv=None) -> int:
  try:
    args = parse_args(argv)
    plan = build_plan(args)
    if args.prepare_training_assets:
      prepare_assets(plan)
      return 0
    reports = preflight(plan)
    print(
      json.dumps(
        dict(
          phase="metadata_preflight",
          plan=plan,
          preflight=reports,
          outputs_written=False,
        ),
        ensure_ascii=False,
        indent=2,
      ),
      flush=True,
    )
    if not all(row["valid"] for row in reports):
      return 2
    if args.run:
      execute_conversion(plan, reports)
    return 0
  except (ValueError, OSError, subprocess.CalledProcessError) as error:
    print(f"FTP-1 batch error: {error}", file=sys.stderr)
    return 2


if __name__ == "__main__":
  raise SystemExit(main())
