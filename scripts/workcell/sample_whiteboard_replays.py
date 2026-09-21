#!/usr/bin/env python3
"""Randomly sample completed Whiteboard episodes and export review videos."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py

DEFAULT_DATASET = Path(
  "/nas/chenxianchi/datasets/sim/whiteboard-wipe/raw/0920_200"
)


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "dataset",
    type=Path,
    nargs="?",
    default=DEFAULT_DATASET,
    help=f"Raw collection root (default: {DEFAULT_DATASET})",
  )
  parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path("artifacts/whiteboard_random_replays"),
    help="New directory that will contain one review directory per sample",
  )
  parser.add_argument("--count", type=int, default=3)
  parser.add_argument(
    "--seed",
    type=int,
    default=20260921,
    help="Sampling seed; use the same value to select the same episodes",
  )
  parser.add_argument("--fps", type=float, default=10.0)
  parser.add_argument("--width", type=int, default=1920)
  parser.add_argument("--height", type=int, default=1080)
  parser.add_argument(
    "--cameras",
    nargs="+",
    help="Saved camera streams to include (default: all standard streams present)",
  )
  parser.add_argument(
    "--max-tactile-age-ms",
    type=float,
    default=50.0,
    help="Maximum causal tactile age accepted for an RGB frame",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Print the selected episodes without creating output",
  )
  args = parser.parse_args(argv)
  if args.count <= 0:
    parser.error("--count must be positive")
  if args.fps <= 0:
    parser.error("--fps must be positive")
  if args.width < 960 or args.height < 540 or args.width % 2 or args.height % 2:
    parser.error("--width/--height must be even and at least 960x540")
  if not 0 <= args.max_tactile_age_ms <= 1000:
    parser.error("--max-tactile-age-ms must be between 0 and 1000")
  return args


def _json_attribute(file: h5py.File, name: str) -> dict[str, Any]:
  value = file.attrs.get(name, "{}")
  if isinstance(value, bytes):
    value = value.decode("utf-8")
  try:
    decoded = json.loads(str(value))
  except (TypeError, ValueError, json.JSONDecodeError):
    return {}
  return decoded if isinstance(decoded, dict) else {}


def _is_completed_whiteboard_episode(path: Path) -> bool:
  """Exclude partial, failed, and non-Whiteboard HDF5 files."""

  try:
    with h5py.File(path, "r") as file:
      metadata = _json_attribute(file, "metadata_json")
      outcome = _json_attribute(file, "outcome_json")
      return (
        metadata.get("scene") == "whiteboard-wipe"
        and outcome.get("success") is True
        and "cameras" in file
        and "state/timestamp" in file
      )
  except (OSError, KeyError):
    return False


def discover_episode_candidates(dataset: Path) -> list[Path]:
  dataset = dataset.expanduser().resolve()
  if not dataset.is_dir():
    raise NotADirectoryError(f"dataset root does not exist: {dataset}")
  # In unified collections, only successful attempts publish raw/episode.h5.
  # Avoid opening every large HDF5 over NAS here; shuffled candidates are
  # validated below until the requested number of completed episodes is found.
  return sorted(dataset.glob("**/raw/episode.h5"))


def select_episodes(dataset: Path, count: int, seed: int) -> tuple[list[Path], int]:
  candidates = discover_episode_candidates(dataset)
  shuffled = list(candidates)
  random.Random(seed).shuffle(shuffled)
  selected: list[Path] = []
  for path in shuffled:
    if _is_completed_whiteboard_episode(path):
      selected.append(path)
      if len(selected) == count:
        break
  if len(selected) < count:
    raise ValueError(
      f"requested {count} episodes, but only {len(selected)} completed "
      f"Whiteboard episodes were found among {len(candidates)} HDF5 candidates "
      f"under {dataset}"
    )
  return selected, len(candidates)


def _sample_name(dataset: Path, episode: Path, ordinal: int) -> str:
  relative = episode.relative_to(dataset)
  episode_id = next(
    (part for part in relative.parts if part.isdigit()), f"sample_{ordinal:02d}"
  )
  attempt = next(
    (part for part in relative.parts if part.startswith("attempt_")), "attempt"
  )
  return f"episode_{episode_id}_{attempt}"


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
  temporary = path.with_suffix(".json.tmp")
  temporary.write_text(
    json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
  )
  temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
  args = _arguments(argv)
  dataset = args.dataset.expanduser().resolve()
  selected, candidate_count = select_episodes(dataset, args.count, args.seed)
  print(f"Found {candidate_count} published episode candidates; seed={args.seed}")
  for index, episode in enumerate(selected, start=1):
    print(f"  [{index}/{args.count}] {episode}")
  if args.dry_run:
    return 0

  output_root = args.output_dir.expanduser().absolute()
  if output_root.exists() or output_root.is_symlink():
    raise FileExistsError(f"output directory already exists: {output_root}")
  output_root.mkdir(parents=True)
  manifest: dict[str, Any] = {
    "schema_version": "whiteboard_random_replays_v1",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "completed": False,
    "dataset": str(dataset),
    "seed": args.seed,
    "candidate_count": candidate_count,
    "sample_count": args.count,
    "fps": args.fps,
    "output_size": [args.width, args.height],
    "requested_cameras": args.cameras,
    "samples": [],
  }
  _write_manifest(output_root / "selection.json", manifest)

  from kaihand_tactile_env.shared.offline_replay import export_multimodal_replay

  try:
    for index, episode in enumerate(selected, start=1):
      name = _sample_name(dataset, episode, index)
      destination = output_root / name
      print(f"Exporting [{index}/{args.count}] {name} ...", flush=True)
      report = export_multimodal_replay(
        episode,
        destination,
        cameras=None if args.cameras is None else tuple(args.cameras),
        fps=args.fps,
        width=args.width,
        height=args.height,
        maximum_tactile_age_s=args.max_tactile_age_ms / 1000.0,
      )
      manifest["samples"].append(
        {
          "source_hdf5": str(episode),
          "output_directory": name,
          "video": f"{name}/review.mp4",
          "camera_names": report["camera_names"],
          "frame_count": report["output_frame_count"],
          "duration_s": report["video_duration_s"],
          "source_sha256": report["source_sha256"],
        }
      )
      _write_manifest(output_root / "selection.json", manifest)
    manifest["completed"] = True
    _write_manifest(output_root / "selection.json", manifest)
  except BaseException as error:
    manifest["error"] = f"{type(error).__name__}: {error}"
    _write_manifest(output_root / "selection.json", manifest)
    raise

  print(f"Wrote {args.count} review videos to {output_root}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
