"""Raw dataset discovery and explicit model-format adapter contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from kaihand_tactile_env.shared.cameras import TRAINING_CAMERA_NAMES

TASKS = (
  "pick-place",
  "poker-draw",
  "usb-insert",
  "bulb-screw",
  "vase-wipe",
  "install-ram",
  "whiteboard-wipe",
  "sponge-grasp",
)
FORMATS = ("egosteer", "pi05", "egotouch", "lerobot-v3")


@dataclass(frozen=True)
class RawDataset:
  root: Path
  episodes: tuple[Path, ...]
  task: str
  cameras: tuple[str, ...]


@dataclass(frozen=True)
class AdapterContract:
  task: str
  output_format: str
  supported_camera_sets: tuple[tuple[str, ...], ...]
  backend: str
  resumable: bool = False


ADAPTERS = {
  ("pick-place", "egosteer"): AdapterContract(
    "pick-place", "egosteer", (("head",),), "convert_to_egosteer.py"
  ),
  ("poker-draw", "egosteer"): AdapterContract(
    "poker-draw", "egosteer", (), "convert_card_to_egosteer.py"
  ),
  ("usb-insert", "egosteer"): AdapterContract(
    "usb-insert", "egosteer", (), "convert_usb_to_egosteer.py"
  ),
  ("bulb-screw", "egosteer"): AdapterContract(
    "bulb-screw", "egosteer", (), "convert_card_to_egosteer.py"
  ),
  ("install-ram", "egosteer"): AdapterContract(
    "install-ram", "egosteer", (), "convert_card_to_egosteer.py"
  ),
  ("whiteboard-wipe", "egosteer"): AdapterContract(
    "whiteboard-wipe", "egosteer", (), "convert_card_to_egosteer.py"
  ),
  ("pick-place", "pi05"): AdapterContract(
    "pick-place", "pi05", (("head",),),
    "convert_pickplace_unified_to_lerobot.py",
    resumable=True,
  ),
  ("poker-draw", "pi05"): AdapterContract(
    "poker-draw", "pi05", (("head", "right_wrist"),),
    "convert_poker_unified_to_lerobot.py",
    resumable=True,
  ),
  ("usb-insert", "pi05"): AdapterContract(
    "usb-insert", "pi05", (("head", "right_wrist"),),
    "convert_usb_unified_to_lerobot.py",
  ),
  ("bulb-screw", "pi05"): AdapterContract(
    "bulb-screw", "pi05", (), "convert_shared_to_lerobot.py",
    resumable=True,
  ),
  ("vase-wipe", "pi05"): AdapterContract(
    "vase-wipe", "pi05", (), "convert_shared_to_lerobot.py",
    resumable=True,
  ),
  ("install-ram", "pi05"): AdapterContract(
    "install-ram", "pi05", (), "convert_shared_to_lerobot.py",
    resumable=True,
  ),
  ("whiteboard-wipe", "pi05"): AdapterContract(
    "whiteboard-wipe", "pi05", (), "convert_shared_to_lerobot.py",
    resumable=True,
  ),
  ("sponge-grasp", "pi05"): AdapterContract(
    "sponge-grasp", "pi05", (("head", "right_wrist"),),
    "convert_shared_to_lerobot.py",
  ),
  **{
    (task, "lerobot-v3"): AdapterContract(
      task, "lerobot-v3", (("head", "right_wrist"),),
      "convert_kaihand_to_lerobot_v3.py", resumable=True,
    )
    for task in (
      "pick-place", "poker-draw", "usb-insert", "bulb-screw",
      "install-ram", "vase-wipe", "whiteboard-wipe", "sponge-grasp",
    )
  },
  ("poker-draw", "egotouch"): AdapterContract(
    "poker-draw", "egotouch", (), "shared.tict_export", resumable=True
  ),
  ("usb-insert", "egotouch"): AdapterContract(
    "usb-insert", "egotouch", (), "shared.tict_export", resumable=True
  ),
  ("whiteboard-wipe", "egotouch"): AdapterContract(
    "whiteboard-wipe", "egotouch", (), "shared.tict_export", resumable=True
  ),
}


def adapter_catalog() -> tuple[dict[str, object], ...]:
  """Return the registered semantic adapters as a stable, serializable catalog.

  An empty ``camera_sets`` list means every canonical camera subset containing
  ``head`` is accepted.  Keeping this information in the adapter registry lets
  the CLI, documentation checks and future model integrations use one source of
  truth instead of maintaining separate task tables.
  """

  return tuple(
    {
      "task": contract.task,
      "format": contract.output_format,
      "backend": contract.backend,
      "camera_sets": [list(names) for names in contract.supported_camera_sets],
      "camera_policy": (
        "any_head_subset"
        if not contract.supported_camera_sets
        else "listed_sets_only"
      ),
      "resumable": contract.resumable,
    }
    for contract in sorted(
      ADAPTERS.values(), key=lambda value: (value.task, value.output_format)
    )
  )


def canonical_cameras(values) -> tuple[str, ...]:
  requested = tuple(values)
  if not requested or len(set(requested)) != len(requested):
    raise ValueError("cameras must be a nonempty set without duplicates")
  unknown = set(requested) - set(TRAINING_CAMERA_NAMES)
  if unknown:
    raise ValueError(f"unsupported training cameras: {sorted(unknown)}")
  if "head" not in requested:
    raise ValueError("camera selection must include head")
  return tuple(name for name in TRAINING_CAMERA_NAMES if name in requested)


def default_cameras_for(task: str, output_format: str) -> tuple[str, ...]:
  """Use a fixed adapter camera contract, otherwise keep the head-only default."""
  try:
    contract = ADAPTERS[(task, output_format)]
  except KeyError as error:
    raise ValueError(
      f"no {output_format} semantic adapter is registered for {task}"
    ) from error
  if len(contract.supported_camera_sets) == 1:
    return contract.supported_camera_sets[0]
  return ("head",)


def _metadata(path: Path) -> tuple[str | None, tuple[str, ...]]:
  import h5py

  with h5py.File(path, "r") as file:
    try:
      metadata = json.loads(file.attrs.get("metadata_json", "{}"))
    except (TypeError, json.JSONDecodeError):
      metadata = {}
    task = metadata.get("scene") if isinstance(metadata, dict) else None
    cameras = tuple(file.get("cameras", {}))
  return task, cameras


def _collection_paths(root: Path) -> tuple[Path, ...] | None:
  """Use the collection summary as authority for published successes."""
  contract_path = root / "collection.json"
  summary_path = root / "summary.json"
  if not contract_path.is_file() or not summary_path.is_file():
    return None
  contract = json.loads(contract_path.read_text(encoding="utf-8"))
  if not isinstance(contract, dict):
    raise ValueError(f"{contract_path}: contract must be an object")
  if contract.get("schema") != "task_collection_v2":
    return None
  summary = json.loads(summary_path.read_text(encoding="utf-8"))
  if not isinstance(summary, dict):
    raise ValueError(f"{summary_path}: summary must be an object")
  rows = summary.get("episodes")
  if not isinstance(rows, list):
    raise ValueError(f"{summary_path}: episodes must be a list")
  tasks = contract.get("tasks")
  counts = summary.get("by_task")
  if not isinstance(tasks, list) or not isinstance(counts, dict):
    raise ValueError("collection tasks and by_task counts must be declared")
  paths = []
  seen = set()
  successes = {name: 0 for name in tasks}
  for row in rows:
    if not isinstance(row, dict) or row.get("status") != "success":
      continue
    row_task = row.get("task")
    if row_task not in successes:
      raise ValueError(f"{summary_path}: success row has undeclared task {row_task!r}")
    successes[row_task] += 1
    names = row.get("hdf5")
    if not isinstance(names, list) or len(names) != 1 or not isinstance(names[0], str):
      raise ValueError(f"{summary_path}: success row must name one HDF5")
    unresolved = root / names[0]
    if unresolved.is_symlink():
      raise ValueError(f"collection HDF5 must not be a symlink: {unresolved}")
    path = unresolved.resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file() or path in seen:
      raise ValueError(f"invalid or duplicate collection HDF5: {names[0]}")
    seen.add(path)
    paths.append(path)
  for name, count in successes.items():
    declared = counts.get(name)
    if not isinstance(declared, dict) or declared.get("success") != count:
      raise ValueError(f"{summary_path}: {name} success count disagrees")
  return tuple(paths)


def inspect_raw_dataset(
  input_dir: str | Path,
  *,
  task: str = "auto",
  cameras=TRAINING_CAMERA_NAMES,
) -> RawDataset:
  root = Path(input_dir).expanduser().resolve()
  if not root.is_dir():
    raise ValueError(f"input directory does not exist: {root}")
  selected_paths = _collection_paths(root)
  candidates = (
    selected_paths if selected_paths is not None else tuple(
      path for path in sorted(root.rglob("*.h5"))
      if not path.name.endswith(".partial")
    )
  )
  if not candidates:
    raise ValueError(f"no finalized HDF5 episodes below {root}")
  observed_tasks = set()
  per_episode_cameras = []
  for path in candidates:
    observed_task, available = _metadata(path)
    # Ignore unrelated analysis HDF5 files below a caller-selected root. A Raw
    # episode is identified by its recorded scene metadata, never its filename.
    if observed_task not in TASKS:
      continue
    observed_tasks.add(observed_task)
    per_episode_cameras.append((path, observed_task, set(available)))
  if not per_episode_cameras:
    raise ValueError(f"no Raw episodes with scene metadata below {root}")
  if task == "auto":
    if len(observed_tasks) != 1:
      raise ValueError(
        "--task auto requires exactly one recorded scene; "
        f"found {sorted(observed_tasks) or 'none'}"
      )
    selected_task = next(iter(observed_tasks))
    selected_rows = per_episode_cameras
  else:
    if task not in TASKS:
      raise ValueError(f"unknown task {task!r}")
    selected_task = task
    selected_rows = [row for row in per_episode_cameras if row[1] == task]
    if not selected_rows:
      raise ValueError(
        f"no Raw episodes for explicit task {task!r}; "
        f"source scenes are {sorted(observed_tasks)}"
      )
  if selected_paths is not None:
    contract = json.loads((root / "collection.json").read_text(encoding="utf-8"))
    if contract.get("collection_mode") == "target-successes":
      summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
      if summary.get("target_met", {}).get(selected_task) is not True:
        raise ValueError(f"{selected_task} success target was not met")
  selected_cameras = canonical_cameras(cameras)
  episodes = tuple(path for path, _, _ in selected_rows)
  missing = {
    str(path): sorted(set(selected_cameras) - available)
    for path, _, available in selected_rows
    if not set(selected_cameras).issubset(available)
  }
  if missing:
    first_path, names = next(iter(missing.items()))
    raise ValueError(f"source episode {first_path} lacks requested cameras {names}")
  return RawDataset(root, episodes, selected_task, selected_cameras)


def adapter_for(dataset: RawDataset, output_format: str) -> AdapterContract:
  try:
    contract = ADAPTERS[(dataset.task, output_format)]
  except KeyError as error:
    raise ValueError(
      f"no {output_format} semantic adapter is registered for {dataset.task}; "
      "the generic path layer deliberately does not invent task actions"
    ) from error
  if (
    contract.supported_camera_sets
    and dataset.cameras not in contract.supported_camera_sets
  ):
    raise ValueError(
      f"{dataset.task} -> {output_format} currently supports camera sets "
      f"{contract.supported_camera_sets}, got {dataset.cameras}"
    )
  return contract
