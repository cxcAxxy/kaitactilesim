"""CPU-only synthetic fixtures; no reads/writes to published training datasets."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import h5py
import numpy as np
import pytest
from kaihand_tactile_env.shared.tactile_task_contracts import task_contract

SCRIPT = (
  Path(__file__).resolve().parents[1]
  / "scripts/workcell/convert_kaihand_raw_to_ftp1_zarr.py"
)
SPEC = importlib.util.spec_from_file_location("ftp1_multitask_conversion", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
converter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = converter
SPEC.loader.exec_module(converter)


def test_usb_runtime_contract_matches_frozen_conversion_instruction():
  runtime = task_contract("usb-insert")
  converted = converter.TASK_CONTRACTS["usb-insert"]

  assert runtime["domain_name"] == "KaiHandSimUsbInsert"
  assert runtime["instruction"] == converted.instruction
  assert runtime["tactile_target_body"] == "usb_plug"


def test_source_change_during_array_read_prevents_publication(tmp_path, monkeypatch):
  task = "bulb-screw-inplace"
  collection = make_collection(tmp_path / "raw", task)
  source = converter.discover_sources(collection, task)[0]
  original = converter.convert_episode
  writer = MagicMock()
  monkeypatch.setattr(converter, "_ShardWriter", lambda *_args: writer)

  def convert_then_change(source, **kwargs):
    episode = original(source, **kwargs)
    stat = source.hdf5_path.stat()
    os.utime(source.hdf5_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    return episode

  monkeypatch.setattr(converter, "convert_episode", convert_then_change)
  output = tmp_path / "result.zarr"
  with pytest.raises(ValueError, match="source changed during conversion"):
    converter._write_shard(
      shard_index=0,
      shard_count=1,
      shard_sources=(source,),
      final_path=output,
      instruction=converter.TASK_CONTRACTS[task].instruction,
      sensor_name=converter.TACTILE_SENSOR,
      verify_source_hash=True,
      task=task,
    )
  writer.add_episode.assert_not_called()
  assert not output.exists()


def make_collection(
  root: Path, task: str, *, episodes: int = 1, both_hands: bool = False
) -> Path:
  """Build a tiny canonical Raw collection usable by independent loader tests."""
  root.mkdir(parents=True, exist_ok=True)
  records = []
  contract = converter.TASK_CONTRACTS[task]
  for episode_index in range(episodes):
    relative = Path(task) / f"{episode_index:06d}" / "episode.h5"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    state_times = np.arange(12, dtype=np.float64) / 100
    camera_times = np.arange(4, dtype=np.float64) / 30
    state_indices = np.searchsorted(state_times, camera_times, side="right") - 1
    right_links = list(converter.RIGHT_TACTILE_LINK_NAMES)
    links = (
      [name.replace("hand_r_", "hand_l_") for name in right_links] + right_links
      if both_hands
      else right_links
    )
    metadata = {"scene": task}
    if task in (converter.TASK, "usb-insert"):
      metadata["side"] = "right"
    if contract.mechanics_version is not None:
      metadata["mechanics_version"] = contract.mechanics_version
    outcome = {"success": True}
    if task == converter.TASK:
      outcome["task_completed"] = True
    if task == "usb-unplug":
      outcome.update(
        {
          key: True
          for key in (
            "fully_extracted",
            "placed_on_table",
            "released",
            "hand_retreated",
          )
        }
      )
    with h5py.File(path, "w") as file:
      file.attrs.update(
        {
          "schema_version": converter.RAW_SCHEMA,
          "camera_hz": 30,
          "physics_hz": 1000 if task.startswith("whiteboard-wipe") else 500,
          "control_hz": 500 if task == "usb-insert" else 100,
          "metadata_json": json.dumps(metadata),
          "outcome_json": json.dumps(outcome),
          "model_fingerprint": contract.model_fingerprint or "b" * 64,
          "contact_force_source": "solver_contact_distributed_taxel_v1",
        }
      )
      file.create_dataset("state/timestamp", data=state_times)
      names = (*converter.RIGHT_ARM_JOINT_NAMES, *converter.RIGHT_HAND_JOINT_NAMES)
      file.create_dataset("state/joint_names", data=names, dtype=h5py.string_dtype())
      positions = (
        np.arange(len(state_times) * len(names), dtype=np.float64).reshape(
          len(state_times), len(names)
        )
        / 1000
      )
      file.create_dataset("state/robot_joint_position", data=positions)
      file.create_dataset(
        "wrist_wrench/side_names", data=("left", "right"), dtype=h5py.string_dtype()
      )
      for camera in ("head", "right_wrist"):
        file.create_dataset(f"cameras/{camera}/timestamp", data=camera_times)
        file.create_dataset(f"cameras/{camera}/state_index", data=state_indices)
        file.create_dataset(
          f"cameras/{camera}/rgb", data=np.zeros((4, 8, 8, 3), dtype=np.uint8)
        )
      file.create_dataset(
        "cameras/head/world_from_wrist", data=np.broadcast_to(np.eye(4), (4, 2, 4, 4))
      )
      file.create_dataset(
        "cameras/head/world_from_camera", data=np.broadcast_to(np.eye(4), (4, 4, 4))
      )
      tactile = file.create_group("tactile_contact_force")
      tactile.attrs.update(
        {
          "force_unit": "N",
          "taxel_unit": "N_per_taxel",
          "is_spatial_estimate": True,
          "clock_semantics": "post-step monitor at state/timestamp",
          "metadata_json": json.dumps(
            {
              "source": "solver_contact_distributed_taxel_v1",
              "force_unit": "N",
              "taxel_force_unit": "N",
              "grid_shape": [7, 5],
              "tangent_basis": ["grid_col_positive", "grid_row_positive"],
              "is_spatial_estimate": True,
            }
          ),
        }
      )
      tactile.create_dataset("timestamp", data=state_times)
      tactile.create_dataset("link_names", data=links, dtype=h5py.string_dtype())
      tactile.create_dataset(
        "normal_axis_local", data=np.broadcast_to([0.0, 0.0, 1.0], (len(links), 3))
      )
      tactile.create_dataset(
        "tangent_basis_local",
        data=np.broadcast_to([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], (len(links), 2, 3)),
      )
      tactile.create_dataset(
        "taxel_positions_local_m", data=np.zeros((len(links), 35, 3))
      )
      normal = np.ones((len(state_times), len(links), 7, 5), dtype=np.float64)
      tangent = np.broadcast_to([0.25, -0.1], (*normal.shape, 2)).copy()
      tactile.create_dataset("normal_taxel_force_n", data=normal)
      tactile.create_dataset("tangent_taxel_force_n", data=tangent)
      tactile.create_dataset("normal_force_n", data=normal.sum(axis=(2, 3)))
      tactile.create_dataset("tangent_force_n", data=tangent.sum(axis=(2, 3)))
    path.with_suffix(".json").write_text(
      json.dumps(
        {
          "episode": path.name,
          "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
          "outcome": outcome,
        }
      )
    )
    records.append(
      {
        "task": task,
        "status": "success",
        "episode_index": episode_index,
        "hdf5": [str(relative)],
      }
    )
  (root / "summary.json").write_text(
    json.dumps({"target_met": {task: True}, "episodes": records})
  )
  return root


def args_for(root: Path, output: Path, task: str, *flags: str):
  return converter._parse_args(
    ["--input-dir", str(root), "--output-dir", str(output), "--task", task, *flags]
  )


def test_poker_default_fingerprint_is_backwards_compatible() -> None:
  contract = converter._conversion_contract(
    instruction=converter.TASK_INSTRUCTION, sensor_name=converter.TACTILE_SENSOR
  )
  encoded = json.dumps(
    contract, ensure_ascii=False, sort_keys=True, separators=(",", ":")
  ).encode()
  assert (
    hashlib.sha256(encoded).hexdigest()
    == "dcd16ba4b124bb0980dbf22b501a1f7dd60471ab74a6a343d014ee4ba4007795"
  )
  source = converter.SourceEpisode(
    7,
    Path("/fixture/episode.h5"),
    "task/000007/episode.h5",
    Path("/fixture/episode.json"),
    "task/000007/episode.json",
    123,
    456,
    "a" * 64,
  )
  assert (
    converter._shard_fingerprint(
      [source],
      instruction=converter.TASK_INSTRUCTION,
      sensor_name=converter.TACTILE_SENSOR,
    )
    == "2ebed1b88dec3151c38fdef96b49711d36502285054acd5c170d3d904693f2e1"
  )


@pytest.mark.parametrize("task", tuple(converter.TASK_CONTRACTS))
@pytest.mark.parametrize("both_hands", [False, True])
def test_all_tasks_select_right_links_and_native_state_actions(
  tmp_path: Path, task: str, both_hands: bool
) -> None:
  root = make_collection(tmp_path / "raw", task, both_hands=both_hands)
  (source,) = converter.discover_sources(root, task)
  episode = converter.convert_episode(source, task=task)
  assert episode.arrays["right_tactile_data_fingers"].shape == (4, 5, 7, 5, 3)
  assert episode.arrays["right_tactile_data_fingers"][..., 2].min() == pytest.approx(
    -0.1
  )
  assert "action" not in episode.arrays
  if task != converter.TASK:
    assert (
      source.source_environment["model_fingerprint"]
      == converter.TASK_CONTRACTS[task].model_fingerprint
    )
    assert (
      "task_completed" not in json.loads(source.sidecar_path.read_text())["outcome"]
    )


@pytest.mark.parametrize("task", tuple(converter.TASK_CONTRACTS))
def test_dry_run_reads_no_pixels_taxels_or_source_hash(
  tmp_path: Path, monkeypatch, task: str, capsys
) -> None:
  root = make_collection(tmp_path / "raw", task)
  output = tmp_path / "ftp1"
  original = h5py.Dataset.__getitem__

  def guarded(dataset, key):
    if dataset.name.endswith(
      ("/rgb", "/normal_taxel_force_n", "/tangent_taxel_force_n")
    ):
      raise AssertionError("metadata preflight must not read image/taxel payload")
    return original(dataset, key)

  monkeypatch.setattr(h5py.Dataset, "__getitem__", guarded)
  monkeypatch.setattr(
    converter, "_sha256_file", lambda *_: pytest.fail("must not hash in dry-run")
  )
  assert (
    converter.run(
      args_for(
        root,
        output,
        task,
        "--dry-run",
        "--plan-json",
        "--verify-source-hash",
        "--expected-episodes",
        "1",
      )
    )
    is None
  )
  plan = json.loads(capsys.readouterr().out)
  assert plan["metadata_preflight_valid"] and not plan["published_complete"]
  assert not output.exists()


@pytest.mark.parametrize(
  "field,value,pattern",
  [
    ("model_fingerprint", "a" * 64, "model_fingerprint"),
    ("contact_force_source", "depth_proxy", "requires solver-contact"),
  ],
)
def test_metadata_rejects_wrong_source_identity(
  tmp_path: Path, field: str, value: str, pattern: str
) -> None:
  root = make_collection(tmp_path / "raw", "bulb-screw-inplace")
  (source,) = converter.discover_sources(root, "bulb-screw-inplace")
  with h5py.File(source.hdf5_path, "r+") as file:
    file.attrs[field] = value
  with pytest.raises(ValueError, match=pattern):
    converter.preflight_episode(source, task="bulb-screw-inplace")


def test_bulb_tasks_cannot_be_relabelled(tmp_path: Path) -> None:
  root = make_collection(tmp_path / "raw", "bulb-screw")
  (source,) = converter.discover_sources(root, "bulb-screw")
  with pytest.raises(ValueError, match="source scene"):
    converter.preflight_episode(source, task="bulb-screw-inplace")


def test_usb_goal_cannot_be_changed_to_new_hold_protocol(tmp_path: Path) -> None:
  root = make_collection(tmp_path / "raw", "usb-unplug")
  with pytest.raises(ValueError, match="recorded task protocol"):
    converter.run(
      args_for(
        root,
        tmp_path / "ftp1",
        "usb-unplug",
        "--dry-run",
        "--instruction",
        "Pull out and hold for two seconds.",
      )
    )
  (source,) = converter.discover_sources(root, "usb-unplug")
  assert (
    source.source_environment["completion_protocol"] == "usb-unplug-return-release-v1"
  )


def test_signed_tangent_conservation_checked_during_conversion(tmp_path: Path) -> None:
  root = make_collection(tmp_path / "raw", "install-ram")
  (source,) = converter.discover_sources(root, "install-ram")
  with h5py.File(source.hdf5_path, "r+") as file:
    file["tactile_contact_force/tangent_force_n"][:] *= -1
  with pytest.raises(ValueError, match="signed tangent taxels"):
    converter.convert_episode(source, task="install-ram")


def test_output_raw_overlap_and_symlinks_rejected(tmp_path: Path) -> None:
  root = make_collection(tmp_path / "raw", "bulb-screw")
  for output in (root, root / "ftp1", tmp_path):
    with pytest.raises(ValueError, match="overlap"):
      converter.run(args_for(root, output, "bulb-screw", "--dry-run"))
  link = tmp_path / "output-link"
  link.symlink_to(root, target_is_directory=True)
  with pytest.raises(ValueError, match="symbolic links"):
    converter.run(args_for(root, link, "bulb-screw", "--dry-run"))


def test_expected_episode_count_is_not_silently_truncated(tmp_path: Path) -> None:
  root = make_collection(tmp_path / "raw", "install-ram")
  with pytest.raises(ValueError, match="expected 200"):
    converter.run(
      args_for(
        root,
        tmp_path / "ftp1",
        "install-ram",
        "--dry-run",
        "--expected-episodes",
        "200",
      )
    )


@pytest.mark.parametrize("task", ["sponge-grasp", "pick-place", "vase-wipe"])
def test_excluded_tasks_are_not_advertised(task: str) -> None:
  with pytest.raises(SystemExit):
    converter._parse_args(
      ["--input-dir", "/fixture", "--output-dir", "/other", "--task", task]
    )


@pytest.mark.parametrize(
  "attribute,value",
  [("force_unit", "Pa"), ("taxel_unit", "depth_m"), ("is_spatial_estimate", False)],
)
def test_force_proxy_units_and_unlabelled_estimates_rejected(
  tmp_path: Path, attribute: str, value
) -> None:
  root = make_collection(tmp_path / "raw", "install-ram")
  (source,) = converter.discover_sources(root, "install-ram")
  with h5py.File(source.hdf5_path, "r+") as file:
    file["tactile_contact_force"].attrs[attribute] = value
  with pytest.raises(ValueError, match="force_unit|taxel_unit|metadata mismatch"):
    converter.preflight_episode(source, task="install-ram")


def test_nonorthogonal_tangent_basis_rejected(tmp_path: Path) -> None:
  root = make_collection(tmp_path / "raw", "bulb-unscrew")
  (source,) = converter.discover_sources(root, "bulb-unscrew")
  with h5py.File(source.hdf5_path, "r+") as file:
    file["tactile_contact_force/tangent_basis_local"][0, 0] = [1.0, 1.0, 0.0]
  with pytest.raises(ValueError, match="orthonormal"):
    converter.preflight_episode(source, task="bulb-unscrew")


def test_camera_cannot_select_future_state(tmp_path: Path) -> None:
  root = make_collection(tmp_path / "raw", "whiteboard-wipe")
  (source,) = converter.discover_sources(root, "whiteboard-wipe")
  with h5py.File(source.hdf5_path, "r+") as file:
    file["cameras/head/state_index"][1] += 1
  with pytest.raises(ValueError, match="latest-nonfuture"):
    converter.preflight_episode(source, task="whiteboard-wipe")


def test_sidecar_cannot_escape_collection_root(tmp_path: Path) -> None:
  root = make_collection(tmp_path / "raw", "bulb-screw-inplace")
  (source,) = converter.discover_sources(root, "bulb-screw-inplace")
  outside = tmp_path / "outside-sidecar.json"
  outside.write_bytes(source.sidecar_path.read_bytes())
  source.sidecar_path.unlink()
  source.sidecar_path.symlink_to(outside)
  with pytest.raises(ValueError, match="path escapes"):
    converter.discover_sources(root, "bulb-screw-inplace")


def test_sidecar_outcome_cannot_disagree_with_native_outcome(tmp_path: Path) -> None:
  root = make_collection(tmp_path / "raw", "bulb-screw")
  (source,) = converter.discover_sources(root, "bulb-screw")
  sidecar = json.loads(source.sidecar_path.read_text())
  sidecar["outcome"]["success"] = False
  source.sidecar_path.write_text(json.dumps(sidecar))
  with pytest.raises(ValueError, match="outcome differs"):
    converter.discover_sources(root, "bulb-screw")


@pytest.mark.parametrize("task", ["bulb-screw-inplace", "usb-unplug", converter.TASK])
def test_publish_resume_preserves_identity_and_rehashes(
  tmp_path: Path, monkeypatch, task: str, capsys
) -> None:
  pytest.importorskip("zarr")
  root = make_collection(tmp_path / "raw", task)
  output = tmp_path / "ftp1"
  converter.run(args_for(root, output, task, "--verify-source-hash"))
  manifest = output / "conversion_manifest.json"
  before = manifest.read_bytes()
  capsys.readouterr()
  converter.run(args_for(root, output, task, "--resume", "--dry-run", "--plan-json"))
  assert json.loads(capsys.readouterr().out)["published_complete"] is True
  verified = []
  original_verify = converter._verify_source

  def verify(source):
    verified.append(source.episode_index)
    original_verify(source)

  monkeypatch.setattr(converter, "_verify_source", verify)
  converter.run(args_for(root, output, task, "--resume", "--verify-source-hash"))
  assert verified == [0]
  assert manifest.read_bytes() == before
  with pytest.raises(FileExistsError):
    converter.run(args_for(root, output, task, "--dry-run"))
  config_path = output / "dataset_config.json"
  config = json.loads(config_path.read_text())
  config["datasets"][0]["enabled"] = False
  config_path.write_text(json.dumps(config))
  with pytest.raises(ValueError, match="configuration identity"):
    converter.run(args_for(root, output, task, "--resume", "--dry-run"))


def test_multiple_processes_receive_explicit_task(tmp_path: Path) -> None:
  pytest.importorskip("zarr")
  root = make_collection(tmp_path / "raw", "usb-unplug", episodes=2)
  output = tmp_path / "ftp1"
  result = subprocess.run(
    [
      sys.executable,
      "-B",
      str(SCRIPT),
      "--input-dir",
      str(root),
      "--output-dir",
      str(output),
      "--task",
      "usb-unplug",
      "--workers",
      "2",
      "--episodes-per-shard",
      "1",
      "--verify-source-hash",
    ],
    capture_output=True,
    text=True,
    timeout=120,
  )
  assert result.returncode == 0, result.stderr
  manifest = json.loads((output / "conversion_manifest.json").read_text())
  assert manifest["task"] == "usb-unplug"
  assert [Path(record["path"]).name for record in manifest["shards"]] == [
    "usb_unplug_00000.zarr",
    "usb_unplug_00001.zarr",
  ]


def test_resume_rejects_changed_store_attributes_even_with_same_fingerprint(
  tmp_path: Path,
) -> None:
  zarr = pytest.importorskip("zarr")
  root = make_collection(tmp_path / "raw", "bulb-screw-inplace")
  output = tmp_path / "ftp1"
  converter.run(args_for(root, output, "bulb-screw-inplace"))
  store = zarr.open_group(str(output / "bulb_screw_inplace_00000.zarr"), mode="r+")
  store.attrs["controller_target_actions_used"] = True
  with pytest.raises(ValueError, match="changed conversion/source attribute"):
    converter.run(args_for(root, output, "bulb-screw-inplace", "--resume", "--dry-run"))


def test_published_complete_requires_verified_manifest_totals(tmp_path: Path) -> None:
  pytest.importorskip("zarr")
  root = make_collection(tmp_path / "raw", "install-ram")
  output = tmp_path / "ftp1"
  converter.run(args_for(root, output, "install-ram"))
  manifest_path = output / "conversion_manifest.json"
  manifest = json.loads(manifest_path.read_text())
  manifest["frames"] += 1
  manifest_path.write_text(json.dumps(manifest))
  with pytest.raises(ValueError, match="identity differs for frames"):
    converter.run(
      args_for(root, output, "install-ram", "--resume", "--dry-run", "--plan-json")
    )
