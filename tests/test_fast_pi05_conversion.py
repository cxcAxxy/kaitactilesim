from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

WORKCELL = Path(__file__).parents[1] / "scripts" / "workcell"
sys.path.insert(0, str(WORKCELL))

import fast_pi05_conversion as fast  # noqa: E402


def test_fast_manifest_schema_requires_explicit_action_contract() -> None:
  assert fast.SCHEMA == "kaihand_fast_pi05_video_v2"


class _Converter:
  REPO_ID = "test/repo"
  TASK_INSTRUCTION = "test instruction"

  @staticmethod
  def _lerobot_features(height, width):
    return {
      "observation.images.head": {
        "dtype": "image",
        "shape": (height, width, 3),
        "names": ["height", "width", "channel"],
      },
      "observation.state": {
        "dtype": "float32",
        "shape": (2,),
        "names": [["a", "b"]],
      },
      "action": {
        "dtype": "float32",
        "shape": (2,),
        "names": [["a", "b"]],
      },
    }


def _plan():
  source = SimpleNamespace(episode_index=42)
  return SimpleNamespace(
    source=source,
    exported_frames=3,
    image_height=4,
    image_width=6,
  )


def test_fast_features_replace_images_with_videos() -> None:
  converter = _Converter()
  fast.install_fast_conversion(converter)
  assert converter._convert is fast._fast_convert
  features = fast._base_features(_plan())
  assert features["observation.images.head"]["dtype"] == "video"
  assert features["observation.state"]["dtype"] == "float32"


def test_episode_arrays_preserve_openpi_alignment() -> None:
  plan = _plan()
  timestamps = np.asarray([1.0, 1.1, 1.2, 1.3])
  state_indices = np.asarray([10, 20, 30, 40])
  positions = np.arange(8, dtype=np.float64).reshape(4, 2)
  actions = positions + 100.0
  arrays = fast._episode_arrays(
    plan,
    (plan, timestamps, state_indices, positions, actions),
    output_episode_index=7,
    global_start_index=50,
  )

  np.testing.assert_array_equal(arrays["observation.state"], positions[:3])
  np.testing.assert_array_equal(arrays["action"], actions[1:])
  np.testing.assert_array_equal(arrays["index"], [50, 51, 52])
  np.testing.assert_array_equal(arrays["episode_index"], [7, 7, 7])
  np.testing.assert_array_equal(arrays["provenance.source_episode_index"], [42, 42, 42])
  np.testing.assert_array_equal(
    arrays["provenance.action_source_state_index"], [20, 30, 40]
  )


def test_source_manifest_rows_preserve_raw_hdf5_mapping() -> None:
  source = SimpleNamespace(
    episode_index=42,
    hdf5=SimpleNamespace(
      path=Path("/raw/bulb.h5"),
      sha256="hdf5-sha256",
      size_bytes=123,
      mtime_ns=1,
      device=2,
      inode=3,
    ),
    sidecar=SimpleNamespace(
      path=Path("/raw/bulb.json"),
      sha256="sidecar-sha256",
      size_bytes=45,
      mtime_ns=4,
      device=5,
      inode=6,
    ),
    task="bulb-screw",
    camera_names=("head", "right_wrist"),
    instruction="screw the bulb",
    repo_id="test/bulb",
    action_contract="recorded-controller-target-v1",
  )
  plan = SimpleNamespace(source=source, exported_frames=1252)

  rows = fast._source_manifest_rows((plan,), verify_source_hash=True)

  assert rows == [
    {
      "output_episode_index": 0,
      "source_episode_index": 42,
      "source_hdf5": "/raw/bulb.h5",
      "source_hdf5_sha256": "hdf5-sha256",
      "source_hdf5_size_bytes": 123,
      "source_sidecar": "/raw/bulb.json",
      "source_hash_verification": "recomputed",
      "task": "bulb-screw",
      "action_contract": "recorded-controller-target-v1",
      "exported_frames_30hz": 1252,
    }
  ]


def test_fast_manifest_uses_openpi_action_contract_authority() -> None:
  contract = "recorded-controller-target-v1"
  converter = _Converter()
  converter._action_contract_manifest = lambda value: {
    "id": value,
    "legacy": False,
    "evaluation_execution": {"physical_substep_interpolation_required": "linear"},
  }
  fast.install_fast_conversion(converter)
  recorded = _plan()
  recorded.source.action_contract = contract

  manifest = fast._conversion_action_contract((recorded,))

  assert manifest["id"] == contract
  assert manifest["legacy"] is False
  assert (
    manifest["evaluation_execution"]["physical_substep_interpolation_required"]
    == "linear"
  )
  legacy = _plan()
  with pytest.raises(ValueError, match="mixes action contracts"):
    fast._conversion_action_contract((recorded, legacy))


def test_fast_manifest_legacy_fallback_is_explicit_and_rejects_new_contract() -> None:
  converter = _Converter()
  fast.install_fast_conversion(converter)
  legacy = _plan()

  manifest = fast._conversion_action_contract((legacy,))

  assert manifest["id"] == fast.LEGACY_ACTION_CONTRACT
  assert manifest["legacy"] is True
  recorded = _plan()
  recorded.source.action_contract = "recorded-controller-target-v1"
  with pytest.raises(ValueError, match="requires an OpenPI manifest authority"):
    fast._conversion_action_contract((recorded,))


@pytest.mark.parametrize("task", ["bulb-screw", "whiteboard-wipe-horizontal-line"])
def test_source_environment_requires_one_bound_mechanics_identity(
  tmp_path: Path, task: str,
) -> None:
  source_path = tmp_path / "source.h5"
  with h5py.File(source_path, "w") as file:
    file.attrs["model_fingerprint"] = "a" * 64
  plan = SimpleNamespace(
    source=SimpleNamespace(
      task=task,
      hdf5=SimpleNamespace(path=source_path),
      metadata={"mechanics_version": "mechanics-v1"},
    )
  )

  assert fast._source_environment((plan,)) == {
    "mechanics_version": "mechanics-v1",
    "model_fingerprint": "a" * 64,
  }

  second_path = tmp_path / "second.h5"
  with h5py.File(second_path, "w") as file:
    file.attrs["model_fingerprint"] = "b" * 64
  second = SimpleNamespace(
    source=SimpleNamespace(
      task=task,
      hdf5=SimpleNamespace(path=second_path),
      metadata={"mechanics_version": "mechanics-v1"},
    )
  )
  with pytest.raises(ValueError, match="mix mechanics/model identities"):
    fast._source_environment((plan, second))

  non_bulb = SimpleNamespace(
    source=SimpleNamespace(task="sponge-grasp", metadata={})
  )
  assert fast._source_environment((non_bulb,)) is None


def _usb_environment_plan(tmp_path: Path, name: str, protocol: object = ...) -> SimpleNamespace:
  from kaihand_tactile_env.tasks.usb_unplug import config

  source_path = tmp_path / f"{name}.h5"
  with h5py.File(source_path, "w") as file:
    file.attrs["model_fingerprint"] = "a" * 64
  metadata = {"mechanics_version": config.MECHANICS_VERSION}
  if protocol is not ...:
    metadata["completion_protocol"] = protocol
  return SimpleNamespace(source=SimpleNamespace(
    task="usb-unplug", hdf5=SimpleNamespace(path=source_path), metadata=metadata,
  ))


def test_usb_source_environment_freezes_current_recorded_completion_protocol(tmp_path: Path):
  from kaihand_tactile_env.tasks.usb_unplug import config

  plan = _usb_environment_plan(tmp_path, "current", config.COMPLETION_PROTOCOL)
  identity = fast._source_environment((plan,))
  assert identity == {
    "mechanics_version": config.MECHANICS_VERSION,
    "model_fingerprint": "a" * 64,
    "completion_protocol": config.COMPLETION_PROTOCOL,
  }
  # The returned frozen identity must not alias a mutable raw metadata object.
  plan.source.metadata["completion_protocol"] = "usb-unplug-legacy-return-release"
  assert identity["completion_protocol"] == config.COMPLETION_PROTOCOL
  with pytest.raises(ValueError, match="completion_protocol"):
    fast._source_environment((plan,))


@pytest.mark.parametrize("protocol", [..., None, "", "usb-unplug-legacy-return-release", True, 2])
def test_usb_source_environment_rejects_legacy_missing_or_unknown_protocol(
  tmp_path: Path, protocol: object,
):
  plan = _usb_environment_plan(tmp_path, "invalid", protocol)
  with pytest.raises(ValueError, match="completion_protocol"):
    fast._source_environment((plan,))


@pytest.mark.parametrize("current_first", [False, True])
def test_usb_source_environment_rejects_mixed_old_and_current_episodes(
  tmp_path: Path, current_first: bool,
):
  from kaihand_tactile_env.tasks.usb_unplug import config

  current = _usb_environment_plan(tmp_path, "current", config.COMPLETION_PROTOCOL)
  legacy = _usb_environment_plan(tmp_path, "legacy")
  plans = (current, legacy) if current_first else (legacy, current)
  with pytest.raises(ValueError, match="legacy or mixed"):
    fast._source_environment(plans)


def test_usb_source_environment_detects_runtime_protocol_drift(tmp_path: Path, monkeypatch):
  from kaihand_tactile_env.tasks.usb_unplug import config

  plan = _usb_environment_plan(tmp_path, "current", config.COMPLETION_PROTOCOL)
  monkeypatch.setattr(config, "COMPLETION_PROTOCOL", "usb-unplug-future-protocol")
  with pytest.raises(ValueError, match="current completion_protocol"):
    fast._source_environment((plan,))


def test_resume_rehashes_committed_episode_artifacts(tmp_path: Path) -> None:
  episode = tmp_path / "episode-000000"
  episode.mkdir()
  artifacts = {}
  for name in ("data.parquet", "stats.json", "head.mp4"):
    path = episode / name
    path.write_bytes(b"original")
    artifacts[name] = {
      "size_bytes": path.stat().st_size,
      "sha256": fast._sha256_file(path),
    }
  source = SimpleNamespace(
    episode_index=42,
    hdf5=SimpleNamespace(sha256="source-hash"),
    camera_names=("head",),
  )
  plan = SimpleNamespace(source=source, exported_frames=3)
  (episode / "done.json").write_text(
    json.dumps(
      {
        "schema": fast.SCHEMA,
        "plan_fingerprint": "plan",
        "output_episode_index": 0,
        "source_episode_index": 42,
        "source_hdf5_sha256": "source-hash",
        "exported_frames": 3,
        "global_start_index": 0,
        "artifacts": artifacts,
      }
    )
  )
  kwargs = dict(
    plan=plan, output_index=0, global_start_index=0, plan_fingerprint="plan"
  )
  assert fast._completed_episode(episode, **kwargs) is not None
  (episode / "data.parquet").write_bytes(b"modified")
  assert fast._completed_episode(episode, **kwargs) is None


def test_plan_fingerprint_tracks_openpi_and_adapter_source(tmp_path: Path) -> None:
  converter_path = tmp_path / "converter.py"
  common_path = tmp_path / "common.py"
  adapter_path = tmp_path / "adapter.py"
  for path in (converter_path, common_path, adapter_path):
    path.write_text("version = 1\n")
  converter = SimpleNamespace(
    __file__=str(converter_path),
    common=SimpleNamespace(__file__=str(common_path)),
  )
  fast.install_fast_conversion(converter, adapter_path=adapter_path)
  before = fast._fingerprint(fast._conversion_source_hashes())
  adapter_path.write_text("version = 2\n")
  assert fast._fingerprint(fast._conversion_source_hashes()) != before
