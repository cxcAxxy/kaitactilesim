from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

WORKCELL = Path(__file__).parents[1] / "scripts" / "workcell"
sys.path.insert(0, str(WORKCELL))

import fast_pi05_conversion as fast  # noqa: E402


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
  np.testing.assert_array_equal(
    arrays["provenance.source_episode_index"], [42, 42, 42]
  )
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
      "exported_frames_30hz": 1252,
    }
  ]


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
  (episode / "done.json").write_text(json.dumps({
    "schema": fast.SCHEMA,
    "plan_fingerprint": "plan",
    "output_episode_index": 0,
    "source_episode_index": 42,
    "source_hdf5_sha256": "source-hash",
    "exported_frames": 3,
    "global_start_index": 0,
    "artifacts": artifacts,
  }))
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
