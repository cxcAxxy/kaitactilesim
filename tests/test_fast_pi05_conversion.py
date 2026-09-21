from __future__ import annotations

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
