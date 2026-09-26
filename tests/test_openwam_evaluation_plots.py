import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from kaihand_tactile_env.shared.openwam_evaluation_plots import (
  RIGHT_FINGERTIP_LINK_NAMES,
  RIGHT_HAND_ACTUATED_JOINT_NAMES,
  SCHEMA_VERSION,
  OpenWAMEvaluationPlots,
  aggregate_fingertip_forces,
  load_openwam_reference,
)
from PIL import Image


def _write_source(path, *, offset):
  state_count = 7
  camera_count = 4
  state_times = np.arange(state_count, dtype=np.float64) * 0.05
  camera_indices = np.array([0, 2, 4, 6], dtype=np.int64)
  camera_times = state_times[camera_indices]
  joint_names = ("unrelated_joint", *reversed(RIGHT_HAND_ACTUATED_JOINT_NAMES))
  tactile_names = (
    "hand_l_thumb_link6",
    "hand_l_index_link4",
    "hand_l_middle_link4",
    "hand_l_ring_link4",
    "hand_l_pinky_link4",
    *reversed(RIGHT_FINGERTIP_LINK_NAMES),
  )

  with h5py.File(path, "w") as file:
    state = file.create_group("state")
    state.create_dataset("timestamp", data=state_times)
    state.create_dataset(
      "joint_names", data=joint_names, dtype=h5py.string_dtype()
    )
    position = np.empty((state_count, len(joint_names)), dtype=np.float64)
    for row in range(state_count):
      position[row] = offset + 100 * row + np.arange(len(joint_names))
    state.create_dataset("robot_joint_position", data=position)

    head = file.create_group("cameras/head")
    head.attrs["side_names_json"] = json.dumps(["left", "right"])
    head.attrs["wrist_sites_json"] = json.dumps(
      ["hand_l_base_link_site", "hand_r_base_link_site"]
    )
    head.create_dataset("timestamp", data=camera_times)
    head.create_dataset("state_index", data=camera_indices)
    wrists = np.broadcast_to(np.eye(4), (camera_count, 2, 4, 4)).copy()
    wrists[:, 1, 0, 3] = offset + np.arange(camera_count)
    wrists[:, 1, 1, 3] = -0.25
    head.create_dataset("world_from_wrist", data=wrists)

    tactile = file.create_group("tactile_contact_force")
    tactile.create_dataset(
      "link_names", data=tactile_names, dtype=h5py.string_dtype()
    )
    normal = np.zeros((state_count, len(tactile_names), 7, 5))
    tangent = np.zeros((*normal.shape, 2))
    for link_index, name in enumerate(tactile_names):
      if name not in RIGHT_FINGERTIP_LINK_NAMES:
        continue
      finger_index = RIGHT_FINGERTIP_LINK_NAMES.index(name)
      for state_index in range(state_count):
        normal[state_index, link_index] = (
          offset + finger_index + state_index
        ) / 35.0
        tangent[state_index, link_index, ..., 0] = (finger_index + 1) / 35.0
        tangent[state_index, link_index, ..., 1] = -(finger_index + 1) / 70.0
    tactile.create_dataset("normal_taxel_force_n", data=normal)
    tactile.create_dataset("tangent_taxel_force_n", data=tangent)


def _dataset(tmp_path):
  root = tmp_path / "lerobot"
  (root / "meta").mkdir(parents=True)
  selected = root / "selected.h5"
  decoy = root / "decoy.h5"
  _write_source(selected, offset=10.0)
  _write_source(decoy, offset=90.0)
  rows = (
    {
      "output_episode_index": 7,
      "source_hdf5": str(decoy),
      "exported_frames_30hz": 3,
    },
    {
      "output_episode_index": 0,
      "source_hdf5": "selected.h5",
      "exported_frames_30hz": 3,
    },
  )
  (root / "meta/kaihand_source_episodes.jsonl").write_text(
    "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
  )
  return root


def _fast_dataset(tmp_path):
  root = tmp_path / "fast_lerobot"
  (root / "meta").mkdir(parents=True)
  raw = tmp_path / "raw_collection"
  source = (
    raw
    / "bulb-screw/000042/attempt_003/data/raw/light_bulb_000000.h5"
  )
  source.parent.mkdir(parents=True)
  _write_source(source, offset=20.0)
  relative = source.relative_to(raw).as_posix()
  (raw / "summary.json").write_text(
    json.dumps({
      "episodes": [
        {
          "task": "bulb-screw",
          "episode_index": 42,
          "attempt_index": 3,
          "status": "success",
          "hdf5": [relative],
        }
      ]
    }),
    encoding="utf-8",
  )
  (root / "meta/kaihand_fast_pi05_conversion.json").write_text(
    json.dumps({
      "schema": "kaihand_fast_pi05_video_v1",
      "input_dir": str(raw),
      "task": "bulb-screw",
      "episodes": [
        {
          "output_episode_index": 0,
          "source_episode_index": 42,
          "frames": 3,
        }
      ],
    }),
    encoding="utf-8",
  )
  return root, source


def test_tangent_components_are_summed_before_resultant_magnitude():
  normal = np.ones((5, 7, 5))
  tangent = np.zeros((5, 7, 5, 2))
  tangent[0, 0, 0] = [2.0, 3.0]
  tangent[0, 0, 1] = [-2.0, -3.0]
  tangent[1, ..., 0] = 1.0
  tangent[1, ..., 1] = -2.0

  fn, ft = aggregate_fingertip_forces(normal, tangent)

  np.testing.assert_allclose(fn, 35.0)
  assert ft[0] == pytest.approx(0.0)
  assert ft[1] == pytest.approx(np.hypot(35.0, -70.0))


def test_reference_uses_manifest_episode_mapping_and_camera_observations(tmp_path):
  root = _dataset(tmp_path)

  reference = load_openwam_reference(root, 0)

  assert reference.source_hdf5 == (root / "selected.h5").resolve()
  assert reference.exported_frames_30hz == 3
  np.testing.assert_allclose(reference.time_s, [0.0, 0.1, 0.2])
  np.testing.assert_allclose(reference.wrist_state[:, 0], [10.0, 11.0, 12.0])
  np.testing.assert_allclose(
    reference.wrist_state[:, 3:],
    np.tile([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], (3, 1)),
  )
  available = ("unrelated_joint", *reversed(RIGHT_HAND_ACTUATED_JOINT_NAMES))
  columns = np.array([available.index(name) for name in RIGHT_HAND_ACTUATED_JOINT_NAMES])
  expected = 10.0 + 100 * np.array([0, 2, 4])[:, None] + columns
  np.testing.assert_allclose(reference.hand_joint_position, expected)
  np.testing.assert_allclose(
    reference.fingertip_normal_force_n[:, 0], [10.0, 12.0, 14.0]
  )
  assert reference.state_29.shape == (3, 29)


def test_reference_falls_back_to_fast_conversion_and_raw_summary(tmp_path):
  root, source = _fast_dataset(tmp_path)

  reference = load_openwam_reference(root, 0)

  assert reference.source_hdf5 == source.resolve()
  assert reference.exported_frames_30hz == 3
  np.testing.assert_allclose(reference.wrist_state[:, 0], [20.0, 21.0, 22.0])
  assert reference.state_29.shape == (3, 29)


def test_canonical_source_manifest_takes_priority_over_fast_conversion(tmp_path):
  root = _dataset(tmp_path)
  (root / "meta/kaihand_fast_pi05_conversion.json").write_text(
    "{\"schema\": \"wrong-and-must-not-be-read\"}", encoding="utf-8"
  )

  reference = load_openwam_reference(root, 0)

  assert reference.source_hdf5 == (root / "selected.h5").resolve()


def test_fast_conversion_rejects_ambiguous_successful_raw_episode(tmp_path):
  root, source = _fast_dataset(tmp_path)
  summary_path = source.parents[5] / "summary.json"
  summary = json.loads(summary_path.read_text(encoding="utf-8"))
  summary["episodes"].append(dict(summary["episodes"][0]))
  summary_path.write_text(json.dumps(summary), encoding="utf-8")

  with pytest.raises(ValueError, match="maps to 2 successful raw records"):
    load_openwam_reference(root, 0)


class _Provider:
  def __init__(self, sample):
    self.sample = sample
    self.seen_data = None

  def read(self, data):
    self.seen_data = data
    return self.sample


def test_capture_exports_three_pngs_and_metadata(tmp_path):
  root = _dataset(tmp_path)
  links = (
    "hand_l_thumb_link6",
    "hand_l_index_link4",
    *reversed(RIGHT_FINGERTIP_LINK_NAMES),
  )
  normal = np.zeros((len(links), 7, 5))
  tangent = np.zeros((len(links), 7, 5, 2))
  for index, name in enumerate(links):
    if name in RIGHT_FINGERTIP_LINK_NAMES:
      normal[index] = RIGHT_FINGERTIP_LINK_NAMES.index(name) + 1
  sample = SimpleNamespace(
    link_names=links,
    normal_taxel_force_n=normal,
    tangent_taxel_force_n=tangent,
  )
  provider = _Provider(sample)
  plots = OpenWAMEvaluationPlots.from_dataset(root, 0, provider)
  data = object()
  simulation = SimpleNamespace(data=data)

  plots.capture(4.0, plots.reference.state_29[0], simulation=simulation)
  plots.capture(
    4.1,
    plots.reference.state_29[1],
    normal_taxel_force_n=np.zeros((5, 7, 5)),
    tangent_taxel_force_n=np.zeros((5, 7, 5, 2)),
  )
  output = tmp_path / "evaluation"
  output.mkdir()
  metadata = plots.finish(output)

  assert provider.seen_data is data
  assert metadata["schema"] == SCHEMA_VERSION
  assert metadata["state"]["right_hand_actuated_dof"] == 20
  assert "signed taxels cancel" in metadata["tactile"]["Ft"]
  assert len(list(output.glob("*.png"))) == 3
  for key in (
    "right_wrist_state",
    "right_hand_actuated_dof",
    "right_fingertip_tactile",
  ):
    artifact = metadata["artifacts"][key]
    with Image.open(artifact) as image:
      assert image.width > 1000
      assert image.height > 1000
  saved = json.loads(
    (output / "openwam_evaluation_plots.json").read_text(encoding="utf-8")
  )
  assert saved["styles"] == {
    "reference": "solid",
    "evaluation_rollout": "dashed",
    "time_axis": "elapsed seconds; each sequence starts at zero",
  }


def test_capture_rejects_nonincreasing_time_and_bad_state(tmp_path):
  plots = OpenWAMEvaluationPlots.from_dataset(_dataset(tmp_path))
  normal = np.zeros((5, 7, 5))
  tangent = np.zeros((5, 7, 5, 2))

  with pytest.raises(ValueError, match=r"shape \(29,\)"):
    plots.capture(
      0.0,
      np.zeros(28),
      normal_taxel_force_n=normal,
      tangent_taxel_force_n=tangent,
    )
  plots.capture(
    0.0,
    plots.reference.state_29[0],
    normal_taxel_force_n=normal,
    tangent_taxel_force_n=tangent,
  )
  with pytest.raises(ValueError, match="strictly increasing"):
    plots.capture(
      0.0,
      plots.reference.state_29[0],
      normal_taxel_force_n=normal,
      tangent_taxel_force_n=tangent,
    )
