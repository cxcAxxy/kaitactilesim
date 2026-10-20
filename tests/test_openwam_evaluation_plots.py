import hashlib
import json
from types import SimpleNamespace

import h5py
import mujoco
import numpy as np
import pytest
from kaihand_tactile_env.shared.config import default_model_path, model_fingerprint
from kaihand_tactile_env.shared.contact_tactile import (
  SolverDistributedTactileProvider,
)
from kaihand_tactile_env.shared.openwam_evaluation_plots import (
  RIGHT_FINGERTIP_LINK_NAMES,
  RIGHT_HAND_ACTUATED_JOINT_NAMES,
  SCHEMA_VERSION,
  OpenWAMEvaluationPlots,
  aggregate_fingertip_forces,
  load_openwam_reference,
)
from kaihand_tactile_env.shared.recording import _qpos_names
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


def _legacy_pickplace_dataset(tmp_path):
  """Write the 0920-era schema without native task-space/taxel streams."""
  root = tmp_path / "legacy_lerobot"
  (root / "meta").mkdir(parents=True)
  source = root / "legacy_pickplace.h5"
  model_path = default_model_path("pick-place")
  model = mujoco.MjModel.from_xml_path(str(model_path))
  data = mujoco.MjData(model)
  state_count = 7
  state_times = np.arange(state_count, dtype=np.float64) * 0.01
  qpos = np.broadcast_to(model.qpos0, (state_count, model.nq)).copy()
  camera_indices = np.asarray([0, 2, 4, 6], dtype=np.int64)
  camera_times = state_times[camera_indices]

  data.qpos[:] = qpos[0]
  mujoco.mj_forward(model, data)
  wrist_id = model.site("hand_r_base_link_site").id
  camera_id = model.camera("right_wrist").id
  expected_wrist = np.eye(4)
  expected_wrist[:3, 3] = data.site_xpos[wrist_id]
  expected_wrist[:3, :3] = data.site_xmat[wrist_id].reshape(3, 3)
  right_camera = np.eye(4)
  right_camera[:3, 3] = data.cam_xpos[camera_id]
  right_camera[:3, :3] = data.cam_xmat[camera_id].reshape(3, 3)

  model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
  fingerprint = model_fingerprint(model_path)
  metadata = {
    "scene": "pick-place",
    "side": "right",
    "recording_contract": "pick_place_stable_terminal_v2",
    "model_layout": "task-isolated-v1",
    "base_model_sha256": model_sha256,
    "base_model_fingerprint": fingerprint,
  }
  with h5py.File(source, "w") as file:
    file.attrs["schema_version"] = "kaihand_tactile_episode_v1"
    file.attrs["metadata_json"] = json.dumps(metadata)
    file.attrs["outcome_json"] = json.dumps(
      {"success": True, "placed_in_box": True}
    )
    file.attrs["model_path"] = str(model_path)
    file.attrs["model_sha256"] = model_sha256
    file.attrs["model_fingerprint"] = fingerprint
    file.attrs["physics_hz"] = 500

    state = file.create_group("state")
    state.create_dataset("timestamp", data=state_times)
    state.create_dataset("qpos", data=qpos)
    state.create_dataset(
      "full_qpos_names", data=_qpos_names(model), dtype=h5py.string_dtype()
    )
    state.create_dataset(
      "joint_names",
      data=RIGHT_HAND_ACTUATED_JOINT_NAMES,
      dtype=h5py.string_dtype(),
    )
    joint_addresses = [
      int(model.jnt_qposadr[model.joint(name).id])
      for name in RIGHT_HAND_ACTUATED_JOINT_NAMES
    ]
    state.create_dataset(
      "robot_joint_position", data=qpos[:, joint_addresses]
    )

    head = file.create_group("cameras/head")
    head.create_dataset("timestamp", data=camera_times)
    head.create_dataset("state_index", data=camera_indices)
    right = file.create_group("cameras/right_wrist")
    right.create_dataset("timestamp", data=camera_times)
    right.create_dataset("state_index", data=camera_indices)
    right.create_dataset(
      "world_from_camera",
      data=np.broadcast_to(right_camera, (len(camera_times), 4, 4)),
    )

    model_group = file.create_group("model")
    geom_names = tuple(model.geom(index).name or "" for index in range(model.ngeom))
    model_group.create_dataset(
      "geom_names", data=geom_names, dtype=h5py.string_dtype()
    )
    pad_id = geom_names.index("hand_r_thumb_link6_tactile_pad_col")
    cylinder_id = geom_names.index("cylinder_geom")
    contacts = file.create_group("contacts")
    contacts.create_dataset("frame_start", data=np.arange(state_count))
    contacts.create_dataset("frame_count", data=np.ones(state_count, dtype=np.int32))
    events = contacts.create_group("events")
    events.attrs["wrench_force_unit"] = "N"
    events.attrs["wrench_action"] = "on geom2; negate for action on geom1"
    events.create_dataset("state_index", data=np.arange(state_count))
    events.create_dataset("geom1_id", data=np.full(state_count, pad_id))
    events.create_dataset("geom2_id", data=np.full(state_count, cylinder_id))
    frames = np.broadcast_to(np.eye(3), (state_count, 3, 3)).copy()
    events.create_dataset("frame_world", data=frames)
    wrench = np.zeros((state_count, 6), dtype=np.float64)
    wrench[:, 0] = np.arange(1, state_count + 1)
    wrench[:, 1:3] = (2.0, -1.0)
    events.create_dataset("wrench_contact_on_geom2", data=wrench)
    events.create_dataset("wrench_world_on_geom2", data=wrench)

  (root / "meta/kaihand_source_episodes.jsonl").write_text(
    json.dumps({
      "output_episode_index": 0,
      "source_hdf5": str(source),
      "exported_frames_30hz": 3,
    })
    + "\n",
    encoding="utf-8",
  )
  provider = SolverDistributedTactileProvider(
    model,
    target_geom_names=("cylinder_geom",),
    link_names=RIGHT_FINGERTIP_LINK_NAMES,
  )
  body_id = model.body(RIGHT_FINGERTIP_LINK_NAMES[0]).id
  rotation = data.xmat[body_id].reshape(3, 3)
  tangent_basis_world = np.einsum(
    "ij,kj->ki", rotation, provider.tangent_basis_local[0]
  )
  expected_tangent = np.linalg.norm(
    tangent_basis_world @ np.asarray([0.0, -2.0, 1.0])
  )
  return root, source, expected_wrist, expected_tangent


def _fast_dataset(tmp_path, *, schema="kaihand_fast_pi05_video_v1"):
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
      "schema": schema,
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


@pytest.mark.parametrize(
  "schema", ("kaihand_fast_pi05_video_v1", "kaihand_fast_pi05_video_v2")
)
def test_reference_falls_back_to_fast_conversion_and_raw_summary(tmp_path, schema):
  root, source = _fast_dataset(tmp_path, schema=schema)

  reference = load_openwam_reference(root, 0)

  assert reference.source_hdf5 == source.resolve()
  assert reference.exported_frames_30hz == 3
  np.testing.assert_allclose(reference.wrist_state[:, 0], [20.0, 21.0, 22.0])
  assert reference.state_29.shape == (3, 29)


def test_legacy_pickplace_reconstructs_hash_bound_fk_and_solver_forces(tmp_path):
  root, _source, expected_wrist, expected_tangent = _legacy_pickplace_dataset(
    tmp_path
  )

  reference = load_openwam_reference(root, 0)

  expected_wrist_state = np.concatenate(
    (
      expected_wrist[:3, 3],
      expected_wrist[:3, 0],
      expected_wrist[:3, 1],
    )
  )
  np.testing.assert_allclose(
    reference.wrist_state,
    np.broadcast_to(expected_wrist_state, (3, 9)),
    atol=1.0e-12,
  )
  np.testing.assert_allclose(
    reference.fingertip_normal_force_n[:, 0], [1.0, 3.0, 5.0]
  )
  np.testing.assert_allclose(reference.fingertip_normal_force_n[:, 1:], 0.0)
  np.testing.assert_allclose(
    reference.fingertip_tangent_force_n[:, 0], expected_tangent
  )
  np.testing.assert_allclose(reference.fingertip_tangent_force_n[:, 1:], 0.0)
  assert reference.wrist_source.startswith("hash-bound MuJoCo FK")
  assert reference.tactile_source.startswith("recorded contacts/events")


def test_legacy_pickplace_rejects_model_provenance_or_camera_fk_drift(tmp_path):
  root, source, _expected_wrist, _expected_tangent = _legacy_pickplace_dataset(
    tmp_path
  )
  with h5py.File(source, "r+") as file:
    file.attrs["model_fingerprint"] = "0" * 64
  with pytest.raises(ValueError, match="model provenance disagree"):
    load_openwam_reference(root, 0)

  root, source, _expected_wrist, _expected_tangent = _legacy_pickplace_dataset(
    tmp_path / "camera"
  )
  with h5py.File(source, "r+") as file:
    file["cameras/right_wrist/world_from_camera"][:, 0, 3] += 0.01
  with pytest.raises(ValueError, match="legacy wrist FK disagrees"):
    load_openwam_reference(root, 0)


def test_non_pickplace_legacy_preserves_missing_reference_gap(tmp_path):
  root, source, _expected_wrist, _expected_tangent = _legacy_pickplace_dataset(
    tmp_path
  )
  with h5py.File(source, "r+") as file:
    metadata = json.loads(file.attrs["metadata_json"])
    metadata["scene"] = "bulb-screw"
    file.attrs["metadata_json"] = json.dumps(metadata)

  with pytest.raises(
    ValueError, match="missing required dataset cameras/head/world_from_wrist"
  ):
    load_openwam_reference(root, 0)


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
