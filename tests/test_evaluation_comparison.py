import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from kaihand_tactile_env.shared import evaluation_comparison
from kaihand_tactile_env.shared.config import FINGERTIP_LINK_NAMES
from kaihand_tactile_env.shared.evaluation_comparison import EvaluationComparisonTrace
from kaihand_tactile_env.shared.openwam_evaluation_plots import (
  RIGHT_FINGERTIP_LINK_NAMES,
  RIGHT_HAND_ACTUATED_JOINT_NAMES,
  OpenWAMReference,
)
from PIL import Image


class _Model:
  def site(self, name):
    assert name == "hand_r_base_link_site"
    return SimpleNamespace(id=1)


class _Provider:
  source = "test_contact_grid"

  def __init__(self):
    self.read_count = 0
    left_links = tuple(
      name for name in FINGERTIP_LINK_NAMES if name.startswith("hand_l_")
    )
    links = (*left_links, *reversed(RIGHT_FINGERTIP_LINK_NAMES))
    normal = np.zeros((len(links), 7, 5))
    tangent = np.zeros((len(links), 7, 5, 2))
    for index, name in enumerate(links):
      if name in RIGHT_FINGERTIP_LINK_NAMES:
        normal[index] = RIGHT_FINGERTIP_LINK_NAMES.index(name) + 1
    # The two signed thumb taxels cancel before the tangent norm is taken.
    thumb = links.index(RIGHT_FINGERTIP_LINK_NAMES[0])
    tangent[thumb, 0, 0] = [3.0, 4.0]
    tangent[thumb, 0, 1] = [-3.0, -4.0]
    index = links.index(RIGHT_FINGERTIP_LINK_NAMES[1])
    tangent[index, :, :, 0] = 1.0
    tangent[index, :, :, 1] = -2.0
    self.sample = SimpleNamespace(
      link_names=links,
      normal_taxel_force_n=normal,
      tangent_taxel_force_n=tangent,
    )

  def read(self, data):
    self.read_count += 1
    return self.sample


def _simulation():
  data = SimpleNamespace(
    time=0.0,
    site_xpos=np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]),
    site_xmat=np.tile(np.eye(3).reshape(1, 9), (2, 1)),
    qpos=np.arange(20, dtype=np.float64),
  )
  return SimpleNamespace(
    model=_Model(),
    data=data,
    _qpos_address={name: index for index, name in enumerate(RIGHT_HAND_ACTUATED_JOINT_NAMES)},
  )


def _reference(tmp_path):
  state = np.tile([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], (3, 1))
  return OpenWAMReference(
    dataset_root=tmp_path,
    output_episode_index=0,
    source_hdf5=tmp_path / "source.h5",
    exported_frames_30hz=3,
    source_start_time_s=100.0,
    time_s=np.array([0.0, 0.1, 0.2]),
    wrist_state=state,
    hand_joint_position=np.tile(np.arange(20, dtype=np.float64), (3, 1)),
    fingertip_normal_force_n=np.zeros((3, 5)),
    fingertip_tangent_force_n=np.zeros((3, 5)),
    joint_names=RIGHT_HAND_ACTUATED_JOINT_NAMES,
  )


def test_capture_saves_control_step_state_and_raw_fingertip_grids(tmp_path):
  simulation = _simulation()
  provider = _Provider()
  trace = EvaluationComparisonTrace(tactile_provider=provider)

  assert trace.capture(simulation)
  assert not trace.capture(simulation)
  assert provider.read_count == 1
  simulation.data.time = 1.0 / 30.0
  simulation.data.qpos[0] = 42.0
  assert trace.capture(simulation)

  metadata = trace.finish(tmp_path)
  assert metadata["status"] == "reference_unavailable"
  assert metadata["reason"] == "reference_dataset was not provided"
  assert metadata["rollout"]["samples"] == 2
  assert metadata["rollout"]["observed_hz"] == pytest.approx(30.0)
  assert metadata["tactile"]["source"] == provider.source
  assert list(tmp_path.glob("*.png")) == []
  with np.load(tmp_path / "evaluation_rollout_trace.npz") as saved:
    assert saved["state_29"].shape == (2, 29)
    np.testing.assert_allclose(
      saved["state_29"][0, :9], [1, 2, 3, 1, 0, 0, 0, 1, 0]
    )
    assert saved["state_29"][1, 9] == 42.0
    assert saved["normal_taxel_force_n"].shape == (2, 5, 7, 5)
    assert saved["tangent_taxel_force_n"].shape == (2, 5, 7, 5, 2)
    np.testing.assert_allclose(saved["fingertip_normal_force_n"][0], [35, 70, 105, 140, 175])
    assert saved["fingertip_tangent_force_n"][0, 0] == pytest.approx(0.0)
    assert saved["fingertip_tangent_force_n"][0, 1] == pytest.approx(np.hypot(35, -70))
  assert json.loads((tmp_path / "evaluation_comparison.json").read_text())["status"] == "reference_unavailable"


def test_video_frame_reuses_canonical_ten_finger_grids_without_provider_read(tmp_path):
  simulation = _simulation()
  provider = _Provider()
  trace = EvaluationComparisonTrace(tactile_provider=provider)
  link_order = [provider.sample.link_names.index(name) for name in FINGERTIP_LINK_NAMES]
  video_normal = provider.sample.normal_taxel_force_n[link_order]
  video_tangent = provider.sample.tangent_taxel_force_n[link_order]
  trace.capture(
    simulation,
    normal_taxel_force_n=video_normal,
    tangent_taxel_force_n=video_tangent,
  )
  assert provider.read_count == 0
  simulation.data.time = 1.0 / 30.0
  trace.capture(simulation)
  assert provider.read_count == 1
  trace.finish(tmp_path)
  with np.load(tmp_path / "evaluation_rollout_trace.npz") as saved:
    np.testing.assert_allclose(
      saved["normal_taxel_force_n"][0], saved["normal_taxel_force_n"][1]
    )


def test_reference_available_writes_three_generic_comparison_plots(tmp_path, monkeypatch):
  simulation = _simulation()
  trace = EvaluationComparisonTrace(tactile_provider=_Provider())
  trace.capture(simulation)
  simulation.data.time = 1.0 / 30.0
  trace.capture(simulation)
  reference = _reference(tmp_path)
  monkeypatch.setattr(evaluation_comparison, "load_openwam_reference", lambda *a, **k: reference)

  metadata = trace.finish(tmp_path, reference_dataset=tmp_path)

  assert metadata["status"] == "ok"
  assert metadata["reference"]["samples"] == 3
  for key in ("right_wrist_state", "right_hand_actuated_dof", "right_fingertip_tactile"):
    artifact = Path(metadata["artifacts"][key])
    assert artifact.name == f"evaluation_{key}.png"
    with Image.open(artifact) as image:
      assert image.width > 1000
      assert image.height > 1000
  assert (tmp_path / "evaluation_comparison.json").exists()


def test_missing_reference_is_reported_but_corrupt_reference_raises(tmp_path, monkeypatch):
  simulation = _simulation()
  absent = EvaluationComparisonTrace(tactile_provider=_Provider())
  absent.capture(simulation)
  missing = tmp_path / "missing-dataset"
  metadata = absent.finish(tmp_path / "absent", reference_dataset=missing)
  assert metadata["status"] == "reference_unavailable"
  assert str(missing) in metadata["reason"]

  corrupt = EvaluationComparisonTrace(tactile_provider=_Provider())
  corrupt.capture(simulation)
  monkeypatch.setattr(
    evaluation_comparison,
    "load_openwam_reference",
    lambda *a, **k: (_ for _ in ()).throw(ValueError("invalid camera alignment")),
  )
  with pytest.raises(ValueError, match="invalid camera alignment"):
    corrupt.finish(tmp_path / "corrupt", reference_dataset=tmp_path)
  saved = json.loads((tmp_path / "corrupt/evaluation_comparison.json").read_text())
  assert saved["status"] == "error"
  assert (tmp_path / "corrupt/evaluation_rollout_trace.npz").exists()


def test_backward_time_is_rejected_and_empty_rollout_is_marked(tmp_path):
  simulation = _simulation()
  trace = EvaluationComparisonTrace(tactile_provider=_Provider())
  trace.capture(simulation, timestamp=0.5)
  with pytest.raises(ValueError, match="increasing"):
    trace.capture(simulation, timestamp=0.4)
  empty = EvaluationComparisonTrace(tactile_provider=_Provider())
  metadata = empty.finish(tmp_path / "empty")
  assert metadata["status"] == "no_samples"
  with np.load(tmp_path / "empty/evaluation_rollout_trace.npz") as saved:
    assert saved["state_29"].shape == (0, 29)
