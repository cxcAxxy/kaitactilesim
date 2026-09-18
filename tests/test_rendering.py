from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
from kaihand_tactile_env.workcell import rendering as workcell_rendering
from kaihand_tactile_env.workcell.config import CameraConfig


class _FakeScene:
  def __init__(self) -> None:
    self.flags = np.ones(int(mujoco.mjtRndFlag.mjNRNDFLAG), dtype=np.uint8)


class _FakeRenderer:
  instances: list[_FakeRenderer] = []

  def __init__(self, _model: mujoco.MjModel, *, height: int, width: int) -> None:
    self.height = height
    self.width = width
    self.scene = _FakeScene()
    self.update_calls: list[tuple[str, np.ndarray | None]] = []
    self.render_shadow_flags: list[bool] = []
    self.closed = False
    self.instances.append(self)

  def disable_depth_rendering(self) -> None:
    pass

  def disable_segmentation_rendering(self) -> None:
    pass

  def enable_depth_rendering(self) -> None:
    pass

  def enable_segmentation_rendering(self) -> None:
    pass

  def update_scene(
    self,
    _data: mujoco.MjData,
    *,
    camera: str,
    scene_option: mujoco.MjvOption | None,
  ) -> None:
    geomgroup = (
      None
      if scene_option is None
      else np.asarray(scene_option.geomgroup).copy()
    )
    self.update_calls.append((camera, geomgroup))

  def render(self) -> np.ndarray:
    shadow_index = int(mujoco.mjtRndFlag.mjRND_SHADOW)
    self.render_shadow_flags.append(bool(self.scene.flags[shadow_index]))
    return np.zeros((self.height, self.width, 3), dtype=np.uint8)

  def close(self) -> None:
    self.closed = True


def _camera_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
  model = mujoco.MjModel.from_xml_string(
    """
    <mujoco>
      <worldbody>
        <camera name="head" pos="0 0 1"/>
      </worldbody>
    </mujoco>
    """
  )
  return model, mujoco.MjData(model)


def test_renderer_applies_geom_groups_and_disables_shadows(
  monkeypatch: Any,
) -> None:
  _FakeRenderer.instances.clear()
  monkeypatch.setattr(workcell_rendering.mujoco, "Renderer", _FakeRenderer)
  model, data = _camera_model()
  camera = CameraConfig("head", width=8, height=6)

  renderer = workcell_rendering.WorkcellRenderer(
    model,
    (camera,),
    visible_geom_groups=(0, 1),
    shadows=False,
  )
  output = renderer.capture(data, camera)

  assert set(output) == {"rgb", "depth", "segmentation"}
  fake = _FakeRenderer.instances[0]
  assert len(fake.update_calls) == 3
  expected_groups = np.array([1, 1, 0, 0, 0, 0], dtype=np.uint8)
  for captured_camera, geomgroup in fake.update_calls:
    assert captured_camera == "head"
    np.testing.assert_array_equal(geomgroup, expected_groups)
  assert fake.render_shadow_flags == [False, False, False]

  renderer.close()
  assert fake.closed


def test_renderer_defaults_keep_scene_options_and_shadows_unchanged(
  monkeypatch: Any,
) -> None:
  _FakeRenderer.instances.clear()
  monkeypatch.setattr(workcell_rendering.mujoco, "Renderer", _FakeRenderer)
  model, data = _camera_model()
  camera = CameraConfig(
    "head",
    width=8,
    height=6,
    depth=False,
    segmentation=False,
  )

  renderer = workcell_rendering.WorkcellRenderer(model, (camera,))
  renderer.capture(data, camera)

  fake = _FakeRenderer.instances[0]
  assert len(fake.update_calls) == 1
  assert fake.update_calls[0][1] is None
  assert fake.render_shadow_flags == [True]
