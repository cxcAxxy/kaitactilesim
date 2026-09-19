"""The decorative strip must leave the existing USB bore and physics intact."""

import struct
from pathlib import Path

import kaihand_tactile_env
import mujoco
import numpy as np
from kaihand_tactile_env.shared.config import default_model_path


def test_power_strip_is_fixed_display_geometry_with_one_existing_usb_target():
  model = mujoco.MjModel.from_xml_path(str(default_model_path("usb-insert")))
  body = model.body("usb_power_strip_visual")
  assert body.jntnum[0] == 0
  assert body.mass[0] == 0
  geoms = np.flatnonzero(model.geom_bodyid == body.id)
  assert len(geoms) > 20
  assert np.all(model.geom_contype[geoms] == 0)
  assert np.all(model.geom_conaffinity[geoms] == 0)
  assert np.all(model.geom_group[geoms] == 1)
  assert body.id > model.body("usb_plug").id
  assert geoms.min() > model.geom("usb_plug_contact_4_visual").id
  socket = model.body("usb_socket")
  np.testing.assert_allclose(socket.pos, [0.620, -0.180, 0.720])
  assert [
    model.site(i).name
    for i in range(model.nsite)
    if "socket_mouth" in model.site(i).name
  ] == ["usb_socket_mouth"]


def test_power_strip_shell_is_closed_and_does_not_cover_the_active_aperture():
  asset = (
    Path(kaihand_tactile_env.__file__).parent
    / "assets/workcell/meshes/usb_power_strip_shell_visual.STL"
  )
  raw = asset.read_bytes()
  count = struct.unpack_from("<I", raw, 80)[0]
  dtype = np.dtype(
    [("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")]
  )
  assert len(raw) == 84 + count * dtype.itemsize
  faces = np.frombuffer(raw, dtype=dtype, count=count, offset=84)["vertices"].astype(
    float
  )
  _, ids = np.unique(faces.reshape(-1, 3), axis=0, return_inverse=True)
  ids = ids.reshape(-1, 3)
  edges = np.sort(
    np.concatenate([ids[:, [0, 1]], ids[:, [1, 2]], ids[:, [2, 0]]]), axis=1
  )
  _, incidence = np.unique(edges, axis=0, return_counts=True)
  assert np.all(incidence == 2)
  top = faces[np.all(np.isclose(faces[:, :, 2], 0.0388), axis=1), :, :2]
  assert len(top) > 0
  edge = np.roll(top, -1, axis=1) - top
  # Probe the center and near all aperture corners, not just the center ray.
  for x in [-0.0033, 0, 0.0033]:
    for y in [-0.0070, 0, 0.0070]:
      offset = np.array([x, y]) - top
      cross = edge[:, :, 0] * offset[:, :, 1] - edge[:, :, 1] * offset[:, :, 0]
      covered = np.all(cross >= -1e-12, axis=1) | np.all(cross <= 1e-12, axis=1)
      assert not covered.any(), "visual cap occludes the existing mechanical USB bore"
