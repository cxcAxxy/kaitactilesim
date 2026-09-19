"""Guard the display-only shell boundary and the generated mesh integrity."""

import hashlib
import json
import struct
from pathlib import Path

import mujoco
import numpy as np
from kaihand_tactile_env.shared.config import default_model_path

ASSETS = Path(__file__).resolve().parents[1] / "src/kaihand_tactile_env/assets/workcell"


def test_closed_shells_preserve_original_cad_and_have_no_open_edges():
  rows = json.loads((ASSETS / "meshes/hand_visual_shells.json").read_text())
  dtype = np.dtype([("normal", "<f4", (3,)), ("v", "<f4", (3, 3)), ("a", "<u2")])
  for row in rows:
    assert (
      hashlib.sha256((ASSETS / "meshes" / row["source"]).read_bytes()).hexdigest()
      == row["source_sha256"]
    )
    raw = (ASSETS / "meshes" / row["output"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == row["output_sha256"]
    count = struct.unpack_from("<I", raw, 80)[0]
    data = np.frombuffer(raw, dtype, count=count, offset=84)
    assert np.isfinite(data["v"]).all() and np.isfinite(data["normal"]).all()
    _, indices = np.unique(data["v"].reshape(-1, 3), axis=0, return_inverse=True)
    faces = indices.reshape(-1, 3)
    edges = np.sort(
      np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1
    )
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert np.all(counts == 2), row["output"]


def test_shells_are_visual_only_and_keep_tactile_contact_meshes():
  model = mujoco.MjModel.from_xml_path(str(default_model_path("usb-insert")))
  shell_geoms = []
  for i in range(model.ngeom):
    mesh_id = int(model.geom_dataid[i])
    if model.geom_type[i] != mujoco.mjtGeom.mjGEOM_MESH:
      continue
    name = model.mesh(mesh_id).name
    if name.endswith("_shell_visual"):
      shell_geoms.append(i)
      assert model.geom_contype[i] == model.geom_conaffinity[i] == 0
      assert model.geom_group[i] == 1
      assert "tactile" not in name
    if "tactile_pad_col" in model.geom(i).name:
      assert not name.endswith("_shell_visual")
      assert model.geom_contype[i] != 0
  assert len(shell_geoms) == 38
