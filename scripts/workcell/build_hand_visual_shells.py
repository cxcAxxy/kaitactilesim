#!/usr/bin/env python3
"""Build closed display shells from existing CAD vertices; never edit collision assets.

Offline asset authoring only: requires scipy.spatial.ConvexHull and numpy.
Runtime simulation loads the generated STL files without those dependencies.
Distal tactile links/pads are intentionally excluded.
"""

import hashlib
import json
import struct
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

MESH_DIR = (
  Path(__file__).resolve().parents[2] / "src/kaihand_tactile_env/assets/workcell/meshes"
)
FACET = np.dtype(
  [("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")]
)


def build(source, scale=(1.0, 1.0, 1.0)):
  raw = source.read_bytes()
  count = struct.unpack_from("<I", raw, 80)[0]
  if len(raw) != 84 + count * FACET.itemsize:
    raise ValueError(f"not a binary STL: {source}")
  vertices = np.unique(
    np.frombuffer(raw, FACET, count=count, offset=84)["vertices"].reshape(-1, 3), axis=0
  ).astype(np.float64)
  vertices *= np.asarray(scale)
  hull = ConvexHull(vertices)
  faces = vertices[hull.simplices].copy()
  normals = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
  inward = np.einsum("ij,ij->i", normals, hull.equations[:, :3]) < 0
  faces[inward] = faces[inward][:, [0, 2, 1]]
  normals = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
  keep = np.linalg.norm(normals, axis=1) > 1e-16
  faces, normals = faces[keep], normals[keep]
  normals /= np.linalg.norm(normals, axis=1)[:, None]
  output = np.zeros(len(faces), dtype=FACET)
  output["normal"] = normals
  output["vertices"] = faces
  target = source.with_name(source.stem + "_shell_visual.STL")
  header = b"KaiHand closed visual shell; collision mesh unchanged".ljust(80, b"\0")
  target.write_bytes(header + struct.pack("<I", len(faces)) + output.tobytes())
  return {
    "source": str(source.relative_to(MESH_DIR)),
    "output": str(target.relative_to(MESH_DIR)),
    "visual_scale": list(scale),
    "source_sha256": hashlib.sha256(raw).hexdigest(),
    "output_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    "source_faces": count,
    "shell_faces": len(faces),
  }


def main():
  names = ["base_link"] + [f"thumb_link{i}" for i in range(1, 6)]
  names += [
    f"{finger}_link{i}"
    for finger in ("index", "middle", "ring", "pinky")
    for i in range(1, 4)
  ]
  report = [
    build(MESH_DIR / f"hand_{side}_{name}.STL") for side in ("l", "r") for name in names
  ]
  # The fixed wrist cameras lie inside the original decorative flange's
  # envelope. A slimmer closed display cuff gives the existing optical center
  # clearance; the original 31 mm collision capsules and inertia are retained.
  report += [
    build(MESH_DIR / f"tianji_m6/m6/Link7_{side}.STL", scale=(0.72, 1.0, 0.72))
    for side in ("L", "R")
  ]
  (MESH_DIR / "hand_visual_shells.json").write_text(json.dumps(report, indent=2) + "\n")
  print(f"Generated {len(report)} visual shells; original CAD files unchanged.")


if __name__ == "__main__":
  main()
