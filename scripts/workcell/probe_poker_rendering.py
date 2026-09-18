#!/usr/bin/env python3
"""Small, serial render-only diagnostic using archived poses, never mj_step."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--source", required=True, type=Path)
  parser.add_argument("--output-dir", required=True, type=Path)
  parser.add_argument("--mode", choices=("hardware", "software"), required=True)
  parser.add_argument("--render-threads", choices=(1, 2), type=int, default=1)
  parser.add_argument("--frames", type=int, choices=(1, 2, 3), default=3)
  args = parser.parse_args()
  if args.output_dir.exists():
    parser.error("diagnostic output already exists")
  for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
  os.environ["MUJOCO_GL"] = "egl"
  os.environ["PYOPENGL_PLATFORM"] = "egl"
  os.environ["LP_NUM_THREADS"] = str(args.render_threads)
  os.environ["KAIHAND_RENDER_BACKEND"] = args.mode
  if args.mode == "software":
    os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
    os.environ["GALLIUM_DRIVER"] = "llvmpipe"
  else:
    os.environ.pop("LIBGL_ALWAYS_SOFTWARE", None)
    os.environ.pop("GALLIUM_DRIVER", None)
  import h5py
  import mujoco
  import numpy as np
  from kaihand_tactile_env.shared.config import default_model_path
  from kaihand_tactile_env.shared.render_backend import prepare_render_backend
  from OpenGL import GL
  from PIL import Image

  args.output_dir.mkdir(parents=True)
  started = time.perf_counter()
  report = {
    "schema_version": "poker-render-only-diagnostic-v1",
    "source": str(args.source.resolve()), "requested_mode": args.mode,
    "render_threads": args.render_threads, "physics_steps": 0,
    "raw_modified": False, "completed": False,
    "pose_semantics": "archived post-step qpos; diagnostic only, not replacement training RGB",
    "hdf5_timing_scope": "memory-backed compressed HDF5; excludes physical disk latency",
  }
  try:
    prepare_render_backend()
    report['egl_device_id'] = os.environ.get('MUJOCO_EGL_DEVICE_ID')
    model = mujoco.MjModel.from_xml_path(str(default_model_path("poker-draw")))
    data = mujoco.MjData(model)
    with h5py.File(args.source, "r") as source, mujoco.Renderer(
      model, width=320, height=240
    ) as renderer, h5py.File("in-memory-render-probe", "w", driver="core", backing_store=False) as memory:
      if source["state/qpos"].shape[1] != model.nq:
        raise ValueError("source and model qpos dimensions differ")
      renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
      report["gl_renderer"] = GL.glGetString(GL.GL_RENDERER).decode()
      report["gl_vendor"] = GL.glGetString(GL.GL_VENDOR).decode()
      report["gl_version"] = GL.glGetString(GL.GL_VERSION).decode()
      report["software_renderer"] = any(
        word in report["gl_renderer"].lower() for word in ("llvmpipe", "softpipe", "swrast", "software")
      )
      if args.mode == "hardware" and report["software_renderer"]:
        raise RuntimeError("hardware requested but EGL selected a software renderer")
      if args.mode == "software" and not report["software_renderer"]:
        raise RuntimeError("software requested but EGL selected a hardware renderer")
      indices = np.linspace(0, len(source["state/qpos"]) - 1, args.frames, dtype=int)
      ds = memory.create_dataset("rgb", shape=(0, 240, 320, 3), maxshape=(None, 240, 320, 3),
                                 dtype="u1", chunks=(1, 240, 320, 3), compression="gzip", compression_opts=1, shuffle=True)
      rows = []
      for i, index in enumerate(indices):
        data.qpos[:] = source["state/qpos"][index]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_camlight(model, data)
        begin = time.perf_counter()
        renderer.update_scene(data, camera="head")
        updated = time.perf_counter()
        rgb = renderer.render().copy()
        rendered = time.perf_counter()
        ds.resize(i + 1, axis=0)
        ds[i] = rgb
        memory.flush()
        written = time.perf_counter()
        Image.fromarray(rgb).save(args.output_dir / f"state_{index:06d}.png")
        rows.append({"state_index": int(index), "scene_update_seconds": updated - begin,
                     "render_readback_seconds": rendered - updated,
                     "hdf5_memory_write_seconds": written - rendered,
                     "pixel_mean": float(rgb.mean()), "pixel_std": float(rgb.std())})
      report["frames"] = rows
      report["warm_render_seconds_mean"] = float(np.mean([
        row["scene_update_seconds"] + row["render_readback_seconds"] for row in rows[1:]
      ])) if len(rows) > 1 else None
      report["completed"] = True
  except Exception as error:
    report["error"] = f"{type(error).__name__}: {error}"
  report["wall_seconds"] = time.perf_counter() - started
  report["probe_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
  with (args.output_dir / "report.json").open("x") as stream:
    json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
    stream.write("\n")
  print(json.dumps(report, ensure_ascii=False), flush=True)
  return 0 if report["completed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
