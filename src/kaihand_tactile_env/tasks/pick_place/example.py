"""Record and publish one pick/place episode with an offline video/force review."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw

from ...shared.config import CameraConfig, WorkcellConfig
from ...shared.contact_tactile import SolverDistributedTactileProvider
from ...shared.poker_review import (
  _probe_video,
  _sha256,
  _validate_probe,
  compose_review_frame,
  plan_review_frames,
)
from ...shared.recording import (
  EpisodeRecorder,
  _append,
  _stream,
  validate_episode,
  wait_until_object_stable,
)
from ...shared.rendering import WorkcellRenderer
from ...shared.simulation import ArmHandSimulation
from ...shared.tactile import RIGHT_FINGERTIP_LINK_NAMES
from ...shared.task_video import _FfmpegPipeWriter
from .task import KnownStateGraspPlanner, PickPlaceExecutor, cylinder_is_in_box

FINGERS = ("thumb", "index", "middle", "ring", "little")
CAMERAS = ("head", "right_wrist", "front", "overhead")


def _json(path, value):
  path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


class PickPlaceExampleRecorder(EpisodeRecorder):
  """Add measured pad forces without changing the production controller clock."""

  def _initialize(self, metadata):
    self.contact_force_provider = SolverDistributedTactileProvider(
      self.sim.model,
      self.sim.genesis_probe_layout,
      link_names=RIGHT_FINGERTIP_LINK_NAMES,
    )
    super()._initialize(metadata)
    force = self._file["tactile_contact_force"]
    _stream(force, "timestamp", (), np.float64)
    force.attrs["timestamp_reference"] = "/tactile_contact_force/timestamp"
    force.attrs["clock_semantics"] = (
      "Initial reset observations at t=0; subsequent cached solver forces and "
      "camera FK at data.time - physics timestep, before integration. "
      "State qpos/qvel timestamps describe post-integration state."
    )
    for camera in self.config.cameras:
      _stream(self._file[f"cameras/{camera.name}"], "pose_timestamp", (), np.float64)
    self._progress_time = 0.0

  def _solver_time(self):
    return max(0.0, float(self.sim.data.time) - self.sim.timestep)

  def _record_state(self, phase):
    super()._record_state(phase)
    _append(self._file["tactile_contact_force/timestamp"], self._solver_time())

  def _capture_cameras(self):
    super()._capture_cameras()
    for camera in self.config.cameras:
      _append(self._file[f"cameras/{camera.name}/pose_timestamp"], self._solver_time())

  def observe(self, simulation, phase):
    super().observe(simulation, phase)
    if simulation.data.time >= self._progress_time:
      print(f"recording t={simulation.data.time:.2f}s phase={phase}", flush=True)
      self._progress_time = simulation.data.time + 2.0


def _curves(file, destination):
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  destination.mkdir()
  group = file["tactile_contact_force"]
  t = group["timestamp"][:]
  fn = group["normal_taxel_force_n"][:].sum(axis=(-2, -1))
  xy = group["tangent_taxel_force_n"][:].sum(axis=(-3, -2))
  ft = np.linalg.norm(xy, axis=-1)
  np.testing.assert_allclose(fn, group["normal_force_n"][:], atol=1e-10)
  np.testing.assert_allclose(xy, group["tangent_force_n"][:], atol=1e-10)
  if not np.isfinite(fn).all() or not np.isfinite(xy).all():
    raise ValueError("nonfinite force samples")
  phases = file["commands/phase"].asstr()[:]
  with (destination / "right_hand_forces.csv").open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(
      ["time_s", "state_time_s", "phase"]
      + [
        f"{finger}_{quantity}_n"
        for finger in FINGERS
        for quantity in ("normal", "tangent", "tangent_x", "tangent_y")
      ]
    )
    for k in range(len(t)):
      writer.writerow(
        [t[k], file["state/timestamp"][k], phases[k]]
        + [value for j in range(5) for value in (fn[k, j], ft[k, j], *xy[k, j])]
      )
  fig, axes = plt.subplots(5, 2, figsize=(14, 12), sharex=True)
  changes = np.flatnonzero(phases[1:] != phases[:-1]) + 1
  for j, finger in enumerate(FINGERS):
    for column, (values, label) in enumerate(((fn, "Normal"), (ft, "Tangential"))):
      ax = axes[j, column]
      ax.plot(t, values[:, j], linewidth=0.8)
      ax.set_ylabel(f"{finger} (N)")
      ax.grid(alpha=0.25)
      for k in changes:
        ax.axvline(t[k], color="grey", linewidth=0.5, alpha=0.4)
      if j == 0:
        ax.set_title(label)
      if j == 4:
        ax.set_xlabel("Simulation time (s)")
  fig.suptitle("Pick and place | right fingertip contact forces | 100 Hz, unfiltered")
  fig.tight_layout()
  fig.savefig(destination / "right_hand_force_curves.png", dpi=150)
  fig.savefig(destination / "right_hand_force_curves.pdf")
  plt.close(fig)
  stats = {
    "sample_count": len(t),
    "force_hz": 100,
    "fingers": FINGERS,
    "source": str(file.attrs["contact_force_source"]),
    "normal_peak_n": fn.max(axis=0).tolist(),
    "tangent_peak_n": ft.max(axis=0).tolist(),
    "normal_mean_n": fn.mean(axis=0).tolist(),
    "tangent_mean_n": ft.mean(axis=0).tolist(),
    "phase_start_s": [
      {"phase": phases[k], "time_s": float(t[k])} for k in np.r_[0, changes]
    ],
    "tangent_definition": "norm of sum of signed local tangential force vectors",
    "filter": "none",
  }
  _json(destination / "force_statistics.json", stats)
  return stats


def export_example(source: Path, root: Path):
  """All review products are derived from saved observations, never a replay."""
  review = root / "review"
  review.mkdir()
  for name in (*CAMERAS, "normal", "tangent", "composite"):
    (review / "frames" / name).mkdir(parents=True)
  (review / "raw").mkdir()
  with h5py.File(source, "r") as file:
    stats = _curves(file, root / "curves")
    frames = plan_review_frames(file, fps=10, second_camera="right_wrist")
    force = file["tactile_contact_force"]
    # One fixed full-range scale for the whole episode; data remain unmodified.
    normal_scale = max(0.1, float(force["normal_taxel_force_n"][:].max()))
    tangent_scale = max(
      0.02, float(np.linalg.norm(force["tangent_taxel_force_n"][:], axis=-1).max())
    )
    writers = {}
    try:
      for name in ("review", "scene"):
        writers[name] = _FfmpegPipeWriter(
          review / f"{name}.mp4", fps=10, width=1280, height=720
        )
      rows = []
      for frame in frames:
        k = frame.tactile_index
        normal = force["normal_taxel_force_n"][k]
        tangent = force["tangent_taxel_force_n"][k]
        phase = file["commands/phase"].asstr()[k]
        rgb = {
          name: file[f"cameras/{name}/rgb"][frame.camera_index] for name in CAMERAS
        }
        composite, panels = compose_review_frame(
          rgb["head"],
          rgb["right_wrist"],
          normal,
          tangent,
          frame=frame,
          width=1280,
          height=720,
          phase=phase,
          heading="PICK AND PLACE",
          second_camera_label="RIGHT WRIST",
          normal_scale_n=normal_scale,
          tangent_scale_n=tangent_scale,
        )
        stem = f"{frame.output_index:06d}"
        for name, array in rgb.items():
          Image.fromarray(array).save(review / "frames" / name / f"{stem}.png")
        for name, panel in {**panels, "composite": composite}.items():
          panel.save(review / "frames" / name / f"{stem}.png")
        np.savez_compressed(
          review / "raw" / f"{stem}.npz",
          normal_taxel_force_n=normal,
          tangent_taxel_force_n=tangent,
        )
        writers["review"].write(np.asarray(composite))
        scene = Image.new("RGB", (1280, 720), (15, 16, 20))
        draw = ImageDraw.Draw(scene)
        for j, name in enumerate(CAMERAS):
          x, y = (j % 2) * 640, (j // 2) * 360
          scene.paste(Image.fromarray(rgb[name]), (x, y))
          draw.rectangle((x, y, x + 640, y + 20), fill=(15, 16, 20))
          draw.text(
            (x + 8, y + 4),
            f"{name.upper()} | {frame.camera_pose_timestamp_s:.3f}s | {phase}",
            fill="white",
          )
        writers["scene"].write(np.asarray(scene))
        rows.append({**asdict(frame), "phase": phase})
      for writer in writers.values():
        writer.finish()
    except BaseException:
      for writer in writers.values():
        writer.abort()
      raise
    with (review / "frames.csv").open("w", newline="") as stream:
      writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
      writer.writeheader()
      writer.writerows(rows)
    probes = {}
    for name in writers:
      probes[name] = _probe_video(review / f"{name}.mp4")
      _validate_probe(probes[name], count=len(frames), fps=10, width=1280, height=720)
    summary = {
      "source": "raw/episode_000000_cylinder_right.h5",
      "source_sha256": _sha256(source),
      "frame_count": len(frames),
      "video_fps": 10,
      "cameras": CAMERAS,
      "normal_taxel_scale_n": normal_scale,
      "tangent_taxel_scale_n": tangent_scale,
      "max_tactile_age_s": max(f.tactile_age_s for f in frames),
      "max_playback_error_s": max(abs(f.playback_time_error_s) for f in frames),
      "videos": probes,
    }
    _json(review / "summary.json", summary)
  return stats, summary


def record_example(output: Path, *, seed: int = 0):
  output = output.expanduser().resolve()
  partial = output.with_name(output.name + ".partial")
  if output.exists() or partial.exists():
    raise FileExistsError(f"Refusing to overwrite {output} or {partial}")
  simulation = ArmHandSimulation(scene="pick-place")
  simulation.reset(seed=seed, object_xy_jitter=0.0, object_yaw_jitter=0.0)
  plan = KnownStateGraspPlanner(simulation).plan_pick_and_place("right")
  capture = WorkcellConfig(
    camera_hz=10,
    cameras=tuple(
      CameraConfig(name, 640, 360, depth=False, segmentation=False) for name in CAMERAS
    ),
  )
  (partial / "raw").mkdir(parents=True)
  source = partial / "raw/episode_000000_cylinder_right.h5"
  with (
    WorkcellRenderer(simulation.model, capture.cameras, shadows=False) as renderer,
    PickPlaceExampleRecorder(
      source,
      simulation,
      capture,
      renderer=renderer,
      metadata={
        "recording_contract": "pick_place_stable_terminal_v2",
        "seed": seed,
        "object": "cylinder",
        "side": "right",
        "object_xy_jitter": 0.0,
        "object_yaw_jitter": 0.0,
        "controller": "KnownStateGraspPlanner + PickPlaceExecutor",
        "grasp_stabilizer": "existing soft carry constraint, disabled before release",
        "render_shadows": False,
      },
    ) as recorder,
  ):
    recorder.record_initial()
    result = PickPlaceExecutor(simulation, observer=recorder.observe).execute(plan)
    stability = wait_until_object_stable(
      simulation, "cylinder", observer=recorder.observe
    )
    recorder.record_terminal()
    pose = simulation.object_pose("cylinder")
    placed = cylinder_is_in_box(simulation, pose)
    outcome = {
      **asdict(result),
      "success": placed,
      "placed_in_box": placed,
      "final_object_pose": pose.tolist(),
      "box_center": result.box_center.tolist(),
      "final_object_twist": simulation.object_twist("cylinder").tolist(),
      "phases": [*result.phases, "terminal_settle"],
      "terminal_stability": asdict(stability),
    }
    if not placed:
      raise RuntimeError("Cylinder did not finish inside the box")
    recorder.set_outcome(outcome)
  report = validate_episode(source)
  if not report.valid:
    raise ValueError(report.errors)
  print(
    f"Recorded successful episode, {report.duration_seconds:.2f}s; exporting review",
    flush=True,
  )
  stats, summary = export_example(source, partial)
  _json(partial / "result.json", {**outcome, "validation": asdict(report)})
  _json(
    partial / "manifest.json",
    {
      "schema": "pick_place_example_v1",
      "success": True,
      "seed": seed,
      "physics_hz": capture.physics_hz,
      "force_hz": capture.control_hz,
      "duration_seconds": report.duration_seconds,
      "sample_count": stats["sample_count"],
      **summary,
      "source_code_sha256": {
        p.name: _sha256(p)
        for p in (
          Path(__file__),
          Path(__file__).with_name("task.py"),
          simulation.model_path,
        )
      },
    },
  )
  (partial / "README.md").write_text(
    "# Pick and place 示例\n\n"
    "同一次成功仿真：右手从桌面抓起圆柱，搬运到蓝色盒子，释放并退回。\n\n"
    "- `index.html`：视频、力曲线和原始数据入口。\n"
    "- `review/review.mp4`：头部/腕部相机与五指触觉热力图，10 fps。\n"
    "- `review/scene.mp4`：头部、腕部、正面、俯视四视角，10 fps。\n"
    "- `review/frames/`、`review/raw/`、`review/frames.csv`：逐帧图像、带符号触觉网格、时间对应。\n"
    "- `curves/`：原始右手法向力/切向力 PNG、PDF、CSV 及统计。\n"
    "- `raw/`：完整 HDF5 与采集清单；`result.json`：放置和稳定性检查。\n\n"
    "物理 500 Hz，状态/力 100 Hz，相机 10 Hz。力为求解器指尖与圆柱接触力，"
    "网格是空间分配估计；曲线不平滑、不裁剪。切向力为带符号切向矢量求和后的模长。"
    "初始观测时间为 0；后续力与相机位姿对应积分前时刻，状态 qpos/qvel 为积分后时刻，"
    "两者相差一个物理步长 2 ms，HDF5 分别保存时钟。\n\n"
    "使用现有程序规划和执行器；搬运期间沿用已有软抓持约束，释放前解除。"
    "本示例初始位置固定，seed 仅用于复现。\n\n"
    "重采到新目录：`MUJOCO_GL=egl .pixi/envs/default/bin/python "
    "scripts/workcell/record_pick_place_example.py --output-dir datasets/pick_place_new`。\n"
  )
  (partial / "index.html").write_text(f"""<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><title>Pick and place 示例</title>
<style>body{{max-width:1280px;margin:32px auto;padding:0 20px;background:#10141c;color:#e5e9f0;font:16px system-ui}}a{{color:#8fc7ff}}video,img{{width:100%;border-radius:8px}}section{{margin:30px 0}}li{{margin:8px 0}}</style>
<h1>Pick and place：抓取并放入盒中</h1>
<p>成功 · 仿真 {report.duration_seconds:.2f} s · 视频 10 fps · 右手接触力 100 Hz · 同一次采集</p>
<section><h2>相机与触觉</h2><video controls preload="metadata" src="review/review.mp4"></video></section>
<section><h2>四视角任务视频</h2><video controls preload="metadata" src="review/scene.mp4"></video></section>
<section><h2>右手五指力曲线</h2><p>法向力与合成切向力，单位 N，未经平滑。</p><a href="curves/right_hand_force_curves.pdf"><img src="curves/right_hand_force_curves.png"></a></section>
<ul><li><a href="curves/right_hand_force_curves.pdf">力曲线 PDF</a> · <a href="curves/right_hand_forces.csv">力数据 CSV</a> · <a href="curves/force_statistics.json">统计</a></li>
<li><a href="raw/episode_000000_cylinder_right.h5">原始 HDF5</a> · <a href="review/frames.csv">逐帧时间对应</a></li>
<li><a href="result.json">任务结果</a> · <a href="manifest.json">清单</a> · <a href="README.md">说明</a></li></ul></html>""")
  # The requested directory appears only after task and artifact checks pass.
  partial.rename(output)
  print(f"Published {output}", flush=True)
