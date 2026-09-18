#!/usr/bin/env python3
"""Interactive bilateral 2x5 fingertip dashboard and grasp controls."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from kaihand_tactile_env.shared.config import CameraConfig
from kaihand_tactile_env.shared.rendering import WorkcellRenderer
from kaihand_tactile_env.shared.simulation import ArmHandSimulation
from kaihand_tactile_env.shared.tactile import (
  GENESIS_PROBE_GEOM_PREFIX,
  GenesisProbeTactileProvider,
)
from kaihand_tactile_env.tasks.pick_place.task import (
  KnownStateGraspPlanner,
  PickPlaceExecutor,
)

FINGERS = (
  ("thumb", "link6"),
  ("index", "link4"),
  ("middle", "link4"),
  ("ring", "link4"),
  ("pinky", "link4"),
)
KEY_HELP = "[1] run pick-and-place   [C] random cylinder   [0] reset home   [Q] quit"
KEY_COMMANDS = {
  "1": "pick_place",
  "c": "random_cylinder",
  "0": "reset_home",
}
BUTTONS = (
  ("1  Pick + place", "pick_place"),
  ("C  Random cylinder", "random_cylinder"),
  ("0  Home", "reset_home"),
  ("Q  Quit", "quit"),
)


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Show all ten 7x5 tactile arrays and control planned grasps."
  )
  parser.add_argument(
    "--layout",
    type=Path,
    help="Accepted bilateral layout; defaults to the workcell layout.",
  )
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--xy-jitter", type=float, default=0.01)
  parser.add_argument("--yaw-jitter", type=float, default=0.05)
  parser.add_argument("--max-depth-mm", type=float, default=3.0)
  parser.add_argument(
    "--refresh-hz",
    type=float,
    default=20.0,
    help="Tactile heatmap drawing rate in simulation time.",
  )
  parser.add_argument(
    "--camera-hz",
    type=float,
    default=2.0,
    help=(
      "Head-camera drawing rate in simulation time. It is intentionally "
      "independent of the tactile heatmaps because offscreen rendering can "
      "be slow without a hardware OpenGL driver."
    ),
  )
  parser.add_argument(
    "--show-probes",
    action="store_true",
    help="Show the 350 diagnostic probe spheres in the MuJoCo viewer.",
  )
  parser.add_argument("--no-sim-viewer", action="store_true")
  parser.add_argument(
    "--task",
    choices=("idle", "pick-place"),
    default="idle",
    help="Optionally start one pick-and-place episode automatically.",
  )
  return parser.parse_args()


def _load_matplotlib() -> tuple[Any, Any]:
  if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
    raise RuntimeError(
      "No graphical display is available. Run this command from the desktop "
      "session or with X11 forwarding enabled."
    )
  import matplotlib

  matplotlib.use("TkAgg")
  import matplotlib.pyplot as plt
  from matplotlib.widgets import Button

  return plt, Button


class TactileDashboard:
  """Matplotlib heatmaps plus a command queue driven by key events."""

  def __init__(
    self,
    simulation: ArmHandSimulation,
    tactile: GenesisProbeTactileProvider,
    viewer: Any,
    *,
    max_depth_mm: float,
    refresh_hz: float,
    camera_hz: float,
    renderer: WorkcellRenderer,
    head_camera: CameraConfig,
  ) -> None:
    self.sim = simulation
    self.tactile = tactile
    self.viewer = viewer
    self.plt, button_type = _load_matplotlib()
    self.refresh_period = 1.0 / refresh_hz
    self.camera_period = 1.0 / camera_hz
    self.renderer = renderer
    self.head_camera = head_camera
    self._next_refresh_sim_time = float(simulation.data.time)
    self._next_camera_sim_time = float(simulation.data.time)
    self._last_sim_time = float(simulation.data.time)
    self._last_viewer_sync = -np.inf
    self._blit_background: Any | None = None
    self._last_input_command = ""
    self._last_input_time = -np.inf
    self._latest_sample = tactile.read(simulation.data)
    self.pending_command: str | None = None
    self.closed = False
    self._probe_geom_ids = np.asarray(
      [
        geom_id
        for geom_id in range(simulation.model.ngeom)
        if simulation.model.geom(geom_id).name.startswith(GENESIS_PROBE_GEOM_PREFIX)
      ],
      dtype=np.int32,
    )
    if len(self._probe_geom_ids) != tactile.layout.count:
      raise RuntimeError("probe geom count differs from the bilateral layout")

    names = np.asarray(tactile.layout.body_names)
    self._indices: dict[str, np.ndarray] = {}
    for side in ("l", "r"):
      for finger, link in FINGERS:
        name = f"hand_{side}_{finger}_{link}"
        indices = np.flatnonzero(names == name)
        if len(indices) != 35:
          raise RuntimeError(f"{name}: expected 35 probes, got {len(indices)}")
        if tactile.layout.grid_shape[int(indices[0])] != (7, 5):
          raise RuntimeError(f"{name}: expected a 7x5 grid")
        self._indices[name] = indices

    self.figure = self.plt.figure(figsize=(18, 9))
    grid = self.figure.add_gridspec(
      2, 6, width_ratios=(1, 1, 1, 1, 1, 2.25), wspace=0.5, hspace=0.82
    )
    axes = np.asarray(
      [
        [self.figure.add_subplot(grid[row, column]) for column in range(5)]
        for row in range(2)
      ]
    )
    camera_axis = self.figure.add_subplot(grid[:, 5])
    camera_axis.set_title("Robot head camera (RGB)")
    camera_axis.axis("off")
    self._head_image = camera_axis.imshow(
      np.zeros((head_camera.height, head_camera.width, 3), dtype=np.uint8),
      animated=True,
    )
    self.figure.canvas.manager.set_window_title("KaiHand bilateral tactile dashboard")
    self.figure.subplots_adjust(
      left=0.05,
      right=0.94,
      bottom=0.22,
      top=0.87,
      wspace=0.50,
      hspace=0.82,
    )
    self._images: dict[str, Any] = {}
    self._empty_labels: dict[str, Any] = {}
    for row, side in enumerate(("l", "r")):
      for column, (finger, link) in enumerate(FINGERS):
        name = f"hand_{side}_{finger}_{link}"
        axis = axes[row, column]
        image = axis.imshow(
          np.zeros((7, 5)),
          origin="lower",
          interpolation="nearest",
          cmap="inferno",
          vmin=0.0,
          vmax=max_depth_mm,
          aspect="equal",
          animated=True,
        )
        axis.set_title(
          f"{'L' if side == 'l' else 'R'} {finger.title()}", fontsize=9, pad=5
        )
        axis.title.set_animated(True)
        axis.set_xticks(range(5))
        axis.set_yticks(range(7))
        axis.tick_params(labelsize=7)
        if column == 0:
          axis.set_ylabel(f"{'Left' if side == 'l' else 'Right'}\nrow")
        self._images[name] = image
        self._empty_labels[name] = axis.text(
          0.5,
          0.5,
          "NO CONTACT",
          color="0.7",
          fontsize=8,
          ha="center",
          va="center",
          transform=axis.transAxes,
          animated=True,
        )
    colorbar = self.figure.colorbar(
      next(iter(self._images.values())), ax=axes.ravel().tolist(), shrink=0.8
    )
    colorbar.set_label("probe depth [mm]")
    self.figure.suptitle("KaiHand 10-fingertip tactile arrays (2 hands × 5 fingers)")
    self._status = self.figure.text(
      0.5,
      0.145,
      "ready: zero depth is shown as black / NO CONTACT",
      ha="center",
      fontsize=10,
      animated=True,
    )
    self.figure.text(0.5, 0.115, KEY_HELP, ha="center", fontsize=9)
    self._buttons: list[Any] = []
    button_margin = 0.035
    button_gap = 0.008
    button_width = (1.0 - 2.0 * button_margin - 6.0 * button_gap) / 7.0
    for index, (label, command) in enumerate(BUTTONS):
      button_axis = self.figure.add_axes(
        [
          button_margin + index * (button_width + button_gap),
          0.035,
          button_width,
          0.05,
        ]
      )
      button = button_type(button_axis, label, hovercolor="0.85")
      button.on_clicked(lambda _event, selected=command: self._queue_command(selected))
      self._buttons.append(button)
    self.figure.canvas.mpl_connect("key_press_event", self._on_key)
    self.figure.canvas.mpl_connect("close_event", self._on_close)
    self.figure.canvas.mpl_connect("draw_event", self._on_draw)
    self._dynamic_artists = [
      *self._images.values(),
      *(image.axes.title for image in self._images.values()),
      *self._empty_labels.values(),
      self._head_image,
      self._status,
    ]
    self.plt.show(block=False)
    # Cache the static axes, labels and buttons once.  Subsequent tactile and
    # cached-camera updates redraw only changing artists instead of the full
    # 18x9-inch Matplotlib canvas.
    self.figure.canvas.draw()
    # Matplotlib's Tk canvas does not always receive focus after the MuJoCo
    # viewer opens.  Bind at the Tk toplevel as a second path and focus the
    # dashboard once on startup.  The visible buttons remain a mouse fallback.
    window = getattr(self.figure.canvas.manager, "window", None)
    widget = getattr(self.figure.canvas, "get_tk_widget", lambda: None)()
    if window is not None:
      window.bind_all("<KeyPress>", self._on_tk_key, add="+")
    if widget is not None:
      widget.focus_set()

  def _on_key(self, event: Any) -> None:
    self._handle_key(str(event.key or ""))

  def _on_tk_key(self, event: Any) -> None:
    self._handle_key(str(event.keysym or ""))

  def _handle_key(self, raw_key: str) -> None:
    key = raw_key.lower()
    key = {"kp_1": "1", "kp_2": "2", "kp_0": "0"}.get(key, key)
    if key == "q":
      self._queue_command("quit")
    elif key in KEY_COMMANDS:
      self._queue_command(KEY_COMMANDS[key])

  def _queue_command(self, command: str) -> None:
    if command == "quit":
      self.closed = True
      self.plt.close(self.figure)
      return
    now = time.monotonic()
    # Tk and Matplotlib can report the same physical key event through both
    # bindings, especially while a redraw is in progress.  Collapse that pair
    # and normal keyboard autorepeat into one command.
    if command == self._last_input_command and now - self._last_input_time < 0.75:
      return
    self._last_input_command = command
    self._last_input_time = now
    self.pending_command = command
    self.set_status(f"input received: {command}")
    self.figure.canvas.flush_events()
    print(f"[tactile-dashboard] input received: {command}", flush=True)

  def _on_close(self, _event: Any) -> None:
    self.closed = True

  def _on_draw(self, _event: Any) -> None:
    """Remember the static canvas used by fast sensor redraws."""
    if self.figure.canvas.supports_blit:
      self._blit_background = self.figure.canvas.copy_from_bbox(self.figure.bbox)

  def _draw_dynamic_artists(self) -> None:
    canvas = self.figure.canvas
    if canvas.supports_blit:
      if self._blit_background is None:
        canvas.draw()
      canvas.restore_region(self._blit_background)
      for artist in self._dynamic_artists:
        if artist.axes is None:
          self.figure.draw_artist(artist)
        else:
          artist.axes.draw_artist(artist)
      canvas.blit(self.figure.bbox)
    else:
      # Animated artists are skipped by a normal full draw, so temporarily
      # include them on backends that do not implement blitting.
      for artist in self._dynamic_artists:
        artist.set_animated(False)
      canvas.draw()
      for artist in self._dynamic_artists:
        artist.set_animated(True)
    canvas.flush_events()

  def take_command(self) -> str | None:
    command = self.pending_command
    self.pending_command = None
    return command

  def set_status(self, message: str) -> None:
    self._status.set_text(message)

  def reset_refresh_timing(self) -> None:
    """Restart simulation-time plot and camera deadlines after a reset."""
    sim_time = float(self.sim.data.time)
    self._last_sim_time = sim_time
    self._next_refresh_sim_time = sim_time
    self._next_camera_sim_time = sim_time

  def sample_tactile(self) -> None:
    """Sample touch at 500 Hz independently of the slower GUI redraw."""
    self._latest_sample = self.tactile.read(self.sim.data)

  def refresh(self, *, force: bool = False) -> None:
    if self.closed:
      return
    sim_time = float(self.sim.data.time)
    if sim_time + self.sim.timestep < self._last_sim_time:
      self.reset_refresh_timing()
    self._last_sim_time = sim_time
    tolerance = 0.5 * self.sim.timestep
    refresh_due = force or sim_time + tolerance >= self._next_refresh_sim_time
    camera_due = sim_time + tolerance >= self._next_camera_sim_time
    if not refresh_due and not camera_due:
      return
    if camera_due:
      self._head_image.set_data(
        self.renderer.capture(self.sim.data, self.head_camera)["rgb"]
      )
      self._next_camera_sim_time = sim_time + self.camera_period
      # Keep the most recent RGB array cached in the image artist.  The costly
      # offscreen capture is low-rate; blitting that cached frame is cheap and
      # avoids a full-canvas redraw whenever a new camera frame arrives.

    if refresh_due:
      sample = self._latest_sample
      for name, indices in self._indices.items():
        depth_mm = self.tactile.probe_depth[indices].reshape(7, 5) * 1000.0
        self._images[name].set_data(depth_mm)
        self._empty_labels[name].set_visible(not np.any(depth_mm > 0.0))
        link_index = sample.link_names.index(name)
        self._images[name].axes.title.set_text(
          f"{'L' if '_l_' in name else 'R'} {name.split('_')[2].title()}\n"
          f"active={sample.contact_count[link_index]}  "
          f"max={depth_mm.max():.2f} mm"
        )
      colors = self.sim.model.geom_rgba[self._probe_geom_ids]
      colors[:] = (0.9, 0.15, 0.05, 0.75)
      colors[self.tactile.probe_contact] = (0.1, 1.0, 0.15, 0.95)
      self._next_refresh_sim_time = sim_time + self.refresh_period

    self._draw_dynamic_artists()

  def process_events(self) -> None:
    if not self.closed:
      self.refresh(force=True)

  def sync_viewer(self) -> None:
    """Refresh MuJoCo once at 60 Hz, independently of sensor-plot redraws."""
    now = time.monotonic()
    if (
      self.viewer is not None
      and self.viewer.is_running()
      and now - self._last_viewer_sync >= 1.0 / 60.0
    ):
      self.viewer.sync()
      self._last_viewer_sync = now


def main() -> None:
  args = _parse_args()
  if (
    args.xy_jitter < 0.0
    or args.yaw_jitter < 0.0
    or args.max_depth_mm <= 0.0
    or args.refresh_hz <= 0.0
    or args.camera_hz <= 0.0
  ):
    raise ValueError(
      "jitter must be non-negative; depth and refresh rates must be positive"
    )
  simulation = ArmHandSimulation(probe_layout_path=args.layout)
  tactile = GenesisProbeTactileProvider(
    simulation.model, simulation.genesis_probe_layout
  )
  head_camera = CameraConfig(
    "head", width=320, height=240, depth=False, segmentation=False
  )
  renderer = WorkcellRenderer(
    simulation.model,
    (head_camera,),
    visible_geom_groups=(0, 1),
    # Shadows dominate software-OpenGL render time and add little information
    # to this small live preview.  Dataset recording keeps the renderer default.
    shadows=False,
  )

  viewer = None
  if not args.no_sim_viewer:
    import mujoco.viewer

    viewer = mujoco.viewer.launch_passive(simulation.model, simulation.data)
    # Group 5 contains non-physical diagnostic probe spheres.  Hiding them has
    # no effect on tactile distance queries.
    viewer.opt.geomgroup[5] = args.show_probes
  dashboard = TactileDashboard(
    simulation,
    tactile,
    viewer,
    max_depth_mm=args.max_depth_mm,
    refresh_hz=args.refresh_hz,
    camera_hz=args.camera_hz,
    renderer=renderer,
    head_camera=head_camera,
  )
  seed = args.seed
  realtime_sim_start = float(simulation.data.time)
  realtime_wall_start = time.monotonic()

  def restart_realtime_clock() -> None:
    nonlocal realtime_sim_start, realtime_wall_start
    realtime_sim_start = float(simulation.data.time)
    realtime_wall_start = time.monotonic()

  def reset(objects: tuple[str, ...], *, randomize: bool) -> None:
    nonlocal seed
    simulation.reset(
      seed=seed,
      object_xy_jitter=args.xy_jitter if randomize else 0.0,
      object_yaw_jitter=args.yaw_jitter if randomize else 0.0,
      randomized_objects=objects,
    )
    seed += 1
    tactile.reset()
    dashboard.sample_tactile()
    dashboard.reset_refresh_timing()
    restart_realtime_clock()
    dashboard.set_status(f"reset: {', '.join(objects) if objects else 'home'}")
    dashboard.refresh(force=True)

  last_phase = ""
  grasp_peak_active = 0
  grasp_peak_depth_mm = 0.0

  def observe(_simulation: ArmHandSimulation, phase: str) -> None:
    nonlocal grasp_peak_active, grasp_peak_depth_mm, last_phase
    if phase != last_phase:
      dashboard.set_status(f"executing: {phase}")
      print(f"[tactile-dashboard] executing: {phase}", flush=True)
      last_phase = phase
    dashboard.sample_tactile()
    dashboard.refresh()
    dashboard.sync_viewer()
    grasp_peak_active = max(
      grasp_peak_active, int(np.count_nonzero(tactile.probe_contact))
    )
    grasp_peak_depth_mm = max(
      grasp_peak_depth_mm, float(tactile.probe_depth.max() * 1000.0)
    )
    target = realtime_wall_start + (simulation.data.time - realtime_sim_start)
    delay = target - time.monotonic()
    if delay > 0.0:
      time.sleep(min(delay, simulation.timestep))

  def pick_place() -> None:
    nonlocal grasp_peak_active, grasp_peak_depth_mm, last_phase
    side = "right"
    task_wall_start = time.monotonic()
    task_sim_start = float(simulation.data.time)
    dashboard.set_status("planning: red cylinder -> blue box")
    dashboard.process_events()
    try:
      plan = KnownStateGraspPlanner(simulation).plan_pick_and_place(side)
      # IK/planning can take wall-clock time while simulation time is frozen.
      # Re-anchor pacing so execution does not race to catch up afterwards.
      restart_realtime_clock()
      last_phase = ""
      grasp_peak_active = 0
      grasp_peak_depth_mm = 0.0
      result = PickPlaceExecutor(simulation, observer=observe).execute(plan)
      task_wall_elapsed = time.monotonic() - task_wall_start
      task_sim_elapsed = float(simulation.data.time) - task_sim_start
      restart_realtime_clock()
      dashboard.set_status(
        f"pick-place: success={result.success}, placed={result.placed_in_box}, "
        f"peak active={grasp_peak_active}, "
        f"peak depth={grasp_peak_depth_mm:.2f} mm, "
        f"sim={task_sim_elapsed:.2f} s, wall={task_wall_elapsed:.2f} s"
      )
      dashboard.refresh(force=True)
      print(
        f"[tactile-dashboard] result: success={result.success}, "
        f"placed={result.placed_in_box}, peak_active={grasp_peak_active}, "
        f"peak_depth_mm={grasp_peak_depth_mm:.3f}, "
        f"sim_elapsed_s={task_sim_elapsed:.3f}, "
        f"wall_elapsed_s={task_wall_elapsed:.3f}",
        flush=True,
      )
    except RuntimeError as error:
      dashboard.set_status(f"planning/execution failed: {error}")

  try:
    dashboard.refresh(force=True)
    if args.task == "pick-place":
      pick_place()
    previous_sim_time = float(simulation.data.time)
    while not dashboard.closed:
      if viewer is not None and not viewer.is_running():
        viewer.close()
        viewer = None
        dashboard.viewer = None
        dashboard.set_status("MuJoCo viewer closed; dashboard controls remain active")
      if simulation.data.time + simulation.timestep < previous_sim_time:
        # Reapply controller targets after the native MuJoCo Reset button.
        reset(("cylinder",), randomize=True)
      command = dashboard.take_command()
      if command == "pick_place":
        pick_place()
      elif command == "random_cylinder":
        reset(("cylinder",), randomize=True)
      elif command == "reset_home":
        reset((), randomize=False)
      else:
        simulation.step()
        dashboard.sample_tactile()
      previous_sim_time = float(simulation.data.time)
      dashboard.refresh()
      dashboard.sync_viewer()
      target = realtime_wall_start + (simulation.data.time - realtime_sim_start)
      delay = target - time.monotonic()
      if delay > 0.0:
        time.sleep(min(delay, simulation.timestep))
  finally:
    if viewer is not None:
      viewer.close()
    renderer.close()
    if not dashboard.closed:
      dashboard.plt.close(dashboard.figure)


if __name__ == "__main__":
  main()
