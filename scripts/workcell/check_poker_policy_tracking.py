"""Serial renderer-free replay of exported measured next-pose targets.

Not actuator replay and never resets live qpos to recorded frames. Stored phases
are diagnostic labels only; the controller must infer its mode from live data.
"""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
from kaihand_tactile_env.shared.config import default_model_path
from kaihand_tactile_env.shared.egosteer_adapter import (
  SITE_FROM_EGOSTEER_WRIST,
  camera_from_world_opencv,
  decode_action_targets,
  model_state_history,
)
from kaihand_tactile_env.shared.egosteer_archive import archived_motion
from kaihand_tactile_env.tasks.poker_draw.mid_full import middle_force_simulation
from kaihand_tactile_env.tasks.poker_draw.randomization import reset_randomized_card
from kaihand_tactile_env.tasks.poker_draw.rollout_eval import (
  PokerOutcomeMonitor,
  observe_simulation,
)
from run_egosteer_policy import RolloutStats, _command_action_step
from run_poker_egosteer_policy import parse_args


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--source",
    type=Path,
    default=Path("datasets/poker_draw_0909/raw/episode_000003_card_right.h5"),
  )
  parser.add_argument(
    "--controller", choices=("legacy", "free", "feedback"), required=True
  )
  parser.add_argument("--seconds", type=float, default=3.0)
  parser.add_argument("--output-dir", type=Path, required=True)
  args = parser.parse_args()
  if not np.isfinite(args.seconds) or not 0 < args.seconds <= 60:
    parser.error("seconds must be in (0,60]")
  args.output_dir.mkdir(parents=True, exist_ok=False)
  with h5py.File(args.source, "r") as file:
    metadata = json.loads(file.attrs["metadata_json"])
    times, wrists, hands = archived_motion(file, SITE_FROM_EGOSTEER_WRIST)
    times = np.asarray(file["cameras/head/timestamp"])
    raw = np.concatenate((wrists, hands), axis=-1)
    camera_from_world = camera_from_world_opencv(
      file["cameras/head/world_from_camera"][0]
    )
    initial_qpos = file["state/qpos"][0]
    state_times = np.asarray(file["state/timestamp"])
    phases = np.asarray(file["commands/phase"]).astype(str)
  stats = RolloutStats()
  controls = parse_args(["--viewer", "none"])
  report = {
    "source": str(args.source),
    "controller": args.controller,
    "action_source": "archived_measured_next_pose_teacher_replay_not_model",
    "privileged_phases_used_for_control": False,
    "frames": 0,
  }
  started = time.monotonic()
  rows = []
  controller = None
  monitor = PokerOutcomeMonitor()
  with middle_force_simulation(default_model_path("poker-draw")) as (sim, _):
    reset_randomized_card(
      sim,
      seed=metadata["seed"],
      xy_jitter_m=metadata["object_xy_jitter"],
      yaw_jitter_rad=metadata["object_yaw_jitter"],
    )
    initial_x = float(sim.object_pose("card")[0])
    report["initial_qpos_max_difference"] = float(
      np.max(np.abs(sim.data.qpos - initial_qpos))
    )
    if report["initial_qpos_max_difference"] > 1e-10:
      raise RuntimeError("reset does not match source; no state teleport allowed")
    if args.controller == "legacy":
      sim.drive_limit_n = 4.0
    elif args.controller == "feedback":
      from kaihand_tactile_env.tasks.poker_draw.policy_control import (
        PokerPolicyController,
      )

      controller = PokerPolicyController(sim)
    try:
      for index in range(1, len(times)):
        if times[index] > args.seconds + 1e-9:
          break
        model_target = model_state_history(raw[index : index + 1], camera_from_world)
        decoded = decode_action_targets(sim, model_target, camera_from_world)
        if controller is not None:
          controller.before_command()
        _command_action_step(sim, decoded, 0, controls, stats)
        if controller is not None:
          controller.after_command(decoded.right, 0)
        while sim.data.time < times[index] - 1e-12:
          sim.step()
          if controller is not None:
            controller.after_step()
            observe_simulation(monitor, controller, initial_x)
        current, _ = sim.current_pose_matrix("right")
        row = {
          "time": float(sim.data.time),
          "source_phase": str(
            phases[max(0, np.searchsorted(state_times, times[index], side="right") - 1)]
          ),
          "wrist_position_error_m": float(
            np.linalg.norm(current - decoded.right.site_positions_world[0])
          ),
          "mode": controller.mode if controller else args.controller,
          "card_position": sim.object_pose("card")[:3].tolist(),
        }
        rows.append(row)
        stats.maximum_object_height = max(
          stats.maximum_object_height, float(sim.object_pose("card")[2])
        )
        stats.action_steps += 1
        if index % 150 == 0:
          print(
            f"frame={index} time={sim.data.time:.3f} mode={row['mode']} error={row['wrist_position_error_m']:.4f}",
            flush=True,
          )
      report["status"] = "completed"
    except Exception as error:
      report.update(status="stopped", error=f"{type(error).__name__}: {error}")
    report.update(
      frames=len(rows),
      sim_time=float(sim.data.time),
      stats=asdict(stats),
      final_card_pose=sim.object_pose("card").tolist(),
      wall_seconds=time.monotonic() - started,
      controller_report=controller.report() if controller else None,
      terminal_metrics=controller.terminal_metrics() if controller else None,
      evaluation=monitor.report() if controller else None,
    )
  report["maximum_wrist_tracking_error_m"] = max(
    (r["wrist_position_error_m"] for r in rows), default=None
  )
  with (args.output_dir / "summary.json").open("x") as file:
    json.dump(report, file, indent=2)
  with (args.output_dir / "tracking.json").open("x") as file:
    json.dump(rows, file, indent=2)
  print(json.dumps(report), flush=True)


if __name__ == "__main__":
  main()
