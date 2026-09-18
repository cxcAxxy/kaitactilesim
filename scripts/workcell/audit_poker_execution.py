"""Replay successful WDS next-pose targets through the 32-step policy interface.

Diagnostic reference actions, NOT a learned-policy rollout or success-rate sample.
No recorded actuator commands, per-stage control calls or live state teleports.
"""

import argparse
import io
import json
import tarfile
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from kaihand_tactile_env.shared.config import default_model_path
from kaihand_tactile_env.shared.egosteer_adapter import (
  _store_pose,
  _wrist_pose,
  decode_action_targets,
  model_state_history,
  raw_unified_from_simulation,
  relative_to_absolute,
)
from kaihand_tactile_env.tasks.poker_draw.execution_audit import execution_sample
from kaihand_tactile_env.tasks.poker_draw.mid_full import middle_force_simulation
from kaihand_tactile_env.tasks.poker_draw.policy_control import PokerPolicyController
from kaihand_tactile_env.tasks.poker_draw.randomization import reset_randomized_card
from kaihand_tactile_env.tasks.poker_draw.rollout_eval import (
  PokerOutcomeMonitor,
  observe_simulation,
)
from run_egosteer_policy import RolloutStats, _command_action_step
from run_poker_egosteer_policy import parse_args


def load_episode(shard, episode):
  prefix = f"episode_{episode:06d}_frame_"
  rows = []
  with tarfile.open(shard, "r|*") as tar:
    for member in tar:
      if member.name.startswith(prefix) and member.name.endswith(".lowdim.npy"):
        index = int(member.name[len(prefix) :].split(".")[0])
        assert index == len(rows), "non-contiguous reference frames"
        row = np.load(io.BytesIO(tar.extractfile(member).read()), allow_pickle=False)
        assert row.shape == (116,) and np.isfinite(row).all()
        rows.append(row)
      elif rows and not member.name.startswith(prefix):
        break
  if not rows:
    raise ValueError("reference episode absent from shard")
  return np.stack(rows)


def encode_reference_relative(state, absolute):
  relative = absolute.copy()
  for side in ("left", "right"):
    _store_pose(
      relative,
      side,
      np.linalg.inv(_wrist_pose(state, side)) @ _wrist_pose(absolute, side),
    )
  relative[:, 18:] -= state[18:]
  return relative


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--shard", type=Path, required=True)
  parser.add_argument("--episode", type=int, default=3)
  parser.add_argument("--seed", type=int, default=90903)
  parser.add_argument("--output-dir", type=Path, required=True)
  args = parser.parse_args()
  data = load_episode(args.shard, args.episode)
  args.output_dir.mkdir(parents=True, exist_ok=False)
  targets_raw = data[:, 48:96]
  # Permit up to two seconds holding the final saved target, no extra motion.
  targets_raw = np.concatenate([targets_raw, np.repeat(targets_raw[-1:], 60, axis=0)])
  camera_from_world = data[0, 96:112].reshape(4, 4)
  report = {
    "source": str(args.shard),
    "episode": args.episode,
    "seed": args.seed,
    "action_source": "successful_WDS_reference_not_model",
    "execute_steps": 32,
    "reference_frames": len(data),
    "extra_final_target_hold_limit_s": 2,
    "recorded_phases_used": False,
    "actuator_replay": False,
    "qpos_teleport": False,
  }
  controls = parse_args(["--viewer", "none", "--execute-steps", "32"])
  stats = RolloutStats()
  monitor = PokerOutcomeMonitor()
  started = time.monotonic()
  tick = 0
  with middle_force_simulation(default_model_path("poker-draw")) as (sim, contact):
    reset_randomized_card(
      sim, seed=args.seed, xy_jitter_m=0.004, yaw_jitter_rad=float(np.deg2rad(0.5))
    )
    delta = float(np.max(np.abs(raw_unified_from_simulation(sim) - data[0, :48])))
    report.update(initial_raw_state_max_difference=delta, contact_model=contact)
    if delta > 1e-7:
      raise RuntimeError("initial robot state does not match reference")
    controller = PokerPolicyController(sim)
    initial_x = sim.object_pose("card")[0]
    try:
      with (args.output_dir / "action_audit.jsonl").open("x") as log:
        for start in range(0, len(targets_raw), 32):
          live = model_state_history(
            raw_unified_from_simulation(sim)[None], camera_from_world
          )[0]
          absolute = model_state_history(
            targets_raw[start : start + 32], camera_from_world
          )
          # Same unnormalized relative action -> absolute -> world -> IK path as policy.
          relative = encode_reference_relative(live, absolute)
          decoded = decode_action_targets(
            sim, relative_to_absolute(live, relative), camera_from_world
          )
          stats.requests += 1
          for index in range(len(relative)):
            controller.before_command()
            _command_action_step(sim, decoded, index, controls, stats)
            controller.after_command(decoded.right, index)
            tick += 1
            while sim.data.time < tick / 30 - 1e-12:
              sim.step()
              controller.after_step()
              observe_simulation(monitor, controller, initial_x)
              if monitor.success:
                break
            stats.action_steps += 1
            stats.maximum_object_height = max(
              stats.maximum_object_height, float(sim.object_pose("card")[2])
            )
            stats.stable_success = monitor.success
            log.write(
              json.dumps(execution_sample(sim, controller, decoded.right, index)) + "\n"
            )
            if monitor.success:
              break
          log.flush()
          if start % 160 == 0:
            print(
              f"reference tick={tick} sim={sim.data.time:.3f} mode={controller.mode}",
              flush=True,
            )
          if monitor.success:
            break
      report["status"] = "success" if monitor.success else "reference_not_completed"
    except Exception as error:
      report.update(status="error", error=f"{type(error).__name__}: {error}")
    report.update(
      sim_seconds=float(sim.data.time),
      wall_seconds=time.monotonic() - started,
      evaluation=monitor.report(),
      stats=asdict(stats),
      controller_report=controller.report(),
      terminal_metrics=controller.terminal_metrics(),
    )
  (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2))
  print(json.dumps(report), flush=True)


if __name__ == "__main__":
  main()
