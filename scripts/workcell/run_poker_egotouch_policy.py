"""Serial model-only EgoTouch poker tests with local private model subprocess."""
import argparse
import base64
import json
import selectors
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from kaihand_tactile_env.shared.config import CameraConfig, default_model_path
from kaihand_tactile_env.shared.egosteer_adapter import (
  CV_FROM_MUJOCO_CAMERA,
  ActionTargets,
  SideActionTargets,
  _base_site_name,
  _fingertip_site_name,
  _named_site_pose,
  _quaternion_wxyz_from_rotation,
)
from kaihand_tactile_env.shared.policy_video import PokerPolicyVideo
from kaihand_tactile_env.shared.rendering import WorkcellRenderer
from kaihand_tactile_env.tasks.poker_draw.execution_audit import execution_sample
from kaihand_tactile_env.tasks.poker_draw.mid_full import middle_force_simulation
from kaihand_tactile_env.tasks.poker_draw.policy_control import (
  CONTROLLER_VERSION,
  PokerPolicyController,
)
from kaihand_tactile_env.tasks.poker_draw.randomization import reset_randomized_card
from kaihand_tactile_env.tasks.poker_draw.review_metrics import PokerReviewMetrics
from kaihand_tactile_env.tasks.poker_draw.rollout_eval import (
  PokerOutcomeMonitor,
  observe_simulation,
)
from run_egosteer_policy import (
  RolloutStats,
  _command_action_step,
  _model_action_horizon,
)

SIDES = ('left', 'right')
FINGERS = ('thumb', 'index', 'middle', 'ring', 'pinky')


def decode_world_targets(sim, response):
  wrists = np.asarray(response['wrists_world'])
  tips = np.asarray(response['tips_world'])
  result = {}
  for side_index, side in enumerate(SIDES):
    xyz, rotation = sim.current_pose_matrix(side)
    control_world = np.eye(4)
    control_world[:3, :3], control_world[:3, 3] = rotation, xyz
    base_world = _named_site_pose(sim, _base_site_name(side))
    # Physical base site -> controlled arm site, not EgoSteer canonical wrist.
    base_from_control = np.linalg.inv(base_world) @ control_world
    control_targets = wrists[:, side_index] @ base_from_control
    rotations = control_targets[:, :3, :3]
    result[side] = SideActionTargets(control_targets[:, :3, 3], rotations,
                                     _quaternion_wxyz_from_rotation(rotations),
                                     tips[:, side_index, :, :3, 3])
  return ActionTargets(**result)


def read_response(process):
  with selectors.DefaultSelector() as selector:
    selector.register(process.stdout, selectors.EVENT_READ)
    if not selector.select(timeout=180):
      raise TimeoutError('EgoTouch private worker response exceeded 180 seconds')
  line = process.stdout.readline()
  if not line:
    raise RuntimeError('EgoTouch worker exited; see model_worker.log')
  return json.loads(line)


def trial(args, process, metadata, seed):
  output = args.output_dir / f'seed_{seed}'
  output.mkdir(exist_ok=False)
  started = time.monotonic()
  stats = RolloutStats()
  monitor = PokerOutcomeMonitor()
  report = dict(seed=seed, model=metadata, controller=CONTROLLER_VERSION,
                execute_steps=args.execute_steps, control_hz=30, max_sim_seconds=args.max_sim_seconds,
                prediction_horizon=metadata['action_horizon'],
                replan_period_s=args.execute_steps/30,
                expert_used=False, tactile_sent_to_model=False, render_shadows=False,
                fingertip_execution='XYZ IK; predicted tip orientation logged but not separately constrained',
                precontact_noise=False)
  recorder = controller = sim = None
  tick = 0
  try:
    with middle_force_simulation(default_model_path('poker-draw')) as (sim, contact):
      report['contact_model'] = contact
      report['randomization'] = reset_randomized_card(sim, seed=seed, xy_jitter_m=.004,
                                                     yaw_jitter_rad=float(np.deg2rad(.5)))
      initial = sim.object_pose('card').copy()
      report['initial_card_pose'] = initial.tolist()
      review_metrics = PokerReviewMetrics(initial[0])
      review_metrics.update(sim)
      controller = PokerPolicyController(sim)
      camera = CameraConfig('head', width=320, height=240, rgb=True, depth=False, segmentation=False)
      with WorkcellRenderer(sim.model, (camera,), shadows=False) as renderer, \
           (output/'requests.jsonl').open('x') as requests, \
           (output/'action_audit.jsonl').open('x') as audit:
        report['render_backend'] = renderer.backend_info
        recorder = PokerPolicyVideo(
          sim,
          output/'review',
          fps=args.record_fps,
          width=args.review_width,
          height=args.review_height,
          render_width=args.review_render_width,
          render_height=args.review_render_height,
          second_camera=args.review_second_camera,
          metrics=review_metrics,
          metadata={
            "checkpoint": str(args.snapshot),
            "model": metadata,
            "prediction_horizon": metadata["action_horizon"],
            "execute_steps": args.execute_steps,
            "control_hz": 30,
            "replan_period_s": args.execute_steps / 30,
          },
        )
        recorder.capture(0, controller.mode)
        while sim.data.time < args.max_sim_seconds and not monitor.success:
          c2w = renderer.calibration(sim.data, camera).world_from_camera @ CV_FROM_MUJOCO_CAMERA
          rgb = renderer.capture(sim.data, camera)['rgb']
          wrists = np.stack([_named_site_pose(sim, _base_site_name(s)) for s in SIDES])
          tips = np.stack([np.stack([_named_site_pose(sim, _fingertip_site_name(s,f)) for f in FINGERS]) for s in SIDES])
          relative = np.linalg.inv(wrists)[:,None] @ tips
          request = dict(rgb=base64.b64encode(rgb.tobytes()).decode(), c2w=c2w.tolist(),
                         wrists=wrists.tolist(), tips_relative=relative.tolist(),
                         noise_seed=seed*10000+stats.requests)
          if stats.requests == 0:
            request['save_input'] = str((output/'first_request.npz').resolve())
          process.stdin.write(json.dumps(request)+'\n')
          process.stdin.flush()
          response = read_response(process)
          stats.requests += 1
          decoded = decode_world_targets(sim, response)
          requests.write(json.dumps(dict(request=stats.requests, sim_time=float(sim.data.time),
                                         inference_seconds=response['seconds'],
                                         roundtrip_max_error=response['roundtrip_max_error'],
                                         right_wrist_targets=decoded.right.site_positions_world.tolist()))+'\n')
          requests.flush()
          if stats.requests == 1 or stats.requests % 10 == 0:
            print(f'[EgoTouch] seed={seed} request={stats.requests} sim={sim.data.time:.3f}s inference={response["seconds"]:.3f}s', flush=True)
          for index in range(args.execute_steps):
            if sim.data.time >= args.max_sim_seconds or monitor.success:
              break
            controller.before_command()
            _command_action_step(sim, decoded, index, args, stats)
            controller.after_command(decoded.right, index)
            tick += 1
            while sim.data.time < tick/30 - 1e-12:
              sim.step()
              controller.after_step()
              observe_simulation(monitor, controller, initial[0])
              review_metrics.update(sim)
              if not np.isfinite(sim.data.qpos).all() or not np.isfinite(sim.data.qvel).all():
                raise RuntimeError('nonfinite physics state')
              if sim.object_pose('card')[2] < .5:
                raise RuntimeError('card fell below work surface')
              if monitor.success:
                break
            stats.action_steps += 1
            stats.maximum_object_height = max(stats.maximum_object_height, float(sim.object_pose('card')[2]))
            row = execution_sample(sim, controller, decoded.right, index)
            predicted = np.asarray(response['tips_world'])[index,1,:,:3,:3]
            actual = np.stack([_named_site_pose(sim, _fingertip_site_name('right', f))[:3,:3] for f in FINGERS])
            relative_rot = np.swapaxes(actual,-1,-2) @ predicted
            row['fingertip_rotation_error_deg'] = np.rad2deg(np.arccos(np.clip((np.trace(relative_rot,axis1=-2,axis2=-1)-1)/2,-1,1))).tolist()
            audit.write(json.dumps(row)+'\n')
            recorder.capture(tick, controller.mode)
        report['status'] = 'success' if monitor.success else 'task_not_completed'
        stats.stable_success = monitor.success
  except Exception as error:
    report.update(status='error', error=f'{type(error).__name__}: {error}')
  finally:
    report['evaluation'] = monitor.report()
    if sim is not None:
      report.update(sim_seconds=float(sim.data.time), final_card_pose=sim.object_pose('card').tolist())
    if controller is not None:
      report['controller_report'] = controller.report()
      report['terminal_metrics'] = controller.terminal_metrics()
    if recorder is not None:
      recorder.capture(tick, controller.mode, force=True)
      report['video'] = recorder.finish(status=report['status'], evaluation=report['evaluation'], error=report.get('error'))
    report.update(stats=asdict(stats), wall_seconds=time.monotonic()-started)
    (output/'summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report.get(k) for k in ['seed','status','error','sim_seconds','wall_seconds','evaluation']}),flush=True)
  return report


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--model-python', required=True)
  parser.add_argument('--model-project', type=Path, required=True)
  parser.add_argument('--snapshot', type=Path, required=True)
  parser.add_argument('--output-dir', type=Path, required=True)
  parser.add_argument('--seeds', type=int, nargs='+', default=[0,1,2])
  parser.add_argument('--execute-steps', type=int, default=5)
  parser.add_argument('--max-sim-seconds', type=float, default=50)
  parser.add_argument('--record-fps', type=int, choices=(5, 10), default=10)
  parser.add_argument('--review-width', type=int, default=1920)
  parser.add_argument('--review-height', type=int, default=1080)
  parser.add_argument('--review-render-width', type=int, default=640)
  parser.add_argument('--review-render-height', type=int, default=480)
  parser.add_argument(
    '--review-second-camera',
    choices=('global', 'right_wrist', 'overhead'),
    default='global',
  )
  args = parser.parse_args()
  if args.execute_steps <= 0 or len(set(args.seeds)) != len(args.seeds):
    parser.error('execute-steps must be positive; seeds must be unique')
  if min(args.review_width, args.review_height, args.review_render_width,
         args.review_render_height) <= 0:
    parser.error('review dimensions must be positive')
  if args.review_width % 2 or args.review_height % 2:
    parser.error('review output dimensions must be even')
  args.control_side='right'
  args.max_wrist_jump=.15
  args.max_arm_position_error=.03
  args.max_arm_orientation_error=.30
  args.hand_ik_tolerance=.0015
  args.max_hand_error=.02
  args.strict_ik=False
  args.output_dir.mkdir(parents=True, exist_ok=False)
  started=time.monotonic()
  with (args.output_dir/'model_worker.log').open('x') as log:
    process=subprocess.Popen([args.model_python, str(Path(__file__).with_name('egotouch_model_worker.py')),
      '--project',str(args.model_project),'--snapshot',str(args.snapshot)],
      stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1)
    try:
      metadata=read_response(process)['ready']
      prediction_horizon = _model_action_horizon(metadata)
      if args.execute_steps > prediction_horizon:
        raise RuntimeError(
          f'--execute-steps={args.execute_steps} exceeds checkpoint horizon '
          f'{prediction_horizon}'
        )
      (args.output_dir/'model.json').write_text(json.dumps(metadata,indent=2))
      reports=[]
      for seed in args.seeds:
        reports.append(trial(args,process,metadata,seed))
        if process.poll() is not None:
          break
      summary=dict(trials=len(reports),successes=sum(r['evaluation']['success'] for r in reports),
                   wall_seconds=time.monotonic()-started,
                   results=[{k:r.get(k) for k in ['seed','status','error','sim_seconds','wall_seconds','evaluation']} for r in reports])
      (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2))
    finally:
      if process.poll() is None:
        try:
          process.stdin.write('{"close":true}\n')
          process.stdin.flush()
          process.wait(timeout=15)
        except (OSError,subprocess.TimeoutExpired):
          process.terminate()
          process.wait(timeout=15)


if __name__ == '__main__':
  main()
