"""Check the physical T-ICT wrist frame, independently of EgoSteer conventions."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def test_physical_wrist_and_future_finger_decoding(monkeypatch):
  directory = Path(__file__).resolve().parents[1] / 'scripts' / 'workcell'
  monkeypatch.syspath_prepend(str(directory))
  spec = importlib.util.spec_from_file_location('poker_tict_runner_test', directory/'run_poker_egotouch_policy.py')
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  rotation = np.array([[0.,-1,0],[1,0,0],[0,0,1]])
  base = np.eye(4); base[:3,:3]=rotation; base[:3,3]=[.4,.2,.9]
  offset = np.eye(4); offset[:3,3]=[.03,-.02,.01]
  control=base@offset
  sim = SimpleNamespace(current_pose_matrix=lambda side:(control[:3,3], control[:3,:3]))
  monkeypatch.setattr(module,'_named_site_pose',lambda sim,name:base.copy())
  wrists=np.broadcast_to(base,(3,2,4,4)).copy()
  wrists[:,:,0,3]+=.05
  tip_relative=np.eye(4);tip_relative[:3,3]=[.04,0,-.1]
  tips=np.broadcast_to(wrists[:,:,None]@tip_relative,(3,2,5,4,4)).copy()
  result=module.decode_world_targets(sim,dict(wrists_world=wrists.tolist(),tips_world=tips.tolist()))
  for side in ('left','right'):
    target=result.for_side(side)
    expected=wrists[:,0]@offset
    np.testing.assert_allclose(target.site_positions_world,expected[:,:3,3])
    np.testing.assert_allclose(target.site_rotations_world,expected[:,:3,:3])
    np.testing.assert_allclose(target.fingertips_world,tips[:,0,:,:3,3])
