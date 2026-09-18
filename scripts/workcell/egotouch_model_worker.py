"""Private stdin/stdout bridge; run with the user's EgoTouch Python environment.

Uses that checkout's training encoders and official H50 sampler/decoder. No
Torch dependency is added to the simulation environment or existing services.
"""
import argparse
import base64
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--project', type=Path, required=True)
  parser.add_argument('--snapshot', type=Path, required=True)
  args = parser.parse_args()
  protocol = sys.stdout
  sys.stdout = sys.stderr
  sys.path.insert(0, str(args.project))
  import numpy as np
  import torch
  from inference.embodiment_policy import sample_h50
  from tools.train_usb_tict_ddp import _model_from_config
  from training.FlowMatchingDataloader import FlowMatchingDataloader
  from utils.tict_schema import decode_taskspace_action, encode_taskspace_action

  torch.set_num_threads(1)
  checkpoint = torch.load(args.snapshot / 'best.pt', map_location='cpu', weights_only=False)
  cfg = SimpleNamespace(**checkpoint['cfg'])
  assert not cfg.use_tactile_input and cfg.action_mode == 'absolute'
  assert cfg.frame_mode == 'camera_frame' and cfg.centric_mode == 'ego_centric'
  assert cfg.pred_horizon == 50 and not cfg.single_hand
  assert list(cfg.image_size) == [240, 320]
  binding = json.loads((args.snapshot / 'egotouch_binding.json').read_text())
  digest = hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
  assert digest == checkpoint['binding_sha256'], 'checkpoint/binding mismatch'
  stats = binding['stats']
  device = torch.device('cuda:0')
  model = _model_from_config(cfg, device)
  model.load_state_dict(checkpoint['model'], strict=True)
  # best.pt model was saved with EMA applied by the training launcher.
  model.eval().requires_grad_(False)
  loader = FlowMatchingDataloader.__new__(FlowMatchingDataloader)
  loader.is_tict = True
  loader.hand_entity_key = 'hands_hawor_v3'
  loader.single_hand = False
  loader.use_object_tokens = False
  loader.use_tactile_input = False
  loader.max_ict, loader.ict_dim = 8, 29
  for prefix, key in [('pos', 'pos'), ('finger_pos', 'finger_pos')]:
    for moment in ['mean', 'std']:
      setattr(loader, prefix + '_' + moment, np.array(stats[key][moment], dtype=np.float32))
  decode_args = dict(wrist_pos_mean=loader.pos_mean, wrist_pos_std=loader.pos_std,
                     fingertip_pos_mean=loader.finger_pos_mean, fingertip_pos_std=loader.finger_pos_std)
  metadata = dict(epoch=checkpoint['epoch'], global_step=checkpoint['global_step'],
                  best=checkpoint['best'], variant=checkpoint['variant'],
                  checkpoint_sha256=hashlib.sha256((args.snapshot/'best.pt').read_bytes()).hexdigest(),
                  binding_sha256=digest, statistics=stats,
                  action_horizon=int(model.pred_horizon),
                  action_dim=int(model.action_dim), flow_steps=cfg.num_inference_steps,
                  rgb='single RGB uint8 / 255, CHW, no JPEG',
                  rotation='official row-major R[:, :2] flattening',
                  model_weights='best.pt model (EMA already applied at save)')
  print(json.dumps({'ready': metadata}), file=protocol, flush=True)
  for line in sys.stdin:
    request = json.loads(line)
    if request.get('close'):
      break
    start = time.monotonic()
    wrists = np.asarray(request['wrists'], dtype=np.float64)
    tips = np.asarray(request['tips_relative'], dtype=np.float64)
    c2w = np.asarray(request['c2w'], dtype=np.float32)
    frame = {'entities': {'hands_hawor_v3': {
      side: {'T_hand_to_world': wrists[i].tolist()} for i, side in enumerate(['left', 'right'])
    }, 'objects': {}}, 'metadata': {}}
    source = {'finger_valid': np.ones((2, 5), dtype=bool), 'T_fingertip_to_wrist': tips}
    loader._get_tict_frame = lambda _, current=source: current
    ict, _, mask = loader._build_ict(frame, np.linalg.inv(c2w))
    fingers, finger_mask = loader._build_finger_tokens('live', frame)
    rgb = np.frombuffer(base64.b64decode(request['rgb']), dtype=np.uint8).reshape(240, 320, 3)
    arrays = {'x_rgb': (rgb.astype(np.float32)/255).transpose(2,0,1),
              'x_ict': ict, 'ict_mask': mask, 'x_finger': fingers, 'finger_mask': finger_mask}
    observation = {k: torch.from_numpy(v.copy()).unsqueeze(0).to(device) for k, v in arrays.items()}
    # Regression check physical poses -> official encode/decode (no MANO wrist correction).
    wrist_ref = np.linalg.inv(c2w) @ wrists
    reconstructed = decode_taskspace_action(encode_taskspace_action(wrist_ref, tips, **decode_args), **decode_args)
    roundtrip = max(float(np.max(np.abs(reconstructed['wrist_T_ref']-wrist_ref))),
                    float(np.max(np.abs(reconstructed['fingertip_T_wrist']-tips))))
    assert roundtrip < 1e-5 and not fingers[:, 12:].any()
    prediction = sample_h50(model, observation, seed=int(request['noise_seed']), steps=cfg.num_inference_steps)
    action = prediction['action'][0].float().cpu().numpy()
    assert action.shape == (metadata['action_horizon'], metadata['action_dim']) and np.isfinite(action).all()
    decoded = decode_taskspace_action(action, **decode_args)
    # Float64 camera conversion, matching current observation's frozen camera.
    world_wrists = np.asarray(request['c2w']) @ decoded['wrist_T_ref']
    world_tips = world_wrists[:, :, None] @ decoded['fingertip_T_wrist']
    output = dict(wrists_world=world_wrists.tolist(), tips_world=world_tips.tolist(),
                  seconds=time.monotonic()-start, roundtrip_max_error=roundtrip)
    if request.get('save_input'):
      np.savez_compressed(request['save_input'], **arrays, action=action,
                          c2w=c2w, wrists_world=wrists, tips_relative=tips)
    print(json.dumps(output, allow_nan=False), file=protocol, flush=True)


if __name__ == '__main__':
  main()
