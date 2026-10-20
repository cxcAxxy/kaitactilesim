"""Private stdin/stdout bridge; run with the user's EgoTouch Python environment.

Uses that checkout's training encoders and official H50 sampler/decoder. No
Torch dependency is added to the simulation environment or existing services.
"""
import argparse
import base64
import binascii
import hashlib
import io
import json
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import numpy as np

VIEW_SLOTS = ('head', 'left_wrist', 'right_wrist')
IMAGE_SHAPE_HWC = (240, 320, 3)
EXTERNAL_CANDIDATE_SCHEMA = 'egotouch-external-sim-candidate-v1'
EXTERNAL_INDEX_SCHEMA = 'egotouch-external-sim-candidate-index-v1'


def parse_args(argv=None):
  parser = argparse.ArgumentParser()
  parser.add_argument('--project', type=Path, required=True)
  source = parser.add_mutually_exclusive_group(required=True)
  source.add_argument(
    '--snapshot', type=Path,
    help='legacy snapshot directory containing best.pt and egotouch_binding.json',
  )
  source.add_argument('--checkpoint', type=Path, help='explicit checkpoint file')
  parser.add_argument(
    '--binding', type=Path,
    help='egotouch_binding.json paired with --checkpoint',
  )
  parser.add_argument('--device', default='cuda:0')
  parser.add_argument('--validate-only', action='store_true')
  args = parser.parse_args(argv)
  if args.checkpoint is not None and args.binding is None:
    parser.error('--checkpoint requires --binding')
  if args.snapshot is not None and args.binding is not None:
    parser.error('--binding is only valid with --checkpoint')
  return args


def _canonical_sha256(value):
  encoded = json.dumps(
    value, sort_keys=True, separators=(',', ':'),
  ).encode('utf-8')
  return hashlib.sha256(encoded).hexdigest()


def _sha256_bytes(encoded):
  return hashlib.sha256(encoded).hexdigest()


def _resolve_artifacts(args):
  if args.snapshot is not None:
    snapshot = args.snapshot.expanduser().resolve(strict=True)
    checkpoint = snapshot / 'best.pt'
    binding = snapshot / 'egotouch_binding.json'
  else:
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    binding = args.binding.expanduser().resolve(strict=True)
  if not checkpoint.is_file():
    raise ValueError(f'EgoTouch checkpoint is not a file: {checkpoint}')
  if not binding.is_file():
    raise ValueError(f'EgoTouch binding is not a file: {binding}')
  return checkpoint, binding


def _load_binding(path):
  encoded = path.read_bytes()
  try:
    binding = json.loads(encoded)
  except (UnicodeDecodeError, json.JSONDecodeError) as error:
    raise ValueError(f'invalid EgoTouch binding JSON: {path}') from error
  if not isinstance(binding, dict):
    raise ValueError('EgoTouch binding must contain a JSON object')
  return binding, _canonical_sha256(binding)


def _record_view_mask(record, *, label):
  if not isinstance(record, Mapping):
    raise ValueError(f'invalid selector record: {label}')
  images = record.get('images')
  if images is None:
    if not isinstance(record.get('image'), Mapping):
      raise ValueError(f'legacy selector record has no head image: {label}')
    return (True, False, False), False
  if not isinstance(images, Mapping) or set(images) != set(VIEW_SLOTS):
    raise ValueError(f'selector view slots differ from {VIEW_SLOTS}: {label}')
  mask = []
  for name in VIEW_SLOTS:
    view = images[name]
    if not isinstance(view, Mapping) or not isinstance(view.get('valid'), bool):
      raise ValueError(f'invalid selector view record: {label}/{name}')
    valid = view['valid']
    if valid and not isinstance(view.get('image'), Mapping):
      raise ValueError(f'valid selector view has no image: {label}/{name}')
    if not valid and set(view) != {'valid'}:
      raise ValueError(f'masked selector view carries payload: {label}/{name}')
    mask.append(valid)
  return tuple(mask), True


def _derive_view_contract(binding):
  """Audit every frozen selector record and return its fixed camera contract."""
  declared = binding.get('view_names')
  if declared is not None and tuple(declared) != VIEW_SLOTS:
    raise ValueError(f'binding view_names differ from {VIEW_SLOTS}')
  records = binding.get('selector_records')
  # Pre-multiview snapshots did not necessarily persist selector records. Their
  # established observation contract is one head RGB tensor.
  if records is None:
    if declared is not None:
      raise ValueError('multiview binding has no selector_records')
    return {
      'view_slots': VIEW_SLOTS,
      'view_mask': (True, False, False),
      'cameras': ('head',),
      'multiview': False,
    }
  if not isinstance(records, Mapping) or not records:
    raise ValueError('binding selector_records must be a nonempty mapping')

  expected_mask = None
  record_count = 0
  saw_multiview = False
  saw_legacy = False
  for session, frames in records.items():
    if not isinstance(frames, Mapping) or not frames:
      raise ValueError(f'selector session has no frame records: {session}')
    for frame, record in frames.items():
      label = f'{session}/{frame}'
      mask, multiview = _record_view_mask(record, label=label)
      if expected_mask is None:
        expected_mask = mask
      elif mask != expected_mask:
        raise ValueError(
          f'selector view mask is not fixed: {label} has {mask}, '
          f'expected {expected_mask}'
        )
      saw_multiview |= multiview
      saw_legacy |= not multiview
      record_count += 1
  if not record_count or expected_mask is None or not any(expected_mask):
    raise ValueError('binding has no usable visual selector records')
  if saw_multiview and saw_legacy:
    raise ValueError('binding mixes legacy and multiview selector records')
  if declared is not None and not saw_multiview:
    raise ValueError('binding declares multiview slots but contains legacy records')
  return {
    'view_slots': VIEW_SLOTS,
    'view_mask': expected_mask,
    'cameras': tuple(
      name
      for name, valid in zip(VIEW_SLOTS, expected_mask, strict=True)
      if valid
    ),
    'multiview': saw_multiview,
  }


def _validate_candidate_index(
  checkpoint_path, checkpoint, checkpoint_bytes, binding_sha256,
):
  if checkpoint.get('schema_version') != EXTERNAL_CANDIDATE_SCHEMA:
    return
  index_path = checkpoint_path.parent / 'index.json'
  if not index_path.is_file():
    return
  try:
    index = json.loads(index_path.read_bytes())
  except (UnicodeDecodeError, json.JSONDecodeError) as error:
    raise ValueError(f'invalid external candidate index: {index_path}') from error
  if (
    not isinstance(index, dict)
    or index.get('schema_version') != EXTERNAL_INDEX_SCHEMA
    or index.get('variant') != checkpoint.get('variant')
    or index.get('binding_sha256') != binding_sha256
    or not isinstance(index.get('candidates'), list)
  ):
    raise ValueError('external candidate index identity mismatch')
  matches = []
  for row in index['candidates']:
    if not isinstance(row, Mapping):
      raise ValueError('external candidate index contains an invalid row')
    reference = row.get('checkpoint_ref')
    if not isinstance(reference, Mapping) or not isinstance(reference.get('path'), str):
      raise ValueError('external candidate index contains an invalid reference')
    try:
      indexed_path = Path(reference['path']).expanduser().resolve(strict=True)
    except OSError:
      continue
    if indexed_path == checkpoint_path:
      matches.append((row, reference))
  if len(matches) != 1:
    raise ValueError('external checkpoint is not uniquely bound by index.json')
  row, reference = matches[0]
  expected = {
    'epoch': checkpoint.get('epoch'),
    'global_step': checkpoint.get('global_step'),
    'model_weights': checkpoint.get('model_weights'),
  }
  actual = {key: row.get(key) for key in expected}
  if actual != expected:
    raise ValueError('external checkpoint metadata differs from index.json')
  if reference.get('bytes') != len(checkpoint_bytes):
    raise ValueError('external checkpoint byte count differs from index.json')
  if reference.get('sha256') != _sha256_bytes(checkpoint_bytes):
    raise ValueError('external checkpoint SHA-256 differs from index.json')


def _validate_checkpoint(checkpoint, checkpoint_path, checkpoint_bytes, binding_digest):
  if not isinstance(checkpoint, Mapping):
    raise ValueError('EgoTouch checkpoint must contain a mapping')
  required = ('epoch', 'global_step', 'model', 'cfg', 'variant', 'binding_sha256')
  missing = [key for key in required if key not in checkpoint]
  if missing:
    raise ValueError(f'EgoTouch checkpoint is missing fields: {missing}')
  if not isinstance(checkpoint['model'], Mapping) or not checkpoint['model']:
    raise ValueError('EgoTouch checkpoint has no model state')
  if not isinstance(checkpoint['cfg'], Mapping):
    raise ValueError('EgoTouch checkpoint cfg must be a mapping')
  digest = checkpoint['binding_sha256']
  if (
    not isinstance(digest, str)
    or len(digest) != 64
    or any(character not in '0123456789abcdef' for character in digest)
  ):
    raise ValueError('checkpoint binding_sha256 is malformed')
  if digest != binding_digest:
    raise ValueError('checkpoint/binding SHA-256 mismatch')
  schema = checkpoint.get('schema_version')
  if schema is not None and schema != EXTERNAL_CANDIDATE_SCHEMA:
    raise ValueError(f'unsupported EgoTouch checkpoint schema: {schema!r}')
  model_weights = checkpoint.get('model_weights')
  if model_weights is None and checkpoint_path.name == 'best.pt':
    # Compatibility with the original frozen best.pt format: its model state
    # was emitted while EMA weights were applied by the training launcher.
    model_weights = 'ema'
  if model_weights not in ('ema', 'train'):
    raise ValueError(f'unsupported checkpoint model_weights: {model_weights!r}')
  _validate_candidate_index(
    checkpoint_path, checkpoint, checkpoint_bytes, binding_digest,
  )
  return model_weights


def _validate_config(cfg):
  expected = {
    'use_tactile_input': False,
    'action_mode': 'absolute',
    'frame_mode': 'camera_frame',
    'centric_mode': 'ego_centric',
    'pred_horizon': 50,
    'single_hand': False,
  }
  mismatches = {
    key: {'expected': value, 'actual': getattr(cfg, key, None)}
    for key, value in expected.items()
    if getattr(cfg, key, None) != value
  }
  if list(getattr(cfg, 'image_size', ())) != [240, 320]:
    mismatches['image_size'] = {
      'expected': [240, 320], 'actual': getattr(cfg, 'image_size', None),
    }
  if mismatches:
    raise ValueError(f'unsupported EgoTouch deployment config: {mismatches}')


def _decode_rgb(encoded, *, label):
  if not isinstance(encoded, str):
    raise ValueError(f'{label} must be a base64 string')
  try:
    raw = base64.b64decode(encoded, validate=True)
  except (binascii.Error, ValueError) as error:
    raise ValueError(f'{label} is not valid base64') from error
  expected_bytes = int(np.prod(IMAGE_SHAPE_HWC))
  if len(raw) != expected_bytes:
    raise ValueError(
      f'{label} has {len(raw)} decoded bytes; expected {expected_bytes}'
    )
  image = np.frombuffer(raw, dtype=np.uint8).reshape(IMAGE_SHAPE_HWC)
  return (image.astype(np.float32) / 255).transpose(2, 0, 1)


def _visual_arrays(request, view_contract):
  if view_contract['multiview']:
    rgbs = request.get('rgbs')
    if not isinstance(rgbs, Mapping):
      raise ValueError('multiview request requires an rgbs mapping')
    expected = set(view_contract['cameras'])
    actual = set(rgbs)
    if actual != expected:
      raise ValueError(
        f'rgbs camera keys differ: expected {sorted(expected)}, '
        f'got {sorted(actual)}'
      )
    views = np.zeros((len(VIEW_SLOTS), 3, 240, 320), dtype=np.float32)
    for index, name in enumerate(VIEW_SLOTS):
      if view_contract['view_mask'][index]:
        views[index] = _decode_rgb(rgbs[name], label=f'rgbs.{name}')
    return {
      'x_rgb_views': views,
      'view_mask': np.asarray(view_contract['view_mask'], dtype=bool),
    }
  if 'rgb' not in request:
    raise ValueError('legacy head-only request requires rgb')
  return {'x_rgb': _decode_rgb(request['rgb'], label='rgb')}


def main(argv=None):
  args = parse_args(argv)
  protocol = sys.stdout
  sys.stdout = sys.stderr
  project = args.project.expanduser().resolve(strict=True)
  checkpoint_path, binding_path = _resolve_artifacts(args)
  sys.path.insert(0, str(project))
  import torch
  from inference.embodiment_policy import sample_h50
  from tools.train_usb_tict_ddp import _model_from_config
  from training.FlowMatchingDataloader import FlowMatchingDataloader
  from utils.tict_schema import decode_taskspace_action, encode_taskspace_action

  torch.set_num_threads(1)
  checkpoint_bytes = checkpoint_path.read_bytes()
  checkpoint = torch.load(
    io.BytesIO(checkpoint_bytes), map_location='cpu', weights_only=False,
  )
  binding, digest = _load_binding(binding_path)
  model_weights = _validate_checkpoint(
    checkpoint, checkpoint_path, checkpoint_bytes, digest,
  )
  view_contract = _derive_view_contract(binding)
  cfg = SimpleNamespace(**checkpoint['cfg'])
  _validate_config(cfg)
  stats = binding.get('stats')
  if not isinstance(stats, Mapping):
    raise ValueError('EgoTouch binding has no training statistics')
  device = torch.device(args.device)
  model = _model_from_config(cfg, device)
  model.load_state_dict(checkpoint['model'], strict=True)
  model.eval().requires_grad_(False)
  loader = FlowMatchingDataloader.__new__(FlowMatchingDataloader)
  loader.is_tict = True
  loader.hand_entity_key = 'hands_hawor_v3'
  loader.single_hand = False
  loader.use_object_tokens = False
  loader.use_tactile_input = False
  loader.max_ict, loader.ict_dim = 8, 29
  for prefix, key in [('pos', 'pos'), ('finger_pos', 'finger_pos')]:
    group = stats.get(key)
    if not isinstance(group, Mapping):
      raise ValueError(f'EgoTouch statistics missing {key}')
    for moment in ['mean', 'std']:
      values = np.asarray(group.get(moment), dtype=np.float32)
      if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError(f'invalid EgoTouch statistics: {key}.{moment}')
      setattr(loader, prefix + '_' + moment, values)
  decode_args = dict(
    wrist_pos_mean=loader.pos_mean, wrist_pos_std=loader.pos_std,
    fingertip_pos_mean=loader.finger_pos_mean,
    fingertip_pos_std=loader.finger_pos_std,
  )
  metadata = dict(
    epoch=int(checkpoint['epoch']), global_step=int(checkpoint['global_step']),
    best=checkpoint.get('best'), variant=checkpoint['variant'],
    checkpoint_path=str(checkpoint_path), binding_path=str(binding_path),
    checkpoint_sha256=_sha256_bytes(checkpoint_bytes),
    binding_sha256=digest, statistics=stats,
    action_horizon=int(model.pred_horizon), action_dim=int(model.action_dim),
    flow_steps=cfg.num_inference_steps,
    cameras=list(view_contract['cameras']),
    view_slots=list(view_contract['view_slots']),
    view_mask=list(view_contract['view_mask']),
    rgb=(
      'rgbs mapping of active camera names to RGB uint8 / 255, HWC raw bytes '
      'encoded as base64'
      if view_contract['multiview']
      else 'single head RGB uint8 / 255, HWC raw bytes encoded as base64'
    ),
    rotation='official row-major R[:, :2] flattening',
    model_weights=model_weights,
  )
  print(json.dumps({'ready': metadata}), file=protocol, flush=True)
  if args.validate_only:
    return

  for line in sys.stdin:
    request = json.loads(line)
    if not isinstance(request, dict):
      raise ValueError('worker request must be a JSON object')
    if request.get('close'):
      break
    start = time.monotonic()
    wrists = np.asarray(request['wrists'], dtype=np.float64)
    tips = np.asarray(request['tips_relative'], dtype=np.float64)
    c2w = np.asarray(request['c2w'], dtype=np.float32)
    if wrists.shape != (2, 4, 4) or tips.shape != (2, 5, 4, 4):
      raise ValueError('invalid live wrist or fingertip shape')
    if c2w.shape != (4, 4):
      raise ValueError('invalid c2w shape')
    if not all(np.isfinite(value).all() for value in (wrists, tips, c2w)):
      raise ValueError('live poses contain nonfinite values')
    frame = {'entities': {'hands_hawor_v3': {
      side: {'T_hand_to_world': wrists[i].tolist()}
      for i, side in enumerate(['left', 'right'])
    }, 'objects': {}}, 'metadata': {}}
    source = {
      'finger_valid': np.ones((2, 5), dtype=bool),
      'T_fingertip_to_wrist': tips,
    }
    loader._get_tict_frame = lambda _, current=source: current
    ict, _, mask = loader._build_ict(frame, np.linalg.inv(c2w))
    fingers, finger_mask = loader._build_finger_tokens('live', frame)
    arrays = {
      **_visual_arrays(request, view_contract),
      'x_ict': ict, 'ict_mask': mask,
      'x_finger': fingers, 'finger_mask': finger_mask,
    }
    observation = {
      key: torch.from_numpy(value.copy()).unsqueeze(0).to(device)
      for key, value in arrays.items()
    }
    # Regression check physical poses -> official encode/decode (no MANO wrist
    # correction).
    wrist_ref = np.linalg.inv(c2w) @ wrists
    reconstructed = decode_taskspace_action(
      encode_taskspace_action(wrist_ref, tips, **decode_args), **decode_args,
    )
    roundtrip = max(
      float(np.max(np.abs(reconstructed['wrist_T_ref'] - wrist_ref))),
      float(np.max(np.abs(reconstructed['fingertip_T_wrist'] - tips))),
    )
    if roundtrip >= 1e-5 or fingers[:, 12:].any():
      raise ValueError('live task-space observation failed round-trip validation')
    prediction = sample_h50(
      model, observation, seed=int(request['noise_seed']),
      steps=cfg.num_inference_steps,
    )
    action = prediction['action'][0].float().cpu().numpy()
    expected_action_shape = (metadata['action_horizon'], metadata['action_dim'])
    if action.shape != expected_action_shape:
      raise ValueError(f'invalid model action output: {action.shape}')
    if not np.isfinite(action).all():
      if request.get('save_input'):
        np.savez_compressed(
          request['save_input'], **arrays, action=action, c2w=c2w,
          wrists_world=wrists, tips_relative=tips,
        )
      print(json.dumps({
        'policy_action_rejected': 'nonfinite_action',
        'error': 'EgoTouch policy actions contain nonfinite values',
        'seconds': time.monotonic() - start,
      }, allow_nan=False), file=protocol, flush=True)
      continue
    decoded = decode_taskspace_action(action, **decode_args)
    # Float64 camera conversion, matching current observation's frozen head
    # camera reference.
    world_wrists = np.asarray(request['c2w']) @ decoded['wrist_T_ref']
    world_tips = world_wrists[:, :, None] @ decoded['fingertip_T_wrist']
    output = dict(
      wrists_world=world_wrists.tolist(), tips_world=world_tips.tolist(),
      seconds=time.monotonic() - start, roundtrip_max_error=roundtrip,
    )
    if request.get('save_input'):
      np.savez_compressed(
        request['save_input'], **arrays, action=action, c2w=c2w,
        wrists_world=wrists, tips_relative=tips,
      )
    print(json.dumps(output, allow_nan=False), file=protocol, flush=True)


if __name__ == '__main__':
  main()
