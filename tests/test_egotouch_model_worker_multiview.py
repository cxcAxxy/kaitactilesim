import argparse
import base64
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

WORKER_PATH = (
  Path(__file__).resolve().parents[1]
  / 'scripts' / 'workcell' / 'egotouch_model_worker.py'
)


def _load_worker():
  spec = importlib.util.spec_from_file_location('egotouch_model_worker_test', WORKER_PATH)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def _image(value):
  array = np.full((240, 320, 3), value, dtype=np.uint8)
  return base64.b64encode(array.tobytes()).decode()


def _view_record(mask=(True, False, True)):
  images = {}
  for name, valid in zip(
    ('head', 'left_wrist', 'right_wrist'), mask, strict=True,
  ):
    images[name] = {'valid': valid, 'image': {'path': name}} if valid else {'valid': False}
  return {'images': images}


def test_derives_fixed_multiview_contract_and_builds_masked_tensor():
  worker = _load_worker()
  binding = {
    'view_names': ['head', 'left_wrist', 'right_wrist'],
    'selector_records': {
      'session_a': {'00000': _view_record(), '00001': _view_record()},
      'session_b': {'00000': _view_record()},
    },
  }
  contract = worker._derive_view_contract(binding)
  assert contract == {
    'view_slots': ('head', 'left_wrist', 'right_wrist'),
    'view_mask': (True, False, True),
    'cameras': ('head', 'right_wrist'),
    'multiview': True,
  }

  arrays = worker._visual_arrays(
    {'rgbs': {'head': _image(51), 'right_wrist': _image(204)}}, contract,
  )
  assert arrays['x_rgb_views'].shape == (3, 3, 240, 320)
  assert arrays['x_rgb_views'].dtype == np.float32
  np.testing.assert_array_equal(arrays['view_mask'], [True, False, True])
  np.testing.assert_allclose(arrays['x_rgb_views'][0], 0.2)
  np.testing.assert_array_equal(arrays['x_rgb_views'][1], 0.0)
  np.testing.assert_allclose(arrays['x_rgb_views'][2], 0.8)


def test_multiview_contract_rejects_drift_and_request_camera_mismatch():
  worker = _load_worker()
  binding = {
    'view_names': list(worker.VIEW_SLOTS),
    'selector_records': {
      'session': {
        '00000': _view_record(),
        '00001': _view_record((True, True, True)),
      },
    },
  }
  with pytest.raises(ValueError, match='view mask is not fixed'):
    worker._derive_view_contract(binding)

  contract = worker._derive_view_contract({
    'view_names': list(worker.VIEW_SLOTS),
    'selector_records': {'session': {'00000': _view_record()}},
  })
  with pytest.raises(ValueError, match='rgbs camera keys differ'):
    worker._visual_arrays({'rgbs': {'head': _image(1)}}, contract)
  with pytest.raises(ValueError, match='rgbs camera keys differ'):
    worker._visual_arrays({
      'rgbs': {
        'head': _image(1), 'left_wrist': _image(1),
        'right_wrist': _image(1),
      },
    }, contract)


def test_legacy_binding_and_rgb_request_remain_supported():
  worker = _load_worker()
  contract = worker._derive_view_contract({
    'selector_records': {
      'session': {'00000': {'image': {'path': 'rgb.png'}}},
    },
  })
  assert contract['multiview'] is False
  assert contract['cameras'] == ('head',)
  assert contract['view_mask'] == (True, False, False)
  arrays = worker._visual_arrays({'rgb': _image(255)}, contract)
  assert set(arrays) == {'x_rgb'}
  assert arrays['x_rgb'].shape == (3, 240, 320)
  np.testing.assert_array_equal(arrays['x_rgb'], 1.0)


def test_checkpoint_binding_identity_and_external_candidate_without_best(tmp_path):
  worker = _load_worker()
  binding = {'stats': {'pos': {'mean': [0, 0, 0]}}}
  binding_digest = worker._canonical_sha256(binding)
  checkpoint_path = tmp_path / 'epoch_0050.pt'
  checkpoint_bytes = b'checkpoint-payload'
  checkpoint_path.write_bytes(checkpoint_bytes)
  checkpoint = {
    'schema_version': worker.EXTERNAL_CANDIDATE_SCHEMA,
    'epoch': 50,
    'global_step': 15800,
    'model': {'weight': object()},
    'cfg': {},
    'variant': 'no_tactile',
    'binding_sha256': binding_digest,
    'model_weights': 'ema',
  }
  index = {
    'schema_version': worker.EXTERNAL_INDEX_SCHEMA,
    'variant': 'no_tactile',
    'binding_sha256': binding_digest,
    'candidates': [{
      'epoch': 50,
      'global_step': 15800,
      'model_weights': 'ema',
      'checkpoint_ref': {
        'path': str(checkpoint_path),
        'bytes': len(checkpoint_bytes),
        'sha256': worker._sha256_bytes(checkpoint_bytes),
      },
    }],
  }
  (tmp_path / 'index.json').write_text(json.dumps(index))
  assert worker._validate_checkpoint(
    checkpoint, checkpoint_path, checkpoint_bytes, binding_digest,
  ) == 'ema'
  assert 'best' not in checkpoint

  with pytest.raises(ValueError, match='checkpoint/binding SHA-256 mismatch'):
    worker._validate_checkpoint(
      checkpoint, checkpoint_path, checkpoint_bytes, '0' * 64,
    )
  index['candidates'][0]['checkpoint_ref']['sha256'] = '0' * 64
  (tmp_path / 'index.json').write_text(json.dumps(index))
  with pytest.raises(ValueError, match='checkpoint SHA-256 differs'):
    worker._validate_checkpoint(
      checkpoint, checkpoint_path, checkpoint_bytes, binding_digest,
    )


def test_artifact_resolution_keeps_snapshot_compatibility(tmp_path):
  worker = _load_worker()
  snapshot = tmp_path / 'snapshot'
  snapshot.mkdir()
  checkpoint = snapshot / 'best.pt'
  binding = snapshot / 'egotouch_binding.json'
  checkpoint.write_bytes(b'checkpoint')
  binding.write_text('{}')
  args = argparse.Namespace(snapshot=snapshot, checkpoint=None, binding=None)
  assert worker._resolve_artifacts(args) == (
    checkpoint.resolve(), binding.resolve(),
  )

  with pytest.raises(SystemExit):
    worker.parse_args([
      '--project', str(tmp_path), '--checkpoint', str(checkpoint),
    ])
