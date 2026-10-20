import threading
from contextlib import contextmanager

import pytest
from kaihand_tactile_env.pipeline.deployment import (
  DeploymentError,
  adapter_for,
  probe_server,
  validate_server_metadata,
  wait_for_server,
)
from kaihand_tactile_env.shared.egosteer_client import pack_payload
from websockets.exceptions import ConnectionClosed
from websockets.sync.server import serve


def manifest_and_metadata():
  manifest = {
    "deployment_id": "a" * 64,
    "checkpoint_sha256": "b" * 64,
    "prediction_horizon": 50,
    "action_dim": 27,
    "observation_contract": {
      "cameras": ["head", "right_wrist"],
      "tactile_sent_to_model": False,
    },
    "action_representation": {"arm": "absolute_joint_target", "hand": "absolute_joint_target"},
  }
  metadata = {
    **manifest,
    "evaluation_task": "pick-place",
    "model_family": "pi0.5",
    "action_horizon": 50,
  }
  return manifest, metadata


@contextmanager
def metadata_server(metadata):
  def connection_handler(connection):
    connection.send(pack_payload(metadata))
    try:
      connection.recv()
    except ConnectionClosed:
      pass

  with serve(connection_handler, "127.0.0.1", 0, logger=None) as server:
    port = server.socket.getsockname()[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
      yield f"ws://127.0.0.1:{port}"
    finally:
      server.shutdown()
      thread.join(timeout=5)


def test_real_websocket_handshake_preserves_frozen_contract():
  manifest, metadata = manifest_and_metadata()
  with metadata_server(metadata) as endpoint:
    observed = probe_server(endpoint, timeout=2)
    assert observed == metadata
    assert wait_for_server(
      endpoint, manifest, adapter_for("pick-place", "pi05"), timeout=2
    ) == metadata


def test_real_websocket_server_with_stale_checkpoint_is_rejected():
  manifest, metadata = manifest_and_metadata()
  metadata["checkpoint_sha256"] = "c" * 64
  with metadata_server(metadata) as endpoint:
    with pytest.raises(DeploymentError, match="checkpoint_sha256"):
      validate_server_metadata(
        probe_server(endpoint, timeout=2), manifest, adapter_for("pick-place", "pi05")
      )


@pytest.mark.parametrize(
  ("field", "replacement"),
  [
    ("observation_contract", {"cameras": ["head"], "tactile_sent_to_model": False}),
    ("action_representation", {"arm": "measured_state", "hand": "absolute_joint_target"}),
  ],
)
def test_matching_id_does_not_allow_a_changed_input_or_action_contract(field, replacement):
  manifest, metadata = manifest_and_metadata()
  metadata[field] = replacement
  with pytest.raises(DeploymentError, match=field):
    validate_server_metadata(metadata, manifest, adapter_for("pick-place", "pi05"))
