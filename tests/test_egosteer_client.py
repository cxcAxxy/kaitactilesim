from __future__ import annotations

import asyncio
import io
from typing import Any

import numpy as np
import pytest
from kaihand_tactile_env.workcell.egosteer_client import (
  EgoSteerPolicyClient,
  ObservationHistory,
  encode_jpeg_sequence,
  pack_payload,
  unpack_payload,
)
from PIL import Image
from websockets.asyncio.server import serve


def test_numpy_msgpack_round_trip_is_writable() -> None:
  source = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
  decoded = unpack_payload(pack_payload({"array": source, "scalar": np.int16(7)}))

  assert np.array_equal(decoded["array"], source)
  assert decoded["array"].flags.writeable
  assert decoded["scalar"] == 7
  decoded["array"][0, 0, 0] = -1


def test_numpy_msgpack_rejects_unsupported_dtype() -> None:
  with pytest.raises(ValueError, match="unsupported NumPy dtype"):
    pack_payload(np.array([object()], dtype=object))


def test_observation_history_uses_repeat_padding_and_stride() -> None:
  history = ObservationHistory(horizon=3, stride=2)
  for value in range(1, 7):
    history.append(
      np.full((2, 3, 3), value, dtype=np.uint8),
      np.full(48, value, dtype=np.float32),
    )

    selected = history.select()
    expected = {
      1: [1, 1, 1],
      2: [1, 1, 2],
      3: [1, 1, 3],
      4: [1, 2, 4],
      5: [1, 3, 5],
      6: [2, 4, 6],
    }[value]
    assert selected.images[:, 0, 0, 0].tolist() == expected
    assert selected.raw_states[:, 0].tolist() == expected

  assert len(history) == history.capacity == 5


def test_jpeg_sequence_schema_and_rgb_order() -> None:
  images = np.zeros((2, 8, 9, 3), dtype=np.uint8)
  images[0, ..., 0] = 255
  images[1, ..., 1] = 200

  encoded = encode_jpeg_sequence(images, quality=95)

  assert encoded["shape"] == (2, 8, 9, 3)
  assert encoded["dtype"] == "uint8"
  assert encoded["color_order"] == "rgb"
  assert encoded["encoded_nbytes"] == sum(map(len, encoded["frames"]))
  decoded = np.asarray(Image.open(io.BytesIO(encoded["frames"][0])))
  assert decoded.shape == (8, 9, 3)
  assert float(decoded[..., 0].mean()) > 245.0
  assert float(decoded[..., 1:].mean()) < 10.0


def test_policy_client_end_to_end_protocol() -> None:
  async def scenario() -> tuple[np.ndarray, dict[str, Any]]:
    observed: dict[str, Any] = {}

    async def handler(websocket: Any) -> None:
      await websocket.send(
        pack_payload({"action_horizon": 32, "action_dim": 48, "model": "fake"})
      )
      message = await websocket.recv()
      assert isinstance(message, bytes)
      observed.update(unpack_payload(message))
      await websocket.send(
        pack_payload(
          {
            "pred_actions": np.full((32, 48), 0.25, dtype=np.float32),
            "server_timing": {"infer_ms": 12.5},
          }
        )
      )

    async with serve(handler, "127.0.0.1", 0) as server:
      port = server.sockets[0].getsockname()[1]
      async with EgoSteerPolicyClient(
        f"ws://127.0.0.1:{port}", response_timeout=2.0
      ) as client:
        response = await client.infer(
          images=np.zeros((6, 12, 16, 3), dtype=np.uint8),
          states=np.zeros((6, 48), dtype=np.float32),
          camera_intrinsics=np.eye(3),
          instruction="Put the red cylinder into the blue box.",
        )
    return response.pred_actions, observed

  actions, observed = asyncio.run(scenario())

  assert actions.shape == (32, 48)
  assert np.all(actions == np.float32(0.25))
  assert observed["states"].shape == (6, 48)
  assert observed["image"]["__image_encoding__"] == "jpeg_sequence"
  assert len(observed["image"]["frames"]) == 6
  assert observed["instruction"] == "Put the red cylinder into the blue box."
  assert observed["action_rtc"] is None


def test_policy_client_multi_camera_protocol() -> None:
  async def scenario() -> dict[str, Any]:
    observed: dict[str, Any] = {}

    async def handler(websocket: Any) -> None:
      await websocket.send(
        pack_payload(
          {
            "action_horizon": 4,
            "action_dim": 48,
            "cameras": ["head", "right_wrist"],
          }
        )
      )
      observed.update(unpack_payload(await websocket.recv()))
      await websocket.send(
        pack_payload({"pred_actions": np.zeros((4, 48), dtype=np.float32)})
      )

    async with serve(handler, "127.0.0.1", 0) as server:
      port = server.sockets[0].getsockname()[1]
      async with EgoSteerPolicyClient(
        f"ws://127.0.0.1:{port}", response_timeout=2.0
      ) as client:
        await client.infer(
          images={
            "head": np.zeros((3, 8, 9, 3), dtype=np.uint8),
            "right_wrist": np.ones((3, 8, 9, 3), dtype=np.uint8),
          },
          states=np.zeros((3, 48), dtype=np.float32),
          camera_intrinsics={"head": np.eye(3), "right_wrist": np.eye(3)},
          instruction="move",
        )
    return observed

  observed = asyncio.run(scenario())

  assert list(observed["image"]) == ["head", "right_wrist"]
  assert list(observed["camera_intrinsics"]) == ["head", "right_wrist"]
  assert len(observed["image"]["head"]["frames"]) == 3
  assert len(observed["image"]["right_wrist"]["frames"]) == 3
