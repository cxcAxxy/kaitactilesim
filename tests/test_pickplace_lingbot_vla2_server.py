"""CPU-only contract tests for the standalone LingBot policy server."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/workcell/serve_pickplace_lingbot_vla2_policy.py"


def _load_server():
  spec = importlib.util.spec_from_file_location("serve_pickplace_lingbot_vla2_policy", SCRIPT)
  module = importlib.util.module_from_spec(spec)
  assert spec.loader is not None
  spec.loader.exec_module(module)
  return module


def _sha(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(value), encoding="utf-8")


def _assets(tmp_path: Path, server) -> tuple[Path, dict, Path]:
  repo = tmp_path / "lingbot-repo"
  (repo / "deploy").mkdir(parents=True)
  (repo / "deploy/lingbot_vla_v2_policy.py").write_text("", encoding="utf-8")
  (repo / "deploy/websocket_policy_server.py").write_text("", encoding="utf-8")
  robot_path = repo / "configs/robot_configs/kaihand_usb.yaml"
  robot = {
    "states": [
      {"observation.state.arm.position": {
        "origin_keys": "observation.state.right_arm_joint_position"
      }},
      *[
        {f"observation.state.hand{suffix}.position": {
          "origin_keys": [{"observation.state.right_hand_joint_position": {
            "start": start, "end": end
          }}]
        }}
        for suffix, start, end in (
          ("", 0, 12), ("_extra_ring", 12, 16), ("_extra_pinky", 16, 20)
        )
      ],
    ],
    "actions": [
      {"action.arm.position": {
        "origin_keys": "auxiliary.action.right_arm_joint_target",
        "subtract_state": True,
        "relative_type": "delta",
      }},
      *[
        {f"action.hand{suffix}.position": {
          "origin_keys": [{"action.right_hand_joint_position": {
            "start": start, "end": end
          }}],
          "subtract_state": False,
        }}
        for suffix, start, end in (
          ("", 0, 12), ("_extra_ring", 12, 16), ("_extra_pinky", 16, 20)
        )
      ],
    ],
    "images": [
      {"observation.images.camera_top": {
        "origin_keys": "observation.images.head"
      }},
      {"observation.images.camera_wrist_right": {
        "origin_keys": "observation.images.right_wrist"
      }},
    ],
    "norm_stats": "unused-generic-normalizer.json",
  }
  _write_json(robot_path, robot)  # JSON is also valid YAML.
  norm_path = tmp_path / "normalizer.json"
  norm = {
    f"{prefix}.{suffix}": {"mean": [0] * dim, "std": [1] * dim}
    for prefix in ("action", "observation.state")
    for suffix, dim in (
      ("arm.position", 7), ("hand.position", 12),
      ("hand_extra_ring.position", 4), ("hand_extra_pinky.position", 4),
    )
  }
  _write_json(norm_path, {"norm_stats": norm})
  dataset = tmp_path / "reference"
  _write_json(dataset / "meta/info.json", {"fps": 30})
  tokenizer = tmp_path / "tokenizer"
  _write_json(tokenizer / "tokenizer_config.json", {})
  run = tmp_path / "finetune"
  checkpoint = run / "checkpoints/global_step_3905/hf_ckpt"
  checkpoint.mkdir(parents=True)
  _write_json(checkpoint / "config.json", {
    "architectures": ["LingbotVlaV2Policy"],
    "action_dim": 55, "max_action_dim": 55, "max_state_dim": 55,
    "chunk_size": 50, "n_action_steps": 50, "post_training": True,
    "tokenizer_path": str(tokenizer),
  })
  (checkpoint / "model-00001-of-00001.safetensors").write_bytes(b"fake-weights")
  _write_json(checkpoint / "model.safetensors.index.json", {
    "weight_map": {"model.weight": "model-00001-of-00001.safetensors"}
  })
  (checkpoint / "chat_template.jinja").write_text(
    "{{ '<|im_start|>' + messages[0]['role'] + '\\n' + "
    "messages[0]['content'] + '<|im_end|>' }}",
    encoding="utf-8",
  )
  training_path = run / "lingbotvla_cli.yaml"
  _write_json(training_path, {
    "model": {
      "config_key": "LingbotVLAV2Config",
      "tokenizer_path": str(tokenizer),
    },
    "data": {
      "data_name": "kaihand_usb",
      "cameras": ["camera_top", "camera_wrist_right"],
      "img_size": 256,
      "train_path": str(dataset),
      "norm_stats_file": str(norm_path),
      "joints": [
        {"arm.position": 14}, {"end.position": 14},
        {"effector.position": 2}, {"hand_extra_pinky.position": 4},
        {"head.position": 2}, {"base.position": 3},
        {"hand.position": 12}, {"hand_extra_ring.position": 4},
      ],
      "norm_type": [
        {"arm.position": "meanstd"},
        {"hand.position": "meanstd"},
        {"hand_extra_ring.position": "meanstd"},
        {"hand_extra_pinky.position": "meanstd"},
      ],
    },
    "train": {
      "chunk_size": 50, "action_dim": 55,
      "max_action_dim": 55, "max_state_dim": 55,
    },
  })
  rows = [
    {"path": path.relative_to(checkpoint).as_posix(),
     "size": path.stat().st_size, "sha256": _sha(path)}
    for path in sorted(checkpoint.iterdir()) if path.is_file()
  ]
  manifest = {
    "schema": server.SCHEMA,
    "task": server.TASK,
    "model_family": server.MODEL_FAMILY,
    "checkpoint_path": str(checkpoint),
    "checkpoint_files": rows,
    "checkpoint_sha256": server._canonical_sha256(rows),
    "training_config_path": str(training_path),
    "training_config_sha256": _sha(training_path),
    "robot_config_path": str(robot_path),
    "robot_config_sha256": _sha(robot_path),
    "norm_stats_path": str(norm_path),
    "norm_stats_sha256": _sha(norm_path),
    "lingbot_repo_path": str(repo),
    "reference_dataset": str(dataset),
    "reference_episode_index": 0,
    "prediction_horizon": server.HORIZON,
    "model_action_dim": server.MODEL_ACTION_DIM,
    "action_dim": server.ACTION_DIM,
    "control_hz": server.CONTROL_HZ,
    "joint_names": list(server.JOINT_NAMES),
    "instruction": server.INSTRUCTION,
    "observation_contract": server.OBSERVATION_CONTRACT,
    "action_representation": server.ACTION_REPRESENTATION,
  }
  manifest["deployment_id"] = server._canonical_sha256(manifest)
  manifest_path = tmp_path / "deployment_manifest.json"
  _write_json(manifest_path, manifest)
  return manifest_path, manifest, repo


def _json_yaml(monkeypatch) -> None:
  monkeypatch.setitem(
    sys.modules, "yaml",
    types.SimpleNamespace(safe_load=json.loads),
  )


def test_server_static_validation_and_metadata(tmp_path, monkeypatch):
  server = _load_server()
  path, _, repo = _assets(tmp_path, server)
  _json_yaml(monkeypatch)
  resolved, manifest = server.load_manifest(path)
  checkpoint = server.validate_runtime_assets(manifest, repo, verify_hashes=True)
  metadata = server.deployment_metadata(resolved, manifest)
  assert checkpoint.name == "hf_ckpt"
  assert metadata["prediction_horizon"] == metadata["action_horizon"] == 50
  assert metadata["model_action_dim"] == 55
  assert metadata["action_dim"] == 27
  assert metadata["observation_contract"]["cameras"] == ["head", "right_wrist"]
  assert metadata["action_representation"]["server_right_arm"] == (
    "absolute_joint_position_target"
  )
  server.main(["--deployment-manifest", str(path), "--validate-only"])


def test_server_rejects_robot_mapping_drift(tmp_path, monkeypatch):
  server = _load_server()
  path, manifest, repo = _assets(tmp_path, server)
  robot_path = Path(manifest["robot_config_path"])
  robot = json.loads(robot_path.read_text(encoding="utf-8"))
  robot["actions"][0]["action.arm.position"]["relative_type"] = "quaternion_local"
  _write_json(robot_path, robot)
  manifest["robot_config_sha256"] = _sha(robot_path)
  manifest["deployment_id"] = server._canonical_sha256({
    key: value for key, value in manifest.items() if key != "deployment_id"
  })
  _write_json(path, manifest)
  _json_yaml(monkeypatch)
  _, loaded = server.load_manifest(path)
  with pytest.raises(RuntimeError, match="robot mapping differs"):
    server.validate_runtime_assets(loaded, repo)


def test_server_rejects_checkpoint_mutation_and_path_traversal(tmp_path, monkeypatch):
  server = _load_server()
  path, manifest, repo = _assets(tmp_path, server)
  _json_yaml(monkeypatch)
  shard = Path(manifest["checkpoint_path"]) / "model-00001-of-00001.safetensors"
  shard.write_bytes(b"different-weights")
  with pytest.raises(RuntimeError, match="checkpoint payload changed"):
    server.validate_runtime_assets(manifest, repo)

  manifest["checkpoint_files"][0]["path"] = "../outside"
  with pytest.raises(ValueError, match="unsafe or duplicate"):
    server._check_checkpoint_files(
      manifest, Path(manifest["checkpoint_path"]), verify_hashes=False
    )


def test_server_rejects_wrong_task_before_model_import(tmp_path):
  server = _load_server()
  path, manifest, _ = _assets(tmp_path, server)
  manifest["task"] = "usb-insert"
  _write_json(path, manifest)
  with pytest.raises(RuntimeError, match="manifest contract mismatch"):
    server.main(["--deployment-manifest", str(path)])


def test_server_rejects_manifest_identity_tampering(tmp_path):
  server = _load_server()
  path, manifest, _ = _assets(tmp_path, server)
  manifest["norm_stats_sha256"] = "f" * 64
  _write_json(path, manifest)
  with pytest.raises(RuntimeError, match="deployment_id does not match"):
    server.load_manifest(path)


def test_server_requires_cuda_before_importing_lingbot(tmp_path, monkeypatch):
  server = _load_server()
  path, manifest, repo = _assets(tmp_path, server)
  monkeypatch.delenv("QWEN3VL_PATH", raising=False)
  monkeypatch.setitem(
    sys.modules, "torch",
    types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)),
  )
  with pytest.raises(RuntimeError, match="requires a CUDA GPU"):
    server.serve(path, manifest, repo, host="127.0.0.1", port=8006)


def test_server_sets_lingbot_cwd_and_resets_robot(tmp_path, monkeypatch):
  server = _load_server()
  path, manifest, repo = _assets(tmp_path, server)
  monkeypatch.delenv("QWEN3VL_PATH", raising=False)
  events: dict = {}

  class FakePolicy:
    def __init__(self, **kwargs):
      events["policy_kwargs"] = kwargs
      events["cwd"] = Path.cwd()
      tokenizer = types.SimpleNamespace(
        chat_template=None,
        apply_chat_template=lambda messages, **_kwargs: (
          "<|im_start|>user\n" + messages[0]["content"] + "<|im_end|>"
        ),
      )
      self.processor = types.SimpleNamespace(tokenizer=tokenizer)
      self.language_tokenizer = tokenizer
      self.vla = types.SimpleNamespace(
        feature_transform=types.SimpleNamespace(tokenizer=tokenizer)
      )
      events["tokenizer"] = tokenizer

    def reset(self, name):
      events["robot_name"] = name

  class FakeWebsocketServer:
    def __init__(self, **kwargs):
      events["server_kwargs"] = kwargs

    def serve_forever(self):
      events["served"] = True

  monkeypatch.setitem(
    sys.modules, "torch",
    types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True)),
  )
  monkeypatch.setitem(sys.modules, "deploy", types.ModuleType("deploy"))
  monkeypatch.setitem(
    sys.modules, "deploy.lingbot_vla_v2_policy",
    types.SimpleNamespace(LingbotVLAv2Server=FakePolicy),
  )
  monkeypatch.setitem(
    sys.modules, "deploy.websocket_policy_server",
    types.SimpleNamespace(WebsocketPolicyServer=FakeWebsocketServer),
  )
  original_cwd = Path.cwd()
  try:
    server.serve(path, manifest, repo, host="127.0.0.1", port=8006)
  finally:
    os.chdir(original_cwd)
  assert events["cwd"] == repo
  assert events["robot_name"] == "kaihand_usb"
  assert events["policy_kwargs"]["use_length"] == 50
  assert events["policy_kwargs"]["chunk_ret"] is True
  assert events["policy_kwargs"]["use_compile"] is False
  assert events["server_kwargs"]["metadata"]["checkpoint_sha256"] == (
    manifest["checkpoint_sha256"]
  )
  assert events["served"] is True
  assert os.environ["QWEN3VL_PATH"] == str(tmp_path / "tokenizer")
  assert "<|im_start|>" in events["tokenizer"].chat_template


def test_server_rejects_unfrozen_qwen_override(tmp_path, monkeypatch):
  server = _load_server()
  path, manifest, repo = _assets(tmp_path, server)
  monkeypatch.setenv("QWEN3VL_PATH", str(tmp_path / "different-qwen-model"))
  with pytest.raises(RuntimeError, match="QWEN3VL_PATH points to"):
    server.serve(path, manifest, repo, host="127.0.0.1", port=8006)
  _json_yaml(monkeypatch)
  with pytest.raises(RuntimeError, match="QWEN3VL_PATH points to"):
    server.main(["--deployment-manifest", str(path), "--validate-only"])


def test_server_rejects_invalid_checkpoint_chat_template(tmp_path):
  server = _load_server()
  _, manifest, _ = _assets(tmp_path, server)
  checkpoint = Path(manifest["checkpoint_path"])
  (checkpoint / "chat_template.jinja").write_text("", encoding="utf-8")
  with pytest.raises(RuntimeError, match="chat template is invalid"):
    server.checkpoint_chat_template(checkpoint)
