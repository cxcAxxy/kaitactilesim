from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/analysis/tactile_task_readiness.py"


@pytest.fixture(scope="module")
def readiness():
  spec = importlib.util.spec_from_file_location("tactile_task_readiness_test", SCRIPT)
  module = importlib.util.module_from_spec(spec)
  assert spec.loader is not None
  spec.loader.exec_module(module)
  return module


def _write_json(path: Path, value: object) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _valid_ftp1_checkpoint(tmp_path: Path, readiness, task: str) -> Path:
  contract = readiness.FTP1_TASK_CONTRACTS[task]
  corpus = tmp_path / "corpus"
  corpus.mkdir(parents=True)
  _write_json(
    corpus / "conversion_manifest.json",
    {
      "task": task,
      "instruction": contract["instruction"],
      "fps": 30,
      "ftp1_action_horizon": 32,
      "action_supervision": {
        "contract": "ftp1-native-future-measured-state-trajectory-v1",
        "source": (
          "measured wrist poses and hand joint positions; actions are built "
          "by the native FTP-1 loader"
        ),
        "controller_targets_used": False,
        "raw_controller_targets": (
          "may be present in Raw; not used as this export's action supervision"
        ),
        "pi05_recorded_controller_target_equivalent": False,
        "online_deployment_ready": False,
      },
    },
  )
  dataset_config = tmp_path / "dataset_config.json"
  _write_json(
    dataset_config,
    {
      "datasets": [
        {"name": contract["domain_name"], "path": str(corpus), "enabled": True}
      ]
    },
  )

  checkpoint = tmp_path / "checkpoint"
  checkpoint.mkdir()
  _write_json(
    checkpoint / "model_config.json",
    {
      "action_dim": 120,
      "action_horizon": 32,
      "discrete_state_input": False,
      "state_input_mode": "adarms",
      "use_tactile_input": True,
      "disable_history": True,
    },
  )
  _write_json(
    checkpoint / "train_config.json",
    {
      "dataset_config_path": str(dataset_config),
      "single_arm_action_rep_dim": 48,
      "reserved_action_dim": 15,
      "disable_history": True,
      "action_down_sample_steps": 1,
      "proprioception_pose_rep": "relative",
      "action_pose_rep": "relative",
      "proprioception_joint_rep": "abs",
      "action_joint_rep": "mix",
      "independent_norm_mode": "all",
      "norm_time_dim": False,
      "norm_type": "zscore",
      "norm_image_tactile_mode": "channel_wise",
    },
  )
  _write_json(
    checkpoint / "tactile_input_config_file.json",
    {
      contract["domain_name"]: {
        "right_tactile_fingers": {
          "shape": [1, 5, 7, 5, 3],
          "function_areas": [0, 1, 2, 3, 4],
          "sensor": "KaiHandSimTaxelV1",
          "type": "matrix",
        }
      }
    },
  )
  _write_json(
    checkpoint
    / "normalization"
    / contract["domain_name"]
    / "independent_norm_stats_all_t0_zscore.json",
    {
      "norm_time_dim": False,
      "norm_type": "zscore",
      "data_dim": {
        "tactile": {"KaiHandSimTaxelV1_matrix_7_5_3": 3},
        "actions": 120,
        "state": 120,
      },
      "params": {
        "state": {},
        "actions": {},
        "tactile": {"KaiHandSimTaxelV1_matrix_7_5_3": {}},
      },
    },
  )
  (checkpoint / "model.safetensors").write_bytes(b"trained-weights")
  hpt = checkpoint / "hpt_tokenizer/KaiHandSimTaxelV1_matrix_7_5_3.safetensors"
  hpt.parent.mkdir()
  hpt.write_bytes(b"trained-tokenizer")
  return checkpoint


def test_valid_checkpoint_requires_model_recipe_domain_and_corpus_binding(
  tmp_path, readiness
):
  checkpoint = _valid_ftp1_checkpoint(tmp_path, readiness, "bulb-screw")

  result = readiness.inspect_ftp_checkpoint(checkpoint, "bulb-screw")

  assert result["matched"] is True
  assert all(result["checks"].values())
  assert result["training_binding"]["task"] == "bulb-screw"
  assert result["weights_loaded"] is False
  assert result["weights_hashed"] is False


def test_normalization_directory_is_never_treated_as_a_trained_checkpoint(
  tmp_path, readiness
):
  contract = readiness.FTP1_TASK_CONTRACTS["vase-wipe"]
  checkpoint = tmp_path / "normalizer-only"
  _write_json(
    checkpoint
    / "normalization"
    / contract["domain_name"]
    / "independent_norm_stats_all_t0_zscore.json",
    {
      "norm_time_dim": False,
      "norm_type": "zscore",
      "data_dim": {
        "tactile": {"KaiHandSimTaxelV1_matrix_7_5_3": 3},
        "actions": 120,
        "state": 120,
      },
      "params": {
        "state": {},
        "actions": {},
        "tactile": {"KaiHandSimTaxelV1_matrix_7_5_3": {}},
      },
    },
  )

  result = readiness.inspect_ftp_checkpoint(checkpoint, "vase-wipe")

  assert result["matched"] is False
  assert result["checks"]["domain_normalization"] is True
  assert result["checks"]["model_weights_present"] is False
  assert result["checks"]["model_h32_d120"] is False
  assert result["normalization_is_not_training_evidence"] is True


def test_wrong_task_dataset_binding_is_rejected(tmp_path, readiness):
  checkpoint = _valid_ftp1_checkpoint(tmp_path, readiness, "bulb-screw")

  result = readiness.inspect_ftp_checkpoint(checkpoint, "bulb-unscrew")

  assert result["matched"] is False
  assert result["checks"]["task_dataset_binding"] is False
  assert any("task_dataset_binding" in error for error in result["errors"])


@pytest.mark.parametrize("corruption", ("missing", "wrong"))
def test_nonpoker_checkpoint_rejects_unproven_action_supervision(
  tmp_path, readiness, corruption
):
  checkpoint = _valid_ftp1_checkpoint(tmp_path, readiness, "bulb-screw")
  train = json.loads((checkpoint / "train_config.json").read_text(encoding="utf-8"))
  dataset_config = json.loads(
    Path(train["dataset_config_path"]).read_text(encoding="utf-8")
  )
  conversion = Path(dataset_config["datasets"][0]["path"]) / "conversion_manifest.json"
  document = json.loads(conversion.read_text(encoding="utf-8"))
  if corruption == "missing":
    document.pop("action_supervision")
  else:
    document["action_supervision"][
      "pi05_recorded_controller_target_equivalent"
    ] = True
  _write_json(conversion, document)

  result = readiness.inspect_ftp_checkpoint(checkpoint, "bulb-screw")

  assert result["matched"] is False
  assert result["checks"]["task_dataset_binding"] is False
  assert any("action_supervision" in error for error in result["errors"])


def test_candidate_scan_is_shallow_and_bounded(tmp_path, readiness):
  checkpoint = _valid_ftp1_checkpoint(
    tmp_path / "root/ftp1/run", readiness, "bulb-screw"
  )
  scan_root = tmp_path / "root"
  nested_checkpoint = checkpoint.parent / "nested" / "too-deep"
  nested_checkpoint.mkdir(parents=True)
  (nested_checkpoint / "model.safetensors").write_bytes(b"not-discovered")

  result = readiness.discover_ftp_candidates(
    "bulb-screw", roots={"bulb-screw": (scan_root,)}
  )

  assert result["max_depth"] == 3
  assert result["truncated"] is False
  assert result["candidates"] == [str(checkpoint.resolve())]


def test_matrix_lists_all_13_tasks_without_claiming_nonpoker_cpm(
  tmp_path, readiness, monkeypatch
):
  poker = _valid_ftp1_checkpoint(tmp_path, readiness, "poker-draw")
  monkeypatch.setattr(readiness, "KNOWN_POKER_FTP1_CHECKPOINT", poker)
  monkeypatch.setattr(
    readiness,
    "inspect_cpm_bundle",
    lambda: {
      "task": "poker-draw",
      "status": "known_checkpoint_matched",
      "matched": True,
      "weights_loaded": False,
      "weights_hashed": False,
    },
  )

  report = readiness.build_readiness(scan_roots={})
  rows = {row["task"]: row for row in report["tasks"]}

  assert report["task_count"] == 13
  assert set(rows) == set(readiness.TASKS)
  assert rows["poker-draw"]["cpm"]["status"] == "known_checkpoint_matched"
  assert rows["poker-draw"]["ftp1"]["status"] == "known_checkpoint_matched"
  assert rows["pick-place"]["ftp1"]["status"] == "adapter_pending"
  for task in readiness.SHARED_TACTILE_TASKS:
    assert rows[task]["ftp1"]["status"] == "adapter_ready_awaiting_checkpoint"
  for task in set(readiness.TASKS) - {"poker-draw"}:
    assert rows[task]["cpm"]["status"] == "awaiting_task_specific_cpm_and_base"


def test_usb_checkpoint_uses_usb_domain_corpus_and_native_action_contract(
  tmp_path, readiness
):
  checkpoint = _valid_ftp1_checkpoint(tmp_path, readiness, "usb-insert")

  result = readiness.inspect_ftp_checkpoint(checkpoint, "usb-insert")

  assert result["matched"] is True
  assert result["domain_name"] == "KaiHandSimUsbInsert"
  assert result["training_binding"]["task"] == "usb-insert"
  assert result["training_binding"]["action_supervision"]["contract"] == (
    "ftp1-native-future-measured-state-trajectory-v1"
  )


def test_explicit_checkpoint_cli_emits_json_without_loading_weights(
  tmp_path, readiness, capsys
):
  checkpoint = _valid_ftp1_checkpoint(tmp_path, readiness, "install-ram")

  assert readiness.main(["--task", "install-ram", "--checkpoint", str(checkpoint)]) == 0
  report = json.loads(capsys.readouterr().out)

  assert report["mode"] == "explicit_ftp1_checkpoint"
  assert report["inspection"]["matched"] is True
  assert report["inspection"]["weights_loaded"] is False
