import json
import subprocess
import sys
from pathlib import Path
from runpy import run_path

import h5py
from kaihand_tactile_env.pipeline.conversion import adapter_catalog

SCRIPT = Path(__file__).parents[1] / "scripts/convert/convert.py"


def _raw_episode(
  path: Path,
  scene: str = "poker-draw",
  cameras=("head", "right_wrist"),
) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with h5py.File(path, "w") as file:
    file.attrs["schema_version"] = "kaihand_raw_v1"
    file.attrs["control_hz"] = 100
    file.attrs["camera_hz"] = 30
    file.attrs["runtime_contract_json"] = json.dumps({
      "physics": {"timestep_s": 0.002},
      "tactile": {"source": "touch-sensors", "unit": "N"},
    })
    file.attrs["metadata_json"] = json.dumps({
      "scene": scene,
      "action_source": "commands/arm_joint_target",
      "camera_contract": "recorded RGB streams",
      "control_contract": "recorded actuator targets",
    })
    camera_group = file.create_group("cameras")
    for camera in cameras:
      camera_group.create_group(camera)
    commands = file.create_group("commands")
    commands.create_dataset("arm_joint_target", shape=(2, 7), dtype="f8")
    commands.create_dataset("hand_joint_target", shape=(2, 20), dtype="f8")
  path.with_suffix(".json").write_text(json.dumps({
    "episode": path.name,
    "sha256": "a" * 64,
  }), encoding="utf-8")


def _adapter(task: str, output_format: str) -> dict:
  return next(
    row for row in adapter_catalog()
    if (row["task"], row["format"]) == (task, output_format)
  )


def test_adapter_catalog_exposes_runtime_capabilities_and_worker_meaning():
  shared = _adapter("whiteboard-wipe", "pi05")
  assert shared["runtime"] == {
    "kind": "pi05-python",
    "backend_location": "workcell",
    "profile_fields": ["python", "project_root", "env"],
  }
  assert shared["workers"]["backend_option"] == "--fingerprint-workers"
  assert "fast episode conversion workers" in shared["workers"]["semantics"]
  assert {"task", "cameras", "openpi_root"}.issubset(
    shared["capabilities"]
  )

  pickplace = _adapter("pick-place", "egosteer")
  assert pickplace["workers"]["backend_option"] is None
  assert "expected_episodes" not in pickplace["capabilities"]


def test_named_runtime_and_raw_contract_are_in_dry_run_plan(tmp_path, capsys):
  source = tmp_path / "raw"
  _raw_episode(source / "episode.h5")
  config_dir = tmp_path / "runtime"
  python = tmp_path / "bin/python"
  python.parent.mkdir()
  python.write_text(f"#!{sys.executable}\n", encoding="utf-8")
  python.chmod(0o755)
  openpi = tmp_path / "openpi"
  openpi.mkdir()
  config_dir.mkdir()
  config = config_dir / "runtimes.toml"
  config.write_text(
    """
[runtimes.pi05]
python = "../bin/python"
project_root = "../openpi"

[runtimes.pi05.env]
API_TOKEN = "do-not-publish"
VISIBLE_SETTING = "kept"
""".strip(),
    encoding="utf-8",
  )

  module = run_path(str(SCRIPT))
  assert module["main"]([
    "--input-dir", str(source),
    "--output-dir", str(tmp_path / "output"),
    "--format", "pi05",
    "--task", "poker-draw",
    "--cameras", "head", "right_wrist",
    "--runtime-config", str(config),
    "--runtime", "pi05",
    "--dry-run",
  ]) == 0
  plan = json.loads(capsys.readouterr().out)
  assert plan["schema"] == "kaihand_conversion_plan_v1"
  assert plan["schema_version"] == 1
  assert plan["stage"] == "convert"
  assert plan["command"][0] == str(python.resolve())
  openpi_index = plan["command"].index("--openpi-root")
  assert plan["command"][openpi_index + 1] == str(openpi.resolve())
  assert plan["runtime"]["profile"]["env"] == {
    "API_TOKEN": "<redacted>",
    "VISIBLE_SETTING": "kept",
  }
  assert plan["runtime"]["pythonpath_prepend"] == [
    str(Path(__file__).parents[1] / "src"),
    str(openpi.resolve() / "src"),
    str(openpi.resolve() / "packages/openpi-client/src"),
  ]
  episode_contract = plan["source_contract"]["episodes"][0]
  assert episode_contract["root_facts"]["runtime_contract_json"]["physics"] == {
    "timestep_s": 0.002
  }
  assert episode_contract["runtime_contract"] == {
    "status": "declared",
    "source_attribute": "runtime_contract_json",
    "value": episode_contract["root_facts"]["runtime_contract_json"],
  }
  assert episode_contract["declared_contracts"]["action_source"] == (
    "commands/arm_joint_target"
  )
  assert episode_contract["available_cameras"] == ["head", "right_wrist"]
  assert episode_contract["command_streams"]["arm_joint_target"]["shape"] == [
    2, 7
  ]


def test_success_publishes_independent_manifest_with_backend_references(
  tmp_path, monkeypatch, capsys
):
  source = tmp_path / "raw"
  _raw_episode(source / "episode.h5")
  output = tmp_path / "output"
  runtime_python = tmp_path / "runtime-python"
  runtime_python.write_text(f"#!{sys.executable}\n", encoding="utf-8")
  runtime_python.chmod(0o755)
  runtime_config = tmp_path / "runtime.json"
  runtime_config.write_text(json.dumps({
    "runtimes": {
      "egosteer": {
        "python": str(runtime_python),
        "env": {
          "CHILD_FLAG": "enabled",
          "PYTHONPATH": "/profile/modules",
        },
      },
    },
  }), encoding="utf-8")
  module = run_path(str(SCRIPT))

  def fake_run(command, *, check, env):
    assert check is False
    assert command[0] == str(runtime_python.resolve())
    assert env["CHILD_FLAG"] == "enabled"
    assert env["PYTHONPATH"].split(":") == [
      str(Path(__file__).parents[1] / "src"),
      "/profile/modules",
    ]
    output.mkdir()
    (output / "source_snapshot_manifest.json").write_text(json.dumps({
      "schema": "backend_source_v1",
      "input_dir": str(source.resolve()),
    }), encoding="utf-8")
    (output / "dataset_manifest.json").write_text(json.dumps({
      "schema_version": "backend_dataset_v2",
      "action_alignment": "next recorded camera frame",
      "cameras": ["head", "right_wrist"],
      "source_manifest": {
        "path": "source_snapshot_manifest.json",
        "sha256": "backend-owned",
      },
    }), encoding="utf-8")
    return subprocess.CompletedProcess(command, 0)

  monkeypatch.setattr(module["subprocess"], "run", fake_run)
  assert module["main"]([
    "--input-dir", str(source),
    "--output-dir", str(output),
    "--format", "egosteer",
    "--task", "poker-draw",
    "--cameras", "head", "right_wrist",
    "--runtime-config", str(runtime_config),
    "--runtime", "egosteer",
  ]) == 0
  json.loads(capsys.readouterr().out)

  backend = json.loads((output / "dataset_manifest.json").read_text())
  assert backend["schema_version"] == "backend_dataset_v2"
  manifest = json.loads((output / "pipeline_conversion.json").read_text())
  assert manifest["schema"] == "kaihand_pipeline_conversion_v1"
  assert manifest["schema_version"] == 1
  assert manifest["status"] == "success"
  assert manifest["source_identity"]["episodes"][0]["capture_sha256"] == (
    "a" * 64
  )
  assert manifest["source_contract"]["episodes"][0][
    "declared_contracts"
  ]["control_contract"] == "recorded actuator targets"
  references = {row["path"]: row for row in manifest["backend_manifests"]}
  assert references["dataset_manifest.json"]["declared_contracts"] == {
    "action_alignment": "next recorded camera frame",
    "cameras": ["head", "right_wrist"],
  }
  assert manifest["backend_source_identity"] == [{
    "path": "source_snapshot_manifest.json",
    "sha256": references["source_snapshot_manifest.json"]["sha256"],
  }]


def test_failed_backend_does_not_publish_pipeline_manifest(
  tmp_path, monkeypatch, capsys
):
  source = tmp_path / "raw"
  _raw_episode(source / "episode.h5")
  output = tmp_path / "output"
  module = run_path(str(SCRIPT))

  def fake_run(command, *, check):
    assert check is False
    output.mkdir()
    (output / "dataset_manifest.json").write_text("{}", encoding="utf-8")
    return subprocess.CompletedProcess(command, 7)

  monkeypatch.setattr(module["subprocess"], "run", fake_run)
  assert module["main"]([
    "--input-dir", str(source),
    "--output-dir", str(output),
    "--format", "egosteer",
    "--task", "poker-draw",
    "--cameras", "head", "right_wrist",
  ]) == 7
  json.loads(capsys.readouterr().out)
  assert not (output / "pipeline_conversion.json").exists()


def test_egotouch_uses_the_same_dry_run_plan_schema(tmp_path, capsys):
  source = tmp_path / "raw"
  episode = source / "episode.h5"
  _raw_episode(episode)
  with h5py.File(episode, "r+") as file:
    del file.attrs["runtime_contract_json"]
  module = run_path(str(SCRIPT))
  assert module["main"]([
    "--input-dir", str(source),
    "--output-dir", str(tmp_path / "output"),
    "--format", "egotouch",
    "--task", "poker-draw",
    "--cameras", "head",
    "--dry-run",
  ]) == 0
  plan = json.loads(capsys.readouterr().out)
  assert plan["schema"] == "kaihand_conversion_plan_v1"
  assert plan["schema_version"] == 1
  assert plan["stage"] == "convert"
  assert plan["mode"] == "dry-run"
  assert plan["command"] is None
  assert plan["source_contract"]["episodes"][0]["runtime_contract"] == {
    "status": "unknown_not_declared",
    "source_attribute": "runtime_contract_json",
    "value": None,
  }
