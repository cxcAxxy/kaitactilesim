import hashlib
import json
import subprocess
import sys
import venv
from pathlib import Path

import pytest
from kaihand_tactile_env.pipeline.runtime import load_runtime_profile


def test_runtime_paths_are_relative_to_config_and_environment_is_isolated(tmp_path):
  config = tmp_path / "runtimes.toml"
  config.write_text(
    'default_runtime = "pi05"\n[runtimes.pi05]\n'
    'python = "env/bin/python"\nproject_root = "openpi"\n'
    '[runtimes.pi05.env]\nCUDA_VISIBLE_DEVICES = "2"\n',
    encoding="utf-8",
  )
  profile = load_runtime_profile(config, None)
  assert profile.python == tmp_path / "env/bin/python"
  assert profile.project_root == tmp_path / "openpi"
  original = {"CUDA_VISIBLE_DEVICES": "0", "PATH": "/bin"}
  assert profile.environment(original) == {"CUDA_VISIBLE_DEVICES": "2", "PATH": "/bin"}
  assert original["CUDA_VISIBLE_DEVICES"] == "0"
  assert profile.as_dict()["config_sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()


def test_runtime_selection_preserves_legacy_defaults_and_rejects_missing_config():
  assert load_runtime_profile(None, None) is None
  with pytest.raises(ValueError, match="requires --runtime-config"):
    load_runtime_profile(None, "pi05")


@pytest.mark.parametrize("relative", [False, True])
def test_runtime_preserves_virtualenv_python_symlink(tmp_path, relative):
  python = tmp_path / "model-env/bin/python"
  python.parent.mkdir(parents=True)
  python.symlink_to(sys.executable)
  project = tmp_path / "model-project"
  project.mkdir()
  project_link = tmp_path / "project-link"
  project_link.symlink_to(project, target_is_directory=True)
  config = tmp_path / "runtime.json"
  config.write_text(json.dumps({"runtimes": {"pi05": {
    "python": str(python.relative_to(tmp_path) if relative else python),
    "project_root": "project-link",
  }}}))

  profile = load_runtime_profile(config, "pi05")

  assert profile.python == python
  assert profile.python != python.resolve()
  assert profile.as_dict()["python"] == str(python)
  assert profile.project_root == project.resolve()
  profile.validate(require_python=True, require_project=True)


def test_runtime_child_keeps_virtualenv_prefix(tmp_path):
  environment = tmp_path / "model-env"
  venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
  python = environment / "bin/python"
  assert python.is_symlink()
  config = tmp_path / "runtime.json"
  config.write_text(json.dumps({"runtimes": {"pi05": {"python": str(python)}}}))
  profile = load_runtime_profile(config, "pi05")
  child_env = profile.environment()
  for key in ("PYTHONHOME", "PYTHONPATH"):
    child_env.pop(key, None)
  result = subprocess.run(
    [str(profile.python), "-B", "-c",
     "import json, sys; print(json.dumps({'prefix': sys.prefix, 'base': sys.base_prefix}))"],
    env=child_env, capture_output=True, text=True, check=True, timeout=15,
  )
  identity = json.loads(result.stdout)
  assert Path(identity["prefix"]) == environment
  assert identity["prefix"] != identity["base"]


@pytest.mark.parametrize(
  ("payload", "name", "message"),
  [
    ({"runtimes": {"pi05": {}}}, None, "select --runtime"),
    ({"runtimes": {"pi05": {}}}, "typo", "unknown runtime"),
    ({"runtimes": {"pi05": {"pyhton": "python"}}}, "pi05", "unknown runtime fields"),
    ({"runtimes": {"pi05": {"python": 2}}}, "pi05", "path string"),
    ({"runtimes": {"pi05": {"env": {"CUDA_VISIBLE_DEVICES": 0}}}}, "pi05", "must be a string"),
    ({"runtimes": {"pi05": {"env": {"A=B": "x"}}}}, "pi05", "invalid environment"),
    ({"runtimes": {"pi05": {"env": []}}}, "pi05", "string overrides"),
  ],
)
def test_invalid_runtime_config_fails_before_launch(tmp_path, payload, name, message):
  config = tmp_path / "runtime.json"
  config.write_text(json.dumps(payload), encoding="utf-8")
  with pytest.raises(ValueError, match=message):
    load_runtime_profile(config, name)


def test_runtime_plan_redacts_credentials_but_child_receives_override(tmp_path):
  config = tmp_path / "runtime.json"
  config.write_text(json.dumps({"runtimes": {"model": {"env": {"HF_TOKEN": "secret"}}}}))
  profile = load_runtime_profile(config, "model")
  assert profile.as_dict()["env"]["HF_TOKEN"] == "<redacted>"
  assert profile.environment({})["HF_TOKEN"] == "secret"


def test_execution_preflight_checks_only_declared_runtime_paths(tmp_path):
  config = tmp_path / "runtime.json"
  config.write_text(json.dumps({"runtimes": {"model": {"python": sys.executable}}}))
  profile = load_runtime_profile(config, "model")
  profile.validate(require_python=True)
  with pytest.raises(ValueError, match="declare project_root"):
    profile.validate(require_project=True)


def test_bad_toml_has_actionable_error(tmp_path):
  config = tmp_path / "runtime.toml"
  config.write_text("[broken")
  with pytest.raises(ValueError, match="invalid runtime configuration"):
    load_runtime_profile(config, "model")
