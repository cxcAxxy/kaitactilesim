from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
  workcell = ROOT / "scripts/workcell"
  sys.path.insert(0, str(workcell))
  try:
    path = workcell / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
  finally:
    sys.path.remove(str(workcell))


@dataclasses.dataclass(frozen=True)
class Assets:
  asset_id: str


@dataclasses.dataclass(frozen=True)
class Data:
  assets: Assets


@dataclasses.dataclass(frozen=True)
class Config:
  data: Data


def test_prepare_uses_shared_kaihand_config_and_checkpoint_normalizer(tmp_path):
  prepare = load_script("prepare_poker_pi05_deployment.py")
  checkpoint = tmp_path / "20000"
  normalizer = checkpoint / "assets/normalizer/norm_stats.json"
  normalizer.parent.mkdir(parents=True)
  normalizer.write_text("{}", encoding="utf-8")

  assert prepare.CONFIG_NAME == "pi05_kaihand"
  assert prepare.normalizer_asset_id(checkpoint) == "normalizer"


def test_server_binds_manifest_normalizer_to_runtime_config():
  server = load_script("serve_poker_pi05_policy.py")
  config = Config(data=Data(assets=Assets(asset_id="kaihand")))

  bound = server.bind_checkpoint_assets(config, {"normalizer_asset_id": "normalizer"})

  assert bound.data.assets.asset_id == "normalizer"
  assert config.data.assets.asset_id == "kaihand"


@pytest.mark.parametrize("asset_id", [None, "", "../normalizer", "a/b", ".", ".."])
def test_server_rejects_invalid_normalizer_asset_id(asset_id):
  server = load_script("serve_poker_pi05_policy.py")
  config = Config(data=Data(assets=Assets(asset_id="kaihand")))
  with pytest.raises(RuntimeError, match="one directory name"):
    server.bind_checkpoint_assets(config, {"normalizer_asset_id": asset_id})


def test_batch_uses_tracked_egl_vendor_configuration():
  batch = load_script("evaluate_poker_pi05_policy_batch.py")
  assert batch.EGL_VENDOR == ROOT / "scripts/collect/nvidia_egl_vendor.json"
  assert batch.EGL_VENDOR.is_file()
