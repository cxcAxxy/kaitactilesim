"""Temporary/mocked fixtures only: never convert source datasets or compute norms."""

import copy
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SCRIPT = (
  Path(__file__).resolve().parents[1] / "scripts/convert/convert_kaihand_ftp1_batch.py"
)
SPEC = importlib.util.spec_from_file_location("ftp1_batch_conversion_test", SCRIPT)
batch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(batch)


class FTP1BatchTests(unittest.TestCase):
  def setUp(self):
    self.temporary = tempfile.TemporaryDirectory()
    self.root = Path(self.temporary.name)
    self.addCleanup(self.temporary.cleanup)
    self.args = batch.parse_args(
      ["--dataset-root", str(self.root / "sim"), "--python-bin", "/usr/bin/python3"]
    )
    self.plan = batch.build_plan(self.args)

  def write(self, path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")

  def dataset(self, group):
    output = Path(group["output_dir"])
    shard = output / "fixture.zarr"
    shard.mkdir(parents=True, exist_ok=True)
    self.write(
      output / "conversion_manifest.json",
      dict(
        task=group["task"],
        episodes=200,
        schema="kaihand_raw_to_ftp1_zarr_v2",
        fps=30,
        ftp1_action_horizon=32,
        ftp1_action_down_sample_steps=1,
        output_root=str(output),
        frames=6400,
        source_episodes=[dict(episode_index=index) for index in range(200)],
        shards=[dict(path=str(shard), episodes=200, frames=6400)],
      ),
    )
    self.write(
      output / "dataset_config.json",
      dict(datasets=[dict(name="KaiHandSimTest", path=str(output), enabled=True)]),
    )

  def assets(self, path, group):
    self.write(path / "norm_params_snapshot.json", batch.snapshot_for(group))
    tactile = dict(
      shape=[1, 5, 7, 5, 3],
      type="matrix",
      function_areas=list(range(5)),
      sensor="KaiHandSimTaxelV1",
    )
    self.write(
      path / "tactile_input_config.json",
      dict(KaiHandSimTest=dict(right_tactile_fingers=tactile)),
    )
    self.write(path / "share_norm_stats_all_t0_zscore.json", dict(params={}))
    stats = dict(mean=[0.0] * 120, std=[1.0] * 120)
    taxel_mean = [[[0.0] * 3 for _ in range(5)] for _ in range(7)]
    taxel_std = [[[1.0] * 3 for _ in range(5)] for _ in range(7)]
    self.write(
      path / "KaiHandSimTest/independent_norm_stats_all_t0_zscore.json",
      dict(
        params=dict(
          actions=stats,
          state=stats,
          tactile={
            "KaiHandSimTaxelV1_matrix_7_5_3": dict(mean=taxel_mean, std=taxel_std)
          },
        )
      ),
    )
    self.write(
      path / "KaiHandSimTest/train_val_split.json",
      dict(train_episode_idx=list(range(198)), val_episode_idx=[198, 199]),
    )
    self.write(path / "dataset_stats.json", dict(episodes=200))

  def test_exact_nine_task_scope_and_usb_native_raw_identity(self):
    self.assertEqual(len(self.plan["groups"]), 9)
    tasks = {row["task"] for row in self.plan["groups"]}
    self.assertFalse(
      tasks.intersection({"poker-draw", "sponge-grasp", "pick-place", "vase-wipe"})
    )
    usb = next(row for row in self.plan["groups"] if row["task"] == "usb-insert")
    self.assertTrue(usb["input_dir"].endswith("usb_insert/raw/0920_200"))
    self.assertTrue(usb["output_dir"].endswith("usb_insert/ftp1/0920_200"))
    self.assertEqual(usb["repo_id"], "usb_insert_0920_200")
    inplace = next(
      row for row in self.plan["groups"] if row["task"] == "bulb-screw-inplace"
    )
    self.assertTrue(inplace["input_dir"].endswith("shared_raw/1018_200"))
    screw = self.plan["groups"][0]
    self.assertTrue(screw["input_dir"].endswith("raw/1012_v15_200_r2"))
    self.assertTrue(screw["output_dir"].endswith("ftp1/1012_200"))
    self.assertFalse((self.root / "sim").exists())

  def test_cli_modes_exclusive_and_unknown_tasks_rejected(self):
    with self.assertRaises(SystemExit):
      batch.parse_args(["--run", "--prepare-training-assets"])
    with self.assertRaises(ValueError):
      batch.build_plan(SimpleNamespace(**dict(vars(self.args), tasks=["poker-draw"])))
    for name in ("workers", "episodes_per_shard"):
      with self.subTest(name=name), self.assertRaises(ValueError):
        batch.build_plan(SimpleNamespace(**dict(vars(self.args), **{name: 0})))

  def test_python_venv_symlink_is_not_resolved(self):
    python = self.root / "venv/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to("/usr/bin/python3")
    self.args.python_bin = python
    self.assertEqual(batch.build_plan(self.args)["python_bin"], str(python))

  def test_plan_all_preflights_cpu_and_creates_no_output(self):
    before = list(self.root.rglob("*"))
    with patch.object(
      batch.subprocess,
      "run",
      return_value=SimpleNamespace(returncode=0, stdout="{}", stderr=""),
    ) as run:
      reports = batch.preflight(self.plan)
    self.assertEqual(run.call_count, 9)
    self.assertTrue(all(row["valid"] for row in reports))
    for args, kwargs in run.call_args_list:
      self.assertIn("--dry-run", args[0])
      self.assertIn("--plan-json", args[0])
      self.assertEqual(args[0][args[0].index("--expected-episodes") + 1], "200")
      self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")
      self.assertEqual(kwargs["env"]["JAX_PLATFORMS"], "cpu")
    self.assertEqual(before, list(self.root.rglob("*")))

  def test_all_failed_preflights_reported_and_none_can_convert(self):
    with patch.object(
      batch.subprocess,
      "run",
      return_value=SimpleNamespace(returncode=2, stdout="", stderr="bad metadata"),
    ) as run:
      reports = batch.preflight(self.plan)
      with self.assertRaisesRegex(ValueError, "nothing was converted"):
        batch.execute_conversion(self.plan, reports)
    self.assertEqual(run.call_count, 9)
    self.assertFalse(any(row["valid"] for row in reports))

  def test_partial_preflight_list_cannot_convert(self):
    with patch.object(batch.subprocess, "run") as run:
      with self.assertRaises(ValueError):
        batch.execute_conversion(self.plan, [dict(valid=True)])
    run.assert_not_called()

  def test_resume_and_source_hash_flags_propagate_in_both_phases(self):
    self.plan.update(resume=True, verify_source_hash=True)
    for dry_run in (False, True):
      command = batch.conversion_command(
        self.plan, self.plan["groups"][0], dry_run=dry_run
      )
      self.assertIn("--resume", command)
      self.assertIn("--verify-source-hash", command)
      self.assertEqual("--dry-run" in command, dry_run)
      self.assertNotIn("--overwrite", command)

  def test_run_converts_only_after_complete_preflight_and_stops_on_error(self):
    reports = [dict(valid=True)] * 9
    with patch.object(batch.subprocess, "run") as run:
      batch.execute_conversion(self.plan, reports)
    self.assertEqual(run.call_count, 9)
    self.assertTrue(all("--dry-run" not in call.args[0] for call in run.call_args_list))

  def test_norm_cli_matches_training_snapshot_and_isolated_assets_base(self):
    group = self.plan["groups"][0]
    command = batch.norm_command(self.plan, group, self.root / "new-staging")
    for option, value in (
      ("--action-joint-rep", "mix"),
      ("--model.action-horizon", "32"),
      ("--norm-sample-ratio", "1.0"),
      ("--norm-type", "zscore"),
      ("--assets-base-dir", str(self.root / "new-staging")),
    ):
      self.assertEqual(command[command.index(option) + 1], value)
    self.assertIn("--disable-history", command)
    self.assertIn("--no-wandb-enabled", command)
    self.assertNotIn("--pytorch-weight-path", command)

  def test_existing_unbound_assets_prevent_all_norm_calls(self):
    for group in self.plan["groups"]:
      self.dataset(group)
    last = self.plan["groups"][-1]
    destination = Path(last["assets_base_dir"]) / "ftp1" / last["repo_id"]
    destination.mkdir(parents=True)
    self.write(destination / "user.json", dict(preserve=True))
    with patch.object(batch.subprocess, "run") as run:
      with self.assertRaisesRegex(ValueError, "not be overwritten"):
        batch.prepare_assets(self.plan)
    run.assert_not_called()
    self.assertEqual(batch.read_json(destination / "user.json"), dict(preserve=True))

  def test_stage_validate_publish_then_reuse_without_recomputing(self):
    self.plan["groups"] = self.plan["groups"][:1]
    group = self.plan["groups"][0]
    self.dataset(group)

    def norm_fixture(command, **kwargs):
      if "--dry-run" in command:
        return SimpleNamespace(
          returncode=0, stdout=json.dumps(dict(published_complete=True)), stderr=""
        )
      staging = Path(command[command.index("--assets-base-dir") + 1])
      self.assertNotEqual(staging, Path(group["assets_base_dir"]))
      self.assets(staging / "ftp1" / group["repo_id"], group)

    with patch.object(batch.subprocess, "run", side_effect=norm_fixture) as run:
      batch.prepare_assets(self.plan)
      batch.prepare_assets(self.plan)
    self.assertEqual(run.call_count, 3)
    self.assertEqual(
      sum("--assets-base-dir" in row.args[0] for row in run.call_args_list), 1
    )
    binding, destination, exists = batch.assets_status(group)
    self.assertTrue(exists)
    self.assertTrue((destination / batch.MARKER).is_file())
    self.assertEqual(binding, batch.dataset_binding(group))
    self.assertEqual(
      list(Path(group["assets_base_dir"]).glob(".kaihand-ftp1-norm-*")), []
    )

  def test_asset_tampering_dataset_change_and_invalid_norms_rejected(self):
    group = self.plan["groups"][0]
    self.dataset(group)
    path = self.root / "assets"
    self.assets(path, group)
    batch.validate_assets(path, group)
    stats_path = path / "KaiHandSimTest/independent_norm_stats_all_t0_zscore.json"
    stats = batch.read_json(stats_path)
    for bad in (0.0, -1.0, math.nan, math.inf):
      changed = copy.deepcopy(stats)
      changed["params"]["actions"]["std"][0] = bad
      self.write(stats_path, changed)
      with self.subTest(bad=bad), self.assertRaises(ValueError):
        batch.validate_assets(path, group)
    self.write(stats_path, stats)
    before = batch.dataset_binding(group)
    changed = batch.read_json(Path(group["output_dir"]) / "conversion_manifest.json")
    changed["changed"] = True
    self.write(Path(group["output_dir"]) / "conversion_manifest.json", changed)
    self.assertNotEqual(before, batch.dataset_binding(group))

  def test_norm_failure_publishes_nothing(self):
    self.plan["groups"] = self.plan["groups"][:1]
    group = self.plan["groups"][0]
    self.dataset(group)

    def fail_norm(command, **kwargs):
      if "--dry-run" in command:
        return SimpleNamespace(
          returncode=0, stdout=json.dumps(dict(published_complete=True)), stderr=""
        )
      raise ValueError("fixture normalization failed")

    with patch.object(batch.subprocess, "run", side_effect=fail_norm):
      with self.assertRaises(ValueError):
        batch.prepare_assets(self.plan)
    destination = Path(group["assets_base_dir"]) / "ftp1" / group["repo_id"]
    self.assertFalse(destination.exists())
    self.assertEqual(
      list(Path(group["assets_base_dir"]).glob(".kaihand-ftp1-norm-*")), []
    )

  def test_incomplete_published_output_blocks_all_norm_writes(self):
    for group in self.plan["groups"]:
      self.dataset(group)
    results = [
      SimpleNamespace(
        returncode=0, stdout=json.dumps(dict(published_complete=index != 8)), stderr=""
      )
      for index in range(9)
    ]
    with patch.object(batch.subprocess, "run", side_effect=results) as run:
      with self.assertRaisesRegex(ValueError, "completely published"):
        batch.prepare_assets(self.plan)
    self.assertEqual(run.call_count, 9)
    self.assertTrue(all("--resume" in row.args[0] for row in run.call_args_list))
    self.assertTrue(all("--dry-run" in row.args[0] for row in run.call_args_list))
    self.assertFalse(
      any(Path(group["assets_base_dir"]).exists() for group in self.plan["groups"])
    )

  def test_asset_symlink_ancestor_refused_without_subprocess(self):
    group = self.plan["groups"][0]
    self.dataset(group)
    target = self.root / "preserve"
    target.mkdir()
    base = Path(group["assets_base_dir"])
    base.symlink_to(target, target_is_directory=True)
    with patch.object(batch.subprocess, "run") as run:
      with self.assertRaisesRegex(ValueError, "Symlinked"):
        batch.prepare_assets(self.plan)
    run.assert_not_called()
    self.assertEqual(list(target.iterdir()), [])

  def test_bad_domain_config_rejected_before_normalization(self):
    group = self.plan["groups"][0]
    self.dataset(group)
    self.write(
      Path(group["output_dir"]) / "dataset_config.json",
      dict(datasets=[dict(name="../../raw", path=group["output_dir"])]),
    )
    with patch.object(batch.subprocess, "run") as run:
      with self.assertRaisesRegex(ValueError, "domain name"):
        batch.prepare_assets(self.plan)
    run.assert_not_called()

  def test_missing_published_shard_blocks_assets(self):
    group = self.plan["groups"][0]
    self.dataset(group)
    (Path(group["output_dir"]) / "fixture.zarr").rmdir()
    with self.assertRaisesRegex(ValueError, "Missing"):
      batch.assets_status(group)

  def test_matrix_shape_sensor_and_train_split_checked(self):
    group = self.plan["groups"][0]
    self.dataset(group)
    path = self.root / "assets"
    self.assets(path, group)
    stats_path = path / "KaiHandSimTest/independent_norm_stats_all_t0_zscore.json"
    original = batch.read_json(stats_path)
    wrong = copy.deepcopy(original)
    wrong["params"]["tactile"]["KaiHandSimTaxelV1_matrix_7_5_3"]["mean"] = [0.0]
    self.write(stats_path, wrong)
    with self.assertRaisesRegex(ValueError, "shared-area"):
      batch.validate_assets(path, group)
    self.write(stats_path, original)
    self.write(
      path / "KaiHandSimTest/train_val_split.json",
      dict(train_episode_idx=list(range(198)), val_episode_idx=[0, 199]),
    )
    with self.assertRaisesRegex(ValueError, "198/2"):
      batch.validate_assets(path, group)

  def test_asset_marker_tampering_and_stale_dataset_preserve_original_assets(self):
    group = self.plan["groups"][0]
    self.dataset(group)
    destination = Path(group["assets_base_dir"]) / "ftp1" / group["repo_id"]
    self.assets(destination, group)
    binding = batch.dataset_binding(group)
    self.write(
      destination / batch.MARKER,
      dict(binding=binding, artifacts=batch.artifact_hashes(destination)),
    )
    self.assertTrue(batch.assets_status(group)[2])
    self.write(destination / "dataset_stats.json", dict(tampered=True))
    with self.assertRaisesRegex(ValueError, "identity/hash"):
      batch.assets_status(group)
    self.assertEqual(
      batch.read_json(destination / "dataset_stats.json"), dict(tampered=True)
    )
    self.assets(destination, group)
    changed = batch.read_json(Path(group["output_dir"]) / "conversion_manifest.json")
    changed["changed"] = True
    self.write(Path(group["output_dir"]) / "conversion_manifest.json", changed)
    with self.assertRaisesRegex(ValueError, "identity/hash"):
      batch.assets_status(group)

  def test_file_ancestor_blocks_assets_before_any_norm_call(self):
    group = self.plan["groups"][0]
    self.dataset(group)
    self.write(Path(group["assets_base_dir"]), dict(user_file=True))
    with patch.object(batch.subprocess, "run") as run:
      with self.assertRaisesRegex(ValueError, "not a directory"):
        batch.prepare_assets(self.plan)
    run.assert_not_called()


if __name__ == "__main__":
  unittest.main()
