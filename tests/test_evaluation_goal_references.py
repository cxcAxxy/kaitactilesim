"""Mechanical chart annotations require the trial's frozen source identity."""
import ast
import copy
import hashlib
import json
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "analysis"))
from evaluation_workbench import goal_references as goals

PROJECT = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def row_for(path, role):
    raw = path.read_bytes()
    return {"path": str(path), "size": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "role": role}


class GoalReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.archive = self.root / "frozen-source"
        self.inventory = []
        for name in goals._REVIEWED:
            relative = goals._source_relative(name)
            destination = self.archive / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PROJECT / relative, destination)
            row = row_for(destination, "unused")
            self.inventory.append({"path": relative, "size": row["size"], "sha256": row["sha256"]})
        self.manifest_path = self.root / "deployment_manifest.json"
        self.summary_path = self.root / "summary.json"
        self.manifest = {"task": "bulb-screw", "checkpoint_sha256": "a" * 64,
                         "deployment_id": "b" * 64, "prediction_horizon": 50,
                         "evaluation_source_files": self.inventory,
                         "evaluation_source_sha256": digest(self.inventory)}
        self.summary = {"task": "bulb-screw", "seed": 0, "prediction_horizon": 50,
                        "checkpoint_sha256": "a" * 64, "deployment_id": "b" * 64,
                        "task_success": False, "deployment_integrity": {
                            "status": "verified", "errors": [], "evaluation_source": {
                                "status": "verified", "root": str(self.archive),
                                "expected_sha256": digest(self.inventory),
                                "actual_sha256": digest(self.inventory)}}}
        self.report = {"trial": {"task": "bulb-screw", "seed": 0, "prediction_horizon": 50,
                                 "cohort": {"checkpoint": "a" * 64}},
                       "recorded_staged_evaluation": {"rule_identity": {
                           "evaluation_source_sha256": digest(self.inventory)}},
                       "outcome": {"task_success": False}}
        self.save()

    def save(self):
        write_json(self.manifest_path, self.manifest)
        manifest_row = row_for(self.manifest_path, "staged_deployment_manifest")
        self.summary["deployment_integrity"]["manifest_sha256"] = manifest_row["sha256"]
        write_json(self.summary_path, self.summary)
        self.report["provenance"] = [manifest_row, row_for(self.summary_path, "prediction_horizon_summary")]

    def rebind_inventory(self):
        bound = digest(self.inventory)
        self.manifest["evaluation_source_sha256"] = bound
        self.summary["deployment_integrity"]["evaluation_source"].update(
            expected_sha256=bound, actual_sha256=bound)
        self.report["recorded_staged_evaluation"]["rule_identity"]["evaluation_source_sha256"] = bound
        self.save()

    def select(self, task):
        self.report["trial"]["task"] = task
        self.summary["task"] = task
        self.manifest["task"] = task
        self.save()

    def build(self, **kwargs):
        return goals.build_goal_references(self.report["trial"]["task"], self.report, **kwargs)

    def unavailable(self, result):
        self.assertEqual(result["status"], "unavailable", result)
        self.assertEqual(result["references"], [])
        self.assertTrue(result["warnings"])

    def test_screw_has_target_lines_and_native_geometric_tolerance_bands(self):
        result = self.build()
        self.assertEqual(result["status"], "verified", result)
        angle, depth = result["references"]
        self.assertEqual(angle["key"], "clockwise_angle_deg")
        self.assertEqual(angle["unit"], "deg")
        self.assertAlmostEqual(angle["target"], 100)
        self.assertAlmostEqual(angle["lower"], 100 - 0.08 * 360)
        self.assertAlmostEqual(angle["upper"], 100 + 0.08 * 360)
        self.assertIn("100°", angle["label"])
        self.assertEqual(depth["key"], "insertion_depth_m")
        self.assertEqual(depth["unit"], "m")
        self.assertAlmostEqual(depth["target"], 0.0071111111111111115)
        self.assertAlmostEqual(depth["upper"] - depth["target"], 0.0003)
        self.assertIn("7.111 mm", depth["label"])
        self.assertTrue(all("不等于" in ref["note"] for ref in result["references"]))
        self.assertFalse(result["binding"]["source_code_executed"])
        self.assertFalse(result["binding"]["rules_reexecuted"])

    def test_unscrew_uses_nominal_131_5_degree_release_not_121_5_minimum(self):
        self.select("bulb-unscrew")
        result = self.build()
        self.assertEqual(result["status"], "verified", result)
        angle, depth = result["references"]
        self.assertEqual((angle["key"], angle["unit"]), ("angle_removed_deg", "deg"))
        self.assertAlmostEqual(angle["target"], 131.5)
        self.assertNotIn("lower", angle)
        self.assertNotIn("upper", angle)
        self.assertIn("名义", angle["label"])
        self.assertIn("不是拧动次数或新的物理判据", angle["note"])
        self.assertEqual(depth["relation"], "at_most")
        self.assertEqual(depth["target"], -0.003)
        self.assertIn("不是放回桌面", depth["note"])

    def test_inplace_absolute_position_not_relative_rotation_gets_reference(self):
        self.select("bulb-screw-inplace")
        result = self.build()
        self.assertEqual(result["status"], "verified", result)
        self.assertEqual({r["key"] for r in result["references"]},
                         {"clockwise_angle_deg", "insertion_depth_m"})
        self.assertTrue(any("实际初始角度" in warning for warning in result["warnings"]))

    def test_ram_depth_has_target_and_native_tolerance_not_minimum_depth(self):
        self.select("install-ram")
        result = self.build()
        self.assertEqual(result["status"], "verified", result)
        depth, orientation = result["references"]
        self.assertEqual((depth["key"], depth["unit"], depth["relation"]),
                         ("insertion_depth_m", "m", "target"))
        self.assertAlmostEqual(depth["target"], 0.006)
        self.assertAlmostEqual(depth["lower"], 0.0058)
        self.assertAlmostEqual(depth["upper"], 0.0062)
        self.assertIn("不是越深越好", depth["note"])
        self.assertEqual((orientation["key"], orientation["unit"], orientation["relation"]),
                         ("orientation_error_rad", "rad", "at_most"))
        self.assertAlmostEqual(orientation["target"], math.pi / 180)
        self.assertEqual(orientation["lower"], 0)
        self.assertEqual(orientation["upper"], orientation["target"])
        self.assertIn("理想 0°", orientation["label"])
        self.assertIn("双侧容差", orientation["note"])
        self.assertIn("完整三维", orientation["note"])
        self.assertFalse(result["binding"]["source_code_executed"])

    def test_poker_current_and_peak_reference_native_50mm_not_expert_80mm(self):
        self.select("poker-draw")
        result = self.build()
        self.assertEqual(result["status"], "verified", result)
        now, peak = result["references"]
        self.assertEqual([now["key"], peak["key"]],
                         ["card_displacement_now_mm", "card_displacement_peak_mm"])
        for ref in (now, peak):
            self.assertEqual(ref["unit"], "mm")
            self.assertEqual(ref["target"], 50)
            self.assertEqual(ref["relation"], "at_least")
            self.assertNotIn("lower", ref)
            self.assertNotIn("upper", ref)
            self.assertIn("40%", ref["note"])
            self.assertIn("不等于", ref["note"])
        self.assertIn("当前净位移", now["note"])
        self.assertIn("之后回退", peak["note"])
        self.assertIn("不能证明当前仍到位", peak["note"])
        self.assertEqual(len(result["provenance"]), 4)
        self.assertFalse(result["binding"]["rules_reexecuted"])

    def test_poker_crop_role_manifest_still_requires_summary_sha_binding(self):
        self.select("poker-draw")
        self.report["provenance"][0]["role"] = "model_crop_deployment_manifest"
        result = self.build()
        self.assertEqual(result["status"], "verified", result)
        self.summary["deployment_integrity"].pop("manifest_sha256")
        write_json(self.summary_path, self.summary)
        self.report["provenance"][1] = row_for(self.summary_path, "prediction_horizon_summary")
        self.unavailable(self.build())

    def test_crop_role_is_not_a_general_unbound_manifest_fallback(self):
        self.select("install-ram")
        self.report["provenance"][0]["role"] = "model_crop_deployment_manifest"
        self.unavailable(self.build())

    def test_new_tasks_do_not_require_unrelated_bulb_archive_files(self):
        for name in goals._REVIEWED:
            if name.startswith("bulb_"):
                (self.archive / (goals._BASE + name)).unlink()
        for task in ("install-ram", "poker-draw"):
            with self.subTest(task=task):
                self.select(task)
                result = self.build()
                self.assertEqual(result["status"], "verified", result)
                paths = [row.get("inventory_path", "") for row in result["provenance"]]
                self.assertFalse(any("bulb_" in path for path in paths))

    def test_new_task_measurement_revision_or_missing_inventory_fails_closed(self):
        for task, name in (("install-ram", "install_ram/task.py"),
                           ("poker-draw", "poker_draw/review_metrics.py")):
            with self.subTest(task=task):
                self.select(task)
                path = self.archive / (goals._BASE + name)
                raw = path.read_bytes()
                path.write_bytes(raw + b"\n# unreviewed measurement revision\n")
                row = next(row for row in self.inventory if row["path"] == goals._BASE + name)
                original_row = copy.deepcopy(row)
                row.update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), size=path.stat().st_size)
                self.rebind_inventory()
                self.unavailable(self.build())
                path.write_bytes(raw)
                row.update(original_row)
                self.inventory.remove(row)
                self.rebind_inventory()
                self.unavailable(self.build())
                self.inventory.append(row)
                self.rebind_inventory()

    def test_ram_static_orientation_reader_rejects_ambiguous_or_disagreeing_gates(self):
        good = ast.parse("angle <= np.deg2rad(1.0)\nstate.orientation_error_rad <= np.deg2rad(1.0)")
        self.assertAlmostEqual(goals._ram_orientation_tolerance(good), math.pi / 180)
        invalid = (
            "angle <= np.deg2rad(1.0)",
            "angle <= np.deg2rad(1.0)\nstate.orientation_error_rad <= np.deg2rad(2.0)",
            "angle <= np.deg2rad(1.0)\nangle <= np.deg2rad(1.0)\nstate.orientation_error_rad <= np.deg2rad(1.0)",
            "angle <= np.deg2rad(danger())\nstate.orientation_error_rad <= np.deg2rad(danger())",
        )
        for source in invalid:
            with self.subTest(source=source), self.assertRaises(ValueError):
                goals._ram_orientation_tolerance(ast.parse(source))

    def test_poker_static_reader_rejects_review_and_outcome_threshold_disagreement(self):
        paths = goals._TASK_FILES["poker-draw"]
        trees = {name: ast.parse((PROJECT / (goals._BASE + name)).read_text()) for name in paths}
        self.assertAlmostEqual(goals._poker_displacement_threshold(trees), 0.05)
        trees["poker_draw/review_metrics.py"] = ast.parse(
            "class PokerReviewMetrics:\n    EDGE_THRESHOLD_M = 0.08\n")
        with self.assertRaises(ValueError):
            goals._poker_displacement_threshold(trees)

    def test_usb_insertion_references_backstop_not_command_depth(self):
        self.select("usb-insert")
        result = self.build()
        self.assertEqual(result["status"], "verified", result)
        depth, orientation = result["references"]
        self.assertEqual((depth["key"], depth["unit"], depth["relation"]),
                         ("insertion_depth_mm", "mm", "target"))
        self.assertEqual(depth["target"], 12)
        self.assertAlmostEqual(depth["lower"], 11.9)
        self.assertAlmostEqual(depth["upper"], 12.1)
        self.assertIn("图中记 0", depth["note"])
        self.assertIn("≤5°", depth["note"])
        self.assertEqual((orientation["key"], orientation["unit"], orientation["relation"]),
                         ("orientation_error_deg", "deg", "at_most"))
        self.assertEqual(orientation["target"], 3)
        self.assertEqual(orientation["lower"], 0)
        self.assertEqual(orientation["upper"], 3)
        self.assertIn("不是图中深度记录", orientation["note"])
        self.assertFalse(result["binding"]["rules_reexecuted"])

    def test_usb_insert_crop_role_requires_binding_and_explicit_missing_root_copy(self):
        self.select("usb-insert")
        self.summary["deployment_integrity"]["evaluation_source"].pop("root")
        self.save()
        self.report["provenance"][0]["role"] = "model_crop_deployment_manifest"
        self.unavailable(self.build())
        result = self.build(source_root=self.archive)
        self.assertEqual(result["status"], "verified", result)
        self.assertEqual(result["binding"]["source_root_selection"], "explicit_inventory_verified_copy")
        self.summary["deployment_integrity"].pop("manifest_sha256")
        write_json(self.summary_path, self.summary)
        self.report["provenance"][1] = row_for(self.summary_path, "prediction_horizon_summary")
        self.unavailable(self.build(source_root=self.archive))

    def test_usb_unplug_clearance_is_negative_single_sided_not_return_target(self):
        self.select("usb-unplug")
        result = self.build()
        self.assertEqual(result["status"], "verified", result)
        self.assertEqual(len(result["references"]), 2)
        reference = result["references"][0]
        self.assertEqual((reference["key"], reference["unit"], reference["relation"]),
                         ("insertion_depth_m", "m", "at_most"))
        self.assertEqual(reference["target"], -0.002)
        self.assertNotIn("lower", reference)
        self.assertNotIn("upper", reference)
        self.assertIn("无插座接触", reference["note"])
        self.assertIn("连续至少 2 s", reference["note"])
        self.assertIn("不要求放回", reference["note"])
        hold = result["references"][1]
        self.assertEqual((hold["key"], hold["unit"], hold["relation"], hold["target"]),
                         ("extraction_hold_duration_s", "s", "at_least", 2.0))

    def test_usb_unplug_historical_config_retains_old_clearance_stage_note(self):
        trees = {"usb_unplug/config.py": ast.parse("FULL_CLEAR_DEPTH_M = -0.002\nFULL_CLEAR_DWELL_S = 0.10\n")}
        refs = goals._references("usb-unplug", trees)
        self.assertEqual(len(refs), 1)
        self.assertIn("0.1 s", refs[0]["note"])
        self.assertIn("不是放回桌面", refs[0]["note"])
        self.assertNotIn("2 s", refs[0]["note"])
        self.assertEqual(goals._REVIEWED["usb_unplug/config.py"],
                         "374ac5656a1c0980ecc2e793f82a09fd2dce2f3a61e214ce88c53bd56cfcfc69")
        self.assertEqual(goals._REVIEWED["usb_unplug/task.py"],
                         "a95d39ab0295187cd7ce19504138a99ba1cf633910ddd87c429f022de52b0f83")

    def test_usb_unplug_unknown_or_changed_completion_protocol_is_not_guessed(self):
        for protocol, hold in (("usb-unplug-unreviewed", 2),
                               ("usb-unplug-held-clear-2s-v2", 1),
                               (True, 2)):
            config = f"FULL_CLEAR_DEPTH_M = -0.002\nCOMPLETION_PROTOCOL = {protocol!r}\nEXTRACTION_HOLD_DWELL_S = {hold!r}\n"
            with self.subTest(protocol=protocol, hold=hold), self.assertRaises(goals._Unavailable):
                goals._references("usb-unplug", {"usb_unplug/config.py": ast.parse(config)})

    def test_usb_unplug_cannot_mix_new_config_with_old_monitor_source_identity(self):
        self.select("usb-unplug")
        relative = goals._source_relative("usb_unplug/task.py")
        row = next(row for row in self.inventory if row["path"] == relative)
        row["sha256"] = goals._REVIEWED["usb_unplug/task.py"]
        self.rebind_inventory()
        result = self.build()
        self.unavailable(result)
        self.assertIn("config/task source pair", result["warnings"][0])

    def test_whiteboard_average_is_ideal_not_sufficient_native_or_staged_success(self):
        for task in ("whiteboard-wipe", "whiteboard-wipe-horizontal", "whiteboard-wipe-horizontal-line"):
            with self.subTest(task=task):
                self.select(task)
                result = self.build()
                self.assertEqual(result["status"], "verified", result)
                average, worst = result["references"]
                self.assertEqual((average["key"], average["target"], average["relation"]),
                                 ("cleaned_fraction", 1, "target"))
                self.assertEqual(average["reference_kind"], "ideal")
                self.assertNotIn("lower", average)
                self.assertNotIn("upper", average)
                self.assertIn("不是实际覆盖面积", average["note"])
                self.assertIn("不能替代", average["note"])
                self.assertEqual((worst["key"], worst["target"], worst["relation"]),
                                 ("maximum_ink_remaining", 1e-9, "at_most"))
                self.assertEqual(worst["lower"], 0)
                self.assertEqual(worst["upper"], 1e-9)
                self.assertIn("只定义拾取阶段", worst["note"])
                self.assertFalse(result["binding"]["rules_reexecuted"])

    def test_whiteboard_missing_or_unreviewed_shared_measurement_fails_closed(self):
        self.select("whiteboard-wipe")
        relative = goals._source_relative("shared/policy_tasks.py")
        path = self.archive / relative
        path.write_bytes(path.read_bytes() + b"\n# unknown shared monitor revision\n")
        row = next(row for row in self.inventory if row["path"] == relative)
        row.update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), size=path.stat().st_size)
        self.rebind_inventory()
        self.unavailable(self.build())
        self.inventory.remove(row)
        self.rebind_inventory()
        self.unavailable(self.build())

    def test_whiteboard_line_crop_audit_manifest_role_still_requires_summary_binding(self):
        self.select("whiteboard-wipe-horizontal-line")
        self.report["provenance"][0]["role"] = "model_crop_deployment_manifest"
        self.assertEqual(self.build()["status"], "verified")
        self.summary["deployment_integrity"].pop("manifest_sha256")
        write_json(self.summary_path, self.summary)
        self.report["provenance"][1] = row_for(self.summary_path, "prediction_horizon_summary")
        self.unavailable(self.build())

    def test_reviewed_usb_adapter_revision_preserves_whiteboard_reference_semantics(self):
        source = PROJECT / "src/kaihand_tactile_env/shared/policy_tasks.py"
        raw = source.read_bytes()
        self.assertIn(hashlib.sha256(raw).hexdigest(), goals._REVIEWED_ALTERNATES["shared/policy_tasks.py"])
        tree = ast.parse(raw)
        # These complete method AST identities match the previously reviewed
        # d0e340... source, not just its threshold literals. No source executes.
        expected = {
            "_whiteboard_state": "c825397e978fdcfbc292209639264dc4d32fa96a4c32283eb7eae5af3883b788",
            "_update_horizontal_pickup": "8ee52a6f2a34450ce4dbfbf4f0dc832624148d78c50ded8c4178c01af69a9369",
            "_whiteboard_release_condition": "660ba0756425044014eb840c8d697da055c5ea8b9e10711f3974a6c5665bd477",
        }
        for name, digest_value in expected.items():
            method = goals._function(tree, name)
            self.assertEqual(hashlib.sha256(ast.dump(method, include_attributes=False).encode()).hexdigest(), digest_value)

    def test_usb_static_reader_rejects_unsupported_or_ambiguous_orientation_gate(self):
        source = "def _near_bottom(state):\n    return state.orientation_error_rad <= np.deg2rad(3.0)\n"
        self.assertEqual(goals._usb_orientation_tolerance(ast.parse(source)), 3)
        invalid = (
            "def _near_bottom(state):\n    return state.orientation_error_rad <= danger()\n",
            "def _near_bottom(state):\n    return state.orientation_error_rad <= np.deg2rad(0.0)\n",
            source + source,
            "def _near_bottom(state):\n    return state.orientation_error_rad <= np.deg2rad(3.0) and state.orientation_error_rad <= np.deg2rad(5.0)\n",
        )
        for value in invalid:
            with self.subTest(source=value), self.assertRaises(ValueError):
                goals._usb_orientation_tolerance(ast.parse(value))

    def test_whiteboard_static_reader_rejects_missing_mean_or_worst_gate(self):
        source = ("def _whiteboard_state(self):\n"
                  "    return State(cleaned_fraction=float(np.mean(1.0 - remaining)), "
                  "success=maximum_remaining <= 1e-9)\n")
        self.assertEqual(goals._whiteboard_ink_tolerance(ast.parse(source)), 1e-9)
        invalid = (source.replace("np.mean", "np.max"),
                   source.replace("1e-9", "danger()"),
                   source.replace("1e-9", "-1e-9"), source + source)
        for value in invalid:
            with self.subTest(source=value), self.assertRaises(ValueError):
                goals._whiteboard_ink_tolerance(ast.parse(value))

    def test_torque_and_contact_load_have_no_invented_terminal_values(self):
        for task in goals._TASKS:
            self.select(task)
            result = self.build()
            self.assertEqual(result["status"], "verified", result)
            self.assertFalse(any("torque" in r["key"] or "load" in r["key"]
                                 for r in result["references"]))

    def test_unsupported_task_does_no_file_io(self):
        with patch.object(goals, "_read_bytes", side_effect=AssertionError("must not read")):
            result = goals.build_goal_references("vase-wipe", {})
        self.assertEqual(result["status"], "not_applicable")
        self.assertEqual(result["references"], [])

    def test_missing_evidence_does_not_fall_back_to_today_defaults(self):
        self.unavailable(goals.build_goal_references("bulb-screw", {"trial": {"task": "bulb-screw"}}))

    def test_no_source_root_is_unavailable_even_if_current_checkout_matches(self):
        self.summary["deployment_integrity"]["evaluation_source"].pop("root")
        self.save()
        self.unavailable(self.build())

    def test_explicit_byte_identical_source_copy_is_allowed_and_declared(self):
        self.summary["deployment_integrity"]["evaluation_source"]["root"] = "/retired/source"
        self.save()
        result = self.build(source_root=self.archive)
        self.assertEqual(result["status"], "verified", result)
        self.assertEqual(result["binding"]["source_root_selection"], "explicit_inventory_verified_copy")

    def test_changed_archive_bytes_are_unavailable(self):
        path = self.archive / (goals._BASE + "bulb_screw/config.py")
        path.write_text(path.read_text() + "\n# changed bytes\n", encoding="utf-8")
        self.unavailable(self.build())

    def test_missing_archive_is_unavailable(self):
        shutil.rmtree(self.archive)
        self.unavailable(self.build())

    def test_same_values_in_unknown_source_revision_do_not_bypass_semantic_review(self):
        path = self.archive / (goals._BASE + "bulb_screw/config.py")
        path.write_text(path.read_text() + "\n# unreviewed revision\n", encoding="utf-8")
        row = next(r for r in self.inventory if r["path"].endswith("bulb_screw/config.py"))
        row.update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), size=path.stat().st_size)
        self.rebind_inventory()
        self.unavailable(self.build())

    def test_changed_manifest_bytes_are_unavailable(self):
        self.manifest_path.write_text(self.manifest_path.read_text() + " ", encoding="utf-8")
        self.unavailable(self.build())

    def test_changed_summary_bytes_are_unavailable(self):
        self.summary_path.write_text(self.summary_path.read_text() + " ", encoding="utf-8")
        self.unavailable(self.build())

    def test_manifest_declaration_cannot_disagree_with_its_bound_bytes(self):
        self.summary["deployment_integrity"]["manifest"] = {
            "status": "verified", "expected_sha256": "f" * 64, "actual_sha256": "f" * 64}
        self.save()
        self.unavailable(self.build())

    def test_manifest_integrity_errors_are_not_hidden_by_a_matching_sha(self):
        self.summary["deployment_integrity"]["manifest"] = {
            "status": "invalid", "errors": ["historical manifest verification failed"]}
        self.save()
        self.unavailable(self.build())

    def test_inventory_digest_mismatch_is_unavailable(self):
        self.manifest["evaluation_source_sha256"] = "f" * 64
        self.save()
        self.unavailable(self.build())

    def test_missing_required_config_inventory_entry_is_unavailable(self):
        self.inventory[:] = [row for row in self.inventory if not row["path"].endswith("bulb_screw/config.py")]
        self.rebind_inventory()
        self.unavailable(self.build())

    def test_duplicate_source_inventory_paths_are_rejected(self):
        self.inventory.append(copy.deepcopy(self.inventory[0]))
        self.rebind_inventory()
        self.unavailable(self.build())

    def test_unsafe_source_inventory_paths_are_rejected(self):
        for path in ("../config.py", "/tmp/config.py", "src//config.py", "src\\config.py"):
            with self.subTest(path=path):
                self.inventory[0]["path"] = path
                self.rebind_inventory()
                self.unavailable(self.build())

    def test_source_symlink_leaf_is_not_followed(self):
        original = self.archive / (goals._BASE + "bulb_screw/config.py")
        copy_path = self.root / "other-config.py"
        original.rename(copy_path)
        original.symlink_to(copy_path)
        self.unavailable(self.build())

    def test_report_task_checkpoint_and_staged_rule_cannot_be_mixed(self):
        base = copy.deepcopy(self.report)
        for changed in ("task", "checkpoint", "rule"):
            self.report = copy.deepcopy(base)
            if changed == "task":
                self.report["trial"]["task"] = "bulb-unscrew"
            elif changed == "checkpoint":
                self.report["trial"]["cohort"]["checkpoint"] = "e" * 64
            else:
                self.report["recorded_staged_evaluation"]["rule_identity"]["evaluation_source_sha256"] = "e" * 64
            with self.subTest(changed=changed):
                self.unavailable(self.build())

    def test_contradictory_recorded_source_verification_is_rejected(self):
        self.summary["deployment_integrity"]["evaluation_source"]["actual_sha256"] = "e" * 64
        self.save()
        self.unavailable(self.build())

    def test_source_files_and_inputs_are_not_mutated(self):
        before = copy.deepcopy(self.report)
        source_bytes = {p: p.read_bytes() for p in self.archive.rglob("*.py")}
        result = self.build()
        self.assertEqual(result["status"], "verified", result)
        self.assertEqual(self.report, before)
        self.assertEqual(source_bytes, {p: p.read_bytes() for p in source_bytes})
        self.assertFalse(self.report["outcome"]["task_success"])

    def test_evidence_includes_actual_sha_bound_config_monitor_and_deployment(self):
        result = self.build()
        self.assertEqual(result["status"], "verified", result)
        self.assertEqual(len(result["provenance"]), 4)
        for row in result["provenance"]:
            self.assertEqual(hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest(), row["sha256"])
        json.dumps(result, allow_nan=False)

    def test_missing_primary_path_can_use_explicit_sha_bound_alias(self):
        self.report["provenance"][0].update(path="/retired/manifest.json", resolved_path=str(self.manifest_path))
        result = self.build()
        self.assertEqual(result["status"], "verified", result)

    def test_conflicting_deployment_provenance_is_not_arbitrarily_selected(self):
        self.report["provenance"].append({**self.report["provenance"][0], "sha256": "e" * 64})
        self.unavailable(self.build())

    def test_malformed_evidence_shapes_fail_closed(self):
        for report in (None, [], {}, {"provenance": None}, {"trial": None}):
            with self.subTest(report=report):
                self.unavailable(goals.build_goal_references("bulb-screw", report))
        self.report["trial"]["cohort"] = None
        self.unavailable(self.build())

    def test_static_numeric_subset_rejects_calls_without_execution(self):
        import ast

        config = goals._NumericConfig(ast.parse("TARGET = danger()"))
        with self.assertRaises(ValueError):
            config.number("TARGET")


if __name__ == "__main__":
    unittest.main()
