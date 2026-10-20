"""Independent task review must not promote recorded monitor flags to truth."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/analysis"))
from evaluation_v1.adapters import TrialData
from evaluation_v2.contracts import catalog, get_contract, list_contracts
from evaluation_v2.task_review import review_task


def trial(task, recorded_success=False, **summary):
    return TrialData(root=Path("/nonexistent/task-review-fixture"), task=task,
                     summary={"status": "task_not_completed", "evaluation": {"success": recorded_success}, **summary})


def evidence(task, records, *, complete=True, kind="measurement"):
    contract = get_contract(task)
    keys = {key for _, values in records for key in values}
    return {"task": task, "task_samples": [{"time_s": time, "values": values, "valid": True,
              "source": "independent-test-fixture"} for time, values in records],
            "complete": complete, "sampling": {"mode": "observations", "nominal_interval_s": .05},
            "field_sources": {key: {"kind": kind, "source": "test instrument", "unit": "declared"} for key in keys},
            "contract": {"contract_id": contract["contract_id"], "contract_sha256": contract["contract_sha256"]},
            "provenance": []}


class ContractTests(unittest.TestCase):
    def test_catalog_covers_exactly_twelve_tasks_and_has_content_hashes(self):
        expected = {"pick-place", "poker-draw", "usb-insert", "usb-unplug", "bulb-screw", "bulb-unscrew",
                    "install-ram", "vase-wipe", "sponge-grasp", "whiteboard-wipe",
                    "whiteboard-wipe-horizontal", "whiteboard-wipe-horizontal-line"}
        self.assertEqual(set(catalog()), expected)
        self.assertEqual(len(list_contracts()), 12)
        for contract in list_contracts():
            checksum = contract.pop("contract_sha256")
            self.assertEqual(checksum, hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest())
            self.assertTrue(contract["conditions"])
            self.assertTrue(contract["phases"])
            self.assertTrue(contract["required_evidence"])
            self.assertTrue(all(len(row["sha256"]) == 64 for row in contract["rule_provenance"]))
        value = catalog(); value["pick-place"]["conditions"].clear()
        self.assertTrue(get_contract("pick-place")["conditions"])

    def test_horizontal_home_rules_are_not_silently_applied_to_tilted_board(self):
        self.assertNotIn("home", get_contract("whiteboard-wipe")["conditions"])
        self.assertIn("home", get_contract("whiteboard-wipe-horizontal")["conditions"])
        self.assertIn("home", get_contract("whiteboard-wipe-horizontal-line")["conditions"])

    def test_usb_hand_held_protocol_has_its_own_identity_and_only_four_primitives(self):
        contract = get_contract("usb-unplug")
        self.assertEqual(contract["contract_id"], "usb-unplug:extracted-and-held-2s-review-v1")
        self.assertNotEqual(contract["contract_sha256"], "1a50bf70de4ae67f014de136c571fb1f8a660b12833e3a316076f691a2c36822")
        self.assertEqual(set(contract["required_evidence"]), {
            "insertion_depth_m", "socket_contact_count", "right_hand_contact_count", "table_contact_count"})
        self.assertEqual([(phase["name"], phase["dwell_s"]) for phase in contract["phases"]],
                         [("extracted_and_held_2s", 2.0)])
        project = Path(__file__).resolve().parents[1]
        for row in contract["rule_provenance"]:
            self.assertEqual(row["sha256"], hashlib.sha256((project / row["path"]).read_bytes()).hexdigest())

    def test_non_usb_contract_identities_are_unchanged(self):
        expected = {
            "pick-place": "92b3995aabc1ab60f3f8ec0c05128308819cbebdb788cf864e5cf87bcfbbefb5",
            "vase-wipe": "ea1d1cda1492b37fcffa46600b61af9081041554099b1418d4baaea548d3c0ff",
            "whiteboard-wipe": "4a3a28fde7b30b3abd20cb118e5696858409ed2eb397fc5766c7de9b4100f61a",
            "whiteboard-wipe-horizontal": "c945bb8487ef601536cdca6fd034652cd87defdb04be299dd9ef925252ce7487",
            "whiteboard-wipe-horizontal-line": "7706c14445a04bd3a5ec1736fbe3244014e66eeddf08569aa744c06b05b5cb2a",
            "install-ram": "401397030f05d78eb2c57fdf81cc25666cd03d09364924870333e992ef713eb4",
            "usb-insert": "a532d0d2e97c739a1e07172dee9987872288d0a62ce65dd6c163ce18fb20c544",
            "bulb-screw": "6217b4755e2b1191e5287bdc1a0f1fe218c3d13a0e59ee5a25edbf52cdf90a78",
            "bulb-unscrew": "a52a7119d14eb0af0ceaf61849e9bd968929dc89d9f4586049030c3251ee0106",
            "sponge-grasp": "3dfc46667719cc2c2e4e9021dfd803fbb3dbccc33277611316614d4121553eca",
            "poker-draw": "30bf6de760cd19689664b1f3834ed97824981615afb5acf143c258325085b711",
        }
        for task, checksum in expected.items():
            with self.subTest(task=task):
                self.assertEqual(get_contract(task)["contract_sha256"], checksum)


class ReviewTests(unittest.TestCase):
    def test_unknown_task_preserves_recorded_outcome_without_guessing_fields(self):
        data = trial("unrecognized", True)
        result = review_task(data)
        self.assertFalse(result["supported_task"])
        self.assertEqual(result["recorded_outcome"]["status"], "pass")
        self.assertEqual(result["review"]["status"], "unknown")

    def test_all_known_tasks_with_monitor_booleans_only_remain_unknown(self):
        for task in catalog():
            with self.subTest(task=task):
                data = trial(task, True)
                data.control_rows = [{"sim_time": 0, "task": {"success": True, "seated": True, "released": True}}]
                result = review_task(data)
                self.assertTrue(result["supported_task"])
                self.assertEqual(result["review"]["status"], "unknown")
                self.assertEqual(result["recorded_outcome"]["status"], "pass")
                self.assertTrue(result["missing_evidence"])

    def test_vase_recomputation_passes_inclusive_threshold_and_detects_disagreement(self):
        data = trial("vase-wipe", False)
        ev = evidence(data.task, [(0, {"dirt_remaining": [0, .1]})])
        result = review_task(data, evidence=ev)
        self.assertEqual(result["review"]["status"], "pass")
        self.assertFalse(result["review"]["matches_recorded_outcome"])
        self.assertEqual(result["recorded_outcome"]["status"], "fail")
        ev["task_samples"][0]["values"]["dirt_remaining"] = [0, .10001]
        self.assertEqual(review_task(data, evidence=ev)["review"]["status"], "fail")

    def test_simulated_clean_state_is_not_claimed_to_be_independent_physics(self):
        data = trial("vase-wipe")
        result = review_task(data, evidence=evidence(data.task, [(0, {"dirt_remaining": [0]})], kind="simulation_oracle"))
        self.assertEqual(result["review"]["basis"], "simulation_oracle_predicate_recomputation")
        self.assertFalse(result["review"]["independent_physics_validation"])

    def test_recorded_label_conflict_or_running_trial_blocks_reviewed_verdict(self):
        for summary in ({"success": True}, {"status": "running"}):
            data = trial("vase-wipe", False, **summary)
            ev = evidence(data.task, [(0, {"dirt_remaining": [0]})])
            result = review_task(data, evidence=ev)
            self.assertEqual(result["review"]["status"], "unknown")
            self.assertEqual(result["recorded_outcome"]["status"], "unknown")

    def test_absent_wrong_or_cross_task_contract_never_selects_current_source_defaults(self):
        data = trial("vase-wipe")
        ev = evidence(data.task, [(0, {"dirt_remaining": [0]})])
        for selection in ({}, {"contract_id": "vase-wipe:observable-review-v2", "contract_sha256": "wrong"},
                          evidence("pick-place", []) ["contract"]):
            with self.subTest(selection=selection):
                ev["contract"] = selection
                self.assertEqual(review_task(data, evidence=ev)["review"]["status"], "unknown")

    def test_cross_task_evidence_is_rejected(self):
        ev = evidence("vase-wipe", [(0, {"dirt_remaining": [0]})])
        result = review_task(trial("whiteboard-wipe"), evidence=ev)
        self.assertEqual(result["review"]["status"], "unknown")
        self.assertIn("differs", result["review"]["reason"])

    def test_missing_invalid_and_monitor_declared_values_never_become_zero(self):
        data = trial("vase-wipe")
        for value in (None, [], [None], [float("nan")], [-.1], [1.1]):
            ev = evidence(data.task, [(0, {"dirt_remaining": value})])
            self.assertEqual(review_task(data, evidence=ev)["review"]["status"], "unknown")
        for kind in ("controller_declaration", "monitor_predicate", "undocumented"):
            ev = evidence(data.task, [(0, {"dirt_remaining": [0]})], kind=kind)
            result = review_task(data, evidence=ev)
            self.assertEqual(result["review"]["status"], "unknown")
            self.assertIn("dirt_remaining", result["coverage"]["rejected_fields"])

    def test_original_whiteboard_numeric_diagnostics_are_not_independent_lift_or_ink_arrays(self):
        data = trial("whiteboard-wipe")
        data.control_rows = [{"sim_time": 0, "task": {"maximum_lift_m": .2, "maximum_ink_remaining": .5,
             "table_support_force_n": .3, "eraser_linear_speed_m_s": 0, "fingertip_normal_force_n": [0] * 5}}]
        result = review_task(data)
        diagnostic = {row["name"]: row for row in result["diagnostic_predicates"]}
        self.assertEqual(diagnostic["lift"]["last_status"], "pass")
        self.assertEqual(diagnostic["clean"]["last_status"], "fail")
        self.assertTrue(diagnostic["lift"]["diagnostic_aliases"])
        self.assertEqual(result["review"]["status"], "unknown")
        self.assertIn("eraser_lift_m", result["missing_evidence"])

    def test_pickplace_stability_and_strict_boundary_are_recomputed(self):
        data = trial("pick-place")
        good = {"box_center_clearance_xy_m": [.003, .003], "box_expected_height_error_m": 0,
                "grasp_constraint_active": False, "linear_speed_m_s": .0199, "angular_speed_rad_s": .1}
        ev = evidence(data.task, [(0, good), (.05, good), (.1, good)])
        self.assertEqual(review_task(data, evidence=ev)["review"]["status"], "pass")
        bad = dict(good, linear_speed_m_s=.02)
        ev = evidence(data.task, [(0, bad), (.05, bad), (.1, bad)])
        self.assertEqual(review_task(data, evidence=ev)["review"]["status"], "fail")
        ev = evidence(data.task, [(0, good), (.05, good)], complete=False)
        self.assertEqual(review_task(data, evidence=ev)["review"]["status"], "unknown")

    def test_clock_rewind_and_gaps_do_not_create_stable_dwell(self):
        data = trial("vase-wipe")
        for times in ([0, 0], [.1, 0], [0, .1, .2, 1]):
            result = review_task(data, evidence=evidence(data.task, [(t, {"dirt_remaining": [0]}) for t in times]))
            self.assertEqual(result["review"]["status"], "unknown")

    def test_ram_stable_timer_overlaps_bottom_dwell_and_history_resets(self):
        data = trial("install-ram")
        good = {"insertion_depth_m": .006, "aperture_clearance_xy_m": [0, 0],
                "orientation_error_rad": 0, "maximum_socket_penetration_m": 0,
                "linear_speed_m_s": 0, "angular_speed_rad_s": 0, "backstop_load_n": 1.2}
        ev = evidence(data.task, [(i / 10, good) for i in range(5)])
        result = review_task(data, evidence=ev)
        self.assertEqual(result["review"]["status"], "pass")
        self.assertEqual(result["progress"]["completed_phase_count"], 2)
        records = [(i / 10, good) for i in range(5)] + [(.5, dict(good, insertion_depth_m=.005)),
                    (.6, dict(good, backstop_load_n=0)), (.7, dict(good, backstop_load_n=0))]
        self.assertEqual(review_task(data, evidence=evidence(data.task, records))["review"]["status"], "fail")

    def test_usb_current_bottom_support_requires_axial_component_too(self):
        data = trial("usb-insert")
        good = {"insertion_depth_m": .012, "aperture_clearance_xy_m": [0, 0],
                "orientation_error_rad": 0, "maximum_socket_penetration_m": 0,
                "linear_speed_m_s": 0, "angular_speed_rad_s": 0,
                "backstop_contact_count": 1, "backstop_normal_load_n": .01, "backstop_axial_resistance_n": .0001}
        ev = evidence(data.task, [(0, good), (.05, good), (.1, good)])
        self.assertEqual(review_task(data, evidence=ev)["review"]["status"], "pass")
        bad = dict(good, backstop_axial_resistance_n=0)
        self.assertEqual(review_task(data, evidence=evidence(data.task, [(0, bad), (.05, bad), (.1, bad)]))["review"]["status"], "fail")

    def test_whiteboard_observed_lift_then_release_and_horizontal_home(self):
        for task in ("whiteboard-wipe", "whiteboard-wipe-horizontal", "whiteboard-wipe-horizontal-line"):
            with self.subTest(task=task):
                records = []
                for i in range(9):
                    loaded = i < 3
                    records.append((i / 20, {"eraser_lift_m": .13, "ink_remaining": [0, 0],
                        "table_support_force_n": 0 if loaded else .21,
                        "eraser_linear_speed_m_s": 0, "fingertip_normal_force_n": [.05] * 5 if loaded else [0] * 5,
                        "arm_home_error_rad": 0, "hand_open_error_rad": 0}))
                self.assertEqual(review_task(trial(task), evidence=evidence(task, records))["review"]["status"], "pass")
                if task != "whiteboard-wipe":
                    bad = [(t, dict(values, arm_home_error_rad=.01)) for t, values in records]
                    self.assertEqual(review_task(trial(task), evidence=evidence(task, bad))["review"]["status"], "fail")

    def test_poker_requires_supported_edge_before_lift_then_a_facing_hold(self):
        task = "poker-draw"
        values = {"finger_card_normal_force_n": [.02], "maximum_table_contact_force_n": .00002,
            "card_up_cosine": .94, "overhang_fraction": .4, "robotward_displacement_m": .05,
            "thumb_bottom_force_n": .05, "fingers_top_force_n": .05, "card_table_clearance_m": .02,
            "face_to_head_cosine": .8, "face_to_robot_cosine": .8, "inspection_position_error_m": .03,
            "linear_speed_m_s": .019, "angular_speed_rad_s": .19}
        records = [(i / 20, values) for i in range(6)]
        self.assertEqual(review_task(trial(task), evidence=evidence(task, records))["review"]["status"], "pass")
        unsupported = [(t, dict(v, maximum_table_contact_force_n=1e-5)) for t, v in records]
        self.assertEqual(review_task(trial(task), evidence=evidence(task, unsupported))["review"]["status"], "fail")

    def test_bulb_screw_recomputes_loaded_stall_then_release_not_lit_flag(self):
        task = "bulb-screw"
        values = {"engaged": True, "insertion_depth_m": .006 + .004 * 100 / 360,
            "clockwise_turns": 100 / 360, "thread_error_m": 0, "lateral_error_m": 0,
            "tilt_rad": 0, "linear_speed_m_s": 0, "angular_speed_rad_s": 0,
            "backstop_load_n": .01, "exposed_thread_m": 0, "shoulder_gap_m": 0,
            "shoulder_contact_load_n": .01, "hand_clockwise_torque_nm": .1,
            "right_hand_nonpad_contact_count": 0, "other_robot_contact_count": 0,
            "right_hand_contact_load_n": 0, "robot_contact_load_n": 0, "robot_contact_count": 0,
            "fingertip_normal_force_n": [4] * 5, "bulb_yaw_rad": 0, "bulb_lit": True}
        records = [(i / 20, dict(values, fingertip_normal_force_n=[4] * 5 if i <= 6 else [0] * 5)) for i in range(13)]
        self.assertEqual(review_task(trial(task), evidence=evidence(task, records))["review"]["status"], "pass")
        rotating = [(t, dict(v, bulb_yaw_rad=t)) for t, v in records]
        self.assertEqual(review_task(trial(task), evidence=evidence(task, rotating))["review"]["status"], "fail")
        partial = evidence(task, records)
        partial["task_samples"][3]["values"].pop("bulb_yaw_rad")
        self.assertEqual(review_task(trial(task), evidence=partial)["review"]["status"], "unknown")

    def test_sponge_support_mean_is_recomputed_from_impulse_and_cannot_be_faked_by_a_monitor_mean(self):
        task = "sponge-grasp"
        records = []
        for i in range(41):
            t = i / 20
            records.append((t, {"fingertip_normal_force_n": [.05] * 5 if t < .75 else [0] * 5,
                "sponge_clearance_m": .12 if .55 <= t < .6 else 0,
                "table_support_force_n": 0,
                "sponge_plate_center_offset_m": .03 if t >= .6 else .2,
                "plate_base_support_force_n": .3 if t >= .6 else 0,
                "recent_plate_base_mean_support_force_n": 999,  # Must not be trusted.
                "sponge_maximum_plate_radius_m": .05, "sponge_bottom_z_m": .69,
                "grasp_patch_enabled": t < .75, "sponge_pin_z_m": .70, "sponge_centroid_z_m": .71,
                "plate_base_support_impulse_ns": max(0, t - .6) * .3}))
        self.assertEqual(review_task(trial(task), evidence=evidence(task, records))["review"]["status"], "pass")
        unsupported = [(t, dict(v, plate_base_support_impulse_ns=0)) for t, v in records]
        result = review_task(trial(task), evidence=evidence(task, unsupported))
        self.assertEqual(result["review"]["status"], "fail")
        decreasing = deepcopy(records)
        decreasing[30][1]["plate_base_support_impulse_ns"] = -.1
        self.assertEqual(review_task(trial(task), evidence=evidence(task, decreasing))["review"]["status"], "unknown")

    def test_usb_unplug_only_requires_complete_clear_hand_held_two_second_window(self):
        task = "usb-unplug"
        good = {"insertion_depth_m": -.002, "socket_contact_count": 0,
                "table_contact_count": 0, "right_hand_contact_count": 1}
        records = [(i / 20, dict(good)) for i in range(41)]
        result = review_task(trial(task), evidence=evidence(task, records))
        self.assertEqual(result["review"]["status"], "pass")
        self.assertEqual([row["name"] for row in result["events"]], ["extracted_and_held_2s"])
        self.assertEqual(review_task(trial(task), evidence=evidence(task, records[:-1]))["review"]["status"], "fail")
        # No initial-state, force, speed, position, release or retreat primitives are needed.
        extra = [(time, dict(values, initially_seated_verified=False, grip_force_n=[0, 0],
                            linear_speed_m_s=8, home_xy_error_m=2, released=False,
                            hand_retreated=False)) for time, values in records]
        self.assertEqual(review_task(trial(task), evidence=evidence(task, extra))["review"]["status"], "pass")

    def test_usb_unplug_hold_restarts_on_each_disqualifying_condition(self):
        task = "usb-unplug"
        good = {"insertion_depth_m": -.002, "socket_contact_count": 0,
                "table_contact_count": 0, "right_hand_contact_count": 1}
        for changed in ({"insertion_depth_m": -.00199}, {"socket_contact_count": 1},
                        {"right_hand_contact_count": 0}, {"table_contact_count": 1}):
            records = [(i / 20, dict(good, **changed) if i == 20 else dict(good)) for i in range(41)]
            with self.subTest(changed=changed):
                self.assertEqual(review_task(trial(task), evidence=evidence(task, records))["review"]["status"], "fail")
        records = [(i / 20, dict(good)) for i in range(41)]
        ev = evidence(task, records)
        ev["task_samples"][20]["values"].pop("table_contact_count")
        self.assertEqual(review_task(trial(task), evidence=ev)["review"]["status"], "unknown")

    def test_usb_old_contract_and_source_witness_do_not_select_new_rules_or_replace_native(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "seed").mkdir()
            manifest = root / "deployment_manifest.json"
            old_sources = {
                "tasks/usb_unplug/task.py": "a95d39ab0295187cd7ce19504138a99ba1cf633910ddd87c429f022de52b0f83",
                "tasks/usb_unplug/config.py": "374ac5656a1c0980ecc2e793f82a09fd2dce2f3a61e214ce88c53bd56cfcfc69",
            }
            manifest.write_text(json.dumps({"evaluation_source_files": [
                {"path": "src/kaihand_tactile_env/" + name, "sha256": checksum}
                for name, checksum in old_sources.items()]}))
            data = trial("usb-unplug", True, deployment_manifest=str(manifest))
            data.root = root / "seed"
            good = {"insertion_depth_m": -.002, "socket_contact_count": 0,
                    "table_contact_count": 0, "right_hand_contact_count": 1}
            ev = evidence(data.task, [(i / 20, good) for i in range(41)])
            ev["contract"] = {"contract_id": "usb-unplug:observable-review-v2",
                              "contract_sha256": "1a50bf70de4ae67f014de136c571fb1f8a660b12833e3a316076f691a2c36822"}
            result = review_task(data, evidence=ev)
            self.assertEqual(result["source_match"]["status"], "unverified")
            self.assertEqual(result["contract_selection"]["status"], "unverified")
            self.assertEqual(result["review"]["status"], "unknown")
            self.assertEqual(result["recorded_outcome"]["status"], "pass")
            self.assertTrue(result["original_outcome_preserved"])

    def test_bulb_unscrew_replays_mechanical_removal_order_and_rejects_premature_disengagement(self):
        task = "bulb-unscrew"
        records = []
        for i in range(25):
            t = i / 20
            grasped = .1 <= t < .8
            turns = 100 / 360 if t < .25 else 100 / 360 - .1 if t < .3 else -.07
            records.append((t, {"engaged": t < .35, "insertion_depth_m": .006 + .004 * 100 / 360 if t < .35 else -.003,
                "clockwise_turns": turns, "thread_error_m": 0, "lateral_error_m": 0, "tilt_rad": 0,
                "linear_speed_m_s": 0, "angular_speed_rad_s": 0, "backstop_load_n": .01,
                "exposed_thread_m": 0, "shoulder_gap_m": 0, "shoulder_contact_load_n": .01,
                "robot_contact_count": 1 if grasped else 0, "fingertip_normal_force_n": [.025] * 5 if grasped else [0] * 5,
                "right_hand_nonpad_contact_count": 0, "turns_removed": max(0, 100 / 360 - turns),
                "hand_counterclockwise_torque_nm": .002 if t >= .25 else 0,
                "socket_contact_load_n": 0, "socket_contact_count": 0 if t >= .35 else 1,
                "table_xy_error_m": 0, "table_z_error_m": 0, "table_contact_load_n": .1,
                "table_contact_count": 1, "right_hand_contact_load_n": .125 if grasped else 0,
                "robot_contact_load_n": .125 if grasped else 0, "arm_home_error_rad": 0,
                "hand_home_error_rad": 0, "home_joint_speed_rad_s": 0, "wrist_object_distance_m": .1}))
        result = review_task(trial(task), evidence=evidence(task, records))
        self.assertEqual(result["review"]["status"], "pass")
        premature = deepcopy(records)
        premature[3][1]["engaged"] = False
        result = review_task(trial(task), evidence=evidence(task, premature))
        self.assertEqual(result["review"]["status"], "fail")
        self.assertEqual(result["progress"]["observed_failure"], "premature_thread_disengagement")

    def test_unknown_condition_and_fully_observed_false_intervals_are_distinct(self):
        data = trial("vase-wipe")
        result = review_task(data, evidence=evidence(data.task, [(0, {"dirt_remaining": [.9]})]))
        self.assertEqual(result["predicate_review"][0]["status"], "not_observed")
        self.assertEqual(result["condition_intervals"], [])
        result = review_task(data, evidence=evidence(data.task, [(0, {"dirt_remaining": None})]))
        self.assertEqual(result["predicate_review"][0]["status"], "missing")
        self.assertEqual(result["condition_intervals"], [])

    def test_bundle_protocol_inputs_not_mutated_and_output_is_finite_json(self):
        data = trial("vase-wipe")
        ev = evidence(data.task, [(0, {"dirt_remaining": [0]})])
        original = deepcopy(ev)
        class Bundle:
            def task_evidence(self):
                return ev
        result = review_task(data, evidence=Bundle())
        self.assertEqual(result["review"]["status"], "pass")
        self.assertEqual(ev, original)
        json.dumps(result, allow_nan=False)

    def test_local_manifest_is_hashed_but_does_not_silently_select_an_acceptance_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "seed").mkdir()
            contract = get_contract("vase-wipe")
            value = {"evaluation_source_files": [{"path": row["path"], "sha256": row["sha256"]} for row in contract["rule_provenance"]]}
            manifest = root / "deployment_manifest.json"
            manifest.write_text(json.dumps(value))
            data = trial("vase-wipe", deployment_manifest=str(manifest))
            data.root = root / "seed"
            result = review_task(data)
            self.assertEqual(result["source_match"]["status"], "matched")
            self.assertEqual(result["contract_selection"]["status"], "unverified")
            self.assertEqual(result["review"]["status"], "unknown")
            self.assertEqual(len(result["provenance"][0]["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
