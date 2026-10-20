"""Versioned, JSON-safe task contracts; importing this module never loads a simulator.

These are explicit review contracts, not a claim that today's simulator rules
were used by an older rollout.  Legacy source identities were captured from the
read-only ``kaitactilesim`` checkout.  USB's explicit held-clear protocol instead
binds its newly reviewed ``kaitactilesim_unified`` source revision.
An independent verdict additionally requires a declared matching contract and
primitive evidence.  Numeric diagnostics alone do not certify a task outcome.
"""
from copy import deepcopy
import hashlib
import json
import math

SCHEMA_VERSION = "evaluation-task-contracts-v2"
SOURCE_ROOT = "/cpfs_infra/user/chenxianchi/code/kaitactilesim"
_ROOT = "src/kaihand_tactile_env/"
_HASHES = {
    "shared/policy_tasks.py": "6ae2b361d2fc9ca4f5bbafdaa316acaf49861c832eaca6aa9060460d174876a0",
    "shared/recording.py": "83668e74b5d4ba9cf2c672f20f3ac47188f0a453d1e0de80a877cfc14052ffd9",
    "tasks/pick_place/config.py": "1afa0cb592db52a5ff98c87029eba9e784b52f59da4aec3eb951cf93acf32d98",
    "tasks/pick_place/task.py": "34eba0b18c923bd86aa840174582a522da0a9c4d8b4548c5dcf8eb72cf598d8b",
    "tasks/poker_draw/rollout_eval.py": "ff399929bdb09d1beb61b1a32635b0d4b81b2f5958134231ebac47420b18d2ba",
    "tasks/poker_draw/policy_control.py": "01f360ad4f918b0bcf8621322f977bcbc3b81ab545b35d4d9212f7e87dea9e4b",
    "tasks/usb_insert/config.py": "9017d39006d0a76b692036aaed1708802134eb7d48e5cee8ddb3da13b0aa8b12",
    "tasks/usb_insert/task.py": "fb89bb3e3bfd37902767acd5abe2cee2f5a2cbf4a3f37dbb3bfb89879396772e",
    "tasks/usb_unplug/config.py": "374ac5656a1c0980ecc2e793f82a09fd2dce2f3a61e214ce88c53bd56cfcfc69",
    "tasks/usb_unplug/task.py": "a95d39ab0295187cd7ce19504138a99ba1cf633910ddd87c429f022de52b0f83",
    "tasks/install_ram/config.py": "b46f59f9c3f127d18dbc09ef9ac1213455f01076231cf086e1274faa17756b1b",
    "tasks/install_ram/task.py": "6092f5918fd3cba9391b46c43590f646107d66a8168a9bffd723ac6eddc95766",
    "tasks/bulb_screw/config.py": "4033b941c798ce6dd31df20109338fafc384140ff0db8d9c8874672ca2914b42",
    "tasks/bulb_screw/task.py": "585426d36475d583fdb43a2b48b36ab1c2381d6e32459e42285421e6869be94a",
    "tasks/bulb_unscrew/config.py": "4c466351b4b4781e16d95c3b8f304db87e502b60556c8cdd636a92959320b880",
    "tasks/bulb_unscrew/task.py": "c5349e3a767f358c8a64ecd42cc3a02ffddc4a3ace1ce65c2c559d05000ae6e9",
    "tasks/vase_wipe/config.py": "ad7dc0002351a11306016c879306d357d44f349abb6eb0786ef8bdc323d0cc17",
    "tasks/vase_wipe/cleaning.py": "dab3047176c38b62f4e276b2f86ee3da3144063dc67324f2c33946f12e85ccbd",
    "tasks/sponge_grasp/config.py": "e8cc93bb8855b231f8255ac82fff1df964b91000dad89f0d5bf737a2174a1f9c",
    "tasks/whiteboard_wipe/config.py": "15b7dc10978695496aa1cf31f33068e980e529a79737f7c293deffc23bd6593c",
    "tasks/whiteboard_wipe_horizontal/config.py": "0c84d542a3c7c575f7a0f5208fcdb7af692fde78a0d72c0965ad97a1192f3b49",
    "tasks/whiteboard_wipe_horizontal_line/config.py": "af7ef79737cfe78c8d1b21a72c44bca5b3eb5162c84716e38c40c5d5ba72a6b8",
}
_SCRIPT = {"scripts/workcell/run_egosteer_policy.py": "7a907e622bba8388f5a683de875102865a2ea0590e829bedf3e99f2b21ef05d8"}


def _p(field, op, value, *, reduce=None, length=None, center=0):
    result = {"field": field, "op": op, "value": value}
    if reduce:
        result["reduce"] = reduce
    if length:
        result["length"] = length
    if center:
        result["center"] = center
    return result


def _phase(name, conditions, dwell=0):
    return {"name": name, "conditions": conditions, "dwell_s": dwell,
            "evidence_kind": "sampled_primitive_predicates", "controller_stage_is_ground_truth": False}


def _contract(task, title, conditions, phases, sources, *, terminal=(), notes=(), engine="ordered",
              contract_id=None, source_root=SOURCE_ROOT, source_hashes=None):
    provenance = []
    for source, locator in sources:
        relative = source if source.startswith("scripts/") else _ROOT + source
        provenance.append({"path": relative, "sha256": (source_hashes or {}).get(source, _SCRIPT.get(source, _HASHES.get(source))),
                           "locator": locator, "source_root_at_catalog_creation": source_root,
                           "relationship_to_trial": "unverified_until_source_or_contract_match"})
    required = sorted({p["field"] for items in conditions.values() for p in items})
    required += {"bulb_screw": ["bulb_yaw_rad"], "sponge": ["plate_base_support_impulse_ns", "sponge_pin_z_m", "sponge_centroid_z_m"]}.get(engine, [])
    data = {"task": task, "title": title, "version": SCHEMA_VERSION,
            "contract_id": contract_id or f"{task}:observable-review-v2", "engine": engine,
            "conditions": conditions, "phases": phases, "terminal_conditions": list(terminal),
            "required_evidence": required, "rule_provenance": provenance,
            "failure_semantics": "Only complete, valid, adequately sampled evidence under the selected contract can establish failure. Timeout and partial trajectories remain in the denominator; missing evidence is unknown, not failure or success.",
            "notes": list(notes), "calibration": "Source-derived simulator thresholds, not independently calibrated real-world tolerances.",
            "scope": "Source-derived observable review contract, not a byte-for-byte replay of the original monitor. Conservative ordered witnesses may differ at sampling boundaries; not independent proof of continuous real-world behavior."}
    data["contract_sha256"] = hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return data


def _build():
    result = {}
    result["pick-place"] = _contract("pick-place", "Released stable cylinder placement", {
        "inside_box": [_p("box_center_clearance_xy_m", "gt", .002, reduce="min", length=2),
                       _p("box_expected_height_error_m", "abs_lt", .05)],
        "released_stable": [_p("grasp_constraint_active", "is", False),
                            _p("linear_speed_m_s", "lt", .02), _p("angular_speed_rad_s", "lt", .2)],
    }, [_phase("stable_placement", ["inside_box", "released_stable"], .1)], [
        ("tasks/pick_place/task.py", "cylinder_is_in_box, lines 898-924"),
        ("scripts/workcell/run_egosteer_policy.py", "StablePlacementDetector, lines 160-203"),
        ("shared/recording.py", "TERMINAL_* constants, lines 42-44")],
        notes=["Box clearance is inner half-width minus cylinder radius minus absolute center offset, before the 2 mm safety margin.",
               "Original placement check does not certify upright orientation or absence of all hand contacts; grasp_constraint_active is an actual constraint state, not a commanded release."])
    result["vase-wipe"] = _contract("vase-wipe", "Vase dirt state", {
        "clean": [_p("dirt_remaining", "le", .05, reduce="mean"),
                  _p("dirt_remaining", "le", .10, reduce="max")],
    }, [_phase("clean", ["clean"])], [("tasks/vase_wipe/cleaning.py", "CleaningProgress.success, lines 54-59"),
        ("tasks/vase_wipe/config.py", "MAX_MEAN_DIRT/MAX_PATCH_DIRT, lines 74-75")], terminal=["clean"],
        notes=["This matches the model-only policy monitor's cleaning-state criterion, not the expert executor's additional pickup checks.",
               "Dirt remaining must be a complete nonempty patch vector in [0,1]; simulated cleaning state is not independently measured real dirt."])
    for task in ("whiteboard-wipe", "whiteboard-wipe-horizontal", "whiteboard-wipe-horizontal-line"):
        horizontal = task != "whiteboard-wipe"
        conditions = {
            "lift": [_p("eraser_lift_m", "gt", .12)],
            "clean": [_p("ink_remaining", "le", 1e-9, reduce="max")],
            "released_support": [_p("table_support_force_n", "gt", .2),
                _p("eraser_linear_speed_m_s", "lt", .01),
                _p("fingertip_normal_force_n", "lt", .05, reduce="sum", length=5)],
        }
        phases = [_phase("lift", ["lift"]), _phase("clean_and_release", ["clean", "released_support"], .1)]
        terminal = ["clean", "released_support"]
        if horizontal:
            conditions.update(opposed_grasp=[_p("thumb_normal_force_n", "ge", .05), _p("other_fingers_normal_force_n", "ge", .15)],
                              free_lift=[_p("eraser_lift_m", "ge", .12), _p("table_support_force_n", "le", .01)],
                              home=[_p("arm_home_error_rad", "lt", .01), _p("hand_open_error_rad", "lt", .06)])
            phases = [_phase("opposed_grasp", ["opposed_grasp"]), _phase("pickup", ["free_lift"]), *phases,
                      _phase("home", ["clean", "released_support", "home"])]
            terminal += ["home"]
        result[task] = _contract(task, "Whiteboard pickup, cleaning and release", conditions, phases,
            [("shared/policy_tasks.py", "_whiteboard_state/_update_horizontal_pickup/_whiteboard_release_condition, lines 383-452"),
             (f"tasks/{task.replace('-', '_')}/config.py", "Task-local geometry and OPEN_HAND_MAX_ERROR_RAD")],
            terminal=terminal, notes=["Tilted-board contract does not require home return; horizontal variants do.",
                "Review requires a witnessed lift before clean release. It does not reuse maximum_lift_m, released_on_table, pickup_verified or controller stages as independent evidence.",
                "Exact loaded-cleaning law replay additionally needs every patch's contact/work inputs; ink state alone only reviews the terminal simulator state."])
    ram_conditions = {
        "near_bottom": [_p("insertion_depth_m", "abs_le", .0002, center=.006),
            _p("aperture_clearance_xy_m", "ge", -1e-6, reduce="min", length=2),
            _p("orientation_error_rad", "le", math.radians(1)), _p("maximum_socket_penetration_m", "le", .0002)],
        "stationary": [_p("linear_speed_m_s", "lt", .001), _p("angular_speed_rad_s", "lt", .05)],
        "bottom_load": [_p("backstop_load_n", "ge", 1.2)],
        "current_bottom_support": [_p("backstop_load_n", "gt", .01)],
    }
    result["install-ram"] = _contract("install-ram", "RAM bottom-out and stable seating", ram_conditions,
        [_phase("bottom_out", ["near_bottom", "stationary", "bottom_load"], .4),
         _phase("stable_seating", ["near_bottom", "stationary"], .3)],
        [("tasks/install_ram/task.py", "RamInstallationMonitor.measure/update, lines 101-248"),
         ("tasks/install_ram/config.py", "TARGET_INSERTION_DEPTH_M and seating constants, lines 33-50")],
        terminal=["near_bottom", "stationary"], engine="ram",
        notes=["Aperture clearance must cover the entire inserted edge, including tilt, not only the center.",
               "Bottom-out history resets when geometry leaves near-bottom; stable seating timer can overlap bottom dwell.",
               "Mechanical seating is not electrical connectivity, latch certification, or proof of release."])
    usb_conditions = {
        "near_bottom": [_p("insertion_depth_m", "ge", .0119), _p("insertion_depth_m", "le", .0121),
            _p("aperture_clearance_xy_m", "ge", 0, reduce="min", length=2),
            _p("orientation_error_rad", "le", math.radians(3)), _p("maximum_socket_penetration_m", "le", .0001)],
        "stationary": [_p("linear_speed_m_s", "le", .0005), _p("angular_speed_rad_s", "le", .05)],
        "current_bottom_support": [_p("backstop_contact_count", "gt", 0), _p("backstop_normal_load_n", "ge", .0001),
                                   _p("backstop_axial_resistance_n", "ge", .0001)],
        "bottom_load": [_p("backstop_axial_resistance_n", "ge", .6)],
    }
    result["usb-insert"] = _contract("usb-insert", "USB mechanical seating", usb_conditions,
        [_phase("bottom_out", ["near_bottom", "stationary", "current_bottom_support", "bottom_load"], .15),
         _phase("stable_seating", ["near_bottom", "stationary"], .1)],
        [("tasks/usb_insert/task.py", "_near_bottom/_stationary/_with_seating/update, lines 52-81 and 238-279"),
         ("tasks/usb_insert/config.py", "BACKSTOP_DEPTH_M and seating constants, lines 74-94")],
        terminal=["near_bottom", "stationary"], engine="usb",
        notes=["Current measured bottom contact or retained qualified bottom-out history is required; friction alone is not seating.",
               "Mechanical seating does not prove electrical connection or hand release."])
    unplug = {
        "fully_clear": [_p("insertion_depth_m", "le", -.002), _p("socket_contact_count", "eq", 0)],
        "hand_held": [_p("right_hand_contact_count", "gt", 0), _p("table_contact_count", "eq", 0)],
    }
    result["usb-unplug"] = _contract("usb-unplug", "USB extract and remain hand-held for 2 seconds", unplug,
        [_phase("extracted_and_held_2s", ["fully_clear", "hand_held"], 2.0)],
        [("tasks/usb_unplug/task.py", "UsbUnplugMonitor.measure/update: continuous clear hand-held window"),
         ("tasks/usb_unplug/config.py", "FULL_CLEAR_DEPTH_M and extraction hold completion protocol")],
        terminal=["fully_clear", "hand_held"],
        contract_id="usb-unplug:extracted-and-held-2s-review-v1",
        source_root="/cpfs_infra/user/chenxianchi/code/kaitactilesim_unified",
        source_hashes={
            "tasks/usb_unplug/config.py": "05692c0debad1f339936690ad8dd4c7aceb49778b773e22f540eb1b3f9be26ee",
            "tasks/usb_unplug/task.py": "b1755b37c9c7d8d24b14245c5b40c8f91bdf6de2b60c1ddb6c80bdf8d8c3efc0",
        },
        notes=["The plug must remain at depth <= -2 mm with no socket contact, right-hand contact and no table contact continuously for at least 2 seconds.",
               "Initial installation, opposed grip, speed, position, table return, release and retreat are not success gates in this protocol.",
               "This is a distinct explicitly selected protocol; old source witnesses or old contract ids do not opt historical trials into the new rule. Archived native outcomes remain unchanged."])
    seated = [_p("engaged", "is", True), _p("insertion_depth_m", "abs_le", .0003, center=.006 + .004 * 100 / 360),
        _p("clockwise_turns", "abs_le", .08, center=100 / 360), _p("thread_error_m", "abs_le", .0003),
        _p("lateral_error_m", "le", .0004), _p("tilt_rad", "le", math.radians(2)),
        _p("linear_speed_m_s", "le", .001), _p("angular_speed_rad_s", "le", .05),
        _p("backstop_load_n", "ge", .01), _p("exposed_thread_m", "le", .0001),
        _p("shoulder_gap_m", "abs_le", .0001), _p("shoulder_contact_load_n", "ge", .01)]
    released = [_p("fingertip_normal_force_n", "le", .01, reduce="max", length=5),
        _p("right_hand_contact_load_n", "le", .01), _p("robot_contact_load_n", "le", .01), _p("robot_contact_count", "eq", 0)]
    bulb = {"seated": seated, "tightening": [_p("hand_clockwise_torque_nm", "ge", .1),
        _p("fingertip_normal_force_n", "ge", 4, reduce="min", length=5),
        _p("right_hand_nonpad_contact_count", "eq", 0), _p("other_robot_contact_count", "eq", 0)],
        "released": released}
    result["bulb-screw"] = _contract("bulb-screw", "Bulb loaded tightening and release", bulb,
        [_phase("tightening", ["seated", "tightening"], .3), _phase("released", ["seated", "released"], .2)],
        [("tasks/bulb_screw/task.py", "measure/_update_tightening/update, lines 434-653"),
         ("tasks/bulb_screw/config.py", "lines 25-58")], terminal=["seated", "released"], engine="bulb_screw",
        notes=["Tightening also requires an unwrapped bulb yaw range <= 0.3 degrees during the same 0.3 s window.",
               "engaged is the measured guide state, not a controller stage; lit/tightening_verified booleans are not accepted as proof.",
               "This explicitly selects policy-mode tightening; mechanics-demo seating-only mode needs another contract."])
    unscrew = {"initial_seated": seated + [_p("robot_contact_count", "eq", 0)],
        "grasp": [_p("fingertip_normal_force_n", "ge", .025, reduce="min", length=5), _p("right_hand_nonpad_contact_count", "eq", 0)],
        "breakaway": [_p("turns_removed", "ge", 30 / 360), _p("hand_counterclockwise_torque_nm", "ge", .002)],
        "fully_clear": [_p("engaged", "is", False), _p("insertion_depth_m", "le", -.003),
            _p("socket_contact_load_n", "le", .01), _p("socket_contact_count", "eq", 0)],
        "placed": [_p("engaged", "is", False), _p("table_xy_error_m", "le", .0015), _p("table_z_error_m", "le", .004),
            _p("tilt_rad", "le", math.radians(8)), _p("linear_speed_m_s", "le", .004), _p("angular_speed_rad_s", "le", .2),
            _p("table_contact_load_n", "ge", .01), _p("table_contact_load_n", "le", 12),
            _p("table_contact_count", "gt", 0), _p("socket_contact_count", "eq", 0)],
        "released": released, "home": [_p("arm_home_error_rad", "le", math.radians(1)),
            _p("hand_home_error_rad", "le", math.radians(4)), _p("home_joint_speed_rad_s", "le", math.radians(2)),
            _p("wrist_object_distance_m", "ge", .1)]}
    result["bulb-unscrew"] = _contract("bulb-unscrew", "Bulb remove, place and return home", unscrew,
        [_phase("initial_seated", ["initial_seated"], .05), _phase("grasp", ["grasp"], .1),
         _phase("breakaway", ["breakaway"]), _phase("fully_clear", ["fully_clear"], .08),
         _phase("placed", ["placed"], .2), _phase("released", ["placed", "released"], .1),
         _phase("home", ["placed", "released", "home"], .1)],
        [("tasks/bulb_unscrew/task.py", "measure/update, lines 325-581"), ("tasks/bulb_unscrew/config.py", "lines 61-101"),
         ("tasks/bulb_screw/config.py", "Inherited thread pitch, turns and release thresholds")],
        terminal=["placed", "released", "home"], engine="bulb_unscrew",
        notes=["A measured engaged->disengaged transition must follow counterclockwise breakaway and sufficient removed turns; forceful premature disengagement fails.",
               "Reset installation validation and the bulb light state are separate simulator diagnostics, not independently certified electrical state."])
    sponge = {
        "grasp": [_p("fingertip_normal_force_n", "ge", .05, reduce="min", length=5)],
        "lift": [_p("sponge_clearance_m", "ge", .12), _p("table_support_force_n", "le", .01)],
        "carry": [_p("sponge_clearance_m", "ge", .08), _p("table_support_force_n", "lt", .05)],
        "placement_zone": [_p("sponge_plate_center_offset_m", "le", .05)],
        "placement_support": [_p("plate_base_support_force_n", "gt", .02)],
        "plate_support": [_p("recent_plate_base_mean_support_force_n", "ge", .08)],
        "released": [_p("fingertip_normal_force_n", "lt", .1, reduce="max", length=5)],
        "inside_plate": [_p("sponge_plate_center_offset_m", "le", .035), _p("sponge_maximum_plate_radius_m", "le", .082),
            _p("sponge_bottom_z_m", "ge", .683), _p("sponge_bottom_z_m", "le", .703)],
        "settled": [_p("grasp_patch_enabled", "is", False), _p("table_support_force_n", "lt", .05)],
    }
    result["sponge-grasp"] = _contract("sponge-grasp", "Five-finger carry and stable plate release", sponge,
        [_phase("grasp", ["grasp"], .5), _phase("lift", ["grasp", "lift"]),
         {**_phase("placement", ["grasp", "plate_support"]), "any_conditions": ["placement_zone", "placement_support"]},
         _phase("released", ["released"], .1), _phase("settled", ["inside_plate", "released", "settled"], 1)],
        [("shared/policy_tasks.py", "_sponge_sample/_update_sponge, lines 453-659"),
         ("tasks/sponge_grasp/config.py", "PLATE_* and grasp/settle constants, lines 28-57")],
        terminal=["inside_plate", "released", "settled"], engine="sponge",
        notes=["Carry contact must persist until qualified placement; initial transport_contact_valid=True is not evidence that transport occurred.",
               "Additional raw plate-support impulse and pin/centroid/bottom heights are required to recompute support means and <=3 mm settling ranges."])
    poker = {"contact": [_p("finger_card_normal_force_n", "gt", .01, reduce="max")],
        "edge": [_p("maximum_table_contact_force_n", "gt", 1e-5), _p("card_up_cosine", "ge", .94),
            _p("overhang_fraction", "ge", .4), _p("robotward_displacement_m", "ge", .05)],
        "opposed": [_p("thumb_bottom_force_n", "ge", .05), _p("fingers_top_force_n", "ge", .05)],
        "lift": [_p("card_table_clearance_m", "ge", .02)],
        "facing_hold": [_p("face_to_head_cosine", "ge", .8), _p("face_to_robot_cosine", "ge", .8),
            _p("inspection_position_error_m", "le", .03), _p("linear_speed_m_s", "lt", .02), _p("angular_speed_rad_s", "lt", .2)]}
    result["poker-draw"] = _contract("poker-draw", "Supported draw, opposed lift and inspection", poker,
        [_phase("contact", ["contact"]), _phase("edge", ["edge"]), _phase("lift", ["opposed", "lift"]),
         _phase("facing_hold", ["opposed", "facing_hold"], .1)],
        [("tasks/poker_draw/rollout_eval.py", "PokerOutcomeMonitor.observe/observe_simulation, lines 9-159"),
         ("tasks/poker_draw/policy_control.py", "_supported/terminal_metrics, lines 136-143 and 321-340")], engine="poker",
        notes=["Selects the model-only outcome contract, not expert force-band acceptance.",
               "Supported/opposed contact predicates require their primitive contact geometry; a controller stage cannot supply them.",
               "Optional original require_clearance_during_hold changes the contract and is never guessed."])
    return result


TASK_CONTRACTS = _build()


def catalog():
    """Return a mutation-safe, finite JSON task directory for the portal."""
    return deepcopy(TASK_CONTRACTS)


def get_contract(task):
    value = TASK_CONTRACTS.get(task)
    return deepcopy(value) if value is not None else None


def list_contracts():
    return list(catalog().values())


__all__ = ["SCHEMA_VERSION", "TASK_CONTRACTS", "catalog", "list_contracts", "get_contract"]
