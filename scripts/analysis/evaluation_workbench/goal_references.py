"""Source-bound mechanical chart references; never rescore a recorded trial.

This module imports no simulator and never executes archived Python.  A chart
reference is emitted only after the trial's frozen deployment, source inventory,
and the reviewed configuration/measurement bytes have all been verified.  A
current checkout may be supplied as a *copy* of those bytes; its current task
defaults are never used as a fallback for missing historical evidence.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path, PurePosixPath

SCHEMA_VERSION = "evaluation-goal-references-v1"
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_LIMIT = 4 * 1024 * 1024
_TASK_FILES = {
    "bulb-screw": ("bulb_screw/config.py", "bulb_screw/task.py"),
    "bulb-unscrew": ("bulb_screw/config.py", "bulb_screw/task.py",
                     "bulb_unscrew/config.py", "bulb_unscrew/task.py"),
    "bulb-screw-inplace": ("bulb_screw/config.py", "bulb_screw/task.py",
                          "bulb_screw_inplace/config.py", "bulb_screw_inplace/task.py"),
    "install-ram": ("install_ram/config.py", "install_ram/task.py"),
    "poker-draw": ("poker_draw/rollout_eval.py", "poker_draw/review_metrics.py"),
    "usb-insert": ("usb_insert/config.py", "usb_insert/task.py", "usb_insert/review_metrics.py"),
    "usb-unplug": ("usb_unplug/config.py", "usb_unplug/task.py"),
    "whiteboard-wipe": ("shared/policy_tasks.py", "whiteboard_wipe/task.py", "whiteboard_wipe/cleaning.py"),
    "whiteboard-wipe-horizontal": ("shared/policy_tasks.py", "whiteboard_wipe/task.py",
                                   "whiteboard_wipe/cleaning.py", "whiteboard_wipe_horizontal/config.py",
                                   "whiteboard_wipe_horizontal/task.py"),
    "whiteboard-wipe-horizontal-line": ("shared/policy_tasks.py", "whiteboard_wipe/task.py",
                                        "whiteboard_wipe/cleaning.py", "whiteboard_wipe_horizontal/config.py",
                                        "whiteboard_wipe_horizontal/task.py", "whiteboard_wipe_horizontal_line/config.py",
                                        "whiteboard_wipe_horizontal_line/task.py"),
}
_TASKS = set(_TASK_FILES)
_BASE = "src/kaihand_tactile_env/tasks/"
# These bytes were reviewed for the meaning and units of the plotted fields,
# not merely for their numerical constants.  Unknown future mechanics fail
# closed until their measurement semantics have been reviewed as well.
_REVIEWED = {
    "bulb_screw/config.py": "4033b941c798ce6dd31df20109338fafc384140ff0db8d9c8874672ca2914b42",
    "bulb_screw/task.py": "585426d36475d583fdb43a2b48b36ab1c2381d6e32459e42285421e6869be94a",
    "bulb_unscrew/config.py": "4c466351b4b4781e16d95c3b8f304db87e502b60556c8cdd636a92959320b880",
    "bulb_unscrew/task.py": "c5349e3a767f358c8a64ecd42cc3a02ffddc4a3ace1ce65c2c559d05000ae6e9",
    "bulb_screw_inplace/config.py": "8d478c11479e83e0553fbc2dd1ac6b1368b98a9a2df2b69b925a84dabeeeb8db",
    "bulb_screw_inplace/task.py": "0d253992e8d3fcbf09e347dd5c01788603e4286d54d5e3800added8103e33265",
    "install_ram/config.py": "b46f59f9c3f127d18dbc09ef9ac1213455f01076231cf086e1274faa17756b1b",
    "install_ram/task.py": "6092f5918fd3cba9391b46c43590f646107d66a8168a9bffd723ac6eddc95766",
    "poker_draw/rollout_eval.py": "ff399929bdb09d1beb61b1a32635b0d4b81b2f5958134231ebac47420b18d2ba",
    "poker_draw/review_metrics.py": "ea8eea1eac83f6e81ce9f17627f8f04be422dbf2cec4528dd5a790dc64b5ac1b",
    "usb_insert/config.py": "9017d39006d0a76b692036aaed1708802134eb7d48e5cee8ddb3da13b0aa8b12",
    "usb_insert/task.py": "fb89bb3e3bfd37902767acd5abe2cee2f5a2cbf4a3f37dbb3bfb89879396772e",
    "usb_insert/review_metrics.py": "95262df2d3306880d6157f0b8ab982edac01178fb56dae460f2a5b2d4286b37c",
    "usb_unplug/config.py": "374ac5656a1c0980ecc2e793f82a09fd2dce2f3a61e214ce88c53bd56cfcfc69",
    "usb_unplug/task.py": "a95d39ab0295187cd7ce19504138a99ba1cf633910ddd87c429f022de52b0f83",
    "shared/policy_tasks.py": "2e9da89f76ec064f30d09ec787efb95c832ca2e81ec2ee2343684640d33b611a",
    "whiteboard_wipe/task.py": "3c560b4bc954302a498fb05912648be6275d8a83a214fe53562018a8a3eb4728",
    "whiteboard_wipe/cleaning.py": "4c0429643f8890eb551a2a8ecb3ad3856f1a5231ad92a482f3f67e6ea8a2a117",
    "whiteboard_wipe_horizontal/config.py": "0c84d542a3c7c575f7a0f5208fcdb7af692fde78a0d72c0965ad97a1192f3b49",
    "whiteboard_wipe_horizontal/task.py": "fa7d3d4132ba5c8dd3fdbc39ef363ac07f177efe35bfc38b8776de6338950573",
    "whiteboard_wipe_horizontal_line/config.py": "af7ef79737cfe78c8d1b21a72c44bca5b3eb5162c84716e38c40c5d5ba72a6b8",
    "whiteboard_wipe_horizontal_line/task.py": "083abb13892a71483f62463a26188d64e26e447d5154c13664354df40bcc505e",
}
# The archived policy adapter and today's adapter differ only by the additive
# bulb-screw-inplace dispatch or the USB held-clear completion protocol.  All
# three whiteboard measurement/success methods were compared as complete ASTs
# against the frozen policy adapter and are identical.  Both source versions are
# reviewed byte-for-byte; neither is a fallback for an unknown source revision.
_REVIEWED_ALTERNATES = {
    "shared/policy_tasks.py": {"05a5b9efd35808326329903bd2d5dda5bfba704cf6fbeffed5dfc91f2dc284f7",
                               "d0e34063b9f49df30fca712faac6ba89dd5a8fea69298280f5a817c1e24c1867",
                               # Adds only bulb display-angle aliases to snapshot;
                               # whiteboard measurement and success ASTs are unchanged.
                               "d794601b1c5589e766f49c2412a6b1c3c0645d96202b55e859ade2dc1313a2f1",
                               # Adds the separate USB-insert policy adapter and
                               # bulb angle aliases. Whiteboard state/pickup/
                               # release ASTs and whiteboard branches match the
                               # reviewed d0e340... revision exactly.
                               "faaa4410cf411bebebddac7de1f909369a03b4c51b41803115ab0616a93421e0"},
    # The original strict return protocol remains reviewed for old archives.
    # The new held-clear protocol is selected only from its bound config AST.
    "usb_unplug/config.py": {"05692c0debad1f339936690ad8dd4c7aceb49778b773e22f540eb1b3f9be26ee"},
    "usb_unplug/task.py": {"b1755b37c9c7d8d24b14245c5b40c8f91bdf6de2b60c1ddb6c80bdf8d8c3efc0"},
}
_USB_UNPLUG_REVIEWED_PAIRS = {
    (_REVIEWED["usb_unplug/config.py"], _REVIEWED["usb_unplug/task.py"]),
    ("05692c0debad1f339936690ad8dd4c7aceb49778b773e22f540eb1b3f9be26ee",
     "b1755b37c9c7d8d24b14245c5b40c8f91bdf6de2b60c1ddb6c80bdf8d8c3efc0"),
}
_DISCLAIMER = "机械位置参考不等于完整任务成功；不重算或替换记录的原生结果、阶段和首次达成时间。"


class _Unavailable(ValueError):
    pass


def _source_relative(name):
    return "src/kaihand_tactile_env/" + name if name.startswith("shared/") else _BASE + name


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _sha(value):
    return isinstance(value, str) and _SHA.fullmatch(value) is not None


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise _Unavailable(f"Duplicate JSON key in bound evidence: {key}")
        result[key] = value
    return result


def _json(raw):
    def invalid(value):
        raise _Unavailable(f"Nonfinite JSON number in bound evidence: {value}")

    result = json.loads(raw, object_pairs_hook=_unique_pairs, parse_constant=invalid)
    if not isinstance(result, dict):
        raise _Unavailable("Bound JSON evidence must be an object")
    return result


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _read_bytes(path, expected):
    """Bound, regular, non-symlink leaf; parent storage aliases are permitted."""
    if not _sha(expected.get("sha256")) or type(expected.get("size")) is not int:
        raise _Unavailable("Evidence requires an explicit SHA256 and byte size")
    size = expected["size"]
    if not 0 < size <= _LIMIT:
        raise _Unavailable("Bound goal-reference evidence exceeds the byte limit")
    path = Path(path).absolute()
    if path.is_symlink():
        raise _Unavailable(f"Evidence leaf must not be a symlink: {path}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                         | getattr(os, "O_NONBLOCK", 0))
    with os.fdopen(descriptor, "rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size != size:
            raise _Unavailable(f"Evidence size/type does not match recorded inventory: {path}")
        raw = handle.read(_LIMIT + 1)
        after = os.fstat(handle.fileno())
        if (_identity(before) != _identity(after)
                or _identity(after) != _identity(path.stat())
                or len(raw) != size):
            raise _Unavailable(f"Evidence changed while reading: {path}")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected["sha256"]:
        raise _Unavailable(f"Evidence SHA256 does not match frozen inventory: {path}")
    return raw, {"path": str(path.resolve()), "size": size, "sha256": digest}


def _read_bound_row(row, role, provenance):
    failures = []
    candidates = []
    for key in ("resolved_path", "canonical_path", "path"):
        value = row.get(key)
        if isinstance(value, str) and value and value not in candidates:
            candidates.append(value)
    for value in candidates:
        try:
            raw, evidence = _read_bytes(value, row)
        except (OSError, _Unavailable) as error:
            failures.append(str(error))
            continue
        provenance.append({**evidence, "role": role,
                           "recorded_path": row.get("path")})
        return _json(raw)
    raise _Unavailable("No readable SHA-bound evidence copy: " + "; ".join(failures))


def _one_evidence(report, roles):
    rows = [row for row in report.get("provenance", [])
            if isinstance(row, dict) and row.get("role") in roles]
    if not rows or len({row.get("sha256") for row in rows}) != 1:
        raise _Unavailable("Missing or conflicting bound evidence: " + ", ".join(sorted(roles)))
    return rows[0]


def _source_inventory(manifest):
    rows = manifest.get("evaluation_source_files")
    if not isinstance(rows, list) or not rows:
        raise _Unavailable("Frozen deployment has no evaluation source inventory")
    if hashlib.sha256(_canonical(rows)).hexdigest() != manifest.get("evaluation_source_sha256"):
        raise _Unavailable("Frozen evaluation source inventory SHA256 mismatch")
    inventory = {}
    for row in rows:
        if not isinstance(row, dict):
            raise _Unavailable("Invalid frozen source inventory row")
        value = row.get("path")
        if not isinstance(value, str) or not value or "\\" in value:
            raise _Unavailable("Unsafe frozen source path")
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or str(path) != value:
            raise _Unavailable("Unsafe or noncanonical frozen source path")
        if (value in inventory or not _sha(row.get("sha256"))
                or type(row.get("size")) is not int or row["size"] < 0):
            raise _Unavailable("Duplicate or invalid frozen source identity")
        inventory[value] = row
    return inventory


def _check_identity(task, report, summary, manifest, inventory_sha):
    trial = report.get("trial", {})
    if trial.get("task") != task or summary.get("task") != task or manifest.get("task") != task:
        raise _Unavailable("Goal reference task identity does not match its trial and deployment")
    for key in ("checkpoint_sha256", "deployment_id", "prediction_horizon"):
        if summary.get(key) is None or summary.get(key) != manifest.get(key):
            raise _Unavailable(f"Summary/deployment identity mismatch: {key}")
    for key in ("seed", "prediction_horizon"):
        if trial.get(key) is not None and trial[key] != summary.get(key):
            raise _Unavailable(f"Report/summary identity mismatch: {key}")
    checkpoint = trial.get("cohort", {}).get("checkpoint")
    if checkpoint is not None and checkpoint != manifest.get("checkpoint_sha256"):
        raise _Unavailable("Report/deployment checkpoint identity mismatch")
    rule = report.get("recorded_staged_evaluation", {}).get("rule_identity", {})
    if rule.get("evaluation_source_sha256") not in (None, inventory_sha):
        raise _Unavailable("Report/deployment frozen source identity mismatch")
    integrity = summary.get("deployment_integrity", {})
    source = integrity.get("evaluation_source", {})
    if (integrity.get("status") != "verified" or integrity.get("errors")
            or source.get("status") != "verified" or source.get("errors")
            or source.get("changed_files") or source.get("error")
            or source.get("expected_sha256") != inventory_sha
            or source.get("actual_sha256") != inventory_sha):
        raise _Unavailable("Recorded deployment source verification is absent or contradictory")


class _NumericConfig:
    """Evaluate only constants, names and elementary arithmetic from reviewed AST."""

    def __init__(self, tree, inherited=None):
        self.definitions = {}
        self.inherited = inherited
        self.active = set()
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target, value = node.targets[0], node.value
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                target, value = node.target, node.value
            else:
                continue
            if isinstance(target, ast.Name):
                if target.id in self.definitions:
                    raise _Unavailable("Ambiguous numeric configuration assignment")
                self.definitions[target.id] = value

    def number(self, name):
        if name in self.active or name not in self.definitions:
            raise _Unavailable(f"Missing or circular archived numeric constant: {name}")
        self.active.add(name)
        try:
            value = self.evaluate(self.definitions[name])
        finally:
            self.active.remove(name)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise _Unavailable(f"Nonfinite archived numeric constant: {name}")
        return float(value)

    def evaluate(self, node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.Name):
            return self.number(node.id)
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id == "screw" and self.inherited is not None):
            return self.inherited.number(node.attr)
        if isinstance(node, ast.UnaryOp):
            value = self.evaluate(node.operand)
            if isinstance(node.op, ast.USub):
                return -value
            if isinstance(node.op, ast.UAdd):
                return value
        if isinstance(node, ast.BinOp):
            left, right = self.evaluate(node.left), self.evaluate(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div) and right != 0:
                return left / right
        raise _Unavailable("Archived goal expression is outside the static numeric subset")


def _turn_tolerance(tree):
    matches = []
    expected = ast.dump(ast.parse("abs(turns - config.TARGET_TURNS)", mode="eval").body)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Compare) and len(node.ops) == 1
                and isinstance(node.ops[0], ast.LtE) and len(node.comparators) == 1
                and ast.dump(node.left) == expected):
            value = node.comparators[0]
            if isinstance(value, ast.Constant) and type(value.value) in (int, float):
                matches.append(float(value.value))
    if len(matches) != 1 or not math.isfinite(matches[0]) or matches[0] <= 0:
        raise _Unavailable("Archived seated-turn tolerance is not uniquely statically verified")
    return matches[0]


def _ram_orientation_tolerance(tree):
    """Read both measure/update gates; never call archived NumPy or Python."""
    expressions = {ast.dump(ast.parse(value, mode="eval").body)
                   for value in ("angle", "state.orientation_error_rad")}
    matches = {}
    for node in ast.walk(tree):
        if (not isinstance(node, ast.Compare) or len(node.ops) != 1
                or not isinstance(node.ops[0], ast.LtE) or len(node.comparators) != 1):
            continue
        left = ast.dump(node.left)
        if left not in expressions:
            continue
        call = node.comparators[0]
        if (not isinstance(call, ast.Call) or call.keywords or len(call.args) != 1
                or ast.dump(call.func) != ast.dump(ast.parse("np.deg2rad", mode="eval").body)
                or left in matches):
            raise _Unavailable("Archived RAM orientation gates are not uniquely statically verified")
        degrees = _NumericConfig(ast.Module(body=[], type_ignores=[])).evaluate(call.args[0])
        if type(degrees) not in (int, float) or not math.isfinite(degrees) or degrees <= 0:
            raise _Unavailable("Archived RAM orientation tolerance has unsupported value")
        matches[left] = float(degrees) * math.pi / 180.0
    if set(matches) != expressions or len(set(matches.values())) != 1:
        raise _Unavailable("Archived RAM measure/update orientation tolerances do not agree")
    return next(iter(matches.values()))


def _poker_displacement_threshold(trees):
    """Bind the displayed review threshold to the native outcome edge gate."""
    review_classes = [node for node in trees["poker_draw/review_metrics.py"].body
                      if isinstance(node, ast.ClassDef) and node.name == "PokerReviewMetrics"]
    monitor_classes = [node for node in trees["poker_draw/rollout_eval.py"].body
                       if isinstance(node, ast.ClassDef) and node.name == "PokerOutcomeMonitor"]
    if len(review_classes) != 1 or len(monitor_classes) != 1:
        raise _Unavailable("Archived poker measurement/monitor classes are not unique")
    review = _NumericConfig(ast.Module(body=review_classes[0].body, type_ignores=[]))
    threshold = review.number("EDGE_THRESHOLD_M")
    methods = [node for node in monitor_classes[0].body
               if isinstance(node, ast.FunctionDef) and node.name == "observe"]
    if len(methods) != 1:
        raise _Unavailable("Archived poker outcome observation method is not unique")
    gates = []
    for node in ast.walk(methods[0]):
        if (isinstance(node, ast.Compare) and isinstance(node.left, ast.Name)
                and node.left.id == "displacement" and len(node.ops) == 1
                and isinstance(node.ops[0], ast.GtE) and len(node.comparators) == 1):
            gates.append(review.evaluate(node.comparators[0]))
    if len(gates) != 1 or not math.isfinite(threshold) or threshold <= 0 or gates[0] != threshold:
        raise _Unavailable("Archived poker review/native displacement thresholds do not agree")
    return threshold


def _function(tree, name):
    methods = [node for node in ast.walk(tree)
               if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(methods) != 1:
        raise _Unavailable(f"Archived measurement function is not unique: {name}")
    return methods[0]


def _usb_orientation_tolerance(tree):
    method = _function(tree, "_near_bottom")
    expected = ast.dump(ast.parse("state.orientation_error_rad", mode="eval").body)
    matches = []
    for node in ast.walk(method):
        if (not isinstance(node, ast.Compare) or len(node.ops) != 1
                or not isinstance(node.ops[0], ast.LtE) or len(node.comparators) != 1
                or ast.dump(node.left) != expected):
            continue
        call = node.comparators[0]
        if (not isinstance(call, ast.Call) or call.keywords or len(call.args) != 1
                or ast.dump(call.func) != ast.dump(ast.parse("np.deg2rad", mode="eval").body)):
            raise _Unavailable("Archived USB seating orientation gate is unsupported")
        matches.append(_NumericConfig(ast.Module(body=[], type_ignores=[])).evaluate(call.args[0]))
    if (len(matches) != 1 or not math.isfinite(matches[0]) or matches[0] <= 0):
        raise _Unavailable("Archived USB seating orientation gate is not unique")
    return float(matches[0])


def _whiteboard_ink_tolerance(tree):
    method = _function(tree, "_whiteboard_state")
    matches = []
    means = []
    expected_mean = ast.dump(ast.parse("float(np.mean(1.0 - remaining))", mode="eval").body)
    for node in ast.walk(method):
        if (isinstance(node, ast.Compare) and isinstance(node.left, ast.Name)
                and node.left.id == "maximum_remaining" and len(node.ops) == 1
                and isinstance(node.ops[0], ast.LtE) and len(node.comparators) == 1):
            matches.append(_NumericConfig(ast.Module(body=[], type_ignores=[])).evaluate(node.comparators[0]))
        if isinstance(node, ast.keyword) and node.arg == "cleaned_fraction":
            means.append(ast.dump(node.value))
    if (len(matches) != 1 or not math.isfinite(matches[0]) or not 0 < matches[0] < 1
            or means != [expected_mean]):
        raise _Unavailable("Archived whiteboard worst-cell cleaning gate or average semantics is unsupported")
    return float(matches[0])


def _references(task, trees):
    if task == "usb-insert":
        config = _NumericConfig(trees["usb_insert/config.py"])
        depth = config.number("BACKSTOP_DEPTH_M") * 1000.0
        lower = depth - config.number("SEATED_DEPTH_TOLERANCE_M") * 1000.0
        upper = depth + config.number("MAX_SOCKET_PENETRATION_M") * 1000.0
        angle = _usb_orientation_tolerance(trees["usb_insert/task.py"])
        if not 0 < lower < depth < upper:
            raise _Unavailable("Archived USB backstop references have unsupported signs")
        return [
            {"key": "insertion_depth_mm", "unit": "mm", "target": depth,
             "relation": "target", "lower": lower, "upper": upper,
             "label": f"USB 机械背挡深度：{depth:g} mm，近底允许 {lower:g}–{upper:g} mm",
             "note": "只在插壳能通过开口且姿态误差 ≤5°时记录深度，否则图中记 0（不代表真实轴位为 0）。"
                     "机械近底另要求姿态 ≤3°、穿透受限；入带不等于测得真实压底、持续稳定就位或完整成功。"},
            {"key": "orientation_error_deg", "unit": "deg", "target": angle,
             "relation": "at_most", "lower": 0.0, "upper": angle,
             "label": f"USB 机械近底姿态误差上限：≤ {angle:g}°；理想 0°",
             "note": "插头相对插座的三维朝向误差。0° 至上限是原生近底几何允许区，"
                     "不是图中深度记录的 ≤5° 对准门槛；入区不等于插入、真实压底、稳定保持或完整成功。"},
        ]
    if task == "usb-unplug":
        config = _NumericConfig(trees["usb_unplug/config.py"])
        depth = config.number("FULL_CLEAR_DEPTH_M")
        protocol = config.definitions.get("COMPLETION_PROTOCOL")
        if protocol is not None:
            if not (isinstance(protocol, ast.Constant)
                    and protocol.value == "usb-unplug-held-clear-2s-v2"):
                raise _Unavailable("Archived USB withdrawal completion protocol is unsupported")
            hold = config.number("EXTRACTION_HOLD_DWELL_S")
            if depth >= 0 or hold != 2.0:
                raise _Unavailable("Archived USB hand-held completion thresholds are unsupported")
            return [
                {"key": "insertion_depth_m", "unit": "m", "target": depth,
                 "relation": "at_most",
                 "label": f"USB 完全清离轴向阈值：≤ {depth * 1000:g} mm",
                 "note": f"本条归档使用拔出后手持协议：轴位 ≤ {depth * 1000:g} mm、无插座接触、右手仍接触且无桌面接触，"
                         f"同时连续至少 {hold:g} s；不要求放回、释放、退手或固定速度。单独越过轴位线不等于完成保持。"},
                {"key": "extraction_hold_duration_s", "unit": "s", "target": hold,
                 "relation": "at_least",
                 "label": f"USB 完全拔出后连续手持目标：≥ {hold:g} s",
                 "note": "这是当前连续合格窗口；插回、插座接触、失去右手接触或接触桌面都会重新计时。"
                         "不是累计时长或历史最大值；原任务结果及阶段曾达到标志分别保留。"},
            ]
        dwell = config.number("FULL_CLEAR_DWELL_S")
        if depth >= 0 or dwell <= 0:
            raise _Unavailable("Archived USB withdrawal references have unsupported signs")
        return [{"key": "insertion_depth_m", "unit": "m", "target": depth,
                 "relation": "at_most",
                 "label": f"USB 完全清离轴向阈值：≤ {depth * 1000:g} mm",
                 "note": f"相对插座入口的有符号轴位；负值代表插头尖端已退到入口外。还须无插座接触并持续 {dwell:g} s。"
                         "这是拔出阶段条件，不是放回桌面的位置终点；越线不等于抓持有效、已放回、已释放或完整成功。"}]
    if task.startswith("whiteboard-wipe"):
        ink = _whiteboard_ink_tolerance(trees["shared/policy_tasks.py"])
        return [
            {"key": "cleaned_fraction", "unit": "fraction", "target": 1.0,
             "relation": "target", "reference_kind": "ideal", "label": "墨迹单元平均清洁进度理想参考：100%",
             "note": "本曲线为各墨迹单元 (1−余量) 的平均值，不是实际覆盖面积。100% 是理想终点参考；"
                     "原生擦净须所有单元都达标（最大单元余量 ≤1e-9），平均值靠近参考不能替代该条件。"
                     "还需拾取、放回释放等条件；这不是 staged 拾取阶段阈值，达到参考不等于完整成功。"},
            {"key": "maximum_ink_remaining", "unit": "fraction", "target": ink,
             "relation": "at_most", "lower": 0.0, "upper": ink,
             "label": f"原生擦净最差单元余量上限：≤ {ink:g}",
             "note": "逐条记录的最大单元墨迹余量，低于上限才满足所有单元擦净这一原生条件；不是平均余量。"
                     "仍需拾取、放回释放（水平类还需回原位）等条件；"
                     "本批 staged 只定义拾取阶段，不能将此线作为该阶段或完整 staged 成功线；越线不等于完整成功。"},
        ]
    if task == "install-ram":
        config = _NumericConfig(trees["install_ram/config.py"])
        depth = config.number("TARGET_INSERTION_DEPTH_M")
        tolerance = config.number("SEATED_DEPTH_TOLERANCE_M")
        angle = _ram_orientation_tolerance(trees["install_ram/task.py"])
        if depth <= 0 or tolerance <= 0 or tolerance >= depth:
            raise _Unavailable("Archived RAM insertion references have unsupported signs")
        return [
            {"key": "insertion_depth_m", "unit": "m", "target": depth,
             "relation": "target", "lower": depth - tolerance, "upper": depth + tolerance,
             "label": f"内存条机械就位深度：{depth * 1000:g} mm，允许 ±{tolerance * 1000:g} mm",
             "note": "相对插槽入口的有符号几何深度，不是越深越好；"
                     "未对准时也只是几何记录。入带不等于真实压底、稳定保持或完整任务成功。"},
            {"key": "orientation_error_rad", "unit": "rad", "target": angle,
             "relation": "at_most", "lower": 0.0, "upper": angle,
             "label": f"内存条—插槽三维姿态误差上限：≤ {angle * 180 / math.pi:g}°；理想 0°",
             "note": "插槽坐标系与内存条完整三维朝向的误差角，允许区为 0° 至上限，"
                     "不是某个轴的误差或关于上限的双侧容差；入区不等于真实压底、稳定保持或完整成功。"},
        ]
    if task == "poker-draw":
        threshold = _poker_displacement_threshold(trees) * 1000.0
        return [
            {"key": "card_displacement_now_mm", "unit": "mm", "target": threshold,
             "relation": "at_least",
             "label": f"模型原生到桌边阶段位移下限：≥ {threshold:g} mm",
             "note": "相对本轨迹初始位置的当前净位移（initial_x − current_x），朝机器人为正，"
                     "不是累计路程或最终持卡位置。还需此前已有指腹接触，且该物理时刻台面支撑、平放及至少 40% 悬边均满足；"
                     "这是原生到桌边条件，不是 staged Stage 1 阈值；不等于到边阶段已通过，更不等于提起、朝向保持或完整成功。"},
            {"key": "card_displacement_peak_mm", "unit": "mm", "target": threshold,
             "relation": "at_least",
             "label": f"历史最大位移的原生到桌边下限参考：≥ {threshold:g} mm",
             "note": "逐物理步记录的历史最大净位移，越线只证明曾达到该位移，之后回退曲线仍保持最大值。"
                     "不能证明当前仍到位，也不能证明当时同时满足支撑、平放和至少 40% 悬边；"
                     "该线不是 staged Stage 1 阈值，不等于阶段或完整任务成功。"},
        ]
    base = _NumericConfig(trees["bulb_screw/config.py"])
    if task == "bulb-unscrew":
        reverse = _NumericConfig(trees["bulb_unscrew/config.py"], base)
        turns = reverse.number("EXPECTED_RELEASE_TURNS")
        depth = reverse.number("FULL_CLEAR_DEPTH_M")
        if turns <= 0 or depth >= 0:
            raise _Unavailable("Archived removal references have unsupported signs")
        angle = turns * 360.0
        return [
            {"key": "angle_removed_deg", "unit": "deg", "target": angle,
             "relation": "target",
             "label": f"名义螺纹脱扣角度：{angle:g}°",
             "note": "从安装位计算的名义净旋出角度，不是拧动次数或新的物理判据。最低有效脱扣角容许量不是终点或对称容差带；"
                     "必须另查实际脱扣、完全取出、放回及释放。到线不等于完整成功。"},
            {"key": "insertion_depth_m", "unit": "m", "target": depth,
             "relation": "at_most",
             "label": f"完全离座轴向阈值：≤ {depth * 1000:g} mm",
             "note": "灯泡参考点高于灯座入口的单侧清离阈值，还需实际脱扣、无灯座接触和低接触载荷。"
                     "这是取出阶段条件，不是放回桌面后的轴向终点，也不等于完整成功。"},
        ]
    turns = base.number("TARGET_TURNS")
    depth = base.number("SEATED_DEPTH_M")
    depth_tolerance = base.number("SEATED_DEPTH_TOLERANCE_M")
    turns_tolerance = _turn_tolerance(trees["bulb_screw/task.py"])
    if turns <= 0 or depth <= 0 or depth_tolerance <= 0:
        raise _Unavailable("Archived seating references have unsupported signs")
    angle = turns * 360.0
    angle_tolerance = turns_tolerance * 360.0
    return [
        {"key": "clockwise_angle_deg", "unit": "deg", "target": angle,
         "relation": "target", "lower": angle - angle_tolerance,
         "upper": angle + angle_tolerance,
         "label": f"名义机械到座旋入角度：{angle:g}°",
         "note": f"螺纹接合后的净顺时针旋入角度；原机械到座角度条件换算为 ±{angle_tolerance:g}°。"
                 "容差带只是多项到座条件之一；还需对准、实际承力、拧紧确认和松手稳定，入带不等于完整成功。"},
        {"key": "insertion_depth_m", "unit": "m", "target": depth,
         "relation": "target", "lower": depth - depth_tolerance,
         "upper": depth + depth_tolerance,
         "label": f"名义机械到座轴向位置：{depth * 1000:.3f} mm",
         "note": f"灯座入口高度减灯泡参考点高度，原到座轴位条件 ±{depth_tolerance * 1000:g} mm；"
                 "不是从入牙开始的新增行程，未对准时也只是几何轴位。入带不等于啮合、拧紧或完整成功。"},
    ]


def build_goal_references(task: str, report: dict, source_root: Path | None = None) -> dict:
    """Return verified mechanical target lines, or an explicit unavailable state.

    ``report`` must carry SHA/size-bound summary and deployment-manifest
    provenance from the same trial.  Unless ``source_root`` explicitly supplies
    another byte-identical copy, the archived root declared in that verified
    summary is read.  Missing sources, changed bytes, unreviewed mechanics, or
    contradictory identities produce **no** reference values.
    """
    result = {"schema_version": SCHEMA_VERSION, "task": task,
              "status": "not_applicable", "references": [], "warnings": [], "provenance": []}
    if task not in _TASKS:
        return result
    result["status"] = "unavailable"
    try:
        if not isinstance(report, dict):
            raise _Unavailable("Goal references require a bound trial report")
        provenance = result["provenance"]
        manifest_roles = {"staged_deployment_manifest", "deployment_manifest"}
        if task in {"poker-draw", "usb-insert", "whiteboard-wipe-horizontal-line"}:
            # Some historical exports retain this verified manifest under its
            # crop-audit role.  The summary's actual/expected manifest SHA and
            # all task/checkpoint/deployment/source identity checks still bind it.
            manifest_roles.add("model_crop_deployment_manifest")
        manifest_row = _one_evidence(report, manifest_roles)
        manifest = _read_bound_row(manifest_row, "goal_reference_deployment_manifest", provenance)
        summary_row = _one_evidence(report, {"prediction_horizon_summary", "staged_summary"})
        summary = _read_bound_row(summary_row, "goal_reference_trial_summary", provenance)
        inventory = _source_inventory(manifest)
        inventory_sha = manifest["evaluation_source_sha256"]
        _check_identity(task, report, summary, manifest, inventory_sha)
        manifest_integrity = summary["deployment_integrity"].get("manifest", {})
        if (not isinstance(manifest_integrity, dict)
                or manifest_integrity.get("status") not in (None, "verified")
                or manifest_integrity.get("error") or manifest_integrity.get("errors")
                or manifest_integrity.get("changed_files")):
            raise _Unavailable("Recorded deployment manifest verification is contradictory")
        declared_hashes = [value for value in (
            summary["deployment_integrity"].get("manifest_sha256"),
            manifest_integrity.get("expected_sha256"), manifest_integrity.get("actual_sha256"))
            if value is not None]
        if not declared_hashes or any(value != manifest_row["sha256"] for value in declared_hashes):
            raise _Unavailable("Summary does not bind the verified deployment manifest SHA256")
        declared_root = summary["deployment_integrity"]["evaluation_source"].get("root")
        chosen_root = source_root if source_root is not None else declared_root
        if chosen_root is None or not Path(chosen_root).is_absolute():
            raise _Unavailable("No explicit archived source root; current checkout is not a fallback")
        root = Path(chosen_root).resolve(strict=True)
        if not root.is_dir():
            raise _Unavailable("Archived evaluation source root is not a directory")
        if task == "usb-unplug":
            pair = tuple(inventory.get(_source_relative(name), {}).get("sha256")
                         for name in _TASK_FILES[task])
            if pair not in _USB_UNPLUG_REVIEWED_PAIRS:
                raise _Unavailable("Archived USB withdrawal config/task source pair has not been reviewed")
        trees = {}
        for name in _TASK_FILES[task]:
            relative = _source_relative(name)
            row = inventory.get(relative)
            if row is None:
                raise _Unavailable(f"Frozen deployment omits required source: {relative}")
            if row["sha256"] not in {_REVIEWED[name], *_REVIEWED_ALTERNATES.get(name, ())}:
                raise _Unavailable(f"Archived mechanics/measurement version has not been reviewed: {relative}")
            path = root / relative
            if not path.resolve().is_relative_to(root):
                raise _Unavailable("Archived source path escapes its declared root")
            raw, evidence = _read_bytes(path, row)
            provenance.append({**evidence, "role": "goal_reference_reviewed_source",
                               "inventory_path": relative})
            trees[name] = ast.parse(raw, filename=str(path))
        references = _references(task, trees)
        result.update(status="verified", references=references, warnings=[_DISCLAIMER],
                      binding={"evaluation_source_sha256": inventory_sha,
                               "source_root": str(root),
                               "source_root_selection": "explicit_inventory_verified_copy" if source_root is not None
                               else "sha_bound_summary_archived_root",
                               "source_code_executed": False, "rules_reexecuted": False})
        if task == "bulb-screw-inplace":
            result["warnings"].append("相对实际初态的净旋入角度未添加固定终点；该量需另扣除本轨迹的实际初始角度。")
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, AttributeError, RecursionError) as error:
        result.update(status="unavailable", references=[],
                      warnings=["未显示机械目标参考：" + str(error), _DISCLAIMER])
    return result
