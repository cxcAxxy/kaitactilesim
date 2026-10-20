"""Audited, read-only expert video bindings for offline multi-task workbenches.

No simulator, model, HDF reader, GPU, or rendering code is imported or invoked.
The registry is an explicit video/replay/frame SHA audit, not filename matching.
Source HDF hashes are declarations in audited metadata, never a claim that this
module rehashed the large HDF files.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import stat
import subprocess

_SCHEMA = "kaihand-offline-multimodal-replay-v1"
_SHA = re.compile(r"[0-9a-f]{64}")
# Audited 2026-10-14: all media metadata was inspected with ffprobe; frame rows
# are contiguous and use their own playback seconds. Publication records, when
# present, additionally bind the current version rather than an archived demo.
_REGISTRY_ROWS = json.loads(r'''[
  {
    "task": "bulb-screw",
    "video": "bulb-screw.mp4",
    "replay": "_metadata/bulb-screw/review_1012_200/replay.json",
    "frames": "_metadata/bulb-screw/review_1012_200/frames.jsonl",
    "video_sha256": "67c7d06dd60a1dcf23de044f967e861ec533e8c3b8f17323726744d736827b90",
    "replay_sha256": "49243eed7a193d1a5a3634e5579e30cdc17c73998e6127de74bc7aab2b50ee1d",
    "frames_sha256": "cbe36a274a910de540d5120f537e9285803117e28f0de855537c6f1160cb2140",
    "source_hdf5": "/nas/chenxianchi/datasets/sim/bulb-screw/raw/1012_v15_200_r2/bulb-screw/000000/attempt_001/data/raw/light_bulb_000000.h5",
    "source_sha256": "eaee9f2cbf3b43e5edc5713c6fca73fb15cdcf850efb43472c56395214813cb4",
    "fps": 30,
    "frame_count": 1273,
    "publication": "_metadata/bulb-screw/publication.json"
  },
  {
    "task": "bulb-unscrew",
    "video": "bulb-unscrew-tabletop.mp4",
    "replay": "_metadata/bulb-unscrew-tabletop/review/replay.json",
    "frames": "_metadata/bulb-unscrew-tabletop/review/frames.jsonl",
    "video_sha256": "17e0867707d807d05d3eb924610a2e67e273e8e822195c46486420923f21f12c",
    "replay_sha256": "c47ccdf021a567fd4ff161934db8eec0271a8594b3405c46b3ceb4934de20948",
    "frames_sha256": "e736d376760b411b92f3bc1813e8a573dd6e31ceb56113a181070f8d92bcef38",
    "source_hdf5": "/nas/chenxianchi/datasets/sim/task_demo/_metadata/bulb-unscrew-tabletop/collection/raw/episode.h5",
    "source_sha256": "c9ef6512ab8ffae9ec6c38f84ccb59c04f42f1a1a2af03cf5f61995b3d22eb21",
    "fps": 30,
    "frame_count": 1150,
    "publication": ""
  },
  {
    "task": "install-ram",
    "video": "install-ram.mp4",
    "replay": "_metadata/install-ram/replay.json",
    "frames": "_metadata/install-ram/frames.jsonl",
    "video_sha256": "26284b014ce0193e4547029d0e246cf878a2b83dd5c16f41698e708c3e2b5f88",
    "replay_sha256": "d31cece168edb9a274be9c9aef327ea62ffa583d411ef2bd3c676ef33bea1fc3",
    "frames_sha256": "e679f8f71f82545f8fb6f1af89d88ef833a253b30c33725c4332a9d14f76c02b",
    "source_hdf5": "/nas/chenxianchi/datasets/sim/install-ram/raw/0920_200/install-ram/000000/attempt_001/data/raw/install_ram_000000.h5",
    "source_sha256": "09635730c9974d48378411663b1e6484ea5fac9b57c4a9779b1f00dd7b050de4",
    "fps": 10,
    "frame_count": 429,
    "publication": ""
  },
  {
    "task": "pick-place",
    "video": "pick-place.mp4",
    "replay": "_metadata/pick-place/replay.json",
    "frames": "_metadata/pick-place/frames.jsonl",
    "video_sha256": "e98c93bd68ee3ba17db58dbf66a858683e6000fbca9436ba1a841ff7b0ebc335",
    "replay_sha256": "eb404a3d7e786c26ea8de5ee33540011af91167bb78086daf5b070ef1b9afd4b",
    "frames_sha256": "9a9326b1db7dbbb2ddf76e0466a93cb6b45595ec1b74d3140a3d89a1d4ad5cff",
    "source_hdf5": "/nas/chenxianchi/datasets/sim/pick-place/raw/0920_200/pick-place/000001/attempt_001/data/episode_000000_cylinder_right.h5",
    "source_sha256": "10c437b81b31823d081f5a69b629d13b1983da117e80d9dbd0afff756c28422a",
    "fps": 10,
    "frame_count": 85,
    "publication": ""
  },
  {
    "task": "poker-draw",
    "video": "poker-draw.mp4",
    "replay": "_metadata/poker-draw/replay.json",
    "frames": "_metadata/poker-draw/frames.jsonl",
    "video_sha256": "49465ad62dfd4a7bc0bd91557f4cdf8c25fca456c11b74cc8b89ad206409ec21",
    "replay_sha256": "4f1853513cf3bd404450af5384f2e7a75fb1ecab5b11ba6774605686e7d637c7",
    "frames_sha256": "cedad46c58bcbd64a8eb6f12ee2cab286c513fb0cc825646752bebc7967b827a",
    "source_hdf5": "/nas/chenxianchi/datasets/sim/poker-draw/raw/0920_200/poker-draw/000000/attempt_001/data/episode_000000_card_right.h5",
    "source_sha256": "fc402667a363994ef5022abb86a121973d9b523b2f6805ffcfd8f0c9aad88e3f",
    "fps": 10,
    "frame_count": 299,
    "publication": ""
  },
  {
    "task": "usb-insert",
    "video": "usb-insert.mp4",
    "replay": "_metadata/usb-insert/replay.json",
    "frames": "_metadata/usb-insert/frames.jsonl",
    "video_sha256": "cd80ea5e8de0024dbdd7fb9b0568e3c8b17a0b9816365a874a18cf2560046a77",
    "replay_sha256": "f5f6bcee79f969b0fd1289b49c3a66a1d15f93a456cf7cc1c6e22a1b9bd8a606",
    "frames_sha256": "421ffc4e5bf3643e7a4f83b707bb6c71bf09185a4e7dcf3bb707e93d837857cd",
    "source_hdf5": "/nas/chenxianchi/datasets/sim/usb_insert/raw/0920_200/usb-insert/000000/attempt_001/data/raw/usb_000000.h5",
    "source_sha256": "3244540c0738e8e1950c9dbf36c1d836448187244fa5c8c0ce200eaeff6f6897",
    "fps": 10,
    "frame_count": 251,
    "publication": ""
  },
  {
    "task": "usb-unplug",
    "video": "usb-unplug.mp4",
    "replay": "_metadata/usb-unplug/review/replay.json",
    "frames": "_metadata/usb-unplug/review/frames.jsonl",
    "video_sha256": "e4cc36c0a63100bf578049fa023f1f27a232285f044317446cb520fcec0baa00",
    "replay_sha256": "c1b8ed69de02f33d87eeeb7b88e52499e5f8c8a90b0ee5efbfe51f228885c3f0",
    "frames_sha256": "b88fcb839077a1ba4fbe2a0171d38274e1562c2fed029506dad4c50a5d31bdbd",
    "source_hdf5": "/nas/chenxianchi/datasets/sim/task_demo/_metadata/usb-unplug/collection/raw/episode.h5",
    "source_sha256": "04042ad0e8567dccada94479dabcee44fa1ff9016d4395ef6370c5a2a25c6a66",
    "fps": 30,
    "frame_count": 712,
    "publication": ""
  },
  {
    "task": "whiteboard-wipe",
    "video": "whiteboard-wipe.mp4",
    "replay": "_metadata/whiteboard-wipe/replay.json",
    "frames": "_metadata/whiteboard-wipe/frames.jsonl",
    "video_sha256": "8b8bf394d02eaf949584092af17c5117968ce6f1f6abc34c6b942b2045846423",
    "replay_sha256": "6276f686f675ae99fc2f79c4158a12606d4543310ab7041c6ffac8113c901fdb",
    "frames_sha256": "caf1a5fb46e0c0eb32515f0c0e1e725d025d95300ac8aa922f5661041788d8a3",
    "source_hdf5": "/nas/chenxianchi/datasets/sim/whiteboard-wipe/raw/0929_200/whiteboard-wipe/000000/attempt_001/data/raw/episode.h5",
    "source_sha256": "6fa506a8bb6ebd0950c1502df178202b8c602ee26533b85ae022987ae5a4bdd6",
    "fps": 10,
    "frame_count": 669,
    "publication": ""
  },
  {
    "task": "whiteboard-wipe-horizontal",
    "video": "whiteboard-wipe-horizontal.mp4",
    "replay": "_metadata/whiteboard-wipe-horizontal/review_1010_200/replay.json",
    "frames": "_metadata/whiteboard-wipe-horizontal/review_1010_200/frames.jsonl",
    "video_sha256": "6cbad19373c23d24e16dfc54b51a3b484b4851e731409462eec437c6ad09be15",
    "replay_sha256": "ce109461a3d5f8a4fbf87740949e508aba5f15c998141e16c652e51bbe10f20d",
    "frames_sha256": "98c4994ccf9de67ff1aff65ce5917992a6025ff3bb01af1e16a5ac00e7663380",
    "source_hdf5": "/nas/chenxianchi/datasets/sim/whiteboard-wipe-horizontal/raw/1010_200/whiteboard-wipe-horizontal/000000/attempt_001/data/raw/episode.h5",
    "source_sha256": "fca98286045717c882e6e149fc33ab71d7c9e5cc4e7a0021a6dcb434c0ac1b90",
    "fps": 30,
    "frame_count": 2506,
    "publication": "_metadata/whiteboard-wipe-horizontal/publication.json"
  },
  {
    "task": "whiteboard-wipe-horizontal-line",
    "video": "whiteboard-wipe-horizontal-line.mp4",
    "replay": "_metadata/whiteboard-wipe-horizontal-line/review_1012_200/replay.json",
    "frames": "_metadata/whiteboard-wipe-horizontal-line/review_1012_200/frames.jsonl",
    "video_sha256": "a93d3b49f6cadda0c02087554dfac97c732cce941dd06543beaf8407248bbda2",
    "replay_sha256": "21a3dcdd60b98191d8d54d8134eecae9aaea38df7b490608f4c35afabc655480",
    "frames_sha256": "2bef1131c6883676cbc27430897d6d400852594c1838d957a44d1a9892445ca4",
    "source_hdf5": "/nas/chenxianchi/datasets/sim/whiteboard-wipe-horizontal-line/raw/1012_200/whiteboard-wipe-horizontal-line/000000/attempt_002/data/raw/episode.h5",
    "source_sha256": "b591007706dd17cc9ccf4edd535c6ee55bc9bd4aee4da88d4e2fc15518b785a0",
    "fps": 30,
    "frame_count": 2138,
    "publication": "_metadata/whiteboard-wipe-horizontal-line/publication.json"
  }
]''')
EXPERT_REGISTRY = {row["task"]: row for row in _REGISTRY_ROWS}
# Each first_frame.png was visually checked against the fixed offline replay
# renderer (two 4:3 RGB panes at left) on 2026-10-14. Crops are bound to exact
# video bytes, not transferable to policy-review-v3 model videos.
_EXPERT_CROP_AUDIT = json.loads(r'''{
  "67c7d06dd60a1dcf23de044f967e861ec533e8c3b8f17323726744d736827b90": {
    "first_frame": "_metadata/bulb-screw/review_1012_200/first_frame.png",
    "first_frame_sha256": "ed2244a8a9bf3e5e8aa26619d2067fe3270377bdd52362c4ac8be54d365972da"
  },
  "17e0867707d807d05d3eb924610a2e67e273e8e822195c46486420923f21f12c": {
    "first_frame": "_metadata/bulb-unscrew-tabletop/review/first_frame.png",
    "first_frame_sha256": "a0ed221b8ed150d5f04292751a008a49f7f0dfdee0c1d9c4b5448d04aeded251"
  },
  "26284b014ce0193e4547029d0e246cf878a2b83dd5c16f41698e708c3e2b5f88": {
    "first_frame": "_metadata/install-ram/first_frame.png",
    "first_frame_sha256": "0d0231dd6bc05f0c8d9d964dcd401d2dbed24b298c2ff4bc734cc3090e1ef0a6"
  },
  "e98c93bd68ee3ba17db58dbf66a858683e6000fbca9436ba1a841ff7b0ebc335": {
    "first_frame": "_metadata/pick-place/first_frame.png",
    "first_frame_sha256": "575fc2c259029fb50a48ea72761a11b163d070b5d86020d42dbc86ca068def2a"
  },
  "49465ad62dfd4a7bc0bd91557f4cdf8c25fca456c11b74cc8b89ad206409ec21": {
    "first_frame": "_metadata/poker-draw/first_frame.png",
    "first_frame_sha256": "85007360ade3daa01fcb9fcedb4450fe59d6de8ca393e2510ba69c5e151f0415"
  },
  "cd80ea5e8de0024dbdd7fb9b0568e3c8b17a0b9816365a874a18cf2560046a77": {
    "first_frame": "_metadata/usb-insert/first_frame.png",
    "first_frame_sha256": "a6ab94b067939c588b31b606edf58c22528491a74841eb11dc32ca7f90c80c89"
  },
  "e4cc36c0a63100bf578049fa023f1f27a232285f044317446cb520fcec0baa00": {
    "first_frame": "_metadata/usb-unplug/review/first_frame.png",
    "first_frame_sha256": "f8054bd8f22118a7f4283cea23fc8e7235c85d9b9deb7bcdf3c9aa0b202d3468"
  },
  "8b8bf394d02eaf949584092af17c5117968ce6f1f6abc34c6b942b2045846423": {
    "first_frame": "_metadata/whiteboard-wipe/first_frame.png",
    "first_frame_sha256": "9dcbb2487f73ea42178d5c287795473f77f2f97f4c534dc323c2029de0fed827"
  },
  "6cbad19373c23d24e16dfc54b51a3b484b4851e731409462eec437c6ad09be15": {
    "first_frame": "_metadata/whiteboard-wipe-horizontal/review_1010_200/first_frame.png",
    "first_frame_sha256": "c53b57ab6c4c34edfa8f23435fbe033e2b29a57510920ff005cb0acef41eac61"
  },
  "a93d3b49f6cadda0c02087554dfac97c732cce941dd06543beaf8407248bbda2": {
    "first_frame": "_metadata/whiteboard-wipe-horizontal-line/review_1012_200/first_frame.png",
    "first_frame_sha256": "3d342770492dc3dd16106cb9b3e269d1e373313fab1097e9f00cba9df8a8d450"
  }
}''')
_MAX_JSON = 4_000_000
_MAX_FRAMES = 64_000_000
_MAX_VIDEO = 128_000_000
_MODEL_RENDERER_PATH = "src/kaihand_tactile_env/shared/evaluation_video.py"
_MODEL_RENDERER_AUDIT = {
    # Read-only frozen source inspected 2026-10-14; the other three current
    # frozen evaluation snapshots have exactly these same renderer bytes.
    "c31e5590abfa93e0cb6052b39db4b92e334771c55759f054e474ac75060a862b":
        "/cpfs_infra/user/chenxianchi/runtime_snapshots/pi05_serial_20261013_111821/kaitactilesim/" + _MODEL_RENDERER_PATH,
    # Archived original bytes also exist in kaitactilesim git object a46defb.
    # The frame fit, three pane positions, and title overlays match the modern
    # renderer; the task metric text changed, not the camera geometry.
    "ec4c74e020942d4366d7330dd74071cda6b2c5709868be75420c81fa49287aac":
        "/cpfs_infra/user/chenxianchi/code/kai_sharpatactilesim/" + _MODEL_RENDERER_PATH,
}
_MODEL_LAYOUT = "enlarged head/right-wrist/global-or-named RGB; enlarged bilateral Fn/|Ft| maps"
_MODEL_VIEWS = [
    {"id": "head", "label": "主视角", "x": 214, "y": 78, "width": 434, "height": 324},
    # The right-wrist title overlays y=410..438, including part of its RGB.
    # Omit this 30px strip and preserve the resulting 434:294 display ratio.
    {"id": "wrist", "label": "右腕视角", "x": 214, "y": 440, "width": 434, "height": 294},
    {"id": "global", "label": "全局视角", "x": 214, "y": 746, "width": 434, "height": 324},
]
_MODEL_CONTRACT_KEYS = (
    "schema", "layout", "output_size", "review_render_size", "model_input_cameras_displayed",
    "review_only_cameras_displayed", "right_wrist_camera_role", "fps", "frame_count", "completed",
)
_MODEL_LEGACY_AUDIT_PATH = Path(__file__).parent / "model_crop_audits/legacy_v3_20261014/audit.json"
_MODEL_LEGACY_AUDIT_SHA256 = "9f3086458f51c350ffb15722e6a85c15d0e3ad98cb961a4bad2b6197e0d4d1d5"


def _read(path, limit):
    path = Path(path)
    if path.is_symlink() or not stat.S_ISREG(path.stat().st_mode):
        raise ValueError(f"Expected a regular non-symlink expert artifact: {path}")
    before = path.stat()
    if before.st_size <= 0 or before.st_size > limit:
        raise ValueError(f"Expert artifact size outside bounds: {path}")
    data = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
            after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError(f"Expert artifact changed while reading: {path}")
    return data, {"path": str(path.resolve()), "size": len(data),
                  "sha256": hashlib.sha256(data).hexdigest()}


def _json(path):
    raw, row = _read(path, _MAX_JSON)
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value, row


def _artifact(root, relative, expected, limit, role):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Unsafe expert registry artifact path")
    path = root / relative
    if not path.resolve(strict=True).is_relative_to(root):
        raise ValueError("Expert artifact escapes the audited demo root")
    data, row = _read(path, limit)
    if not isinstance(expected, str) or not _SHA.fullmatch(expected) or row["sha256"] != expected:
        raise ValueError(f"Expert {role} SHA256 mismatch: {path}")
    row["role"] = role
    return data, row


def _probe_video(path):
    """Metadata-only probe; never decode/render/re-run the simulation."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate,nb_frames,duration:format=duration",
         "-of", "json", str(path)], check=True, capture_output=True,
        text=True, timeout=20)
    if len(result.stdout) > 100_000:
        raise ValueError("Oversized ffprobe output")
    return json.loads(result.stdout)


def _unavailable(task, reason):
    return {"status": "unavailable", "task": task,
            "title": "专家示范不可用", "reason": reason,
            "clock": "independent_video_seconds", "copy_jobs": []}


def _reference(summary):
    video = summary.get("video")
    if not isinstance(video, dict):
        return {}
    comparison = video.get("comparison_plots")
    ref = comparison.get("reference") if isinstance(comparison, dict) else None
    return ref if isinstance(ref, dict) else {}


def _case_key(summary):
    ref = _reference(summary)
    values = (summary.get("task"), ref.get("dataset_root", ""),
              ref.get("source_hdf5", ""), ref.get("output_episode_index"))
    if not isinstance(values[0], str) or any(
            not isinstance(value, str) for value in values[1:3]):
        raise ValueError("Invalid task/reference identity")
    if values[3] is not None and type(values[3]) is not int:
        raise ValueError("Invalid reference episode index")
    # Keep per-trial witnesses attached to their own seed/checkpoint even when
    # multiple trials share the same reference recording.
    checkpoint, seed = summary.get("checkpoint_sha256"), summary.get("seed")
    if not isinstance(checkpoint, str) or not _SHA.fullmatch(checkpoint) or type(seed) is not int:
        raise ValueError("Invalid model checkpoint/seed identity")
    return values + (checkpoint, seed)


def _h50(summary):
    horizon = summary.get("prediction_horizon")
    if horizon is None:
        server = summary.get("server_metadata", {})
        horizon = server.get("prediction_horizon", server.get("action_horizon"))
    return type(horizon) is int and horizon == 50 and type(summary.get("execute_steps")) is int and summary["execute_steps"] == 50


def _expert_views(root, replay, video_sha, provenance):
    views = {"status": "full_frame_only", "sha256": video_sha,
             "source_width": 1920, "source_height": 1080, "views": []}
    crop = _EXPERT_CROP_AUDIT.get(video_sha)
    if not crop:
        return views
    raw, witness = _artifact(root, crop["first_frame"], crop["first_frame_sha256"],
                             20_000_000, "expert_visually_audited_crop_first_frame")
    if (replay.get("camera_names") != ["head", "right_wrist"]
            or raw[:8] != b"\x89PNG\r\n\x1a\n" or raw[12:16] != b"IHDR"
            or int.from_bytes(raw[16:20], "big") != 1920
            or int.from_bytes(raw[20:24], "big") != 1080):
        raise ValueError("Expert crop first-frame/camera geometry differs from audit")
    provenance.append(witness)
    views.update(status="verified", views=[{
        "id": "head", "label": "主视角", "x": 10, "y": 108,
        "width": 632, "height": 456,
    }], audit="2026-10-14 visual first-frame + fixed offline replay renderer audit; exact video SHA-bound crop. Full mosaic remains available.")
    return views


def _verify_registered(root, audit):
    video_raw, video_row = _artifact(root, audit["video"], audit["video_sha256"],
                                    _MAX_VIDEO, "expert_recorded_video")
    del video_raw
    replay_raw, replay_row = _artifact(root, audit["replay"], audit["replay_sha256"],
                                      _MAX_JSON, "expert_replay_manifest")
    frame_raw, frame_row = _artifact(root, audit["frames"], audit["frames_sha256"],
                                    _MAX_FRAMES, "expert_playback_frame_clock")
    replay = json.loads(replay_raw)
    if (not isinstance(replay, dict) or replay.get("schema_version") != _SCHEMA
            or replay.get("completed") is not True or replay.get("task") != audit["task"]
            or replay.get("source_modified") is not False or replay.get("physics_rerun") is not False
            or replay.get("source_hdf5") != audit["source_hdf5"]
            or replay.get("source_sha256") != audit["source_sha256"]
            or replay.get("output_size") != [1920, 1080]
            or replay.get("fps") != audit["fps"]
            or replay.get("output_frame_count") != audit["frame_count"]):
        raise ValueError("Expert replay identity/layout/clock differs from explicit audit")
    fps, count = audit["fps"], audit["frame_count"]
    if type(count) is not int or count < 1 or count > 100_000 or type(fps) is not int or not 1 <= fps <= 120:
        raise ValueError("Invalid expert audited frame count/fps")
    frames = [json.loads(line) for line in frame_raw.splitlines() if line.strip()]
    if len(frames) != count:
        raise ValueError("Expert frame count differs from audited replay")
    camera_times = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError("Invalid expert frame row")
        playback, camera = frame.get("playback_timestamp_s"), frame.get("camera_pose_timestamp_s")
        if (type(frame.get("output_index")) is not int or frame["output_index"] != index
                or type(playback) not in (int, float) or not math.isfinite(playback)
                or abs(playback - index / fps) > 1e-6
                or type(camera) not in (int, float) or not math.isfinite(camera)):
            raise ValueError("Expert playback frame clock is not contiguous/finite")
        camera_times.append(camera)
    if any(b < a for a, b in zip(camera_times, camera_times[1:])):
        raise ValueError("Expert camera clock runs backwards")
    declared_duration = replay.get("video_duration_s")
    if (type(declared_duration) not in (int, float) or not math.isfinite(declared_duration)
            or abs(declared_duration - count / fps) > 1e-6):
        raise ValueError("Expert replay duration does not equal frame count/fps")
    probe = _probe_video(Path(video_row["path"]))
    streams = probe.get("streams", [])
    if len(streams) != 1:
        raise ValueError("Expected one expert video stream")
    stream = streams[0]
    try:
        num, den = (float(x) for x in stream["r_frame_rate"].split("/"))
        actual_fps = num / den
        encoded_duration = float(probe["format"]["duration"])
        stream_duration = float(stream["duration"])
        encoded_frames = int(stream["nb_frames"])
    except (KeyError, ValueError, TypeError, ZeroDivisionError) as error:
        raise ValueError("Incomplete expert video probe metadata") from error
    if (stream.get("width") != 1920 or stream.get("height") != 1080
            or encoded_frames != count or abs(actual_fps - fps) > 1e-6
            or not math.isfinite(encoded_duration) or not math.isfinite(stream_duration)
            or abs(encoded_duration - declared_duration) > .002
            or abs(stream_duration - declared_duration) > .002):
        raise ValueError("Encoded expert video disagrees with replay layout/clock")
    provenance = [video_row, replay_row, frame_row]
    if audit.get("publication"):
        publication_path = root / audit["publication"]
        if not publication_path.resolve(strict=True).is_relative_to(root):
            raise ValueError("Expert publication escapes demo root")
        publication, publication_row = _json(publication_path)
        if (publication.get("schema") != "kaihand_task_demo_publication_v1"
                or publication.get("task") != audit["task"]
                or publication.get("video_sha256") != video_row["sha256"]
                or publication.get("replay_sha256") != replay_row["sha256"]
                or publication.get("canonical_video") != str(root / audit["video"])
                or publication.get("source_review_dir") != str((root / audit["replay"]).parent)):
            raise ValueError("Expert publication does not bind the selected current replay/video")
        publication_row["role"] = "expert_current_publication"
        provenance.append(publication_row)
    views = _expert_views(root, replay, video_row["sha256"], provenance)
    return {
        "status": "available", "task": audit["task"], "url": "media/expert.mp4",
        "clock": "independent_video_seconds", "duration_s": encoded_duration,
        "manifest_duration_s": declared_duration, "stream_duration_s": stream_duration,
        "frame_count": count, "width": 1920, "height": 1080, "fps": fps,
        "sha256": video_row["sha256"], "source_hdf5": audit["source_hdf5"],
        "source_sha256": audit["source_sha256"], "source_episode_index": replay.get("source_episode_index"),
        "source_verification": "Source HDF SHA256 is declared by the SHA-verified replay; the large HDF was not reread or rehashed by this attachment.",
        "description": "记录的脚本示范轨迹离线回放，不是真机人工示教；单条示范不代表专家总体表现。",
        "role": "scripted_recorded_demonstration", "provenance": provenance,
        "tactile_source_kind": replay.get("tactile_source_kind"),
        "tactile_heatmaps": deepcopy(replay.get("tactile_heatmaps", [])),
        "curve_unit": replay.get("curve_unit"),
        "media_binding": "Explicit audited task/video/replay/frame SHA registry; never inferred from filenames.",
        "frame_playback_range_s": [frames[0]["playback_timestamp_s"], frames[-1]["playback_timestamp_s"]],
        "frame_camera_range_s": [camera_times[0], camera_times[-1]],
        "media_views": views,
        "copy_jobs": [{"source": Path(video_row["path"]), "url": "media/expert.mp4",
                       "sha256": video_row["sha256"], "size": video_row["size"]}],
    }


def prepare_expert_bindings(trials: list[Path], task_demo_root: Path) -> dict:
    """Audit selected trial identities and each registered expert once.

    Keys retain task, exact recorded reference dataset/HDF, and episode index.
    A different reference recording is never silently relabelled as the blue
    PNG curve; it is an explicitly labelled independent same-task example.
    This function reads only metadata/videos, writes nothing, and invokes only
    ffprobe. Unknown tasks are represented by explicit unavailable descriptors.
    """
    if not isinstance(trials, (list, tuple)) or len(trials) > 10_000:
        raise ValueError("Expected a bounded trial list")
    root = Path(task_demo_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Expert demo root must be a directory")
    bindings, cache, seen = {}, {}, set()
    for trial in trials:
        trial = Path(trial).resolve(strict=True)
        if trial in seen:
            continue
        seen.add(trial)
        summary, summary_row = _json(trial / "summary.json")
        key = _case_key(summary)
        if not _h50(summary):
            bindings[key] = _unavailable(key[0], "模型记录未明确验证为 H50K50；未按目录名匹配专家。")
            continue
        review_dir = trial if (trial / "review.json").is_file() else trial / "review"
        review, review_row = _json(review_dir / "review.json")
        comparison, comparison_row = _json(review_dir / "evaluation_comparison.json")
        if (review.get("task") != summary.get("task") or review.get("seed") != summary.get("seed")
                or review.get("checkpoint_sha256") != summary.get("checkpoint_sha256")
                or review.get("completed") is not True
                or comparison.get("reference", {}) != _reference(summary)
                or review.get("comparison_plots", {}).get("reference", {}) != _reference(summary)):
            raise ValueError("Trial summary/review/reference identity mismatch")
        task = key[0]
        if task not in EXPERT_REGISTRY:
            bindings[key] = _unavailable(task, "没有经明确任务身份与 SHA 审计注册的专家示范。")
            continue
        if task not in cache:
            cache[task] = _verify_registered(root, EXPERT_REGISTRY[task])
        descriptor = deepcopy(cache[task])
        same = bool(key[2]) and key[2] == descriptor["source_hdf5"] and key[3] == 0
        descriptor["reference_relation"] = "same_recording" if same else "same_task_other_recording"
        descriptor["title"] = "专家参考回放" if same else "同任务专家示范"
        source_parts = Path(descriptor["source_hdf5"]).parts
        collection = "collection 独立录制" if "collection" in source_parts else "/".join(source_parts[-7:-3])
        descriptor["source_label"] = f"{task} · 脚本示范录制 · {collection}" + ("" if same else " · 与当前参考不同")
        if comparison.get("status") == "reference_unavailable":
            comparison_note = (
                "与原记录声明的参考源不同；本页未绘制专家参考线。初始条件可能不同。"
                if not same else "与原记录声明的参考源相同；本页未绘制专家参考线，初始条件仍不保证相同。"
            )
        else:
            comparison_note = (
                "与 PNG 蓝线对应同一源录制；仍不代表与模型初始条件相同。"
                if same else
                "同任务的另一条录制，不是 PNG 蓝线对应 episode；初始条件可能不同。"
            )
        descriptor["comparison_note"] = comparison_note + "独立播放，不进行阶段对齐，不参与模型评分。"
        if descriptor["tactile_source_kind"] == "genesis_probe":
            descriptor["comparison_note"] += "专家拼图为 Depth（mm）/Contact（0/1）及 Genesis 局部代理量，不是 Fn/Ft（N），不可与模型力图作数值比较。"
        elif descriptor["tactile_source_kind"] == "spatial_force":
            descriptor["comparison_note"] += "完整拼图的力图色标可能与模型不同，请以图内单位与色标为准。"
        descriptor["model_reference"] = deepcopy(_reference(summary))
        # Provenance for shared media is kept separate from per-trial reference
        # witnesses so callers can copy one immutable media artifact per case.
        descriptor["reference_binding_evidence"] = [
            {**summary_row, "role": "model_reference_summary"},
            {**review_row, "role": "model_reference_review"},
            {**comparison_row, "role": "model_reference_comparison"},
        ]
        bindings[key] = descriptor
    return bindings


def expert_for_case(summary: dict, bindingdict: dict) -> dict:
    """Return a defensive copy; no task-only or filename-derived fallback."""
    if not isinstance(summary, dict) or not _h50(summary):
        return _unavailable(summary.get("task") if isinstance(summary, dict) else None,
                            "模型记录未明确验证为 H50K50。")
    key = _case_key(summary)
    descriptor = bindingdict.get(key)
    if descriptor is None:
        return _unavailable(key[0], "此任务及确切参考身份尚未审计，未自动匹配同名视频。")
    return deepcopy(descriptor)


def _model_renderer_views(summary, result):
    """Verify the recorded source inventory, then the audited renderer bytes."""
    try:
        integrity = summary.get("deployment_integrity", {})
        source = integrity.get("evaluation_source", {})
        digest = integrity.get("manifest", {}).get("actual_sha256") or integrity.get("manifest_sha256")
        if (integrity.get("status") != "verified" or source.get("status") != "verified"
                or not isinstance(digest, str) or not _SHA.fullmatch(digest)
                or not source.get("expected_sha256")
                or source.get("expected_sha256") != source.get("actual_sha256")):
            return result
        manifest, manifest_row = _json(Path(summary["deployment_manifest"]))
        if (manifest_row["sha256"] != digest or manifest.get("task") != summary.get("task")
                or manifest.get("checkpoint_sha256") != summary.get("checkpoint_sha256")
                or manifest.get("deployment_id") != summary.get("deployment_id")
                or manifest.get("prediction_horizon") != 50):
            return result
        rows = manifest.get("evaluation_source_files")
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            return result
        inventory_sha = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if inventory_sha != manifest.get("evaluation_source_sha256") or inventory_sha != source["actual_sha256"]:
            return result
        renderer_rows = [row for row in rows if row.get("path") == _MODEL_RENDERER_PATH]
        if len(renderer_rows) != 1:
            return result
        renderer = renderer_rows[0]
        audited_path = _MODEL_RENDERER_AUDIT.get(renderer.get("sha256"))
        if audited_path is None:
            return result
        # Staged batches retain the audited c31e renderer, despite different
        # observer/source-inventory SHAs. Prefer their own frozen renderer as
        # witness when an absolute root is explicitly recorded. The known-byte
        # renderer allowlist remains mandatory; a new task name is not evidence.
        project = source.get("root")
        if isinstance(project, str) and Path(project).is_absolute():
            declared_project = manifest.get("evaluation_project_path")
            if declared_project is not None and declared_project != project:
                return result
            candidate = Path(project) / _MODEL_RENDERER_PATH
            if candidate.is_file():
                audited_path = str(candidate)
        _, renderer_witness = _read(Path(audited_path), _MAX_JSON)
        if renderer_witness["sha256"] != renderer["sha256"] or renderer_witness["size"] != renderer.get("size"):
            return result
        manifest_row["role"] = "model_crop_deployment_manifest"
        renderer_witness["role"] = "model_crop_audited_renderer_bytes"
        result.update(status="verified", views=deepcopy(_MODEL_VIEWS),
            reason="已核验三相机 v3 布局与存档渲染器；腕部裁剪去除覆盖标签，按 434:294 显示。仅空间裁剪，不声明时间对齐。",
            renderer_sha256=renderer["sha256"], evaluation_source_inventory_sha256=inventory_sha,
            provenance=[manifest_row, renderer_witness])
    except (OSError, ValueError, TypeError, KeyError):
        # Missing/moved legacy manifests or unmatched renderer versions must
        # not remove the usable video or inherit a crop from another trial.
        return result
    return result


def _audited_model_file(witness, limit, role):
    """Recheck persistent audit witnesses; never trust a stale file path."""
    raw, actual = _read(Path(witness["path"]), limit)
    if actual["sha256"] != witness.get("sha256") or actual["size"] != witness.get("size"):
        raise ValueError("Model crop witness changed after explicit media audit")
    actual["role"] = role
    return raw, actual


def _legacy_model_views(summary, video_sha, result):
    """Exact-video registry for archived trials whose deployment was moved.

    No filename/task-only fallback: the immutable audit couples every video to
    its summary, review, saved first frame, CPU-decoded first frame and source
    protocol. The latter must declare the actual archived renderer hash.
    """
    try:
        audit, audit_row = _json(_MODEL_LEGACY_AUDIT_PATH)
        if audit_row["sha256"] != _MODEL_LEGACY_AUDIT_SHA256:
            raise ValueError("Registered model media audit digest differs")
        if (audit.get("schema") != "evaluation-workbench-model-crop-audit-v1"
                or audit.get("views") != _MODEL_VIEWS
                or not audit.get("representative_visual_review")):
            raise ValueError("Registered model media audit schema/geometry differs")
        recording = audit.get("recordings", {}).get(video_sha)
        if recording is None:
            return result
        if (any(recording.get(k) != summary.get(k) for k in
                ("task", "seed", "checkpoint_sha256", "deployment_id"))
                or recording.get("contract") != {k: summary["video"].get(k) for k in _MODEL_CONTRACT_KEYS}
                or recording.get("pane_label_signatures_verified") is not True
                or not 0 <= recording.get("decoded_vs_saved_first_rgb_mae", math.inf) < 3
                or recording.get("video", {}).get("sha256") != video_sha):
            raise ValueError("Exact model recording/layout identity differs from audit")
        raw, summary_row = _audited_model_file(recording["summary"], _MAX_JSON, "model_crop_recorded_summary")
        if json.loads(raw) != summary:
            raise ValueError("Caller summary differs from the SHA-bound source summary")
        raw, review_row = _audited_model_file(recording["review"], _MAX_JSON, "model_crop_recorded_review")
        review = json.loads(raw)
        if ({k: review.get(k) for k in _MODEL_CONTRACT_KEYS} != recording["contract"]
                or any(review.get(k) != summary.get(k) for k in
                       ("task", "seed", "checkpoint_sha256", "deployment_id"))):
            raise ValueError("Review identity differs from the registered source summary")
        raw, first_row = _audited_model_file(recording["first_frame"], 20_000_000, "model_crop_recorded_first_frame")
        if (raw[:8] != b"\x89PNG\r\n\x1a\n" or raw[12:16] != b"IHDR"
                or int.from_bytes(raw[16:20], "big") != 1920 or int.from_bytes(raw[20:24], "big") != 1080):
            raise ValueError("Registered first-frame dimensions differ")
        _, video_row = _audited_model_file(recording["video"], _MAX_VIDEO, "model_crop_recorded_video")
        raw, protocol_row = _audited_model_file(recording["protocol"], _MAX_JSON, "model_crop_recorded_protocol")
        protocol = json.loads(raw)
        renderer = audit["renderer"]
        if (any(protocol.get(k) != summary.get(k) for k in ("task", "checkpoint_sha256", "deployment_id"))
                or protocol.get("prediction_horizon") != 50 or protocol.get("execute_steps") != 50
                or summary["seed"] not in protocol.get("seeds", [])
                or protocol.get("source_hashes", {}).get(_MODEL_RENDERER_PATH) != renderer["sha256"]
                or _MODEL_RENDERER_AUDIT.get(renderer["sha256"]) != renderer["path"]
                or protocol.get("review", {}).get("resolution") != [1920, 1080]
                or protocol.get("review", {}).get("fps") != summary["video"].get("fps")
                or protocol.get("review", {}).get("second_camera") != "global"):
            raise ValueError("Archived protocol does not bind this model/renderer/layout")
        _, renderer_row = _audited_model_file(renderer, _MAX_JSON, "model_crop_audited_renderer_bytes")
        _, sheet_row = _audited_model_file(audit["contact_sheet"], 20_000_000, "model_crop_visual_contact_sheet")
        audit_row["role"] = "model_crop_exact_video_audit"
        result.update(status="verified", views=deepcopy(_MODEL_VIEWS),
            reason="已核验本视频 SHA、实际首帧和相机标签，以及存档协议/渲染器；腕部去除标签后按 434:294 显示。仅空间裁剪，不声明时间对齐。",
            renderer_sha256=renderer["sha256"],
            provenance=[audit_row, summary_row, review_row, first_row, video_row,
                        protocol_row, renderer_row, sheet_row])
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        result["reason"] = "完整拼图；既有视频裁剪审计或其来源文件未通过本次复核，未套用其他轨迹的裁剪框。"
    return result


def model_views_for_case(summary: dict, video_sha: str) -> dict:
    """SHA-bound head/wrist/global crops; unsupported layouts remain complete.

    The caller supplies the SHA of the video it actually exports. The normal
    chain verifies the deployment source inventory and archived renderer. The
    explicitly registered legacy alternative verifies exact video bytes,
    frame-zero evidence and its original protocol. Neither proves clock sync.
    """
    if not isinstance(video_sha, str) or not _SHA.fullmatch(video_sha):
        raise ValueError("Expected model video SHA256")
    video = summary.get("video") if isinstance(summary, dict) else None
    size = video.get("output_size") if isinstance(video, dict) else None
    if not isinstance(size, list) or len(size) != 2 or any(type(x) is not int or x <= 0 for x in size):
        size = [1920, 1080]
    result = {"status": "full_frame_only", "sha256": video_sha,
              "source_width": size[0], "source_height": size[1], "views": [],
              "reason": "完整拼图；没有通过来源核验的相机裁剪证据。"}
    if (not isinstance(video, dict) or not _h50(summary) or size != [1920, 1080]
            or video.get("schema") != "kaihand-policy-evaluation-review-v3"
            or video.get("layout") != _MODEL_LAYOUT
            or video.get("review_render_size") != [640, 480]
            or video.get("model_input_cameras_displayed") != ["head", "right_wrist"]
            or video.get("review_only_cameras_displayed") != ["global"]
            or video.get("right_wrist_camera_role") != "model_input"
            or video.get("completed") is not True):
        result["reason"] = "完整拼图；记录未明确声明已审计的 H50 三相机布局或真实右腕模型视角。"
        return result
    _model_renderer_views(summary, result)
    if result["status"] != "verified":
        _legacy_model_views(summary, video_sha, result)
    return result

