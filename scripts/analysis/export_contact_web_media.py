#!/usr/bin/env python3
"""嵌入专家与模型原始视频，保留每一视频帧与仿真时间的真实映射。

需 ffprobe；不重新编码视频、不修改原始评估文件。播放器应按 simTime
左取已观测帧，再跳至对应 pts，不能将视频编码秒数直接当作仿真秒数。
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def probe_video(path: Path) -> tuple[dict, list[float]]:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError("导出媒体需要 ffprobe，以核实真实编码帧时间")
    completed = subprocess.run([
        ffprobe, "-v", "error", "-select_streams", "v:0", "-show_frames",
        "-show_entries", "stream=codec_name,width,height,nb_frames,duration,pix_fmt:frame=best_effort_timestamp_time",
        "-of", "json", str(path),
    ], check=True, capture_output=True, text=True)
    result = json.loads(completed.stdout)
    stream = result["streams"][0]
    if stream["codec_name"] != "h264":
        raise ValueError(f"浏览器媒体暂仅支持已核实的 H.264 文件：{path.name}")
    pts = [float(frame["best_effort_timestamp_time"]) for frame in result["frames"]]
    validate_times(pts, "视频帧时间")
    return stream, pts


def validate_times(values: list[float], label: str) -> None:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError(f"{label}为空或存在非有限值")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise ValueError(f"{label}必须严格递增")


def media_source(video: Path, frames: Path, *, label: str,
                 simulation_key: str, index_key: str, expected_end: float) -> dict:
    rows = [json.loads(line) for line in frames.read_text(encoding="utf-8").splitlines() if line.strip()]
    sim_time = [float(row[simulation_key]) for row in rows]
    validate_times(sim_time, f"{label}仿真时间")
    if any(row[index_key] != index for index, row in enumerate(rows)):
        raise ValueError(f"{label}帧索引不连续")
    if abs(sim_time[0]) > 1e-7 or abs(sim_time[-1] - expected_end) > 1e-7:
        raise ValueError(f"{label}视频元数据与触觉报告的起止时间不一致")
    stream, pts = probe_video(video)
    if len(pts) != len(rows) or int(stream["nb_frames"]) != len(rows):
        raise ValueError(f"{label}视频解码帧数与元数据行数不一致")
    raw = video.read_bytes()
    return {
        "label": label,
        "filename": "expert_whiteboard.mp4" if label == "专家示例" else "model_evaluation.mp4",
        "originalFilename": video.name,
        "mime": "video/mp4",
        "src": "data:video/mp4;base64," + base64.b64encode(raw).decode("ascii"),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "width": stream["width"], "height": stream["height"],
        "codec": stream["codec_name"], "pixelFormat": stream["pix_fmt"],
        "durationS": float(stream["duration"]),
        "endS": round(sim_time[-1], 9),
        "frameCount": len(rows),
        "simTime": [round(value, 9) for value in sim_time],
        "pts": [round(value, 9) for value in pts],
        "mapping": "metadata simulation time to decoded video PTS, observed left frame; no interpolation or time stretching",
    }


def export_media(trial: Path, reference_video: Path | None = None) -> dict:
    report = read_json(trial / "tactile_contact_comparison.json")
    reference = report["reference"]
    metadata = reference["metadata"]
    replay_path = Path(metadata["source_replay_json"])
    replay = read_json(replay_path)
    if Path(replay["source_hdf5"]).resolve() != Path(metadata["source_hdf5"]).resolve():
        raise ValueError("专家视频与触觉参考不是同一原始 HDF5 轨迹")
    if replay.get("source_sha256") != metadata.get("declared_source_sha256"):
        raise ValueError("专家视频与触觉报告声明的轨迹摘要不一致")
    if not replay.get("completed"):
        raise ValueError("专家原始回放未完成")
    if reference_video is None:
        reference_video = replay_path.parent.parent.parent / f"{replay['task']}.mp4"
    model_dir = trial / "review" if (trial / "review" / "review.mp4").is_file() else trial
    sources = {
        "reference": media_source(reference_video, replay_path.parent / "frames.jsonl",
                                  label="专家示例", simulation_key="camera_pose_timestamp_s",
                                  index_key="output_index", expected_end=reference["last_observation_time_s"]),
        "rollout": media_source(model_dir / "review.mp4", model_dir / "frames.jsonl",
                                label="模型评估", simulation_key="simulation_time_s",
                                index_key="frame", expected_end=report["rollout"]["last_observation_time_s"]),
    }
    sources["rollout"]["filename"] = f"model_{trial.name}.mp4"
    if sources["reference"]["frameCount"] != replay["output_frame_count"]:
        raise ValueError("专家回放声明的帧数与视频不一致")
    return {
        "schemaVersion": "contact-media-v1", "sources": sources,
        "referenceIdentity": "video replay metadata and tactile report declare the same source HDF5 and SHA256; raw HDF5 not rehashed",
        "playback": "Unified simulation clock; video frame selection by metadata; video end freezes only its image, not the data timeline.",
        "colorScaleNote": "原视频为不同时间生成的回放，其触觉图内嵌色标不相同；用力数值比较请看本页统一色标的触觉与曲线。",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-dir", type=Path, required=True)
    parser.add_argument("--reference-video", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"不覆盖已有文件：{args.output}")
    payload = export_media(args.trial_dir, args.reference_video)
    content = "window.CONTACT_MEDIA = " + json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + ";\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(content)
    print(json.dumps({"output": str(args.output), "bytes": len(content.encode("utf-8")),
                      "sources": {key: {field: source[field] for field in ("frameCount", "endS", "durationS", "sha256")}
                                  for key, source in payload["sources"].items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
