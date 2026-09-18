#!/usr/bin/env python3
"""Strictly validate a fixed-camera EgoSteer WebDataset export.

The validator never extracts archive members.  It checks the dataset manifest,
the physical tar layout, every serialized sample, and temporal relationships
that cannot be checked by WebDataset's decoder alone.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, UnidentifiedImageError

EXPECTED_FPS = 30.0
BASE_LOWDIM_DIM = 96
CAMERA_BLOCK_DIM = 20
CANONICAL_CAMERA_ORDER = ("head", "left_wrist", "right_wrist")
DEFAULT_CAMERAS = ("head",)
EXPECTED_SCHEMA_VERSION = "egosteer_webdataset_head_v1"
EXPECTED_SAMPLE_KEY = "episode_{episode_index:06d}_frame_{frame_index:06d}"
MAX_MEMBER_BYTES = {
  "image.jpg": 64 * 1024 * 1024,
  "left_wrist_image.jpg": 64 * 1024 * 1024,
  "right_wrist_image.jpg": 64 * 1024 * 1024,
  "lowdim.npy": 64 * 1024,
  "meta.json": 1024 * 1024,
}
SAMPLE_RE = re.compile(
  r"^(?P<prefix>episode_(?P<episode>[0-9]{6})_frame_(?P<frame>[0-9]{6}))"
  r"\.(?P<suffix>image\.jpg|left_wrist_image\.jpg|right_wrist_image\.jpg|"
  r"lowdim\.npy|meta\.json)$"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ValidationIssue:
  """One validation failure with a stable machine-readable code."""

  code: str
  location: str
  message: str


@dataclass(frozen=True)
class SampleRecord:
  """The validated fields needed for temporal and aggregate checks."""

  shard: str
  split: str
  prefix: str
  episode_index: int
  frame_index: int
  dataset_name: str
  instruction_signature: str
  cameras: tuple[str, ...]
  lowdim: np.ndarray | None
  image_size: tuple[int, int] | None


@dataclass(frozen=True)
class ShardRecord:
  """Observed contents of one tar shard."""

  relative_path: str
  split: str
  samples: tuple[SampleRecord, ...]
  episode_keys: tuple[tuple[str, int], ...]


class DatasetValidator:
  """Accumulate all strict validation checks into one bounded report."""

  def __init__(
    self,
    root: Path,
    *,
    max_reported_errors: int = 100,
    numeric_atol: float = 1e-5,
    precomputed_sha256: Mapping[str, str] | None = None,
  ) -> None:
    self.root = root.resolve()
    self.max_reported_errors = max_reported_errors
    self.numeric_atol = numeric_atol
    self.error_count = 0
    self.issues: list[ValidationIssue] = []
    self.manifest: dict[str, Any] | None = None
    self.manifest_path = self.root / "dataset_manifest.json"
    self.expected_cameras = DEFAULT_CAMERAS
    self.expected_lowdim_dim = BASE_LOWDIM_DIM + CAMERA_BLOCK_DIM
    self.expected_suffixes = self._member_order(DEFAULT_CAMERAS)
    self.expected_image_size: tuple[int, int] | None = None
    self.observed_image_size: tuple[int, int] | None = None
    self.shards: list[ShardRecord] = []
    # In-process converters may supply hashes they just computed from complete
    # shard files. Standalone validation leaves this empty and re-hashes every
    # file, preserving its independent strict-validation behavior.
    self._sha256_cache: dict[str, str] = dict(precomputed_sha256 or {})

  @staticmethod
  def _parse_cameras(value: object) -> tuple[str, ...] | None:
    if not isinstance(value, list) or not value:
      return None
    declared = tuple(value)
    if not all(isinstance(camera, str) for camera in declared):
      return None
    if "head" not in declared or len(set(declared)) != len(declared):
      return None
    if any(camera not in CANONICAL_CAMERA_ORDER for camera in declared):
      return None
    canonical = tuple(
      camera for camera in CANONICAL_CAMERA_ORDER if camera in declared
    )
    return declared if declared == canonical else None

  @staticmethod
  def _camera_image_member(camera: str) -> str:
    return "image.jpg" if camera == "head" else f"{camera}_image.jpg"

  @classmethod
  def _member_order(cls, cameras: Sequence[str]) -> tuple[str, ...]:
    return tuple(cls._camera_image_member(camera) for camera in cameras) + (
      "lowdim.npy",
      "meta.json",
    )

  def issue(self, code: str, location: str, message: str) -> None:
    """Record an error while keeping console output bounded."""

    self.error_count += 1
    if len(self.issues) < self.max_reported_errors:
      self.issues.append(ValidationIssue(code, location, message))

  def validate(self) -> dict[str, Any]:
    """Run all checks and return a JSON-serializable summary."""

    if not self.root.is_dir():
      self.issue("dataset.not_directory", str(self.root), "dataset root is absent")
      return self._report()

    self._read_manifest()
    tar_paths = sorted(self.root.rglob("*.tar"))
    if not tar_paths:
      self.issue("dataset.no_shards", str(self.root), "no .tar shards were found")
    for tar_path in tar_paths:
      if tar_path.is_symlink():
        self.issue(
          "tar.symlink",
          self._relative(tar_path),
          "dataset shards must be regular files, not symbolic links",
        )
        continue
      shard = self._validate_shard(tar_path)
      if shard is not None:
        self.shards.append(shard)

    self._validate_global_episodes()
    self._validate_manifest_statistics(tar_paths)
    return self._report()

  def _read_manifest(self) -> None:
    if not self.manifest_path.is_file():
      self.issue(
        "manifest.missing",
        self._relative(self.manifest_path),
        "dataset_manifest.json is required to prove the fixed 30 fps export",
      )
      return
    if self.manifest_path.is_symlink():
      self.issue(
        "manifest.symlink",
        self._relative(self.manifest_path),
        "dataset_manifest.json must not be a symbolic link",
      )
      return
    try:
      value = json.loads(self.manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
      self.issue(
        "manifest.invalid_json",
        self._relative(self.manifest_path),
        f"cannot read JSON: {error}",
      )
      return
    if not isinstance(value, dict):
      self.issue(
        "manifest.not_object",
        self._relative(self.manifest_path),
        "top-level manifest value must be an object",
      )
      return
    self.manifest = value
    location = self._relative(self.manifest_path)

    fps = value.get("fps")
    if not _is_number(fps) or not math.isclose(float(fps), EXPECTED_FPS):
      self.issue("manifest.fps", location, f"fps must equal {EXPECTED_FPS:g}")

    cameras = value.get("cameras")
    parsed_cameras = self._parse_cameras(cameras)
    if parsed_cameras is None:
      self.issue(
        "manifest.cameras",
        location,
        (
          "cameras must be a canonical subset of "
          f"{list(CANONICAL_CAMERA_ORDER)!r} containing head"
        ),
      )
    else:
      self.expected_cameras = parsed_cameras
      self.expected_lowdim_dim = BASE_LOWDIM_DIM + CAMERA_BLOCK_DIM * len(
        parsed_cameras
      )
      self.expected_suffixes = self._member_order(parsed_cameras)

    lowdim_dim = value.get("lowdim_dim")
    if not _is_int(lowdim_dim) or lowdim_dim != self.expected_lowdim_dim:
      self.issue(
        "manifest.lowdim_dim",
        location,
        f"lowdim_dim must equal {self.expected_lowdim_dim}",
      )

    image_size = value.get("image_size")
    if (
      not isinstance(image_size, list)
      or len(image_size) != 2
      or not all(_is_int(part) and part > 0 for part in image_size)
    ):
      self.issue(
        "manifest.image_size",
        location,
        "image_size must be [width, height] with positive integers",
      )
    else:
      self.expected_image_size = (image_size[0], image_size[1])

    schema_version = value.get("schema_version")
    if schema_version != EXPECTED_SCHEMA_VERSION:
      self.issue(
        "manifest.schema_version",
        location,
        f"schema_version must equal {EXPECTED_SCHEMA_VERSION!r}",
      )

    expected_values = {
      "action_alignment": "next_30hz_frame",
      "depth_included": False,
      "lowdim_dtype": "float32",
      "member_order": list(self.expected_suffixes),
      "sample_key": EXPECTED_SAMPLE_KEY,
      "tactile_included": False,
    }
    for field, expected in expected_values.items():
      if value.get(field) != expected:
        self.issue(
          f"manifest.{field}",
          location,
          f"{field} must equal {expected!r}",
        )

    dataset_name = value.get("dataset_name")
    if not isinstance(dataset_name, str) or not dataset_name.strip():
      self.issue(
        "manifest.dataset_name", location, "dataset_name must be a non-empty string"
      )
    instruction = value.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
      self.issue(
        "manifest.instruction", location, "instruction must be a non-empty string"
      )

    image_encoding = value.get("image_encoding")
    if not isinstance(image_encoding, dict) or image_encoding.get("format") != "JPEG":
      self.issue(
        "manifest.image_encoding",
        location,
        "image_encoding.format must equal 'JPEG'",
      )

    splits = value.get("splits")
    if not isinstance(splits, dict) or not splits:
      self.issue(
        "manifest.splits",
        location,
        "splits must be a non-empty object",
      )

    sources = value.get("sources")
    if not isinstance(sources, list) or not sources:
      self.issue(
        "manifest.sources",
        location,
        "sources must be a non-empty list with per-episode resampling counts",
      )

  def _validate_shard(self, tar_path: Path) -> ShardRecord | None:
    relative_path = self._relative(tar_path)
    split = self._split_for(tar_path)
    try:
      archive = tarfile.open(tar_path, mode="r:*")
    except (OSError, tarfile.TarError) as error:
      self.issue("tar.unreadable", relative_path, str(error))
      return None

    samples: list[SampleRecord] = []
    episode_order: list[tuple[str, int]] = []
    seen_names: set[str] = set()
    seen_prefixes: set[str] = set()
    current_prefix: str | None = None
    current_members: list[tuple[str, tarfile.TarInfo]] = []

    try:
      for member in archive.getmembers():
        member_location = f"{relative_path}:{member.name}"
        if not _safe_tar_name(member.name):
          self.issue(
            "tar.unsafe_name",
            member_location,
            "member must be a top-level relative POSIX path without traversal",
          )
          continue
        if not member.isreg() or member.sparse is not None:
          self.issue(
            "tar.non_regular_member",
            member_location,
            "only non-sparse regular files are permitted",
          )
          continue
        if member.name in seen_names:
          self.issue("tar.duplicate_member", member_location, "duplicate member name")
          continue
        seen_names.add(member.name)

        match = SAMPLE_RE.fullmatch(member.name)
        if match is None:
          self.issue(
            "tar.unexpected_member",
            member_location,
            "member does not match episode_NNNNNN_frame_NNNNNN.<field>",
          )
          continue
        prefix = match.group("prefix")
        suffix = match.group("suffix")
        if member.size <= 0 or member.size > MAX_MEMBER_BYTES[suffix]:
          self.issue(
            "tar.member_size",
            member_location,
            f"invalid {suffix} size {member.size} bytes",
          )

        if current_prefix is None:
          current_prefix = prefix
        elif prefix != current_prefix:
          record = self._finish_sample(
            archive,
            relative_path,
            split,
            current_prefix,
            current_members,
          )
          if record is not None:
            samples.append(record)
          seen_prefixes.add(current_prefix)
          if prefix in seen_prefixes:
            self.issue(
              "tar.noncontiguous_sample",
              member_location,
              f"members for {prefix} are not contiguous",
            )
          current_prefix = prefix
          current_members = []
        current_members.append((suffix, member))

      if current_prefix is not None:
        record = self._finish_sample(
          archive,
          relative_path,
          split,
          current_prefix,
          current_members,
        )
        if record is not None:
          samples.append(record)
    except (OSError, tarfile.TarError) as error:
      self.issue("tar.read_error", relative_path, str(error))
    finally:
      archive.close()

    current_episode: tuple[str, int] | None = None
    closed_episodes: set[tuple[str, int]] = set()
    expected_frame = 0
    episode_signature: tuple[str, tuple[str, ...]] | None = None
    for sample in samples:
      episode_key = (sample.dataset_name, sample.episode_index)
      if episode_key != current_episode:
        if current_episode is not None:
          closed_episodes.add(current_episode)
        if episode_key in closed_episodes:
          self.issue(
            "episode.noncontiguous",
            f"{relative_path}:{sample.prefix}",
            f"episode {episode_key!r} appears in multiple tar blocks",
          )
        current_episode = episode_key
        episode_order.append(episode_key)
        expected_frame = 0
        episode_signature = (sample.instruction_signature, sample.cameras)
      if sample.frame_index != expected_frame:
        self.issue(
          "episode.frame_sequence",
          f"{relative_path}:{sample.prefix}",
          f"expected frame {expected_frame:06d}, got {sample.frame_index:06d}",
        )
        expected_frame = sample.frame_index
      expected_frame += 1
      signature = (sample.instruction_signature, sample.cameras)
      if episode_signature is not None and signature != episode_signature:
        self.issue(
          "episode.metadata_changed",
          f"{relative_path}:{sample.prefix}",
          "instruction or cameras changed inside one episode",
        )

    return ShardRecord(
      relative_path=relative_path,
      split=split,
      samples=tuple(samples),
      episode_keys=tuple(dict.fromkeys(episode_order)),
    )

  def _finish_sample(
    self,
    archive: tarfile.TarFile,
    shard: str,
    split: str,
    prefix: str,
    members: list[tuple[str, tarfile.TarInfo]],
  ) -> SampleRecord | None:
    location = f"{shard}:{prefix}"
    suffixes = tuple(suffix for suffix, _ in members)
    if suffixes != self.expected_suffixes:
      self.issue(
        "sample.members",
        location,
        (
          f"expected exactly {self.expected_suffixes!r} in that order, "
          f"got {suffixes!r}"
        ),
      )
    by_suffix: dict[str, tarfile.TarInfo] = {}
    for suffix, member in members:
      if suffix in by_suffix:
        self.issue(
          "sample.duplicate_field",
          f"{shard}:{member.name}",
          f"sample has more than one {suffix}",
        )
      else:
        by_suffix[suffix] = member
    if any(suffix not in by_suffix for suffix in self.expected_suffixes):
      return None

    contents: dict[str, bytes] = {}
    for suffix in self.expected_suffixes:
      member = by_suffix[suffix]
      extracted = archive.extractfile(member)
      if extracted is None:
        self.issue(
          "sample.unreadable_field",
          f"{shard}:{member.name}",
          "regular member could not be read",
        )
        return None
      try:
        contents[suffix] = extracted.read(MAX_MEMBER_BYTES[suffix] + 1)
      except (OSError, tarfile.TarError) as error:
        self.issue("sample.unreadable_field", f"{shard}:{member.name}", str(error))
        return None
      finally:
        extracted.close()
      if len(contents[suffix]) != member.size:
        self.issue(
          "sample.truncated_field",
          f"{shard}:{member.name}",
          f"header says {member.size} bytes, read {len(contents[suffix])}",
        )
        return None

    match = SAMPLE_RE.fullmatch(f"{prefix}.image.jpg")
    if match is None:
      self.issue("sample.bad_prefix", location, "invalid sample prefix")
      return None
    filename_episode = int(match.group("episode"))
    frame_index = int(match.group("frame"))
    meta = self._validate_meta(
      contents["meta.json"],
      location,
      filename_episode=filename_episode,
    )
    dataset_name = ""
    instruction_signature = ""
    cameras = self.expected_cameras
    if meta is not None:
      dataset_name = meta.get("dataset_name", "")
      instruction_signature = json.dumps(
        meta.get("instruction"), ensure_ascii=False, sort_keys=True
      )
      cameras = tuple(meta.get("cameras", ["head"]))
    image_sizes = {}
    for camera in cameras:
      suffix = self._camera_image_member(camera)
      if suffix in contents:
        image_sizes[camera] = self._validate_jpeg(
          contents[suffix], f"{location}:{suffix}"
        )
    image_size = image_sizes.get("head")
    lowdim = self._validate_lowdim(
      contents["lowdim.npy"], location, cameras, image_sizes
    )
    return SampleRecord(
      shard=shard,
      split=split,
      prefix=prefix,
      episode_index=filename_episode,
      frame_index=frame_index,
      dataset_name=dataset_name,
      instruction_signature=instruction_signature,
      cameras=cameras,
      lowdim=lowdim,
      image_size=image_size,
    )

  def _validate_jpeg(
    self, data: bytes, location: str
  ) -> tuple[int, int] | None:
    try:
      with Image.open(io.BytesIO(data)) as image:
        image.load()
        image_format = image.format
        mode = image.mode
        size = image.size
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as error:
      self.issue("image.invalid_jpeg", location, str(error))
      return None
    if image_format != "JPEG":
      self.issue("image.format", location, f"expected JPEG, got {image_format!r}")
    if mode != "RGB":
      self.issue("image.mode", location, f"expected RGB JPEG, got {mode!r}")
    if size[0] <= 0 or size[1] <= 0:
      self.issue("image.size", location, f"invalid image size {size!r}")
      return None
    if self.observed_image_size is None:
      self.observed_image_size = size
    elif size != self.observed_image_size:
      self.issue(
        "image.inconsistent_size",
        location,
        f"expected dataset size {self.observed_image_size!r}, got {size!r}",
      )
    if self.expected_image_size is not None and size != self.expected_image_size:
      self.issue(
        "image.manifest_size_mismatch",
        location,
        f"manifest says {self.expected_image_size!r}, JPEG is {size!r}",
      )
    return size

  def _validate_lowdim(
    self,
    data: bytes,
    location: str,
    cameras: tuple[str, ...],
    image_sizes: dict[str, tuple[int, int] | None],
  ) -> np.ndarray | None:
    stream = io.BytesIO(data)
    try:
      value = np.load(stream, allow_pickle=False)
    except (OSError, ValueError, EOFError) as error:
      self.issue("lowdim.invalid_npy", location, str(error))
      return None
    if stream.tell() != len(data):
      self.issue(
        "lowdim.trailing_bytes",
        location,
        f"NPY decoder consumed {stream.tell()} of {len(data)} bytes",
      )
    if value.dtype != np.dtype(np.float32):
      self.issue(
        "lowdim.dtype",
        location,
        f"expected native float32, got {value.dtype}",
      )
      return None
    expected_dim = BASE_LOWDIM_DIM + CAMERA_BLOCK_DIM * len(cameras)
    if value.shape != (expected_dim,):
      self.issue(
        "lowdim.shape",
        location,
        f"expected ({expected_dim},), got {value.shape!r}",
      )
      return None
    if not np.isfinite(value).all():
      bad = np.flatnonzero(~np.isfinite(value))
      self.issue(
        "lowdim.nonfinite",
        location,
        f"non-finite values at indices {bad[:8].tolist()}",
      )
      return None

    for label, start in (
      ("wrist_state.left", 6),
      ("wrist_state.right", 12),
      ("wrist_action.left", 54),
      ("wrist_action.right", 60),
    ):
      self._validate_rot6d(value[start : start + 6], location, label)

    for camera_index, camera in enumerate(cameras):
      offset = BASE_LOWDIM_DIM + CAMERA_BLOCK_DIM * camera_index
      self._validate_camera_calibration(
        value[offset : offset + CAMERA_BLOCK_DIM],
        location,
        camera,
        image_sizes.get(camera),
      )
    return value

  def _validate_camera_calibration(
    self,
    value: np.ndarray,
    location: str,
    camera: str,
    image_size: tuple[int, int] | None,
  ) -> None:
    extrinsic = value[:16].reshape(4, 4).astype(np.float64)
    rotation = extrinsic[:3, :3]
    if not np.allclose(
      extrinsic[3],
      np.array([0.0, 0.0, 0.0, 1.0]),
      atol=self.numeric_atol,
      rtol=0.0,
    ):
      self.issue(
        "extrinsic.last_row",
        location,
        f"{camera} world-to-camera last row is {extrinsic[3].tolist()!r}",
      )
    if not np.allclose(
      rotation.T @ rotation,
      np.eye(3),
      atol=self.numeric_atol,
      rtol=0.0,
    ):
      self.issue(
        "extrinsic.rotation_orthonormal",
        location,
        f"{camera} world-to-camera rotation is not orthonormal",
      )
    determinant = float(np.linalg.det(rotation))
    if not math.isclose(determinant, 1.0, abs_tol=self.numeric_atol):
      self.issue(
        "extrinsic.rotation_determinant",
        location,
        f"{camera} world-to-camera rotation determinant is {determinant:.8g}",
      )

    fx, fy, cx, cy = (float(part) for part in value[16:20])
    if fx <= 0.0 or fy <= 0.0:
      self.issue(
        "intrinsic.focal_length", location, f"{camera} invalid fx/fy: {fx}, {fy}"
      )
    if image_size is not None:
      width, height = image_size
      if not 0.0 <= cx < width or not 0.0 <= cy < height:
        self.issue(
          "intrinsic.principal_point",
          location,
          f"{camera} principal point ({cx}, {cy}) is outside {width}x{height}",
        )

  def _validate_rot6d(
    self, value: np.ndarray, location: str, label: str
  ) -> None:
    first = value[:3].astype(np.float64)
    second = value[3:].astype(np.float64)
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    dot = float(first @ second)
    if not math.isclose(first_norm, 1.0, abs_tol=self.numeric_atol):
      self.issue(
        "rot6d.first_norm",
        location,
        f"{label} first column norm is {first_norm:.8g}",
      )
    if not math.isclose(second_norm, 1.0, abs_tol=self.numeric_atol):
      self.issue(
        "rot6d.second_norm",
        location,
        f"{label} second column norm is {second_norm:.8g}",
      )
    if not math.isclose(dot, 0.0, abs_tol=self.numeric_atol):
      self.issue(
        "rot6d.columns_not_orthogonal",
        location,
        f"{label} column dot product is {dot:.8g}",
      )

  def _validate_meta(
    self,
    data: bytes,
    location: str,
    *,
    filename_episode: int,
  ) -> dict[str, Any] | None:
    try:
      value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
      self.issue("meta.invalid_json", location, str(error))
      return None
    if not isinstance(value, dict):
      self.issue("meta.not_object", location, "meta.json must contain an object")
      return None

    instruction = value.get("instruction")
    instruction_num = value.get("instruction_num")
    if isinstance(instruction, str):
      valid_instruction = bool(instruction.strip())
      expected_instruction_num = 1
    elif isinstance(instruction, list):
      valid_instruction = bool(instruction) and all(
        isinstance(item, str) and bool(item.strip()) for item in instruction
      )
      expected_instruction_num = len(instruction)
    else:
      valid_instruction = False
      expected_instruction_num = None
    if not valid_instruction:
      self.issue(
        "meta.instruction",
        location,
        "instruction must be a non-empty string or list of non-empty strings",
      )
    if not _is_int(instruction_num) or instruction_num <= 0:
      self.issue(
        "meta.instruction_num",
        location,
        "instruction_num must be a positive integer",
      )
    elif expected_instruction_num is not None and instruction_num != expected_instruction_num:
      self.issue(
        "meta.instruction_num_mismatch",
        location,
        f"instruction_num={instruction_num}, expected {expected_instruction_num}",
      )

    episode_index = value.get("episode_index")
    if not _is_int(episode_index) or episode_index < 0:
      self.issue(
        "meta.episode_index",
        location,
        "episode_index must be a non-negative integer",
      )
    elif episode_index != filename_episode:
      self.issue(
        "meta.episode_mismatch",
        location,
        f"filename episode is {filename_episode}, metadata says {episode_index}",
      )

    dataset_name = value.get("dataset_name", "")
    if not isinstance(dataset_name, str):
      self.issue("meta.dataset_name", location, "dataset_name must be a string")
      value["dataset_name"] = ""

    cameras = value.get("cameras", ["head"])
    if cameras != list(self.expected_cameras):
      self.issue(
        "meta.cameras",
        location,
        f"manifest requires cameras={list(self.expected_cameras)!r}",
      )
      value["cameras"] = list(self.expected_cameras)

    if "high_quality" in value:
      high_quality = value["high_quality"]
      if not _is_int(high_quality) or high_quality not in (0, 1):
        self.issue("meta.high_quality", location, "high_quality must be 0 or 1")
    return value

  def _validate_global_episodes(self) -> None:
    locations: dict[tuple[str, int], str] = {}
    episodes: dict[tuple[str, int], list[SampleRecord]] = {}
    for shard in self.shards:
      for episode_key in shard.episode_keys:
        previous = locations.get(episode_key)
        if previous is not None and previous != shard.relative_path:
          self.issue(
            "episode.crosses_shards",
            shard.relative_path,
            f"episode {episode_key!r} also appears in {previous}",
          )
        else:
          locations[episode_key] = shard.relative_path
      for sample in shard.samples:
        episodes.setdefault((sample.dataset_name, sample.episode_index), []).append(sample)

    for episode_key, samples in episodes.items():
      samples.sort(key=lambda sample: sample.frame_index)
      for current, following in zip(samples, samples[1:], strict=False):
        if following.frame_index != current.frame_index + 1:
          continue
        if current.lowdim is None or following.lowdim is None:
          continue
        if not np.allclose(
          current.lowdim[48:96],
          following.lowdim[0:48],
          atol=self.numeric_atol,
          rtol=0.0,
        ):
          maximum_error = float(
            np.max(np.abs(current.lowdim[48:96] - following.lowdim[0:48]))
          )
          self.issue(
            "action.not_next_state",
            f"{current.shard}:{current.prefix}",
            (
              f"action does not equal frame {following.frame_index:06d} state "
              f"for episode {episode_key!r}; max abs error={maximum_error:.8g}"
            ),
          )

  def _validate_manifest_statistics(self, tar_paths: list[Path]) -> None:
    if self.manifest is None:
      return
    location = self._relative(self.manifest_path)
    splits_value = self.manifest.get("splits")
    if not isinstance(splits_value, dict):
      return

    actual_by_split: dict[str, list[ShardRecord]] = {}
    for shard in self.shards:
      actual_by_split.setdefault(shard.split, []).append(shard)
    declared_splits = set(splits_value)
    actual_splits = set(actual_by_split)
    declared_nonempty_splits = {
      split
      for split, value in splits_value.items()
      if isinstance(value, dict)
      and any(value.get(field) for field in ("episodes", "samples", "shards"))
    }
    if declared_nonempty_splits != actual_splits:
      self.issue(
        "manifest.split_set",
        location,
        (
          f"declared non-empty splits {sorted(declared_nonempty_splits)!r}, "
          f"actual {sorted(actual_splits)!r}"
        ),
      )

    for split in sorted(declared_splits | actual_splits):
      actual_shards = actual_by_split.get(split, [])
      split_location = f"{location}:splits.{split}"
      declared = splits_value.get(split)
      if not isinstance(declared, dict):
        self.issue("manifest.split", split_location, "split entry must be an object")
        continue
      actual_episode_keys = {
        episode_key for shard in actual_shards for episode_key in shard.episode_keys
      }
      actual_samples = sum(len(shard.samples) for shard in actual_shards)
      self._check_integer_stat(
        declared, "episodes", len(actual_episode_keys), split_location
      )
      self._check_integer_stat(declared, "samples", actual_samples, split_location)
      expected_episode_indexes = [
        episode_index
        for shard in actual_shards
        for _, episode_index in shard.episode_keys
      ]
      if declared.get("episode_indices") != expected_episode_indexes:
        self.issue(
          "manifest.split_episode_indexes",
          split_location,
          (
            f"episode_indices must equal {expected_episode_indexes!r}, "
            f"got {declared.get('episode_indices')!r}"
          ),
        )

      declared_shards = declared.get("shards")
      if not isinstance(declared_shards, list):
        self.issue(
          "manifest.shards",
          split_location,
          "shards must be a list of per-shard objects",
        )
        continue
      actual_lookup = {shard.relative_path: shard for shard in actual_shards}
      declared_paths: set[str] = set()
      for index, entry in enumerate(declared_shards):
        entry_location = f"{split_location}:shards[{index}]"
        if not isinstance(entry, dict):
          self.issue("manifest.shard", entry_location, "shard entry must be an object")
          continue
        path_value = entry.get("path")
        normalized = _normalize_manifest_path(path_value, split)
        if normalized is None:
          self.issue(
            "manifest.shard_path",
            entry_location,
            "path must be a safe relative .tar path",
          )
          continue
        if normalized in declared_paths:
          self.issue("manifest.duplicate_shard", entry_location, normalized)
        declared_paths.add(normalized)
        actual = actual_lookup.get(normalized)
        if actual is None:
          self.issue(
            "manifest.missing_shard",
            entry_location,
            f"declared shard {normalized!r} was not observed",
          )
          continue
        self._check_shard_episode_stat(entry, actual, entry_location)
        self._check_integer_stat(entry, "samples", len(actual.samples), entry_location)
        self._check_sha256(entry, self.root / normalized, entry_location)
      if declared_paths != set(actual_lookup):
        self.issue(
          "manifest.shard_set",
          split_location,
          f"declared {sorted(declared_paths)!r}, actual {sorted(actual_lookup)!r}",
        )

    manifest_tar_paths = {
      shard.relative_path for records in actual_by_split.values() for shard in records
    }
    physical_tar_paths = {self._relative(path) for path in tar_paths}
    if manifest_tar_paths != physical_tar_paths:
      self.issue(
        "manifest.unreadable_tar_set",
        location,
        (
          f"readable shards {sorted(manifest_tar_paths)!r}, "
          f"physical shards {sorted(physical_tar_paths)!r}"
        ),
      )

    self._validate_source_statistics()

  def _validate_source_statistics(self) -> None:
    assert self.manifest is not None
    sources = self.manifest.get("sources")
    if not isinstance(sources, list):
      return
    location = f"{self._relative(self.manifest_path)}:sources"
    actual: dict[int, tuple[str, int]] = {}
    duplicate_indexes: set[int] = set()
    for shard in self.shards:
      counts: dict[int, int] = {}
      for sample in shard.samples:
        counts[sample.episode_index] = counts.get(sample.episode_index, 0) + 1
      for episode_index, count in counts.items():
        if episode_index in actual:
          duplicate_indexes.add(episode_index)
        actual[episode_index] = (shard.split, count)

    declared: dict[int, tuple[str, int]] = {}
    for index, source in enumerate(sources):
      source_location = f"{location}[{index}]"
      if not isinstance(source, dict):
        self.issue("manifest.source", source_location, "source must be an object")
        continue
      episode_index = source.get("episode_index")
      split = source.get("split")
      exported = source.get("exported_samples")
      strict_prefix = source.get("strict_prefix_frames")
      source_frames = source.get("source_frames")
      if not _is_int(episode_index) or episode_index < 0:
        self.issue(
          "manifest.source_episode", source_location, "invalid episode_index"
        )
        continue
      if not isinstance(split, str) or not split:
        self.issue("manifest.source_split", source_location, "invalid split")
        split = ""
      if not _is_int(exported) or exported <= 0:
        self.issue(
          "manifest.source_exported", source_location, "exported_samples must be > 0"
        )
        exported = -1
      if not _is_int(strict_prefix) or strict_prefix <= 1:
        self.issue(
          "manifest.source_prefix",
          source_location,
          "strict_prefix_frames must be an integer greater than one",
        )
      elif exported >= 0 and strict_prefix != exported + 1:
        self.issue(
          "manifest.source_action_tail",
          source_location,
          (
            f"strict_prefix_frames={strict_prefix} must equal "
            f"exported_samples+1={exported + 1}"
          ),
        )
      if (
        not _is_int(source_frames)
        or not _is_int(strict_prefix)
        or source_frames < strict_prefix
      ):
        self.issue(
          "manifest.source_frames",
          source_location,
          "source_frames must be >= strict_prefix_frames",
        )
      off_grid = source.get("off_grid_frames_discarded")
      if (
        not _is_int(off_grid)
        or not _is_int(source_frames)
        or not _is_int(strict_prefix)
        or off_grid != source_frames - strict_prefix
      ):
        self.issue(
          "manifest.source_off_grid",
          source_location,
          "off_grid_frames_discarded must equal source_frames-strict_prefix_frames",
        )
      maximum_grid_error = source.get("maximum_grid_error_seconds")
      if (
        not _is_number(maximum_grid_error)
        or not math.isfinite(float(maximum_grid_error))
        or not 0.0 <= float(maximum_grid_error) < 0.5 / EXPECTED_FPS
      ):
        self.issue(
          "manifest.source_grid_error",
          source_location,
          "maximum_grid_error_seconds must be finite and below half a 30 Hz period",
        )
      for hash_field in ("hdf5_sha256", "model_sha256"):
        digest = source.get(hash_field)
        if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
          self.issue(
            f"manifest.source_{hash_field}",
            source_location,
            f"{hash_field} must be 64 lowercase hex digits",
          )
      if episode_index in declared:
        self.issue(
          "manifest.duplicate_source_episode",
          source_location,
          f"episode {episode_index} occurs more than once",
        )
      declared[episode_index] = (split, exported)
      observed = actual.get(episode_index)
      if observed is None:
        self.issue(
          "manifest.source_missing_episode",
          source_location,
          f"episode {episode_index} was not found in shards",
        )
      elif observed != (split, exported):
        self.issue(
          "manifest.source_count",
          source_location,
          f"declared {(split, exported)!r}, actual {observed!r}",
        )

    if duplicate_indexes:
      self.issue(
        "manifest.ambiguous_episode_indexes",
        location,
        f"episode indexes occur in multiple shards: {sorted(duplicate_indexes)!r}",
      )
    if set(declared) != set(actual):
      self.issue(
        "manifest.source_set",
        location,
        f"declared episodes {sorted(declared)!r}, actual {sorted(actual)!r}",
      )

  def _check_integer_stat(
    self,
    container: dict[str, Any],
    field: str,
    expected: int,
    location: str,
  ) -> None:
    value = container.get(field)
    if not _is_int(value) or value != expected:
      self.issue(
        f"manifest.{field}_count",
        location,
        f"{field} must equal {expected}, got {value!r}",
      )

  def _check_shard_episode_stat(
    self, entry: dict[str, Any], shard: ShardRecord, location: str
  ) -> None:
    value = entry.get("episodes")
    indexes = [episode_index for _, episode_index in shard.episode_keys]
    if entry.get("episode_indices") != indexes:
      self.issue(
        "manifest.episode_indexes",
        location,
        f"episode_indices must equal {indexes!r}, got {entry.get('episode_indices')!r}",
      )
    if _is_int(value):
      if value != len(shard.episode_keys):
        self.issue(
          "manifest.episodes_count",
          location,
          f"episodes must equal {len(shard.episode_keys)}, got {value}",
        )
      return
    if isinstance(value, list) and all(_is_int(item) for item in value):
      if value != indexes:
        self.issue(
          "manifest.episode_indexes",
          location,
          f"episode list must equal {indexes!r}, got {value!r}",
        )
      return
    self.issue(
      "manifest.episodes",
      location,
      "episodes must be the count or the ordered episode-index list",
    )

  def _check_sha256(
    self, entry: dict[str, Any], path: Path, location: str
  ) -> None:
    expected = entry.get("sha256")
    if not isinstance(expected, str) or SHA256_RE.fullmatch(expected) is None:
      self.issue("manifest.sha256", location, "sha256 must be 64 lowercase hex digits")
      return
    relative = self._relative(path)
    actual = self._sha256_cache.get(relative)
    if actual is None:
      try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
          for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        actual = digest.hexdigest()
        self._sha256_cache[relative] = actual
      except OSError as error:
        self.issue("manifest.sha256_unreadable", location, str(error))
        return
    if actual != expected:
      self.issue(
        "manifest.sha256_mismatch",
        location,
        f"declared {expected}, actual {actual}",
      )

  def _report(self) -> dict[str, Any]:
    samples = [sample for shard in self.shards for sample in shard.samples]
    episode_keys = {
      (sample.dataset_name, sample.episode_index) for sample in samples
    }
    split_stats: dict[str, dict[str, int]] = {}
    for shard in self.shards:
      stats = split_stats.setdefault(
        shard.split, {"shards": 0, "episodes": 0, "samples": 0}
      )
      stats["shards"] += 1
      stats["episodes"] += len(shard.episode_keys)
      stats["samples"] += len(shard.samples)
    checks = {
      "safe_tar_members": True,
      "exact_camera_members_per_frame": True,
      "float32_finite_lowdim": True,
      "rot6d_and_camera_calibration": True,
      "episode_layout": True,
      "next_frame_actions": True,
      "manifest_fixed_fps_and_counts": True,
      "jpeg_dimensions": True,
    }
    for issue in self.issues:
      for check in _checks_for_issue(issue.code):
        checks[check] = False
    if self.error_count > len(self.issues):
      # Detailed errors may have been truncated, so individual categories can
      # no longer be proven independently even though overall validity is kept.
      checks = {check: False for check in checks}
    return {
      "valid": self.error_count == 0,
      "dataset_root": str(self.root),
      "manifest": self._relative(self.manifest_path),
      "fps": self.manifest.get("fps") if self.manifest is not None else None,
      "image_size": (
        list(self.observed_image_size)
        if self.observed_image_size is not None
        else None
      ),
      "lowdim_dim": self.expected_lowdim_dim,
      "cameras": list(self.expected_cameras),
      "shards": len(self.shards),
      "episodes": len(episode_keys),
      "samples": len(samples),
      "splits": split_stats,
      "checks": checks,
      "error_count": self.error_count,
      "reported_error_count": len(self.issues),
      "errors_truncated": self.error_count > len(self.issues),
      "errors": [
        {
          "code": issue.code,
          "location": issue.location,
          "message": issue.message,
        }
        for issue in self.issues
      ],
    }

  def _relative(self, path: Path) -> str:
    try:
      return path.resolve().relative_to(self.root).as_posix()
    except ValueError:
      return str(path)

  def _split_for(self, path: Path) -> str:
    relative = path.resolve().relative_to(self.root)
    if len(relative.parts) < 2:
      return "root"
    return relative.parts[0]


def _is_int(value: object) -> bool:
  return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
  return isinstance(value, (int, float)) and not isinstance(value, bool)


def _safe_tar_name(name: str) -> bool:
  if not name or "\\" in name or "\x00" in name:
    return False
  path = PurePosixPath(name)
  return (
    not path.is_absolute()
    and len(path.parts) == 1
    and all(part not in ("", ".", "..") for part in path.parts)
  )


def _normalize_manifest_path(value: object, split: str) -> str | None:
  if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
    return None
  path = PurePosixPath(value)
  if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
    return None
  if path.suffix != ".tar":
    return None
  if len(path.parts) == 1:
    path = PurePosixPath(split) / path
  return path.as_posix()


def _checks_for_issue(code: str) -> tuple[str, ...]:
  if code.startswith("manifest.") or code.startswith("dataset."):
    return ("manifest_fixed_fps_and_counts",)
  if code.startswith("image."):
    return ("jpeg_dimensions",)
  if code.startswith("lowdim."):
    return ("float32_finite_lowdim",)
  if code.startswith(("rot6d.", "extrinsic.", "intrinsic.")):
    return ("rot6d_and_camera_calibration",)
  if code.startswith("episode."):
    return ("episode_layout",)
  if code.startswith("action."):
    return ("next_frame_actions",)
  if code.startswith("sample."):
    return ("exact_camera_members_per_frame",)
  if code.startswith("tar."):
    return ("safe_tar_members",)
  return tuple()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Strictly validate a fixed-camera EgoSteer WebDataset export."
  )
  parser.add_argument("dataset", type=Path, help="directory containing the manifest")
  parser.add_argument(
    "--max-errors",
    type=int,
    default=100,
    help="maximum detailed errors included in the JSON report",
  )
  parser.add_argument(
    "--numeric-atol",
    type=float,
    default=1e-5,
    help="absolute tolerance for geometry and next-state comparisons",
  )
  args = parser.parse_args(argv)
  if args.max_errors <= 0:
    parser.error("--max-errors must be positive")
  if not math.isfinite(args.numeric_atol) or args.numeric_atol <= 0.0:
    parser.error("--numeric-atol must be finite and positive")
  return args


def main(argv: Sequence[str] | None = None) -> None:
  args = _parse_args(argv)
  validator = DatasetValidator(
    args.dataset,
    max_reported_errors=args.max_errors,
    numeric_atol=args.numeric_atol,
  )
  report = validator.validate()
  print(json.dumps(report, indent=2, ensure_ascii=False))
  raise SystemExit(0 if report["valid"] else 1)


if __name__ == "__main__":
  main()
