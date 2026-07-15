#!/usr/bin/env python3
"""Fail-closed H100 -> workstation -> Columbus training-artifact mirror.

This program is intentionally independent of the training process.  It only
reads the H100 run directory and publishes immutable, byte-verified snapshots
through a workstation staging area to a protected Columbus storage root.  A
checkpoint is called recoverable only when the checkpoint itself and every
checkpoint referenced by its selection state are verified on both tiers.

The normal entry point is ``watch`` or ``once``.  The ``--internal-*`` entry
points are implementation details: the same source file is streamed to a
remote Python interpreter over SSH, so remote validation uses exactly the same
rules as local validation without installing or mutating code on either host.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import dataclasses
import datetime as dt
import fcntl
import getpass
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
CHECKPOINT_RE = re.compile(r"^steps_(0|[1-9][0-9]*)$")
EVAL_RE = re.compile(r"^step_([0-9]{8,})\.json$")
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,191}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

REQUIRED_CHECKPOINT_FILES = frozenset(
    {
        "model.safetensors",
        "optimizer.bin",
        "scheduler.bin",
        "trainer_state.json",
        *(f"random_states_{rank}.pkl" for rank in range(8)),
    }
)
OPTIONAL_CHECKPOINT_FILES = frozenset({"selection_state.json"})
ALLOWED_CHECKPOINT_FILES = REQUIRED_CHECKPOINT_FILES | OPTIONAL_CHECKPOINT_FILES

REQUIRED_RUN_EVIDENCE = frozenset(
    {
        "config.yaml",
        "config.json",
        "resolved_training_schedule.json",
        "dataset_statistics.json",
        "dataset_provenance.json",
        "heldout_eval_windows.json",
        "heldout_focused_eval_windows.json",
        "summary.jsonl",
    }
)
OPTIONAL_RUN_EVIDENCE = frozenset(
    {
        "best_checkpoint.json",
        "launch.env",
        "production_preflight_manifest.txt",
    }
)
MUTABLE_RUN_EVIDENCE = frozenset({"summary.jsonl", "best_checkpoint.json"})
RUN_EVIDENCE_DIRS: frozenset[str] = frozenset()

BASE_HELDOUT_REPORT_FIELDS = frozenset(
    {
    "schema_version",
    "purpose",
    "view",
    "algorithm",
    "observation_mode",
    "evaluation_video_offsets",
    "action_offset_range_inclusive",
    "frames_per_episode",
    "seed_sha256",
    "window_selection_sha256",
    "observation_count",
    "action_evaluable_observation_count",
    "action_dim",
    "valid_action_timestep_count",
    "valid_action_element_count",
    "subtask_observation_counts",
    "subtask_evaluable_observation_counts",
    "subtask_action_timestep_counts_by_horizon",
    "subtask_valid_action_element_counts_by_horizon",
    "zero_valid_action_episodes",
    "episode_split_provenance",
    "production_valid",
    "checkpoint_selection_eligible",
    "windows",
    }
)
FOCUSED_HELDOUT_REPORT_FIELDS = frozenset(
    {
    "open_to_close_transition_count_h10",
    "close_to_open_transition_count_h10",
    "open_to_close_transition_window_count_h10",
    "close_to_open_transition_window_count_h10",
    "arm_movement_element_count_h10",
    "arm_movement_hold_abs_sum_h10",
    "movement_threshold_normalized",
    "focused_subtasks",
    }
)
PROVENANCE_FIELDS = frozenset(
    {
        "dataset_name",
        "manifest_path",
        "manifest_sha256",
        "role",
        "selected_episode_count",
        "selected_episode_set_sha256",
        "selected_frame_count",
        "train_episode_count",
        "train_episode_set_sha256",
        "train_frame_count",
        "holdout_episode_count",
        "holdout_episode_set_sha256",
        "full_catalog_sha256",
        "train_statistics_path",
        "train_statistics_sha256",
    }
)

SAFE_COLUMBUS_PARENT = Path("/mnt/vla-jepa/h100-relay")
SAFE_COLUMBUS_BASE = SAFE_COLUMBUS_PARENT / "checkpoint-mirror-storage"
DEFAULT_DISK_RESERVE_BYTES = 100_000_000_000
DEFAULT_RETAIN = 3
DEFAULT_FULL_SCRUB_HOURS = 6.0
MIRROR_STATE_SCHEMA_VERSION = 4


class MirrorError(RuntimeError):
    """A validation, consistency, or transport invariant failed."""


def _metadata_fingerprint(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _light_stat(path: Path) -> dict[str, Any]:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return {"exists": False}
    if stat.S_ISREG(info.st_mode):
        kind = "file"
    elif stat.S_ISDIR(info.st_mode):
        kind = "directory"
    elif stat.S_ISLNK(info.st_mode):
        kind = "symlink"
    else:
        kind = "special"
    return {
        "exists": True,
        "kind": kind,
        "device": info.st_dev,
        "inode": info.st_ino,
        "size": info.st_size,
        "ctime_ns": info.st_ctime_ns,
        "mtime_ns": info.st_mtime_ns,
        "mode": stat.S_IMODE(info.st_mode),
        "nlink": info.st_nlink,
        "uid": info.st_uid,
        "gid": info.st_gid,
    }


def _light_directory(path: Path) -> dict[str, Any]:
    root_stat = _light_stat(path)
    if root_stat.get("kind") != "directory":
        return {"root": root_stat, "entries": {}}
    entries = {
        entry.name: _light_stat(path / entry.name)
        for entry in sorted(os.scandir(path), key=lambda item: item.name)
    }
    return {"root": root_stat, "entries": entries}


def lightweight_source_inventory(run_dir: Path | str) -> dict[str, Any]:
    """Stat-only source inventory; never opens or hashes artifact contents."""

    root = Path(run_dir)
    _validate_root_directory(root, label="source run")
    checkpoints: dict[str, Any] = {}
    unexpected_checkpoint_entries: list[str] = []
    checkpoint_root = root / "checkpoints"
    if checkpoint_root.exists() or checkpoint_root.is_symlink():
        _validate_root_directory(checkpoint_root, label="source checkpoints")
        for entry in sorted(os.scandir(checkpoint_root), key=lambda item: item.name):
            if CHECKPOINT_RE.fullmatch(entry.name):
                tree = _light_directory(checkpoint_root / entry.name)
                tree["fingerprint"] = _metadata_fingerprint(tree)
                checkpoints[entry.name] = tree
            elif not entry.name.startswith(".incoming-"):
                unexpected_checkpoint_entries.append(entry.name)
    evals: dict[str, Any] = {}
    unexpected_eval_entries: list[str] = []
    eval_root = root / "heldout_eval_metrics"
    if eval_root.exists() or eval_root.is_symlink():
        _validate_root_directory(eval_root, label="source eval directory")
        for entry in sorted(os.scandir(eval_root), key=lambda item: item.name):
            if EVAL_RE.fullmatch(entry.name):
                try:
                    eval_step_from_name(entry.name)
                except MirrorError:
                    unexpected_eval_entries.append(entry.name)
                else:
                    file_stat = _light_stat(eval_root / entry.name)
                    evals[entry.name] = {
                        **file_stat,
                        "fingerprint": _metadata_fingerprint(file_stat),
                    }
            elif not entry.name.startswith(".incoming-"):
                unexpected_eval_entries.append(entry.name)
    evidence: dict[str, Any] = {}
    for name in sorted(REQUIRED_RUN_EVIDENCE | OPTIONAL_RUN_EVIDENCE):
        file_stat = _light_stat(root / name)
        if file_stat.get("exists"):
            evidence[name] = file_stat
    evidence_fingerprint = _metadata_fingerprint(evidence)
    pointer = _light_stat(root / "best_checkpoint.json")
    return {
        "schema_version": SCHEMA_VERSION,
        "run_dir": str(root.resolve()),
        "run_root": _light_stat(root),
        "checkpoints": checkpoints,
        "unexpected_checkpoint_entries": unexpected_checkpoint_entries,
        "eval_artifacts": evals,
        "unexpected_eval_entries": unexpected_eval_entries,
        "evidence": evidence,
        "evidence_fingerprint": evidence_fingerprint,
        "best_checkpoint_pointer": {
            **pointer,
            "fingerprint": _metadata_fingerprint(pointer),
        },
    }


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_write(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    """Durably replace one local file without exposing partial contents."""

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temp_path = path.parent / f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    try:
        fd = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temp_path.unlink()


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _loads_json_object(payload_bytes: bytes, *, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str):
        raise ValueError(f"non-finite JSON constant: {value}")

    try:
        payload = json.loads(
            payload_bytes.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise MirrorError(f"{label} is not canonical JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise MirrorError(f"{label} must be a JSON object")
    return payload


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload_bytes = path.read_bytes()
    except OSError as exc:
        raise MirrorError(f"{label} is not readable: {path}: {exc}") from exc
    try:
        return _loads_json_object(payload_bytes, label=label)
    except MirrorError as exc:
        raise MirrorError(f"{exc}: {path}") from exc


def _strict_int(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise MirrorError(f"{label} must be an integer >= {minimum}; got {value!r}")
    return value


def _require_schema_version(value: Any, *, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != SCHEMA_VERSION:
        raise MirrorError(f"{label} must be the exact integer {SCHEMA_VERSION}; got {value!r}")


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MirrorError(f"{label} must be a finite number; got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise MirrorError(f"{label} must be finite; got {value!r}")
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _file_prefix_sha256(path: Path, length: int) -> dict[str, Any]:
    prefix_length = _strict_int(length, label="file prefix length")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_size < prefix_length
        ):
            raise MirrorError("append-only evidence file is unsafe or truncated")
        digest = hashlib.sha256()
        remaining = prefix_length
        while remaining:
            chunk = os.read(descriptor, min(8 * 1024 * 1024, remaining))
            if not chunk:
                raise MirrorError("append-only evidence file truncated during hashing")
            digest.update(chunk)
            remaining -= len(chunk)
        post = os.fstat(descriptor)
        if (
            not stat.S_ISREG(post.st_mode)
            or post.st_nlink != 1
            or post.st_size < prefix_length
        ):
            raise MirrorError("append-only evidence file changed unsafely during hashing")
        return {
            "size": post.st_size,
            "prefix_length": prefix_length,
            "prefix_sha256": digest.hexdigest(),
        }
    finally:
        os.close(descriptor)


def checkpoint_step_from_name(name: str) -> int:
    match = CHECKPOINT_RE.fullmatch(name)
    if match is None:
        raise MirrorError(f"checkpoint name must be steps_N with canonical N: {name!r}")
    return int(match.group(1))


def eval_step_from_name(name: str) -> int:
    match = EVAL_RE.fullmatch(name)
    if match is None:
        raise MirrorError(
            "eval artifact name must be the zero-padded canonical "
            f"step_NNNNNNNN.json form: {name!r}"
        )
    step_text = match.group(1)
    step = int(step_text)
    if name != f"step_{step:08d}.json":
        raise MirrorError(f"eval artifact step spelling is not canonical: {name!r}")
    return step


@dataclasses.dataclass(frozen=True)
class FileRecord:
    path: str
    size: int
    sha256: str

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class TreeManifest:
    kind: str
    files: tuple[FileRecord, ...]
    manifest_sha256: str
    total_size: int
    step: int | None = None
    selection_best_step: int | None = None
    selection_best_value: float | None = None
    checkpoint_schema: str | None = None

    @classmethod
    def create(
        cls,
        *,
        kind: str,
        records: Iterable[FileRecord],
        step: int | None = None,
        selection_best_step: int | None = None,
        selection_best_value: float | None = None,
        checkpoint_schema: str | None = None,
    ) -> "TreeManifest":
        ordered = tuple(sorted(records, key=lambda item: item.path.encode("utf-8")))
        if len({item.path for item in ordered}) != len(ordered):
            raise MirrorError("manifest contains duplicate relative paths")
        body: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "kind": kind,
            "files": [item.as_dict() for item in ordered],
        }
        if step is not None:
            body["step"] = step
        if selection_best_step is not None:
            body["selection_best_step"] = selection_best_step
            body["selection_best_value"] = _finite_number(
                selection_best_value, label="manifest selection_best_value"
            )
        elif selection_best_value is not None:
            raise MirrorError("manifest selection best step/value must be set together")
        if checkpoint_schema is not None:
            if kind != "checkpoint" or checkpoint_schema not in {"legacy", "selection_v1"}:
                raise MirrorError(f"invalid checkpoint schema marker: {checkpoint_schema!r}")
            body["checkpoint_schema"] = checkpoint_schema
        digest = hashlib.sha256(_canonical_json_bytes(body)).hexdigest()
        return cls(
            kind=kind,
            files=ordered,
            manifest_sha256=digest,
            total_size=sum(item.size for item in ordered),
            step=step,
            selection_best_step=selection_best_step,
            selection_best_value=selection_best_value,
            checkpoint_schema=checkpoint_schema,
        )

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "kind": self.kind,
            "files": [record.as_dict() for record in self.files],
            "manifest_sha256": self.manifest_sha256,
            "total_size": self.total_size,
        }
        if self.step is not None:
            payload["step"] = self.step
        if self.selection_best_step is not None:
            payload["selection_best_step"] = self.selection_best_step
            payload["selection_best_value"] = self.selection_best_value
        if self.checkpoint_schema is not None:
            payload["checkpoint_schema"] = self.checkpoint_schema
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TreeManifest":
        _require_schema_version(payload.get("schema_version"), label="manifest schema_version")
        kind = payload.get("kind")
        if not isinstance(kind, str) or not kind:
            raise MirrorError("manifest kind must be a non-empty string")
        expected_fields = {
            "schema_version",
            "kind",
            "files",
            "manifest_sha256",
            "total_size",
        }
        if "step" in payload:
            expected_fields.add("step")
        if "selection_best_step" in payload or "selection_best_value" in payload:
            expected_fields.update({"selection_best_step", "selection_best_value"})
        if "checkpoint_schema" in payload:
            expected_fields.add("checkpoint_schema")
        if set(payload) != expected_fields:
            raise MirrorError("manifest has incomplete or unexpected fields")
        raw_files = payload.get("files")
        if not isinstance(raw_files, list):
            raise MirrorError("manifest files must be a list")
        records: list[FileRecord] = []
        for index, raw in enumerate(raw_files):
            if not isinstance(raw, dict):
                raise MirrorError(f"manifest files[{index}] must be an object")
            if set(raw) != {"path", "size", "sha256"}:
                raise MirrorError(f"manifest files[{index}] has unexpected fields")
            path = raw.get("path")
            size = raw.get("size")
            sha256 = raw.get("sha256")
            if not isinstance(path, str) or not _safe_relative_path(path):
                raise MirrorError(f"manifest files[{index}].path is unsafe: {path!r}")
            size_int = _strict_int(size, label=f"manifest files[{index}].size", minimum=1)
            if not isinstance(sha256, str) or SHA256_RE.fullmatch(sha256) is None:
                raise MirrorError(f"manifest files[{index}].sha256 is invalid")
            records.append(FileRecord(path=path, size=size_int, sha256=sha256))
        raw_step = payload.get("step")
        step = None if raw_step is None else _strict_int(raw_step, label="manifest step")
        raw_best = payload.get("selection_best_step")
        best = (
            None
            if raw_best is None
            else _strict_int(raw_best, label="manifest selection_best_step")
        )
        raw_best_value = payload.get("selection_best_value")
        best_value = (
            None
            if raw_best_value is None
            else _finite_number(raw_best_value, label="manifest selection_best_value")
        )
        raw_checkpoint_schema = payload.get("checkpoint_schema")
        checkpoint_schema = (
            None if raw_checkpoint_schema is None else str(raw_checkpoint_schema)
        )
        rebuilt = cls.create(
            kind=kind,
            records=records,
            step=step,
            selection_best_step=best,
            selection_best_value=best_value,
            checkpoint_schema=checkpoint_schema,
        )
        if payload.get("manifest_sha256") != rebuilt.manifest_sha256:
            raise MirrorError("manifest_sha256 does not match canonical manifest contents")
        if payload.get("total_size") != rebuilt.total_size:
            raise MirrorError("manifest total_size does not match file records")
        return rebuilt

    def file(self, relative_path: str) -> FileRecord:
        for record in self.files:
            if record.path == relative_path:
                return record
        raise MirrorError(f"manifest does not contain {relative_path!r}")


def _safe_relative_path(path: str) -> bool:
    if (
        not path
        or "\x00" in path
        or "\\" in path
        or any(ord(character) < 32 or ord(character) == 127 for character in path)
    ):
        return False
    pure = PurePosixPath(path)
    return (
        not pure.is_absolute()
        and str(pure) == path
        and all(part not in {"", ".", ".."} for part in pure.parts)
    )


def _validate_root_directory(path: Path, *, label: str) -> os.stat_result:
    if path.is_absolute():
        current = Path(path.anchor)
        for part in path.parts[1:]:
            current = current / part
            try:
                component = current.lstat()
            except FileNotFoundError:
                break
            if stat.S_ISLNK(component.st_mode):
                raise MirrorError(f"{label} rejects symlinked ancestor: {current}")
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise MirrorError(f"{label} does not exist: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise MirrorError(f"{label} must be a real directory, not a link/special file: {path}")
    return info


def _regular_record(path: Path, *, relative_path: str, label: str) -> FileRecord:
    if not _safe_relative_path(relative_path):
        raise MirrorError(f"{label} has unsafe relative path {relative_path!r}")
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise MirrorError(f"{label} rejects symlink or special file: {path}")
    if info.st_nlink != 1:
        raise MirrorError(f"{label} rejects hard-linked file (nlink={info.st_nlink}): {path}")
    if info.st_size <= 0:
        raise MirrorError(f"{label} requires non-empty file: {path}")
    if not os.access(path, os.R_OK):
        raise MirrorError(f"{label} file is not readable: {path}")
    return FileRecord(
        path=relative_path,
        size=info.st_size,
        sha256=_sha256_file(path),
    )


@dataclasses.dataclass(frozen=True)
class SelectionPointer:
    metric_name: str
    metric_mode: str
    metric_value: float
    best_step: int
    checkpoint_relative_path: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "best_metric_name": self.metric_name,
            "best_metric_mode": self.metric_mode,
            "best_metric_value": self.metric_value,
            "best_metric_step": self.best_step,
            "checkpoint_relative_path": self.checkpoint_relative_path,
        }


def validate_selection_payload(
    payload: Mapping[str, Any],
    *,
    metric_name: str,
    metric_mode: str,
    maximum_step: int | None = None,
) -> SelectionPointer:
    expected_fields = {
        "schema_version",
        "best_metric_name",
        "best_metric_mode",
        "best_metric_value",
        "best_metric_step",
        "checkpoint_relative_path",
    }
    if set(payload) != expected_fields:
        raise MirrorError("selection state has incomplete or unexpected fields")
    _require_schema_version(
        payload.get("schema_version"), label="selection state schema_version"
    )
    if payload.get("best_metric_name") != metric_name:
        raise MirrorError("selection state best_metric_name does not match configured metric")
    if payload.get("best_metric_mode") != metric_mode or metric_mode not in {"min", "max"}:
        raise MirrorError("selection state best_metric_mode does not match configured mode")
    value = _finite_number(payload.get("best_metric_value"), label="best_metric_value")
    step = _strict_int(payload.get("best_metric_step"), label="best_metric_step")
    if maximum_step is not None and step > maximum_step:
        raise MirrorError(
            f"selection best step {step} is newer than containing checkpoint step {maximum_step}"
        )
    expected_path = f"checkpoints/steps_{step}"
    if payload.get("checkpoint_relative_path") != expected_path:
        raise MirrorError(
            "selection checkpoint_relative_path must exactly match best_metric_step; "
            f"expected {expected_path!r}"
        )
    return SelectionPointer(
        metric_name=metric_name,
        metric_mode=metric_mode,
        metric_value=value,
        best_step=step,
        checkpoint_relative_path=expected_path,
    )


def validate_checkpoint_tree(
    checkpoint_path: Path | str,
    *,
    metric_name: str,
    metric_mode: str,
    expected_step: int | None = None,
) -> TreeManifest:
    root = Path(checkpoint_path)
    _validate_root_directory(root, label="checkpoint")
    step = (
        checkpoint_step_from_name(root.name)
        if expected_step is None
        else _strict_int(expected_step, label="expected checkpoint step")
    )
    entries = list(os.scandir(root))
    names = {entry.name for entry in entries}
    missing = sorted(REQUIRED_CHECKPOINT_FILES - names)
    unexpected = sorted(names - ALLOWED_CHECKPOINT_FILES)
    if missing or unexpected:
        raise MirrorError(
            f"checkpoint contract mismatch at {root}: missing={missing}, unexpected={unexpected}"
        )
    # Validate cheap metadata and JSON finalization before hashing multi-GB
    # model/optimizer bytes.  A complete-shaped checkpoint caught during an
    # atomic metadata rewrite must remain a cheap pending poll.
    for entry in entries:
        path = root / entry.name
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise MirrorError(f"checkpoint rejects symlink or special file: {path}")
        if info.st_nlink != 1 or info.st_size <= 0 or not os.access(path, os.R_OK):
            raise MirrorError(f"checkpoint file metadata is unsafe/incomplete: {path}")

    trainer_state = _load_json_object(root / "trainer_state.json", label="trainer state")
    completed_steps = _strict_int(
        trainer_state.get("completed_steps"), label="trainer_state.completed_steps"
    )
    if completed_steps != step:
        raise MirrorError(
            f"trainer_state.completed_steps={completed_steps} does not match steps_{step}"
        )

    marker = trainer_state.get("selection_state_schema_version")
    checkpoint_schema = "legacy" if marker is None else "selection_v1"
    selection_best_step = None
    selection_best_value = None
    selection_path = root / "selection_state.json"
    if marker is None and selection_path.is_file():
        raise MirrorError(
            "legacy trainer_state cannot be combined with selection_state.json"
        )
    if marker is None and set(trainer_state) != {"completed_steps"}:
        raise MirrorError(
            "legacy trainer_state has partial/unexpected selection fields; "
            "production legacy parsing is intentionally rejected"
        )
    if marker is not None:
        _require_schema_version(
            marker, label="trainer_state.selection_state_schema_version"
        )
        if trainer_state.get("best_metric_name") != metric_name:
            raise MirrorError("trainer_state best_metric_name does not match configured metric")
        if trainer_state.get("best_metric_mode") != metric_mode:
            raise MirrorError("trainer_state best_metric_mode does not match configured mode")
        expected_trainer_keys = {
            "completed_steps",
            "selection_state_schema_version",
            "best_metric_name",
            "best_metric_mode",
            "best_metric_value",
            "best_metric_step",
        }
        if set(trainer_state) != expected_trainer_keys:
            raise MirrorError(
                "selection-v1 trainer_state.json has incomplete or unexpected fields"
            )
        if not selection_path.is_file():
            raise MirrorError(
                "trainer_state declares selection state schema 1 but "
                "selection_state.json is missing"
            )
    if selection_path.is_file():
        selection_payload = _load_json_object(selection_path, label="selection state")
        # Before the first eligible evaluation, schema-1 state contains explicit
        # null best fields.  That is valid and has no dependency closure yet.
        raw_best_step = selection_payload.get("best_metric_step")
        raw_best_value = selection_payload.get("best_metric_value")
        if raw_best_step is None and raw_best_value is None:
            if set(selection_payload) != {
                "schema_version",
                "best_metric_name",
                "best_metric_mode",
                "best_metric_value",
                "best_metric_step",
                "checkpoint_relative_path",
            }:
                raise MirrorError("empty selection state shape is invalid")
            _require_schema_version(
                selection_payload.get("schema_version"),
                label="empty selection state schema_version",
            )
            if selection_payload.get("best_metric_name") != metric_name:
                raise MirrorError("empty selection state metric name mismatch")
            if selection_payload.get("best_metric_mode") != metric_mode:
                raise MirrorError("empty selection state metric mode mismatch")
            if selection_payload.get("checkpoint_relative_path") is not None:
                raise MirrorError("empty selection state cannot point to a checkpoint")
        elif raw_best_step is None or raw_best_value is None:
            raise MirrorError("selection state must set or clear best step/value together")
        else:
            selection_pointer = validate_selection_payload(
                selection_payload,
                metric_name=metric_name,
                metric_mode=metric_mode,
                maximum_step=step,
            )
            selection_best_step = selection_pointer.best_step
            selection_best_value = selection_pointer.metric_value

    records = [
        _regular_record(root / entry.name, relative_path=entry.name, label="checkpoint")
        for entry in entries
    ]

    return TreeManifest.create(
        kind="checkpoint",
        records=records,
        step=step,
        selection_best_step=selection_best_step,
        selection_best_value=selection_best_value,
        checkpoint_schema=checkpoint_schema,
    )


@dataclasses.dataclass(frozen=True)
class EvalArtifact:
    step: int
    trainer_state_sha256: str
    metric_name: str
    metric_mode: str
    metric_value: float
    sha256: str
    size: int
    cryptographically_bound: bool
    production_eligible: bool
    archival_reason: str | None = None
    source_kind: str = "checkpoint"

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvalArtifact":
        required = {field.name for field in dataclasses.fields(cls)}
        if set(payload) != required:
            raise MirrorError("cached eval artifact shape is invalid")
        artifact = cls(
            step=_strict_int(payload["step"], label="cached eval step"),
            trainer_state_sha256=str(payload["trainer_state_sha256"]),
            metric_name=str(payload["metric_name"]),
            metric_mode=str(payload["metric_mode"]),
            metric_value=_finite_number(
                payload["metric_value"], label="cached eval metric value"
            ),
            sha256=str(payload["sha256"]),
            size=_strict_int(payload["size"], label="cached eval size", minimum=1),
            cryptographically_bound=payload["cryptographically_bound"] is True,
            production_eligible=payload["production_eligible"] is True,
            archival_reason=(
                None
                if payload["archival_reason"] is None
                else str(payload["archival_reason"])
            ),
            source_kind=str(payload["source_kind"]),
        )
        if SHA256_RE.fullmatch(artifact.sha256) is None or SHA256_RE.fullmatch(
            artifact.trainer_state_sha256
        ) is None:
            raise MirrorError("cached eval SHA-256 is invalid")
        if artifact.metric_mode not in {"min", "max"} or not artifact.metric_name:
            raise MirrorError("cached eval metric identity is invalid")
        if artifact.production_eligible and not artifact.cryptographically_bound:
            raise MirrorError("cached eval falsely claims production eligibility")
        if artifact.archival_reason is not None and artifact.production_eligible:
            raise MirrorError("cached archival eval cannot be production eligible")
        if artifact.source_kind not in {"checkpoint", "live_in_memory_model"}:
            raise MirrorError("cached eval source_kind is invalid")
        if artifact.production_eligible and artifact.source_kind != "checkpoint":
            raise MirrorError("non-checkpoint eval cannot be recoverable-selection eligible")
        return artifact


def validate_eval_artifact(
    artifact_path: Path | str,
    *,
    checkpoint_manifest: TreeManifest,
    metric_name: str,
    metric_mode: str,
    run_id: str,
    run_evidence: RunEvidenceIdentity,
    allow_legacy_archive: bool = False,
    expected_step: int | None = None,
) -> EvalArtifact:
    path = Path(artifact_path)
    record = _regular_record(path, relative_path=path.name, label="eval artifact")
    if expected_step is None:
        step = eval_step_from_name(path.name)
    else:
        step = _strict_int(expected_step, label="expected eval step")
        if path.name not in {
            f"step_{step:08d}.json",
            f".incoming-step_{step:08d}.json",
        }:
            raise MirrorError("staged eval filename does not match its expected step")
    if checkpoint_manifest.kind != "checkpoint" or checkpoint_manifest.step != step:
        raise MirrorError("eval artifact step is not bound to the supplied checkpoint manifest")
    payload = _load_json_object(path, label="eval artifact")
    expected_top_fields = {
        "schema_version",
        "checkpoint_step",
        "checkpoint_relative_path",
        "checkpoint",
        "run",
        "sampling_reports",
        "production_valid",
        "checkpoint_selection_eligible",
        "selection_metric",
        "metrics",
    }
    if set(payload) != expected_top_fields:
        raise MirrorError("eval artifact has incomplete or unexpected top-level fields")
    _require_schema_version(payload.get("schema_version"), label="eval artifact schema_version")
    if payload.get("checkpoint_step") != step:
        raise MirrorError("eval artifact checkpoint_step does not match filename")
    if payload.get("checkpoint_relative_path") != f"checkpoints/steps_{step}":
        raise MirrorError("eval artifact checkpoint_relative_path is not canonical")
    checkpoint = payload.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise MirrorError("eval artifact checkpoint evidence must be an object")
    modern_checkpoint_fields = {
        "step",
        "source_path",
        "source_kind",
        "trainer_state_sha256",
        "model_file",
        "model_file_size_bytes",
        "model_file_sha256",
    }
    legacy_checkpoint_fields = modern_checkpoint_fields - {"model_file_sha256"}
    checkpoint_fields = set(checkpoint)
    cryptographically_bound = checkpoint_fields == modern_checkpoint_fields
    if not cryptographically_bound and checkpoint_fields != legacy_checkpoint_fields:
        raise MirrorError("eval artifact checkpoint identity shape is invalid")
    if not cryptographically_bound and not allow_legacy_archive:
        raise MirrorError(
            "eval artifact lacks checkpoint.model_file_sha256 and is archival-only"
        )
    if checkpoint.get("step") != step or checkpoint.get("source_kind") != "checkpoint":
        raise MirrorError("eval artifact checkpoint identity is invalid")
    expected_source_path = f"{run_evidence.source_output_dir}/checkpoints/steps_{step}"
    if checkpoint.get("source_path") != expected_source_path:
        raise MirrorError("eval artifact checkpoint source_path does not match run evidence")
    expected_trainer_sha = checkpoint_manifest.file("trainer_state.json").sha256
    if checkpoint.get("trainer_state_sha256") != expected_trainer_sha:
        raise MirrorError(
            "eval artifact trainer_state_sha256 does not bind to checkpoint bytes"
        )
    expected_model = checkpoint_manifest.file("model.safetensors")
    if checkpoint.get("model_file") != "model.safetensors":
        raise MirrorError("eval artifact model_file is not model.safetensors")
    if checkpoint.get("model_file_size_bytes") != expected_model.size:
        raise MirrorError("eval artifact model size does not bind to checkpoint bytes")
    if cryptographically_bound and checkpoint.get("model_file_sha256") != expected_model.sha256:
        raise MirrorError("eval artifact model_file_sha256 does not bind to checkpoint bytes")
    run = payload.get("run")
    if not isinstance(run, dict) or set(run) != {
        "run_id",
        "output_dir",
        "seed",
        "config_path",
        "config_sha256",
        "resolved_training_schedule",
        "source_training_config",
    }:
        raise MirrorError("eval artifact run identity shape is invalid")
    if run_id != run_evidence.run_id or run.get("run_id") != run_evidence.run_id:
        raise MirrorError("eval artifact run_id does not match the mirrored run")
    if run.get("output_dir") != run_evidence.source_output_dir:
        raise MirrorError("eval artifact output_dir does not match mirrored evidence")
    if run.get("seed") != run_evidence.seed or isinstance(run.get("seed"), bool):
        raise MirrorError("eval artifact run seed does not match config evidence")
    if (
        run.get("config_path") != "config.yaml"
        or run.get("config_sha256") != run_evidence.config_sha256
    ):
        raise MirrorError("eval artifact config identity does not match mirrored evidence")
    if run.get("resolved_training_schedule") != {
        "path": "resolved_training_schedule.json",
        "sha256": run_evidence.schedule_sha256,
    }:
        raise MirrorError("eval artifact schedule identity does not match mirrored evidence")
    if run.get("source_training_config") is not None:
        raise MirrorError("live training eval artifact claims eval-only source config")
    production_valid = payload.get("production_valid")
    selection_eligible = payload.get("checkpoint_selection_eligible")
    if not isinstance(production_valid, bool) or not isinstance(selection_eligible, bool):
        raise MirrorError("eval artifact top-level validity flags must be booleans")
    reports = payload.get("sampling_reports")
    expected_reports = {
        "unbiased": dict(run_evidence.unbiased_sampling_report),
        "focused": dict(run_evidence.focused_sampling_report),
    }
    if not isinstance(reports, dict) or reports != expected_reports:
        raise MirrorError("eval artifact sampling_reports do not match heldout evidence")
    report_production: list[bool] = []
    report_eligible: list[bool] = []
    for report_name in ("unbiased", "focused"):
        report = reports[report_name]
        if not isinstance(report, dict):
            raise MirrorError("eval artifact sampling report must be an object")
        report_prod = report.get("production_valid")
        report_select = report.get("checkpoint_selection_eligible")
        if not isinstance(report_prod, bool) or not isinstance(report_select, bool):
            raise MirrorError("sampling report validity flags must be exact booleans")
        report_production.append(report_prod)
        report_eligible.append(report_select)
    if production_valid != all(report_production):
        raise MirrorError("eval artifact production_valid disagrees with sampling reports")
    if selection_eligible != all(report_eligible):
        raise MirrorError(
            "eval artifact checkpoint_selection_eligible disagrees with sampling reports"
        )
    if selection_eligible and not production_valid:
        raise MirrorError("selection-eligible eval cannot be production-invalid")
    selection = payload.get("selection_metric")
    if not isinstance(selection, dict) or set(selection) != {
        "name",
        "mode",
        "eligible",
        "value",
    }:
        raise MirrorError("eval artifact selection_metric must be an object")
    if selection.get("name") != metric_name or selection.get("mode") != metric_mode:
        raise MirrorError("eval artifact selection metric configuration mismatch")
    if selection.get("eligible") is not selection_eligible:
        raise MirrorError("eval artifact selection metric eligibility is inconsistent")
    metric_value = _finite_number(selection.get("value"), label="selection metric value")

    metric_occurrences: list[float] = []
    metric_groups = payload.get("metrics")
    if not isinstance(metric_groups, dict) or set(metric_groups) != {"unbiased", "focused"}:
        raise MirrorError("eval artifact metrics must be an object")
    for group_name, group in metric_groups.items():
        if not isinstance(group, dict):
            raise MirrorError(f"eval metric group {group_name} must be an object")
        expected_prefix = "heldout_eval_" if group_name == "unbiased" else "heldout_focused_eval_"
        for recorded_name, recorded_value in group.items():
            if not isinstance(recorded_name, str) or not recorded_name.startswith(expected_prefix):
                raise MirrorError(f"eval metric is placed in the wrong group: {recorded_name!r}")
            _finite_number(recorded_value, label=f"metrics.{group_name}.{recorded_name}")
        if metric_name in group:
            metric_occurrences.append(
                _finite_number(group[metric_name], label=f"metrics.{metric_name}")
            )
    if metric_name not in metric_groups["focused"] or metric_name in metric_groups["unbiased"]:
        raise MirrorError("configured focused selection metric is placed incorrectly")
    if len(metric_occurrences) != 1 or metric_occurrences[0] != metric_value:
        raise MirrorError(
            "configured selection metric must occur exactly once in metric groups "
            "and equal selection_metric.value"
        )
    return EvalArtifact(
        step=step,
        trainer_state_sha256=expected_trainer_sha,
        metric_name=metric_name,
        metric_mode=metric_mode,
        metric_value=metric_value,
        sha256=record.sha256,
        size=record.size,
        cryptographically_bound=cryptographically_bound,
        production_eligible=bool(
            cryptographically_bound and production_valid and selection_eligible
        ),
        archival_reason=(
            None if cryptographically_bound else "missing_checkpoint_model_file_sha256"
        ),
        source_kind="checkpoint",
    )


def validate_baseline_eval_artifact(
    artifact_path: Path | str,
    *,
    metric_name: str,
    metric_mode: str,
    run_id: str,
    run_evidence: RunEvidenceIdentity,
) -> EvalArtifact:
    """Authenticate the step-0 live-model audit without inventing a checkpoint."""

    path = Path(artifact_path)
    if path.name != "step_00000000.json":
        raise MirrorError("live-model baseline must use canonical step_00000000.json")
    record = _regular_record(path, relative_path=path.name, label="baseline eval artifact")
    payload = _load_json_object(path, label="baseline eval artifact")
    expected_top_fields = {
        "schema_version",
        "checkpoint_step",
        "checkpoint_relative_path",
        "checkpoint",
        "run",
        "sampling_reports",
        "production_valid",
        "checkpoint_selection_eligible",
        "selection_metric",
        "metrics",
    }
    if set(payload) != expected_top_fields:
        raise MirrorError("baseline eval has incomplete or unexpected top-level fields")
    _require_schema_version(payload.get("schema_version"), label="baseline eval schema_version")
    if payload.get("checkpoint_step") != 0 or payload.get("checkpoint_relative_path") is not None:
        raise MirrorError("baseline eval must not claim a checkpoint path")
    checkpoint = payload.get("checkpoint")
    if not isinstance(checkpoint, dict) or checkpoint != {
        "step": 0,
        "source_path": None,
        "source_kind": "live_in_memory_model",
    }:
        raise MirrorError("baseline eval live-model identity is invalid")
    run = payload.get("run")
    if not isinstance(run, dict) or set(run) != {
        "run_id",
        "output_dir",
        "seed",
        "config_path",
        "config_sha256",
        "resolved_training_schedule",
        "source_training_config",
    }:
        raise MirrorError("baseline eval run identity shape is invalid")
    if (
        run_id != run_evidence.run_id
        or run.get("run_id") != run_evidence.run_id
        or run.get("output_dir") != run_evidence.source_output_dir
        or run.get("seed") != run_evidence.seed
        or isinstance(run.get("seed"), bool)
        or run.get("config_path") != "config.yaml"
        or run.get("config_sha256") != run_evidence.config_sha256
        or run.get("resolved_training_schedule")
        != {
            "path": "resolved_training_schedule.json",
            "sha256": run_evidence.schedule_sha256,
        }
        or run.get("source_training_config") is not None
    ):
        raise MirrorError("baseline eval run identity does not match run evidence")
    reports = payload.get("sampling_reports")
    if reports != {
        "unbiased": dict(run_evidence.unbiased_sampling_report),
        "focused": dict(run_evidence.focused_sampling_report),
    }:
        raise MirrorError("baseline eval sampling reports do not match run evidence")
    if payload.get("production_valid") is not True or payload.get(
        "checkpoint_selection_eligible"
    ) is not True:
        raise MirrorError("baseline eval must carry exact production-valid flags")
    selection = payload.get("selection_metric")
    if not isinstance(selection, dict) or set(selection) != {"name", "mode", "eligible", "value"}:
        raise MirrorError("baseline eval selection metric shape is invalid")
    if (
        selection.get("name") != metric_name
        or selection.get("mode") != metric_mode
        or selection.get("eligible") is not True
    ):
        raise MirrorError("baseline eval selection metric identity is invalid")
    metric_value = _finite_number(selection.get("value"), label="baseline selection metric")
    metric_groups = payload.get("metrics")
    if not isinstance(metric_groups, dict) or set(metric_groups) != {"unbiased", "focused"}:
        raise MirrorError("baseline eval metrics shape is invalid")
    occurrences: list[float] = []
    for group_name, group in metric_groups.items():
        if not isinstance(group, dict):
            raise MirrorError("baseline eval metric group is invalid")
        prefix = "heldout_eval_" if group_name == "unbiased" else "heldout_focused_eval_"
        for name, value in group.items():
            if not isinstance(name, str) or not name.startswith(prefix):
                raise MirrorError("baseline eval metric is placed in the wrong group")
            numeric = _finite_number(value, label=f"baseline metrics.{group_name}.{name}")
            if name == metric_name:
                occurrences.append(numeric)
    if metric_name not in metric_groups["focused"] or metric_name in metric_groups["unbiased"]:
        raise MirrorError("baseline selection metric is placed incorrectly")
    if occurrences != [metric_value]:
        raise MirrorError("baseline selection metric must occur exactly once")
    return EvalArtifact(
        step=0,
        trainer_state_sha256="0" * 64,
        metric_name=metric_name,
        metric_mode=metric_mode,
        metric_value=metric_value,
        sha256=record.sha256,
        size=record.size,
        cryptographically_bound=True,
        production_eligible=False,
        archival_reason="live_in_memory_baseline_not_checkpoint_recoverable",
        source_kind="live_in_memory_model",
    )


def validate_pointer_file(
    path: Path | str,
    *,
    metric_name: str,
    metric_mode: str,
    maximum_step: int | None = None,
) -> tuple[SelectionPointer, FileRecord]:
    pointer_path = Path(path)
    record = _regular_record(
        pointer_path, relative_path=pointer_path.name, label="best checkpoint pointer"
    )
    payload = _load_json_object(pointer_path, label="best checkpoint pointer")
    pointer = validate_selection_payload(
        payload,
        metric_name=metric_name,
        metric_mode=metric_mode,
        maximum_step=maximum_step,
    )
    return pointer, record


def _walk_regular_tree(root: Path, *, label: str) -> list[FileRecord]:
    _validate_root_directory(root, label=label)
    records: list[FileRecord] = []
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        for directory in list(directories):
            path = current_path / directory
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise MirrorError(f"{label} rejects linked/special directory: {path}")
        for filename in files:
            path = current_path / filename
            relative = path.relative_to(root).as_posix()
            records.append(_regular_record(path, relative_path=relative, label=label))
    if not records:
        raise MirrorError(f"{label} tree has no files: {root}")
    return records


def validate_run_evidence(run_dir: Path | str) -> TreeManifest:
    root = Path(run_dir)
    _validate_root_directory(root, label="source run")
    missing = sorted(name for name in REQUIRED_RUN_EVIDENCE if not (root / name).is_file())
    if missing:
        raise MirrorError(f"run evidence is incomplete; missing={missing}")
    records: list[FileRecord] = []
    for name in sorted(REQUIRED_RUN_EVIDENCE | OPTIONAL_RUN_EVIDENCE):
        path = root / name
        if path.exists() or path.is_symlink():
            records.append(_regular_record(path, relative_path=name, label="run evidence"))
    log_records = 0
    for directory in sorted(RUN_EVIDENCE_DIRS):
        path = root / directory
        if path.exists() or path.is_symlink():
            for record in _walk_regular_tree(path, label="run log evidence"):
                records.append(
                    FileRecord(
                        path=f"{directory}/{record.path}",
                        size=record.size,
                        sha256=record.sha256,
                    )
                )
                log_records += 1
    if log_records == 0 and not any(record.path == "summary.jsonl" for record in records):
        raise MirrorError("run evidence has no TensorBoard or summary log evidence")
    return TreeManifest.create(kind="run_evidence", records=records)


def validate_evidence_snapshot(snapshot_dir: Path | str) -> TreeManifest:
    root = Path(snapshot_dir)
    records = _walk_regular_tree(root, label="evidence snapshot")
    return TreeManifest.create(kind="run_evidence", records=records)


def _heldout_report_evidence(payload: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    digest = payload.get("window_selection_sha256")
    if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
        raise MirrorError(f"{label} lacks a valid window_selection_sha256")
    if "episode_split_provenance" not in payload:
        raise MirrorError(f"{label} lacks episode_split_provenance")
    try:
        evidence = json.loads(
            json.dumps(payload, sort_keys=True, ensure_ascii=True, allow_nan=False)
        )
    except (TypeError, ValueError) as exc:
        raise MirrorError(f"{label} is not strict JSON evidence") from exc
    if not isinstance(evidence, dict):
        raise MirrorError(f"{label} is not an object")
    return evidence


def _strict_count(value: Any, *, label: str, minimum: int = 0) -> int:
    return _strict_int(value, label=label, minimum=minimum)


def _canonical_count_map(value: Any, *, label: str) -> dict[int, int]:
    if not isinstance(value, dict) or not value:
        raise MirrorError(f"{label} must be a non-empty object")
    result: dict[int, int] = {}
    for raw_key, raw_value in value.items():
        if not isinstance(raw_key, str) or not re.fullmatch(r"0|[1-9][0-9]*", raw_key):
            raise MirrorError(f"{label} has a noncanonical integer key")
        key = int(raw_key)
        result[key] = _strict_count(raw_value, label=f"{label}.{raw_key}")
    return result


def _canonical_horizon_counts(value: Any, *, label: str) -> dict[int, dict[int, int]]:
    if not isinstance(value, dict) or not value:
        raise MirrorError(f"{label} must be a non-empty object")
    allowed = {1, 5, 10, 20, 50}
    result: dict[int, dict[int, int]] = {}
    for raw_horizon, raw_counts in value.items():
        if (
            not isinstance(raw_horizon, str)
            or not re.fullmatch(r"[1-9][0-9]*", raw_horizon)
            or int(raw_horizon) not in allowed
        ):
            raise MirrorError(f"{label} has an unsupported/noncanonical horizon")
        result[int(raw_horizon)] = _canonical_count_map(
            raw_counts, label=f"{label}.{raw_horizon}"
        )
    if 10 not in result:
        raise MirrorError(f"{label} lacks required H10 coverage")
    return result


def _validate_split_provenance(value: Any, *, observations: int, label: str) -> None:
    if not isinstance(value, list) or not value:
        raise MirrorError(f"{label} split provenance is missing")
    sha_fields = {
        "manifest_sha256",
        "selected_episode_set_sha256",
        "train_episode_set_sha256",
        "holdout_episode_set_sha256",
        "full_catalog_sha256",
        "train_statistics_sha256",
    }
    count_fields = {
        "selected_episode_count",
        "selected_frame_count",
        "train_episode_count",
        "train_frame_count",
        "holdout_episode_count",
    }
    selected_total = 0
    for index, entry in enumerate(value):
        if not isinstance(entry, dict) or set(entry) != PROVENANCE_FIELDS:
            raise MirrorError(f"{label} provenance[{index}] shape is invalid")
        if not isinstance(entry["dataset_name"], str) or not entry["dataset_name"]:
            raise MirrorError(f"{label} provenance[{index}] dataset_name is invalid")
        if str(entry["role"]).lower() not in {
            "eval",
            "evaluation",
            "val",
            "validation",
            "test",
            "holdout",
        }:
            raise MirrorError(f"{label} provenance[{index}] is not holdout-role")
        for field in ("manifest_path", "train_statistics_path"):
            if not isinstance(entry[field], str) or not entry[field]:
                raise MirrorError(f"{label} provenance[{index}].{field} is invalid")
        for field in sha_fields:
            if not isinstance(entry[field], str) or SHA256_RE.fullmatch(entry[field]) is None:
                raise MirrorError(f"{label} provenance[{index}].{field} is invalid")
        for field in count_fields:
            _strict_count(entry[field], label=f"{label} provenance[{index}].{field}")
        selected_total += int(entry["selected_episode_count"])
    if selected_total != observations:
        raise MirrorError(
            f"{label} split provenance does not bind all {observations} observations"
        )


def _validate_window_report_contracts(
    unbiased: Mapping[str, Any],
    focused: Mapping[str, Any],
    *,
    expected_observations: int | None = None,
) -> None:
    window_fields = {
        "dataset_index",
        "dataset_name",
        "episode_id",
        "base_index",
        "valid_base_index_min",
        "valid_base_index_max",
        "structural_candidate_count",
        "evaluable_candidate_count",
        "selection_pool_candidate_count",
        "valid_action_timesteps",
        "valid_action_elements",
        "anchor_subtask_index",
        "action_subtask_indices",
        "valid_action_elements_per_timestep",
        "open_to_close_transitions_h10",
        "close_to_open_transitions_h10",
        "open_to_close_window_h10",
        "close_to_open_window_h10",
        "arm_movement_elements_h10",
        "arm_movement_hold_abs_h10",
    }
    normalized_windows: dict[str, list[dict[str, Any]]] = {}
    for label, report, expected_view, expected_purpose in (
        (
            "unbiased",
            unbiased,
            "unbiased",
            "one_window_per_manifest_holdout_episode_checkpoint_eval",
        ),
        (
            "focused",
            focused,
            "focused",
            "h10_transition_stage_focused_manifest_holdout_checkpoint_eval",
        ),
    ):
        _require_schema_version(
            report.get("schema_version"), label=f"{label} window schema_version"
        )
        expected_fields = set(BASE_HELDOUT_REPORT_FIELDS)
        if label == "focused":
            expected_fields.update(FOCUSED_HELDOUT_REPORT_FIELDS)
        if set(report) != expected_fields:
            raise MirrorError(
                f"{label} heldout window report has incomplete/unexpected fields: "
                f"missing={sorted(expected_fields - set(report))}, "
                f"unexpected={sorted(set(report) - expected_fields)}"
            )
        if report.get("view") != expected_view or report.get("purpose") != expected_purpose:
            raise MirrorError(f"{label} heldout window view/purpose is invalid")
        expected_algorithm = (
            "nonzero_valid_unpadded_uniform_v1"
            if label == "unbiased"
            else "h10_gripper_transition_stage_balanced_v1"
        )
        if report.get("algorithm") != expected_algorithm:
            raise MirrorError(f"{label} heldout window algorithm is invalid")
        if report.get("production_valid") is not True or report.get(
            "checkpoint_selection_eligible"
        ) is not True:
            raise MirrorError(f"{label} heldout window report is not production eligible")
        observations = _strict_count(
            report.get("observation_count"),
            label=f"{label}.observation_count",
            minimum=1,
        )
        if expected_observations is not None and observations != expected_observations:
            raise MirrorError(
                f"{label} report does not contain the resolved global batch: "
                f"{observations} != {expected_observations}"
            )
        windows = report.get("windows")
        if not isinstance(windows, list) or observations != len(windows) or not all(
            isinstance(window, dict) for window in windows
        ):
            raise MirrorError(f"{label} heldout window cardinality is inconsistent")
        action_range = report.get("action_offset_range_inclusive")
        if (
            not isinstance(action_range, list)
            or len(action_range) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in action_range)
            or action_range[1] < action_range[0]
        ):
            raise MirrorError(f"{label} action offset range must be two ordered integers")
        action_horizon = action_range[1] - action_range[0] + 1
        normalized: list[dict[str, Any]] = []
        seen_episodes: set[tuple[int, str, int]] = set()
        for window_index, raw_window in enumerate(windows):
            if set(raw_window) != window_fields:
                raise MirrorError(
                    f"{label} window[{window_index}] has incomplete/unexpected fields"
                )
            dataset_index = _strict_count(
                raw_window["dataset_index"],
                label=f"{label}.windows[{window_index}].dataset_index",
            )
            dataset_name = raw_window["dataset_name"]
            if not isinstance(dataset_name, str) or not dataset_name:
                raise MirrorError(f"{label} window[{window_index}] dataset_name is invalid")
            episode_id = _strict_count(
                raw_window["episode_id"],
                label=f"{label}.windows[{window_index}].episode_id",
            )
            identity = (dataset_index, dataset_name, episode_id)
            if identity in seen_episodes:
                raise MirrorError(f"{label} contains duplicate episode window {identity!r}")
            seen_episodes.add(identity)
            base_index = _strict_count(
                raw_window["base_index"], label=f"{label}.windows[{window_index}].base_index"
            )
            valid_min = _strict_count(
                raw_window["valid_base_index_min"],
                label=f"{label}.windows[{window_index}].valid_base_index_min",
            )
            valid_max = _strict_count(
                raw_window["valid_base_index_max"],
                label=f"{label}.windows[{window_index}].valid_base_index_max",
            )
            if valid_min > base_index or base_index > valid_max:
                raise MirrorError(f"{label} window[{window_index}] base index is out of range")
            structural = _strict_count(
                raw_window["structural_candidate_count"],
                label=f"{label}.windows[{window_index}].structural_candidate_count",
                minimum=1,
            )
            evaluable = _strict_count(
                raw_window["evaluable_candidate_count"],
                label=f"{label}.windows[{window_index}].evaluable_candidate_count",
                minimum=1,
            )
            pool = _strict_count(
                raw_window["selection_pool_candidate_count"],
                label=f"{label}.windows[{window_index}].selection_pool_candidate_count",
                minimum=1,
            )
            if (
                evaluable > structural
                or pool > evaluable
                or (label == "unbiased" and pool != evaluable)
            ):
                raise MirrorError(f"{label} window[{window_index}] candidate counts are inconsistent")
            valid_per_timestep = raw_window["valid_action_elements_per_timestep"]
            action_subtasks = raw_window["action_subtask_indices"]
            if (
                not isinstance(valid_per_timestep, list)
                or not isinstance(action_subtasks, list)
                or len(valid_per_timestep) != action_horizon
                or len(action_subtasks) != action_horizon
            ):
                raise MirrorError(f"{label} window[{window_index}] action metadata horizon is invalid")
            if any(
                isinstance(item, bool) or not isinstance(item, int)
                for item in action_subtasks
            ):
                raise MirrorError(f"{label} window[{window_index}] action subtask labels are invalid")
            valid_counts = [
                _strict_count(
                    item,
                    label=(
                        f"{label}.windows[{window_index}]"
                        ".valid_action_elements_per_timestep"
                    ),
                )
                for item in valid_per_timestep
            ]
            if any(item > int(report["action_dim"]) for item in valid_counts):
                raise MirrorError(f"{label} window[{window_index}] valid element count exceeds action_dim")
            window_timesteps = _strict_count(
                raw_window["valid_action_timesteps"],
                label=f"{label}.windows[{window_index}].valid_action_timesteps",
                minimum=1,
            )
            window_elements = _strict_count(
                raw_window["valid_action_elements"],
                label=f"{label}.windows[{window_index}].valid_action_elements",
                minimum=1,
            )
            if window_timesteps != sum(item > 0 for item in valid_counts) or window_elements != sum(valid_counts):
                raise MirrorError(f"{label} window[{window_index}] valid-action aggregates are inconsistent")
            anchor = raw_window["anchor_subtask_index"]
            if isinstance(anchor, bool) or not isinstance(anchor, int):
                raise MirrorError(f"{label} window[{window_index}] anchor subtask is invalid")
            for field in (
                "open_to_close_transitions_h10",
                "close_to_open_transitions_h10",
                "arm_movement_elements_h10",
            ):
                _strict_count(raw_window[field], label=f"{label}.windows[{window_index}].{field}")
            for transition_field, flag_field in (
                ("open_to_close_transitions_h10", "open_to_close_window_h10"),
                ("close_to_open_transitions_h10", "close_to_open_window_h10"),
            ):
                if not isinstance(raw_window[flag_field], bool) or raw_window[flag_field] != (
                    raw_window[transition_field] > 0
                ):
                    raise MirrorError(f"{label} window[{window_index}] transition flag is inconsistent")
            movement_hold = _finite_number(
                raw_window["arm_movement_hold_abs_h10"],
                label=f"{label}.windows[{window_index}].arm_movement_hold_abs_h10",
            )
            if movement_hold < 0:
                raise MirrorError(f"{label} window[{window_index}] movement hold is negative")
            normalized.append(dict(raw_window))
        normalized_windows[label] = normalized
        ordered_identities = [
            (window["dataset_name"], window["episode_id"], window["dataset_index"])
            for window in normalized
        ]
        if ordered_identities != sorted(ordered_identities):
            raise MirrorError(f"{label} heldout windows are not in canonical producer order")
        if label == "unbiased":
            for window_index, window in enumerate(normalized):
                if any(
                    window[field] != 0
                    for field in (
                        "open_to_close_transitions_h10",
                        "close_to_open_transitions_h10",
                        "arm_movement_elements_h10",
                    )
                ) or any(
                    window[field] is not False
                    for field in (
                        "open_to_close_window_h10",
                        "close_to_open_window_h10",
                    )
                ) or float(window["arm_movement_hold_abs_h10"]) != 0.0:
                    raise MirrorError(
                        f"unbiased window[{window_index}] carries focused diagnostics"
                    )
        action_evaluable = _strict_count(
            report.get("action_evaluable_observation_count"),
            label=f"{label}.action_evaluable_observation_count",
        )
        if action_evaluable != observations or report.get("zero_valid_action_episodes") != []:
            raise MirrorError(f"{label} report contains a zero-supervision observation")
        action_dim = _strict_count(
            report.get("action_dim"), label=f"{label}.action_dim", minimum=1
        )
        valid_timesteps = _strict_count(
            report.get("valid_action_timestep_count"),
            label=f"{label}.valid_action_timestep_count",
            minimum=1,
        )
        valid_elements = _strict_count(
            report.get("valid_action_element_count"),
            label=f"{label}.valid_action_element_count",
            minimum=1,
        )
        if valid_elements > valid_timesteps * action_dim:
            raise MirrorError(f"{label} valid-action counts are impossible")
        if valid_timesteps != sum(
            int(window["valid_action_timesteps"]) for window in normalized
        ) or valid_elements != sum(
            int(window["valid_action_elements"]) for window in normalized
        ):
            raise MirrorError(f"{label} valid-action totals do not match its windows")
        subtask_counts = _canonical_count_map(
            report.get("subtask_observation_counts"),
            label=f"{label}.subtask_observation_counts",
        )
        evaluable_counts = _canonical_count_map(
            report.get("subtask_evaluable_observation_counts"),
            label=f"{label}.subtask_evaluable_observation_counts",
        )
        if sum(subtask_counts.values()) != observations or sum(
            evaluable_counts.values()
        ) != observations:
            raise MirrorError(f"{label} subtask observation counts are inconsistent")
        derived_anchor_counts: dict[int, int] = {}
        for window in normalized:
            key = int(window["anchor_subtask_index"])
            derived_anchor_counts[key] = derived_anchor_counts.get(key, 0) + 1
        if subtask_counts != derived_anchor_counts or evaluable_counts != derived_anchor_counts:
            raise MirrorError(f"{label} subtask counts do not match window anchors")
        timestep_horizons = _canonical_horizon_counts(
            report.get("subtask_action_timestep_counts_by_horizon"),
            label=f"{label}.subtask_action_timestep_counts_by_horizon",
        )
        element_horizons = _canonical_horizon_counts(
            report.get("subtask_valid_action_element_counts_by_horizon"),
            label=f"{label}.subtask_valid_action_element_counts_by_horizon",
        )
        if set(timestep_horizons) != set(element_horizons):
            raise MirrorError(f"{label} action/elements horizon sets differ")
        expected_horizons = {value for value in (1, 5, 10, 20, 50) if value <= action_horizon}
        if set(timestep_horizons) != expected_horizons:
            raise MirrorError(f"{label} action horizon coverage is incomplete")
        for horizon, counts in timestep_horizons.items():
            if sum(counts.values()) != observations * horizon:
                raise MirrorError(f"{label} H{horizon} timestep coverage is incomplete")
            element_total = sum(element_horizons[horizon].values())
            if element_total <= 0 or element_total > observations * horizon * action_dim:
                raise MirrorError(f"{label} H{horizon} valid-element coverage is invalid")
            derived_timesteps: dict[int, int] = {}
            derived_elements: dict[int, int] = {}
            for window in normalized:
                labels = window["action_subtask_indices"][:horizon]
                valid_counts = window["valid_action_elements_per_timestep"][:horizon]
                for subtask, element_count in zip(labels, valid_counts):
                    key = int(subtask)
                    derived_timesteps[key] = derived_timesteps.get(key, 0) + 1
                    derived_elements[key] = derived_elements.get(key, 0) + int(element_count)
            if counts != derived_timesteps or element_horizons[horizon] != derived_elements:
                raise MirrorError(f"{label} H{horizon} subtask aggregates do not match windows")
        if report.get("frames_per_episode") != 1:
            raise MirrorError(f"{label}.frames_per_episode must be exactly 1")
        if report.get("observation_mode") != "deployment_action_current_qwen_rgb_v1":
            raise MirrorError(f"{label}.observation_mode is not deployment-exact")
        if report.get("evaluation_video_offsets") != [0]:
            raise MirrorError(f"{label}.evaluation_video_offsets must be exactly [0]")
        for field in ("evaluation_video_offsets", "action_offset_range_inclusive"):
            sequence = report.get(field)
            if not isinstance(sequence, list) or not sequence or any(
                isinstance(item, bool) or not isinstance(item, int) for item in sequence
            ):
                raise MirrorError(f"{label}.{field} is invalid")
        seed = report.get("seed_sha256")
        if not isinstance(seed, str) or SHA256_RE.fullmatch(seed) is None:
            raise MirrorError(f"{label} heldout window seed digest is invalid")
        _validate_split_provenance(
            report.get("episode_split_provenance"),
            observations=observations,
            label=label,
        )
        provenance = report["episode_split_provenance"]
        if len({entry["dataset_name"] for entry in provenance}) != len(provenance):
            raise MirrorError(f"{label} split provenance has duplicate dataset names")
        per_dataset: dict[int, int] = {}
        for window in normalized:
            dataset_index = int(window["dataset_index"])
            if dataset_index >= len(provenance) or provenance[dataset_index]["dataset_name"] != window[
                "dataset_name"
            ]:
                raise MirrorError(f"{label} window dataset index/name is not provenance-bound")
            per_dataset[dataset_index] = per_dataset.get(dataset_index, 0) + 1
        if any(
            per_dataset.get(index, 0) != entry["selected_episode_count"]
            for index, entry in enumerate(provenance)
        ):
            raise MirrorError(f"{label} per-dataset window counts do not match provenance")
    shared_fields = {
        "observation_mode",
        "evaluation_video_offsets",
        "action_offset_range_inclusive",
        "frames_per_episode",
        "seed_sha256",
        "observation_count",
        "action_dim",
        "episode_split_provenance",
    }
    if any(unbiased[field] != focused[field] for field in shared_fields):
        raise MirrorError("focused and unbiased immutable sampling contracts differ")
    identity = lambda window: (
        window["dataset_index"], window["dataset_name"], window["episode_id"]
    )
    if {identity(window) for window in normalized_windows["unbiased"]} != {
        identity(window) for window in normalized_windows["focused"]
    }:
        raise MirrorError("focused and unbiased reports cover different heldout episodes")
    focused_subtasks = focused["focused_subtasks"]
    if (
        not isinstance(focused_subtasks, list)
        or not focused_subtasks
        or any(isinstance(item, bool) or not isinstance(item, int) for item in focused_subtasks)
        or focused_subtasks != sorted(set(focused_subtasks))
    ):
        raise MirrorError("focused_subtasks must be sorted unique integers")
    for field in (
        "open_to_close_transition_count_h10",
        "close_to_open_transition_count_h10",
        "open_to_close_transition_window_count_h10",
        "close_to_open_transition_window_count_h10",
        "arm_movement_element_count_h10",
    ):
        _strict_count(focused[field], label=f"focused.{field}", minimum=1)
    if (
        focused["open_to_close_transition_window_count_h10"]
        > focused["observation_count"]
        or focused["close_to_open_transition_window_count_h10"]
        > focused["observation_count"]
    ):
        raise MirrorError("focused transition-window counts exceed observation count")
    if focused["open_to_close_transition_count_h10"] < focused[
        "open_to_close_transition_window_count_h10"
    ] or focused["close_to_open_transition_count_h10"] < focused[
        "close_to_open_transition_window_count_h10"
    ]:
        raise MirrorError("focused transition counts are inconsistent")
    hold_abs = _finite_number(
        focused["arm_movement_hold_abs_sum_h10"],
        label="focused.arm_movement_hold_abs_sum_h10",
    )
    threshold = _finite_number(
        focused["movement_threshold_normalized"],
        label="focused.movement_threshold_normalized",
    )
    if hold_abs <= 0 or threshold < 0:
        raise MirrorError("focused arm movement/threshold coverage is invalid")
    focused_windows = normalized_windows["focused"]
    aggregate_fields = {
        "open_to_close_transition_count_h10": sum(
            int(window["open_to_close_transitions_h10"]) for window in focused_windows
        ),
        "close_to_open_transition_count_h10": sum(
            int(window["close_to_open_transitions_h10"]) for window in focused_windows
        ),
        "open_to_close_transition_window_count_h10": sum(
            bool(window["open_to_close_window_h10"]) for window in focused_windows
        ),
        "close_to_open_transition_window_count_h10": sum(
            bool(window["close_to_open_window_h10"]) for window in focused_windows
        ),
        "arm_movement_element_count_h10": sum(
            int(window["arm_movement_elements_h10"]) for window in focused_windows
        ),
    }
    if any(focused[field] != value for field, value in aggregate_fields.items()):
        raise MirrorError("focused transition/movement aggregates do not match windows")
    if hold_abs != sum(
        float(window["arm_movement_hold_abs_h10"]) for window in focused_windows
    ):
        raise MirrorError("focused arm movement hold aggregate does not match windows")
    unbiased_contract = {
        "algorithm": unbiased.get("algorithm"),
        "observation_mode": unbiased.get("observation_mode"),
        "evaluation_video_offsets": unbiased.get("evaluation_video_offsets"),
        "action_offset_range_inclusive": unbiased.get("action_offset_range_inclusive"),
        "frames_per_episode": unbiased.get("frames_per_episode"),
        "seed_sha256": unbiased.get("seed_sha256"),
        "windows": unbiased.get("windows"),
    }
    unbiased_digest = hashlib.sha256(_canonical_json_bytes(unbiased_contract)).hexdigest()
    if unbiased.get("window_selection_sha256") != unbiased_digest:
        raise MirrorError("unbiased window_selection_sha256 does not match its windows")
    if len(focused["windows"]) != len(unbiased["windows"]):
        raise MirrorError("focused and unbiased heldout cardinalities differ")
    focused_contract = {
        "algorithm": focused.get("algorithm"),
        "transition_horizon": 10,
        "focused_subtasks": focused.get("focused_subtasks"),
        "manifest_seed_sha256": focused.get("seed_sha256"),
        "unbiased_window_selection_sha256": unbiased_digest,
        "windows": focused.get("windows"),
    }
    focused_digest = hashlib.sha256(_canonical_json_bytes(focused_contract)).hexdigest()
    if focused.get("window_selection_sha256") != focused_digest:
        raise MirrorError("focused window_selection_sha256 does not match its windows")


@dataclasses.dataclass(frozen=True)
class RunEvidenceIdentity:
    manifest: TreeManifest
    source_output_dir: str
    config_sha256: str
    schedule_sha256: str
    run_id: str
    seed: int
    unbiased_sampling_report: Mapping[str, Any]
    focused_sampling_report: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "manifest": self.manifest.as_dict(),
            "source_output_dir": self.source_output_dir,
            "config_sha256": self.config_sha256,
            "schedule_sha256": self.schedule_sha256,
            "run_id": self.run_id,
            "seed": self.seed,
            "sampling_reports": {
                "unbiased": dict(self.unbiased_sampling_report),
                "focused": dict(self.focused_sampling_report),
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RunEvidenceIdentity":
        _require_schema_version(
            payload.get("schema_version"), label="run evidence identity schema_version"
        )
        if set(payload) != {
            "schema_version",
            "manifest",
            "source_output_dir",
            "config_sha256",
            "schedule_sha256",
            "run_id",
            "seed",
            "sampling_reports",
        }:
            raise MirrorError("run evidence identity has incomplete/unexpected fields")
        source_output = payload.get("source_output_dir")
        if not isinstance(source_output, str) or not _safe_absolute_posix_path(source_output):
            raise MirrorError("run evidence source_output_dir is unsafe")
        config_sha = payload.get("config_sha256")
        schedule_sha = payload.get("schedule_sha256")
        if not isinstance(config_sha, str) or SHA256_RE.fullmatch(config_sha) is None:
            raise MirrorError("run evidence config SHA-256 is invalid")
        if not isinstance(schedule_sha, str) or SHA256_RE.fullmatch(schedule_sha) is None:
            raise MirrorError("run evidence schedule SHA-256 is invalid")
        run_id = payload.get("run_id")
        if not isinstance(run_id, str) or RUN_ID_RE.fullmatch(run_id) is None:
            raise MirrorError("run evidence run_id is invalid")
        seed = _strict_int(payload.get("seed"), label="run evidence seed")
        reports = payload.get("sampling_reports")
        if not isinstance(reports, dict) or set(reports) != {"unbiased", "focused"}:
            raise MirrorError("run evidence sampling report shape is invalid")
        if not all(isinstance(reports[key], dict) for key in reports):
            raise MirrorError("run evidence sampling reports must be objects")
        _validate_window_report_contracts(reports["unbiased"], reports["focused"])
        manifest = TreeManifest.from_dict(payload["manifest"])
        if manifest.kind != "run_evidence" or manifest.step is not None:
            raise MirrorError("run evidence identity carries the wrong manifest kind")
        evidence_paths = {record.path for record in manifest.files}
        if not REQUIRED_RUN_EVIDENCE <= evidence_paths or not evidence_paths <= (
            REQUIRED_RUN_EVIDENCE | OPTIONAL_RUN_EVIDENCE
        ):
            raise MirrorError("run evidence identity manifest has invalid evidence paths")
        if manifest.file("config.yaml").sha256 != config_sha:
            raise MirrorError("run evidence config SHA-256 contradicts its manifest")
        if manifest.file("resolved_training_schedule.json").sha256 != schedule_sha:
            raise MirrorError("run evidence schedule SHA-256 contradicts its manifest")
        return cls(
            manifest=manifest,
            source_output_dir=source_output,
            config_sha256=config_sha,
            schedule_sha256=schedule_sha,
            run_id=run_id,
            seed=seed,
            unbiased_sampling_report=dict(reports["unbiased"]),
            focused_sampling_report=dict(reports["focused"]),
        )


def _safe_absolute_posix_path(path: str) -> bool:
    pure = PurePosixPath(path)
    return (
        pure.is_absolute()
        and str(pure) == path
        and "\x00" not in path
        and "\\" not in path
        and not any(ord(character) < 32 or ord(character) == 127 for character in path)
        and all(part not in {"", ".", ".."} for part in pure.parts)
    )


def resolve_source_root_binding(logical_path: str) -> dict[str, Any]:
    """Resolve an operator path once while pinning every allowed ancestor link."""

    if not _safe_absolute_posix_path(logical_path):
        raise MirrorError("source run logical path is not canonical and absolute")
    logical = Path(logical_path)
    try:
        final_info = logical.lstat()
        resolved = logical.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise MirrorError(f"source run path cannot be resolved: {logical}") from exc
    if stat.S_ISLNK(final_info.st_mode) or not stat.S_ISDIR(final_info.st_mode):
        raise MirrorError("source run endpoint must itself be a real directory")
    resolved_text = str(resolved)
    if not _safe_absolute_posix_path(resolved_text):
        raise MirrorError("resolved source run path is not canonical and absolute")
    resolved_info = _validate_root_directory(resolved, label="resolved source run")
    descriptor = os.open(logical, os.O_RDONLY | os.O_DIRECTORY)
    try:
        opened_info = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        not stat.S_ISDIR(opened_info.st_mode)
        or (opened_info.st_dev, opened_info.st_ino)
        != (resolved_info.st_dev, resolved_info.st_ino)
    ):
        raise MirrorError("source run changed while resolving its real path")
    symlink_ancestors: list[dict[str, str]] = []
    current = Path(logical.anchor)
    for part in logical.parts[1:-1]:
        current = current / part
        try:
            component = current.lstat()
        except FileNotFoundError as exc:
            raise MirrorError(f"source ancestor disappeared: {current}") from exc
        if stat.S_ISLNK(component.st_mode):
            symlink_ancestors.append(
                {"path": str(current), "target": os.readlink(current)}
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "logical_path": logical_path,
        "resolved_path": resolved_text,
        "device": resolved_info.st_dev,
        "inode": resolved_info.st_ino,
        "uid": resolved_info.st_uid,
        "symlink_ancestors": symlink_ancestors,
    }


def validate_run_evidence_identity(
    run_dir: Path | str, *, source_output_dir: str | None = None
) -> RunEvidenceIdentity:
    root = Path(run_dir)
    manifest = validate_run_evidence(root)
    source_output = str(root.resolve()) if source_output_dir is None else source_output_dir
    if not _safe_absolute_posix_path(source_output):
        raise MirrorError("source output directory is not a normalized absolute path")
    config_record = manifest.file("config.yaml")
    schedule_record = manifest.file("resolved_training_schedule.json")
    schedule = _load_json_object(
        root / "resolved_training_schedule.json", label="resolved training schedule"
    )
    _require_schema_version(
        schedule.get("schema_version"), label="resolved schedule schema_version"
    )
    if set(schedule.get("source_config", {})) != {"path", "sha256"}:
        raise MirrorError("resolved schedule source_config shape is invalid")
    if schedule["source_config"] != {
        "path": "config.yaml",
        "sha256": config_record.sha256,
    }:
        raise MirrorError("resolved schedule is not bound to config.yaml")
    resolved = schedule.get("resolved")
    required_resolved = {
        "effective_global_batch_size",
        "eval_interval",
        "max_train_steps",
        "num_warmup_steps",
        "save_interval",
    }
    if not isinstance(resolved, dict) or not required_resolved <= set(resolved):
        raise MirrorError("resolved schedule lacks required production fields")
    unbiased_raw = _load_json_object(
        root / "heldout_eval_windows.json", label="unbiased heldout window evidence"
    )
    focused_raw = _load_json_object(
        root / "heldout_focused_eval_windows.json",
        label="focused heldout window evidence",
    )
    expected_observations = _strict_count(
        resolved["effective_global_batch_size"],
        label="resolved effective_global_batch_size",
        minimum=1,
    )
    _validate_window_report_contracts(
        unbiased_raw,
        focused_raw,
        expected_observations=expected_observations,
    )
    config_json = _load_json_object(root / "config.json", label="run config JSON")
    evidence_run_id = config_json.get("run_id")
    evidence_seed = config_json.get("seed")
    if not isinstance(evidence_run_id, str) or RUN_ID_RE.fullmatch(evidence_run_id) is None:
        raise MirrorError("config.json run_id is invalid")
    if PurePosixPath(source_output).name != evidence_run_id:
        raise MirrorError("config.json run_id does not match source output basename")
    if config_json.get("output_dir") != source_output:
        raise MirrorError("config.json output_dir does not match source run directory")
    evidence_seed = _strict_int(evidence_seed, label="config.json seed")
    return RunEvidenceIdentity(
        manifest=manifest,
        source_output_dir=source_output,
        config_sha256=config_record.sha256,
        schedule_sha256=schedule_record.sha256,
        run_id=evidence_run_id,
        seed=evidence_seed,
        unbiased_sampling_report=_heldout_report_evidence(
            unbiased_raw, label="unbiased heldout window evidence"
        ),
        focused_sampling_report=_heldout_report_evidence(
            focused_raw, label="focused heldout window evidence"
        ),
    )


def _evidence_b64(evidence: RunEvidenceIdentity) -> str:
    return base64.urlsafe_b64encode(_canonical_json_bytes(evidence.as_dict())).decode(
        "ascii"
    )


def _eval_validation_context_sha256(
    manifest: TreeManifest | None, evidence: RunEvidenceIdentity
) -> str:
    """Bind an eval cache entry to every semantic input used to validate it."""

    return hashlib.sha256(
        _canonical_json_bytes(
            {
                "checkpoint_manifest_sha256": (
                    None if manifest is None else manifest.manifest_sha256
                ),
                "source_output_dir": evidence.source_output_dir,
                "config_sha256": evidence.config_sha256,
                "schedule_sha256": evidence.schedule_sha256,
                "run_id": evidence.run_id,
                "seed": evidence.seed,
                "sampling_reports": {
                    "unbiased": dict(evidence.unbiased_sampling_report),
                    "focused": dict(evidence.focused_sampling_report),
                },
                "immutable_evidence_sha256": _immutable_evidence_sha256(evidence),
            }
        )
    ).hexdigest()


def _immutable_evidence_files(evidence: RunEvidenceIdentity) -> dict[str, dict[str, Any]]:
    records = {
        record.path: {"size": record.size, "sha256": record.sha256}
        for record in evidence.manifest.files
        if record.path not in MUTABLE_RUN_EVIDENCE
    }
    required_immutable = REQUIRED_RUN_EVIDENCE - MUTABLE_RUN_EVIDENCE
    if not required_immutable <= set(records):
        raise MirrorError("run evidence immutable file set is incomplete")
    return dict(sorted(records.items()))


def _immutable_evidence_sha256(evidence: RunEvidenceIdentity) -> str:
    return hashlib.sha256(
        _canonical_json_bytes({"files": _immutable_evidence_files(evidence)})
    ).hexdigest()


def _evidence_from_b64(value: str) -> RunEvidenceIdentity:
    try:
        payload = json.loads(base64.urlsafe_b64decode(value.encode("ascii")))
    except Exception as exc:
        raise MirrorError("invalid encoded run evidence identity") from exc
    if not isinstance(payload, dict):
        raise MirrorError("encoded run evidence identity is not an object")
    return RunEvidenceIdentity.from_dict(payload)


def validate_recovery_closure(
    checkpoint_step: int,
    manifests: Mapping[int, TreeManifest],
) -> tuple[int, ...]:
    """Return the transitive selection dependency closure, or fail closed."""

    if checkpoint_step not in manifests:
        raise MirrorError(f"checkpoint steps_{checkpoint_step} is absent")
    visiting: set[int] = set()
    visited: set[int] = set()

    def visit(step: int) -> None:
        if step in visiting:
            raise MirrorError(f"selection dependency cycle includes steps_{step}")
        if step in visited:
            return
        manifest = manifests.get(step)
        if manifest is None:
            raise MirrorError(
                f"checkpoint steps_{checkpoint_step} depends on missing steps_{step}"
            )
        if manifest.kind != "checkpoint" or manifest.step != step:
            raise MirrorError(f"invalid checkpoint manifest for steps_{step}")
        visiting.add(step)
        if manifest.selection_best_step is not None and manifest.selection_best_step != step:
            visit(manifest.selection_best_step)
        visiting.remove(step)
        visited.add(step)

    visit(checkpoint_step)
    return tuple(sorted(visited))


def strict_global_best(
    evals: Mapping[int, EvalArtifact],
    *,
    metric_mode: str,
    maximum_step: int | None = None,
) -> tuple[int, float] | None:
    if metric_mode not in {"min", "max"}:
        raise MirrorError(f"unsupported metric mode: {metric_mode}")
    best: tuple[int, float] | None = None
    for step in sorted(evals):
        artifact = evals[step]
        if step != artifact.step:
            raise MirrorError("eval map key does not match artifact step")
        if maximum_step is not None and step > maximum_step:
            continue
        if not artifact.production_eligible:
            continue
        if best is None:
            best = (step, artifact.metric_value)
            continue
        improves = (
            artifact.metric_value < best[1]
            if metric_mode == "min"
            else artifact.metric_value > best[1]
        )
        if improves:
            best = (step, artifact.metric_value)
        # Equality deliberately retains the earlier prior pointer.
    return best


def authenticate_selection_history(
    manifests: Mapping[int, TreeManifest],
    evals: Mapping[int, EvalArtifact],
    pointer: SelectionPointer | None,
    *,
    metric_mode: str,
) -> None:
    global_best = strict_global_best(evals, metric_mode=metric_mode)
    if global_best is None:
        if pointer is not None:
            raise MirrorError("best pointer exists without an authenticated eligible eval")
    else:
        if pointer is None:
            raise MirrorError("authenticated eligible eval history lacks best pointer")
        if (pointer.best_step, pointer.metric_value) != global_best:
            raise MirrorError(
                "best pointer is not the strict global argmin/argmax over eligible evals"
            )
        best_eval = evals.get(pointer.best_step)
        if best_eval is None or not best_eval.cryptographically_bound:
            raise MirrorError("best pointer is not authenticated by its checkpoint eval")

    for step, manifest in sorted(manifests.items()):
        if manifest.checkpoint_schema != "selection_v1":
            continue
        expected = strict_global_best(
            evals, metric_mode=metric_mode, maximum_step=step
        )
        actual = (
            None
            if manifest.selection_best_step is None
            else (manifest.selection_best_step, manifest.selection_best_value)
        )
        if actual != expected:
            raise MirrorError(
                f"steps_{step} selection dependency/value does not match "
                "authenticated strict-best eval history"
            )


def retained_checkpoint_steps(
    verified_steps: Iterable[int],
    *,
    best_step: int | None,
    limit: int,
) -> tuple[int, ...]:
    """Keep the selected best plus newest recoverable checkpoints within cap."""

    if isinstance(limit, bool) or limit <= 0:
        raise MirrorError("checkpoint retention limit must be a positive integer")
    steps = sorted(set(verified_steps))
    if best_step is not None and best_step not in steps:
        raise MirrorError(f"selected best steps_{best_step} is not verified")
    if len(steps) <= limit:
        return tuple(steps)
    newest = list(steps[-limit:])
    if best_step is not None and best_step not in newest:
        if limit < 2:
            raise MirrorError(
                "retention cap cannot preserve selected best and newest checkpoint"
            )
        newest = [best_step, *steps[-(limit - 1) :]]
    return tuple(sorted(set(newest)))


def retained_recovery_steps(
    manifests: Mapping[int, TreeManifest],
    verified_steps: Iterable[int],
    *,
    best_step: int | None,
    limit: int,
) -> tuple[int, ...]:
    """Keep latest/best plus every selection dependency, all within ``limit``."""

    verified = sorted(set(verified_steps))
    if not verified:
        return ()
    if best_step is not None and best_step not in verified:
        raise MirrorError(f"selected best steps_{best_step} is not verified")
    latest = verified[-1]
    priority: list[int] = [latest]
    if best_step is not None and best_step != latest:
        priority.append(best_step)
    priority.extend(step for step in reversed(verified[:-1]) if step not in priority)

    closure: set[int] = set()
    for step in priority:
        candidate = closure | set(validate_recovery_closure(step, manifests))
        if len(candidate) <= limit:
            closure = candidate
        elif step in {latest, best_step}:
            raise MirrorError(
                "checkpoint retention cap cannot preserve latest/selected-best "
                f"recovery closure: required={sorted(candidate)}, limit={limit}"
            )
        if len(closure) == limit:
            break
    return tuple(sorted(closure))


def validate_columbus_root(run_id: str, root: Path | str) -> Path:
    if RUN_ID_RE.fullmatch(run_id) is None:
        raise MirrorError(f"unsafe run_id: {run_id!r}")
    path = Path(root)
    expected = SAFE_COLUMBUS_BASE / run_id
    if (
        not path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts[1:])
        or path != expected
    ):
        raise MirrorError(
            "Columbus root is fixed to the protected new relay namespace; "
            f"expected {expected}, got {path}"
        )
    return path


def require_free_space(path: Path | str, *, incoming_bytes: int, reserve_bytes: int) -> int:
    incoming = _strict_int(incoming_bytes, label="incoming_bytes")
    reserve = _strict_int(reserve_bytes, label="reserve_bytes")
    stats = os.statvfs(path)
    available = stats.f_bavail * stats.f_frsize
    required = incoming + reserve
    if available < required:
        raise MirrorError(
            f"insufficient free space at {path}: available={available}, required={required} "
            f"(incoming={incoming}, reserve={reserve})"
        )
    return available


class MirrorState:
    """Atomic, independently tiered verification and watcher state."""

    def __init__(
        self,
        path: Path,
        *,
        run_id: str,
        source_run_dir: str,
        local_root: str,
        columbus_root: str,
        metric_name: str,
        metric_mode: str,
        h100_identity: Mapping[str, Any],
        columbus_identity: Mapping[str, Any],
    ) -> None:
        self.path = path
        identity = {
            "run_id": run_id,
            "source_run_dir": source_run_dir,
            "local_root": local_root,
            "columbus_root": columbus_root,
            "metric_name": metric_name,
            "metric_mode": metric_mode,
            "h100_endpoint": dict(h100_identity),
            "columbus_endpoint": dict(columbus_identity),
        }
        if path.exists() or path.is_symlink():
            _secure_regular_file(path, label="mirror state")
            payload = _load_json_object(path, label="mirror state")
            expected_state_fields = {
                "schema_version",
                "identity",
                "created_at",
                "updated_at",
                "checkpoints",
                "eval_artifacts",
                "evidence_snapshots",
                "best_pointer",
                "pointer_history",
                "current_source_steps",
                "source_inventory",
                "source_evidence",
                "source_pointer",
                "pending_candidates",
                "last_full_scrub_at",
                "prune_pending",
                "prune_receipts",
                "restore_index_generation",
                "restore_index_sha256",
                "restore_index_intent",
                "pointer_publication_intent",
                "heartbeat",
            }
            if set(payload) != expected_state_fields:
                raise MirrorError("mirror state has incomplete or unexpected fields")
            if payload.get("schema_version") != MIRROR_STATE_SCHEMA_VERSION:
                raise MirrorError(
                    f"mirror state schema_version must be {MIRROR_STATE_SCHEMA_VERSION}"
                )
            if payload.get("identity") != identity:
                raise MirrorError("mirror state identity/configuration drift")
            self.payload = payload
        else:
            self.payload: dict[str, Any] = {
                "schema_version": MIRROR_STATE_SCHEMA_VERSION,
                "identity": identity,
                "created_at": utc_now(),
                "updated_at": utc_now(),
                "checkpoints": {},
                "eval_artifacts": {},
                "evidence_snapshots": {},
                "best_pointer": None,
                "pointer_history": [],
                "current_source_steps": [],
                "source_inventory": None,
                "source_evidence": None,
                "source_pointer": None,
                "pending_candidates": {},
                "last_full_scrub_at": None,
                "prune_pending": {},
                "prune_receipts": {},
                "restore_index_generation": 0,
                "restore_index_sha256": None,
                "restore_index_intent": None,
                "pointer_publication_intent": None,
                "heartbeat": None,
            }

    def set_source_inventory(self, steps: Iterable[int]) -> None:
        self.payload["current_source_steps"] = sorted(
            {_strict_int(step, label="source inventory step") for step in steps}
        )

    def set_source_inventory_payload(self, inventory: Mapping[str, Any]) -> None:
        self.payload["source_inventory"] = dict(inventory)
        self.set_source_inventory(
            checkpoint_step_from_name(name)
            for name in inventory.get("checkpoints", {})
        )

    def checkpoint_seen(
        self, manifest: TreeManifest, *, source_fingerprint: str | None = None
    ) -> None:
        assert manifest.step is not None
        key = str(manifest.step)
        record = self.payload["checkpoints"].setdefault(
            key,
            {
                "first_seen_at": utc_now(),
                "local": {"verified": False, "pruned": False},
                "columbus": {"verified": False, "pruned": False},
                "dual_verified": False,
                "recoverable": False,
            },
        )
        old_digest = record.get("manifest_sha256")
        if old_digest is not None and old_digest != manifest.manifest_sha256:
            if (
                record["local"].get("verified")
                or record["columbus"].get("verified")
                or record["local"].get("pruned")
                or record["columbus"].get("pruned")
                or record.get("recovery_intent") is not None
            ):
                raise MirrorError(
                    f"source checkpoint steps_{manifest.step} changed after mirroring began: "
                    f"{old_digest} -> {manifest.manifest_sha256}"
                )
            record["candidate_replaced_at"] = utc_now()
        record.update(
            {
                "manifest_sha256": manifest.manifest_sha256,
                "total_size": manifest.total_size,
                "selection_best_step": manifest.selection_best_step,
                "selection_best_value": manifest.selection_best_value,
                "checkpoint_schema": manifest.checkpoint_schema,
                "source_fingerprint": source_fingerprint,
                "manifest": manifest.as_dict(),
                "last_source_verified_at": utc_now(),
            }
        )

    def eval_seen(
        self,
        artifact: EvalArtifact,
        *,
        source_fingerprint: str,
        validation_context_sha256: str,
    ) -> None:
        if SHA256_RE.fullmatch(validation_context_sha256) is None:
            raise MirrorError("eval validation context SHA-256 is invalid")
        key = str(artifact.step)
        record = self.payload["eval_artifacts"].setdefault(
            key,
            {
                "local": False,
                "columbus": False,
                "dual_verified": False,
            },
        )
        old_sha = record.get("sha256")
        if old_sha is not None and old_sha != artifact.sha256:
            raise MirrorError(
                f"source eval step {artifact.step} changed after authentication"
            )
        record.update(
            {
                **artifact.as_dict(),
                "source_fingerprint": source_fingerprint,
                "validation_context_sha256": validation_context_sha256,
                "artifact": artifact.as_dict(),
                "last_source_verified_at": utc_now(),
            }
        )

    def verify_tier(self, step: int, tier: str, manifest_sha256: str) -> None:
        if tier not in {"local", "columbus"}:
            raise MirrorError(f"unknown mirror tier: {tier}")
        record = self.payload["checkpoints"].get(str(step))
        if not isinstance(record, dict) or record.get("manifest_sha256") != manifest_sha256:
            raise MirrorError(f"cannot verify unknown/mismatched steps_{step} on {tier}")
        record[tier] = {
            "verified": True,
            "pruned": False,
            "manifest_sha256": manifest_sha256,
            "verified_at": utc_now(),
        }
        record["dual_verified"] = bool(
            record["local"].get("verified")
            and not record["local"].get("pruned")
            and record["columbus"].get("verified")
            and not record["columbus"].get("pruned")
        )

    def mark_recoverable(
        self,
        step: int,
        closure: Sequence[int],
        *,
        evidence_manifest_sha256: str,
        receipt_sha256: str,
    ) -> None:
        record = self.payload["checkpoints"].get(str(step), {})
        if record.get("checkpoint_schema") != "selection_v1":
            raise MirrorError(
                f"steps_{step} is archival-only; production recovery requires selection_v1"
            )
        eval_record = self.payload["eval_artifacts"].get(str(step), {})
        if not eval_record.get("dual_verified") or not eval_record.get(
            "production_eligible"
        ):
            raise MirrorError(
                f"steps_{step} cannot be recoverable without its dual-verified, "
                "cryptographically bound production eval"
            )
        evidence_record = self.payload["evidence_snapshots"].get(
            evidence_manifest_sha256, {}
        )
        if not evidence_record.get("dual_verified"):
            raise MirrorError(
                f"steps_{step} cannot be recoverable without dual-verified run evidence"
            )
        if not isinstance(receipt_sha256, str) or SHA256_RE.fullmatch(receipt_sha256) is None:
            raise MirrorError("recovery receipt SHA-256 is invalid")
        for dependency in closure:
            dependency_record = self.payload["checkpoints"].get(str(dependency), {})
            if (
                not dependency_record.get("dual_verified")
                or dependency_record.get("checkpoint_schema") != "selection_v1"
            ):
                raise MirrorError(
                    f"steps_{step} cannot be recoverable; dependency steps_{dependency} "
                    "is not dual verified"
                )
        record["recovery_closure"] = list(closure)
        record["evidence_manifest_sha256"] = evidence_manifest_sha256
        record["receipt_sha256"] = receipt_sha256
        record["recoverable"] = True
        record["recoverable_at"] = utc_now()

    def mark_pruned(self, step: int, tier: str) -> None:
        record = self.payload["checkpoints"].get(str(step))
        if not isinstance(record, dict):
            raise MirrorError(f"cannot prune unknown steps_{step}")
        record[tier]["verified"] = False
        record[tier]["pruned"] = True
        record[tier]["pruned_at"] = utc_now()
        record["dual_verified"] = False
        record["recoverable"] = False

    def record_eval(self, artifact: EvalArtifact, *, tier: str) -> None:
        if tier not in {"local", "columbus"}:
            raise MirrorError(f"unknown eval mirror tier: {tier}")
        record = self.payload["eval_artifacts"].setdefault(
            str(artifact.step),
            {
                "sha256": artifact.sha256,
                "metric_name": artifact.metric_name,
                "metric_mode": artifact.metric_mode,
                "metric_value": artifact.metric_value,
                "trainer_state_sha256": artifact.trainer_state_sha256,
                "cryptographically_bound": artifact.cryptographically_bound,
                "production_eligible": artifact.production_eligible,
                "archival_reason": artifact.archival_reason,
                "local": False,
                "columbus": False,
            },
        )
        for key in (
            "sha256",
            "metric_name",
            "metric_mode",
            "metric_value",
            "trainer_state_sha256",
            "cryptographically_bound",
            "production_eligible",
            "archival_reason",
        ):
            if record.get(key) != getattr(artifact, key):
                raise MirrorError(f"eval artifact state conflict for step {artifact.step}: {key}")
        record[tier] = True
        record[f"{tier}_verified_at"] = utc_now()
        record["dual_verified"] = bool(record["local"] and record["columbus"])

    def record_evidence(self, manifest: TreeManifest, *, tier: str) -> None:
        record = self.payload["evidence_snapshots"].setdefault(
            manifest.manifest_sha256,
            {"total_size": manifest.total_size, "local": False, "columbus": False},
        )
        if record.get("total_size") != manifest.total_size:
            raise MirrorError("evidence snapshot state conflict")
        record["manifest"] = manifest.as_dict()
        record[tier] = True
        record[f"{tier}_verified_at"] = utc_now()
        record["dual_verified"] = bool(record["local"] and record["columbus"])

    def set_pointer_tier(
        self, pointer: SelectionPointer, sha256: str, *, tier: str
    ) -> None:
        if tier not in {"local", "columbus"}:
            raise MirrorError(f"unknown pointer mirror tier: {tier}")
        history = self.payload.setdefault("pointer_history", [])
        existing = self.payload.get("best_pointer")
        identity = {**pointer.as_dict(), "sha256": sha256}
        if existing is not None:
            for key, value in identity.items():
                if existing.get(key) != value:
                    existing = None
                    break
        if existing is None:
            if history:
                prior = history[-1]
                improves = (
                    pointer.metric_value < float(prior["best_metric_value"])
                    if pointer.metric_mode == "min"
                    else pointer.metric_value > float(prior["best_metric_value"])
                )
                if pointer.best_step <= int(prior["best_metric_step"]) or not improves:
                    raise MirrorError(
                        "best-pointer history must advance to a later strict improvement; "
                        "ties retain the prior pointer"
                    )
            existing = {
                **identity,
                "local": False,
                "columbus": False,
                "dual_verified": False,
                "first_observed_at": utc_now(),
            }
            self.payload["best_pointer"] = existing
            history.append(existing)
        existing[tier] = True
        existing[f"{tier}_verified_at"] = utc_now()
        existing["dual_verified"] = bool(existing["local"] and existing["columbus"])
        if history and history[-1].get("sha256") == sha256:
            history[-1] = dict(existing)

    def heartbeat(self, *, status: str, error: str | None = None) -> None:
        source_steps = list(self.payload.get("current_source_steps", []))
        local_steps = [
            int(key)
            for key, value in self.payload["checkpoints"].items()
            if value["local"].get("verified") and not value["local"].get("pruned")
        ]
        columbus_steps = [
            int(key)
            for key, value in self.payload["checkpoints"].items()
            if value["columbus"].get("verified") and not value["columbus"].get("pruned")
        ]
        dual_steps = [
            int(key)
            for key, value in self.payload["checkpoints"].items()
            if value.get("dual_verified")
        ]
        recoverable_steps = [
            int(key)
            for key, value in self.payload["checkpoints"].items()
            if value.get("recoverable")
        ]
        intentionally_pruned_steps = [
            int(key)
            for key, value in self.payload["checkpoints"].items()
            if value.get("local", {}).get("pruned")
            and value.get("columbus", {}).get("pruned")
        ]
        backlog = sorted(
            set(source_steps)
            - set(recoverable_steps)
            - set(intentionally_pruned_steps)
        )
        self.payload["heartbeat"] = {
            "at": utc_now(),
            "status": status,
            "error": error,
            "source_steps": sorted(source_steps),
            "local_steps": sorted(local_steps),
            "columbus_steps": sorted(columbus_steps),
            "dual_steps": sorted(dual_steps),
            "recoverable_steps": sorted(recoverable_steps),
            "backlog_steps": backlog,
            "backlog_count": len(backlog),
        }

    def save(self) -> None:
        self.payload["updated_at"] = utc_now()
        serialized = json.dumps(
            self.payload, indent=2, sort_keys=True, allow_nan=False
        ).encode("utf-8") + b"\n"
        _atomic_write(self.path, serialized)


@contextlib.contextmanager
def exclusive_lock(path: Path):
    _reject_linked_ancestors(path.parent, label="local mirror state directory")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    _ensure_secure_directory(path.parent, create=False, label="local mirror state directory")
    _reject_linked_ancestors(path, label="local mirror lock")
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
        os.close(descriptor)
        raise MirrorError(f"local mirror lock owner/type is unsafe: {path}")
    handle = os.fdopen(descriptor, "a+b")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise MirrorError(f"another mirror owns lock {path}") from exc
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


class CommandRunner:
    """Subprocess wrapper that records dry-run mutations and pulses heartbeat."""

    def __init__(self, *, dry_run: bool = False, heartbeat_seconds: int = 60) -> None:
        self.dry_run = dry_run
        self.heartbeat_seconds = heartbeat_seconds
        self.planned_commands: list[tuple[str, ...]] = []

    def run(
        self,
        args: Sequence[str],
        *,
        input_bytes: bytes | None = None,
        mutation: bool = False,
        heartbeat: Callable[[], None] | None = None,
        allowed_returncodes: frozenset[int] = frozenset({0}),
    ) -> subprocess.CompletedProcess[str]:
        command = tuple(str(arg) for arg in args)
        if self.dry_run and mutation:
            self.planned_commands.append(command)
            return subprocess.CompletedProcess(command, 0, "", "")
        with tempfile.TemporaryFile() as output:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE if input_bytes is not None else subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
            )
            if input_bytes is not None:
                assert process.stdin is not None
                process.stdin.write(input_bytes)
                process.stdin.close()
            while True:
                try:
                    return_code = process.wait(timeout=self.heartbeat_seconds)
                    break
                except subprocess.TimeoutExpired:
                    if heartbeat is not None:
                        heartbeat()
            output.seek(0)
            decoded = output.read().decode("utf-8", errors="replace")
        if return_code not in allowed_returncodes:
            raise MirrorError(
                f"command failed with exit {return_code}: {shlex.join(command)}\n{decoded[-8000:]}"
            )
        return subprocess.CompletedProcess(command, return_code, decoded, "")


class RemoteEndpoint:
    """SSH/rsync endpoint using this exact file as its remote validator."""

    def __init__(self, host: str, runner: CommandRunner) -> None:
        if (
            not host
            or host.startswith("-")
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,252}", host) is None
        ):
            raise MirrorError(f"unsafe SSH host spelling: {host!r}")
        self.host = host
        self.runner = runner
        self.helper_source = Path(__file__).read_bytes()
        self.ssh_options = (
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            "ConnectTimeout=20",
        )

    def invoke(
        self,
        arguments: Sequence[str],
        *,
        mutation: bool = False,
        heartbeat: Callable[[], None] | None = None,
    ) -> str:
        remote_command = shlex.join(("python3", "-", *map(str, arguments)))
        result = self.runner.run(
            ("ssh", *self.ssh_options, self.host, remote_command),
            input_bytes=self.helper_source,
            mutation=mutation,
            heartbeat=heartbeat,
        )
        return result.stdout

    def invoke_json(
        self,
        arguments: Sequence[str],
        *,
        mutation: bool = False,
        heartbeat: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        output = self.invoke(arguments, mutation=mutation, heartbeat=heartbeat)
        try:
            payload = json.loads(output)
        except json.JSONDecodeError as exc:
            raise MirrorError(
                f"remote helper on {self.host} returned invalid JSON: {output[-2000:]}"
            ) from exc
        if not isinstance(payload, dict):
            raise MirrorError(f"remote helper on {self.host} did not return an object")
        return payload

    @property
    def rsync_shell(self) -> str:
        return shlex.join(("ssh", *self.ssh_options))

    def rsync_to_local(
        self,
        source: str,
        destination: Path,
        *,
        files_from: Path | None = None,
        delete: bool = False,
        heartbeat: Callable[[], None] | None = None,
    ) -> None:
        args = [
            "rsync",
            "--recursive",
            "--links",
            "--times",
            "--protect-args",
            "--partial",
            "--partial-dir=.rsync-partial",
            "--fsync",
            "--no-inplace",
            "--checksum",
            "--chmod=D700,F600",
            "-e",
            self.rsync_shell,
        ]
        if files_from is not None:
            args.extend(("--from0", f"--files-from={files_from}"))
        if delete:
            args.append("--delete-delay")
            if files_from is not None:
                args.append("--delete-excluded")
        args.extend((f"{self.host}:{source}", str(destination)))
        self.runner.run(args, mutation=True, heartbeat=heartbeat)

    def rsync_from_local(
        self,
        source: Path | str,
        destination: str,
        *,
        delete: bool = False,
        heartbeat: Callable[[], None] | None = None,
    ) -> None:
        args = (
            "rsync",
            "--recursive",
            "--links",
            "--times",
            "--protect-args",
            "--partial",
            "--partial-dir=.rsync-partial",
            "--fsync",
            "--no-inplace",
            "--checksum",
            "--chmod=D700,F600",
            "-e",
            self.rsync_shell,
        )
        mutable_args = list(args)
        if delete:
            mutable_args.append("--delete-delay")
        mutable_args.extend((str(source), f"{self.host}:{destination}"))
        self.runner.run(mutable_args, mutation=True, heartbeat=heartbeat)


def _parse_ssh_config(output: str) -> dict[str, list[str]]:
    parsed: dict[str, list[str]] = {}
    for line in output.splitlines():
        key, separator, value = line.partition(" ")
        if not separator:
            continue
        parsed.setdefault(key.strip().lower(), []).append(value.strip())
    return parsed


def resolve_endpoint_identity(endpoint: RemoteEndpoint) -> dict[str, Any]:
    """Resolve and authenticate the SSH alias without accepting new host keys."""

    result = endpoint.runner.run(("ssh", "-G", "--", endpoint.host))
    config = _parse_ssh_config(result.stdout)
    try:
        hostname = config["hostname"][-1]
        user = config["user"][-1]
        port_text = config["port"][-1]
    except KeyError as exc:
        raise MirrorError(f"ssh -G omitted endpoint identity for {endpoint.host}") from exc
    if (
        not hostname
        or hostname.startswith("-")
        or any(character.isspace() for character in hostname)
    ):
        raise MirrorError(f"resolved SSH hostname is unsafe: {hostname!r}")
    port = _strict_int(int(port_text), label="resolved SSH port", minimum=1)
    if port > 65535 or not user or user.startswith("-") or any(c.isspace() for c in user):
        raise MirrorError("resolved SSH user/port is unsafe")
    known_host_lines: list[str] = []
    host_key_alias = config.get("hostkeyalias", [""])[-1]
    lookup_host = host_key_alias or hostname
    if not host_key_alias and port != 22:
        lookup_host = f"[{hostname}]:{port}"
    known_host_files = config.get("userknownhostsfile", [])
    for configured in known_host_files:
        for raw_path in shlex.split(configured):
            known_path = Path(os.path.expanduser(raw_path))
            if not known_path.is_file():
                continue
            lookup = endpoint.runner.run(
                ("ssh-keygen", "-F", lookup_host, "-f", str(known_path)),
                allowed_returncodes=frozenset({0, 1}),
            )
            known_host_lines.extend(
                line for line in lookup.stdout.splitlines() if line and not line.startswith("#")
            )
    if not known_host_lines:
        raise MirrorError(
            f"no pinned known-host key found for resolved endpoint {hostname!r}"
        )
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8") as handle:
        handle.write("\n".join(sorted(set(known_host_lines))) + "\n")
        handle.flush()
        fingerprints = endpoint.runner.run(("ssh-keygen", "-lf", handle.name)).stdout
    fingerprint_lines: list[str] = []
    for line in fingerprints.splitlines():
        fields = line.split()
        if len(fields) < 4:
            continue
        # Store only stable key size, fingerprint, and algorithm.  Known-host
        # comments are mutable and are not endpoint identity.
        fingerprint_lines.append(f"{fields[0]} {fields[1]} {fields[-1]}")
    fingerprint_lines = sorted(set(fingerprint_lines))
    if not fingerprint_lines:
        raise MirrorError(f"unable to fingerprint known-host key for {hostname}")
    remote = endpoint.invoke_json(("--internal-endpoint-identity",))
    if set(remote) != {"user", "hostname", "fqdn", "python_version"}:
        raise MirrorError("remote endpoint identity response shape is invalid")
    if remote.get("user") != user:
        raise MirrorError(
            f"SSH resolved user {user!r} differs from remote user {remote.get('user')!r}"
        )
    return {
        "alias": endpoint.host,
        "resolved_hostname": hostname,
        "resolved_user": user,
        "resolved_port": port,
        "host_key_alias": host_key_alias or None,
        "known_host_fingerprints": fingerprint_lines,
        "remote_hostname": remote["hostname"],
        "remote_fqdn": remote["fqdn"],
        "remote_python_version": remote["python_version"],
    }


def resolve_remote_source_root_binding(
    endpoint: RemoteEndpoint, logical_path: str
) -> dict[str, Any]:
    payload = endpoint.invoke_json(("--internal-resolve-source-root", logical_path))
    if set(payload) != {
        "schema_version",
        "logical_path",
        "resolved_path",
        "device",
        "inode",
        "uid",
        "symlink_ancestors",
    }:
        raise MirrorError("remote source-root binding shape is invalid")
    _require_schema_version(
        payload.get("schema_version"), label="source-root binding schema_version"
    )
    if payload.get("logical_path") != logical_path:
        raise MirrorError("remote source-root binding logical path mismatch")
    resolved_path = payload.get("resolved_path")
    if not isinstance(resolved_path, str) or not _safe_absolute_posix_path(
        resolved_path
    ):
        raise MirrorError("remote source-root binding resolved path is unsafe")
    for field in ("device", "inode", "uid"):
        _strict_int(payload.get(field), label=f"source-root binding {field}")
    ancestors = payload.get("symlink_ancestors")
    if not isinstance(ancestors, list):
        raise MirrorError("remote source-root binding ancestor list is malformed")
    for ancestor in ancestors:
        if (
            not isinstance(ancestor, dict)
            or set(ancestor) != {"path", "target"}
            or not isinstance(ancestor.get("path"), str)
            or not _safe_absolute_posix_path(ancestor["path"])
            or not isinstance(ancestor.get("target"), str)
            or not ancestor["target"]
            or "\x00" in ancestor["target"]
        ):
            raise MirrorError("remote source-root binding ancestor is malformed")
    return dict(payload)


def _manifest_b64(manifest: TreeManifest) -> str:
    return base64.urlsafe_b64encode(_canonical_json_bytes(manifest.as_dict())).decode("ascii")


def _manifest_from_b64(value: str) -> TreeManifest:
    try:
        payload = json.loads(base64.urlsafe_b64decode(value.encode("ascii")))
    except Exception as exc:
        raise MirrorError("invalid encoded manifest") from exc
    if not isinstance(payload, dict):
        raise MirrorError("encoded manifest is not an object")
    return TreeManifest.from_dict(payload)


def _ensure_secure_directory(path: Path, *, create: bool, label: str) -> None:
    if create:
        path.mkdir(mode=0o700, parents=False, exist_ok=True)
    info = _validate_root_directory(path, label=label)
    if info.st_uid != os.getuid():
        raise MirrorError(f"{label} is not owned by current uid: {path}")
    if stat.S_IMODE(info.st_mode) & 0o077:
        raise MirrorError(f"{label} must not grant group/world access: {path}")


def _reject_linked_ancestors(path: Path, *, label: str, stop: Path | None = None) -> None:
    """lstat every existing component; resolving a symlink is never acceptable."""

    if not path.is_absolute():
        raise MirrorError(f"{label} must be absolute: {path}")
    current = Path(path.anchor)
    stop_path = stop
    for part in path.parts[1:]:
        current = current / part
        if stop_path is not None and current == stop_path.parent:
            continue
        try:
            info = current.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode):
            raise MirrorError(f"{label} rejects symlinked ancestor: {current}")


def _secure_regular_file(path: Path, *, label: str) -> os.stat_result:
    _reject_linked_ancestors(path, label=label)
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise MirrorError(f"{label} does not exist: {path}") from exc
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise MirrorError(f"{label} must be a real regular file: {path}")
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise MirrorError(f"{label} owner/mode is not private: {path}")
    return info


def _ensure_columbus_root(run_id: str, root: Path) -> None:
    validate_columbus_root(run_id, root)
    _ensure_secure_directory(SAFE_COLUMBUS_PARENT, create=False, label="protected relay parent")
    if not SAFE_COLUMBUS_BASE.exists():
        SAFE_COLUMBUS_BASE.mkdir(mode=0o700)
    _ensure_secure_directory(SAFE_COLUMBUS_BASE, create=False, label="mirror storage base")
    if not root.exists():
        root.mkdir(mode=0o700)
    _ensure_secure_directory(root, create=False, label="per-run mirror root")
    for child in (
        "checkpoints",
        "heldout_eval_metrics",
        "evidence",
        "manifests",
        "best_checkpoint_history",
    ):
        path = root / child
        if not path.exists():
            path.mkdir(mode=0o700)
        _ensure_secure_directory(path, create=False, label=f"mirror {child} directory")


def _require_contained(path: Path, parent: Path, *, label: str) -> None:
    for candidate, candidate_label in ((path, label), (parent, f"{label} parent")):
        if not candidate.is_absolute() or any(
            part in {"", ".", ".."} for part in candidate.parts[1:]
        ):
            raise MirrorError(
                f"{candidate_label} must be a canonical absolute path without dot segments: "
                f"{candidate}"
            )
    try:
        path.relative_to(parent)
    except ValueError as exc:
        raise MirrorError(f"{label} escapes protected root: {path}") from exc


def _remove_stage(path: Path, parent: Path) -> None:
    _require_contained(path, parent, label="staging path")
    if not path.name.startswith(".incoming-"):
        raise MirrorError(f"refusing to remove non-staging path: {path}")
    if path.is_symlink():
        raise MirrorError(f"refusing linked staging path: {path}")
    if path.is_dir():
        shutil.rmtree(path)
    else:
        with contextlib.suppress(FileNotFoundError):
            path.unlink()


def _prepare_local_stage(path: Path, *, directory: bool) -> None:
    if not path.name.startswith(".incoming-"):
        raise MirrorError(f"local staging basename is invalid: {path.name}")
    if path.exists() or path.is_symlink():
        info = path.lstat()
        valid = stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)
        if stat.S_ISLNK(info.st_mode) or not valid:
            raise MirrorError(f"existing local staging object has wrong type: {path}")
        if not directory and info.st_nlink != 1:
            raise MirrorError(f"existing local file stage is hard-linked: {path}")
        return
    if directory:
        path.mkdir(mode=0o700)


def _promote_local_tree(
    stage: Path,
    final: Path,
    *,
    expected: TreeManifest,
    metric_name: str,
    metric_mode: str,
) -> None:
    if stage.parent != final.parent or not stage.name.startswith(".incoming-"):
        raise MirrorError("tree staging and promotion must use a hidden same-directory path")
    if expected.kind == "checkpoint":
        if expected.step is None:
            raise MirrorError("checkpoint promotion manifest lacks step")
        actual = validate_checkpoint_tree(
            stage,
            metric_name=metric_name,
            metric_mode=metric_mode,
            expected_step=expected.step,
        )
    elif expected.kind == "run_evidence":
        actual = validate_evidence_snapshot(stage)
    else:
        raise MirrorError(f"unsupported tree promotion kind: {expected.kind}")
    if actual.manifest_sha256 != expected.manifest_sha256:
        raise MirrorError("staged tree bytes do not match source manifest")
    if final.exists() or final.is_symlink():
        if expected.kind == "checkpoint":
            existing = validate_checkpoint_tree(
                final, metric_name=metric_name, metric_mode=metric_mode
            )
        else:
            existing = validate_evidence_snapshot(final)
        if existing.manifest_sha256 != expected.manifest_sha256:
            raise MirrorError(f"immutable destination conflict: {final}")
        _remove_stage(stage, final.parent)
        return
    os.replace(stage, final)
    directory_fd = os.open(final.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _copy_file_atomic_local(stage: Path, final: Path, *, expected_sha256: str) -> None:
    record = _regular_record(stage, relative_path=stage.name, label="staged file")
    if record.sha256 != expected_sha256:
        raise MirrorError("staged file bytes do not match source SHA-256")
    if final.exists() or final.is_symlink():
        existing = _regular_record(final, relative_path=final.name, label="destination file")
        if existing.sha256 != expected_sha256:
            raise MirrorError(f"immutable destination conflict: {final}")
        stage.unlink()
        return
    os.replace(stage, final)
    directory_fd = os.open(final.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _local_mirror_root(path: Path, *, create: bool) -> None:
    if not path.is_absolute():
        raise MirrorError("local mirror root must be absolute")
    _reject_linked_ancestors(path, label="local mirror root")
    if create:
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
    _ensure_secure_directory(path, create=False, label="local mirror root")
    for child in ("checkpoints", "heldout_eval_metrics", "evidence", "manifests"):
        subdir = path / child
        if create:
            subdir.mkdir(mode=0o700, exist_ok=True)
        _ensure_secure_directory(
            subdir, create=False, label=f"local mirror {child}"
        )


def _remote_checkpoint_manifest(
    endpoint: RemoteEndpoint,
    path: str,
    *,
    metric_name: str,
    metric_mode: str,
) -> TreeManifest:
    payload = endpoint.invoke_json(
        ("--internal-inspect-checkpoint", path, metric_name, metric_mode)
    )
    return TreeManifest.from_dict(payload)


def _remote_evidence_identity(
    endpoint: RemoteEndpoint,
    access_run_dir: str,
    *,
    source_output_dir: str,
) -> RunEvidenceIdentity:
    payload = endpoint.invoke_json(
        ("--internal-inspect-evidence", access_run_dir, source_output_dir)
    )
    return RunEvidenceIdentity.from_dict(payload)


def _remote_eval(
    endpoint: RemoteEndpoint,
    path: str,
    *,
    manifest: TreeManifest,
    metric_name: str,
    metric_mode: str,
    run_id: str,
    run_evidence: RunEvidenceIdentity,
    allow_legacy_archive: bool = False,
) -> EvalArtifact:
    payload = endpoint.invoke_json(
        (
            "--internal-inspect-eval",
            path,
            _manifest_b64(manifest),
            metric_name,
            metric_mode,
            run_id,
            _evidence_b64(run_evidence),
            "1" if allow_legacy_archive else "0",
        )
    )
    try:
        return EvalArtifact(
            step=_strict_int(payload["step"], label="remote eval step"),
            trainer_state_sha256=str(payload["trainer_state_sha256"]),
            metric_name=str(payload["metric_name"]),
            metric_mode=str(payload["metric_mode"]),
            metric_value=_finite_number(payload["metric_value"], label="remote eval metric"),
            sha256=str(payload["sha256"]),
            size=_strict_int(payload["size"], label="remote eval size", minimum=1),
            cryptographically_bound=payload.get("cryptographically_bound") is True,
            production_eligible=payload.get("production_eligible") is True,
            archival_reason=(
                None
                if payload.get("archival_reason") is None
                else str(payload.get("archival_reason"))
            ),
            source_kind=str(payload.get("source_kind", "checkpoint")),
        )
    except KeyError as exc:
        raise MirrorError("remote eval result is incomplete") from exc


def _remote_baseline_eval(
    endpoint: RemoteEndpoint,
    path: str,
    *,
    metric_name: str,
    metric_mode: str,
    run_id: str,
    run_evidence: RunEvidenceIdentity,
) -> EvalArtifact:
    payload = endpoint.invoke_json(
        (
            "--internal-inspect-baseline-eval",
            path,
            metric_name,
            metric_mode,
            run_id,
            _evidence_b64(run_evidence),
        )
    )
    return EvalArtifact.from_dict(payload)


def _remote_pointer(
    endpoint: RemoteEndpoint,
    path: str,
    *,
    metric_name: str,
    metric_mode: str,
    maximum_step: int | None,
) -> tuple[SelectionPointer, FileRecord, bytes]:
    payload = endpoint.invoke_json(
        (
            "--internal-inspect-pointer",
            path,
            metric_name,
            metric_mode,
            "none" if maximum_step is None else str(maximum_step),
        )
    )
    pointer_payload = payload.get("pointer")
    record_payload = payload.get("record")
    encoded = payload.get("content_b64")
    if (
        not isinstance(pointer_payload, dict)
        or not isinstance(record_payload, dict)
        or not isinstance(encoded, str)
    ):
        raise MirrorError("remote pointer inspection is incomplete")
    pointer = validate_selection_payload(
        pointer_payload,
        metric_name=metric_name,
        metric_mode=metric_mode,
        maximum_step=maximum_step,
    )
    record = FileRecord(
        path=str(record_payload["path"]),
        size=_strict_int(record_payload["size"], label="pointer size", minimum=1),
        sha256=str(record_payload["sha256"]),
    )
    if SHA256_RE.fullmatch(record.sha256) is None:
        raise MirrorError("remote pointer SHA-256 is invalid")
    content = base64.b64decode(encoded)
    if len(content) != record.size or hashlib.sha256(content).hexdigest() != record.sha256:
        raise MirrorError("remote pointer returned bytes do not match its digest")
    return pointer, record, content


def _remote_available_bytes(endpoint: RemoteEndpoint, path: str) -> int:
    payload = endpoint.invoke_json(("--internal-available-bytes", path))
    return _strict_int(payload.get("available_bytes"), label="remote available bytes")


def _remote_file_prefix_sha256(
    endpoint: RemoteEndpoint, path: str, length: int
) -> dict[str, Any]:
    payload = endpoint.invoke_json(
        ("--internal-file-prefix-sha256", path, str(length))
    )
    expected_length = _strict_int(length, label="expected prefix length")
    if set(payload) != {"size", "prefix_length", "prefix_sha256"}:
        raise MirrorError("remote append-only prefix result is malformed")
    size = _strict_int(payload.get("size"), label="remote append-only file size")
    prefix_length = _strict_int(
        payload.get("prefix_length"), label="remote append-only prefix length"
    )
    digest = payload.get("prefix_sha256")
    if (
        prefix_length != expected_length
        or size < prefix_length
        or not isinstance(digest, str)
        or SHA256_RE.fullmatch(digest) is None
    ):
        raise MirrorError("remote append-only prefix result is invalid")
    return {
        "size": size,
        "prefix_length": prefix_length,
        "prefix_sha256": digest,
    }


def _require_remote_space(
    endpoint: RemoteEndpoint,
    path: str,
    *,
    incoming_bytes: int,
    reserve_bytes: int,
) -> None:
    available = _remote_available_bytes(endpoint, path)
    required = incoming_bytes + reserve_bytes
    if available < required:
        raise MirrorError(
            f"insufficient free space on {endpoint.host}:{path}: "
            f"available={available}, required={required}"
        )


@dataclasses.dataclass(frozen=True)
class MirrorConfig:
    run_id: str
    h100_host: str
    source_run_dir: str
    local_root: Path
    columbus_host: str
    columbus_root: Path
    state_dir: Path
    metric_name: str
    metric_mode: str
    retain: int = DEFAULT_RETAIN
    reserve_bytes: int = DEFAULT_DISK_RESERVE_BYTES
    max_backlog: int = 2
    full_scrub_hours: float = DEFAULT_FULL_SCRUB_HOURS
    pending_timeout_seconds: int = 1800

    def validate(self) -> None:
        if RUN_ID_RE.fullmatch(self.run_id) is None:
            raise MirrorError(f"unsafe run_id: {self.run_id!r}")
        source = PurePosixPath(self.source_run_dir)
        if (
            not _safe_absolute_posix_path(self.source_run_dir)
            or source.name != self.run_id
        ):
            raise MirrorError(
                "source run directory must be absolute and end in the exact run_id"
            )
        if self.metric_mode not in {"min", "max"} or not self.metric_name:
            raise MirrorError("selection metric name/mode are invalid")
        if self.retain < 2:
            raise MirrorError("retention must be at least 2 to preserve best plus newest")
        if self.reserve_bytes < 0 or self.max_backlog < 0:
            raise MirrorError("reserve/backlog values must be non-negative")
        if self.pending_timeout_seconds < 60:
            raise MirrorError("pending finalization timeout must be at least 60 seconds")
        validate_columbus_root(self.run_id, self.columbus_root)
        if (
            not self.local_root.is_absolute()
            or self.local_root.name != self.run_id
            or any(part in {".", ".."} for part in self.local_root.parts)
        ):
            raise MirrorError("local mirror root must be absolute and end in run_id")
        if not self.state_dir.is_absolute() or any(
            part in {".", ".."} for part in self.state_dir.parts
        ):
            raise MirrorError("state directory must be canonical and absolute")
        if not math.isfinite(self.full_scrub_hours) or self.full_scrub_hours < 1.0:
            raise MirrorError("full scrub interval must be at least one hour")


@dataclasses.dataclass(frozen=True)
class SourceScan:
    manifests: Mapping[int, TreeManifest]
    evals: Mapping[int, EvalArtifact]
    pointer: SelectionPointer | None
    pointer_record: FileRecord | None
    pointer_content: bytes | None
    evidence: RunEvidenceIdentity
    finalized_steps: tuple[int, ...]
    pending_steps: tuple[int, ...]
    inventory: Mapping[str, Any]
    full_scrub: bool
    changed: bool


def _parse_utc(value: Any, *, label: str) -> dt.datetime:
    if not isinstance(value, str):
        raise MirrorError(f"{label} must be an ISO UTC timestamp")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise MirrorError(f"{label} is not a valid timestamp") from exc
    if parsed.tzinfo is None:
        raise MirrorError(f"{label} must include a timezone")
    return parsed.astimezone(dt.timezone.utc)


def _checkpoint_light_complete(tree: Mapping[str, Any]) -> tuple[bool, str | None]:
    root = tree.get("root")
    entries = tree.get("entries")
    if not isinstance(root, dict) or root.get("kind") != "directory":
        return False, "checkpoint path is not a real directory"
    if not isinstance(entries, dict):
        return False, "checkpoint directory inventory is malformed"
    names = set(entries)
    missing = sorted(REQUIRED_CHECKPOINT_FILES - names)
    unexpected = sorted(names - ALLOWED_CHECKPOINT_FILES)
    if missing or unexpected:
        return False, f"missing={missing}, unexpected={unexpected}"
    for name, item in entries.items():
        if not isinstance(item, dict) or item.get("kind") != "file":
            return False, f"{name} is not a regular file"
        if item.get("nlink") != 1 or not isinstance(item.get("size"), int) or item["size"] <= 0:
            return False, f"{name} is empty or hard-linked"
    return True, None


def _cached_manifests(state: MirrorState) -> dict[int, TreeManifest]:
    result: dict[int, TreeManifest] = {}
    for key, record in state.payload.get("checkpoints", {}).items():
        if not isinstance(record, dict) or not isinstance(record.get("manifest"), dict):
            continue
        step = _strict_int(int(key), label="cached checkpoint step")
        manifest = TreeManifest.from_dict(record["manifest"])
        if manifest.step != step:
            raise MirrorError("cached checkpoint manifest key/step mismatch")
        result[step] = manifest
    return result


def _cached_evals(state: MirrorState) -> dict[int, EvalArtifact]:
    result: dict[int, EvalArtifact] = {}
    for key, record in state.payload.get("eval_artifacts", {}).items():
        if not isinstance(record, dict) or not isinstance(record.get("artifact"), dict):
            continue
        step = _strict_int(int(key), label="cached eval step")
        artifact = EvalArtifact.from_dict(record["artifact"])
        if artifact.step != step:
            raise MirrorError("cached eval artifact key/step mismatch")
        result[step] = artifact
    return result


class MirrorController:
    def __init__(self, config: MirrorConfig, runner: CommandRunner) -> None:
        config.validate()
        self.config = config
        self.runner = runner
        self.source = RemoteEndpoint(config.h100_host, runner)
        self.columbus = RemoteEndpoint(config.columbus_host, runner)
        self.state_path = config.state_dir / config.run_id / "state.json"
        self.lock_path = config.state_dir / config.run_id / "mirror.lock"
        self.heartbeat_path = config.state_dir / config.run_id / "heartbeat.json"
        self.state: MirrorState | None = None
        self._remote_lock_token: str | None = None
        self.h100_identity: dict[str, Any] | None = None
        self.columbus_identity: dict[str, Any] | None = None
        self.source_access_dir: str | None = None

    def _resolve_h100_identity(self) -> dict[str, Any]:
        identity = resolve_endpoint_identity(self.source)
        binding = resolve_remote_source_root_binding(
            self.source, self.config.source_run_dir
        )
        resolved_path = binding["resolved_path"]
        if PurePosixPath(resolved_path).name != self.config.run_id:
            raise MirrorError("resolved source run path does not end in the exact run_id")
        return {**identity, "source_root_binding": binding}

    def _set_source_access_from_identity(self, identity: Mapping[str, Any]) -> None:
        binding = identity.get("source_root_binding")
        if (
            not isinstance(binding, dict)
            or binding.get("logical_path") != self.config.source_run_dir
            or not isinstance(binding.get("resolved_path"), str)
            or not _safe_absolute_posix_path(binding["resolved_path"])
            or PurePosixPath(binding["resolved_path"]).name != self.config.run_id
        ):
            raise MirrorError("pinned H100 source-root binding is invalid")
        self.source_access_dir = binding["resolved_path"]

    def _pulse(self, status: str = "working") -> None:
        if self.state is None:
            return
        self.state.heartbeat(status=status)
        self.state.save()
        heartbeat = self.state.payload.get("heartbeat")
        assert isinstance(heartbeat, dict)
        _atomic_write(
            self.heartbeat_path,
            json.dumps(heartbeat, indent=2, sort_keys=True).encode("utf-8") + b"\n",
        )

    def _init_state(self, *, defer_h100_resolution: bool = False) -> bool:
        # A persisted state can supply the already-pinned H100 identity while
        # crash-pending Columbus prune work is reconciled.  This is deliberately
        # only a deferral: the source alias/key is re-resolved before any source
        # inventory access later in the same poll.
        pinned_h100: dict[str, Any] | None = None
        if defer_h100_resolution and (self.state_path.exists() or self.state_path.is_symlink()):
            _secure_regular_file(self.state_path, label="mirror state")
            raw_state = _load_json_object(self.state_path, label="mirror state")
            identity = raw_state.get("identity")
            if not isinstance(identity, dict) or not isinstance(
                identity.get("h100_endpoint"), dict
            ):
                raise MirrorError("mirror state lacks a pinned H100 endpoint identity")
            pinned_h100 = dict(identity["h100_endpoint"])
        h100_resolved = pinned_h100 is None
        self.h100_identity = self._resolve_h100_identity() if h100_resolved else pinned_h100
        assert self.h100_identity is not None
        self._set_source_access_from_identity(self.h100_identity)
        self.columbus_identity = resolve_endpoint_identity(self.columbus)
        assert self.h100_identity is not None and self.columbus_identity is not None
        if (
            self.h100_identity["remote_fqdn"]
            == self.columbus_identity["remote_fqdn"]
            or self.h100_identity["remote_hostname"]
            == self.columbus_identity["remote_hostname"]
        ):
            raise MirrorError("H100 source and Columbus destination resolve to the same machine")
        self.state = MirrorState(
            self.state_path,
            run_id=self.config.run_id,
            source_run_dir=self.config.source_run_dir,
            local_root=str(self.config.local_root),
            columbus_root=str(self.config.columbus_root),
            metric_name=self.config.metric_name,
            metric_mode=self.config.metric_mode,
            h100_identity=self.h100_identity,
            columbus_identity=self.columbus_identity,
        )
        return h100_resolved

    def _revalidate_h100_identity(self) -> None:
        assert self.state is not None and self.columbus_identity is not None
        resolved = self._resolve_h100_identity()
        expected = self.state.payload["identity"]["h100_endpoint"]
        if resolved != expected:
            raise MirrorError("mirror state identity/configuration drift")
        if (
            resolved["remote_fqdn"] == self.columbus_identity["remote_fqdn"]
            or resolved["remote_hostname"] == self.columbus_identity["remote_hostname"]
        ):
            raise MirrorError("H100 source and Columbus destination resolve to the same machine")
        self.h100_identity = resolved
        self._set_source_access_from_identity(resolved)

    def _source_listing(self) -> dict[str, Any]:
        if self.source_access_dir is None:
            raise MirrorError("H100 source-root binding is not initialized")
        return self.source.invoke_json(
            ("--internal-light-inventory", self.source_access_dir)
        )

    def _full_scrub_due(self) -> bool:
        assert self.state is not None
        value = self.state.payload.get("last_full_scrub_at")
        if value is None:
            checkpoints = self.state.payload.get("checkpoints", {})
            checkpoint_verification_exists = any(
                isinstance(record, dict)
                and (
                    record.get("dual_verified") is True
                    or record.get("recoverable") is True
                    or record.get("local", {}).get("verified") is True
                    or record.get("columbus", {}).get("verified") is True
                )
                for record in checkpoints.values()
            )
            eval_verification_exists = any(
                isinstance(record, dict)
                and (
                    record.get("dual_verified") is True
                    or record.get("local") is True
                    or record.get("columbus") is True
                )
                for record in self.state.payload.get("eval_artifacts", {}).values()
            )
            evidence_verification_exists = any(
                isinstance(record, dict)
                and (
                    record.get("dual_verified") is True
                    or record.get("local") is True
                    or record.get("columbus") is True
                )
                for record in self.state.payload.get("evidence_snapshots", {}).values()
            )
            return bool(
                checkpoint_verification_exists
                or eval_verification_exists
                or evidence_verification_exists
                or self.state.payload.get("restore_index_generation", 0)
                or (
                    isinstance(self.state.payload.get("best_pointer"), dict)
                    and self.state.payload["best_pointer"].get("dual_verified") is True
                )
            )
        age = dt.datetime.now(dt.timezone.utc) - _parse_utc(
            value, label="last_full_scrub_at"
        )
        return age.total_seconds() >= self.config.full_scrub_hours * 3600.0

    def inspect_source(self, *, force_full_scrub: bool = False) -> SourceScan:
        """Authenticate changed/finalized candidates while unchanged polls stay stat-only."""

        if self.state is None:
            raise MirrorError("mirror state must be initialized before source inspection")
        inventory = self._source_listing()
        if set(inventory) != {
            "schema_version",
            "run_dir",
            "run_root",
            "checkpoints",
            "unexpected_checkpoint_entries",
            "eval_artifacts",
            "unexpected_eval_entries",
            "evidence",
            "evidence_fingerprint",
            "best_checkpoint_pointer",
        }:
            raise MirrorError("source lightweight inventory shape is malformed")
        _require_schema_version(
            inventory.get("schema_version"), label="source inventory schema_version"
        )
        if self.source_access_dir is None:
            raise MirrorError("H100 source-root binding is not initialized")
        source_dir = self.source_access_dir
        if inventory.get("run_dir") != source_dir:
            raise MirrorError("source inventory resolved path differs from configured path")
        binding = self.h100_identity.get("source_root_binding", {})
        root_identity = inventory.get("run_root")
        if (
            not isinstance(binding, dict)
            or not isinstance(root_identity, dict)
            or root_identity.get("kind") != "directory"
            or root_identity.get("device") != binding.get("device")
            or root_identity.get("inode") != binding.get("inode")
            or root_identity.get("uid") != binding.get("uid")
        ):
            raise MirrorError("source inventory root differs from pinned realpath binding")
        unexpected = inventory.get("unexpected_checkpoint_entries")
        if not isinstance(unexpected, list) or any(
            not isinstance(item, str) for item in unexpected
        ):
            raise MirrorError("source unexpected-entry inventory is malformed")
        if unexpected:
            raise MirrorError(
                f"source checkpoints directory has unexpected entries: {unexpected}"
            )
        unexpected_evals = inventory.get("unexpected_eval_entries")
        if not isinstance(unexpected_evals, list) or any(
            not isinstance(item, str) for item in unexpected_evals
        ):
            raise MirrorError("source unexpected-eval inventory is malformed")
        if unexpected_evals:
            raise MirrorError(
                f"source eval directory has unexpected/noncanonical entries: "
                f"{unexpected_evals}"
            )
        checkpoint_inventory = inventory.get("checkpoints")
        eval_inventory = inventory.get("eval_artifacts")
        if not isinstance(checkpoint_inventory, dict) or not isinstance(
            eval_inventory, dict
        ):
            raise MirrorError("source checkpoint/eval inventory is malformed")

        full_scrub = force_full_scrub or self._full_scrub_due()
        cached_manifest_map = _cached_manifests(self.state)
        cached_eval_map = _cached_evals(self.state)
        manifests = dict(cached_manifest_map)
        evals = dict(cached_eval_map)
        changed = full_scrub
        old_pending = self.state.payload.get("pending_candidates", {})
        if not isinstance(old_pending, dict):
            raise MirrorError("pending candidate state is malformed")

        evidence_fingerprint = inventory.get("evidence_fingerprint")
        if not isinstance(evidence_fingerprint, str) or SHA256_RE.fullmatch(
            evidence_fingerprint
        ) is None:
            raise MirrorError("source evidence metadata fingerprint is invalid")
        evidence_cache = self.state.payload.get("source_evidence")
        if (
            not full_scrub
            and isinstance(evidence_cache, dict)
            and evidence_cache.get("fingerprint") == evidence_fingerprint
            and isinstance(evidence_cache.get("identity"), dict)
        ):
            evidence = RunEvidenceIdentity.from_dict(evidence_cache["identity"])
        else:
            evidence = _remote_evidence_identity(
                self.source,
                source_dir,
                source_output_dir=self.config.source_run_dir,
            )
            if evidence.run_id != self.config.run_id:
                raise MirrorError("run evidence run_id differs from configured run_id")
            changed = True
        immutable_evidence_files = _immutable_evidence_files(evidence)
        immutable_evidence_sha256 = _immutable_evidence_sha256(evidence)
        if isinstance(evidence_cache, dict) and isinstance(
            evidence_cache.get("identity"), dict
        ):
            previous_evidence = RunEvidenceIdentity.from_dict(evidence_cache["identity"])
            previous_files = _immutable_evidence_files(previous_evidence)
            previous_sha256 = _immutable_evidence_sha256(previous_evidence)
            stored_files = evidence_cache.get("immutable_files", previous_files)
            stored_sha256 = evidence_cache.get(
                "immutable_evidence_sha256", previous_sha256
            )
            if stored_files != previous_files or stored_sha256 != previous_sha256:
                raise MirrorError("cached immutable run evidence binding is corrupt")
            if (
                immutable_evidence_files != previous_files
                or immutable_evidence_sha256 != previous_sha256
            ):
                raise MirrorError(
                    "immutable run evidence changed after authentication"
                )
            previous_summary = previous_evidence.manifest.file("summary.jsonl")
            current_summary = evidence.manifest.file("summary.jsonl")
            if current_summary.size < previous_summary.size:
                raise MirrorError("append-only summary.jsonl was truncated")
            if (
                current_summary.size == previous_summary.size
                and current_summary.sha256 != previous_summary.sha256
            ):
                raise MirrorError("append-only summary.jsonl was rewritten")
            if current_summary.size > previous_summary.size:
                prefix = _remote_file_prefix_sha256(
                    self.source,
                    f"{source_dir}/summary.jsonl",
                    previous_summary.size,
                )
                if prefix["prefix_sha256"] != previous_summary.sha256:
                    raise MirrorError(
                        "append-only summary.jsonl no longer preserves its authenticated prefix"
                    )

        source_steps: list[int] = []
        pending: set[int] = set()
        pending_reasons: dict[int, str] = {}
        eval_source_fingerprints: dict[int, str] = {}
        eval_validation_contexts: dict[int, str] = {}
        for name in checkpoint_inventory:
            if not isinstance(name, str):
                raise MirrorError("source checkpoint inventory contains a non-string key")
            source_steps.append(checkpoint_step_from_name(name))
        source_steps.sort()
        latest_step = source_steps[-1] if source_steps else None

        def may_be_transient_newest(step: int) -> bool:
            if step != latest_step:
                return False
            record = self.state.payload["checkpoints"].get(str(step), {})
            if not isinstance(record, dict):
                return False
            previously_finalized = bool(record.get("manifest")) and str(step) not in old_pending
            already_published = bool(
                record.get("recoverable")
                or record.get("dual_verified")
                or record.get("local", {}).get("verified")
                or record.get("columbus", {}).get("verified")
            )
            return not previously_finalized and not already_published

        for step in source_steps:
            name = f"steps_{step}"
            tree = checkpoint_inventory[name]
            if not isinstance(tree, dict):
                raise MirrorError(f"source inventory for {name} is malformed")
            fingerprint = tree.get("fingerprint")
            if not isinstance(fingerprint, str) or SHA256_RE.fullmatch(fingerprint) is None:
                raise MirrorError(f"source inventory fingerprint for {name} is invalid")
            complete, reason = _checkpoint_light_complete(tree)
            if not complete:
                if not may_be_transient_newest(step):
                    raise MirrorError(
                        f"finalized/non-latest checkpoint {name} is incomplete/corrupt: {reason}"
                    )
                pending.add(step)
                pending_reasons[step] = str(reason)
                continue
            cached_record = self.state.payload["checkpoints"].get(str(step), {})
            if (
                not full_scrub
                and isinstance(cached_record, dict)
                and cached_record.get("source_fingerprint") == fingerprint
                and step in cached_manifest_map
            ):
                manifest = cached_manifest_map[step]
            else:
                try:
                    manifest = _remote_checkpoint_manifest(
                        self.source,
                        f"{source_dir}/checkpoints/{name}",
                        metric_name=self.config.metric_name,
                        metric_mode=self.config.metric_mode,
                    )
                except MirrorError as exc:
                    if not may_be_transient_newest(step):
                        raise
                    manifests.pop(step, None)
                    evals.pop(step, None)
                    pending.add(step)
                    pending_reasons[step] = (
                        f"checkpoint finalization validation pending: {exc}"
                    )
                    changed = True
                    continue
                changed = True
            if manifest.step != step:
                raise MirrorError("source checkpoint manifest step drift")
            if manifest.checkpoint_schema != "selection_v1":
                raise MirrorError(
                    f"{name} is legacy archival-only; production mirror refuses it"
                )
            manifests[step] = manifest

            eval_name = f"step_{step:08d}.json"
            eval_stat = eval_inventory.get(eval_name)
            if eval_stat is None:
                if not may_be_transient_newest(step):
                    raise MirrorError(f"finalized/non-latest {name} lacks its required eval artifact")
                pending.add(step)
                pending_reasons[step] = "same-step eval artifact is absent"
                continue
            if not isinstance(eval_stat, dict) or eval_stat.get("kind") != "file":
                raise MirrorError(f"eval inventory for {eval_name} is malformed")
            eval_fingerprint = eval_stat.get("fingerprint")
            if not isinstance(eval_fingerprint, str) or SHA256_RE.fullmatch(
                eval_fingerprint
            ) is None:
                raise MirrorError(f"eval metadata fingerprint for {eval_name} is invalid")
            eval_source_fingerprints[step] = eval_fingerprint
            validation_context = _eval_validation_context_sha256(manifest, evidence)
            eval_validation_contexts[step] = validation_context
            cached_eval_record = self.state.payload["eval_artifacts"].get(str(step), {})
            if (
                not full_scrub
                and isinstance(cached_eval_record, dict)
                and cached_eval_record.get("source_fingerprint") == eval_fingerprint
                and cached_eval_record.get("validation_context_sha256")
                == validation_context
                and step in cached_eval_map
            ):
                artifact = cached_eval_map[step]
            else:
                try:
                    artifact = _remote_eval(
                        self.source,
                        f"{source_dir}/heldout_eval_metrics/{eval_name}",
                        manifest=manifest,
                        metric_name=self.config.metric_name,
                        metric_mode=self.config.metric_mode,
                        run_id=self.config.run_id,
                        run_evidence=evidence,
                    )
                except MirrorError as exc:
                    if not may_be_transient_newest(step):
                        raise
                    evals.pop(step, None)
                    eval_source_fingerprints.pop(step, None)
                    eval_validation_contexts.pop(step, None)
                    pending.add(step)
                    pending_reasons[step] = f"eval finalization validation pending: {exc}"
                    changed = True
                    continue
                changed = True
            if not artifact.production_eligible or not artifact.cryptographically_bound:
                raise MirrorError(f"{eval_name} is not production-selection eligible")
            evals[step] = artifact

        # Eval artifacts intentionally outlive source checkpoint retention.  A
        # complete strict-best ledger therefore authenticates every eval file,
        # using a cached checkpoint manifest when its H100 tree was pruned.
        for eval_name, eval_stat in eval_inventory.items():
            step = eval_step_from_name(eval_name)
            if step in eval_source_fingerprints:
                continue
            if step in pending and step == latest_step:
                continue
            manifest = manifests.get(step)
            if manifest is None and step != 0:
                raise MirrorError(
                    f"{eval_name} has no source or cached checkpoint manifest; "
                    "the global eval ledger cannot be authenticated"
                )
            if not isinstance(eval_stat, dict) or eval_stat.get("kind") != "file":
                raise MirrorError(f"eval inventory for {eval_name} is malformed")
            fingerprint = eval_stat.get("fingerprint")
            if not isinstance(fingerprint, str) or SHA256_RE.fullmatch(fingerprint) is None:
                raise MirrorError(f"eval metadata fingerprint for {eval_name} is invalid")
            eval_source_fingerprints[step] = fingerprint
            validation_context = _eval_validation_context_sha256(manifest, evidence)
            eval_validation_contexts[step] = validation_context
            cached_eval_record = self.state.payload["eval_artifacts"].get(str(step), {})
            if (
                not full_scrub
                and isinstance(cached_eval_record, dict)
                and cached_eval_record.get("source_fingerprint") == fingerprint
                and cached_eval_record.get("validation_context_sha256")
                == validation_context
                and step in cached_eval_map
            ):
                artifact = cached_eval_map[step]
            elif step == 0 and manifest is None:
                artifact = _remote_baseline_eval(
                    self.source,
                    f"{source_dir}/heldout_eval_metrics/{eval_name}",
                    metric_name=self.config.metric_name,
                    metric_mode=self.config.metric_mode,
                    run_id=self.config.run_id,
                    run_evidence=evidence,
                )
                changed = True
            else:
                assert manifest is not None
                artifact = _remote_eval(
                    self.source,
                    f"{source_dir}/heldout_eval_metrics/{eval_name}",
                    manifest=manifest,
                    metric_name=self.config.metric_name,
                    metric_mode=self.config.metric_mode,
                    run_id=self.config.run_id,
                    run_evidence=evidence,
                )
                changed = True
            if step == 0 and manifest is None:
                if (
                    artifact.source_kind != "live_in_memory_model"
                    or not artifact.cryptographically_bound
                    or artifact.production_eligible
                    or artifact.archival_reason
                    != "live_in_memory_baseline_not_checkpoint_recoverable"
                ):
                    raise MirrorError("step-0 live-model baseline classification is invalid")
                evals[step] = artifact
                continue
            if not artifact.production_eligible or not artifact.cryptographically_bound:
                raise MirrorError(f"{eval_name} is not production-selection eligible")
            evals[step] = artifact

        pointer_stat = inventory.get("best_checkpoint_pointer")
        if not isinstance(pointer_stat, dict):
            raise MirrorError("source pointer inventory is malformed")
        pointer_fingerprint = pointer_stat.get("fingerprint")
        if not isinstance(pointer_fingerprint, str) or SHA256_RE.fullmatch(
            pointer_fingerprint
        ) is None:
            raise MirrorError("source pointer metadata fingerprint is invalid")
        pointer: SelectionPointer | None = None
        pointer_record: FileRecord | None = None
        pointer_content: bytes | None = None
        pointer_cache = self.state.payload.get("source_pointer")

        def load_cached_pointer() -> tuple[
            SelectionPointer | None, FileRecord | None, bytes | None
        ]:
            if not (
                isinstance(pointer_cache, dict)
                and isinstance(pointer_cache.get("pointer"), dict)
                and isinstance(pointer_cache.get("record"), dict)
                and isinstance(pointer_cache.get("content_b64"), str)
            ):
                return None, None, None
            cached_pointer = validate_selection_payload(
                pointer_cache["pointer"],
                metric_name=self.config.metric_name,
                metric_mode=self.config.metric_mode,
                maximum_step=max(manifests) if manifests else None,
            )
            cached_record = FileRecord(**pointer_cache["record"])
            try:
                cached_content = base64.b64decode(
                    pointer_cache["content_b64"], validate=True
                )
            except (ValueError, TypeError) as exc:
                raise MirrorError("cached source pointer base64 is corrupt") from exc
            if (
                len(cached_content) != cached_record.size
                or hashlib.sha256(cached_content).hexdigest() != cached_record.sha256
            ):
                raise MirrorError("cached source pointer bytes are corrupt")
            return cached_pointer, cached_record, cached_content

        pointer_unreadable_pending = False
        if pointer_stat.get("exists") is True:
            if pointer_stat.get("kind") != "file" or pointer_stat.get("nlink") != 1:
                raise MirrorError("source best pointer is not a private regular file")
            if (
                not full_scrub
                and isinstance(pointer_cache, dict)
                and pointer_cache.get("fingerprint") == pointer_fingerprint
                and isinstance(pointer_cache.get("pointer"), dict)
                and isinstance(pointer_cache.get("record"), dict)
                and isinstance(pointer_cache.get("content_b64"), str)
            ):
                pointer, pointer_record, pointer_content = load_cached_pointer()
            else:
                try:
                    pointer, pointer_record, pointer_content = _remote_pointer(
                        self.source,
                        f"{source_dir}/best_checkpoint.json",
                        metric_name=self.config.metric_name,
                        metric_mode=self.config.metric_mode,
                        maximum_step=max(manifests) if manifests else None,
                    )
                except MirrorError as exc:
                    if latest_step is None or not may_be_transient_newest(latest_step):
                        raise
                    pending.add(latest_step)
                    pending_reasons[latest_step] = (
                        f"global best pointer finalization pending: {exc}"
                    )
                    pointer, pointer_record, pointer_content = load_cached_pointer()
                    pointer_unreadable_pending = True
                changed = True
        elif pointer_stat.get("exists") is not False:
            raise MirrorError("source best pointer existence flag is invalid")

        # A checkpoint is finalized only after its eval, selection dependency,
        # and global pointer all agree.  Only the maximum source step may lag.
        for step in source_steps:
            if step in pending:
                continue
            manifest = manifests[step]
            artifact = evals.get(step)
            if artifact is None:
                raise MirrorError(f"steps_{step} has no authenticated eval")
            expected = strict_global_best(
                evals, metric_mode=self.config.metric_mode, maximum_step=step
            )
            actual = (
                None
                if manifest.selection_best_step is None
                else (manifest.selection_best_step, manifest.selection_best_value)
            )
            if actual != expected:
                if not may_be_transient_newest(step):
                    raise MirrorError(
                        f"finalized/non-latest steps_{step} selection dependency/value is corrupt"
                    )
                pending.add(step)
                pending_reasons[step] = "checkpoint selection state is not finalized"

        auth_evals = {step: value for step, value in evals.items() if step not in pending}
        auth_manifests = {
            step: value for step, value in manifests.items() if step not in pending
        }
        expected_pointer = strict_global_best(
            auth_evals, metric_mode=self.config.metric_mode
        )
        full_pointer = strict_global_best(evals, metric_mode=self.config.metric_mode)
        pointer_tuple = (
            None if pointer is None else (pointer.best_step, pointer.metric_value)
        )
        if pointer_tuple != expected_pointer and not pointer_unreadable_pending:
            if (
                latest_step is not None
                and latest_step not in pending
                and may_be_transient_newest(latest_step)
            ):
                pending.add(latest_step)
                pending_reasons[latest_step] = "global best pointer is not finalized"
                auth_evals.pop(latest_step, None)
                auth_manifests.pop(latest_step, None)
                expected_pointer = strict_global_best(
                    auth_evals, metric_mode=self.config.metric_mode
                )
                full_pointer = strict_global_best(
                    evals, metric_mode=self.config.metric_mode
                )
            if pointer_tuple not in {expected_pointer, full_pointer}:
                raise MirrorError(
                    "best pointer is not the strict global argmin/argmax over all "
                    "finalized eligible checkpoint evals"
                )

        history_pointer = (
            None
            if expected_pointer is None
            else SelectionPointer(
                metric_name=self.config.metric_name,
                metric_mode=self.config.metric_mode,
                metric_value=expected_pointer[1],
                best_step=expected_pointer[0],
                checkpoint_relative_path=f"checkpoints/steps_{expected_pointer[0]}",
            )
        )
        authenticate_selection_history(
            auth_manifests,
            auth_evals,
            history_pointer,
            metric_mode=self.config.metric_mode,
        )
        for step in auth_manifests:
            validate_recovery_closure(step, auth_manifests)

        now = dt.datetime.now(dt.timezone.utc)
        pending_payload: dict[str, Any] = {}
        for step in sorted(pending):
            if step != latest_step:
                raise MirrorError(f"only the newest checkpoint may be pending: steps_{step}")
            previous = old_pending.get(str(step), {})
            first_seen = previous.get("first_seen_at", utc_now())
            if (
                isinstance(previous, dict)
                and previous
                and (now - _parse_utc(first_seen, label="pending first_seen_at")).total_seconds()
                > self.config.pending_timeout_seconds
            ):
                raise MirrorError(f"pending steps_{step} exceeded finalization timeout")
            pending_payload[str(step)] = {
                "first_seen_at": first_seen,
                "last_seen_at": utc_now(),
                "reason": pending_reasons[step],
                "source_fingerprint": checkpoint_inventory[f"steps_{step}"].get(
                    "fingerprint"
                ),
            }
        for old_step_text, previous in old_pending.items():
            old_step = _strict_int(int(old_step_text), label="prior pending step")
            if not isinstance(previous, dict):
                raise MirrorError(f"prior pending steps_{old_step} record is malformed")
            if old_step in pending:
                continue
            newer_steps = [step for step in source_steps if step > old_step]
            if newer_steps:
                raise MirrorError(
                    f"a newer checkpoint appeared before pending steps_{old_step} finalized"
                )
            if old_step in source_steps:
                # The exact pending step was observed fully authenticated on
                # this scan, so its tombstone may now be retired.
                continue
            first_seen = previous.get("first_seen_at")
            if not isinstance(first_seen, str):
                raise MirrorError(
                    f"prior pending steps_{old_step} lacks first_seen_at"
                )
            if (
                now - _parse_utc(first_seen, label="pending first_seen_at")
            ).total_seconds() > self.config.pending_timeout_seconds:
                raise MirrorError(f"pending steps_{old_step} exceeded finalization timeout")
            # A vanished in-progress directory is not evidence that it ever
            # finalized. Keep a durable tombstone so a later step cannot make
            # this gap disappear from state or heartbeat status.
            pending_payload[str(old_step)] = {
                "first_seen_at": first_seen,
                "last_seen_at": previous.get("last_seen_at", first_seen),
                "reason": "pending checkpoint disappeared before authentication",
                "source_fingerprint": previous.get("source_fingerprint"),
                "tombstoned_at": previous.get("tombstoned_at", utc_now()),
            }

        self.state.set_source_inventory_payload(inventory)
        self.state.payload["source_evidence"] = {
            "fingerprint": evidence_fingerprint,
            "identity": evidence.as_dict(),
            "immutable_files": immutable_evidence_files,
            "immutable_evidence_sha256": immutable_evidence_sha256,
        }
        if not pointer_unreadable_pending:
            self.state.payload["source_pointer"] = (
                None
                if pointer is None
                else {
                    "fingerprint": pointer_fingerprint,
                    "pointer": pointer.as_dict(),
                    "record": pointer_record.as_dict(),
                    "content_b64": base64.b64encode(pointer_content).decode("ascii"),
                }
            )
        self.state.payload["pending_candidates"] = pending_payload
        # Cache every fully validated candidate, even while its eval, selection
        # state, or pointer is still pending.  This makes unchanged minute polls
        # stat-only without granting candidate bytes authenticated/recoverable
        # status.
        for step in source_steps:
            if step not in manifests:
                continue
            fingerprint = checkpoint_inventory[f"steps_{step}"]["fingerprint"]
            self.state.checkpoint_seen(
                manifests[step], source_fingerprint=fingerprint
            )
        for step, eval_fingerprint in sorted(eval_source_fingerprints.items()):
            if step in evals:
                self.state.eval_seen(
                    evals[step],
                    source_fingerprint=eval_fingerprint,
                    validation_context_sha256=eval_validation_contexts[step],
                )

        finalized = tuple(step for step in source_steps if step not in pending)
        return SourceScan(
            manifests=auth_manifests,
            evals=auth_evals,
            pointer=(
                pointer
                if not pointer_unreadable_pending and pointer_tuple == expected_pointer
                else None
            ),
            pointer_record=(
                pointer_record
                if not pointer_unreadable_pending and pointer_tuple == expected_pointer
                else None
            ),
            pointer_content=(
                pointer_content
                if not pointer_unreadable_pending and pointer_tuple == expected_pointer
                else None
            ),
            evidence=evidence,
            finalized_steps=finalized,
            pending_steps=tuple(sorted(int(step) for step in pending_payload)),
            inventory=inventory,
            full_scrub=full_scrub,
            changed=changed,
        )

    @staticmethod
    def _nearest_existing_directory(path: Path) -> Path:
        candidate = path
        while not candidate.exists():
            if candidate.parent == candidate:
                raise MirrorError(f"no existing ancestor for path {path}")
            candidate = candidate.parent
        _validate_root_directory(candidate, label="free-space ancestor")
        return candidate

    def preflight(self) -> dict[str, Any]:
        """Read-only endpoint/tool/storage/inventory check.  It performs zero writes."""

        before_paths = {
            "state_run_dir": _internal_path_status(self.state_path.parent),
            "local_root": _internal_path_status(self.config.local_root),
        }
        h100_identity = self._resolve_h100_identity()
        self._set_source_access_from_identity(h100_identity)
        assert self.source_access_dir is not None
        columbus_identity = resolve_endpoint_identity(self.columbus)
        if (
            h100_identity["remote_fqdn"] == columbus_identity["remote_fqdn"]
            or h100_identity["remote_hostname"]
            == columbus_identity["remote_hostname"]
        ):
            raise MirrorError("H100 source and Columbus destination are the same machine")
        source_report = self.source.invoke_json(
            ("--internal-preflight-source", self.source_access_dir)
        )
        columbus_report = self.columbus.invoke_json(
            (
                "--internal-preflight-columbus",
                self.config.run_id,
                str(self.config.columbus_root),
            )
        )
        inventory = source_report.get("inventory")
        if not isinstance(inventory, dict):
            raise MirrorError("source preflight did not return an inventory")
        binding = h100_identity["source_root_binding"]
        root_identity = inventory.get("run_root")
        if (
            inventory.get("run_dir") != self.source_access_dir
            or not isinstance(root_identity, dict)
            or root_identity.get("kind") != "directory"
            or root_identity.get("device") != binding["device"]
            or root_identity.get("inode") != binding["inode"]
            or root_identity.get("uid") != binding["uid"]
        ):
            raise MirrorError("source preflight root differs from pinned realpath binding")
        for label, report in (
            ("H100", source_report),
            ("Columbus", columbus_report),
        ):
            tools = report.get("tools")
            if not isinstance(tools, dict) or not tools.get("rsync_path"):
                raise MirrorError(f"{label} preflight cannot find rsync")
            options = tools.get("rsync_required_options")
            if not isinstance(options, dict) or not all(options.values()):
                raise MirrorError(
                    f"{label} rsync lacks required safe options: {options}"
                )
        local_tools = _tool_compatibility_report()
        if not local_tools.get("rsync_path"):
            raise MirrorError("workstation preflight cannot find rsync")
        if not all(local_tools.get("rsync_required_options", {}).values()):
            raise MirrorError(
                "workstation rsync lacks required safe options: "
                f"{local_tools.get('rsync_required_options')}"
            )
        if columbus_report.get("destination_conflicts"):
            raise MirrorError(
                "Columbus destination has unexpected entries: "
                f"{columbus_report['destination_conflicts']}"
            )
        checkpoint_inventory = inventory.get("checkpoints", {})
        eval_inventory = inventory.get("eval_artifacts", {})
        if not isinstance(checkpoint_inventory, dict) or not isinstance(
            eval_inventory, dict
        ):
            raise MirrorError("source preflight inventory is malformed")
        steps = sorted(checkpoint_step_from_name(name) for name in checkpoint_inventory)
        latest = steps[-1] if steps else None
        candidates: list[int] = []
        pending: list[dict[str, Any]] = []
        incoming_bytes = 0
        for step in steps:
            tree = checkpoint_inventory[f"steps_{step}"]
            complete, reason = _checkpoint_light_complete(tree)
            has_eval = f"step_{step:08d}.json" in eval_inventory
            if complete and has_eval:
                candidates.append(step)
                incoming_bytes += sum(
                    int(item.get("size", 0))
                    for item in tree.get("entries", {}).values()
                    if isinstance(item, dict)
                )
            else:
                if step != latest:
                    raise MirrorError(
                        f"preflight found non-latest incomplete steps_{step}: "
                        f"{reason or 'eval missing'}"
                    )
                pending.append(
                    {
                        "step": step,
                        "reason": reason or "same-step eval artifact is absent",
                    }
                )
        local_space_root = self._nearest_existing_directory(self.config.local_root)
        local_stats = os.statvfs(local_space_root)
        local_available = local_stats.f_bavail * local_stats.f_frsize
        required = incoming_bytes + self.config.reserve_bytes
        columbus_available = _strict_int(
            columbus_report.get("available_bytes"),
            label="Columbus available bytes",
        )
        if local_available < required or columbus_available < required:
            raise MirrorError(
                "preflight free-space gate failed: "
                f"local={local_available}, Columbus={columbus_available}, required={required}"
            )
        if before_paths["local_root"]["exists"]:
            info = self.config.local_root.lstat()
            if (
                before_paths["local_root"]["kind"] != "directory"
                or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) & 0o077
            ):
                raise MirrorError("existing local mirror root owner/mode/type is unsafe")
        after_paths = {
            "state_run_dir": _internal_path_status(self.state_path.parent),
            "local_root": _internal_path_status(self.config.local_root),
        }
        if after_paths != before_paths:
            raise MirrorError("preflight unexpectedly changed local filesystem state")
        return {
            "schema_version": SCHEMA_VERSION,
            "mode": "preflight",
            "mutations_executed": False,
            "run_id": self.config.run_id,
            "h100_identity": h100_identity,
            "columbus_identity": columbus_identity,
            "source": source_report,
            "columbus": columbus_report,
            "local": {
                "space_checked_at": str(local_space_root),
                "available_bytes": local_available,
                "state_run_dir": before_paths["state_run_dir"],
                "mirror_root": before_paths["local_root"],
                "lock": _internal_path_status(self.lock_path),
                "tools": local_tools,
            },
            "candidate_steps": candidates,
            "pending_candidates": pending,
            "estimated_incoming_bytes": incoming_bytes,
            "reserve_bytes": self.config.reserve_bytes,
            "retention": self.config.retain,
        }

    def _remote_exists(self, endpoint: RemoteEndpoint, path: str) -> bool:
        payload = endpoint.invoke_json(("--internal-path-status", path))
        value = payload.get("exists")
        if not isinstance(value, bool):
            raise MirrorError("remote path status is malformed")
        return value

    def _persist_immutable_record(
        self, relative_path: str, payload: Mapping[str, Any]
    ) -> str:
        if not _safe_relative_path(relative_path) or not relative_path.startswith(
            "manifests/"
        ):
            raise MirrorError(f"unsafe manifest/receipt path: {relative_path!r}")
        _require_schema_version(
            payload.get("schema_version"), label="immutable record schema_version"
        )
        content = _canonical_json_bytes(payload) + b"\n"
        digest = hashlib.sha256(content).hexdigest()
        local_path = self.config.local_root / relative_path
        _reject_linked_ancestors(
            local_path.parent, label="local manifest record directory"
        )
        local_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        _ensure_secure_directory(
            local_path.parent, create=False, label="local manifest record directory"
        )
        if local_path.exists() or local_path.is_symlink():
            existing = _regular_record(
                local_path, relative_path=local_path.name, label="manifest record"
            )
            if existing.sha256 != digest:
                raise MirrorError(f"immutable manifest record conflict: {local_path}")
        else:
            _atomic_write(local_path, content)
        self.columbus.invoke_json(
            (
                "--internal-put-record",
                self.config.run_id,
                str(self.config.columbus_root),
                relative_path,
                base64.urlsafe_b64encode(content).decode("ascii"),
                digest,
                "immutable",
            ),
            mutation=True,
        )
        return digest

    def _prepare_recovery_intent(
        self,
        *,
        step: int,
        manifest: TreeManifest,
        artifact: EvalArtifact,
        evidence: RunEvidenceIdentity,
        closure: Sequence[int],
    ) -> tuple[TreeManifest, EvalArtifact, RunEvidenceIdentity, tuple[int, ...]]:
        """Durably pin receipt inputs before writing any immutable receipt record."""

        assert self.state is not None
        record = self.state.payload["checkpoints"].get(str(step))
        if not isinstance(record, dict):
            raise MirrorError(f"cannot plan recovery receipt for unknown steps_{step}")
        intent = record.get("recovery_intent")
        if intent is None:
            intent = {
                "schema_version": SCHEMA_VERSION,
                "record_kind": "recovery_publication_intent",
                "run_id": self.config.run_id,
                "step": step,
                "checkpoint_manifest": manifest.as_dict(),
                "eval_artifact": artifact.as_dict(),
                "evidence_identity": evidence.as_dict(),
                "recovery_closure": list(closure),
            }
            record["recovery_intent"] = intent
            # This save is the write-ahead boundary.  A restart must reuse this
            # exact evidence snapshot even when mutable summary/pointer evidence
            # has legitimately advanced in the meantime.
            self.state.save()
        if not isinstance(intent, dict) or set(intent) != {
            "schema_version",
            "record_kind",
            "run_id",
            "step",
            "checkpoint_manifest",
            "eval_artifact",
            "evidence_identity",
            "recovery_closure",
        }:
            raise MirrorError(f"steps_{step} recovery publication intent is malformed")
        _require_schema_version(
            intent.get("schema_version"),
            label="recovery publication intent schema_version",
        )
        if (
            intent.get("record_kind") != "recovery_publication_intent"
            or intent.get("run_id") != self.config.run_id
            or intent.get("step") != step
        ):
            raise MirrorError(f"steps_{step} recovery publication intent identity mismatch")
        pinned_manifest = TreeManifest.from_dict(intent["checkpoint_manifest"])
        pinned_artifact = EvalArtifact.from_dict(intent["eval_artifact"])
        pinned_evidence = RunEvidenceIdentity.from_dict(intent["evidence_identity"])
        raw_closure = intent.get("recovery_closure")
        if not isinstance(raw_closure, list):
            raise MirrorError(f"steps_{step} recovery publication closure is malformed")
        pinned_closure = tuple(
            _strict_int(value, label="recovery publication dependency")
            for value in raw_closure
        )
        expected_closure = validate_recovery_closure(step, _cached_manifests(self.state))
        if (
            pinned_manifest != manifest
            or pinned_manifest.step != step
            or pinned_artifact != artifact
            or pinned_artifact.step != step
            or pinned_closure != tuple(closure)
            or pinned_closure != expected_closure
        ):
            raise MirrorError(f"steps_{step} recovery publication intent drift")
        if (
            pinned_evidence.run_id != self.config.run_id
            or pinned_evidence.source_output_dir != self.config.source_run_dir
        ):
            raise MirrorError(f"steps_{step} recovery evidence identity mismatch")
        evidence_record = self.state.payload["evidence_snapshots"].get(
            pinned_evidence.manifest.manifest_sha256, {}
        )
        if not isinstance(evidence_record, dict) or not evidence_record.get(
            "dual_verified"
        ):
            raise MirrorError(
                f"steps_{step} recovery intent lacks its dual-verified evidence snapshot"
            )
        return pinned_manifest, pinned_artifact, pinned_evidence, pinned_closure

    def _persist_recovery_receipt(
        self,
        *,
        step: int,
        manifest: TreeManifest,
        artifact: EvalArtifact,
        evidence: RunEvidenceIdentity,
        closure: Sequence[int],
    ) -> str:
        checkpoint_manifest_path = f"manifests/checkpoints/steps_{step}.json"
        eval_manifest_path = f"manifests/evals/step_{step:08d}.json"
        evidence_manifest_path = (
            "manifests/evidence/"
            f"sha256-{evidence.manifest.manifest_sha256}.json"
        )
        checkpoint_record_sha = self._persist_immutable_record(
            checkpoint_manifest_path,
            {
                "schema_version": SCHEMA_VERSION,
                "record_kind": "checkpoint_manifest",
                "run_id": self.config.run_id,
                "manifest": manifest.as_dict(),
            },
        )
        eval_record_sha = self._persist_immutable_record(
            eval_manifest_path,
            {
                "schema_version": SCHEMA_VERSION,
                "record_kind": "eval_manifest",
                "run_id": self.config.run_id,
                "artifact": artifact.as_dict(),
            },
        )
        evidence_record_sha = self._persist_immutable_record(
            evidence_manifest_path,
            {
                "schema_version": SCHEMA_VERSION,
                "record_kind": "evidence_manifest",
                "run_id": self.config.run_id,
                "identity": evidence.as_dict(),
            },
        )
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "record_kind": "recovery_receipt",
            "run_id": self.config.run_id,
            "step": step,
            "production_eligible": True,
            "recovery_closure": list(closure),
            "checkpoint": {
                "relative_path": f"checkpoints/steps_{step}",
                "manifest_sha256": manifest.manifest_sha256,
                "manifest_record_path": checkpoint_manifest_path,
                "manifest_record_sha256": checkpoint_record_sha,
            },
            "eval": {
                "relative_path": f"heldout_eval_metrics/step_{step:08d}.json",
                "sha256": artifact.sha256,
                "manifest_record_path": eval_manifest_path,
                "manifest_record_sha256": eval_record_sha,
            },
            "evidence": {
                "relative_path": (
                    "evidence/snapshots/"
                    f"sha256-{evidence.manifest.manifest_sha256}"
                ),
                "manifest_sha256": evidence.manifest.manifest_sha256,
                "manifest_record_path": evidence_manifest_path,
                "manifest_record_sha256": evidence_record_sha,
            },
        }
        return self._persist_immutable_record(
            f"manifests/receipts/checkpoint-steps_{step}.json", receipt
        )

    def _reconcile_recovery_intents(self) -> tuple[int, ...]:
        """Complete receipt transactions from pinned state without source access."""

        assert self.state is not None
        reconciled: list[int] = []
        for step_text, record in sorted(
            self.state.payload["checkpoints"].items(), key=lambda item: int(item[0])
        ):
            if not isinstance(record, dict):
                raise MirrorError("checkpoint state record is malformed")
            intent = record.get("recovery_intent")
            if (
                intent is None
                or record.get("recoverable")
                or (
                    record.get("local", {}).get("pruned")
                    and record.get("columbus", {}).get("pruned")
                )
            ):
                continue
            step = _strict_int(int(step_text), label="recovery intent step")
            if not isinstance(intent, dict):
                raise MirrorError(f"steps_{step} recovery publication intent is malformed")
            try:
                manifest = TreeManifest.from_dict(intent["checkpoint_manifest"])
                artifact = EvalArtifact.from_dict(intent["eval_artifact"])
                evidence = RunEvidenceIdentity.from_dict(intent["evidence_identity"])
                raw_closure = intent["recovery_closure"]
            except KeyError as exc:
                raise MirrorError(
                    f"steps_{step} recovery publication intent is incomplete"
                ) from exc
            if not isinstance(raw_closure, list):
                raise MirrorError(f"steps_{step} recovery publication closure is malformed")
            closure = tuple(
                _strict_int(value, label="recovery publication dependency")
                for value in raw_closure
            )
            manifest, artifact, evidence, closure = self._prepare_recovery_intent(
                step=step,
                manifest=manifest,
                artifact=artifact,
                evidence=evidence,
                closure=closure,
            )
            local_checkpoint = (
                self.config.local_root / "checkpoints" / f"steps_{step}"
            )
            local_eval = (
                self.config.local_root
                / "heldout_eval_metrics"
                / f"step_{step:08d}.json"
            )
            # Both copy routines can finish Columbus entirely from authenticated
            # local bytes.  Do not call either one before its local source exists,
            # because this startup phase must not fall back to the H100.
            if not (local_checkpoint.exists() or local_checkpoint.is_symlink()) or not (
                local_eval.exists() or local_eval.is_symlink()
            ):
                continue
            self._mirror_checkpoint(step, manifest)
            self._mirror_eval(step, artifact, manifest, evidence)
            checkpoint_record = self.state.payload["checkpoints"].get(str(step), {})
            eval_record = self.state.payload["eval_artifacts"].get(str(step), {})
            if not checkpoint_record.get("dual_verified") or not eval_record.get(
                "dual_verified"
            ):
                raise MirrorError(
                    f"steps_{step} recovery copy reconciliation did not dual-verify"
                )
            receipt_sha = self._persist_recovery_receipt(
                step=step,
                manifest=manifest,
                artifact=artifact,
                evidence=evidence,
                closure=closure,
            )
            self.state.mark_recoverable(
                step,
                closure,
                evidence_manifest_sha256=evidence.manifest.manifest_sha256,
                receipt_sha256=receipt_sha,
            )
            self.state.save()
            reconciled.append(step)
        return tuple(reconciled)

    def _build_restore_index(self, *, excluded_steps: Iterable[int] = ()) -> dict[str, Any]:
        assert self.state is not None
        excluded = set(excluded_steps)
        entries: list[dict[str, Any]] = []
        for step_text, record in sorted(
            self.state.payload["checkpoints"].items(), key=lambda item: int(item[0])
        ):
            step = int(step_text)
            if step in excluded or not record.get("recoverable"):
                continue
            eval_record = self.state.payload["eval_artifacts"].get(step_text, {})
            if (
                record.get("checkpoint_schema") != "selection_v1"
                or not record.get("dual_verified")
                or not eval_record.get("dual_verified")
                or not eval_record.get("production_eligible")
                or SHA256_RE.fullmatch(str(record.get("receipt_sha256", ""))) is None
            ):
                raise MirrorError(f"state falsely marks steps_{step} recoverable")
            closure = record.get("recovery_closure")
            if not isinstance(closure, list) or any(
                not self.state.payload["checkpoints"].get(str(dependency), {}).get(
                    "dual_verified"
                )
                for dependency in closure
            ):
                raise MirrorError(f"steps_{step} recovery closure is unavailable")
            evidence_digest = record.get("evidence_manifest_sha256")
            entries.append(
                {
                    "step": step,
                    "receipt_relative_path": (
                        f"manifests/receipts/checkpoint-steps_{step}.json"
                    ),
                    "receipt_sha256": record["receipt_sha256"],
                    "checkpoint_relative_path": f"checkpoints/steps_{step}",
                    "checkpoint_manifest_sha256": record["manifest_sha256"],
                    "eval_relative_path": (
                        f"heldout_eval_metrics/step_{step:08d}.json"
                    ),
                    "eval_sha256": eval_record["sha256"],
                    "evidence_relative_path": (
                        f"evidence/snapshots/sha256-{evidence_digest}"
                    ),
                    "evidence_manifest_sha256": evidence_digest,
                    "recovery_closure": list(closure),
                }
            )
        current_pointer = self.state.payload.get("best_pointer")
        pointer = None
        if isinstance(current_pointer, dict) and current_pointer.get("dual_verified"):
            pointer = {
                "relative_path": "best_checkpoint.json",
                "sha256": current_pointer["sha256"],
                "best_step": current_pointer["best_metric_step"],
                "metric_name": current_pointer["best_metric_name"],
                "metric_mode": current_pointer["best_metric_mode"],
                "metric_value": current_pointer["best_metric_value"],
            }
        return {
            "schema_version": SCHEMA_VERSION,
            "record_kind": "restore_index",
            "run_id": self.config.run_id,
            "metric_name": self.config.metric_name,
            "metric_mode": self.config.metric_mode,
            "generation": int(self.state.payload.get("restore_index_generation", 0)) + 1,
            "checkpoints": entries,
            "best_pointer": pointer,
        }

    def _restore_index_needs_publication(self) -> bool:
        """Return whether durable recovery state is newer than the public index."""

        assert self.state is not None
        generation = _strict_int(
            self.state.payload.get("restore_index_generation", 0),
            label="restore-index state generation",
        )
        recorded_sha = self.state.payload.get("restore_index_sha256")
        current_path = self.config.local_root / "restore-index.json"
        desired = self._build_restore_index()
        if generation == 0:
            if recorded_sha is not None:
                raise MirrorError("restore-index generation/SHA state is inconsistent")
            if current_path.exists() or current_path.is_symlink():
                raise MirrorError("untracked local restore index exists before generation 1")
            return bool(desired["checkpoints"] or desired["best_pointer"] is not None)
        if not isinstance(recorded_sha, str) or SHA256_RE.fullmatch(recorded_sha) is None:
            raise MirrorError("restore-index state SHA-256 is invalid")
        _secure_regular_file(current_path, label="current local restore index")
        content = current_path.read_bytes()
        if hashlib.sha256(content).hexdigest() != recorded_sha:
            raise MirrorError("current local restore index disagrees with durable state")
        current = _load_json_object(current_path, label="current local restore index")
        if set(current) != {
            "schema_version",
            "record_kind",
            "run_id",
            "metric_name",
            "metric_mode",
            "generation",
            "checkpoints",
            "best_pointer",
        }:
            raise MirrorError("current local restore index shape is invalid")
        _require_schema_version(
            current.get("schema_version"), label="current local restore index schema"
        )
        if (
            current.get("record_kind") != "restore_index"
            or current.get("run_id") != self.config.run_id
            or current.get("metric_name") != self.config.metric_name
            or current.get("metric_mode") != self.config.metric_mode
            or current.get("generation") != generation
        ):
            raise MirrorError("current local restore index identity/state mismatch")
        desired["generation"] = generation
        return current != desired

    def _durable_selection_state(
        self,
    ) -> tuple[bool, int | None]:
        """Return whether cached recoverable state has a complete best pointer."""

        assert self.state is not None
        manifests: dict[int, TreeManifest] = {}
        evals: dict[int, EvalArtifact] = {}
        for step_text, record in self.state.payload["checkpoints"].items():
            if not isinstance(record, dict) or not record.get("recoverable"):
                continue
            step = _strict_int(int(step_text), label="durable checkpoint step")
            raw_manifest = record.get("manifest")
            eval_record = self.state.payload["eval_artifacts"].get(step_text, {})
            if not isinstance(raw_manifest, dict) or not isinstance(eval_record, dict):
                raise MirrorError(f"recoverable steps_{step} lacks cached identity")
            raw_artifact = eval_record.get("artifact")
            if not isinstance(raw_artifact, dict):
                raise MirrorError(f"recoverable steps_{step} lacks cached eval identity")
            manifests[step] = TreeManifest.from_dict(raw_manifest)
            evals[step] = EvalArtifact.from_dict(raw_artifact)
        pointer_record = self.state.payload.get("best_pointer")
        pointer: SelectionPointer | None = None
        if isinstance(pointer_record, dict) and pointer_record.get("dual_verified") is True:
            pointer = SelectionPointer(
                metric_name=str(pointer_record.get("best_metric_name")),
                metric_mode=str(pointer_record.get("best_metric_mode")),
                metric_value=_finite_number(
                    pointer_record.get("best_metric_value"),
                    label="durable best-pointer value",
                ),
                best_step=_strict_int(
                    pointer_record.get("best_metric_step"),
                    label="durable best-pointer step",
                ),
                checkpoint_relative_path=str(
                    pointer_record.get("checkpoint_relative_path")
                ),
            )
        expected = strict_global_best(evals, metric_mode=self.config.metric_mode)
        if expected is None:
            return pointer is None, None
        if (
            pointer is None
            or pointer.metric_name != self.config.metric_name
            or pointer.metric_mode != self.config.metric_mode
            or pointer.checkpoint_relative_path
            != f"checkpoints/steps_{pointer.best_step}"
            or (pointer.best_step, pointer.metric_value) != expected
        ):
            return False, None
        try:
            authenticate_selection_history(
                manifests, evals, pointer, metric_mode=self.config.metric_mode
            )
        except MirrorError:
            return False, None
        return True, pointer.best_step

    def _validated_restore_index_intent(
        self,
    ) -> tuple[dict[str, Any], tuple[int, ...], str] | None:
        assert self.state is not None
        intent = self.state.payload.get("restore_index_intent")
        if intent is None:
            return None
        if not isinstance(intent, dict) or set(intent) != {
            "schema_version",
            "record_kind",
            "run_id",
            "generation",
            "sha256",
            "excluded_steps",
            "payload",
        }:
            raise MirrorError("restore-index publication intent is malformed")
        _require_schema_version(
            intent.get("schema_version"),
            label="restore-index publication intent schema_version",
        )
        generation = _strict_int(
            intent.get("generation"),
            label="restore-index publication generation",
            minimum=1,
        )
        expected_generation = int(
            self.state.payload.get("restore_index_generation", 0)
        ) + 1
        if (
            intent.get("record_kind") != "restore_index_publication_intent"
            or intent.get("run_id") != self.config.run_id
            or generation != expected_generation
        ):
            raise MirrorError("restore-index publication intent identity mismatch")
        raw_excluded = intent.get("excluded_steps")
        if not isinstance(raw_excluded, list):
            raise MirrorError("restore-index publication exclusions are malformed")
        excluded = tuple(
            _strict_int(value, label="restore-index excluded step")
            for value in raw_excluded
        )
        if excluded != tuple(sorted(set(excluded))):
            raise MirrorError("restore-index publication exclusions are noncanonical")
        payload = intent.get("payload")
        if not isinstance(payload, dict):
            raise MirrorError("restore-index publication payload is malformed")
        expected_payload = self._build_restore_index(excluded_steps=excluded)
        if payload != expected_payload or payload.get("generation") != generation:
            raise MirrorError("restore-index publication intent drift")
        content = _canonical_json_bytes(payload) + b"\n"
        digest = hashlib.sha256(content).hexdigest()
        if intent.get("sha256") != digest:
            raise MirrorError("restore-index publication intent SHA-256 mismatch")
        return dict(payload), excluded, digest

    def _commit_restore_index_intent(self) -> str:
        assert self.state is not None
        validated = self._validated_restore_index_intent()
        if validated is None:
            raise MirrorError("restore-index publication has no durable intent")
        payload, _excluded, digest = validated
        content = _canonical_json_bytes(payload) + b"\n"
        generation = payload["generation"]
        immutable_path = f"manifests/restore_indexes/generation_{generation:08d}.json"
        immutable_sha = self._persist_immutable_record(immutable_path, payload)
        if immutable_sha != digest:
            raise MirrorError("restore-index canonical digest disagreement")
        local_current = self.config.local_root / "restore-index.json"
        _atomic_write(local_current, content)
        self.columbus.invoke_json(
            (
                "--internal-put-record",
                self.config.run_id,
                str(self.config.columbus_root),
                "restore-index.json",
                base64.urlsafe_b64encode(content).decode("ascii"),
                digest,
                "mutable",
            ),
            mutation=True,
        )
        self.state.payload["restore_index_generation"] = generation
        self.state.payload["restore_index_sha256"] = digest
        self.state.payload["restore_index_intent"] = None
        self.state.save()
        return digest

    def _reconcile_restore_index_intent(self) -> str | None:
        assert self.state is not None
        if self.state.payload.get("restore_index_intent") is None:
            return None
        return self._commit_restore_index_intent()

    def _publish_restore_index(self, *, excluded_steps: Iterable[int] = ()) -> str:
        assert self.state is not None
        excluded = tuple(
            sorted(
                {
                    _strict_int(step, label="restore-index excluded step")
                    for step in excluded_steps
                }
            )
        )
        existing = self._validated_restore_index_intent()
        if existing is not None:
            _payload, pending_excluded, _digest = existing
            committed = self._commit_restore_index_intent()
            if pending_excluded == excluded:
                return committed
        payload = self._build_restore_index(excluded_steps=excluded)
        content = _canonical_json_bytes(payload) + b"\n"
        digest = hashlib.sha256(content).hexdigest()
        self.state.payload["restore_index_intent"] = {
            "schema_version": SCHEMA_VERSION,
            "record_kind": "restore_index_publication_intent",
            "run_id": self.config.run_id,
            "generation": payload["generation"],
            "sha256": digest,
            "excluded_steps": list(excluded),
            "payload": payload,
        }
        # Publish intent before the immutable generation.  A restart reconciles
        # this transaction before reading a source that may have advanced.
        self.state.save()
        return self._commit_restore_index_intent()

    def _validated_pointer_publication_intent(
        self,
    ) -> tuple[SelectionPointer, FileRecord, bytes, int] | None:
        assert self.state is not None
        intent = self.state.payload.get("pointer_publication_intent")
        if intent is None:
            return None
        if not isinstance(intent, dict) or set(intent) != {
            "schema_version",
            "record_kind",
            "run_id",
            "pointer",
            "record",
            "content_b64",
            "maximum_step",
        }:
            raise MirrorError("pointer publication intent is malformed")
        _require_schema_version(
            intent.get("schema_version"),
            label="pointer publication intent schema_version",
        )
        if (
            intent.get("record_kind") != "pointer_publication_intent"
            or intent.get("run_id") != self.config.run_id
        ):
            raise MirrorError("pointer publication intent identity mismatch")
        maximum_step = _strict_int(
            intent.get("maximum_step"), label="pointer publication maximum_step"
        )
        pointer_payload = intent.get("pointer")
        record_payload = intent.get("record")
        encoded_content = intent.get("content_b64")
        if not isinstance(pointer_payload, dict):
            raise MirrorError("pointer publication intent lacks a pointer object")
        pointer = validate_selection_payload(
            pointer_payload,
            metric_name=self.config.metric_name,
            metric_mode=self.config.metric_mode,
            maximum_step=maximum_step,
        )
        if not isinstance(record_payload, dict) or set(record_payload) != {
            "path",
            "size",
            "sha256",
        }:
            raise MirrorError("pointer publication file record is malformed")
        record = FileRecord(
            path=str(record_payload.get("path")),
            size=_strict_int(
                record_payload.get("size"),
                label="pointer publication content size",
                minimum=1,
            ),
            sha256=str(record_payload.get("sha256")),
        )
        if (
            record.path != "best_checkpoint.json"
            or SHA256_RE.fullmatch(record.sha256) is None
            or not isinstance(encoded_content, str)
        ):
            raise MirrorError("pointer publication file identity is invalid")
        try:
            content = base64.b64decode(encoded_content, validate=True)
        except (ValueError, TypeError) as exc:
            raise MirrorError("pointer publication content base64 is corrupt") from exc
        if (
            len(content) != record.size
            or hashlib.sha256(content).hexdigest() != record.sha256
        ):
            raise MirrorError("pointer publication content digest mismatch")
        content_payload = _loads_json_object(
            content, label="pointer publication content"
        )
        if content_payload != pointer.as_dict():
            raise MirrorError("pointer publication content/object mismatch")
        cached_evals = {
            step: artifact
            for step, artifact in _cached_evals(self.state).items()
            if step <= maximum_step
        }
        cached_manifests = {
            step: manifest
            for step, manifest in _cached_manifests(self.state).items()
            if step <= maximum_step
        }
        authenticate_selection_history(
            cached_manifests,
            cached_evals,
            pointer,
            metric_mode=self.config.metric_mode,
        )
        return pointer, record, content, maximum_step

    def _prepare_pointer_publication_intent(
        self,
        *,
        pointer: SelectionPointer,
        record: FileRecord,
        content: bytes,
        maximum_step: int,
    ) -> None:
        assert self.state is not None
        payload = {
            "schema_version": SCHEMA_VERSION,
            "record_kind": "pointer_publication_intent",
            "run_id": self.config.run_id,
            "pointer": pointer.as_dict(),
            "record": record.as_dict(),
            "content_b64": base64.b64encode(content).decode("ascii"),
            "maximum_step": maximum_step,
        }
        existing = self.state.payload.get("pointer_publication_intent")
        if existing is None:
            self.state.payload["pointer_publication_intent"] = payload
            # This write-ahead boundary precedes any new recoverable marker.
            # A restart can publish the exact authenticated pointer and then
            # rebuild the index without contacting the H100.
            self.state.save()
        elif existing != payload:
            raise MirrorError("pointer publication intent drift")
        self._validated_pointer_publication_intent()

    def _reconcile_pointer_publication_intent(
        self, *, verify_source: bool
    ) -> bool:
        assert self.state is not None
        validated = self._validated_pointer_publication_intent()
        if validated is None:
            return True
        pointer, record, content, maximum_step = validated
        target = self.state.payload["checkpoints"].get(str(pointer.best_step), {})
        if not isinstance(target, dict) or not (
            target.get("dual_verified") and target.get("recoverable")
        ):
            # The checkpoint/eval transfer may have crashed before enough
            # local bytes existed. Keep the intent and continue to the source;
            # recovery reconciliation will retry it after those bytes arrive.
            return False
        recoverable_evals: dict[int, EvalArtifact] = {}
        for step_text, checkpoint_record in self.state.payload["checkpoints"].items():
            if not isinstance(checkpoint_record, dict) or not checkpoint_record.get(
                "recoverable"
            ):
                continue
            eval_record = self.state.payload["eval_artifacts"].get(step_text, {})
            if not isinstance(eval_record, dict) or not isinstance(
                eval_record.get("artifact"), dict
            ):
                raise MirrorError("recoverable checkpoint lacks cached eval identity")
            step = _strict_int(int(step_text), label="recoverable pointer step")
            recoverable_evals[step] = EvalArtifact.from_dict(eval_record["artifact"])
        if strict_global_best(
            recoverable_evals, metric_mode=self.config.metric_mode
        ) != (pointer.best_step, pointer.metric_value):
            raise MirrorError("pointer publication intent is stale for recoverable state")
        self._mirror_pointer(
            pointer,
            record,
            content,
            maximum_step,
            force_verify=True,
            verify_source=verify_source,
        )
        durable = self.state.payload.get("best_pointer")
        if not isinstance(durable, dict) or not (
            durable.get("dual_verified") is True
            and durable.get("sha256") == record.sha256
        ):
            raise MirrorError("pointer publication did not become dual verified")
        self.state.payload["pointer_publication_intent"] = None
        self.state.save()
        return True

    def _reconcile_recoverable_pointer_from_checkpoint(self) -> bool:
        """Repair an incomplete durable pointer using receipt-complete local bytes."""

        assert self.state is not None
        complete, _best_step = self._durable_selection_state()
        if complete:
            return False
        manifests: dict[int, TreeManifest] = {}
        evals: dict[int, EvalArtifact] = {}
        for step_text, record in self.state.payload["checkpoints"].items():
            if not isinstance(record, dict) or not record.get("recoverable"):
                continue
            step = _strict_int(int(step_text), label="offline pointer checkpoint step")
            raw_manifest = record.get("manifest")
            eval_record = self.state.payload["eval_artifacts"].get(step_text, {})
            if not isinstance(raw_manifest, dict) or not isinstance(eval_record, dict):
                raise MirrorError("recoverable state lacks offline pointer identity")
            raw_artifact = eval_record.get("artifact")
            if not isinstance(raw_artifact, dict):
                raise MirrorError("recoverable state lacks offline pointer eval")
            manifests[step] = TreeManifest.from_dict(raw_manifest)
            evals[step] = EvalArtifact.from_dict(raw_artifact)
        if not manifests:
            return False
        latest_step = max(manifests)
        latest_manifest = manifests[latest_step]
        selection_identity = latest_manifest.file("selection_state.json")
        selection_path = (
            self.config.local_root
            / "checkpoints"
            / f"steps_{latest_step}"
            / "selection_state.json"
        )
        actual = _regular_record(
            selection_path,
            relative_path="best_checkpoint.json",
            label="offline checkpoint-bound best pointer",
        )
        if (
            actual.size != selection_identity.size
            or actual.sha256 != selection_identity.sha256
        ):
            raise MirrorError("offline checkpoint-bound best pointer bytes changed")
        content = selection_path.read_bytes()
        pointer = validate_selection_payload(
            _loads_json_object(content, label="offline checkpoint-bound best pointer"),
            metric_name=self.config.metric_name,
            metric_mode=self.config.metric_mode,
            maximum_step=latest_step,
        )
        authenticate_selection_history(
            manifests, evals, pointer, metric_mode=self.config.metric_mode
        )
        self._mirror_pointer(
            pointer,
            actual,
            content,
            latest_step,
            force_verify=True,
            verify_source=False,
        )
        complete, _best_step = self._durable_selection_state()
        if not complete:
            raise MirrorError("offline checkpoint-bound pointer repair was incomplete")
        return True

    def _acquire_remote_lock(self) -> None:
        if self._remote_lock_token is not None:
            return
        token = f"{os.getpid()}-{uuid.uuid4().hex}"
        self.columbus.invoke_json(
            (
                "--internal-acquire-lock",
                self.config.run_id,
                str(self.config.columbus_root),
                token,
            ),
            mutation=True,
        )
        self._remote_lock_token = token

    def _release_remote_lock(self) -> None:
        if self._remote_lock_token is None:
            return
        token = self._remote_lock_token
        self._remote_lock_token = None
        self.columbus.invoke_json(
            (
                "--internal-release-lock",
                self.config.run_id,
                str(self.config.columbus_root),
                token,
            ),
            mutation=True,
        )

    def _mirror_checkpoint(
        self, step: int, expected: TreeManifest, *, force_verify: bool = False
    ) -> None:
        assert self.state is not None
        if self.source_access_dir is None:
            raise MirrorError("H100 source-root binding is not initialized")
        state_record = self.state.payload["checkpoints"].get(str(step), {})
        source_path = f"{self.source_access_dir}/checkpoints/steps_{step}"
        local_final = self.config.local_root / "checkpoints" / f"steps_{step}"
        local_verified = bool(
            isinstance(state_record, dict)
            and state_record.get("local", {}).get("verified")
            and state_record.get("manifest_sha256") == expected.manifest_sha256
        )
        if local_verified and not force_verify:
            pass
        elif local_final.exists() or local_final.is_symlink():
            local_manifest = validate_checkpoint_tree(
                local_final,
                metric_name=self.config.metric_name,
                metric_mode=self.config.metric_mode,
            )
            if local_manifest.manifest_sha256 != expected.manifest_sha256:
                raise MirrorError(f"local immutable checkpoint conflict: {local_final}")
        else:
            require_free_space(
                self.config.local_root,
                incoming_bytes=expected.total_size,
                reserve_bytes=self.config.reserve_bytes,
            )
            stage = local_final.parent / f".incoming-steps_{step}"
            _prepare_local_stage(stage, directory=True)
            try:
                self._pulse(f"h100_to_local_steps_{step}")
                self.source.rsync_to_local(
                    f"{source_path}/",
                    Path(f"{stage}/"),
                    delete=True,
                    heartbeat=lambda: self._pulse(f"h100_to_local_steps_{step}"),
                )
                local_manifest = validate_checkpoint_tree(
                    stage,
                    metric_name=self.config.metric_name,
                    metric_mode=self.config.metric_mode,
                    expected_step=step,
                )
                post_source = _remote_checkpoint_manifest(
                    self.source,
                    source_path,
                    metric_name=self.config.metric_name,
                    metric_mode=self.config.metric_mode,
                )
                if (
                    local_manifest.manifest_sha256 != expected.manifest_sha256
                    or post_source.manifest_sha256 != expected.manifest_sha256
                ):
                    raise MirrorError(f"steps_{step} changed during source transfer")
                _promote_local_tree(
                    stage,
                    local_final,
                    expected=expected,
                    metric_name=self.config.metric_name,
                    metric_mode=self.config.metric_mode,
                )
            except Exception:
                # Preserve the hidden, protected stage so checksum-mode rsync
                # can resume or repair it on the next locked cycle.
                raise
        if not local_verified or force_verify:
            self.state.verify_tier(step, "local", expected.manifest_sha256)
            self.state.save()

        remote_final = f"{self.config.columbus_root}/checkpoints/steps_{step}"
        state_record = self.state.payload["checkpoints"].get(str(step), {})
        remote_verified = bool(
            isinstance(state_record, dict)
            and state_record.get("columbus", {}).get("verified")
            and state_record.get("manifest_sha256") == expected.manifest_sha256
        )
        if remote_verified and not force_verify:
            pass
        elif self._remote_exists(self.columbus, remote_final):
            remote_manifest = _remote_checkpoint_manifest(
                self.columbus,
                remote_final,
                metric_name=self.config.metric_name,
                metric_mode=self.config.metric_mode,
            )
            if remote_manifest.manifest_sha256 != expected.manifest_sha256:
                raise MirrorError(f"Columbus immutable checkpoint conflict: {remote_final}")
        else:
            _require_remote_space(
                self.columbus,
                str(self.config.columbus_root),
                incoming_bytes=expected.total_size,
                reserve_bytes=self.config.reserve_bytes,
            )
            remote_stage = (
                f"{self.config.columbus_root}/checkpoints/"
                f".incoming-steps_{step}"
            )
            self.columbus.invoke_json(
                (
                    "--internal-create-stage",
                    self.config.run_id,
                    str(self.config.columbus_root),
                    remote_stage,
                    "directory",
                ),
                mutation=True,
            )
            self._pulse(f"local_to_columbus_steps_{step}")
            self.columbus.rsync_from_local(
                f"{local_final}/",
                f"{remote_stage}/",
                delete=True,
                heartbeat=lambda: self._pulse(f"local_to_columbus_steps_{step}"),
            )
            self.columbus.invoke_json(
                (
                    "--internal-promote-tree",
                    self.config.run_id,
                    str(self.config.columbus_root),
                    remote_stage,
                    remote_final,
                    _manifest_b64(expected),
                    self.config.metric_name,
                    self.config.metric_mode,
                ),
                mutation=True,
            )
            remote_manifest = _remote_checkpoint_manifest(
                self.columbus,
                remote_final,
                metric_name=self.config.metric_name,
                metric_mode=self.config.metric_mode,
            )
            if remote_manifest.manifest_sha256 != expected.manifest_sha256:
                raise MirrorError(f"Columbus post-promotion verification failed: steps_{step}")
        if not remote_verified or force_verify:
            self.state.verify_tier(step, "columbus", expected.manifest_sha256)
            self.state.save()

    def _mirror_eval(
        self,
        step: int,
        artifact: EvalArtifact,
        manifest: TreeManifest,
        run_evidence: RunEvidenceIdentity,
        *,
        force_verify: bool = False,
        local_only: bool = False,
    ) -> None:
        assert self.state is not None
        if self.source_access_dir is None:
            raise MirrorError("H100 source-root binding is not initialized")
        if not artifact.production_eligible or not artifact.cryptographically_bound:
            raise MirrorError("archival-only eval cannot enter the production mirror")
        eval_state = self.state.payload["eval_artifacts"].get(str(step), {})
        name = f"step_{step:08d}.json"
        source_path = f"{self.source_access_dir}/heldout_eval_metrics/{name}"
        local_final = self.config.local_root / "heldout_eval_metrics" / name
        local_verified = bool(
            isinstance(eval_state, dict)
            and eval_state.get("local")
            and eval_state.get("sha256") == artifact.sha256
        )
        if local_verified and not force_verify:
            pass
        elif local_final.exists() or local_final.is_symlink():
            local_artifact = validate_eval_artifact(
                local_final,
                checkpoint_manifest=manifest,
                metric_name=self.config.metric_name,
                metric_mode=self.config.metric_mode,
                run_id=self.config.run_id,
                run_evidence=run_evidence,
            )
            if local_artifact.sha256 != artifact.sha256:
                raise MirrorError(f"local immutable eval conflict: {local_final}")
        else:
            require_free_space(
                self.config.local_root,
                incoming_bytes=artifact.size,
                reserve_bytes=self.config.reserve_bytes,
            )
            stage = local_final.parent / f".incoming-{name}"
            _prepare_local_stage(stage, directory=False)
            self.source.rsync_to_local(
                source_path,
                stage,
                heartbeat=lambda: self._pulse(f"h100_to_local_eval_{step}"),
            )
            local_artifact = validate_eval_artifact(
                stage,
                checkpoint_manifest=manifest,
                metric_name=self.config.metric_name,
                metric_mode=self.config.metric_mode,
                run_id=self.config.run_id,
                run_evidence=run_evidence,
                expected_step=step,
            )
            post = _remote_eval(
                self.source,
                source_path,
                manifest=manifest,
                metric_name=self.config.metric_name,
                metric_mode=self.config.metric_mode,
                run_id=self.config.run_id,
                run_evidence=run_evidence,
            )
            if local_artifact.sha256 != artifact.sha256 or post.sha256 != artifact.sha256:
                raise MirrorError(f"eval artifact step {step} changed during transfer")
            _copy_file_atomic_local(stage, local_final, expected_sha256=artifact.sha256)
        if not local_verified or force_verify:
            self.state.record_eval(artifact, tier="local")
            self.state.save()

        if local_only:
            return

        remote_final = f"{self.config.columbus_root}/heldout_eval_metrics/{name}"
        eval_state = self.state.payload["eval_artifacts"].get(str(step), {})
        remote_verified = bool(
            isinstance(eval_state, dict)
            and eval_state.get("columbus")
            and eval_state.get("sha256") == artifact.sha256
        )
        if remote_verified and not force_verify:
            pass
        elif self._remote_exists(self.columbus, remote_final):
            remote_artifact = _remote_eval(
                self.columbus,
                remote_final,
                manifest=manifest,
                metric_name=self.config.metric_name,
                metric_mode=self.config.metric_mode,
                run_id=self.config.run_id,
                run_evidence=run_evidence,
            )
            if remote_artifact.sha256 != artifact.sha256:
                raise MirrorError(f"Columbus immutable eval conflict: {remote_final}")
        else:
            _require_remote_space(
                self.columbus,
                str(self.config.columbus_root),
                incoming_bytes=artifact.size,
                reserve_bytes=self.config.reserve_bytes,
            )
            remote_stage = (
                f"{self.config.columbus_root}/heldout_eval_metrics/"
                f".incoming-{name}"
            )
            self.columbus.invoke_json(
                (
                    "--internal-create-stage",
                    self.config.run_id,
                    str(self.config.columbus_root),
                    remote_stage,
                    "file",
                ),
                mutation=True,
            )
            self.columbus.rsync_from_local(
                local_final,
                remote_stage,
                heartbeat=lambda: self._pulse(f"local_to_columbus_eval_{step}"),
            )
            self.columbus.invoke_json(
                (
                    "--internal-promote-eval",
                    self.config.run_id,
                    str(self.config.columbus_root),
                    remote_stage,
                    remote_final,
                    _manifest_b64(manifest),
                    artifact.sha256,
                    self.config.metric_name,
                    self.config.metric_mode,
                    self.config.run_id,
                    _evidence_b64(run_evidence),
                ),
                mutation=True,
            )
            remote_artifact = _remote_eval(
                self.columbus,
                remote_final,
                manifest=manifest,
                metric_name=self.config.metric_name,
                metric_mode=self.config.metric_mode,
                run_id=self.config.run_id,
                run_evidence=run_evidence,
            )
            if remote_artifact.sha256 != artifact.sha256:
                raise MirrorError(f"Columbus eval post-promotion failed: step {step}")
        if not remote_verified or force_verify:
            self.state.record_eval(artifact, tier="columbus")
            self.state.save()

    def _mirror_pointer(
        self,
        pointer: SelectionPointer,
        record: FileRecord,
        content: bytes,
        maximum_step: int,
        *,
        force_verify: bool = False,
        verify_source: bool = True,
    ) -> None:
        assert self.state is not None
        if verify_source and self.source_access_dir is None:
            raise MirrorError("H100 source-root binding is not initialized")
        source_path = (
            None
            if self.source_access_dir is None
            else f"{self.source_access_dir}/best_checkpoint.json"
        )
        if len(content) != record.size or hashlib.sha256(content).hexdigest() != record.sha256:
            raise MirrorError("source pointer content does not match its authenticated record")
        best_record = self.state.payload["checkpoints"].get(str(pointer.best_step), {})
        if not best_record.get("dual_verified") or not best_record.get("recoverable"):
            raise MirrorError(
                f"refusing to publish best pointer before steps_{pointer.best_step} "
                "and its recovery closure are dual verified"
            )

        existing_pointer = self.state.payload.get("best_pointer")
        if (
            not force_verify
            and isinstance(existing_pointer, dict)
            and existing_pointer.get("sha256") == record.sha256
            and existing_pointer.get("dual_verified")
        ):
            return

        local_history = self.config.local_root / "best_checkpoint_history"
        local_history.mkdir(mode=0o700, exist_ok=True)
        _ensure_secure_directory(
            local_history, create=False, label="local best-pointer history"
        )
        history_file = local_history / f"sha256-{record.sha256}.json"
        if history_file.exists() or history_file.is_symlink():
            existing = _regular_record(
                history_file, relative_path=history_file.name, label="pointer history"
            )
            if existing.sha256 != record.sha256:
                raise MirrorError(f"immutable pointer history conflict: {history_file}")
        else:
            _atomic_write(history_file, content)
        local_pointer, local_record = validate_pointer_file(
            history_file,
            metric_name=self.config.metric_name,
            metric_mode=self.config.metric_mode,
            maximum_step=maximum_step,
        )
        if local_pointer != pointer or local_record.sha256 != record.sha256:
            raise MirrorError("local best pointer verification failed")
        _atomic_write(self.config.local_root / "best_checkpoint.json", content)
        self.state.set_pointer_tier(pointer, record.sha256, tier="local")
        self.state.save()

        remote_history = (
            f"{self.config.columbus_root}/best_checkpoint_history/"
            f"sha256-{record.sha256}.json"
        )
        if not self._remote_exists(self.columbus, remote_history):
            remote_stage = (
                f"{self.config.columbus_root}/best_checkpoint_history/"
                f".incoming-pointer-{record.sha256}"
            )
            self.columbus.invoke_json(
                (
                    "--internal-create-stage",
                    self.config.run_id,
                    str(self.config.columbus_root),
                    remote_stage,
                    "file",
                ),
                mutation=True,
            )
            self.columbus.rsync_from_local(history_file, remote_stage)
            self.columbus.invoke_json(
                (
                    "--internal-promote-pointer",
                    self.config.run_id,
                    str(self.config.columbus_root),
                    remote_stage,
                    remote_history,
                    record.sha256,
                    self.config.metric_name,
                    self.config.metric_mode,
                    str(maximum_step),
                    "immutable",
                ),
                mutation=True,
            )
        remote_pointer, remote_record, _ = _remote_pointer(
            self.columbus,
            remote_history,
            metric_name=self.config.metric_name,
            metric_mode=self.config.metric_mode,
            maximum_step=maximum_step,
        )
        if remote_pointer != pointer or remote_record.sha256 != record.sha256:
            raise MirrorError("Columbus pointer-history verification failed")

        # The current pointer is deliberately mutable, but every version is
        # retained above under its content digest.  Installation validates that
        # the referenced best checkpoint exists in the protected mirror.
        remote_stage = (
            f"{self.config.columbus_root}/best_checkpoint_history/"
            ".incoming-current-pointer"
        )
        self.columbus.invoke_json(
            (
                "--internal-create-stage",
                self.config.run_id,
                str(self.config.columbus_root),
                remote_stage,
                "file",
            ),
            mutation=True,
        )
        self.columbus.rsync_from_local(history_file, remote_stage)
        self.columbus.invoke_json(
            (
                "--internal-promote-pointer",
                self.config.run_id,
                str(self.config.columbus_root),
                remote_stage,
                f"{self.config.columbus_root}/best_checkpoint.json",
                record.sha256,
                self.config.metric_name,
                self.config.metric_mode,
                str(maximum_step),
                "mutable",
            ),
            mutation=True,
        )
        current_pointer, current_record, _ = _remote_pointer(
            self.columbus,
            f"{self.config.columbus_root}/best_checkpoint.json",
            metric_name=self.config.metric_name,
            metric_mode=self.config.metric_mode,
            maximum_step=maximum_step,
        )
        if current_pointer != pointer or current_record.sha256 != record.sha256:
            raise MirrorError("Columbus current best-pointer verification failed")
        if verify_source:
            assert source_path is not None
            post_pointer, post_record, post_content = _remote_pointer(
                self.source,
                source_path,
                metric_name=self.config.metric_name,
                metric_mode=self.config.metric_mode,
                maximum_step=maximum_step,
            )
            if (
                post_pointer != pointer
                or post_record.sha256 != record.sha256
                or post_content != content
            ):
                raise MirrorError("source best pointer changed during mirror")
        self.state.set_pointer_tier(pointer, record.sha256, tier="columbus")
        self.state.save()

    def _mirror_evidence(
        self, identity: RunEvidenceIdentity, *, force_verify: bool = False
    ) -> None:
        assert self.state is not None
        if self.source_access_dir is None:
            raise MirrorError("H100 source-root binding is not initialized")
        expected = identity.manifest
        evidence_state = self.state.payload["evidence_snapshots"].get(
            expected.manifest_sha256, {}
        )
        snapshot_name = f"sha256-{expected.manifest_sha256}"
        local_parent = self.config.local_root / "evidence" / "snapshots"
        local_parent.mkdir(mode=0o700, exist_ok=True)
        _ensure_secure_directory(
            local_parent, create=False, label="local evidence snapshots"
        )
        local_final = local_parent / snapshot_name
        local_verified = bool(
            isinstance(evidence_state, dict) and evidence_state.get("local")
        )
        if local_verified and not force_verify:
            pass
        elif local_final.exists() or local_final.is_symlink():
            local_manifest = validate_evidence_snapshot(local_final)
            if local_manifest.manifest_sha256 != expected.manifest_sha256:
                raise MirrorError(f"local immutable evidence conflict: {local_final}")
        else:
            require_free_space(
                self.config.local_root,
                incoming_bytes=expected.total_size,
                reserve_bytes=self.config.reserve_bytes,
            )
            stage = local_parent / f".incoming-evidence-{expected.manifest_sha256}"
            _prepare_local_stage(stage, directory=True)
            with tempfile.NamedTemporaryFile(dir=self.state_path.parent, delete=False) as handle:
                files_from = Path(handle.name)
                for record in expected.files:
                    handle.write(record.path.encode("utf-8") + b"\x00")
                handle.flush()
                os.fsync(handle.fileno())
            try:
                self.source.rsync_to_local(
                    f"{self.source_access_dir}/",
                    Path(f"{stage}/"),
                    files_from=files_from,
                    delete=True,
                    heartbeat=lambda: self._pulse("h100_to_local_evidence"),
                )
                local_manifest = validate_evidence_snapshot(stage)
                post_source = _remote_evidence_identity(
                    self.source,
                    self.source_access_dir,
                    source_output_dir=self.config.source_run_dir,
                )
                if (
                    local_manifest.manifest_sha256 != expected.manifest_sha256
                    or post_source.manifest.manifest_sha256
                    != expected.manifest_sha256
                ):
                    raise MirrorError("run evidence changed during source transfer")
                _promote_local_tree(
                    stage,
                    local_final,
                    expected=expected,
                    metric_name=self.config.metric_name,
                    metric_mode=self.config.metric_mode,
                )
            except Exception:
                # Evidence snapshots use deterministic stages for crash-safe
                # checksum repair on the next cycle.
                raise
            finally:
                with contextlib.suppress(FileNotFoundError):
                    files_from.unlink()
        if not local_verified or force_verify:
            self.state.record_evidence(expected, tier="local")
            self.state.save()

        self._mirror_evidence_manifest_to_columbus(
            expected, local_final, force_verify=force_verify
        )

    def _mirror_evidence_manifest_to_columbus(
        self,
        expected: TreeManifest,
        local_final: Path,
        *,
        force_verify: bool,
    ) -> None:
        """Finish an immutable evidence snapshot using authenticated local bytes."""

        assert self.state is not None
        if expected.kind != "run_evidence":
            raise MirrorError("evidence reconciliation received a non-evidence manifest")
        actual_local = validate_evidence_snapshot(local_final)
        if actual_local.manifest_sha256 != expected.manifest_sha256:
            raise MirrorError("local evidence reconciliation bytes changed")

        snapshot_name = f"sha256-{expected.manifest_sha256}"
        remote_final = f"{self.config.columbus_root}/evidence/snapshots/{snapshot_name}"
        evidence_state = self.state.payload["evidence_snapshots"].get(
            expected.manifest_sha256, {}
        )
        remote_verified = bool(
            isinstance(evidence_state, dict) and evidence_state.get("columbus")
        )
        if remote_verified and not force_verify:
            pass
        elif self._remote_exists(self.columbus, remote_final):
            remote_manifest = TreeManifest.from_dict(
                self.columbus.invoke_json(
                    ("--internal-inspect-snapshot", remote_final)
                )
            )
            if remote_manifest.manifest_sha256 != expected.manifest_sha256:
                raise MirrorError(f"Columbus immutable evidence conflict: {remote_final}")
        else:
            _require_remote_space(
                self.columbus,
                str(self.config.columbus_root),
                incoming_bytes=expected.total_size,
                reserve_bytes=self.config.reserve_bytes,
            )
            remote_stage = (
                f"{self.config.columbus_root}/evidence/snapshots/"
                f".incoming-evidence-{expected.manifest_sha256}"
            )
            self.columbus.invoke_json(
                (
                    "--internal-create-stage",
                    self.config.run_id,
                    str(self.config.columbus_root),
                    remote_stage,
                    "directory",
                ),
                mutation=True,
            )
            self.columbus.rsync_from_local(
                f"{local_final}/",
                f"{remote_stage}/",
                delete=True,
                heartbeat=lambda: self._pulse("local_to_columbus_evidence"),
            )
            self.columbus.invoke_json(
                (
                    "--internal-promote-tree",
                    self.config.run_id,
                    str(self.config.columbus_root),
                    remote_stage,
                    remote_final,
                    _manifest_b64(expected),
                    self.config.metric_name,
                    self.config.metric_mode,
                ),
                mutation=True,
            )
            remote_manifest = TreeManifest.from_dict(
                self.columbus.invoke_json(("--internal-inspect-snapshot", remote_final))
            )
            if remote_manifest.manifest_sha256 != expected.manifest_sha256:
                raise MirrorError("Columbus evidence post-promotion verification failed")
        if not remote_verified or force_verify:
            self.state.record_evidence(expected, tier="columbus")
            self.state.save()

    def _reconcile_partial_evidence_snapshots(self) -> tuple[str, ...]:
        """Complete every journaled one-tier evidence copy before source access."""

        assert self.state is not None
        reconciled: list[str] = []
        for digest, record in sorted(self.state.payload["evidence_snapshots"].items()):
            if not isinstance(record, dict) or not isinstance(record.get("manifest"), dict):
                raise MirrorError("partial evidence state lacks its canonical manifest")
            expected = TreeManifest.from_dict(record["manifest"])
            if expected.manifest_sha256 != digest:
                raise MirrorError("partial evidence state digest/key mismatch")
            if record.get("dual_verified") is True:
                continue
            if record.get("local") is not True:
                raise MirrorError(
                    "partial evidence cannot be recovered without authenticated local bytes"
                )
            local_final = (
                self.config.local_root
                / "evidence"
                / "snapshots"
                / f"sha256-{digest}"
            )
            self._mirror_evidence_manifest_to_columbus(
                expected, local_final, force_verify=True
            )
            repaired = self.state.payload["evidence_snapshots"].get(digest, {})
            if not isinstance(repaired, dict) or repaired.get("dual_verified") is not True:
                raise MirrorError("partial evidence reconciliation did not dual-verify")
            reconciled.append(digest)
        return tuple(reconciled)

    def _scrub_all_evidence_snapshots(self) -> None:
        assert self.state is not None
        for digest, record in self.state.payload["evidence_snapshots"].items():
            if not isinstance(record, dict) or not isinstance(record.get("manifest"), dict):
                raise MirrorError("cached evidence snapshot lacks its canonical manifest")
            expected = TreeManifest.from_dict(record["manifest"])
            if expected.manifest_sha256 != digest:
                raise MirrorError("cached evidence snapshot digest/key mismatch")
            local = self.config.local_root / "evidence" / "snapshots" / f"sha256-{digest}"
            actual_local = validate_evidence_snapshot(local)
            if actual_local.manifest_sha256 != digest:
                raise MirrorError(f"local evidence scrub failed: {digest}")
            remote_path = f"{self.config.columbus_root}/evidence/snapshots/sha256-{digest}"
            actual_remote = TreeManifest.from_dict(
                self.columbus.invoke_json(("--internal-inspect-snapshot", remote_path))
            )
            if actual_remote.manifest_sha256 != digest:
                raise MirrorError(f"Columbus evidence scrub failed: {digest}")
            self.state.record_evidence(expected, tier="local")
            self.state.record_evidence(expected, tier="columbus")
        self.state.save()

    def _scrub_published_recovery_closure(self) -> None:
        """Authenticate both copies of every object reachable from the live index."""

        assert self.state is not None
        generation = _strict_int(
            self.state.payload.get("restore_index_generation", 0),
            label="full-scrub restore-index generation",
        )
        recorded_sha = self.state.payload.get("restore_index_sha256")
        if generation == 0:
            if recorded_sha is not None:
                raise MirrorError("full scrub found inconsistent empty restore-index state")
            return
        if not isinstance(recorded_sha, str) or SHA256_RE.fullmatch(recorded_sha) is None:
            raise MirrorError("full scrub restore-index SHA-256 state is invalid")
        local = verify_restore_index_at_root(
            self.config.local_root,
            run_id=self.config.run_id,
            metric_name=self.config.metric_name,
            metric_mode=self.config.metric_mode,
        )
        remote = self.columbus.invoke_json(
            (
                "--internal-verify-index",
                self.config.run_id,
                str(self.config.columbus_root),
                self.config.metric_name,
                self.config.metric_mode,
                "full",
            )
        )
        if local != remote:
            raise MirrorError("local and Columbus full recovery scrubs disagree")
        if (
            local.get("verified") is not True
            or local.get("generation") != generation
            or local.get("restore_index_sha256") != recorded_sha
        ):
            raise MirrorError("full recovery scrub disagrees with durable index state")
        expected_steps = sorted(
            int(step)
            for step, record in self.state.payload["checkpoints"].items()
            if isinstance(record, dict) and record.get("recoverable") is True
        )
        durable_pointer = self.state.payload.get("best_pointer")
        expected_best = (
            int(durable_pointer["best_metric_step"])
            if isinstance(durable_pointer, dict)
            and durable_pointer.get("dual_verified") is True
            else None
        )
        if (
            local.get("checkpoint_steps") != expected_steps
            or local.get("best_step") != expected_best
        ):
            raise MirrorError("full recovery scrub closure differs from durable state")

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _reconcile_prune_pending(self) -> None:
        assert self.state is not None
        pending = self.state.payload.get("prune_pending", {})
        if not isinstance(pending, dict):
            raise MirrorError("prune_pending state is malformed")
        active_steps = sorted(
            (int(key) for key, value in pending.items() if value.get("phase") != "complete"),
            reverse=True,
        )
        if not active_steps:
            return
        if any(pending[str(step)]["phase"] == "planned" for step in active_steps):
            self._publish_restore_index(excluded_steps=active_steps)
            withdrawn_generation = int(
                self.state.payload["restore_index_generation"]
            )
            for step in active_steps:
                if pending[str(step)]["phase"] == "planned":
                    pending[str(step)]["phase"] = "index_withdrawn"
                    pending[str(step)]["index_generation"] = withdrawn_generation
                    pending[str(step)]["index_withdrawn_at"] = utc_now()
            self.state.save()

        for step in active_steps:
            intent = pending[str(step)]
            digest = intent["manifest_sha256"]
            suffix = digest[:16]
            local_live = self.config.local_root / "checkpoints" / f"steps_{step}"
            local_trash = local_live.parent / f".trash-steps_{step}-{suffix}"
            remote_live = f"{self.config.columbus_root}/checkpoints/steps_{step}"
            remote_trash = (
                f"{self.config.columbus_root}/checkpoints/.trash-steps_{step}-{suffix}"
            )
            phase = intent["phase"]
            if phase == "index_withdrawn":
                if local_live.exists() or local_live.is_symlink():
                    current = validate_checkpoint_tree(
                        local_live,
                        metric_name=self.config.metric_name,
                        metric_mode=self.config.metric_mode,
                    )
                    if current.manifest_sha256 != digest:
                        raise MirrorError(f"refusing to prune changed local steps_{step}")
                    if local_trash.exists() or local_trash.is_symlink():
                        raise MirrorError(f"deterministic local trash conflict: {local_trash}")
                    os.replace(local_live, local_trash)
                    self._fsync_directory(local_live.parent)
                elif not local_trash.is_dir():
                    raise MirrorError(
                        f"prune reconciliation lost local steps_{step} and its trash"
                    )
                intent["phase"] = "local_trashed"
                intent["local_trashed_at"] = utc_now()
                self.state.save()
                phase = "local_trashed"
            if phase == "local_trashed":
                self.columbus.invoke_json(
                    (
                        "--internal-trash-checkpoint",
                        self.config.run_id,
                        str(self.config.columbus_root),
                        remote_live,
                        remote_trash,
                        digest,
                        self.config.metric_name,
                        self.config.metric_mode,
                    ),
                    mutation=True,
                )
                intent["phase"] = "columbus_trashed"
                intent["columbus_trashed_at"] = utc_now()
                self.state.save()
                phase = "columbus_trashed"
            if phase == "columbus_trashed":
                if local_trash.exists() or local_trash.is_symlink():
                    if local_trash.is_symlink() or not local_trash.is_dir():
                        raise MirrorError(f"local prune trash has unsafe type: {local_trash}")
                    shutil.rmtree(local_trash)
                    self._fsync_directory(local_trash.parent)
                intent["phase"] = "local_deleted"
                intent["local_deleted_at"] = utc_now()
                self.state.save()
                phase = "local_deleted"
            if phase == "local_deleted":
                self.columbus.invoke_json(
                    (
                        "--internal-delete-trash",
                        self.config.run_id,
                        str(self.config.columbus_root),
                        remote_trash,
                    ),
                    mutation=True,
                )
                intent["phase"] = "columbus_deleted"
                intent["columbus_deleted_at"] = utc_now()
                self.state.save()
                phase = "columbus_deleted"
            if phase == "columbus_deleted":
                self.state.mark_pruned(step, "local")
                self.state.mark_pruned(step, "columbus")
                receipt = {
                    "schema_version": SCHEMA_VERSION,
                    "record_kind": "prune_receipt",
                    "run_id": self.config.run_id,
                    "step": step,
                    "manifest_sha256": digest,
                    "index_generation": intent["index_generation"],
                    "terminal_phase": "complete",
                }
                receipt_sha = self._persist_immutable_record(
                    f"manifests/prune_receipts/steps_{step}.json", receipt
                )
                self.state.payload["prune_receipts"][str(step)] = {
                    **receipt,
                    "receipt_sha256": receipt_sha,
                }
                intent["phase"] = "complete"
                intent["completed_at"] = utc_now()
                intent["receipt_sha256"] = receipt_sha
                self.state.save()

    def _retention_victims(
        self, manifests: Mapping[int, TreeManifest], best_step: int | None
    ) -> tuple[int, ...]:
        assert self.state is not None
        all_manifests = _cached_manifests(self.state)
        all_manifests.update(manifests)
        verified = [
            int(step)
            for step, record in self.state.payload["checkpoints"].items()
            if record.get("dual_verified") and record.get("recoverable")
        ]
        keep = set(
            retained_recovery_steps(
                all_manifests,
                verified,
                best_step=best_step,
                limit=self.config.retain,
            )
        )
        return tuple(sorted(set(verified) - keep, reverse=True))

    def _prune(self, manifests: Mapping[int, TreeManifest], best_step: int | None) -> None:
        assert self.state is not None
        self._reconcile_prune_pending()
        victims = self._retention_victims(manifests, best_step)
        if not victims:
            return
        generation = int(self.state.payload.get("restore_index_generation", 0)) + 1
        pending = self.state.payload["prune_pending"]
        for step in victims:
            record = self.state.payload["checkpoints"][str(step)]
            record["recoverable"] = False
            pending[str(step)] = {
                "step": step,
                "phase": "planned",
                "planned_at": utc_now(),
                "manifest_sha256": record["manifest_sha256"],
                "index_generation": generation,
            }
        self.state.save()
        self._reconcile_prune_pending()

    def run_once(self, *, force_full_scrub: bool = False) -> dict[str, Any]:
        if self.runner.dry_run:
            return self.preflight()
        with exclusive_lock(self.lock_path):
            _local_mirror_root(self.config.local_root, create=True)
            h100_resolved = self._init_state(defer_h100_resolution=True)
            assert self.state is not None
            self._pulse("starting")
            try:
                # Finish crash-pending destructive work before touching the
                # source.  Source outages must not strand a checkpoint between
                # local and Columbus trash phases after its index withdrawal.
                active_prune_before_source = any(
                    isinstance(item, dict) and item.get("phase") != "complete"
                    for item in self.state.payload.get("prune_pending", {}).values()
                )
                index_publication_before_source = (
                    self.state.payload.get("restore_index_intent") is not None
                )
                pointer_publication_before_source = (
                    self.state.payload.get("pointer_publication_intent") is not None
                )
                evidence_reconciliation_before_source = any(
                    isinstance(record, dict)
                    and record.get("dual_verified") is not True
                    and (
                        record.get("local") is True
                        or record.get("columbus") is True
                    )
                    for record in self.state.payload.get(
                        "evidence_snapshots", {}
                    ).values()
                )
                recovery_publication_before_source = any(
                    isinstance(item, dict)
                    and item.get("recovery_intent") is not None
                    and not item.get("recoverable")
                    and not (
                        item.get("local", {}).get("pruned")
                        and item.get("columbus", {}).get("pruned")
                    )
                    for item in self.state.payload.get("checkpoints", {}).values()
                )
                if (
                    active_prune_before_source
                    or index_publication_before_source
                    or pointer_publication_before_source
                    or evidence_reconciliation_before_source
                    or recovery_publication_before_source
                ):
                    self._acquire_remote_lock()
                    self._reconcile_restore_index_intent()
                    self._reconcile_partial_evidence_snapshots()
                    self._reconcile_recovery_intents()
                    self._reconcile_pointer_publication_intent(
                        verify_source=False
                    )
                if active_prune_before_source:
                    self._reconcile_prune_pending()
                durable_selection_complete, durable_best_step = self._durable_selection_state()
                if not durable_selection_complete:
                    has_recoverable = any(
                        isinstance(record, dict) and record.get("recoverable") is True
                        for record in self.state.payload["checkpoints"].values()
                    )
                    if has_recoverable:
                        self._acquire_remote_lock()
                        self._reconcile_recoverable_pointer_from_checkpoint()
                        durable_selection_complete, durable_best_step = (
                            self._durable_selection_state()
                        )
                if durable_selection_complete:
                    pre_source_victims = self._retention_victims(
                        {}, durable_best_step
                    )
                    pre_source_index_work = (
                        self._restore_index_needs_publication()
                    )
                    if pre_source_victims:
                        self._acquire_remote_lock()
                        self._prune({}, durable_best_step)
                    elif pre_source_index_work:
                        self._acquire_remote_lock()
                        self._publish_restore_index()
                if not h100_resolved:
                    self._revalidate_h100_identity()
                scan = self.inspect_source(force_full_scrub=force_full_scrub)
                self.state.save()
                active_prune = any(
                    item.get("phase") != "complete"
                    for item in self.state.payload.get("prune_pending", {}).values()
                )
                unfinished = [
                    step
                    for step in scan.finalized_steps
                    if (
                        not self.state.payload["checkpoints"]
                        .get(str(step), {})
                        .get("recoverable")
                        and not (
                            self.state.payload["checkpoints"]
                            .get(str(step), {})
                            .get("local", {})
                            .get("pruned")
                            and self.state.payload["checkpoints"]
                            .get(str(step), {})
                            .get("columbus", {})
                            .get("pruned")
                        )
                    )
                ]
                unresolved_recovery_steps = sorted(
                    int(step_text)
                    for step_text, record in self.state.payload["checkpoints"].items()
                    if isinstance(record, dict)
                    and record.get("recovery_intent") is not None
                    and not record.get("recoverable")
                    and not (
                        record.get("local", {}).get("pruned")
                        and record.get("columbus", {}).get("pruned")
                    )
                )
                missing_recovery_sources = sorted(
                    set(unresolved_recovery_steps) - set(scan.finalized_steps)
                )
                if missing_recovery_sources:
                    raise MirrorError(
                        "unfinished recovery intents lost their finalized source bytes: "
                        f"{missing_recovery_sources}"
                    )
                evidence_state = self.state.payload["evidence_snapshots"].get(
                    scan.evidence.manifest.manifest_sha256, {}
                )
                pointer_needs_work = bool(
                    scan.pointer is not None
                    and (
                        not isinstance(self.state.payload.get("best_pointer"), dict)
                        or self.state.payload["best_pointer"].get("sha256")
                        != scan.pointer_record.sha256
                        or not self.state.payload["best_pointer"].get("dual_verified")
                    )
                )
                index_needs_work = self._restore_index_needs_publication()
                durable_pointer = self.state.payload.get("best_pointer")
                durable_best_step = (
                    int(durable_pointer["best_metric_step"])
                    if isinstance(durable_pointer, dict)
                    and durable_pointer.get("dual_verified") is True
                    else None
                )
                retention_needs_work = bool(
                    self._retention_victims(
                        scan.manifests,
                        durable_best_step,
                    )
                )
                work_required = bool(
                    unfinished
                    or scan.full_scrub
                    or active_prune
                    or pointer_needs_work
                    or index_needs_work
                    or retention_needs_work
                    or (unfinished and not evidence_state.get("dual_verified"))
                )
                if work_required:
                    self._acquire_remote_lock()
                    self._reconcile_prune_pending()
                    if unfinished or scan.full_scrub:
                        self._mirror_evidence(
                            scan.evidence, force_verify=scan.full_scrub
                        )
                    if scan.full_scrub:
                        self._scrub_all_evidence_snapshots()
                    scrub_steps = [
                        int(step)
                        for step, record in self.state.payload["checkpoints"].items()
                        if scan.full_scrub
                        and record.get("dual_verified")
                        and not record.get("local", {}).get("pruned")
                        and not record.get("columbus", {}).get("pruned")
                    ]
                    for step in sorted(set(unfinished) | set(scrub_steps)):
                        manifest = scan.manifests.get(step)
                        artifact = scan.evals.get(step)
                        if manifest is None or artifact is None:
                            raise MirrorError(
                                f"cannot verify steps_{step}; cached manifest/eval is missing"
                            )
                        record = self.state.payload["checkpoints"][str(step)]
                        receipt_inputs: tuple[
                            TreeManifest,
                            EvalArtifact,
                            RunEvidenceIdentity,
                            tuple[int, ...],
                        ] | None = None
                        if not record.get("recoverable"):
                            closure = validate_recovery_closure(step, scan.manifests)
                            # Journal the authenticated receipt inputs before
                            # either tier copy.  If both copies finish and the
                            # trainer then removes its source checkpoint, a
                            # restart can still finalize recovery from state.
                            receipt_inputs = self._prepare_recovery_intent(
                                step=step,
                                manifest=manifest,
                                artifact=artifact,
                                evidence=scan.evidence,
                                closure=closure,
                            )
                        # Pin the tiny eval locally before the long checkpoint
                        # two-hop. If the trainer then applies source retention,
                        # a dual-verified checkpoint can never be stranded only
                        # because its eval was not captured yet.
                        self._mirror_eval(
                            step,
                            artifact,
                            manifest,
                            scan.evidence,
                            force_verify=scan.full_scrub,
                            local_only=True,
                        )
                        self._mirror_checkpoint(
                            step, manifest, force_verify=scan.full_scrub
                        )
                        self._mirror_eval(
                            step,
                            artifact,
                            manifest,
                            scan.evidence,
                            force_verify=scan.full_scrub,
                        )
                        if receipt_inputs is not None:
                            (
                                receipt_manifest,
                                receipt_artifact,
                                receipt_evidence,
                                receipt_closure,
                            ) = receipt_inputs
                            receipt_sha = self._persist_recovery_receipt(
                                step=step,
                                manifest=receipt_manifest,
                                artifact=receipt_artifact,
                                evidence=receipt_evidence,
                                closure=receipt_closure,
                            )
                            self.state.mark_recoverable(
                                step,
                                receipt_closure,
                                evidence_manifest_sha256=(
                                    receipt_evidence.manifest.manifest_sha256
                                ),
                                receipt_sha256=receipt_sha,
                            )
                            self.state.save()
                    if self.state.payload.get("pointer_publication_intent") is not None:
                        if not self._reconcile_pointer_publication_intent(
                            verify_source=True
                        ):
                            raise MirrorError(
                                "pointer publication target is not recoverable after mirroring"
                            )
                    durable_pointer = self.state.payload.get("best_pointer")
                    current_pointer_matches = bool(
                        scan.pointer is not None
                        and isinstance(durable_pointer, dict)
                        and durable_pointer.get("dual_verified") is True
                        and durable_pointer.get("sha256")
                        == scan.pointer_record.sha256
                    )
                    if scan.pointer is not None and not current_pointer_matches:
                        assert scan.pointer_record is not None
                        assert scan.pointer_content is not None
                        self._prepare_pointer_publication_intent(
                            pointer=scan.pointer,
                            record=scan.pointer_record,
                            content=scan.pointer_content,
                            maximum_step=max(scan.manifests),
                        )
                        if not self._reconcile_pointer_publication_intent(
                            verify_source=True
                        ):
                            raise MirrorError(
                                "source pointer target is not recoverable after mirroring"
                            )
                    elif scan.pointer is not None and scan.full_scrub:
                        assert scan.pointer_record is not None
                        assert scan.pointer_content is not None
                        self._mirror_pointer(
                            scan.pointer,
                            scan.pointer_record,
                            scan.pointer_content,
                            max(scan.manifests),
                            force_verify=scan.full_scrub,
                        )
                    if (
                        index_needs_work
                        or retention_needs_work
                        or scan.pointer is not None
                        or not scan.evals
                    ):
                        self._publish_restore_index()
                        self._prune(
                            scan.manifests,
                            None if scan.pointer is None else scan.pointer.best_step,
                        )
                    if scan.full_scrub or self.state.payload.get("last_full_scrub_at") is None:
                        self._scrub_published_recovery_closure()
                        self.state.payload["last_full_scrub_at"] = utc_now()
                status = "healthy_pending" if scan.pending_steps else "healthy"
                self.state.heartbeat(status=status)
                heartbeat = self.state.payload["heartbeat"]
                assert isinstance(heartbeat, dict)
                heartbeat["pending_steps"] = list(scan.pending_steps)
                heartbeat = self.state.payload["heartbeat"]
                assert isinstance(heartbeat, dict)
                finalized_backlog = sorted(
                    set(heartbeat["backlog_steps"]) - set(scan.pending_steps)
                )
                if len(finalized_backlog) > self.config.max_backlog:
                    self.state.heartbeat(status="degraded_backlog")
                    heartbeat = self.state.payload["heartbeat"]
                    heartbeat["pending_steps"] = list(scan.pending_steps)
                self.state.save()
                _atomic_write(
                    self.heartbeat_path,
                    json.dumps(
                        self.state.payload["heartbeat"], indent=2, sort_keys=True
                    ).encode("utf-8")
                    + b"\n",
                )
                return dict(self.state.payload["heartbeat"])
            except Exception as exc:
                self.state.heartbeat(status="failed", error=str(exc))
                self.state.save()
                _atomic_write(
                    self.heartbeat_path,
                    json.dumps(
                        self.state.payload["heartbeat"], indent=2, sort_keys=True
                    ).encode("utf-8")
                    + b"\n",
                )
                raise
            finally:
                self._release_remote_lock()


def _secure_mkdirs_below(root: Path, directory: Path) -> None:
    _require_contained(directory, root, label="remote directory")
    relative = directory.relative_to(root)
    current = root
    for part in relative.parts:
        if part in {"", ".", ".."}:
            raise MirrorError(f"unsafe directory component: {part!r}")
        current = current / part
        if not current.exists():
            current.mkdir(mode=0o700)
        _ensure_secure_directory(current, create=False, label="remote mirror directory")


def _validate_remote_target(root: Path, path: Path, *, staging: bool = False) -> None:
    _require_contained(path, root, label="remote target")
    relative = path.relative_to(root)
    if not relative.parts or relative.parts[0] not in {
        "checkpoints",
        "heldout_eval_metrics",
        "evidence",
        "manifests",
        "best_checkpoint_history",
        "best_checkpoint.json",
        "restore-index.json",
        ".mirror.lock",
    }:
        raise MirrorError(f"remote target is outside the mirror allowlist: {path}")
    if staging and not path.name.startswith(".incoming-"):
        raise MirrorError(f"remote staging basename is invalid: {path.name}")


def _internal_list_source(run_dir: Path) -> dict[str, Any]:
    _validate_root_directory(run_dir, label="source run")
    checkpoints_dir = run_dir / "checkpoints"
    eval_dir = run_dir / "heldout_eval_metrics"
    checkpoints: list[str] = []
    evals: list[str] = []
    if checkpoints_dir.exists() or checkpoints_dir.is_symlink():
        _validate_root_directory(checkpoints_dir, label="source checkpoints")
        for entry in os.scandir(checkpoints_dir):
            if CHECKPOINT_RE.fullmatch(entry.name):
                checkpoints.append(entry.name)
    if eval_dir.exists() or eval_dir.is_symlink():
        _validate_root_directory(eval_dir, label="source eval directory")
        for entry in os.scandir(eval_dir):
            if EVAL_RE.fullmatch(entry.name):
                evals.append(entry.name)
    pointer_path = run_dir / "best_checkpoint.json"
    return {
        "checkpoints": sorted(checkpoints, key=checkpoint_step_from_name),
        "eval_artifacts": sorted(evals, key=eval_step_from_name),
        "best_checkpoint_pointer": pointer_path.exists() or pointer_path.is_symlink(),
    }


def _internal_endpoint_identity() -> dict[str, Any]:
    return {
        "user": getpass.getuser(),
        "hostname": socket.gethostname(),
        "fqdn": socket.getfqdn(),
        "python_version": sys.version.split()[0],
    }


def _tool_compatibility_report() -> dict[str, Any]:
    rsync_path = shutil.which("rsync")
    rsync_version = None
    required_options: dict[str, bool] = {}
    if rsync_path is not None:
        result = subprocess.run(
            (rsync_path, "--version"),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if result.returncode == 0 and result.stdout:
            rsync_version = result.stdout.splitlines()[0]
        help_result = subprocess.run(
            (rsync_path, "--help"),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        help_text = help_result.stdout if help_result.returncode == 0 else ""
        protect_args_result = subprocess.run(
            (rsync_path, "--protect-args", "--version"),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        required_options = {
            "--fsync": "--fsync" in help_text,
            # Modern rsync help advertises the canonical --secluded-args name,
            # even though the --protect-args spelling used by our transfer
            # commands remains supported. Probe the exact runtime spelling.
            "--protect-args": protect_args_result.returncode == 0,
            "--partial-dir": "--partial-dir" in help_text,
        }
    return {
        "python_version": sys.version.split()[0],
        "rsync_path": rsync_path,
        "rsync_version": rsync_version,
        "rsync_required_options": required_options,
    }


def _nearest_existing_remote_directory(path: Path) -> Path:
    candidate = path
    while not candidate.exists():
        if candidate.parent == candidate:
            raise MirrorError(f"no existing directory ancestor for {path}")
        candidate = candidate.parent
    _validate_root_directory(candidate, label="remote free-space ancestor")
    return candidate


def _internal_preflight_source(run_dir: Path) -> dict[str, Any]:
    inventory = lightweight_source_inventory(run_dir)
    root = _nearest_existing_remote_directory(run_dir)
    stats = os.statvfs(root)
    return {
        "identity": _internal_endpoint_identity(),
        "tools": _tool_compatibility_report(),
        "available_bytes": stats.f_bavail * stats.f_frsize,
        "inventory": inventory,
    }


def _internal_preflight_columbus(run_id: str, root: Path) -> dict[str, Any]:
    validate_columbus_root(run_id, root)
    ancestor = _nearest_existing_remote_directory(root)
    stats = os.statvfs(ancestor)
    statuses = {
        "protected_parent": _internal_path_status(SAFE_COLUMBUS_PARENT),
        "storage_base": _internal_path_status(SAFE_COLUMBUS_BASE),
        "run_root": _internal_path_status(root),
        "lock": _internal_path_status(root / ".mirror.lock"),
        "restore_index": _internal_path_status(root / "restore-index.json"),
    }
    if not statuses["protected_parent"]["exists"]:
        raise MirrorError(
            f"protected Columbus relay parent must be provisioned first: {SAFE_COLUMBUS_PARENT}"
        )
    for label in ("protected_parent", "storage_base", "run_root"):
        path = {
            "protected_parent": SAFE_COLUMBUS_PARENT,
            "storage_base": SAFE_COLUMBUS_BASE,
            "run_root": root,
        }[label]
        if statuses[label]["exists"]:
            info = path.lstat()
            statuses[label].update(
                {"uid": info.st_uid, "mode": stat.S_IMODE(info.st_mode)}
            )
            if (
                statuses[label]["kind"] != "directory"
                or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) & 0o077
            ):
                raise MirrorError(f"Columbus {label} owner/mode/type is unsafe")
    conflicts: list[str] = []
    if root.is_dir():
        allowed = {
            "checkpoints",
            "heldout_eval_metrics",
            "evidence",
            "manifests",
            "best_checkpoint_history",
            "best_checkpoint.json",
            "restore-index.json",
            ".mirror.lock",
        }
        conflicts = sorted(entry.name for entry in os.scandir(root) if entry.name not in allowed)
    return {
        "identity": _internal_endpoint_identity(),
        "tools": _tool_compatibility_report(),
        "available_bytes": stats.f_bavail * stats.f_frsize,
        "space_checked_at": str(ancestor),
        "paths": statuses,
        "destination_conflicts": conflicts,
    }


def _internal_path_status(path: Path) -> dict[str, Any]:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return {"exists": False, "kind": None}
    if stat.S_ISLNK(info.st_mode):
        kind = "symlink"
    elif stat.S_ISDIR(info.st_mode):
        kind = "directory"
    elif stat.S_ISREG(info.st_mode):
        kind = "file"
    else:
        kind = "special"
    return {"exists": True, "kind": kind}


def _internal_acquire_lock(run_id: str, root: Path, token: str) -> dict[str, Any]:
    if not token or not re.fullmatch(r"[A-Za-z0-9-]{8,128}", token):
        raise MirrorError("remote lock token is unsafe")
    _ensure_columbus_root(run_id, root)
    lock_dir = root / ".mirror.lock"
    try:
        lock_dir.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise MirrorError(
            f"Columbus mirror lock already exists (manual stale-lock audit required): {lock_dir}"
        ) from exc
    _atomic_write(
        lock_dir / "owner.json",
        json.dumps({"token": token, "created_at": utc_now()}, sort_keys=True).encode("utf-8")
        + b"\n",
    )
    return {"locked": True, "path": str(lock_dir)}


def _internal_release_lock(run_id: str, root: Path, token: str) -> dict[str, Any]:
    validate_columbus_root(run_id, root)
    lock_dir = root / ".mirror.lock"
    _validate_root_directory(lock_dir, label="remote mirror lock")
    owner = _load_json_object(lock_dir / "owner.json", label="remote mirror lock owner")
    if owner.get("token") != token:
        raise MirrorError("refusing to release a remote lock owned by another token")
    (lock_dir / "owner.json").unlink()
    lock_dir.rmdir()
    return {"released": True}


def _internal_create_stage(
    run_id: str,
    root: Path,
    stage: Path,
    object_kind: str,
) -> dict[str, Any]:
    _ensure_columbus_root(run_id, root)
    _validate_remote_target(root, stage, staging=True)
    _secure_mkdirs_below(root, stage.parent)
    if stage.exists() or stage.is_symlink():
        info = stage.lstat()
        expected_directory = object_kind == "directory"
        valid = stat.S_ISDIR(info.st_mode) if expected_directory else stat.S_ISREG(info.st_mode)
        if stat.S_ISLNK(info.st_mode) or not valid:
            raise MirrorError(f"remote staging target has wrong type: {stage}")
        if not expected_directory and info.st_nlink != 1:
            raise MirrorError(f"remote staging file is hard-linked: {stage}")
        return {"prepared": True, "reused": True, "path": str(stage), "kind": object_kind}
    if object_kind == "directory":
        stage.mkdir(mode=0o700)
    elif object_kind != "file":
        raise MirrorError(f"unsupported staging object kind: {object_kind}")
    return {"prepared": True, "reused": False, "path": str(stage), "kind": object_kind}


def _internal_put_record(
    run_id: str,
    root: Path,
    relative_path: str,
    content_b64: str,
    expected_sha256: str,
    behavior: str,
) -> dict[str, Any]:
    _ensure_columbus_root(run_id, root)
    if not _safe_relative_path(relative_path) or not (
        relative_path.startswith("manifests/")
        or relative_path == "restore-index.json"
    ):
        raise MirrorError(f"remote record path is outside its allowlist: {relative_path!r}")
    if SHA256_RE.fullmatch(expected_sha256) is None:
        raise MirrorError("remote record expected SHA-256 is invalid")
    try:
        content = base64.urlsafe_b64decode(content_b64.encode("ascii"))
    except Exception as exc:
        raise MirrorError("remote record content is not valid base64") from exc
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise MirrorError("remote record content SHA-256 mismatch")
    target = root / relative_path
    _validate_remote_target(root, target)
    _secure_mkdirs_below(root, target.parent)
    if behavior == "immutable" and (target.exists() or target.is_symlink()):
        current = _regular_record(
            target, relative_path=target.name, label="remote immutable record"
        )
        if current.sha256 != expected_sha256:
            raise MirrorError(f"remote immutable record conflict: {target}")
    elif behavior == "immutable":
        _atomic_write(target, content)
    elif behavior == "mutable":
        _atomic_write(target, content)
    else:
        raise MirrorError(f"unknown remote record behavior: {behavior!r}")
    return {"path": str(target), "sha256": expected_sha256, "behavior": behavior}


def _internal_promote_tree(
    run_id: str,
    root: Path,
    stage: Path,
    final: Path,
    expected: TreeManifest,
    metric_name: str,
    metric_mode: str,
) -> dict[str, Any]:
    _ensure_columbus_root(run_id, root)
    _validate_remote_target(root, stage, staging=True)
    _validate_remote_target(root, final)
    if stage.parent != final.parent:
        raise MirrorError("tree promotion must rename within the same filesystem directory")
    _promote_local_tree(
        stage,
        final,
        expected=expected,
        metric_name=metric_name,
        metric_mode=metric_mode,
    )
    return {"promoted": True, "path": str(final), "manifest_sha256": expected.manifest_sha256}


def _internal_promote_eval(
    run_id: str,
    root: Path,
    stage: Path,
    final: Path,
    manifest: TreeManifest,
    expected_sha256: str,
    metric_name: str,
    metric_mode: str,
    evidence: RunEvidenceIdentity,
) -> dict[str, Any]:
    _ensure_columbus_root(run_id, root)
    _validate_remote_target(root, stage, staging=True)
    _validate_remote_target(root, final)
    if stage.parent != final.parent:
        raise MirrorError("eval promotion must rename within one directory")
    artifact = validate_eval_artifact(
        stage,
        checkpoint_manifest=manifest,
        metric_name=metric_name,
        metric_mode=metric_mode,
        run_id=run_id,
        run_evidence=evidence,
        expected_step=manifest.step,
    )
    if artifact.sha256 != expected_sha256:
        raise MirrorError("staged eval SHA-256 does not match expected source bytes")
    _copy_file_atomic_local(stage, final, expected_sha256=expected_sha256)
    return {"promoted": True, "path": str(final), "sha256": expected_sha256}


def _internal_promote_pointer(
    run_id: str,
    root: Path,
    stage: Path,
    final: Path,
    expected_sha256: str,
    metric_name: str,
    metric_mode: str,
    maximum_step: int,
    behavior: str,
) -> dict[str, Any]:
    _ensure_columbus_root(run_id, root)
    _validate_remote_target(root, stage, staging=True)
    _validate_remote_target(root, final)
    pointer, record = validate_pointer_file(
        stage,
        metric_name=metric_name,
        metric_mode=metric_mode,
        maximum_step=maximum_step,
    )
    if record.sha256 != expected_sha256:
        raise MirrorError("staged pointer SHA-256 mismatch")
    best_path = root / pointer.checkpoint_relative_path
    manifests: dict[int, TreeManifest] = {}
    for candidate in (root / "checkpoints").glob("steps_*"):
        candidate_step = checkpoint_step_from_name(candidate.name)
        manifests[candidate_step] = validate_checkpoint_tree(
            candidate, metric_name=metric_name, metric_mode=metric_mode
        )
    if (
        pointer.best_step not in manifests
        or best_path != root / "checkpoints" / f"steps_{pointer.best_step}"
    ):
        raise MirrorError("pointer references a missing/noncanonical Columbus best checkpoint")
    validate_recovery_closure(pointer.best_step, manifests)
    if behavior == "immutable":
        _copy_file_atomic_local(stage, final, expected_sha256=expected_sha256)
    elif behavior == "mutable":
        payload = stage.read_bytes()
        _atomic_write(final, payload)
        stage.unlink()
    else:
        raise MirrorError(f"unknown pointer promotion behavior: {behavior}")
    return {"promoted": True, "path": str(final), "best_step": pointer.best_step}


def _internal_delete_checkpoint(
    run_id: str,
    root: Path,
    path: Path,
    expected_sha256: str,
    metric_name: str,
    metric_mode: str,
) -> dict[str, Any]:
    _ensure_columbus_root(run_id, root)
    if path.parent != root / "checkpoints":
        raise MirrorError("checkpoint deletion path is outside exact checkpoints directory")
    step = checkpoint_step_from_name(path.name)
    current = validate_checkpoint_tree(path, metric_name=metric_name, metric_mode=metric_mode)
    if current.manifest_sha256 != expected_sha256:
        raise MirrorError("refusing to delete checkpoint whose bytes changed")
    pointer_path = root / "best_checkpoint.json"
    if pointer_path.exists() or pointer_path.is_symlink():
        pointer, _ = validate_pointer_file(
            pointer_path,
            metric_name=metric_name,
            metric_mode=metric_mode,
        )
        if pointer.best_step == step:
            raise MirrorError("refusing to delete the selected best checkpoint")
    for candidate in (root / "checkpoints").glob("steps_*"):
        if candidate == path:
            continue
        candidate_manifest = validate_checkpoint_tree(
            candidate, metric_name=metric_name, metric_mode=metric_mode
        )
        if candidate_manifest.selection_best_step == step:
            raise MirrorError(
                f"refusing to delete steps_{step}; {candidate.name} depends on it"
            )
    trash = path.parent / f".trash-{path.name}-{uuid.uuid4().hex}"
    os.replace(path, trash)
    shutil.rmtree(trash)
    return {"deleted": True, "step": step}


def _internal_trash_checkpoint(
    run_id: str,
    root: Path,
    live: Path,
    trash: Path,
    expected_sha256: str,
    metric_name: str,
    metric_mode: str,
) -> dict[str, Any]:
    _ensure_columbus_root(run_id, root)
    if live.parent != root / "checkpoints" or trash.parent != live.parent:
        raise MirrorError("checkpoint trash paths are outside exact checkpoints directory")
    step = checkpoint_step_from_name(live.name)
    expected_trash = f".trash-steps_{step}-{expected_sha256[:16]}"
    if trash.name != expected_trash:
        raise MirrorError("checkpoint trash name is not deterministic")
    if live.exists() or live.is_symlink():
        current = validate_checkpoint_tree(
            live, metric_name=metric_name, metric_mode=metric_mode
        )
        if current.manifest_sha256 != expected_sha256:
            raise MirrorError("refusing to trash changed Columbus checkpoint")
        pointer_path = root / "best_checkpoint.json"
        if pointer_path.exists() or pointer_path.is_symlink():
            pointer, _ = validate_pointer_file(
                pointer_path, metric_name=metric_name, metric_mode=metric_mode
            )
            if pointer.best_step == step:
                raise MirrorError("refusing to trash the selected best checkpoint")
        for candidate in (root / "checkpoints").glob("steps_*"):
            if candidate == live:
                continue
            manifest = validate_checkpoint_tree(
                candidate, metric_name=metric_name, metric_mode=metric_mode
            )
            if manifest.selection_best_step == step:
                raise MirrorError(
                    f"refusing to trash steps_{step}; {candidate.name} depends on it"
                )
        if trash.exists() or trash.is_symlink():
            raise MirrorError(f"Columbus deterministic trash conflict: {trash}")
        os.replace(live, trash)
        descriptor = os.open(live.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    elif trash.is_dir() and not trash.is_symlink():
        current = validate_checkpoint_tree(
            trash,
            metric_name=metric_name,
            metric_mode=metric_mode,
            expected_step=step,
        )
        if current.manifest_sha256 != expected_sha256:
            raise MirrorError("Columbus checkpoint trash bytes changed")
    else:
        raise MirrorError("Columbus checkpoint and deterministic trash are both absent")
    return {"trashed": True, "step": step, "path": str(trash)}


def _internal_delete_trash(run_id: str, root: Path, trash: Path) -> dict[str, Any]:
    _ensure_columbus_root(run_id, root)
    if trash.parent != root / "checkpoints" or re.fullmatch(
        r"\.trash-steps_(0|[1-9][0-9]*)-[0-9a-f]{16}", trash.name
    ) is None:
        raise MirrorError("remote trash deletion path is unsafe")
    if trash.exists() or trash.is_symlink():
        info = trash.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise MirrorError("remote checkpoint trash has unsafe type")
        shutil.rmtree(trash)
        descriptor = os.open(trash.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return {"deleted": True, "path": str(trash)}


def verify_restore_index_at_root(
    root: Path | str,
    *,
    run_id: str,
    metric_name: str,
    metric_mode: str,
) -> dict[str, Any]:
    """Full Columbus-only verification rooted exclusively in restore-index.json."""

    base = Path(root)
    _validate_root_directory(base, label="restore root")
    for current, directories, files in os.walk(base, topdown=True, followlinks=False):
        current_path = Path(current)
        for name in directories:
            path = current_path / name
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise MirrorError(f"restore root contains an unsafe directory: {path}")
        for name in files:
            path = current_path / name
            info = path.lstat()
            if (
                stat.S_ISLNK(info.st_mode)
                or not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
            ):
                raise MirrorError(f"restore root contains an unsafe file: {path}")
    index_path = base / "restore-index.json"
    index_record = _regular_record(
        index_path, relative_path="restore-index.json", label="restore index"
    )
    index = _load_json_object(index_path, label="restore index")
    if set(index) != {
        "schema_version",
        "record_kind",
        "run_id",
        "metric_name",
        "metric_mode",
        "generation",
        "checkpoints",
        "best_pointer",
    }:
        raise MirrorError("restore index has incomplete or unexpected fields")
    _require_schema_version(index.get("schema_version"), label="restore index schema")
    if (
        index.get("record_kind") != "restore_index"
        or index.get("run_id") != run_id
        or index.get("metric_name") != metric_name
        or index.get("metric_mode") != metric_mode
    ):
        raise MirrorError("restore index identity/configuration mismatch")
    generation = _strict_int(
        index.get("generation"), label="restore index generation", minimum=1
    )
    generation_path = (
        base / "manifests" / "restore_indexes" / f"generation_{generation:08d}.json"
    )
    generation_record = _regular_record(
        generation_path,
        relative_path=generation_path.name,
        label="generation restore index",
    )
    if generation_record.sha256 != index_record.sha256 or (
        generation_path.read_bytes() != index_path.read_bytes()
    ):
        raise MirrorError("current restore index does not match its immutable generation")
    raw_entries = index.get("checkpoints")
    if not isinstance(raw_entries, list):
        raise MirrorError("restore index checkpoints must be a list")
    manifests: dict[int, TreeManifest] = {}
    evals: dict[int, EvalArtifact] = {}
    reconstructed_evidence: dict[str, RunEvidenceIdentity] = {}
    seen_steps: set[int] = set()
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict) or set(raw_entry) != {
            "step",
            "receipt_relative_path",
            "receipt_sha256",
            "checkpoint_relative_path",
            "checkpoint_manifest_sha256",
            "eval_relative_path",
            "eval_sha256",
            "evidence_relative_path",
            "evidence_manifest_sha256",
            "recovery_closure",
        }:
            raise MirrorError("restore index checkpoint entry shape is invalid")
        step = _strict_int(raw_entry["step"], label="restore checkpoint step")
        if step in seen_steps:
            raise MirrorError("restore index contains duplicate checkpoint steps")
        seen_steps.add(step)
        expected_paths = {
            "receipt_relative_path": f"manifests/receipts/checkpoint-steps_{step}.json",
            "checkpoint_relative_path": f"checkpoints/steps_{step}",
            "eval_relative_path": f"heldout_eval_metrics/step_{step:08d}.json",
            "evidence_relative_path": (
                "evidence/snapshots/"
                f"sha256-{raw_entry['evidence_manifest_sha256']}"
            ),
        }
        if any(raw_entry[key] != value for key, value in expected_paths.items()):
            raise MirrorError(f"restore index paths for steps_{step} are noncanonical")
        for sha_key in (
            "receipt_sha256",
            "checkpoint_manifest_sha256",
            "eval_sha256",
            "evidence_manifest_sha256",
        ):
            if SHA256_RE.fullmatch(str(raw_entry[sha_key])) is None:
                raise MirrorError(f"restore index {sha_key} is invalid")
        receipt_path = base / raw_entry["receipt_relative_path"]
        receipt_record = _regular_record(
            receipt_path, relative_path=receipt_path.name, label="recovery receipt"
        )
        if receipt_record.sha256 != raw_entry["receipt_sha256"]:
            raise MirrorError(f"steps_{step} recovery receipt SHA-256 mismatch")
        receipt = _load_json_object(receipt_path, label="recovery receipt")
        if set(receipt) != {
            "schema_version",
            "record_kind",
            "run_id",
            "step",
            "production_eligible",
            "recovery_closure",
            "checkpoint",
            "eval",
            "evidence",
        }:
            raise MirrorError("recovery receipt shape is invalid")
        _require_schema_version(
            receipt.get("schema_version"), label="recovery receipt schema_version"
        )
        if (
            receipt.get("record_kind") != "recovery_receipt"
            or receipt.get("run_id") != run_id
            or receipt.get("step") != step
            or receipt.get("production_eligible") is not True
            or receipt.get("recovery_closure") != raw_entry["recovery_closure"]
        ):
            raise MirrorError(f"steps_{step} recovery receipt identity mismatch")
        checkpoint_receipt = receipt.get("checkpoint")
        eval_receipt = receipt.get("eval")
        evidence_receipt = receipt.get("evidence")
        if not all(
            isinstance(item, dict)
            for item in (checkpoint_receipt, eval_receipt, evidence_receipt)
        ):
            raise MirrorError("recovery receipt object references are malformed")
        if set(checkpoint_receipt) != {
            "relative_path",
            "manifest_sha256",
            "manifest_record_path",
            "manifest_record_sha256",
        } or set(eval_receipt) != {
            "relative_path",
            "sha256",
            "manifest_record_path",
            "manifest_record_sha256",
        } or set(evidence_receipt) != {
            "relative_path",
            "manifest_sha256",
            "manifest_record_path",
            "manifest_record_sha256",
        }:
            raise MirrorError("recovery receipt reference shapes are invalid")

        expected_manifest_record_paths = {
            "checkpoint": f"manifests/checkpoints/steps_{step}.json",
            "eval": f"manifests/evals/step_{step:08d}.json",
            "evidence": (
                "manifests/evidence/"
                f"sha256-{raw_entry['evidence_manifest_sha256']}.json"
            ),
        }
        for kind, reference in (
            ("checkpoint", checkpoint_receipt),
            ("eval", eval_receipt),
            ("evidence", evidence_receipt),
        ):
            if reference.get("manifest_record_path") != expected_manifest_record_paths[kind]:
                raise MirrorError(f"{kind} manifest record path is noncanonical")

        def read_manifest_record(reference: Mapping[str, Any], kind: str) -> dict[str, Any]:
            path_value = reference.get("manifest_record_path")
            sha_value = reference.get("manifest_record_sha256")
            if not isinstance(path_value, str) or not _safe_relative_path(path_value):
                raise MirrorError(f"{kind} manifest record path is unsafe")
            record_path = base / path_value
            record = _regular_record(
                record_path, relative_path=record_path.name, label=f"{kind} manifest record"
            )
            if record.sha256 != sha_value:
                raise MirrorError(f"{kind} manifest record SHA-256 mismatch")
            payload = _load_json_object(record_path, label=f"{kind} manifest record")
            _require_schema_version(
                payload.get("schema_version"),
                label=f"{kind} manifest record schema_version",
            )
            if payload.get("run_id") != run_id:
                raise MirrorError(f"{kind} manifest record run_id mismatch")
            return payload

        checkpoint_record_payload = read_manifest_record(
            checkpoint_receipt, "checkpoint"
        )
        if set(checkpoint_record_payload) != {
            "schema_version",
            "record_kind",
            "run_id",
            "manifest",
        } or checkpoint_record_payload.get("record_kind") != "checkpoint_manifest":
            raise MirrorError("checkpoint manifest record shape is invalid")
        manifest = TreeManifest.from_dict(checkpoint_record_payload["manifest"])
        if (
            manifest.step != step
            or manifest.checkpoint_schema != "selection_v1"
            or manifest.manifest_sha256 != raw_entry["checkpoint_manifest_sha256"]
            or checkpoint_receipt.get("relative_path")
            != raw_entry["checkpoint_relative_path"]
            or checkpoint_receipt.get("manifest_sha256")
            != manifest.manifest_sha256
        ):
            raise MirrorError(f"steps_{step} checkpoint receipt/manifest mismatch")
        actual_manifest = validate_checkpoint_tree(
            base / raw_entry["checkpoint_relative_path"],
            metric_name=metric_name,
            metric_mode=metric_mode,
        )
        if actual_manifest.manifest_sha256 != manifest.manifest_sha256:
            raise MirrorError(f"steps_{step} checkpoint bytes failed full verification")

        evidence_record_payload = read_manifest_record(evidence_receipt, "evidence")
        if set(evidence_record_payload) != {
            "schema_version",
            "record_kind",
            "run_id",
            "identity",
        } or evidence_record_payload.get("record_kind") != "evidence_manifest":
            raise MirrorError("evidence manifest record shape is invalid")
        evidence = RunEvidenceIdentity.from_dict(evidence_record_payload["identity"])
        if (
            evidence.manifest.manifest_sha256
            != raw_entry["evidence_manifest_sha256"]
            or evidence_receipt.get("relative_path")
            != raw_entry["evidence_relative_path"]
            or evidence_receipt.get("manifest_sha256")
            != evidence.manifest.manifest_sha256
        ):
            raise MirrorError(f"steps_{step} evidence receipt/manifest mismatch")
        actual_evidence = validate_evidence_snapshot(
            base / raw_entry["evidence_relative_path"]
        )
        if actual_evidence.manifest_sha256 != evidence.manifest.manifest_sha256:
            raise MirrorError(f"steps_{step} evidence snapshot failed full verification")
        evidence_digest = evidence.manifest.manifest_sha256
        rebuilt_evidence = reconstructed_evidence.get(evidence_digest)
        if rebuilt_evidence is None:
            rebuilt_evidence = validate_run_evidence_identity(
                base / raw_entry["evidence_relative_path"],
                source_output_dir=evidence.source_output_dir,
            )
            reconstructed_evidence[evidence_digest] = rebuilt_evidence
        if rebuilt_evidence.as_dict() != evidence.as_dict() or rebuilt_evidence.run_id != run_id:
            raise MirrorError(
                f"steps_{step} recorded evidence identity does not match snapshot bytes"
            )

        eval_record_payload = read_manifest_record(eval_receipt, "eval")
        if set(eval_record_payload) != {
            "schema_version",
            "record_kind",
            "run_id",
            "artifact",
        } or eval_record_payload.get("record_kind") != "eval_manifest":
            raise MirrorError("eval manifest record shape is invalid")
        recorded_eval = EvalArtifact.from_dict(eval_record_payload["artifact"])
        actual_eval = validate_eval_artifact(
            base / raw_entry["eval_relative_path"],
            checkpoint_manifest=manifest,
            metric_name=metric_name,
            metric_mode=metric_mode,
            run_id=run_id,
            run_evidence=rebuilt_evidence,
        )
        if (
            recorded_eval != actual_eval
            or actual_eval.sha256 != raw_entry["eval_sha256"]
            or not actual_eval.production_eligible
            or eval_receipt.get("relative_path") != raw_entry["eval_relative_path"]
            or eval_receipt.get("sha256") != actual_eval.sha256
        ):
            raise MirrorError(f"steps_{step} eval failed full verification")
        manifests[step] = manifest
        evals[step] = actual_eval

    ordered_entries = sorted(raw_entries, key=lambda item: item["step"])
    for step, raw_entry in zip(sorted(manifests), ordered_entries):
        expected_closure = validate_recovery_closure(step, manifests)
        if tuple(raw_entry["recovery_closure"]) != expected_closure:
            raise MirrorError(f"steps_{step} recovery closure is incomplete")
    pointer_payload = index.get("best_pointer")
    pointer: SelectionPointer | None = None
    if pointer_payload is not None:
        if not isinstance(pointer_payload, dict) or set(pointer_payload) != {
            "relative_path",
            "sha256",
            "best_step",
            "metric_name",
            "metric_mode",
            "metric_value",
        }:
            raise MirrorError("restore-index best pointer shape is invalid")
        if pointer_payload.get("relative_path") != "best_checkpoint.json":
            raise MirrorError("restore-index best pointer path is noncanonical")
        pointer, pointer_record = validate_pointer_file(
            base / "best_checkpoint.json",
            metric_name=metric_name,
            metric_mode=metric_mode,
            maximum_step=max(manifests) if manifests else None,
        )
        if (
            pointer_record.sha256 != pointer_payload.get("sha256")
            or pointer.best_step != pointer_payload.get("best_step")
            or pointer.metric_value != pointer_payload.get("metric_value")
        ):
            raise MirrorError("restore-index best pointer bytes/identity mismatch")
    authenticate_selection_history(
        manifests, evals, pointer, metric_mode=metric_mode
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "verified": True,
        "run_id": run_id,
        "generation": index["generation"],
        "restore_index_sha256": index_record.sha256,
        "checkpoint_steps": sorted(manifests),
        "best_step": None if pointer is None else pointer.best_step,
    }


def authenticated_restore_files(root: Path | str) -> tuple[str, ...]:
    """Return the exact regular-file closure authenticated by restore-index.json."""

    base = Path(root)
    index_path = base / "restore-index.json"
    index = _load_json_object(index_path, label="restore index copy plan")
    generation = _strict_int(
        index.get("generation"), label="restore index generation", minimum=1
    )
    files: set[str] = {
        "restore-index.json",
        f"manifests/restore_indexes/generation_{generation:08d}.json",
    }
    pointer = index.get("best_pointer")
    if pointer is not None:
        if not isinstance(pointer, dict) or pointer.get("relative_path") != "best_checkpoint.json":
            raise MirrorError("restore copy plan has a noncanonical best pointer")
        files.add("best_checkpoint.json")
    entries = index.get("checkpoints")
    if not isinstance(entries, list):
        raise MirrorError("restore copy plan checkpoints are malformed")
    for entry in entries:
        if not isinstance(entry, dict):
            raise MirrorError("restore copy plan entry is malformed")
        receipt_relative = entry.get("receipt_relative_path")
        if not isinstance(receipt_relative, str) or not _safe_relative_path(receipt_relative):
            raise MirrorError("restore copy plan receipt path is unsafe")
        files.add(receipt_relative)
        receipt = _load_json_object(base / receipt_relative, label="restore copy plan receipt")
        for kind in ("checkpoint", "eval", "evidence"):
            reference = receipt.get(kind)
            if not isinstance(reference, dict):
                raise MirrorError(f"restore copy plan {kind} reference is malformed")
            manifest_path = reference.get("manifest_record_path")
            if not isinstance(manifest_path, str) or not _safe_relative_path(manifest_path):
                raise MirrorError(f"restore copy plan {kind} manifest path is unsafe")
            files.add(manifest_path)
        checkpoint_record = _load_json_object(
            base / receipt["checkpoint"]["manifest_record_path"],
            label="restore copy plan checkpoint manifest",
        )
        checkpoint_manifest = TreeManifest.from_dict(checkpoint_record["manifest"])
        checkpoint_root = entry.get("checkpoint_relative_path")
        if not isinstance(checkpoint_root, str) or not _safe_relative_path(checkpoint_root):
            raise MirrorError("restore copy plan checkpoint root is unsafe")
        files.update(
            f"{checkpoint_root}/{record.path}" for record in checkpoint_manifest.files
        )
        eval_path = entry.get("eval_relative_path")
        if not isinstance(eval_path, str) or not _safe_relative_path(eval_path):
            raise MirrorError("restore copy plan eval path is unsafe")
        files.add(eval_path)
        evidence_record = _load_json_object(
            base / receipt["evidence"]["manifest_record_path"],
            label="restore copy plan evidence manifest",
        )
        evidence = RunEvidenceIdentity.from_dict(evidence_record["identity"])
        evidence_root = entry.get("evidence_relative_path")
        if not isinstance(evidence_root, str) or not _safe_relative_path(evidence_root):
            raise MirrorError("restore copy plan evidence root is unsafe")
        files.update(f"{evidence_root}/{record.path}" for record in evidence.manifest.files)
    ordered = tuple(sorted(files))
    for relative in ordered:
        if not _safe_relative_path(relative):
            raise MirrorError(f"restore copy plan contains unsafe path: {relative!r}")
        _regular_record(
            base / relative,
            relative_path=relative,
            label="authenticated restore file",
        )
    return ordered


def _assert_exact_restore_files(root: Path, expected_files: Sequence[str]) -> None:
    expected = set(expected_files)
    expected_directories: set[str] = set()
    for relative in expected:
        parent = PurePosixPath(relative).parent
        while str(parent) != ".":
            expected_directories.add(str(parent))
            parent = parent.parent
    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        for name in directories:
            path = current_path / name
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise MirrorError(f"restored tree has unsafe directory: {path}")
            actual_directories.add(path.relative_to(root).as_posix())
        for name in files:
            path = current_path / name
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise MirrorError(f"restored tree has unsafe file: {path}")
            actual_files.add(path.relative_to(root).as_posix())
    if actual_files != expected or actual_directories != expected_directories:
        raise MirrorError(
            "restored tree differs from authenticated file plan: "
            f"missing={sorted(expected - actual_files)}, "
            f"extra={sorted(actual_files - expected)}, "
            f"extra_dirs={sorted(actual_directories - expected_directories)}"
        )


def _restore_stage_path(target: Path) -> Path:
    return target.parent / f".incoming-restore-{target.name}-{uuid.uuid4().hex}"


def _publish_restore_stage(stage: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        raise MirrorError(f"restore destination appeared during restore: {target}")
    os.replace(stage, target)
    descriptor = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def restore_from_local_root(
    source_root: Path | str,
    destination: Path | str,
    *,
    run_id: str,
    metric_name: str,
    metric_mode: str,
) -> dict[str, Any]:
    source = Path(source_root)
    target = Path(destination)
    before = verify_restore_index_at_root(
        source, run_id=run_id, metric_name=metric_name, metric_mode=metric_mode
    )
    plan = authenticated_restore_files(source)
    if not target.is_absolute() or any(part in {"", ".", ".."} for part in target.parts[1:]):
        raise MirrorError("restore destination must be a canonical absolute path")
    if target.exists() or target.is_symlink():
        raise MirrorError(f"restore destination must not already exist: {target}")
    _reject_linked_ancestors(target.parent, label="restore destination parent")
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    stage = _restore_stage_path(target)
    try:
        stage.mkdir(mode=0o700)
        for relative in plan:
            source_path = source / relative
            destination_path = stage / relative
            destination_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            shutil.copy2(source_path, destination_path)
        _assert_exact_restore_files(stage, plan)
        after = verify_restore_index_at_root(
            stage, run_id=run_id, metric_name=metric_name, metric_mode=metric_mode
        )
        if before != after:
            raise MirrorError("restored tree verification result differs from Columbus source")
        _publish_restore_stage(stage, target)
        return {**after, "restored_to": str(target)}
    except Exception:
        if stage.exists() and not stage.is_symlink():
            shutil.rmtree(stage)
        raise


def _run_internal(argv: Sequence[str]) -> int | None:
    if not argv or not argv[0].startswith("--internal-"):
        return None
    command, *args = argv
    if command == "--internal-endpoint-identity" and len(args) == 0:
        result: Any = _internal_endpoint_identity()
    elif command == "--internal-resolve-source-root" and len(args) == 1:
        result = resolve_source_root_binding(args[0])
    elif command == "--internal-light-inventory" and len(args) == 1:
        result = lightweight_source_inventory(Path(args[0]))
    elif command == "--internal-preflight-source" and len(args) == 1:
        result = _internal_preflight_source(Path(args[0]))
    elif command == "--internal-preflight-columbus" and len(args) == 2:
        result = _internal_preflight_columbus(args[0], Path(args[1]))
    elif command == "--internal-list-source" and len(args) == 1:
        result = _internal_list_source(Path(args[0]))
    elif command == "--internal-inspect-checkpoint" and len(args) == 3:
        result = validate_checkpoint_tree(
            Path(args[0]), metric_name=args[1], metric_mode=args[2]
        ).as_dict()
    elif command == "--internal-inspect-evidence" and len(args) == 2:
        result = validate_run_evidence_identity(
            Path(args[0]), source_output_dir=args[1]
        ).as_dict()
    elif command == "--internal-file-prefix-sha256" and len(args) == 2:
        result = _file_prefix_sha256(
            Path(args[0]), _strict_int(int(args[1]), label="file prefix length")
        )
    elif command == "--internal-inspect-snapshot" and len(args) == 1:
        result = validate_evidence_snapshot(Path(args[0])).as_dict()
    elif command == "--internal-inspect-eval" and len(args) == 7:
        result = validate_eval_artifact(
            Path(args[0]),
            checkpoint_manifest=_manifest_from_b64(args[1]),
            metric_name=args[2],
            metric_mode=args[3],
            run_id=args[4],
            run_evidence=_evidence_from_b64(args[5]),
            allow_legacy_archive=args[6] == "1",
        ).as_dict()
        if args[6] not in {"0", "1"}:
            raise MirrorError("legacy archive flag must be 0 or 1")
    elif command == "--internal-inspect-baseline-eval" and len(args) == 5:
        result = validate_baseline_eval_artifact(
            Path(args[0]),
            metric_name=args[1],
            metric_mode=args[2],
            run_id=args[3],
            run_evidence=_evidence_from_b64(args[4]),
        ).as_dict()
    elif command == "--internal-inspect-pointer" and len(args) == 4:
        maximum = None if args[3] == "none" else _strict_int(int(args[3]), label="maximum step")
        pointer, record = validate_pointer_file(
            Path(args[0]),
            metric_name=args[1],
            metric_mode=args[2],
            maximum_step=maximum,
        )
        content = Path(args[0]).read_bytes()
        result = {
            "pointer": pointer.as_dict(),
            "record": record.as_dict(),
            "content_b64": base64.b64encode(content).decode("ascii"),
        }
    elif command == "--internal-path-status" and len(args) == 1:
        result = _internal_path_status(Path(args[0]))
    elif command == "--internal-available-bytes" and len(args) == 1:
        path = Path(args[0])
        _validate_root_directory(path, label="free-space target")
        stats = os.statvfs(path)
        result = {"available_bytes": stats.f_bavail * stats.f_frsize}
    elif command == "--internal-acquire-lock" and len(args) == 3:
        result = _internal_acquire_lock(args[0], Path(args[1]), args[2])
    elif command == "--internal-release-lock" and len(args) == 3:
        result = _internal_release_lock(args[0], Path(args[1]), args[2])
    elif command == "--internal-create-stage" and len(args) == 4:
        result = _internal_create_stage(args[0], Path(args[1]), Path(args[2]), args[3])
    elif command == "--internal-put-record" and len(args) == 6:
        result = _internal_put_record(
            args[0], Path(args[1]), args[2], args[3], args[4], args[5]
        )
    elif command == "--internal-promote-tree" and len(args) == 7:
        result = _internal_promote_tree(
            args[0],
            Path(args[1]),
            Path(args[2]),
            Path(args[3]),
            _manifest_from_b64(args[4]),
            args[5],
            args[6],
        )
    elif command == "--internal-promote-eval" and len(args) == 10:
        if args[8] != args[0]:
            raise MirrorError("eval promotion run_id argument mismatch")
        result = _internal_promote_eval(
            args[0],
            Path(args[1]),
            Path(args[2]),
            Path(args[3]),
            _manifest_from_b64(args[4]),
            args[5],
            args[6],
            args[7],
            _evidence_from_b64(args[9]),
        )
    elif command == "--internal-promote-pointer" and len(args) == 9:
        result = _internal_promote_pointer(
            args[0],
            Path(args[1]),
            Path(args[2]),
            Path(args[3]),
            args[4],
            args[5],
            args[6],
            _strict_int(int(args[7]), label="maximum step"),
            args[8],
        )
    elif command == "--internal-delete-checkpoint" and len(args) == 6:
        result = _internal_delete_checkpoint(
            args[0], Path(args[1]), Path(args[2]), args[3], args[4], args[5]
        )
    elif command == "--internal-trash-checkpoint" and len(args) == 7:
        result = _internal_trash_checkpoint(
            args[0],
            Path(args[1]),
            Path(args[2]),
            Path(args[3]),
            args[4],
            args[5],
            args[6],
        )
    elif command == "--internal-delete-trash" and len(args) == 3:
        result = _internal_delete_trash(args[0], Path(args[1]), Path(args[2]))
    elif command == "--internal-verify-index" and len(args) == 5:
        validate_columbus_root(args[0], Path(args[1]))
        result = verify_restore_index_at_root(
            Path(args[1]),
            run_id=args[0],
            metric_name=args[2],
            metric_mode=args[3],
        )
        if args[4] != "full":
            raise MirrorError("restore-index verification mode must be full")
    elif command == "--internal-read-restore-index" and len(args) == 4:
        validate_columbus_root(args[0], Path(args[1]))
        if (Path(args[1]) / ".mirror.lock").exists():
            raise MirrorError("refusing restore while the Columbus mirror lock is held")
        verification = verify_restore_index_at_root(
            Path(args[1]),
            run_id=args[0],
            metric_name=args[2],
            metric_mode=args[3],
        )
        content = (Path(args[1]) / "restore-index.json").read_bytes()
        result = {
            "verification": verification,
            "content_b64": base64.b64encode(content).decode("ascii"),
            "relative_files": list(authenticated_restore_files(Path(args[1]))),
        }
    else:
        raise MirrorError(f"invalid internal command/arity: {command}")
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


def _canonical_absolute_path_arg(value: str) -> Path:
    if not _safe_absolute_posix_path(value):
        raise argparse.ArgumentTypeError(
            "path must be a canonical absolute POSIX path without links such as '..'"
        )
    return Path(value)


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=(
            "preflight",
            "once",
            "watch",
            "status",
            "verify",
            "restore-index",
            "restore",
        ),
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--h100-host")
    parser.add_argument("--source-run-dir")
    parser.add_argument("--local-root", type=_canonical_absolute_path_arg)
    parser.add_argument(
        "--columbus-host",
        default="columbus-8xa100.us-east5-a.yondu-general-workspace",
    )
    parser.add_argument("--columbus-root", type=_canonical_absolute_path_arg)
    parser.add_argument("--state-dir", type=_canonical_absolute_path_arg)
    parser.add_argument("--selection-metric-name")
    parser.add_argument("--selection-metric-mode", choices=("min", "max"))
    parser.add_argument("--retain", type=int, default=DEFAULT_RETAIN)
    parser.add_argument("--disk-reserve-bytes", type=int, default=DEFAULT_DISK_RESERVE_BYTES)
    parser.add_argument("--max-backlog", type=int, default=2)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--stale-heartbeat-seconds", type=int, default=900)
    parser.add_argument("--full-scrub-hours", type=float, default=DEFAULT_FULL_SCRUB_HOURS)
    parser.add_argument("--pending-timeout-seconds", type=int, default=1800)
    parser.add_argument("--full-scrub", action="store_true")
    parser.add_argument("--restore-destination", type=_canonical_absolute_path_arg)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _config_from_args(args: argparse.Namespace) -> MirrorConfig:
    required = {
        "h100_host": args.h100_host,
        "source_run_dir": args.source_run_dir,
        "local_root": args.local_root,
        "columbus_root": args.columbus_root,
        "state_dir": args.state_dir,
        "selection_metric_name": args.selection_metric_name,
        "selection_metric_mode": args.selection_metric_mode,
    }
    missing = sorted(key for key, value in required.items() if value is None)
    if missing:
        raise MirrorError(f"mirror mode lacks required arguments: {missing}")
    return MirrorConfig(
        run_id=args.run_id,
        h100_host=args.h100_host,
        source_run_dir=args.source_run_dir.rstrip("/"),
        local_root=args.local_root,
        columbus_host=args.columbus_host,
        columbus_root=args.columbus_root,
        state_dir=args.state_dir,
        metric_name=args.selection_metric_name,
        metric_mode=args.selection_metric_mode,
        retain=args.retain,
        reserve_bytes=args.disk_reserve_bytes,
        max_backlog=args.max_backlog,
        full_scrub_hours=args.full_scrub_hours,
        pending_timeout_seconds=args.pending_timeout_seconds,
    )


def _status(state_dir: Path, run_id: str, *, stale_seconds: int) -> int:
    state_path = state_dir / run_id / "state.json"
    _secure_regular_file(state_path, label="mirror state")
    state = _load_json_object(state_path, label="mirror state")
    heartbeat = state.get("heartbeat")
    if not isinstance(heartbeat, dict) or not isinstance(heartbeat.get("at"), str):
        raise MirrorError("mirror has no heartbeat")
    parsed = dt.datetime.fromisoformat(heartbeat["at"].replace("Z", "+00:00"))
    age = (dt.datetime.now(dt.timezone.utc) - parsed).total_seconds()
    result = {**heartbeat, "heartbeat_age_seconds": age, "stale": age > stale_seconds}
    print(json.dumps(result, indent=2, sort_keys=True))
    healthy_status = heartbeat.get("status") in {"healthy", "healthy_pending"}
    return 1 if result["stale"] or not healthy_status else 0


def _columbus_public_command(args: argparse.Namespace) -> int:
    required = {
        "columbus_root": args.columbus_root,
        "selection_metric_name": args.selection_metric_name,
        "selection_metric_mode": args.selection_metric_mode,
    }
    missing = sorted(key for key, value in required.items() if value is None)
    if missing:
        raise MirrorError(f"{args.mode} lacks required arguments: {missing}")
    validate_columbus_root(args.run_id, args.columbus_root)
    runner = CommandRunner()
    endpoint = RemoteEndpoint(args.columbus_host, runner)
    identity = resolve_endpoint_identity(endpoint)
    if args.mode == "verify":
        result = endpoint.invoke_json(
            (
                "--internal-verify-index",
                args.run_id,
                str(args.columbus_root),
                args.selection_metric_name,
                args.selection_metric_mode,
                "full",
            )
        )
        print(json.dumps({**result, "columbus_identity": identity}, indent=2, sort_keys=True))
        return 0
    remote = endpoint.invoke_json(
        (
            "--internal-read-restore-index",
            args.run_id,
            str(args.columbus_root),
            args.selection_metric_name,
            args.selection_metric_mode,
        )
    )
    content = base64.b64decode(remote["content_b64"])
    relative_files = remote.get("relative_files")
    if (
        not isinstance(relative_files, list)
        or not relative_files
        or relative_files != sorted(set(relative_files))
        or any(not isinstance(path, str) or not _safe_relative_path(path) for path in relative_files)
    ):
        raise MirrorError("Columbus returned an invalid authenticated restore plan")
    if args.mode == "restore-index":
        print(content.decode("utf-8"), end="")
        return 0
    if args.restore_destination is None or not args.restore_destination.is_absolute():
        raise MirrorError("restore requires an absolute --restore-destination")
    destination = args.restore_destination
    if destination.exists() or destination.is_symlink():
        raise MirrorError(f"restore destination must not already exist: {destination}")
    _reject_linked_ancestors(destination.parent, label="restore destination parent")
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    stage = _restore_stage_path(destination)
    try:
        stage.mkdir(mode=0o700)
        with tempfile.NamedTemporaryFile() as file_list:
            file_list.write(
                b"".join(path.encode("utf-8") + b"\x00" for path in relative_files)
            )
            file_list.flush()
            endpoint.rsync_to_local(
                f"{args.columbus_root}/",
                Path(f"{stage}/"),
                files_from=Path(file_list.name),
                delete=True,
            )
        _assert_exact_restore_files(stage, relative_files)
        local_result = verify_restore_index_at_root(
            stage,
            run_id=args.run_id,
            metric_name=args.selection_metric_name,
            metric_mode=args.selection_metric_mode,
        )
        remote_after = endpoint.invoke_json(
            (
                "--internal-read-restore-index",
                args.run_id,
                str(args.columbus_root),
                args.selection_metric_name,
                args.selection_metric_mode,
            )
        )
        if (
            remote_after.get("content_b64") != remote["content_b64"]
            or remote_after.get("relative_files") != relative_files
        ):
            raise MirrorError("Columbus restore index changed during restore; discard the copy")
        _publish_restore_stage(stage, destination)
    except Exception:
        if stage.exists() and not stage.is_symlink():
            shutil.rmtree(stage)
        raise
    print(
        json.dumps(
            {**local_result, "restored_to": str(destination), "columbus_identity": identity},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    internal = _run_internal(arguments)
    if internal is not None:
        return internal
    args = _argument_parser().parse_args(arguments)
    if args.dry_run and args.mode not in {"preflight", "once"}:
        raise MirrorError("--dry-run is only a compatibility alias for preflight")
    if args.mode == "status":
        if args.state_dir is None:
            raise MirrorError("status requires --state-dir")
        return _status(
            args.state_dir, args.run_id, stale_seconds=args.stale_heartbeat_seconds
        )
    if args.mode in {"verify", "restore-index", "restore"}:
        if args.dry_run:
            raise MirrorError(f"{args.mode} does not accept --dry-run")
        return _columbus_public_command(args)
    config = _config_from_args(args)
    config.validate()
    runner = CommandRunner(dry_run=args.dry_run)
    controller = MirrorController(config, runner)
    if args.mode == "preflight" or args.dry_run:
        print(json.dumps(controller.preflight(), indent=2, sort_keys=True))
        return 0
    if args.mode == "once":
        print(
            json.dumps(
                controller.run_once(force_full_scrub=args.full_scrub),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.poll_seconds < 1 or args.poll_seconds > 60:
        raise MirrorError("watch poll interval must be between 1 and 60 seconds")
    while True:
        try:
            result = controller.run_once(force_full_scrub=args.full_scrub)
            args.full_scrub = False
            print(json.dumps(result, sort_keys=True), flush=True)
        except Exception as exc:
            print(f"mirror cycle failed: {exc}", file=sys.stderr, flush=True)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MirrorError as exc:
        print(f"checkpoint mirror refused: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
