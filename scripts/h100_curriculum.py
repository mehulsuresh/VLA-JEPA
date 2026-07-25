#!/usr/bin/env python3
"""Config-owned, fail-closed orchestration for sequential H100 curricula.

This launcher deliberately does not accept training hyperparameter overrides.
Each stage is an ordinary, independently reviewable H100 training YAML.  The
curriculum YAML owns only ordering, immutable shared-contract assertions, and
the rule used to authenticate the natural-final checkpoint handed to the next
stage.  Offline best-checkpoint selection remains diagnostic and never rewinds
the sequential exposure curriculum.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from omegaconf import DictConfig, OmegaConf
from starVLA.action_representation import (
    load_openpi_realman_union_statistics,
)

try:
    import h100_training
except ModuleNotFoundError:  # Imported as ``scripts.h100_curriculum`` in tests.
    from scripts import h100_training


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CURRICULUM = (
    REPO_ROOT
    / "scripts/config/h100/realman_realsource_intervention_hq_curriculum_v1.yaml"
)
CURRICULUM_ID_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{2,127}")
RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{2,191}")
STAGE_ID_RE = re.compile(r"[a-z0-9][a-z0-9_-]{1,63}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
APPROVED_ROLE_SEQUENCES = {
    ("pretrain", "adapt", "finetune"),
}
NATURAL_FINAL_HANDOFF = "natural_final"
PRODUCTION_CURRICULUM = "production_curriculum"
CHECKPOINT_HANDOFF_SMOKE = "checkpoint_handoff_smoke"
HANDOFF_SMOKE_STAGE_SCHEMA = "realman-checkpoint-handoff-smoke-stage-v1"


class CurriculumError(RuntimeError):
    """Raised when a curriculum or a stage handoff is not production-valid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _state_payload_sha256(payload: Mapping[str, Any]) -> str:
    unsigned = copy.deepcopy(dict(payload))
    unsigned.pop("state_payload_sha256", None)
    serialized = json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _write_curriculum_state(path: Path, state: dict[str, Any]) -> None:
    state["state_payload_sha256"] = _state_payload_sha256(state)
    _atomic_json(path, state)


def _load_curriculum_state(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise CurriculumError(
            f"curriculum resume state must be a regular file: {path}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CurriculumError(
            f"curriculum resume state is unreadable: {path}"
        ) from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 3:
        raise CurriculumError(
            "curriculum resume state must use schema_version=3"
        )
    expected_sha = payload.get("state_payload_sha256")
    if (
        not isinstance(expected_sha, str)
        or SHA256_RE.fullmatch(expected_sha) is None
        or _state_payload_sha256(payload) != expected_sha
    ):
        raise CurriculumError(
            "curriculum resume state self-authentication failed"
        )
    return payload


def _plain(value: Any) -> Any:
    if isinstance(value, (DictConfig,)):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _require_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    value = _plain(value)
    if not isinstance(value, Mapping):
        raise CurriculumError(f"{field} must be an object")
    return value


def _require_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CurriculumError(f"{field} must be a non-empty string")
    return value.strip()


def _require_int(value: Any, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CurriculumError(
            f"{field} must be an integer greater than or equal to {minimum}"
        )
    return int(value)


def _resolve_repo_file(value: Any, *, field: str) -> Path:
    raw = _require_string(value, field=field)
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve(strict=True)
    if not path.is_relative_to(REPO_ROOT):
        raise CurriculumError(f"{field} must be inside the repository: {path}")
    if path.is_symlink() or not path.is_file():
        raise CurriculumError(f"{field} must be a regular non-symlink file: {path}")
    return path


def _resolve_absolute_file(value: Any, *, field: str) -> Path:
    raw = _require_string(value, field=field)
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve(strict=True)
    if path.is_symlink() or not path.is_file():
        raise CurriculumError(f"{field} must be a regular non-symlink file: {path}")
    return path


def _validate_shared_statistics_artifact(
    *,
    statistics_path: Path,
    expected_statistics_sha256: str,
    expected_holdout_sha256: str,
    expected_contract_sha256: str,
    expected_normalization: str,
) -> Mapping[str, Any]:
    """Authenticate the shared statistics and its exact leakage holdout."""

    try:
        artifact = load_openpi_realman_union_statistics(
            statistics_path,
            expected_statistics_sha256,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise CurriculumError(
            f"shared normalization artifact is invalid: {exc}"
        ) from exc
    if artifact["contract_sha256"] != expected_contract_sha256:
        raise CurriculumError(
            "shared normalization artifact action contract does not match "
            "the curriculum"
        )
    if artifact["normalization"] != expected_normalization:
        raise CurriculumError(
            "shared normalization artifact normalization mode does not match "
            "the curriculum"
        )
    artifact_holdout_sha256 = artifact["population"][
        "holdout_manifest_sha256"
    ]
    if artifact_holdout_sha256 != expected_holdout_sha256:
        raise CurriculumError(
            "shared normalization artifact was built against a different "
            "statistics holdout manifest: "
            f"{artifact_holdout_sha256} != {expected_holdout_sha256}"
        )
    return artifact


def _nested(payload: Mapping[str, Any], dotted: str) -> Any:
    current: Any = payload
    for component in dotted.split("."):
        if not isinstance(current, Mapping) or component not in current:
            raise CurriculumError(f"stage config is missing {dotted}")
        current = current[component]
    return current


def _optional_nested(
    payload: Mapping[str, Any], dotted: str, default: Any = None
) -> Any:
    current: Any = payload
    for component in dotted.split("."):
        if not isinstance(current, Mapping) or component not in current:
            return default
        current = current[component]
    return current


def _load_curriculum(path: Path) -> tuple[DictConfig, Mapping[str, Any]]:
    path = path.expanduser().resolve(strict=True)
    if path.is_symlink() or not path.is_file():
        raise CurriculumError(
            f"curriculum config must be a regular non-symlink file: {path}"
        )
    try:
        cfg = OmegaConf.load(path)
        payload = _plain(cfg)
    except Exception as exc:
        raise CurriculumError(f"unable to load curriculum config {path}: {exc}") from exc
    if not isinstance(cfg, DictConfig) or not isinstance(payload, Mapping):
        raise CurriculumError("curriculum config root must be an object")
    return cfg, payload


def _stage_shared_contract(
    stage_payload: Mapping[str, Any],
) -> dict[str, Any]:
    data = _require_mapping(
        _nested(stage_payload, "datasets.vla_data"),
        field="datasets.vla_data",
    )
    return {
        "state_dim": _nested(stage_payload, "framework.action_model.state_dim"),
        "action_dim": _nested(stage_payload, "framework.action_model.action_dim"),
        "action_horizon": _nested(
            stage_payload, "framework.action_model.action_horizon"
        ),
        "action_type": data.get("action_type"),
        "action_delta_anchor": data.get("action_delta_anchor"),
        "gripper_action_type": data.get("gripper_action_type"),
        "state_action_normalization": data.get(
            "state_action_normalization",
            data.get("sidecar_normalization"),
        ),
        "normalization_statistics_artifact": data.get(
            "normalization_statistics_artifact"
        ),
        "normalization_statistics_artifact_sha256": data.get(
            "normalization_statistics_artifact_sha256"
        ),
        "action_representation_contract_sha256": data.get(
            "action_representation_contract_sha256"
        ),
    }


def _model_architecture_contract(
    stage_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the fully resolved model-construction contract.

    The entire framework tree is intentional.  Comparing only policy
    dimensions misses incompatible Qwen, V-JEPA, MoGe, attention, RTC, and
    inference-step settings whose checkpoints cannot be continued strictly.
    """

    framework = _require_mapping(
        _nested(stage_payload, "framework"),
        field="framework",
    )
    contract = {
        "schema": "starvla-resolved-framework-architecture-v1",
        "framework": copy.deepcopy(dict(framework)),
    }
    try:
        json.dumps(
            contract,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CurriculumError(
            f"framework architecture is not deterministic JSON: {exc}"
        ) from exc
    return contract


def _model_architecture_sha256(
    stage_payload: Mapping[str, Any],
) -> str:
    serialized = json.dumps(
        _model_architecture_contract(stage_payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _require_identical_model_architectures(
    stages: Sequence[Mapping[str, Any]],
) -> str:
    """Fail closed unless every reviewed stage has one architecture SHA."""

    if not stages:
        raise CurriculumError(
            "curriculum must contain at least one stage architecture"
        )
    architecture_sha256 = _require_string(
        stages[0].get("model_architecture_sha256"),
        field="stages[0].model_architecture_sha256",
    )
    mismatches: dict[str, str] = {}
    for index, stage in enumerate(stages[1:], start=1):
        stage_id = _require_string(
            stage.get("id"),
            field=f"stages[{index}].id",
        )
        stage_sha256 = _require_string(
            stage.get("model_architecture_sha256"),
            field=f"stages[{index}].model_architecture_sha256",
        )
        if stage_sha256 != architecture_sha256:
            mismatches[stage_id] = stage_sha256
    if mismatches:
        raise CurriculumError(
            "curriculum stages do not instantiate the same fully resolved "
            "framework/model architecture; Stage A checkpoints cannot be "
            "strictly continued by later stages: "
            f"stage1={architecture_sha256}, mismatches={mismatches}"
        )
    return architecture_sha256


def _require_approved_role_sequence(
    stages: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    role_sequence = tuple(
        _require_string(stage.get("role"), field=f"stages[{index}].role")
        for index, stage in enumerate(stages)
    )
    if role_sequence not in APPROVED_ROLE_SEQUENCES:
        raise CurriculumError(
            "stage roles must be exactly the approved production sequence: "
            "pretrain → adapt → finetune"
        )
    return role_sequence


def _require_comparable_training_contract(
    stages: Sequence[Mapping[str, Any]],
) -> tuple[str, int, int]:
    """Authenticate model, seed, and global batch across sequential stages."""

    architecture_sha256 = _require_identical_model_architectures(stages)
    experiment_seed = _require_int(
        stages[0].get("seed"),
        field="stages[0].seed",
        minimum=0,
    )
    global_batch_size = _require_int(
        stages[0].get("global_batch_size"),
        field="stages[0].global_batch_size",
        minimum=1,
    )
    for index, stage in enumerate(stages[1:], start=1):
        stage_id = _require_string(
            stage.get("id"),
            field=f"stages[{index}].id",
        )
        seed = _require_int(
            stage.get("seed"),
            field=f"stages[{index}].seed",
            minimum=0,
        )
        if seed != experiment_seed:
            raise CurriculumError(
                f"stage {stage_id} seed differs from the first stage: "
                f"{seed} != {experiment_seed}"
            )
        global_batch = _require_int(
            stage.get("global_batch_size"),
            field=f"stages[{index}].global_batch_size",
            minimum=1,
        )
        if global_batch != global_batch_size:
            raise CurriculumError(
                f"stage {stage_id} global batch differs from the first "
                f"stage: {global_batch} != {global_batch_size}"
            )
    return architecture_sha256, experiment_seed, global_batch_size


def _stage_local_evaluation_manifest(
    stage_payload: Mapping[str, Any],
) -> str:
    """Return the stage-local split artifact.

    The shared union-statistics holdout and the loader-specific evaluation
    split are intentionally different concepts.  The former prevents
    cross-stage normalization leakage.  The latter can use backend-specific
    identities while remaining immutable and hash-bound.
    """

    data = _require_mapping(
        _nested(stage_payload, "datasets.vla_data"),
        field="datasets.vla_data",
    )
    eval_data = _optional_nested(stage_payload, "datasets.eval_vla_data")
    if eval_data is None:
        eval_data = data
    eval_data = _require_mapping(eval_data, field="datasets.eval_vla_data")
    # Canonical configs deliberately declare ``episode_split_manifest: null``
    # alongside their real ``canonical_eval_manifest``.  ``dict.get`` with a
    # default does not fall back when the key exists with a null value, so
    # select the first populated backend-specific binding explicitly.
    value = eval_data.get("episode_split_manifest")
    if value is None:
        value = eval_data.get("canonical_eval_manifest")
    return _require_string(value, field="stage local evaluation manifest")


def _validate_exhaustive_view(
    *,
    view_path: Path,
    expected_manifest_sha256: str,
    stage_id: str,
) -> tuple[Mapping[str, Any], int]:
    """Validate the immutable population that defines one logical epoch."""

    actual_view_sha = _sha256(view_path)
    if actual_view_sha != expected_manifest_sha256:
        raise CurriculumError(
            f"{stage_id} frozen train-view SHA mismatch: "
            f"{actual_view_sha} != {expected_manifest_sha256}"
        )
    view = json.loads(view_path.read_text(encoding="utf-8"))
    if not isinstance(view, Mapping):
        raise CurriculumError(f"{stage_id} frozen train view must be an object")
    epoch_contract = _require_mapping(
        view.get("epoch_contract"),
        field=f"{stage_id}.frozen_view.epoch_contract",
    )
    required_epoch_contract = {
        "mode": "all_exhaustive",
        "epoch_passes": 1,
        "replacement": False,
        "drop_last": False,
        "shuffle": "deterministic_bijection_per_epoch",
        "ddp_tail": "duplicated_padding_reported_separately",
    }
    mismatches = {
        key: {
            "view": epoch_contract.get(key),
            "required": expected,
        }
        for key, expected in required_epoch_contract.items()
        if epoch_contract.get(key) != expected
    }
    if mismatches:
        raise CurriculumError(
            f"{stage_id} frozen view is not a one-pass exhaustive epoch: "
            f"{mismatches}"
        )
    rows = _require_mapping(
        view.get("rows"),
        field=f"{stage_id}.frozen_view.rows",
    )
    eligible_windows = _require_int(
        rows.get("row_count"),
        field=f"{stage_id}.frozen_view.rows.row_count",
        minimum=1,
    )
    unique_samples = _require_int(
        rows.get("unique_sample_count"),
        field=f"{stage_id}.frozen_view.rows.unique_sample_count",
        minimum=1,
    )
    if unique_samples != eligible_windows:
        raise CurriculumError(
            f"{stage_id} frozen view contains repeated sample identities "
            f"({unique_samples} unique for {eligible_windows} rows)"
        )
    raw_ledger = _require_string(
        rows.get("path"),
        field=f"{stage_id}.frozen_view.rows.path",
    )
    ledger_path = (view_path.parent / raw_ledger).resolve(strict=True)
    if not ledger_path.is_relative_to(view_path.parent.resolve()):
        raise CurriculumError(f"{stage_id} frozen ledger escapes its directory")
    if ledger_path.is_symlink() or not ledger_path.is_file():
        raise CurriculumError(
            f"{stage_id} frozen ledger must be a regular file: {ledger_path}"
        )
    expected_ledger_sha = _require_string(
        rows.get("sha256"),
        field=f"{stage_id}.frozen_view.rows.sha256",
    )
    if SHA256_RE.fullmatch(expected_ledger_sha) is None:
        raise CurriculumError(f"{stage_id} frozen ledger SHA is malformed")
    actual_ledger_sha = _sha256(ledger_path)
    if actual_ledger_sha != expected_ledger_sha:
        raise CurriculumError(
            f"{stage_id} frozen ledger SHA mismatch: "
            f"{actual_ledger_sha} != {expected_ledger_sha}"
        )
    return view, eligible_windows


def _canonical_json_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _validate_canonical_view_eval_holdout_binding(
    *,
    view_path: Path,
    view: Mapping[str, Any],
    evaluation_path: Path,
    evaluation_sha256: str,
    eligible_windows: int,
    stage_id: str,
) -> Mapping[str, Any]:
    """Prove the frozen canonical population excludes its exact eval split."""

    try:
        evaluation = json.loads(
            evaluation_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise CurriculumError(
            f"{stage_id} canonical evaluation manifest is invalid: {exc}"
        ) from exc
    if (
        not isinstance(evaluation, Mapping)
        or evaluation.get("schema_version") != 1
        or evaluation.get("purpose") != "heldout"
    ):
        raise CurriculumError(
            f"{stage_id} canonical evaluation manifest must be a heldout "
            "schema_version=1 artifact"
        )
    windows = evaluation.get("windows")
    if not isinstance(windows, list) or not windows:
        raise CurriculumError(
            f"{stage_id} canonical evaluation manifest contains no windows"
        )
    eval_identities: set[tuple[str, str, str, str, int]] = set()
    eval_windows: set[tuple[str, str, str, str, int, int]] = set()
    for index, raw_window in enumerate(windows):
        if not isinstance(raw_window, Mapping):
            raise CurriculumError(
                f"{stage_id} canonical eval window {index} is malformed"
            )
        strings = tuple(
            raw_window.get(field)
            for field in ("dataset_id", "sid", "revision", "data_file")
        )
        episode_index = raw_window.get("episode_index")
        base_index = raw_window.get("base_index")
        if (
            any(
                not isinstance(value, str) or not value
                for value in strings
            )
            or isinstance(episode_index, bool)
            or not isinstance(episode_index, int)
            or episode_index < 0
            or isinstance(base_index, bool)
            or not isinstance(base_index, int)
            or base_index < 0
        ):
            raise CurriculumError(
                f"{stage_id} canonical eval window {index} has an invalid "
                "episode identity"
            )
        identity = (*strings, episode_index)
        eval_identities.add(identity)
        eval_windows.add((*identity, base_index))
    if len(eval_windows) != len(windows):
        raise CurriculumError(
            f"{stage_id} canonical evaluation manifest repeats windows"
        )

    selection = _require_mapping(
        view.get("selection"),
        field=f"{stage_id}.frozen_view.selection",
    )
    binding = _require_mapping(
        selection.get("evaluation_holdout"),
        field=f"{stage_id}.frozen_view.selection.evaluation_holdout",
    )
    if (
        binding.get("schema")
        != "realsource-canonical-eval-holdout-binding-v1"
        or binding.get("manifest_sha256") != evaluation_sha256
        or binding.get("source_manifest_sha256")
        != evaluation.get("source_manifest_sha256")
        or binding.get("window_count") != len(eval_windows)
        or binding.get("episode_count") != len(eval_identities)
        or binding.get("episode_identities_sha256")
        != _canonical_json_sha256(
            [list(identity) for identity in sorted(eval_identities)]
        )
    ):
        raise CurriculumError(
            f"{stage_id} frozen view is not bound to the exact canonical "
            "evaluation episode population"
        )
    binding_sha256 = binding.get("sha256")
    unsigned_binding = dict(binding)
    unsigned_binding.pop("sha256", None)
    if (
        not isinstance(binding_sha256, str)
        or SHA256_RE.fullmatch(binding_sha256) is None
        or binding_sha256 != _canonical_json_sha256(unsigned_binding)
    ):
        raise CurriculumError(
            f"{stage_id} frozen view evaluation-holdout binding hash is invalid"
        )
    if binding.get("copy_detection") != [
        "episode_identity",
        "episode_lineage_id",
        "episode_content_id",
    ]:
        raise CurriculumError(
            f"{stage_id} frozen view does not exclude content-identical "
            "heldout copies"
        )

    exclusions = _require_mapping(
        view.get("holdout_exclusions"),
        field=f"{stage_id}.frozen_view.holdout_exclusions",
    )
    unsigned_exclusions = dict(exclusions)
    exclusions_sha256 = unsigned_exclusions.pop("sha256", None)
    lineages = exclusions.get("lineage_ids")
    contents = exclusions.get("content_ids")
    if (
        exclusions.get("schema")
        != "vla-dataset-view-holdout-exclusions-v1"
        or exclusions.get("source_id")
        != f"canonical_eval_manifest:{evaluation_sha256}"
        or not isinstance(lineages, list)
        or not lineages
        or not isinstance(contents, list)
        or not contents
        or any(
            not isinstance(value, str)
            or SHA256_RE.fullmatch(value) is None
            for value in [*lineages, *contents]
        )
        or exclusions_sha256 != _canonical_json_sha256(unsigned_exclusions)
    ):
        raise CurriculumError(
            f"{stage_id} frozen view has invalid or empty dual-identity "
            "holdout exclusions"
        )

    rows = _require_mapping(
        view.get("rows"),
        field=f"{stage_id}.frozen_view.rows",
    )
    ledger_path = (view_path.parent / str(rows["path"])).resolve()
    encoding = rows.get("encoding", "expanded_rows_v1")
    observed_records = 0
    observed_logical_rows = 0
    overlap: list[tuple[str, str, str, str, int]] = []
    lineage_set = set(lineages)
    content_set = set(contents)
    with ledger_path.open("r", encoding="utf-8") as handle:
        for ordinal, line in enumerate(handle):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CurriculumError(
                    f"{stage_id} frozen ledger row {ordinal} is invalid: {exc}"
                ) from exc
            identity = tuple(
                row.get(field)
                for field in (
                    "dataset_id",
                    "sid",
                    "revision",
                    "data_file",
                    "episode_index",
                )
            )
            if identity in eval_identities:
                overlap.append(identity)
            if row.get("episode_lineage_id") in lineage_set:
                overlap.append(identity)
            if row.get("episode_content_id") in content_set:
                overlap.append(identity)
            observed_records += 1
            if encoding == "episode_ranges_v1":
                count = row.get("sample_count")
                if (
                    isinstance(count, bool)
                    or not isinstance(count, int)
                    or count <= 0
                ):
                    raise CurriculumError(
                        f"{stage_id} frozen range row {ordinal} has an "
                        "invalid sample_count"
                    )
                observed_logical_rows += count
            else:
                observed_logical_rows += 1
    if overlap:
        raise CurriculumError(
            f"{stage_id} frozen train view leaks canonical eval episodes: "
            f"{sorted(set(overlap))[:5]}"
        )
    expected_records = rows.get("record_count", eligible_windows)
    if (
        observed_records != expected_records
        or observed_logical_rows != eligible_windows
    ):
        raise CurriculumError(
            f"{stage_id} frozen ledger counts do not match its reviewed "
            f"population: records={observed_records}/{expected_records}, "
            f"rows={observed_logical_rows}/{eligible_windows}"
        )
    return binding


def _validate_action_supervision_audit(
    *,
    view: Mapping[str, Any],
    data_cfg: Mapping[str, Any],
    stage_id: str,
) -> Mapping[str, Any]:
    """Require provenance-backed action labels for intervention training."""

    suffix = (
        "`valid_state` alone is insufficient: it is state-quality metadata "
        "unless a reviewed, SHA-bound dataset-specific action-label semantics "
        "contract proves otherwise."
    )
    raw_audit = _plain(view.get("action_supervision_audit"))
    if not isinstance(raw_audit, Mapping):
        raise CurriculumError(
            f"{stage_id} frozen view lacks an action-supervision audit. "
            f"{suffix}"
        )
    audit = raw_audit
    if audit.get("schema") != "realman-action-supervision-audit-v1":
        raise CurriculumError(
            f"{stage_id} has an unknown action-supervision audit schema. "
            f"{suffix}"
        )
    if audit.get("status") != "verified":
        reasons = audit.get("reasons")
        raise CurriculumError(
            f"{stage_id} action supervision is not verified: {reasons!r}. "
            f"{suffix}"
        )
    mode = audit.get("verification_mode")
    expected_label_key: str
    if mode == "explicit_valid_action_and_expert_owner":
        explicit = _require_mapping(
            audit.get("explicit_action_labels"),
            field=f"{stage_id}.action_supervision_audit.explicit_action_labels",
        )
        expected_label_key = _require_string(
            explicit.get("validity_column"),
            field=f"{stage_id}.explicit_action_labels.validity_column",
        )
        if expected_label_key != "valid_action":
            raise CurriculumError(
                f"{stage_id} explicit action supervision must use "
                f"valid_action. {suffix}"
            )
        if _require_int(
            explicit.get(
                "invalid_state_with_explicit_expert_action_row_count"
            ),
            field=f"{stage_id}.explicit recovery action count",
            minimum=1,
        ) < 1:
            raise CurriculumError(
                f"{stage_id} has no explicit expert recovery action. {suffix}"
            )
    elif mode == "reviewed_valid_state_action_semantics_contract":
        expected_label_key = "valid_state"
        contract = _require_mapping(
            audit.get("action_label_semantics_contract"),
            field=(
                f"{stage_id}.action_supervision_audit."
                "action_label_semantics_contract"
            ),
        )
        contract_sha = _require_string(
            contract.get("sha256"),
            field=f"{stage_id}.action_label_semantics_contract.sha256",
        )
        if SHA256_RE.fullmatch(contract_sha) is None:
            raise CurriculumError(
                f"{stage_id} action-label contract SHA is malformed. {suffix}"
            )
    else:
        raise CurriculumError(
            f"{stage_id} action-supervision verification mode {mode!r} is "
            f"not accepted. {suffix}"
        )

    recovery = _require_mapping(
        audit.get("recovery_from_invalid_state_supervision"),
        field=(
            f"{stage_id}.action_supervision_audit."
            "recovery_from_invalid_state_supervision"
        ),
    )
    if recovery.get("status") != "verified":
        raise CurriculumError(
            f"{stage_id} does not prove recovery-from-invalid-state action "
            f"supervision. {suffix}"
        )
    if mode == "reviewed_valid_state_action_semantics_contract":
        anchors = _require_int(
            recovery.get("recovery_anchor_window_count"),
            field=f"{stage_id}.recovery_anchor_window_count",
            minimum=1,
        )
        nonzero = _require_int(
            recovery.get(
                "recovery_anchor_with_nonzero_action_mask_count"
            ),
            field=(
                f"{stage_id}."
                "recovery_anchor_with_nonzero_action_mask_count"
            ),
            minimum=1,
        )
        minimum_coverage = _require_int(
            recovery.get(
                "minimum_supervised_action_timesteps_per_anchor"
            ),
            field=(
                f"{stage_id}."
                "minimum_supervised_action_timesteps_per_anchor"
            ),
            minimum=1,
        )
        if nonzero != anchors or minimum_coverage < 1:
            raise CurriculumError(
                f"{stage_id} recovery-anchored windows are not all "
                f"action-supervised. {suffix}"
            )

    if data_cfg.get("use_action_validity_prefix_mask") is not True:
        raise CurriculumError(
            f"{stage_id} must enable the audited action-validity mask. "
            f"{suffix}"
        )
    if data_cfg.get("action_validity_fail_closed") is not True:
        raise CurriculumError(
            f"{stage_id} must set action_validity_fail_closed=true so "
            "missing, malformed, or non-finite action labels cannot silently "
            f"become supervised actions. {suffix}"
        )
    if (
        data_cfg.get("frozen_train_view_require_data_shard_hashes")
        is not True
    ):
        raise CurriculumError(
            f"{stage_id} must require byte hashes for every selected "
            f"LeRobot data shard. {suffix}"
        )
    sources = view.get("sources")
    if not isinstance(sources, list) or len(sources) != 1:
        raise CurriculumError(
            f"{stage_id} action-supervision view must contain exactly one "
            f"source. {suffix}"
        )
    selected_data_shards = sources[0].get("selected_data_shards")
    if (
        not isinstance(selected_data_shards, list)
        or not selected_data_shards
    ):
        raise CurriculumError(
            f"{stage_id} action-supervision view does not bind the actual "
            f"selected parquet-shard bytes. {suffix}"
        )
    if data_cfg.get("action_validity_label_key") != expected_label_key:
        raise CurriculumError(
            f"{stage_id} loader action_validity_label_key does not match the "
            f"audited {expected_label_key!r} column. {suffix}"
        )
    if data_cfg.get("action_validity_positive_is_valid") is not True:
        raise CurriculumError(
            f"{stage_id} must interpret the audited positive label as valid. "
            f"{suffix}"
        )
    audited_invalid_run = _require_int(
        recovery.get("invalid_run_length"),
        field=f"{stage_id}.recovery invalid_run_length",
        minimum=1,
    )
    configured_invalid_run = _require_int(
        data_cfg.get("action_validity_invalid_run_length"),
        field=f"{stage_id}.action_validity_invalid_run_length",
        minimum=1,
    )
    if configured_invalid_run != audited_invalid_run:
        raise CurriculumError(
            f"{stage_id} loader invalid-run length {configured_invalid_run} "
            f"does not match audited mask coverage {audited_invalid_run}. "
            f"{suffix}"
        )
    return audit


def _validate_source_stage(
    *,
    curriculum_path: Path,
    stage_index: int,
    stage: Mapping[str, Any],
    shared: Mapping[str, Any],
    workflow_kind: str,
    validate_artifacts: bool,
) -> dict[str, Any]:
    prefix = f"stages[{stage_index}]"
    stage_id = _require_string(stage.get("id"), field=f"{prefix}.id")
    if STAGE_ID_RE.fullmatch(stage_id) is None:
        raise CurriculumError(f"{prefix}.id is invalid: {stage_id!r}")
    role = _require_string(stage.get("role"), field=f"{prefix}.role")
    if role not in {"pretrain", "adapt", "finetune"}:
        raise CurriculumError(
            f"{prefix}.role must be pretrain, adapt, or finetune"
        )
    require_verified_action_supervision = stage.get(
        "require_verified_action_supervision"
    )
    if role == "adapt" and require_verified_action_supervision is not True:
        raise CurriculumError(
            f"{stage_id} adapt stage must set "
            "require_verified_action_supervision=true; `valid_state` alone "
            "is insufficient action-ownership provenance"
        )
    initialization = _require_string(
        stage.get("initialization"), field=f"{prefix}.initialization"
    )
    expected_initialization = (
        "upstream" if stage_index == 0 else "previous_stage_final"
    )
    if initialization != expected_initialization:
        raise CurriculumError(
            f"{prefix}.initialization must be {expected_initialization!r}"
        )
    handoff_checkpoint_policy = _require_string(
        stage.get("handoff_checkpoint"),
        field=f"{prefix}.handoff_checkpoint",
    )
    if handoff_checkpoint_policy != NATURAL_FINAL_HANDOFF:
        raise CurriculumError(
            f"{prefix}.handoff_checkpoint must be "
            f"{NATURAL_FINAL_HANDOFF!r}; sequential exposure must not rewind "
            "to an earlier offline-selected checkpoint"
        )
    expected_epochs = _require_int(
        stage.get("expected_full_dataset_epochs"),
        field=f"{prefix}.expected_full_dataset_epochs",
        minimum=1,
    )
    config_path = _resolve_repo_file(stage.get("config"), field=f"{prefix}.config")
    try:
        stage_cfg, stage_payload = h100_training._load_config(config_path)
        plan = h100_training.resolve_plan(
            config_path, validate_artifacts=validate_artifacts
        )
    except (h100_training.PlanError, OSError) as exc:
        raise CurriculumError(f"{stage_id} config is invalid: {exc}") from exc
    del stage_cfg

    epochs = _require_int(
        _nested(stage_payload, "trainer.epochs"),
        field=f"{stage_id}.trainer.epochs",
        minimum=1,
    )
    if epochs != expected_epochs:
        raise CurriculumError(
            f"{stage_id} expected_full_dataset_epochs={expected_epochs} but "
            f"trainer.epochs={epochs}"
        )
    if _nested(stage_payload, "trainer.max_train_steps") != "auto":
        raise CurriculumError(
            f"{stage_id} must set trainer.max_train_steps=auto so an epoch is "
            "the complete frozen train view"
        )
    data_cfg = _require_mapping(
        _nested(stage_payload, "datasets.vla_data"),
        field=f"{stage_id}.datasets.vla_data",
    )
    if data_cfg.get("epoch_sampling_strategy") != "all_sources_exhaustive":
        raise CurriculumError(
            f"{stage_id} must use all_sources_exhaustive"
        )
    if data_cfg.get("fail_on_sample_error") is not True:
        raise CurriculumError(f"{stage_id} must set fail_on_sample_error=true")
    if data_cfg.get("drop_last") is not False:
        raise CurriculumError(f"{stage_id} must set drop_last=false")
    if data_cfg.get("shuffle") is not False:
        raise CurriculumError(f"{stage_id} must set shuffle=false")
    if _optional_nested(stage_payload, "trainer.pretrained_checkpoint") is not None:
        raise CurriculumError(
            f"{stage_id} source config must leave trainer.pretrained_checkpoint "
            "null; the authenticated handoff materializes it"
        )
    if (
        _optional_nested(
            stage_payload, "trainer.pretrained_checkpoint_sha256"
        )
        is not None
    ):
        raise CurriculumError(
            f"{stage_id} source config must leave "
            "trainer.pretrained_checkpoint_sha256 null; the authenticated "
            "handoff materializes it"
        )
    if _nested(stage_payload, "trainer.reload_modules") is not None:
        raise CurriculumError(
            f"{stage_id} source config must set trainer.reload_modules=null; "
            "a sequential handoff always loads the complete authenticated "
            "model and must not silently become a partial-module reload"
        )
    if _nested(stage_payload, "trainer.is_resume") is not False:
        raise CurriculumError(f"{stage_id} source config must be a fresh train")
    if _nested(stage_payload, "trainer.resume_from_checkpoint") is not None:
        raise CurriculumError(
            f"{stage_id} source config must set resume_from_checkpoint=null"
        )
    eval_before_train = _nested(
        stage_payload, "trainer.eval_before_train"
    )
    expected_eval_before_train = (
        workflow_kind != CHECKPOINT_HANDOFF_SMOKE
    )
    if eval_before_train is not expected_eval_before_train:
        if expected_eval_before_train:
            raise CurriculumError(
                f"{stage_id} must evaluate the frozen holdout before training"
            )
        raise CurriculumError(
            f"{stage_id} checkpoint-handoff smoke must set "
            "trainer.eval_before_train=false; this workflow validates only "
            "the natural-final A→B→C checkpoint transfer"
        )
    if _nested(stage_payload, "trainer.allow_training_stream_eval") is not False:
        raise CurriculumError(
            f"{stage_id} must not evaluate or select on its training stream"
        )
    if _nested(stage_payload, "trainer.save_final_model") is not True:
        raise CurriculumError(f"{stage_id} must save its natural-final model")
    if (
        _optional_nested(
            stage_payload,
            "trainer.checkpoint_eval_milestones_only",
            False,
        )
        is not True
    ):
        raise CurriculumError(
            f"{stage_id} must set checkpoint_eval_milestones_only=true so "
            "inherited periodic intervals cannot add unreviewed boundaries"
        )
    if _nested(stage_payload, "trainer.checkpoint_max_to_keep") != 0:
        raise CurriculumError(
            f"{stage_id} must set checkpoint_max_to_keep=0 so intended "
            "milestones and the selected best cannot be pruned"
        )
    if _nested(stage_payload, "trainer.save_interval") != _nested(
        stage_payload, "trainer.eval_interval"
    ):
        raise CurriculumError(
            f"{stage_id} save_interval and eval_interval must coincide so "
            "every selectable evaluation is checkpoint-backed"
        )

    # The union holdout manifest is a curriculum-level provenance input to
    # the shared statistics artifact; it is validated once above and is not a
    # loader setting repeated in every stage YAML.  Compare only the contract
    # fields that a stage can actually declare.
    actual_shared = _stage_shared_contract(stage_payload)
    mismatches = {
        key: {"stage": actual_shared[key], "curriculum": shared[key]}
        for key in actual_shared
        if actual_shared[key] != shared[key]
    }
    if mismatches:
        raise CurriculumError(
            f"{stage_id} violates the shared representation/eval contract: "
            f"{mismatches}"
        )
    local_evaluation_path = _resolve_absolute_file(
        _stage_local_evaluation_manifest(stage_payload),
        field=f"{stage_id}.local_evaluation_manifest",
    )
    expected_local_evaluation_sha = _require_string(
        stage.get("local_evaluation_manifest_sha256"),
        field=f"{prefix}.local_evaluation_manifest_sha256",
    )
    if SHA256_RE.fullmatch(expected_local_evaluation_sha) is None:
        raise CurriculumError(
            f"{prefix}.local_evaluation_manifest_sha256 must be a SHA256"
        )
    actual_local_evaluation_sha = _sha256(local_evaluation_path)
    if actual_local_evaluation_sha != expected_local_evaluation_sha:
        raise CurriculumError(
            f"{stage_id} local evaluation manifest SHA mismatch: "
            f"{actual_local_evaluation_sha} != "
            f"{expected_local_evaluation_sha}"
        )

    view_path = _resolve_absolute_file(
        data_cfg.get("frozen_train_view_manifest"),
        field=f"{stage_id}.datasets.vla_data.frozen_train_view_manifest",
    )
    expected_view_sha = _require_string(
        data_cfg.get("frozen_train_view_manifest_sha256"),
        field=(
            f"{stage_id}.datasets.vla_data."
            "frozen_train_view_manifest_sha256"
        ),
    )
    if SHA256_RE.fullmatch(expected_view_sha) is None:
        raise CurriculumError(
            f"{stage_id} frozen train-view SHA is malformed"
        )
    view, eligible_windows = _validate_exhaustive_view(
        view_path=view_path,
        expected_manifest_sha256=expected_view_sha,
        stage_id=stage_id,
    )
    view_purpose = view.get("purpose")
    if workflow_kind == CHECKPOINT_HANDOFF_SMOKE:
        if view_purpose != CHECKPOINT_HANDOFF_SMOKE:
            raise CurriculumError(
                f"{stage_id} handoff smoke must use a purpose="
                f"{CHECKPOINT_HANDOFF_SMOKE!r} derived view"
            )
        smoke_contract = _require_mapping(
            stage_payload.get("checkpoint_handoff_smoke"),
            field=f"{stage_id}.checkpoint_handoff_smoke",
        )
        expected_smoke_contract = {
            "schema": HANDOFF_SMOKE_STAGE_SCHEMA,
            "scope": "checkpoint_handoff_only",
            "model_quality_claim_allowed": False,
            "expected_global_batch_rows": 128,
            "expected_optimizer_steps": 1,
        }
        if dict(smoke_contract) != expected_smoke_contract:
            raise CurriculumError(
                f"{stage_id} checkpoint_handoff_smoke contract must be "
                f"exactly {expected_smoke_contract}"
            )
    elif view_purpose == CHECKPOINT_HANDOFF_SMOKE:
        raise CurriculumError(
            f"{stage_id} production curriculum cannot consume a "
            "checkpoint-handoff smoke view"
        )
    actual_view_sha = expected_view_sha
    canonical_eval_holdout_binding = (
        _validate_canonical_view_eval_holdout_binding(
            view_path=view_path,
            view=view,
            evaluation_path=local_evaluation_path,
            evaluation_sha256=actual_local_evaluation_sha,
            eligible_windows=eligible_windows,
            stage_id=stage_id,
        )
        if data_cfg.get("dataset_py") == "canonical_subset_vla"
        else None
    )
    action_supervision_audit = (
        _validate_action_supervision_audit(
            view=view,
            data_cfg=data_cfg,
            stage_id=stage_id,
        )
        if require_verified_action_supervision is True
        else None
    )
    if stage.get("require_useful_subtask_coverage", False) is True:
        selection = _require_mapping(
            view.get("selection"),
            field=f"{stage_id}.frozen_view.selection",
        )
        coverage = _require_mapping(
            selection.get("subtask_prompt_coverage"),
            field=(
                f"{stage_id}.frozen_view.selection."
                "subtask_prompt_coverage"
            ),
        )
        coverage_rows = _require_int(
            coverage.get("row_count"),
            field=f"{stage_id}.subtask_prompt_coverage.row_count",
            minimum=1,
        )
        if coverage_rows != eligible_windows:
            raise CurriculumError(
                f"{stage_id} subtask coverage row count does not match view"
            )
        useful_rows = _require_int(
            coverage.get("useful_nonzero_subtask_row_count"),
            field=(
                f"{stage_id}.subtask_prompt_coverage."
                "useful_nonzero_subtask_row_count"
            ),
            minimum=1,
        )
        eligible_prompt_rows = _require_int(
            coverage.get("prompt_eligible_row_count"),
            field=(
                f"{stage_id}.subtask_prompt_coverage."
                "prompt_eligible_row_count"
            ),
            minimum=1,
        )
        distinct_subtasks = _require_int(
            coverage.get("distinct_useful_subtask_count"),
            field=(
                f"{stage_id}.subtask_prompt_coverage."
                "distinct_useful_subtask_count"
            ),
            minimum=2,
        )
        if useful_rows > eligible_windows or eligible_prompt_rows > useful_rows:
            raise CurriculumError(
                f"{stage_id} subtask coverage counts are inconsistent"
            )
        useful_fraction = useful_rows / eligible_windows
        if useful_fraction < 0.95:
            raise CurriculumError(
                f"{stage_id} useful subtask coverage is only "
                f"{useful_fraction:.2%}; refusing a mislabeled stage"
            )
        probability = data_cfg.get("subtask_prompt_append_probability")
        if probability != 0.7:
            raise CurriculumError(
                f"{stage_id} must append eligible subtasks with probability 0.7"
            )
        expected_appended_rows = eligible_prompt_rows * float(probability)
        if expected_appended_rows <= 0:
            raise CurriculumError(
                f"{stage_id} has no prompt-eligible rows at 70% prompting"
            )
    global_batch = int(plan["training"]["global_batch_size"])
    seed = _require_int(
        stage_payload.get("seed"),
        field=f"{stage_id}.seed",
        minimum=0,
    )
    subtask_prompt_probability = data_cfg.get(
        "subtask_prompt_append_probability"
    )
    if subtask_prompt_probability != 0.7:
        raise CurriculumError(
            f"{stage_id} must append eligible subtasks with probability 0.7"
        )
    steps_per_epoch = math.ceil(int(eligible_windows) / global_batch)
    if workflow_kind == CHECKPOINT_HANDOFF_SMOKE:
        if (
            expected_epochs != 1
            or int(eligible_windows) != global_batch
            or steps_per_epoch != 1
            or global_batch != int(
                smoke_contract["expected_global_batch_rows"]
            )
            or steps_per_epoch * expected_epochs
            != int(smoke_contract["expected_optimizer_steps"])
        ):
            raise CurriculumError(
                f"{stage_id} checkpoint-handoff smoke must be exactly one "
                "complete global batch, one exhaustive epoch, and one "
                "optimizer step"
            )
    monitoring = _require_mapping(
        stage.get("monitoring"),
        field=f"{prefix}.monitoring",
    )
    raw_fractions = monitoring.get("first_epoch_exposure_fractions")
    if not isinstance(raw_fractions, Sequence) or isinstance(
        raw_fractions, (str, bytes)
    ):
        raise CurriculumError(
            f"{prefix}.monitoring.first_epoch_exposure_fractions must be a list"
        )
    fractions: list[float] = []
    for fraction in raw_fractions:
        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
            raise CurriculumError(
                f"{prefix}.monitoring exposure fractions must be numeric"
            )
        value = float(fraction)
        if not math.isfinite(value) or value <= 0.0 or value > 1.0:
            raise CurriculumError(
                f"{prefix}.monitoring exposure fractions must be in (0, 1]"
            )
        fractions.append(value)
    if fractions != sorted(set(fractions)) or not fractions or fractions[-1] != 1.0:
        raise CurriculumError(
            f"{prefix}.monitoring exposure fractions must be sorted, unique, "
            "and end at 1.0"
        )
    if monitoring.get("full_epoch_required") is not True:
        raise CurriculumError(
            f"{prefix}.monitoring.full_epoch_required must be true"
        )
    if monitoring.get("partial_pass_is_epoch") is not False:
        raise CurriculumError(
            f"{prefix}.monitoring.partial_pass_is_epoch must be false"
        )
    configured_fractions = _optional_nested(
        stage_payload,
        "trainer.checkpoint_eval_milestone_fractions",
    )
    if configured_fractions != raw_fractions:
        raise CurriculumError(
            f"{stage_id} trainer.checkpoint_eval_milestone_fractions must "
            "exactly match curriculum monitoring fractions"
        )
    if (
        _optional_nested(
            stage_payload,
            "trainer.checkpoint_eval_milestone_steps",
        )
        is not None
    ):
        raise CurriculumError(
            f"{stage_id} source config must leave resolved milestone steps "
            "null; the curriculum derives them from its authenticated view"
        )
    first_epoch_checkpoints = sorted(
        {
            min(
                steps_per_epoch,
                max(1, math.ceil(steps_per_epoch * fraction)),
            )
            for fraction in fractions
        }
    )
    full_epoch_boundaries = [
        steps_per_epoch * epoch
        for epoch in range(1, expected_epochs + 1)
    ]
    checkpoint_eval_milestone_steps = sorted(
        set(first_epoch_checkpoints) | set(full_epoch_boundaries)
    )
    model_architecture_sha256 = _model_architecture_sha256(stage_payload)
    return {
        "id": stage_id,
        "workflow_kind": workflow_kind,
        "role": role,
        "initialization": initialization,
        "handoff_checkpoint_policy": handoff_checkpoint_policy,
        "config_path": str(config_path),
        "config_sha256": plan["config_sha256"],
        "model_architecture_sha256": model_architecture_sha256,
        "expected_full_dataset_epochs": expected_epochs,
        "eligible_window_count": int(eligible_windows),
        "steps_per_epoch": steps_per_epoch,
        "planned_optimizer_steps": steps_per_epoch * expected_epochs,
        "seed": seed,
        "global_batch_size": global_batch,
        "subtask_prompt_append_probability": float(
            subtask_prompt_probability
        ),
        "first_epoch_exposure_fractions": fractions,
        "first_epoch_checkpoint_steps": first_epoch_checkpoints,
        "full_epoch_boundary_steps": full_epoch_boundaries,
        "checkpoint_eval_milestone_steps": (
            checkpoint_eval_milestone_steps
        ),
        "plan": plan,
        "payload": stage_payload,
        "frozen_train_view_manifest": str(view_path),
        "frozen_train_view_manifest_sha256": actual_view_sha,
        "frozen_train_view_id": view.get("view_id"),
        "local_evaluation_manifest": str(local_evaluation_path),
        "local_evaluation_manifest_sha256": (
            actual_local_evaluation_sha
        ),
        "subtask_prompt_coverage": (
            coverage
            if stage.get("require_useful_subtask_coverage", False) is True
            else None
        ),
        "action_supervision_audit": action_supervision_audit,
        "canonical_eval_holdout_binding": (
            canonical_eval_holdout_binding
        ),
    }


def resolve_curriculum(
    path: Path, *, validate_stage_artifacts: bool = False
) -> dict[str, Any]:
    _, payload = _load_curriculum(path)
    schema_version = payload.get("schema_version")
    if type(schema_version) is not int or schema_version != 2:
        raise CurriculumError("curriculum schema_version must be exactly 2")
    curriculum_id = _require_string(
        payload.get("curriculum_id"), field="curriculum_id"
    )
    if CURRICULUM_ID_RE.fullmatch(curriculum_id) is None:
        raise CurriculumError(f"invalid curriculum_id: {curriculum_id!r}")
    workflow_kind = _require_string(
        payload.get("workflow_kind", PRODUCTION_CURRICULUM),
        field="workflow_kind",
    )
    if workflow_kind not in {
        PRODUCTION_CURRICULUM,
        CHECKPOINT_HANDOFF_SMOKE,
    }:
        raise CurriculumError(
            "workflow_kind must be production_curriculum or "
            "checkpoint_handoff_smoke"
        )
    state_root = Path(
        _require_string(payload.get("state_root_dir"), field="state_root_dir")
    ).expanduser()
    if not state_root.is_absolute():
        raise CurriculumError("state_root_dir must be absolute")
    bootstrap = dict(
        _require_mapping(payload.get("bootstrap"), field="bootstrap")
    )
    if set(bootstrap) != {
        "stage_config",
        "container_image",
        "scratch_root",
    }:
        raise CurriculumError(
            "bootstrap keys must be exactly stage_config, container_image, "
            "and scratch_root"
        )
    bootstrap_stage_config = _resolve_repo_file(
        bootstrap["stage_config"], field="bootstrap.stage_config"
    )
    bootstrap_image = _require_string(
        bootstrap["container_image"], field="bootstrap.container_image"
    )
    bootstrap_scratch = Path(
        _require_string(
            bootstrap["scratch_root"], field="bootstrap.scratch_root"
        )
    ).expanduser()
    if not bootstrap_scratch.is_absolute():
        raise CurriculumError("bootstrap.scratch_root must be absolute")
    shared = dict(
        _require_mapping(payload.get("shared_contract"), field="shared_contract")
    )
    expected_shared = {
        "state_dim",
        "action_dim",
        "action_horizon",
        "action_type",
        "action_delta_anchor",
        "gripper_action_type",
        "state_action_normalization",
        "normalization_statistics_artifact",
        "normalization_statistics_artifact_sha256",
        "action_representation_contract_sha256",
        "statistics_holdout_manifest",
        "statistics_holdout_manifest_sha256",
    }
    if set(shared) != expected_shared:
        raise CurriculumError(
            "shared_contract keys must be exact: "
            f"missing={sorted(expected_shared - set(shared))}, "
            f"extra={sorted(set(shared) - expected_shared)}"
        )
    for key in (
        "normalization_statistics_artifact_sha256",
        "action_representation_contract_sha256",
        "statistics_holdout_manifest_sha256",
    ):
        value = _require_string(shared[key], field=f"shared_contract.{key}")
        if SHA256_RE.fullmatch(value) is None:
            raise CurriculumError(f"shared_contract.{key} must be a SHA256")
    statistics_path = _resolve_absolute_file(
        shared["normalization_statistics_artifact"],
        field="shared_contract.normalization_statistics_artifact",
    )
    holdout_path = _resolve_absolute_file(
        shared["statistics_holdout_manifest"],
        field="shared_contract.statistics_holdout_manifest",
    )
    actual_holdout_sha256 = _sha256(holdout_path)
    if actual_holdout_sha256 != shared[
        "statistics_holdout_manifest_sha256"
    ]:
        raise CurriculumError("shared statistics holdout manifest SHA mismatch")
    _validate_shared_statistics_artifact(
        statistics_path=statistics_path,
        expected_statistics_sha256=shared[
            "normalization_statistics_artifact_sha256"
        ],
        expected_holdout_sha256=actual_holdout_sha256,
        expected_contract_sha256=shared[
            "action_representation_contract_sha256"
        ],
        expected_normalization=shared["state_action_normalization"],
    )
    shared["normalization_statistics_artifact"] = str(statistics_path)
    shared["statistics_holdout_manifest"] = str(holdout_path)

    raw_stages = payload.get("stages")
    if not isinstance(raw_stages, Sequence) or isinstance(
        raw_stages, (str, bytes)
    ):
        raise CurriculumError("stages must be a list")
    if len(raw_stages) != 3:
        raise CurriculumError(
            "a RealMan production or handoff-smoke workflow must have "
            "exactly three stages"
        )
    stages = []
    for index, raw_stage in enumerate(raw_stages):
        stage = _require_mapping(raw_stage, field=f"stages[{index}]")
        stages.append(
            _validate_source_stage(
                curriculum_path=path,
                stage_index=index,
                stage=stage,
                shared=shared,
                workflow_kind=workflow_kind,
                validate_artifacts=validate_stage_artifacts,
            )
        )
    stage_ids = [stage["id"] for stage in stages]
    if len(set(stage_ids)) != len(stage_ids):
        raise CurriculumError(f"stage IDs must be unique: {stage_ids}")
    _require_approved_role_sequence(stages)
    (
        architecture_sha256,
        experiment_seed,
        global_batch_size,
    ) = _require_comparable_training_contract(stages)

    first_runtime = stages[0]["plan"]["runtime"]
    if Path(stages[0]["config_path"]) != bootstrap_stage_config:
        raise CurriculumError(
            "bootstrap.stage_config must be the first curriculum stage"
        )
    if bootstrap_image != first_runtime["container_image"]:
        raise CurriculumError(
            "bootstrap.container_image differs from first-stage runtime"
        )
    if str(bootstrap_scratch) != str(first_runtime["scratch_root"]):
        raise CurriculumError(
            "bootstrap.scratch_root differs from first-stage runtime"
        )
    for stage in stages[1:]:
        runtime = stage["plan"]["runtime"]
        for key in (
            "container_image",
            "scratch_root",
            "expected_gpu_count",
            "num_processes",
            "num_machines",
            "mixed_precision",
            "dynamo_backend",
            "use_deepspeed",
            "torch_compile_environment",
        ):
            if runtime[key] != first_runtime[key]:
                raise CurriculumError(
                    f"stage {stage['id']} runtime.{key} differs from stage 1"
                )
    scratch_root = Path(str(first_runtime["scratch_root"])).expanduser()
    if not state_root.is_relative_to(scratch_root):
        raise CurriculumError(
            "state_root_dir must be inside the shared runtime.scratch_root"
        )
    total_steps = sum(stage["planned_optimizer_steps"] for stage in stages)
    return {
        "schema_version": 2,
        "curriculum_id": curriculum_id,
        "workflow_kind": workflow_kind,
        "curriculum_path": str(path.expanduser().resolve()),
        "curriculum_sha256": _sha256(path.expanduser().resolve()),
        "state_root_dir": str(state_root),
        "bootstrap": {
            "stage_config": str(bootstrap_stage_config),
            "container_image": bootstrap_image,
            "scratch_root": str(bootstrap_scratch),
        },
        "shared_contract": shared,
        "model_architecture_sha256": architecture_sha256,
        "experiment_seed": experiment_seed,
        "global_batch_size": global_batch_size,
        "stages": stages,
        "runtime": first_runtime,
        "total_planned_optimizer_steps": total_steps,
    }


def _public_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in plan.items()
        if key not in {"stages"}
    } | {
        "stages": [
            {
                key: copy.deepcopy(value)
                for key, value in stage.items()
                if key not in {"plan", "payload"}
            }
            for stage in plan["stages"]
        ]
    }


def print_plan(plan: Mapping[str, Any]) -> None:
    print(f"Curriculum                 : {plan['curriculum_id']}")
    print(f"Curriculum SHA256          : {plan['curriculum_sha256']}")
    print(f"State root                 : {plan['state_root_dir']}")
    print(
        "Shared policy contract      : "
        f"{plan['shared_contract']['state_dim']}D state / "
        f"{plan['shared_contract']['action_dim']}D action / "
        f"H={plan['shared_contract']['action_horizon']}"
    )
    print(
        "Shared normalization SHA    : "
        f"{plan['shared_contract']['normalization_statistics_artifact_sha256']}"
    )
    print(
        "Shared model architecture SHA: "
        f"{plan['model_architecture_sha256']}"
    )
    print(
        "Statistics leakage holdout   : "
        f"{plan['shared_contract']['statistics_holdout_manifest']}"
    )
    for index, stage in enumerate(plan["stages"], 1):
        print(
            f"Stage {index} ({stage['id']})".ljust(28)
            + ": "
            + f"{stage['role']}, epochs={stage['expected_full_dataset_epochs']}, "
            + f"windows={stage['eligible_window_count']}, "
            + f"steps/epoch={stage['steps_per_epoch']}, "
            + f"planned_steps={stage['planned_optimizer_steps']}"
        )
        print(f"  config                    : {stage['config_path']}")
        print(
            "  frozen view SHA           : "
            f"{stage['frozen_train_view_manifest_sha256']}"
        )
        print(
            "  one-pass monitor steps    : "
            f"{stage['first_epoch_checkpoint_steps']} "
            f"(fractions={stage['first_epoch_exposure_fractions']})"
        )
        print(
            "  full epoch boundaries     : "
            f"{stage['full_epoch_boundary_steps']}"
        )
        print(
            "  save/eval milestone steps : "
            f"{stage['checkpoint_eval_milestone_steps']}"
        )
        print(
            "  stage-local eval SHA      : "
            f"{stage['local_evaluation_manifest_sha256']}"
        )
    print(f"Total planned steps         : {plan['total_planned_optimizer_steps']}")


def _canonical_stage_run_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only fields injected by the distributed runtime before freeze."""

    canonical = copy.deepcopy(dict(payload))
    canonical.pop("output_dir", None)
    trainer = canonical.get("trainer")
    if isinstance(trainer, dict):
        for key in (
            "_accelerate_distributed_type",
            "_accelerate_gradient_accumulation_steps",
            "_accelerate_num_processes",
            "_accelerate_step_scheduler_with_optimizer",
        ):
            trainer.pop(key, None)
    return canonical


def _validate_materialized_run_config(
    *,
    run_config: Path,
    materialized_config: Path,
    expected_materialized_config_sha256: str,
) -> None:
    if SHA256_RE.fullmatch(expected_materialized_config_sha256) is None:
        raise CurriculumError(
            "expected materialized stage config SHA-256 is malformed"
        )
    materialized_config = materialized_config.expanduser().resolve(strict=True)
    if materialized_config.is_symlink() or not materialized_config.is_file():
        raise CurriculumError(
            "materialized stage config must be a regular non-symlink file: "
            f"{materialized_config}"
        )
    actual_materialized_sha = _sha256(materialized_config)
    if actual_materialized_sha != expected_materialized_config_sha256:
        raise CurriculumError(
            "materialized stage config changed after launch: "
            f"{actual_materialized_sha} != "
            f"{expected_materialized_config_sha256}"
        )
    try:
        materialized_payload = _plain(OmegaConf.load(materialized_config))
        run_payload = _plain(OmegaConf.load(run_config))
    except Exception as exc:
        raise CurriculumError(
            "unable to parse materialized/run stage config for identity "
            f"validation: {exc}"
        ) from exc
    if not isinstance(materialized_payload, Mapping) or not isinstance(
        run_payload, Mapping
    ):
        raise CurriculumError(
            "materialized and immutable run configs must be objects"
        )
    if _canonical_stage_run_config(
        run_payload
    ) != _canonical_stage_run_config(materialized_payload):
        raise CurriculumError(
            "immutable run config does not match the expected materialized "
            "stage config after removing documented runtime-only fields"
        )


def _validate_selection_handoff(
    run_dir: Path,
    *,
    materialized_config: Path | None = None,
    expected_materialized_config_sha256: str | None = None,
) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve(strict=True)
    pointer_path = run_dir / "best_checkpoint.json"
    if pointer_path.is_symlink() or not pointer_path.is_file():
        raise CurriculumError(
            f"completed stage lacks a regular best_checkpoint.json: {run_dir}"
        )
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    if not isinstance(pointer, Mapping) or pointer.get("schema_version") != 1:
        raise CurriculumError("best_checkpoint.json must use schema_version=1")
    step = _require_int(
        pointer.get("best_metric_step"),
        field="best_checkpoint.best_metric_step",
        minimum=0,
    )
    relative = pointer.get("checkpoint_relative_path")
    if relative != f"checkpoints/steps_{step}":
        raise CurriculumError(
            "best checkpoint relative path does not match the selected step"
        )
    checkpoint = (run_dir / relative).resolve(strict=True)
    if not checkpoint.is_relative_to(run_dir) or checkpoint.is_symlink():
        raise CurriculumError("best checkpoint escapes the run or is a symlink")
    model_path = checkpoint / "model.safetensors"
    trainer_state_path = checkpoint / "trainer_state.json"
    selection_path = checkpoint / "selection_state.json"
    for path in (model_path, trainer_state_path, selection_path):
        if path.is_symlink() or not path.is_file():
            raise CurriculumError(f"best checkpoint is incomplete: {path}")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection != pointer:
        raise CurriculumError(
            "checkpoint selection_state.json does not exactly match the run pointer"
        )
    trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    if trainer_state.get("completed_steps") != step:
        raise CurriculumError(
            "best checkpoint trainer_state completed_steps mismatches its path"
        )
    if trainer_state.get("selection_state_schema_version") != 1:
        raise CurriculumError(
            "best checkpoint trainer_state lacks selection schema v1"
        )

    eval_path = run_dir / "heldout_eval_metrics" / f"step_{step:08d}.json"
    if eval_path.is_symlink() or not eval_path.is_file():
        raise CurriculumError(
            f"best checkpoint lacks same-step heldout evaluation: {eval_path}"
        )
    evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
    if evaluation.get("schema_version") != 1:
        raise CurriculumError("same-step heldout evaluation schema is invalid")
    if evaluation.get("production_valid") is not True:
        raise CurriculumError("same-step heldout evaluation is not production-valid")
    if evaluation.get("checkpoint_selection_eligible") is not True:
        raise CurriculumError("same-step heldout evaluation is not selection-eligible")
    if evaluation.get("checkpoint_step") != step:
        raise CurriculumError("same-step heldout evaluation step mismatches pointer")
    if evaluation.get("checkpoint_relative_path") != relative:
        raise CurriculumError(
            "same-step heldout evaluation checkpoint path mismatches pointer"
        )
    selection_metric = _require_mapping(
        evaluation.get("selection_metric"), field="selection_metric"
    )
    for eval_key, pointer_key in (
        ("name", "best_metric_name"),
        ("mode", "best_metric_mode"),
        ("value", "best_metric_value"),
    ):
        if selection_metric.get(eval_key) != pointer.get(pointer_key):
            raise CurriculumError(
                f"same-step selection metric {eval_key} mismatches pointer"
            )
    checkpoint_evidence = _require_mapping(
        evaluation.get("checkpoint"), field="checkpoint"
    )
    model_sha = _sha256(model_path)
    trainer_state_sha = _sha256(trainer_state_path)
    if checkpoint_evidence.get("model_file") != "model.safetensors":
        raise CurriculumError("evaluation did not bind model.safetensors")
    if checkpoint_evidence.get("model_file_sha256") != model_sha:
        raise CurriculumError("evaluation model hash mismatches selected checkpoint")
    if checkpoint_evidence.get("trainer_state_sha256") != trainer_state_sha:
        raise CurriculumError(
            "evaluation trainer-state hash mismatches selected checkpoint"
        )
    run_config = run_dir / "config.yaml"
    if run_config.is_symlink() or not run_config.is_file():
        raise CurriculumError("completed stage lacks immutable config.yaml")
    recorded_config_sha = _optional_nested(
        evaluation, "run.config_sha256"
    )
    if recorded_config_sha != _sha256(run_config):
        raise CurriculumError(
            "same-step heldout evaluation does not bind immutable run config"
        )
    if (materialized_config is None) != (
        expected_materialized_config_sha256 is None
    ):
        raise CurriculumError(
            "materialized config path and expected SHA-256 must be supplied "
            "together"
        )
    if materialized_config is not None:
        _validate_materialized_run_config(
            run_config=run_config,
            materialized_config=materialized_config,
            expected_materialized_config_sha256=str(
                expected_materialized_config_sha256
            ),
        )
    return {
        "schema_version": 1,
        "run_dir": str(run_dir),
        "run_config": str(run_config),
        "run_config_sha256": _sha256(run_config),
        "model_architecture_sha256": _model_architecture_sha256(
            _plain(OmegaConf.load(run_config))
        ),
        "materialized_config_path": (
            None
            if materialized_config is None
            else str(materialized_config.expanduser().resolve())
        ),
        "materialized_config_sha256": (
            expected_materialized_config_sha256
        ),
        "selection_pointer": str(pointer_path),
        "selection_pointer_sha256": _sha256(pointer_path),
        "checkpoint_step": step,
        "checkpoint_relative_path": relative,
        "checkpoint_path": str(checkpoint),
        "model_path": str(model_path),
        "model_sha256": model_sha,
        "trainer_state_sha256": trainer_state_sha,
        "heldout_eval_path": str(eval_path),
        "heldout_eval_sha256": _sha256(eval_path),
        "best_metric_name": pointer["best_metric_name"],
        "best_metric_mode": pointer["best_metric_mode"],
        "best_metric_value": pointer["best_metric_value"],
    }


def _validate_natural_final_handoff(
    run_dir: Path,
    *,
    final_step: int,
    materialized_config: Path,
    expected_materialized_config_sha256: str,
) -> dict[str, Any]:
    """Authenticate the exact last optimizer-step checkpoint for a handoff.

    Offline best-checkpoint selection is useful diagnostics, but it must not
    silently rewind a sequential data curriculum.  Stage B therefore starts
    from the weights produced after *all* Stage-A examples, and Stage C starts
    from the weights produced after *all* Stage-B examples.
    """

    run_dir = run_dir.expanduser().resolve(strict=True)
    selected = _validate_selection_handoff(
        run_dir,
        materialized_config=materialized_config,
        expected_materialized_config_sha256=(
            expected_materialized_config_sha256
        ),
    )
    step = _require_int(
        final_step, field="natural_final.final_step", minimum=1
    )
    relative = f"checkpoints/steps_{step}"
    checkpoint = (run_dir / relative).resolve(strict=True)
    if not checkpoint.is_relative_to(run_dir) or checkpoint.is_symlink():
        raise CurriculumError(
            "natural-final checkpoint escapes the run or is a symlink"
        )
    model_path = checkpoint / "model.safetensors"
    trainer_state_path = checkpoint / "trainer_state.json"
    for path in (model_path, trainer_state_path):
        if path.is_symlink() or not path.is_file():
            raise CurriculumError(
                f"natural-final checkpoint is incomplete: {path}"
            )
    trainer_state = json.loads(
        trainer_state_path.read_text(encoding="utf-8")
    )
    if trainer_state.get("completed_steps") != step:
        raise CurriculumError(
            "natural-final trainer_state completed_steps mismatches its path"
        )
    if trainer_state.get("selection_state_schema_version") != 1:
        raise CurriculumError(
            "natural-final trainer_state lacks selection schema v1"
        )

    eval_path = (
        run_dir / "heldout_eval_metrics" / f"step_{step:08d}.json"
    )
    if eval_path.is_symlink() or not eval_path.is_file():
        raise CurriculumError(
            "natural-final checkpoint lacks same-step heldout evaluation: "
            f"{eval_path}"
        )
    evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
    if (
        evaluation.get("schema_version") != 1
        or evaluation.get("production_valid") is not True
        or evaluation.get("checkpoint_selection_eligible") is not True
        or evaluation.get("checkpoint_step") != step
        or evaluation.get("checkpoint_relative_path") != relative
    ):
        raise CurriculumError(
            "natural-final same-step heldout evaluation is not "
            "production-valid and checkpoint-bound"
        )
    checkpoint_evidence = _require_mapping(
        evaluation.get("checkpoint"), field="natural_final.checkpoint"
    )
    model_sha = _sha256(model_path)
    trainer_state_sha = _sha256(trainer_state_path)
    if (
        checkpoint_evidence.get("model_file") != "model.safetensors"
        or checkpoint_evidence.get("model_file_sha256") != model_sha
        or checkpoint_evidence.get("trainer_state_sha256")
        != trainer_state_sha
    ):
        raise CurriculumError(
            "natural-final evaluation does not authenticate the exact "
            "checkpoint model/trainer-state bytes"
        )
    run_config = run_dir / "config.yaml"
    if run_config.is_symlink() or not run_config.is_file():
        raise CurriculumError("completed stage lacks immutable config.yaml")
    if _optional_nested(evaluation, "run.config_sha256") != _sha256(
        run_config
    ):
        raise CurriculumError(
            "natural-final evaluation does not bind immutable run config"
        )
    metric = _require_mapping(
        evaluation.get("selection_metric"),
        field="natural_final.selection_metric",
    )
    metric_name = _require_string(
        metric.get("name"), field="natural_final.selection_metric.name"
    )
    metric_mode = _require_string(
        metric.get("mode"), field="natural_final.selection_metric.mode"
    )
    metric_value = metric.get("value")
    if (
        metric_mode not in {"min", "max"}
        or isinstance(metric_value, bool)
        or not isinstance(metric_value, (int, float))
        or not math.isfinite(float(metric_value))
    ):
        raise CurriculumError(
            "natural-final selection metric must be finite with min/max mode"
        )
    final_model_path = run_dir / "final_model" / "pytorch_model.pt"
    if final_model_path.is_symlink() or not final_model_path.is_file():
        raise CurriculumError(
            f"completed stage lacks natural-final model artifact: "
            f"{final_model_path}"
        )
    return {
        "schema_version": 2,
        "handoff_checkpoint_policy": NATURAL_FINAL_HANDOFF,
        "run_dir": str(run_dir),
        "run_config": str(run_config),
        "run_config_sha256": _sha256(run_config),
        "model_architecture_sha256": selected[
            "model_architecture_sha256"
        ],
        "materialized_config_path": str(
            materialized_config.expanduser().resolve()
        ),
        "materialized_config_sha256": (
            expected_materialized_config_sha256
        ),
        "checkpoint_step": step,
        "checkpoint_relative_path": relative,
        "checkpoint_path": str(checkpoint),
        "model_path": str(model_path),
        "model_sha256": model_sha,
        "trainer_state_sha256": trainer_state_sha,
        "heldout_eval_path": str(eval_path),
        "heldout_eval_sha256": _sha256(eval_path),
        "handoff_metric_name": metric_name,
        "handoff_metric_mode": metric_mode,
        "handoff_metric_value": float(metric_value),
        "best_metric_name": selected["best_metric_name"],
        "best_metric_mode": selected["best_metric_mode"],
        "best_metric_value": selected["best_metric_value"],
        "selection_diagnostics": {
            "selection_pointer": selected["selection_pointer"],
            "selection_pointer_sha256": selected[
                "selection_pointer_sha256"
            ],
            "best_checkpoint_step": selected["checkpoint_step"],
            "best_checkpoint_relative_path": selected[
                "checkpoint_relative_path"
            ],
            "best_model_sha256": selected["model_sha256"],
            "best_heldout_eval_sha256": selected[
                "heldout_eval_sha256"
            ],
        },
        "final_model_path": str(final_model_path),
        "final_model_sha256": _sha256(final_model_path),
    }


def _validate_completed_stage(
    *,
    stage: Mapping[str, Any],
    state_stage: Mapping[str, Any],
) -> dict[str, Any]:
    resolved_path = Path(
        _require_string(
            state_stage.get("resolved_config_path"),
            field=f"{stage['id']}.resolved_config_path",
        )
    )
    expected_resolved_sha = _require_string(
        state_stage.get("resolved_config_sha256"),
        field=f"{stage['id']}.resolved_config_sha256",
    )
    run_dir = Path(
        _require_string(
            state_stage.get("run_dir"),
            field=f"{stage['id']}.run_dir",
        )
    ).expanduser()
    planned_final_step = int(stage["planned_optimizer_steps"])
    if (
        stage.get("handoff_checkpoint_policy")
        != NATURAL_FINAL_HANDOFF
    ):
        raise CurriculumError(
            f"completed stage {stage['id']} does not declare an authenticated "
            "natural-final handoff"
        )
    handoff = _validate_natural_final_handoff(
        run_dir,
        final_step=planned_final_step,
        materialized_config=resolved_path,
        expected_materialized_config_sha256=expected_resolved_sha,
    )
    if (
        handoff["model_architecture_sha256"]
        != stage["model_architecture_sha256"]
    ):
        raise CurriculumError(
            f"completed stage {stage['id']} immutable run architecture "
            "does not match the reviewed stage plan"
        )
    return handoff


def _validate_state_against_plan(
    *,
    state: Mapping[str, Any],
    plan: Mapping[str, Any],
    run_id: str,
) -> None:
    if (
        state.get("curriculum_id") != plan["curriculum_id"]
        or state.get("curriculum_run_id") != run_id
        or state.get("curriculum_config_path") != plan["curriculum_path"]
        or state.get("curriculum_config_sha256")
        != plan["curriculum_sha256"]
    ):
        raise CurriculumError(
            "curriculum resume state does not match the requested "
            "curriculum config/run identity"
        )
    state_stages = state.get("stages")
    if not isinstance(state_stages, list) or len(state_stages) != len(
        plan["stages"]
    ):
        raise CurriculumError(
            "curriculum resume state stage list does not match the plan"
        )
    saw_incomplete = False
    for planned, recorded in zip(
        plan["stages"], state_stages, strict=True
    ):
        if not isinstance(recorded, Mapping):
            raise CurriculumError(
                "curriculum resume state contains a malformed stage"
            )
        if (
            recorded.get("id") != planned["id"]
            or recorded.get("role") != planned["role"]
            or recorded.get("source_config_path") != planned["config_path"]
            or recorded.get("source_config_sha256")
            != planned["config_sha256"]
            or recorded.get("model_architecture_sha256")
            != planned["model_architecture_sha256"]
        ):
            raise CurriculumError(
                f"curriculum resume state drift for stage {planned['id']}"
            )
        status = recorded.get("status")
        if status not in {"pending", "running", "failed", "complete"}:
            raise CurriculumError(
                f"curriculum resume stage {planned['id']} has invalid status "
                f"{status!r}"
            )
        if status != "complete":
            saw_incomplete = True
        elif saw_incomplete:
            raise CurriculumError(
                "curriculum resume state has a completed stage after an "
                "incomplete predecessor"
            )


def _latest_complete_resume_checkpoint(
    *,
    run_dir: Path,
    expected_ranks: int,
) -> Path:
    candidates: list[tuple[int, Path]] = []
    checkpoint_root = run_dir / "checkpoints"
    if checkpoint_root.is_dir():
        for candidate in checkpoint_root.iterdir():
            match = re.fullmatch(r"steps_(\d+)", candidate.name)
            if match is not None and candidate.is_dir():
                candidates.append((int(match.group(1)), candidate))
    errors: list[str] = []
    for _, candidate in sorted(candidates, reverse=True):
        try:
            validated, _ = h100_training._validate_checkpoint(
                candidate, expected_ranks
            )
            return validated
        except (h100_training.PlanError, OSError) as exc:
            errors.append(f"{candidate.name}: {exc}")
    detail = "; ".join(errors) if errors else "no steps_N directories"
    raise CurriculumError(
        "interrupted curriculum stage has no complete full-state checkpoint "
        f"to resume ({detail})"
    )


def _materialize_stage_resume_config(
    *,
    run_dir: Path,
    checkpoint: Path,
    output_path: Path,
) -> str:
    immutable_config = run_dir / "config.yaml"
    if immutable_config.is_symlink() or not immutable_config.is_file():
        raise CurriculumError(
            f"interrupted stage lacks immutable config.yaml: {run_dir}"
        )
    cfg = OmegaConf.load(immutable_config)
    cfg.trainer.is_resume = True
    cfg.trainer.resume_from_checkpoint = str(checkpoint)
    payload = OmegaConf.to_yaml(cfg, resolve=True).encode("utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(
        f".{output_path.name}.tmp-{os.getpid()}"
    )
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output_path)
    return hashlib.sha256(payload).hexdigest()


def _pid_is_alive(value: Any) -> bool:
    if type(value) is not int or value <= 0:
        return False
    try:
        os.kill(value, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _restore_completed_stage_handoff(
    *,
    stage: Mapping[str, Any],
    state_stage: Mapping[str, Any],
) -> dict[str, Any]:
    handoff_path = Path(
        _require_string(
            state_stage.get("handoff_path"),
            field=f"{stage['id']}.handoff_path",
        )
    ).expanduser()
    expected_handoff_sha = _require_string(
        state_stage.get("handoff_sha256"),
        field=f"{stage['id']}.handoff_sha256",
    )
    if (
        handoff_path.is_symlink()
        or not handoff_path.is_file()
        or _sha256(handoff_path) != expected_handoff_sha
    ):
        raise CurriculumError(
            f"completed stage {stage['id']} handoff artifact is missing or "
            "has changed"
        )
    saved = json.loads(handoff_path.read_text(encoding="utf-8"))
    recomputed = _validate_completed_stage(
        stage=stage,
        state_stage=state_stage,
    )
    if saved != recomputed:
        raise CurriculumError(
            f"completed stage {stage['id']} handoff no longer matches its "
            "authenticated run artifacts"
        )
    if (
        state_stage.get("selected_checkpoint_step")
        != recomputed["checkpoint_step"]
        or state_stage.get("selected_model_sha256")
        != recomputed["model_sha256"]
    ):
        raise CurriculumError(
            f"completed stage {stage['id']} selection identity drifted"
        )
    return recomputed


def _materialize_stage_config(
    *,
    curriculum_plan: Mapping[str, Any],
    stage: Mapping[str, Any],
    run_id: str,
    output_path: Path,
    previous_handoff: Mapping[str, Any] | None,
) -> dict[str, Any]:
    _, source_payload = h100_training._load_config(Path(stage["config_path"]))
    cfg = OmegaConf.create(copy.deepcopy(source_payload))
    cfg.run_id = run_id
    cfg.trainer.is_resume = False
    cfg.trainer.resume_from_checkpoint = None
    cfg.trainer.pretrained_checkpoint = (
        None if previous_handoff is None else previous_handoff["model_path"]
    )
    cfg.trainer.pretrained_checkpoint_sha256 = (
        None
        if previous_handoff is None
        else previous_handoff["model_sha256"]
    )
    # A stage boundary is a complete-model handoff.  Optimizer/scheduler/RNG
    # are fresh, but no subset of model modules may be silently skipped.
    cfg.trainer.reload_modules = None
    cfg.trainer.checkpoint_eval_milestone_steps = list(
        stage["checkpoint_eval_milestone_steps"]
    )
    handoff_payload = {
        "schema_version": 1,
        "curriculum_id": curriculum_plan["curriculum_id"],
        "curriculum_config_path": curriculum_plan["curriculum_path"],
        "curriculum_config_sha256": curriculum_plan["curriculum_sha256"],
        "stage_id": stage["id"],
        "stage_role": stage["role"],
        "source_stage_config_path": stage["config_path"],
        "source_stage_config_sha256": stage["config_sha256"],
        "model_architecture_sha256": stage[
            "model_architecture_sha256"
        ],
        "frozen_train_view_manifest": stage["frozen_train_view_manifest"],
        "frozen_train_view_manifest_sha256": stage[
            "frozen_train_view_manifest_sha256"
        ],
        "shared_normalization_statistics_artifact": curriculum_plan[
            "shared_contract"
        ]["normalization_statistics_artifact"],
        "shared_normalization_statistics_artifact_sha256": curriculum_plan[
            "shared_contract"
        ]["normalization_statistics_artifact_sha256"],
        "statistics_holdout_manifest": curriculum_plan["shared_contract"][
            "statistics_holdout_manifest"
        ],
        "statistics_holdout_manifest_sha256": curriculum_plan[
            "shared_contract"
        ]["statistics_holdout_manifest_sha256"],
        "stage_local_evaluation_manifest": stage[
            "local_evaluation_manifest"
        ],
        "stage_local_evaluation_manifest_sha256": stage[
            "local_evaluation_manifest_sha256"
        ],
        "epoch_contract": {
            "one_epoch_is_entire_frozen_view": True,
            "eligible_window_count": stage["eligible_window_count"],
            "steps_per_epoch": stage["steps_per_epoch"],
            "expected_full_dataset_epochs": stage[
                "expected_full_dataset_epochs"
            ],
            "first_epoch_exposure_fractions": stage[
                "first_epoch_exposure_fractions"
            ],
            "first_epoch_checkpoint_steps": stage[
                "first_epoch_checkpoint_steps"
            ],
            "full_epoch_boundary_steps": stage[
                "full_epoch_boundary_steps"
            ],
            "checkpoint_eval_milestone_steps": stage[
                "checkpoint_eval_milestone_steps"
            ],
            "partial_pass_is_epoch": False,
        },
        "initialization": stage["initialization"],
        "handoff_checkpoint_policy": stage[
            "handoff_checkpoint_policy"
        ],
        "previous_stage_handoff": (
            None
            if previous_handoff is None
            else {
                key: previous_handoff[key]
                for key in (
                    "handoff_checkpoint_policy",
                    "run_dir",
                    "run_config_sha256",
                    "model_architecture_sha256",
                    "checkpoint_step",
                    "checkpoint_relative_path",
                    "model_path",
                    "model_sha256",
                    "heldout_eval_sha256",
                    "best_metric_name",
                    "best_metric_mode",
                    "best_metric_value",
                    "handoff_metric_name",
                    "handoff_metric_mode",
                    "handoff_metric_value",
                )
            }
        ),
        "optimizer_scheduler_rng_reset": True,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    cfg.curriculum_handoff = handoff_payload
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        OmegaConf.to_yaml(cfg, resolve=True),
        encoding="utf-8",
    )
    return handoff_payload


def _training_environment(plan: Mapping[str, Any]) -> dict[str, str]:
    runtime = plan["runtime"]
    env = os.environ.copy()
    env["STARVLA_USE_DEEPSPEED"] = (
        "1" if bool(runtime["use_deepspeed"]) else "0"
    )
    if runtime["torch_compile_environment"] == "disabled":
        env["TORCH_COMPILE_DISABLE"] = "1"
        env["TORCHDYNAMO_DISABLE"] = "1"
        env["STARVLA_ALLOW_TORCH_COMPILE"] = "0"
    interface = str(runtime["network_interface"])
    if interface == "auto":
        interface = h100_training._discover_default_interface()
    env["NCCL_SOCKET_IFNAME"] = interface
    env["GLOO_SOCKET_IFNAME"] = interface
    return env


def run_curriculum(
    plan: Mapping[str, Any],
    *,
    run_id: str | None,
    resume: bool = False,
) -> Path:
    if resume and run_id is None:
        raise CurriculumError(
            "curriculum resume requires the original --run-id"
        )
    if run_id is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_id = f"{plan['curriculum_id']}_{timestamp}"
    if RUN_ID_RE.fullmatch(run_id) is None:
        raise CurriculumError(f"invalid curriculum run ID: {run_id!r}")
    if not run_id.startswith(f"{plan['curriculum_id']}_"):
        raise CurriculumError(
            f"curriculum run ID must start with {plan['curriculum_id']}_"
        )
    state_dir = Path(plan["state_root_dir"]) / run_id
    state_path = state_dir / "curriculum_state.json"
    if resume:
        if not state_dir.is_dir():
            raise CurriculumError(
                f"curriculum resume state directory does not exist: {state_dir}"
            )
        state = _load_curriculum_state(state_path)
        _validate_state_against_plan(state=state, plan=plan, run_id=run_id)
        state["status"] = "running"
        state.pop("ended_utc", None)
        state["last_resumed_utc"] = datetime.now(timezone.utc).isoformat()
        _write_curriculum_state(state_path, state)
    else:
        if state_dir.exists():
            raise CurriculumError(
                f"curriculum state directory already exists: {state_dir}; "
                "use --resume with the same --run-id after authenticating it"
            )
        state_dir.mkdir(parents=True)
        state = {
            "schema_version": 3,
            "curriculum_id": plan["curriculum_id"],
            "curriculum_run_id": run_id,
            "curriculum_config_path": plan["curriculum_path"],
            "curriculum_config_sha256": plan["curriculum_sha256"],
            "status": "running",
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "stages": [
                {
                    "id": stage["id"],
                    "role": stage["role"],
                    "status": "pending",
                    "source_config_path": stage["config_path"],
                    "source_config_sha256": stage["config_sha256"],
                    "model_architecture_sha256": stage[
                        "model_architecture_sha256"
                    ],
                }
                for stage in plan["stages"]
            ],
        }
        _write_curriculum_state(state_path, state)

    previous_handoff: Mapping[str, Any] | None = None
    for stage, state_stage in zip(
        plan["stages"], state["stages"], strict=True
    ):
        if state_stage["status"] != "complete":
            break
        previous_handoff = _restore_completed_stage_handoff(
            stage=stage,
            state_stage=state_stage,
        )
    if state.get("status") == "complete":
        print(f"Curriculum already complete: {state_path}", flush=True)
        return state_path

    active_process: subprocess.Popen[bytes] | None = None

    def forward_signal(signum: int, _frame: Any) -> None:
        nonlocal active_process
        if active_process is not None and active_process.poll() is None:
            active_process.send_signal(signum)

    old_handlers = {
        signum: signal.signal(signum, forward_signal)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        for index, stage in enumerate(plan["stages"]):
            state_stage = state["stages"][index]
            if state_stage["status"] == "complete":
                continue
            stage_run_id = (
                f"{stage['plan']['training']['run_id_prefix']}_"
                f"{run_id}_{index + 1:02d}_{stage['id']}"
            )
            if RUN_ID_RE.fullmatch(stage_run_id) is None:
                raise CurriculumError(
                    f"materialized stage run ID is invalid: {stage_run_id!r}"
                )
            run_dir = (
                Path(stage["plan"]["training"]["run_root_dir"])
                / stage_run_id
            )
            resolved_path = (
                state_dir
                / "resolved_configs"
                / f"{index + 1:02d}_{stage['id']}.yaml"
            )
            launch_config = resolved_path
            resumed_from_checkpoint: Path | None = None

            if state_stage["status"] == "pending":
                if run_dir.exists():
                    raise CurriculumError(
                        f"fresh stage output already exists: {run_dir}"
                    )
                handoff_input = _materialize_stage_config(
                    curriculum_plan=plan,
                    stage=stage,
                    run_id=stage_run_id,
                    output_path=resolved_path,
                    previous_handoff=previous_handoff,
                )
                state_stage.update(
                    {
                        "run_id": stage_run_id,
                        "run_dir": str(run_dir),
                        "resolved_config_path": str(resolved_path),
                        "resolved_config_sha256": _sha256(resolved_path),
                        "handoff_input": handoff_input,
                        "started_utc": datetime.now(
                            timezone.utc
                        ).isoformat(),
                        "launch_attempt": 1,
                    }
                )
            else:
                if state_stage.get("run_id") != stage_run_id or Path(
                    str(state_stage.get("run_dir", ""))
                ) != run_dir:
                    raise CurriculumError(
                        f"interrupted stage {stage['id']} run identity drifted"
                    )
                if _pid_is_alive(state_stage.get("active_process_pid")):
                    raise CurriculumError(
                        f"stage {stage['id']} still has live process "
                        f"{state_stage['active_process_pid']}; refusing a "
                        "duplicate resume"
                    )
                if (
                    not resolved_path.is_file()
                    or _sha256(resolved_path)
                    != state_stage.get("resolved_config_sha256")
                ):
                    raise CurriculumError(
                        f"interrupted stage {stage['id']} materialized config "
                        "is missing or changed"
                    )
                resumed_from_checkpoint = (
                    _latest_complete_resume_checkpoint(
                        run_dir=run_dir,
                        expected_ranks=int(
                            stage["plan"]["runtime"]["num_processes"]
                        ),
                    )
                )
                launch_config = (
                    state_dir
                    / "resume_configs"
                    / (
                        f"{index + 1:02d}_{stage['id']}_"
                        f"{resumed_from_checkpoint.name}.yaml"
                    )
                )
                resume_sha = _materialize_stage_resume_config(
                    run_dir=run_dir,
                    checkpoint=resumed_from_checkpoint,
                    output_path=launch_config,
                )
                state_stage["resume_config_path"] = str(launch_config)
                state_stage["resume_config_sha256"] = resume_sha
                state_stage["resumed_from_checkpoint"] = str(
                    resumed_from_checkpoint
                )
                state_stage["launch_attempt"] = (
                    int(state_stage.get("launch_attempt", 1)) + 1
                )

            state_stage["status"] = "running"
            state_stage.pop("ended_utc", None)
            state_stage.pop("exit_code", None)
            _write_curriculum_state(state_path, state)
            command = h100_training._accelerate_command(
                stage["plan"], launch_config
            )
            print(
                f"Starting curriculum stage {index + 1}/"
                f"{len(plan['stages'])}: {stage['id']}",
                flush=True,
            )
            print(f"Run directory: {run_dir}", flush=True)
            if resumed_from_checkpoint is not None:
                print(
                    f"Full-state resume: {resumed_from_checkpoint}",
                    flush=True,
                )
            print(f"Command: {' '.join(command)}", flush=True)
            active_process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=_training_environment(stage["plan"]),
            )
            state_stage["active_process_pid"] = active_process.pid
            _write_curriculum_state(state_path, state)
            return_code = active_process.wait()
            active_process = None
            state_stage["active_process_pid"] = None
            state_stage["exit_code"] = return_code
            state_stage["ended_utc"] = datetime.now(
                timezone.utc
            ).isoformat()
            if return_code != 0:
                state_stage["status"] = "failed"
                state["status"] = "failed"
                _write_curriculum_state(state_path, state)
                raise CurriculumError(
                    f"stage {stage['id']} exited with code {return_code}; "
                    f"state is preserved at {state_path}"
                )
            handoff = _validate_completed_stage(
                stage=stage,
                state_stage=state_stage,
            )
            handoff_path = (
                state_dir
                / "handoffs"
                / f"{index + 1:02d}_{stage['id']}.json"
            )
            _atomic_json(handoff_path, handoff)
            state_stage.update(
                {
                    "status": "complete",
                    "handoff_path": str(handoff_path),
                    "handoff_sha256": _sha256(handoff_path),
                    "selected_checkpoint_step": handoff[
                        "checkpoint_step"
                    ],
                    "selected_model_sha256": handoff["model_sha256"],
                    "handoff_checkpoint_policy": handoff[
                        "handoff_checkpoint_policy"
                    ],
                    "handoff_metric": {
                        "name": handoff["handoff_metric_name"],
                        "mode": handoff["handoff_metric_mode"],
                        "value": handoff["handoff_metric_value"],
                    },
                    "selection_metric": {
                        "name": handoff["best_metric_name"],
                        "mode": handoff["best_metric_mode"],
                        "value": handoff["best_metric_value"],
                    },
                }
            )
            _write_curriculum_state(state_path, state)
            previous_handoff = handoff
        state["status"] = "complete"
        state["completed_utc"] = datetime.now(timezone.utc).isoformat()
        state["final_handoff"] = previous_handoff
        _write_curriculum_state(state_path, state)
        print(f"Curriculum complete: {state_path}", flush=True)
        return state_path
    except Exception:
        if state.get("status") == "running":
            state["status"] = "failed"
            state["ended_utc"] = datetime.now(timezone.utc).isoformat()
            _write_curriculum_state(state_path, state)
        raise
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)


def setup(plan: Mapping[str, Any]) -> None:
    seen: set[tuple[str, str]] = set()
    for stage in plan["stages"]:
        helpers = stage["plan"]["runtime"]["helper_repositories"]
        identity = tuple(
            sorted(
                (
                    name,
                    f"{entry['url']}@{entry['commit']}:{entry['path']}",
                )
                for name, entry in helpers.items()
            )
        )
        if identity in seen:
            continue
        seen.add(identity)
        h100_training.setup_dependencies(Path(stage["config_path"]))


def check(plan: Mapping[str, Any]) -> None:
    for index, stage in enumerate(plan["stages"]):
        # The expensive CUDA/model preflight is identical across the three
        # stages, so run it once. Every stage still gets full config/artifact,
        # Git, data, GCS, hardware, and port validation.
        h100_training.check_plan(
            Path(stage["config_path"]),
            deep=index == 0,
        )
    print("Three-stage curriculum preflight: PASS")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plan, validate, or run the config-owned RealSource → intervention "
            "→ high-quality H100 curriculum."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "setup", "check", "run"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--config", type=Path, default=DEFAULT_CURRICULUM)
        if name == "plan":
            sub.add_argument("--json", action="store_true")
        elif name == "run":
            sub.add_argument("--run-id")
            sub.add_argument(
                "--resume",
                action="store_true",
                help=(
                    "authenticate the existing curriculum state and resume "
                    "the first incomplete stage from its newest complete "
                    "full-state checkpoint"
                ),
            )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        validate_artifacts = args.command in {"check", "run"}
        plan = resolve_curriculum(
            args.config,
            validate_stage_artifacts=validate_artifacts,
        )
        if args.command == "plan":
            if args.json:
                print(
                    json.dumps(
                        _public_plan(plan),
                        indent=2,
                        sort_keys=True,
                        allow_nan=False,
                    )
                )
            else:
                print_plan(plan)
        elif args.command == "setup":
            setup(plan)
        elif args.command == "check":
            check(plan)
        elif args.command == "run":
            run_curriculum(
                plan,
                run_id=args.run_id,
                resume=bool(args.resume),
            )
        else:  # pragma: no cover
            raise AssertionError(args.command)
    except (
        CurriculumError,
        h100_training.PlanError,
        OSError,
        subprocess.CalledProcessError,
        json.JSONDecodeError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
