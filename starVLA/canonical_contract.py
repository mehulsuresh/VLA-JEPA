"""Pure, shared canonical adapter/action cache contract helpers.

This module deliberately uses only the Python standard library so the human
launcher, manifest generator, and training dataloader compute byte-identical
contracts without importing Torch, OpenCV, or the dataloader package.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


CANONICAL_EVAL_SELECTION_ALGORITHM = "sha256_episode_rank_dense_window_v1"
SHARD_Q01_Q99_UNCLIPPED = "shard_q01_q99_unclipped"
DEFAULT_ABSOLUTE_ACTION_REFERENCES = (
    "absolute",
    "absolute_derived_from_future_observation",
    "mixed",
    "source_action_command",
)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def canonical_adapter_contract_sha256(
    data_cfg: Any,
    *,
    local_repo_root: str | Path | None = None,
) -> str:
    """Hash every definition and code path that projects 53-D/49-D values."""

    canonical_root = Path(
        _cfg_get(
            data_cfg,
            "dataset_canonicalization_root",
            "/home/mehul/work/dataset-canonicalization",
        )
    ).expanduser()
    adapter_dir = Path(
        _cfg_get(
            data_cfg,
            "adapter_dir",
            canonical_root / "configs/dataset_adapters",
        )
    ).expanduser()
    if not adapter_dir.is_dir():
        raise FileNotFoundError(
            f"Canonical adapter directory does not exist: {adapter_dir}"
        )
    manifest_path = adapter_dir / "MANIFEST.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise FileNotFoundError(
            f"Canonical adapter MANIFEST.json is missing: {manifest_path}"
        )
    adapter_candidates = sorted(
        child
        for child in adapter_dir.rglob("*")
        if child.suffix.lower() in {".json", ".yaml", ".yml"}
    )
    symlinked_adapters = [
        path.as_posix() for path in adapter_candidates if path.is_symlink()
    ]
    if symlinked_adapters:
        raise FileNotFoundError(
            "Canonical adapter definitions must not be symlinks: "
            f"{symlinked_adapters}"
        )
    adapter_files = [path for path in adapter_candidates if path.is_file()]
    external_semantic_files = [
        canonical_root / "src/model_v0/data/adapters.py",
        canonical_root / "src/model_v0/data/unified_schema.py",
    ]
    repo_root = (
        Path(local_repo_root).expanduser().resolve()
        if local_repo_root is not None
        else Path(__file__).resolve().parents[1]
    )
    local_semantic_files = [
        repo_root / "starVLA/canonical_contract.py",
        repo_root / "starVLA/dataloader/canonical_subset_dataset.py",
    ]
    missing = [
        path.as_posix()
        for path in (*external_semantic_files, *local_semantic_files)
        if not path.is_file() or path.is_symlink()
    ]
    if missing:
        raise FileNotFoundError(
            "Canonical adapter semantic code is missing or symlinked: "
            f"{missing}"
        )
    payload = {
        "version": 2,
        "adapter_files": [
            {
                "path": child.relative_to(adapter_dir).as_posix(),
                "sha256": _file_sha256(child),
            }
            for child in adapter_files
        ],
        "external_semantic_code_files": [
            {
                "path": child.relative_to(canonical_root).as_posix(),
                "sha256": _file_sha256(child),
            }
            for child in external_semantic_files
        ],
        "local_semantic_code_files": [
            {
                "path": child.relative_to(repo_root).as_posix(),
                "sha256": _file_sha256(child),
            }
            for child in local_semantic_files
        ],
    }
    return _stable_json_sha256(payload)


def canonical_action_sidecar_variant(
    data_cfg: Any,
    *,
    action_horizon: int,
    canonical_eval_manifest_sha256: str | None,
    exclude_eval_episodes_from_training: bool,
    adapter_contract_sha256: str | None = None,
    local_repo_root: str | Path | None = None,
) -> str:
    """Hash canonical action/statistics semantics into a cache variant."""

    if adapter_contract_sha256 is None:
        adapter_contract_sha256 = canonical_adapter_contract_sha256(
            data_cfg,
            local_repo_root=local_repo_root,
        )
    dtype_string = (
        "<class 'numpy.float16'>"
        if str(_cfg_get(data_cfg, "sidecar_dtype", "float16")) == "float16"
        else "<class 'numpy.float32'>"
    )
    payload = {
        "action_type": str(
            _cfg_get(data_cfg, "action_type", "dataset_native")
        ).lower(),
        "action_delta_anchor": str(
            _cfg_get(data_cfg, "action_delta_anchor", "chunk_start_state")
        ).lower(),
        "gripper_action_type": str(
            _cfg_get(data_cfg, "gripper_action_type", "absolute")
        ).lower(),
        "absolute_action_references": sorted(
            str(value).lower()
            for value in _as_list(
                _cfg_get(
                    data_cfg,
                    "absolute_action_references",
                    list(DEFAULT_ABSOLUTE_ACTION_REFERENCES),
                )
            )
        ),
        "action_horizon": int(action_horizon),
        "sample_stride": max(1, int(_cfg_get(data_cfg, "sample_stride", 1))),
        "normalization": str(
            _cfg_get(
                data_cfg,
                "sidecar_normalization",
                SHARD_Q01_Q99_UNCLIPPED,
            )
        ).lower(),
        "dtype": dtype_string,
        "representation_contract_version": 2,
        "adapter_contract_sha256": str(adapter_contract_sha256),
        "canonical_eval_manifest_sha256": canonical_eval_manifest_sha256,
        "exclude_eval_episodes_from_training": bool(
            exclude_eval_episodes_from_training
        ),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
