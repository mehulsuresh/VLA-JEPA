#!/usr/bin/env python3
"""Materialize the fail-closed RealSource -> intervention -> HQ handoff smoke.

The reviewed YAML templates own every training choice.  This helper performs
only mechanical artifact binding: it authenticates the three 128-row real-data
views, isolated smoke statistics/holdout, and immutable evaluation manifests;
then writes stage configs plus a runnable curriculum containing their exact
paths and SHA-256 digests.

It intentionally cannot change epochs, learning rates, masking, prompts,
batching, dimensions, normalization, or handoff policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence
import uuid

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import h100_curriculum, h100_training  # noqa: E402
from starVLA.action_representation import (  # noqa: E402
    Q01_Q99_UNCLIPPED,
    REALMAN_18D_ACTION_CONTRACT,
    load_openpi_realman_union_statistics,
)
from starVLA.dataloader import dataset_view  # noqa: E402


TEMPLATE_ROOT = REPO_ROOT / "scripts/config/h100"
CURRICULUM_TEMPLATE = (
    TEMPLATE_ROOT / "realman_checkpoint_handoff_smoke_v1.template.yaml"
)
STAGE_TEMPLATE_ROOT = TEMPLATE_ROOT / "realman_curriculum/handoff_smoke"
STAGE_SPECS = (
    {
        "id": "realsource",
        "template": STAGE_TEMPLATE_ROOT / "realsource_one_step_v1.yaml",
        "output": "01_realsource_one_step_v1.yaml",
        "view_field": "canonical_eval_manifest",
    },
    {
        "id": "intervention",
        "template": STAGE_TEMPLATE_ROOT / "intervention_one_step_v1.yaml",
        "output": "02_intervention_one_step_v1.yaml",
        "view_field": "episode_split_manifest",
    },
    {
        "id": "hq",
        "template": STAGE_TEMPLATE_ROOT / "hq_one_step_v1.yaml",
        "output": "03_hq_one_step_v1.yaml",
        "view_field": "episode_split_manifest",
    },
)
DEFAULT_OUTPUT_DIR = (
    TEMPLATE_ROOT / "generated/realman_checkpoint_handoff_smoke_v1"
)
RESULT_PREFIX = "REALMAN_HANDOFF_SMOKE_MATERIALIZATION="
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SMOKE_STATISTICS_SCHEMA = "realman-handoff-smoke-statistics-v1"
_SMOKE_STATISTICS_SCOPE = "checkpoint_handoff_validation_only"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_file(value: str | Path, *, label: str) -> Path:
    path = Path(value).expanduser()
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {path}")
    try:
        path = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{label} does not exist: {path}") from exc
    if not path.is_file():
        raise ValueError(f"{label} must be a regular file: {path}")
    return path


def _repo_relative(path: Path) -> str:
    resolved = path.expanduser().resolve()
    if not resolved.is_relative_to(REPO_ROOT):
        raise ValueError(
            f"Generated config/template must be inside repository: {resolved}"
        )
    return str(resolved.relative_to(REPO_ROOT))


def _config_owned_path(path: Path) -> str:
    """Use mount-stable repository paths when an artifact lives in-tree."""

    resolved = path.expanduser().resolve()
    if resolved.is_relative_to(REPO_ROOT):
        return _repo_relative(resolved)
    return str(resolved)


def _atomic_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = OmegaConf.to_yaml(
        OmegaConf.create(dict(payload)),
        resolve=True,
        sort_keys=False,
    ).encode("utf-8")
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_mapping(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be an object: {path}")
    return payload


def _validate_smoke_view(
    path: Path, *, expected_source: str | None = None
) -> str:
    view = dataset_view.load_frozen_view(
        path,
        expected_representation_contract_sha256=(
            REALMAN_18D_ACTION_CONTRACT.sha256()
        ),
        verify_ledger=True,
    )
    descriptor = view.descriptor
    if descriptor.get("purpose") != h100_curriculum.CHECKPOINT_HANDOFF_SMOKE:
        raise ValueError(
            f"Smoke view has wrong purpose: {path}: "
            f"{descriptor.get('purpose')!r}"
        )
    if view.row_count != 128 or view.unique_sample_count != 128:
        raise ValueError(
            f"Smoke view must contain exactly 128 unique rows: {path}"
        )
    epoch = descriptor.get("epoch_contract")
    if not isinstance(epoch, Mapping) or {
        "mode": epoch.get("mode"),
        "epoch_passes": epoch.get("epoch_passes"),
        "drop_last": epoch.get("drop_last"),
        "replacement": epoch.get("replacement"),
        "shuffle": epoch.get("shuffle"),
        "ddp_tail": epoch.get("ddp_tail"),
    } != {
        "mode": dataset_view.ALL_EXHAUSTIVE_MODE,
        "epoch_passes": 1,
        "drop_last": False,
        "replacement": False,
        "shuffle": "deterministic_bijection_per_epoch",
        "ddp_tail": "duplicated_padding_reported_separately",
    }:
        raise ValueError(
            f"Smoke view is not an exact one-pass exhaustive population: {path}"
        )
    usage = descriptor.get("usage_contract")
    if (
        not isinstance(usage, Mapping)
        or usage.get("training_allowed") is not True
        or usage.get("scope") != "checkpoint_handoff_validation_only"
        or usage.get("model_quality_claim_allowed") is not False
        or usage.get("statistics_accumulation") != "forbidden"
    ):
        raise ValueError(f"Smoke view has an unsafe usage contract: {path}")
    selection = descriptor.get("selection")
    if (
        not isinstance(selection, Mapping)
        or selection.get("schema")
        != "realman-checkpoint-handoff-smoke-view-v1"
        or selection.get("requested_logical_row_count") != 128
        or not isinstance(selection.get("parent_manifest_sha256"), str)
        or _SHA256_RE.fullmatch(
            str(selection.get("parent_manifest_sha256"))
        )
        is None
        or not isinstance(selection.get("parent_ledger_sha256"), str)
        or _SHA256_RE.fullmatch(str(selection.get("parent_ledger_sha256")))
        is None
    ):
        raise ValueError(
            f"Smoke view lacks authenticated production-parent lineage: {path}"
        )
    sources = descriptor.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError(f"Smoke view has no source descriptors: {path}")
    if expected_source == "realsource":
        if any(
            source.get("backend") != "canonical"
            or not str(source.get("dataset_id", "")).startswith(
                "RealSourceData/RealSource-World/"
            )
            for source in sources
        ):
            raise ValueError(
                f"RealSource smoke view contains a non-RealSource source: {path}"
            )
    elif expected_source in {"intervention", "hq"}:
        expected_dataset = {
            "intervention": (
                "magna_training_data_with_interventions_final_subtask_labelled"
            ),
            "hq": "latest_high_quality_magna_data_final_subtask_labelled",
        }[expected_source]
        if (
            len(sources) != 1
            or sources[0].get("backend") != "lerobot"
            or sources[0].get("dataset_name") != expected_dataset
        ):
            raise ValueError(
                f"{expected_source} smoke view binds the wrong dataset: {path}"
            )
    elif expected_source is not None:
        raise ValueError(f"Unknown expected smoke source {expected_source!r}")
    return view.manifest_sha256


def _validate_smoke_statistics_population(
    statistics_payload: Mapping[str, Any],
    *,
    view_hashes: Sequence[str],
) -> None:
    """Require statistics derived from the exact three smoke populations."""

    population = statistics_payload.get("population")
    if not isinstance(population, Mapping):
        raise ValueError("Smoke statistics lack population provenance")
    if population.get("unique_base_frames") != 128 * len(STAGE_SPECS):
        raise ValueError(
            "Smoke statistics must contain exactly 384 unique base frames"
        )
    sources = population.get("sources")
    if (
        not isinstance(sources, list)
        or [source.get("id") for source in sources] != [
            str(spec["id"]) for spec in STAGE_SPECS
        ]
        or len(view_hashes) != len(STAGE_SPECS)
    ):
        raise ValueError(
            "Smoke statistics sources do not exactly match the handoff stages"
        )
    for source, expected_view_sha256 in zip(
        sources, view_hashes, strict=True
    ):
        provenance = source.get("provenance")
        smoke = (
            provenance.get("handoff_smoke")
            if isinstance(provenance, Mapping)
            else None
        )
        expected = {
            "schema": _SMOKE_STATISTICS_SCHEMA,
            "scope": _SMOKE_STATISTICS_SCOPE,
            "model_quality_claim_allowed": False,
            "exact_training_logical_rows": 128,
            "parent_smoke_view_sha256": expected_view_sha256,
        }
        mismatches = {
            field: {
                "statistics": (
                    smoke.get(field)
                    if isinstance(smoke, Mapping)
                    else None
                ),
                "expected": value,
            }
            for field, value in expected.items()
            if not isinstance(smoke, Mapping)
            or smoke.get(field) != value
        }
        if mismatches:
            raise ValueError(
                "Smoke statistics are not derived from the exact supplied "
                f"view: {mismatches}"
            )


def _stage_override(
    *,
    template: Path,
    view_path: Path,
    view_sha256: str,
    statistics_path: Path,
    statistics_sha256: str,
    evaluation_path: Path,
    evaluation_field: str,
) -> dict[str, Any]:
    return {
        "extends": [_repo_relative(template)],
        "datasets": {
            "vla_data": {
                "frozen_train_view_manifest": _config_owned_path(view_path),
                "frozen_train_view_manifest_sha256": view_sha256,
                "normalization_statistics_artifact": _config_owned_path(
                    statistics_path
                ),
                "normalization_statistics_artifact_sha256": (
                    statistics_sha256
                ),
                evaluation_field: _config_owned_path(evaluation_path),
            }
        },
    }


def materialize(
    *,
    output_dir: Path | str,
    realsource_view: Path | str,
    intervention_view: Path | str,
    hq_view: Path | str,
    statistics_artifact: Path | str,
    statistics_holdout: Path | str,
    realsource_eval_manifest: Path | str,
    intervention_eval_manifest: Path | str,
    hq_eval_manifest: Path | str,
    overwrite: bool = False,
) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    if not output.is_relative_to(REPO_ROOT):
        raise ValueError("output_dir must be inside the repository")
    if output.is_symlink():
        raise ValueError("output_dir must not be a symlink")
    output_files = [
        output / str(spec["output"]) for spec in STAGE_SPECS
    ]
    curriculum_output = output / "curriculum.yaml"
    output_files.append(curriculum_output)
    symlinks = [path for path in output_files if path.is_symlink()]
    if symlinks:
        raise ValueError(
            "Materialized outputs must not be symlinks: "
            + ", ".join(str(path) for path in symlinks)
        )
    existing = [path for path in output_files if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite materialized smoke config(s): "
            + ", ".join(str(path) for path in existing)
        )

    views = [
        _regular_file(realsource_view, label="RealSource smoke view"),
        _regular_file(intervention_view, label="intervention smoke view"),
        _regular_file(hq_view, label="HQ smoke view"),
    ]
    view_hashes = [
        _validate_smoke_view(path, expected_source=str(spec["id"]))
        for path, spec in zip(views, STAGE_SPECS, strict=True)
    ]
    statistics = _regular_file(
        statistics_artifact, label="smoke-only union statistics"
    )
    holdout = _regular_file(
        statistics_holdout, label="smoke-only union holdout"
    )
    statistics_sha256 = _sha256(statistics)
    holdout_sha256 = _sha256(holdout)
    statistics_payload = load_openpi_realman_union_statistics(
        statistics, statistics_sha256
    )
    if statistics_payload.get("normalization") != Q01_Q99_UNCLIPPED:
        raise ValueError("Smoke statistics do not use q01/q99 unclipped")
    if (
        statistics_payload.get("population", {}).get(
            "holdout_manifest_sha256"
        )
        != holdout_sha256
    ):
        raise ValueError(
            "Smoke statistics were built against a different holdout"
        )
    if statistics_payload.get("population", {}).get("source_order") != [
        "realsource",
        "intervention",
        "hq",
    ]:
        raise ValueError(
            "Smoke statistics must bind RealSource, intervention, then HQ"
        )
    _validate_smoke_statistics_population(
        statistics_payload,
        view_hashes=view_hashes,
    )

    evaluations = [
        _regular_file(
            realsource_eval_manifest,
            label="RealSource evaluation manifest",
        ),
        _regular_file(
            intervention_eval_manifest,
            label="intervention evaluation manifest",
        ),
        _regular_file(hq_eval_manifest, label="HQ evaluation manifest"),
    ]
    evaluation_hashes = [_sha256(path) for path in evaluations]
    for view_path, evaluation_sha256 in zip(
        views, evaluation_hashes, strict=True
    ):
        view = dataset_view.load_frozen_view(view_path, verify_ledger=False)
        binding = view.descriptor.get("selection", {}).get(
            "evaluation_holdout"
        )
        if (
            not isinstance(binding, Mapping)
            or binding.get("manifest_sha256") != evaluation_sha256
        ):
            raise ValueError(
                "Smoke view is not derived from the production view bound "
                f"to its supplied evaluation manifest: {view_path}"
            )

    output.mkdir(parents=True, exist_ok=True)
    generated_stage_paths: list[Path] = []
    for spec, view, view_sha, evaluation in zip(
        STAGE_SPECS, views, view_hashes, evaluations, strict=True
    ):
        template = _regular_file(
            Path(str(spec["template"])),
            label=f"{spec['id']} stage template",
        )
        destination = output / str(spec["output"])
        _atomic_yaml(
            destination,
            _stage_override(
                template=template,
                view_path=view,
                view_sha256=view_sha,
                statistics_path=statistics,
                statistics_sha256=statistics_sha256,
                evaluation_path=evaluation,
                evaluation_field=str(spec["view_field"]),
            ),
        )
        generated_stage_paths.append(destination)

    curriculum = _load_mapping(
        _regular_file(CURRICULUM_TEMPLATE, label="curriculum template")
    )
    stage_paths = [_repo_relative(path) for path in generated_stage_paths]
    curriculum["bootstrap"]["stage_config"] = stage_paths[0]
    shared = curriculum["shared_contract"]
    shared["normalization_statistics_artifact"] = str(statistics)
    shared["normalization_statistics_artifact_sha256"] = (
        statistics_sha256
    )
    shared["statistics_holdout_manifest"] = str(holdout)
    shared["statistics_holdout_manifest_sha256"] = holdout_sha256
    for stage, stage_path, evaluation_sha in zip(
        curriculum["stages"],
        stage_paths,
        evaluation_hashes,
        strict=True,
    ):
        stage["config"] = stage_path
        stage["local_evaluation_manifest_sha256"] = evaluation_sha
    _atomic_yaml(curriculum_output, curriculum)

    # This is the final gate: compose every generated stage, authenticate all
    # artifacts, and prove each stage is exactly one global batch/step.
    plan = h100_curriculum.resolve_curriculum(
        curriculum_output, validate_stage_artifacts=False
    )
    if (
        plan.get("workflow_kind")
        != h100_curriculum.CHECKPOINT_HANDOFF_SMOKE
        or [stage["planned_optimizer_steps"] for stage in plan["stages"]]
        != [1, 1, 1]
    ):
        raise AssertionError("Materialized smoke did not resolve to 1/1/1 steps")

    return {
        "schema": "realman-checkpoint-handoff-smoke-materialization-v1",
        "curriculum": {
            "path": str(curriculum_output),
            "sha256": _sha256(curriculum_output),
        },
        "stage_configs": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in generated_stage_paths
        ],
        "views": [
            {"path": str(path), "sha256": digest}
            for path, digest in zip(views, view_hashes, strict=True)
        ],
        "statistics": {
            "path": str(statistics),
            "sha256": statistics_sha256,
            "holdout_path": str(holdout),
            "holdout_sha256": holdout_sha256,
        },
        "evaluation_manifest_sha256": evaluation_hashes,
        "planned_optimizer_steps": [1, 1, 1],
        "model_quality_claim_allowed": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--realsource-view", type=Path, required=True)
    parser.add_argument("--intervention-view", type=Path, required=True)
    parser.add_argument("--hq-view", type=Path, required=True)
    parser.add_argument("--statistics-artifact", type=Path, required=True)
    parser.add_argument("--statistics-holdout", type=Path, required=True)
    parser.add_argument(
        "--realsource-eval-manifest", type=Path, required=True
    )
    parser.add_argument(
        "--intervention-eval-manifest", type=Path, required=True
    )
    parser.add_argument("--hq-eval-manifest", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = materialize(
        output_dir=args.output_dir,
        realsource_view=args.realsource_view,
        intervention_view=args.intervention_view,
        hq_view=args.hq_view,
        statistics_artifact=args.statistics_artifact,
        statistics_holdout=args.statistics_holdout,
        realsource_eval_manifest=args.realsource_eval_manifest,
        intervention_eval_manifest=args.intervention_eval_manifest,
        hq_eval_manifest=args.hq_eval_manifest,
        overwrite=args.overwrite,
    )
    print(RESULT_PREFIX + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
