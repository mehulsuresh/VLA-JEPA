#!/usr/bin/env python3
"""Derive a tiny immutable handoff-only view from a production frozen view.

The output still has an ``all_exhaustive`` epoch: every row in the derived
population is consumed exactly once.  The population is deliberately one
global batch so the RealSource -> intervention -> HQ checkpoint handoff can be
tested with one real optimizer step per stage.  This tool never changes source
labels, validity masks, actions, states, or episode metadata.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
_DATASET_VIEW_PATH = REPO_ROOT / "starVLA/dataloader/dataset_view.py"
_DATASET_VIEW_SPEC = importlib.util.spec_from_file_location(
    "_realman_handoff_smoke_dataset_view", _DATASET_VIEW_PATH
)
if _DATASET_VIEW_SPEC is None or _DATASET_VIEW_SPEC.loader is None:
    raise ImportError(
        f"Could not load dataset-view helpers from {_DATASET_VIEW_PATH}"
    )
dataset_view = importlib.util.module_from_spec(_DATASET_VIEW_SPEC)
sys.modules[_DATASET_VIEW_SPEC.name] = dataset_view
_DATASET_VIEW_SPEC.loader.exec_module(dataset_view)


HANDOFF_SMOKE_PURPOSE = "checkpoint_handoff_smoke"
HANDOFF_SMOKE_SCHEMA = "realman-checkpoint-handoff-smoke-view-v1"
RESULT_PREFIX = "REALMAN_HANDOFF_SMOKE_VIEW_RESULT="


def _strip_generated(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(row)
    payload.pop("schema", None)
    payload.pop("ordinal", None)
    return payload


def _expanded_prefix(
    parent: dataset_view.FrozenDatasetView,
    logical_rows: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in parent.iter_rows():
        if len(selected) >= logical_rows:
            break
        selected.append(_strip_generated(row))
    if len(selected) != logical_rows:
        raise ValueError(
            f"Parent view has only {len(selected)} rows; "
            f"{logical_rows} were requested."
        )
    return selected


def _range_prefix(
    parent: dataset_view.FrozenDatasetView,
    logical_rows: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    remaining = logical_rows
    for raw in parent.iter_rows():
        if remaining <= 0:
            break
        row = _strip_generated(raw)
        available = int(row["sample_count"])
        take = min(available, remaining)
        row["base_stop"] = int(row["base_start"]) + (
            take * int(row["base_step"])
        )
        row["sample_count"] = take
        row["range_id"] = dataset_view.make_range_id(
            episode_content_id=str(row["episode_content_id"]),
            base_start=int(row["base_start"]),
            base_stop=int(row["base_stop"]),
            base_step=int(row["base_step"]),
            horizon=int(row["horizon"]),
            target_fps=int(row["target_fps"]),
            representation_contract_sha256=str(
                parent.descriptor["representation"]["contract_sha256"]
            ),
            end_clamp=row["end_clamp_policy"] == "repeat_last",
        )
        selected.append(row)
        remaining -= take
    if remaining:
        raise ValueError(
            f"Parent range view is {remaining} logical rows short of "
            f"the requested {logical_rows}."
        )
    return selected


def _exact_selected_sources(
    *,
    parent_sources: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Copy referenced sources while narrowing LeRobot shard commitments."""

    referenced_sources = {str(row["source_id"]) for row in selected}
    rows_by_source: dict[str, list[Mapping[str, Any]]] = {}
    for row in selected:
        rows_by_source.setdefault(str(row["source_id"]), []).append(row)
    result: list[dict[str, Any]] = []
    for raw_source in parent_sources:
        source_id = str(raw_source["source_id"])
        if source_id not in referenced_sources:
            continue
        source = deepcopy(dict(raw_source))
        if source.get("backend") == "lerobot":
            bindings = source.get("selected_data_shards")
            if not isinstance(bindings, list) or not bindings:
                raise ValueError(
                    f"LeRobot parent source {source_id!r} must bind "
                    "selected_data_shards."
                )
            by_path = {str(binding["path"]): binding for binding in bindings}
            selected_paths: set[str] = set()
            for row in rows_by_source[source_id]:
                data_file = row.get("data_file")
                if not isinstance(data_file, str) or not data_file:
                    raise ValueError(
                        f"Selected LeRobot row for {source_id!r} lacks "
                        "data_file shard identity."
                    )
                selected_paths.add(data_file)
            missing = selected_paths - set(by_path)
            if missing:
                raise ValueError(
                    f"Selected LeRobot rows for {source_id!r} reference "
                    "shards absent from the authenticated parent: "
                    f"{sorted(missing)}."
                )
            source["selected_data_shards"] = [
                deepcopy(dict(by_path[path]))
                for path in sorted(selected_paths)
            ]
        result.append(source)
    if {str(source["source_id"]) for source in result} != referenced_sources:
        raise ValueError("Selected rows reference a source absent from parent.")
    return result


def _descriptor(
    parent: dataset_view.FrozenDatasetView,
    *,
    selected: Sequence[Mapping[str, Any]],
    logical_rows: int,
) -> dict[str, Any]:
    descriptor = parent.descriptor
    sources = _exact_selected_sources(
        parent_sources=descriptor["sources"],
        selected=selected,
    )
    referenced_sources = {str(row["source_id"]) for row in selected}
    episode_indices = sorted(
        {int(row["episode_index"]) for row in selected}
    )
    selection: dict[str, Any] = {
        "schema": HANDOFF_SMOKE_SCHEMA,
        "algorithm": (
            "ordered_logical_prefix_from_authenticated_parent_v1"
        ),
        "parent_manifest": str(parent.manifest_path),
        "parent_manifest_sha256": parent.manifest_sha256,
        "parent_view_id": parent.view_id,
        "parent_ledger_sha256": descriptor["rows"]["sha256"],
        "requested_logical_row_count": logical_rows,
        "selected_episode_indices": episode_indices,
        "selected_source_ids": sorted(referenced_sources),
    }
    parent_selection = descriptor.get("selection")
    if isinstance(parent_selection, Mapping):
        # Canonical training validation still proves that the tiny derived
        # training population excludes the exact immutable production eval
        # episodes.  Preserve only that authenticated binding; the smoke
        # selection itself remains separately parent-hash bound above.
        evaluation_holdout = parent_selection.get("evaluation_holdout")
        if isinstance(evaluation_holdout, Mapping):
            selection["evaluation_holdout"] = deepcopy(
                dict(evaluation_holdout)
            )

    result: dict[str, Any] = {
        "view_name": (
            f"{descriptor['view_name']}_handoff_smoke_{logical_rows}_v1"
        ),
        "description": (
            "Handoff-only real-data prefix derived from an authenticated "
            "production frozen view. It is exactly one reviewed global batch "
            "and is not a model-quality experiment."
        ),
        "purpose": HANDOFF_SMOKE_PURPOSE,
        "sources": sources,
        "representation": deepcopy(descriptor["representation"]),
        "holdout_exclusions": deepcopy(
            descriptor["holdout_exclusions"]
        ),
        "selection": selection,
        "epoch_contract": {
            "mode": dataset_view.ALL_EXHAUSTIVE_MODE,
            "epoch_passes": 1,
            "drop_last": False,
            "replacement": False,
            "shuffle": "deterministic_bijection_per_epoch",
            "ddp_tail": "duplicated_padding_reported_separately",
            "global_membership": (
                "every_derived_ledger_row_once_before_ddp_padding"
            ),
            "partial_pass_is_epoch": False,
        },
        "usage_contract": {
            "training_allowed": True,
            "scope": "checkpoint_handoff_validation_only",
            "model_quality_claim_allowed": False,
            "statistics_accumulation": "forbidden",
        },
        "generator": {
            "schema": HANDOFF_SMOKE_SCHEMA,
            "script": "scripts/build_realman_handoff_smoke_view.py",
            "logical_row_count": logical_rows,
        },
    }
    if "action_supervision_audit" in descriptor:
        result["action_supervision_audit"] = deepcopy(
            descriptor["action_supervision_audit"]
        )
        result["handoff_smoke_action_audit_scope"] = {
            "schema": HANDOFF_SMOKE_SCHEMA,
            "scope": "authenticated_parent_view",
            "parent_view_id": parent.view_id,
            "note": (
                "The smoke test preserves the production mask/config but "
                "does not make a new label-quality claim from one batch."
            ),
        }
    return result


def build_handoff_smoke_view(
    *,
    parent_manifest: Path | str,
    output_manifest: Path | str,
    logical_rows: int,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dataset_view.FrozenDatasetViewBuild:
    if isinstance(logical_rows, bool) or logical_rows <= 0:
        raise ValueError("logical_rows must be a positive integer.")
    parent = dataset_view.load_frozen_view(parent_manifest)
    if parent.descriptor.get("purpose") in {
        dataset_view.STATISTICS_POPULATION_CANDIDATE_PURPOSE,
        dataset_view.EVAL_SELECTION_POPULATION_CANDIDATE_PURPOSE,
        HANDOFF_SMOKE_PURPOSE,
    }:
        raise ValueError(
            "The parent must be a production training view, not a candidate "
            "or another smoke view."
        )
    if parent.row_count < logical_rows:
        raise ValueError(
            f"Parent has {parent.row_count} rows, fewer than {logical_rows}."
        )
    if parent.encoding == dataset_view.EXPANDED_ROWS_ENCODING:
        selected = _expanded_prefix(parent, logical_rows)
    elif parent.encoding == dataset_view.EPISODE_RANGES_ENCODING:
        selected = _range_prefix(parent, logical_rows)
    else:  # pragma: no cover - load_frozen_view already rejects this.
        raise ValueError(f"Unsupported parent encoding: {parent.encoding}")
    descriptor = _descriptor(
        parent, selected=selected, logical_rows=logical_rows
    )
    if parent.encoding == dataset_view.EXPANDED_ROWS_ENCODING:
        return dataset_view.write_frozen_view(
            output_manifest,
            descriptor=descriptor,
            rows=selected,
            overwrite=overwrite,
            dry_run=dry_run,
        )
    return dataset_view.write_frozen_range_view(
        output_manifest,
        descriptor=descriptor,
        ranges=selected,
        overwrite=overwrite,
        dry_run=dry_run,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--logical-rows", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = build_handoff_smoke_view(
        parent_manifest=args.parent_manifest,
        output_manifest=args.output,
        logical_rows=args.logical_rows,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    print(RESULT_PREFIX + json.dumps(result.to_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
