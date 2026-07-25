#!/usr/bin/env python3
"""Materialize the authenticated three-source RealMan statistics contract.

The command derives one global holdout from the exact RealSource,
intervention, and high-quality evaluation manifests.  Every source is checked
against its separately materialized statistics-population candidate view.
It then writes the matching frozen-parquet union population manifest consumed
by ``compute_openpi_realman_union_stats.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from starVLA.action_representation import (  # noqa: E402
    OPENPI_REALMAN_UNION_POPULATION_SCHEMA,
    REALMAN_18D_ACTION_CONTRACT,
    deterministic_json_bytes,
)
from starVLA.realman_union_holdout import (  # noqa: E402
    CANONICAL_SOURCE_KIND,
    LEROBOT_SOURCE_KIND,
    REALMAN_UNION_HOLDOUT_SCHEMA,
    derive_source_holdout,
    write_global_holdout_manifest,
)


SOURCE_ORDER = ("realsource", "intervention", "hq")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_input_file(value: str | Path, *, label: str) -> Path:
    path = Path(value).expanduser()
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {path}")
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{label} does not exist: {path}") from exc
    if not resolved.is_file():
        raise ValueError(f"{label} must be a regular file: {resolved}")
    return resolved


def _portable_path(path: Path, *, base_dir: Path) -> str:
    try:
        return str(path.relative_to(base_dir))
    except ValueError:
        return str(path)


def _write_canonical_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    overwrite: bool,
) -> str:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}")
    serialized = deterministic_json_bytes(dict(payload)) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        temporary.write_bytes(serialized)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(serialized).hexdigest()


def _population_source(
    *,
    derived: Mapping[str, Any],
    candidate_view: Path,
    population_dir: Path,
    reader_options: Mapping[str, Any],
) -> dict[str, Any]:
    reader = {
        "kind": "frozen_parquet_view",
        "view_manifest": _portable_path(
            candidate_view, base_dir=population_dir
        ),
        "view_manifest_sha256": derived[
            "statistics_population_candidate_view_sha256"
        ],
        **dict(reader_options),
    }
    return {
        "id": derived["id"],
        "catalog_sha256": derived["catalog_sha256"],
        "reader": reader,
        "provenance": {
            "global_holdout_schema": REALMAN_UNION_HOLDOUT_SCHEMA,
            "evaluation_manifest_sha256": derived[
                "evaluation_manifest_sha256"
            ],
            "statistics_population_candidate_view_id": derived[
                "statistics_population_candidate_view_id"
            ],
        },
    }


def build_union_contract(
    *,
    holdout_output: str | Path,
    population_output: str | Path,
    realsource_eval_manifest: str | Path,
    realsource_candidate_view: str | Path,
    realsource_cache_dir: str | Path,
    intervention_eval_manifest: str | Path,
    intervention_candidate_view: str | Path,
    intervention_dataset_root: str | Path,
    hq_eval_manifest: str | Path,
    hq_candidate_view: str | Path,
    hq_dataset_root: str | Path,
    allow_gcs_download: bool = False,
    gcs_download_timeout_seconds: int = 900,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Derive and atomically write matching holdout/population manifests."""

    holdout_path = Path(holdout_output).expanduser().resolve()
    population_path = Path(population_output).expanduser().resolve()
    if holdout_path == population_path:
        raise ValueError("Holdout and population outputs must be different.")
    if not overwrite:
        existing = [
            path for path in (holdout_path, population_path) if path.exists()
        ]
        if existing:
            raise FileExistsError(
                "Refusing to overwrite existing union contract output(s): "
                + ", ".join(str(path) for path in existing)
            )
    timeout = int(gcs_download_timeout_seconds)
    if timeout <= 0:
        raise ValueError("gcs_download_timeout_seconds must be positive.")

    inputs = {
        "realsource": {
            "kind": CANONICAL_SOURCE_KIND,
            "eval": _require_input_file(
                realsource_eval_manifest,
                label="RealSource evaluation manifest",
            ),
            "view": _require_input_file(
                realsource_candidate_view,
                label="RealSource 50% statistics candidate view",
            ),
        },
        "intervention": {
            "kind": LEROBOT_SOURCE_KIND,
            "eval": _require_input_file(
                intervention_eval_manifest,
                label="intervention evaluation manifest",
            ),
            "view": _require_input_file(
                intervention_candidate_view,
                label="intervention statistics candidate view",
            ),
        },
        "hq": {
            "kind": LEROBOT_SOURCE_KIND,
            "eval": _require_input_file(
                hq_eval_manifest,
                label="HQ evaluation manifest",
            ),
            "view": _require_input_file(
                hq_candidate_view,
                label="HQ statistics candidate view",
            ),
        },
    }
    derived_sources = [
        derive_source_holdout(
            source_id=source_id,
            kind=inputs[source_id]["kind"],
            evaluation_manifest=inputs[source_id]["eval"],
            evaluation_manifest_sha256=_file_sha256(
                inputs[source_id]["eval"]
            ),
            candidate_view_manifest=inputs[source_id]["view"],
            candidate_view_manifest_sha256=_file_sha256(
                inputs[source_id]["view"]
            ),
            manifest_dir=holdout_path.parent,
        )
        for source_id in SOURCE_ORDER
    ]
    _, holdout_sha256 = write_global_holdout_manifest(
        holdout_path,
        derived_sources,
        overwrite=overwrite,
    )

    population_sources = [
        _population_source(
            derived=derived_sources[0],
            candidate_view=inputs["realsource"]["view"],
            population_dir=population_path.parent,
            reader_options={
                "cache_dir": str(
                    Path(realsource_cache_dir).expanduser().resolve()
                ),
                "allow_gcs_download": bool(allow_gcs_download),
                "gcs_download_timeout_seconds": timeout,
            },
        ),
        _population_source(
            derived=derived_sources[1],
            candidate_view=inputs["intervention"]["view"],
            population_dir=population_path.parent,
            reader_options={
                "dataset_root": str(
                    Path(intervention_dataset_root).expanduser().resolve()
                ),
            },
        ),
        _population_source(
            derived=derived_sources[2],
            candidate_view=inputs["hq"]["view"],
            population_dir=population_path.parent,
            reader_options={
                "dataset_root": str(
                    Path(hq_dataset_root).expanduser().resolve()
                ),
            },
        ),
    ]
    episode_keys = sorted(
        key
        for source in derived_sources
        for key in source["episode_keys"]
    )
    population = {
        "schema": OPENPI_REALMAN_UNION_POPULATION_SCHEMA,
        "contract_sha256": REALMAN_18D_ACTION_CONTRACT.sha256(),
        "source_order": list(SOURCE_ORDER),
        "sources": population_sources,
        "holdout": {
            "manifest": _portable_path(
                holdout_path, base_dir=population_path.parent
            ),
            "manifest_sha256": holdout_sha256,
            "episode_keys": episode_keys,
        },
    }
    population_sha256 = _write_canonical_json(
        population_path,
        population,
        overwrite=overwrite,
    )
    return {
        "holdout": {
            "path": str(holdout_path),
            "sha256": holdout_sha256,
            "episode_count": len(episode_keys),
        },
        "population": {
            "path": str(population_path),
            "sha256": population_sha256,
            "source_order": list(SOURCE_ORDER),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout-output", required=True)
    parser.add_argument("--population-output", required=True)
    parser.add_argument("--realsource-eval-manifest", required=True)
    parser.add_argument("--realsource-candidate-view", required=True)
    parser.add_argument("--realsource-cache-dir", required=True)
    parser.add_argument("--intervention-eval-manifest", required=True)
    parser.add_argument("--intervention-candidate-view", required=True)
    parser.add_argument("--intervention-dataset-root", required=True)
    parser.add_argument("--hq-eval-manifest", required=True)
    parser.add_argument("--hq-candidate-view", required=True)
    parser.add_argument("--hq-dataset-root", required=True)
    parser.add_argument(
        "--allow-gcs-download",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--gcs-download-timeout-seconds",
        type=int,
        default=900,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = build_union_contract(
        holdout_output=args.holdout_output,
        population_output=args.population_output,
        realsource_eval_manifest=args.realsource_eval_manifest,
        realsource_candidate_view=args.realsource_candidate_view,
        realsource_cache_dir=args.realsource_cache_dir,
        intervention_eval_manifest=args.intervention_eval_manifest,
        intervention_candidate_view=args.intervention_candidate_view,
        intervention_dataset_root=args.intervention_dataset_root,
        hq_eval_manifest=args.hq_eval_manifest,
        hq_candidate_view=args.hq_candidate_view,
        hq_dataset_root=args.hq_dataset_root,
        allow_gcs_download=args.allow_gcs_download,
        gcs_download_timeout_seconds=args.gcs_download_timeout_seconds,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
