from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import build_realman_dataset_views as builder
from scripts import h100_curriculum
from scripts.build_realman_dataset_views import (
    REALSOURCE_VIEW_SELECTION_ALGORITHM,
    RealSourceEpisode,
    RealSourceTaskCatalog,
    _align_realsource_annotations,
    build_realsource_canonical_view,
    _select_realsource_episodes,
    _target_20hz_row_count,
)


def _episode(task: int, episode_index: int, rows: int) -> RealSourceEpisode:
    digest = f"{task * 10_000 + episode_index + 1:064x}"
    return RealSourceEpisode(
        dataset_id=f"RealSourceData/RealSource-World/task_{task:02d}",
        sid=f"sid_{task:02d}",
        revision="main",
        episode_index=episode_index,
        length=max(1, (3 * rows) // 2),
        target_row_count=rows,
        data_file="data/chunk-000/file-000.parquet",
        annotation_ordinal=episode_index,
        annotation_episode_index=episode_index,
        annotation_sha256=digest,
        lineage_id=f"{task * 20_000 + episode_index + 1:064x}",
        content_id=f"{task * 30_000 + episode_index + 1:064x}",
    )


def _catalog(task: int, *, episodes: int, rows: int) -> RealSourceTaskCatalog:
    selected = tuple(
        _episode(task, episode_index, rows)
        for episode_index in range(episodes)
    )
    return RealSourceTaskCatalog(
        dataset_id=f"RealSourceData/RealSource-World/task_{task:02d}",
        sid=f"sid_{task:02d}",
        revision="main",
        fps=30,
        metadata_path=Path(f"/metadata/{task}"),
        metadata_sha256=f"{task + 1:064x}",
        annotation_path=Path(f"/annotations/{task}"),
        annotation_sha256=f"{task + 101:064x}",
        subtask_segments_path=Path(f"/subtask-segments/{task}"),
        subtask_segments_sha256=f"{task + 201:064x}",
        subtask_segments_summary={
            "schema": builder.SUBTASK_SEGMENTS_SCHEMA,
            "row_count": len(selected),
            "episode_count": len(selected),
            "usable_episode_count": len(selected),
            "zero_length_span_count": 0,
            "frame_coordinates": "raw_source_frames",
            "boundary_semantics": "start_inclusive_end_exclusive",
            "labels_synthesized": False,
        },
        alignment={"schema": "fixture"},
        catalog_episode_count=len(selected),
        catalog_raw_frame_count=sum(item.length for item in selected),
        quality_value_counts={"VALID": len(selected)},
        valid_episodes=selected,
    )


def test_exact_30_to_20_row_count_is_not_naive_ceil() -> None:
    assert [_target_20hz_row_count(length) for length in range(1, 7)] == [
        1,
        1,
        2,
        3,
        3,
        4,
    ]
    for length in range(1, 100):
        mapped = [
            target
            for target in range(length + 2)
            if (3 * target + 1) // 2 < length
        ]
        assert len(mapped) == _target_20hz_row_count(length)


def test_realsource_source_binding_changes_with_subtask_sidecar() -> None:
    catalog = _catalog(0, episodes=2, rows=7)
    first = builder._realsource_source_content_sha256(catalog)
    changed = builder._realsource_source_content_sha256(
        replace(catalog, subtask_segments_sha256="f" * 64)
    )

    assert first != changed


def test_collect_mail_extra_annotation_alignment_is_content_verified() -> None:
    episodes = [
        {"episode_index": index, "length": 1000 + index}
        for index in range(390)
    ]
    annotations = []
    for ordinal in range(391):
        if ordinal <= 121:
            data_index = ordinal
        elif ordinal == 122:
            data_index = None
        else:
            data_index = ordinal - 1
        annotations.append(
            {
                "episode_index": ordinal,
                "total_frames": (
                    777 if data_index is None else 1000 + data_index
                ),
                "quality_assessments": {"overall_valid": "VALID"},
            }
        )

    aligned, contract = _align_realsource_annotations(
        dataset_id="RealSourceData/RealSource-World/Collect_the_mail",
        episode_rows=episodes,
        annotations=annotations,
    )

    assert len(aligned) == 390
    assert aligned[121][1] == 121
    assert aligned[122][1] == 123
    assert aligned[-1][1] == 390
    assert contract["dropped_annotation_ordinal"] == 122
    assert "drop_ordinal_122" in contract["algorithm"]


def test_breadth_fraction_is_whole_episode_deterministic_and_all_tasks() -> None:
    # Two small tasks saturate under water-fill. The remaining task budgets
    # are redistributed evenly instead of sampling globally from the largest
    # task.
    catalogs = tuple(
        _catalog(
            task,
            episodes=(3 if task < 2 else 100),
            rows=(10 if task < 2 else 100),
        )
        for task in range(35)
    )
    first, first_budgets = _select_realsource_episodes(
        catalogs,
        fraction=Decimal("0.10"),
        seed=73,
    )
    second, second_budgets = _select_realsource_episodes(
        catalogs,
        fraction=Decimal("0.10"),
        seed=73,
    )

    assert first == second
    assert first_budgets == second_budgets
    assert len(first) == 35
    assert all(first[task.dataset_id] for task in catalogs)
    assert first_budgets[catalogs[0].dataset_id] == 30
    assert first_budgets[catalogs[1].dataset_id] == 30
    assert min(
        first_budgets[item.dataset_id] for item in catalogs[2:]
    ) >= 998
    assert all(
        tuple(sorted(ep.episode_index for ep in episodes))
        == tuple(ep.episode_index for ep in episodes)
        for episodes in first.values()
    )
    assert REALSOURCE_VIEW_SELECTION_ALGORITHM.endswith("_v1")


def _write_eval_manifest(
    path: Path,
    *,
    source_sha256: str,
    episode: RealSourceEpisode,
    configured_identities: tuple[
        tuple[str, str, str, str, int], ...
    ] | None = None,
) -> None:
    if configured_identities is None:
        configured_identities = (
            (
                episode.dataset_id,
                episode.sid,
                episode.revision,
                episode.data_file,
                episode.episode_index,
            ),
        )
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "purpose": "heldout",
                "source_manifest_sha256": source_sha256,
                "selection": {
                    "window_count": 1,
                    "holdout_episode_count": 1,
                    "configured_episode_count": len(
                        configured_identities
                    ),
                    "configured_episode_catalog_sha256": (
                        h100_curriculum._canonical_json_sha256(
                            sorted(configured_identities)
                        )
                    ),
                },
                "windows": [
                    {
                        "dataset_id": episode.dataset_id,
                        "sid": episode.sid,
                        "revision": episode.revision,
                        "data_file": episode.data_file,
                        "episode_index": episode.episode_index,
                        "base_index": 0,
                    }
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_realsource_train_view_requires_and_excludes_eval_holdout_and_copies(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    catalog = _catalog(0, episodes=3, rows=7)
    heldout = catalog.valid_episodes[0]
    content_copy = replace(
        catalog.valid_episodes[1],
        content_id=heldout.content_id,
    )
    catalog = replace(
        catalog,
        valid_episodes=(heldout, content_copy, catalog.valid_episodes[2]),
    )
    source_manifest = tmp_path / "catalog.jsonl.gz"
    source_manifest.write_bytes(b"immutable-catalog")
    source_sha256 = hashlib.sha256(source_manifest.read_bytes()).hexdigest()
    adapter = tmp_path / "adapter.yaml"
    adapter.write_text("fixture: true\n", encoding="utf-8")
    eval_manifest = tmp_path / "eval.json"
    _write_eval_manifest(
        eval_manifest,
        source_sha256=source_sha256,
        episode=heldout,
        configured_identities=(
            (
                heldout.dataset_id,
                heldout.sid,
                heldout.revision,
                heldout.data_file,
                heldout.episode_index,
            ),
            (
                catalog.valid_episodes[2].dataset_id,
                catalog.valid_episodes[2].sid,
                catalog.valid_episodes[2].revision,
                catalog.valid_episodes[2].data_file,
                catalog.valid_episodes[2].episode_index,
            ),
        ),
    )

    monkeypatch.setattr(
        builder,
        "_load_realsource_catalog",
        lambda **_: ([catalog], source_sha256, "a" * 64),
    )
    monkeypatch.setattr(
        builder,
        "_select_realsource_episodes",
        lambda *_args, **_kwargs: (
            {catalog.dataset_id: catalog.valid_episodes},
            {catalog.dataset_id: sum(
                episode.target_row_count
                for episode in catalog.valid_episodes
            )},
        ),
    )

    with pytest.raises(ValueError, match="eval_holdout_manifest"):
        build_realsource_canonical_view(
            canonical_manifest=source_manifest,
            adapter_path=adapter,
            cache_dir=tmp_path,
            output_manifest=tmp_path / "missing-holdout.json",
        )

    output = tmp_path / "train-view.json"
    built = build_realsource_canonical_view(
        canonical_manifest=source_manifest,
        adapter_path=adapter,
        cache_dir=tmp_path,
        eval_holdout_manifest=eval_manifest,
        output_manifest=output,
    )
    view = json.loads(output.read_text(encoding="utf-8"))
    assert built.row_count == 7
    assert view["rows"]["episode_count"] == 1
    assert view["selection"]["evaluation_holdout"][
        "selected_population_overlap_episode_count"
    ] == 2
    assert view["selection"]["evaluation_holdout"]["copy_detection"] == [
        "episode_identity",
        "episode_lineage_id",
        "episode_content_id",
    ]
    assert view["holdout_exclusions"]["content_ids"] == [
        heldout.content_id
    ]
    ledger = [
        json.loads(line)
        for line in built.ledger_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["episode_index"] for row in ledger] == [2]
    binding = h100_curriculum._validate_canonical_view_eval_holdout_binding(
        view_path=output,
        view=view,
        evaluation_path=eval_manifest,
        evaluation_sha256=hashlib.sha256(
            eval_manifest.read_bytes()
        ).hexdigest(),
        eligible_windows=7,
        stage_id="realsource_pretrain",
    )
    assert binding["episode_count"] == 1

    candidate_output = tmp_path / "statistics-candidate-view.json"
    candidate = build_realsource_canonical_view(
        canonical_manifest=source_manifest,
        adapter_path=adapter,
        cache_dir=tmp_path,
        eval_holdout_manifest=eval_manifest,
        output_manifest=candidate_output,
        statistics_population_candidate=True,
    )
    candidate_view = json.loads(
        candidate_output.read_text(encoding="utf-8")
    )
    assert candidate_view["purpose"] == (
        builder.dataset_view.STATISTICS_POPULATION_CANDIDATE_PURPOSE
    )
    assert candidate_view["usage_contract"]["training_allowed"] is False
    assert candidate.row_count == 14
    candidate_ledger = [
        json.loads(line)
        for line in candidate.ledger_path.read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    # Exact heldout identity is present for content authentication; its
    # differently indexed content copy remains excluded.
    assert [row["episode_index"] for row in candidate_ledger] == [0, 2]
    assert candidate_view["selection"][
        "authenticated_holdout_episode_indices_in_ledger"
    ] == [heldout.episode_index]


def test_realsource_eval_selection_candidate_is_nontrainable_and_needs_no_holdout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    catalog = _catalog(0, episodes=3, rows=7)
    source_manifest = tmp_path / "catalog.jsonl.gz"
    source_manifest.write_bytes(b"immutable-catalog")
    source_sha256 = hashlib.sha256(source_manifest.read_bytes()).hexdigest()
    adapter = tmp_path / "adapter.yaml"
    adapter.write_text("fixture: true\n", encoding="utf-8")
    monkeypatch.setattr(
        builder,
        "_load_realsource_catalog",
        lambda **_: ([catalog], source_sha256, "a" * 64),
    )
    monkeypatch.setattr(
        builder,
        "_select_realsource_episodes",
        lambda *_args, **_kwargs: (
            {catalog.dataset_id: catalog.valid_episodes},
            {
                catalog.dataset_id: sum(
                    episode.target_row_count
                    for episode in catalog.valid_episodes
                )
            },
        ),
    )

    output = tmp_path / "eval-selection-candidate.json"
    built = build_realsource_canonical_view(
        canonical_manifest=source_manifest,
        adapter_path=adapter,
        cache_dir=tmp_path,
        output_manifest=output,
        eval_selection_population_candidate=True,
    )
    view = json.loads(output.read_text(encoding="utf-8"))
    assert view["purpose"] == (
        builder.dataset_view.EVAL_SELECTION_POPULATION_CANDIDATE_PURPOSE
    )
    assert view["usage_contract"] == {
        "training_allowed": False,
        "eval_manifest_generation": True,
        "statistics_accumulation": (
            "forbidden_eval_selection_bootstrap_only"
        ),
    }
    assert view["holdout_exclusions"]["episode_indices"] == []
    assert view["sources"][0]["subtask_segments_sha256"] == (
        catalog.subtask_segments_sha256
    )
    assert view["sources"][0]["subtask_segments_schema"] == (
        builder.SUBTASK_SEGMENTS_SCHEMA
    )
    assert built.episode_count == 3
    assert built.row_count == 21

    with pytest.raises(
        ValueError, match="must not receive eval_holdout_manifest"
    ):
        build_realsource_canonical_view(
            canonical_manifest=source_manifest,
            adapter_path=adapter,
            cache_dir=tmp_path,
            eval_holdout_manifest=tmp_path / "not-used.json",
            output_manifest=tmp_path / "invalid.json",
            eval_selection_population_candidate=True,
        )


def test_realsource_cli_forwards_eval_selection_candidate_only_to_canonical(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, object] = {}

    def fake_build(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(to_dict=lambda: {"status": "ok"})

    monkeypatch.setattr(builder, "build_realsource_canonical_view", fake_build)
    assert (
        builder.main(
            [
                "realsource-canonical",
                "--output",
                str(tmp_path / "candidate.json"),
                "--eval-selection-population-candidate",
            ]
        )
        == 0
    )
    assert captured["eval_selection_population_candidate"] is True
    assert captured["statistics_population_candidate"] is False
    assert "eval_selection_population_candidate" not in vars(
        builder._parser().parse_args(
            [
                "intervention-incremental",
                "--dataset-root",
                str(tmp_path),
                "--output",
                str(tmp_path / "intervention.json"),
                "--eval-holdout-manifest",
                str(tmp_path / "eval.json"),
            ]
        )
    )
    assert builder.RESULT_PREFIX in capsys.readouterr().out


def test_curriculum_gate_rejects_empty_or_unbound_canonical_holdout(
    tmp_path: Path,
) -> None:
    eval_manifest = tmp_path / "eval.json"
    heldout = _episode(0, 0, 7)
    source_sha256 = "b" * 64
    _write_eval_manifest(
        eval_manifest,
        source_sha256=source_sha256,
        episode=heldout,
    )
    ledger = tmp_path / "rows.jsonl"
    ledger.write_text(
        json.dumps(
            {
                "dataset_id": heldout.dataset_id,
                "sid": heldout.sid,
                "revision": heldout.revision,
                "data_file": heldout.data_file,
                "episode_index": heldout.episode_index,
                "episode_lineage_id": heldout.lineage_id,
                "episode_content_id": heldout.content_id,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    view = {
        "selection": {},
        "holdout_exclusions": {
            "schema": "vla-dataset-view-holdout-exclusions-v1",
            "source_id": "gcs_realsource_strict_valid",
            "episode_indices": [],
            "lineage_ids": [],
            "content_ids": [],
            "sha256": "0" * 64,
        },
        "rows": {
            "path": ledger.name,
            "encoding": "expanded_rows_v1",
            "row_count": 1,
        },
    }
    with pytest.raises(
        h100_curriculum.CurriculumError,
        match="evaluation_holdout",
    ):
        h100_curriculum._validate_canonical_view_eval_holdout_binding(
            view_path=tmp_path / "view.json",
            view=view,
            evaluation_path=eval_manifest,
            evaluation_sha256=hashlib.sha256(
                eval_manifest.read_bytes()
            ).hexdigest(),
            eligible_windows=1,
            stage_id="realsource_pretrain",
        )
