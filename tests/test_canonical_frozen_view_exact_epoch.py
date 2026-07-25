from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from starVLA.action_representation import REALMAN_18D_ACTION_CONTRACT
from starVLA.dataloader.canonical_subset_dataset import (
    CanonicalSubsetVLADataset,
    EpisodeSpec,
    ShardSpec,
)
from starVLA.dataloader.dataset_view import (
    file_sha256,
    load_frozen_view,
    make_episode_content_id,
    make_episode_lineage_id,
    make_holdout_exclusions,
    make_sample_id,
    make_range_id,
    write_frozen_view,
    write_frozen_range_view,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _metadata_cache_key_dataset(
    tmp_path: Path,
) -> CanonicalSubsetVLADataset:
    manifest = tmp_path / "canonical-manifest.jsonl.gz"
    manifest.write_bytes(b"immutable canonical source manifest\n")
    adapter_dir = tmp_path / "adapters"
    adapter_dir.mkdir()
    (adapter_dir / "MANIFEST.json").write_text(
        '{"adapters":[]}\n',
        encoding="utf-8",
    )
    dataset = object.__new__(CanonicalSubsetVLADataset)
    dataset.manifest_path = manifest
    dataset.frozen_train_view_manifest_path = tmp_path / "view-a.json"
    dataset.frozen_train_view_manifest_sha256 = _sha("view-a")
    dataset.adapter_dir = adapter_dir
    dataset.cache_dir = tmp_path / "cache"
    dataset.bucket_root = "gs://fixture"
    dataset.dataset_id_list = []
    dataset.exclude_dataset_id_list = []
    dataset.exclude_dataset_ids_path_list = []
    dataset.exclude_sid_list = []
    dataset.exclude_sid_path_list = []
    dataset.adapter_group_ids = set()
    dataset.camera_slots = ["left", "right", "main"]
    dataset.qwen_camera_slots = ["main", "left", "right"]
    dataset.vjepa_camera_slots = ["left", "right", "main"]
    dataset.append_subtask_to_prompt = True
    dataset.preferred_fps = {30.0}
    dataset.allow_gcs_download = True
    dataset.max_shards = 0
    dataset.max_shards_per_dataset = 0
    dataset.max_windows = 0
    dataset.max_windows_per_dataset = 0
    dataset.canonical_eval_min_episodes_per_shard = 2
    dataset.sample_stride = 1
    dataset.video_horizon = 8
    dataset.action_horizon = 50
    dataset.action_type = "joint_delta_gripper_absolute"
    dataset.action_delta_anchor = "chunk_start_state"
    dataset.absolute_action_references = {"absolute_qpos"}
    dataset.action_sidecar_variant = "fixture"
    dataset.video_frame_stride = 1
    dataset.video_target_shift_steps = 0
    dataset.lazy_cache_shards = True
    dataset.index_windows_lazily = True
    return dataset


def test_metadata_cache_key_binds_frozen_view_path_and_sha256(
    tmp_path: Path,
) -> None:
    dataset = _metadata_cache_key_dataset(tmp_path)
    first = dataset._build_metadata_index_cache_key()

    dataset.frozen_train_view_manifest_path = tmp_path / "view-b.json"
    second = dataset._build_metadata_index_cache_key()

    dataset.frozen_train_view_manifest_path = tmp_path / "view-a.json"
    dataset.frozen_train_view_manifest_sha256 = _sha("view-b")
    third = dataset._build_metadata_index_cache_key()

    assert first != second
    assert first != third
    assert second != third


def _build_view(
    tmp_path: Path,
    *,
    source_manifest_sha256: str,
    row_sids: tuple[str, ...] = ("sid0", "sid0", "sid0"),
    purpose: str | None = None,
) -> Path:
    lineage = make_episode_lineage_id(
        backend="canonical",
        source_id="source0",
        catalog_sha256=_sha("catalog"),
        episode_index=7,
        length=100,
        episode_metadata_sha256=_sha("episode-metadata"),
    )
    content = make_episode_content_id(
        frame_content_sha256=_sha("frames"),
        length=100,
        content_contract="canonical-test-v1",
    )
    representation_sha = REALMAN_18D_ACTION_CONTRACT.sha256()
    rows = []
    for base_index, sid in enumerate(row_sids):
        rows.append(
            {
                "backend": "canonical",
                "source_id": "source0",
                "dataset_id": "realman",
                "sid": sid,
                "revision": "rev0",
                "data_file": "data/chunk-000/file-000.parquet",
                "adapter_sha256": _sha("adapter"),
                "episode_index": 7,
                "base_index": base_index,
                "horizon": 50,
                "target_fps": 20,
                "episode_lineage_id": lineage,
                "episode_content_id": content,
                "sample_id": make_sample_id(
                    episode_content_id=content,
                    base_index=base_index,
                    horizon=50,
                    target_fps=20,
                    representation_contract_sha256=representation_sha,
                    end_clamp=True,
                ),
                "end_clamp_policy": "repeat_last",
            }
        )
    manifest = tmp_path / "view.json"
    write_frozen_view(
        manifest,
        descriptor={
            "view_name": "canonical-test",
            **({"purpose": purpose} if purpose is not None else {}),
            **(
                {
                    "usage_contract": {
                        "training_allowed": False,
                        "eval_manifest_generation": True,
                        "statistics_accumulation": (
                            "forbidden_eval_selection_bootstrap_only"
                        ),
                    }
                }
                if purpose == "eval_selection_population_candidate"
                else {}
            ),
            "sources": [
                {
                    "source_id": "source0",
                    "backend": "canonical",
                    "catalog_sha256": _sha("catalog"),
                    "annotation_sha256": _sha("annotations"),
                    "source_content_sha256": _sha("source-content"),
                    "manifest_sha256": source_manifest_sha256,
                    "dataset_id": "realman",
                    "sid": "sid0",
                    "revision": "rev0",
                }
            ],
            "representation": {
                "contract_sha256": representation_sha,
                "state_dim": 18,
                "action_dim": 18,
                "horizon": 50,
                "target_fps": 20,
                "action_type": "joint_delta_gripper_absolute",
                "normalization": "q01_q99_unclipped",
            },
            "epoch_contract": {
                "mode": "all_exhaustive",
                "epoch_passes": 1,
                "drop_last": False,
                "replacement": False,
                "shuffle": "deterministic_bijection_per_epoch",
                "ddp_tail": "duplicated_padding_reported_separately",
            },
            "holdout_exclusions": make_holdout_exclusions(
                episode_indices=[],
                lineage_ids=[],
                content_ids=[],
                source_id="source0",
            ),
            "selection": {"algorithm": "test-all-rows-v1"},
        },
        rows=rows,
        overwrite=True,
    )
    return manifest


def test_canonical_training_rejects_statistics_population_candidate(
    tmp_path: Path,
) -> None:
    source_manifest = tmp_path / "canonical-manifest.jsonl.gz"
    source_manifest.write_bytes(b"immutable canonical source manifest\n")
    manifest = _build_view(
        tmp_path,
        source_manifest_sha256=file_sha256(source_manifest),
        purpose="statistics_population_candidate",
    )
    dataset = object.__new__(CanonicalSubsetVLADataset)
    dataset.epoch_sampling_strategy = "all_sources_exhaustive"
    dataset.normalization_statistics = {"selected": {}}
    dataset.allow_eval_selection_population_candidate = False
    dataset.manifest_path = source_manifest
    with pytest.raises(
        ValueError, match="statistics_population_candidate"
    ):
        dataset._load_frozen_train_view_descriptor(
            manifest,
            expected_manifest_sha256=file_sha256(manifest),
        )


def test_canonical_training_rejects_eval_selection_candidate_unless_generator(
    tmp_path: Path,
) -> None:
    source_manifest = tmp_path / "canonical-manifest.jsonl.gz"
    source_manifest.write_bytes(b"immutable canonical source manifest\n")
    manifest = _build_view(
        tmp_path,
        source_manifest_sha256=file_sha256(source_manifest),
        purpose="eval_selection_population_candidate",
    )
    dataset = object.__new__(CanonicalSubsetVLADataset)
    dataset.epoch_sampling_strategy = "all_sources_exhaustive"
    dataset.normalization_statistics = None
    dataset.allow_eval_selection_population_candidate = False
    dataset.manifest_path = source_manifest
    with pytest.raises(
        ValueError, match="eval_selection_population_candidate"
    ):
        dataset._load_frozen_train_view_descriptor(
            manifest,
            expected_manifest_sha256=file_sha256(manifest),
        )

    # The only permitted consumer is the eval generator's explicit code path.
    dataset.allow_eval_selection_population_candidate = True
    dataset._load_frozen_train_view_descriptor(
        manifest,
        expected_manifest_sha256=file_sha256(manifest),
    )
    assert dataset.frozen_train_view is not None


def _dataset(
    tmp_path: Path,
    *,
    row_sids: tuple[str, ...] = ("sid0", "sid0", "sid0"),
) -> CanonicalSubsetVLADataset:
    source_manifest = tmp_path / "canonical-manifest.jsonl.gz"
    source_manifest.write_bytes(b"immutable canonical source manifest\n")
    manifest = _build_view(
        tmp_path,
        source_manifest_sha256=file_sha256(source_manifest),
        row_sids=row_sids,
    )
    episode = EpisodeSpec(
        local_start=0,
        length=100,
        task="move the chain",
        video_paths={},
        video_base_frames={},
        episode_index=7,
    )
    shard = ShardSpec(
        dataset_id="realman",
        sid="sid0",
        revision="rev0",
        adapter_group_id="realman",
        adapter_path=tmp_path / "adapter.yaml",
        root=tmp_path,
        gcs_prefix="gs://test",
        data_relative_path="data/chunk-000/file-000.parquet",
        data_path=tmp_path / "data.parquet",
        sidecar_path=tmp_path / "sidecar.npz",
        fps=30.0,
        camera_source_keys={},
        qwen_camera_slots=(),
        vjepa_camera_slots=(),
        decode_camera_slots=(),
        task_map={0: "move the chain"},
        episodes=[episode],
        adapter_sha256=_sha("adapter"),
    )

    dataset = object.__new__(CanonicalSubsetVLADataset)
    dataset.data_cfg = {
        "frozen_train_view_index_cache_dir": str(
            tmp_path / "frozen-index"
        )
    }
    dataset.metadata_index_cache_dir = tmp_path / "metadata-index"
    dataset.manifest_path = source_manifest
    dataset.adapter_contract_sha256 = _sha("adapter-contract")
    dataset._metadata_index_cache_key = _sha("metadata-index")
    dataset.shards = [shard]
    dataset.append_subtask_to_prompt = False
    dataset.subtask_prompt_source_column = "subtask"
    dataset.subtask_prompt_label_column = "subtask_label"
    dataset.frozen_train_view = load_frozen_view(
        manifest, verify_ledger=False
    )
    dataset.frozen_train_view_manifest_path = manifest
    dataset.frozen_train_view_manifest_sha256 = file_sha256(manifest)
    dataset.frozen_train_view_index_path = None
    dataset.frozen_train_view_index_metadata_path = None
    dataset.frozen_train_view_index_sha256 = None
    dataset.frozen_train_view_identity_catalog_sha256 = None
    dataset._frozen_view_offsets = None
    dataset._frozen_view_ledger_handle = None
    dataset._frozen_view_episode_lookup = {}
    dataset.mode = "train"
    dataset.epoch_sampling_strategy = "all_sources_exhaustive"
    dataset.epoch_sampling_algorithm_version = (
        "all_sources_exhaustive_frozen_view_affine_v1"
    )
    dataset.seed = 17
    dataset.fail_on_sample_error = True
    dataset.index_windows_lazily = True
    dataset.total_windows = dataset.frozen_train_view.row_count
    dataset.windows = []
    dataset._window_ranges = []
    dataset._window_range_ends = []
    dataset._action_offsets = np.arange(50, dtype=np.int64)
    dataset.video_horizon = 4
    dataset.video_frame_stride = 1
    dataset.video_target_shift_steps = 0
    dataset._compact_offsets_cache = None
    dataset._initialize_frozen_train_view_index()
    dataset._initialize_epoch_schedule()
    return dataset


def _range_dataset(
    tmp_path: Path,
    *,
    selected_episode_count: int,
) -> CanonicalSubsetVLADataset:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source_manifest = tmp_path / "canonical-range-manifest.jsonl.gz"
    source_manifest.write_bytes(b"immutable compact canonical catalog\n")
    source_manifest_sha256 = file_sha256(source_manifest)
    representation_sha = REALMAN_18D_ACTION_CONTRACT.sha256()
    annotation_sha256 = _sha("annotation-catalog")
    ranges = []
    episodes = []
    target_count = (2 * 10 + 1) // 3
    for episode_index in range(selected_episode_count):
        lineage = make_episode_lineage_id(
            backend="canonical",
            source_id="source0",
            catalog_sha256=_sha("range-catalog"),
            episode_index=episode_index,
            length=10,
            episode_metadata_sha256=_sha(
                f"range-episode-{episode_index}"
            ),
        )
        content = make_episode_content_id(
            frame_content_sha256=_sha(
                f"range-content-{episode_index}"
            ),
            length=10,
            content_contract="canonical-range-test-v1",
        )
        ranges.append(
            {
                "backend": "canonical",
                "source_id": "source0",
                "dataset_id": "realman",
                "sid": "sid0",
                "revision": "rev0",
                "data_file": "data/chunk-000/file-000.parquet",
                "adapter_sha256": _sha("adapter"),
                "episode_index": episode_index,
                "base_start": 0,
                "base_stop": target_count,
                "base_step": 1,
                "sample_count": target_count,
                "horizon": 50,
                "target_fps": 20,
                "source_fps": 30.0,
                "source_episode_length": 10,
                "annotation_ordinal": episode_index,
                "annotation_episode_index": episode_index,
                "annotation_sha256": annotation_sha256,
                "selection_kind": "whole_episode_fraction",
                "episode_lineage_id": lineage,
                "episode_content_id": content,
                "range_id": make_range_id(
                    episode_content_id=content,
                    base_start=0,
                    base_stop=target_count,
                    base_step=1,
                    horizon=50,
                    target_fps=20,
                    representation_contract_sha256=representation_sha,
                    end_clamp=True,
                ),
                "end_clamp_policy": "repeat_last",
            }
        )
        episodes.append(
            EpisodeSpec(
                local_start=episode_index * 10,
                length=10,
                task="move the chain",
                video_paths={},
                video_base_frames={},
                episode_index=episode_index,
            )
        )

    manifest = tmp_path / "range-view.json"
    write_frozen_range_view(
        manifest,
        descriptor={
            "view_name": (
                f"canonical-{selected_episode_count}-episode-range-test"
            ),
            "sources": [
                {
                    "source_id": "source0",
                    "backend": "canonical",
                    "catalog_sha256": _sha("range-catalog"),
                    "annotation_sha256": annotation_sha256,
                    "source_content_sha256": _sha(
                        "range-source-content"
                    ),
                    "manifest_sha256": source_manifest_sha256,
                    "dataset_id": "realman",
                    "sid": "sid0",
                    "revision": "rev0",
                }
            ],
            "representation": {
                "contract_sha256": representation_sha,
                "state_dim": 18,
                "action_dim": 18,
                "horizon": 50,
                "target_fps": 20,
                "action_type": "joint_delta_gripper_absolute",
                "normalization": "q01_q99_unclipped",
            },
            "epoch_contract": {
                "mode": "all_exhaustive",
                "epoch_passes": 1,
                "drop_last": False,
                "replacement": False,
                "shuffle": "deterministic_bijection_per_epoch",
                "ddp_tail": "duplicated_padding_reported_separately",
            },
            "holdout_exclusions": make_holdout_exclusions(
                episode_indices=[],
                lineage_ids=[],
                content_ids=[],
                source_id="source0",
            ),
            "selection": {
                "algorithm": "whole_episode_fraction_test_v1",
                "selected_episode_count": selected_episode_count,
            },
        },
        ranges=ranges,
        overwrite=True,
    )
    # Exercise the dependency-light verifier before the training loader.
    loaded = load_frozen_view(manifest, verify_ledger=True)
    assert loaded.record_count == selected_episode_count
    assert loaded.row_count == selected_episode_count * target_count

    shard = ShardSpec(
        dataset_id="realman",
        sid="sid0",
        revision="rev0",
        adapter_group_id="realman",
        adapter_path=tmp_path / "adapter.yaml",
        root=tmp_path,
        gcs_prefix="gs://test",
        data_relative_path="data/chunk-000/file-000.parquet",
        data_path=tmp_path / "data.parquet",
        sidecar_path=tmp_path / "sidecar.npz",
        fps=30.0,
        camera_source_keys={},
        qwen_camera_slots=(),
        vjepa_camera_slots=(),
        decode_camera_slots=(),
        task_map={0: "move the chain"},
        episodes=episodes,
        adapter_sha256=_sha("adapter"),
    )
    dataset = object.__new__(CanonicalSubsetVLADataset)
    dataset.data_cfg = {
        "frozen_train_view_index_cache_dir": str(
            tmp_path / "range-index"
        )
    }
    dataset.metadata_index_cache_dir = tmp_path / "metadata-index"
    dataset.manifest_path = source_manifest
    dataset.adapter_contract_sha256 = _sha("adapter-contract")
    dataset._metadata_index_cache_key = _sha("metadata-index")
    dataset.shards = [shard]
    dataset.append_subtask_to_prompt = False
    dataset.subtask_prompt_source_column = "subtask"
    dataset.subtask_prompt_label_column = "subtask_label"
    dataset.frozen_train_view = loaded
    dataset.frozen_train_view_manifest_path = manifest
    dataset.frozen_train_view_manifest_sha256 = file_sha256(manifest)
    dataset.frozen_train_view_index_path = None
    dataset.frozen_train_view_index_metadata_path = None
    dataset.frozen_train_view_index_sha256 = None
    dataset.frozen_train_view_identity_catalog_sha256 = None
    dataset._frozen_view_offsets = None
    dataset._frozen_view_ledger_handle = None
    dataset._frozen_view_episode_lookup = {}
    dataset.mode = "train"
    dataset.epoch_sampling_strategy = "all_sources_exhaustive"
    dataset.epoch_sampling_algorithm_version = (
        "all_sources_exhaustive_frozen_view_affine_v1"
    )
    dataset.seed = 23
    dataset.fail_on_sample_error = True
    dataset.index_windows_lazily = True
    dataset.total_windows = loaded.row_count
    dataset.windows = []
    dataset._window_ranges = []
    dataset._window_range_ends = []
    dataset._action_offsets = np.arange(50, dtype=np.int64)
    dataset.video_horizon = 4
    dataset.video_frame_stride = 1
    dataset.video_target_shift_steps = 0
    dataset._compact_offsets_cache = None
    dataset._initialize_frozen_train_view_index()
    dataset._initialize_epoch_schedule()
    return dataset


def test_frozen_canonical_epoch_is_lazy_and_exhaustive(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path)

    assert dataset.windows == []
    assert dataset._window_ranges == []
    assert dataset.frozen_train_view_index_path.stat().st_size == 4 * 8
    assert dataset._frozen_view_row(1)["base_index"] == 1

    epoch_zero = [
        dataset._frozen_view_row(dataset._epoch_window_index(index))[
            "ordinal"
        ]
        for index in range(len(dataset))
    ]
    dataset.set_epoch(1)
    epoch_one = [
        dataset._frozen_view_row(dataset._epoch_window_index(index))[
            "ordinal"
        ]
        for index in range(len(dataset))
    ]

    assert sorted(epoch_zero) == [0, 1, 2]
    assert sorted(epoch_one) == [0, 1, 2]
    assert epoch_zero != epoch_one


def test_frozen_canonical_maps_target_fps_without_materializing_schedule(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path)
    captured = {}

    def capture(window, **kwargs):
        captured["window"] = window
        captured.update(kwargs)
        return {"window": window}

    dataset._sample_context_for_window = capture
    context = dataset._sample_context_for_frozen_row(
        dataset._frozen_view_row(1)
    )

    assert context["frozen_view_ordinal"] == 1
    assert captured["window"].base_index == 1
    assert captured["source_base_index"] == 2
    np.testing.assert_array_equal(
        captured["action_episode_indices"][:4],
        [2, 3, 5, 6],
    )
    np.testing.assert_array_equal(
        captured["compact_episode_indices"],
        [2, 3, 5, 6],
    )


def test_frozen_canonical_identity_mismatch_fails_closed(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="does not match its source"):
        _dataset(
            tmp_path,
            row_sids=("sid0", "wrong-sid", "sid0"),
        )


def test_frozen_canonical_corrupt_cached_offset_index_fails_closed(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path)
    index_path = dataset.frozen_train_view_index_path
    dataset.close_video_readers()
    payload = bytearray(index_path.read_bytes())
    payload[8] ^= 1
    index_path.write_bytes(payload)

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        _dataset(tmp_path)


def test_frozen_canonical_offset_cache_isolated_by_adapter_contract(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path)
    first_path = dataset.frozen_train_view_index_path
    ledger_sha256 = str(
        dataset.frozen_train_view.descriptor["rows"]["sha256"]
    )

    dataset.adapter_contract_sha256 = _sha("new-adapter-contract")
    second_path, _, _ = dataset._frozen_index_cache_paths(ledger_sha256)

    assert first_path != second_path
    assert dataset.adapter_contract_sha256[:16] in second_path.name


def test_frozen_view_and_index_hashes_bind_resume_provenance(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path)
    provenance = dataset.dataset_provenance()["frozen_train_view"]

    assert provenance["row_count"] == 3
    assert provenance["view_id"] == dataset.frozen_train_view.view_id
    assert provenance["manifest_sha256"] == file_sha256(
        dataset.frozen_train_view_manifest_path
    )
    assert provenance["ledger_sha256"] == file_sha256(
        dataset.frozen_train_view.ledger_path
    )
    assert provenance["offset_index_sha256"] == file_sha256(
        dataset.frozen_train_view_index_path
    )
    assert len(provenance["identity_catalog_sha256"]) == 64


@pytest.mark.parametrize(
    ("selection_fraction", "selected_episode_count"),
    ((0.10, 1), (0.50, 5)),
)
def test_compact_range_views_expand_exact_10_and_50_percent_epochs(
    tmp_path: Path,
    selection_fraction: float,
    selected_episode_count: int,
) -> None:
    dataset = _range_dataset(
        tmp_path / f"selection-{selection_fraction}",
        selected_episode_count=selected_episode_count,
    )
    samples_per_episode = 7
    logical_count = selected_episode_count * samples_per_episode

    assert dataset.frozen_train_view.encoding == "episode_ranges_v1"
    assert dataset.frozen_train_view.record_count == selected_episode_count
    assert len(dataset) == logical_count
    assert dataset.windows == []
    assert dataset._window_ranges == []
    assert dataset.frozen_train_view_index_path.stat().st_size == (
        (selected_episode_count + 1) * 2 * 8
    )

    # Random access crosses a compact range boundary without expanding the
    # range ledger in memory.
    assert dataset._frozen_view_row(0)["base_index"] == 0
    assert dataset._frozen_view_row(6)["base_index"] == 6
    if selected_episode_count > 1:
        next_episode = dataset._frozen_view_row(7)
        assert next_episode["episode_index"] == 1
        assert next_episode["base_index"] == 0

    epoch_zero = [
        dataset._frozen_view_row(dataset._epoch_window_index(index))[
            "sample_id"
        ]
        for index in range(len(dataset))
    ]
    dataset.set_epoch(1)
    epoch_one = [
        dataset._frozen_view_row(dataset._epoch_window_index(index))[
            "sample_id"
        ]
        for index in range(len(dataset))
    ]
    assert len(set(epoch_zero)) == logical_count
    assert set(epoch_one) == set(epoch_zero)
    assert epoch_one != epoch_zero

    provenance = dataset.dataset_provenance()["frozen_train_view"]
    assert provenance["ledger_encoding"] == "episode_ranges_v1"
    assert provenance["record_count"] == selected_episode_count
    assert provenance["row_count"] == logical_count
    assert provenance["offset_index_sha256"] == file_sha256(
        dataset.frozen_train_view_index_path
    )


def test_compact_range_cached_cumulative_index_corruption_fails_closed(
    tmp_path: Path,
) -> None:
    dataset = _range_dataset(
        tmp_path,
        selected_episode_count=5,
    )
    index_path = dataset.frozen_train_view_index_path
    dataset.close_video_readers()
    payload = bytearray(index_path.read_bytes())
    # Second uint64 column is the cumulative logical count.
    payload[8] ^= 1
    index_path.write_bytes(payload)

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        _range_dataset(tmp_path, selected_episode_count=5)
