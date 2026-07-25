import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from starVLA.dataloader.canonical_eval_manifest import (
    build_canonical_eval_manifest_payload,
    write_canonical_eval_manifest,
)
from starVLA.dataloader.canonical_subset_dataset import (
    ACTION_DIM,
    CANONICAL_EVAL_SELECTION_ALGORITHM,
    SHARD_Q01_Q99_UNCLIPPED,
    STATE_DIM,
    CanonicalEvalManifest,
    CanonicalEvalSelection,
    CanonicalEvalWindow,
    CanonicalSubsetVLADataset,
    DeterministicCanonicalEvalDataset,
    EpisodeSpec,
    JOINT_DELTA_GRIPPER_ABSOLUTE,
    ShardSpec,
    _canonical_eval_metric_groups,
    canonical_action_sidecar_variant,
    canonical_adapter_contract_sha256,
    load_canonical_eval_manifest,
)
from starVLA.training.train_starvla import (
    VLATrainer,
    _validate_canonical_checkpoint_selection_metric,
    _validate_heldout_report_coverage,
    prepare_heldout_eval_data,
)
from scripts import h100_training


def _selection(
    *,
    window_count: int = 1,
    seed: int = 42,
    candidate_count: int = 32,
    action_horizon: int = 2,
    action_dim: int = ACTION_DIM,
    configured_episode_count: int = 2,
    configured_episode_catalog_sha256: str = "b" * 64,
) -> CanonicalEvalSelection:
    return CanonicalEvalSelection(
        algorithm=CANONICAL_EVAL_SELECTION_ALGORITHM,
        seed=seed,
        window_count=window_count,
        candidate_count=candidate_count,
        action_horizon=action_horizon,
        action_dim=action_dim,
        action_type=JOINT_DELTA_GRIPPER_ABSOLUTE,
        normalization=SHARD_Q01_Q99_UNCLIPPED,
        adapter_contract_sha256="c" * 64,
        action_sidecar_variant="a" * 16,
        configured_episode_count=configured_episode_count,
        configured_episode_catalog_sha256=(
            configured_episode_catalog_sha256
        ),
    )


def _selection_payload(**overrides) -> dict:
    payload = vars(_selection())
    payload.update(overrides)
    return payload


def _episode(local_start: int, episode_index: int) -> EpisodeSpec:
    return EpisodeSpec(
        local_start=local_start,
        length=2,
        task="move the object",
        video_paths={},
        video_base_frames={},
        episode_index=episode_index,
        dataset_from_index=local_start,
    )


def _shard(tmp_path: Path) -> ShardSpec:
    return ShardSpec(
        dataset_id="fixture/canonical",
        sid="sid-1",
        revision="r1",
        adapter_group_id="fixture",
        adapter_path=tmp_path / "adapter.yaml",
        root=tmp_path,
        gcs_prefix="gs://fixture",
        data_relative_path="data/chunk.parquet",
        data_path=tmp_path / "data/chunk.parquet",
        sidecar_path=tmp_path / "sidecar.npz",
        fps=30.0,
        camera_source_keys={},
        qwen_camera_slots=(),
        vjepa_camera_slots=(),
        decode_camera_slots=(),
        task_map={0: "move the object"},
        episodes=[_episode(0, 10), _episode(2, 20)],
    )


def _source(tmp_path: Path) -> CanonicalSubsetVLADataset:
    source = object.__new__(CanonicalSubsetVLADataset)
    source.mode = "eval"
    source.shards = [_shard(tmp_path)]
    source.action_horizon = 2
    source._action_offsets = np.arange(2, dtype=np.int64)
    source.action_type = JOINT_DELTA_GRIPPER_ABSOLUTE
    source.sidecar_normalization = SHARD_Q01_Q99_UNCLIPPED
    source.adapter_contract_sha256 = "c" * 64
    source.action_sidecar_variant = "a" * 16
    source.data_cfg = {}
    source._full_episode_identities = source._episode_identity_set(source.shards)

    state = np.zeros((4, STATE_DIM), dtype=np.float32)
    state_mask = np.zeros_like(state, dtype=bool)
    state_mask[:, 0] = True
    action = np.zeros((4, ACTION_DIM), dtype=np.float32)
    action_mask = np.zeros_like(action, dtype=bool)
    action_mask[:, 0] = True
    delta_mask = np.zeros((ACTION_DIM,), dtype=bool)
    delta_mask[0] = True
    action_to_state = np.full((ACTION_DIM,), -1, dtype=np.int64)
    action_to_state[0] = 0
    shard_data = SimpleNamespace(
        state=state,
        state_mask=state_mask,
        action=action,
        action_mask=action_mask,
        state_low=np.full((STATE_DIM,), -1.0, dtype=np.float32),
        state_high=np.full((STATE_DIM,), 1.0, dtype=np.float32),
        action_low=np.full((ACTION_DIM,), -1.0, dtype=np.float32),
        action_high=np.full((ACTION_DIM,), 1.0, dtype=np.float32),
        action_delta_mask=delta_mask,
        action_to_state_indices=action_to_state,
    )
    source._get_shard_data = lambda _index: shard_data
    source.close_video_readers = lambda: None
    heldout_window = CanonicalEvalWindow(
        dataset_id="fixture/canonical",
        sid="sid-1",
        revision="r1",
        data_file="data/chunk.parquet",
        episode_index=20,
        base_index=0,
    )
    source.canonical_eval_manifest = CanonicalEvalManifest(
        path=tmp_path / "heldout.json",
        sha256="a" * 64,
        purpose="heldout",
        source_manifest_sha256="b" * 64,
        selection=_selection(),
        windows=(heldout_window,),
    )

    def sample_context(window):
        return {
            "window": window,
            "shard": source.shards[window.shard_index],
            "episode": source.shards[window.shard_index].episodes[
                window.episode_index
            ],
            "shard_data": shard_data,
            "row_base": 2,
        }

    source._sample_context_for_window = sample_context
    source._sample_from_context = lambda _context: {
        "action": np.zeros((2, ACTION_DIM), dtype=np.float32),
        "action_mask": action_mask[2:4].copy(),
        "action_is_pad": np.zeros((2,), dtype=bool),
        "state": np.zeros((1, STATE_DIM), dtype=np.float32),
    }
    return source


def _many_episode_source(
    tmp_path: Path,
    *,
    episode_count: int,
    episode_length: int = 6,
) -> CanonicalSubsetVLADataset:
    source = object.__new__(CanonicalSubsetVLADataset)
    source.mode = "eval"
    shard = _shard(tmp_path)
    shard.episodes = [
        EpisodeSpec(
            local_start=index * episode_length,
            length=episode_length,
            task="move the object",
            video_paths={},
            video_base_frames={},
            episode_index=index,
            dataset_from_index=index * episode_length,
        )
        for index in range(episode_count)
    ]
    source.shards = [shard]
    source.action_horizon = 2
    source._action_offsets = np.arange(2, dtype=np.int64)
    source.action_type = JOINT_DELTA_GRIPPER_ABSOLUTE
    source.sidecar_normalization = SHARD_Q01_Q99_UNCLIPPED
    source.adapter_contract_sha256 = "c" * 64
    source.action_sidecar_variant = "a" * 16
    source.data_cfg = {}
    source.canonical_eval_manifest = None
    source._full_episode_identities = source._episode_identity_set(
        source.shards
    )
    row_count = episode_count * episode_length
    state = np.zeros((row_count, STATE_DIM), dtype=np.float32)
    state_mask = np.ones_like(state, dtype=bool)
    action = np.zeros((row_count, ACTION_DIM), dtype=np.float32)
    action_mask = np.ones_like(action, dtype=bool)
    delta_mask = np.zeros((ACTION_DIM,), dtype=bool)
    action_to_state = np.full((ACTION_DIM,), -1, dtype=np.int64)
    shard_data = SimpleNamespace(
        state=state,
        state_mask=state_mask,
        action=action,
        action_mask=action_mask,
        state_low=np.full((STATE_DIM,), -1.0, dtype=np.float32),
        state_high=np.full((STATE_DIM,), 1.0, dtype=np.float32),
        action_low=np.full((ACTION_DIM,), -1.0, dtype=np.float32),
        action_high=np.full((ACTION_DIM,), 1.0, dtype=np.float32),
        action_delta_mask=delta_mask,
        action_to_state_indices=action_to_state,
    )
    source._get_shard_data = lambda _index: shard_data
    source.close_video_readers = lambda: None
    source.manifest_path = tmp_path / "catalog.jsonl.gz"
    source.manifest_path.write_bytes(b"immutable canonical catalog")
    return source


def test_canonical_eval_manifest_is_bound_and_one_window_per_episode(tmp_path):
    source_manifest = tmp_path / "catalog.jsonl.gz"
    source_manifest.write_bytes(b"immutable catalog")
    source_sha = hashlib.sha256(source_manifest.read_bytes()).hexdigest()
    window = {
        "dataset_id": "fixture/canonical",
        "sid": "sid-1",
        "revision": "r1",
        "data_file": "data/chunk.parquet",
        "episode_index": 20,
        "base_index": 0,
    }
    eval_manifest = tmp_path / "eval.json"
    eval_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "purpose": "heldout",
                "source_manifest_sha256": source_sha,
                "selection": _selection_payload(),
                "windows": [window],
            }
        ),
        encoding="utf-8",
    )

    loaded = load_canonical_eval_manifest(
        eval_manifest,
        source_manifest_path=source_manifest,
    )

    assert loaded.source_manifest_sha256 == source_sha
    assert loaded.windows[0].episode_index == 20
    assert loaded.selection.seed == 42

    payload = json.loads(eval_manifest.read_text(encoding="utf-8"))
    payload["windows"].append({**window, "base_index": 1})
    eval_manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one deterministic window"):
        load_canonical_eval_manifest(
            eval_manifest,
            source_manifest_path=source_manifest,
        )


@pytest.mark.parametrize(
    ("field", "manifest_value", "expected_value"),
    [
        ("seed", 43, 42),
        ("candidate_count", 31, 32),
        ("action_horizon", 49, 50),
        ("action_dim", 48, ACTION_DIM),
        ("adapter_contract_sha256", "d" * 64, "c" * 64),
    ],
)
def test_canonical_eval_manifest_rejects_stale_selection_contract(
    tmp_path,
    field,
    manifest_value,
    expected_value,
):
    source_manifest = tmp_path / "catalog.jsonl.gz"
    source_manifest.write_bytes(b"immutable catalog")
    selection = _selection_payload(action_horizon=50)
    selection[field] = manifest_value
    eval_manifest = tmp_path / "eval.json"
    eval_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "purpose": "heldout",
                "source_manifest_sha256": hashlib.sha256(
                    source_manifest.read_bytes()
                ).hexdigest(),
                "selection": selection,
                "windows": [
                    {
                        "dataset_id": "fixture/canonical",
                        "sid": "sid-1",
                        "revision": "r1",
                        "data_file": "data/chunk.parquet",
                        "episode_index": 20,
                        "base_index": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="selection contract"):
        load_canonical_eval_manifest(
            eval_manifest,
            source_manifest_path=source_manifest,
            expected_selection={
                field: expected_value,
            },
        )


def test_canonical_eval_manifest_rejects_stale_window_count(tmp_path):
    source_manifest = tmp_path / "catalog.jsonl.gz"
    source_manifest.write_bytes(b"immutable catalog")
    eval_manifest = tmp_path / "eval.json"
    eval_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "purpose": "heldout",
                "source_manifest_sha256": hashlib.sha256(
                    source_manifest.read_bytes()
                ).hexdigest(),
                "selection": _selection_payload(window_count=2),
                "windows": [
                    {
                        "dataset_id": "fixture/canonical",
                        "sid": "sid-1",
                        "revision": "r1",
                        "data_file": "data/chunk.parquet",
                        "episode_index": 20,
                        "base_index": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="window_count does not match"):
        load_canonical_eval_manifest(
            eval_manifest,
            source_manifest_path=source_manifest,
        )


def test_adapter_semantic_drift_changes_canonical_sidecar_contract(tmp_path):
    canonical_root = tmp_path / "dataset-canonicalization"
    adapter_dir = canonical_root / "configs/dataset_adapters"
    semantic_dir = canonical_root / "src/model_v0/data"
    adapter_dir.mkdir(parents=True)
    semantic_dir.mkdir(parents=True)
    (adapter_dir / "MANIFEST.json").write_text(
        '{"adapters":[{"path":"adapter.json"}]}',
        encoding="utf-8",
    )
    adapter_path = adapter_dir / "adapter.json"
    adapter_path.write_text('{"scale":1}', encoding="utf-8")
    (semantic_dir / "adapters.py").write_text(
        "def apply_unified_adapter(): pass\n",
        encoding="utf-8",
    )
    (semantic_dir / "unified_schema.py").write_text(
        "STATE_DIM = 53\nACTION_DIM = 49\n",
        encoding="utf-8",
    )
    data_cfg = {
        "dataset_canonicalization_root": str(canonical_root),
        "adapter_dir": str(adapter_dir),
        "action_type": JOINT_DELTA_GRIPPER_ABSOLUTE,
        "action_delta_anchor": "chunk_start_state",
        "gripper_action_type": "absolute",
        "sidecar_normalization": SHARD_Q01_Q99_UNCLIPPED,
    }

    first_contract = canonical_adapter_contract_sha256(data_cfg)
    first_variant = canonical_action_sidecar_variant(
        data_cfg,
        action_horizon=50,
        canonical_eval_manifest_sha256=None,
        exclude_eval_episodes_from_training=False,
        adapter_contract_sha256=first_contract,
    )
    adapter_path.write_text('{"scale":2}', encoding="utf-8")
    second_contract = canonical_adapter_contract_sha256(data_cfg)
    second_variant = canonical_action_sidecar_variant(
        data_cfg,
        action_horizon=50,
        canonical_eval_manifest_sha256=None,
        exclude_eval_episodes_from_training=False,
        adapter_contract_sha256=second_contract,
    )

    assert first_contract != second_contract
    assert first_variant != second_variant
    assert (
        h100_training.canonical_adapter_contract_sha256(data_cfg)
        == second_contract
    )
    assert (
        h100_training.canonical_action_sidecar_variant(
            data_cfg,
            action_horizon=50,
            canonical_eval_manifest_sha256=None,
            exclude_eval_episodes_from_training=False,
            adapter_contract_sha256=second_contract,
        )
        == second_variant
    )


def test_adapter_file_drift_rejects_an_indexed_sidecar_shard(tmp_path):
    dataset = object.__new__(CanonicalSubsetVLADataset)
    shard = _shard(tmp_path)
    shard.adapter_path.write_text('{"scale":1}', encoding="utf-8")
    shard.adapter_sha256 = hashlib.sha256(
        shard.adapter_path.read_bytes()
    ).hexdigest()
    shard.adapter_path.write_text('{"scale":2}', encoding="utf-8")

    with pytest.raises(RuntimeError, match="changed after shard indexing"):
        dataset._ensure_sidecar(shard)


def test_window_cap_keeps_two_episodes_for_holdout_and_training(tmp_path):
    dataset = object.__new__(CanonicalSubsetVLADataset)
    dataset.sample_stride = 1
    dataset.canonical_eval_min_episodes_per_shard = 2
    dataset.append_subtask_to_prompt = False
    episodes = pd.DataFrame(
        [
            {
                "dataset_from_index": 0,
                "length": 100,
                "episode_index": 10,
                "task_index": 0,
            },
            {
                "dataset_from_index": 100,
                "length": 20,
                "episode_index": 20,
                "task_index": 0,
            },
            {
                "dataset_from_index": 120,
                "length": 20,
                "episode_index": 30,
                "task_index": 0,
            },
        ]
    )

    specs = dataset._build_episode_specs(
        root=tmp_path,
        gcs_prefix="gs://fixture",
        episodes=episodes,
        camera_source_keys={},
        task_map={0: "move the object"},
        fps=30.0,
        max_windows_remaining=4,
        lazy_cache=True,
    )

    # The first episode alone exceeds the four-window smoke limit. We still
    # index exactly one more episode so a heldout selection cannot remove all
    # train/statistics data from this shard, then stop without scanning all.
    assert [spec.episode_index for spec in specs] == [10, 20]
    dataset.shards = [
        SimpleNamespace(
            dataset_id="fixture/canonical",
            sid="sid",
            revision="r1",
            data_relative_path="data/chunk.parquet",
            episodes=specs,
        )
    ]
    heldout = CanonicalEvalWindow(
        dataset_id="fixture/canonical",
        sid="sid",
        revision="r1",
        data_file="data/chunk.parquet",
        episode_index=10,
        base_index=0,
    )
    dataset.canonical_eval_manifest = CanonicalEvalManifest(
        path=tmp_path / "eval.json",
        sha256="a" * 64,
        purpose="heldout",
        source_manifest_sha256="b" * 64,
        selection=_selection(configured_episode_count=2),
        windows=(heldout,),
    )
    filtered = dataset._without_heldout_episodes(dataset.shards)
    assert [episode.episode_index for episode in filtered[0].episodes] == [20]


def test_canonical_eval_manifest_generator_is_deterministic_and_immutable(
    tmp_path,
):
    source = _source(tmp_path)
    source.canonical_eval_manifest = None
    source.manifest_path = tmp_path / "catalog.jsonl.gz"
    source.manifest_path.write_bytes(b"immutable canonical catalog")

    first = build_canonical_eval_manifest_payload(
        source,
        window_count=1,
        seed=42,
        candidate_count=3,
    )
    second = build_canonical_eval_manifest_payload(
        source,
        window_count=1,
        seed=42,
        candidate_count=3,
    )

    assert first == second
    assert first["selection"]["window_count"] == 1
    assert len(first["windows"]) == 1
    heldout_episode = first["windows"][0]["episode_index"]
    assert heldout_episode in {10, 20}
    assert {10, 20} - {heldout_episode}
    output = tmp_path / "eval.json"
    path, created = write_canonical_eval_manifest(output, first)
    assert path == output.resolve()
    assert created is True
    _, created = write_canonical_eval_manifest(output, second)
    assert created is False

    drifted = json.loads(json.dumps(first))
    drifted["selection"]["seed"] = 43
    with pytest.raises(RuntimeError, match="manifest drift"):
        write_canonical_eval_manifest(output, drifted)

    with pytest.raises(
        ValueError,
        match="reserve_train_statistics",
    ):
        build_canonical_eval_manifest_payload(
            source,
            window_count=2,
            seed=42,
            candidate_count=3,
        )


def test_canonical_eval_manifest_is_ranked_only_within_bootstrap_candidate(
    tmp_path,
):
    source = _many_episode_source(tmp_path, episode_count=10)
    candidate_indices = {2, 5, 8}
    rows = [
        {
            "dataset_id": "fixture/canonical",
            "sid": "sid-1",
            "revision": "r1",
            "data_file": "data/chunk.parquet",
            "episode_index": index,
        }
        for index in sorted(candidate_indices)
    ]
    source.frozen_train_view = SimpleNamespace(
        descriptor={
            "purpose": "eval_selection_population_candidate",
            "usage_contract": {
                "training_allowed": False,
                "eval_manifest_generation": True,
            },
        },
        episode_count=len(rows),
        iter_rows=lambda: iter(rows),
    )
    source.policy_action_dim = 18

    payload = build_canonical_eval_manifest_payload(
        source,
        window_count=1,
        seed=42,
        candidate_count=3,
    )
    assert payload["selection"]["configured_episode_count"] == 3
    assert payload["selection"]["action_dim"] == 18
    assert payload["windows"][0]["episode_index"] in candidate_indices


def test_canonical_eval_manifest_balances_exact_128_windows_with_remainder(
    tmp_path,
):
    source = _many_episode_source(tmp_path, episode_count=1000)
    policy = {
        "algorithm": "dataset_fraction_divisor_v1",
        "minimum_episode_fraction": 0.05,
        "maximum_episode_fraction": 0.08,
        "episode_count_multiple": 8,
        "max_episode_count": 128,
        "evaluation_observation_count": 128,
    }

    payload = build_canonical_eval_manifest_payload(
        source,
        window_count=128,
        seed=42,
        candidate_count=3,
        holdout_sampling_policy=policy,
    )
    selection = payload["selection"]

    assert selection["holdout_episode_count"] == 56
    assert selection["base_frames_per_episode"] == 2
    assert selection["extra_window_episode_count"] == 16
    assert selection["maximum_frames_per_episode"] == 3
    assert "frames_per_episode" not in selection
    assert (
        selection["window_allocation_algorithm"]
        == "balanced_digest_rank_v1"
    )
    assert len(payload["windows"]) == 128
    assert len(selection["extra_window_episode_identities"]) == 16

    counts = {}
    for window in payload["windows"]:
        identity = (
            window["dataset_id"],
            window["sid"],
            window["revision"],
            window["data_file"],
            window["episode_index"],
        )
        counts[identity] = counts.get(identity, 0) + 1
    extras = {
        tuple(identity)
        for identity in selection["extra_window_episode_identities"]
    }
    assert len(counts) == 56
    assert sum(count == 3 for count in counts.values()) == 16
    assert all(
        count == 2 + int(identity in extras)
        for identity, count in counts.items()
    )

    output = tmp_path / "balanced-eval.json"
    write_canonical_eval_manifest(output, payload)
    loaded = load_canonical_eval_manifest(
        output,
        source_manifest_path=source.manifest_path,
        expected_selection=selection,
    )
    assert len(loaded.windows) == 128
    assert loaded.selection.frames_per_episode is None
    assert loaded.selection.extra_window_episode_identities == tuple(
        tuple(identity)
        for identity in selection["extra_window_episode_identities"]
    )


def test_canonical_eval_manifest_rejects_drifted_extra_episode_binding(
    tmp_path,
):
    source = _many_episode_source(tmp_path, episode_count=1000)
    policy = {
        "algorithm": "dataset_fraction_divisor_v1",
        "minimum_episode_fraction": 0.05,
        "maximum_episode_fraction": 0.08,
        "episode_count_multiple": 8,
        "max_episode_count": 128,
        "evaluation_observation_count": 128,
    }
    payload = build_canonical_eval_manifest_payload(
        source,
        window_count=128,
        seed=42,
        candidate_count=3,
        holdout_sampling_policy=policy,
    )
    payload["selection"]["extra_window_episode_identities"][0] = [
        "fixture/canonical",
        "sid-1",
        "r1",
        "data/chunk.parquet",
        999999,
    ]
    output = tmp_path / "drifted-extra.json"
    output.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="absent from the manifest"):
        load_canonical_eval_manifest(
            output,
            source_manifest_path=source.manifest_path,
        )


def test_canonical_train_filter_removes_entire_heldout_episode(tmp_path):
    dataset = object.__new__(CanonicalSubsetVLADataset)
    shard = _shard(tmp_path)
    heldout = CanonicalEvalWindow(
        dataset_id=shard.dataset_id,
        sid=shard.sid,
        revision=shard.revision,
        data_file=shard.data_relative_path,
        episode_index=20,
        base_index=0,
    )
    dataset.canonical_eval_manifest = CanonicalEvalManifest(
        path=tmp_path / "eval.json",
        sha256="a" * 64,
        purpose="heldout",
        source_manifest_sha256="b" * 64,
        selection=_selection(),
        windows=(heldout,),
    )

    filtered = dataset._without_heldout_episodes([shard])

    assert [episode.episode_index for episode in filtered[0].episodes] == [10]
    assert [episode.episode_index for episode in shard.episodes] == [10, 20]

    statistics_episodes, statistics_rows = (
        dataset._statistics_scope_for_shard(shard, row_count=4)
    )
    assert [episode.episode_index for episode in statistics_episodes] == [10]
    assert statistics_rows.tolist() == [True, True, False, False]


def test_unclipped_q01_q99_normalization_preserves_outlier_signal():
    values = np.asarray([[3.0, 99.0]], dtype=np.float32)
    mask = np.asarray([[True, False]])
    low = np.asarray([-1.0, -1.0], dtype=np.float32)
    high = np.asarray([1.0, 1.0], dtype=np.float32)

    unclipped = CanonicalSubsetVLADataset._normalize(
        values,
        mask,
        low,
        high,
        clip=False,
    )
    clipped = CanonicalSubsetVLADataset._normalize(
        values,
        mask,
        low,
        high,
        clip=True,
    )

    assert unclipped.tolist() == [[3.0, 0.0]]
    assert clipped.tolist() == [[1.0, 0.0]]


def test_canonical_eval_report_proves_no_leakage_and_uses_compact_metrics(
    tmp_path,
):
    dataset = DeterministicCanonicalEvalDataset(_source(tmp_path))

    report = dataset.sampling_report()

    assert report["observation_count"] == 1
    assert report["valid_action_element_count"] == 2
    assert report["train_holdout_disjoint"] is True
    assert report["normalization_excludes_holdout"] is True
    assert report["subtask_labels_required"] is False
    assert report["metric_horizons"] == [10, 50]
    assert report["metric_groups"] == ["all_action", "arm", "hand"]
    assert (
        report["episode_split_provenance"][0]["train_episode_set_sha256"]
        != report["episode_split_provenance"][0]["holdout_episode_set_sha256"]
    )
    _validate_heldout_report_coverage(
        report,
        expected_observations=1,
        required_subtasks=(2, 3, 4),
        minimum_per_subtask=1,
        label="Canonical heldout",
    )

    sample = dataset[0]
    assert "_heldout_eval_hold_action" not in sample
    assert "_heldout_eval_action_midpoint" not in sample
    assert "_heldout_eval_subtask_index" not in sample


def test_18d_canonical_eval_report_requests_gripper_not_legacy_hand(tmp_path):
    source = _source(tmp_path)
    source.policy_state_dim = 18
    source.policy_action_dim = 18
    source.normalization_statistics = {}
    source.normalization_statistics_artifact_sha256 = "d" * 64

    report = DeterministicCanonicalEvalDataset(source).sampling_report()

    assert report["action_dim"] == 18
    assert report["metric_groups"] == ["all_action", "arm", "gripper"]


@pytest.mark.parametrize(
    ("action_dim", "expected"),
    (
        (18, ["all_action", "arm", "gripper"]),
        (ACTION_DIM, ["all_action", "arm", "hand"]),
    ),
)
def test_canonical_eval_metric_groups_match_trainer_action_layout(
    action_dim,
    expected,
):
    trainer = object.__new__(VLATrainer)
    trainer.config = OmegaConf.create({"trainer": {}})

    requested = _canonical_eval_metric_groups(action_dim)
    available = trainer._eval_action_groups(action_dim)

    assert requested == expected
    assert set(requested).issubset(available)


def test_canonical_eval_metric_groups_reject_unknown_action_width():
    with pytest.raises(ValueError, match="18-D RealMan or 49-D"):
        _canonical_eval_metric_groups(22)


class _CanonicalEvalModel(torch.nn.Module):
    def predict_action(self, *, batch, **_kwargs):
        actions = np.stack([example["action"] for example in batch])
        return {"normalized_actions": np.zeros_like(actions)}


class _Accelerator:
    device = torch.device("cpu")

    @staticmethod
    def unwrap_model(model):
        return model

    @staticmethod
    def reduce(value, reduction):
        assert reduction == "sum"
        return value

    @staticmethod
    def gather(value):
        return value


def test_trainer_accepts_49d_canonical_eval_without_control_heuristics():
    trainer = object.__new__(VLATrainer)
    trainer.config = OmegaConf.create(
        {
            "framework": {"action_model": {"num_inference_timesteps": 2}},
            "trainer": {},
        }
    )
    trainer.accelerator = _Accelerator()
    trainer.model = _CanonicalEvalModel()
    action = np.ones((50, ACTION_DIM), dtype=np.float32)
    action_mask = np.zeros_like(action, dtype=bool)
    action_mask[:, :14] = True
    report = {
        "metric_horizons": [10, 50],
        "metric_groups": ["all_action", "arm", "hand"],
        "control_metadata_required": False,
    }

    metrics = trainer._evaluate_action_batches(
        [
            [
                {
                    "action": action,
                    "action_mask": action_mask,
                    "action_is_pad": np.zeros((50,), dtype=bool),
                    "_heldout_eval_index": 0,
                }
            ]
        ],
        step_metrics={},
        metric_prefix="heldout_eval",
        expected_observations=1,
        expected_valid_observations=1,
        expected_valid_elements=50 * 14,
        require_heldout_indices=True,
        sampling_report=report,
        evaluation_seed=1,
    )

    assert metrics["heldout_eval_normalized_action_mae"] == pytest.approx(1.0)
    assert metrics["heldout_eval_normalized_arm_mae_h10"] == pytest.approx(1.0)
    assert "heldout_eval_normalized_arm_mae_h1" not in metrics
    assert not any("task_success" in key for key in metrics)


def test_canonical_manifest_enables_periodic_heldout_loader(monkeypatch):
    expected_loader = object()
    calls = []

    def fake_build_dataloader(**kwargs):
        calls.append(kwargs)
        return expected_loader

    monkeypatch.setattr(
        "starVLA.dataloader.build_dataloader",
        fake_build_dataloader,
    )
    cfg = OmegaConf.create(
        {
            "datasets": {
                "vla_data": {
                    "dataset_py": "canonical_subset_vla",
                    "canonical_eval_manifest": "heldout.json",
                }
            },
            "trainer": {},
        }
    )
    accelerator = SimpleNamespace(
        dataloader_config=SimpleNamespace(
            even_batches=True,
            dispatch_batches=True,
        )
    )

    loader, focused = prepare_heldout_eval_data(
        cfg,
        accelerator,
        output_dir=None,
    )

    assert loader is expected_loader
    assert focused is None
    assert calls[0]["mode"] == "eval"
    assert calls[0]["dataset_py"] == "canonical_subset_vla"


def test_canonical_checkpoint_selection_metric_must_be_emitted_error():
    report = {
        "purpose": "canonical_manifest_heldout_training_health",
        "metric_horizons": [10, 50],
        "metric_groups": ["all_action", "arm", "hand"],
    }

    _validate_canonical_checkpoint_selection_metric(
        report,
        metric_name="heldout_eval_normalized_action_mae",
        metric_mode="min",
    )
    _validate_canonical_checkpoint_selection_metric(
        report,
        metric_name="heldout_eval_normalized_arm_mae_h10",
        metric_mode="min",
    )
    with pytest.raises(ValueError, match="cannot emit"):
        _validate_canonical_checkpoint_selection_metric(
            report,
            metric_name="mae_score",
            metric_mode="min",
        )
    with pytest.raises(ValueError, match="best_metric_mode=min"):
        _validate_canonical_checkpoint_selection_metric(
            report,
            metric_name="heldout_eval_normalized_action_mae",
            metric_mode="max",
        )


@pytest.mark.parametrize(
    "config_name",
    sorted(
        path.name
        for path in (
            Path(__file__).parents[1] / "scripts" / "config"
        ).glob("vlajepa_robot_ft_canonical*.yaml")
    ),
)
def test_canonical_configs_enable_prepared_holdout_and_select_emitted_metric(
    config_name,
):
    config_path = (
        Path(__file__).parents[1] / "scripts" / "config" / config_name
    )
    cfg = OmegaConf.load(config_path)
    if cfg.get("extends", None):
        from scripts.h100_training import _load_config

        cfg, _ = _load_config(config_path)

    assert cfg.datasets.vla_data.canonical_eval_manifest
    manifest_before_run_id_change = str(
        cfg.datasets.vla_data.canonical_eval_manifest
    )
    cfg.run_id = f"{cfg.run_id}_timestamped_launch"
    assert (
        str(cfg.datasets.vla_data.canonical_eval_manifest)
        == manifest_before_run_id_change
    )
    assert (
        cfg.datasets.vla_data.canonical_exclude_eval_episodes_from_training
        is True
    )
    assert int(cfg.datasets.vla_data.canonical_eval_selection_seed) == int(
        cfg.seed
    )
    assert int(cfg.datasets.vla_data.canonical_eval_candidate_count) == 32
    assert (
        int(
            cfg.datasets.vla_data.canonical_eval_min_episodes_per_shard
        )
        == 2
    )
    assert (
        cfg.trainer.best_metric_name
        == "heldout_eval_normalized_action_mae"
    )
    assert cfg.trainer.best_metric_mode == "min"


def test_canonical_h100_config_uses_fractional_exact_128_holdout():
    config_path = (
        Path(__file__).parents[1]
        / "scripts"
        / "config"
        / "h100"
        / "vlajepa_robot_ft_canonical_full_h100x8_qwen_full_rawddp_moge_vits.yaml"
    )
    from scripts.h100_training import _load_config

    cfg, _ = _load_config(config_path)

    assert OmegaConf.to_container(
        cfg.datasets.vla_data.holdout_sampling,
        resolve=True,
    ) == {
        "algorithm": "dataset_fraction_divisor_v1",
        "minimum_episode_fraction": 0.05,
        "maximum_episode_fraction": 0.08,
        "episode_count_multiple": 8,
        "max_episode_count": 128,
        "evaluation_observation_count": 128,
    }
    assert int(cfg.datasets.vla_data.eval_per_device_batch_size) == 16
    assert Path(
        str(cfg.datasets.vla_data.canonical_eval_manifest)
    ).name == "canonical_full_gcs_fractional_eval128_v2.json"
