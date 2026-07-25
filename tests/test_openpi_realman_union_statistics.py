from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.compute_openpi_realman_union_stats import (
    build_union_statistics,
    write_union_statistics,
)
from starVLA.action_representation import (
    CANONICAL_REALMAN_ACTION_SOURCE_INDICES,
    CANONICAL_REALMAN_STATE_SOURCE_INDICES,
    OPENPI_REALMAN_UNION_POPULATION_SCHEMA,
    PiCompatibleMaskedRunningStats,
    PiCompatibleRunningStats,
    REALMAN_18D_ACTION_CONTRACT,
    deterministic_json_bytes,
    load_openpi_realman_union_statistics,
    select_canonical_realman_policy_action_mask,
    select_canonical_realman_policy_actions,
    select_canonical_realman_policy_state,
    select_canonical_realman_policy_state_mask,
    serialize_openpi_realman_union_statistics,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_npz(
    path: Path,
    *,
    state: np.ndarray,
    action: np.ndarray,
    state_mask: np.ndarray | None = None,
    action_mask: np.ndarray | None = None,
) -> str:
    kwargs = {"state": state, "action": action}
    if state_mask is not None:
        kwargs["state_mask"] = state_mask
    if action_mask is not None:
        kwargs["action_mask"] = action_mask
    np.savez(path, **kwargs)
    return _sha256(path)


def _policy_episode(
    *,
    length: int,
    offset: float,
    head_present: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    state = np.zeros((length, 18), dtype=np.float32)
    action = np.zeros((length, 18), dtype=np.float32)
    for frame in range(length):
        state[frame] = (
            offset
            + np.arange(18, dtype=np.float32) * np.float32(0.01)
            + np.float32(frame)
        )
        action[frame] = state[frame] + np.float32(frame + 1) * np.float32(0.1)
        action[frame, 7] = np.float32(0.2 + 0.1 * frame)
        action[frame, 15] = np.float32(0.8 - 0.1 * frame)
    state_mask = np.ones_like(state, dtype=bool)
    action_mask = np.ones_like(action, dtype=bool)
    if not head_present:
        state[:, 16:18] = 0
        action[:, 16:18] = 0
        state_mask[:, 16:18] = False
        action_mask[:, 16:18] = False
    return state, action, state_mask, action_mask


def _canonical_from_policy(
    state: np.ndarray,
    action: np.ndarray,
    state_mask: np.ndarray,
    action_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    canonical_state = np.zeros((len(state), 53), dtype=np.float32)
    canonical_action = np.zeros((len(action), 49), dtype=np.float32)
    canonical_state_mask = np.zeros((len(state), 53), dtype=bool)
    canonical_action_mask = np.zeros((len(action), 49), dtype=bool)
    state_indices = np.asarray(CANONICAL_REALMAN_STATE_SOURCE_INDICES)
    action_indices = np.asarray(CANONICAL_REALMAN_ACTION_SOURCE_INDICES)
    canonical_state[:, state_indices] = state
    canonical_action[:, action_indices] = action
    canonical_state_mask[:, state_indices] = state_mask
    canonical_action_mask[:, action_indices] = action_mask
    return (
        canonical_state,
        canonical_action,
        canonical_state_mask,
        canonical_action_mask,
    )


def _source(
    source_id: str,
    *,
    representation: str,
    episodes: list[dict],
    catalog_digit: str,
) -> dict:
    return {
        "id": source_id,
        "catalog_sha256": catalog_digit * 64,
        "provenance": {"fixture": source_id},
        "reader": {
            "kind": "npz_episodes",
            "representation": representation,
            "episodes": episodes,
        },
    }


def _episode_entry(
    path: Path,
    episode_id: str,
    *,
    duplicate_of: str | None = None,
    base_frame_indices: str | list[int] = "all",
) -> dict:
    entry = {
        "id": episode_id,
        "path": path.name,
        "sha256": _sha256(path),
        "base_frame_indices": base_frame_indices,
    }
    if duplicate_of is not None:
        entry["duplicate_of"] = duplicate_of
    return entry


def _write_population_manifest(
    path: Path,
    *,
    sources: list[dict],
    holdout_keys: list[str],
) -> Path:
    holdout_path = path.with_name(f"{path.stem}.holdout.json")
    holdout_path.write_bytes(
        deterministic_json_bytes(
            {
                "schema": "test-realman-union-holdout-v1",
                "episode_keys": sorted(holdout_keys),
            }
        )
    )
    payload = {
        "schema": OPENPI_REALMAN_UNION_POPULATION_SCHEMA,
        "contract_sha256": REALMAN_18D_ACTION_CONTRACT.sha256(),
        "source_order": [source["id"] for source in sources],
        "sources": sources,
        "holdout": {
            "manifest": holdout_path.name,
            "manifest_sha256": _sha256(holdout_path),
            "episode_keys": sorted(holdout_keys),
        },
    }
    path.write_bytes(deterministic_json_bytes(payload))
    return path


def _build_three_pool_fixture(tmp_path: Path) -> Path:
    rs = _policy_episode(length=3, offset=0.0, head_present=False)
    canonical_rs = _canonical_from_policy(*rs)
    rs_path = tmp_path / "rs.npz"
    _write_npz(
        rs_path,
        state=canonical_rs[0],
        action=canonical_rs[1],
        state_mask=canonical_rs[2],
        action_mask=canonical_rs[3],
    )
    intervention_path = tmp_path / "intervention_duplicate.npz"
    _write_npz(
        intervention_path,
        state=rs[0],
        action=rs[1],
        state_mask=rs[2],
        action_mask=rs[3],
    )
    hq_train = _policy_episode(length=3, offset=10.0, head_present=True)
    hq_train_path = tmp_path / "hq_train.npz"
    _write_npz(
        hq_train_path,
        state=hq_train[0],
        action=hq_train[1],
        state_mask=hq_train[2],
        action_mask=hq_train[3],
    )
    hq_holdout = _policy_episode(length=2, offset=20.0, head_present=True)
    hq_holdout_path = tmp_path / "hq_holdout.npz"
    _write_npz(
        hq_holdout_path,
        state=hq_holdout[0],
        action=hq_holdout[1],
        state_mask=hq_holdout[2],
        action_mask=hq_holdout[3],
    )
    sources = [
        _source(
            "realsource",
            representation="canonical_realman",
            episodes=[_episode_entry(rs_path, "000")],
            catalog_digit="a",
        ),
        _source(
            "intervention",
            representation="policy18",
            episodes=[
                _episode_entry(
                    intervention_path,
                    "000",
                    duplicate_of="realsource/000",
                )
            ],
            catalog_digit="b",
        ),
        _source(
            "hq",
            representation="policy18",
            episodes=[
                _episode_entry(hq_train_path, "000-train"),
                _episode_entry(hq_holdout_path, "999-holdout"),
            ],
            catalog_digit="c",
        ),
    ]
    return _write_population_manifest(
        tmp_path / "population.json",
        sources=sources,
        holdout_keys=["hq/999-holdout"],
    )


def test_canonical_projection_uses_semantic_indices_and_preserves_missing_head():
    state = np.arange(53, dtype=np.float32)
    action = np.arange(49, dtype=np.float32)
    state_mask = np.ones(53, dtype=bool)
    action_mask = np.ones(49, dtype=bool)
    state_mask[40:42] = False
    action_mask[40:42] = False

    np.testing.assert_array_equal(
        select_canonical_realman_policy_state(state),
        state[np.asarray(CANONICAL_REALMAN_STATE_SOURCE_INDICES)],
    )
    np.testing.assert_array_equal(
        select_canonical_realman_policy_actions(action),
        action[np.asarray(CANONICAL_REALMAN_ACTION_SOURCE_INDICES)],
    )
    np.testing.assert_array_equal(
        select_canonical_realman_policy_state_mask(state_mask)[16:18],
        False,
    )
    np.testing.assert_array_equal(
        select_canonical_realman_policy_action_mask(action_mask)[16:18],
        False,
    )


def test_masked_openpi_statistics_do_not_count_missing_head_channels():
    running = PiCompatibleMaskedRunningStats(18)
    first = np.arange(4 * 18, dtype=np.float32).reshape(4, 18)
    first_mask = np.ones_like(first, dtype=bool)
    first_mask[:, 16:18] = False
    second = np.arange(3 * 18, dtype=np.float32).reshape(3, 18) + 100

    running.update(first, first_mask)
    running.update(second, np.ones_like(second, dtype=bool))
    statistics = running.get_statistics()

    assert statistics["count"][:16] == [7] * 16
    assert statistics["count"][16:18] == [3, 3]
    np.testing.assert_allclose(
        statistics["mean"][16:18],
        second[:, 16:18].mean(axis=0),
    )


def test_direct_union_quantile_is_not_weighted_average_of_source_quantiles():
    direct = PiCompatibleMaskedRunningStats(1)
    low = np.zeros((100, 1), dtype=np.float32)
    high = np.full((2, 1), 100.0, dtype=np.float32)
    direct.update(low)
    direct.update(high)

    low_running = PiCompatibleRunningStats()
    high_running = PiCompatibleRunningStats()
    low_running.update(low)
    high_running.update(high)
    weighted_average = (
        low_running.get_statistics()["q99"][0]
        + high_running.get_statistics()["q99"][0]
    ) / 2

    union_q99 = direct.get_statistics()["q99"][0]
    assert union_q99 > 90
    assert weighted_average < 60


def test_union_builder_deduplicates_expected_overlap_and_excludes_holdout(tmp_path):
    manifest = _build_three_pool_fixture(tmp_path)

    artifact, ledger = build_union_statistics(manifest)

    population = artifact["population"]
    assert population["candidate_base_frames"] == 11
    assert population["unique_base_frames"] == 6
    assert population["duplicate_base_frames"] == 3
    assert population["holdout_excluded_base_frames"] == 2
    assert artifact["selected"]["state"]["count"][:16] == [6] * 16
    # RealSource's absent head values are not zero observations.
    assert artifact["selected"]["state"]["count"][16:18] == [3, 3]
    assert artifact["selected"]["action"]["count"][:16] == [300] * 16
    assert artifact["selected"]["action"]["count"][16:18] == [150, 150]
    assert ledger["episodes"][1]["duplicate_base_frames"] == [0, 1, 2]
    assert ledger["episodes"][-1]["holdout_base_frames"] == [0, 1]


def test_h50_targets_are_chunk_start_deltas_and_grippers_stay_absolute(tmp_path):
    state = np.zeros((2, 18), dtype=np.float32)
    action = np.zeros((2, 18), dtype=np.float32)
    state[:, 0] = [10.0, 20.0]
    state[:, 7] = [100.0, 200.0]
    action[:, 0] = [11.0, 13.0]
    action[:, 7] = [0.25, 0.75]
    # Give all other channels at least two finite observations.
    state[:, 1:] += np.asarray([0.0, 1.0], dtype=np.float32)[:, None]
    action[:, 1:] += np.asarray([1.0, 2.0], dtype=np.float32)[:, None]
    action[:, 7] = [0.25, 0.75]
    train_path = tmp_path / "train.npz"
    _write_npz(train_path, state=state, action=action)
    holdout = _policy_episode(length=2, offset=30, head_present=True)
    holdout_path = tmp_path / "holdout.npz"
    _write_npz(
        holdout_path,
        state=holdout[0],
        action=holdout[1],
        state_mask=holdout[2],
        action_mask=holdout[3],
    )
    source = _source(
        "hq",
        representation="policy18",
        episodes=[
            _episode_entry(holdout_path, "000-holdout"),
            _episode_entry(train_path, "001-train"),
        ],
        catalog_digit="d",
    )
    manifest = _write_population_manifest(
        tmp_path / "population.json",
        sources=[source],
        holdout_keys=["hq/000-holdout"],
    )

    artifact, _ = build_union_statistics(manifest)
    action_stats = artifact["selected"]["action"]

    assert action_stats["count"] == [100] * 18
    assert action_stats["min"][0] == pytest.approx(-7.0)
    assert action_stats["max"][0] == pytest.approx(3.0)
    assert action_stats["min"][7] == pytest.approx(0.25)
    assert action_stats["max"][7] == pytest.approx(0.75)


def test_unmarked_duplicate_and_holdout_content_leak_fail_closed(tmp_path):
    episode = _policy_episode(length=2, offset=0, head_present=True)
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    for path in (first, second):
        _write_npz(
            path,
            state=episode[0],
            action=episode[1],
            state_mask=episode[2],
            action_mask=episode[3],
        )
    sources = [
        _source(
            "one",
            representation="policy18",
            episodes=[_episode_entry(first, "000")],
            catalog_digit="1",
        ),
        _source(
            "two",
            representation="policy18",
            episodes=[_episode_entry(second, "000")],
            catalog_digit="2",
        ),
    ]
    manifest = _write_population_manifest(
        tmp_path / "unmarked.json",
        sources=sources,
        holdout_keys=["two/000"],
    )
    with pytest.raises(ValueError, match="Unmarked duplicate episode content"):
        build_union_statistics(manifest)

    sources[1]["reader"]["episodes"][0]["duplicate_of"] = "one/000"
    manifest = _write_population_manifest(
        tmp_path / "leak.json",
        sources=sources,
        holdout_keys=["two/000"],
    )
    with pytest.raises(ValueError, match="Holdout content leak"):
        build_union_statistics(manifest)


def test_artifact_bytes_are_deterministic_and_file_mutation_is_rejected(tmp_path):
    manifest = _build_three_pool_fixture(tmp_path)
    first, _ = build_union_statistics(manifest)
    second, _ = build_union_statistics(manifest)

    first_bytes = serialize_openpi_realman_union_statistics(first)
    assert first_bytes == serialize_openpi_realman_union_statistics(second)

    output = tmp_path / "union.json"
    artifact_sha256, _ = write_union_statistics(
        manifest_path=manifest,
        output_path=output,
    )
    loaded = load_openpi_realman_union_statistics(output, artifact_sha256)
    assert loaded == first

    output.write_bytes(output.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_openpi_realman_union_statistics(output, artifact_sha256)


def test_population_holdout_file_is_hash_authenticated(tmp_path):
    manifest = _build_three_pool_fixture(tmp_path)
    holdout_path = manifest.with_name(f"{manifest.stem}.holdout.json")
    holdout_path.write_bytes(holdout_path.read_bytes() + b" ")

    with pytest.raises(ValueError, match="Holdout manifest SHA-256 mismatch"):
        build_union_statistics(manifest)


def test_authenticated_holdout_keys_must_match_population_exclusions(tmp_path):
    manifest = _build_three_pool_fixture(tmp_path)
    holdout_path = manifest.with_name(f"{manifest.stem}.holdout.json")
    holdout_path.write_bytes(
        deterministic_json_bytes(
            {
                "schema": "test-realman-union-holdout-v1",
                "episode_keys": ["hq/000-train"],
            }
        )
    )
    population = json.loads(manifest.read_text(encoding="utf-8"))
    population["holdout"]["manifest_sha256"] = _sha256(holdout_path)
    manifest.write_bytes(deterministic_json_bytes(population))

    with pytest.raises(
        ValueError,
        match="Authenticated holdout episode keys do not exactly match",
    ):
        build_union_statistics(manifest)


def test_population_episode_file_mutation_is_rejected(tmp_path):
    manifest = _build_three_pool_fixture(tmp_path)
    rs_path = tmp_path / "rs.npz"
    with rs_path.open("ab") as handle:
        handle.write(b"drift")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        build_union_statistics(manifest)
