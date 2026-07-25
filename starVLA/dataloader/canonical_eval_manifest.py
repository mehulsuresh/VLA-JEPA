from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..eval_sampling_policy import (
    derive_episode_holdout_sampling_plan,
    validate_holdout_sampling_policy,
)
from .canonical_subset_dataset import (
    ACTION_DIM,
    CANONICAL_EVAL_SELECTION_ALGORITHM,
    JOINT_DELTA_GRIPPER_ABSOLUTE,
    CanonicalSubsetVLADataset,
    _stable_json_sha256,
)
from .dataset_view import EVAL_SELECTION_POPULATION_CANDIDATE_PURPOSE


GENERATOR_SCHEMA_VERSION = 1


def _stable_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _episode_key(source: CanonicalSubsetVLADataset, shard: Any, episode: Any) -> tuple:
    return source._episode_identity(shard, episode)


def _eval_selection_candidate_episode_identities(
    source: CanonicalSubsetVLADataset,
) -> frozenset[tuple[str, str, str, str, int]] | None:
    """Return the exact frozen bootstrap population, when configured."""

    view = getattr(source, "frozen_train_view", None)
    if view is None:
        return None
    if (
        view.descriptor.get("purpose")
        != EVAL_SELECTION_POPULATION_CANDIDATE_PURPOSE
    ):
        return None
    usage_contract = view.descriptor.get("usage_contract")
    if (
        not isinstance(usage_contract, dict)
        or usage_contract.get("training_allowed") is not False
        or usage_contract.get("eval_manifest_generation") is not True
    ):
        raise ValueError(
            "Canonical eval-selection candidate has an invalid usage contract."
        )

    identities: set[tuple[str, str, str, str, int]] = set()
    for ordinal, row in enumerate(view.iter_rows()):
        values = tuple(
            row.get(field)
            for field in ("dataset_id", "sid", "revision", "data_file")
        )
        episode_index = row.get("episode_index")
        if (
            any(not isinstance(value, str) or not value for value in values)
            or isinstance(episode_index, bool)
            or not isinstance(episode_index, int)
            or episode_index < 0
        ):
            raise ValueError(
                "Canonical eval-selection candidate ledger contains an "
                f"invalid episode identity at record {ordinal}."
            )
        identities.add((*values, int(episode_index)))
    if not identities:
        raise ValueError(
            "Canonical eval-selection candidate contains no episodes."
        )
    if len(identities) != int(view.episode_count):
        raise ValueError(
            "Canonical eval-selection candidate episode cardinality does not "
            "match its authenticated descriptor."
        )
    return frozenset(identities)


def _rank_digest(seed: int, *values: Any) -> str:
    return hashlib.sha256(
        _stable_json_bytes([int(seed), *values])
    ).hexdigest()


def _candidate_base_indices(
    *,
    episode_length: int,
    action_horizon: int,
    seed: int,
    episode_identity: Sequence[Any],
    candidate_count: int,
) -> tuple[int, ...]:
    """Return deterministic, spread-out candidate anchors for one episode."""

    maximum = max(int(episode_length) - int(action_horizon), 0)
    if maximum <= 0:
        return (0,)
    if maximum + 1 <= int(candidate_count):
        return tuple(range(maximum + 1))

    positions = {0, maximum, maximum // 2}
    digest_index = 0
    while len(positions) < int(candidate_count):
        digest = _rank_digest(
            seed,
            *episode_identity,
            "base",
            digest_index,
        )
        positions.add(int(digest[:16], 16) % (maximum + 1))
        digest_index += 1
    return tuple(sorted(positions))


def _valid_action_count(
    source: CanonicalSubsetVLADataset,
    *,
    shard_index: int,
    episode_index: int,
    base_index: int,
) -> int:
    shard = source.shards[int(shard_index)]
    episode = shard.episodes[int(episode_index)]
    shard_data = source._get_shard_data(int(shard_index))
    available_rows = min(
        len(shard_data.state),
        len(shard_data.action),
        len(shard_data.action_mask),
    )
    if available_rows <= 0:
        return 0

    row_base = int(episode.local_start) + int(base_index)
    if row_base < 0 or row_base >= available_rows:
        return 0
    local_indices = int(base_index) + source._action_offsets
    action_is_pad = np.logical_or(
        local_indices < 0,
        local_indices >= int(episode.length),
    )
    action_rows = int(episode.local_start) + np.clip(
        local_indices,
        0,
        int(episode.length) - 1,
    )
    action_is_pad |= action_rows >= available_rows
    action_rows = np.clip(action_rows, 0, available_rows - 1)
    action_mask = shard_data.action_mask[action_rows].astype(bool, copy=True)
    if source.action_type == JOINT_DELTA_GRIPPER_ABSOLUTE:
        delta_dimensions = np.flatnonzero(shard_data.action_delta_mask)
        mapped_state = shard_data.action_to_state_indices[delta_dimensions]
        anchor_valid = shard_data.state_mask[row_base, mapped_state]
        action_mask[:, delta_dimensions[~anchor_valid]] = False
    action_mask &= ~action_is_pad[:, None]
    return int(action_mask.sum())


def build_canonical_eval_manifest_payload(
    source: CanonicalSubsetVLADataset,
    *,
    window_count: int,
    seed: int,
    candidate_count: int = 32,
    holdout_sampling_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Select exact, supervised windows from immutable canonical episodes.

    Episode ordering is SHA-256 based instead of depending on local cache
    population or Python hash randomization. For each episode, a bounded set of
    spread-out H-step anchors is inspected and the densest supervised anchors
    are selected without replacement. Ties are broken by another stable digest.
    """

    if source.canonical_eval_manifest is not None:
        raise ValueError(
            "Manifest generation requires a canonical source without an existing "
            "canonical_eval_manifest."
        )
    if isinstance(window_count, bool) or int(window_count) <= 0:
        raise ValueError("window_count must be a positive integer.")
    if isinstance(candidate_count, bool) or int(candidate_count) < 3:
        raise ValueError("candidate_count must be an integer of at least 3.")
    if int(source.action_horizon) <= 0:
        raise ValueError("Canonical action_horizon must be positive.")
    if len(source._action_offsets) != int(source.action_horizon):
        raise ValueError("Canonical action offsets do not match action_horizon.")

    candidate_episode_identities = (
        _eval_selection_candidate_episode_identities(source)
    )
    ranked_episodes: list[tuple[str, int, int, tuple[Any, ...]]] = []
    seen_episode_identities: set[tuple[Any, ...]] = set()
    for shard_index, shard in enumerate(source.shards):
        for episode_index, episode in enumerate(shard.episodes):
            identity = _episode_key(source, shard, episode)
            if (
                candidate_episode_identities is not None
                and identity not in candidate_episode_identities
            ):
                continue
            if identity in seen_episode_identities:
                raise ValueError(
                    "Canonical configured stream contains a duplicate immutable "
                    f"episode identity: {identity}."
                )
            seen_episode_identities.add(identity)
            ranked_episodes.append(
                (
                    _rank_digest(seed, *identity, "episode"),
                    int(shard_index),
                    int(episode_index),
                    identity,
                )
            )
    if candidate_episode_identities is not None:
        missing_candidate_episodes = (
            candidate_episode_identities - seen_episode_identities
        )
        if missing_candidate_episodes:
            raise ValueError(
                "Canonical eval-selection candidate references episodes "
                "outside the resolved canonical source: "
                f"{sorted(missing_candidate_episodes)[:5]}"
            )
    ranked_episodes.sort(key=lambda value: (value[0], value[3]))
    normalized_holdout_sampling_policy: dict[str, Any] | None
    if holdout_sampling_policy is None:
        holdout_episode_count = int(window_count)
        base_frames_per_episode = 1
        extra_window_episode_count = 0
        maximum_frames_per_episode = 1
        window_allocation_algorithm = "uniform_per_episode_v1"
        holdout_sampling_plan = None
        normalized_holdout_sampling_policy = None
    else:
        normalized_holdout_sampling_policy = (
            validate_holdout_sampling_policy(holdout_sampling_policy)
        )
        holdout_sampling_plan = derive_episode_holdout_sampling_plan(
            total_episode_count=len(ranked_episodes),
            evaluation_observation_count=int(window_count),
            policy=normalized_holdout_sampling_policy,
        )
        holdout_episode_count = int(
            holdout_sampling_plan["holdout_episode_count"]
        )
        base_frames_per_episode = int(
            holdout_sampling_plan["base_frames_per_episode"]
        )
        extra_window_episode_count = int(
            holdout_sampling_plan["extra_window_episode_count"]
        )
        maximum_frames_per_episode = int(
            holdout_sampling_plan["maximum_frames_per_episode"]
        )
        window_allocation_algorithm = str(
            holdout_sampling_plan["window_allocation_algorithm"]
        )
    if int(candidate_count) < maximum_frames_per_episode:
        raise ValueError(
            "candidate_count must be at least maximum_frames_per_episode: "
            f"{candidate_count} < {maximum_frames_per_episode}"
        )
    if len(ranked_episodes) < holdout_episode_count:
        raise ValueError(
            "Canonical stream does not contain enough distinct episodes for "
            f"the holdout: available={len(ranked_episodes)}, "
            f"requested={holdout_episode_count}."
        )

    selected: list[dict[str, Any]] = []
    selected_per_shard: dict[int, int] = {}
    selected_episode_count = 0
    extra_window_episode_identities: list[list[Any]] = []
    skipped_without_supervision = 0
    skipped_to_reserve_train_statistics = 0
    for _, shard_index, episode_index, identity in ranked_episodes:
        shard = source.shards[shard_index]
        episode = shard.episodes[episode_index]
        # Canonical normalization is derived independently per data shard.
        # Holding out every episode in one shard would leave no train rows from
        # which eval can reconstruct that shard's q01/q99 sidecar.
        maximum_heldout_for_shard = max(len(shard.episodes) - 1, 0)
        if selected_per_shard.get(shard_index, 0) >= maximum_heldout_for_shard:
            skipped_to_reserve_train_statistics += 1
            continue
        candidates = _candidate_base_indices(
            episode_length=int(episode.length),
            action_horizon=int(source.action_horizon),
            seed=int(seed),
            episode_identity=identity,
            candidate_count=int(candidate_count),
        )
        scored = [
            (
                _valid_action_count(
                    source,
                    shard_index=shard_index,
                    episode_index=episode_index,
                    base_index=base_index,
                ),
                _rank_digest(seed, *identity, "tie", int(base_index)),
                int(base_index),
            )
            for base_index in candidates
        ]
        ranked_scored = sorted(
            (value for value in scored if value[0] > 0),
            key=lambda value: (value[0], value[1]),
            reverse=True,
        )
        episode_window_count = base_frames_per_episode + int(
            selected_episode_count < extra_window_episode_count
        )
        if len(ranked_scored) < episode_window_count:
            skipped_without_supervision += 1
            continue
        for _, _, base_index in ranked_scored[:episode_window_count]:
            selected.append(
                {
                    "dataset_id": str(shard.dataset_id),
                    "sid": str(shard.sid),
                    "revision": str(shard.revision),
                    "data_file": str(shard.data_relative_path),
                    "episode_index": int(episode.episode_index),
                    "base_index": int(base_index),
                }
            )
        if episode_window_count > base_frames_per_episode:
            extra_window_episode_identities.append(list(identity))
        selected_per_shard[shard_index] = (
            selected_per_shard.get(shard_index, 0) + 1
        )
        selected_episode_count += 1
        if selected_episode_count == holdout_episode_count:
            break

    if (
        len(selected) != int(window_count)
        or selected_episode_count != holdout_episode_count
    ):
        raise ValueError(
            "Canonical stream does not contain enough supervised episodes for "
            f"heldout evaluation: selected_episodes={selected_episode_count}, "
            f"requested_episodes={holdout_episode_count}, "
            f"selected_windows={len(selected)}, requested_windows="
            f"{int(window_count)}, "
            f"skipped_without_supervision={skipped_without_supervision}, "
            "skipped_to_reserve_train_statistics="
            f"{skipped_to_reserve_train_statistics}."
        )

    source_manifest_path = Path(source.manifest_path).expanduser().resolve()
    return {
        "schema_version": GENERATOR_SCHEMA_VERSION,
        "purpose": "heldout",
        "source_manifest_sha256": _file_sha256(source_manifest_path),
        "selection": {
            "algorithm": CANONICAL_EVAL_SELECTION_ALGORITHM,
            "seed": int(seed),
            "window_count": int(window_count),
            "holdout_episode_count": int(holdout_episode_count),
            **(
                {
                    # Compatibility alias is valid only when every heldout
                    # episode really receives this exact count.
                    "frames_per_episode": int(base_frames_per_episode)
                }
                if extra_window_episode_count == 0
                else {}
            ),
            "base_frames_per_episode": int(base_frames_per_episode),
            "extra_window_episode_count": int(
                extra_window_episode_count
            ),
            "maximum_frames_per_episode": int(
                maximum_frames_per_episode
            ),
            "window_allocation_algorithm": window_allocation_algorithm,
            "extra_window_episode_identities": (
                extra_window_episode_identities
            ),
            "holdout_sampling_policy": normalized_holdout_sampling_policy,
            "holdout_sampling_plan": holdout_sampling_plan,
            "candidate_count": int(candidate_count),
            "action_horizon": int(source.action_horizon),
            "action_dim": int(
                getattr(source, "policy_action_dim", ACTION_DIM)
            ),
            "action_type": str(source.action_type),
            "normalization": str(source.sidecar_normalization),
            "adapter_contract_sha256": str(
                source.adapter_contract_sha256
            ),
            "action_sidecar_variant": str(source.action_sidecar_variant),
            "configured_episode_count": len(seen_episode_identities),
            "configured_episode_catalog_sha256": _stable_json_sha256(
                sorted(seen_episode_identities)
            ),
        },
        "windows": selected,
    }


def write_canonical_eval_manifest(
    output_path: str | Path,
    payload: dict[str, Any],
) -> tuple[Path, bool]:
    """Write an immutable manifest, reusing an identical existing file.

    Returns ``(path, created)``. Any semantic drift at an existing path fails
    closed so a resume or repeated prepare cannot silently change its split.
    """

    path = Path(output_path).expanduser().resolve()
    serialized = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    def verify_existing() -> None:
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(
                f"Canonical evaluation manifest is not a regular file: {path}"
            )
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Existing canonical evaluation manifest is unreadable: {path}"
            ) from exc
        if existing != payload:
            raise RuntimeError(
                "Canonical evaluation manifest drift detected; refusing to "
                f"overwrite immutable split: {path}"
            )

    if path.exists():
        verify_existing()
        return path, False

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            # Hard-link creation is atomic and, unlike os.replace(), cannot
            # overwrite another prepare process that won the race.
            os.link(temporary, path)
            created = True
        except FileExistsError:
            verify_existing()
            created = False
    finally:
        temporary.unlink(missing_ok=True)
    return path, created
