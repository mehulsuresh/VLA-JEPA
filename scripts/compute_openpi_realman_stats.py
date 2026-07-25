#!/usr/bin/env python3
"""Compute OpenPI-compatible 18-D RealMan state/action statistics.

The output is tied to one immutable episode-split manifest.  It scans only the
manifest's training episodes, updates statistics once per episode in ascending
episode order, and constructs H=50 action chunks with episode-end clamping.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from starVLA.action_representation import (  # noqa: E402
    PiCompatibleRunningStats,
    REALMAN_18D_ACTION_CONTRACT,
    encode_actions,
    select_realman_policy_actions,
    select_realman_policy_state,
    split_manifest_sha256_without_statistics_binding,
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(path)


def _list_column(column) -> np.ndarray:
    array = column.combine_chunks()
    if not (
        pa.types.is_list(array.type)
        or pa.types.is_large_list(array.type)
        or pa.types.is_fixed_size_list(array.type)
    ):
        raise ValueError(f"Expected a list column, got {array.type}.")
    lengths = np.asarray(pc.list_value_length(array).to_numpy(), dtype=np.int64)
    if lengths.size == 0 or np.any(lengths != lengths[0]):
        raise ValueError(f"Expected a non-empty fixed-width list column, got {array.type}.")
    values = np.asarray(pc.list_flatten(array).to_numpy(zero_copy_only=False))
    return values.reshape(len(array), int(lengths[0]))


def _split_statistics(statistics: dict[str, Any], start: int, end: int) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in statistics.items():
        if key == "count":
            output[key] = list(value)
        else:
            output[key] = np.asarray(value)[start:end].astype(np.float64).tolist()
    return output


def compute_statistics(
    *,
    dataset_root: Path,
    train_episode_ids: set[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    episode_payloads: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for parquet_path in sorted((dataset_root / "data").glob("**/*.parquet")):
        table = pq.read_table(
            parquet_path,
            columns=[
                "episode_index",
                "frame_index",
                "source.observation.state",
                "source.action",
            ],
        )
        episode_ids = np.asarray(
            table.column("episode_index").to_numpy(), dtype=np.int64
        )
        states = _list_column(table.column("source.observation.state"))
        actions = _list_column(table.column("source.action"))
        frame_indices = np.asarray(
            table.column("frame_index").to_numpy(), dtype=np.int64
        )
        for episode_id in np.unique(episode_ids):
            episode_id = int(episode_id)
            if episode_id not in train_episode_ids:
                continue
            if episode_id in episode_payloads:
                raise ValueError(
                    f"Episode {episode_id} spans multiple parquet shards; "
                    "the parity scan requires one deterministic episode batch."
                )
            mask = episode_ids == episode_id
            order = np.argsort(frame_indices[mask], kind="stable")
            ordered_frames = frame_indices[mask][order]
            if not np.array_equal(
                ordered_frames,
                np.arange(ordered_frames.size, dtype=np.int64),
            ):
                raise ValueError(
                    f"Episode {episode_id} frame indices are not contiguous from zero."
                )
            episode_payloads[episode_id] = (states[mask][order], actions[mask][order])

    missing = sorted(train_episode_ids - set(episode_payloads))
    extra = sorted(set(episode_payloads) - train_episode_ids)
    if missing or extra:
        raise ValueError(
            f"Train episode scan mismatch: missing={missing[:20]}, extra={extra[:20]}."
        )

    state_running = PiCompatibleRunningStats()
    action_running = PiCompatibleRunningStats()
    frame_count = 0
    source_state_dims: set[int] = set()
    source_action_dims: set[int] = set()
    horizon = REALMAN_18D_ACTION_CONTRACT.action_horizon
    for episode_id in sorted(episode_payloads):
        source_state, source_action = episode_payloads[episode_id]
        source_state_dims.add(int(source_state.shape[-1]))
        source_action_dims.add(int(source_action.shape[-1]))
        state = select_realman_policy_state(source_state)
        action = select_realman_policy_actions(source_action)
        if state.shape[0] != action.shape[0] or state.shape[0] <= 0:
            raise ValueError(
                f"Episode {episode_id} state/action lengths differ: "
                f"{state.shape} vs {action.shape}."
            )
        length = state.shape[0]
        gather = np.minimum(
            np.arange(length, dtype=np.int64)[:, None]
            + np.arange(horizon, dtype=np.int64)[None, :],
            length - 1,
        )
        action_chunks = encode_actions(action[gather], state)
        state_running.update(state)
        action_running.update(action_chunks)
        frame_count += length

    state_statistics = state_running.get_statistics()
    action_statistics = action_running.get_statistics()
    return (
        {
            "state": state_statistics,
            "action": action_statistics,
        },
        {
            "episode_count": len(episode_payloads),
            "frame_count": frame_count,
            "action_value_count": frame_count * horizon,
            "source_state_dims": sorted(source_state_dims),
            "source_action_dims": sorted(source_action_dims),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--update-manifest", action="store_true")
    args = parser.parse_args()

    manifest_path = args.manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("datasets")
    if not isinstance(entries, list) or len(entries) != 1:
        raise ValueError("The RealMan statistics command requires exactly one dataset entry.")
    entry = entries[0]
    dataset_root = (
        args.dataset_root.expanduser().resolve()
        if args.dataset_root is not None
        else Path(entry["dataset_root"]).expanduser().resolve()
    )
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    holdout_ids = {int(value) for value in entry["holdout_episode_indices"]}
    episode_paths = sorted((dataset_root / "meta/episodes").glob("**/*.parquet"))
    catalog_ids: set[int] = set()
    for episode_path in episode_paths:
        catalog_ids.update(
            int(value)
            for value in pq.read_table(episode_path, columns=["episode_index"])
            .column("episode_index")
            .to_pylist()
        )
    train_ids = catalog_ids - holdout_ids
    if len(train_ids) != int(entry["train_episode_count"]):
        raise ValueError("Manifest train episode count does not match the dataset catalog.")

    statistics, observed = compute_statistics(
        dataset_root=dataset_root,
        train_episode_ids=train_ids,
    )
    if observed["frame_count"] != int(entry["train_frame_count"]):
        raise ValueError(
            f"Observed {observed['frame_count']} train frames, expected "
            f"{entry['train_frame_count']}."
        )

    output_path = args.output
    if output_path is None:
        output_path = (
            manifest_path.parent
            / "artifacts"
            / f"{manifest_path.stem}_openpi_realman18_stats.json"
        )
    output_path = output_path.expanduser().resolve()
    contract = REALMAN_18D_ACTION_CONTRACT
    payload = {
        "schema": "openpi-realman-18d-statistics-v1",
        "contract": contract.to_dict(),
        "contract_sha256": contract.sha256(),
        "provenance": {
            "manifest_path": str(manifest_path),
            "split_manifest_sha256_without_statistics_binding": (
                split_manifest_sha256_without_statistics_binding(manifest)
            ),
            "full_catalog_sha256": entry["full_catalog_sha256"],
            "train_catalog_sha256": entry["train_catalog_sha256"],
            "train_episode_count": observed["episode_count"],
            "train_frame_count": observed["frame_count"],
            "action_value_count": observed["action_value_count"],
            "observed_source_state_dims": observed["source_state_dims"],
            "observed_source_action_dims": observed["source_action_dims"],
            "episode_order": "episode_id_ascending",
            "update_batch": "one_episode",
            "action_padding": "repeat_episode_final_frame",
            "quantile_bins": 5000,
        },
        "selected": statistics,
        "modalities": {
            "state": {
                "source": _split_statistics(statistics["state"], 0, 18),
            },
            "action": {
                "source_controls": _split_statistics(statistics["action"], 0, 16),
                "source_head": _split_statistics(statistics["action"], 16, 18),
            },
        },
    }
    _write_json(output_path, payload)
    output_sha256 = _file_sha256(output_path)

    if args.update_manifest:
        entry["action_representation_statistics"] = {
            "path": output_path.relative_to(manifest_path.parent).as_posix(),
            "sha256": output_sha256,
            "frame_count": observed["frame_count"],
            "catalog_sha256": entry["train_catalog_sha256"],
            "contract_sha256": contract.sha256(),
            "normalization": contract.normalization,
        }
        _write_json(manifest_path, manifest)
        report_path = manifest_path.with_name(f"{manifest_path.stem}_report.json")
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["manifest_sha256"] = _file_sha256(manifest_path)
            report["action_representation_statistics"] = {
                "path": str(output_path),
                "sha256": output_sha256,
                "contract_sha256": contract.sha256(),
                "normalization": contract.normalization,
                "train_frame_count": observed["frame_count"],
                "action_value_count": observed["action_value_count"],
            }
            _write_json(report_path, report)

    print(f"output={output_path}")
    print(f"sha256={output_sha256}")
    print(f"contract_sha256={contract.sha256()}")
    print(f"train_episodes={observed['episode_count']}")
    print(f"train_frames={observed['frame_count']}")
    print(f"action_values={observed['action_value_count']}")


if __name__ == "__main__":
    main()
