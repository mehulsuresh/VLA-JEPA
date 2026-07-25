#!/usr/bin/env python3
"""Fail-closed numeric parity gate for OpenPI and VLA-JEPA RealMan inputs.

The command reads deterministic raw LeRobot fixtures, sends them through the
real OpenPI RealMan slicing/delta/normalization transforms and VLA-JEPA's real
state/action transforms, and writes channel-level evidence on any mismatch.
Image resizing and prompt tokenization are intentionally outside this gate.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
from pathlib import Path
import sys
import types
from typing import Any

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from starVLA.action_representation import (  # noqa: E402
    PiCompatibleRunningStats,
    Q01_Q99_UNCLIPPED,
    REALMAN_18D_ACTION_CONTRACT,
    select_realman_policy_actions,
    select_realman_policy_state,
)


DELTA_MASK = np.asarray(
    REALMAN_18D_ACTION_CONTRACT.action_to_state_indices, dtype=np.int64
) >= 0


def _list_column(column: Any) -> np.ndarray:
    array = column.combine_chunks()
    lengths = np.asarray(pc.list_value_length(array).to_numpy(), dtype=np.int64)
    if not lengths.size or np.any(lengths != lengths[0]):
        raise ValueError(f"Expected fixed-width lists, got {array.type}.")
    values = np.asarray(pc.list_flatten(array).to_numpy(zero_copy_only=False))
    return values.reshape(len(array), int(lengths[0]))


def _load_episodes(dataset_root: Path, episode_ids: set[int]) -> dict[int, dict[str, np.ndarray]]:
    output: dict[int, dict[str, np.ndarray]] = {}
    columns = [
        "episode_index",
        "frame_index",
        "timestamp",
        "source.observation.state",
        "source.action",
        "valid_state",
    ]
    for path in sorted((dataset_root / "data").glob("**/*.parquet")):
        table = pq.read_table(path, columns=columns)
        episode = np.asarray(table.column("episode_index").to_numpy(), dtype=np.int64)
        selected = set(int(value) for value in np.unique(episode)) & episode_ids
        if not selected:
            continue
        state = _list_column(table.column("source.observation.state"))
        action = _list_column(table.column("source.action"))
        frames = np.asarray(table.column("frame_index").to_numpy(), dtype=np.int64)
        timestamps = np.asarray(table.column("timestamp").to_numpy(), dtype=np.float64)
        valid = np.asarray(table.column("valid_state").to_numpy(), dtype=bool)
        for episode_id in sorted(selected):
            mask = episode == episode_id
            if episode_id in output:
                raise ValueError(f"Episode {episode_id} spans multiple parquet files.")
            order = np.argsort(frames[mask], kind="stable")
            output[episode_id] = {
                "frame": frames[mask][order],
                "timestamp": timestamps[mask][order],
                "state": state[mask][order],
                "action": action[mask][order],
                "valid": valid[mask][order],
            }
    missing = episode_ids - set(output)
    if missing:
        raise ValueError(f"Missing fixture episodes: {sorted(missing)}.")
    return output


def _fixture_frames(episodes: dict[int, dict[str, np.ndarray]]) -> list[tuple[str, int, int]]:
    fixtures: list[tuple[str, int, int]] = []
    for episode_id in sorted(episodes):
        episode = episodes[episode_id]
        action = select_realman_policy_actions(episode["action"])
        gripper_change = np.flatnonzero(
            np.any(np.abs(np.diff(action[:, [7, 15]], axis=0)) > 0.5, axis=1)
        )
        invalid = np.flatnonzero(~episode["valid"])
        if not fixtures:
            fixtures.append(("normal_motion", episode_id, 0))
        if gripper_change.size and not any(name == "gripper_transition" for name, *_ in fixtures):
            fixtures.append(("gripper_transition", episode_id, int(gripper_change[0])))
        if invalid.size and not any(name == "invalid_intervention" for name, *_ in fixtures):
            fixtures.append(("invalid_intervention", episode_id, int(invalid[0])))
        if not any(name == "episode_end_clamp" for name, *_ in fixtures):
            fixtures.append(("episode_end_clamp", episode_id, len(action) - 1))
        if len({name for name, *_ in fixtures}) == 4:
            break
    required = {"normal_motion", "gripper_transition", "invalid_intervention", "episode_end_clamp"}
    present = {name for name, *_ in fixtures}
    if present != required:
        raise ValueError(f"Could not construct all parity fixtures; missing {sorted(required - present)}.")
    return fixtures


def _load_vla_transforms() -> tuple[Any, Any, Any]:
    """Load the production VLA transform without optional video dependencies."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment failure
        raise RuntimeError("The parity environment must provide PyTorch.") from exc

    # starVLA.dataloader only needs Accelerate to construct its logger, while
    # the numeric transform itself does not depend on Accelerate.  OpenPI's
    # reference venv intentionally omits that training-only dependency.
    try:
        importlib.import_module("accelerate.logging")
    except ImportError:
        accelerate = types.ModuleType("accelerate")
        accelerate_logging = types.ModuleType("accelerate.logging")
        accelerate_logging.get_logger = logging.getLogger
        accelerate.logging = accelerate_logging
        sys.modules["accelerate"] = accelerate
        sys.modules["accelerate.logging"] = accelerate_logging

    # Avoid importing the transform package's image augmentation registry;
    # parity is intentionally measured before image/model transforms.
    package_name = "starVLA.dataloader.gr00t_lerobot.transform"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [
            str(REPO_ROOT / "starVLA/dataloader/gr00t_lerobot/transform")
        ]
        sys.modules[package_name] = package
    module = importlib.import_module(f"{package_name}.state_action")
    return module.AnchorRelativeActionTransform, module.Normalizer, torch


def _vla_transform(
    state: np.ndarray,
    action: np.ndarray,
    transform_cls: Any,
) -> np.ndarray:
    transform = transform_cls(
        apply_to=["action.source_controls", "action.source_head"],
        mappings={
            "source_controls": {
                "state_key": "source",
                "state_indices": list(range(16)),
                "delta_mask": [True] * 7
                + [False]
                + [True] * 7
                + [False],
            },
            "source_head": {
                "state_key": "source",
                "state_indices": [16, 17],
                "delta_mask": [True, True],
            },
        },
    )
    transformed = transform.apply(
        {
            "state.source": state[None, :].copy(),
            "action.source_controls": action[..., :16].copy(),
            "action.source_head": action[..., 16:18].copy(),
        }
    )
    return np.concatenate(
        [
            transformed["action.source_controls"],
            transformed["action.source_head"],
        ],
        axis=-1,
    )


def _statistics_from_fixtures(values: list[np.ndarray], running_cls: Any) -> dict[str, np.ndarray]:
    running = running_cls()
    for value in values:
        running.update(value)
    stats = running.get_statistics()
    if hasattr(stats, "mean"):
        return {
            "mean": np.asarray(stats.mean),
            "std": np.asarray(stats.std),
            "q01": np.asarray(stats.q01),
            "q99": np.asarray(stats.q99),
        }
    return {key: np.asarray(stats[key]) for key in ("mean", "std", "q01", "q99")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--openpi-root", type=Path, default=Path("/home/mehul/work/reward_model_small/pi0"))
    parser.add_argument("--output-dir", type=Path, default=Path("local_eval_reports/openpi_parity"))
    parser.add_argument("--episode", type=int, action="append", default=[])
    args = parser.parse_args()

    manifest_path = args.manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["datasets"][0]
    dataset_root = Path(entry["dataset_root"]).expanduser().resolve()
    holdout = {int(value) for value in entry["holdout_episode_indices"]}
    all_ids: set[int] = set()
    for catalog_path in sorted((dataset_root / "meta/episodes").glob("**/*.parquet")):
        catalog = pq.read_table(catalog_path, columns=["episode_index"])
        all_ids.update(int(value) for value in catalog.column("episode_index").to_pylist())
    train_ids = sorted(all_ids - holdout)
    requested = set(args.episode or train_ids[:256])
    if not requested <= set(train_ids):
        raise ValueError("Parity fixtures must come from the immutable training split.")
    episodes = _load_episodes(dataset_root, requested)
    fixtures = _fixture_frames(episodes)

    openpi_src = args.openpi_root.expanduser().resolve() / "src"
    openpi_client_src = (
        args.openpi_root.expanduser().resolve() / "packages/openpi-client/src"
    )
    sys.path.insert(0, str(openpi_client_src))
    sys.path.insert(0, str(openpi_src))
    pi_transforms = importlib.import_module("openpi.transforms")
    pi_policy = importlib.import_module("openpi.policies.realman_policy")
    pi_normalize = importlib.import_module("openpi.shared.normalize")
    vla_transform_cls, vla_normalizer_cls, torch = _load_vla_transforms()

    info = json.loads((dataset_root / "meta/info.json").read_text(encoding="utf-8"))
    camera_order = [
        key.removeprefix("observation.images.")
        for key, feature in info.get("features", {}).items()
        if isinstance(feature, dict) and feature.get("dtype") == "video"
    ]
    expected_camera_order = ["head", "wrist_left", "wrist_right"]
    if camera_order != expected_camera_order:
        raise ValueError(
            f"Unexpected source camera ordering {camera_order}; expected "
            f"{expected_camera_order}."
        )

    stats_binding = entry.get("action_representation_statistics")
    if not isinstance(stats_binding, dict):
        raise ValueError("Manifest does not bind action_representation_statistics.")
    stats_path = manifest_path.parent / stats_binding["path"]
    bound_stats = json.loads(stats_path.read_text(encoding="utf-8"))["selected"]

    arrays: dict[str, np.ndarray] = {}
    records: list[dict[str, Any]] = []
    pi_states: list[np.ndarray] = []
    pi_actions: list[np.ndarray] = []
    vla_states: list[np.ndarray] = []
    vla_actions: list[np.ndarray] = []
    failures: list[str] = []
    channel_report: dict[str, Any] = {}
    horizon = REALMAN_18D_ACTION_CONTRACT.action_horizon

    def compare(
        label: str,
        left: np.ndarray,
        right: np.ndarray,
        *,
        atol: float,
        exact: bool = False,
    ) -> None:
        left = np.asarray(left)
        right = np.asarray(right)
        if left.shape != right.shape:
            failures.append(f"{label}: shape {left.shape} != {right.shape}")
            channel_report[label] = {
                "left_shape": list(left.shape),
                "right_shape": list(right.shape),
                "passed": False,
            }
            return
        error = np.abs(left.astype(np.float64) - right.astype(np.float64))
        if error.ndim:
            reduce_axes = tuple(range(error.ndim - 1))
            per_channel = (
                np.max(error, axis=reduce_axes) if reduce_axes else error
            )
        else:
            per_channel = np.asarray([float(error)])
        passed = bool(
            np.array_equal(left, right)
            if exact
            else np.allclose(left, right, atol=atol, rtol=0.0)
        )
        channel_report[label] = {
            "passed": passed,
            "exact": exact,
            "atol": atol,
            "max_abs_by_channel": np.asarray(per_channel).reshape(-1).tolist(),
        }
        if not passed:
            failures.append(
                f"{label}: max_abs={float(np.max(error)) if error.size else 0.0}"
            )

    vla_delta_mask = np.asarray(
        [True] * 7 + [False] + [True] * 7 + [False, True, True],
        dtype=bool,
    )
    compare("delta_mask", DELTA_MASK, vla_delta_mask, atol=0.0, exact=True)
    arrays["pi_delta_mask"] = DELTA_MASK
    arrays["vla_delta_mask"] = vla_delta_mask

    for fixture_name, episode_id, local_index in fixtures:
        episode = episodes[episode_id]
        length = len(episode["frame"])
        pi_indices = np.minimum(local_index + np.arange(horizon), length - 1)
        vla_indices = np.clip(local_index + np.arange(horizon), 0, length - 1)
        compare(
            f"{fixture_name}:horizon_indices",
            pi_indices,
            vla_indices,
            atol=0.0,
            exact=True,
        )
        raw_state = episode["state"][local_index]
        raw_action = episode["action"][pi_indices]

        pi_state = np.asarray(pi_policy._slice_state(raw_state), dtype=np.float32)
        pi_action = np.asarray(pi_policy._slice_actions(raw_action), dtype=np.float32)
        pi_mixed = pi_transforms.DeltaActions(DELTA_MASK.copy())(
            {"state": pi_state.copy(), "actions": pi_action.copy()}
        )["actions"]
        vla_state = select_realman_policy_state(raw_state)
        vla_action = select_realman_policy_actions(raw_action)
        vla_mixed = _vla_transform(vla_state, vla_action, vla_transform_cls)

        for label, left, right, atol, exact in (
            ("state", pi_state, vla_state, 0.0, True),
            ("action", pi_action, vla_action, 0.0, True),
            ("mixed", pi_mixed, vla_mixed, 1e-6, False),
        ):
            compare(
                f"{fixture_name}:{label}",
                left,
                right,
                atol=atol,
                exact=exact,
            )
            arrays[f"{fixture_name}_pi_{label}"] = left
            arrays[f"{fixture_name}_vla_{label}"] = right
        arrays[f"{fixture_name}_pi_indices"] = pi_indices
        arrays[f"{fixture_name}_vla_indices"] = vla_indices
        pi_states.append(pi_state[None, :])
        vla_states.append(vla_state[None, :])
        pi_actions.append(pi_mixed)
        vla_actions.append(vla_mixed)
        records.append(
            {
                "name": fixture_name,
                "episode_id": episode_id,
                "frame_index": int(episode["frame"][local_index]),
                "timestamp": float(episode["timestamp"][local_index]),
                "camera_order": camera_order,
                "camera_timestamps": {
                    name: float(episode["timestamp"][local_index])
                    for name in camera_order
                },
                "horizon_indices": pi_indices.tolist(),
                "end_clamped": bool(np.any(pi_indices == length - 1)),
                "valid_state": bool(episode["valid"][local_index]),
            }
        )

    pi_state_stats = _statistics_from_fixtures(pi_states, pi_normalize.RunningStats)
    vla_state_stats = _statistics_from_fixtures(vla_states, PiCompatibleRunningStats)
    pi_action_stats = _statistics_from_fixtures(pi_actions, pi_normalize.RunningStats)
    vla_action_stats = _statistics_from_fixtures(vla_actions, PiCompatibleRunningStats)
    for modality, pi_stats, vla_stats, source_values in (
        ("state", pi_state_stats, vla_state_stats, np.concatenate(pi_states, axis=0)),
        ("action", pi_action_stats, vla_action_stats, np.concatenate(pi_actions, axis=0)),
    ):
        span = np.max(source_values, axis=0) - np.min(source_values, axis=0)
        bin_width = span / 5000.0
        for name in ("q01", "q99"):
            error = np.abs(pi_stats[name] - vla_stats[name])
            passed = bool(np.all(error <= bin_width + 1e-12))
            channel_report[f"statistics:{modality}:{name}"] = {
                "passed": passed,
                "max_abs_by_channel": error.tolist(),
                "histogram_bin_width_by_channel": bin_width.tolist(),
            }
            if not passed:
                failures.append(f"{modality}:{name}: exceeds one histogram bin")

    for fixture_name, *_ in fixtures:
        pi_state = arrays[f"{fixture_name}_pi_state"]
        pi_mixed = arrays[f"{fixture_name}_pi_mixed"]
        pi_state_norm = pi_transforms.Normalize(
            {"state": pi_normalize.NormStats(**{
                key: np.asarray(bound_stats["state"][key])
                for key in ("mean", "std", "q01", "q99")
            })},
            use_quantiles=True,
            strict=True,
        )({"state": pi_state.copy()})["state"]
        pi_action_norm = pi_transforms.Normalize(
            {"actions": pi_normalize.NormStats(**{
                key: np.asarray(bound_stats["action"][key])
                for key in ("mean", "std", "q01", "q99")
            })},
            use_quantiles=True,
            strict=True,
        )({"actions": pi_mixed.copy()})["actions"]
        vla_state_norm = vla_normalizer_cls(
            Q01_Q99_UNCLIPPED,
            {
                key: np.asarray(value).copy()
                for key, value in bound_stats["state"].items()
            },
        ).forward(
            torch.as_tensor(arrays[f"{fixture_name}_vla_state"])
        ).cpu().numpy()
        vla_action_norm = vla_normalizer_cls(
            Q01_Q99_UNCLIPPED,
            {
                key: np.asarray(value).copy()
                for key, value in bound_stats["action"].items()
            },
        ).forward(
            torch.as_tensor(arrays[f"{fixture_name}_vla_mixed"])
        ).cpu().numpy()
        for label, left, right in (
            ("state_normalized", pi_state_norm, vla_state_norm),
            ("action_normalized", pi_action_norm, vla_action_norm),
        ):
            arrays[f"{fixture_name}_pi_{label}"] = left
            arrays[f"{fixture_name}_vla_{label}"] = right
            compare(
                f"{fixture_name}:{label}", left, right, atol=1e-5
            )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "openpi-vlajepa-realman-parity-v1",
        "passed": not failures,
        "contract": REALMAN_18D_ACTION_CONTRACT.to_dict(),
        "contract_sha256": REALMAN_18D_ACTION_CONTRACT.sha256(),
        "manifest": str(manifest_path),
        "statistics": str(stats_path),
        "fixtures": records,
        "tolerances": {
            "selected_raw": 0.0,
            "transformed_action_atol": 1e-6,
            "quantile": "one_histogram_bin",
            "normalized_atol": 1e-5,
        },
        "channel_report": channel_report,
        "failures": failures,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if failures:
        np.savez_compressed(output_dir / "failure_outputs.npz", **arrays)
        raise SystemExit("OpenPI parity gate failed: " + "; ".join(failures))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
