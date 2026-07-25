#!/usr/bin/env python3
"""Measure RealMan left/right pickup accuracy and prompt sensitivity.

This is an offline, teacher-forced diagnostic.  Every condition for a fixture
uses the same recorded images and state; only the language instruction changes.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from deployment.realman.pipeline import (
    MAGNA_DEFAULT_INSTRUCTION,
    REALMAN_CAMERA_ORDER,
    build_policy_payload,
    realman_continuous_unnormalize,
    realman_policy_actions_to_absolute,
    resolve_qwen_frame_size,
    validate_realman_policy_payload,
    validate_realman_server_metadata,
)
from deployment.trossen.pipeline import (
    resolve_action_stats,
    resolve_norm_mode,
    resolve_state_stats,
)
from scripts.replay_realman_eval_episode import (
    _load_episode_subtasks,
    _load_subtask_text,
)
from scripts.replay_realman_training_episode import (
    _decode_anchor_images,
    _load_episode_metadata,
    _load_episode_rows,
)
from starVLA.action_representation import (
    REALMAN_18D_ACTION_CONTRACT,
    normalize_q01_q99_unclipped,
    select_realman_policy_actions,
    select_realman_policy_state,
)


PICKUP_LEFT = 1
PICKUP_RIGHT = 5
ARM_DIMS = np.asarray(tuple(range(7)) + tuple(range(8, 15)), dtype=np.int64)
LEFT_ARM_DIMS = np.arange(0, 7, dtype=np.int64)
RIGHT_ARM_DIMS = np.arange(8, 15, dtype=np.int64)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--holdout-manifest", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--episodes-per-side", type=int, default=8)
    parser.add_argument("--anchors-per-episode", type=int, default=2)
    parser.add_argument("--compare-window", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--matched-noise",
        action="store_true",
        help=(
            "Send the same deterministic inference seed for every language "
            "condition of a fixture. Requires the analysis-only seeded server."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _longest_span(values: np.ndarray, label: int) -> tuple[int, int] | None:
    mask = values == int(label)
    if not np.any(mask):
        return None
    boundaries = np.flatnonzero(mask[1:] != mask[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(mask)]))
    spans = [
        (int(start), int(end))
        for start, end in zip(starts, ends, strict=True)
        if bool(mask[start])
    ]
    return max(spans, key=lambda span: span[1] - span[0])


def _select_evenly(values: list[int], count: int) -> list[int]:
    if len(values) <= count:
        return values
    positions = np.linspace(0, len(values) - 1, count)
    return [values[int(round(position))] for position in positions]


def _candidate_episodes(
    dataset_root: Path,
    holdout_indices: list[int],
    *,
    episodes_per_side: int,
) -> dict[int, list[dict[str, Any]]]:
    candidates: dict[int, list[dict[str, Any]]] = {
        PICKUP_LEFT: [],
        PICKUP_RIGHT: [],
    }
    for episode_index in holdout_indices:
        episode_meta = _load_episode_metadata(dataset_root, episode_index)
        subtasks = _load_episode_subtasks(dataset_root, episode_meta)
        present = [
            label
            for label in (PICKUP_LEFT, PICKUP_RIGHT)
            if np.any(subtasks == label)
        ]
        if len(present) != 1:
            continue
        label = present[0]
        span = _longest_span(subtasks, label)
        if span is None:
            continue
        candidates[label].append(
            {
                "episode_index": int(episode_index),
                "episode_meta": episode_meta,
                "subtasks": subtasks,
                "span": span,
            }
        )

    selected: dict[int, list[dict[str, Any]]] = {}
    for label, entries in candidates.items():
        entries.sort(key=lambda entry: int(entry["episode_index"]))
        selected_ids = set(
            _select_evenly(
                [int(entry["episode_index"]) for entry in entries],
                episodes_per_side,
            )
        )
        selected[label] = [
            entry for entry in entries if int(entry["episode_index"]) in selected_ids
        ]
    return selected


def _fixture_anchors(
    *,
    span: tuple[int, int],
    episode_length: int,
    action_horizon: int,
    compare_window: int,
    count: int,
) -> np.ndarray:
    start, end = span
    maximum = min(end - compare_window, episode_length - action_horizon)
    if maximum < start:
        return np.empty((0,), dtype=np.int64)
    candidates = np.arange(start, maximum + 1, dtype=np.int64)
    if len(candidates) <= count:
        return candidates
    positions = np.linspace(0, len(candidates) - 1, count)
    return np.unique(candidates[np.rint(positions).astype(np.int64)])


def _masked_mae(
    prediction: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    dims: np.ndarray,
) -> float:
    errors = np.abs(prediction[:, dims] - target[:, dims])
    errors = errors[np.broadcast_to(valid[:, None], errors.shape)]
    return float(errors.mean()) if errors.size else math.nan


def _cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return math.nan
    return float(1.0 - np.dot(left, right) / denominator)


def main() -> None:
    args = _parse_args()
    if args.episodes_per_side <= 0 or args.anchors_per_episode <= 0:
        raise ValueError("episodes-per-side and anchors-per-episode must be positive")
    if args.compare_window <= 0 or args.repeats <= 0:
        raise ValueError("compare-window and repeats must be positive")

    dataset_root = args.dataset_root.resolve()
    manifest_path = args.holdout_manifest.resolve()
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    holdout_indices = sorted(
        int(value) for value in manifest["datasets"][0]["holdout_episode_indices"]
    )
    subtask_text = _load_subtask_text(dataset_root)

    client = WebsocketClientPolicy(host=args.host, port=args.port, timeout=60)
    try:
        metadata = client.get_server_metadata()
        warnings = validate_realman_server_metadata(
            metadata,
            require_input_contract=True,
        )
        if warnings:
            raise RuntimeError("Policy metadata mismatch: " + "; ".join(warnings))
        action_horizon = int(metadata["action_horizon"])
        action_dim = int(metadata["action_dim"])
        if action_horizon != REALMAN_18D_ACTION_CONTRACT.action_horizon:
            raise RuntimeError(f"Unexpected action horizon {action_horizon}")
        if action_dim != REALMAN_18D_ACTION_CONTRACT.action_dim:
            raise RuntimeError(f"Unexpected action dimension {action_dim}")
        if args.compare_window > action_horizon:
            raise ValueError("compare-window exceeds the model horizon")

        action_stats = resolve_action_stats(metadata, None)
        state_stats = resolve_state_stats(metadata, None)
        action_norm_mode = resolve_norm_mode(metadata, "action", "auto")
        state_norm_mode = resolve_norm_mode(metadata, "state", "auto")
        if action_norm_mode != "q01_q99_unclipped":
            raise RuntimeError(f"Unexpected action normalization {action_norm_mode}")
        if state_norm_mode != "q01_q99_unclipped":
            raise RuntimeError(f"Unexpected state normalization {state_norm_mode}")
        image_size = resolve_qwen_frame_size(metadata)

        selected = _candidate_episodes(
            dataset_root,
            holdout_indices,
            episodes_per_side=args.episodes_per_side,
        )
        fixture_results: list[dict[str, Any]] = []
        for pickup_label in (PICKUP_LEFT, PICKUP_RIGHT):
            opposite_label = (
                PICKUP_RIGHT if pickup_label == PICKUP_LEFT else PICKUP_LEFT
            )
            for entry in selected[pickup_label]:
                episode_index = int(entry["episode_index"])
                episode_meta = entry["episode_meta"]
                episode = _load_episode_rows(dataset_root, episode_meta)
                episode_length = len(episode["frame_index"])
                anchors = _fixture_anchors(
                    span=entry["span"],
                    episode_length=episode_length,
                    action_horizon=action_horizon,
                    compare_window=args.compare_window,
                    count=args.anchors_per_episode,
                )
                if anchors.size == 0:
                    continue
                anchor_images = _decode_anchor_images(
                    dataset_root,
                    episode_meta,
                    anchors,
                    float(manifest["datasets"][0].get("fps", 20.0)),
                )
                policy_states = select_realman_policy_state(episode["source_state"])
                absolute_actions = select_realman_policy_actions(
                    episode["source_action"]
                )
                base_instruction = (
                    (episode_meta.get("tasks") or [MAGNA_DEFAULT_INSTRUCTION])[0]
                )

                for anchor in anchors:
                    conditions = {
                        "true": (
                            f"{base_instruction} | {subtask_text[pickup_label]}"
                        ),
                        "opposite": (
                            f"{base_instruction} | {subtask_text[opposite_label]}"
                        ),
                        "base_only": base_instruction,
                    }
                    condition_outputs: dict[str, dict[str, Any]] = {}
                    for condition_name, instruction in conditions.items():
                        repeated_actions: list[np.ndarray] = []
                        repeated_tokens: list[np.ndarray] = []
                        ensemble_draw_std: list[float] = []
                        for repeat_index in range(args.repeats):
                            observation = {
                                "source.observation.state": episode["source_state"][
                                    anchor, :19
                                ],
                                **{
                                    f"observation.images.{camera}": anchor_images[
                                        int(anchor)
                                    ][camera]
                                    for camera in REALMAN_CAMERA_ORDER
                                },
                            }
                            payload = build_policy_payload(
                                observation,
                                instruction=instruction,
                                image_size=image_size,
                            )
                            payload["state"] = np.ascontiguousarray(
                                normalize_q01_q99_unclipped(
                                    policy_states[anchor],
                                    state_stats,
                                )[None, None, :],
                                dtype=np.float32,
                            )
                            if args.matched_noise:
                                payload["inference_seed"] = int(
                                    1_000_003
                                    + episode_index * 10_007
                                    + int(anchor) * 101
                                    + repeat_index
                                )
                            validate_realman_policy_payload(payload, metadata)
                            response = client.infer(payload)
                            if not response.get("ok", False):
                                raise RuntimeError(
                                    f"Inference failed for episode={episode_index}, "
                                    f"anchor={anchor}, condition={condition_name}: {response}"
                                )
                            data = response["data"]
                            normalized = np.asarray(
                                data["normalized_actions"],
                                dtype=np.float32,
                            )[0]
                            mixed = realman_continuous_unnormalize(
                                normalized,
                                action_stats,
                                mode=action_norm_mode,
                            )
                            absolute = realman_policy_actions_to_absolute(
                                mixed,
                                policy_states[anchor],
                                action_type=str(metadata["action_type"]),
                            )
                            repeated_actions.append(absolute)
                            if "embodied_action_tokens" in data:
                                repeated_tokens.append(
                                    np.asarray(
                                        data["embodied_action_tokens"],
                                        dtype=np.float32,
                                    )[0]
                                )
                            if "policy_ensemble" in data:
                                ensemble_draw_std.append(
                                    float(
                                        data["policy_ensemble"][
                                            "normalized_draw_std"
                                        ]
                                    )
                                )
                        condition_outputs[condition_name] = {
                            "instruction": instruction,
                            "absolute_mean": np.mean(repeated_actions, axis=0),
                            "absolute_repeat_std": float(
                                np.std(repeated_actions, axis=0).mean()
                            ),
                            "tokens_mean": (
                                np.mean(repeated_tokens, axis=0)
                                if repeated_tokens
                                else None
                            ),
                            "ensemble_normalized_draw_std_mean": (
                                float(np.mean(ensemble_draw_std))
                                if ensemble_draw_std
                                else None
                            ),
                        }

                    target = absolute_actions[
                        anchor : anchor + args.compare_window
                    ]
                    valid = np.asarray(
                        episode["valid_state"][
                            anchor : anchor + args.compare_window
                        ],
                        dtype=bool,
                    )
                    fixture_metrics: dict[str, Any] = {}
                    for condition_name, output in condition_outputs.items():
                        prediction = output["absolute_mean"][: args.compare_window]
                        fixture_metrics[condition_name] = {
                            "arm_mae_rad": _masked_mae(
                                prediction, target, valid, ARM_DIMS
                            ),
                            "left_arm_mae_rad": _masked_mae(
                                prediction, target, valid, LEFT_ARM_DIMS
                            ),
                            "right_arm_mae_rad": _masked_mae(
                                prediction, target, valid, RIGHT_ARM_DIMS
                            ),
                            "absolute_repeat_std": output[
                                "absolute_repeat_std"
                            ],
                            "ensemble_normalized_draw_std_mean": output[
                                "ensemble_normalized_draw_std_mean"
                            ],
                        }

                    true_actions = condition_outputs["true"]["absolute_mean"][
                        : args.compare_window
                    ]
                    opposite_actions = condition_outputs["opposite"][
                        "absolute_mean"
                    ][: args.compare_window]
                    base_actions = condition_outputs["base_only"]["absolute_mean"][
                        : args.compare_window
                    ]
                    true_tokens = condition_outputs["true"]["tokens_mean"]
                    opposite_tokens = condition_outputs["opposite"]["tokens_mean"]
                    base_tokens = condition_outputs["base_only"]["tokens_mean"]
                    fixture_metrics["prompt_effect"] = {
                        "true_vs_opposite_arm_mae_rad": float(
                            np.abs(
                                true_actions[:, ARM_DIMS]
                                - opposite_actions[:, ARM_DIMS]
                            ).mean()
                        ),
                        "true_vs_base_arm_mae_rad": float(
                            np.abs(
                                true_actions[:, ARM_DIMS]
                                - base_actions[:, ARM_DIMS]
                            ).mean()
                        ),
                        "true_vs_opposite_token_cosine_distance": (
                            _cosine_distance(true_tokens, opposite_tokens)
                            if true_tokens is not None
                            and opposite_tokens is not None
                            else None
                        ),
                        "true_vs_base_token_cosine_distance": (
                            _cosine_distance(true_tokens, base_tokens)
                            if true_tokens is not None and base_tokens is not None
                            else None
                        ),
                    }
                    fixture_results.append(
                        {
                            "episode_index": episode_index,
                            "pickup_subtask": pickup_label,
                            "pickup_side": (
                                "left"
                                if pickup_label == PICKUP_LEFT
                                else "right"
                            ),
                            "anchor": int(anchor),
                            "span": entry["span"],
                            "valid_timesteps": int(valid.sum()),
                            "metrics": fixture_metrics,
                        }
                    )
                    print(
                        f"episode={episode_index:03d} side="
                        f"{fixture_results[-1]['pickup_side']:5s} anchor={int(anchor):4d} "
                        f"true_mae={fixture_metrics['true']['arm_mae_rad']:.4f} "
                        f"flip_mae={fixture_metrics['opposite']['arm_mae_rad']:.4f} "
                        f"prompt_delta="
                        f"{fixture_metrics['prompt_effect']['true_vs_opposite_arm_mae_rad']:.4f}",
                        flush=True,
                    )
    finally:
        client.close()

    aggregate: dict[str, Any] = {}
    for side in ("left", "right"):
        fixtures = [
            item for item in fixture_results if item["pickup_side"] == side
        ]
        if not fixtures:
            continue
        aggregate[side] = {
            "fixture_count": len(fixtures),
            "episode_count": len(
                {int(item["episode_index"]) for item in fixtures}
            ),
        }
        for condition in ("true", "opposite", "base_only"):
            for metric in (
                "arm_mae_rad",
                "left_arm_mae_rad",
                "right_arm_mae_rad",
                "absolute_repeat_std",
            ):
                values = np.asarray(
                    [
                        item["metrics"][condition][metric]
                        for item in fixtures
                    ],
                    dtype=np.float64,
                )
                aggregate[side][f"{condition}_{metric}_mean"] = float(
                    np.mean(values)
                )
                aggregate[side][f"{condition}_{metric}_median"] = float(
                    np.median(values)
                )
        for metric in (
            "true_vs_opposite_arm_mae_rad",
            "true_vs_base_arm_mae_rad",
            "true_vs_opposite_token_cosine_distance",
            "true_vs_base_token_cosine_distance",
        ):
            values = np.asarray(
                [
                    item["metrics"]["prompt_effect"][metric]
                    for item in fixtures
                    if item["metrics"]["prompt_effect"][metric] is not None
                ],
                dtype=np.float64,
            )
            aggregate[side][f"{metric}_mean"] = (
                float(np.mean(values)) if values.size else None
            )
            aggregate[side][f"{metric}_median"] = (
                float(np.median(values)) if values.size else None
            )

    result = {
        "schema_version": 1,
        "checkpoint_path": metadata.get("checkpoint_path"),
        "run_id": metadata.get("run_id"),
        "policy_ensemble": metadata.get("policy_ensemble"),
        "dataset_root": str(dataset_root),
        "holdout_manifest": str(manifest_path),
        "configuration": {
            "episodes_per_side": args.episodes_per_side,
            "anchors_per_episode": args.anchors_per_episode,
            "compare_window": args.compare_window,
            "repeats": args.repeats,
            "matched_noise": bool(args.matched_noise),
        },
        "aggregate": aggregate,
        "fixtures": fixture_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(result), handle, indent=2)
    print(json.dumps(_jsonable(aggregate), indent=2), flush=True)
    print(f"output={args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
