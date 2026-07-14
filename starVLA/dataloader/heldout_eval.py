"""Deterministic, one-window-per-episode LeRobot holdout evaluation.

The training mixture deliberately maps an integer index to a pseudo-random
episode and frame.  That is appropriate for training, but it cannot express
the checkpoint-evaluation contract: visit every manifest-selected holdout
episode exactly once at one immutable, structurally valid frame.  This module
adapts a separately constructed ``mode="eval"`` mixture without sharing any
dataset object, iterator, sampler, or RNG with the training loader.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Sequence

import numpy as np
import torch

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotMixtureDataset


EVAL_SAMPLING_ALGORITHM = "nonzero_valid_unpadded_uniform_v1"
EVAL_OBSERVATION_MODE = "deployment_action_current_qwen_rgb_v1"
EVAL_CANDIDATE_POLICY = (
    "structurally unpadded frames with at least one valid action-mask element"
)
EVAL_ALL_INVALID_FALLBACK = (
    "uniform over all structurally unpadded frames; report zero valid elements"
)
EVAL_SAMPLING_REPORT_SCHEMA_VERSION = 1
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class EvaluationSamplingContract:
    algorithm: str
    frames_per_episode: int
    seed_sha256: str
    observation_mode: str
    evaluation_video_offsets: tuple[int, ...]
    action_offset_range_inclusive: tuple[int, int]

    @property
    def torch_seed(self) -> int:
        # Keep the seed in torch's portable positive signed-64-bit range.
        return int(self.seed_sha256[:16], 16) % (2**63 - 1)


@dataclass(frozen=True, slots=True)
class HeldoutWindowReference:
    dataset_index: int
    dataset_name: str
    episode_id: int
    base_index: int
    valid_base_index_min: int
    valid_base_index_max: int
    structural_candidate_count: int
    evaluable_candidate_count: int
    selection_pool_candidate_count: int
    valid_action_timesteps: int
    valid_action_elements: int


def load_evaluation_sampling_contract(
    manifest_path: Path | str,
) -> EvaluationSamplingContract:
    """Load and strictly validate the manifest's checkpoint sampling rule."""

    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Episode split manifest does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Episode split manifest is invalid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Episode split manifest root must be a JSON object.")

    sampling = payload.get("evaluation_sampling")
    if not isinstance(sampling, dict):
        raise ValueError(
            "Episode split manifest must contain a top-level evaluation_sampling object."
        )
    algorithm = sampling.get("algorithm")
    if algorithm != EVAL_SAMPLING_ALGORITHM:
        raise ValueError(
            "evaluation_sampling.algorithm must be "
            f"{EVAL_SAMPLING_ALGORITHM!r}, got {algorithm!r}."
        )
    frames_per_episode = sampling.get("frames_per_episode")
    if isinstance(frames_per_episode, bool) or frames_per_episode != 1:
        raise ValueError(
            "In-training checkpoint evaluation requires exactly one frame per "
            f"holdout episode; got frames_per_episode={frames_per_episode!r}."
        )
    seed_sha256 = sampling.get("seed_sha256")
    if not isinstance(seed_sha256, str) or _SHA256_RE.fullmatch(seed_sha256) is None:
        raise ValueError(
            "evaluation_sampling.seed_sha256 must be exactly 64 lowercase hex characters."
        )
    observation_mode = sampling.get("observation_mode")
    if observation_mode != EVAL_OBSERVATION_MODE:
        raise ValueError(
            "evaluation_sampling.observation_mode must be "
            f"{EVAL_OBSERVATION_MODE!r}, got {observation_mode!r}."
        )
    evaluation_video_offsets = sampling.get("evaluation_video_offsets")
    if evaluation_video_offsets != [0]:
        raise ValueError(
            "Deployment-style in-training action eval requires exactly "
            "evaluation_video_offsets=[0]."
        )
    raw_action_range = sampling.get("action_offset_range_inclusive")
    if (
        not isinstance(raw_action_range, list)
        or len(raw_action_range) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in raw_action_range)
        or int(raw_action_range[0]) > int(raw_action_range[1])
    ):
        raise ValueError(
            "evaluation_sampling.action_offset_range_inclusive must be two "
            "ordered integer offsets."
        )
    if sampling.get("candidate_policy") != EVAL_CANDIDATE_POLICY:
        raise ValueError(
            "evaluation_sampling.candidate_policy does not match the supported "
            "nonzero-valid unpadded sampling contract."
        )
    if sampling.get("all_invalid_episode_fallback") != EVAL_ALL_INVALID_FALLBACK:
        raise ValueError(
            "evaluation_sampling.all_invalid_episode_fallback does not match "
            "the supported zero-metric forwarding contract."
        )
    return EvaluationSamplingContract(
        algorithm=algorithm,
        frames_per_episode=1,
        seed_sha256=seed_sha256,
        observation_mode=observation_mode,
        evaluation_video_offsets=(0,),
        action_offset_range_inclusive=(
            int(raw_action_range[0]),
            int(raw_action_range[1]),
        ),
    )


def _used_modality_offsets(dataset, modality: str) -> np.ndarray:
    """Return offsets actually consumed by the mixture's packed sample.

    Action chunks consume every configured action offset.  State and language
    are fetched at the observation horizon but the mixture packs only their
    first (current) row, so only that first offset constrains a valid window.
    """

    keys = list(dataset.modality_keys.get(modality, ()))
    if not keys:
        return np.empty(0, dtype=np.int64)
    offsets = [np.asarray(dataset.delta_indices[key], dtype=np.int64) for key in keys]
    if modality in {"state", "language"}:
        return np.asarray([int(values[0]) for values in offsets], dtype=np.int64)
    return np.concatenate(offsets)


def configure_deployment_current_frame_observation(
    mixture: LeRobotMixtureDataset,
) -> None:
    """Make the independent eval view match deployed action-policy observation.

    Training's compact V-JEPA clip is auxiliary world-model context.  Action
    deployment sends only the current RGB frame to Qwen, so checkpoint action
    eval must not decode or constrain anchors by that eight-frame clip.
    """

    mixture.video_target_shift_steps = 0
    for dataset in mixture.datasets:
        for video_key in dataset.modality_keys.get("video", ()):
            dataset.delta_indices[video_key] = np.asarray([0], dtype=np.int64)


def _validate_sampling_offsets(
    mixture: LeRobotMixtureDataset,
    contract: EvaluationSamplingContract,
) -> None:
    for dataset in mixture.datasets:
        for video_key in dataset.modality_keys.get("video", ()):
            video_offsets = tuple(
                int(value)
                for value in np.asarray(
                    dataset.delta_indices[video_key], dtype=np.int64
                ).reshape(-1)
            )
            if video_offsets != contract.evaluation_video_offsets:
                raise ValueError(
                    f"Dataset {dataset.dataset_name!r} eval video offsets "
                    f"{video_offsets} do not match manifest "
                    f"{contract.evaluation_video_offsets}."
                )
        action_offsets = _used_modality_offsets(dataset, "action")
        if action_offsets.size == 0:
            raise ValueError(f"Dataset {dataset.dataset_name!r} has no action offsets.")
        action_range = (int(action_offsets.min()), int(action_offsets.max()))
        if action_range != contract.action_offset_range_inclusive:
            raise ValueError(
                f"Dataset {dataset.dataset_name!r} action offset range "
                f"{action_range} does not match manifest "
                f"{contract.action_offset_range_inclusive}."
            )


def required_window_offsets(
    mixture: LeRobotMixtureDataset,
    dataset,
) -> np.ndarray:
    """Offsets that must remain inside an episode for an unpadded eval sample."""

    parts = [
        _used_modality_offsets(dataset, "action"),
        _used_modality_offsets(dataset, "state"),
        _used_modality_offsets(dataset, "language"),
    ]

    video_keys = list(dataset.modality_keys.get("video", ()))
    if video_keys:
        target_shift = int(getattr(mixture, "video_target_shift_steps", 0) or 0)
        if target_shift > 0:
            horizons = {
                len(np.asarray(dataset.delta_indices[key]).reshape(-1))
                for key in video_keys
            }
            if len(horizons) != 1:
                raise ValueError(
                    f"Dataset {dataset.dataset_name!r} has inconsistent video horizons: "
                    f"{sorted(horizons)}."
                )
            video_horizon = next(iter(horizons))
            context_horizon = video_horizon - target_shift
            if context_horizon <= 0:
                raise ValueError(
                    f"Dataset {dataset.dataset_name!r} video horizon {video_horizon} "
                    f"must exceed video_target_shift_steps={target_shift}."
                )
            stride = max(int(getattr(mixture, "video_frame_stride", 1)), 1)
            parts.append(
                np.arange(
                    -(context_horizon - 1),
                    target_shift + 1,
                    dtype=np.int64,
                )
                * stride
            )
        else:
            parts.extend(
                np.asarray(dataset.delta_indices[key], dtype=np.int64)
                for key in video_keys
            )

    nonempty = [values.reshape(-1) for values in parts if values.size]
    if not nonempty:
        raise ValueError(
            f"Dataset {dataset.dataset_name!r} has no configured evaluation modalities."
        )
    return np.unique(np.concatenate(nonempty))


def _episode_candidate_steps(
    *,
    dataset,
    episode_id: int,
    episode_length: int,
    required_offsets: np.ndarray,
) -> tuple[list[int], int, int]:
    valid_min = max(0, -int(required_offsets.min()))
    valid_max = min(
        int(episode_length) - 1,
        int(episode_length) - 1 - int(required_offsets.max()),
    )
    if valid_max < valid_min:
        raise ValueError(
            f"Heldout episode {dataset.dataset_name}/{episode_id} has length "
            f"{episode_length}, but the training context/action offsets "
            f"[{int(required_offsets.min())}, {int(required_offsets.max())}] leave "
            "no structurally valid unpadded evaluation window."
        )

    # ``all_steps`` incorporates any dataset-specific candidate filtering (for
    # example language validity or pause-frame deletion).  Intersecting it with
    # the structural interval avoids silently selecting a padded window.
    candidates = sorted(
        {
            int(base_index)
            for trajectory_id, base_index in dataset.all_steps
            if int(trajectory_id) == int(episode_id)
            and valid_min <= int(base_index) <= valid_max
        }
    )
    if not candidates:
        raise ValueError(
            f"Heldout episode {dataset.dataset_name}/{episode_id} has no valid "
            f"candidate steps in [{valid_min}, {valid_max}]."
        )
    return candidates, valid_min, valid_max


def _candidate_valid_action_counts(
    *,
    mixture: LeRobotMixtureDataset,
    dataset,
    episode_id: int,
    candidates: Sequence[int],
    action_dim: int,
) -> list[tuple[int, int, int]]:
    """Return ``(step, valid_timesteps, valid_elements)`` using train masking."""

    action_keys = list(dataset.modality_keys.get("action", ()))
    if not action_keys:
        raise ValueError(f"Dataset {dataset.dataset_name!r} has no action modality.")
    action_offsets = [
        np.asarray(dataset.delta_indices[key], dtype=np.int64).reshape(-1)
        for key in action_keys
    ]
    if any(not np.array_equal(action_offsets[0], values) for values in action_offsets[1:]):
        raise ValueError(
            f"Dataset {dataset.dataset_name!r} action keys have inconsistent horizons."
        )
    action_horizon = int(action_offsets[0].size)
    if action_horizon <= 0:
        raise ValueError(f"Dataset {dataset.dataset_name!r} has an empty action horizon.")

    # Load only the episode parquet.  The training mask helper reads labels from
    # curr_traj_data and never decodes video or builds model image inputs.
    dataset.curr_traj_data = dataset.get_trajectory_data(int(episode_id))
    dummy_action = np.zeros((action_horizon, int(action_dim)), dtype=np.float32)
    no_padding = np.zeros(action_horizon, dtype=bool)
    counts: list[tuple[int, int, int]] = []
    for step in candidates:
        mask = mixture._build_action_validity_mask(
            dataset,
            step=int(step),
            action=dummy_action,
            action_is_pad=no_padding,
        )
        if mask is None:
            valid_timesteps = action_horizon
            valid_elements = action_horizon * int(action_dim)
        else:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != dummy_action.shape:
                raise ValueError(
                    f"Training action-validity mask for {dataset.dataset_name}/{episode_id} "
                    f"has shape {mask.shape}, expected {dummy_action.shape}."
                )
            valid_elements = int(mask.sum())
            valid_timesteps = int(mask.any(axis=1).sum())
        counts.append((int(step), valid_timesteps, valid_elements))
    return counts


def _uniform_candidate_index(
    *,
    contract: EvaluationSamplingContract,
    dataset_name: str,
    episode_id: int,
    candidate_count: int,
) -> int:
    digest = hashlib.sha256(
        _canonical_json_bytes(
            {
                "algorithm": contract.algorithm,
                "seed_sha256": contract.seed_sha256,
                "dataset_name": str(dataset_name),
                "episode_id": int(episode_id),
            }
        )
    ).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % candidate_count


def select_heldout_window_references(
    mixture: LeRobotMixtureDataset,
    contract: EvaluationSamplingContract,
    *,
    action_dim: int,
) -> tuple[HeldoutWindowReference, ...]:
    """Select one stable, unpadded base index from every heldout episode."""

    references: list[HeldoutWindowReference] = []
    seen_identities: set[tuple[str, int]] = set()
    for dataset_index, dataset in enumerate(mixture.datasets):
        selection = getattr(dataset, "_episode_split_selection", None)
        if selection is None:
            raise ValueError(
                f"Evaluation dataset {dataset.dataset_name!r} was not selected by "
                "an episode split manifest."
            )
        if str(getattr(selection, "role", "")) not in {
            "eval",
            "evaluation",
            "val",
            "validation",
            "test",
            "holdout",
        }:
            raise ValueError(
                f"Evaluation dataset {dataset.dataset_name!r} has non-holdout "
                f"episode split role {getattr(selection, 'role', None)!r}."
            )

        selected_ids = tuple(int(value) for value in dataset.trajectory_ids.tolist())
        manifest_ids = tuple(int(value) for value in selection.selected_episode_ids)
        if set(selected_ids) != set(manifest_ids) or len(selected_ids) != len(manifest_ids):
            raise ValueError(
                f"Evaluation dataset {dataset.dataset_name!r} episode IDs do not "
                "exactly match its manifest-selected holdout IDs."
            )
        lengths_by_id = {
            int(episode_id): int(length)
            for episode_id, length in zip(
                dataset.trajectory_ids.tolist(),
                dataset.trajectory_lengths.tolist(),
            )
        }
        offsets = required_window_offsets(mixture, dataset)
        for episode_id in sorted(selected_ids):
            identity = (str(dataset.dataset_name), int(episode_id))
            if identity in seen_identities:
                raise ValueError(f"Duplicate heldout episode identity {identity!r}.")
            seen_identities.add(identity)
            candidates, valid_min, valid_max = _episode_candidate_steps(
                dataset=dataset,
                episode_id=episode_id,
                episode_length=lengths_by_id[episode_id],
                required_offsets=offsets,
            )
            candidate_counts = _candidate_valid_action_counts(
                mixture=mixture,
                dataset=dataset,
                episode_id=episode_id,
                candidates=candidates,
                action_dim=action_dim,
            )
            evaluable = [entry for entry in candidate_counts if entry[2] > 0]
            # Sample uniformly over every structurally unpadded candidate with
            # at least one supervised action element.  This avoids optimistic
            # max-valid anchor bias.  A manifest-random episode that is entirely
            # invalid falls back to all structural candidates, is still
            # forwarded, and is explicitly excluded from action metrics.
            selection_pool = evaluable if evaluable else candidate_counts
            choice = _uniform_candidate_index(
                contract=contract,
                dataset_name=dataset.dataset_name,
                episode_id=episode_id,
                candidate_count=len(selection_pool),
            )
            selected_step, valid_timesteps, valid_elements = selection_pool[choice]
            references.append(
                HeldoutWindowReference(
                    dataset_index=int(dataset_index),
                    dataset_name=str(dataset.dataset_name),
                    episode_id=int(episode_id),
                    base_index=int(selected_step),
                    valid_base_index_min=int(valid_min),
                    valid_base_index_max=int(valid_max),
                    structural_candidate_count=len(candidates),
                    evaluable_candidate_count=len(evaluable),
                    selection_pool_candidate_count=len(selection_pool),
                    valid_action_timesteps=int(valid_timesteps),
                    valid_action_elements=int(valid_elements),
                )
            )
    if not references:
        raise ValueError("The manifest-selected holdout contains no episodes.")
    return tuple(
        sorted(
            references,
            key=lambda ref: (ref.dataset_name, ref.episode_id, ref.dataset_index),
        )
    )


def validate_global_eval_observation_count(
    *,
    holdout_episode_count: int,
    per_device_batch_size: int,
    world_size: int,
    gradient_accumulation_steps: int,
) -> int:
    """Require exactly one effective global training batch of holdout episodes."""

    values = {
        "holdout_episode_count": holdout_episode_count,
        "per_device_batch_size": per_device_batch_size,
        "world_size": world_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
    }
    for name, value in values.items():
        if isinstance(value, bool) or int(value) <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    expected = (
        int(per_device_batch_size)
        * int(world_size)
        * int(gradient_accumulation_steps)
    )
    if int(holdout_episode_count) != expected:
        raise ValueError(
            "The manifest holdout must contain exactly one effective global "
            "training batch: "
            f"holdout episodes={int(holdout_episode_count)}, expected={expected} "
            f"({int(per_device_batch_size)} per device * {int(world_size)} ranks * "
            f"{int(gradient_accumulation_steps)} gradient accumulation steps)."
        )
    return expected


class DeterministicHeldoutEvalDataset(LeRobotMixtureDataset):
    """A separate eval mixture whose index is a fixed episode/window reference."""

    def __init__(
        self,
        source: LeRobotMixtureDataset,
        *,
        sampling_contract: EvaluationSamplingContract,
        action_dim: int,
    ) -> None:
        if not isinstance(source, LeRobotMixtureDataset):
            raise TypeError(
                "Deterministic heldout evaluation currently supports only "
                f"LeRobotMixtureDataset, got {type(source).__name__}."
            )
        # The source was constructed independently with mode="eval".  Copying
        # its attributes avoids re-running expensive catalog/statistics setup;
        # it does not share with the separately constructed training dataset.
        self.__dict__.update(source.__dict__)
        self.mode = "eval"
        if isinstance(action_dim, bool) or int(action_dim) <= 0:
            raise ValueError(f"action_dim must be a positive integer, got {action_dim!r}.")
        self.heldout_eval_action_dim = int(action_dim)
        self.evaluation_sampling_contract = sampling_contract
        configure_deployment_current_frame_observation(self)
        _validate_sampling_offsets(self, sampling_contract)
        self.heldout_window_references = select_heldout_window_references(
            self,
            sampling_contract,
            action_dim=self.heldout_eval_action_dim,
        )
        self.heldout_window_digest = _canonical_sha256(
            {
                "algorithm": sampling_contract.algorithm,
                "observation_mode": sampling_contract.observation_mode,
                "evaluation_video_offsets": list(
                    sampling_contract.evaluation_video_offsets
                ),
                "action_offset_range_inclusive": list(
                    sampling_contract.action_offset_range_inclusive
                ),
                "frames_per_episode": sampling_contract.frames_per_episode,
                "seed_sha256": sampling_contract.seed_sha256,
                "windows": [asdict(reference) for reference in self.heldout_window_references],
            }
        )
        self.close_eval_caches()

    @classmethod
    def from_manifest(
        cls,
        source: LeRobotMixtureDataset,
        manifest_path: Path | str,
        *,
        action_dim: int,
    ) -> "DeterministicHeldoutEvalDataset":
        return cls(
            source,
            sampling_contract=load_evaluation_sampling_contract(manifest_path),
            action_dim=action_dim,
        )

    def __len__(self) -> int:
        return len(self.heldout_window_references)

    def sample_step(self, index: int):
        reference = getattr(self, "_active_heldout_window_reference", None)
        if reference is None:
            reference = self.heldout_window_references[int(index)]
        return (
            self.datasets[reference.dataset_index],
            reference.episode_id,
            reference.base_index,
        )

    def __getitem__(self, index: int) -> dict:
        eval_index = int(index)
        reference = self.heldout_window_references[eval_index]
        sentinel = object()
        previous = getattr(self, "_active_heldout_window_reference", sentinel)
        self._active_heldout_window_reference = reference
        try:
            # The parent has retry handling.  Pinning the active reference makes
            # every retry target the same episode/window instead of its fallback
            # random index, so a corrupt heldout sample fails closed.
            sample = super().__getitem__(eval_index)
        finally:
            if previous is sentinel:
                del self._active_heldout_window_reference
            else:
                self._active_heldout_window_reference = previous
        sample["_heldout_eval_index"] = eval_index
        sample["_heldout_eval_dataset_name"] = reference.dataset_name
        sample["_heldout_eval_episode_id"] = reference.episode_id
        sample["_heldout_eval_base_index"] = reference.base_index
        return sample

    def sampling_report(self) -> dict[str, Any]:
        return {
            "schema_version": EVAL_SAMPLING_REPORT_SCHEMA_VERSION,
            "purpose": "one_window_per_manifest_holdout_episode_checkpoint_eval",
            "algorithm": self.evaluation_sampling_contract.algorithm,
            "observation_mode": self.evaluation_sampling_contract.observation_mode,
            "evaluation_video_offsets": list(
                self.evaluation_sampling_contract.evaluation_video_offsets
            ),
            "action_offset_range_inclusive": list(
                self.evaluation_sampling_contract.action_offset_range_inclusive
            ),
            "frames_per_episode": self.evaluation_sampling_contract.frames_per_episode,
            "seed_sha256": self.evaluation_sampling_contract.seed_sha256,
            "observation_count": len(self),
            "action_evaluable_observation_count": sum(
                reference.valid_action_elements > 0
                for reference in self.heldout_window_references
            ),
            "action_dim": self.heldout_eval_action_dim,
            "valid_action_timestep_count": sum(
                reference.valid_action_timesteps
                for reference in self.heldout_window_references
            ),
            "valid_action_element_count": sum(
                reference.valid_action_elements
                for reference in self.heldout_window_references
            ),
            "zero_valid_action_episodes": [
                {
                    "dataset_name": reference.dataset_name,
                    "episode_id": reference.episode_id,
                    "base_index": reference.base_index,
                }
                for reference in self.heldout_window_references
                if reference.valid_action_elements == 0
            ],
            "window_selection_sha256": self.heldout_window_digest,
            "windows": [asdict(reference) for reference in self.heldout_window_references],
        }

    def close_eval_caches(self) -> None:
        """Release eval-only parquet/video readers after selection or a pass."""

        for dataset in self.datasets:
            close_parquet = getattr(dataset, "close_parquet_cache", None)
            if callable(close_parquet):
                close_parquet()
            close_video = getattr(dataset, "close_video_readers", None)
            if callable(close_video):
                close_video()

    def save_sampling_report(self, path: Path | str) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.sampling_report(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def make_torch_generator(self) -> torch.Generator:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.evaluation_sampling_contract.torch_seed)
        return generator


def build_heldout_eval_dataset(
    source: LeRobotMixtureDataset,
    *,
    manifest_path: Path | str,
    action_dim: int,
) -> DeterministicHeldoutEvalDataset:
    return DeterministicHeldoutEvalDataset.from_manifest(
        source,
        manifest_path,
        action_dim=action_dim,
    )


def sampling_seed_from_dataset(dataset: Any, default: int = 0) -> int:
    """Resolve the dedicated inference seed through common loader wrappers."""

    visited: set[int] = set()
    pending = [dataset]
    while pending:
        current = pending.pop()
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))
        contract = getattr(current, "evaluation_sampling_contract", None)
        if contract is not None:
            return int(contract.torch_seed)
        for attr in ("dataset", "base_dataloader", "dataloader"):
            nested = getattr(current, attr, None)
            if nested is not None:
                pending.append(nested)
    return int(default)
