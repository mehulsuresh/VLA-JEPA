import copy
import hashlib
import json
import math
import multiprocessing as mp
import os
import signal
import sys
from accelerate.logging import get_logger
import atexit
import faulthandler
from functools import partial
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.distributed as dist
from pathlib import Path

try:
    import av as _av

    _av.logging.set_level(_av.logging.PANIC)
except Exception:
    pass

logger = get_logger(__name__)

_EXHAUSTIVE_EPOCH_SAMPLING_STRATEGIES = frozenset(
    {"primary_exhaustive", "all_sources_exhaustive"}
)
_CANONICAL_REALMAN_ACTION_DIM = 18
_CANONICAL_SEMANTIC_FLAT_ACTION_DIM = 49
_EXHAUSTIVE_GLOBAL_AFFINE = "global_affine"
_EXHAUSTIVE_VIDEO_LOCAL_BLOCKS = "video_local_blocks"
_CANONICAL_VIDEO_LOCAL_SAMPLER_VERSION = (
    "all_sources_exhaustive_frozen_view_video_local_sampler_v1"
)
_CANONICAL_FROZEN_AFFINE_VERSION = (
    "all_sources_exhaustive_frozen_view_affine_v1"
)


class _CanonicalVideoLocalExhaustiveSampler(
    torch.utils.data.SequentialSampler
):
    """Reorder exact canonical epochs by episode without changing row semantics.

    The canonical dataset already maps each loader index through a deterministic
    affine bijection. This sampler emits the inverse affine loader index for a
    desired raw row, allowing compact episode ranges to remain contiguous while
    preserving the dataset's exact once-per-epoch contract.
    """

    algorithm_version = _CANONICAL_VIDEO_LOCAL_SAMPLER_VERSION

    def __init__(self, dataset) -> None:
        super().__init__(dataset)
        if (
            getattr(dataset, "epoch_sampling_algorithm_version", None)
            != _CANONICAL_FROZEN_AFFINE_VERSION
        ):
            raise ValueError(
                "Canonical video-local exhaustive sampling requires the "
                "authenticated frozen-view affine dataset schedule."
            )
        frozen_view = getattr(dataset, "frozen_train_view", None)
        if (
            frozen_view is None
            or getattr(frozen_view, "encoding", None)
            != "episode_ranges_v1"
        ):
            raise ValueError(
                "Canonical exhaustive_window_order='video_local_blocks' "
                "requires a frozen episode_ranges_v1 view; expanded row views "
                "do not authenticate episode block boundaries."
            )
        offsets, _ = dataset._open_frozen_view_readers()
        cumulative = np.asarray(offsets[:, 1], dtype=np.int64).copy()
        length = int(len(dataset))
        if (
            cumulative.ndim != 1
            or cumulative.size < 2
            or int(cumulative[0]) != 0
            or int(cumulative[-1]) != length
            or np.any(cumulative[1:] <= cumulative[:-1])
        ):
            raise RuntimeError(
                "Canonical video-local exhaustive block boundaries are "
                "invalid or do not cover the exact logical epoch."
            )
        self._block_starts = cumulative[:-1]
        self._block_ends = cumulative[1:]
        self._length = length
        self.epoch = int(getattr(dataset, "current_epoch", 0))
        self.source_algorithm_version = str(
            dataset.epoch_sampling_algorithm_version
        )
        dataset.epoch_sampling_algorithm_version = self.algorithm_version
        dataset.exhaustive_window_order = (
            _EXHAUSTIVE_VIDEO_LOCAL_BLOCKS
        )

    @staticmethod
    def _affine_parameters(length: int, seed: int) -> tuple[int, int]:
        if length <= 1:
            return 1, 0
        multiplier = int(seed % length) or 1
        while math.gcd(multiplier, length) != 1:
            multiplier += 1
            if multiplier >= length:
                multiplier = 1
        offset = int((seed >> 64) % length)
        return multiplier, offset

    def _dataset_inverse_affine(self, epoch: int) -> tuple[int, int]:
        if self._length <= 1:
            return 1, 0
        dataset = self.data_source
        seed_payload = (
            f"{dataset.epoch_sampling_strategy}|{int(epoch)}|"
            f"{int(dataset.seed)}|{self._length}"
        ).encode("utf-8")
        permutation_seed = int.from_bytes(
            hashlib.sha256(seed_payload).digest()[:16],
            byteorder="big",
            signed=False,
        )
        multiplier, offset = self._affine_parameters(
            self._length, permutation_seed
        )
        return pow(multiplier, -1, self._length), offset

    def _block_order(self, epoch: int) -> np.ndarray:
        block_count = int(self._block_starts.size)
        seed_payload = (
            f"{self.algorithm_version}|{int(self.data_source.seed)}|"
            f"{self._length}|{block_count}"
        ).encode("utf-8")
        permutation_seed = int.from_bytes(
            hashlib.sha256(seed_payload).digest()[:16],
            byteorder="big",
            signed=False,
        )
        multiplier, offset = self._affine_parameters(
            block_count, permutation_seed
        )
        positions = (
            np.arange(block_count, dtype=np.int64) + int(epoch)
        ) % block_count
        return (multiplier * positions + offset) % block_count

    def __iter__(self):
        dataset = self.data_source
        dataset.set_epoch(self.epoch)
        inverse_multiplier, dataset_offset = (
            self._dataset_inverse_affine(self.epoch)
        )
        for block_index in self._block_order(self.epoch):
            start = int(self._block_starts[int(block_index)])
            stop = int(self._block_ends[int(block_index)])
            for raw_index in range(start, stop):
                yield int(
                    inverse_multiplier
                    * ((raw_index - dataset_offset) % self._length)
                    % self._length
                )

    def __len__(self) -> int:
        return self._length

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or int(epoch) < 0:
            raise ValueError(
                f"epoch must be a non-negative integer, got {epoch!r}"
            )
        self.epoch = int(epoch)
        self.data_source.set_epoch(self.epoch)


def _canonical_eval_metric_groups(action_dim: int) -> list[str]:
    """Return compact canonical groups supported by the trainer action layout."""

    if int(action_dim) == _CANONICAL_REALMAN_ACTION_DIM:
        return ["all_action", "arm", "gripper"]
    if int(action_dim) == _CANONICAL_SEMANTIC_FLAT_ACTION_DIM:
        return ["all_action", "arm", "hand"]
    raise ValueError(
        "Canonical heldout evaluation supports compact metric groups only for "
        "the 18-D RealMan or 49-D semantic-flat action layouts; "
        f"got action_dim={action_dim}."
    )


def _normalize_canonical_eval_metric_groups(dataset) -> None:
    """Bind the runtime canonical report to its projected policy action layout.

    The canonical dataset's adapter semantics and immutable cache hashes are
    deliberately independent of trainer diagnostics.  Normalize the report in
    this construction wrapper so both the trainer and the saved
    ``heldout_eval_windows.json`` see the correct group names without changing
    canonical sample, adapter, or sidecar semantics.
    """

    report = getattr(dataset, "_sampling_report", None)
    if not isinstance(report, dict):
        raise RuntimeError(
            "Deterministic canonical eval dataset lacks its runtime sampling "
            "report."
        )
    raw_action_dim = report.get("action_dim")
    if isinstance(raw_action_dim, bool) or not isinstance(raw_action_dim, int):
        raise RuntimeError(
            "Canonical runtime sampling report has an invalid action_dim: "
            f"{raw_action_dim!r}."
        )
    expected = _canonical_eval_metric_groups(raw_action_dim)
    configured = report.get("metric_groups")
    allowed_pre_normalization = (
        expected,
        ["all_action", "arm", "hand"],
    )
    if configured not in allowed_pre_normalization:
        raise RuntimeError(
            "Canonical runtime sampling report has unexpected compact metric "
            f"groups for action_dim={raw_action_dim}: {configured!r}."
        )
    report["metric_groups"] = expected


def _identity_collate(batch):
    return batch


def _resolve_epoch_loader_contract(
    vla_dataset_cfg,
    *,
    dataset_py: str,
    is_eval: bool,
) -> tuple[bool, bool, bool]:
    """Resolve loader ordering without weakening exhaustive epoch coverage.

    The dataset owns the deterministic per-epoch permutation.  A second
    DataLoader shuffle would obscure the authenticated cursor, while
    ``drop_last`` would silently omit the final selected rows.  Window caps and
    sample strides alter the selected population itself, so those remain
    explicit configuration errors instead of being silently rewritten.
    """

    if is_eval:
        return False, False, False
    strategy = str(
        vla_dataset_cfg.get("epoch_sampling_strategy", "with_replacement")
    ).strip().lower()
    exhaustive = strategy in _EXHAUSTIVE_EPOCH_SAMPLING_STRATEGIES
    if not exhaustive:
        return (
            False,
            bool(vla_dataset_cfg.get("shuffle", True)),
            bool(vla_dataset_cfg.get("drop_last", True)),
        )

    incompatible: list[str] = []
    for key in (
        "max_shards",
        "max_shards_per_dataset",
        "max_windows",
        "max_windows_per_dataset",
    ):
        if int(vla_dataset_cfg.get(key, 0) or 0) != 0:
            incompatible.append(key)
    if bool(vla_dataset_cfg.get("shuffle_shards", False)):
        incompatible.append("shuffle_shards")
    if int(vla_dataset_cfg.get("sample_stride", 1)) != 1:
        incompatible.append("sample_stride")
    if incompatible:
        raise ValueError(
            f"{dataset_py} epoch_sampling_strategy={strategy!r} must expose "
            "every selected row exactly once per logical epoch; incompatible "
            f"settings: {', '.join(incompatible)}."
        )

    # These are deliberately resolved from the sampling strategy, rather than
    # trusted as independent booleans, so omitted or stale legacy defaults
    # cannot shorten or reshuffle an exhaustive epoch.
    return True, False, False


def _validate_exhaustive_dataset_schedule(dataset) -> None:
    """Fail closed if a dataset's public exact-epoch schedule is inconsistent."""

    strategy = str(
        getattr(dataset, "epoch_sampling_strategy", "with_replacement")
    ).strip().lower()
    if strategy not in _EXHAUSTIVE_EPOCH_SAMPLING_STRATEGIES:
        return
    algorithm_version = getattr(
        dataset, "epoch_sampling_algorithm_version", None
    )
    if not isinstance(algorithm_version, str) or not algorithm_version:
        raise RuntimeError(
            "Exhaustive dataset lacks epoch_sampling_algorithm_version."
        )
    if not callable(getattr(dataset, "set_epoch", None)):
        raise RuntimeError("Exhaustive dataset lacks set_epoch().")
    if not bool(getattr(dataset, "fail_on_sample_error", False)):
        raise RuntimeError(
            "Exhaustive dataset must set fail_on_sample_error=true; retry "
            "substitution would break exact epoch coverage."
        )

    lengths = np.asarray(
        getattr(dataset, "dataset_lengths", ()), dtype=np.int64
    )
    counts = np.asarray(
        getattr(dataset, "epoch_dataset_counts", ()), dtype=np.int64
    )
    primary = np.asarray(
        getattr(dataset, "primary_dataset_indices", ()), dtype=np.bool_
    )
    if (
        lengths.ndim != 1
        or lengths.size == 0
        or counts.shape != lengths.shape
        or primary.shape != lengths.shape
        or np.any(lengths <= 0)
        or np.any(counts < 0)
    ):
        raise RuntimeError(
            "Exhaustive dataset schedule arrays are missing or inconsistent."
        )
    if strategy == "all_sources_exhaustive" and not np.array_equal(
        counts, lengths
    ):
        raise RuntimeError(
            "all_sources_exhaustive must schedule every retained dataset row "
            "exactly once."
        )
    scheduled_rows = int(counts.sum(dtype=np.int64))
    if int(len(dataset)) != scheduled_rows:
        raise RuntimeError(
            "Exhaustive dataset length differs from its logical epoch: "
            f"len={len(dataset)}, scheduled_rows={scheduled_rows}."
        )


def _validate_exhaustive_dataloader(dataset, dataloader: DataLoader) -> None:
    """Verify the concrete loader preserves the dataset-owned epoch schedule."""

    strategy = str(
        getattr(dataset, "epoch_sampling_strategy", "with_replacement")
    ).strip().lower()
    if strategy not in _EXHAUSTIVE_EPOCH_SAMPLING_STRATEGIES:
        return
    if bool(dataloader.drop_last):
        raise RuntimeError(
            "Exhaustive epoch DataLoader must use drop_last=false."
        )
    if not isinstance(
        dataloader.sampler, torch.utils.data.SequentialSampler
    ):
        raise RuntimeError(
            "Exhaustive epoch DataLoader must use shuffle=false; the dataset "
            "owns the deterministic epoch permutation."
        )
    window_order = str(
        getattr(
            dataset,
            "exhaustive_window_order",
            _EXHAUSTIVE_GLOBAL_AFFINE,
        )
    )
    if (
        window_order == _EXHAUSTIVE_VIDEO_LOCAL_BLOCKS
        and not isinstance(
            dataloader.sampler,
            _CanonicalVideoLocalExhaustiveSampler,
        )
    ):
        raise RuntimeError(
            "Canonical video-local exhaustive training lost its authenticated "
            "episode-block sampler."
        )


def _resolve_canonical_exhaustive_sampler(
    vla_dataset_cfg,
    dataset,
    *,
    exhaustive_training: bool,
    is_eval: bool,
):
    raw_order = vla_dataset_cfg.get(
        "exhaustive_window_order",
        _EXHAUSTIVE_GLOBAL_AFFINE,
    )
    if not isinstance(raw_order, str) or not raw_order.strip():
        raise ValueError(
            "datasets.vla_data.exhaustive_window_order must be a non-empty "
            "string."
        )
    window_order = raw_order.strip().lower()
    if window_order not in {
        _EXHAUSTIVE_GLOBAL_AFFINE,
        _EXHAUSTIVE_VIDEO_LOCAL_BLOCKS,
    }:
        raise ValueError(
            "Canonical exhaustive_window_order must be "
            f"{_EXHAUSTIVE_GLOBAL_AFFINE!r} or "
            f"{_EXHAUSTIVE_VIDEO_LOCAL_BLOCKS!r}; got "
            f"{window_order!r}."
        )
    if is_eval:
        return None
    if not exhaustive_training:
        if window_order != _EXHAUSTIVE_GLOBAL_AFFINE:
            raise ValueError(
                "Canonical exhaustive_window_order="
                f"{window_order!r} requires "
                "epoch_sampling_strategy='all_sources_exhaustive'."
            )
        return None
    dataset.exhaustive_window_order = window_order
    if window_order == _EXHAUSTIVE_VIDEO_LOCAL_BLOCKS:
        return _CanonicalVideoLocalExhaustiveSampler(dataset)
    return None


def _resolve_worker_multiprocessing_context(
    vla_dataset_cfg,
    *,
    is_eval: bool,
    num_workers: int,
) -> str | None:
    """Validate the process start method before a worker can be created.

    Training constructs its model (and therefore may initialize CUDA and
    native helper threads) before the first DataLoader iterator is made.
    ``fork`` is consequently never safe here.  ``spawn`` and ``forkserver``
    preserve the existing pickle-based dataset contract, while ``forkserver``
    avoids spawning every worker directly from a large CUDA rank.
    """

    if num_workers <= 0:
        return None
    config_key = (
        "eval_multiprocessing_context" if is_eval else "multiprocessing_context"
    )
    raw_context = vla_dataset_cfg.get(
        config_key,
        vla_dataset_cfg.get("multiprocessing_context", "spawn"),
    )
    if not isinstance(raw_context, str) or not raw_context.strip():
        raise ValueError(
            f"datasets.vla_data.{config_key} must be a non-empty start-method string"
        )
    context = raw_context.strip().lower()
    supported_contexts = set(mp.get_all_start_methods())
    if context not in supported_contexts:
        raise ValueError(
            f"DataLoader multiprocessing context {context!r} is unavailable; "
            f"supported contexts are {sorted(supported_contexts)}"
        )
    if context == "fork":
        raise ValueError(
            "DataLoader multiprocessing_context='fork' is forbidden because "
            "workers start after model/CUDA initialization. Use 'forkserver' "
            "or 'spawn'."
        )
    return context


def _host_memory_gib() -> float | None:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    return float(line.split()[1]) / 1024.0 / 1024.0
    except OSError:
        return None
    return None


def _distributed_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return max(1, int(dist.get_world_size()))
    for key in ("WORLD_SIZE", "SLURM_NTASKS"):
        value = os.environ.get(key)
        if value:
            try:
                return max(1, int(value))
            except ValueError:
                continue
    return 1


def _positive_loader_integer(
    vla_dataset_cfg,
    key: str,
    *,
    default: int | None = None,
) -> int:
    """Read one worker knob without silently repairing invalid YAML.

    Older non-H100 profiles may still omit these settings and use the legacy
    default.  Once a value is present, however, zero, negative, boolean, and
    string values are configuration errors rather than values to clamp.
    """

    if key in vla_dataset_cfg and vla_dataset_cfg.get(key) is not None:
        value = vla_dataset_cfg.get(key)
    elif default is not None:
        value = default
    else:
        raise ValueError(f"datasets.vla_data.{key} is required")
    if type(value) is not int or value <= 0:
        raise ValueError(
            f"datasets.vla_data.{key} must be a positive integer; got {value!r}"
        )
    return value


def _loader_prefetch_factor(vla_dataset_cfg, *, is_eval: bool) -> int:
    key = "eval_prefetch_factor" if is_eval else "prefetch_factor"
    if is_eval and (
        key not in vla_dataset_cfg or vla_dataset_cfg.get(key) is None
    ):
        key = "prefetch_factor"
    return _positive_loader_integer(vla_dataset_cfg, key, default=2)


def _maybe_clamp_canonical_workers_for_memory(vla_dataset_cfg, num_workers: int) -> int:
    if type(num_workers) is not int or num_workers < 0:
        raise ValueError(
            "canonical_subset_vla num_workers must be a non-negative integer"
        )
    enforce_budget = vla_dataset_cfg.get(
        "enforce_worker_memory_budget", True
    )
    if type(enforce_budget) is not bool:
        raise ValueError(
            "datasets.vla_data.enforce_worker_memory_budget must be boolean"
        )
    if not enforce_budget or num_workers == 0:
        return num_workers

    raw_worker_budget_gib = vla_dataset_cfg.get(
        "estimated_worker_memory_gb", 5.0
    )
    raw_host_fraction = vla_dataset_cfg.get(
        "worker_memory_budget_fraction", 0.65
    )
    if (
        isinstance(raw_worker_budget_gib, bool)
        or not isinstance(raw_worker_budget_gib, (int, float))
        or not math.isfinite(float(raw_worker_budget_gib))
        or float(raw_worker_budget_gib) <= 0.0
    ):
        raise ValueError(
            "datasets.vla_data.estimated_worker_memory_gb must be a positive "
            f"finite number; got {raw_worker_budget_gib!r}"
        )
    if (
        isinstance(raw_host_fraction, bool)
        or not isinstance(raw_host_fraction, (int, float))
        or not math.isfinite(float(raw_host_fraction))
        or not 0.0 < float(raw_host_fraction) <= 1.0
    ):
        raise ValueError(
            "datasets.vla_data.worker_memory_budget_fraction must be a finite "
            f"number in (0, 1]; got {raw_host_fraction!r}"
        )
    worker_budget_gib = float(raw_worker_budget_gib)
    host_fraction = float(raw_host_fraction)

    total_gib = _host_memory_gib()
    if total_gib is None or total_gib <= 0:
        return num_workers

    world_size = _distributed_world_size()

    max_total_workers = int(
        (total_gib * host_fraction) // worker_budget_gib
    )
    max_workers_per_rank = max_total_workers // world_size
    if num_workers <= max_workers_per_rank:
        return num_workers

    raise ValueError(
        "canonical_subset_vla num_workers exceeds the config-owned host RAM "
        "budget: "
        f"requested={num_workers}, maximum={max_workers_per_rank} "
        f"(MemTotal={total_gib:.1f}GiB, world_size={world_size}, "
        f"estimated_worker_memory_gb={worker_budget_gib:.1f}, "
        f"worker_memory_budget_fraction={host_fraction:.2f}). "
        "Refusing to silently change the reviewed worker topology. Set an "
        "explicitly reviewed lower datasets.vla_data.num_workers value, or set "
        "datasets.vla_data.enforce_worker_memory_budget=false in YAML."
    )


def save_dataset_statistics(dataset_statistics, run_dir):
    """Saves a `dataset_statistics.json` file."""
    out_path = run_dir / "dataset_statistics.json"
    with open(out_path, "w") as f_json:
        for _, stats in dataset_statistics.items():
            for k in stats["action"].keys():
                if isinstance(stats["action"][k], np.ndarray):
                    stats["action"][k] = stats["action"][k].tolist()
            if "proprio" in stats:
                for k in stats["proprio"].keys():
                    if isinstance(stats["proprio"][k], np.ndarray):
                        stats["proprio"][k] = stats["proprio"][k].tolist()
            if "num_trajectories" in stats:
                if isinstance(stats["num_trajectories"], np.ndarray):
                    stats["num_trajectories"] = stats["num_trajectories"].item()
            if "num_transitions" in stats:
                if isinstance(stats["num_transitions"], np.ndarray):
                    stats["num_transitions"] = stats["num_transitions"].item()
        json.dump(dataset_statistics, f_json, indent=2)
    logger.info(f"Saved dataset statistics file at path {out_path}")


def _resolve_output_dir(cfg) -> Path | None:
    if "output_dir" in cfg:
        return Path(cfg.output_dir)
    if "run_root_dir" in cfg and "run_id" in cfg:
        return Path(cfg.run_root_dir) / cfg.run_id
    return None


def _build_eval_lerobot_data_cfg(vla_dataset_cfg):
    """Clone shared config for immutable holdout evaluation.

    The frozen view is a train-row ledger and the LeRobot loader correctly
    rejects it for ``episode_split_role="eval"``.  Evaluation remains bound
    by the copied episode-split manifest and shared normalization artifact.
    """

    eval_cfg = copy.deepcopy(vla_dataset_cfg)
    eval_cfg["lerobot_v3_parquet_cache_size"] = 1
    eval_cfg["frozen_train_view_manifest"] = None
    eval_cfg["frozen_train_view_manifest_sha256"] = None
    eval_cfg["frozen_train_view_require_data_shard_hashes"] = False
    return eval_cfg


def _close_worker_dataset_readers(dataset, visited: set[int] | None = None) -> None:
    """Best-effort cleanup for native video readers held by worker-local dataset copies."""
    if dataset is None:
        return
    if visited is None:
        visited = set()
    dataset_id = id(dataset)
    if dataset_id in visited:
        return
    visited.add(dataset_id)
    wrapped = getattr(dataset, "dataset", None)
    if wrapped is not None and wrapped is not dataset:
        _close_worker_dataset_readers(wrapped, visited)
    children = getattr(dataset, "datasets", None)
    if children is not None:
        for child in children:
            _close_worker_dataset_readers(child, visited)
    close_readers = getattr(dataset, "close_video_readers", None)
    if callable(close_readers):
        close_readers()
    close_parquet_cache = getattr(dataset, "close_parquet_cache", None)
    if callable(close_parquet_cache):
        close_parquet_cache()
    readers = getattr(dataset, "_decord_readers", None)
    if readers is not None:
        try:
            readers.clear()
        except Exception:
            pass


def _configure_lerobot_worker(worker_id: int, *, torch_threads: int, cv2_threads: int) -> None:
    """
    Keep each LeRobot dataloader worker close to single-threaded so we do not
    accidentally multiply 20 workers into hundreds of native helper threads.
    Also ask Linux to terminate the worker if its training-rank parent dies, so
    failed runs do not leave multi-GB orphan workers behind.
    """
    try:
        faulthandler.enable(file=sys.stderr, all_threads=True)
        faulthandler.register(signal.SIGUSR2, file=sys.stderr, all_threads=True, chain=False)
        print(f"DataLoader worker {worker_id} started pid={os.getpid()}", file=sys.stderr, flush=True)
    except Exception:
        pass

    try:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            atexit.register(_close_worker_dataset_readers, worker_info.dataset)
    except Exception:
        pass

    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        PR_SET_PDEATHSIG = 1
        if libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0) != 0:
            logger.warning(
                "Unable to install worker parent-death signal; orphaned workers may survive rank crashes"
            )
    except Exception:
        pass

    os.environ["OMP_NUM_THREADS"] = str(torch_threads)
    os.environ["MKL_NUM_THREADS"] = str(torch_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(torch_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(torch_threads)

    try:
        torch.set_num_threads(torch_threads)
    except Exception:
        pass

    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    try:
        import cv2

        cv2.setNumThreads(cv2_threads)
        if hasattr(cv2, "ocl"):
            cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass



def build_dataloader(cfg, dataset_py="lerobot_datasets", model=None, *, mode: str = "train"):

    mode_aliases = {
        "train": "train",
        "eval": "eval",
        "evaluation": "eval",
        "val": "eval",
        "validation": "eval",
        "holdout": "eval",
    }
    normalized_mode = mode_aliases.get(str(mode).lower())
    if normalized_mode is None:
        raise ValueError(
            f"Unsupported dataloader mode {mode!r}; expected one of {sorted(mode_aliases)}."
        )

    if dataset_py == "lerobot_datasets":
        from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn
        vla_dataset_cfg = cfg.datasets.vla_data
        is_eval = normalized_mode == "eval"
        (
            exhaustive_training,
            loader_shuffle,
            drop_last,
        ) = _resolve_epoch_loader_contract(
            vla_dataset_cfg,
            dataset_py=dataset_py,
            is_eval=is_eval,
        )
        if is_eval and not vla_dataset_cfg.get("episode_split_manifest", None):
            raise ValueError(
                "Heldout evaluation requires datasets.vla_data.episode_split_manifest."
            )
        if is_eval and str(vla_dataset_cfg.get("lerobot_statistics_source", "")) != "split_train":
            raise ValueError(
                "Heldout evaluation must use the manifest-bound TRAIN normalization "
                "statistics; set datasets.vla_data.lerobot_statistics_source=split_train."
            )

        if is_eval:
            num_workers = int(vla_dataset_cfg.get("eval_num_workers", 0))
            pin_memory = bool(
                vla_dataset_cfg.get(
                    "eval_pin_memory",
                    vla_dataset_cfg.get("pin_memory", torch.cuda.is_available()),
                )
            )
            # Never allow a loader/sampler implementation detail to discard a
            # heldout episode.  Exact global/local counts are validated below.
            drop_last = False
        else:
            num_workers = int(vla_dataset_cfg.get("num_workers", 8))
            pin_memory = bool(vla_dataset_cfg.get("pin_memory", torch.cuda.is_available()))

        dataset_build_cfg = vla_dataset_cfg
        if is_eval:
            # Single-dataset config objects are retained by each child.  Give
            # eval an independent copy so its one-shard label scan/cache policy
            # can never mutate training config or worker behavior.
            dataset_build_cfg = _build_eval_lerobot_data_cfg(vla_dataset_cfg)

        vla_dataset = get_vla_dataset(
            data_cfg=dataset_build_cfg,
            mode=normalized_mode,
            action_horizon=cfg.framework.action_model.action_horizon,
            video_horizon=cfg.framework.vj2_model.num_frames,
            video_frame_stride=vla_dataset_cfg.get("video_frame_stride", 1),
        )
        if exhaustive_training:
            _validate_exhaustive_dataset_schedule(vla_dataset)
        loader_generator = None
        focused_eval_dataset = None
        batch_size = int(
            vla_dataset_cfg.get(
                "eval_per_device_batch_size",
                vla_dataset_cfg.per_device_batch_size,
            )
            if is_eval
            else vla_dataset_cfg.per_device_batch_size
        )
        if is_eval:
            from starVLA.dataloader.heldout_eval import (
                build_heldout_eval_dataset,
                validate_global_eval_observation_count,
            )

            focused_eval_enabled = bool(
                cfg.trainer.get("heldout_focused_eval_enabled", False)
            )
            legacy_underfilled_eval = bool(
                cfg.trainer.get("eval_only_legacy_underfilled_holdout", False)
            )
            legacy_excluded_ids = tuple(
                int(value)
                for value in cfg.trainer.get(
                    "eval_only_legacy_excluded_zero_valid_episode_ids", ()
                )
            )
            if legacy_underfilled_eval and not bool(
                cfg.trainer.get("eval_only", False)
            ):
                raise ValueError(
                    "Legacy underfilled holdout mode is restricted to eval-only."
                )
            focused_subtasks = (
                tuple(
                    int(value)
                    for value in cfg.trainer.get(
                        "heldout_focused_eval_required_subtasks",
                        (2, 3, 4, 5, 6, 7),
                    )
                )
                if focused_eval_enabled
                else None
            )
            vla_dataset = build_heldout_eval_dataset(
                vla_dataset,
                manifest_path=vla_dataset_cfg.episode_split_manifest,
                action_dim=int(cfg.framework.action_model.action_dim),
                focused_subtasks=focused_subtasks,
                movement_threshold=float(
                    cfg.trainer.get("heldout_eval_movement_threshold", 0.02)
                ),
                legacy_underfilled_eval=legacy_underfilled_eval,
                legacy_excluded_zero_valid_episode_ids=legacy_excluded_ids,
            )
            world_size = _distributed_world_size()
            expected_global_observations = validate_global_eval_observation_count(
                holdout_episode_count=vla_dataset.manifest_holdout_episode_count,
                observation_count=(
                    vla_dataset.evaluation_sampling_contract.observation_count
                ),
                frames_per_episode=(
                    vla_dataset.evaluation_sampling_contract.frames_per_episode
                ),
                per_device_batch_size=batch_size,
                world_size=world_size,
                gradient_accumulation_steps=1,
            )
            if legacy_underfilled_eval and len(vla_dataset) != (
                expected_global_observations - len(legacy_excluded_ids)
            ):
                raise ValueError(
                    "Legacy heldout audit cardinality must equal the original "
                    "effective global batch minus explicit zero-valid exclusions."
                )
            loader_generator = vla_dataset.make_torch_generator()
            if focused_eval_enabled:
                focused_eval_dataset = vla_dataset.make_focused_view()
                if legacy_underfilled_eval:
                    if len(focused_eval_dataset) != len(vla_dataset):
                        raise ValueError(
                            "Legacy focused/unbiased eval views must have equal "
                            "underfilled cardinality."
                        )
                else:
                    validate_global_eval_observation_count(
                        holdout_episode_count=(
                            focused_eval_dataset.manifest_holdout_episode_count
                        ),
                        observation_count=(
                            focused_eval_dataset.evaluation_sampling_contract
                            .observation_count
                        ),
                        frames_per_episode=(
                            focused_eval_dataset.evaluation_sampling_contract
                            .frames_per_episode
                        ),
                        per_device_batch_size=batch_size,
                        world_size=world_size,
                        gradient_accumulation_steps=1,
                    )
            logger.info(
                "Heldout checkpoint eval: "
                f"{len(vla_dataset)} windows from "
                f"{vla_dataset.manifest_holdout_episode_count} episodes "
                f"(base="
                f"{vla_dataset.evaluation_sampling_contract.base_frames_per_episode}, "
                f"extra_episodes="
                f"{vla_dataset.evaluation_sampling_contract.extra_window_episode_count}), "
                f"world_size={world_size}, local_microbatch={batch_size}, "
                f"distributed_microbatches="
                f"{expected_global_observations // (batch_size * world_size)}, "
                f"window_selection_sha256={vla_dataset.heldout_window_digest}"
            )
            if focused_eval_dataset is not None:
                focused_report = focused_eval_dataset.sampling_report()
                logger.info(
                    "Focused heldout checkpoint eval: "
                    f"{len(focused_eval_dataset)} deterministic H10 "
                    "transition/stage-focused windows from "
                    f"{focused_eval_dataset.manifest_holdout_episode_count} "
                    "episodes, "
                    f"window_selection_sha256="
                    f"{focused_eval_dataset.heldout_window_digest}, "
                    f"open_to_close_h10="
                    f"{focused_report['open_to_close_transition_count_h10']}, "
                    f"close_to_open_h10="
                    f"{focused_report['close_to_open_transition_count_h10']}"
                )
            sampling_report = vla_dataset.sampling_report()
            zero_valid_episodes = sampling_report["zero_valid_action_episodes"]
            if zero_valid_episodes:
                raise ValueError(
                    "Every manifest-heldout episode must contribute supervised "
                    "action elements so each eval view remains one full effective "
                    "global batch. Episodes with no evaluable structural window: "
                    f"{zero_valid_episodes}"
                )
        if bool(vla_dataset_cfg.get("gpu_video_decode_on_rank", False)):
            logger.info(
                "LeRobot dataloader will hand off video decode specs to each training rank; "
                "video frames are decoded on the rank device instead of inside dataloader workers"
            )
        elif bool(vla_dataset_cfg.get("cpu_video_decode_drop_worker_images", False)):
            logger.info(
                "LeRobot dataloader will keep CPU video decode in workers but skip worker-built image payloads; "
                "the training rank will derive Qwen inputs from the returned video tensors"
            )

        loader_kwargs = dict(
            dataset=vla_dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=loader_shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            generator=loader_generator,
        )
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = _loader_prefetch_factor(
                vla_dataset_cfg,
                is_eval=is_eval,
            )
            # Eval workers are transient by default: holding a second complete
            # set of dataset/video-reader processes for thousands of train steps
            # would defeat the low-overhead one-batch design.
            if is_eval and bool(vla_dataset_cfg.get("eval_persistent_workers", False)):
                raise ValueError(
                    "eval_persistent_workers=true is incompatible with deterministic, "
                    "low-footprint checkpoint evaluation."
                )
            loader_kwargs["persistent_workers"] = (
                False
                if is_eval
                else bool(vla_dataset_cfg.get("persistent_workers", True))
            )
            loader_kwargs["multiprocessing_context"] = (
                _resolve_worker_multiprocessing_context(
                    vla_dataset_cfg,
                    is_eval=is_eval,
                    num_workers=num_workers,
                )
            )
            timeout_key = "eval_dataloader_timeout_seconds" if is_eval else "dataloader_timeout_seconds"
            dataloader_timeout_seconds = int(
                vla_dataset_cfg.get(
                    timeout_key,
                    vla_dataset_cfg.get("dataloader_timeout_seconds", 0),
                )
            )
            if dataloader_timeout_seconds > 0:
                loader_kwargs["timeout"] = dataloader_timeout_seconds
            loader_kwargs["worker_init_fn"] = partial(
                _configure_lerobot_worker,
                torch_threads=_positive_loader_integer(
                    vla_dataset_cfg,
                    "worker_torch_threads",
                    default=1,
                ),
                cv2_threads=_positive_loader_integer(
                    vla_dataset_cfg,
                    "worker_cv2_threads",
                    default=1,
                ),
            )
            logger.info(
                "LeRobot DataLoader worker contract: "
                f"mode={normalized_mode}, workers={num_workers}, "
                f"persistent={loader_kwargs['persistent_workers']}, "
                f"context={loader_kwargs['multiprocessing_context']}, "
                f"prefetch_factor={loader_kwargs['prefetch_factor']}"
            )

        vla_train_dataloader = DataLoader(**loader_kwargs)
        if exhaustive_training:
            _validate_exhaustive_dataloader(
                vla_dataset, vla_train_dataloader
            )
        focused_eval_dataloader = None
        if focused_eval_dataset is not None:
            focused_loader_kwargs = dict(loader_kwargs)
            focused_loader_kwargs["dataset"] = focused_eval_dataset
            focused_loader_kwargs["generator"] = (
                focused_eval_dataset.make_torch_generator()
            )
            focused_eval_dataloader = DataLoader(**focused_loader_kwargs)
        if not dist.is_initialized() or dist.get_rank() == 0:
            output_dir = _resolve_output_dir(cfg)
            if output_dir is not None:
                output_dir.mkdir(parents=True, exist_ok=True)
                if is_eval:
                    vla_dataset.save_sampling_report(
                        output_dir / "heldout_eval_windows.json"
                    )
                    if focused_eval_dataset is not None:
                        focused_eval_dataset.save_sampling_report(
                            output_dir / "heldout_focused_eval_windows.json"
                        )
                else:
                    vla_dataset.save_dataset_provenance(output_dir / "dataset_provenance.json")
                    vla_dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")
        if focused_eval_dataloader is not None:
            return vla_train_dataloader, focused_eval_dataloader
        return vla_train_dataloader
    elif dataset_py == "canonical_subset_vla":
        from starVLA.dataloader.canonical_subset_dataset import (
            DeterministicCanonicalEvalDataset,
            collate_fn,
            get_vla_dataset,
        )

        vla_dataset_cfg = cfg.datasets.vla_data
        is_eval = normalized_mode == "eval"
        (
            exhaustive_training,
            loader_shuffle,
            drop_last,
        ) = _resolve_epoch_loader_contract(
            vla_dataset_cfg,
            dataset_py=dataset_py,
            is_eval=is_eval,
        )
        if is_eval and not vla_dataset_cfg.get(
            "canonical_eval_manifest", None
        ):
            raise ValueError(
                "Canonical checkpoint evaluation requires "
                "datasets.vla_data.canonical_eval_manifest."
            )
        num_workers = int(
            vla_dataset_cfg.get(
                "eval_num_workers" if is_eval else "num_workers",
                0,
            )
        )
        num_workers = _maybe_clamp_canonical_workers_for_memory(vla_dataset_cfg, num_workers)
        pin_memory = bool(
            vla_dataset_cfg.get(
                "eval_pin_memory" if is_eval else "pin_memory",
                vla_dataset_cfg.get("pin_memory", torch.cuda.is_available()),
            )
        )

        vla_dataset = get_vla_dataset(
            data_cfg=vla_dataset_cfg,
            mode=normalized_mode,
            action_horizon=cfg.framework.action_model.action_horizon,
            video_horizon=cfg.framework.vj2_model.num_frames,
            video_frame_stride=vla_dataset_cfg.get("video_frame_stride", 1),
        )
        if exhaustive_training:
            _validate_exhaustive_dataset_schedule(vla_dataset)
        canonical_sampler = _resolve_canonical_exhaustive_sampler(
            vla_dataset_cfg,
            vla_dataset,
            exhaustive_training=exhaustive_training,
            is_eval=is_eval,
        )
        loader_generator = None
        batch_size = int(
            vla_dataset_cfg.get(
                "eval_per_device_batch_size",
                vla_dataset_cfg.per_device_batch_size,
            )
            if is_eval
            else vla_dataset_cfg.per_device_batch_size
        )
        if is_eval:
            vla_dataset = DeterministicCanonicalEvalDataset(vla_dataset)
            _normalize_canonical_eval_metric_groups(vla_dataset)
            from starVLA.dataloader.heldout_eval import (
                validate_global_eval_observation_count,
            )

            sampling_report = vla_dataset.sampling_report()
            validate_global_eval_observation_count(
                holdout_episode_count=int(
                    sampling_report["holdout_episode_count"]
                ),
                observation_count=int(
                    sampling_report["observation_count"]
                ),
                frames_per_episode=(
                    int(sampling_report["frames_per_episode"])
                    if (
                        "frames_per_episode" in sampling_report
                        and int(
                            sampling_report.get(
                                "extra_window_episode_count",
                                0,
                            )
                        )
                        == 0
                    )
                    else None
                ),
                per_device_batch_size=batch_size,
                world_size=_distributed_world_size(),
                gradient_accumulation_steps=1,
            )
            loader_generator = vla_dataset.make_torch_generator()
            logger.info(
                "Canonical heldout checkpoint eval: "
                f"{len(vla_dataset)} exact manifest windows, "
                f"window_selection_sha256={vla_dataset.heldout_window_digest}"
            )
        try:
            logger.info(
                "Canonical subset dataloader will use the existing CPU-worker video_compact path; "
                "gpu_video_decode_on_rank is intentionally not required"
            )
        except RuntimeError:
            print(
                "Canonical subset dataloader will use the existing CPU-worker video_compact path; "
                "gpu_video_decode_on_rank is intentionally not required"
            )

        loader_kwargs = dict(
            dataset=vla_dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=loader_shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            generator=loader_generator,
        )
        if canonical_sampler is not None:
            loader_kwargs["sampler"] = canonical_sampler
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = _loader_prefetch_factor(
                vla_dataset_cfg,
                is_eval=is_eval,
            )
            loader_kwargs["persistent_workers"] = (
                False
                if is_eval
                else bool(vla_dataset_cfg.get("persistent_workers", True))
            )
            loader_kwargs["multiprocessing_context"] = (
                _resolve_worker_multiprocessing_context(
                    vla_dataset_cfg,
                    is_eval=is_eval,
                    num_workers=num_workers,
                )
            )
            dataloader_timeout_seconds = int(
                vla_dataset_cfg.get(
                    (
                        "eval_dataloader_timeout_seconds"
                        if is_eval
                        else "dataloader_timeout_seconds"
                    ),
                    vla_dataset_cfg.get("dataloader_timeout_seconds", 0),
                )
            )
            if dataloader_timeout_seconds > 0:
                loader_kwargs["timeout"] = dataloader_timeout_seconds
            loader_kwargs["worker_init_fn"] = partial(
                _configure_lerobot_worker,
                torch_threads=_positive_loader_integer(
                    vla_dataset_cfg,
                    "worker_torch_threads",
                    default=1,
                ),
                cv2_threads=_positive_loader_integer(
                    vla_dataset_cfg,
                    "worker_cv2_threads",
                    default=1,
                ),
            )

        vla_train_dataloader = DataLoader(**loader_kwargs)
        if exhaustive_training:
            _validate_exhaustive_dataloader(
                vla_dataset, vla_train_dataloader
            )
        if not dist.is_initialized() or dist.get_rank() == 0:
            output_dir = _resolve_output_dir(cfg)
            if output_dir is not None:
                output_dir.mkdir(parents=True, exist_ok=True)
                if is_eval:
                    vla_dataset.save_sampling_report(
                        output_dir / "heldout_eval_windows.json"
                    )
                else:
                    vla_dataset.save_dataset_provenance(output_dir / "dataset_provenance.json")
                    vla_dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")
        return vla_train_dataloader
    elif normalized_mode != "train":
        raise ValueError(
            "In-training heldout evaluation is not implemented for "
            f"dataset_py={dataset_py!r}. Use lerobot_datasets or "
            "canonical_subset_vla for immutable checkpoint evaluation."
        )
    elif dataset_py == "preprocessed_subtask_dataset":
        from starVLA.dataloader.preprocessed_subtask_dataset import (
            PreprocessedSubtaskCollator,
            PreprocessedSubtaskVLADataset,
        )

        vla_dataset_cfg = cfg.datasets.vla_data
        num_workers = int(vla_dataset_cfg.get("num_workers", 8))
        pin_memory = bool(vla_dataset_cfg.get("pin_memory", torch.cuda.is_available()))
        drop_last = bool(vla_dataset_cfg.get("drop_last", True))

        vla_dataset = PreprocessedSubtaskVLADataset(
            data_root_dir=vla_dataset_cfg.data_root_dir,
            action_horizon=cfg.framework.action_model.action_horizon,
            video_horizon=cfg.framework.vj2_model.num_frames,
            video_frame_stride=vla_dataset_cfg.get("video_frame_stride", 1),
            video_target_shift_steps=(
                int(getattr(model.vj_encoder, "tubelet_size", 0))
                if model is not None and hasattr(model, "vj_encoder")
                else int(vla_dataset_cfg.get("video_target_shift_steps", cfg.framework.vj2_model.get("tubelet_size", 2)))
            ),
            resolution_size=vla_dataset_cfg.get("resolution_size", 224),
            video_resolution_size=vla_dataset_cfg.get("video_resolution_size", 384),
            instruction_text=vla_dataset_cfg.get("instruction_text", "Complete the task successfully."),
            current_cameras=vla_dataset_cfg.get("current_cameras", None),
            frame_cache_size=vla_dataset_cfg.get("frame_cache_size", 256),
            data_cfg=vla_dataset_cfg,
        )

        collate_fn = _identity_collate
        if model is not None:
            collate_fn = PreprocessedSubtaskCollator(
                model_id=cfg.framework.qwenvl.base_vlm,
                prompt_template=vla_dataset_cfg.get("CoT_prompt", ""),
                replace_prompt=model.replace_prompt,
                embodied_replace_prompt=model.embodied_replace_prompt,
                state_replace_prompt=getattr(model, "qwen_state_replace_prompt", ""),
                geometry_replace_prompt=getattr(model, "geometry_replace_prompt", ""),
                special_action_token=cfg.framework.vj2_model.special_action_token,
                max_action_tokens=cfg.framework.action_model.action_horizon * 4,
                embodied_action_token=cfg.framework.vj2_model.get(
                    "embodied_action_token", "<|embodied_action|>"
                ),
                extra_special_tokens=[
                    *getattr(model, "geometry_tokens", []),
                    *getattr(model, "qwen_state_tokens", []),
                ],
            )
            safe_worker_cap = int(vla_dataset_cfg.get("safe_num_workers_cap", 2))
            if num_workers > safe_worker_cap:
                logger.warning(
                    "Clamping preprocessed_subtask_dataset num_workers from "
                    f"{num_workers} to {safe_worker_cap} to avoid worker RAM blowups"
                )
                num_workers = safe_worker_cap

        loader_kwargs = dict(
            dataset=vla_dataset,
            batch_size=cfg.datasets.vla_data.per_device_batch_size,
            collate_fn=collate_fn,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = _loader_prefetch_factor(
                vla_dataset_cfg,
                is_eval=False,
            )
            loader_kwargs["persistent_workers"] = False
            loader_kwargs["multiprocessing_context"] = (
                _resolve_worker_multiprocessing_context(
                    vla_dataset_cfg,
                    is_eval=False,
                    num_workers=num_workers,
                )
            )

        vla_train_dataloader = DataLoader(**loader_kwargs)
        if not dist.is_initialized() or dist.get_rank() == 0:
            output_dir = _resolve_output_dir(cfg)
            if output_dir is not None:
                output_dir.mkdir(parents=True, exist_ok=True)
                vla_dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")
        return vla_train_dataloader
