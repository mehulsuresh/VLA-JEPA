import json
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

from starVLA.dataloader.vlm_datasets import make_vlm_dataloader

logger = get_logger(__name__)


def _identity_collate(batch):
    return batch

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


def _close_worker_dataset_readers(dataset) -> None:
    """Best-effort cleanup for native video readers held by worker-local dataset copies."""
    if dataset is None:
        return
    wrapped = getattr(dataset, "dataset", None)
    if wrapped is not None and wrapped is not dataset:
        _close_worker_dataset_readers(wrapped)
    close_readers = getattr(dataset, "close_video_readers", None)
    if callable(close_readers):
        close_readers()
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



def build_dataloader(cfg, dataset_py="lerobot_datasets_oxe", model=None): # TODO now here only is get dataset, we need mv dataloader to here

    if dataset_py == "lerobot_datasets":
        from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn
        vla_dataset_cfg = cfg.datasets.vla_data
        num_workers = int(vla_dataset_cfg.get("num_workers", 8))
        pin_memory = bool(vla_dataset_cfg.get("pin_memory", torch.cuda.is_available()))
        drop_last = bool(vla_dataset_cfg.get("drop_last", True))

        vla_dataset = get_vla_dataset(
            data_cfg=vla_dataset_cfg,
            action_horizon=cfg.framework.action_model.action_horizon,
            video_horizon=cfg.framework.vj2_model.num_frames,
            video_frame_stride=vla_dataset_cfg.get("video_frame_stride", 1),
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
            batch_size=cfg.datasets.vla_data.per_device_batch_size,
            collate_fn=collate_fn,
            shuffle=bool(vla_dataset_cfg.get("shuffle", True)),
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = max(2, int(vla_dataset_cfg.get("prefetch_factor", 2)))
            loader_kwargs["persistent_workers"] = bool(vla_dataset_cfg.get("persistent_workers", True))
            loader_kwargs["multiprocessing_context"] = vla_dataset_cfg.get("multiprocessing_context", "spawn")
            dataloader_timeout_seconds = int(vla_dataset_cfg.get("dataloader_timeout_seconds", 0))
            if dataloader_timeout_seconds > 0:
                loader_kwargs["timeout"] = dataloader_timeout_seconds
            loader_kwargs["worker_init_fn"] = partial(
                _configure_lerobot_worker,
                torch_threads=max(1, int(vla_dataset_cfg.get("worker_torch_threads", 1))),
                cv2_threads=max(1, int(vla_dataset_cfg.get("worker_cv2_threads", 1))),
            )

        vla_train_dataloader = DataLoader(**loader_kwargs)
        if not dist.is_initialized() or dist.get_rank() == 0:
            output_dir = _resolve_output_dir(cfg)
            if output_dir is not None:
                output_dir.mkdir(parents=True, exist_ok=True)
                vla_dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")
        return vla_train_dataloader
    elif dataset_py == "canonical_subset_vla":
        from starVLA.dataloader.canonical_subset_dataset import get_vla_dataset, collate_fn

        vla_dataset_cfg = cfg.datasets.vla_data
        num_workers = int(vla_dataset_cfg.get("num_workers", 0))
        pin_memory = bool(vla_dataset_cfg.get("pin_memory", torch.cuda.is_available()))
        drop_last = bool(vla_dataset_cfg.get("drop_last", True))

        vla_dataset = get_vla_dataset(
            data_cfg=vla_dataset_cfg,
            action_horizon=cfg.framework.action_model.action_horizon,
            video_horizon=cfg.framework.vj2_model.num_frames,
            video_frame_stride=vla_dataset_cfg.get("video_frame_stride", 1),
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
            batch_size=cfg.datasets.vla_data.per_device_batch_size,
            collate_fn=collate_fn,
            shuffle=bool(vla_dataset_cfg.get("shuffle", True)),
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = max(2, int(vla_dataset_cfg.get("prefetch_factor", 2)))
            loader_kwargs["persistent_workers"] = bool(vla_dataset_cfg.get("persistent_workers", True))
            loader_kwargs["multiprocessing_context"] = vla_dataset_cfg.get("multiprocessing_context", "spawn")
            dataloader_timeout_seconds = int(vla_dataset_cfg.get("dataloader_timeout_seconds", 0))
            if dataloader_timeout_seconds > 0:
                loader_kwargs["timeout"] = dataloader_timeout_seconds
            loader_kwargs["worker_init_fn"] = partial(
                _configure_lerobot_worker,
                torch_threads=max(1, int(vla_dataset_cfg.get("worker_torch_threads", 1))),
                cv2_threads=max(1, int(vla_dataset_cfg.get("worker_cv2_threads", 1))),
            )

        vla_train_dataloader = DataLoader(**loader_kwargs)
        if not dist.is_initialized() or dist.get_rank() == 0:
            output_dir = _resolve_output_dir(cfg)
            if output_dir is not None:
                output_dir.mkdir(parents=True, exist_ok=True)
                vla_dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")
        return vla_train_dataloader
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
        )

        collate_fn = _identity_collate
        if model is not None:
            collate_fn = PreprocessedSubtaskCollator(
                model_id=cfg.framework.qwenvl.base_vlm,
                prompt_template=vla_dataset_cfg.get("CoT_prompt", ""),
                replace_prompt=model.replace_prompt,
                embodied_replace_prompt=model.embodied_replace_prompt,
                special_action_token=cfg.framework.vj2_model.special_action_token,
                max_action_tokens=cfg.framework.action_model.action_horizon * 4,
                embodied_action_token=cfg.framework.vj2_model.get(
                    "embodied_action_token", "<|embodied_action|>"
                ),
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
            loader_kwargs["prefetch_factor"] = max(2, int(vla_dataset_cfg.get("prefetch_factor", 2)))
            loader_kwargs["persistent_workers"] = False
            loader_kwargs["multiprocessing_context"] = vla_dataset_cfg.get("multiprocessing_context", "spawn")

        vla_train_dataloader = DataLoader(**loader_kwargs)
        if not dist.is_initialized() or dist.get_rank() == 0:
            output_dir = _resolve_output_dir(cfg)
            if output_dir is not None:
                output_dir.mkdir(parents=True, exist_ok=True)
                vla_dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")
        return vla_train_dataloader
    elif dataset_py == "vlm_datasets":
        vlm_data_module = make_vlm_dataloader(cfg)
        vlm_train_dataloader = vlm_data_module["train_dataloader"]
        
        return vlm_train_dataloader
    elif dataset_py == "lerobot_v3_datasets":
        from starVLA.dataloader.lerobot_v3_datasets import get_lerobot_v3_datasets, collate_fn
        vla_dataset_cfg = cfg.datasets.vla_data

        vla_dataset = get_lerobot_v3_datasets(data_cfg=vla_dataset_cfg)

        custom_collate_fn = partial(collate_fn, 
            img_keys=cfg.datasets.vla_data.img_keys,
            state_key=cfg.datasets.vla_data.state_key if "state_key" in cfg.datasets.vla_data else None,
            action_key=cfg.datasets.vla_data.action_key if cfg.datasets.vla_data.action_key else None,
            task_key=cfg.datasets.vla_data.task_key if cfg.datasets.vla_data.task_key else None,
            resize_size=cfg.datasets.vla_data.resize_size)

        vla_train_dataloader = DataLoader(
            vla_dataset,
            batch_size=cfg.datasets.vla_data.per_device_batch_size,
            collate_fn=custom_collate_fn,
            num_workers=16,
            shuffle=True,
        )      
        #if dist.get_rank() == 0: 
        #    for batch in vla_train_dataloader:
        #        print(batch)
        #        for k, v in batch.items():
        #            print(f"{k}: {v.shape if isinstance(v, torch.Tensor) else v}")
        #        break
        return vla_train_dataloader
    elif dataset_py == "video_datasets":
        from starVLA.dataloader.video_datasets import VideoFolderDataset, collate_fn

        video_dataset_cfg = cfg.datasets.video_data

        video_dataset = VideoFolderDataset(
            video_dir=video_dataset_cfg.video_dir,
            text_file=video_dataset_cfg.text_file,
            n_frames=cfg.framework.vj2_model.num_frames,
            extensions=tuple(video_dataset_cfg.extensions),
            crop_h_size=video_dataset_cfg.video_resolution_size,
            crop_w_size=video_dataset_cfg.video_resolution_size,
            max_retry=10,
        )

        video_collate_fn = partial(collate_fn, 
            n_views=2,
            resolution_size=video_dataset_cfg.resolution_size)
        video_train_dataloader = DataLoader(
            video_dataset,
            batch_size=video_dataset_cfg.per_device_batch_size,
            collate_fn=video_collate_fn,
            num_workers=16,
            shuffle=True,
        )        
        return video_train_dataloader
