#!/usr/bin/env python3
"""Exercise the production DataLoader under an Accelerate multi-rank launch.

This intentionally builds no model.  It reproduces the part of production
startup that creates the manifest-bound dataset, lets Accelerate shard the
loader, starts every persistent worker, and transfers real batches through the
collate function.  Run it in the production image with ``torchrun`` before
changing a worker start method or count.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from accelerate import Accelerator
from omegaconf import OmegaConf

from starVLA.dataloader import build_dataloader


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-yaml", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument(
        "--multiprocessing-context",
        choices=("spawn", "forkserver"),
        required=True,
    )
    parser.add_argument("--data-root-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/starvla-loader-probe"))
    return parser.parse_args()


def _worker_pids(loader) -> list[int]:
    base_loader = getattr(loader, "base_dataloader", loader)
    iterator = getattr(base_loader, "_iterator", None)
    workers = getattr(iterator, "_workers", ()) if iterator is not None else ()
    return sorted(
        int(worker.pid)
        for worker in workers
        if getattr(worker, "pid", None) is not None
    )


def main() -> None:
    args = _parse_args()
    if args.workers <= 0 or args.batches <= 0:
        raise ValueError("workers and batches must be positive")

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    accelerator = Accelerator(cpu=True)
    cfg = OmegaConf.load(args.config_yaml)
    cfg.datasets.vla_data.num_workers = int(args.workers)
    cfg.datasets.vla_data.persistent_workers = True
    cfg.datasets.vla_data.multiprocessing_context = args.multiprocessing_context
    if args.data_root_dir is not None:
        cfg.datasets.vla_data.data_root_dir = str(args.data_root_dir.resolve())
    cfg.output_dir = str(args.output_dir.resolve())
    cfg.run_root_dir = str(args.output_dir.resolve().parent)
    cfg.run_id = args.output_dir.name

    loader = build_dataloader(
        cfg,
        dataset_py=str(cfg.datasets.vla_data.dataset_py),
    )
    loader = accelerator.prepare(loader)
    iterator = iter(loader)
    observed_batches = 0
    observed_samples = 0
    for _ in range(args.batches):
        batch = next(iterator)
        observed_batches += 1
        observed_samples += len(batch)

    pids = _worker_pids(loader)
    if len(pids) != args.workers:
        raise RuntimeError(
            f"rank {accelerator.process_index}: observed workers {pids}, "
            f"expected {args.workers}"
        )
    print(
        f"rank={accelerator.process_index} context={args.multiprocessing_context} "
        f"workers={args.workers} batches={observed_batches} "
        f"samples={observed_samples} worker_pids={pids}",
        flush=True,
    )
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print("MULTIRANK_TRAINING_DATALOADER_PROBE_PASS", flush=True)


if __name__ == "__main__":
    main()
