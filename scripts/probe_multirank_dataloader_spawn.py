#!/usr/bin/env python3
"""Exercise multi-rank DataLoader worker startup without training a model."""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset


class ProbeDataset(Dataset):
    def __init__(self, length: int) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.tensor((index, os.getpid()), dtype=torch.int64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--multiprocessing-context",
        choices=("spawn", "forkserver"),
        default="spawn",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers <= 0 or args.batches <= 0 or args.batch_size <= 0:
        raise ValueError("workers, batches, and batch-size must be positive")

    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    loader = DataLoader(
        ProbeDataset(args.batches * args.batch_size * 4),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        prefetch_factor=2,
        persistent_workers=True,
        multiprocessing_context=args.multiprocessing_context,
        generator=torch.Generator().manual_seed(42 + rank),
    )
    iterator = iter(loader)
    observed = 0
    worker_pids: set[int] = set()
    for _ in range(args.batches):
        batch = next(iterator)
        observed += int(batch.shape[0])
        worker_pids.update(int(pid) for pid in batch[:, 1].tolist())

    expected = args.batches * args.batch_size
    if observed != expected:
        raise RuntimeError(f"rank {rank}: observed {observed}, expected {expected}")
    if len(worker_pids) != args.workers:
        raise RuntimeError(
            f"rank {rank}: observed worker PIDs {sorted(worker_pids)}, "
            f"expected {args.workers} workers"
        )
    print(
        f"rank={rank} context={args.multiprocessing_context} "
        f"workers={args.workers} worker_pids={sorted(worker_pids)}",
        flush=True,
    )
    dist.barrier()
    if rank == 0:
        print("MULTIRANK_DATALOADER_PROBE_PASS", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
