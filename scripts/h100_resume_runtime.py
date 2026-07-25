#!/usr/bin/env python3
"""Resume one frozen H100 run with a reviewed DataLoader worker correction.

Fresh launches and exact resumes remain owned by ``h100_training.py``.  This
separate helper exists so a run whose immutable holdout provenance binds the
full SHA-256 of that launcher can make one narrowly scoped runtime correction
without changing the bound launcher file.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
from typing import Any, Mapping

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import h100_training  # noqa: E402


RUNTIME_SCHEMA_VERSION = 1
METADATA_SCHEMA = "starvla-resume-runtime-override-v1"
IMAGE_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")


def _validate_resume_runtime_config(
    config_path: Path,
) -> tuple[Path, int, str, str]:
    """Validate one narrowly scoped, repository-owned runtime YAML."""

    candidate = config_path.expanduser()
    if candidate.is_symlink():
        raise h100_training.PlanError(
            "resume runtime config must be a regular non-symlink file: "
            f"{candidate}"
        )
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise h100_training.PlanError(
            f"resume runtime config does not exist: {candidate}"
        ) from exc
    if not resolved.is_relative_to(REPO_ROOT):
        raise h100_training.PlanError(
            "resume runtime config must resolve inside the repository: "
            f"{resolved}"
        )
    if not resolved.is_file() or resolved.is_symlink():
        raise h100_training.PlanError(
            "resume runtime config must be a regular non-symlink file: "
            f"{resolved}"
        )
    try:
        raw = OmegaConf.load(resolved)
        payload = OmegaConf.to_container(
            raw,
            resolve=False,
            throw_on_missing=True,
        )
    except Exception as exc:
        raise h100_training.PlanError(
            f"resume runtime config is invalid YAML: {resolved}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise h100_training.PlanError(
            "resume runtime config root must be a mapping"
        )
    if set(payload) != {"schema_version", "datasets"}:
        raise h100_training.PlanError(
            "resume runtime config may contain only schema_version and datasets"
        )
    schema_version = payload["schema_version"]
    if type(schema_version) is not int or schema_version != RUNTIME_SCHEMA_VERSION:
        raise h100_training.PlanError(
            "resume runtime config schema_version must be exactly "
            f"{RUNTIME_SCHEMA_VERSION}"
        )
    datasets = payload["datasets"]
    if not isinstance(datasets, Mapping) or set(datasets) != {"vla_data"}:
        raise h100_training.PlanError(
            "resume runtime config datasets may contain only vla_data"
        )
    vla_data = datasets["vla_data"]
    if not isinstance(vla_data, Mapping) or set(vla_data) != {
        "num_workers",
        "multiprocessing_context",
    }:
        raise h100_training.PlanError(
            "resume runtime config datasets.vla_data must contain exactly "
            "num_workers and multiprocessing_context"
        )
    num_workers = vla_data["num_workers"]
    if type(num_workers) is not int or num_workers != 4:
        raise h100_training.PlanError(
            "resume runtime config datasets.vla_data.num_workers must be "
            "exactly 4"
        )
    multiprocessing_context = vla_data["multiprocessing_context"]
    if (
        not isinstance(multiprocessing_context, str)
        or multiprocessing_context != "forkserver"
    ):
        raise h100_training.PlanError(
            "resume runtime config datasets.vla_data."
            "multiprocessing_context must be exactly forkserver"
        )
    return (
        resolved,
        num_workers,
        multiprocessing_context,
        h100_training._sha256(resolved),
    )


def _git_commit() -> str:
    status = subprocess.run(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "status",
            "--porcelain",
            "--untracked-files=all",
        ],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    if status.strip():
        raise h100_training.PlanError(
            "resume runtime launch requires a clean repository so source_commit "
            "identifies the exact running code"
        )
    commit = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if COMMIT_RE.fullmatch(commit) is None:
        raise h100_training.PlanError(
            f"could not resolve a full repository commit SHA: {commit!r}"
        )
    return commit


def _container_identity(plan: Mapping[str, Any]) -> tuple[str, str | None]:
    configured_image = str(plan["runtime"]["container_image"])
    current_image = os.environ.get(
        "STARVLA_CONTAINER_IMAGE",
        configured_image,
    ).strip()
    if not current_image:
        raise h100_training.PlanError(
            "current container image identity is empty"
        )
    if current_image != configured_image:
        raise h100_training.PlanError(
            "current container image does not match runtime.container_image: "
            f"{current_image!r} != {configured_image!r}"
        )
    digest = os.environ.get("STARVLA_CONTAINER_IMAGE_DIGEST", "").strip()
    if not digest:
        return current_image, None
    if IMAGE_DIGEST_RE.fullmatch(digest) is None:
        raise h100_training.PlanError(
            "STARVLA_CONTAINER_IMAGE_DIGEST must be sha256:<64 lowercase hex>"
        )
    return current_image, digest


def _resolved_resume_config(
    source_config: Path,
    plan: Mapping[str, Any],
    *,
    checkpoint: Path,
    resume_runtime_config: Path,
) -> tuple[Path, str]:
    """Build an auditable resume invocation without mutating frozen artifacts."""

    checkpoint, immutable_config = h100_training._validate_checkpoint(
        checkpoint,
        int(plan["runtime"]["num_processes"]),
    )
    cfg = OmegaConf.load(immutable_config)
    run_id = str(cfg.get("run_id", ""))
    if h100_training.RUN_ID_RE.fullmatch(run_id) is None:
        raise h100_training.PlanError(
            f"immutable run config has an invalid run ID: {run_id!r}"
        )

    human_launch = cfg.get("human_launch", {})
    recorded_source_sha = human_launch.get("source_config_sha256")
    if recorded_source_sha != plan["config_sha256"]:
        raise h100_training.PlanError(
            "resume source profile SHA does not match the run's recorded profile"
        )
    launcher_path = Path(h100_training.__file__).resolve()
    launcher_sha256 = h100_training._sha256(launcher_path)
    if human_launch.get("launcher_sha256") != launcher_sha256:
        raise h100_training.PlanError(
            "current frozen H100 launcher SHA does not match the run's "
            "recorded launcher SHA"
        )

    (
        runtime_config_path,
        resumed_num_workers,
        resumed_multiprocessing_context,
        runtime_config_sha256,
    ) = _validate_resume_runtime_config(resume_runtime_config)
    previous_num_workers = cfg.get("datasets", {}).get("vla_data", {}).get(
        "num_workers"
    )
    if type(previous_num_workers) is not int or previous_num_workers < 0:
        raise h100_training.PlanError(
            "immutable run config has an invalid "
            "datasets.vla_data.num_workers value"
        )
    if previous_num_workers != 1:
        raise h100_training.PlanError(
            "this resume runtime correction is source-bound to "
            "datasets.vla_data.num_workers=1"
        )
    if resumed_num_workers == previous_num_workers:
        raise h100_training.PlanError(
            "resume runtime config must change datasets.vla_data.num_workers"
        )
    previous_multiprocessing_context = (
        cfg.get("datasets", {})
        .get("vla_data", {})
        .get("multiprocessing_context")
    )
    if (
        not isinstance(previous_multiprocessing_context, str)
        or not previous_multiprocessing_context
    ):
        raise h100_training.PlanError(
            "immutable run config has an invalid "
            "datasets.vla_data.multiprocessing_context value"
        )
    if previous_multiprocessing_context != "spawn":
        raise h100_training.PlanError(
            "this resume runtime correction is source-bound to "
            "datasets.vla_data.multiprocessing_context=spawn"
        )
    if resumed_multiprocessing_context == previous_multiprocessing_context:
        raise h100_training.PlanError(
            "resume runtime config must change datasets.vla_data."
            "multiprocessing_context"
        )

    current_image, image_digest = _container_identity(plan)
    recorded_image = human_launch.get("container_image")
    if recorded_image is not None and str(recorded_image) != current_image:
        raise h100_training.PlanError(
            "current container image does not match the frozen run's recorded "
            "container image"
        )
    helper_path = Path(__file__).resolve()
    source_commit = _git_commit()

    cfg.trainer.is_resume = True
    cfg.trainer.resume_from_checkpoint = str(checkpoint)
    cfg.datasets.vla_data.num_workers = resumed_num_workers
    cfg.datasets.vla_data.multiprocessing_context = (
        resumed_multiprocessing_context
    )
    cfg.resume_runtime_override = {
        "schema": METADATA_SCHEMA,
        "resume_helper_path": str(helper_path),
        "resume_helper_sha256": h100_training._sha256(helper_path),
        "source_commit": source_commit,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "container_image": current_image,
        "container_image_digest": image_digest,
        "runtime_config_path": str(runtime_config_path),
        "runtime_config_sha256": runtime_config_sha256,
        "changes": {
            "datasets.vla_data.num_workers": {
                "previous": previous_num_workers,
                "resumed": resumed_num_workers,
            },
            "datasets.vla_data.multiprocessing_context": {
                "previous": previous_multiprocessing_context,
                "resumed": resumed_multiprocessing_context,
            },
        },
    }

    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f"starvla-{run_id}-resume-runtime-",
        suffix=".yaml",
        dir="/tmp",
        delete=False,
    )
    try:
        handle.write(OmegaConf.to_yaml(cfg, resolve=True))
    finally:
        handle.close()
    return Path(handle.name), run_id


def launch(
    config_path: Path,
    *,
    checkpoint: Path,
    resume_runtime_config: Path,
    print_command_only: bool,
) -> None:
    """Run the original launch gates, then exec the same Accelerate command."""

    plan = h100_training.check_plan(
        config_path,
        deep=not print_command_only,
    )
    resolved_config, run_id = _resolved_resume_config(
        config_path.resolve(),
        plan,
        checkpoint=checkpoint,
        resume_runtime_config=resume_runtime_config,
    )
    command = h100_training._accelerate_command(plan, resolved_config)
    resolved = OmegaConf.load(resolved_config)
    metadata = resolved.resume_runtime_override
    change = metadata.changes["datasets.vla_data.num_workers"]
    context_change = metadata.changes[
        "datasets.vla_data.multiprocessing_context"
    ]
    print(f"Resolved run ID             : {run_id}")
    print(f"Resolved resume config      : {resolved_config}")
    print(
        "Resume runtime contract     : "
        f"{metadata.runtime_config_path} ({metadata.runtime_config_sha256})"
    )
    print(
        "Resume helper               : "
        f"{metadata.resume_helper_path} ({metadata.resume_helper_sha256})"
    )
    print(
        "Resume worker change        : "
        f"{change.previous} -> {change.resumed} per rank"
    )
    print(
        "Resume worker context       : "
        f"{context_change.previous} -> {context_change.resumed}"
    )
    print(f"Training command            : {shlex.join(command)}")
    if print_command_only:
        return

    runtime = plan["runtime"]
    env = os.environ.copy()
    env["STARVLA_USE_DEEPSPEED"] = "1" if runtime["use_deepspeed"] else "0"
    if runtime["torch_compile_environment"] == "disabled":
        env["TORCH_COMPILE_DISABLE"] = "1"
        env["TORCHDYNAMO_DISABLE"] = "1"
        env["STARVLA_ALLOW_TORCH_COMPILE"] = "0"
    interface = str(runtime["network_interface"])
    if interface == "auto":
        interface = h100_training._discover_default_interface()
    env["NCCL_SOCKET_IFNAME"] = interface
    env["GLOO_SOCKET_IFNAME"] = interface
    print(f"Network interface           : {interface}")
    sys.stdout.flush()
    os.chdir(REPO_ROOT)
    os.execvpe(command[0], command, env)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Resume one frozen H100 run with a schema-validated DataLoader "
            "worker correction."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--resume-runtime-config",
        type=Path,
        required=True,
    )
    parser.add_argument("--print-command", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        launch(
            args.config,
            checkpoint=args.checkpoint,
            resume_runtime_config=args.resume_runtime_config,
            print_command_only=bool(args.print_command),
        )
    except (
        h100_training.PlanError,
        subprocess.CalledProcessError,
        OSError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
