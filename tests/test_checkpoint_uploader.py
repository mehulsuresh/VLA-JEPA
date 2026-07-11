import os
import shlex
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
UPLOADER = REPO_ROOT / "scripts/watch_and_upload_checkpoints_gcs.sh"


def _write_fake_gcloud(
    bin_dir: Path,
    calls_path: Path,
    checkpoint_listing: tuple[str, ...] = (),
) -> None:
    gcloud = bin_dir / "gcloud"
    script = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"printf '%s\\n' \"$*\" >> {shlex.quote(str(calls_path))}",
    ]
    if checkpoint_listing:
        quoted_listing = " ".join(shlex.quote(path) for path in checkpoint_listing)
        script.extend(
            [
                'if [[ "${1:-}" == "storage" && "${2:-}" == "ls" ]]; then',
                f"  printf '%s\\n' {quoted_listing}",
                "fi",
            ]
        )
    gcloud.write_text("\n".join(script) + "\n", encoding="utf-8")
    gcloud.chmod(0o755)


def test_uploader_copies_checkpoints_final_model_and_logs(tmp_path):
    run_dir = tmp_path / "run"
    checkpoint = run_dir / "checkpoints/steps_1"
    final_model = run_dir / "final_model"
    tensorboard = run_dir / "starvla"
    extra_metadata = tmp_path / "extra-metadata"
    checkpoint.mkdir(parents=True)
    final_model.mkdir()
    tensorboard.mkdir()
    extra_metadata.mkdir()

    (run_dir / "config.yaml").write_text("trainer: {}\n", encoding="utf-8")
    (extra_metadata / "launch.env").write_text("RUN_ID=test-run\n", encoding="utf-8")
    (extra_metadata / "production_preflight_manifest.txt").write_text(
        "repo_head=test\n", encoding="utf-8"
    )
    (run_dir / "summary.jsonl").write_text('{"steps": 1}\n', encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"checkpoint")
    (checkpoint / "trainer_state.json").write_text('{"completed_steps": 1}\n', encoding="utf-8")
    (final_model / "pytorch_model.pt").write_bytes(b"final")
    (tensorboard / "events.out.tfevents.test").write_bytes(b"events")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls_path = tmp_path / "gcloud_calls.txt"
    _write_fake_gcloud(bin_dir, calls_path)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "RUN_ONCE": "1",
            "STABLE_SECONDS": "0",
            "LOG_SYNC_SECONDS": "0",
            "CLOUDSDK_CONFIG": str(tmp_path / "gcloud-config"),
            "EXTRA_METADATA_DIR": str(extra_metadata),
        }
    )
    subprocess.run(
        ["bash", str(UPLOADER), str(run_dir), "gs://test-bucket/test-run"],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )

    calls = calls_path.read_text(encoding="utf-8")
    assert f"storage rsync --recursive {checkpoint}" in calls
    assert "gs://test-bucket/test-run/checkpoints/steps_1" in calls
    assert f"storage rsync --recursive {final_model}" in calls
    assert "gs://test-bucket/test-run/final_model" in calls
    assert f"storage rsync --recursive {tensorboard}" in calls
    assert "gs://test-bucket/test-run/logs/starvla" in calls
    assert f"storage cp {run_dir / 'summary.jsonl'}" in calls

    state_dir = run_dir / ".upload_state"
    assert (state_dir / "uploaded_steps_1").exists()
    assert (state_dir / "uploaded_final_model").exists()
    assert (state_dir / "uploaded_runtime_logs").exists()
    assert (state_dir / "metadata/launch.env").exists()
    assert (state_dir / "metadata/production_preflight_manifest.txt").exists()


def test_uploader_waits_for_stable_checkpoint_and_final_model(tmp_path):
    run_dir = tmp_path / "run"
    checkpoint = run_dir / "checkpoints/steps_1"
    final_model = run_dir / "final_model"
    checkpoint.mkdir(parents=True)
    final_model.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"checkpoint")
    (final_model / "pytorch_model.pt").write_bytes(b"final")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls_path = tmp_path / "gcloud_calls.txt"
    _write_fake_gcloud(bin_dir, calls_path)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "RUN_ONCE": "1",
            "STABLE_SECONDS": "3600",
            "CLOUDSDK_CONFIG": str(tmp_path / "gcloud-config"),
        }
    )
    subprocess.run(
        ["bash", str(UPLOADER), str(run_dir), "gs://test-bucket/test-run"],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )

    assert not calls_path.exists()
    state_dir = run_dir / ".upload_state"
    assert not (state_dir / "uploaded_steps_1").exists()
    assert not (state_dir / "uploaded_final_model").exists()


def test_uploader_resynchronizes_metadata_created_after_startup(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "launch.env").write_text("RUN_ID=test-run\n", encoding="utf-8")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls_path = tmp_path / "gcloud_calls.txt"
    _write_fake_gcloud(bin_dir, calls_path)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "RUN_ONCE": "1",
            "CLOUDSDK_CONFIG": str(tmp_path / "gcloud-config"),
        }
    )
    command = ["bash", str(UPLOADER), str(run_dir), "gs://test-bucket/test-run"]
    subprocess.run(command, check=True, env=env, text=True, capture_output=True)
    first_fingerprint = (run_dir / ".upload_state/uploaded_metadata").read_text(
        encoding="utf-8"
    )

    (run_dir / "config.yaml").write_text("trainer: {}\n", encoding="utf-8")
    subprocess.run(command, check=True, env=env, text=True, capture_output=True)

    state_dir = run_dir / ".upload_state"
    assert (state_dir / "metadata/config.yaml").exists()
    assert (state_dir / "uploaded_metadata").read_text(encoding="utf-8") != first_fingerprint
    calls = calls_path.read_text(encoding="utf-8")
    assert calls.count("storage rsync --recursive") == 2


def test_uploader_supports_read_only_container_owned_run_directory(tmp_path):
    run_dir = tmp_path / "container-owned-run"
    checkpoint = run_dir / "checkpoints/steps_1"
    extra_metadata = tmp_path / "external-metadata"
    checkpoint.mkdir(parents=True)
    extra_metadata.mkdir()
    (run_dir / "config.yaml").write_text("trainer: {}\n", encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"checkpoint")
    (extra_metadata / "launch.env").write_text(
        "RUN_ID=container-owned-run\n", encoding="utf-8"
    )

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls_path = tmp_path / "gcloud_calls.txt"
    _write_fake_gcloud(bin_dir, calls_path)
    state_root = tmp_path / "external-upload-state"

    run_dir.chmod(0o555)
    try:
        env = os.environ.copy()
        env.update(
            {
                "PATH": f"{bin_dir}:{env['PATH']}",
                "RUN_ONCE": "1",
                "STABLE_SECONDS": "0",
                "CLOUDSDK_CONFIG": str(tmp_path / "gcloud-config"),
                "UPLOAD_STATE_ROOT": str(state_root),
                "EXTRA_METADATA_DIR": str(extra_metadata),
            }
        )
        subprocess.run(
            ["bash", str(UPLOADER), str(run_dir), "gs://test-bucket/test-run"],
            check=True,
            env=env,
            text=True,
            capture_output=True,
        )
    finally:
        run_dir.chmod(0o755)

    state_dir = state_root / run_dir.name
    assert (state_dir / "uploaded_metadata").exists()
    assert (state_dir / "uploaded_steps_1").exists()
    assert (state_dir / "metadata/config.yaml").exists()
    assert (state_dir / "metadata/launch.env").exists()


def test_uploader_prunes_only_oldest_remote_step_checkpoint(tmp_path):
    run_dir = tmp_path / "run"
    checkpoint = run_dir / "checkpoints/steps_40"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(b"checkpoint")

    remote_root = "gs://test-bucket/test-run/checkpoints"
    checkpoint_listing = tuple(
        f"{remote_root}/steps_{step}/" for step in (2, 10, 20, 40)
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls_path = tmp_path / "gcloud_calls.txt"
    _write_fake_gcloud(bin_dir, calls_path, checkpoint_listing)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "RUN_ONCE": "1",
            "STABLE_SECONDS": "0",
            "REMOTE_CHECKPOINT_MAX_TO_KEEP": "3",
            "CLOUDSDK_CONFIG": str(tmp_path / "gcloud-config"),
        }
    )
    subprocess.run(
        ["bash", str(UPLOADER), str(run_dir), "gs://test-bucket/test-run"],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )

    calls = calls_path.read_text(encoding="utf-8")
    assert f"storage rm --recursive {remote_root}/steps_2/" in calls
    assert f"storage rm --recursive {remote_root}/steps_10/" not in calls
    assert f"storage rm --recursive {remote_root}/steps_20/" not in calls
    assert f"storage rm --recursive {remote_root}/steps_40/" not in calls
