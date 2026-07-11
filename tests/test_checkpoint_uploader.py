import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
UPLOADER = REPO_ROOT / "scripts/watch_and_upload_checkpoints_gcs.sh"


def _write_fake_gcloud(bin_dir: Path, calls_path: Path) -> None:
    gcloud = bin_dir / "gcloud"
    gcloud.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"printf '%s\\n' \"$*\" >> {calls_path!s}\n",
        encoding="utf-8",
    )
    gcloud.chmod(0o755)


def test_uploader_copies_checkpoints_final_model_and_logs(tmp_path):
    run_dir = tmp_path / "run"
    checkpoint = run_dir / "checkpoints/steps_1"
    final_model = run_dir / "final_model"
    tensorboard = run_dir / "starvla"
    checkpoint.mkdir(parents=True)
    final_model.mkdir()
    tensorboard.mkdir()

    (run_dir / "config.yaml").write_text("trainer: {}\n", encoding="utf-8")
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


def test_uploader_supports_read_only_container_owned_run_directory(tmp_path):
    run_dir = tmp_path / "container-owned-run"
    checkpoint = run_dir / "checkpoints/steps_1"
    checkpoint.mkdir(parents=True)
    (run_dir / "config.yaml").write_text("trainer: {}\n", encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"checkpoint")

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
