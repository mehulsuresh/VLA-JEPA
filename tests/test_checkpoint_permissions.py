import os

from starVLA.training.train_starvla import _make_artifact_tree_host_readable


def test_artifact_tree_is_readable_outside_container_owner(tmp_path):
    artifact_dir = tmp_path / "steps_1"
    nested_dir = artifact_dir / "nested"
    nested_dir.mkdir(parents=True)
    model_path = artifact_dir / "model.safetensors"
    state_path = nested_dir / "state.bin"
    model_path.write_bytes(b"model")
    state_path.write_bytes(b"state")

    artifact_dir.chmod(0o700)
    nested_dir.chmod(0o700)
    model_path.chmod(0o600)
    state_path.chmod(0o600)

    _make_artifact_tree_host_readable(str(artifact_dir))

    assert os.stat(artifact_dir).st_mode & 0o055 == 0o055
    assert os.stat(nested_dir).st_mode & 0o055 == 0o055
    assert os.stat(model_path).st_mode & 0o044 == 0o044
    assert os.stat(state_path).st_mode & 0o044 == 0o044
