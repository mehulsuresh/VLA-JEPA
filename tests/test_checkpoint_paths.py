import json
import tempfile
from pathlib import Path

from starVLA.model.framework.share_tools import read_mode_config as read_framework_mode_config
from starVLA.model.tools import read_mode_config as read_legacy_mode_config


def _write_run_with_nested_safetensors(tmpdir: str) -> Path:
    run_root = Path(tmpdir)
    (run_root / "config.yaml").write_text(
        "framework:\n"
        "  action_model:\n"
        "    future_action_window_size: 49\n",
        encoding="utf-8",
    )
    (run_root / "dataset_statistics.json").write_text(
        json.dumps({"new_embodiment": {"action": {"min": [0.0], "max": [1.0]}}}),
        encoding="utf-8",
    )
    ckpt_file = run_root / "checkpoints" / "steps_2500" / "model.safetensors"
    ckpt_file.parent.mkdir(parents=True)
    ckpt_file.write_bytes(b"test")
    return ckpt_file


def test_framework_read_mode_config_accepts_nested_safetensors_checkpoint():
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_file = _write_run_with_nested_safetensors(tmpdir)
        model_config, norm_stats = read_framework_mode_config(ckpt_file)

    assert model_config["framework"]["action_model"]["future_action_window_size"] == 49
    assert norm_stats["new_embodiment"]["action"]["max"] == [1.0]


def test_legacy_read_mode_config_accepts_nested_safetensors_checkpoint():
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_file = _write_run_with_nested_safetensors(tmpdir)
        model_config, norm_stats = read_legacy_mode_config(ckpt_file)

    assert model_config["framework"]["action_model"]["future_action_window_size"] == 49
    assert norm_stats["new_embodiment"]["action"]["min"] == [0.0]
