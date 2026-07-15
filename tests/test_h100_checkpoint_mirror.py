import base64
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/mirror_h100_checkpoint_run.py"
SPEC = importlib.util.spec_from_file_location("h100_checkpoint_mirror", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
mirror = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mirror
SPEC.loader.exec_module(mirror)

METRIC = "heldout_focused_eval_task_failure_score_h10"
MODE = "min"
RUN_ID = "robot_ft_test_20260714"


def _selection_payload(best_step=None, value=None):
    return {
        "schema_version": 1,
        "best_metric_name": METRIC,
        "best_metric_mode": MODE,
        "best_metric_value": value,
        "best_metric_step": best_step,
        "checkpoint_relative_path": (
            None if best_step is None else f"checkpoints/steps_{best_step}"
        ),
    }


def _checkpoint(
    root: Path,
    step: int,
    *,
    best_step=None,
    best_value=None,
    legacy=False,
) -> Path:
    path = root / "checkpoints" / f"steps_{step}"
    path.mkdir(parents=True)
    for name in (
        "model.safetensors",
        "optimizer.bin",
        "scheduler.bin",
        *(f"random_states_{rank}.pkl" for rank in range(8)),
    ):
        (path / name).write_bytes(f"{step}:{name}".encode())
    trainer_state = {"completed_steps": step}
    if not legacy:
        trainer_state.update(
            {
                "selection_state_schema_version": 1,
                "best_metric_name": METRIC,
                "best_metric_mode": MODE,
                "best_metric_value": best_value,
                "best_metric_step": best_step,
            }
        )
        (path / "selection_state.json").write_text(
            json.dumps(_selection_payload(best_step, best_value)), encoding="utf-8"
        )
    (path / "trainer_state.json").write_text(
        json.dumps(trainer_state), encoding="utf-8"
    )
    return path


def _manifest(root: Path, step: int, *, best_step=None, best_value=None, legacy=False):
    checkpoint = _checkpoint(
        root,
        step,
        best_step=best_step,
        best_value=best_value,
        legacy=legacy,
    )
    return mirror.validate_checkpoint_tree(
        checkpoint, metric_name=METRIC, metric_mode=MODE
    )


def _production_reports(batch_size=2):
    seed = "1" * 64
    provenance = [
        {
            "dataset_name": "fixture",
            "manifest_path": "/fixtures/manifest.json",
            "role": "holdout",
            "selected_episode_count": batch_size,
            "selected_frame_count": batch_size * 50,
            "train_episode_count": 10,
            "train_frame_count": 500,
            "holdout_episode_count": batch_size,
            "train_statistics_path": "/fixtures/stats.json",
            "manifest_sha256": "2" * 64,
            "selected_episode_set_sha256": "3" * 64,
            "train_episode_set_sha256": "4" * 64,
            "holdout_episode_set_sha256": "5" * 64,
            "full_catalog_sha256": "6" * 64,
            "train_statistics_sha256": "7" * 64,
        }
    ]
    def window(index, *, focused=False):
        return {
            "dataset_index": 0,
            "dataset_name": "fixture",
            "episode_id": index,
            "base_index": index * 100 + 20,
            "valid_base_index_min": index * 100,
            "valid_base_index_max": index * 100 + 50,
            "structural_candidate_count": 51,
            "evaluable_candidate_count": 50,
            "selection_pool_candidate_count": 50,
            "valid_action_timesteps": 50,
            "valid_action_elements": 900,
            "anchor_subtask_index": 2,
            "action_subtask_indices": [2] * 50,
            "valid_action_elements_per_timestep": [18] * 50,
            "open_to_close_transitions_h10": 1 if focused else 0,
            "close_to_open_transitions_h10": 1 if focused else 0,
            "open_to_close_window_h10": focused,
            "close_to_open_window_h10": focused,
            "arm_movement_elements_h10": 10 if focused else 0,
            "arm_movement_hold_abs_h10": 1.0 if focused else 0.0,
        }

    windows = [window(index) for index in range(batch_size)]
    focused_windows = [window(index, focused=True) for index in range(batch_size)]
    horizons = (1, 5, 10, 20, 50)
    common = {
        "schema_version": 1,
        "seed_sha256": seed,
        "observation_count": batch_size,
        "action_evaluable_observation_count": batch_size,
        "action_dim": 18,
        "valid_action_timestep_count": batch_size * 50,
        "valid_action_element_count": batch_size * 900,
        "subtask_observation_counts": {"2": batch_size},
        "subtask_evaluable_observation_counts": {"2": batch_size},
        "subtask_action_timestep_counts_by_horizon": {
            str(horizon): {"2": batch_size * horizon} for horizon in horizons
        },
        "subtask_valid_action_element_counts_by_horizon": {
            str(horizon): {"2": batch_size * horizon * 18} for horizon in horizons
        },
        "zero_valid_action_episodes": [],
        "production_valid": True,
        "checkpoint_selection_eligible": True,
        "episode_split_provenance": provenance,
        "windows": windows,
    }
    unbiased = {
        **common,
        "algorithm": "nonzero_valid_unpadded_uniform_v1",
        "purpose": "one_window_per_manifest_holdout_episode_checkpoint_eval",
        "view": "unbiased",
        "observation_mode": "deployment_action_current_qwen_rgb_v1",
        "evaluation_video_offsets": [0],
        "action_offset_range_inclusive": [0, 49],
        "frames_per_episode": 1,
    }
    unbiased_contract = {
        "algorithm": unbiased["algorithm"],
        "observation_mode": unbiased["observation_mode"],
        "evaluation_video_offsets": unbiased["evaluation_video_offsets"],
        "action_offset_range_inclusive": unbiased["action_offset_range_inclusive"],
        "frames_per_episode": unbiased["frames_per_episode"],
        "seed_sha256": seed,
        "windows": windows,
    }
    unbiased["window_selection_sha256"] = hashlib.sha256(
        mirror._canonical_json_bytes(unbiased_contract)
    ).hexdigest()
    focused = {
        **common,
        "algorithm": "h10_gripper_transition_stage_balanced_v1",
        "purpose": "h10_transition_stage_focused_manifest_holdout_checkpoint_eval",
        "view": "focused",
        "observation_mode": "deployment_action_current_qwen_rgb_v1",
        "evaluation_video_offsets": [0],
        "action_offset_range_inclusive": [0, 49],
        "frames_per_episode": 1,
        "windows": focused_windows,
        "open_to_close_transition_count_h10": batch_size,
        "close_to_open_transition_count_h10": batch_size,
        "open_to_close_transition_window_count_h10": batch_size,
        "close_to_open_transition_window_count_h10": batch_size,
        "arm_movement_element_count_h10": batch_size * 10,
        "arm_movement_hold_abs_sum_h10": float(batch_size),
        "movement_threshold_normalized": 0.01,
        "focused_subtasks": [2],
    }
    focused_contract = {
        "algorithm": focused["algorithm"],
        "transition_horizon": 10,
        "focused_subtasks": focused["focused_subtasks"],
        "manifest_seed_sha256": seed,
        "unbiased_window_selection_sha256": unbiased["window_selection_sha256"],
        "windows": focused_windows,
    }
    focused["window_selection_sha256"] = hashlib.sha256(
        mirror._canonical_json_bytes(focused_contract)
    ).hexdigest()
    return unbiased, focused


def _reseal_report_digests(unbiased, focused):
    unbiased_contract = {
        "algorithm": unbiased["algorithm"],
        "observation_mode": unbiased["observation_mode"],
        "evaluation_video_offsets": unbiased["evaluation_video_offsets"],
        "action_offset_range_inclusive": unbiased["action_offset_range_inclusive"],
        "frames_per_episode": unbiased["frames_per_episode"],
        "seed_sha256": unbiased["seed_sha256"],
        "windows": unbiased["windows"],
    }
    unbiased["window_selection_sha256"] = hashlib.sha256(
        mirror._canonical_json_bytes(unbiased_contract)
    ).hexdigest()
    focused_contract = {
        "algorithm": focused["algorithm"],
        "transition_horizon": 10,
        "focused_subtasks": focused["focused_subtasks"],
        "manifest_seed_sha256": focused["seed_sha256"],
        "unbiased_window_selection_sha256": unbiased["window_selection_sha256"],
        "windows": focused["windows"],
    }
    focused["window_selection_sha256"] = hashlib.sha256(
        mirror._canonical_json_bytes(focused_contract)
    ).hexdigest()


def _write_canonical_json(path: Path, payload) -> str:
    content = mirror._canonical_json_bytes(payload) + b"\n"
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _rewrite_restore_indexes(root: Path, index) -> None:
    generation = index["generation"]
    _write_canonical_json(root / "restore-index.json", index)
    _write_canonical_json(
        root / f"manifests/restore_indexes/generation_{generation:08d}.json",
        index,
    )


def _run_evidence(run_dir: Path):
    run_dir.mkdir(parents=True, exist_ok=True)
    config_bytes = b"trainer:\n  seed: 17\n"
    (run_dir / "config.yaml").write_bytes(config_bytes)
    (run_dir / "config.json").write_text(
        json.dumps({"run_id": RUN_ID, "output_dir": str(run_dir), "seed": 17})
    )
    (run_dir / "resolved_training_schedule.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "configured": {},
                "resolved": {
                    "effective_global_batch_size": 2,
                    "eval_interval": 10,
                    "max_train_steps": 100,
                    "num_warmup_steps": 1,
                    "save_interval": 10,
                },
                "source_config": {
                    "path": "config.yaml",
                    "sha256": hashlib.sha256(config_bytes).hexdigest(),
                },
            }
        )
    )
    unbiased, focused = _production_reports()
    (run_dir / "heldout_eval_windows.json").write_text(json.dumps(unbiased))
    (run_dir / "heldout_focused_eval_windows.json").write_text(json.dumps(focused))
    (run_dir / "dataset_statistics.json").write_text("{}\n")
    (run_dir / "dataset_provenance.json").write_text("{}\n")
    (run_dir / "summary.jsonl").write_text("{}\n")
    return mirror.validate_run_evidence_identity(
        run_dir, source_output_dir=str(run_dir)
    )


def _eval_artifact(run_dir: Path, manifest, *, value=0.25, include_model_hash=True):
    evidence = _run_evidence(run_dir)
    assert manifest.step is not None
    step = manifest.step
    path = run_dir / "heldout_eval_metrics" / f"step_{step:08d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "step": step,
        "source_kind": "checkpoint",
        "source_path": str(run_dir / "checkpoints" / f"steps_{step}"),
        "trainer_state_sha256": manifest.file("trainer_state.json").sha256,
        "model_file": "model.safetensors",
        "model_file_size_bytes": manifest.file("model.safetensors").size,
    }
    if include_model_hash:
        checkpoint["model_file_sha256"] = manifest.file("model.safetensors").sha256
    payload = {
        "schema_version": 1,
        "checkpoint_step": step,
        "checkpoint_relative_path": f"checkpoints/steps_{step}",
        "checkpoint": checkpoint,
        "run": {
            "run_id": RUN_ID,
            "output_dir": str(run_dir),
            "seed": 17,
            "config_path": "config.yaml",
            "config_sha256": evidence.config_sha256,
            "resolved_training_schedule": {
                "path": "resolved_training_schedule.json",
                "sha256": evidence.schedule_sha256,
            },
            "source_training_config": None,
        },
        "sampling_reports": {
            "unbiased": dict(evidence.unbiased_sampling_report),
            "focused": dict(evidence.focused_sampling_report),
        },
        "production_valid": True,
        "checkpoint_selection_eligible": True,
        "selection_metric": {
            "name": METRIC,
            "mode": MODE,
            "eligible": True,
            "value": value,
        },
        "metrics": {"unbiased": {}, "focused": {METRIC: value}},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, evidence


def _pointer(run_dir: Path, step: int, value: float):
    (run_dir / "best_checkpoint.json").write_text(
        json.dumps(_selection_payload(step, value)), encoding="utf-8"
    )


def _baseline_eval_artifact(run_dir: Path, *, value=0.40) -> Path:
    source = run_dir / "heldout_eval_metrics/step_00000010.json"
    if not source.is_file():
        raise AssertionError("fixture requires the checkpoint-bound step-10 eval first")
    payload = json.loads(source.read_text())
    payload["checkpoint_step"] = 0
    payload["checkpoint_relative_path"] = None
    payload["checkpoint"] = {
        "step": 0,
        "source_path": None,
        "source_kind": "live_in_memory_model",
    }
    payload["selection_metric"]["value"] = value
    payload["metrics"]["focused"][METRIC] = value
    target = run_dir / "heldout_eval_metrics/step_00000000.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    return target


def _identity(name):
    return {
        "alias": name,
        "resolved_hostname": f"{name}.example",
        "resolved_user": "mehul",
        "resolved_port": 22,
        "host_key_alias": None,
        "known_host_fingerprints": ["256 SHA256:fixture (ED25519)"],
        "remote_hostname": name,
        "remote_fqdn": f"{name}.example",
        "remote_python_version": sys.version.split()[0],
    }


def _state(path: Path, **overrides):
    kwargs = {
        "run_id": RUN_ID,
        "source_run_dir": f"/source/{RUN_ID}",
        "local_root": f"/local/{RUN_ID}",
        "columbus_root": str(mirror.SAFE_COLUMBUS_BASE / RUN_ID),
        "metric_name": METRIC,
        "metric_mode": MODE,
        "h100_identity": _identity("h100"),
        "columbus_identity": _identity("columbus"),
    }
    kwargs.update(overrides)
    return mirror.MirrorState(path, **kwargs)


def test_checkpoint_and_eval_are_cryptographically_bound(tmp_path):
    run_dir = tmp_path / RUN_ID
    manifest = _manifest(run_dir, 5, best_step=5, best_value=0.25)
    artifact_path, evidence = _eval_artifact(run_dir, manifest)
    artifact = mirror.validate_eval_artifact(
        artifact_path,
        checkpoint_manifest=manifest,
        metric_name=METRIC,
        metric_mode=MODE,
        run_id=RUN_ID,
        run_evidence=evidence,
    )
    assert artifact.cryptographically_bound and artifact.production_eligible


def test_missing_model_hash_is_archival_only_and_never_recoverable(tmp_path):
    run_dir = tmp_path / RUN_ID
    manifest = _manifest(run_dir, 5, best_step=5, best_value=0.25)
    artifact_path, evidence = _eval_artifact(
        run_dir, manifest, include_model_hash=False
    )
    with pytest.raises(mirror.MirrorError, match="archival-only"):
        mirror.validate_eval_artifact(
            artifact_path,
            checkpoint_manifest=manifest,
            metric_name=METRIC,
            metric_mode=MODE,
            run_id=RUN_ID,
            run_evidence=evidence,
        )
    artifact = mirror.validate_eval_artifact(
        artifact_path,
        checkpoint_manifest=manifest,
        metric_name=METRIC,
        metric_mode=MODE,
        run_id=RUN_ID,
        run_evidence=evidence,
        allow_legacy_archive=True,
    )
    assert not artifact.production_eligible and artifact.archival_reason


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda p: p["checkpoint"].update(model_file_sha256="0" * 64), "model_file_sha256"),
        (lambda p: p["run"].update(config_sha256="0" * 64), "config identity"),
        (
            lambda p: p["sampling_reports"]["focused"].update(
                production_valid=False
            ),
            "sampling_reports",
        ),
        (
            lambda p: p["metrics"]["unbiased"].update({METRIC: 0.25}),
            "wrong group|exactly once",
        ),
    ],
)
def test_eval_rejects_binding_and_placement_drift(tmp_path, mutation, message):
    run_dir = tmp_path / RUN_ID
    manifest = _manifest(run_dir, 10, best_step=10, best_value=0.25)
    path, evidence = _eval_artifact(run_dir, manifest)
    payload = json.loads(path.read_text())
    mutation(payload)
    path.write_text(json.dumps(payload))
    with pytest.raises(mirror.MirrorError, match=message):
        mirror.validate_eval_artifact(
            path,
            checkpoint_manifest=manifest,
            metric_name=METRIC,
            metric_mode=MODE,
            run_id=RUN_ID,
            run_evidence=evidence,
        )


def test_window_digests_are_recomputed_from_complete_evidence(tmp_path):
    run_dir = tmp_path / RUN_ID
    _run_evidence(run_dir)
    payload = json.loads((run_dir / "heldout_eval_windows.json").read_text())
    payload["windows"][0]["episode_id"] = 999
    (run_dir / "heldout_eval_windows.json").write_text(json.dumps(payload))
    with pytest.raises(
        mirror.MirrorError,
        match="window_selection_sha256|different heldout episodes|canonical producer order",
    ):
        mirror.validate_run_evidence_identity(run_dir, source_output_dir=str(run_dir))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda unbiased, focused: unbiased["windows"][0].pop("dataset_index"),
            "incomplete/unexpected fields",
        ),
        (
            lambda unbiased, focused: unbiased["windows"][0].update(
                valid_action_elements=899
            ),
            "valid-action aggregates",
        ),
        (
            lambda unbiased, focused: (
                unbiased["subtask_action_timestep_counts_by_horizon"].pop("1"),
                unbiased["subtask_valid_action_element_counts_by_horizon"].pop("1"),
            ),
            "horizon coverage",
        ),
        (
            lambda unbiased, focused: focused.update(
                open_to_close_transition_count_h10=999
            ),
            "aggregates do not match windows",
        ),
    ],
)
def test_inner_window_shapes_and_derived_aggregates_are_exact(mutation, message):
    unbiased, focused = _production_reports()
    mutation(unbiased, focused)
    _reseal_report_digests(unbiased, focused)
    with pytest.raises(mirror.MirrorError, match=message):
        mirror._validate_window_report_contracts(
            unbiased, focused, expected_observations=2
        )


def test_serialized_run_evidence_hashes_must_match_its_own_manifest(tmp_path):
    evidence = _run_evidence(tmp_path / RUN_ID)
    payload = evidence.as_dict()
    payload["config_sha256"] = "0" * 64
    with pytest.raises(mirror.MirrorError, match="contradicts its manifest"):
        mirror.RunEvidenceIdentity.from_dict(payload)


def test_eval_cache_is_invalidated_when_semantic_run_evidence_changes(
    tmp_path, monkeypatch
):
    controller, source, _local, _columbus = _fake_controller(tmp_path, monkeypatch)
    controller.run_once()
    config_path = source / "config.yaml"
    config_path.write_bytes(config_path.read_bytes() + b"# semantic drift\n")
    schedule_path = source / "resolved_training_schedule.json"
    schedule = json.loads(schedule_path.read_text())
    schedule["source_config"]["sha256"] = hashlib.sha256(
        config_path.read_bytes()
    ).hexdigest()
    schedule_path.write_text(json.dumps(schedule))
    # The eval bytes did not change and still claim the old config/schedule.
    # Reusing them by file fingerprint alone would incorrectly report healthy.
    with pytest.raises(
        mirror.MirrorError,
        match="immutable run evidence|config identity|schedule identity",
    ):
        controller.run_once()


def test_immutable_run_evidence_drift_is_rejected(tmp_path, monkeypatch):
    controller, source, _local, _columbus = _fake_controller(tmp_path, monkeypatch)
    controller.run_once()
    statistics = source / "dataset_statistics.json"
    statistics.write_bytes(statistics.read_bytes() + b"{\"drift\":true}\n")
    with pytest.raises(mirror.MirrorError, match="immutable run evidence changed"):
        controller.run_once()


def test_mutable_summary_evidence_may_advance_after_authentication(
    tmp_path, monkeypatch
):
    controller, source, _local, _columbus = _fake_controller(tmp_path, monkeypatch)
    controller.run_once()
    old_digest = controller.state.payload["source_evidence"]["identity"]["manifest"][
        "manifest_sha256"
    ]
    summary = source / "summary.jsonl"
    summary.write_bytes(summary.read_bytes() + b"{\"step\":20}\n")
    result = controller.run_once()
    new_digest = controller.state.payload["source_evidence"]["identity"]["manifest"][
        "manifest_sha256"
    ]
    assert result["status"] == "healthy"
    assert old_digest != new_digest


@pytest.mark.parametrize(
    "replacement",
    [b"\n", b"[]\n", b'{"rewritten":true}\n'],
)
def test_append_only_summary_rejects_truncation_or_rewrite(
    tmp_path, monkeypatch, replacement
):
    controller, source, _local, _columbus = _fake_controller(tmp_path, monkeypatch)
    controller.run_once()
    (source / "summary.jsonl").write_bytes(replacement)
    with pytest.raises(mirror.MirrorError, match="append-only summary.jsonl"):
        controller.run_once()


def test_checkpoint_contract_rejects_legacy_marker_mix_and_extra_keys(tmp_path):
    checkpoint = _checkpoint(tmp_path / "run", 10, legacy=True)
    (checkpoint / "selection_state.json").write_text(json.dumps(_selection_payload()))
    with pytest.raises(mirror.MirrorError, match="legacy trainer_state"):
        mirror.validate_checkpoint_tree(checkpoint, metric_name=METRIC, metric_mode=MODE)
    (checkpoint / "selection_state.json").unlink()
    trainer = json.loads((checkpoint / "trainer_state.json").read_text())
    trainer["best_metric_step"] = 2
    (checkpoint / "trainer_state.json").write_text(json.dumps(trainer))
    with pytest.raises(mirror.MirrorError, match="legacy trainer_state|unexpected"):
        # Markerless, half-populated legacy state is archival-only and the
        # production controller rejects its resulting legacy manifest.
        manifest = mirror.validate_checkpoint_tree(
            checkpoint, metric_name=METRIC, metric_mode=MODE
        )
        assert manifest.checkpoint_schema != "selection_v1"


def _eval(step, value, *, eligible=True):
    return mirror.EvalArtifact(
        step=step,
        trainer_state_sha256="1" * 64,
        metric_name=METRIC,
        metric_mode=MODE,
        metric_value=value,
        sha256=(f"{step:064x}"[-64:]),
        size=10,
        cryptographically_bound=True,
        production_eligible=eligible,
    )


def test_strict_global_best_retains_prior_on_ties_and_detects_bad_pointer(tmp_path):
    evals = {10: _eval(10, 0.05), 20: _eval(20, 0.05), 30: _eval(30, 0.15)}
    assert mirror.strict_global_best(evals, metric_mode="min") == (10, 0.05)
    manifests = {
        10: _manifest(tmp_path / "run", 10, best_step=10, best_value=0.05),
        20: _manifest(tmp_path / "run", 20, best_step=10, best_value=0.05),
        30: _manifest(tmp_path / "run", 30, best_step=10, best_value=0.05),
    }
    bad = mirror.SelectionPointer(METRIC, MODE, 0.10, 20, "checkpoints/steps_20")
    with pytest.raises(mirror.MirrorError, match="strict global"):
        mirror.authenticate_selection_history(
            manifests, evals, bad, metric_mode=MODE
        )


def test_every_checkpoint_dependency_must_match_best_eval_at_its_step(tmp_path):
    manifests = {
        10: _manifest(tmp_path / "run", 10, best_step=10, best_value=0.2),
        20: _manifest(tmp_path / "run", 20, best_step=20, best_value=0.3),
    }
    evals = {10: _eval(10, 0.2), 20: _eval(20, 0.3)}
    pointer = mirror.SelectionPointer(METRIC, MODE, 0.2, 10, "checkpoints/steps_10")
    with pytest.raises(mirror.MirrorError, match="dependency/value"):
        mirror.authenticate_selection_history(
            manifests, evals, pointer, metric_mode=MODE
        )


def test_state_pins_endpoint_identity_and_pointer_history_is_monotonic(tmp_path):
    path = tmp_path / "state" / "state.json"
    state = _state(path)
    first = mirror.SelectionPointer(METRIC, MODE, 0.2, 10, "checkpoints/steps_10")
    state.set_pointer_tier(first, "1" * 64, tier="local")
    state.set_pointer_tier(first, "1" * 64, tier="columbus")
    state.save()
    with pytest.raises(mirror.MirrorError, match="identity/configuration drift"):
        _state(path, h100_identity=_identity("replacement"))
    reloaded = _state(path)
    tie = mirror.SelectionPointer(METRIC, MODE, 0.2, 20, "checkpoints/steps_20")
    with pytest.raises(mirror.MirrorError, match="strict improvement"):
        reloaded.set_pointer_tier(tie, "2" * 64, tier="local")


def test_mark_recoverable_requires_modern_eval_evidence_and_receipt(tmp_path):
    manifest = _manifest(tmp_path / "run", 10, best_step=10, best_value=0.2)
    state = _state(tmp_path / "state" / "state.json")
    state.checkpoint_seen(manifest)
    state.verify_tier(10, "local", manifest.manifest_sha256)
    state.verify_tier(10, "columbus", manifest.manifest_sha256)
    with pytest.raises(mirror.MirrorError, match="production eval"):
        state.mark_recoverable(
            10, (10,), evidence_manifest_sha256="2" * 64, receipt_sha256="3" * 64
        )


@pytest.mark.parametrize(
    "path",
    ["a/../b", "a//b", "/absolute", "a\\b", "a/./b", "a\x00b"],
)
def test_relative_paths_reject_dotdot_noncanonical_and_controls(path):
    assert not mirror._safe_relative_path(path)


def test_json_loader_rejects_duplicate_keys_and_nonfinite(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a":1,"a":2}')
    with pytest.raises(mirror.MirrorError, match="duplicate"):
        mirror._load_json_object(duplicate, label="fixture")
    duplicate.write_text('{"a":NaN}')
    with pytest.raises(mirror.MirrorError, match="non-finite"):
        mirror._load_json_object(duplicate, label="fixture")


def test_lightweight_inventory_has_full_stat_fingerprint_and_detects_incomplete_latest(tmp_path):
    run_dir = tmp_path / RUN_ID
    checkpoint = _checkpoint(run_dir, 10)
    inventory = mirror.lightweight_source_inventory(run_dir)
    stat_payload = inventory["checkpoints"]["steps_10"]["entries"]["model.safetensors"]
    assert {"device", "inode", "ctime_ns", "mtime_ns", "uid", "nlink"} <= set(
        stat_payload
    )
    (checkpoint / "optimizer.bin").unlink()
    inventory = mirror.lightweight_source_inventory(run_dir)
    complete, reason = mirror._checkpoint_light_complete(
        inventory["checkpoints"]["steps_10"]
    )
    assert not complete and "missing" in reason


def test_remote_endpoint_rejects_option_like_hosts():
    with pytest.raises(mirror.MirrorError, match="unsafe SSH host"):
        mirror.RemoteEndpoint("-oProxyCommand=evil", mirror.CommandRunner())


def test_symlinked_state_and_history_directories_are_rejected(tmp_path):
    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(mirror.MirrorError, match="symlink"):
        mirror._local_mirror_root(linked, create=False)


def test_retention_uses_cached_manifests_after_source_removal(tmp_path):
    manifests = {
        10: _manifest(tmp_path / "run", 10, best_step=10, best_value=0.2),
        20: _manifest(tmp_path / "run", 20, best_step=10, best_value=0.2),
        30: _manifest(tmp_path / "run", 30, best_step=10, best_value=0.2),
        40: _manifest(tmp_path / "run", 40, best_step=10, best_value=0.2),
    }
    assert mirror.retained_recovery_steps(
        manifests, manifests, best_step=10, limit=3
    ) == (10, 30, 40)


class FakeEndpoint:
    def __init__(self, root: Path, *, role: str):
        self.root = root
        self.role = role
        self.host = role

    def invoke_json(self, args, *, mutation=False, heartbeat=None):
        command, *values = map(str, args)
        if command == "--internal-resolve-source-root":
            return mirror.resolve_source_root_binding(values[0])
        if command == "--internal-light-inventory":
            return mirror.lightweight_source_inventory(Path(values[0]))
        if command == "--internal-inspect-checkpoint":
            return mirror.validate_checkpoint_tree(
                Path(values[0]), metric_name=values[1], metric_mode=values[2]
            ).as_dict()
        if command == "--internal-inspect-evidence":
            return mirror.validate_run_evidence_identity(
                Path(values[0]), source_output_dir=values[1]
            ).as_dict()
        if command == "--internal-file-prefix-sha256":
            return mirror._file_prefix_sha256(Path(values[0]), int(values[1]))
        if command == "--internal-inspect-snapshot":
            return mirror.validate_evidence_snapshot(Path(values[0])).as_dict()
        if command == "--internal-inspect-eval":
            return mirror.validate_eval_artifact(
                Path(values[0]),
                checkpoint_manifest=mirror._manifest_from_b64(values[1]),
                metric_name=values[2],
                metric_mode=values[3],
                run_id=values[4],
                run_evidence=mirror._evidence_from_b64(values[5]),
                allow_legacy_archive=values[6] == "1",
            ).as_dict()
        if command == "--internal-inspect-baseline-eval":
            return mirror.validate_baseline_eval_artifact(
                Path(values[0]),
                metric_name=values[1],
                metric_mode=values[2],
                run_id=values[3],
                run_evidence=mirror._evidence_from_b64(values[4]),
            ).as_dict()
        if command == "--internal-inspect-pointer":
            maximum = None if values[3] == "none" else int(values[3])
            pointer, record = mirror.validate_pointer_file(
                Path(values[0]),
                metric_name=values[1],
                metric_mode=values[2],
                maximum_step=maximum,
            )
            content = Path(values[0]).read_bytes()
            return {
                "pointer": pointer.as_dict(),
                "record": record.as_dict(),
                "content_b64": base64.b64encode(content).decode(),
            }
        if command == "--internal-path-status":
            return mirror._internal_path_status(Path(values[0]))
        if command == "--internal-available-bytes":
            stats = os.statvfs(Path(values[0]))
            return {"available_bytes": stats.f_bavail * stats.f_frsize}
        if command == "--internal-acquire-lock":
            return mirror._internal_acquire_lock(values[0], Path(values[1]), values[2])
        if command == "--internal-release-lock":
            return mirror._internal_release_lock(values[0], Path(values[1]), values[2])
        if command == "--internal-create-stage":
            return mirror._internal_create_stage(
                values[0], Path(values[1]), Path(values[2]), values[3]
            )
        if command == "--internal-promote-tree":
            return mirror._internal_promote_tree(
                values[0],
                Path(values[1]),
                Path(values[2]),
                Path(values[3]),
                mirror._manifest_from_b64(values[4]),
                values[5],
                values[6],
            )
        if command == "--internal-promote-eval":
            return mirror._internal_promote_eval(
                values[0],
                Path(values[1]),
                Path(values[2]),
                Path(values[3]),
                mirror._manifest_from_b64(values[4]),
                values[5],
                values[6],
                values[7],
                mirror._evidence_from_b64(values[9]),
            )
        if command == "--internal-promote-pointer":
            return mirror._internal_promote_pointer(
                values[0],
                Path(values[1]),
                Path(values[2]),
                Path(values[3]),
                values[4],
                values[5],
                values[6],
                int(values[7]),
                values[8],
            )
        if command == "--internal-put-record":
            return mirror._internal_put_record(
                values[0],
                Path(values[1]),
                values[2],
                values[3],
                values[4],
                values[5],
            )
        if command == "--internal-trash-checkpoint":
            return mirror._internal_trash_checkpoint(
                values[0],
                Path(values[1]),
                Path(values[2]),
                Path(values[3]),
                values[4],
                values[5],
                values[6],
            )
        if command == "--internal-delete-trash":
            return mirror._internal_delete_trash(
                values[0], Path(values[1]), Path(values[2])
            )
        if command == "--internal-verify-index":
            if values[4] != "full":
                raise mirror.MirrorError("restore-index verification mode must be full")
            return mirror.verify_restore_index_at_root(
                Path(values[1]),
                run_id=values[0],
                metric_name=values[2],
                metric_mode=values[3],
            )
        raise AssertionError(f"unhandled fake endpoint command: {args}")

    def rsync_to_local(
        self, source, destination, *, files_from=None, delete=False, heartbeat=None
    ):
        source_path = Path(str(source).rstrip("/"))
        destination_path = Path(str(destination).rstrip("/"))
        if files_from is not None:
            destination_path.mkdir(parents=True, exist_ok=True)
            names = Path(files_from).read_bytes().split(b"\x00")
            for raw_name in names:
                if not raw_name:
                    continue
                relative = raw_name.decode()
                target = destination_path / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path / relative, target)
        elif source_path.is_dir():
            destination_path.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source_path, destination_path, dirs_exist_ok=True)
        else:
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination_path)

    def rsync_from_local(self, source, destination, *, delete=False, heartbeat=None):
        source_path = Path(str(source).rstrip("/"))
        destination_path = Path(str(destination).rstrip("/"))
        if source_path.is_dir():
            destination_path.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source_path, destination_path, dirs_exist_ok=True)
        else:
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination_path)


def _fake_controller(tmp_path, monkeypatch, *, retain=3):
    relay = tmp_path / "relay"
    relay.mkdir(mode=0o700)
    base = relay / "checkpoint-mirror-storage"
    monkeypatch.setattr(mirror, "SAFE_COLUMBUS_PARENT", relay)
    monkeypatch.setattr(mirror, "SAFE_COLUMBUS_BASE", base)
    source = tmp_path / "source" / RUN_ID
    manifest = _manifest(source, 10, best_step=10, best_value=0.25)
    _eval_artifact(source, manifest, value=0.25)
    _pointer(source, 10, 0.25)
    local = tmp_path / "local" / RUN_ID
    columbus = base / RUN_ID
    config = mirror.MirrorConfig(
        run_id=RUN_ID,
        h100_host="h100",
        source_run_dir=str(source),
        local_root=local,
        columbus_host="columbus",
        columbus_root=columbus,
        state_dir=tmp_path / "state",
        metric_name=METRIC,
        metric_mode=MODE,
        retain=retain,
        reserve_bytes=0,
    )
    controller = mirror.MirrorController(config, mirror.CommandRunner())
    controller.source = FakeEndpoint(source, role="h100")
    controller.columbus = FakeEndpoint(columbus, role="columbus")
    monkeypatch.setattr(
        mirror,
        "resolve_endpoint_identity",
        lambda endpoint: _identity(endpoint.role),
    )
    return controller, source, local, columbus


def test_logical_source_with_symlinked_mount_is_realpath_pinned_end_to_end(
    tmp_path, monkeypatch
):
    relay = tmp_path / "relay"
    relay.mkdir(mode=0o700)
    base = relay / "checkpoint-mirror-storage"
    monkeypatch.setattr(mirror, "SAFE_COLUMBUS_PARENT", relay)
    monkeypatch.setattr(mirror, "SAFE_COLUMBUS_BASE", base)

    actual_mount = tmp_path / "actual-vla-jepa"
    actual_mount.mkdir()
    logical_parent = tmp_path / "mnt"
    logical_parent.mkdir()
    logical_mount = logical_parent / "vla-jepa"
    logical_mount.symlink_to(actual_mount, target_is_directory=True)
    actual_source = actual_mount / RUN_ID
    logical_source = logical_mount / RUN_ID
    manifest = _manifest(actual_source, 10, best_step=10, best_value=0.25)
    eval_path, _ = _eval_artifact(actual_source, manifest, value=0.25)

    config_json_path = actual_source / "config.json"
    config_json = json.loads(config_json_path.read_text())
    config_json["output_dir"] = str(logical_source)
    config_json_path.write_text(json.dumps(config_json))
    evidence = mirror.validate_run_evidence_identity(
        actual_source, source_output_dir=str(logical_source)
    )
    eval_payload = json.loads(eval_path.read_text())
    eval_payload["checkpoint"]["source_path"] = str(
        logical_source / "checkpoints/steps_10"
    )
    eval_payload["run"]["output_dir"] = str(logical_source)
    eval_payload["run"]["config_sha256"] = evidence.config_sha256
    eval_payload["run"]["resolved_training_schedule"][
        "sha256"
    ] = evidence.schedule_sha256
    eval_path.write_text(json.dumps(eval_payload))
    _pointer(actual_source, 10, 0.25)

    local = tmp_path / "local" / RUN_ID
    columbus = base / RUN_ID
    controller = mirror.MirrorController(
        mirror.MirrorConfig(
            run_id=RUN_ID,
            h100_host="h100",
            source_run_dir=str(logical_source),
            local_root=local,
            columbus_host="columbus",
            columbus_root=columbus,
            state_dir=tmp_path / "state",
            metric_name=METRIC,
            metric_mode=MODE,
            reserve_bytes=0,
        ),
        mirror.CommandRunner(),
    )
    controller.source = FakeEndpoint(actual_source, role="h100")
    controller.columbus = FakeEndpoint(columbus, role="columbus")
    monkeypatch.setattr(
        mirror, "resolve_endpoint_identity", lambda endpoint: _identity(endpoint.role)
    )

    with pytest.raises(mirror.MirrorError, match="symlinked ancestor"):
        mirror._validate_root_directory(logical_source, label="unbound logical source")
    result = controller.run_once()
    binding = controller.state.payload["identity"]["h100_endpoint"][
        "source_root_binding"
    ]
    assert binding["logical_path"] == str(logical_source)
    assert binding["resolved_path"] == str(actual_source)
    assert binding["symlink_ancestors"] == [
        {"path": str(logical_mount), "target": str(actual_mount)}
    ]
    assert result["recoverable_steps"] == [10]
    assert mirror.verify_restore_index_at_root(
        columbus, run_id=RUN_ID, metric_name=METRIC, metric_mode=MODE
    )["checkpoint_steps"] == [10]

    replacement_mount = tmp_path / "replacement-vla-jepa"
    replacement_mount.mkdir()
    shutil.copytree(actual_source, replacement_mount / RUN_ID)
    logical_mount.unlink()
    logical_mount.symlink_to(replacement_mount, target_is_directory=True)
    with pytest.raises(mirror.MirrorError, match="identity/configuration drift"):
        controller.run_once()


def test_recovery_receipt_crash_reuses_pinned_evidence_after_summary_advance(
    tmp_path, monkeypatch
):
    controller, source, local, columbus = _fake_controller(tmp_path, monkeypatch)
    real_mark_recoverable = mirror.MirrorState.mark_recoverable
    crashed = False

    def crash_after_receipt(state, step, closure, **kwargs):
        nonlocal crashed
        if not crashed:
            crashed = True
            raise RuntimeError("synthetic crash after recovery receipt")
        return real_mark_recoverable(state, step, closure, **kwargs)

    monkeypatch.setattr(mirror.MirrorState, "mark_recoverable", crash_after_receipt)
    with pytest.raises(RuntimeError, match="synthetic crash after recovery receipt"):
        controller.run_once()
    record = controller.state.payload["checkpoints"]["10"]
    intent = record["recovery_intent"]
    pinned_digest = intent["evidence_identity"]["manifest"]["manifest_sha256"]
    receipt = local / "manifests/receipts/checkpoint-steps_10.json"
    assert receipt.is_file()
    assert record["recoverable"] is False

    summary = source / "summary.jsonl"
    summary.write_bytes(summary.read_bytes() + b"{\"step\":11}\n")
    monkeypatch.setattr(
        mirror.MirrorState, "mark_recoverable", real_mark_recoverable
    )
    result = controller.run_once()
    record = controller.state.payload["checkpoints"]["10"]
    current_digest = controller.state.payload["source_evidence"]["identity"][
        "manifest"
    ]["manifest_sha256"]
    assert result["recoverable_steps"] == [10]
    assert current_digest != pinned_digest
    assert record["evidence_manifest_sha256"] == pinned_digest
    assert mirror.verify_restore_index_at_root(
        columbus, run_id=RUN_ID, metric_name=METRIC, metric_mode=MODE
    )["checkpoint_steps"] == [10]


@pytest.mark.parametrize("improves", [False, True])
def test_copy_complete_crash_recovers_after_trainer_removes_source_checkpoint(
    tmp_path, monkeypatch, improves
):
    controller, source, _local, columbus = _fake_controller(tmp_path, monkeypatch)
    controller.run_once()
    best_step = 20 if improves else 10
    best_value = 0.20 if improves else 0.25
    eval_value = 0.20 if improves else 0.30
    manifest = _manifest(
        source, 20, best_step=best_step, best_value=best_value
    )
    _eval_artifact(source, manifest, value=eval_value)
    _pointer(source, best_step, best_value)
    real_persist_receipt = controller._persist_recovery_receipt
    crashed = False

    def crash_after_dual_copy(**kwargs):
        nonlocal crashed
        if kwargs["step"] == 20 and not crashed:
            crashed = True
            raise RuntimeError("synthetic crash after dual checkpoint/eval copy")
        return real_persist_receipt(**kwargs)

    monkeypatch.setattr(
        controller, "_persist_recovery_receipt", crash_after_dual_copy
    )
    with pytest.raises(RuntimeError, match="synthetic crash after dual"):
        controller.run_once()
    record = controller.state.payload["checkpoints"]["20"]
    eval_record = controller.state.payload["eval_artifacts"]["20"]
    assert record["dual_verified"] is True
    assert eval_record["dual_verified"] is True
    assert record["recovery_intent"] is not None
    assert record["recoverable"] is False

    shutil.rmtree(source / "checkpoints/steps_20")
    (source / "heldout_eval_metrics/step_00000020.json").unlink()
    result = controller.run_once()
    assert result["recoverable_steps"] == [10, 20]
    verified = mirror.verify_restore_index_at_root(
        columbus, run_id=RUN_ID, metric_name=METRIC, metric_mode=MODE
    )
    assert verified["checkpoint_steps"] == [10, 20]
    assert verified["best_step"] == best_step


def test_local_partial_copy_finishes_columbus_after_source_removal(
    tmp_path, monkeypatch
):
    controller, source, _local, columbus = _fake_controller(tmp_path, monkeypatch)
    controller.run_once()
    manifest = _manifest(source, 20, best_step=10, best_value=0.25)
    _eval_artifact(source, manifest, value=0.30)
    _pointer(source, 10, 0.25)
    real_rsync = controller.columbus.rsync_from_local
    failed = False

    def fail_first_remote_eval(
        source_path, destination, *, delete=False, heartbeat=None
    ):
        nonlocal failed
        if "heldout_eval_metrics" in str(destination) and not failed:
            failed = True
            raise RuntimeError("synthetic Columbus eval transfer crash")
        return real_rsync(
            source_path,
            destination,
            delete=delete,
            heartbeat=heartbeat,
        )

    monkeypatch.setattr(
        controller.columbus, "rsync_from_local", fail_first_remote_eval
    )
    with pytest.raises(RuntimeError, match="synthetic Columbus eval"):
        controller.run_once()
    record = controller.state.payload["checkpoints"]["20"]
    eval_record = controller.state.payload["eval_artifacts"]["20"]
    assert record["dual_verified"] is True
    assert eval_record["local"] is True and eval_record["columbus"] is False
    assert record["recovery_intent"] is not None

    shutil.rmtree(source / "checkpoints/steps_20")
    (source / "heldout_eval_metrics/step_00000020.json").unlink()
    monkeypatch.setattr(controller.columbus, "rsync_from_local", real_rsync)
    result = controller.run_once()
    assert result["recoverable_steps"] == [10, 20]
    assert mirror.verify_restore_index_at_root(
        columbus, run_id=RUN_ID, metric_name=METRIC, metric_mode=MODE
    )["checkpoint_steps"] == [10, 20]


def test_restore_index_crash_reconciles_before_source_advance(
    tmp_path, monkeypatch
):
    controller, source, local, columbus = _fake_controller(tmp_path, monkeypatch)
    real_persist = controller._persist_immutable_record
    crashed = False

    def crash_after_generation(relative_path, payload):
        nonlocal crashed
        digest = real_persist(relative_path, payload)
        if relative_path.startswith("manifests/restore_indexes/") and not crashed:
            crashed = True
            raise RuntimeError("synthetic crash after restore-index generation")
        return digest

    monkeypatch.setattr(
        controller, "_persist_immutable_record", crash_after_generation
    )
    with pytest.raises(RuntimeError, match="synthetic crash after restore-index"):
        controller.run_once()
    assert controller.state.payload["restore_index_generation"] == 0
    assert controller.state.payload["restore_index_intent"] is not None
    first_generation = local / "manifests/restore_indexes/generation_00000001.json"
    assert first_generation.is_file()
    assert [
        entry["step"] for entry in json.loads(first_generation.read_text())["checkpoints"]
    ] == [10]

    manifest = _manifest(source, 20, best_step=20, best_value=0.20)
    _eval_artifact(source, manifest, value=0.20)
    _pointer(source, 20, 0.20)
    result = controller.run_once()
    assert result["recoverable_steps"] == [10, 20]
    assert controller.state.payload["restore_index_generation"] == 2
    assert controller.state.payload["restore_index_intent"] is None
    current = json.loads((columbus / "restore-index.json").read_text())
    assert current["generation"] == 2
    assert [entry["step"] for entry in current["checkpoints"]] == [10, 20]
    assert mirror.verify_restore_index_at_root(
        columbus, run_id=RUN_ID, metric_name=METRIC, metric_mode=MODE
    )["checkpoint_steps"] == [10, 20]


@pytest.mark.parametrize("improves", [False, True])
def test_checkpoint_or_pointer_publication_gap_is_repaired_on_restart(
    tmp_path, monkeypatch, improves
):
    controller, source, _local, columbus = _fake_controller(tmp_path, monkeypatch)
    controller.run_once()
    best_step = 20 if improves else 10
    best_value = 0.20 if improves else 0.25
    eval_value = 0.20 if improves else 0.30
    manifest = _manifest(
        source, 20, best_step=best_step, best_value=best_value
    )
    _eval_artifact(source, manifest, value=eval_value)
    _pointer(source, best_step, best_value)
    real_publish = controller._publish_restore_index
    crashed = False

    def crash_before_publication(*, excluded_steps=()):
        nonlocal crashed
        if not crashed:
            crashed = True
            raise RuntimeError("synthetic crash before restore-index publication")
        return real_publish(excluded_steps=excluded_steps)

    monkeypatch.setattr(controller, "_publish_restore_index", crash_before_publication)
    with pytest.raises(RuntimeError, match="synthetic crash before restore-index"):
        controller.run_once()
    assert controller.state.payload["checkpoints"]["20"]["recoverable"] is True
    assert controller.state.payload["restore_index_generation"] == 1
    assert [
        entry["step"]
        for entry in json.loads((columbus / "restore-index.json").read_text())[
            "checkpoints"
        ]
    ] == [10]

    result = controller.run_once()
    assert result["recoverable_steps"] == [10, 20]
    assert controller.state.payload["restore_index_generation"] == 2
    verified = mirror.verify_restore_index_at_root(
        columbus, run_id=RUN_ID, metric_name=METRIC, metric_mode=MODE
    )
    assert verified["checkpoint_steps"] == [10, 20]
    assert verified["best_step"] == best_step


def test_stale_index_is_repaired_before_h100_outage(tmp_path, monkeypatch):
    controller, source, _local, columbus = _fake_controller(tmp_path, monkeypatch)
    controller.run_once()
    manifest = _manifest(source, 20, best_step=20, best_value=0.20)
    _eval_artifact(source, manifest, value=0.20)
    _pointer(source, 20, 0.20)
    real_publish = controller._publish_restore_index
    crashed = False

    def crash_before_publication(*, excluded_steps=()):
        nonlocal crashed
        if not crashed:
            crashed = True
            raise RuntimeError("synthetic crash before restore-index publication")
        return real_publish(excluded_steps=excluded_steps)

    monkeypatch.setattr(controller, "_publish_restore_index", crash_before_publication)
    with pytest.raises(RuntimeError, match="synthetic crash before restore-index"):
        controller.run_once()

    def source_outage(endpoint):
        if endpoint.role == "h100":
            raise mirror.MirrorError("synthetic H100 endpoint outage")
        return _identity("columbus")

    monkeypatch.setattr(mirror, "resolve_endpoint_identity", source_outage)
    with pytest.raises(mirror.MirrorError, match="synthetic H100 endpoint outage"):
        controller.run_once()
    verified = mirror.verify_restore_index_at_root(
        columbus, run_id=RUN_ID, metric_name=METRIC, metric_mode=MODE
    )
    assert verified["checkpoint_steps"] == [10, 20]
    assert verified["best_step"] == 20


@pytest.mark.parametrize("initial_checkpoint", [True, False])
def test_pointer_crash_is_reconciled_and_indexed_before_h100_outage(
    tmp_path, monkeypatch, initial_checkpoint
):
    controller, source, _local, columbus = _fake_controller(tmp_path, monkeypatch)
    if not initial_checkpoint:
        controller.run_once()
        manifest = _manifest(source, 20, best_step=20, best_value=0.20)
        _eval_artifact(source, manifest, value=0.20)
        _pointer(source, 20, 0.20)

    real_mirror_pointer = controller._mirror_pointer
    crashed = False

    def crash_at_pointer(*args, **kwargs):
        nonlocal crashed
        if kwargs.get("verify_source") and not crashed:
            crashed = True
            raise RuntimeError("synthetic crash at pointer publication")
        return real_mirror_pointer(*args, **kwargs)

    monkeypatch.setattr(controller, "_mirror_pointer", crash_at_pointer)
    with pytest.raises(RuntimeError, match="pointer publication"):
        controller.run_once()
    expected_step = 10 if initial_checkpoint else 20
    assert controller.state.payload["checkpoints"][str(expected_step)][
        "recoverable"
    ] is True
    assert controller.state.payload["pointer_publication_intent"] is not None
    if initial_checkpoint:
        assert controller.state.payload["restore_index_generation"] == 0
        assert not (columbus / "restore-index.json").exists()
    else:
        assert [
            entry["step"]
            for entry in json.loads((columbus / "restore-index.json").read_text())[
                "checkpoints"
            ]
        ] == [10]

    monkeypatch.setattr(controller, "_mirror_pointer", real_mirror_pointer)

    def source_outage(endpoint):
        if endpoint.role == "h100":
            raise mirror.MirrorError("synthetic H100 endpoint outage")
        return _identity("columbus")

    monkeypatch.setattr(mirror, "resolve_endpoint_identity", source_outage)
    with pytest.raises(mirror.MirrorError, match="synthetic H100 endpoint outage"):
        controller.run_once()
    assert controller.state.payload["pointer_publication_intent"] is None
    verified = mirror.verify_restore_index_at_root(
        columbus, run_id=RUN_ID, metric_name=METRIC, metric_mode=MODE
    )
    assert verified["checkpoint_steps"] == (
        [10] if initial_checkpoint else [10, 20]
    )
    assert verified["best_step"] == expected_step


def test_partial_multistep_loop_indexes_receipt_complete_prefix_before_h100_outage(
    tmp_path, monkeypatch
):
    controller, source, _local, columbus = _fake_controller(tmp_path, monkeypatch)
    controller.run_once()
    manifest20 = _manifest(source, 20, best_step=20, best_value=0.20)
    _eval_artifact(source, manifest20, value=0.20)
    manifest30 = _manifest(source, 30, best_step=30, best_value=0.15)
    _eval_artifact(source, manifest30, value=0.15)
    _pointer(source, 30, 0.15)
    real_mirror_checkpoint = controller._mirror_checkpoint

    def crash_before_step30(step, *args, **kwargs):
        if step == 30:
            raise RuntimeError("synthetic crash before step30 checkpoint copy")
        return real_mirror_checkpoint(step, *args, **kwargs)

    monkeypatch.setattr(controller, "_mirror_checkpoint", crash_before_step30)
    with pytest.raises(RuntimeError, match="step30 checkpoint"):
        controller.run_once()
    assert controller.state.payload["checkpoints"]["20"]["recoverable"] is True
    assert controller.state.payload["checkpoints"]["30"]["recoverable"] is False
    assert controller.state.payload["pointer_publication_intent"] is None
    monkeypatch.setattr(controller, "_mirror_checkpoint", real_mirror_checkpoint)

    def source_outage(endpoint):
        if endpoint.role == "h100":
            raise mirror.MirrorError("synthetic H100 endpoint outage")
        return _identity("columbus")

    monkeypatch.setattr(mirror, "resolve_endpoint_identity", source_outage)
    with pytest.raises(mirror.MirrorError, match="synthetic H100 endpoint outage"):
        controller.run_once()
    verified = mirror.verify_restore_index_at_root(
        columbus, run_id=RUN_ID, metric_name=METRIC, metric_mode=MODE
    )
    assert verified["checkpoint_steps"] == [10, 20]
    assert verified["best_step"] == 20


def test_local_eval_is_pinned_before_checkpoint_dual_copy_source_retention(
    tmp_path, monkeypatch
):
    controller, source, _local, columbus = _fake_controller(tmp_path, monkeypatch)
    controller.run_once()
    manifest = _manifest(source, 20, best_step=20, best_value=0.20)
    _eval_artifact(source, manifest, value=0.20)
    _pointer(source, 20, 0.20)
    real_mirror_eval = controller._mirror_eval
    crashed = False

    def crash_before_remote_eval(step, *args, **kwargs):
        nonlocal crashed
        if step == 20 and not kwargs.get("local_only") and not crashed:
            crashed = True
            checkpoint_record = controller.state.payload["checkpoints"]["20"]
            assert checkpoint_record["dual_verified"] is True
            raise RuntimeError("synthetic crash after checkpoint dual copy")
        return real_mirror_eval(step, *args, **kwargs)

    monkeypatch.setattr(controller, "_mirror_eval", crash_before_remote_eval)
    with pytest.raises(RuntimeError, match="checkpoint dual copy"):
        controller.run_once()
    checkpoint_record = controller.state.payload["checkpoints"]["20"]
    eval_record = controller.state.payload["eval_artifacts"]["20"]
    assert checkpoint_record["dual_verified"] is True
    assert checkpoint_record["recoverable"] is False
    assert eval_record["local"] is True and eval_record["columbus"] is False

    shutil.rmtree(source / "checkpoints/steps_20")
    monkeypatch.setattr(controller, "_mirror_eval", real_mirror_eval)
    result = controller.run_once()
    assert result["recoverable_steps"] == [10, 20]
    assert (source / "heldout_eval_metrics/step_00000020.json").is_file()
    verified = mirror.verify_restore_index_at_root(
        columbus, run_id=RUN_ID, metric_name=METRIC, metric_mode=MODE
    )
    assert verified["checkpoint_steps"] == [10, 20]
    assert verified["best_step"] == 20


def test_retention_gap_before_prune_is_repaired_on_restart(tmp_path, monkeypatch):
    controller, source, _local, columbus = _fake_controller(
        tmp_path, monkeypatch, retain=2
    )
    controller.run_once()
    manifest = _manifest(source, 20, best_step=20, best_value=0.20)
    _eval_artifact(source, manifest, value=0.20)
    _pointer(source, 20, 0.20)
    controller.run_once()
    manifest = _manifest(source, 30, best_step=30, best_value=0.15)
    _eval_artifact(source, manifest, value=0.15)
    _pointer(source, 30, 0.15)
    real_prune = controller._prune
    crashed = False

    def crash_before_prune(manifests, best_step):
        nonlocal crashed
        if not crashed:
            crashed = True
            raise RuntimeError("synthetic crash before prune intent")
        return real_prune(manifests, best_step)

    monkeypatch.setattr(controller, "_prune", crash_before_prune)
    with pytest.raises(RuntimeError, match="synthetic crash before prune"):
        controller.run_once()
    assert controller.state.payload["prune_pending"] == {}
    assert [
        entry["step"]
        for entry in json.loads((columbus / "restore-index.json").read_text())[
            "checkpoints"
        ]
    ] == [10, 20, 30]

    result = controller.run_once()
    assert result["recoverable_steps"] == [20, 30]
    assert controller.state.payload["prune_pending"]["10"]["phase"] == "complete"
    assert mirror.verify_restore_index_at_root(
        columbus, run_id=RUN_ID, metric_name=METRIC, metric_mode=MODE
    )["checkpoint_steps"] == [20, 30]
    steady = controller.run_once()
    assert steady["status"] == "healthy"
    assert steady["backlog_count"] == 0
    assert not (controller.config.local_root / "checkpoints/steps_10").exists()
    assert not (columbus / "checkpoints/steps_10").exists()


def test_end_to_end_mirror_stat_only_second_poll_restore_and_corruption_refusal(
    tmp_path, monkeypatch
):
    controller, source, local, columbus = _fake_controller(tmp_path, monkeypatch)
    first = controller.run_once()
    assert first["recoverable_steps"] == [10]
    assert mirror.verify_restore_index_at_root(
        columbus, run_id=RUN_ID, metric_name=METRIC, metric_mode=MODE
    )["checkpoint_steps"] == [10]

    hashes = 0
    real_hash = mirror._sha256_file

    def count_hash(path):
        nonlocal hashes
        hashes += 1
        return real_hash(path)

    monkeypatch.setattr(mirror, "_sha256_file", count_hash)
    second = controller.run_once()
    assert second["recoverable_steps"] == [10]
    assert hashes == 0, "an unchanged one-minute poll must hash zero artifact bytes"
    monkeypatch.setattr(mirror, "_sha256_file", real_hash)

    shutil.rmtree(source)
    shutil.rmtree(local)
    shutil.rmtree(controller.config.state_dir)
    restored = tmp_path / "restored" / RUN_ID
    result = mirror.restore_from_local_root(
        columbus,
        restored,
        run_id=RUN_ID,
        metric_name=METRIC,
        metric_mode=MODE,
    )
    assert result["verified"] is True
    assert (restored / "checkpoints/steps_10/model.safetensors").read_bytes() == (
        columbus / "checkpoints/steps_10/model.safetensors"
    ).read_bytes()

    model = columbus / "checkpoints/steps_10/model.safetensors"
    data = bytearray(model.read_bytes())
    data[0] ^= 1
    model.write_bytes(data)
    with pytest.raises(mirror.MirrorError, match="verification|manifest|bytes"):
        mirror.verify_restore_index_at_root(
            columbus, run_id=RUN_ID, metric_name=METRIC, metric_mode=MODE
        )


def test_scheduled_scrub_detects_same_size_tampering(tmp_path, monkeypatch):
    controller, _source, local, _columbus = _fake_controller(tmp_path, monkeypatch)
    controller.run_once()
    model = local / "checkpoints/steps_10/model.safetensors"
    data = bytearray(model.read_bytes())
    data[-1] ^= 1
    model.write_bytes(data)
    with pytest.raises(mirror.MirrorError, match="conflict|scrub|bytes"):
        controller.run_once(force_full_scrub=True)


def test_pending_latest_does_not_block_finalized_older_and_newer_step_escalates(
    tmp_path, monkeypatch
):
    controller, source, _local, _columbus = _fake_controller(tmp_path, monkeypatch)
    pending = _checkpoint(source, 20, best_step=10, best_value=0.25)
    (pending / "optimizer.bin").unlink()
    result = controller.run_once()
    assert result["status"] == "healthy_pending"
    assert result["recoverable_steps"] == [10]
    assert result["pending_steps"] == [20]

    _checkpoint(source, 30, best_step=10, best_value=0.25)
    with pytest.raises(mirror.MirrorError, match="non-latest checkpoint"):
        controller.run_once()


def test_pending_valid_checkpoint_is_cached_and_second_poll_hashes_zero_bytes(
    tmp_path, monkeypatch
):
    controller, source, _local, _columbus = _fake_controller(tmp_path, monkeypatch)
    _checkpoint(source, 20, best_step=10, best_value=0.25)
    first = controller.run_once()
    assert first["status"] == "healthy_pending"
    assert first["pending_steps"] == [20]
    pending_record = controller.state.payload["checkpoints"]["20"]
    assert pending_record["manifest"]["step"] == 20
    assert pending_record["recoverable"] is False

    hashes = 0
    real_hash = mirror._sha256_file

    def count_hash(path):
        nonlocal hashes
        hashes += 1
        return real_hash(path)

    monkeypatch.setattr(mirror, "_sha256_file", count_hash)
    second = controller.run_once()
    assert second["status"] == "healthy_pending"
    assert second["pending_steps"] == [20]
    assert hashes == 0


def test_step_zero_live_model_baseline_without_checkpoint_is_authenticated_but_excluded(
    tmp_path, monkeypatch
):
    controller, source, _local, columbus = _fake_controller(tmp_path, monkeypatch)
    baseline_path = _baseline_eval_artifact(
        source, value=0.9688526547054578
    )
    production_shape = json.loads(baseline_path.read_text())
    assert set(production_shape) == {
        "checkpoint",
        "checkpoint_relative_path",
        "checkpoint_selection_eligible",
        "checkpoint_step",
        "metrics",
        "production_valid",
        "run",
        "sampling_reports",
        "schema_version",
        "selection_metric",
    }
    assert production_shape["checkpoint"] == {
        "source_kind": "live_in_memory_model",
        "source_path": None,
        "step": 0,
    }
    assert production_shape["checkpoint_relative_path"] is None
    assert production_shape["checkpoint_step"] == 0
    assert production_shape["production_valid"] is True
    assert production_shape["checkpoint_selection_eligible"] is True
    assert production_shape["selection_metric"] == {
        "eligible": True,
        "mode": "min",
        "name": METRIC,
        "value": 0.9688526547054578,
    }
    assert not (source / "checkpoints/steps_0").exists()
    result = controller.run_once()
    assert result["status"] == "healthy"
    baseline = mirror.EvalArtifact.from_dict(
        controller.state.payload["eval_artifacts"]["0"]["artifact"]
    )
    assert baseline.source_kind == "live_in_memory_model"
    assert baseline.cryptographically_bound is True
    assert baseline.production_eligible is False
    assert result["recoverable_steps"] == [10]
    assert not (columbus / "heldout_eval_metrics/step_00000000.json").exists()

    changed = json.loads(baseline_path.read_text())
    changed["selection_metric"]["value"] = 0.39
    changed["metrics"]["focused"][METRIC] = 0.39
    baseline_path.write_text(json.dumps(changed))
    with pytest.raises(mirror.MirrorError, match="changed after authentication"):
        controller.run_once()


@pytest.mark.parametrize("artifact", ["checkpoint", "eval"])
def test_newest_complete_shaped_midwrite_json_is_pending_but_older_recovers(
    tmp_path, monkeypatch, artifact
):
    controller, source, _local, _columbus = _fake_controller(tmp_path, monkeypatch)
    manifest = _manifest(source, 20, best_step=10, best_value=0.25)
    if artifact == "checkpoint":
        (source / "checkpoints/steps_20/trainer_state.json").write_text("{")
    else:
        eval_path, _ = _eval_artifact(source, manifest, value=0.30)
        eval_path.write_text("{")
    result = controller.run_once()
    assert result["status"] == "healthy_pending"
    assert result["recoverable_steps"] == [10]
    assert result["pending_steps"] == [20]


def test_midwrite_checkpoint_metadata_never_rehashes_large_candidate_bytes(
    tmp_path, monkeypatch
):
    controller, source, _local, _columbus = _fake_controller(tmp_path, monkeypatch)
    controller.run_once()
    checkpoint = _checkpoint(source, 20, best_step=10, best_value=0.25)
    (checkpoint / "trainer_state.json").write_text("{")
    hashes = 0
    real_hash = mirror._sha256_file

    def count_hash(path):
        nonlocal hashes
        hashes += 1
        return real_hash(path)

    monkeypatch.setattr(mirror, "_sha256_file", count_hash)
    first = controller.run_once()
    second = controller.run_once()
    assert first["status"] == second["status"] == "healthy_pending"
    assert hashes == 0


def test_newest_midwrite_pointer_is_pending_and_reuses_prior_authenticated_pointer(
    tmp_path, monkeypatch
):
    controller, source, _local, _columbus = _fake_controller(tmp_path, monkeypatch)
    controller.run_once()
    manifest = _manifest(source, 20, best_step=10, best_value=0.25)
    _eval_artifact(source, manifest, value=0.30)
    (source / "best_checkpoint.json").write_text("{")
    result = controller.run_once()
    assert result["status"] == "healthy_pending"
    assert result["recoverable_steps"] == [10]
    assert result["pending_steps"] == [20]


def test_previously_finalized_latest_parse_failure_is_not_downgraded_to_pending(
    tmp_path, monkeypatch
):
    controller, source, _local, _columbus = _fake_controller(tmp_path, monkeypatch)
    controller.run_once()
    (source / "checkpoints/steps_10/trainer_state.json").write_text("{")
    with pytest.raises(mirror.MirrorError, match="canonical JSON|validation"):
        controller.run_once()


def test_endpoint_identity_is_reresolved_and_drift_refused_each_poll(
    tmp_path, monkeypatch
):
    controller, _source, _local, _columbus = _fake_controller(tmp_path, monkeypatch)
    controller.run_once()

    def drifted(endpoint):
        return _identity("replacement") if endpoint.role == "h100" else _identity("columbus")

    monkeypatch.setattr(mirror, "resolve_endpoint_identity", drifted)
    with pytest.raises(mirror.MirrorError, match="identity/configuration drift"):
        controller.run_once()


def test_missing_scrub_baseline_is_due_after_verified_artifacts(tmp_path, monkeypatch):
    controller, _source, local, _columbus = _fake_controller(tmp_path, monkeypatch)
    controller.run_once()
    controller.state.payload["last_full_scrub_at"] = None
    controller.state.save()
    model = local / "checkpoints/steps_10/model.safetensors"
    damaged = bytearray(model.read_bytes())
    damaged[0] ^= 1
    model.write_bytes(damaged)
    with pytest.raises(mirror.MirrorError, match="conflict|scrub|bytes"):
        controller.run_once()


def test_missing_scrub_baseline_covers_evidence_only_crash_state(
    tmp_path, monkeypatch
):
    controller, _source, local, _columbus = _fake_controller(tmp_path, monkeypatch)
    real_mirror_eval = controller._mirror_eval

    def crash_before_first_eval(*args, **kwargs):
        if kwargs.get("local_only"):
            raise RuntimeError("synthetic crash after evidence-only dual verification")
        return real_mirror_eval(*args, **kwargs)

    monkeypatch.setattr(controller, "_mirror_eval", crash_before_first_eval)
    with pytest.raises(RuntimeError, match="evidence-only"):
        controller.run_once()
    assert controller.state.payload["last_full_scrub_at"] is None
    evidence_records = controller.state.payload["evidence_snapshots"]
    assert len(evidence_records) == 1
    digest, evidence_record = next(iter(evidence_records.items()))
    assert evidence_record["dual_verified"] is True
    assert not controller.state.payload["eval_artifacts"]["10"].get("local")

    config = local / "evidence" / "snapshots" / f"sha256-{digest}" / "config.yaml"
    damaged = bytearray(config.read_bytes())
    damaged[0] ^= 1
    config.write_bytes(damaged)
    monkeypatch.setattr(controller, "_mirror_eval", real_mirror_eval)
    assert controller._full_scrub_due() is True
    with pytest.raises(mirror.MirrorError, match="evidence|conflict|bytes"):
        controller.run_once()
    assert controller.state.payload["last_full_scrub_at"] is None


def test_partial_evidence_copy_reconciles_before_advanced_source_scrub(
    tmp_path, monkeypatch
):
    controller, source, local, columbus = _fake_controller(tmp_path, monkeypatch)
    real_rsync = controller.columbus.rsync_from_local
    crashed = False

    def crash_first_evidence_copy(
        source_path, destination, *, delete=False, heartbeat=None
    ):
        nonlocal crashed
        if "evidence/snapshots" in str(destination) and not crashed:
            crashed = True
            raise RuntimeError("synthetic crash during Columbus evidence copy")
        return real_rsync(
            source_path,
            destination,
            delete=delete,
            heartbeat=heartbeat,
        )

    monkeypatch.setattr(
        controller.columbus, "rsync_from_local", crash_first_evidence_copy
    )
    with pytest.raises(RuntimeError, match="Columbus evidence copy"):
        controller.run_once()
    assert controller.state.payload["last_full_scrub_at"] is None
    old_digest, old_record = next(
        iter(controller.state.payload["evidence_snapshots"].items())
    )
    assert old_record["local"] is True
    assert old_record["columbus"] is False
    assert old_record["dual_verified"] is False
    assert (
        local / "evidence" / "snapshots" / f"sha256-{old_digest}"
    ).is_dir()

    summary = source / "summary.jsonl"
    summary.write_bytes(summary.read_bytes() + b'{"step":20}\n')
    monkeypatch.setattr(controller.columbus, "rsync_from_local", real_rsync)
    result = controller.run_once()
    evidence_records = controller.state.payload["evidence_snapshots"]
    assert len(evidence_records) == 2
    assert all(record["dual_verified"] is True for record in evidence_records.values())
    assert (
        columbus / "evidence" / "snapshots" / f"sha256-{old_digest}"
    ).is_dir()
    assert result["recoverable_steps"] == [10]
    assert controller.state.payload["last_full_scrub_at"] is not None


@pytest.mark.parametrize(
    "published_record",
    ["receipt", "checkpoint_manifest", "eval_manifest", "evidence_manifest"],
)
def test_full_scrub_authenticates_all_published_recovery_records(
    tmp_path, monkeypatch, published_record
):
    controller, _source, _local, columbus = _fake_controller(tmp_path, monkeypatch)
    controller.run_once()
    old_scrub_at = controller.state.payload["last_full_scrub_at"]
    evidence_digest = controller.state.payload["checkpoints"]["10"][
        "evidence_manifest_sha256"
    ]
    relative_paths = {
        "receipt": "manifests/receipts/checkpoint-steps_10.json",
        "checkpoint_manifest": "manifests/checkpoints/steps_10.json",
        "eval_manifest": "manifests/evals/step_00000010.json",
        "evidence_manifest": (
            f"manifests/evidence/sha256-{evidence_digest}.json"
        ),
    }
    target = columbus / relative_paths[published_record]
    original = target.read_bytes()
    replacement_run_id = ("x" + RUN_ID[1:]).encode()
    damaged = original.replace(RUN_ID.encode(), replacement_run_id, 1)
    assert damaged != original and len(damaged) == len(original)
    target.write_bytes(damaged)

    with pytest.raises(mirror.MirrorError, match="receipt|manifest|scrub|SHA-256"):
        controller.run_once(force_full_scrub=True)
    assert controller.state.payload["last_full_scrub_at"] == old_scrub_at


def test_vanished_pending_checkpoint_remains_tombstoned_and_blocks_newer(
    tmp_path, monkeypatch
):
    controller, source, _local, _columbus = _fake_controller(tmp_path, monkeypatch)
    pending = _checkpoint(source, 20, best_step=10, best_value=0.25)
    (pending / "optimizer.bin").unlink()
    first = controller.run_once()
    assert first["pending_steps"] == [20]

    shutil.rmtree(pending)
    vanished = controller.run_once()
    assert vanished["status"] == "healthy_pending"
    assert vanished["pending_steps"] == [20]
    tombstone = controller.state.payload["pending_candidates"]["20"]
    assert tombstone["reason"] == "pending checkpoint disappeared before authentication"
    assert "tombstoned_at" in tombstone

    manifest = _manifest(source, 30, best_step=10, best_value=0.25)
    _eval_artifact(source, manifest, value=0.30)
    _pointer(source, 10, 0.25)
    with pytest.raises(mirror.MirrorError, match="newer checkpoint appeared"):
        controller.run_once()


def test_require_contained_rejects_lexical_dotdot_even_when_prefix_matches(tmp_path):
    root = tmp_path / "protected"
    escaped = root / "checkpoints" / ".." / "outside" / ".incoming-stage"
    with pytest.raises(mirror.MirrorError, match="dot segments|canonical"):
        mirror._require_contained(escaped, root, label="adversarial stage")


def test_only_latest_checkpoint_may_be_pending(tmp_path):
    run_dir = tmp_path / RUN_ID
    first = _checkpoint(run_dir, 10)
    _checkpoint(run_dir, 20)
    (first / "optimizer.bin").unlink()
    inventory = mirror.lightweight_source_inventory(run_dir)
    steps = sorted(mirror.checkpoint_step_from_name(name) for name in inventory["checkpoints"])
    incomplete = [
        step
        for step in steps
        if not mirror._checkpoint_light_complete(
            inventory["checkpoints"][f"steps_{step}"]
        )[0]
    ]
    assert incomplete == [10] and incomplete[0] != max(steps)


def test_prune_trash_state_machine_helpers_are_idempotent(tmp_path, monkeypatch):
    relay = tmp_path / "relay"
    relay.mkdir(mode=0o700)
    base = relay / "checkpoint-mirror-storage"
    monkeypatch.setattr(mirror, "SAFE_COLUMBUS_PARENT", relay)
    monkeypatch.setattr(mirror, "SAFE_COLUMBUS_BASE", base)
    root = base / RUN_ID
    mirror._ensure_columbus_root(RUN_ID, root)
    path = _checkpoint(root, 10)
    manifest = mirror.validate_checkpoint_tree(path, metric_name=METRIC, metric_mode=MODE)
    trash = path.parent / f".trash-steps_10-{manifest.manifest_sha256[:16]}"
    first = mirror._internal_trash_checkpoint(
        RUN_ID, root, path, trash, manifest.manifest_sha256, METRIC, MODE
    )
    second = mirror._internal_trash_checkpoint(
        RUN_ID, root, path, trash, manifest.manifest_sha256, METRIC, MODE
    )
    assert first["trashed"] and second["trashed"]
    mirror._internal_delete_trash(RUN_ID, root, trash)
    mirror._internal_delete_trash(RUN_ID, root, trash)
    assert not trash.exists()


def test_crash_pending_prune_reconciles_before_source_outage(
    tmp_path, monkeypatch
):
    controller, source, local, columbus = _fake_controller(tmp_path, monkeypatch)
    controller.run_once()
    manifest = _manifest(source, 20, best_step=20, best_value=0.20)
    _eval_artifact(source, manifest, value=0.20)
    _pointer(source, 20, 0.20)
    controller.run_once()

    record = controller.state.payload["checkpoints"]["10"]
    record["recoverable"] = False
    controller.state.payload["prune_pending"]["10"] = {
        "step": 10,
        "phase": "planned",
        "planned_at": mirror.utc_now(),
        "manifest_sha256": record["manifest_sha256"],
        "index_generation": controller.state.payload["restore_index_generation"] + 1,
    }
    controller.state.save()

    def endpoint_outage(endpoint):
        if endpoint.role == "h100":
            raise mirror.MirrorError("synthetic H100 endpoint outage")
        return _identity("columbus")

    monkeypatch.setattr(mirror, "resolve_endpoint_identity", endpoint_outage)
    with pytest.raises(mirror.MirrorError, match="synthetic H100 endpoint outage"):
        controller.run_once()
    assert controller.state.payload["prune_pending"]["10"]["phase"] == "complete"
    assert not (local / "checkpoints/steps_10").exists()
    assert not (columbus / "checkpoints/steps_10").exists()


def test_restore_copies_only_authenticated_indexed_files(tmp_path, monkeypatch):
    controller, _source, _local, columbus = _fake_controller(tmp_path, monkeypatch)
    controller.run_once()
    extras = {
        "checkpoints/.trash-steps_999-deadbeefdeadbeef/payload.bin": b"trash",
        "evidence/.incoming-orphan/payload.bin": b"stage",
        "best_checkpoint_history/unindexed.json": b"{}\n",
        "manifests/restore_indexes/generation_99999999.json": b"{}\n",
        "manifests/arbitrary.json": b"{}\n",
    }
    for relative, content in extras.items():
        path = columbus / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    # Unindexed regular trees do not invalidate the indexed Columbus closure,
    # but they must never cross the restore boundary.
    assert mirror.verify_restore_index_at_root(
        columbus, run_id=RUN_ID, metric_name=METRIC, metric_mode=MODE
    )["verified"]
    restored = tmp_path / "indexed-restore" / RUN_ID
    mirror.restore_from_local_root(
        columbus,
        restored,
        run_id=RUN_ID,
        metric_name=METRIC,
        metric_mode=MODE,
    )
    plan = mirror.authenticated_restore_files(columbus)
    mirror._assert_exact_restore_files(restored, plan)
    for relative in extras:
        assert not (restored / relative).exists()


@pytest.mark.parametrize("record_kind", ["receipt", "checkpoint", "eval", "evidence"])
def test_restore_envelope_schema_versions_are_exact_after_hash_chain_resealed(
    tmp_path, monkeypatch, record_kind
):
    controller, _source, _local, columbus = _fake_controller(tmp_path, monkeypatch)
    controller.run_once()
    index_path = columbus / "restore-index.json"
    index = json.loads(index_path.read_text())
    entry = index["checkpoints"][0]
    receipt_path = columbus / entry["receipt_relative_path"]
    receipt = json.loads(receipt_path.read_text())
    if record_kind == "receipt":
        receipt["schema_version"] = 999
    else:
        record_path = columbus / receipt[record_kind]["manifest_record_path"]
        record = json.loads(record_path.read_text())
        record["schema_version"] = 999
        receipt[record_kind]["manifest_record_sha256"] = _write_canonical_json(
            record_path, record
        )
    entry["receipt_sha256"] = _write_canonical_json(receipt_path, receipt)
    _rewrite_restore_indexes(columbus, index)
    with pytest.raises(mirror.MirrorError, match="schema_version"):
        mirror.verify_restore_index_at_root(
            columbus, run_id=RUN_ID, metric_name=METRIC, metric_mode=MODE
        )


def test_columbus_verifier_reconstructs_evidence_semantics_from_snapshot_bytes(
    tmp_path, monkeypatch
):
    controller, _source, _local, columbus = _fake_controller(tmp_path, monkeypatch)
    controller.run_once()
    index = json.loads((columbus / "restore-index.json").read_text())
    entry = index["checkpoints"][0]
    receipt_path = columbus / entry["receipt_relative_path"]
    receipt = json.loads(receipt_path.read_text())

    evidence_record_path = columbus / receipt["evidence"]["manifest_record_path"]
    evidence_record = json.loads(evidence_record_path.read_text())
    evidence_record["identity"]["seed"] = 99
    receipt["evidence"]["manifest_record_sha256"] = _write_canonical_json(
        evidence_record_path, evidence_record
    )

    eval_path = columbus / entry["eval_relative_path"]
    eval_payload = json.loads(eval_path.read_text())
    eval_payload["run"]["seed"] = 99
    eval_sha = _write_canonical_json(eval_path, eval_payload)
    entry["eval_sha256"] = eval_sha
    receipt["eval"]["sha256"] = eval_sha

    eval_record_path = columbus / receipt["eval"]["manifest_record_path"]
    eval_record = json.loads(eval_record_path.read_text())
    eval_record["artifact"]["sha256"] = eval_sha
    eval_record["artifact"]["size"] = eval_path.stat().st_size
    receipt["eval"]["manifest_record_sha256"] = _write_canonical_json(
        eval_record_path, eval_record
    )
    entry["receipt_sha256"] = _write_canonical_json(receipt_path, receipt)
    _rewrite_restore_indexes(columbus, index)

    with pytest.raises(mirror.MirrorError, match="snapshot bytes"):
        mirror.verify_restore_index_at_root(
            columbus, run_id=RUN_ID, metric_name=METRIC, metric_mode=MODE
        )


def test_preflight_columbus_probe_is_strictly_read_only(tmp_path, monkeypatch):
    relay = tmp_path / "relay"
    relay.mkdir(mode=0o700)
    base = relay / "checkpoint-mirror-storage"
    monkeypatch.setattr(mirror, "SAFE_COLUMBUS_PARENT", relay)
    monkeypatch.setattr(mirror, "SAFE_COLUMBUS_BASE", base)
    root = base / RUN_ID
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    report = mirror._internal_preflight_columbus(RUN_ID, root)
    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert report["paths"]["run_root"]["exists"] is False
    assert before == after


def test_rsync_protect_args_support_is_probed_when_help_uses_new_name(monkeypatch):
    rsync_path = "/usr/bin/rsync"
    monkeypatch.setattr(mirror.shutil, "which", lambda name: rsync_path)

    def fake_run(args, **_kwargs):
        assert args[0] == rsync_path
        if args[1:] == ("--version",):
            output = "rsync  version 3.2.7  protocol version 31\n"
            return mirror.subprocess.CompletedProcess(args, 0, stdout=output)
        if args[1:] == ("--help",):
            output = "--fsync --secluded-args --partial-dir\n"
            return mirror.subprocess.CompletedProcess(args, 0, stdout=output)
        if args[1:] == ("--protect-args", "--version"):
            output = "rsync  version 3.2.7  protocol version 31\n"
            return mirror.subprocess.CompletedProcess(args, 0, stdout=output)
        raise AssertionError(f"unexpected rsync probe: {args}")

    monkeypatch.setattr(mirror.subprocess, "run", fake_run)
    report = mirror._tool_compatibility_report()

    assert report["rsync_path"] == rsync_path
    assert report["rsync_required_options"] == {
        "--fsync": True,
        "--protect-args": True,
        "--partial-dir": True,
    }
