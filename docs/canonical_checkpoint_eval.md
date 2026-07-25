# Canonical checkpoint evaluation

Canonical/streaming training uses a compact heldout check. It is intended to
answer three questions:

1. Are exact examples excluded from training and normalization statistics?
2. Are the action targets finite and structurally valid?
3. Is prediction error on those same examples improving?

It does not estimate robot task success. It does not compute the RealMan
hold-position composite, gripper-transition score, or subtask score.

## Configuration

Canonical configs name an immutable manifest outside the fresh run directory
and exclude every selected episode from training:

```yaml
canonical_eval_split_id: canonical_full_gcs_heldout_v1
datasets:
  vla_data:
    canonical_eval_manifest: ${run_root_dir}/eval_manifests/${canonical_eval_split_id}.json
    canonical_exclude_eval_episodes_from_training: true
    holdout_sampling:
      algorithm: dataset_fraction_divisor_v1
      minimum_episode_fraction: 0.05
      maximum_episode_fraction: 0.08
      episode_count_multiple: 8
      max_episode_count: 128
      evaluation_observation_count: 128
    eval_per_device_batch_size: 16
    canonical_eval_selection_seed: ${seed}
    canonical_eval_candidate_count: 32
    canonical_eval_min_episodes_per_shard: 2
```

The split ID is deliberately independent of `run_id`: launchers may timestamp
the run ID, but that must not redirect training to a different or nonexistent
holdout artifact.

Prepare it explicitly before `check` or `start`:

```bash
python scripts/generate_canonical_eval_manifest.py \
  --config scripts/config/vlajepa_robot_ft_canonical_full_a100x8_qwen_full_zero3_moge_vits.yaml \
  --world-size 8
```

The generator reads canonical metadata/action sidecars without decoding videos.
It deterministically ranks distinct episodes, holds out about 5–8% of the
catalog in a multiple of eight (capped at 128 episodes), and distributes
exactly 128 H=50 evaluation windows as evenly as possible over those episodes.
For a non-zero division remainder, a content-hash-ranked subset receives one
additional window. Repeating the command verifies and reuses a semantically
identical file.
If the catalog, filters, seed, dimensions, or selected windows drift, it
refuses to overwrite the existing split. `check` and `start` should remain
read-only and fail if this prepare step was not completed.

The generator also leaves at least one training episode in every selected data
shard, because canonical q01/q99 values are train-derived per shard. To
intentionally replace a drifted split for a new run, remove or relocate that
explicit manifest artifact during `prepare`, choose a new
`canonical_eval_split_id`, and normally use a new `run_id` too; the generator
never overwrites it.

For bounded smoke profiles, the loader honors `max_windows` and
`max_windows_per_dataset` only after it has indexed two valid episodes from a
selected shard. Thus a long first episode cannot consume the cap and make the
shard impossible to split. It stops immediately after the second episode when
the window cap is already met; this does not scan or download the full shard.

The manifest window count is independent of the effective training batch. It
must equal `holdout_sampling.evaluation_observation_count` and be divisible by
the distributed evaluation batch:

```text
evaluation_observation_count %
    (eval_per_device_batch_size * world_size) == 0
```

Schema:

```json
{
  "schema_version": 1,
  "purpose": "heldout",
  "source_manifest_sha256": "<sha256 of dataset_manifests.jsonl.gz>",
  "selection": {
    "algorithm": "sha256_episode_rank_dense_window_v1",
    "seed": 42,
    "window_count": 128,
    "holdout_episode_count": 56,
    "base_frames_per_episode": 2,
    "extra_window_episode_count": 16,
    "maximum_frames_per_episode": 3,
    "window_allocation_algorithm": "balanced_digest_rank_v1",
    "extra_window_episode_identities": [
      ["organization/dataset", "source-id", "revision", "data.parquet", 12]
    ],
    "holdout_sampling_policy": {
      "algorithm": "dataset_fraction_divisor_v1",
      "minimum_episode_fraction": 0.05,
      "maximum_episode_fraction": 0.08,
      "episode_count_multiple": 8,
      "max_episode_count": 128,
      "evaluation_observation_count": 128
    },
    "holdout_sampling_plan": {
      "...": "full derived policy result"
    },
    "candidate_count": 32,
    "action_horizon": 50,
    "action_dim": 49,
    "action_type": "joint_delta_gripper_absolute",
    "normalization": "shard_q01_q99_unclipped",
    "adapter_contract_sha256": "<adapter definitions + projection-code sha256>",
    "action_sidecar_variant": "<16-character contract hash>",
    "configured_episode_count": 1000,
    "configured_episode_catalog_sha256": "<episode-catalog sha256>"
  },
  "windows": [
    {
      "dataset_id": "organization/dataset",
      "sid": "source-id",
      "revision": "revision",
      "data_file": "data/chunk-000/file-000.parquet",
      "episode_index": 12,
      "base_index": 40
    }
  ]
}
```

The loader and H100 launcher fail closed if the sampling policy, derived
episode/window allocation, selection seed, candidate count, evaluation window
count, action horizon, action dimension, action
representation, normalization, sidecar contract, or configured episode
catalog no longer match. Startup also fails if a window is missing,
duplicated, outside its episode, not in the configured stream, or has no
supervised action elements. The selected episodes are removed from the
training window index and from per-shard q01/q99 statistics.

The adapter contract hash is content-based and portable across machines. It
binds all adapter JSON/YAML files, the adapter manifest, external
`adapters.py`/`unified_schema.py`, and the local canonical projection/cache
implementation. Changing any of those bytes produces a new sidecar directory
and makes an older eval manifest fail before training.

Canonical checkpoint selection must use an error metric emitted by this
evaluator:

```yaml
trainer:
  best_metric_name: heldout_eval_normalized_action_mae
  best_metric_mode: min
```

Trainer startup rejects a stale or impossible canonical best-metric name
instead of running evaluation without updating the best-checkpoint pointer.

## Logged signals

The canonical view logs:

- normalized action MAE and RMSE;
- H10 and H50 MAE for all actions, joint arms, and hands;
- target and prediction normalized mean-absolute value and RMS;
- valid action fraction and exact valid element/observation counts.

NaN or infinity in any supervised target or prediction fails evaluation.
`heldout_eval_windows.json` records the exact window hash, train/holdout set
hashes, train-statistics hash, per-channel mask coverage, and source-manifest
binding.
