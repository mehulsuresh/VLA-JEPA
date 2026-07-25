# RealMan 18-D three-stage curriculum

This production workflow trains one policy sequentially:

1. one exhaustive pass over the task-balanced 10% RealSource view;
2. two exhaustive passes over the Magna intervention view;
3. four exhaustive passes over the high-quality Magna view.

All stages use the same 18-D state/action contract, H=50 chunk-start deltas,
absolute grippers, union OpenPI q01/q99 statistics, and 70% sample-local
subtask prompting. Base and lift are outside the model. Stage handoffs use the
authenticated natural-final checkpoint from the preceding stage; optimizer
and scheduler state intentionally restart for the next stage.

## Why this sequence

The ordering follows published robot-policy results; it does not require a
new pre-training experiment:

- [π0](https://www.physicalintelligence.company/download/pi0.pdf) separates
  diverse pre-training from high-quality post-training. Its ablations report
  that the combined recipe performs best, that hard tasks benefit especially
  from pre-training, and that high-quality-only training is brittle when the
  policy must recover from mistakes.
- [Octo](https://octo-models.github.io/) likewise uses diverse robot-policy
  pre-training followed by target-domain fine-tuning. Across its six
  fine-tuning setups, the pretrained policy averaged 0.72 success versus 0.20
  from scratch.
- [IntervenGen](https://arxiv.org/abs/2405.01472) shows why corrective
  intervention data belongs between broad pre-training and clean target
  specialization: recovery examples cover policy-error states that clean
  demonstrations normally omit.
- [π0.5](https://www.physicalintelligence.company/download/pi05.pdf) shows
  that high-level/subtask training data materially improves downstream
  behavior, including when the runtime policy does not explicitly generate a
  high-level plan.
- A large real-robot
  [data-scaling study](https://arxiv.org/abs/2410.18647) finds that environment
  and object diversity matters more than repeatedly adding demonstrations
  after a per-setting saturation point. This supports a deterministic,
  task-balanced RealSource subset under the available time budget.

These papers support the **broad/diverse -> corrective/recovery -> clean
target-style** structure. They do not prove that 10%, two epochs, and four
epochs are uniquely optimal. Those exposure counts are a reviewed,
time-budgeted engineering choice. The prior 50% RealSource view contains
10,215,064 windows (79,806 optimizer steps at global batch 128), which would
overwhelm the 8,220 intervention and 7,524 HQ specialization steps. The v2
curriculum instead uses a deterministic 10% view with 2,043,024 windows,
2,437 episodes, and all 35 tasks: 15,962 optimizer steps for one complete
pass.

The pre-launch empirical gate is a real production-data checkpoint-handoff
validation: ten optimizer steps on each dataset, proving that A's
authenticated natural-final model initializes B and B's initializes C while
each stage creates fresh optimizer, scheduler, and RNG state. It is an
integration test and makes no model-quality claim.

## What an epoch means

An epoch is one global, no-replacement pass over **every row of the stage's
immutable frozen view**. DDP ranks shard that single schedule. The uneven tail
is retained and any duplicated DDP padding is reported separately.

We never call an interrupted 25% or 50% pass an epoch. Those percentages are
monitoring checkpoints inside the first exhaustive RealSource pass. They do
not change the natural-final handoff. At global batch 128, the materialized
view row counts determine exact optimizer steps; `plan` prints those counts
before launch.

The final stage is pure HQ so the final operator demonstrations control
deployment style. Do not add dataset-identity prompt tokens or implicit replay.
Training replay is only a memorization diagnostic. Heldout metrics and guarded
robot rollouts remain deployment gates, but they do not choose an earlier
cross-stage handoff checkpoint.

## Intervention-label gate

The Magna intervention dataset has a dataset-specific convention confirmed by
its operator: `valid_state=0` marks a mistake action and `valid_state=1` marks
an expert/recovery action.  That meaning is **not** inferred from the column
name.  In every other dataset, `valid_state` remains state-quality metadata and
is insufficient action-ownership provenance.

`intervention-incremental` always writes an `action_supervision_audit`.  It is
verified by either:

1. explicit `valid_action` plus `action_owner`/`action_source` columns, where
   every supervised action has a recognized expert/human owner and at least
   one invalid-state expert recovery action exists; or
2. a reviewed JSON `realman-action-label-semantics-v1` artifact bound to the
   exact source ID, episode-catalog SHA, annotation SHA, and source-content
   SHA.

Running the builder without either proof is intentional: it produces the view
and all hashes needed for review, but records `status: unverified`.  Stage B
sets `require_verified_action_supervision: true`, so `plan`, `check`, and
`start` reject that view.  A bare command-line boolean cannot approve it.

For the Magna convention, the contract must bind these exact semantics:

```json
{
  "schema": "realman-action-label-semantics-v1",
  "status": "verified",
  "dataset": {
    "source_id": "<from the unverified view audit>",
    "catalog_sha256": "<from the unverified view audit>",
    "annotation_sha256": "<from the unverified view audit>",
    "source_content_sha256": "<from the unverified view audit>"
  },
  "semantics": {
    "validity_column": "valid_state",
    "mistake_value": 0,
    "mistake_meaning": "mistake_action_do_not_supervise",
    "supervised_value": 1,
    "supervised_meaning": "expert_or_recovery_action_supervise",
    "recovery_anchor_definition": "first_valid_frame_after_invalid_frame"
  },
  "masking": {
    "policy": "chunk_prefix_until_first_sustained_invalid_or_padding",
    "invalid_run_length": 10,
    "recovery_windows": "windows_anchored_at_valid_recovery_frames_are_supervised"
  },
  "review": {
    "reviewer": "<human reviewer>",
    "reviewed_at_utc": "<ISO-8601 time>",
    "evidence": "<labeling report or review reference>"
  }
}
```

Rebuild with
`--action-label-semantics-contract /absolute/path/to/contract.json`.  The
manifest hashes the contract and reports every 0→1 recovery anchor, the exact
number of nonzero action-mask timesteps/elements retained for its H=50 window,
and whether every recovery-anchored window remains supervised.  No separate
recovery flag is required.  Pre-mistake windows may still mask a later recovery
suffix: the future error state was not visible at their chunk-start anchor, so
that suffix would be causally invalid supervision.  Windows anchored once the
label returns to 1 retain recovery supervision.

Stage B preserves the established
`action_validity_invalid_run_length=10` behavior. Individual 0-labelled
mistake actions are masked, but a chunk suffix is truncated only after ten
consecutive invalid actions. This keeps valid recovery targets following
short mistakes in the same H=50 chunk. Recovery is also supervised from the
exhaustive window anchored at its own later 1-labelled frame, and the audit
requires every such recovery window to retain a nonzero action mask.

## Learning-rate contract

| Stage | backbone base | Qwen interface | policy heads | warmup | cosine floor |
|---|---:|---:|---:|---:|---:|
| RealSource | `2e-5` | `1e-5` | `1e-4` | 5% | 5% of each group |
| intervention | `1e-5` | `5e-6` | `5e-5` | 5% | 5% |
| HQ | `4e-6` | `2e-6` | `3e-5` | 5% | 5% |

The optimizer and scheduler restart at each authenticated natural-final
handoff. This is a new delta-action curriculum; do not resume an older
absolute-action or 19-D checkpoint.

The world-model auxiliary objective does not restart its initial high-weight
ramp in Stages B or C. Stage A uses the inherited `wm_initial=0.3` to
`wm=0.1` warmup. Stages B and C set `wm_initial=wm=0.1` and
`wm_warmup_steps=0`; their fresh optimizer warmup remains active, but the
behavior-specific phases do not temporarily triple the world-model loss.

## Materialize the immutable contracts

### Break the canonical holdout/statistics bootstrap cycle

The existing RealSource v4 holdout was selected from the deterministic
task-balanced 50% population. That population is a strict superset of the
v2 production 10% view. The 10% view authenticates all 128 holdout identities,
excludes every identity/content copy that intersects it, and retains all
35 tasks. The broader fixed holdout remains useful for evaluating
generalization beyond the exact Stage-A subset.
Do not use an empty-exclusion training view to break this cycle. First create
a distinct non-trainable selection candidate, then generate the holdout from
only that candidate's authenticated ledger:

```bash
REALSOURCE_CANDIDATE=/data/mehul-vla-jepa/data_contracts/views/realsource_strict_valid_balanced_50_eval_selection_candidate_v1.json
REALSOURCE_EVAL=/data/mehul-vla-jepa/checkpoints/eval_manifests/realsource_strict_valid_fractional_eval128_v2.json

python scripts/build_realman_dataset_views.py realsource-canonical \
  --canonical-manifest /data/mehul-vla-jepa/src/dataset-canonicalization-training-d9c3298/configs/manifests/dataset_manifests.jsonl.gz \
  --adapter-path /data/mehul-vla-jepa/src/dataset-canonicalization-training-d9c3298/configs/dataset_adapters/RealSourceData_RealSource-World__607cbd4f6adf.json \
  --cache-dir /data/mehul-vla-jepa/datasets/canonical_gcs \
  --fraction 0.50 \
  --seed 0 \
  --eval-selection-population-candidate \
  --output "$REALSOURCE_CANDIDATE"

REALSOURCE_CANDIDATE_SHA=$(sha256sum "$REALSOURCE_CANDIDATE" | awk '{print $1}')

python scripts/generate_canonical_eval_manifest.py \
  --config scripts/config/h100/realman_curriculum/realsource_production_50_v1.yaml \
  --output "$REALSOURCE_EVAL" \
  --world-size 8 \
  --eval-selection-view-manifest "$REALSOURCE_CANDIDATE" \
  --eval-selection-view-manifest-sha256 "$REALSOURCE_CANDIDATE_SHA"
```

The `v2` filename is intentional. The canonical adapter/action-sidecar
contract changed after `v1` was frozen, so the generator created a new
immutable manifest instead of overwriting history. Its 128 selected
episode/window identities are identical to `v1`; only the authenticated
adapter and sidecar contract bindings changed.

The candidate has
`purpose: eval_selection_population_candidate`,
`usage_contract.training_allowed: false`, and no holdout exclusions because
the holdout does not exist yet. The canonical training loader rejects it
unless the eval generator uses its explicit code-only bootstrap gate. The
generator clears the not-yet-existing union-statistics reference, inspects raw
18-D supervision masks, and ranks only episodes present in this frozen
candidate. It never samples from the full canonical catalog.

After the eval manifest is immutable, materialize the actual holdout-free
v2 production view. Its row count is the only row count used for
optimizer-step planning:

```bash
python scripts/build_realman_dataset_views.py realsource-canonical \
  --canonical-manifest /data/mehul-vla-jepa/src/dataset-canonicalization-training-d9c3298/configs/manifests/dataset_manifests.jsonl.gz \
  --adapter-path /data/mehul-vla-jepa/src/dataset-canonicalization-training-d9c3298/configs/dataset_adapters/RealSourceData_RealSource-World__607cbd4f6adf.json \
  --cache-dir /data/mehul-vla-jepa/datasets/canonical_gcs \
  --eval-holdout-manifest "$REALSOURCE_EVAL" \
  --fraction 0.10 \
  --seed 0 \
  --output /data/mehul-vla-jepa/data_contracts/views/realsource_strict_valid_balanced_10_v1.json
```

Then build the separate task-balanced 50% RealSource
`statistics_population_candidate` shown below and compute the shared union
statistics. Training, eval selection, and statistics therefore use three
different purpose-bound artifacts; none can be substituted for another.

Create the remaining views on persistent storage:

```bash
python scripts/build_realman_dataset_views.py intervention-incremental \
  --dataset-root /data/mehul-vla-jepa/datasets/magna_training_data_with_interventions_final_subtask_labelled \
  --eval-holdout-manifest /data/mehul-vla-jepa/checkpoints/eval_manifests/magna_intervention_labelled_holdout_global_batch128_v1.json \
  --action-label-semantics-contract /data/mehul-vla-jepa/data_contracts/magna_intervention_action_labels_v1.json \
  --output /data/mehul-vla-jepa/data_contracts/views/magna_intervention_incremental_v1.json

python scripts/build_realman_dataset_views.py hq-clean-h50 \
  --dataset-root /data/mehul-vla-jepa/datasets/latest_high_quality_magna_data_final_subtask_labelled \
  --eval-holdout-manifest /data/mehul-vla-jepa/checkpoints/eval_manifests/magna_hq_subtasks_delta_fractional_holdout_global_batch128_v1.json \
  --output /data/mehul-vla-jepa/data_contracts/views/magna_hq_clean_h50_v1.json

```

For the first audit pass, omit `--action-label-semantics-contract`, copy the
four exact dataset-binding values from the emitted unverified manifest into
the reviewed contract above, then rerun with the flag.

The local builders derive holdout episode IDs from
`--eval-holdout-manifest`; the production CLI has no hard-coded-ID fallback.
The split entry must name the exact labelled dataset directory and match its
episode-catalog, info, train-complement, count, and content hashes. Generate a
new intervention split for the labelled Stage-B dataset rather than reusing
the older unlabelled dataset's manifest. The frozen view binds the exact split
file SHA-256, and the training loader validates the complete split again.

The RealSource generator uses the task-stratified `0.10` production selection.
It filters exact
`quality_assessments.overall_valid == "VALID"`, preserves all 35 tasks, selects
whole episodes deterministically, and maps 30 Hz to 20 Hz with exact half-up
indices.

The shared normalization artifact intentionally remains the holdout-clean
50%-RealSource + intervention + HQ union. It gives all three stages one fixed
same-robot scale and avoids changing the meaning of normalized model outputs
at a handoff. The RealSource evaluation episodes are excluded from that
statistics population.

The three training views above are deliberately holdout-free and cannot be
used as union-statistics sources. Materialize a separate candidate view for
each source. A candidate view has
`purpose: statistics_population_candidate`, contains the exact authenticated
holdout episode references, and is rejected by both training loaders. The
union builder is the only consumer: it hashes every candidate episode,
detects content copies, and contributes zero state/action samples for keys in
the immutable holdout.

```bash
python scripts/build_realman_dataset_views.py realsource-canonical \
  --canonical-manifest /data/mehul-vla-jepa/src/dataset-canonicalization-training-d9c3298/configs/manifests/dataset_manifests.jsonl.gz \
  --adapter-path /data/mehul-vla-jepa/src/dataset-canonicalization-training-d9c3298/configs/dataset_adapters/RealSourceData_RealSource-World__607cbd4f6adf.json \
  --cache-dir /data/mehul-vla-jepa/datasets/canonical_gcs \
  --eval-holdout-manifest /data/mehul-vla-jepa/checkpoints/eval_manifests/realsource_strict_valid_fractional_eval128_v2.json \
  --fraction 0.50 \
  --statistics-population-candidate \
  --output /data/mehul-vla-jepa/data_contracts/views/realsource_strict_valid_balanced_50_statistics_population_candidate_v1.json

python scripts/build_realman_dataset_views.py intervention-incremental \
  --dataset-root /data/mehul-vla-jepa/datasets/magna_training_data_with_interventions_final_subtask_labelled \
  --eval-holdout-manifest /data/mehul-vla-jepa/checkpoints/eval_manifests/magna_intervention_labelled_holdout_global_batch128_v1.json \
  --action-label-semantics-contract /data/mehul-vla-jepa/data_contracts/magna_intervention_action_labels_v1.json \
  --statistics-population-candidate \
  --output /data/mehul-vla-jepa/data_contracts/views/magna_intervention_statistics_population_candidate_v1.json

python scripts/build_realman_dataset_views.py hq-clean-h50 \
  --dataset-root /data/mehul-vla-jepa/datasets/latest_high_quality_magna_data_final_subtask_labelled \
  --eval-holdout-manifest /data/mehul-vla-jepa/checkpoints/eval_manifests/magna_hq_subtasks_delta_fractional_holdout_global_batch128_v1.json \
  --statistics-population-candidate \
  --output /data/mehul-vla-jepa/data_contracts/views/magna_hq_statistics_population_candidate_v1.json
```

Materialize the authenticated holdout and matching population manifest from
the exact production populations: task-balanced RealSource 50%, intervention,
and HQ. The command re-derives every heldout episode from all three evaluation
manifests, verifies that every key is present in the corresponding
statistics-candidate ledger, and binds the exact eval, catalog, view, and
representation hashes. A stale holdout file, a copied digest with different
keys, or a population manifest that names a different candidate view fails
closed.

```bash
python scripts/build_openpi_realman_union_contract.py \
  --holdout-output /data/mehul-vla-jepa/data_contracts/realman_union_holdout_v1.json \
  --population-output /data/mehul-vla-jepa/data_contracts/realman_union_population_v1.json \
  --realsource-eval-manifest /data/mehul-vla-jepa/checkpoints/eval_manifests/realsource_strict_valid_fractional_eval128_v2.json \
  --realsource-candidate-view /data/mehul-vla-jepa/data_contracts/views/realsource_strict_valid_balanced_50_statistics_population_candidate_v1.json \
  --realsource-cache-dir /data/mehul-vla-jepa/datasets/canonical_gcs \
  --intervention-eval-manifest /data/mehul-vla-jepa/checkpoints/eval_manifests/magna_intervention_labelled_holdout_global_batch128_v1.json \
  --intervention-candidate-view /data/mehul-vla-jepa/data_contracts/views/magna_intervention_statistics_population_candidate_v1.json \
  --intervention-dataset-root /data/mehul-vla-jepa/datasets/magna_training_data_with_interventions_final_subtask_labelled \
  --hq-eval-manifest /data/mehul-vla-jepa/checkpoints/eval_manifests/magna_hq_subtasks_delta_fractional_holdout_global_batch128_v1.json \
  --hq-candidate-view /data/mehul-vla-jepa/data_contracts/views/magna_hq_statistics_population_candidate_v1.json \
  --hq-dataset-root /data/mehul-vla-jepa/datasets/latest_high_quality_magna_data_final_subtask_labelled
```

The curriculum launcher also requires the resulting statistics artifact to
name the same holdout digest as
`shared_contract.statistics_holdout_manifest_sha256`.

```bash
python scripts/compute_openpi_realman_union_stats.py \
  --manifest /data/mehul-vla-jepa/data_contracts/realman_union_population_v1.json \
  --output /data/mehul-vla-jepa/data_contracts/realman_union_openpi_q01q99_v1.json
```

Copy the emitted SHA-256 values into the reviewed stage and curriculum YAMLs.
The checked-in `REPLACE_WITH_MATERIALIZED_*` values are deliberate fail-closed
markers: `plan`, `check`, and `start` must refuse to run until every path and
hash is real. Never replace them with an unverified all-zero digest.

Verify each view after transfer:

```bash
python scripts/build_realman_dataset_views.py verify \
  /data/mehul-vla-jepa/data_contracts/views/magna_intervention_incremental_v1.json
```

## Human launch

Use the reviewed v2 production curriculum. The v1/50% file is retained only
as an immutable historical contract and must not be used for the new launch:

```bash
CURRICULUM=scripts/config/h100/realman_realsource_intervention_hq_curriculum_v2.yaml
```

Then run:

```bash
./scripts/h100_curriculum.sh setup --config "$CURRICULUM"
./scripts/h100_curriculum.sh plan --config "$CURRICULUM"
./scripts/h100_curriculum.sh check --config "$CURRICULUM"
./scripts/h100_curriculum.sh start --config "$CURRICULUM" --detach
./scripts/h100_curriculum.sh logs
```

`setup` builds the pinned container and helper repositories from the first
stage. `plan` verifies the exact views and prints rows, steps per full epoch,
first-epoch exposure milestones, and full-epoch boundaries. `check` runs every
stage preflight. `start` runs those stages sequentially and admits the next
stage only after authenticating the previous stage's natural-final
checkpoint.

If the curriculum container or host process stops after at least one complete
full-state checkpoint, resume with the exact same reviewed YAML and original
curriculum run ID:

```bash
./scripts/h100_curriculum.sh resume \
  --config "$CURRICULUM" \
  --run-id ORIGINAL_CURRICULUM_RUN_ID \
  --detach
```

`resume` authenticates the saved curriculum state, every already-completed
stage handoff, each materialized stage-config hash, and the newest complete
checkpoint before launching the first incomplete stage. It does not accept
learning-rate, batch-size, epoch, dataset, or other training overrides.

The wrapper intentionally accepts no batch, LR, epoch, prompt, dataset, or
normalization overrides. Change the YAML, review the diff, regenerate hashes
when data changes, then rerun `plan` and `check`.
