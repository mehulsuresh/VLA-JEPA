# Frozen RealMan curriculum views

These manifests define immutable row populations. A logical epoch is an
exhaustive pass over every ledger row; source weights, replacement sampling,
and implicit `max_windows` caps are forbidden.

In distributed training, "every row" means the union of the rows processed by
all ranks. Each rank processes its shard of that same finite schedule; it must
not independently replay the complete dataset. The final uneven DDP batch is
kept (`drop_last: false`) and any distributed padding is reported separately.
Changing the deterministic permutation between epochs changes only order, not
membership: every logical epoch still covers the complete frozen view.

Fractional exposure points such as 5%, 10%, 25%, and 50% are monitoring
checkpoints inside an epoch. They are not smaller sampled datasets and they do
not redefine the epoch boundary. A completed epoch always reaches 100% of the
frozen view.

Generate the local views outside the repository because their JSONL ledgers
contain hundreds of thousands of rows:

```bash
python scripts/build_realman_dataset_views.py intervention-incremental \
  --dataset-root /home/mehul/work/reward_model_small/magna_training_data_with_interventions_final_subtask_labelled \
  --output /mnt/data/reward_model_small/dataset_views/magna_intervention_incremental_v1.json

python scripts/build_realman_dataset_views.py hq-clean-h50 \
  --dataset-root /home/mehul/work/reward_model_small/latest_high_quality_magna_data_final_subtask_labelled \
  --output /mnt/data/reward_model_small/dataset_views/magna_hq_clean_h50_v1.json
```

Use `--dry-run` to compute the exact view identity and population without
writing artifacts. Verify a materialized artifact with:

```bash
python scripts/build_realman_dataset_views.py verify /path/to/view.json
```

`realsource_full_strict_valid_v1.template.json` is deliberately not a usable
manifest. It specifies the selection contract for a future canonical/GCS
generator. That generator must include every strict-valid episode and every
20 Hz row; it must not apply a task cap or balanced curation.

The production local datasets were dry-run verified on 2026-07-24:

- intervention incremental: 148 contributing episodes and 521,135 rows;
- clean HQ H50: 485 contributing episodes and 240,718 windows.

These counts are assertions about the currently bound source catalogs, not
magic truncation limits. Regeneration after a source change must produce new
catalog, annotation, content, ledger, and view hashes.
