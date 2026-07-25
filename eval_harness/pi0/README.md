PI0 / OpenPI LIBERO-Plus Eval
=============================

This folder contains a thin local launcher for evaluating the OpenPI
`pi05_libero` checkpoint with the same LIBERO-Plus all-category benchmark
configuration used for VLA-JEPA runs.

The policy server loads:

```text
gs://openpi-assets/checkpoints/pi05_libero
```

The checkpoint is cached by OpenPI under `~/.cache/openpi`, so the first launch
downloads about 12 GiB and later launches reuse the local copy.

Typical launch:

```bash
./eval_harness/pi0/scripts/run_pi05_libero_plus_all_categories.sh
```

Useful overrides:

```bash
NUM_SHARDS=24 PORT=8015 \
OUTPUT_DIR=/home/mehul/work/vjepa/eval_videos/pi05_libero_plus_all_categories \
./eval_harness/pi0/scripts/run_pi05_libero_plus_all_categories.sh
```

Monitor progress:

```bash
DB=/path/to/output/recording-pi05-libero-plus-YYYYmmdd-HHMMSS.sqlite
watch -d -n 0.5 'sqlite3 "$DB" "select status,count(*) from episode_results group by status; select printf('\''%.2f%%'\'',100.0*sum(status='\''success'\'')/count(*)) as success_rate, count(*) as completed, 10030-count(*) as remaining from episode_results;"'
```
