# H100 Checkpoint And Eval Mirror

`scripts/mirror_h100_checkpoint_run.py` is the fail-closed fallback backup path
for a training run when unattended object-storage authentication is unavailable.
It performs two independently verified hops:

```text
H100 run (read only) -> workstation mirror -> protected Columbus storage
```

The configured H100 run path may traverse the production `/mnt/vla-jepa`
symlink. Preflight resolves that logical path once, pins every symlink ancestor
plus the canonical target device/inode/uid in watcher identity, and performs all
artifact reads through the resolved path. The logical path remains the run/eval
identity recorded by training. Every poll re-resolves the binding and compares
the stat-only inventory root; retargeting is refused. This exception applies
only to the pinned read-only H100 source—state, staging, restore, and Columbus
paths continue to reject symlinked ancestors.

The Columbus destination is fixed to the new protected namespace:

```text
/mnt/vla-jepa/h100-relay/checkpoint-mirror-storage/<RUN_ID>
```

It does not use the old inbound relay, credentials, or GCS. The Columbus SSH
user must already own `/mnt/vla-jepa/h100-relay` with mode `0700`. Read-only
preflight deliberately refuses to create or repair that protected parent.

## Production invariants

A checkpoint is published as recoverable only after all of these hold:

- the exact full-state file contract is present, with no empty, linked,
  hard-linked, special, or unexpected files;
- the trainer's schema-1 selection state is exact and agrees with the strict
  global argmin/argmax over every authenticated checkpoint-bound eligible eval
  through that step; equal metrics retain the earlier winner;
- the eval binds both `trainer_state.json` and `model.safetensors` by SHA-256,
  and its run ID, output path, config SHA, resolved-schedule SHA, seed, complete
  sampling reports, validity flags, and focused metric placement match the
  immutable run evidence;
- the checkpoint, same-step eval, evidence snapshot, and every selection
  dependency are byte-verified on both the workstation and Columbus;
- canonical manifests and an immutable recovery receipt exist on both tiers;
- an atomic generation-numbered Columbus restore index references the receipt.

Legacy checkpoints or evals missing `checkpoint.model_file_sha256` are
archival-only and are never recoverable, selectable, prunable as a recovery
copy, or included in the public restore index. There is no production option
to waive same-step eval evidence.

The production step-0 `live_in_memory_model` baseline has no checkpoint by
design. Its exact run/sampling/metric evidence is authenticated and cached, but
it is explicitly nonrecoverable: it cannot enter checkpoint selection history,
retention dependencies, receipts, or a restore index. A live-model artifact at
any other step, or a step-0 artifact that claims checkpoint bytes, is refused.

Normal one-minute polls use only `lstat` metadata fingerprints and cached
canonical manifests. Unchanged polls do not hash model bytes on any tier. A
full source/workstation/Columbus scrub runs every six hours by default; use
`once --full-scrub` for an immediate scrub. A scrub authenticates both complete
restore-index closures, including recovery receipts and checkpoint, eval, and
evidence manifest records, before advancing its durable completion timestamp.

Run-definition evidence (config, resolved schedule, dataset statistics and
provenance, heldout windows, launch/preflight records) is pinned on first
authentication and may never drift. Append-only `summary.jsonl` and the current
best pointer may advance; every new summary is required to preserve the exact
previously authenticated byte prefix, so rewrite and truncation fail closed.
Each checkpoint's recovery-publication intent pins the exact checkpoint, eval,
and evidence identity before either tier copy begins, so a crash, source
retention, and later summary/pointer advance cannot lose or change that receipt.
The small eval is pinned locally before the checkpoint's longer two-hop copy.
If a crash leaves a receipt-complete checkpoint ahead of the public pointer,
startup reconstructs the exact checkpoint-bound selection bytes and publishes
that recoverable prefix without H100 access.
One-tier evidence snapshots are likewise completed from their authenticated
workstation copy before reading possibly advanced source evidence.

Only the numerically newest checkpoint may be temporarily incomplete while the
trainer finishes its eval and best-pointer update. It is reported as
`healthy_pending`; older finalized checkpoints continue mirroring. A non-latest
incomplete checkpoint, unexpected file/link, cryptographic change, newer step
appearing ahead of an incomplete predecessor, or finalization timeout fails
closed. If a pending source directory disappears, its identity remains as a
durable pending tombstone; it cannot be silently cleared or bypassed by a later
checkpoint.

## Read-only preflight

Set the exact launched run ID. Preflight contacts both endpoints, pins their SSH
alias/resolved user/host/port/known-host fingerprint and remote identity, checks
Python/rsync, protected-root ownership and mode, existing locks/conflicts,
local and Columbus free space plus reserve, the stat-only source inventory,
candidate/pending steps, and retention. It creates no directory, lock, state,
or transfer:

```bash
RUN_ID=<exact-launched-run-id>
METRIC=heldout_focused_eval_task_failure_score_h10

python scripts/mirror_h100_checkpoint_run.py preflight \
  --run-id "$RUN_ID" \
  --h100-host reward-model-small \
  --source-run-dir "/mnt/vla-jepa/checkpoints/$RUN_ID" \
  --local-root "/home/mehul/work/checkpoint-mirror/$RUN_ID" \
  --columbus-host columbus-8xa100.us-east5-a.yondu-general-workspace \
  --columbus-root "/mnt/vla-jepa/h100-relay/checkpoint-mirror-storage/$RUN_ID" \
  --state-dir /home/mehul/.local/state/vlajepa-checkpoint-mirror \
  --selection-metric-name "$METRIC" \
  --selection-metric-mode min \
  --retain 3 \
  --disk-reserve-bytes 100000000000
```

`once --dry-run` is retained only as a compatibility alias for `preflight`.

## Watch

After reviewing a clean preflight, use the same arguments with `watch`:

```bash
python scripts/mirror_h100_checkpoint_run.py watch \
  --run-id "$RUN_ID" \
  --h100-host reward-model-small \
  --source-run-dir "/mnt/vla-jepa/checkpoints/$RUN_ID" \
  --local-root "/home/mehul/work/checkpoint-mirror/$RUN_ID" \
  --columbus-host columbus-8xa100.us-east5-a.yondu-general-workspace \
  --columbus-root "/mnt/vla-jepa/h100-relay/checkpoint-mirror-storage/$RUN_ID" \
  --state-dir /home/mehul/.local/state/vlajepa-checkpoint-mirror \
  --selection-metric-name "$METRIC" \
  --selection-metric-mode min \
  --retain 3 \
  --disk-reserve-bytes 100000000000 \
  --poll-seconds 60 \
  --max-backlog 2 \
  --full-scrub-hours 6 \
  --pending-timeout-seconds 1800
```

Rsync uses checksum repair, protected partial files, `--fsync`, and
`--no-inplace`. Immutable destination conflicts fail rather than overwrite.
The current best pointer is the sole mutable artifact, but every version is
first retained by content SHA-256 and pointer history can advance only to a
later strict improvement.

State and heartbeat files are atomically stored under:

```text
<STATE_DIR>/<RUN_ID>/state.json
<STATE_DIR>/<RUN_ID>/heartbeat.json
```

State schema 4 journals the full next restore-index payload and any authenticated
source-pointer publication before mutation. Startup completes pending recovery,
pointer, and index work under the Columbus lock before inspecting a source that
may have advanced. Recovery receipts use the same write-ahead rule inside their
checkpoint state record. If a global source-pointer intent targets bytes not yet
copied, startup can still publish the highest receipt-complete prefix from its
checkpoint-bound selection state and retain the later intent for retry. Every
poll also compares durable recoverable/pointer state with the authenticated
local index and reevaluates retention, so crashes immediately before index
publication or prune planning are repaired before a healthy result.
Intentionally pruned checkpoints are not re-mirrored merely because the trainer
still retains its source copy.

Check the local heartbeat without contacting either endpoint:

```bash
python scripts/mirror_h100_checkpoint_run.py status \
  --run-id "$RUN_ID" \
  --state-dir /home/mehul/.local/state/vlajepa-checkpoint-mirror \
  --stale-heartbeat-seconds 900
```

The Columbus ownership lock deliberately survives an abnormal process exit.
Do not remove it until the original process and all associated SSH/rsync
children are proven absent.

## Verification and restore from Columbus alone

Full verification needs neither H100 access, workstation mirror state, nor the
workstation artifact copy:

```bash
python scripts/mirror_h100_checkpoint_run.py verify \
  --run-id "$RUN_ID" \
  --columbus-host columbus-8xa100.us-east5-a.yondu-general-workspace \
  --columbus-root "/mnt/vla-jepa/h100-relay/checkpoint-mirror-storage/$RUN_ID" \
  --selection-metric-name "$METRIC" \
  --selection-metric-mode min
```

Print the authenticated current restore index with `restore-index` and the same
arguments. Restore into a new, absent absolute directory with:

```bash
python scripts/mirror_h100_checkpoint_run.py restore \
  --run-id "$RUN_ID" \
  --columbus-host columbus-8xa100.us-east5-a.yondu-general-workspace \
  --columbus-root "/mnt/vla-jepa/h100-relay/checkpoint-mirror-storage/$RUN_ID" \
  --selection-metric-name "$METRIC" \
  --selection-metric-mode min \
  --restore-destination "/home/mehul/work/restored-checkpoints/$RUN_ID"
```

Restore first verifies Columbus, refuses while the mirror lock is held, derives
an exact file plan from authenticated checkpoint/eval/evidence manifests and
receipts, and copies only that indexed closure into a hidden sibling stage.
Unindexed stages, trash, old history, obsolete generations, and arbitrary
regular files never cross the restore boundary. The staged file and directory
inventory must be exact, full semantic/byte verification must pass, and the
remote index and file plan must remain unchanged before an atomic publish.

Retention pruning is crash-consistent:

```text
planned -> index_withdrawn -> local_trashed -> columbus_trashed
        -> local_deleted -> columbus_deleted -> complete
```

Each transition is durable, trash is deterministic and same-filesystem,
directories are fsynced, a receipt is mirrored, and startup reconciles every
intermediate phase before doing new work.
