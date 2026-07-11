# Repository Agent Instructions

## Production Training

Before inspecting, preparing, launching, resuming, or handing off a Magna A100x8
training run, read and follow
[`docs/magna_a100x8_training_runbook.md`](docs/magna_a100x8_training_runbook.md).
It is the living operational source of truth; do not duplicate its volatile run
state in this file.

- Inspect live processes, containers, tmux sessions, exit markers, and logs
  before changing anything on a training node.
- Never launch or resume over an active run, and never interrupt an unfamiliar
  training, upload, transfer, build, evaluation, or audit process.
- Production must use a clean pushed commit, an immutable per-run worktree, a
  non-secret launch environment, named containers, and verified durable
  checkpoint uploads.
- When operational experience reveals a reusable lesson, correct the runbook's
  durable procedure, refresh its bounded UTC handoff snapshot, run its handoff
  validations, and commit/push/synchronize the intentional changes.

For LIBERO simulation evaluation work, use
[`docs/libero_sim_eval_agent_runbook.md`](docs/libero_sim_eval_agent_runbook.md).
