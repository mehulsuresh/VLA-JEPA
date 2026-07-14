# Late-Friday Realman model diagnosis (2026-07-13)

## Decision

The task and demonstrations are learnable. The present failure is best explained by a training/deployment objective mismatch, not by an incapable model and not by corrupt late-Friday data.

The dominant defect is fully-ramped RTC prefix training without RTC prefix inference. At and after step 60,000, the training configuration makes the first action contribute to the action loss only 1/11 (9.1%) of the time. The first five actions receive 27.3% average supervision, while positions 10 through 49 remain fully supervised. The synchronous Realman rollout sends no prior action prefix. This predicts the measured behavior: poor immediate/static control and delayed gripper closure, despite much better long-horizon predictions.

Do not treat the currently served checkpoint as ready for another hardware rollout. The server is on step 62,500; the newest complete training checkpoint was step 65,000 at the time of this audit, and neither has passed a pickup-focused selection gate.

## Data audit

The late-Friday session is `20260710_112359`, final episodes 1625 through 1637:

- 13 episodes and 46,553 frames, recorded 11:24:12 through 12:25:00 PDT.
- All 13 episodes visibly contain the complete reach, pickup, carry, place, align, remove, and destination sequence. Episode 1631 contains pickup retries.
- All 39 camera streams are valid 640x480 H.264 at 20 fps with exact frame counts.
- Raw-to-final action/state conversion is exact; the maximum numeric difference is 0.
- Sampled raw/final video comparisons show no camera swap or temporal offset.
- Late-session action/state ranges and motion statistics lie inside the broader training distribution.

The material data defects are weighting and labels:

- Every late-Friday frame has `subtask_index=0`; all appended episodes 1600 through 1637 lack stage labels.
- Late Friday is only 1.898% of all training frames.
- Stage 6 is 35.83% of the full dataset; pickup stage 3 is about 7.77%.

Therefore the latest successful full-task demonstrations are present but lightly weighted and contribute no stage-conditioned pickup supervision.

## Fixed offline checkpoint results

The uniform late-Friday regression used the same deterministic frame plan for every checkpoint (84 valid frames; plan SHA-256 `f4edb...a86f`). Arm MAE is in raw joint radians, averaged over both 7-DoF arms.

| Checkpoint/baseline | h1 | h5 | h10 | h20 | h50 |
|---|---:|---:|---:|---:|---:|
| Step 57,500 | 0.05334 | 0.05667 | 0.06013 | 0.06188 | 0.06605 |
| Step 60,000 | 0.05381 | 0.05722 | 0.06093 | 0.06322 | 0.06785 |
| Step 62,500 | 0.05394 | 0.05688 | 0.06081 | 0.06316 | 0.06705 |
| Hold current state | 0.04790 | 0.05366 | 0.06108 | 0.07333 | 0.11308 |

Step 57,500 is best on this newest session. Step 62,500 is still worse than simply holding at h1 and h5, but much better than hold at long horizons. On an older labeled complete-stage set, step 62,500 improves over step 57,500. This non-monotonic tradeoff means “latest” is not “best.”

The pickup-specific plan uses 48 manually bracketed frames from 12 clean late-Friday pickups, with 50 valid future actions per frame (plan SHA-256 `af20c983...0889`).

| Step 62,500 method/baseline | h1 | h5 | h10 | h20 | h50 |
|---|---:|---:|---:|---:|---:|
| One stochastic draw | 0.03758 | 0.03889 | 0.04028 | 0.04219 | 0.05217 |
| Median of four draws | 0.03570 | 0.03727 | 0.03842 | 0.03954 | 0.04776 |
| Hold current state | 0.02522 | 0.02813 | 0.03065 | 0.03902 | 0.07499 |

Four-draw median ensembling reduces stochastic arm error, but the median remains 41.5%, 32.5%, and 25.3% worse than hold at h1, h5, and h10. It only becomes competitive around h20 and is strong by h50. Ensembling is useful variance reduction, but it does not repair the learned early-horizon defect.

## Pickup/gripper result

For the four-draw median on the pickup plan (positive means open; a false positive means the target closes but the policy stays open):

| Executed prefix | Required closed labels | Correctly closed | Missed closures | Close recall |
|---|---:|---:|---:|---:|
| h1 | 32 | 26 | 6 | 81.25% |
| h5 | 176 | 137 | 39 | 77.84% |
| h10 | 356 | 284 | 72 | 79.78% |

There are no false-close errors in the first ten steps; the model is specifically biased toward remaining open. Four of the six h1 misses predict a fully open value near 1.0 in all four stochastic draws. The remaining two are ambiguous near 0.55. Several missed examples predict closure only 7 to 10 actions later, while others remain open through all ten actions. This is learned timing/phase error, not a threshold or sampling-noise problem.

Across all 2,452,692 training frames, each gripper is closed only about 15% of the time. Open-to-close transitions occur in approximately 0.1% of frames per gripper. Grippers are only 2 of 18 action channels and receive the same continuous squared flow-matching loss as every arm/head channel; there is no class, transition, or per-channel weighting.

## Why RTC is the primary cause

The active configuration uses a 50-step horizon and RTC with `max_delay=11`, uniform delay, `rtc_prob=1.0`, a 10k warmup, and a 50k ramp. At full ramp the sampled delay is uniformly 0 through 10. Positions before that delay are inserted as a clean known prefix and explicitly removed from loss.

| Action position | Probability it receives loss at full ramp |
|---:|---:|
| 0 | 9.1% |
| 1 | 18.2% |
| 2 | 27.3% |
| 3 | 36.4% |
| 4 | 45.5% |
| Mean, positions 0-4 | 27.3% |
| 9 | 90.9% |
| 10-49 | 100% |

At step 57,500, h0 still received about 13.6% supervision because the RTC ramp was not quite complete. From step 60,000 onward it receives 9.1%. The live request contract contains only `instructions`, `qwen_frames`, and `state`; it contains neither normalized `prev_actions` nor `prefix_len`. The client itself warns that it is synchronous and does not stitch the overlapping RTC prefix.

This creates exactly the observed failure mode: training teaches the network that early actions are usually already-known prefix placeholders, while deployment asks those same positions to initiate reach and grasp from a fresh observation.

## Other verified deployment facts

- The image-format correction is active. Live requests use `qwen_frames`, three RGB `uint8` 384x384 views in `head`, `wrist_left`, `wrist_right` order. The obsolete 224-pixel PIL path is rejected.
- State and action use the checkpoint's min/max statistics and exact 19D/18D contracts.
- The server preserves FP32 checkpoint parameters and uses BF16 autocast, matching training more closely than a global BF16 cast.
- The most recent live request has no RTC prefix.
- The current server is step 62,500 with a single stochastic draw. It was not restarted during this audit and had no robot client connected at the final check.
- Training remained healthy and untouched around step 65,097/76,644; the newest complete remote checkpoint was step 65,000.
- The training run has no real held-out selection signal. Its configured best metric remains null; the built-in check consumes one training batch, uses a different inference-step count, and aggregates all 50 horizons/channels, hiding h1 and closure recall.

## Correction plan

For a synchronous controller, the lowest-risk next model change is to fine-tune with RTC disabled. If RTC is retained, deployment must implement true overlapping-prefix scheduling, and training must preserve a substantial unconditioned h1/h5 objective rather than masking h0 91% of the time.

The next training/fine-tuning run should also:

1. Weight h1/h5 strongly or add a short-horizon control head alongside the 50-step planner.
2. Add a class-balanced gripper objective and transition weighting; oversample pickup/closure frames.
3. Materialize stage labels for episodes 1600-1637 and use stage-balanced sampling.
4. Hold out complete episodes and select checkpoints on pickup h1/h5 arm error, close recall, static hold-relative error, and full-stage metrics.
5. Match the hardware command rate to the 20 fps data cadence and use finite chunk-1 then chunk-5 gates only after offline metrics pass.

A practical recovery path is a short, lower-learning-rate fine-tune from the best fixed-evaluation checkpoint with RTC disabled, stage/pickup balancing, and an explicit gripper loss. Re-evaluate step 65,000 first; do not assume it improved simply because it is newer.

## Artifacts

- Data audit: `late_friday_20260710_112359_data_audit_v1.md`
- Late-Friday episode manifest: `late_friday_20260710_112359_regression_v1.json`
- Pickup manifest: `late_friday_20260710_112359_pickup_valid_v1.json`
- Step 62,500 pickup, one draw: `logs/realman_steps62500_late_friday_pickup_valid_deployment_b1.json`
- Step 62,500 pickup, four draws: `logs/realman_steps62500_late_friday_pickup_valid_deployment_b4.json`

No training, policy-server, or robot-control process was stopped, signaled, restarted, or modified during this audit.
