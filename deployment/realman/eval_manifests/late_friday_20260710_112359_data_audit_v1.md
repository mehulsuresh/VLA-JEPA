# Late-Friday Realman data audit (2026-07-10)

## Scope and provenance

"Late Friday" means the final and largest locally available Friday collection session, `20260710_112359`. Its authoritative per-frame `sync.timestamp_ns` values span 2026-07-10 11:24:12.134 through 12:25:00.842 America/Los_Angeles. Raw episodes 0-12 map one-to-one to final training episode IDs 1625-1637. No later local session is present in the fetched raw source.

All 13 episodes use the same global task text:

> reach into the bin, lift the chain, put it in the jig, then remove it from the jig and put it in the other bin

| Final ID | Raw ID | First frame (PDT) | Last frame (PDT) | Frames | Nominal video duration | `valid_state=1` |
|---:|---:|---|---|---:|---:|---:|
| 1625 | 0 | 11:24:12.134 | 11:29:18.832 | 5,833 | 291.65 s | 76.94% |
| 1626 | 1 | 11:30:05.879 | 11:34:04.029 | 4,745 | 237.25 s | 70.26% |
| 1627 | 2 | 11:37:42.355 | 11:39:52.904 | 2,553 | 127.65 s | 88.95% |
| 1628 | 3 | 11:40:14.081 | 11:42:07.731 | 2,241 | 112.05 s | 88.98% |
| 1629 | 4 | 11:42:38.047 | 11:45:43.098 | 3,702 | 185.10 s | 81.06% |
| 1630 | 5 | 11:54:54.344 | 11:57:57.144 | 3,655 | 182.75 s | 75.92% |
| 1631 | 6 | 11:58:20.669 | 12:01:16.169 | 2,933 | 146.65 s | 74.91% |
| 1632 | 7 | 12:01:41.566 | 12:05:04.866 | 3,399 | 169.95 s | 62.25% |
| 1633 | 8 | 12:05:27.594 | 12:09:04.044 | 4,213 | 210.65 s | 80.51% |
| 1634 | 9 | 12:09:26.648 | 12:11:14.148 | 2,080 | 104.00 s | 94.18% |
| 1635 | 10 | 12:11:36.575 | 12:14:53.725 | 3,865 | 193.25 s | 84.58% |
| 1636 | 11 | 12:15:17.949 | 12:17:37.200 | 2,460 | 123.00 s | 95.93% |
| 1637 | 12 | 12:20:46.342 | 12:25:00.842 | 4,874 | 243.70 s | 85.13% |

Total: 46,553 frames, 2,327.65 nominal video seconds, and 37,305 valid frames (80.13%). This session is 1.898% of the 2,452,692-frame training dataset. All 38 final appended episodes (1600-1637) are 145,683 frames, or 5.940% of training.

## Structural and numeric integrity

- All 39 raw camera streams (13 episodes x head/left wrist/right wrist) are H.264 YUV420P, 640x480, 20 fps. Every stream's frame count exactly equals its episode length.
- Matched-frame visual review shows coherent head, left-wrist, and right-wrist viewpoints without evidence of a camera swap. Training order is `head`, `wrist_left`, `wrist_right`; the training transform produces three 384x384 RGB views.
- The preserved dataset fields are `source.action` float32 `[22]` and `source.observation.state` float32 `[19]`. This checkpoint consumes state indices `[0:19]` (19 dimensions) and action indices `[0:16] + [19:21]` (18 dimensions: both arms/grippers plus two head joints; base and lift action are omitted).
- All late-session source and official action/state values are finite. Gripper commands exercise both states, with open-state fractions of 92.72% on the left and 80.16% on the right.
- Raw-to-final validation across all 46,553 rows is exact: maximum absolute error is 0 for preserved source action/state and for the selected official arm action/state fields.
- Raw-to-final video validation sampled first/middle/last frames for every episode and camera (117 comparisons). All decoded; median uint8 pixel MAE was 2.44 and maximum was 3.66, consistent with H.264 re-encoding, with no offset or camera mismatch.
- The raw session contains a zero-byte, video-only `episode_000013` for each camera and no matching parquet. The clean manifest explicitly excluded it; it is not in the final training dataset.

The late data's dynamics closely match the preceding 25 appended episodes:

| Metric | Episodes 1600-1624 | Episodes 1625-1637 |
|---|---:|---:|
| Observed near-stationary frames, max arm delta <= 0.001 rad/frame | 21.93% | 20.06% |
| Observed slow frames, max arm delta <= 0.01 rad/frame | 60.42% | 60.23% |
| Command near-hold, max arm `abs(action-state)` <= 0.01 rad | 1.33% | 1.16% |
| Command active, max arm `abs(action-state)` > 0.05 rad | 73.96% | 72.71% |
| Valid frames | 64.44% | 80.13% |

Raw capture cadence is imperfect but not newly degraded: late median/p95/p99/max frame intervals are 50.02/98.35/100.21/199.85 ms, versus 50.03/99.72/100.22/200.83 ms in the preceding appended data. Late camera-skew p95/max are 45.77/71.99 ms versus 44.60/77.49 ms earlier.

## Critical labeling result

Every one of the 46,553 late-Friday frames has `subtask_index=0` and `valid_state_source=0`. There are no materialized stage-2-through-stage-7 labels or formal stage boundaries in these episodes. In fact, all appended episodes 1600-1637 are only label 0.

Visual review nevertheless confirms that all 13 videos contain the full observable sequence: reach, grasp/lift, carry, place, align, and removal to the destination. Episode 1631 contains repeated pickup retries before completing the sequence. Thus the late data contains learnable full-task behavior, but it contributes no stage-conditioned prompt supervision.

This matters because the entire dataset is strongly stage-imbalanced: stage 6 has 878,780 frames (35.83%), while stage 7 has 95,498 (3.89%). The latest full-task demonstrations cannot improve those labeled stage counts. Their 1.90% total weight also limits how much they can correct the older distribution under a global prompt.

## Evaluation artifacts

- Episode-selection manifest: `late_friday_20260710_112359_regression_v1.json`
- Episode selection plus 72 visually reviewed, stage-balanced, 50-step-valid representative frames: `late_friday_20260710_112359_stage_balanced_v1.json`
- Full machine-readable audit: `/tmp/late_friday_20260710_112359_data_audit.json`
- Stage contact sheet: `/tmp/late_friday_stage_balanced_v1.jpg`

The current `audit_checkpoint_offline.py` accepts both episode-only manifests and strict explicit frame maps. With `--seed 20260713 --frames-per-episode 12`, the episode-only deterministic stratum plan has SHA-256 `07291e85ecbbec381e0f9ae30f8bb4e5a8dcd9615dc868de2e2bc30597784d29`. Use global prompt mode for the authoritative late-data regression; source-derived subtask-explicit mode cannot work here because all source labels are 0.

## Data-side conclusion

There is no evidence that late-Friday numeric conversion, camera mapping, video alignment, schema, or ranges are corrupt. The demonstrations visibly perform the task, so the dataset is not intrinsically incapable of representing it. The material data problem is missing stage labels on every appended episode, combined with a small late-data weight and severe stage imbalance. Offline checkpoint errors on these episodes—especially pickup/reach frames—are therefore the next useful discriminator between model underfitting/phase ambiguity and any remaining deployment issue.
