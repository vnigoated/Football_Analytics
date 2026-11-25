# Phase 2 · A6: end to end, broadcast frames → game state (GS-HOTA)

**Notebook:** `colab/06_pipeline.ipynb` · **Code:** `pitchlens.vision.pipeline`, `pitchlens video` · **Run 1:** pending
(Colab A100)

## Pipeline

| Step | From | What |
|---|---|---|
| People boxes | A2 run 2 | RF-DETR Medium, 1120 px, score ≥ 0.05 (cached per clip) |
| Tracks | A3 | OC-SORT with camera-motion compensation |
| Homography per frame | A5 | 33 keypoints, RANSAC, at least 6 inliers |
| **Homography over time** | new | drop fits that disagree with their neighbours, fill short gaps, optional smoothing |
| Pitch positions | | box bottom middle through the frame's homography |
| Teams | A4 | shirt colour without lightness, one feature per track, k-means |
| **Roles and sides** | new | tracks fitting neither team: goalkeeper in a penalty area, referee elsewhere; the team further left on average is "left" |
| Jersey numbers | not yet | predicted as null |

**Homography over time.** Each homography is compared through where it sends nine fixed image points in the lower
part of the frame. A fit more than `outlier_m` metres (median) from the median of its neighbours within 6 frames is
dropped. Gaps up to `max_gap` frames are filled by interpolating those mapped points linearly and refitting. At the
start and end of a clip the nearest homography is held. An optional moving average over ±`smooth` frames reduces
jitter.

## Metric

GS-HOTA (Somers et al., CVPRW 2024) is HOTA with similarity `LocSim × IdSim`:

- `LocSim = exp(ln(0.05) · d² / τ²)`, where d is the pitch distance and τ = 5 m. Pairs scoring at or below 0.05 are
  never matched.
- `IdSim = 1` only if role, team and jersey number all match, otherwise 0. An attribute missing from the labels is not
  checked, for example a referee's team or an unreadable number. The ball is not scored.

Ground truth comes from SN-GSR `bbox_pitch`, in every labelled frame. Because we predict no jersey numbers, we also
report three relaxed variants: without jersey, role only, and position only. These mirror the SN-GSR paper's
ablation (Table 1). We also report oracle rows that replace one stage with the labels.

Published reference points:
- SoccerNet baseline: 22.26 on test and 18.05 on valid. Its position-only score is 42.65 on test.
- 2024 challenge winner: 63.81 on the challenge set.
- Choi (2026): 35.2 on 11 test sequences.

## Protocol

Settings are tuned on the even-numbered valid clips and reported on the odd-numbered clips, the same halves as A3
and A4:
1. Homography settings, chosen by position-only HOTA.
2. Role settings, chosen by HOTA without jersey.

Test is then run once with those settings.

## Run 1 (2026-11-25, Colab A100-SXM4-40GB, 12 CPUs; torch 2.14.1+cu130, rfdetr 1.12.1, trackers 2.7.0)

Time: keypoints on every frame took 871 s for valid (58 clips, 43,500 frames, 50 frames/s). Tracking and colours
took 294 s on 12 CPUs. Test: detections 807 s, keypoints 664 s, tracking 271 s (49 clips).

### Label checks (valid)

- The "left" team stands further towards negative pitch x in **56 / 58 clips**, so `left_is_negative_x = True`.
- Roles: 610k player, 61k referee, 25k goalkeeper and 399 "other" boxes.
- **87.2% of player boxes carry a jersey number**, so a pipeline without numbers loses most of its GS-HOTA matches.

### Homographies over time (tuning half, every frame)

| Setting | Position-only HOTA | Frames with H | Median error | Mean | Within 2 m |
|---|---|---|---|---|---|
| A5 alone (6+ inliers, no filling) | 55.60 | 80.4% | 0.65 m | 1.08 m | 88.2% |
| + fill gaps ≤ 50 frames | 59.92 | 89.5% | 0.67 m | 1.14 m | 87.6% |
| + drop fits > 3 m from neighbours | 61.19 | 89.5% | 0.65 m | 1.04 m | 89.0% |
| + gaps ≤ 100, moving average ± 5 frames (**chosen**) | **62.93** | **92.2%** | 0.62 m | 1.07 m | 89.8% |
| 5+ inliers, gaps ≤ 50, no outlier check | 58.80 | 94.4% | 0.72 m | 3.29 m | 83.9% |

Using time adds **+7.3 HOTA points**. Each step helps: filling raises coverage, the outlier check removes the wrong
fits that filling would otherwise spread, and smoothing removes jitter. Five inliers give more frames but worse
positions unless the outlier check is on, so six inliers stay the default.

### Roles and teams (tuning half, HOTA without jersey)

| k | Distance rule | HOTA without jersey |
|---|---|---|
| 2 | off | 52.13 |
| 2 | z = 2 | 59.24 |
| 3 | off | 58.85 |
| **3** | **z = 3** (pitch margin 5 m, **chosen**) | **59.42** |

Setting referees and goalkeepers apart is worth **+7 points**. With k = 2 everybody is forced into a team, which the
A4 accuracy did not penalise but GS-HOTA does. The pitch margin makes almost no difference.

### GS-HOTA (%)

| Setting | GS-HOTA | Without jersey | Role only | Position only |
|---|---|---|---|---|
| **Ours, test (49 clips)** | **23.50** | **52.61** | 58.36 | **61.57** |
| Ours, valid held-out half (29 clips) | 25.38 | 61.66 | 67.46 | 70.10 |
| Ours, all valid (58 clips, half used for tuning) | 24.41 | 60.58 | 63.81 | 66.59 |
| SoccerNet baseline, test (Somers et al. 2024) | 22.26 | – | – | 42.65 |
| SoccerNet baseline, valid | 18.05 | – | – | – |

On test, the components: GS-HOTA DetA 9.46 and AssA 58.39; without jersey, DetA 49.85 and AssA 55.58.

**Oracle rows (valid held-out half):** each row replaces one stage with the labels.

| Replaced by labels | GS-HOTA | Without jersey | Position only |
|---|---|---|---|
| nothing (ours) | 25.38 | 61.66 | 70.10 |
| homography | 27.24 | 66.88 | 77.20 |
| role + team | 27.29 | 69.65 | 70.10 |
| role + team + **jersey** | **69.35** | 69.65 | 70.10 |
| homography + all attributes (detection and tracking left) | 76.25 | 76.64 | 77.20 |

### Visible players per frame (B1 preview, valid held-out half)

Players and goalkeepers per frame:
- labels: 14.85; ours: 13.52;
- mean difference −1.33; mean absolute difference 1.43;
- within ±1 in 70.6% of frames.

We undercount by about one player. Likely causes: missed or merged boxes in crowds, frames without a homography,
and goalkeepers or players set apart as referees. To analyse in B1.

### Per clip (held out, without jersey)

- The best clips score 79–80.
- The worst four are SNGS-058 (19.5), 092 (21.3), 054 (22.8) and 038 (30.2). Their DetA is 15–21, so these are
  position failures, probably homography (A5's SNGS-055 type) or close-up views. To inspect.

## Findings

1. **Already above the SoccerNet baseline without reading a single jersey number:** 23.5 vs 22.26 on test.
   Position only, the gap is much larger: 61.6 vs 42.7. Detection, tracking and calibration are the strong part.
2. **Jersey numbers are the missing piece.** 87% of player boxes have a number in the labels. With correct numbers,
   held-out GS-HOTA would rise from 25.4 to 69.4, and the other stages are already good enough for that. This is
   the next work package (A7: jersey number recognition, aggregated per track).
3. **Time matters for calibration.** Checking, filling and smoothing homographies over time adds 7 points and
   raises coverage from 80% to 92% of frames.
4. **Roles matter.** Setting referees and goalkeepers apart adds 7 points.
5. Remaining losses without jersey: about 5 points from the homography and 8 from role and team (oracle rows),
   then detection and tracking (77 with everything else perfect).
6. **Comparison caveat.** These numbers come from our own GS-HOTA implementation: the paper's similarity, the HOTA
   code from `trackers`, `bbox_pitch` ground truth and all labelled frames. Before the paper, re-score the saved
   predictions with the official SoccerNet evaluation (`sn-gamestate` / TrackEval), because its handling of null
   attributes and of frame filtering may differ slightly. Choi (2026, 35.2) used 11 test clips, not all 49.

Outputs (Drive `outputs/pipeline/`):
- tuning tables: `homography_tune.csv`, `roles_tune.csv`;
- results: `gs_hota_valid.csv`, `gs_hota_test.csv`, `per_clip_held_out.csv`, `visible_players_held_out.csv`;
- `settings.json`;
- caches: `keypoints_*`, `cpu_*`.
